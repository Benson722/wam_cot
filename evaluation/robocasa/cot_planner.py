# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
"""DeepSeek high-level planner for WAM-CoT Route-1 (External Semantic CoT).

Implements the PDF's 路线一: a VLM acts as a high-level planner that, from the
task instruction and the current camera frame, produces (a) an explicit
physical-constraint / occlusion *reasoning* trace and (b) an ordered list of
atomic sub-task instructions phrased in the short imperative style the
LIBERO-trained low-level WAM understands. A second `monitor` call provides the
VLM-monitored sub-task-completion / replanning signal.

Transport is a minimal OpenAI-compatible chat-completions call over stdlib
``urllib`` (no extra dependency in the lingbot conda env). DeepSeek's API is
OpenAI-compatible, so this also works against any OpenAI-style endpoint.

Every call (messages, raw response, latency, token usage) is appended to a
JSONL log so the report can include 推理时间与计算开销统计.
"""
from __future__ import annotations

import base64
import dataclasses
import io
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np


# ===========================================================================
# >>> HARDCODED DEEPSEEK CREDENTIALS — EDIT THESE ONCE, NO `export` NEEDED <<<
#
# Paste your real DeepSeek API key on the next line (keep the quotes). The
# exact "v4 pro" API model id is endpoint-specific — change it here if the
# default below is not what your account exposes. An environment variable, if
# set, still takes precedence over these (handy for one-off overrides).
# ===========================================================================
HARDCODED_DEEPSEEK_API_KEY = "sk-bf71c76a9b5a4baf82ea78759ba2a0fa"
HARDCODED_DEEPSEEK_MODEL = "deepseek-v4-pro"
HARDCODED_DEEPSEEK_BASE_URL = "https://api.deepseek.com"


def _cred(env_name: str, hardcoded: str) -> str:
    """env var wins if non-empty, else the hardcoded constant above."""
    v = os.environ.get(env_name, "").strip()
    return v if v else hardcoded


# ---------------------------------------------------------------------------
@dataclasses.dataclass
class PlannerConfig:
    base_url: str = _cred("DEEPSEEK_BASE_URL", HARDCODED_DEEPSEEK_BASE_URL)
    api_key: str = _cred("DEEPSEEK_API_KEY", HARDCODED_DEEPSEEK_API_KEY)
    model: str = _cred("DEEPSEEK_MODEL", HARDCODED_DEEPSEEK_MODEL)
    temperature: float = 0.2
    max_tokens: int = 1024
    timeout_s: int = 90
    max_retries: int = 4
    multimodal: bool = True          # set False / use blind mode for text-only
    log_path: Optional[str] = None   # JSONL call log


def _img_to_data_url(img: np.ndarray, max_side: int = 512) -> str:
    """uint8 HxWx3 (already vertically-correct) -> JPEG data URL."""
    from PIL import Image

    arr = np.ascontiguousarray(np.asarray(img)[..., :3]).astype(np.uint8)
    pil = Image.fromarray(arr)
    if max(pil.size) > max_side:
        s = max_side / max(pil.size)
        pil = pil.resize((int(pil.size[0] * s), int(pil.size[1] * s)))
    buf = io.BytesIO()
    pil.save(buf, format="JPEG", quality=85)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


_PLAN_SYS = """You are the high-level PLANNER for a single-arm robot performing \
kitchen manipulation in the RoboCasa simulator. You receive a task instruction \
and the current camera image. Think about PHYSICAL CONSTRAINTS before acting: \
occlusion, containment (a container must be opened before something is placed \
inside / removed), order (move a blocking or heavy object before grasping the \
target), reachability, and stability.

Decompose the task into an ORDERED list of ATOMIC sub-tasks. Each sub-task MUST:
 - be ONE primitive only: a single pick, place, open, close, push, turn, or press;
 - be a SHORT imperative instruction in the same style as low-level robot \
language commands, e.g. "open the cabinet door", "pick up the apple from the \
counter", "place the apple inside the cabinet", "close the cabinet door";
 - name concrete visible objects/locations (no pronouns, no multi-step verbs).

Respond with STRICT JSON only, no markdown, of the form:
{"reasoning": "<concise physical-constraint analysis>",
 "subtasks": [{"instruction": "<atomic imperative>", "max_steps": <int>}, ...]}
Keep 2-6 sub-tasks. max_steps is a generous per-sub-task safety budget \
(e.g. 120-300)."""

_MONITOR_SYS = """You are the high-level MONITOR for a single-arm RoboCasa robot. \
Given the overall task, the sub-task currently being executed, the remaining \
planned sub-tasks, and the current camera image, judge progress from what is \
VISIBLE.

Respond with STRICT JSON only, no markdown:
{"subtask_done": <bool>,        // current sub-task visually achieved
 "task_success": <bool>,        // the whole task is visually achieved
 "need_replan": <bool>,         // scene contradicts the remaining plan
 "revised_plan": [{"instruction": "...", "max_steps": <int>}, ...] | null,
 "reason": "<one short sentence>"}
Be conservative: only set subtask_done/task_success true when clearly visible."""


class CoTPlanner:
    """Route-1 semantic planner. Stateless w.r.t. the env; the orchestrator
    owns plan progression."""

    def __init__(self, cfg: PlannerConfig):
        self.cfg = cfg
        self.calls = 0
        self.total_latency_ms = 0.0
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        if cfg.log_path:
            Path(cfg.log_path).parent.mkdir(parents=True, exist_ok=True)

    # -- low-level transport ----------------------------------------------
    def _chat(self, system: str, user_text: str,
              image: Optional[np.ndarray]) -> str:
        content: list = [{"type": "text", "text": user_text}]
        if image is not None and self.cfg.multimodal:
            content.append(
                {"type": "image_url",
                 "image_url": {"url": _img_to_data_url(image)}}
            )
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": content},
        ]
        payload = {
            "model": self.cfg.model,
            "messages": messages,
            "temperature": self.cfg.temperature,
            "max_tokens": self.cfg.max_tokens,
            "stream": False,
        }
        data = json.dumps(payload).encode("utf-8")
        url = self.cfg.base_url.rstrip("/") + "/v1/chat/completions"
        if self.cfg.base_url.rstrip("/").endswith("/v1"):
            url = self.cfg.base_url.rstrip("/") + "/chat/completions"

        last_err = None
        for attempt in range(self.cfg.max_retries):
            t0 = time.monotonic()
            try:
                req = urllib.request.Request(
                    url, data=data,
                    headers={
                        "Content-Type": "application/json",
                        "Authorization": f"Bearer {self.cfg.api_key}",
                    },
                    method="POST",
                )
                with urllib.request.urlopen(req, timeout=self.cfg.timeout_s) as r:
                    resp = json.loads(r.read().decode("utf-8"))
                dt = (time.monotonic() - t0) * 1000.0
                txt = resp["choices"][0]["message"]["content"]
                usage = resp.get("usage", {}) or {}
                self.calls += 1
                self.total_latency_ms += dt
                self.total_prompt_tokens += int(usage.get("prompt_tokens", 0))
                self.total_completion_tokens += int(
                    usage.get("completion_tokens", 0)
                )
                self._log(system, user_text, image is not None, txt, dt, usage)
                return txt
            except (urllib.error.URLError, KeyError, TimeoutError,
                    json.JSONDecodeError) as e:  # noqa: PERF203
                last_err = e
                time.sleep(min(2 ** attempt, 10))
        raise RuntimeError(
            f"DeepSeek call failed after {self.cfg.max_retries} retries: {last_err}"
        )

    def _log(self, system, user_text, had_image, response, dt, usage):
        if not self.cfg.log_path:
            return
        with open(self.cfg.log_path, "a") as f:
            f.write(json.dumps({
                "ts": time.time(),
                "model": self.cfg.model,
                "system": system[:200],
                "user": user_text[:1000],
                "had_image": had_image,
                "latency_ms": round(dt, 1),
                "usage": usage,
                "response": response,
            }, default=str) + "\n")

    @staticmethod
    def _parse_json(txt: str) -> dict:
        s = txt.strip()
        if s.startswith("```"):
            s = s.split("```", 2)[1]
            if s.lstrip().lower().startswith("json"):
                s = s.lstrip()[4:]
        a, b = s.find("{"), s.rfind("}")
        if a != -1 and b != -1:
            s = s[a:b + 1]
        return json.loads(s)

    # -- high-level API ----------------------------------------------------
    def plan(self, task_instruction: str,
             image: Optional[np.ndarray]) -> Dict:
        """-> {"reasoning": str, "subtasks": [{"instruction","max_steps"}...]}"""
        user = (
            f"TASK: {task_instruction}\n"
            f"{'(no image provided; reason from the task text only)' if image is None or not self.cfg.multimodal else 'The image shows the current scene.'}\n"
            "Produce the physical-constraint reasoning and the ordered atomic "
            "sub-task plan as STRICT JSON."
        )
        raw = self._chat(_PLAN_SYS, user, image)
        try:
            obj = self._parse_json(raw)
            subs = obj.get("subtasks") or []
            norm = []
            for s in subs:
                if isinstance(s, str):
                    norm.append({"instruction": s, "max_steps": 240})
                else:
                    norm.append({
                        "instruction": str(s.get("instruction", "")).strip(),
                        "max_steps": int(s.get("max_steps", 240) or 240),
                    })
            norm = [s for s in norm if s["instruction"]]
            if not norm:  # planner produced nothing usable
                norm = [{"instruction": task_instruction, "max_steps": 600}]
                obj["planner_failed"] = True
            obj["subtasks"] = norm
            obj.setdefault("reasoning", "")
            return obj
        except Exception as e:  # noqa: BLE001
            return {
                "reasoning": f"<plan parse error: {e}> raw={raw[:300]}",
                "subtasks": [{"instruction": task_instruction,
                              "max_steps": 600}],
                "planner_failed": True,
            }

    def monitor(self, task_instruction: str, current_subtask: str,
                remaining: List[str], image: Optional[np.ndarray]) -> Dict:
        """-> {subtask_done, task_success, need_replan, revised_plan, reason}"""
        user = (
            f"TASK: {task_instruction}\n"
            f"CURRENT SUB-TASK (being executed now): {current_subtask}\n"
            f"REMAINING PLAN: {json.dumps(remaining)}\n"
            "Judge progress from the image. STRICT JSON only."
        )
        raw = self._chat(_MONITOR_SYS, user, image)
        try:
            obj = self._parse_json(raw)
            return {
                "subtask_done": bool(obj.get("subtask_done", False)),
                "task_success": bool(obj.get("task_success", False)),
                "need_replan": bool(obj.get("need_replan", False)),
                "revised_plan": obj.get("revised_plan"),
                "reason": str(obj.get("reason", "")),
            }
        except Exception as e:  # noqa: BLE001
            # On monitor parse failure, do not advance/replan (fail safe).
            return {
                "subtask_done": False, "task_success": False,
                "need_replan": False, "revised_plan": None,
                "reason": f"<monitor parse error: {e}>",
            }

    def stats(self) -> Dict:
        return {
            "vlm_calls": self.calls,
            "vlm_total_latency_ms": round(self.total_latency_ms, 1),
            "vlm_avg_latency_ms": round(
                self.total_latency_ms / max(self.calls, 1), 1
            ),
            "vlm_prompt_tokens": self.total_prompt_tokens,
            "vlm_completion_tokens": self.total_completion_tokens,
        }
