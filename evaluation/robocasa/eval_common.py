# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
"""Shared evaluation plumbing for the Robocasa baseline & WAM-CoT clients.

Keeps the *protocol-critical* bits (websocket session, the LIBERO-identical
action-chunk consumption + KV-cache replanning loop, video/metrics IO) in one
place so the Baseline client and the WAM-CoT client differ only in their
high-level orchestration (single fixed prompt vs. VLM-planned sub-task
sequence with mid-episode soft prompt-switch).
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

# repo root on path so `python evaluation/robocasa/client.py` and
# `python -m evaluation.robocasa.client` both work.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from wan_va.utils.Simple_Remote_Infer.deploy.websocket_client_policy import (  # noqa: E402
    WebsocketClientPolicy,
)


# ---------------------------------------------------------------------------
# WAM server session
# ---------------------------------------------------------------------------
class WamSession:
    """Typed wrapper over the msgpack websocket policy, exposing the four
    message types the LingBot server understands (the 4th, ``switch_prompt``,
    is the WAM-CoT soft prompt-switch added in ``wan_va_server.py``)."""

    def __init__(self, port: int, host: str = "0.0.0.0"):
        self._client = WebsocketClientPolicy(host=host, port=port)

    def reset(self, prompt: str) -> None:
        self._client.infer(dict(reset=True, prompt=prompt))

    def switch_prompt(self, prompt: str) -> None:
        """WAM-CoT: advance to a new sub-task instruction WITHOUT discarding
        the autoregressive world-model context (KV cache / frame_st_id)."""
        self._client.infer(dict(switch_prompt=True, prompt=prompt))

    def infer_action(self, obs: Dict[str, np.ndarray], prompt: str) -> np.ndarray:
        ret = self._client.infer(dict(obs=obs, prompt=prompt))
        return ret["action"]

    def compute_kv_cache(self, key_frames: List[Dict], state: np.ndarray) -> None:
        self._client.infer(
            dict(obs=key_frames, compute_kv_cache=True, imagine=False, state=state)
        )


# ---------------------------------------------------------------------------
# Action-chunk consumption (identical semantics to evaluation/libero/client.py)
# ---------------------------------------------------------------------------
# on_keyframe(step_global, env, lingbot_obs) -> str | None
#   return "stop_success" / "stop_advance" / "stop_replan" to break the chunk
#   early (used by the WAM-CoT VLM monitor); return None/"continue" otherwise.
KeyframeHook = Callable[[int, "object", Dict[str, np.ndarray]], Optional[str]]


def execute_action_chunk(
    env,
    action: np.ndarray,
    first: bool,
    full_obs_list: List[Dict],
    on_keyframe: Optional[KeyframeHook] = None,
    global_step_offset: int = 0,
) -> Tuple[List[Dict], bool, int, Optional[str]]:
    """Run one predicted action chunk through the env.

    Mirrors the LIBERO client's nested ``for i in shape[1] / for j in shape[2]``
    loop and key-frame sampling so the zero-shot LIBERO checkpoint sees exactly
    the protocol it was trained/served with.

    Returns ``(key_frame_list, done, n_env_steps, hook_signal)``.
    """
    assert action.shape[2] % 4 == 0, f"action.shape[2]={action.shape[2]} not %4"
    action_per_frame = action.shape[2] // 4
    key_frame_list: List[Dict] = []
    done = False
    n_steps = 0
    signal: Optional[str] = None

    start_idx = 1 if first else 0
    for i in range(start_idx, action.shape[1]):
        for j in range(action.shape[2]):
            ee_action = action[:, i, j]
            obs, done = env.step(ee_action)
            n_steps += 1
            if done:
                break
            if (j + 1) % action_per_frame == 0:
                full_obs_list.append(obs)
                key_frame_list.append(obs)
                if on_keyframe is not None:
                    signal = on_keyframe(
                        global_step_offset + n_steps, env, obs
                    )
                    if signal in ("stop_success", "stop_advance", "stop_replan"):
                        return key_frame_list, done, n_steps, signal
        if done:
            break
    return key_frame_list, done, n_steps, signal


# ---------------------------------------------------------------------------
# IO helpers
# ---------------------------------------------------------------------------
def now_str() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def write_json(obj: dict, path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)


def _stack_views(obs: Dict[str, np.ndarray], size: Tuple[int, int]) -> np.ndarray:
    import cv2

    keys = [
        "observation.images.agentview_rgb",
        "observation.images.eye_in_hand_rgb",
    ]
    tiles = [
        cv2.resize(np.ascontiguousarray(obs[k]), size).astype(np.uint8)
        for k in keys
        if k in obs
    ]
    return np.hstack(tiles)


def save_views_video(
    real_obs_list: List[Dict],
    save_path,
    fps: int = 30,
    overlay_timeline: Optional[List[Tuple[int, str]]] = None,
    reasoning: str = "",
) -> None:
    """Save the stacked 2-camera rollout. ``overlay_timeline`` is a list of
    ``(frame_index, subtask_text)`` transitions used to burn the WAM-CoT
    sub-task plan onto the video for the demo deliverable."""
    if not real_obs_list:
        print("[video] no frames to save")
        return
    import cv2
    import imageio

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    h, w = next(iter(real_obs_list[0].values())).shape[:2]
    size = (w, h)

    # build a step -> subtask label lookup
    label_at = {}
    if overlay_timeline:
        cur = ""
        ti = 0
        for fi in range(len(real_obs_list)):
            while ti < len(overlay_timeline) and overlay_timeline[ti][0] <= fi:
                cur = overlay_timeline[ti][1]
                ti += 1
            label_at[fi] = cur

    frames = []
    for fi, obs in enumerate(real_obs_list):
        frame = _stack_views(obs, size)
        if overlay_timeline is not None:
            bar = np.zeros((34, frame.shape[1], 3), dtype=np.uint8)
            txt = (label_at.get(fi, "") or "")[:90]
            cv2.putText(
                bar, txt, (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (0, 255, 0), 1, cv2.LINE_AA,
            )
            frame = np.vstack([bar, frame])
        frames.append(frame.astype(np.uint8))
    imageio.mimsave(str(save_path), frames, fps=fps)
    print(f"[video] saved {len(frames)} frames -> {save_path}")


# ---------------------------------------------------------------------------
# Failure taxonomy (PDF: 失败类型统计 / 针对失败案例的机制分析)
# ---------------------------------------------------------------------------
FAILURE_TAGS = (
    "success",
    "timeout_no_progress",        # ran out of steps, never near goal
    "wrong_object_or_location",   # manipulated wrong thing (planner/grounding)
    "grasp_failure",              # approached but failed to grasp
    "subtask_stuck",              # a sub-task never completed (CoT-specific)
    "planner_error",              # VLM produced unusable / empty plan
    "env_error",                  # exception in env / server
)


def classify_failure(
    success: bool,
    completed_subtasks: int,
    total_subtasks: int,
    env_steps: int,
    max_steps: int,
    planner_failed: bool = False,
    env_errored: bool = False,
) -> str:
    if env_errored:
        return "env_error"
    if success:
        return "success"
    if planner_failed:
        return "planner_error"
    if total_subtasks and completed_subtasks < total_subtasks:
        return "subtask_stuck"
    if env_steps >= max_steps:
        return "timeout_no_progress"
    return "wrong_object_or_location"
