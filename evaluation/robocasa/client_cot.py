# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
"""WAM-CoT client for Robocasa — PDF Phase-2, Route-1 (External Semantic CoT).

Pipeline per episode:
  1. reset Robocasa; read the task language + first frame.
  2. DeepSeek PLANNER: physical-constraint reasoning -> ordered atomic
     sub-task instructions (Route-1 semantic chain-of-thought).
  3. WAM executes sub-task[0] (server `reset(prompt=subtask)`).
  4. Every K key-frames the DeepSeek MONITOR judges the frame:
        subtask_done -> soft `switch_prompt` to the next sub-task
                        (KV / world-model context preserved);
        need_replan  -> re-plan from the current frame;
        task_success -> stop.
  5. Per-sub-task step budgets are a safety cap.

Ablations (PDF 消融实验 · 必做) via --ablation:
  none            full WAM-CoT
  no_cot          single prompt = full task (== Baseline, same harness)
  shuffle_subtasks   shuffle the planned sub-task order
  no_monitor      no VLM feedback; advance only on per-sub-task step budget
                  (open-loop plan -> ablates the CoT *observation* model)
  blind_planner   planner gets NO image (text-only) -> degraded perception
  hard_reset      `reset` instead of `switch_prompt` at sub-task boundaries
                  -> ablates the soft-switch world-model-context carryover

Run (internet-capable 4090, server already up):
  export DEEPSEEK_API_KEY=sk-...
  python evaluation/robocasa/client_cot.py \
     --tasks PnPCounterToCab --port 29056 --test-num 25 \
     --ablation none --out-dir outputs/robocasa/cot
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import traceback
from pathlib import Path

from tqdm import tqdm

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
)

from evaluation.robocasa.cot_planner import CoTPlanner, PlannerConfig  # noqa: E402
from evaluation.robocasa.eval_common import (  # noqa: E402
    WamSession,
    classify_failure,
    execute_action_chunk,
    now_str,
    save_views_video,
    write_json,
)
from evaluation.robocasa.robocasa_env import (  # noqa: E402
    DEFAULT_TASKS,
    RobocasaConfig,
    RobocasaEnv,
)

ABLATIONS = (
    "none", "no_cot", "shuffle_subtasks", "no_monitor",
    "blind_planner", "hard_reset",
)


def _make_plan(planner, language, first_img, ablation):
    """Return (reasoning, [ {instruction,max_steps}... ], planner_failed)."""
    if ablation == "no_cot":
        return ("(no_cot ablation: single full-task prompt)",
                [{"instruction": language, "max_steps": 100000}], False)
    img = None if ablation == "blind_planner" else first_img
    plan = planner.plan(language, img)
    subs = plan.get("subtasks", [])
    if ablation == "shuffle_subtasks" and len(subs) > 1:
        subs = subs[:]
        random.shuffle(subs)
    return (plan.get("reasoning", ""), subs,
            bool(plan.get("planner_failed", False)))


def run_one(session, planner, env_name, ep, out_dir, env_overrides,
            max_steps, ablation, monitor_every_kf):
    cfg = RobocasaConfig(env_name=env_name, max_env_steps=max_steps)
    cfg = cfg.merge_overrides(env_overrides)
    use_monitor = ablation not in ("no_cot", "no_monitor")
    hard_reset = ablation == "hard_reset"

    env = None
    full_obs_list = []
    overlay = []                 # [(frame_idx, subtask_text)]
    success = False
    completed = 0
    planner_failed = False
    env_errored = False
    language = env_name
    plan_trace = {}
    n_total = 0

    try:
        env = RobocasaEnv(cfg)
        first_obs = env.reset()
        language = env.task_language
        first_img = first_obs.get("observation.images.agentview_rgb")
        print(f"[cot:{ablation}] {env_name} ep{ep} :: '{language}'")

        reasoning, subtasks, planner_failed = _make_plan(
            planner, language, first_img, ablation
        )
        plan_trace = {
            "language": language, "ablation": ablation,
            "reasoning": reasoning,
            "subtasks": [s["instruction"] for s in subtasks],
            "events": [],
        }
        if not subtasks:
            subtasks = [{"instruction": language, "max_steps": max_steps}]
            planner_failed = True

        # ---- start first sub-task ----
        si = 0
        cur = subtasks[si]
        session.reset(prompt=cur["instruction"])
        overlay.append((0, f"[1/{len(subtasks)}] {cur['instruction']}"))
        first = True
        steps_in_sub = 0
        replans = 0
        max_replans = 6
        # mutable holder so the keyframe hook can stash the monitor verdict
        verdict = {"signal": None, "reason": ""}

        def on_kf(step_global, _env, obs):
            nonlocal steps_in_sub
            # per-sub-task safety budget (only advance mechanism if no monitor)
            if steps_in_sub >= cur["max_steps"]:
                verdict["reason"] = "budget_exhausted"
                return "stop_advance"
            if not use_monitor:
                return None
            kf_idx = len(full_obs_list)
            if kf_idx % max(monitor_every_kf, 1) != 0:
                return None
            m = planner.monitor(
                language, cur["instruction"],
                [s["instruction"] for s in subtasks[si + 1:]],
                None if ablation == "blind_planner" else
                obs.get("observation.images.agentview_rgb"),
            )
            verdict["reason"] = m.get("reason", "")
            if m["task_success"]:
                return "stop_success"
            if m["need_replan"]:
                verdict["replan"] = m.get("revised_plan")
                return "stop_replan"
            if m["subtask_done"]:
                return "stop_advance"
            return None

        while env.ep_steps < max_steps:
            action = session.infer_action(first_obs, cur["instruction"])
            pre = env.ep_steps
            key_frames, done, steps, signal = execute_action_chunk(
                env, action, first, full_obs_list,
                on_keyframe=on_kf, global_step_offset=pre,
            )
            steps_in_sub += (env.ep_steps - pre)
            n_total = env.ep_steps
            first = False

            if env.check_success() or signal == "stop_success":
                success = True
                plan_trace["events"].append(
                    {"step": n_total, "event": "task_success",
                     "reason": verdict.get("reason", "")}
                )
                break
            if done:
                break

            if signal == "stop_replan" and replans < max_replans:
                replans += 1
                completed = max(completed, si)
                rp = verdict.get("replan")
                if rp:
                    subtasks = [
                        {"instruction": str(s.get("instruction", s)),
                         "max_steps": int(s.get("max_steps", 240))
                         if isinstance(s, dict) else 240}
                        for s in rp
                    ]
                else:
                    _, subtasks, _ = _make_plan(
                        planner, language,
                        full_obs_list[-1].get(
                            "observation.images.agentview_rgb"),
                        ablation,
                    )
                si = 0
                cur = subtasks[si]
                steps_in_sub = 0
                plan_trace["events"].append(
                    {"step": n_total, "event": "replan",
                     "new_plan": [s["instruction"] for s in subtasks]})
                overlay.append(
                    (len(full_obs_list),
                     f"[replan] {cur['instruction']}"))
                _advance_server(session, cur["instruction"], hard_reset)
                first = bool(hard_reset)
                if hard_reset:
                    first_obs = full_obs_list[-1]
                continue

            if signal == "stop_advance":
                completed += 1
                plan_trace["events"].append(
                    {"step": n_total, "event": "subtask_done",
                     "subtask": cur["instruction"],
                     "reason": verdict.get("reason", "")})
                si += 1
                if si >= len(subtasks):
                    break  # plan exhausted; let final success check decide
                cur = subtasks[si]
                steps_in_sub = 0
                overlay.append(
                    (len(full_obs_list),
                     f"[{si + 1}/{len(subtasks)}] {cur['instruction']}"))
                _advance_server(session, cur["instruction"], hard_reset)
                first = bool(hard_reset)
                if hard_reset:
                    first_obs = full_obs_list[-1]
                continue

            # normal chunk consumed -> feed executed frames back as context
            if key_frames:
                session.compute_kv_cache(key_frames, state=action)
            else:
                break

        success = success or env.check_success()
        if success:
            completed = len(subtasks)
    except Exception as e:  # noqa: BLE001
        env_errored = True
        print(f"[cot:{ablation}] EXCEPTION {env_name} ep{ep}: {e}")
        traceback.print_exc()
    finally:
        if env is not None:
            n_total = env.ep_steps
            env.close()

    n_sub = len(plan_trace.get("subtasks", [])) or 1
    tag = classify_failure(
        success=success, completed_subtasks=completed, total_subtasks=n_sub,
        env_steps=n_total, max_steps=max_steps,
        planner_failed=planner_failed, env_errored=env_errored,
    )
    stem = Path(out_dir) / env_name / f"{ep}_{success}_{tag}"
    save_views_video(
        full_obs_list, str(stem) + ".mp4", fps=30,
        overlay_timeline=overlay, reasoning=plan_trace.get("reasoning", ""),
    )
    write_json(plan_trace, str(stem) + ".plan.json")
    return success, tag, n_total, completed, n_sub, language


def _advance_server(session, instruction, hard_reset):
    if hard_reset:
        session.reset(prompt=instruction)            # ablation: drop context
    else:
        session.switch_prompt(prompt=instruction)    # WAM-CoT soft switch


def run(tasks, port, out_dir, test_num, env_overrides, max_steps,
        ablation, planner_cfg, monitor_every_kf):
    session = WamSession(port=port)
    planner = CoTPlanner(planner_cfg)
    summary = {}
    for env_name in tasks:
        succ = 0.0
        sub_done = 0
        sub_tot = 0
        tags = {}
        steps_list = []
        for ep in tqdm(range(test_num), desc=f"{env_name}[{ablation}]"):
            ok, tag, steps, c, nsub, lang = run_one(
                session, planner, env_name, ep, out_dir, env_overrides,
                max_steps, ablation, monitor_every_kf,
            )
            succ += float(ok)
            sub_done += c
            sub_tot += nsub
            steps_list.append(steps)
            tags[tag] = tags.get(tag, 0) + 1
            res = {
                "task": env_name, "language": lang, "method": "wam_cot",
                "ablation": ablation,
                "succ_num": succ, "total_num": ep + 1.0,
                "succ_rate": succ / (ep + 1),
                "subtask_progress_rate": sub_done / max(sub_tot, 1),
                "avg_steps": sum(steps_list) / len(steps_list),
                "failure_tags": tags,
                **planner.stats(),
            }
            write_json(res, Path(out_dir) / f"{env_name}.json")
            print(f"[cot:{ablation}] {env_name}: SR={succ/(ep+1):.3f} "
                  f"subSR={sub_done/max(sub_tot,1):.3f} tags={tags}")
        summary[env_name] = res
    write_json(
        {"created": now_str(), "method": "wam_cot", "ablation": ablation,
         "tasks": summary, "planner": planner.stats()},
        Path(out_dir) / "summary.json",
    )
    print(f"[cot:{ablation}] done.")


def main():
    ap = argparse.ArgumentParser(description="Robocasa WAM-CoT (Route-1) client")
    ap.add_argument("--tasks", nargs="+", default=list(DEFAULT_TASKS))
    ap.add_argument("--port", type=int, default=29056)
    ap.add_argument("--test-num", type=int, default=25)
    ap.add_argument("--max-steps", type=int, default=1500)
    ap.add_argument("--out-dir", default="outputs/robocasa/cot")
    ap.add_argument("--ablation", choices=ABLATIONS, default="none")
    ap.add_argument("--env-overrides", default=None,
                    help="JSON overriding RobocasaConfig (post probe_env.py)")
    ap.add_argument("--monitor-every-keyframes", type=int, default=2,
                    help="VLM-monitor cadence in key-frames (API cost knob)")
    # planner / DeepSeek
    ap.add_argument("--vlm-model",
                    default=os.environ.get("DEEPSEEK_MODEL", "deepseek-v4-pro"))
    ap.add_argument("--vlm-base-url",
                    default=os.environ.get("DEEPSEEK_BASE_URL",
                                           "https://api.deepseek.com"))
    ap.add_argument("--vlm-text-only", action="store_true",
                    help="force text-only planner (model not multimodal)")
    args = ap.parse_args()

    overrides = json.loads(args.env_overrides) if args.env_overrides else None
    pcfg = PlannerConfig(
        base_url=args.vlm_base_url,
        model=args.vlm_model,
        multimodal=not args.vlm_text_only,
        log_path=str(Path(args.out_dir) / "vlm_calls.jsonl"),
    )
    if not pcfg.api_key and args.ablation != "no_cot":
        print("WARNING: DEEPSEEK_API_KEY is empty; only --ablation no_cot "
              "will work without it.")
    run(args.tasks, args.port, args.out_dir, args.test_num, overrides,
        args.max_steps, args.ablation, pcfg, args.monitor_every_keyframes)


if __name__ == "__main__":
    main()
