# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
"""Phase-1 Baseline WAM client for Robocasa.

Direct port of ``evaluation/libero/client.py`` to Robocasa: the world-action
model receives the language instruction + visual observation and outputs a
low-level action chunk, with NO explicit intermediate reasoning. This is the
control group the WAM-CoT (Route-1) client is compared against.

Run (on the internet-capable 4090, after launching the server):
    python evaluation/robocasa/client.py \
        --tasks PickPlaceCounterToCabinet PickPlaceCounterToMicrowave \
        --port 29056 --test-num 25 --out-dir outputs/robocasa/baseline
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from pathlib import Path

from tqdm import tqdm

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
)

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


def run_one(session, env_name, episode_idx, out_dir, env_overrides, max_steps):
    """One Baseline episode. Returns (success, failure_tag, n_steps, language)."""
    cfg = RobocasaConfig(env_name=env_name, max_env_steps=max_steps)
    cfg = cfg.merge_overrides(env_overrides)
    env = None
    full_obs_list = []
    success = False
    n_total = 0
    env_errored = False
    language = env_name
    try:
        env = RobocasaEnv(cfg)
        first_obs = env.reset()
        language = env.task_language
        print(f"[baseline] {env_name} ep{episode_idx} :: '{language}'")

        session.reset(prompt=language)
        done = False
        first = True
        replans = 0
        max_replans = max_steps  # safety; env enforces real step cap
        while not done and env.ep_steps < max_steps and replans < max_replans:
            action = session.infer_action(first_obs, language)
            pre = env.ep_steps
            key_frames, done, steps, _ = execute_action_chunk(
                env, action, first, full_obs_list,
                on_keyframe=None, global_step_offset=pre,
            )
            n_total = env.ep_steps
            first = False
            replans += 1
            if done:
                break
            if key_frames:
                session.compute_kv_cache(key_frames, state=action)
            else:
                break  # no progress this chunk -> avoid spin
        success = env.check_success()
    except Exception as e:  # noqa: BLE001
        env_errored = True
        print(f"[baseline] EXCEPTION {env_name} ep{episode_idx}: {e}")
        traceback.print_exc()
    finally:
        if env is not None:
            n_total = env.ep_steps
            env.close()

    tag = classify_failure(
        success=success,
        completed_subtasks=1 if success else 0,
        total_subtasks=1,
        env_steps=n_total,
        max_steps=max_steps,
        env_errored=env_errored,
    )
    out_file = (
        Path(out_dir) / env_name
        / f"{episode_idx}_{success}_{tag}.mp4"
    )
    save_views_video(full_obs_list, out_file, fps=30)
    return success, tag, n_total, language


def run(tasks, port, out_dir, test_num, env_overrides, max_steps):
    session = WamSession(port=port)
    summary = {}
    for env_name in tasks:
        succ_num = 0.0
        tags = {}
        steps_list = []
        for ep in tqdm(range(test_num), desc=env_name):
            ok, tag, steps, lang = run_one(
                session, env_name, ep, out_dir, env_overrides, max_steps
            )
            succ_num += float(ok)
            tags[tag] = tags.get(tag, 0) + 1
            steps_list.append(steps)
            rate = succ_num / (ep + 1)
            res = {
                "task": env_name,
                "language": lang,
                "succ_num": succ_num,
                "total_num": ep + 1.0,
                "succ_rate": rate,
                "avg_steps": sum(steps_list) / len(steps_list),
                "failure_tags": tags,
                "method": "baseline",
            }
            write_json(res, Path(out_dir) / f"{env_name}.json")
            print(
                f"[baseline] {env_name}: SR={rate:.3f} "
                f"({int(succ_num)}/{ep + 1}) tags={tags}"
            )
        summary[env_name] = res
    write_json(
        {"created": now_str(), "method": "baseline", "tasks": summary},
        Path(out_dir) / "summary.json",
    )
    print("[baseline] done.")


def main():
    ap = argparse.ArgumentParser(description="Robocasa Baseline WAM client")
    ap.add_argument("--tasks", nargs="+", default=list(DEFAULT_TASKS),
                    help="Robocasa env names (see `python -m robocasa.demos.demo_tasks`)")
    ap.add_argument("--port", type=int, default=29056)
    ap.add_argument("--test-num", type=int, default=25, help="episodes / task")
    ap.add_argument("--max-steps", type=int, default=1200)
    ap.add_argument("--out-dir", default="outputs/robocasa/baseline")
    ap.add_argument(
        "--env-overrides", default=None,
        help="JSON dict overriding RobocasaConfig fields (camera_map, "
        "arm_action_slice, gripper_index, robots, ... — use after probe_env.py)",
    )
    args = ap.parse_args()
    overrides = json.loads(args.env_overrides) if args.env_overrides else None
    run(args.tasks, args.port, args.out_dir, args.test_num, overrides,
        args.max_steps)


if __name__ == "__main__":
    main()
