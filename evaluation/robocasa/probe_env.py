# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
"""Robocasa environment probe.

Run this ON THE SERVER (conda env `robocasa`) BEFORE the first evaluation.
It introspects the real Robocasa / robosuite install — versions, action
dimension & spec, observation keys, camera names, the robot's composite
controller action split, the per-episode language instruction and a sample
frame — and writes everything to JSON.

Paste the JSON back so the camera_map / action layout in
`evaluation/robocasa/robocasa_env.py` (RobocasaConfig) can be finalized; the
defaults target PandaOmron + recent robosuite but the install may differ.

Usage:
    conda activate robocasa
    cd /inspire/qb-ilm2/project/26summer-camp-11/public/group3/robocasa_suite/robocasa
    python -m evaluation.robocasa.probe_env --env PnPCounterToCab \
        --out outputs/robocasa_probe.json
    # (or run from the lingbot repo with PYTHONPATH including it)
"""
from __future__ import annotations

import argparse
import json
import os
import traceback
from typing import Any

import numpy as np


def _safe(fn, default=None):
    try:
        return fn()
    except Exception as e:  # noqa: BLE001
        return f"<error: {e}>" if default is None else default


def _jsonable(x: Any):
    if isinstance(x, (str, int, float, bool)) or x is None:
        return x
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return {"__ndarray__": True, "shape": list(x.shape), "dtype": str(x.dtype)}
    return str(x)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default="PnPCounterToCab", help="Robocasa env name")
    ap.add_argument("--robots", default="PandaOmron")
    ap.add_argument("--controller", default=None)
    ap.add_argument("--height", type=int, default=256)
    ap.add_argument("--width", type=int, default=256)
    ap.add_argument("--out", default="outputs/robocasa_probe.json")
    ap.add_argument("--save-frames", action="store_true",
                    help="also dump the resolved agentview/eye_in_hand PNGs")
    args = ap.parse_args()

    report: dict = {"env_name": args.env, "robots": args.robots}

    # --- versions ---------------------------------------------------------
    import robocasa  # registers envs
    import robosuite

    report["versions"] = {
        "robosuite": _safe(lambda: robosuite.__version__, "?"),
        "robocasa": _safe(lambda: robocasa.__version__, "?"),
        "numpy": np.__version__,
    }
    report["robosuite_file"] = _safe(lambda: robosuite.__file__)

    # --- controller config ------------------------------------------------
    ctrl_cfg = None
    ctrl_api = None
    try:
        from robosuite.controllers import load_composite_controller_config

        ctrl_cfg = load_composite_controller_config(
            controller=args.controller, robot=args.robots
        )
        ctrl_api = "load_composite_controller_config"
    except Exception:
        try:
            from robosuite.controllers import load_controller_config

            ctrl_cfg = load_controller_config(
                default_controller=args.controller or "OSC_POSE"
            )
            ctrl_api = "load_controller_config"
        except Exception as e:
            report["controller_error"] = f"{e}\n{traceback.format_exc()}"
    report["controller_api"] = ctrl_api
    report["controller_config"] = _jsonable(ctrl_cfg)

    # --- make env (resilient to unknown kwargs) ---------------------------
    base_kwargs = dict(
        env_name=args.env,
        robots=args.robots,
        controller_configs=ctrl_cfg,
        has_renderer=False,
        has_offscreen_renderer=True,
        use_camera_obs=True,
        use_object_obs=True,
        camera_heights=args.height,
        camera_widths=args.width,
        control_freq=20,
        ignore_done=True,
        camera_names=[
            "robot0_agentview_left",
            "robot0_agentview_right",
            "robot0_agentview_center",
            "robot0_eye_in_hand",
            "robot0_frontview",
        ],
    )
    env = None
    last_err = None
    for drop in ([], ["use_object_obs"], ["ignore_done", "use_object_obs"]):
        kw = {k: v for k, v in base_kwargs.items() if k not in drop}
        try:
            env = robosuite.make(**kw)
            report["make_kwargs_used"] = sorted(kw.keys())
            break
        except Exception as e:  # noqa: BLE001
            last_err = f"{e}"
    if env is None:
        report["make_error"] = f"{last_err}\n{traceback.format_exc()}"
        _dump(report, args.out)
        return

    # --- action space -----------------------------------------------------
    report["action_dim"] = _safe(lambda: int(env.action_dim))
    try:
        low, high = env.action_spec
        report["action_spec"] = {
            "low": np.asarray(low).round(4).tolist(),
            "high": np.asarray(high).round(4).tolist(),
        }
    except Exception as e:
        report["action_spec_error"] = str(e)

    # --- robot / controller action split ----------------------------------
    try:
        robot = env.robots[0]
        report["robot_class"] = type(robot).__name__
        cc = getattr(robot, "composite_controller", None)
        report["composite_controller_class"] = type(cc).__name__ if cc else None
        split = getattr(robot, "_action_split_indexes", None) or getattr(
            robot, "action_split_indexes", None
        )
        report["action_split_indexes"] = _jsonable(split)
        if cc is not None:
            report["controller_parts"] = _safe(
                lambda: list(cc.part_controllers.keys())
            )
    except Exception as e:
        report["robot_introspect_error"] = str(e)

    # --- reset + observation ---------------------------------------------
    try:
        obs = env.reset()
        report["obs_keys"] = sorted(obs.keys())
        report["image_keys"] = sorted(k for k in obs if k.endswith("_image"))
        report["obs_shapes"] = {
            k: {"shape": list(np.asarray(v).shape), "dtype": str(np.asarray(v).dtype)}
            for k, v in obs.items()
            if hasattr(v, "shape")
        }
    except Exception as e:
        report["reset_error"] = f"{e}\n{traceback.format_exc()}"
        _dump(report, args.out)
        env.close()
        return

    # --- language instruction --------------------------------------------
    report["task_language"] = _safe(
        lambda: (env.get_ep_meta() or {}).get("lang"), "<no get_ep_meta>"
    )
    report["ep_meta_keys"] = _safe(lambda: sorted((env.get_ep_meta() or {}).keys()))

    # --- success api ------------------------------------------------------
    report["has__check_success"] = hasattr(env, "_check_success")
    report["sample_check_success"] = _safe(lambda: env._check_success())

    # --- one zero-action step --------------------------------------------
    try:
        a = np.zeros(int(env.action_dim))
        o2, r, d, info = env.step(a)
        report["step_return"] = {
            "reward": float(r),
            "done": bool(d),
            "info_keys": sorted(info.keys()) if isinstance(info, dict) else None,
        }
    except Exception as e:
        report["step_error"] = str(e)

    # --- optional frame dump ---------------------------------------------
    if args.save_frames:
        try:
            import imageio

            for k in report["image_keys"]:
                imageio.imwrite(
                    os.path.join(
                        os.path.dirname(args.out) or ".", f"probe_{k}.png"
                    ),
                    np.asarray(obs[k])[::-1],
                )
        except Exception as e:
            report["frame_dump_error"] = str(e)

    env.close()
    _dump(report, args.out)


def _dump(report: dict, out: str) -> None:
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(json.dumps(report, indent=2, default=str))
    print(f"\n[probe] written to {out}")


if __name__ == "__main__":
    main()
