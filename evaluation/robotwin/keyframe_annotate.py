# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
"""Offline keyframe annotation for Latent-CoT component #1 (see latent_plan.md).

Scans a LeRobot-format training set and, for every episode, extracts the
**keyframe raw-frame indices** the implicit-physical-CoT aux head will regress
distances to. Default (and most robust, per the plan) keyframe source is the
**gripper open/close transition** — purely from the recorded action vector, no
VLM/LLM, strongly aligned with physical manipulation events. Optionally also
adds `action_config` segment boundaries (deepseek/stage-change style) as
keyframes.

Output: ``<dataset>/meta/keyframes.jsonl``, one JSON object per episode:

    {"episode_index": 0, "length": 450,
     "keyframes": [37, 88, 210, 449],
     "types": ["grasp", "release", "grasp", "stage"]}

This file is consumed lazily and **backward-compatibly** by
``wan_va/dataset/lerobot_latent_dataset.py`` (only when ``cfg.kf_aux`` is set);
if absent, training is unaffected.

Run (RoboTwin / lingbot env, from the LingBot repo root):

    python evaluation/robotwin/keyframe_annotate.py \
        --dataset /path/to/robotwin-clean-and-aug-lerobot \
        --gripper-idx 7 15 --with-stage-boundaries
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np


def _load_episodes_meta(meta_dir: Path) -> List[dict]:
    """Read meta/episodes.jsonl (LeRobot v2). Each line: episode_index,
    length, tasks, action_config[{start_frame,end_frame,...}]."""
    fp = meta_dir / "episodes.jsonl"
    if not fp.exists():
        raise FileNotFoundError(f"{fp} not found (expected LeRobot meta).")
    out = []
    with open(fp, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def _episode_chunk(episode_index: int, chunks_size: int = 1000) -> int:
    return episode_index // chunks_size


def _load_episode_actions(dataset: Path, episode_index: int,
                          chunks_size: int = 1000) -> Optional[np.ndarray]:
    """Load the (T, action_dim) action array for one episode from the
    LeRobot parquet. Returns None if unreadable (caller skips)."""
    try:
        import pandas as pd  # noqa: PLC0415
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(
            "keyframe_annotate needs pandas+pyarrow to read LeRobot parquet: "
            f"{e}")
    chunk = _episode_chunk(episode_index, chunks_size)
    cand = [
        dataset / "data" / f"chunk-{chunk:03d}"
        / f"episode_{episode_index:06d}.parquet",
        dataset / "data" / f"chunk-{chunk:03d}"
        / f"episode_{episode_index:06d}.parquet",
    ]
    fp = next((c for c in cand if c.exists()), None)
    if fp is None:
        return None
    df = pd.read_parquet(fp, columns=["action"])
    arr = np.stack([np.asarray(a, dtype=np.float64) for a in df["action"]])
    return arr  # (T, action_dim)


def _binary_gripper(g: np.ndarray) -> np.ndarray:
    """Binarize a 1-D gripper channel into open(0)/closed(1).

    Uses the midpoint of the channel's own observed range as the threshold,
    which is robust to differing gripper conventions/scales across embodiments
    (RoboTwin aloha-agilex vs others)."""
    g = np.asarray(g, dtype=np.float64).reshape(-1)
    lo, hi = float(np.min(g)), float(np.max(g))
    if hi - lo < 1e-6:
        return np.zeros_like(g, dtype=np.int64)  # never moves -> no switch
    thr = 0.5 * (lo + hi)
    # Convention-agnostic: we only care about *transitions*, not which side is
    # "closed". Pick the labeling so the first frame is state 0.
    b = (g > thr).astype(np.int64)
    if b[0] == 1:
        b = 1 - b
    return b


def _transitions(b: np.ndarray) -> List[int]:
    """Frame indices where the binary state changes (the change is recorded
    at the first frame of the new state)."""
    if b.size < 2:
        return []
    chg = np.nonzero(b[1:] != b[:-1])[0] + 1
    return chg.tolist()


def extract_keyframes(
    actions: np.ndarray,
    gripper_idx: Tuple[int, ...],
    stage_boundaries: Optional[List[int]] = None,
) -> Tuple[List[int], List[str]]:
    """Return (sorted unique keyframe frame indices, parallel type labels).

    Keyframe sources:
      - gripper transition for EACH configured gripper channel
        ('grasp' when entering the closed-ish state, else 'release')
      - optional stage/segment boundaries ('stage')
    """
    T = actions.shape[0]
    kf: dict = {}  # frame -> type (later keyframe at same frame keeps first)
    for gi in gripper_idx:
        if gi < 0 or gi >= actions.shape[1]:
            continue
        b = _binary_gripper(actions[:, gi])
        for t in _transitions(b):
            # entering state 1 = "closing" (grasp), entering 0 = "release"
            kf.setdefault(int(t), "grasp" if b[t] == 1 else "release")
    for sb in (stage_boundaries or []):
        if 0 < sb < T:
            kf.setdefault(int(sb), "stage")
    if not kf:
        return [], []
    frames = sorted(kf.keys())
    return frames, [kf[f] for f in frames]


def annotate_dataset(
    dataset: Path,
    gripper_idx: Tuple[int, ...],
    with_stage_boundaries: bool,
    out_name: str = "keyframes.jsonl",
) -> Path:
    meta_dir = dataset / "meta"
    episodes = _load_episodes_meta(meta_dir)
    out_fp = meta_dir / out_name
    n_ok, n_skip, n_kf = 0, 0, 0
    with open(out_fp, "w", encoding="utf-8") as fout:
        for ep in episodes:
            ei = int(ep["episode_index"])
            length = int(ep.get("length", 0))
            actions = _load_episode_actions(dataset, ei)
            if actions is None or actions.shape[0] < 2:
                n_skip += 1
                continue
            stage_b = None
            if with_stage_boundaries:
                stage_b = sorted({
                    int(ac["start_frame"])
                    for ac in ep.get("action_config", [])
                    if int(ac.get("start_frame", 0)) > 0
                })
            frames, types = extract_keyframes(actions, gripper_idx, stage_b)
            # Always treat the final frame as a terminal keyframe so every
            # timestep has a well-defined "distance to next keyframe".
            if not frames or frames[-1] != actions.shape[0] - 1:
                frames.append(int(actions.shape[0] - 1))
                types.append("end")
            fout.write(json.dumps({
                "episode_index": ei,
                "length": length or int(actions.shape[0]),
                "keyframes": frames,
                "types": types,
            }) + "\n")
            n_ok += 1
            n_kf += len(frames)
    print(f"[keyframe_annotate] episodes ok={n_ok} skipped={n_skip} "
          f"avg_keyframes={n_kf / max(n_ok, 1):.2f}")
    print(f"[keyframe_annotate] wrote {out_fp}")
    return out_fp


def main():
    ap = argparse.ArgumentParser(description="Offline keyframe annotation "
                                 "(Latent-CoT #1).")
    ap.add_argument("--dataset", required=True,
                    help="LeRobot dataset root (has meta/ and data/).")
    ap.add_argument("--gripper-idx", type=int, nargs="+", default=[7, 15],
                    help="Gripper channel indices in the raw action vector "
                         "(RoboTwin aloha-agilex default: 7=left, 15=right).")
    ap.add_argument("--with-stage-boundaries", action="store_true",
                    help="Also add action_config segment starts as keyframes.")
    ap.add_argument("--out-name", default="keyframes.jsonl")
    args = ap.parse_args()
    annotate_dataset(Path(args.dataset), tuple(args.gripper_idx),
                     args.with_stage_boundaries, args.out_name)


if __name__ == "__main__":
    main()
