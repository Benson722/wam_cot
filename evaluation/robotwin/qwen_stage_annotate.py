# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
"""Phase A — VLM-assisted offline STAGE annotation (Latent-CoT, VLM in the
data-generation loop).

For each episode: uniformly sample K frames from the cam_high video, send
them (multimodal) to the local Qwen3.5-27B served by serve_qwen.py
(OpenAI-compatible :8000/v1; offline, no internet), and ask for the ORDERED
manipulation STAGES with the frame index where each starts. Write
``<dataset>/meta/stages.jsonl`` (parallel to keyframes.jsonl) — this is a
RICHER, SEMANTIC supervision signal than the gripper-switch keyframe
distance, used by the training stage-classification head (Phase B).

Reuses cot_planner's vLLM endpoint constants + image encoder + JSON parser
(no new transport/credential code). Runs where serve_qwen is reachable
(H200 GPU0). ``--recursive`` annotates every task dir under a parent
(curated *_stable symlink set) for multi-task training.

Run (env with serve_qwen reachable, repo root):
  python evaluation/robotwin/qwen_stage_annotate.py \
      --dataset /inspire/.../lerobot_robotwin_eef_aug_500_stable \
      --recursive --frames 6
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from evaluation.robocasa.cot_planner import (  # noqa: E402
    HARDCODED_VLLM_BASE_URL,
    HARDCODED_VLLM_MODEL,
    CoTPlanner,
    _cred,
    _img_to_data_url,
)

_SYS = """You segment a single-arm/dual-arm robot manipulation episode into \
ORDERED semantic STAGES from sampled video frames. A stage is a coherent \
sub-phase of the task (e.g. "approach object", "grasp", "lift / move", \
"place", "retract"). Use the visible motion + the task instruction. The 1st \
stage starts at frame 0. Give 2-6 stages, strictly increasing start_frame in
[0, length). Respond with STRICT JSON only, no markdown:
{"stages":[{"name":"<short>","start_frame":<int>}, ...]}"""


def _episodes(meta_dir: Path, episodes_file: str):
    fp = meta_dir / episodes_file
    if not fp.exists():
        raise FileNotFoundError(fp)
    out = []
    with open(fp, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def _video_path(dataset: Path, ei: int, cam: str, chunks_size: int = 1000):
    chunk = ei // chunks_size
    return (dataset / "videos" / f"chunk-{chunk:03d}" / cam
            / f"episode_{ei:06d}.mp4")


def _read_frames(mp4: Path, idxs):
    """Decode given frame indices. Try imageio(ffmpeg) then pyav."""
    idxs = sorted(set(int(i) for i in idxs))
    try:
        import imageio.v2 as iio
        rdr = iio.get_reader(str(mp4), "ffmpeg")
        out = []
        for i in idxs:
            try:
                out.append(np.asarray(rdr.get_data(i)))
            except Exception:  # noqa: BLE001
                out.append(np.asarray(rdr.get_data(0)))
        rdr.close()
        return out
    except Exception:  # noqa: BLE001
        pass
    try:
        import imageio.v3 as iio3
        frames = iio3.imread(str(mp4), plugin="pyav")  # [T,H,W,C]
        return [np.asarray(frames[min(i, len(frames) - 1)]) for i in idxs]
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(
            f"cannot decode {mp4} ({e}); install imageio-ffmpeg or av "
            "(av1 support).")


def _vlm_stage_call(base_url, model, api_key, task, length, idxs, frames,
                    timeout=120, retries=4):
    content = [{"type": "text", "text":
                f"TASK: {task}\nEPISODE LENGTH (frames): {length}\n"
                f"The {len(frames)} images are frames at indices "
                f"{list(idxs)} (chronological). Output the ordered stages "
                "with integer start_frame."}]
    for fr in frames:
        content.append({"type": "image_url",
                        "image_url": {"url": _img_to_data_url(fr)}})
    payload = {"model": model, "temperature": 0.1, "max_tokens": 512,
               "stream": False,
               "messages": [{"role": "system", "content": _SYS},
                            {"role": "user", "content": content}]}
    base = base_url.rstrip("/")
    url = base + ("/chat/completions" if base.endswith("/v1")
                  else "/v1/chat/completions")
    data = json.dumps(payload).encode()
    last = None
    for a in range(retries):
        try:
            req = urllib.request.Request(
                url, data=data, method="POST",
                headers={"Content-Type": "application/json",
                         "Authorization": f"Bearer {api_key}"})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                resp = json.loads(r.read().decode())
            return resp["choices"][0]["message"]["content"]
        except (urllib.error.URLError, KeyError, TimeoutError,
                json.JSONDecodeError) as e:  # noqa: PERF203
            last = e
            time.sleep(min(2 ** a, 8))
    raise RuntimeError(f"VLM call failed after {retries}: {last}")


def _norm_stages(raw, length):
    obj = CoTPlanner._parse_json(raw)
    st = obj.get("stages") or []
    out = []
    for s in st:
        if isinstance(s, dict):
            nm = str(s.get("name", "stage")).strip()[:40]
            sf = int(round(float(s.get("start_frame", 0))))
        else:
            nm, sf = str(s)[:40], 0
        out.append({"name": nm, "start_frame": max(0, min(sf, length - 1))})
    out.sort(key=lambda x: x["start_frame"])
    # dedup / enforce strictly increasing, first stage at 0
    fixed, prev = [], -1
    for s in out:
        if s["start_frame"] <= prev:
            continue
        fixed.append(s)
        prev = s["start_frame"]
    if not fixed or fixed[0]["start_frame"] != 0:
        fixed = [{"name": "start", "start_frame": 0}] + [
            s for s in fixed if s["start_frame"] > 0]
    return fixed[:6]


def annotate(dataset: Path, cam: str, frames_k: int, episodes_file: str,
             out_name: str, base_url, model, api_key):
    meta = dataset / "meta"
    eps = _episodes(meta, episodes_file)
    out_fp = meta / out_name
    ok = skip = nstage = 0
    with open(out_fp, "w", encoding="utf-8") as fo:
        for ep in eps:
            ei = int(ep["episode_index"])
            L = int(ep.get("length", 0))
            task = (ep.get("tasks") or [""])[0]
            mp4 = _video_path(dataset, ei, cam)
            if L < 2 or not mp4.exists():
                skip += 1
                continue
            idxs = np.linspace(0, L - 1, num=min(frames_k, L),
                               dtype=int).tolist()
            try:
                frs = _read_frames(mp4, idxs)
                raw = _vlm_stage_call(base_url, model, api_key, task, L,
                                      idxs, frs)
                stages = _norm_stages(raw, L)
            except Exception as e:  # noqa: BLE001
                print(f"[stage] ep{ei} SKIP: {e}")
                skip += 1
                continue
            fo.write(json.dumps({"episode_index": ei, "length": L,
                                 "tasks": ep.get("tasks"),
                                 "stages": stages}) + "\n")
            fo.flush()
            ok += 1
            nstage += len(stages)
            if ok % 20 == 0:
                print(f"[stage] {ok} done (avg_stages="
                      f"{nstage / max(ok,1):.2f})")
    print(f"[stage] {dataset.name}: ok={ok} skipped={skip} "
          f"avg_stages={nstage / max(ok,1):.2f} -> {out_fp}")


def main():
    ap = argparse.ArgumentParser(description="VLM offline stage annotation")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--cam-key", default="observation.images.cam_high")
    ap.add_argument("--frames", type=int, default=6,
                    help="frames sampled per episode for the VLM")
    ap.add_argument("--episodes-file", default="episodes.jsonl")
    ap.add_argument("--out-name", default="stages.jsonl")
    ap.add_argument("--recursive", action="store_true",
                    help="annotate every task dir with meta/<episodes-file> "
                         "under --dataset (multi-task, follows symlinks)")
    ap.add_argument("--base-url",
                    default=_cred("VLLM_BASE_URL", HARDCODED_VLLM_BASE_URL))
    ap.add_argument("--model",
                    default=_cred("VLLM_MODEL", HARDCODED_VLLM_MODEL))
    ap.add_argument("--api-key", default=_cred("VLLM_API_KEY", "EMPTY"))
    args = ap.parse_args()

    root = Path(args.dataset)
    if args.recursive:
        targets = sorted({
            Path(dp) for dp, _d, _f in os.walk(root, followlinks=True)
            if (Path(dp) / "meta" / args.episodes_file).exists()})
        if not targets:
            raise SystemExit(f"--recursive: no task dir under {root}")
        print(f"[stage] recursive: {len(targets)} task dirs")
        for i, t in enumerate(targets):
            print(f"[stage] ({i+1}/{len(targets)}) {t}")
            try:
                annotate(t, args.cam_key, args.frames, args.episodes_file,
                         args.out_name, args.base_url, args.model,
                         args.api_key)
            except Exception as e:  # noqa: BLE001
                print(f"[stage] SKIP {t}: {e}")
    else:
        annotate(root, args.cam_key, args.frames, args.episodes_file,
                 args.out_name, args.base_url, args.model, args.api_key)


if __name__ == "__main__":
    main()
