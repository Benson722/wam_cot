# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
"""Offline loss-curve plotter (no internet / no W&B account needed).

The training tqdm postfix already records every logged step, e.g.:
  Training: ... latent_loss=0.1597, action_loss=0.0009, kf_loss=0.0027,
            step=877, grad_norm=0.10, lr=1.00e-05
This scans a captured stdout/torchrun log, extracts those scalars, and
writes a CSV + a PNG (latent / action / kf loss + grad_norm) for the report.
Works fully offline from data already on disk.

Run (lingbot venv, repo root):
  python evaluation/robotwin/plot_losses.py \
      --log train_out/torchrun_logs/<run>/attempt_0/0/stdout.log \
      --out experiments/loss_curves
  # or point --log at any file containing the training console output
"""
from __future__ import annotations

import argparse
import csv
import glob
import os
import re
from pathlib import Path

# step=NN with the four scalars (order-independent, tolerant of spacing).
_PAT = re.compile(
    r"latent_loss=([0-9.eE+-]+).*?"
    r"action_loss=([0-9.eE+-]+).*?"
    r"kf_loss=([0-9.eE+-]+).*?"
    r"step=(\d+).*?"
    r"grad_norm=([0-9.eE+-]+)")


def _latest_stdout() -> str | None:
    cands = sorted(
        glob.glob("train_out/torchrun_logs/*/attempt_*/0/stdout.log"),
        key=os.path.getmtime)
    return cands[-1] if cands else None


def parse(log_path: str):
    rows = {}  # step -> (latent, action, kf, grad)
    with open(log_path, "r", errors="ignore") as f:
        text = f.read()
    for m in _PAT.finditer(text):
        lat, act, kf, step, gn = m.groups()
        rows[int(step)] = (float(lat), float(act), float(kf), float(gn))
    return [(s, *rows[s]) for s in sorted(rows)]


def main():
    ap = argparse.ArgumentParser(description="Offline training loss plotter")
    ap.add_argument("--log", default=None,
                    help="training stdout/torchrun log (default: latest "
                         "train_out/torchrun_logs/*/attempt_*/0/stdout.log)")
    ap.add_argument("--out", default="experiments/loss_curves")
    ap.add_argument("--smooth", type=int, default=20,
                    help="moving-average window for the smoothed overlay")
    args = ap.parse_args()

    log_path = args.log or _latest_stdout()
    if not log_path or not os.path.isfile(log_path):
        raise SystemExit(
            f"log not found: {log_path!r}. Pass --log <file> (a captured "
            "training console / torchrun stdout.log).")
    data = parse(log_path)
    if not data:
        raise SystemExit(
            f"No 'latent_loss=... step=... ' lines parsed from {log_path}. "
            "Point --log at the file with the training progress output.")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    csv_fp = out / "loss_curves.csv"
    with open(csv_fp, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["step", "latent_loss", "action_loss", "kf_loss",
                    "grad_norm"])
        w.writerows(data)
    print(f"[plot] {len(data)} steps -> {csv_fp}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except Exception as e:  # noqa: BLE001
        print(f"[plot] matplotlib/numpy unavailable ({e}); CSV written only.")
        return

    a = np.array(data, dtype=float)
    steps = a[:, 0]
    names = ["latent_loss", "action_loss", "kf_loss", "grad_norm"]

    def smooth(y):
        k = max(1, int(args.smooth))
        if len(y) < k:
            return y
        c = np.convolve(y, np.ones(k) / k, mode="valid")
        return np.concatenate([np.full(k - 1, c[0]), c])

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    for ax, j, nm in zip(axes.flat, range(1, 5), names):
        ax.plot(steps, a[:, j], lw=0.6, alpha=0.35, label="raw")
        ax.plot(steps, smooth(a[:, j]), lw=1.8,
                label=f"MA{args.smooth}")
        ax.set_title(nm)
        ax.set_xlabel("step")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
        if nm in ("action_loss", "kf_loss"):
            ax.set_yscale("log")  # tiny, fast-converging -> log y is clearer
    fig.suptitle(f"Training losses ({len(data)} logged steps) — {log_path}")
    fig.tight_layout()
    png = out / "loss_curves.png"
    fig.savefig(png, dpi=140)
    plt.close(fig)
    print(f"[plot] wrote {png}")
    print("[plot] summary (last logged step):")
    s, lat, act, kf, gn = data[-1]
    print(f"[plot]   step={s} latent={lat:.4f} action={act:.4f} "
          f"kf={kf:.4f} grad_norm={gn:.3f}")


if __name__ == "__main__":
    main()
