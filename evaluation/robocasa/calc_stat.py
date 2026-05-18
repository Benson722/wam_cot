# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
"""Aggregate Baseline vs WAM-CoT (+ ablations) into the report deliverables.

Produces (PDF 成果产出: 对比实验结果 / 消融对比图表 / 推理时间与计算开销统计):
  - comparison.csv          per (run,task): SR, sub-task progress, avg steps,
                            failure-tag counts, VLM cost
  - sr_comparison.png       grouped success-rate bar chart (baseline vs runs)
  - failure_breakdown.png   stacked failure-type bars per run
  - report.md               a ready-to-paste markdown summary table

Each run dir is one invocation of client.py / client_cot.py (it holds
`<task>.json`, `summary.json`, per-episode `*.mp4` / `*.plan.json`,
`vlm_calls.jsonl`). Episode video filenames encode ground truth as
`<ep>_<True|False>_<failuretag>.mp4`, used to cross-check the json counters.

Usage:
  python evaluation/robocasa/calc_stat.py \
     --runs baseline=outputs/robocasa/baseline \
            cot=outputs/robocasa/cot \
            cot_no_monitor=outputs/robocasa/abl_no_monitor \
            cot_shuffle=outputs/robocasa/abl_shuffle \
            cot_hard_reset=outputs/robocasa/abl_hard_reset \
     --out outputs/robocasa/report
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple


def _scan_videos(task_dir: Path) -> Tuple[int, int, Dict[str, int]]:
    """(succ, total, tag_counts) recounted from `*.mp4` filenames."""
    succ = total = 0
    tags: Dict[str, int] = defaultdict(int)
    if not task_dir.is_dir():
        return 0, 0, {}
    for f in task_dir.glob("*.mp4"):
        parts = f.stem.split("_")
        if len(parts) < 2:
            continue
        is_succ = parts[1] == "True"
        tag = "_".join(parts[2:]) if len(parts) > 2 else (
            "success" if is_succ else "unknown")
        total += 1
        succ += int(is_succ)
        tags[tag] += 1
    return succ, total, dict(tags)


def _load_run(run_dir: Path) -> Dict[str, dict]:
    """task -> merged record (prefers <task>.json counters; falls back to
    recounting videos)."""
    out: Dict[str, dict] = {}
    if not run_dir.is_dir():
        print(f"[stat] WARNING: run dir missing: {run_dir}")
        return out
    for jf in run_dir.glob("*.json"):
        if jf.name in ("summary.json",):
            continue
        try:
            rec = json.loads(jf.read_text())
        except Exception:
            continue
        task = rec.get("task", jf.stem)
        v_s, v_t, v_tags = _scan_videos(run_dir / task)
        rec["video_succ"] = v_s
        rec["video_total"] = v_t
        if v_tags:
            rec["failure_tags"] = v_tags
        out[task] = rec
    return out


def _vlm_cost(run_dir: Path) -> Dict[str, float]:
    log = run_dir / "vlm_calls.jsonl"
    if not log.exists():
        return {}
    calls = lat = pt = ct = 0
    for line in log.read_text().splitlines():
        try:
            r = json.loads(line)
        except Exception:
            continue
        calls += 1
        lat += float(r.get("latency_ms", 0))
        u = r.get("usage", {}) or {}
        pt += int(u.get("prompt_tokens", 0))
        ct += int(u.get("completion_tokens", 0))
    return {
        "vlm_calls": calls,
        "vlm_total_latency_ms": round(lat, 1),
        "vlm_avg_latency_ms": round(lat / max(calls, 1), 1),
        "vlm_prompt_tokens": pt,
        "vlm_completion_tokens": ct,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True,
                    help="label=dir pairs (first one is the baseline)")
    ap.add_argument("--out", default="outputs/robocasa/report")
    args = ap.parse_args()

    runs: List[Tuple[str, Path]] = []
    for spec in args.runs:
        label, _, d = spec.partition("=")
        runs.append((label, Path(d)))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    data = {label: _load_run(d) for label, d in runs}
    cost = {label: _vlm_cost(d) for label, d in runs}
    all_tasks = sorted({t for r in data.values() for t in r})

    # ---- comparison.csv ----
    csv_path = out / "comparison.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["run", "task", "succ", "total", "succ_rate",
                    "subtask_progress_rate", "avg_steps",
                    "failure_tags", "vlm_calls", "vlm_avg_latency_ms",
                    "vlm_prompt_tokens", "vlm_completion_tokens"])
        for label, _ in runs:
            for task in all_tasks:
                rec = data[label].get(task, {})
                if not rec:
                    continue
                c = cost.get(label, {})
                w.writerow([
                    label, task,
                    rec.get("succ_num", rec.get("video_succ", 0)),
                    rec.get("total_num", rec.get("video_total", 0)),
                    round(rec.get("succ_rate", 0), 4),
                    round(rec.get("subtask_progress_rate", float("nan")), 4)
                    if "subtask_progress_rate" in rec else "",
                    round(rec.get("avg_steps", 0), 1),
                    json.dumps(rec.get("failure_tags", {})),
                    c.get("vlm_calls", ""), c.get("vlm_avg_latency_ms", ""),
                    c.get("vlm_prompt_tokens", ""),
                    c.get("vlm_completion_tokens", ""),
                ])
    print(f"[stat] wrote {csv_path}")

    # ---- charts ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        labels = [l for l, _ in runs]
        x = np.arange(len(all_tasks))
        bw = 0.8 / max(len(labels), 1)
        fig, ax = plt.subplots(figsize=(max(8, 1.6 * len(all_tasks)), 5))
        for i, label in enumerate(labels):
            srs = [data[label].get(t, {}).get("succ_rate", 0)
                   for t in all_tasks]
            ax.bar(x + i * bw, srs, bw, label=label)
        ax.set_xticks(x + bw * (len(labels) - 1) / 2)
        ax.set_xticklabels(all_tasks, rotation=20, ha="right")
        ax.set_ylabel("Success Rate")
        ax.set_ylim(0, 1)
        ax.set_title("Baseline vs WAM-CoT vs ablations")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out / "sr_comparison.png", dpi=140)
        plt.close(fig)

        # failure breakdown (aggregated over tasks)
        tagset = sorted({tg for label in labels for t in all_tasks
                         for tg in data[label].get(t, {})
                         .get("failure_tags", {})})
        fig2, ax2 = plt.subplots(figsize=(max(7, 1.2 * len(labels)), 5))
        bottoms = np.zeros(len(labels))
        for tg in tagset:
            vals = []
            for label in labels:
                vals.append(sum(
                    data[label].get(t, {}).get("failure_tags", {}).get(tg, 0)
                    for t in all_tasks))
            ax2.bar(labels, vals, bottom=bottoms, label=tg)
            bottoms += np.array(vals)
        ax2.set_ylabel("episodes")
        ax2.set_title("Failure-type breakdown")
        ax2.legend(fontsize=8)
        fig2.tight_layout()
        fig2.savefig(out / "failure_breakdown.png", dpi=140)
        plt.close(fig2)
        print(f"[stat] wrote {out/'sr_comparison.png'}, "
              f"{out/'failure_breakdown.png'}")
    except Exception as e:  # noqa: BLE001
        print(f"[stat] chart skipped ({e})")

    # ---- report.md ----
    lines = ["# Robocasa: Baseline vs WAM-CoT\n",
             "| run | task | SR | sub-task progress | avg steps | VLM calls |",
             "|---|---|---|---|---|---|"]
    for label, _ in runs:
        for task in all_tasks:
            rec = data[label].get(task)
            if not rec:
                continue
            lines.append(
                f"| {label} | {task} | "
                f"{rec.get('succ_rate', 0):.3f} | "
                f"{rec.get('subtask_progress_rate', float('nan')):.3f} | "
                f"{rec.get('avg_steps', 0):.0f} | "
                f"{cost.get(label, {}).get('vlm_calls', 0)} |")
    lines.append("\n## VLM cost (推理时间与计算开销)\n")
    for label, _ in runs:
        c = cost.get(label, {})
        if c:
            lines.append(f"- **{label}**: {c}")
    (out / "report.md").write_text("\n".join(lines))
    print(f"[stat] wrote {out/'report.md'}")


if __name__ == "__main__":
    main()
