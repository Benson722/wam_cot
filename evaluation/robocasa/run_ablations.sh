#!/usr/bin/env bash
# Full PDF deliverable matrix: Baseline + WAM-CoT + 必做 ablations, then
# aggregate into the comparison report. Server must already be up.
# Run in the *robocasa* conda env on the internet-capable 4090.
set -e

# DeepSeek creds are hardcoded in evaluation/robocasa/cot_planner.py.
PORT=${PORT:-29056}
TEST_NUM=${TEST_NUM:-25}
TASKS=${TASKS:-"PickPlaceCounterToCabinet PickPlaceCounterToMicrowave OpenDrawer"}
ROOT=${ROOT:-outputs/robocasa}
ENV_OVERRIDES=${ENV_OVERRIDES:-}

run_cot () {  # $1=ablation $2=outdir
  OUT_DIR="$2" ABLATION="$1" PORT="$PORT" TEST_NUM="$TEST_NUM" \
    TASKS="$TASKS" ENV_OVERRIDES="$ENV_OVERRIDES" \
    bash evaluation/robocasa/launch_client_cot.sh
}

# 1. Baseline WAM (Phase 1 control group)
OUT_DIR="$ROOT/baseline" PORT="$PORT" TEST_NUM="$TEST_NUM" \
  TASKS="$TASKS" ENV_OVERRIDES="$ENV_OVERRIDES" \
  bash evaluation/robocasa/launch_client.sh

# 2. Full WAM-CoT
run_cot none            "$ROOT/cot"
# 3. Ablations (PDF 消融实验 · 必做)
run_cot no_monitor      "$ROOT/abl_no_monitor"      # 去除观察反馈(开环计划)
run_cot shuffle_subtasks "$ROOT/abl_shuffle"        # 打乱子任务顺序
run_cot blind_planner   "$ROOT/abl_blind_planner"   # CoT 观察模型退化
run_cot hard_reset      "$ROOT/abl_hard_reset"      # 去除软切换/世界模型上下文

# 4. Aggregate -> report deliverables
python evaluation/robocasa/calc_stat.py \
  --runs baseline="$ROOT/baseline" \
         cot="$ROOT/cot" \
         cot_no_monitor="$ROOT/abl_no_monitor" \
         cot_shuffle="$ROOT/abl_shuffle" \
         cot_blind_planner="$ROOT/abl_blind_planner" \
         cot_hard_reset="$ROOT/abl_hard_reset" \
  --out "$ROOT/report"

echo "All runs complete. See $ROOT/report/{comparison.csv,sr_comparison.png,report.md}"
