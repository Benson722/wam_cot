#!/usr/bin/env bash
# Phase-2 WAM-CoT (Route-1, Semantic CoT) rollout on Robocasa.
# Run in the *robocasa* conda env on the internet-capable 4090 (DeepSeek API).
set -e

# DeepSeek creds are hardcoded in evaluation/robocasa/cot_planner.py
# (HARDCODED_DEEPSEEK_*). No `export` needed. Env vars, if set, still win.
DEEPSEEK_MODEL=${DEEPSEEK_MODEL:-deepseek-v4-pro}
DEEPSEEK_BASE_URL=${DEEPSEEK_BASE_URL:-https://api.deepseek.com}

PORT=${PORT:-29056}
TEST_NUM=${TEST_NUM:-25}
OUT_DIR=${OUT_DIR:-outputs/robocasa/cot}
ABLATION=${ABLATION:-none}
TASKS=${TASKS:-"PnPCounterToCab PnPCounterToMicrowave OpenDrawer"}
ENV_OVERRIDES=${ENV_OVERRIDES:-}

ARGS=(--tasks $TASKS --port "$PORT" --test-num "$TEST_NUM"
      --out-dir "$OUT_DIR" --ablation "$ABLATION"
      --vlm-model "$DEEPSEEK_MODEL" --vlm-base-url "$DEEPSEEK_BASE_URL")
[ -n "$ENV_OVERRIDES" ] && ARGS+=(--env-overrides "$ENV_OVERRIDES")
[ -n "$VLM_TEXT_ONLY" ] && ARGS+=(--vlm-text-only)

python evaluation/robocasa/client_cot.py "${ARGS[@]}"
