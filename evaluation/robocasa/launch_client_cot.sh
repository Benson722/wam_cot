#!/usr/bin/env bash
# Phase-2 WAM-CoT (Route-1, Semantic CoT) rollout on Robocasa.
# Run in the *robocasa* conda env on the internet-capable 4090 (DeepSeek API).
set -e

# Planner backend:
#   PLANNER=deepseek (default) -> text-only DeepSeek V4 Pro, reasons over the
#                                 symbolic scene-text (no pixels).
#   PLANNER=vllm               -> local Qwen3.5-27B multimodal via vLLM
#                                 (start the vLLM server first; see COT_DESIGN.md).
# All creds/endpoints are hardcoded in cot_planner.py (HARDCODED_DEEPSEEK_* /
# HARDCODED_VLLM_*). No `export` needed; env vars (DEEPSEEK_*/VLLM_*) still win.
PLANNER=${PLANNER:-deepseek}

PORT=${PORT:-29056}
TEST_NUM=${TEST_NUM:-25}
OUT_DIR=${OUT_DIR:-outputs/robocasa/cot}
ABLATION=${ABLATION:-none}
TASKS=${TASKS:-"PickPlaceCounterToCabinet PickPlaceCounterToMicrowave OpenDrawer"}
ENV_OVERRIDES=${ENV_OVERRIDES:-}

ARGS=(--tasks $TASKS --port "$PORT" --test-num "$TEST_NUM"
      --out-dir "$OUT_DIR" --ablation "$ABLATION" --planner "$PLANNER")
[ -n "$VLM_MODEL" ]    && ARGS+=(--vlm-model "$VLM_MODEL")
[ -n "$VLM_BASE_URL" ] && ARGS+=(--vlm-base-url "$VLM_BASE_URL")
[ -n "$ENV_OVERRIDES" ] && ARGS+=(--env-overrides "$ENV_OVERRIDES")
[ -n "$VLM_TEXT_ONLY" ] && ARGS+=(--vlm-text-only)

python evaluation/robocasa/client_cot.py "${ARGS[@]}"
