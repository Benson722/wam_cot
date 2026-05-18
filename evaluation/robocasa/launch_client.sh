#!/usr/bin/env bash
# Phase-1 BASELINE WAM rollout on Robocasa.
# Run in the *robocasa* conda env (server must already be up).
set -e

PORT=${PORT:-29056}
TEST_NUM=${TEST_NUM:-25}
OUT_DIR=${OUT_DIR:-outputs/robocasa/baseline}
TASKS=${TASKS:-"PickPlaceCounterToCabinet PickPlaceCounterToMicrowave OpenDrawer"}
# After running probe_env.py, pass corrections here, e.g.:
#   ENV_OVERRIDES='{"camera_map":{"robot0_agentview_center":"observation.images.agentview_rgb","robot0_eye_in_hand":"observation.images.eye_in_hand_rgb"},"arm_action_slice":[0,6],"gripper_index":6}'
ENV_OVERRIDES=${ENV_OVERRIDES:-}

ARGS=(--tasks $TASKS --port "$PORT" --test-num "$TEST_NUM" --out-dir "$OUT_DIR")
[ -n "$ENV_OVERRIDES" ] && ARGS+=(--env-overrides "$ENV_OVERRIDES")

python evaluation/robocasa/client.py "${ARGS[@]}"
