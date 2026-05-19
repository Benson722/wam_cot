#!/bin/bash
# RoboTwin WAM-CoT (PDF Phase-2, Route-1 External Semantic CoT).
#
# Same harness as launch_client.sh (native RoboTwin client) but with the
# high-level VLM planner enabled (`--cot True`). Default planner = the local
# Qwen3.5-27B served by serve_qwen.py (OpenAI-compatible :8000/v1, multimodal).
#
# Run in the RoboTwin conda env, from the LingBot repo root, AFTER:
#   - the LingBot server is up:        bash evaluation/robotwin/launch_server.sh
#   - the Qwen VLM server is up:       python /inspire/qb-ilm2/project/26summer-camp-11/serve_qwen.py
export LD_LIBRARY_PATH=/usr/lib64:/usr/lib:$LD_LIBRARY_PATH

save_root=${1:-'./results_cot'}
task_name=${2:-"adjust_bottle"}

policy_name=ACT
task_config=demo_clean
train_config_name=0
model_name=0
seed=0
PORT=${PORT:-29056}

# --- WAM-CoT knobs ---------------------------------------------------------
PLANNER=${PLANNER:-vllm}                       # vllm (local Qwen) | deepseek
VLM_BASE_URL=${VLM_BASE_URL:-http://127.0.0.1:8000/v1}
VLM_MODEL=${VLM_MODEL:-Qwen3.5-27B}
MONITOR_EVERY=${MONITOR_EVERY:-2}              # VLM monitor cadence (chunks)
COT_ABLATION=${COT_ABLATION:-none}             # none|no_cot|no_monitor|
                                               # shuffle_subtasks|hard_reset
TEST_NUM=${TEST_NUM:-100}

PYTHONWARNINGS=ignore::UserWarning \
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 python -m evaluation.robotwin.eval_polict_client_openpi --config policy/$policy_name/deploy_policy.yml \
    --overrides \
    --task_name ${task_name} \
    --task_config ${task_config} \
    --train_config_name ${train_config_name} \
    --model_name ${model_name} \
    --ckpt_setting ${model_name} \
    --seed ${seed} \
    --policy_name ${policy_name} \
    --save_root ${save_root} \
    --video_guidance_scale 5 \
    --action_guidance_scale 1 \
    --test_num ${TEST_NUM} \
    --port ${PORT} \
    --cot True \
    --planner ${PLANNER} \
    --vlm_base_url ${VLM_BASE_URL} \
    --vlm_model ${VLM_MODEL} \
    --monitor_every ${MONITOR_EVERY} \
    --cot_ablation ${COT_ABLATION}
