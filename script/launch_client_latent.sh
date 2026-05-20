#!/bin/bash
# Latent-eval client(含 dream_video 消费),针对本项目两个 Latent-CoT ckpt。
# 完全沿用 eval_env 下 launch_client_latent.sh 的形式,加 TAG + TASK + TEST_NUM 参数。
#
# 用法(4090 实例;先在另一终端跑对应 TAG 的 server):
#   TAG=M1  TASK=hanging_mug                  bash script/launch_client_latent.sh
#   TAG=M1v TASK=lift_pot     TEST_NUM=20     bash script/launch_client_latent.sh
#   TAG=M1v TASK=handover_block PORT=29066    bash script/launch_client_latent.sh
#
# 6 个任务循环示例(脚本会逐个跑,共用同一 server):
#   for t in handover_block handover_mic hanging_mug blocks_ranking_size beat_block_hammer lift_pot; do
#     TAG=M1 TASK=$t TEST_NUM=10 bash script/launch_client_latent.sh
#   done

TAG=${TAG:-M1}
TASK=${TASK:-adjust_bottle}
TEST_NUM=${TEST_NUM:-10}
PORT=${PORT:-29056}

EVAL_ENV=${EVAL_ENV:-/inspire/qb-ilm2/project/26summer-camp-11/public/group3/eval_env/sii_wam_cot/lingbot-va_goal_cond_cot}

# 预检
case $TAG in M1|M1v) ;; *) echo "[client] TAG 必须是 M1 或 M1v (当前: $TAG)"; exit 1 ;; esac
[ -d "$EVAL_ENV/evaluation/robotwin" ] || { echo "[client] 缺 $EVAL_ENV/evaluation/robotwin"; exit 1; }
[ -f "$EVAL_ENV/evaluation/robotwin/eval_polict_client_openpi_latent.py" ] \
  || { echo "[client] 缺 eval_polict_client_openpi_latent.py"; exit 1; }

# RoboTwin sapien 需要系统 libs
export LD_LIBRARY_PATH=/usr/lib64:/usr/lib:$LD_LIBRARY_PATH

# DeepSeek key (planner 只用来记 subgoals,可空)
export DEEPSEEK_API_KEY=${DEEPSEEK_API_KEY:-}

policy_name=ACT
task_config=demo_clean
train_config_name=0
model_name=0
seed=0

cd "$EVAL_ENV"
echo "[client] TAG=$TAG TASK=$TASK TEST_NUM=$TEST_NUM PORT=$PORT"

PYTHONWARNINGS=ignore::UserWarning \
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
exec python -m evaluation.robotwin.eval_polict_client_openpi_latent \
    --config policy/$policy_name/deploy_policy.yml \
    --overrides \
    --task_name ${TASK} \
    --task_config ${task_config} \
    --train_config_name ${train_config_name} \
    --model_name ${model_name} \
    --ckpt_setting ${TAG} \
    --seed ${seed} \
    --policy_name ${policy_name} \
    --save_root "./results_latent_${TAG}" \
    --outputs_root "./outputs_latent_${TAG}" \
    --video_guidance_scale 5 \
    --action_guidance_scale 1 \
    --test_num ${TEST_NUM} \
    --port ${PORT}
