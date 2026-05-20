#!/bin/bash
# Ablation-3 (错误阶段数据 / 隐式 CoT)
#
# 用与正常 M1v 完全相同的训练管线,但 dataset emit vlm_stage 时对每集做
# deterministic 置换 (vlm_stage_corrupt='shuffle' in
# va_robotwin_train_wrongstage_cfg.py)。然后跑 ① 训练 → ② 探针 → ③ 在线 SR,
# 与原 M1v / M1 / stock 三档对照。
#
# 期望(报告原话):若 stage_loss 收敛到更高 / 探针 val_acc 退化到 stock 附近
# / SR 不优于 stock,则 M1v 的提升来自**正确的** VLM 信号,非"加 aux head"。
#
# 用法:
#   PHASE=train  bash script/run_ablation_implicit.sh   # 训练 (H200, 8 卡, ~1.5h)
#   PHASE=probe  bash script/run_ablation_implicit.sh   # 离线探针 (H200 或 4090)
#   PHASE=eval   bash script/run_ablation_implicit.sh   # 在线 SR (4090, RoboTwin)
#   PHASE=all    bash script/run_ablation_implicit.sh   # 全跑

set -e
PHASE=${PHASE:-all}
REPO=/inspire/hdd/project/26summer-camp-11/26220077/lingbot-va
cd "$REPO"

NGPU_TRAIN=${NGPU_TRAIN:-8}
CUDA_TRAIN=${CUDA_TRAIN:-0,1,2,3,4,5,6,7}
MASTER_PORT_TRAIN=${MASTER_PORT_TRAIN:-29550}

# Wrong-stage ckpt 落到的目录(由 va_robotwin_train_wrongstage_cfg 的
# exp_name='robotwin_kf0.1_vlmstage0.1_WRONG' 决定)
WRONG_CKPT_DIR=$REPO/train_out/checkpoints/robotwin_kf0.1_vlmstage0.1_WRONG
WRONG_CKPT=${WRONG_CKPT:-$WRONG_CKPT_DIR/checkpoint_step_1200}
BS=/inspire/hdd/project/26summer-camp-11/26220077/lingbot-va/checkpoints/lingbot-va-posttrain-robotwin
M1V_CKPT=$REPO/train_out/checkpoints/robotwin_kf0.1_vlmstage0.1/checkpoint_step_1200
M1_CKPT=$REPO/train_out/checkpoints/checkpoint_step_1200

# ============ Phase TRAIN: 用错误标签训 M1v_WRONG ============
if [ "$PHASE" = "train" ] || [ "$PHASE" = "all" ]; then
  echo "##################  PHASE: train (wrongstage)  ##################"
  pkill -9 -f 'wan_va\.train|torch\.distributed\.run' 2>/dev/null || true
  sleep 3
  NGPU=$NGPU_TRAIN CONFIG_NAME=robotwin_train_wrongstage \
  CUDA_VISIBLE_DEVICES=$CUDA_TRAIN MASTER_PORT=$MASTER_PORT_TRAIN \
    bash script/run_va_posttrain.sh
  # 训练循环到 num_steps=50000;按 §7.7 step 1200 即收敛,Ctrl-C 一次安全退出
  # 或等 save_interval=200 累积到 1200 后手动 Ctrl-C。
  echo "[ablation/implicit/train] 完成。期望:stage_loss 收敛到比 M1v 高,"
  echo "  因为标签置乱后只能学到 marginal 分布,无视觉-阶段对应。"
fi

# ============ Phase PROBE: collect_hidden + 线性探针 ============
if [ "$PHASE" = "probe" ] || [ "$PHASE" = "all" ]; then
  echo "##################  PHASE: probe (wrongstage)  ##################"
  if [ ! -d "$WRONG_CKPT/transformer" ]; then
    echo "ERROR: $WRONG_CKPT/transformer 不存在,先跑 PHASE=train"
    exit 1
  fi
  pkill -9 -f 'wan_va\.train|torch\.distributed\.run' 2>/dev/null || true
  sleep 3
  mkdir -p ./train_out/probe

  # 收集 wrongstage ckpt 的 h_t + vlm_stage(注意:loader 仍按 cfg.vlm_stage_corrupt
  # 输出标签,所以这里 dump 的标签也是被腐化的。我们做"真实标签上的探针",
  # 因此 collect 时**临时改用** robotwin_train 配置 (vlm_stage_corrupt='none'),
  # 只是用 --probe-ckpt 替换权重。)
  NGPU=1 CONFIG_NAME=robotwin_train CUDA_VISIBLE_DEVICES=0 MASTER_PORT=29543 \
    bash script/run_va_posttrain.sh \
    --probe-ckpt "$WRONG_CKPT" \
    --probe-collect ./train_out/probe/h_wrongstage.pt \
    --probe-collect-batches 200

  echo "[ablation/implicit/probe] 跑线性探针 (label=vlm_stage, 真实 VLM 标签)"
  python evaluation/robotwin/latent_probe.py --config robotwin_train \
    --features h_hidden --label vlm_stage \
    --hidden-dump ./train_out/probe/h_wrongstage.pt \
    --out-dir ./train_out/probe/out_h_wrongstage

  echo
  echo "============= 探针对照(val_acc, label=vlm_stage, 6 类, chance=0.167) ============="
  for tag in h_stock h_kf h_kfvlm h_wrongstage; do
    fp=./train_out/probe/out_${tag}/results_robotwin_train_h_hidden_vlm_stage.json
    if [ -f "$fp" ]; then
      v=$(python -c "import json; d=json.load(open('$fp')); print(f\"{d['val_acc']:.3f} (train {d['train_acc']:.3f}, gap {d['train_acc']-d['val_acc']:+.3f})\")")
      printf "  %-18s val_acc=%s\n" "$tag" "$v"
    else
      printf "  %-18s (no result)\n" "$tag"
    fi
  done
  echo "期望:wrongstage < kfvlm,接近或低于 stock → 证明 M1v 提升来自正确 VLM 信号。"
fi

# ============ Phase EVAL: 在线 SR (4090, RoboTwin) ============
if [ "$PHASE" = "eval" ] || [ "$PHASE" = "all" ]; then
  echo "##################  PHASE: eval (wrongstage SR on RoboTwin)  ##################"
  if [ ! -d "$WRONG_CKPT/transformer" ]; then
    echo "ERROR: $WRONG_CKPT/transformer 不存在,先跑 PHASE=train"
    exit 1
  fi

  # 补 ckpt 自包含
  for s in vae tokenizer text_encoder; do ln -sfn "$BS/$s" "$WRONG_CKPT/$s"; done

  TEST_NUM=${TEST_NUM:-10}
  TASKS=${TASKS:-"handover_block handover_mic hanging_mug blocks_ranking_size beat_block_hammer lift_pot"}
  START_PORT=${START_PORT:-29086}
  MASTER_PORT_EVAL=${MASTER_PORT_EVAL:-29091}

  pkill -9 -f 'wan_va_server|wan_va_server_predvideo|torch\.distributed\.run|eval_polict_client_openpi' 2>/dev/null || true
  sleep 5

  port_listen () { (exec 3<>/dev/tcp/127.0.0.1/$1) 2>/dev/null && { exec 3<&- 3>&-; return 0; }; return 1; }
  wait_port_up () { for i in $(seq 1 300); do port_listen $1 && return 0
                    kill -0 $SRV 2>/dev/null || return 2; sleep 2; done; return 1; }

  # 用 latent server 与 M1v / M1 / stock 对齐(eval_env 内,见 §12.6)
  EVAL_ENV=/inspire/qb-ilm2/project/26summer-camp-11/public/group3/eval_env/sii_wam_cot/lingbot-va_goal_cond_cot
  EVAL_CFG="$EVAL_ENV/wan_va/configs/va_robotwin_cfg.py"
  [ -f "$EVAL_CFG.bak" ] || cp "$EVAL_CFG" "$EVAL_CFG.bak"
  sed -i "s|^va_robotwin_cfg\.wan22_pretrained_model_name_or_path = .*|va_robotwin_cfg.wan22_pretrained_model_name_or_path = \"$WRONG_CKPT\"|" "$EVAL_CFG"

  cd "$EVAL_ENV"
  CUDA_VISIBLE_DEVICES=0 START_PORT=$START_PORT MASTER_PORT=$MASTER_PORT_EVAL \
    bash evaluation/robotwin/launch_server_pred_latent.sh \
    > "$REPO/train_out/srv_M1v_WRONG.log" 2>&1 &
  SRV=$!
  wait_port_up $START_PORT || { tail -n 60 "$REPO/train_out/srv_M1v_WRONG.log"; kill -9 $SRV; exit 1; }
  echo "[ablation/implicit/eval] server LISTEN :$START_PORT"

  for t in $TASKS; do
    echo "=== M1v_WRONG :: $t (N=$TEST_NUM) ==="
    export LD_LIBRARY_PATH=/usr/lib64:/usr/lib:$LD_LIBRARY_PATH
    PYTHONWARNINGS=ignore::UserWarning XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
    python -m evaluation.robotwin.eval_polict_client_openpi_latent \
      --config policy/ACT/deploy_policy.yml --overrides \
      --task_name $t --task_config demo_clean \
      --train_config_name 0 --model_name 0 --ckpt_setting M1v_WRONG --seed 0 \
      --policy_name ACT \
      --save_root ./results_latent_M1v_WRONG --outputs_root ./outputs_latent_M1v_WRONG \
      --video_guidance_scale 5 --action_guidance_scale 1 \
      --test_num $TEST_NUM --port $START_PORT
  done

  kill -9 $SRV 2>/dev/null || true
  pkill -9 -f 'wan_va_server_predvideo\.py|torch\.distributed\.run' 2>/dev/null || true
  wait $SRV 2>/dev/null || true
  cp "$EVAL_CFG.bak" "$EVAL_CFG"   # 复原 eval env 配置

  ROBOTWIN=/inspire/qb-ilm2/project/26summer-camp-11/public/group3/RoboTwin
  echo
  echo "============= M1v_WRONG SR(对照 M0/M1/M1v 已有结果) ============="
  grep -H "" "$ROBOTWIN"/eval_result/*/ACT/demo_clean/{M0,M1,M1v,M1v_WRONG}/*/_result.txt 2>/dev/null || true
fi

echo "all done (PHASE=$PHASE)"
