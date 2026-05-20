#!/bin/bash
# 启动 latent-pred server(出 dream_video),针对本项目两个 Latent-CoT ckpt:
#   TAG=M1   -> train_out/checkpoints/checkpoint_step_1200                       (kf only)
#   TAG=M1v  -> train_out/checkpoints/robotwin_kf0.1_vlmstage0.1/checkpoint_step_1200 (kf+VLM)
#
# 用法(4090 实例,RoboTwin/eval_env 镜像):
#   TAG=M1  bash script/launch_server_pred_latent.sh
#   TAG=M1v bash script/launch_server_pred_latent.sh
#   TAG=M1v START_PORT=29066 MASTER_PORT=29071 CUDA_VISIBLE_DEVICES=1 bash ...
#
# 设计:
#   - 复用 eval_env 下的 wan_va/wan_va_server_predvideo.py(reference 脚本同款)
#   - 通过 sed 临时改 eval_env 的 va_robotwin_cfg.py 切到本项目 ckpt
#   - 自动补齐 ckpt 的 vae/tokenizer/text_encoder 软链(从 BASE)
#   - 配套 client: script/launch_client_latent.sh

TAG=${TAG:-M1}

REPO=${REPO:-/inspire/hdd/project/26summer-camp-11/26220077/lingbot-va}
EVAL_ENV=${EVAL_ENV:-/inspire/qb-ilm2/project/26summer-camp-11/public/group3/eval_env/sii_wam_cot/lingbot-va_goal_cond_cot}
BS=${BS:-$REPO/checkpoints/lingbot-va-posttrain-robotwin}
EVAL_CFG="$EVAL_ENV/wan_va/configs/va_robotwin_cfg.py"

case $TAG in
  M1)  CKPT=$REPO/train_out/checkpoints/checkpoint_step_1200 ;;
  M1v) CKPT=$REPO/train_out/checkpoints/robotwin_kf0.1_vlmstage0.1/checkpoint_step_1200 ;;
  *)   echo "[server] TAG 必须是 M1 或 M1v (当前: $TAG)"; exit 1 ;;
esac

START_PORT=${START_PORT:-29056}
MASTER_PORT=${MASTER_PORT:-29061}

# 预检
[ -d "$CKPT/transformer" ] || { echo "[server] 缺 $CKPT/transformer"; exit 1; }
[ -d "$EVAL_ENV/wan_va" ]  || { echo "[server] 缺 $EVAL_ENV (eval_env 路径不对)"; exit 1; }
[ -f "$EVAL_CFG" ]         || { echo "[server] 缺 $EVAL_CFG"; exit 1; }

# 补 ckpt 自包含(server 单目录加载 vae/tokenizer/text_encoder/transformer)
for s in vae tokenizer text_encoder; do
  [ -e "$CKPT/$s" ] || ln -sfn "$BS/$s" "$CKPT/$s"
done

# 备份 + 切 ckpt 路径
[ -f "$EVAL_CFG.bak" ] || cp "$EVAL_CFG" "$EVAL_CFG.bak"
sed -i "s|^va_robotwin_cfg\.wan22_pretrained_model_name_or_path = .*|va_robotwin_cfg.wan22_pretrained_model_name_or_path = \"$CKPT\"|" "$EVAL_CFG"
echo "[server] TAG=$TAG  CKPT=$CKPT"
echo "[server] CFG -> $(grep -E '^va_robotwin_cfg\.wan22_pretrained' "$EVAL_CFG")"

# dream_video 落地目录(per-tag)
save_root="visualization_predvideo_${TAG}/"
mkdir -p "$EVAL_ENV/$save_root"

# === Pre-flight: 打印模型参数量(EVAL_ENV server 源码不动,在我们 wrapper
#                  里靠 safetensors metadata 算,~100ms,不加载权重) ===
PRINT_PARAMS_PY=${PRINT_PARAMS_PY:-$REPO/script/print_model_params.py}
if [ -f "$PRINT_PARAMS_PY" ]; then
  echo "[server] pre-flight: model parameter counts (TAG=$TAG)"
  python "$PRINT_PARAMS_PY" --ckpt "$CKPT" --tag "$TAG" \
    || echo "[server] (param 打印失败,不阻断 server 启动)"
fi

# Ctrl-C 时复原 eval_env 配置(注意:**不能 exec**,否则 bash 被替换,EXIT
# trap 不会触发,CFG 不复原。改用普通 python 调用 + wait,trap 才有效。)
trap 'cp "$EVAL_CFG.bak" "$EVAL_CFG"; echo "[server] CFG restored from .bak"' EXIT INT TERM

# 起 server(完全沿用 reference 的 torch.distributed.run 形式)
cd "$EVAL_ENV"
echo "[server] starting on START_PORT=$START_PORT MASTER_PORT=$MASTER_PORT (Ctrl-C 退出会自动复原 CFG)"

python -m torch.distributed.run \
    --nproc_per_node 1 \
    --master_port $MASTER_PORT \
    wan_va/wan_va_server_predvideo.py \
    --config-name robotwin \
    --port $START_PORT \
    --save_root "$save_root"
