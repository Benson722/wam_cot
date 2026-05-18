#!/usr/bin/env bash
# LingBot-VA inference server for Robocasa (zero-shot from LIBERO-Long ckpt).
#
# Run this in the *lingbot* conda env on the internet-capable 4090.
# Before launching:
#   1) edit wan_va/configs/va_robocasa_cfg.py ->
#        wan22_pretrained_model_name_or_path = <LIBERO-Long ckpt dir>
#   2) set that ckpt's transformer/config.json "attn_mode": "torch"
set -e

SAVE_ROOT=${SAVE_ROOT:-'visualization/robocasa/'}
PORT=${PORT:-29056}
MASTER_PORT=${MASTER_PORT:-29061}
mkdir -p "$SAVE_ROOT"

python -m torch.distributed.run \
    --nproc_per_node 1 \
    --master_port "$MASTER_PORT" \
    wan_va/wan_va_server.py \
    --config-name robocasa \
    --port "$PORT" \
    --save_root "$SAVE_ROOT"
