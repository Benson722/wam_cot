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

# Fix: system /usr/lib cuDNN shadows the cuDNN bundled with torch 2.9 (cu126),
# causing `libcudnn_graph.so.9: undefined symbol: cudnnGetLibConfig` -> SIGABRT
# on the first GPU forward. Prepend torch's own bundled NVIDIA libs so the
# matched cuDNN/cuBLAS are loaded first. (Disable with NO_CUDNN_FIX=1.)
if [ "${NO_CUDNN_FIX:-0}" != "1" ]; then
  NV_LIBS=$(python - <<'PY'
import os, glob
try:
    import nvidia
    base = os.path.dirname(nvidia.__file__)
    dirs = sorted(set(os.path.dirname(p) for p in
                      glob.glob(os.path.join(base, "*", "lib", "*.so*"))))
    print(":".join(dirs))
except Exception:
    print("")
PY
)
  if [ -n "$NV_LIBS" ]; then
    export LD_LIBRARY_PATH="$NV_LIBS${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    echo "[launch_server] LD_LIBRARY_PATH prepended with torch NVIDIA libs:"
    echo "  $NV_LIBS"
  fi
fi

python -m torch.distributed.run \
    --nproc_per_node 1 \
    --master_port "$MASTER_PORT" \
    wan_va/wan_va_server.py \
    --config-name robocasa \
    --port "$PORT" \
    --save_root "$SAVE_ROOT"
