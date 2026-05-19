#!/usr/bin/bash

set -x

umask 007
 
NGPU=${NGPU:-"8"}
MASTER_PORT=${MASTER_PORT:-"29501"}
PORT=${PORT:-"1106"}
LOG_RANK=${LOG_RANK:-"0"}
TORCHFT_LIGHTHOUSE=${TORCHFT_LIGHTHOUSE:-"http://localhost:29510"}
CONFIG_NAME=${CONFIG_NAME:-"robotwin_train"} # robotwin_train, libero_train

overrides=""
if [ $# -ne 0 ]; then
    overrides="$*"
fi

# WandB is OFFLINE by default (local logs, no login/network). Sync later:
#   wandb sync <save_root>/wandb/wandb/offline-run-*
# For cloud streaming instead: set WANDB_MODE=online plus real WANDB_API_KEY
# / WANDB_BASE_URL / WANDB_TEAM_NAME before launching (do NOT use the old
# "your key" placeholders — train.py strips those defensively anyway).
export WANDB_MODE=${WANDB_MODE:-offline}
export WANDB_PROJECT=${WANDB_PROJECT:-va_robotwin}

## node setting
num_gpu=${NGPU}
master_port=${MASTER_PORT}
log_rank=${LOG_RANK}
torchft_lighthouse=${TORCHFT_LIGHTHOUSE}
config_name=${CONFIG_NAME}

## cuDNN fix: system /usr/lib cuDNN shadows torch 2.9(cu126)'s bundled cuDNN
## -> `libcudnn_graph.so.9: undefined symbol: cudnnGetLibConfig` SIGABRT on
## first GPU forward (same venv as the inference server). Force torch's own
## matched cuDNN first via LD_LIBRARY_PATH + LD_PRELOAD. (NO_CUDNN_FIX=1 off.)
if [ "${NO_CUDNN_FIX:-0}" != "1" ]; then
  eval "$(python - <<'PY'
import glob, os, sysconfig
cand = []
try:
    import nvidia.cudnn
    cand.append(os.path.join(os.path.dirname(nvidia.cudnn.__file__), "lib"))
except Exception:
    pass
for key in ("purelib", "platlib"):
    base = sysconfig.get_paths().get(key, "")
    if base:
        for p in glob.glob(os.path.join(base, "**", "libcudnn*.so*"),
                           recursive=True):
            cand.append(os.path.dirname(p))
try:
    import torch
    cand.append(os.path.join(os.path.dirname(torch.__file__), "lib"))
except Exception:
    pass
seen, dirs = set(), []
for d in cand:
    if d and d not in seen and glob.glob(os.path.join(d, "libcudnn*.so*")):
        seen.add(d); dirs.append(d)
pre = []
if dirs:
    for pat in ("libcudnn.so*", "libcudnn_graph.so*",
                "libcudnn_engines_precompiled.so*", "libcudnn_ops.so*",
                "libcudnn_adv.so*", "libcudnn_cnn.so*",
                "libcudnn_heuristic.so*",
                "libcudnn_engines_runtime_compiled.so*"):
        m = sorted(glob.glob(os.path.join(dirs[0], pat)))
        if m:
            pre.append(m[-1])
print("CUDNN_DIRS=%r" % (":".join(dirs),))
print("CUDNN_PRELOAD=%r" % (":".join(pre),))
PY
)"
  echo "[run_va_posttrain] cuDNN dirs: ${CUDNN_DIRS:-<none found>}"
  if [ -n "$CUDNN_DIRS" ]; then
    export LD_LIBRARY_PATH="$CUDNN_DIRS${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    [ -n "$CUDNN_PRELOAD" ] && export LD_PRELOAD="$CUDNN_PRELOAD${LD_PRELOAD:+:$LD_PRELOAD}"
    echo "[run_va_posttrain] LD_PRELOAD=${LD_PRELOAD:-<none>}"
  fi
fi

## cmd setting
export TOKENIZERS_PARALLELISM=false
PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" TORCHFT_LIGHTHOUSE=${torchft_lighthouse} \
python -m torch.distributed.run \
    --nproc_per_node=${num_gpu} \
    --local-ranks-filter=${log_rank} \
    --master_port ${master_port} \
    --tee 3 \
    -m wan_va.train --config-name ${config_name} $overrides
