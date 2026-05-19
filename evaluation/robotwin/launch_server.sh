START_PORT=${START_PORT:-29056}
MASTER_PORT=${MASTER_PORT:-29061}

save_root='visualization/'
mkdir -p $save_root

# Fix: system /usr/lib/x86_64-linux-gnu cuDNN shadows the cuDNN bundled with
# torch 2.9 (cu126) -> `libcudnn_graph.so.9: undefined symbol:
# cudnnGetLibConfig` -> SIGABRT on the first GPU forward. We locate torch's
# OWN matched cuDNN and BOTH prepend its dir to LD_LIBRARY_PATH and LD_PRELOAD
# the exact libcudnn*.so.9 so the linker cannot fall back to the system copy.
# Always prints what it found. (Disable with NO_CUDNN_FIX=1.)
if [ "${NO_CUDNN_FIX:-0}" != "1" ]; then
  eval "$(python - <<'PY'
import glob, os, sysconfig
cand = []
try:
    import nvidia.cudnn  # torch cu126 wheels bundle nvidia-cudnn-cu12 here
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
preload = []
if dirs:
    for pat in ("libcudnn.so*", "libcudnn_graph.so*", "libcudnn_engines_precompiled.so*",
                "libcudnn_ops.so*", "libcudnn_adv.so*", "libcudnn_cnn.so*",
                "libcudnn_heuristic.so*", "libcudnn_engines_runtime_compiled.so*"):
        m = sorted(glob.glob(os.path.join(dirs[0], pat)))
        if m:
            preload.append(m[-1])
print("CUDNN_DIRS=%r" % (":".join(dirs),))
print("CUDNN_PRELOAD=%r" % (":".join(preload),))
PY
)"
  echo "[launch_server] cuDNN dirs: ${CUDNN_DIRS:-<none found>}"
  if [ -n "$CUDNN_DIRS" ]; then
    export LD_LIBRARY_PATH="$CUDNN_DIRS${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    if [ -n "$CUDNN_PRELOAD" ]; then
      export LD_PRELOAD="$CUDNN_PRELOAD${LD_PRELOAD:+:$LD_PRELOAD}"
    fi
    echo "[launch_server] LD_PRELOAD=${LD_PRELOAD:-<none>}"
  else
    echo "[launch_server] WARNING: no bundled cuDNN found in this env; the"
    echo "  system cuDNN will SIGABRT. Check: python -c \"import torch;"
    echo "  print(torch.backends.cudnn.version(), torch.__file__)\""
  fi
fi

python -m torch.distributed.run \
    --nproc_per_node 1 \
    --master_port $MASTER_PORT \
    wan_va/wan_va_server.py \
    --config-name robotwin \
    --port $START_PORT \
    --save_root $save_root


