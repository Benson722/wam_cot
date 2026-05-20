#!/bin/bash
# Ablation-1 (丢弃 CoT) + Ablation-2 (推理时打乱 CoT 顺序) —— 显式 CoT
#
# 同一 M1 server (kf-only ckpt) 跑 3 档客户端,VLM 接公网
# Qwen3-VL-4B-Instruct (参考 qwen_api.py 的端点配置):
#   A0  cot_full   = 完整 VLM CoT          (基准,供 ΔSR 对照)
#   A1  no_cot     = 推理时不调用 VLM/丢弃子任务 (Ablation-1)
#   A2  shuffle    = VLM 出子任务后 shuffle 顺序 (Ablation-2)
#
# 运行环境: 4090 实例 (RoboTwin sim + 公网 Qwen3-VL 端点都在此)
# Python 环境: LingBot env (server) + RoboTwin env (client) —— 同实例
#
# 用法:
#   bash script/run_ablation_explicit.sh
#   TEST_NUM=20 bash script/run_ablation_explicit.sh
#   TASKS="adjust_bottle hanging_mug" TEST_NUM=5 bash script/run_ablation_explicit.sh

# 注意:刻意 *不* 用 set -e(它会和 `wait_port_up; rc=$?` 这种"先调用后检查"
# 的模式冲突,导致 wait_port_up 返回非 0 时脚本静默退出、连诊断都不打)。
# 关键失败路径都已用显式 if / || 处理。

# ============ 可调参数 ============
TEST_NUM=${TEST_NUM:-10}
TASKS=${TASKS:-"handover_block handover_mic hanging_mug blocks_ranking_size beat_block_hammer lift_pot"}
CKPT=${CKPT:-/inspire/hdd/project/26summer-camp-11/26220077/lingbot-va/train_out/checkpoints/checkpoint_step_1200}
START_PORT=${START_PORT:-29056}
MASTER_PORT=${MASTER_PORT:-29061}

# 外部 VLM (Qwen3-VL-4B-Instruct, 公网, 来自 qwen_api.py)
export PLANNER=vllm
export VLM_BASE_URL=${VLM_BASE_URL:-http://106.12.146.172:8271/v1}
export VLM_MODEL=${VLM_MODEL:-Qwen3-VL-4B-Instruct}

# bypass 任何 HTTP 代理(对应 qwen_api.py 的 trust_env=False)
unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy no_proxy NO_PROXY 2>/dev/null || true

REPO=/inspire/hdd/project/26summer-camp-11/26220077/lingbot-va
LOG_DIR=$REPO/train_out/ablation_explicit
mkdir -p "$LOG_DIR"
cd "$REPO"

# ============ helpers (bash /dev/tcp 探活, 不依赖 ss) ============
port_listen () { (exec 3<>/dev/tcp/127.0.0.1/$1) 2>/dev/null && { exec 3<&- 3>&-; return 0; }; return 1; }
wait_port_up () {
  for i in $(seq 1 300); do
    port_listen $1 && return 0
    kill -0 $SRV 2>/dev/null || return 2
    sleep 2
  done
  return 1
}
wait_port_free () { for i in $(seq 1 60); do port_listen $1 || return 0; sleep 1; done; return 1; }

# ============ 1) 清残留 ============
echo "[ablation/explicit] cleaning leftovers"
pkill -9 -f 'wan_va_server|wan_va_server_predvideo|torch\.distributed\.run|torch/distributed/run|eval_polict_client_openpi' 2>/dev/null || true
sleep 5
if ! wait_port_free $START_PORT; then
  echo "port $START_PORT 仍被占, 手动 pkill 再重跑"; exit 1
fi
if ! wait_port_free $MASTER_PORT; then
  echo "port $MASTER_PORT 仍被占, 手动 pkill 再重跑"; exit 1
fi

# ============ 2) 启动 M1 server (三档共享) ============
echo "[ablation/explicit] starting M1 server (VA_EVAL_CKPT=$CKPT) on :$START_PORT"
echo "[ablation/explicit] server 日志: $LOG_DIR/srv.log"
echo "[ablation/explicit] (loading 模型 ~30s, wait_port_up 最长等 10 min)"
VA_EVAL_CKPT="$CKPT" CUDA_VISIBLE_DEVICES=0 \
START_PORT=$START_PORT MASTER_PORT=$MASTER_PORT \
  bash evaluation/robotwin/launch_server.sh > "$LOG_DIR/srv.log" 2>&1 &
SRV=$!
if ! wait_port_up $START_PORT; then
  rc=$?
  echo "[ablation/explicit] server 起不来 (rc=$rc), 最后 80 行日志:"
  echo "------------------------------------------------------------"
  tail -n 80 "$LOG_DIR/srv.log"
  echo "------------------------------------------------------------"
  kill -9 $SRV 2>/dev/null || true
  pkill -9 -f 'wan_va_server\.py|torch\.distributed\.run|torch/distributed/run' 2>/dev/null || true
  exit 1
fi
echo "[ablation/explicit] server LISTEN :$START_PORT (PID=$SRV)"

# ============ 3) 三档 × 6 任务 (一个 server, 不重启) ============
# (tag : --cot_ablation 字符串)
CONDS=("cot_full:none" "no_cot:no_cot" "shuffle:shuffle_subtasks")

for cond_spec in "${CONDS[@]}"; do
  tag=${cond_spec%%:*}
  ablation=${cond_spec##*:}
  echo
  echo "============= condition: $tag  (cot_ablation=$ablation) ============="
  for t in $TASKS; do
    echo "--- $tag :: $t (N=$TEST_NUM) ---"
    # 直接调 python (不走 launch_cot_client.sh, 这样 ckpt_setting 可被
    # 我们改成 M1_<tag>, eval_result/.../M1_<tag>/<ts>/ 自然区分)
    export LD_LIBRARY_PATH=/usr/lib64:/usr/lib:$LD_LIBRARY_PATH
    PYTHONWARNINGS=ignore::UserWarning DEEPSEEK_API_KEY=${DEEPSEEK_API_KEY:-} \
    XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
    python -m evaluation.robotwin.eval_polict_client_openpi \
      --config policy/ACT/deploy_policy.yml --overrides \
      --task_name $t --task_config demo_clean \
      --train_config_name 0 --model_name 0 \
      --ckpt_setting "M1_${tag}" --seed 0 \
      --policy_name ACT \
      --save_root ./results_explicit_${tag} \
      --video_guidance_scale 5 --action_guidance_scale 1 \
      --test_num $TEST_NUM --port $START_PORT \
      --cot True --planner vllm \
      --vlm_base_url "$VLM_BASE_URL" --vlm_model "$VLM_MODEL" \
      --monitor_every 2 --cot_ablation "$ablation" \
      2>&1 | tee -a "$LOG_DIR/${tag}_${t}.log"
  done
done

# ============ 4) 关 server ============
echo
echo "[ablation/explicit] killing server"
kill -9 $SRV 2>/dev/null || true
pkill -9 -f 'wan_va_server\.py' 2>/dev/null || true
pkill -9 -f 'torch\.distributed\.run|torch/distributed/run' 2>/dev/null || true
wait $SRV 2>/dev/null || true
wait_port_free $START_PORT || true
wait_port_free $MASTER_PORT || true

# ============ 5) SR 汇总 ============
ROBOTWIN=/inspire/qb-ilm2/project/26summer-camp-11/public/group3/RoboTwin
echo
echo "============= SR 汇总 ============="
python - <<PY
import os, glob, json
ROBOTWIN = "$ROBOTWIN"
TAGS = ["M1_cot_full", "M1_no_cot", "M1_shuffle"]
TASKS = "$TASKS".split()
TEST_NUM = $TEST_NUM
rows = {tag: {} for tag in TAGS}
for tag in TAGS:
    for t in TASKS:
        pat = os.path.join(ROBOTWIN, "eval_result", t, "ACT",
                           "demo_clean", tag, "*", "_result.txt")
        files = sorted(glob.glob(pat))
        if not files:
            rows[tag][t] = None
            continue
        # 用最新的 timestamp
        fp = files[-1]
        try:
            lines = [l.strip() for l in open(fp) if l.strip()]
            sr = float(lines[-1]) if lines else None
        except Exception:
            sr = None
        rows[tag][t] = sr
# 打印表
hdr = ["Task"] + TAGS + ["ΔA1=cot-no_cot", "ΔA2=cot-shuffle"]
print("| " + " | ".join(f"{h:>18s}" for h in hdr) + " |")
print("|" + "|".join(["-" * 20] * len(hdr)) + "|")
for t in TASKS:
    cells = [t]
    for tag in TAGS:
        v = rows[tag].get(t)
        cells.append("--" if v is None else f"{v:.2f}")
    cot = rows["M1_cot_full"].get(t); nc = rows["M1_no_cot"].get(t); sh = rows["M1_shuffle"].get(t)
    d1 = "--" if cot is None or nc is None else f"{cot-nc:+.2f}"
    d2 = "--" if cot is None or sh is None else f"{cot-sh:+.2f}"
    cells.extend([d1, d2])
    print("| " + " | ".join(f"{c:>18s}" for c in cells) + " |")
# 均值
def mean(vs): vs=[v for v in vs if v is not None]; return sum(vs)/len(vs) if vs else None
mrow = ["Mean"]
for tag in TAGS:
    m = mean(rows[tag].values())
    mrow.append("--" if m is None else f"{m:.3f}")
cm, nm, sm = mean(rows["M1_cot_full"].values()), mean(rows["M1_no_cot"].values()), mean(rows["M1_shuffle"].values())
mrow.append("--" if cm is None or nm is None else f"{cm-nm:+.3f}")
mrow.append("--" if cm is None or sm is None else f"{cm-sm:+.3f}")
print("| " + " | ".join(f"{c:>18s}" for c in mrow) + " |")

# 也写 json 方便后续抓
out = "$LOG_DIR/ablation_explicit_summary.json"
with open(out, "w") as f:
    json.dump({"test_num": TEST_NUM, "tasks": TASKS, "results": rows}, f, indent=2)
print(f"\n汇总写入: {out}")
PY

echo
echo "明细日志: $LOG_DIR/<tag>_<task>.log"
echo "执行视频: $ROBOTWIN/eval_result/<task>/ACT/demo_clean/M1_<tag>/<ts>/episode*.mp4"
echo "完成."
