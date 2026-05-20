#!/bin/bash
# ============================================================================
#  eval_route2_4090_robotwin.sh — 4090 实例 RoboTwin 一键评测(M1 / M1v)
# ----------------------------------------------------------------------------
#  专门给 4090 镜像(sii 公开镜像 `26summer-robocasa:260516` 或同等):该镜像
#  默认 Python 已装好 sapien + RoboTwin + lingbot 依赖,服务器(世界模型)
#  和客户端(sapien 仿真)可在**同一进程空间**跑。
#
#  与姐妹脚本 `eval_route2_latent_cot.sh` 区别:
#    本脚本                                      eval_route2_latent_cot.sh
#    --------------------------------------------------------------------------
#    用主仓库 launch_server.sh + 常规 client   eval_env 的 predvideo+latent
#    无 dream_video                            出 dream_video
#    不依赖 EVAL_ENV(更少 unknown 失败)      需要 EVAL_ENV 全套文件 OK
#    *4090 上 100% 可跑*                       依赖 eval_polict_client_openpi_latent
#                                              的实际可 import 性(已知不稳)
#    专为出 SR 表设计                          想要 dream_video 时才用
#
#  使用:
#    bash script/eval_route2_4090_robotwin.sh                # 默认 TEST_NUM=10
#    TEST_NUM=5   bash script/eval_route2_4090_robotwin.sh   # 冒烟
#    TEST_NUM=25  bash script/eval_route2_4090_robotwin.sh   # 严格统计
#    TASKS="adjust_bottle hanging_mug" bash ...              # 任务子集
# ============================================================================

# ============ 用户可调 ============
TEST_NUM=${TEST_NUM:-10}
TASKS=${TASKS:-"handover_block handover_mic hanging_mug blocks_ranking_size beat_block_hammer lift_pot"}

# ============ 路径 ============
REPO=${REPO:-/inspire/hdd/project/26summer-camp-11/26220077/lingbot-va}
ROBOTWIN=${ROBOTWIN:-/inspire/qb-ilm2/project/26summer-camp-11/public/group3/RoboTwin}
LINGBOT_VENV=${LINGBOT_VENV:-/inspire/qb-ilm2/project/26summer-camp-11/26220077/lingbot-va/.venv}

# Ckpts
BS=${BS:-$REPO/checkpoints/lingbot-va-posttrain-robotwin}
M1_CKPT=${M1_CKPT:-$REPO/train_out/checkpoints/checkpoint_step_1200}
M1V_CKPT=${M1V_CKPT:-$REPO/train_out/checkpoints/robotwin_kf0.1_vlmstage0.1/checkpoint_step_1200}

LOG_DIR=${LOG_DIR:-$REPO/train_out/eval_route2_4090}
mkdir -p "$LOG_DIR"

# 端口(被占就改)
START_PORT_M1=${START_PORT_M1:-29056};   MASTER_PORT_M1=${MASTER_PORT_M1:-29061}
START_PORT_M1V=${START_PORT_M1V:-29066}; MASTER_PORT_M1V=${MASTER_PORT_M1V:-29071}

# ============ 激活 venv ============
if [ -f "$LINGBOT_VENV/bin/activate" ]; then
  # shellcheck disable=SC1091
  source "$LINGBOT_VENV/bin/activate"
  echo "[eval-4090] activated venv: $LINGBOT_VENV"
fi
echo "[eval-4090] python = $(command -v python)"

# RoboTwin sapien 必需 /usr/lib64 + /usr/lib
export LD_LIBRARY_PATH=/usr/lib64:/usr/lib:$LD_LIBRARY_PATH

# ============ 硬性预检(缺一项就立即退出,不浪费时间) ============
err=0
for ck in "$BS" "$M1_CKPT" "$M1V_CKPT"; do
  if [ ! -d "$ck/transformer" ]; then
    echo "[eval-4090] ERROR: checkpoint 缺 transformer 子目录: $ck"; err=1
  fi
done
[ -d "$REPO" ]     || { echo "[eval-4090] ERROR: REPO 不存在: $REPO"; err=1; }
[ -d "$ROBOTWIN" ] || { echo "[eval-4090] ERROR: ROBOTWIN 不存在: $ROBOTWIN"; err=1; }

# 必须能 import torch + diffusers + websockets + sapien(后两者决定能否跑 server/client)
if ! python -c "import torch, diffusers, websockets" 2>/tmp/_eval_imp.err; then
  echo "[eval-4090] ERROR: 缺 torch / diffusers / websockets:"; cat /tmp/_eval_imp.err; err=1
fi
if ! python -c "import sapien" 2>/tmp/_eval_sap.err; then
  echo "[eval-4090] ERROR: 缺 sapien —— 此脚本必须在 4090 实例(镜像 26summer-robocasa)运行"
  echo "             当前不是 4090 / 镜像 / venv 不含 sapien;请切到 4090 实例后重跑"
  cat /tmp/_eval_sap.err
  err=1
fi

# 主仓库 client 必须可 import
if ! ( cd "$REPO" && python -c "import evaluation.robotwin.eval_polict_client_openpi" 2>/tmp/_eval_cli.err ); then
  echo "[eval-4090] ERROR: 主仓库的 eval_polict_client_openpi 不可 import:"
  cat /tmp/_eval_cli.err
  err=1
fi

if [ $err -ne 0 ]; then exit 1; fi
echo "[eval-4090] 预检通过(sapien + ckpt + client 模块全 OK)"

# ============ 一次性预备 ============
# (a) ckpt 自包含
for CK in "$M1_CKPT" "$M1V_CKPT"; do
  for s in vae tokenizer text_encoder; do
    [ -e "$CK/$s" ] || ln -sfn "$BS/$s" "$CK/$s"
  done
done
echo "[eval-4090] checkpoints 自包含 OK"

# (b) RoboTwin 视频开关
RW_YAML="$ROBOTWIN/task_config/demo_clean.yml"
if [ -f "$RW_YAML" ]; then
  sed -i 's/^\(\s*eval_video_log\s*:\s*\).*/\1True/' "$RW_YAML"
  echo "[eval-4090] RoboTwin eval_video_log: True"
fi

# (c) 退出/中断自动清理
cleanup () {
  echo
  echo "[eval-4090] cleanup: 杀残留 server"
  pkill -9 -f 'wan_va_server\.py' 2>/dev/null
  pkill -9 -f 'wan_va_server_predvideo\.py' 2>/dev/null
  pkill -9 -f 'torch\.distributed\.run|torch/distributed/run' 2>/dev/null
}
trap cleanup EXIT INT TERM

# ============ helpers ============
port_listen () { (exec 3<>/dev/tcp/127.0.0.1/$1) 2>/dev/null && { exec 3<&- 3>&-; return 0; }; return 1; }
wait_port_up   () { for i in $(seq 1 300); do port_listen $1 && return 0
                    kill -0 $SRV 2>/dev/null || return 2; sleep 2; done; return 1; }
wait_port_free () { for i in $(seq 1 60); do port_listen $1 || return 0; sleep 1; done; return 1; }

# ============ run_one (常规 server + 常规 client) ============
run_one () {            # $1=tag $2=ckpt $3=start_port $4=master_port
  tag=$1; ckpt=$2; sp=$3; mp=$4
  echo
  echo "================  $tag  (server :$sp, N=$TEST_NUM)  ================"
  echo "[eval-4090] ckpt = $ckpt"

  if ! wait_port_free $sp; then echo "[eval-4090] port $sp 仍被占"; return 1; fi
  if ! wait_port_free $mp; then echo "[eval-4090] port $mp 仍被占"; return 1; fi

  # 启常规 server(主仓库 launch_server.sh,无 predvideo)
  cd "$REPO" || return 1
  VA_EVAL_CKPT="$ckpt" CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} \
  START_PORT=$sp MASTER_PORT=$mp \
    bash evaluation/robotwin/launch_server.sh > "$LOG_DIR/srv_$tag.log" 2>&1 &
  SRV=$!
  echo "[eval-4090] server $tag launched (PID=$SRV), waiting LISTEN :$sp..."
  if ! wait_port_up $sp; then
    rc=$?
    echo "[eval-4090] server $tag 起不来 (rc=$rc), 最后 80 行日志:"
    echo "------------------------------------------------------------"
    tail -n 80 "$LOG_DIR/srv_$tag.log"
    echo "------------------------------------------------------------"
    kill -9 $SRV 2>/dev/null
    return 1
  fi
  echo "[eval-4090] server $tag LISTEN :$sp"

  # 逐任务跑常规 client
  for t in $TASKS; do
    echo
    echo "--- $tag :: $t (N=$TEST_NUM) ---"
    PYTHONWARNINGS=ignore::UserWarning XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
    python -m evaluation.robotwin.eval_polict_client_openpi \
      --config policy/ACT/deploy_policy.yml --overrides \
      --task_name $t --task_config demo_clean \
      --train_config_name 0 --model_name 0 --ckpt_setting "$tag" --seed 0 \
      --policy_name ACT \
      --save_root "./results_$tag" \
      --video_guidance_scale 5 --action_guidance_scale 1 \
      --test_num $TEST_NUM --port $sp \
      2>&1 | tee "$LOG_DIR/${tag}_${t}.log"
  done

  # 关本档 server
  kill -9 $SRV 2>/dev/null
  pkill -9 -f 'wan_va_server\.py' 2>/dev/null
  pkill -9 -f 'torch\.distributed\.run|torch/distributed/run' 2>/dev/null
  wait $SRV 2>/dev/null
  wait_port_free $sp
  wait_port_free $mp
  echo "[eval-4090] server $tag closed"
}

# ============ 主流程 ============
echo "[eval-4090] cleaning leftover servers/clients"
pkill -9 -f 'wan_va_server|wan_va_server_predvideo|torch\.distributed\.run|eval_polict_client_openpi' 2>/dev/null
sleep 5

echo
echo "########################################################################"
echo "#  4090 RoboTwin 评测 (regular server + regular client, 无 dream_video)"
echo "#    M1  (kf only)        :  $M1_CKPT"
echo "#    M1v (kf + VLM stage) :  $M1V_CKPT"
echo "#  TEST_NUM=$TEST_NUM, 任务($(echo $TASKS | wc -w) 个):"
echo "#    $TASKS"
echo "#  日志:$LOG_DIR"
echo "########################################################################"

run_one M1  "$M1_CKPT"  "$START_PORT_M1"  "$MASTER_PORT_M1"
run_one M1v "$M1V_CKPT" "$START_PORT_M1V" "$MASTER_PORT_M1V"

# ============ SR 汇总 ============
echo
echo "============================  SR 汇总  ============================"
python - <<PY
import os, glob, json
ROBOTWIN = "$ROBOTWIN"
LOG_DIR  = "$LOG_DIR"
TAGS = ["M1", "M1v"]
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
        fp = files[-1]
        try:
            lines = [l.strip() for l in open(fp) if l.strip()]
            sr = float(lines[-1]) if lines else None
        except Exception:
            sr = None
        rows[tag][t] = sr

hdr = ["Task", "M1 (kf)", "M1v (kf+VLM)", "Δ (M1v - M1)"]
w = [28, 12, 14, 14]
print("| " + " | ".join(f"{h:<{w[i]}s}" for i,h in enumerate(hdr)) + " |")
print("|" + "|".join("-"*(c+2) for c in w) + "|")
for t in TASKS:
    m1 = rows["M1"].get(t); m1v = rows["M1v"].get(t)
    s1 = "--" if m1 is None else f"{m1:.3f}"
    s2 = "--" if m1v is None else f"{m1v:.3f}"
    d  = "--" if (m1 is None or m1v is None) else f"{m1v-m1:+.3f}"
    print("| " + " | ".join(f"{c:<{w[i]}s}" for i,c in enumerate([t, s1, s2, d])) + " |")

def mean(vs):
    vs = [v for v in vs if v is not None]
    return sum(vs)/len(vs) if vs else None
m1m, m1vm = mean(rows["M1"].values()), mean(rows["M1v"].values())
sm1 = "--" if m1m is None else f"{m1m:.3f}"
sm2 = "--" if m1vm is None else f"{m1vm:.3f}"
sdm = "--" if (m1m is None or m1vm is None) else f"{m1vm-m1m:+.3f}"
print("| " + " | ".join(f"{c:<{w[i]}s}" for i,c in enumerate(["Mean", sm1, sm2, sdm])) + " |")

out = os.path.join(LOG_DIR, "eval_route2_4090_summary.json")
with open(out, "w", encoding="utf-8") as f:
    json.dump({"test_num": TEST_NUM, "tasks": TASKS, "results": rows,
               "mean": {"M1": m1m, "M1v": m1vm,
                        "delta_M1v_minus_M1": (None if (m1m is None or m1vm is None) else m1vm-m1m)}},
              f, indent=2, ensure_ascii=False)
print(f"\n[eval-4090] 汇总写入: {out}")
PY

# ============ 产物位置 ============
echo
echo "============================  产物位置  ============================"
echo "  SR 文本     : $ROBOTWIN/eval_result/<task>/ACT/demo_clean/{M1,M1v}/<ts>/_result.txt"
echo "  RoboTwin 视频: $ROBOTWIN/eval_result/<task>/ACT/demo_clean/{M1,M1v}/<ts>/episode*.mp4"
echo "  想象 vs 真实: $ROBOTWIN/results_{M1,M1v}/stseed-*/visualization/<task>/*_True|False.mp4"
echo "  per-task log: $LOG_DIR/{M1,M1v}_<task>.log"
echo "  server log  : $LOG_DIR/srv_{M1,M1v}.log"
echo "  汇总 JSON   : $LOG_DIR/eval_route2_4090_summary.json"
echo
echo "[eval-4090] all done."
