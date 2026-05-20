#!/bin/bash
# ============================================================================
#  eval_route2_latent_cot.sh
# ----------------------------------------------------------------------------
#  一键评测本项目训练的两个 Latent-CoT 检查点(路线二、隐式物理 CoT):
#    - M1  = Latent-CoT #1 (仅 keyframe 辅助头)         checkpoint_step_1200
#    - M1v = Latent-CoT Phase B (kf + VLM 语义阶段头)    robotwin_kf0.1_vlmstage0.1/
#                                                       checkpoint_step_1200
#
#  对 6 个代表性 RoboTwin 任务各跑 TEST_NUM 集,出 SR 汇总表 + 执行视频 +
#  dream_video(模型"想象"的未来 latent decode 出来的视频)。
#
#  使用:
#    bash script/eval_route2_latent_cot.sh                   # 默认 TEST_NUM=10
#    TEST_NUM=5  bash script/eval_route2_latent_cot.sh       # 冒烟
#    TEST_NUM=25 bash script/eval_route2_latent_cot.sh       # 严格统计
#    TASKS="adjust_bottle hanging_mug" TEST_NUM=10 bash ...  # 任务子集
#
#  必须在 SII **4090 实例**运行(RoboTwin sapien 仿真在这里;镜像
#  `26summer-robocasa:260516` 默认 Python 已具备 torch + diffusers + sapien)。
#
#  作者:WAM-CoT 项目组 (26 夏令营,路线二)
#  约定:`bash` 直接调用,**无 set -e**,失败路径有显式诊断与清理。
# ============================================================================

# ============ 用户可调(env 变量覆盖即可) ============
TEST_NUM=${TEST_NUM:-10}
TASKS=${TASKS:-"handover_block handover_mic hanging_mug blocks_ranking_size beat_block_hammer lift_pot"}

# ============ 固定路径(SII 服务器,本项目默认部署) ============
REPO=${REPO:-/inspire/hdd/project/26summer-camp-11/26220077/lingbot-va}
EVAL_ENV=${EVAL_ENV:-/inspire/qb-ilm2/project/26summer-camp-11/public/group3/eval_env/sii_wam_cot/lingbot-va_goal_cond_cot}
ROBOTWIN=${ROBOTWIN:-/inspire/qb-ilm2/project/26summer-camp-11/public/group3/RoboTwin}
LINGBOT_VENV=${LINGBOT_VENV:-/inspire/qb-ilm2/project/26summer-camp-11/26220077/lingbot-va/.venv}

# 三个检查点目录
BS=${BS:-$REPO/checkpoints/lingbot-va-posttrain-robotwin}                       # 基座(BASE)
M1_CKPT=${M1_CKPT:-$REPO/train_out/checkpoints/checkpoint_step_1200}            # M1
M1V_CKPT=${M1V_CKPT:-$REPO/train_out/checkpoints/robotwin_kf0.1_vlmstage0.1/checkpoint_step_1200}  # M1v

# eval_env 内的推理配置(我们用 sed 临时改 wan22_pretrained 这一行切换 ckpt)
EVAL_CFG="$EVAL_ENV/wan_va/configs/va_robotwin_cfg.py"

# 日志根
LOG_DIR=${LOG_DIR:-$REPO/train_out/eval_route2}
mkdir -p "$LOG_DIR"

# 端口(失败重跑时如端口被占,可覆盖 START_PORT_M1 等)
START_PORT_M1=${START_PORT_M1:-29056};  MASTER_PORT_M1=${MASTER_PORT_M1:-29061}
START_PORT_M1V=${START_PORT_M1V:-29066}; MASTER_PORT_M1V=${MASTER_PORT_M1V:-29071}

# ============ 环境激活(LingBot venv;含 torch 2.9 + diffusers + websockets) ============
if [ -f "$LINGBOT_VENV/bin/activate" ]; then
  # shellcheck disable=SC1091
  source "$LINGBOT_VENV/bin/activate"
  echo "[eval] activated venv: $LINGBOT_VENV"
else
  echo "[eval] WARNING: $LINGBOT_VENV 不存在,沿用当前 shell 的 python"
fi
echo "[eval] python = $(command -v python)"

# RoboTwin sapien 需要系统 /usr/lib64 + /usr/lib (vulkan/EGL/libstdc++)
export LD_LIBRARY_PATH=/usr/lib64:/usr/lib:$LD_LIBRARY_PATH

# ============ 预检 ============
err=0
for ck in "$BS" "$M1_CKPT" "$M1V_CKPT"; do
  if [ ! -d "$ck/transformer" ]; then
    echo "[eval] ERROR: checkpoint 缺少 transformer 子目录: $ck"; err=1
  fi
done
if [ ! -d "$EVAL_ENV" ]; then
  echo "[eval] ERROR: eval_env 不存在: $EVAL_ENV"; err=1
fi
if [ ! -d "$ROBOTWIN" ]; then
  echo "[eval] ERROR: RoboTwin 不存在: $ROBOTWIN"; err=1
fi
if [ ! -f "$EVAL_CFG" ]; then
  echo "[eval] ERROR: eval_env 推理配置不存在: $EVAL_CFG"; err=1
fi
# python 依赖快速 import 检查(给清晰错误)
python -c "import torch, diffusers, websockets" 2>/tmp/_eval_imp.err
if [ $? -ne 0 ]; then
  echo "[eval] ERROR: python 缺少 torch / diffusers / websockets:"
  cat /tmp/_eval_imp.err
  echo "       请激活 LingBot venv 或检查 LINGBOT_VENV 路径"
  err=1
fi
python -c "import sapien" 2>/dev/null
if [ $? -ne 0 ]; then
  echo "[eval] WARNING: python 缺少 sapien;RoboTwin client 会 ImportError"
  echo "       这一脚本要求 server 和 client 都用同一 python(4090 镜像默认就行)"
fi
if [ $err -ne 0 ]; then exit 1; fi

# ============ Auto-detect: latent client/server 是否存在 ============
# latent 版需要 eval_env 下两个文件(client + server),任缺一个就回退到主仓库的
# 常规 client + 常规 server(失去 dream_video,但 SR 表正常出)。
# 想强制 latent 路径(确保 dream_video):DREAM_VIDEO=1 bash ...
DREAM_VIDEO=${DREAM_VIDEO:-auto}
LATENT_CLIENT_FILE="$EVAL_ENV/evaluation/robotwin/eval_polict_client_openpi_latent.py"
LATENT_SERVER_SH="$EVAL_ENV/evaluation/robotwin/launch_server_pred_latent.sh"

if [ "$DREAM_VIDEO" = "1" ]; then
  USE_LATENT=1
elif [ "$DREAM_VIDEO" = "0" ]; then
  USE_LATENT=0
else
  # auto: 两个 latent 文件齐全 -> 用 latent,否则回退
  if [ -f "$LATENT_CLIENT_FILE" ] && [ -f "$LATENT_SERVER_SH" ]; then
    USE_LATENT=1
  else
    USE_LATENT=0
  fi
fi

if [ $USE_LATENT -eq 1 ]; then
  SERVER_CWD="$EVAL_ENV"
  SERVER_SH="evaluation/robotwin/launch_server_pred_latent.sh"
  CLIENT_MODULE="evaluation.robotwin.eval_polict_client_openpi_latent"
  CLIENT_EXTRA_ARGS=(--outputs_root ./outputs_latent_TAG)   # TAG 占位,后面替换
  SAVE_ROOT_PREFIX="./results_latent_"
  MODE_DESC="latent (含 dream_video,server=predvideo, client=latent)"
else
  SERVER_CWD="$REPO"
  SERVER_SH="evaluation/robotwin/launch_server.sh"
  CLIENT_MODULE="evaluation.robotwin.eval_polict_client_openpi"
  CLIENT_EXTRA_ARGS=()
  SAVE_ROOT_PREFIX="./results_"
  MODE_DESC="regular (无 dream_video, 但 SR 完整;主仓库 client+server)"
  if [ ! -f "$LATENT_CLIENT_FILE" ]; then
    echo "[eval] note: $LATENT_CLIENT_FILE 不存在 -> 自动回退到 regular 模式"
  fi
fi
echo "[eval] mode = $MODE_DESC"
echo "[eval] server cwd = $SERVER_CWD"
echo "[eval] server sh  = $SERVER_SH"
echo "[eval] client mod = $CLIENT_MODULE"

# ============ 一次性预备工作 ============
# 1) 把 M1 / M1v 补成"server 可单目录加载"(vae/tokenizer/text_encoder 从 BASE 软链)
for CK in "$M1_CKPT" "$M1V_CKPT"; do
  for s in vae tokenizer text_encoder; do
    [ -e "$CK/$s" ] || ln -sfn "$BS/$s" "$CK/$s"
  done
done
echo "[eval] checkpoints 已自包含(vae/tokenizer/text_encoder 软链 BASE)"

# 2) 打开 RoboTwin 视频开关(否则 episodeN.mp4 不出)
RW_YAML="$ROBOTWIN/task_config/demo_clean.yml"
if [ -f "$RW_YAML" ]; then
  sed -i 's/^\(\s*eval_video_log\s*:\s*\).*/\1True/' "$RW_YAML"
  echo "[eval] RoboTwin eval_video_log: True($RW_YAML)"
fi

# 3) 备份 eval_env 推理配置(脚本会用 sed 切 ckpt,退出时复原)
[ -f "$EVAL_CFG.bak" ] || cp "$EVAL_CFG" "$EVAL_CFG.bak"

# 4) 退出/中断时:复原配置 + 杀掉残留 server
cleanup () {
  echo
  echo "[eval] cleanup: 杀残留 server,复原 eval_env 配置"
  pkill -9 -f 'wan_va_server_predvideo\.py' 2>/dev/null
  pkill -9 -f 'wan_va_server\.py' 2>/dev/null
  pkill -9 -f 'torch\.distributed\.run|torch/distributed/run' 2>/dev/null
  [ -f "$EVAL_CFG.bak" ] && cp "$EVAL_CFG.bak" "$EVAL_CFG" && \
    echo "[eval] $EVAL_CFG 已从 .bak 复原"
}
trap cleanup EXIT INT TERM

# ============ helpers(bash 内建 /dev/tcp 探活,不依赖 ss/nc) ============
port_listen () { (exec 3<>/dev/tcp/127.0.0.1/$1) 2>/dev/null && { exec 3<&- 3>&-; return 0; }; return 1; }
wait_port_up   () { for i in $(seq 1 300); do port_listen $1 && return 0
                    kill -0 $SRV 2>/dev/null || return 2; sleep 2; done; return 1; }
wait_port_free () { for i in $(seq 1 60); do port_listen $1 || return 0; sleep 1; done; return 1; }
set_ckpt () {
  sed -i "s|^va_robotwin_cfg\.wan22_pretrained_model_name_or_path = .*|va_robotwin_cfg.wan22_pretrained_model_name_or_path = \"$1\"|" "$EVAL_CFG"
  echo "[eval] CFG -> $(grep -E '^va_robotwin_cfg\.wan22_pretrained' "$EVAL_CFG")"
}

# ============ run_one: 一档 server + 全部任务的 client ============
run_one () {            # $1=tag  $2=ckpt  $3=start_port  $4=master_port
  tag=$1; ckpt=$2; sp=$3; mp=$4
  echo
  echo "================  $tag  (server :$sp, N=$TEST_NUM)  ================"
  echo "[eval] ckpt = $ckpt"

  if ! wait_port_free $sp; then echo "[eval] port $sp 仍被占"; return 1; fi
  if ! wait_port_free $mp; then echo "[eval] port $mp 仍被占"; return 1; fi
  set_ckpt "$ckpt"

  # 启 server(latent 或 regular,见 USE_LATENT)
  cd "$SERVER_CWD" || return 1
  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} START_PORT=$sp MASTER_PORT=$mp \
    bash "$SERVER_SH" \
    > "$LOG_DIR/srv_$tag.log" 2>&1 &
  SRV=$!
  echo "[eval] server $tag launched (PID=$SRV), waiting LISTEN :$sp..."
  if ! wait_port_up $sp; then
    rc=$?
    echo "[eval] ERROR: server $tag 起不来 (rc=$rc),最后 80 行日志:"
    echo "------------------------------------------------------------"
    tail -n 80 "$LOG_DIR/srv_$tag.log"
    echo "------------------------------------------------------------"
    kill -9 $SRV 2>/dev/null
    return 1
  fi
  echo "[eval] server $tag LISTEN :$sp"

  # 逐任务跑 client
  for t in $TASKS; do
    echo
    echo "--- $tag :: $t (N=$TEST_NUM) ---"
    extras=()
    if [ $USE_LATENT -eq 1 ]; then
      extras+=(--outputs_root "./outputs_latent_$tag")
    fi
    PYTHONWARNINGS=ignore::UserWarning XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
    python -m "$CLIENT_MODULE" \
      --config policy/ACT/deploy_policy.yml --overrides \
      --task_name $t --task_config demo_clean \
      --train_config_name 0 --model_name 0 --ckpt_setting "$tag" --seed 0 \
      --policy_name ACT \
      --save_root "${SAVE_ROOT_PREFIX}${tag}" \
      "${extras[@]}" \
      --video_guidance_scale 5 --action_guidance_scale 1 \
      --test_num $TEST_NUM --port $sp \
      2>&1 | tee "$LOG_DIR/${tag}_${t}.log"
  done

  # 关本档 server(latent 和 regular 用不同进程名)
  kill -9 $SRV 2>/dev/null
  pkill -9 -f 'wan_va_server_predvideo\.py' 2>/dev/null
  pkill -9 -f 'wan_va_server\.py' 2>/dev/null
  pkill -9 -f 'torch\.distributed\.run|torch/distributed/run' 2>/dev/null
  wait $SRV 2>/dev/null
  wait_port_free $sp
  wait_port_free $mp
  echo "[eval] server $tag closed"
}

# ============ 主流程 ============
# 先清残留
echo "[eval] cleaning leftover servers/clients"
pkill -9 -f 'wan_va_server|wan_va_server_predvideo|torch\.distributed\.run|eval_polict_client_openpi' 2>/dev/null
sleep 5

echo
echo "########################################################################"
echo "#  开始评测两个 Latent-CoT ckpt:"
echo "#    M1  (kf only)        :  $M1_CKPT"
echo "#    M1v (kf + VLM stage) :  $M1V_CKPT"
echo "#  TEST_NUM=$TEST_NUM,任务($(echo $TASKS | wc -w) 个):"
echo "#    $TASKS"
echo "#  日志根:$LOG_DIR"
echo "########################################################################"

run_one M1  "$M1_CKPT"  "$START_PORT_M1"  "$MASTER_PORT_M1"
run_one M1v "$M1V_CKPT" "$START_PORT_M1V" "$MASTER_PORT_M1V"

# ============ SR 汇总表 ============
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
        # eval_polict_client_openpi_latent 用 ckpt_setting=tag,目录就是 .../tag/<时间戳>/
        pat = os.path.join(ROBOTWIN, "eval_result", t, "ACT",
                           "demo_clean", tag, "*", "_result.txt")
        files = sorted(glob.glob(pat))
        if not files:
            rows[tag][t] = None
            continue
        fp = files[-1]   # 最新时间戳
        try:
            lines = [l.strip() for l in open(fp) if l.strip()]
            sr = float(lines[-1]) if lines else None
        except Exception:
            sr = None
        rows[tag][t] = sr

# 打表
hdr = ["Task", "M1 (kf)", "M1v (kf+VLM)", "Δ (M1v - M1)"]
w = [28, 12, 14, 14]
print("| " + " | ".join(f"{h:<{w[i]}s}" for i,h in enumerate(hdr)) + " |")
print("|" + "|".join("-"*(c+2) for c in w) + "|")
for t in TASKS:
    m1 = rows["M1"].get(t); m1v = rows["M1v"].get(t)
    s1 = "--" if m1 is None else f"{m1:.3f}"
    s2 = "--" if m1v is None else f"{m1v:.3f}"
    d  = "--" if (m1 is None or m1v is None) else f"{m1v-m1:+.3f}"
    cells = [t, s1, s2, d]
    print("| " + " | ".join(f"{c:<{w[i]}s}" for i,c in enumerate(cells)) + " |")

def mean(vs):
    vs = [v for v in vs if v is not None]
    return sum(vs)/len(vs) if vs else None
m1m, m1vm = mean(rows["M1"].values()), mean(rows["M1v"].values())
sm1 = "--" if m1m is None else f"{m1m:.3f}"
sm2 = "--" if m1vm is None else f"{m1vm:.3f}"
sdm = "--" if (m1m is None or m1vm is None) else f"{m1vm-m1m:+.3f}"
print("| " + " | ".join(f"{c:<{w[i]}s}" for i,c in enumerate(["Mean", sm1, sm2, sdm])) + " |")

# 写 JSON 便于后续抓取
out = os.path.join(LOG_DIR, "eval_route2_summary.json")
with open(out, "w", encoding="utf-8") as f:
    json.dump({"test_num": TEST_NUM, "tasks": TASKS, "results": rows,
               "mean": {"M1": m1m, "M1v": m1vm,
                        "delta_M1v_minus_M1": (None if (m1m is None or m1vm is None) else m1vm-m1m)}},
              f, indent=2, ensure_ascii=False)
print(f"\n[eval] 汇总写入: {out}")
PY

# ============ 结尾:产物位置提示 ============
echo
echo "============================  产物位置  ============================"
echo "  SR 文本     : $ROBOTWIN/eval_result/<task>/ACT/demo_clean/{M1,M1v}/<ts>/_result.txt"
echo "  执行视频    : $ROBOTWIN/eval_result/<task>/ACT/demo_clean/{M1,M1v}/<ts>/episode*.mp4"
echo "  想象 vs 真实: $ROBOTWIN/results_latent_{M1,M1v}/stseed-*/visualization/<task>/*_True|False.mp4"
echo "  dream_video : $EVAL_ENV/visualization_predvideo/"
echo "                $ROBOTWIN/outputs_latent_{M1,M1v}/"
echo "  per-task log: $LOG_DIR/{M1,M1v}_<task>.log"
echo "  server log  : $LOG_DIR/srv_{M1,M1v}.log"
echo "  汇总 JSON   : $LOG_DIR/eval_route2_summary.json"
echo
echo "[eval] all done."
