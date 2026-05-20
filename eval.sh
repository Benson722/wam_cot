#!/bin/bash
# ============================================================================
#  eval.sh —— 报告复现统一入口(WAM-CoT 路线二·Latent-CoT)
# ----------------------------------------------------------------------------
#  一个脚本跑通报告里所有"测试"类工作。把已有的零散 script/*.sh 包成统一接口,
#  保证审稿人/同组成员**一条命令**就能验到对应章节的数字。
#
#  设计原则:
#    1) 默认参数 = 报告里报的数字(seed=0, TEST_NUM=10, 6 个代表任务...)
#    2) 不重写已有的子脚本逻辑,只 dispatch;改子脚本不影响这层接口
#    3) 实例自检:probe 任何机器跑得动;sr/ablation 强制要 4090
#    4) 失败"快返回"——缺 ckpt / 缺 sapien / 端口被占,立刻报告退出,不浪费时间
#    5) 所有产物默认落到 $REPO/train_out/ 下分目录(eval_route2/, eval_route2_4090/,
#       ablation_explicit/, ablation_implicit/, probe/reproduce_out/, judge/)
#
#  ==========  用法  ==========
#    bash eval.sh --help                       # 看本帮助
#    bash eval.sh probe                        # §6/§10 探针(CPU,~1 min,4/4 PASS)
#    bash eval.sh sr                           # §9 在线 SR(M1 vs M1v,4090,~1.5h)
#    bash eval.sh sr --dream                   # 同上,出 dream_video(需要 eval_env 完整)
#    bash eval.sh ablation explicit            # §10.1/§10.2 显式 CoT 消融(A0/A1/A2)
#    bash eval.sh ablation implicit train      # §10.3 隐式消融:训 M1v_WRONG(H200)
#    bash eval.sh ablation implicit probe      # §10.3:对 4 ckpt 跑探针(已有 dump 用)
#    bash eval.sh ablation implicit eval       # §10.3:WRONG 在线 SR
#    bash eval.sh ablation implicit all        # 全跑(train→probe→eval)
#    bash eval.sh judge                        # §8 VLM 过程性评判(Qwen3-VL 逐子目标)
#    bash eval.sh all                          # probe + sr + ablation explicit + judge
#
#  ==========  常用变量覆盖  ==========
#    TEST_NUM=25                         # 严格统计;默认 10(冒烟用 5)
#    TASKS="adjust_bottle hanging_mug"   # 任务子集;默认 6 个代表任务
#    SEED=0                              # 探针种子(canonical 用 0)
#    REPO=...                            # 主仓库路径(默认服务器约定路径)
#    EVAL_ENV=...                        # eval_env 路径(latent server/client)
#    ROBOTWIN=...                        # RoboTwin 安装路径
#    LINGBOT_VENV=...                    # python venv
#    CUDA_VISIBLE_DEVICES=0              # 评测用哪块 GPU(默认 0)
#
#  ==========  期望数字(seed=0)  ==========
#    probe:   stock 0.652 / kf 0.648 / **kfvlm 0.778** / wrongstage 0.638
#             (容忍 ±0.01,4/4 PASS Δ=+0.000)
#    sr:      6 任务平均 — M1 vs M1v,详见 §9 表
#
#  作者:WAM-CoT 项目组 (26 夏令营,路线二·Latent-CoT)
# ============================================================================

# ----------- 路径默认值 -----------
REPO=${REPO:-/inspire/hdd/project/26summer-camp-11/26220077/lingbot-va}
EVAL_ENV=${EVAL_ENV:-/inspire/qb-ilm2/project/26summer-camp-11/public/group3/eval_env/sii_wam_cot/lingbot-va_goal_cond_cot}
ROBOTWIN=${ROBOTWIN:-/inspire/qb-ilm2/project/26summer-camp-11/public/group3/RoboTwin}
LINGBOT_VENV=${LINGBOT_VENV:-/inspire/qb-ilm2/project/26summer-camp-11/26220077/lingbot-va/.venv}

# 报告里报的 6 个代表任务
DEFAULT_TASKS="handover_block handover_mic hanging_mug blocks_ranking_size beat_block_hammer lift_pot"
TASKS=${TASKS:-$DEFAULT_TASKS}
TEST_NUM=${TEST_NUM:-10}
SEED=${SEED:-0}

# ----------- 颜色 / 工具 -----------
if [ -t 1 ]; then
  C_TITLE='\033[1;36m'; C_OK='\033[0;32m'; C_ERR='\033[0;31m'
  C_WARN='\033[0;33m'; C_DIM='\033[0;90m'; C_RST='\033[0m'
else
  C_TITLE=''; C_OK=''; C_ERR=''; C_WARN=''; C_DIM=''; C_RST=''
fi
banner () { echo -e "\n${C_TITLE}################  $*  ################${C_RST}"; }
say     () { echo -e "${C_DIM}[eval.sh]${C_RST} $*"; }
ok      () { echo -e "${C_OK}[eval.sh OK]${C_RST} $*"; }
warn    () { echo -e "${C_WARN}[eval.sh WARN]${C_RST} $*"; }
die     () { echo -e "${C_ERR}[eval.sh ERROR]${C_RST} $*" >&2; exit 1; }

# ----------- venv 激活 -----------
maybe_activate_venv () {
  if [ -f "$LINGBOT_VENV/bin/activate" ]; then
    # shellcheck disable=SC1091
    source "$LINGBOT_VENV/bin/activate"
    say "activated venv: $LINGBOT_VENV"
  else
    warn "$LINGBOT_VENV 不存在,沿用当前 shell 的 python"
  fi
  say "python = $(command -v python)"
}

# ----------- 实例自检 -----------
need_4090 () {
  if ! python -c "import sapien" 2>/dev/null; then
    die "缺 sapien —— sr / ablation eval 必须在 4090 实例(镜像 26summer-robocasa:260516)运行"
  fi
}
need_cpu_deps () {
  python -c "import numpy, torch, sklearn, matplotlib" 2>/tmp/_evl_cpu.err \
    || { cat /tmp/_evl_cpu.err; die "缺依赖:pip install numpy torch scikit-learn matplotlib"; }
}
need_repo () {
  [ -d "$REPO" ]     || die "REPO 不存在: $REPO"
  [ -d "$REPO/script" ] || die "$REPO/script 不存在(REPO 路径不对?)"
}

# ----------- help -----------
print_help () {
  sed -n '2,55p' "$0"
}

# ----------- mode: probe -----------
do_probe () {
  banner "PROBE  (Latent-CoT #4,字节级再现 §6 / §10)"
  need_cpu_deps
  need_repo
  local script="$REPO/script/reproduce_probe.sh"
  [ -f "$script" ] || die "$script 不存在"
  say "运行:SEED=$SEED bash $script"
  SEED=$SEED bash "$script" || die "probe 失败"
  ok "probe 复现完成 — 看 $REPO/train_out/probe/reproduce_out/"
}

# ----------- mode: sr -----------
do_sr () {
  banner "SR  (在线 RoboTwin 评测,M1 vs M1v)"
  need_4090
  need_repo
  local dream=0
  for a in "$@"; do
    case $a in
      --dream) dream=1 ;;
      *)       warn "sr: 未知参数 $a(忽略)" ;;
    esac
  done

  if [ $dream -eq 1 ]; then
    local script="$REPO/script/eval_route2_latent_cot.sh"
    [ -f "$script" ] || die "$script 不存在"
    say "运行(latent 路径,出 dream_video):DREAM_VIDEO=1 TEST_NUM=$TEST_NUM bash $script"
    DREAM_VIDEO=1 TEST_NUM=$TEST_NUM TASKS="$TASKS" bash "$script" || die "sr (dream) 失败"
  else
    local script="$REPO/script/eval_route2_4090_robotwin.sh"
    [ -f "$script" ] || die "$script 不存在"
    say "运行(常规 server + 常规 client,纯 SR):TEST_NUM=$TEST_NUM bash $script"
    TEST_NUM=$TEST_NUM TASKS="$TASKS" bash "$script" || die "sr 失败"
  fi
  ok "sr 完成 — SR 文本: $ROBOTWIN/eval_result/<task>/ACT/demo_clean/{M1,M1v}/<ts>/_result.txt"
  ok "        汇总 JSON: $REPO/train_out/eval_route2_4090/eval_route2_4090_summary.json"
}

# ----------- mode: ablation -----------
do_ablation () {
  local subtype=$1; shift || true
  need_repo
  case $subtype in
    explicit)
      banner "ABLATION-EXPLICIT  (A0/A1/A2:full / no_cot / shuffle,§10.1-§10.2)"
      need_4090
      local script="$REPO/script/run_ablation_explicit.sh"
      [ -f "$script" ] || die "$script 不存在"
      say "运行:TEST_NUM=$TEST_NUM bash $script"
      TEST_NUM=$TEST_NUM TASKS="$TASKS" bash "$script" || die "ablation explicit 失败"
      ok "explicit 完成 — 日志: $REPO/train_out/ablation_explicit/"
      ;;
    implicit)
      local phase=${1:-all}
      banner "ABLATION-IMPLICIT  (Ablation-3: vlm_stage shuffle,§10.3,PHASE=$phase)"
      case $phase in
        train) need_repo ;;
        probe) need_cpu_deps ;;
        eval)  need_4090 ;;
        all)   need_repo ;;
        *) die "ablation implicit: PHASE 必须是 train|probe|eval|all (当前: $phase)" ;;
      esac
      local script="$REPO/script/run_ablation_implicit.sh"
      [ -f "$script" ] || die "$script 不存在"
      say "运行:PHASE=$phase bash $script"
      PHASE=$phase TEST_NUM=$TEST_NUM TASKS="$TASKS" bash "$script" || die "ablation implicit ($phase) 失败"
      ok "implicit ($phase) 完成 — 日志: $REPO/train_out/ablation_implicit/"
      ;;
    "")
      die "ablation: 子类必填(explicit | implicit [train|probe|eval|all])"
      ;;
    *)
      die "ablation: 未知子类 '$subtype'(用 explicit 或 implicit)"
      ;;
  esac
}

# ----------- mode: judge -----------
do_judge () {
  banner "JUDGE  (VLM 过程性评判,Qwen3-VL,§8)"
  need_repo
  local judge_py="$REPO/evaluation/robotwin/judge_completion.py"
  [ -f "$judge_py" ] || die "$judge_py 不存在(本仓库或路径不正确)"
  # 走公网 Qwen3-VL-4B-Instruct(qwen_api.py 端点)
  export VLM_BASE_URL=${VLM_BASE_URL:-http://106.12.146.172:8271/v1}
  export VLM_MODEL=${VLM_MODEL:-Qwen3-VL-4B-Instruct}
  unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy 2>/dev/null || true
  local out_dir=${JUDGE_OUT:-$REPO/train_out/judge}
  mkdir -p "$out_dir"
  say "运行:python $judge_py --videos-root $ROBOTWIN/eval_result --out $out_dir"
  python "$judge_py" \
      --videos-root "$ROBOTWIN/eval_result" \
      --tags M1 M1v \
      --tasks $TASKS \
      --out "$out_dir" \
      --vlm-base-url "$VLM_BASE_URL" --vlm-model "$VLM_MODEL" \
    || die "judge 失败"
  ok "judge 完成 — 看 $out_dir/judge_summary.json + per-task JSONL"
}

# ----------- mode: all -----------
do_all () {
  banner "ALL  (probe + sr + ablation explicit + judge)"
  do_probe
  do_sr
  do_ablation explicit
  do_judge
  ok "ALL 完成 — 报告里 §6 / §9 / §10.1-2 / §8 的数字应全部就位"
}

# ============================================================================
#  dispatch
# ============================================================================
if [ $# -eq 0 ]; then print_help; exit 0; fi

# 激活 venv(probe 也用,因为 reproduce_probe.sh 内部会再激活一次,无副作用)
maybe_activate_venv

MODE=$1; shift
case $MODE in
  -h|--help|help) print_help ;;
  probe)          do_probe "$@" ;;
  sr)             do_sr "$@" ;;
  ablation)       do_ablation "$@" ;;
  judge)          do_judge "$@" ;;
  all)            do_all "$@" ;;
  *)              die "未知 mode: $MODE(用 --help 看用法)" ;;
esac
