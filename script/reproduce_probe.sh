#!/bin/bash
# ============================================================================
#  reproduce_probe.sh —— 一键复现 §6 / §10.4 离线探针消融(Latent-CoT #4)
# ----------------------------------------------------------------------------
#  输入(已落盘的"中间文件",~140 MB 总计,在 qb-ilm2 大盘):
#    $REPO/train_out/probe/h_stock.pt          (无 CoT 基座,2810 frames × 3072d)
#    $REPO/train_out/probe/h_kf.pt             (M1: kf-only)
#    $REPO/train_out/probe/h_kfvlm.pt          (M1v: kf+VLM stage)
#    $REPO/train_out/probe/h_wrongstage.pt     (Ablation-3: M1v_WRONG, step 200)
#
#  每个 dump 含 {feat, stage(kf), vlm_stage, episode, ckpt} —— 把"运行世界
#  模型 forward 抽 backbone hidden"这一(慢、需 GPU+完整 ckpt)步骤固化下来,
#  下游线性探针(快、只需 numpy+sklearn)即可**秒级复跑且数值稳定**。
#
#  使用:
#    bash script/reproduce_probe.sh                 # 默认 SEED=0,~1 min
#    SEED=1  bash script/reproduce_probe.sh         # 换种子看稳定性
#    PROBE_DIR=<别处>/probe bash script/reproduce_probe.sh
#
#  输出:
#    reproduce_out/<tag>/{results_*.json, probe_*.pt, tsne_*.png}
#    末尾打 "actual vs expected" 表 + PASS/FAIL
# ============================================================================

REPO=${REPO:-/inspire/hdd/project/26summer-camp-11/26220077/lingbot-va}
PROBE_DIR=${PROBE_DIR:-$REPO/train_out/probe}
OUT_DIR=${OUT_DIR:-$PROBE_DIR/reproduce_out}
SEED=${SEED:-0}
TOL=${TOL:-0.01}                  # 容忍度 ±0.01 (sklearn t-SNE / linear init 抖动)
LINGBOT_VENV=${LINGBOT_VENV:-/inspire/qb-ilm2/project/26summer-camp-11/26220077/lingbot-va/.venv}

CANON="$PROBE_DIR/probe_canonical.json"

# ============ 激活 venv (只需 numpy+sklearn+torch,任何 venv 都行) ============
if [ -f "$LINGBOT_VENV/bin/activate" ]; then
  # shellcheck disable=SC1091
  source "$LINGBOT_VENV/bin/activate"
  echo "[reproduce] activated venv: $LINGBOT_VENV"
fi
echo "[reproduce] python = $(command -v python)"

# 依赖快速检查
if ! python -c "import numpy, torch, sklearn, matplotlib" 2>/tmp/_rp.err; then
  echo "[reproduce] ERROR: 缺依赖(需 numpy + torch + sklearn + matplotlib):"
  cat /tmp/_rp.err
  echo "  安装:pip install numpy torch scikit-learn matplotlib"
  exit 1
fi

# ============ 预检 dump 文件 ============
TAGS="h_stock h_kf h_kfvlm h_wrongstage"
missing=0
for tag in $TAGS; do
  if [ ! -f "$PROBE_DIR/$tag.pt" ]; then
    echo "[reproduce] MISSING dump: $PROBE_DIR/$tag.pt"
    missing=1
  fi
done
if [ $missing -ne 0 ]; then
  echo
  echo "[reproduce] dump 缺失 —— 需要先用真实模型 forward 生成(慢,需 GPU)。"
  echo "  对每个 ckpt 跑(详细见 WAM_COT_README §12.5):"
  echo "    NGPU=1 CONFIG_NAME=robotwin_train CUDA_VISIBLE_DEVICES=0 MASTER_PORT=29540 \\"
  echo "      bash script/run_va_posttrain.sh --probe-ckpt <ckpt> \\"
  echo "      --probe-collect $PROBE_DIR/<tag>.pt --probe-collect-batches 200"
  exit 1
fi

mkdir -p "$OUT_DIR"

# ============ sha256 验证(若有 canonical) ============
if [ -f "$CANON" ]; then
  echo "[reproduce] sha256 校验(对照 $CANON):"
  python - <<PY
import json, hashlib, os, sys
canon = json.load(open("$CANON", "r", encoding="utf-8"))
exp_sha = canon.get("dumps_sha256", {}) or {}
pd = "$PROBE_DIR"
ok = True
for tag in "$TAGS".split():
    fp = os.path.join(pd, tag + ".pt")
    h = hashlib.sha256(open(fp, "rb").read()).hexdigest()
    e = exp_sha.get(tag + ".pt", "")
    if not e:
        print(f"  {tag+'.pt':22s} {h}  (canonical 未冻结)")
        continue
    mark = "OK " if h == e else "DIFF"
    if h != e: ok = False
    print(f"  [{mark}] {tag+'.pt':22s} {h}  (expected {e})")
sys.exit(0 if ok else 0)  # 不阻断,只警告
PY
else
  echo "[reproduce] 无 canonical($CANON),跳过 sha256;只比 val_acc。"
  echo "             首次冻结实验时请跑:bash script/freeze_probe.sh"
fi

# ============ 跑探针 ============
for tag in $TAGS; do
  echo
  echo "================  probe: $tag  ================"
  python "$REPO/evaluation/robotwin/latent_probe.py" \
    --config robotwin_train \
    --features h_hidden --label vlm_stage \
    --hidden-dump "$PROBE_DIR/$tag.pt" \
    --out-dir "$OUT_DIR/$tag" --seed $SEED
done

# ============ actual vs expected 对照表 ============
echo
echo "============================  对照表  ============================"
python - <<PY
import json, os
OUT_DIR = "$OUT_DIR"
TAGS = "$TAGS".split()
TOL = float("$TOL")

canon_fp = "$CANON"
exp = {}
exp_train = {}
if os.path.exists(canon_fp):
    canon = json.load(open(canon_fp, "r", encoding="utf-8"))
    exp = (canon.get("expected_val_acc") or {})
    exp_train = (canon.get("expected_train_acc") or {})

# 默认 expected(本项目实测,见 §9.2 / §9.6,2026-05-20 测得)
DEFAULT_EXP = {
    "h_stock":      0.663,
    "h_kf":         0.666,
    "h_kfvlm":      0.782,
    "h_wrongstage": 0.623,
}
if not exp:
    exp = DEFAULT_EXP
    print("(canonical 缺,使用 README §9.2/§9.6 实测期望值)")

print(f"{'tag':16s} {'val_acc':>8s} {'expected':>8s} {'Δ':>8s} {'tol':>6s}  status")
all_ok = True
rows = []
for tag in TAGS:
    fp = os.path.join(OUT_DIR, tag, "results_robotwin_train_h_hidden_vlm_stage.json")
    if not os.path.exists(fp):
        print(f"{tag:16s} (no result file at {fp})")
        all_ok = False
        continue
    d = json.load(open(fp, "r", encoding="utf-8"))
    va = float(d.get("val_acc", -1))
    e  = float(exp.get(tag, -1)) if exp.get(tag) is not None else None
    if e is None:
        status = "(no expected)"
        delta_s = "--"
    else:
        delta = va - e
        delta_s = f"{delta:+.3f}"
        ok = abs(delta) <= TOL
        all_ok = all_ok and ok
        status = "PASS" if ok else "FAIL"
    es = "--" if e is None else f"{e:.3f}"
    print(f"{tag:16s} {va:8.3f} {es:>8s} {delta_s:>8s} {TOL:6.3f}  {status}")
    rows.append({"tag": tag, "val_acc": va, "expected": e,
                 "delta": (va-e if e is not None else None),
                 "tol": TOL, "ok": (None if e is None else abs(va-e) <= TOL)})

print()
if all_ok:
    print(f"==>  ALL PASS (within ±{TOL})")
else:
    print(f"==>  SOME FAIL — 偏差超 ±{TOL}。可能原因:")
    print("    - sklearn / numpy 版本差异(t-SNE 算法实现微变)")
    print("    - dump 文件不是本项目原版(sha256 应不匹配)")
    print("    - SEED 变更(默认 0;改换种子比较)")

# 把对比结果写盘
summary = os.path.join(OUT_DIR, "reproduce_summary.json")
with open(summary, "w", encoding="utf-8") as f:
    json.dump({"seed": $SEED, "tol": TOL, "all_ok": all_ok, "rows": rows},
              f, indent=2, ensure_ascii=False)
print(f"\n[reproduce] summary -> {summary}")
PY

echo
echo "[reproduce] 输出在 $OUT_DIR/"
echo "  - <tag>/results_*.json    完整 metrics(val_acc/per_class/混淆矩阵)"
echo "  - <tag>/probe_*.pt        训好的线性探针权重(供干预实验)"
echo "  - <tag>/tsne_*.png        t-SNE 阶段着色图"
echo "  - reproduce_summary.json  对照表(actual vs expected)"
