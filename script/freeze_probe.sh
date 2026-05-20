#!/bin/bash
# ============================================================================
#  freeze_probe.sh —— 冻结探针消融的"标准答案"(供后续 reproduce_probe.sh 校验)
# ----------------------------------------------------------------------------
#  当你已经跑出满意的 §6 探针消融结果(train_out/probe/out_h_*/results_*.json
#  存在,且实测数 ≈ §9.2/§9.6 的 0.663/0.666/0.782/0.623),跑这个脚本把:
#    - 每个 h_*.pt 的 sha256
#    - 每个 results_*.json 的 val_acc / train_acc
#  写到 train_out/probe/probe_canonical.json。reproduce_probe.sh 后续会读它。
#
#  使用:
#    bash script/freeze_probe.sh                  # 首次冻结
#    SEED=0 bash script/freeze_probe.sh           # 显式记录种子
# ============================================================================

REPO=${REPO:-/inspire/hdd/project/26summer-camp-11/26220077/lingbot-va}
PROBE_DIR=${PROBE_DIR:-$REPO/train_out/probe}
SEED=${SEED:-0}
LINGBOT_VENV=${LINGBOT_VENV:-/inspire/qb-ilm2/project/26summer-camp-11/26220077/lingbot-va/.venv}
CANON="$PROBE_DIR/probe_canonical.json"

# 激活 venv(latent_probe 需 numpy/sklearn/torch/matplotlib)
if [ -f "$LINGBOT_VENV/bin/activate" ]; then
  # shellcheck disable=SC1091
  source "$LINGBOT_VENV/bin/activate"
fi

TAGS="h_stock h_kf h_kfvlm h_wrongstage"

# 预检 dump
for tag in $TAGS; do
  fp=$PROBE_DIR/$tag.pt
  if [ ! -f "$fp" ]; then echo "[freeze] MISSING: $fp"; exit 1; fi
done

# 关键:**用当前代码 + SEED 跑一次 latent_probe**,覆盖旧 out_h_*/results_*.json
# 这样 freeze 写入的 expected = 之后 reproduce 拿到的同一来源,真正 PASS。
echo "[freeze] 用 SEED=$SEED 重跑 latent_probe(确保 canonical 与 reproduce 同源)..."
for tag in $TAGS; do
  python "$REPO/evaluation/robotwin/latent_probe.py" \
    --config robotwin_train \
    --features h_hidden --label vlm_stage \
    --hidden-dump "$PROBE_DIR/$tag.pt" \
    --out-dir "$PROBE_DIR/out_$tag" \
    --seed "$SEED" > /tmp/_freeze_probe_$tag.log 2>&1
  if [ $? -ne 0 ]; then
    echo "[freeze] FAIL on $tag:"
    tail -n 30 /tmp/_freeze_probe_$tag.log
    exit 1
  fi
  va=$(python -c "import json; d=json.load(open('$PROBE_DIR/out_$tag/results_robotwin_train_h_hidden_vlm_stage.json')); print(f\"{d['val_acc']:.3f}\")")
  echo "  [refresh] $tag: val_acc=$va"
done

python - <<PY
import json, hashlib, os
PROBE_DIR = "$PROBE_DIR"
CANON     = "$CANON"
SEED      = int("$SEED")
TAGS      = "$TAGS".split()

dumps_sha = {}
exp_val_acc = {}
exp_train_acc = {}
n_samples = {}
n_episodes = {}
ckpt_of = {}
for tag in TAGS:
    pt = os.path.join(PROBE_DIR, tag + ".pt")
    js = os.path.join(PROBE_DIR, "out_" + tag,
                      "results_robotwin_train_h_hidden_vlm_stage.json")
    h = hashlib.sha256(open(pt, "rb").read()).hexdigest()
    dumps_sha[tag + ".pt"] = h
    d = json.load(open(js, "r", encoding="utf-8"))
    exp_val_acc[tag]   = float(d["val_acc"])
    exp_train_acc[tag] = float(d["train_acc"])
    n_samples[tag]     = int(d["n_samples"])
    n_episodes[tag]    = int(d["n_episodes"])
    ckpt_of[tag]       = d.get("dataset_path", "?")
    print(f"[freeze] {tag}: val_acc={d['val_acc']:.3f} "
          f"train_acc={d['train_acc']:.3f} sha={h[:16]}...")

canon = {
    "description": "Latent-CoT #4 offline probe ablation - canonical reference",
    "label": "vlm_stage",
    "num_classes_actual": 6,
    "chance": 0.16667,
    "split_seed": SEED,
    "tolerance": 0.01,
    "n_samples": n_samples,
    "n_episodes": n_episodes,
    "ckpt_paths": ckpt_of,
    "expected_val_acc": exp_val_acc,
    "expected_train_acc": exp_train_acc,
    "dumps_sha256": dumps_sha,
    "note": ("val_acc 容忍 ±0.01;探针训练对 SEED+输入字节级确定,跨 sklearn/"
             "numpy 版本可能有 <0.005 抖动。"),
}
with open(CANON, "w", encoding="utf-8") as f:
    json.dump(canon, f, indent=2, ensure_ascii=False)
print(f"\n[freeze] wrote {CANON}")
PY

echo
echo "[freeze] done. 之后 bash script/reproduce_probe.sh 会自动:"
echo "  - sha256 校验 h_*.pt 是否被改"
echo "  - 重跑 latent_probe.py 并与 expected_val_acc 对比"
