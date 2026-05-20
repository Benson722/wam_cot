# 26 夏令营 WAM-CoT 项目复现说明(从零起步)

> 本文档目标:任何人(考官/同行/未来的你)按本文从零跑通**全部**:
> 数据生成 → 训练 → 评测 → 离线探针 → 消融 → 日志归档。**只写命令 + 期望
> 输出**;原理与细节见 `WAM_COT_README.md`,训练专项见 `H200_TRAINING.md`。
>
> 三类复现路径任选:
> - 🟢 **最小复现**(无 GPU、~1 分钟):只验证 §6 探针消融数字 → 见 §10
> - 🟡 **评测复现**(4090 实例、~3 小时):用已训 ckpt 出 SR + dream_video → 见 §5
> - 🔴 **完全复现**(H200 + 4090、~10 小时):数据 → 训练 → 评测 → 探针全跑 → 见 §1–§9 顺序执行

---

## 目录

1. [环境](#1-环境)
2. [数据准备](#2-数据准备)
3. [训练](#3-训练3-个-ckptm1--m1v--m1v_wrong)
4. [推理日志(模型参数 + 计算开销自动打印)](#4-推理日志模型参数--计算开销自动打印)
5. [在线 RoboTwin 评测](#5-在线-robotwin-评测)
6. [离线探针消融(§6 必做消融)](#6-离线探针消融6-必做消融)
7. [三项正式消融(PDF 必做)](#7-三项正式消融pdf-必做)
8. [VLM 过程性评判(可选)](#8-vlm-过程性评判可选)
9. [产物位置一览](#9-产物位置一览)
10. [最小复现路径(考官 1 分钟,无 GPU)](#10-最小复现路径考官-1-分钟无-gpu)
11. [常见问题速查](#11-常见问题速查)

---

## 1. 环境

### 1.1 两台实例(分别用途)

| 实例 | 用途 | GPU | 外网 | venv |
|---|---|---|---|---|
| **H200** | **训练 / 数据生成 / 离线探针** | 8 × H200 (141 GB) | 否 | `/inspire/qb-ilm2/project/26summer-camp-11/26220077/lingbot-va/.venv` |
| **4090**(镜像 `26summer-robocasa:260516`) | **在线 RoboTwin 评测**(sapien 仿真在此) | 1 × 4090 (48 GB) | 是 | 镜像默认 Python(含 sapien + RoboTwin) |

### 1.2 仓库 / 路径速查(SII 共享 /inspire 文件系统)

```bash
# 主仓库(训练 / 通用代码)
REPO=/inspire/hdd/project/26summer-camp-11/26220077/lingbot-va

# LingBot venv(含 torch 2.9 / diffusers / FSDP / wandb / safetensors / sklearn ...)
LINGBOT_VENV=/inspire/qb-ilm2/project/26summer-camp-11/26220077/lingbot-va/.venv

# RoboTwin 仓库(4090 实例,sapien 仿真在这)
ROBOTWIN=/inspire/qb-ilm2/project/26summer-camp-11/public/group3/RoboTwin

# EVAL_ENV(latent 评测原 reference,本项目不动它,只 cp 入 wrapper 见 §5)
EVAL_ENV=/inspire/qb-ilm2/project/26summer-camp-11/public/group3/eval_env/sii_wam_cot/lingbot-va_goal_cond_cot

# 数据集(12 任务主集 + 4 任务 _latsup 子集)
DS_MAIN=/inspire/qb-ilm2/project/26summer-camp-11/public/group3/lingbot-robotwin-clean-and-aug-lerobot/lerobot_robotwin_eef_aug_500_stable
DS_LATSUP=/inspire/qb-ilm2/project/26summer-camp-11/public/group3/lingbot-robotwin-clean-and-aug-lerobot/lerobot_robotwin_eef_aug_500_stable_latsup

# 基座 ckpt(官方 LingBot-VA RoboTwin posttrain)
BS=$REPO/checkpoints/lingbot-va-posttrain-robotwin

# 本项目训出的 ckpt(我们的 3 个交付)
M1_CKPT=$REPO/train_out/checkpoints/checkpoint_step_1200                              # kf-only
M1V_CKPT=$REPO/train_out/checkpoints/robotwin_kf0.1_vlmstage0.1/checkpoint_step_1200  # kf + VLM (主交付)
M1V_WRONG_CKPT=$REPO/train_out/checkpoints/robotwin_kf0.1_vlmstage0.1_WRONG/checkpoint_step_200
```

### 1.3 一次性环境准备(H200 第一次部署时跑一遍)

```bash
# 训练输出目录软链到 qb-ilm2 大盘(防 hdd 11G 配额爆,~50 GB ckpt+wandb+logs)
ln -sfn /inspire/qb-ilm2/project/26summer-camp-11/26220077/lingbot-va/train_out \
        $REPO/train_out

# 公开 venv 装 imageio(qwen_stage_annotate.py 解码视频用)
/inspire/qb-ilm2/project/26summer-camp-11/.venv/bin/pip install imageio imageio-ffmpeg Pillow

# 验证基座 ckpt 4 子目录齐全
ls -d $BS/{vae,tokenizer,text_encoder,transformer}
```

---

## 2. 数据准备

> 主数据集 12 任务已有 keyframes.jsonl + stages.jsonl(本项目已生成);
> 若要**从零重做**,按下面两步。

### 2.1 关键帧标注(Latent-CoT #1,M1/M1v/WRONG 都需要,~5 分钟)

```bash
python $REPO/evaluation/robotwin/keyframe_annotate.py \
  --dataset $DS_MAIN \
  --recursive --gripper-idx 7 15
# 产出 12 个 meta/keyframes.jsonl
```

### 2.2 Phase B VLM 阶段标注(M1v/WRONG 需要,M1 不需要,~1 小时,**8 GPU 并行**)

```bash
LOG=$REPO/train_out
mkdir -p "$LOG"

# 起 8 个 serve_qwen(GPU 0–7,端口 8000–8007)
for k in $(seq 0 7); do
  CUDA_VISIBLE_DEVICES=$k PORT=$((8000+k)) nohup \
    /inspire/qb-ilm2/project/26summer-camp-11/.venv/bin/python \
    /inspire/qb-ilm2/project/26summer-camp-11/serve_qwen.py \
    > "$LOG/serveqwen_gpu${k}.log" 2>&1 &
  sleep 2
done
for k in $(seq 0 7); do
  until grep -q "Application startup complete" "$LOG/serveqwen_gpu${k}.log" 2>/dev/null; do sleep 3; done
  echo "server $k ready"
done

# 8 分片客户端
for k in $(seq 0 7); do
  nohup python $REPO/evaluation/robotwin/qwen_stage_annotate.py \
    --dataset "$DS_MAIN" --recursive --frames 4 --max-tokens 256 --timeout 120 --resume \
    --num-shards 8 --shard $k --base-url http://127.0.0.1:$((8000+k))/v1 \
    > "$LOG/stage_shard${k}.log" 2>&1 &
  sleep 1
done
tail -f "$LOG"/stage_shard*.log    # 等 pgrep -fa qwen_stage_annotate.py 空 = 全跑完

# 收尾去重 + 关 8 server
python - <<'PY'
import json, glob, os
DS="/inspire/qb-ilm2/project/26summer-camp-11/public/group3/lingbot-robotwin-clean-and-aug-lerobot/lerobot_robotwin_eef_aug_500_stable"
for sj in sorted(glob.glob(os.path.join(DS,"*","meta","stages.jsonl"))):
    raw=[json.loads(l) for l in open(sj) if l.strip()]
    uniq={int(r["episode_index"]): r for r in raw}
    n_ep=sum(1 for _ in open(os.path.join(os.path.dirname(sj),"episodes.jsonl")))
    with open(sj,"w") as f:
        for ei in sorted(uniq): f.write(json.dumps(uniq[ei])+"\n")
    print(f"{os.path.basename(os.path.dirname(os.path.dirname(sj))):55s} {len(raw)} -> {len(uniq)} / {n_ep}")
PY
pkill -f serve_qwen.py
```

**期望**:12 任务每个 `500 -> 500 / 500`,无缺失。

---

## 3. 训练(3 个 ckpt:M1 / M1v / M1v_WRONG)

> 详见 `H200_TRAINING.md`。每个 ~2 小时到 step 1200(`save_interval=200`),
> **按一次 Ctrl-C 安全存档退出**。三个串行 ≈ 6 小时。

### 3.1 M1(baseline,仅 kf 辅助头)

先在 `wan_va/configs/va_robotwin_train_cfg.py` 临时关闭 vlm_stage_aux:
```python
va_robotwin_train_cfg.vlm_stage_aux    = False
va_robotwin_train_cfg.vlm_stage_weight = 0.0
```
然后:
```bash
cd $REPO && \
NGPU=8 CONFIG_NAME=robotwin_train CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
MASTER_PORT=29533 \
  bash script/run_va_posttrain.sh
# 期望: latent_loss ~0.12, action_loss ~1e-3, kf_loss ~2e-3 (stage_loss=0,被关掉)
# Ctrl-C → train_out/checkpoints/robotwin_kf0.1/checkpoint_step_1200/
# (注:本项目历史命名为 checkpoint_step_1200 直接放 train_out/checkpoints/)
```
事后**改回 `vlm_stage_aux=True`**(给 M1v 用)。

### 3.2 M1v(主交付,kf + VLM 阶段双辅助头)

确认配置默认开启 kf_aux + vlm_stage_aux,然后:
```bash
cd $REPO && \
NGPU=8 CONFIG_NAME=robotwin_train CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
MASTER_PORT=29534 \
  bash script/run_va_posttrain.sh
# 期望: 四 loss 同时下降, stage_loss 收敛 ~0.03 (vs chance 0.208)
# Ctrl-C → train_out/checkpoints/robotwin_kf0.1_vlmstage0.1/checkpoint_step_1200/
```

### 3.3 M1v_WRONG(Ablation-3,kf + 错误 VLM 阶段)

用专用配置 `robotwin_train_wrongstage`(`vlm_stage_corrupt='shuffle'`):
```bash
cd $REPO && \
NGPU=8 CONFIG_NAME=robotwin_train_wrongstage CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
MASTER_PORT=29535 \
  bash script/run_va_posttrain.sh
# 关键期望: stage_loss **完全卡在 ~0.208** (chance 0.1×ln 8) 不下降 (= Ablation 起效)
# kf_loss 正常下降到 ~2e-3 (kf 信号完好,只 VLM 阶段被毁)
# 等 step 200 或 1200 → Ctrl-C → train_out/checkpoints/robotwin_kf0.1_vlmstage0.1_WRONG/checkpoint_step_<N>/
```

---

## 4. 推理日志(模型参数 + 计算开销自动打印)

**所有 server 启动时都会自动打**一份"模型规模 + 计算开销"报表(从 safetensors
metadata 算,~100ms,不加载权重)。两条路径:

| Server 实现 | 谁打? |
|---|---|
| 主仓库 `wan_va/wan_va_server.py`(M0 用 / `launch_server.sh`)| `_log_param_counts` 在 `__init__` 内 inline 打 |
| EVAL_ENV `wan_va_server_predvideo.py`(M1/M1v dream_video,**EVAL_ENV 源码不动**)| 我们 wrapper `script/launch_server_pred_latent.sh` 在 cd EVAL_ENV 之前 pre-flight 调 `script/print_model_params.py` |

两条路径输出**完全一致**:
```
================  Model Parameter Counts  ================
  TAG  : M1v
  ckpt : /inspire/hdd/.../checkpoints/robotwin_kf0.1_vlmstage0.1/checkpoint_step_1200
  Wan2.2 VAE                : 247.0 M   (   247,000,000)   [1 file(s)]
  UMT5 Text Encoder         :  5.50 B   ( 5,500,000,000)   [2 file(s)]
  Transformer backbone      :  1.80 B   ( 1,799,212,279)
  + kf_aux_head (Latent #1) : 393.3 K   (       393,345)
  + stage_head  (Phase B)   : 394.4 K   (       394,376)
  ──────────────────────────────────────────────────────────
  Transformer (subtotal)    :  1.80 B   ( 1,800,000,000)   [1 file(s)]
  TOTAL (VAE + UMT5 + Xfmr) :  7.55 B   ( 7,547,000,000)
==========================================================
================  Compute Cost (estimates)  ==============
  Memory footprint(仅权重,不含 KV cache / activations / 梯度):
    bf16/fp16 (inference): VAE 471 MB | UMT5 10.24 GB | Xfmr 3.35 GB | TOTAL 14.06 GB
    fp32      (training) : VAE 942 MB | UMT5 20.49 GB | Xfmr 6.71 GB | TOTAL 28.11 GB
  Forward FLOPs(Kaplan ≈ 6 × P × N_tokens):
    Transformer / token                : 10.80 GFLOPs
    Transformer / forward (~1500 tok)  : 16.20 TFLOPs
    Transformer / episode (~100 chunk) :  1.62 PFLOPs
==========================================================
```

**手动单跑**(任何 ckpt,无 server,~100 ms):
```bash
python $REPO/script/print_model_params.py --ckpt $M1V_CKPT --tag M1v
python $REPO/script/print_model_params.py --ckpt $M1_CKPT  --tag M1
python $REPO/script/print_model_params.py --ckpt $BS       --tag stock
```

---

## 5. 在线 RoboTwin 评测(4090 实例,~3 小时全跑)

详 `H200_TRAINING.md` 不涵盖,见此处。两套路径:

### 5.1 两步式手动(最稳,推荐)

**终端 1 — M1 server**:
```bash
TAG=M1 bash $REPO/script/launch_server_pred_latent.sh
# 等 "Model Parameter Counts" + "server listening on 0.0.0.0:29056" 出现 → 就绪
```

**终端 2 — M1 client 6 任务循环**:
```bash
for t in handover_block handover_mic hanging_mug blocks_ranking_size beat_block_hammer lift_pot; do
  TAG=M1 TASK=$t TEST_NUM=10 PORT=29056 \
    bash $REPO/script/launch_client_latent.sh
done
```
跑完 → 终端 1 按一次 `Ctrl-C` 关 server。

**M1v(换端口)**:
```bash
# 终端 1
TAG=M1v START_PORT=29066 MASTER_PORT=29071 \
  bash $REPO/script/launch_server_pred_latent.sh
# 终端 2 (PORT 同步换)
for t in handover_block handover_mic hanging_mug blocks_ranking_size beat_block_hammer lift_pot; do
  TAG=M1v TASK=$t TEST_NUM=10 PORT=29066 \
    bash $REPO/script/launch_client_latent.sh
done
```

### 5.2 一键编排(自动两个 ckpt × 6 任务)

```bash
TEST_NUM=10 bash $REPO/script/eval_route2_latent_cot.sh
# 末尾自动打 SR 对照表 + JSON
```

### 5.3 SR 汇总

```bash
ROBOTWIN=/inspire/qb-ilm2/project/26summer-camp-11/public/group3/RoboTwin
python - <<'PY'
import os, glob, json
ROBOTWIN = "/inspire/qb-ilm2/project/26summer-camp-11/public/group3/RoboTwin"
TAGS  = ["M1", "M1v"]
TASKS = "handover_block handover_mic hanging_mug blocks_ranking_size beat_block_hammer lift_pot".split()
rows = {tag: {} for tag in TAGS}
for tag in TAGS:
    for t in TASKS:
        fs = sorted(glob.glob(os.path.join(ROBOTWIN, "eval_result", t, "ACT",
                                           "demo_clean", tag, "*", "_result.txt")))
        if not fs: rows[tag][t] = None; continue
        lines = [l.strip() for l in open(fs[-1]) if l.strip()]
        rows[tag][t] = float(lines[-1]) if lines else None
w=[26,10,12,12]; hdr=["Task","M1","M1v","Δ(M1v-M1)"]
print("| " + " | ".join(f"{h:<{w[i]}s}" for i,h in enumerate(hdr)) + " |")
print("|" + "|".join("-"*(c+2) for c in w) + "|")
for t in TASKS:
    m1, m1v = rows["M1"].get(t), rows["M1v"].get(t)
    s1="--" if m1 is None else f"{m1:.3f}"; s2="--" if m1v is None else f"{m1v:.3f}"
    d="--" if (m1 is None or m1v is None) else f"{m1v-m1:+.3f}"
    print("| " + " | ".join(f"{c:<{w[i]}s}" for i,c in enumerate([t,s1,s2,d])) + " |")
PY
```

---

## 6. 离线探针消融(§6 必做消融)

> Latent-CoT #4 冻结-backbone 线性探针。两步:① collect h_t dump(慢,GPU)
> → ② 跑线性探针(快,CPU,确定性)。

### 6.1 Collect h_t dump(对每个 ckpt 跑一次,~5 min/个)

```bash
cd $REPO

# stock(无 CoT 基座)
NGPU=1 CONFIG_NAME=robotwin_train CUDA_VISIBLE_DEVICES=0 MASTER_PORT=29540 \
  bash script/run_va_posttrain.sh \
  --probe-ckpt $BS \
  --probe-collect ./train_out/probe/h_stock.pt --probe-collect-batches 200

# M1 (kf-only)
NGPU=1 CONFIG_NAME=robotwin_train CUDA_VISIBLE_DEVICES=0 MASTER_PORT=29541 \
  bash script/run_va_posttrain.sh \
  --probe-ckpt $M1_CKPT \
  --probe-collect ./train_out/probe/h_kf.pt --probe-collect-batches 200

# M1v (kf + VLM)
NGPU=1 CONFIG_NAME=robotwin_train CUDA_VISIBLE_DEVICES=0 MASTER_PORT=29542 \
  bash script/run_va_posttrain.sh \
  --probe-ckpt $M1V_CKPT \
  --probe-collect ./train_out/probe/h_kfvlm.pt --probe-collect-batches 200

# M1v_WRONG (Ablation-3)
NGPU=1 CONFIG_NAME=robotwin_train CUDA_VISIBLE_DEVICES=0 MASTER_PORT=29543 \
  bash script/run_va_posttrain.sh \
  --probe-ckpt $M1V_WRONG_CKPT \
  --probe-collect ./train_out/probe/h_wrongstage.pt --probe-collect-batches 200
```

### 6.2 跑探针(任何机器,无 GPU,~1 min)

```bash
for tag in h_stock h_kf h_kfvlm h_wrongstage; do
  python $REPO/evaluation/robotwin/latent_probe.py --config robotwin_train \
    --features h_hidden --label vlm_stage \
    --hidden-dump $REPO/train_out/probe/$tag.pt \
    --out-dir $REPO/train_out/probe/out_$tag \
    --seed 0
done
```

### 6.3 冻结 canonical + 复现校验

```bash
bash $REPO/script/freeze_probe.sh        # 把 sha256+expected 写 probe_canonical.json
bash $REPO/script/reproduce_probe.sh     # 4/4 PASS Δ=+0.000 = 全对
```

**期望数字**(seed=0,canonical):
| ckpt | val_acc | 与 stock 差 |
|---|---|---|
| stock | 0.652 | 0 |
| kf (M1) | 0.648 | −0.004 |
| **kf+VLM (M1v)** | **0.778** | **+0.126** |
| wrongstage | 0.638 | −0.014(错误监督有害) |

---

## 7. 三项正式消融(PDF 必做)

### 7.1 Ablation-1 + 2:显式 CoT(4090,无需重训,~3 h)

```bash
TEST_NUM=10 bash $REPO/script/run_ablation_explicit.sh
# 一个 M1 server + 6 任务 × 3 档客户端 (cot_full / no_cot / shuffle_subtasks)
# VLM 自动接 Qwen3-VL-4B-Instruct 公网端点
# 末尾自动出 ΔA1 / ΔA2 对照表 + train_out/ablation_explicit/ablation_explicit_summary.json
```

### 7.2 Ablation-3:错误标记(隐式,H200 重训 + 4090 评测,~5 h)

```bash
# Phase TRAIN (H200, ~2 h 到 step 1200)
PHASE=train bash $REPO/script/run_ablation_implicit.sh
# Ctrl-C 后存到 train_out/checkpoints/robotwin_kf0.1_vlmstage0.1_WRONG/

# Phase PROBE (任何机, ~15 min)
PHASE=probe bash $REPO/script/run_ablation_implicit.sh
# 末尾自动打 4 ckpt 对照表 (stock/kf/kfvlm/wrongstage)

# Phase EVAL (4090, ~1.5 h)
PHASE=eval bash $REPO/script/run_ablation_implicit.sh
```

---

## 8. VLM 过程性评判(可选,4090)

VLM 看真实 rollout 视频逐子目标打 0–1 分(env SR 二值之外的过程性评估):

```bash
# 一次性装依赖
pip install openai httpx imageio imageio-ffmpeg Pillow numpy

# 全量(~15-30 min)
python $REPO/evaluation/robotwin/judge_completion.py \
  --log-root /inspire/qb-ilm2/project/26summer-camp-11/public/group3/RoboTwin/outputs_infonce/log \
  --frames 8 --resume

# 查看汇总
cat /inspire/qb-ilm2/project/26summer-camp-11/public/group3/RoboTwin/outputs_infonce/log/judge/summary.json
```

---

## 9. 产物位置一览

| 类型 | 位置 |
|---|---|
| 训练 ckpt(M1) | `$REPO/train_out/checkpoints/checkpoint_step_1200/transformer/` |
| 训练 ckpt(M1v) | `$REPO/train_out/checkpoints/robotwin_kf0.1_vlmstage0.1/checkpoint_step_1200/transformer/` |
| 训练 ckpt(M1v_WRONG) | `$REPO/train_out/checkpoints/robotwin_kf0.1_vlmstage0.1_WRONG/checkpoint_step_<N>/transformer/` |
| 训练 ckpt 元信息 | 同上目录的 `meta.json`(自描述 exp_tag/base/λ/timestamp 等) |
| wandb 离线日志 | `$REPO/train_out/wandb/wandb/offline-run-*` |
| 探针 dump (~140 MB) | `$REPO/train_out/probe/h_{stock,kf,kfvlm,wrongstage}.pt` |
| 探针结果 + t-SNE | `$REPO/train_out/probe/out_h_<tag>/{results,probe,tsne}_*.{json,pt,png}` |
| 探针 canonical | `$REPO/train_out/probe/probe_canonical.json` |
| RoboTwin SR 文本 | `$ROBOTWIN/eval_result/<task>/ACT/demo_clean/<tag>/<ts>/_result.txt` |
| RoboTwin sapien 视频 | `$ROBOTWIN/eval_result/<task>/ACT/demo_clean/<tag>/<ts>/episode*.mp4` |
| dream_video(M1/M1v latent eval) | `$EVAL_ENV/visualization_predvideo_<tag>/` 和 `$ROBOTWIN/outputs_latent_<tag>/` |
| 想象 vs 真实对比 | `$ROBOTWIN/results_latent_<tag>/stseed-*/visualization/<task>/*_True\|False.mp4` |
| Ablation-1/2 SR 汇总 | `$REPO/train_out/ablation_explicit/ablation_explicit_summary.json` |
| 完整探针归档(分发用) | `$REPO/train_out/archive/wam_cot_probe_full_<ts>.tgz` + `README_<ts>.md`(62 MB tarball) |

---

## 10. 最小复现路径(考官 1 分钟,无 GPU)

只验证 §6 探针消融的全部数字 ——**任何机器,无 GPU,~1 分钟**:

```bash
# 1) 获取 archive tarball (~62 MB,共享盘 / 邮件 / 网盘任你选)
wget <wherever you uploaded the tarball>   # 或:cp /inspire/.../archive/wam_cot_probe_full_<ts>.tgz .

# 2) 解压
TGT=/some/path
mkdir -p $TGT
tar xzf wam_cot_probe_full_*.tgz -C $TGT
ls -R $TGT/probe   # 应见 h_*.pt × 4 + out_h_* × 4 + probe_canonical.json + reproduce_out/

# 3) 装最小依赖(无 torch GPU,纯 CPU 即可)
pip install numpy torch scikit-learn matplotlib safetensors

# 4) clone 仓库 + 跑复现脚本
git clone <项目 repo url>
cd <repo>
PROBE_DIR=$TGT/probe bash script/reproduce_probe.sh

# 期望末尾输出:
#   tag               val_acc expected        Δ    tol  status
#   h_stock             0.652    0.652   +0.000  0.010  PASS
#   h_kf                0.648    0.648   +0.000  0.010  PASS
#   h_kfvlm             0.778    0.778   +0.000  0.010  PASS
#   h_wrongstage        0.638    0.638   +0.000  0.010  PASS
#   ==>  ALL PASS (within ±0.01)
```

**这就是 PDF "必做消融" 的硬证据**,字节级可复现,无需 GPU、无需 RoboTwin 仿真、无需重新训练。

---

## 11. 常见问题速查

| 现象 | 原因 | 解决 |
|---|---|---|
| `EADDRINUSE port 29501/29061` | torchrun MASTER_PORT 残留 | `pkill -9 -f 'wan_va\.train\|torch\.distributed\.run'; sleep 3` 或换 `MASTER_PORT=29533` |
| `Disk quota exceeded` 训到 step 80 崩 | `/inspire/hdd/.../26220077` 11G 满 | `§1.3 (a)` 软链 `train_out` 到 qb-ilm2;`run_va_posttrain.sh` 已设 HF cache 到大盘 |
| `libcudnn_graph.so.9: undefined symbol cudnnGetLibConfig` SIGABRT | 系统 cuDNN 覆盖 torch 自带 | `run_va_posttrain.sh` 已内嵌 LD_PRELOAD 修复(全自动) |
| `qwen_stage_annotate` 全 SKIP `No module named imageio` | 当前 venv 没装 imageio | `§1.3` 给公开 venv 装,或切 LingBot venv |
| `Cannot copy out of meta tensor` | 新加 head 留 meta | `load_transformer` 已修(扫子模块 to_empty + reset_parameters) |
| client `No module named eval_polict_client_openpi_latent` | 当前 cwd 不在 EVAL_ENV / 路径错 | 用 `script/launch_client_latent.sh` 自动 cd EVAL_ENV |
| 4090 无 `ss` 命令 | iproute2 未装 | wrapper 已改用 bash 内建 `/dev/tcp/`(免 ss) |
| `Ctrl-C` 训练后 ckpt 损坏 | 按了两次 Ctrl-C 硬杀 | **只按一次**,等日志显示 `Interrupt: saving checkpoint ... then exiting.` |
| 探针跨次跑数字漂 ~0.02 | `_train_probe` nn.Linear init 未受 seed 控制 | `latent_probe.py` 已修(`torch.manual_seed(args.seed)`),`reproduce_probe.sh` 4/4 PASS Δ=+0.000 |

---

## 文档关系

- `WAM_COT_README.md` — **项目交付主 README**(全面,1800+ 行)
- `H200_TRAINING.md` — 训练专项快速参考
- `EXPERIMENT_RESULTS.md` — 主 SR 表 + 探针消融完整结果
- `MODEL_AND_DATA.md` — 模型架构 + 数据 + 损失数学详细推导
- `TEAM_ROLES.md` — 团队分工
- `latent_plan*.md` — 设计思路与进度
- **本文档** — **复现说明**(从零起步全流程,只写命令)

---

## 一句话最后总结

**最小复现**:`bash script/reproduce_probe.sh` ~1 分钟无 GPU 4/4 PASS。
**评测复现**(4090,~3 h):`bash script/eval_route2_latent_cot.sh`。
**完整复现**(H200+4090,~10 h):按 §2 → §3 → §5 → §6 → §7 顺序执行。
