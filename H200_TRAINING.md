# H200 训练命令(WAM-CoT 路线二,Latent CoT)

> 本文档**专注训练**:从数据前置到三个 ckpt 出炉的完整命令链。
> 评测在 4090,见 `script/launch_server_pred_latent.sh` 等;表/分析见
> `WAM_COT_README.md` §9–§10、`EXPERIMENT_RESULTS.md`。

---

## 0. 总览:三个模型 = 三个配置 + 一条命令

| 模型 | 配置 (`CONFIG_NAME=`) | 辅助监督 | 输出目录(`exp_name` 自动派生) | 用途 |
|---|---|---|---|---|
| **M1**(baseline) | `robotwin_train` + 配置里**临时**关掉 vlm_stage_aux | 仅 kf 关键帧距离 | `train_out/checkpoints/robotwin_kf0.1/checkpoint_step_<N>/`(本项目里历史命名为 `checkpoint_step_1200`)| Latent-CoT #1 baseline |
| **M1v**(main) | `robotwin_train`(默认) | kf + VLM 语义阶段 | `train_out/checkpoints/robotwin_kf0.1_vlmstage0.1/checkpoint_step_<N>/` | Latent-CoT Phase B,主交付 |
| **M1v_WRONG**(Ablation-3)| `robotwin_train_wrongstage` | kf + **错误置乱**的 VLM 阶段 | `train_out/checkpoints/robotwin_kf0.1_vlmstage0.1_WRONG/checkpoint_step_<N>/` | 必做消融:错误标记 |

**机器**:H200 实例(8 卡,无外网),仓库 `/inspire/hdd/project/26summer-camp-11/26220077/lingbot-va`,venv `/inspire/qb-ilm2/project/26summer-camp-11/26220077/lingbot-va/.venv`。
**单次时长**:~2 小时收敛到 step 1200(`save_interval=200`,Ctrl-C 安全存档)。
**总训练时长**:M1(~2 h) + M1v(~2 h) + M1v_WRONG(~2 h) = **~6 小时**(M1 已存在则跳过第一项)。

---

## 1. 一次性环境准备

```bash
# (a) 输出目录软链到 qb-ilm2 大盘(防 hdd 11G 配额爆;若已是软链可跳)
ln -sfn /inspire/qb-ilm2/project/26summer-camp-11/26220077/lingbot-va/train_out \
        /inspire/hdd/project/26summer-camp-11/26220077/lingbot-va/train_out
ls -ld /inspire/hdd/project/26summer-camp-11/26220077/lingbot-va/train_out
# 应输出 ... -> /inspire/qb-ilm2/.../train_out

# (b) 确保数据集软链 _stable 父目录存在(12 任务,本项目训练源)
ls /inspire/qb-ilm2/project/26summer-camp-11/public/group3/lingbot-robotwin-clean-and-aug-lerobot/lerobot_robotwin_eef_aug_500_stable
# 应见 12 个 *-aloha-agilex_randomized_500-1000 软链

# (c) 共享 empty_emb.pt(CFG null-prompt UMT5 嵌入)
ls -lh /inspire/qb-ilm2/project/26summer-camp-11/public/group3/lingbot-robotwin-clean-and-aug-lerobot/empty_emb.pt

# (d) 基座 ckpt(LingBot 官方 RoboTwin posttrain)
ls -d /inspire/hdd/project/26summer-camp-11/26220077/lingbot-va/checkpoints/lingbot-va-posttrain-robotwin/{vae,tokenizer,text_encoder,transformer}
# 4 子目录必须都在
```

---

## 2. 数据前置(M1 / M1v / M1v_WRONG 三者都依赖)

### 2.1 Latent-CoT #1 关键帧标注(快,无 VLM,M1 / M1v / M1v_WRONG 都需要)

```bash
python /inspire/hdd/project/26summer-camp-11/26220077/lingbot-va/evaluation/robotwin/keyframe_annotate.py \
  --dataset /inspire/qb-ilm2/project/26summer-camp-11/public/group3/lingbot-robotwin-clean-and-aug-lerobot/lerobot_robotwin_eef_aug_500_stable \
  --recursive --gripper-idx 7 15
# 产出: 每任务 meta/keyframes.jsonl, ~5 分钟全 12 任务搞定
```

### 2.2 Phase B VLM 阶段标注(M1v / M1v_WRONG 需要,M1 不需要)

8 GPU 并行(详 `WAM_COT_README.md §12.3`),~1 小时全 12 任务:

```bash
# 起 8 个 serve_qwen
LOG=/inspire/qb-ilm2/project/26summer-camp-11/26220077/lingbot-va/train_out
DS=/inspire/qb-ilm2/project/26summer-camp-11/public/group3/lingbot-robotwin-clean-and-aug-lerobot/lerobot_robotwin_eef_aug_500_stable
for k in $(seq 0 7); do
  CUDA_VISIBLE_DEVICES=$k PORT=$((8000+k)) nohup \
    /inspire/qb-ilm2/project/26summer-camp-11/.venv/bin/python \
    /inspire/qb-ilm2/project/26summer-camp-11/serve_qwen.py \
    > "$LOG/serveqwen_gpu${k}.log" 2>&1 &
  sleep 2
done
for k in $(seq 0 7); do
  until grep -q "Application startup complete" "$LOG/serveqwen_gpu${k}.log" 2>/dev/null; do sleep 3; done
done

# 8 shard 客户端
for k in $(seq 0 7); do
  nohup python /inspire/hdd/project/26summer-camp-11/26220077/lingbot-va/evaluation/robotwin/qwen_stage_annotate.py \
    --dataset "$DS" --recursive --frames 4 --max-tokens 256 --timeout 120 --resume \
    --num-shards 8 --shard $k --base-url http://127.0.0.1:$((8000+k))/v1 \
    > "$LOG/stage_shard${k}.log" 2>&1 &
  sleep 1
done
tail -f "$LOG"/stage_shard*.log     # 等 pgrep -fa qwen_stage_annotate.py 为空

# 跑完去重 + 关 server
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

**预期**:12 任务 × 500 ep,全部 `OK`,12 个 `stages.jsonl` 各 500 行。

---

## 3. 训练 M1(baseline,Latent-CoT #1,仅 kf 辅助头)

> M1 = 你历史上第一次训出的 `checkpoint_step_1200`(纯 kf-only)。若该 ckpt 已存在,本节可跳。

**临时关 VLM 阶段头**(M1 baseline 不要 VLM 监督)——改 `wan_va/configs/va_robotwin_train_cfg.py`:
```python
va_robotwin_train_cfg.vlm_stage_aux    = False     # ← M1 临时关
va_robotwin_train_cfg.vlm_stage_weight = 0.0       # ← 安全
```

启动训练:
```bash
cd /inspire/hdd/project/26summer-camp-11/26220077/lingbot-va && \
NGPU=8 CONFIG_NAME=robotwin_train CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
MASTER_PORT=29533 \
  bash script/run_va_posttrain.sh
```

进度条期望:`latent_loss ~0.12, action_loss ~1e-3, kf_loss 从 ~0.3 降到 ~2e-3`(stage_loss = 0,因为被禁用)。

收敛(step 1200)→ **按一次 Ctrl-C** → 自动安全存档退出 → ckpt 落:
```
train_out/checkpoints/robotwin_kf0.1/checkpoint_step_1200/transformer/{diffusion_pytorch_model.safetensors, config.json}
                                                          /meta.json
```
**事后必做**:把 `va_robotwin_train_cfg.py` 改回 `vlm_stage_aux=True`(为后面 M1v 用)。

---

## 4. 训练 M1v(主交付,Latent-CoT Phase B,kf + VLM 阶段双头)

确认 `va_robotwin_train_cfg.py` 默认:
```python
va_robotwin_train_cfg.kf_aux        = True
va_robotwin_train_cfg.kf_aux_weight = 0.1
va_robotwin_train_cfg.kf_file       = 'keyframes.jsonl'
va_robotwin_train_cfg.vlm_stage_aux    = True              # ← Phase B 开
va_robotwin_train_cfg.vlm_stage_weight = 0.1
va_robotwin_train_cfg.vlm_stage_file   = 'stages.jsonl'
va_robotwin_train_cfg.vlm_num_stages   = 8
va_robotwin_train_cfg.exp_name = None    # 自动派生 robotwin_kf0.1_vlmstage0.1
```

启动:
```bash
cd /inspire/hdd/project/26summer-camp-11/26220077/lingbot-va && \
NGPU=8 CONFIG_NAME=robotwin_train CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
MASTER_PORT=29534 \
  bash script/run_va_posttrain.sh
```

进度条期望(所有 4 项 loss 同时下降):
```
latent_loss ~0.12   action_loss ~1e-3   kf_loss ~2e-3   stage_loss ~0.03
                                                       (vs chance 0.208)
```

step 1200 → Ctrl-C 一次 → 存档落:
```
train_out/checkpoints/robotwin_kf0.1_vlmstage0.1/checkpoint_step_1200/
```

---

## 5. 训练 M1v_WRONG(Ablation-3:错误标记)

**完全不动 cfg**——用专用配置 `robotwin_train_wrongstage`(`wan_va/configs/va_robotwin_train_wrongstage_cfg.py`),继承 train_cfg 但加 `vlm_stage_corrupt='shuffle'`(dataset loader 在 emit `vlm_stage` 时做 per-episode deterministic 置换)。

启动:
```bash
cd /inspire/hdd/project/26summer-camp-11/26220077/lingbot-va && \
NGPU=8 CONFIG_NAME=robotwin_train_wrongstage CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
MASTER_PORT=29535 \
  bash script/run_va_posttrain.sh
# 或一键封装(等价):
# PHASE=train bash script/run_ablation_implicit.sh
```

进度条**关键预期**(Ablation-3 的成功标志):
- `stage_loss` 卡在 **~0.208**(理论 chance = `0.1 × ln 8`)**不下降** → 错误标签下信号被毁,模型完全学不到
- `kf_loss` 正常下降到 ~2e-3 → kf 监督完好,只 VLM 阶段被毁
- 与 M1v 对比可写在报告

step 1200(或保守 step 400 已稳)→ Ctrl-C 一次 → 存档落:
```
train_out/checkpoints/robotwin_kf0.1_vlmstage0.1_WRONG/checkpoint_step_<N>/
```

---

## 6. 监控、检查点、Ctrl-C 安全保存

### 6.1 训练时进度

- **stdout 进度条**(rank 0,torchrun 自动 redirect 到本终端):
  `latent_loss / action_loss / kf_loss / stage_loss / step / grad_norm / lr`
- **逐 rank 日志**:`train_out/torchrun_logs/<run>/<rank>/{stdout,stderr}.log`(rank>0 的真实 traceback 也落盘)
- **wandb 离线**:`train_out/wandb/wandb/offline-run-*`,事后:
  ```bash
  wandb sync /inspire/qb-ilm2/project/26summer-camp-11/26220077/lingbot-va/train_out/wandb/wandb/offline-run-*
  ```

### 6.2 检查点目录(都软链到 qb-ilm2 大盘)

```
train_out/checkpoints/
├── robotwin_kf0.1/checkpoint_step_<N>/       ← M1
├── robotwin_kf0.1_vlmstage0.1/checkpoint_step_<N>/        ← M1v
└── robotwin_kf0.1_vlmstage0.1_WRONG/checkpoint_step_<N>/  ← M1v_WRONG

每个 checkpoint_step_<N>/:
├── transformer/{diffusion_pytorch_model.safetensors, config.json}
└── meta.json   ← 自描述: exp_tag/base_ckpt/dataset/λ_kf/λ_st/lr/timestamp/...
```

`meta.json` 让目录"自证身份"——即便文件夹被移动 / 改名,打开 meta.json 就知道当年用什么 cfg 训的。

### 6.3 Ctrl-C 安全保存

`train.py` 注册了 SIGINT/SIGTERM 处理器:
1. 按一次 `Ctrl-C` → 只置标志位
2. 在**下个 optimizer step 边界**全 rank `all_reduce` 决定停 → 集体 `save_checkpoint()` → 干净退出
3. **不要按两次**(两次会硬杀,可能损坏 FSDP 集体写中的 .safetensors)

看到日志:
```
Interrupt: saving checkpoint at step N then exiting.
```
即安全。

### 6.4 提前停 vs 跑满 50000 步

- 收敛标准(看 `WAM_COT_README.md §7.7`):`action_loss ≈ 1e-3` 且 `kf_loss ≈ 2e-3` 且 `latent_loss` 触地板 ≈ 0.12 → 通常 step 1000-1200 已稳
- 跑满 50000 步**没必要**(~53h),浪费算力
- **推荐**:`save_interval=200`,等 step 1200 自动存档后 Ctrl-C 一次

---

## 7. 训练完跑探针自检(可选,~2 分钟)

跑完一个新 ckpt 后,立刻验证 backbone 表征有没有"学到 VLM 阶段":
```bash
# 收集 backbone hidden + 跑线性探针
NGPU=1 CONFIG_NAME=robotwin_train CUDA_VISIBLE_DEVICES=0 MASTER_PORT=29540 \
  bash script/run_va_posttrain.sh \
  --probe-ckpt <新 ckpt 路径> \
  --probe-collect ./train_out/probe/h_<tag>.pt --probe-collect-batches 200

python evaluation/robotwin/latent_probe.py --config robotwin_train \
  --features h_hidden --label vlm_stage \
  --hidden-dump ./train_out/probe/h_<tag>.pt \
  --out-dir ./train_out/probe/out_h_<tag>
```

期望 `val_acc`:
- M1 / kf-only:~0.65(几乎与 stock 持平,kf 与 VLM 阶段是正交信号)
- M1v / kf+VLM:**~0.78**(显著 +0.13,VLM 监督生效)
- M1v_WRONG:~0.64(比 stock 还低 0.01-0.02,错误监督有害)

详 §9.2 / §9.6 of `WAM_COT_README.md`。

---

## 8. 常见问题(踩过的坑)

| 现象 | 原因 | 解决 |
|---|---|---|
| `EADDRINUSE port 29501` | 上次 torchrun MASTER_PORT 残留 | `pkill -9 -f 'torch\.distributed\.run\|wan_va\.train'; sleep 3` 或换 `MASTER_PORT=29533` 等 |
| `Disk quota exceeded` 训到 step 80 崩 | `/inspire/hdd/.../26220077` 11G 配额满(ckpt+wandb+logs 写爆) | §1 (a) 软链 `train_out` 到 qb-ilm2 大盘;`run_va_posttrain.sh` 已设 `HF_HOME/HF_LEROBOT_HOME` 到大盘 |
| `libcudnn_graph.so.9: undefined symbol cudnnGetLibConfig` SIGABRT | 系统 `/usr/lib` 的 cuDNN 覆盖 torch 2.9 自带 cuDNN | `run_va_posttrain.sh` 已内嵌 LD_PRELOAD 修复;若关掉(`NO_CUDNN_FIX=1`)就崩 |
| wandb `ModuleNotFoundError: click` 或 `your url ValidationError` | repo 用 `--no-deps` 装 wandb;占位 `WANDB_BASE_URL` 不合法 | `train.py` 已 lazy import + 异常降级 `config.enable_wandb=False`,**绝不会拖垮训练** |
| `Cannot copy out of meta tensor` | diffusers 低内存加载下,新加的 `kf_aux_head`/`stage_head` 留 meta | `load_transformer` 已修:加载后扫所有子模块,对仍在 meta 的 `to_empty(cpu) + reset_parameters() + to(dtype)` |
| `Pool(128) fork-after-CUDA` 死锁(`Setting up datasets...` 显存涨但 0% util) | `MultiLatentLeRobotDataset` 默认 worker=128 | dataset 已改串行 + skip-incomplete-repo |
| `kf_loss = 0.0000` 训了一阵 | `kf_aux=False`(被同步覆盖回 default)| 确认 `va_robotwin_train_cfg.py` 里 `kf_aux=True, kf_aux_weight=0.1` |

---

## 9. 文件索引(训练相关)

| 角色 | 路径 |
|---|---|
| 训练主程序 | `wan_va/train.py` |
| 模型 + 辅助头 | `wan_va/modules/model.py`(`kf_aux_head`, `stage_head`, `forward_train` 返回 5 元组) |
| Meta-tensor 修复 + load_transformer/vae/text_encoder | `wan_va/modules/utils.py` |
| 多任务 dataset + kf/vlm_stage 钩子 + Ablation-3 腐化 | `wan_va/dataset/lerobot_latent_dataset.py` |
| **训练配置(主)** | `wan_va/configs/va_robotwin_train_cfg.py` |
| **训练配置(Ablation-3)** | `wan_va/configs/va_robotwin_train_wrongstage_cfg.py` |
| 训练启动 shell(torchrun + cuDNN 修复 + HF cache 重定向) | `script/run_va_posttrain.sh` |
| Ablation-3 三阶段一键封装(train/probe/eval) | `script/run_ablation_implicit.sh` |
| 关键帧标注脚本 | `evaluation/robotwin/keyframe_annotate.py` |
| VLM 阶段标注脚本 | `evaluation/robotwin/qwen_stage_annotate.py` |
| 训完自检探针 | `evaluation/robotwin/latent_probe.py`(+ `wan_va/train.py:collect_hidden`) |
| 探针可复现包 | `script/{reproduce_probe.sh, freeze_probe.sh}` + `train_out/probe/probe_canonical.json` |

---

## 10. 一图流总结

```
                    H200 训练流水线(本项目)
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│  数据集(共享 qb-ilm2):                                          │
│    lerobot_robotwin_eef_aug_500_stable/  (12 任务 × 500 ep)     │
│         ↓                                                       │
│  [§2.1] keyframe_annotate.py --recursive  → meta/keyframes.jsonl│
│  [§2.2] qwen_stage_annotate.py + 8 卡并行 → meta/stages.jsonl   │
│         ↓                                                       │
│  基座: lingbot-va-posttrain-robotwin (LingBot 官方,不动)        │
│         ↓                                                       │
│  ┌─────────────┬───────────────────┬───────────────────────┐    │
│  │  [§3] M1    │  [§4] M1v         │  [§5] M1v_WRONG       │    │
│  │  kf 头      │  kf + VLM 阶段    │  kf + 错误 VLM 阶段   │    │
│  │  ~2 h       │  ~2 h             │  ~2 h                 │    │
│  └─────────────┴───────────────────┴───────────────────────┘    │
│         ↓             ↓                      ↓                  │
│  train_out/checkpoints/                                         │
│    robotwin_kf0.1/...1200                  ← M1 (报告 baseline) │
│    robotwin_kf0.1_vlmstage0.1/...1200      ← M1v (主交付)       │
│    robotwin_kf0.1_vlmstage0.1_WRONG/...1200 ← Ablation-3        │
│         ↓                                                       │
│  [§7] 各自跑探针 → val_acc 应分别 ~0.65 / 0.78 / 0.64           │
│  → 4090 评测出 SR(见 `script/eval_route2_latent_cot.sh` 等)    │
└─────────────────────────────────────────────────────────────────┘
```

---

**报告"训练配置与实现细节"章节直接引用本文档**(尤其 §3 / §4 / §5 三条命令 +
§6.3 Ctrl-C 安全保存机制 + §8 工程坑表),即可完整描述训练流程。
