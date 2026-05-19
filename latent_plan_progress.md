# Latent-CoT 实施进度 (companion of latent_plan.md)

> `latent_plan.md` 为只读规格文档，本文件记录**增量落地进度**。
> 原则: 5 个组件是 6 周研究计划，**增量落地、不盲写不可验证的训练代码**。
> 已落地部分全部 **opt-in / 向后兼容**——开关关闭时 baseline 训练逐字节不变。

---

## ✅ 已完成: #1 的数据地基 (latent_plan 第 1 周)

> 注意: `latent_plan.md`"数据基础"一节假设
> `evaluation/robotwin/deepseek_stage_planner.py` **已存在——实际并不存在**。
> 已改用计划自己也推荐的、**更鲁棒且零 LLM 依赖**的 gripper 开合切换作为
> 关键帧来源（计划原文: "二值变化最直接, 与物理操作强对齐"）。

1. **离线标注脚本** `evaluation/robotwin/keyframe_annotate.py`
   - 扫 LeRobot 训练集，按 **gripper 开/合跳变**提关键帧：左/右夹爪通道
     默认 `idx 7/15`，阈值取该通道自身值域中点 → 跨 embodiment 鲁棒、与
     "哪侧算合"无关，只认*跳变*；类型 `grasp`/`release`，可选 `stage`、
     末帧 `end`（保证每个 timestep 都有"到下一关键帧距离"）。
   - `--with-stage-boundaries` 叠加 `action_config` 段边界（stage-change）。
   - 输出 `<dataset>/meta/keyframes.jsonl`（每 episode 一行：raw 帧索引+类型）。
   - 纯离线、无 GPU/LLM；核心逻辑已本地单测通过
     (grasp/release/stage/末帧、convention-agnostic 二值化)。
   - 运行:
     ```
     python evaluation/robotwin/keyframe_annotate.py \
         --dataset <lerobot_dataset_root> \
         --gripper-idx 7 15 [--with-stage-boundaries]
     ```

2. **dataset loader 钩子** `wan_va/dataset/lerobot_latent_dataset.py`
   - 新增 `_load_keyframes()`（懒加载+缓存）；`__getitem__` 末尾按**与模型
     一致的 latent 时间步**（`latent_frame_ids` 每 ~4× 取代表帧，与 action
     `f=latent_frame_num` 对齐）算 `kf_dist` / `kf_mask`。
   - 仅当 `cfg.kf_aux=True` 才加 `out_dict['kf_dist'/'kf_mask']`；标注文件
     缺失 → 全 mask（零 loss）。关闭时 libero/demo 等配置完全无感知。

3. **训练配置开关** `wan_va/configs/va_robotwin_train_cfg.py`
   - `kf_aux=False`、`kf_aux_weight=0.0`、`kf_file='keyframes.jsonl'`
   - 默认严格 NO-OP。启用 = 跑标注脚本 + 置 `kf_aux=True, kf_aux_weight>0`
     (λ_kf，计划建议 0.1 → 收敛后降 0.05)。

py_compile 全通过；标注核心逻辑本地单测通过。

---

## ✅ 已完成: #1 模型侧 (基于真实数据 schema 实现，py_compile 通过)

> 数据 I/O 已对真实数据校验：`data_example/.../meta/info.json` 确认
> `action` 16 维、`left_gripper=idx7 / right_gripper=idx15`（与
> `keyframe_annotate.py --gripper-idx 7 15` 默认一致）、parquet 路径
> `data/chunk-{c:03d}/episode_{i:06d}.parquet`、`chunks_size=1000`、
> `episodes.jsonl` 结构。`va_robotwin_train_cfg.dataset_path` 已指向 H200
> 任务路径 `/inspire/.../adjust_bottle-aloha-agilex_randomized_500-1000`。
> `keyframe_annotate.py` 加 `--episodes-file`（demo 的细粒度多阶段在
> `episodes_ori.jsonl`）。

实现（全部 opt-in；`kf_aux_weight=0` 时 `kf_loss` 为常数 0、aux head 不接收
梯度 → 训练动力学与 baseline 逐字节一致）：

- **取 hidden** `wan_va/modules/model.py::forward_train`：在所有 block +
  norm_out + scale/shift 之后、`torch.split` 取出 `latent_hidden_states`
  （**proj_out 之前**的主干隐状态）；token 顺序 `(f h w)`（由
  `_input_embed` 决定），`rearrange('1 (b f s) d -> b f s d')` 后对空间
  维 `s` mean-pool → `[B, F_lat, d]`。
- **aux head** `__init__`：`nn.Sequential(Linear(inner_dim,128), GELU,
  Linear(128,1))`，恒构建（极小；从无此 head 的 ckpt `from_pretrained`
  会随机初始化+warn，推理不调用）；`forward_train` 多返回 `kf_pred
  [B,F_lat]`。
- **loss 接入** `wan_va/train.py`：`_prepare_input_dict` 透传
  `kf_dist/kf_mask`（缺失=None）；`compute_loss` 解包 3 元组，
  `SmoothL1(log1p(kf_dist), kf_pred)` masked-mean × `cfg.kf_aux_weight`，
  长度做防御性对齐；`_train_step` `loss = latent+action+kf`，日志加
  `kf_loss`。仅 `cfg.kf_aux & kf_aux_weight>0 & 有标注` 时生效。
- **推理**：aux head 不参与 inference（`forward` 不走 `forward_train`）；
  其输出可作 client 端 progress monitor（后续可选）。

启用步骤：① `python evaluation/robotwin/keyframe_annotate.py --dataset
<task_dir> --gripper-idx 7 15`（可加 `--episodes-file episodes_ori.jsonl
--with-stage-boundaries`）→ 生成 `meta/keyframes.jsonl`；② 训练配置置
`kf_aux=True, kf_aux_weight=0.1`（计划建议训稳后降 0.05）。

---

## ⏭️ 延后 (计划列为"成本高/需重训"; 待 #1 验证收益后再做)

- **#2** predictability (BYOL/InfoNCE) ~80 行、训练慢 ~5%
- **#3** subgoal token 改架构 ~300 行 + 重训 + KV-cache 分区（计划列最大改动）
- **#5** two-stage mask + sparse 推理 ~150 行 + finetune

理由: 三者改训练/架构且**本地不可验证**，盲注入风险高；`latent_plan.md`
本身要求"先 explicit 验证收益再 implicit"、预算不足优先 #1+#4+#5。建议顺序:
**#1 数据跑通 → #1 模型侧训练 → #4 probing 量化 latent 是否编码 stage →
再决定 #2/#3/#5**。

## ✅ #4 probing (z_t 离线基线已落地、可立即跑；h_t 钩子已留)

- loader 在 kf 钩子里追加 `kf_stage`（每 latent frame 已过关键帧数 = 阶段
  idx）与 `kf_episode`（轨迹安全 split 用），同 `cfg.kf_aux` 守卫。
- `model.py::forward_train` 现额外返回 `kf_feat`（pre-head 主干隐状态，
  4th 元素；`compute_loss` 只读 pred[0:3]，无副作用）→ 供日后 `h_t` probe。
- `evaluation/robotwin/latent_probe.py`：
  - `--features z_latent`（默认，**零 GPU/模型依赖、可立即跑**）：对 Wan-VAE
    latent 做空间 mean-pool → 每 latent frame `[C]`，按 `kf_stage` 训线性
    探针；**按 episode 轨迹切分** train/val（防 plan 风险 #2 的泄漏）；
    输出 `results_*.json`（val_acc / chance / 每类 acc / 混淆矩阵）+
    `tsne_*.png`（sklearn t-SNE，缺失则 torch PCA-2D 回退）+ probe 权重。
  - `--features h_hidden`：保留（`forward_train` 已暴露 `kf_feat`），待 #1
    训出 ckpt 后做 stock-vs-#1 对比，当前主动 NotImplementedError 指引。
- 运行: `python evaluation/robotwin/latent_probe.py --config robotwin_train
  --num-samples 400 --out-dir experiments/probing`
- **预期/解读**：`adjust_bottle` 仅 2 阶段（grasp 前/后）→ chance=0.50。
  val_acc 显著 >0.5（如 >0.65）= 即便 stock 表征，VAE latent 已线性可分
  manipulation 阶段（隐式物理编码的弱证据/基线）；接近 0.5 = 不可分。
  强 probing 需多/长程任务（更多关键帧→更多阶段，chance 更低更有区分度）。
  z_t 是"喂给世界模型的表征"基线；h_t（主干隐状态）+#1 训练后的提升才是
  计划要的核心对比。

---

## 变更文件清单 (本次)

| 文件 | 改动 | 兼容性 |
|---|---|---|
| `evaluation/robotwin/keyframe_annotate.py` | 新增（离线标注） | 独立脚本 |
| `wan_va/dataset/lerobot_latent_dataset.py` | `import json`；`_load_keyframes()`；`__getitem__` 末尾 kf 钩子 | `kf_aux` 关 → NO-OP |
| `wan_va/configs/va_robotwin_train_cfg.py` | `kf_aux/kf_aux_weight/kf_file` | 默认 NO-OP |
