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

## ⏭️ 下一步 (需在可训练环境做; 已给精确接线点): #1 模型侧 ~50 行

未盲改 30 层 diffusion-transformer（本地无法训练验证）。精确接线规格:

- **取 hidden**: `wan_va/modules/model.py` 主干 forward 倒数第二层 block 输出
  `h`（`[B, seq, d]`）；按 latent-token 区段切出每 latent frame 的 token，
  时间维池化 → `h_t ∈ R^{B, F_lat, d}`。
- **aux head**: `MLP(d→128→1)` → `pred_log_dist`；目标 `log1p(kf_dist)`，
  `SmoothL1`(Huber)，`× kf_mask` 后对 mask 求均值。可选第二头
  `MLP(d→128→K)` 预测 keyframe type（CE，同样 mask）。
- **loss 接入**: `wan_va/train.py` `train_one_step`（约 L311
  `loss = latent_loss + action_loss`）→ `+ cfg.kf_aux_weight * L_kf`；
  仅 `cfg.kf_aux and cfg.kf_aux_weight>0` 时才构建/调用 head（关 = 零影响）。
  `kf_dist/kf_mask` 已在 `batch_dict`，需在 `_prepare_input_dict` / forward
  透传到 `compute_loss`（与 `actions_mask` 同路径）。
- **推理**: aux head 可不调用；其输出可作 client 端 progress monitor。

---

## ⏭️ 延后 (计划列为"成本高/需重训"; 待 #1 验证收益后再做)

- **#2** predictability (BYOL/InfoNCE) ~80 行、训练慢 ~5%
- **#3** subgoal token 改架构 ~300 行 + 重训 + KV-cache 分区（计划列最大改动）
- **#5** two-stage mask + sparse 推理 ~150 行 + finetune

理由: 三者改训练/架构且**本地不可验证**，盲注入风险高；`latent_plan.md`
本身要求"先 explicit 验证收益再 implicit"、预算不足优先 #1+#4+#5。建议顺序:
**#1 数据跑通 → #1 模型侧训练 → #4 probing 量化 latent 是否编码 stage →
再决定 #2/#3/#5**。

## #4 probing

待 #1 模型侧训出 ckpt 后做（依赖训练产物 + stage 标签；stage 标签可用
`keyframe_annotate.py --with-stage-boundaries` 的 `stage` 类型，或 deepseek
阶段标注，二者交叉验证，见 latent_plan.md "风险点 1"）。

---

## 变更文件清单 (本次)

| 文件 | 改动 | 兼容性 |
|---|---|---|
| `evaluation/robotwin/keyframe_annotate.py` | 新增（离线标注） | 独立脚本 |
| `wan_va/dataset/lerobot_latent_dataset.py` | `import json`；`_load_keyframes()`；`__getitem__` 末尾 kf 钩子 | `kf_aux` 关 → NO-OP |
| `wan_va/configs/va_robotwin_train_cfg.py` | `kf_aux/kf_aux_weight/kf_file` | 默认 NO-OP |
