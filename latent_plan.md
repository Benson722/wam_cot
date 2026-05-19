# Latent CoT 实施计划 — lingbot-va WAM

目标: 在 lingbot-va (video+action world model, WAN 2.2 VAE + 48-ch latent diffusion + action diffusion head) 中注入 **隐式物理 CoT (Implicit Physical Chain-of-Thought)**, 让 latent rollout 自发编码任务阶段与未来动力学, 提升 long-horizon 任务成功率。

数据基础: 已有离线阶段标注管线 `evaluation/robotwin/deepseek_stage_planner.py` (deepseek-v4-pro 调用), 可对训练集每段 trajectory 生成 `stages` (阶段名 + goal) + 每 chunk `stage_idx`。

下面五条独立可串联, 按代价由低到高排序。

---

## 1. 关键帧辅助 loss (Keyframe Auxiliary Head)

**核心**: 数据集预先标注每帧到下一个关键状态的距离 (帧数), 模型加 aux head 回归这一距离, 强迫 backbone latent 编码任务进度。

**关键帧来源** (三选一或融合):
- **deepseek 阶段切换点**: stage_idx 变化时刻
- **gripper open/close 切换帧**: 二值变化最直接, 与物理操作强对齐
- **接触事件**: 末端力突变 / 速度极小值 (停顿)

**数据制作**:
- 离线脚本扫训练集, 输出 `keyframe_dist.jsonl`:
  - 每条轨迹 `[T]` 长度的 int 数组 `dist_to_next_keyframe[t]`
  - 可选: `next_keyframe_type[t]` (one-hot: grasp / release / contact / stage-change)
- 与现有 `wan_va/dataset/` loader 对齐, 按 latent 时间下采样比 (~4×) 重采样到 latent timestep

**模型改造**:
- 在 transformer backbone 取倒数第二层 hidden state `h_t ∈ R^d`
- 加 `aux_head = MLP(d → 128 → 1)` 出 log(dist + 1) (Huber loss)
- 可选第二头预测 keyframe type (CE loss)
- 训练 loss:
  ```
  L_total = L_video + L_action + λ_kf · (L_dist + L_type)
  ```
  λ_kf 起始 0.1, 训稳后可降到 0.05

**预期效果**:
- backbone 被迫在 latent 中编码 "还有多久到下一阶段切换", 即时间维度的物理 CoT
- 推理时 aux head 可选不调用, 但 backbone 已 shaped
- 副产品: aux head 输出可作为 progress monitor 给 client

**成本**: 数据标注脚本 ~200 行, 模型改 ~50 行, 训练时间不变 (只多一个小 head)

---

## 2. Latent Contrastive / Predictability Loss

**核心**: 在 latent 序列上加自监督 predictability 约束 — 当前 latent 应能预测未来 latent, 类似 BYOL/VICReg 但在时序维度。强迫 latent 编码可预测的动力学规律, 不只表观像素。

**两种实现路径**:

### 路 A: InfoNCE (contrastive)
- 取 latent 序列 `z_1, z_2, ..., z_T` (来自 VAE encoder 输出, shape `[1, 48, T_lat, H, W]`)
- 投影头 `g_θ`: 把 `z_t` flatten + MLP → `R^256`
- 预测头 `p_φ`: `R^256 → R^256`, 输入 `g(z_t)`, 目标 `g(z_{t+k})` (stop-gradient)
- 负样本: 同 batch 其它轨迹的 `g(z_{t+k})`
- Loss: InfoNCE, τ=0.07

### 路 B: 简单回归 + stop-gradient (BYOL 风格, 更稳)
- `p_φ(g_θ(z_t)) ≈ sg(g_θ(z_{t+k}))`
- L2 loss, 无负样本, 无 collapse 风险 (predictor + stop-grad 起防 collapse)

**关键设计**:
- `k` 取 1-4 latent timestep (相当 4-16 真实帧, 覆盖一个 chunk)
- 投影头只用于自监督 loss, 推理不参与
- 与 video diffusion loss 并行, 权重 λ_pred = 0.05

**预期效果**:
- latent 表征对短期未来可预测, 间接约束了物理一致性
- 探针实验 (见 #4) 上 stage 分类准确率显著高于无此 loss

**对比 #1**:
- #1 用外部标签 (keyframe distance), 显式监督
- #2 全自监督, 无需标注, 可堆叠使用
- 论文参照: V-JEPA, JEPA-Predictor, Predictive Coding

**成本**: 标注 0, 模型 ~80 行 (投影头 + predictor + loss), 训练慢 ~5%

---

## 3. 显式 Subgoal Token 注入 (Latent Coconut)

**核心**: transformer 序列里插入特殊 `[SUBGOAL]` slot, 该位置在训练时监督到 "下一关键帧的 latent" (或其压缩), 推理时模型自回归生成 SUBGOAL 后再出 action / video chunk。等同 Meta Coconut "continuous thought" 思路, 但在 video latent space。

**序列结构改造**:
- 当前: `[obs_latent][action_latent]`
- 新增: `[obs_latent][SUBGOAL_1][SUBGOAL_2]...[SUBGOAL_K][action_latent]`
- K 取 2-4 (子目标数, 对应 future keyframe 数)
- SUBGOAL slot 用可学 query token 初始化, transformer 自回归填充

**监督目标**:
- `SUBGOAL_i` 位置的 transformer 输出 `h_subgoal_i ∈ R^d`
- 投影到 latent space: `proj(h_subgoal_i) ∈ R^{48·H·W}`
- 监督到训练集中第 i 个关键帧的真实 latent `z_{kf_i}` (由 VAE encode 得到)
- Loss: MSE 或 flow-matching loss (与主 video diffusion 同 scheduler)

**数据来源**:
- 与 #1 共用关键帧标注 (gripper switch / stage change)
- 对每段 video 取前 K 个未来关键帧, VAE 离线 encode 缓存到磁盘 (避免训练时反复 encode)

**推理流程**:
- t 时刻 client 发 obs → server 先去噪 SUBGOAL tokens (少步 diffusion, e.g. 5 步)
- 再去噪 video latent (条件 SUBGOAL) + action (条件两者)
- SUBGOAL 输出可解码看 client 是否要可视化 ("imagined keyframes")

**对模型改动 (lingbot-va 具体位置)**:
- `wan_va/modules/` transformer 加 subgoal slot 处理 (sequence packing 改, attention mask 改)
- `wan_va/configs/` 加 `num_subgoal_tokens` 参数
- KV cache 逻辑要更新, subgoal token 在 cache 里独立分区

**预期效果**:
- 模型在生成动作前先 "想象" 几个关键未来状态, 显式 latent plan
- 当任务阶段切换时, SUBGOAL 突变 → 给 action head 强信号
- 推理可解码 SUBGOAL 看模型 "脑中" 的规划, 便于 debug

**成本**: 改动最大 (~300 行 + 训练流程改), 但效果最显式

---

## 4. Probing 实验 + t-SNE 可视化

**目的**: 验证训完模型的 latent 是否真的隐式编码了任务阶段。冻结 backbone, 在 latent 上训小分类器, 高准确率 = implicit CoT 成功。

**Probing 协议**:

### 数据
- 用 deepseek 阶段标注作 ground truth, 每条 trajectory 每 latent timestep 有 `stage_idx ∈ {0,...,S-1}` (S 通常 3-6)
- 平衡采样: 每个 stage 各取 ~1k 样本
- train/val 8:2 split (按 trajectory 而非 frame 划分, 防泄漏)

### Probe
- 冻 backbone 全部参数
- 提取每个 latent timestep 对应的 transformer hidden state `h_t ∈ R^d`
- 训 linear probe: `softmax(W · h_t + b)`, S 类 cross-entropy
- 训 50 epoch, AdamW lr=1e-3

### 比较 baseline
| 模型 | 期望 probe acc |
|---|---|
| 原 stock lingbot-va | 35-50% (chance 1/S ~ 20-33%) |
| + #1 keyframe aux | 60-70% |
| + #2 predictability | 55-65% |
| + #3 subgoal token | 70-80% |
| + #1 #2 #3 全开 | 80%+ |

### t-SNE 可视化
- 抽 5-10 个 unseen 任务 × 10 episodes
- 每 episode 取所有 latent timestep 的 `h_t`, 着色 = `stage_idx`
- t-SNE perplexity=30, 出图 `figures/latent_tsne_<exp_name>.png`
- 期望: 不同 stage 形成可分簇; 同一 stage 跨任务也聚集 (说明跨任务 stage 语义共享)

### 干预实验 (可选, 进阶)
- 在 latent 上找 "stage direction" (linear probe 权重 W)
- 推理时把当前 latent 沿某 stage 方向加扰动, 观察 action 是否对应改变
- 类似 ROME / activation patching, 证明 latent 因果性而非只相关

**输出物**:
- `experiments/probing/results.json`: 各模型 probe acc
- `figures/latent_tsne_*.png`: 可视化
- `experiments/probing/probe_<exp_name>.pt`: 训好的 probe (供干预用)

**成本**: 一周内可跑完所有 baseline + probe + 可视化

---

## 5. Two-Stage 训练 — Action Decoder Mask + 稀疏条件推理

**核心**: 训练时部分 step 给 action decoder mask 掉真实 obs latent, 强迫它只能依赖 imagined/predicted latent 解码动作。测试时 action decoder 在稀疏 obs 条件下仍能根据 latent rollout 出 action, 实现 "看一眼想很远再动" 的能力。

**训练改造**:

### 阶段 A (warmup, 现有流程)
- 正常 video + action 联合训练, action head 看到完整 real obs latent
- 保留 N 个 epoch 让基础模型收敛

### 阶段 B (mask schedule)
- 每个 training step 以概率 `p_mask` (从 0.1 ramp 到 0.5):
  - 把 obs latent 的后 50-100% timestep mask 掉 (置 0 或学到的 [MASK] token)
  - action head 仍要预测全部 action chunk
- backbone 必须用 imagined latent (自己 rollout 出的) 补全缺失 obs, action head 必须接受这些 imagined latent
- 关键: video diffusion loss 也保留, 让 imagined latent 接近真 obs latent

### Loss
```
L = L_video(unmasked timesteps) +
    λ_imag · L_video(masked timesteps, predict from prev) +
    L_action(all timesteps)
```

**推理改造 (稀疏条件)**:
- 当前: 每 chunk 都给 server 发 `key_frame_list` 做 `compute_kv_cache` 更新
- 新模式: 每 K 个 chunk 才发一次 obs, 中间 K-1 个 chunk server 用自己上一轮 imagined latent 当 obs
- 实际收益: 减少模拟器步进等待, 加速 rollout; 同时检验 latent rollout 的稳定性

**两种推理对比**:
| 模式 | server 输入 | 优势 | 风险 |
|---|---|---|---|
| Dense (现行) | 每 chunk 真 obs | 准 | 慢, 依赖模拟器 |
| Sparse (本方案) | 每 K chunk 真 obs | 快, 接近 real-world async | 误差累积 |

**评估指标**:
- 不同 K 值 (1, 2, 4, 8) 下 SR
- imagined latent vs real latent 的 reconstruction L2 (衡量误差累积)
- 与 explicit stagecond + dense 对比

**对模型的间接好处**:
- 强 action decoder 依赖 latent rollout, 不会偷懒 shortcut 用 obs
- backbone latent 必须自洽 (imagined 与 real 接近), 提升物理一致性
- 等同 implicit CoT 的 stress test

**成本**: 训练阶段改 ~100 行 + 重训 (或 finetune 现有 ckpt), 推理 client/server ~50 行

---

## 实施路线建议 (六周)

| 周 | 任务 |
|---|---|
| 1 | #1 数据标注脚本 + dataset loader; explicit stagecond baseline SR 验证 |
| 2 | #1 模型改 + 训练; #4 probing 框架搭好 |
| 3 | #2 contrastive loss 接入, 训练; probe #1 vs #2 |
| 4 | #3 subgoal token 架构改造 + 训练 |
| 5 | #4 完整 probing + t-SNE + 干预; 论文 figure 出 |
| 6 | #5 two-stage mask 训练 + sparse 推理评测 + ablation |

## 风险点

1. **deepseek 阶段标注质量**: 若 stage 边界与真实物理切换不齐, #1 #4 都受影响。缓解: 与 gripper 切换交叉验证, 不一致丢弃。
2. **probing 高 acc 可能源自任务标识而非 stage**: 必须按 trajectory split, 跨任务 probe 才说明 stage 语义。
3. **#3 subgoal token 训练不稳**: warmup 时先冻 subgoal slot, 收敛后再放开。
4. **#5 mask 太激进导致塌缩**: p_mask 必须 schedule, 不可一开始就 0.5。
5. **算力**: #1 #2 加 head 影响小; #3 改架构需重训; #5 需 finetune。预算够建议全做, 不够优先 #1 + #4 + #5。

## 与 explicit stagecond (已实现) 的关系

- explicit stagecond = 推理时调 deepseek 做 online CoT, 验证 "阶段条件能否提升 SR"
- implicit (本计划) = 把 CoT 灌进模型权重, 推理零外部依赖
- 先 explicit 验证收益, 再 implicit 工程化。若 explicit 不涨, 检查阶段质量 / 提示词, 不要急着 implicit
