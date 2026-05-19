# 实验结果记录与主表填写说明（5.1 主表）

> 配套：`MODEL_AND_DATA.md`（模型/数据/损失）、`latent_plan_progress.md`
> （路线二进度）、`evaluation/robocasa/COT_DESIGN.md`（路线一设计）。
> ⚠️ 现状：仅完成 **M1 训练 + z_t 离线探针**。主表里 TSR/SSR/ASC/Latency
> 需跑 RoboTwin 评测才能得到 —— 现在**全部为"待测"**，命令见 §4。

---

## 1. 列定义（与 PDF "任务成功率/阶段成功率/平均执行步数/推理开销" 对齐）

| 列 | 含义 | 来源 |
|---|---|---|
| **T1/T2/T3 TSR** | 三组代表性任务的 **Task Success Rate**（任务成功率，`成功局数/总局数`）。建议按 PDF"挑代表性任务"分三档：T1=短程原子（如 `adjust_bottle`）、T2=中等（如 `beat_block_hammer`/`handover_block`）、T3=长程/遮挡（如 `lift_pot`/`hanging_mug`/多阶段任务）。 | `evaluation/robotwin/calc_stat.py` 统计视频文件名 `*_True/False.mp4`，或客户端写的 `res.json` `succ_rate` |
| **Mean SSR** | 平均 **Stage Success Rate**（阶段成功率）：完成的子阶段数 / 总阶段数，跨任务取均值。子阶段 = CoT 的 subtask（M2/M3/M4）或关键帧分段（M0/M1）。 | CoT 客户端 `.plan.json`/`subtask_progress_rate`；M0/M1 用关键帧分段判定 |
| **ASC** | **Average Steps to Complete**（平均执行步数，越低越好；失败局可记为 step 上限）。 | 客户端 `avg_steps`（`eval_polict_client_openpi.py` / `client_cot.py` 已统计） |
| **Latency** | 推理时间与计算开销：M0/M1 = WAM 每步/每局推理耗时；M2/M3/M4 还需叠加 **VLM 规划器** 调用延迟与 token 开销（`cot_planner` 已记 `vlm_*` 统计到 `vlm_calls.jsonl`/`planner.stats()`）。 | server `server_timing` + `planner.stats()` |

---

## 2. M0–M4 → 本项目的具体方法（这几格现在就能填：Method/设计/改动）

| 行 | 我们的实现 | 一句话 Method 说明（可填表/报告） |
|---|---|---|
| **M0 Baseline** | 原始 `lingbot-va-posttrain-robotwin` 权重 + 原生 RoboTwin 客户端 `eval_polict_client_openpi.py`（**不开 `--cot`**）。纯 WAM，无任何思维链。 | 基线 WAM：直接接收指令+观测，输出底层动作，无显式/隐式中间推理。 |
| **M1 Latent CoT** | **路线二（隐式物理 CoT）**：在 `forward_train` 主干隐状态上加 `kf_aux_head`，用"到下一关键帧（夹爪 grasp/release/end）距离"做 `log1p`+Huber 辅助损失（`λ_kf=0.1`），从 `posttrain-robotwin` 续训得 `checkpoint_step_*`。推理用与 M0 同一客户端（**思维链已灌进权重，推理零外部依赖**）。 | 隐式 CoT：辅助损失迫使世界模型 latent 编码任务进度/阶段，提升长程稳定性。 |
| **M2 Semantic CoT** | **路线一（外部语义 CoT）**：stock WAM + 高层规划器（本地 Qwen3.5-27B 多模态，`client_cot.py --ablation none`）把任务分解为有序原子子任务，经 `switch_prompt` 软切换驱动 WAM。 | 外部 VLM 规划器产出物理约束推理与子任务序列，逐子任务驱动底层 WAM。 |
| **M3 Dual (Bonus)** | M1 的 ckpt（隐式 CoT 权重）+ 路线一外部规划器叠加（`client_cot.py` 指向 M1 server）。 | 隐式+显式 CoT 协同。 |
| **M4 Replan (Bonus)** | 路线一 + VLM 监控重规划（`client_cot` 的 `need_replan` 路径）。 | 执行中 VLM 监控，场景偏离即重规划，鲁棒性最佳。 |

> 设计思路 / 改动（可直接进报告"训练配置与实现细节"）：见 `MODEL_AND_DATA.md`
> §1（基座 LingBot-VA + posttrain-robotwin 权重）、§2（A 工程修复 / B 路线二
> 方法改造 / C 探针，每条含原因）、§2.5（L_video/L_action/L_kf 数学定义与
> 物理意义）、§3/§4（数据集 = RoboTwin2.0 多任务 LeRobot + 血缘）。

---

## 3. 现在已有的、可写进报告的结果（不属于主表，属"分析/消融"）

- **z_t latent 探针**（`latent_probe.py --features z_latent`，adjust_bottle 400
  ep，轨迹切分）：val_acc ≈ **0.797**（chance 0.50，+0.30），per-class
  pre-grasp 0.92 / post-grasp 0.70 → 证明 VAE latent 已线性可分操作阶段
  （隐式编码基线证据）。这放进"latent 分析"小表，不是主 SR 表。
- **训练曲线**：`plot_losses.py` 产出 `loss_curves.png`（latent 触地板
  ~0.1、action ~1e-3、**kf_loss 从 ~0.30 → ~0.002**，证明 #1 辅助信号被学到）。
- 主表数值列：**尚无**（未跑 RoboTwin 评测）。

---

## 4. 怎么把主表每一格"测出来"（待执行）

前提：RoboTwin 可跑的机器（你之前把推理放 4090），server 加载对应 ckpt。

**M0 / M1 行（TSR/SSR/ASC/Latency）—— 原生客户端，逐任务跑**：
```bash
# server: M0=stock ckpt；M1=把 va_robotwin_cfg 的 ckpt 指向 checkpoint_step_*
bash evaluation/robotwin/launch_server.sh
# client: 对 T1/T2/T3 各代表任务各跑 N 局
bash evaluation/robotwin/launch_client.sh ./results_M0 <task>     # M0
# (M1: server 换 ckpt 后) bash evaluation/robotwin/launch_client.sh ./results_M1 <task>
python evaluation/robotwin/calc_stat.py   # 汇总 succ_rate/avg_steps -> 填 TSR/ASC
```
**M2 / M3 / M4 行 —— CoT 客户端 + 本地 Qwen 规划器**：
```bash
PLANNER=vllm ABLATION=none      bash evaluation/robotwin/launch_cot_client.sh ./results_M2 <task>   # M2 (stock server)
PLANNER=vllm ABLATION=none      bash evaluation/robotwin/launch_cot_client.sh ./results_M3 <task>   # M3 (M1 server)
PLANNER=vllm ABLATION=replan?   ...                                                                 # M4 (need_replan 开启)
# .plan.json -> subtask_progress_rate=Mean SSR；vlm_calls.jsonl/planner.stats() -> Latency
```
每格填：`TSR=succ_rate`、`Mean SSR=subtask_progress_rate`、`ASC=avg_steps`、
`Latency=` server 每步耗时（+M2/3/4 的 VLM 调用均延迟）。建议每任务 ≥25 局、
3 个种子取均值±std。

---

## 5. 关键诚实结论

- 现在能交付的是 **Method 定义 + 设计/改动 + 训练完成(M1) + z_t 探针 +
  loss 曲线**；**主表 SR 数值需按 §4 跑评测**，我无法凭训练日志/wandb 估出
  TSR（必须真跑 RoboTwin）。
- 报告里主表先填 Method 列与"期望"列（图中已有），数值列标"进行中"，附
  z_t 探针小表 + loss 曲线作为已完成的支撑证据。
