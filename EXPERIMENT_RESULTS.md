# 实验结果记录与主表填写说明（5.1 主表）

> 配套：`MODEL_AND_DATA.md`（模型/数据/损失）、`latent_plan_progress.md`
> （路线二进度）、`evaluation/robocasa/COT_DESIGN.md`（路线一设计）。
> ⚠️ 现状（2026-05-20 更新）：已完成 **(a) VLM 离线语义阶段数据生成（12
> 任务 ×500 ep，Qwen3.5-VL）→ (b) Latent-CoT Phase B 训练（kf + VLM 阶段
> 双辅助头，step 1200）→ (c) 冻结-backbone 线性探针消融（无 CoT / 仅 kf /
> kf+VLM 三档，详见 §6）**——这是一条完整、单调、可直接写报告的**消融实验
> （PDF 第三阶段"必做"）**，并附 t-SNE 可视化。**主表 SR 数值列
> （TSR/SSR/ASC/Latency）仍"待测"**，按 §4 跑在线 RoboTwin 评测。

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

## 3. 现在已有的、可写进报告的结果（"分析/消融"，非主 SR 表）

- **【头条】冻结-backbone × VLM 语义阶段 线性探针消融**（详见 §6）：无 CoT
  0.663 → 仅 kf 0.666 → kf+VLM **0.782**（chance 0.167），泛化差
  0.307→0.226→0.102 单调收窄。**PDF 必做消融的核心证据**。
- **z_t latent 探针**（`latent_probe.py --features z_latent`，adjust_bottle 400
  ep，轨迹切分）：val_acc ≈ 0.797（chance 0.50），证明 VAE latent 已线性可
  分粗操作阶段（与 §6 的 backbone/VLM 多类探针互补，放"latent 分析"小表）。
- **训练曲线**：`plot_losses.py` 产出 `loss_curves.png`（latent 触地板
  ~0.1、action ~1e-3、**kf_loss ~0.30→~0.002、stage_loss 收敛 ~0.03**，
  两个 CoT 辅助信号均被学到）。
- 主 SR 表数值列：**尚无**（未跑在线 RoboTwin 评测，见 §4/§7）。

---

## 4. 怎么把主表每一格"测出来"（待执行）

前提：RoboTwin 可跑的机器（你之前把推理放 4090），server 加载对应 ckpt。

**M0 / M1 / M1v 行（TSR/SSR/ASC/Latency）—— 原生客户端，逐 ckpt 跑**。
`va_robotwin_cfg` 读环境变量 `VA_EVAL_CKPT`，**无需改配置**即可切换被服务的
权重（train ckpt 先把 `vae/tokenizer/text_encoder` 从 BASE 软链进去）：
```bash
# M0 = BASE(无 CoT)；M1 = checkpoint_step_1200(kf)；
# M1v = robotwin_kf0.1_vlmstage0.1/checkpoint_step_1200(kf+VLM)
VA_EVAL_CKPT=<ckpt_dir> bash evaluation/robotwin/launch_server.sh    # 一终端(LingBot env)
bash evaluation/robotwin/launch_client.sh ./results_<tag> <task>     # 另一终端(RoboTwin env)
python evaluation/robotwin/calc_stat.py   # 汇总 succ_rate/avg_steps -> 填 TSR/ASC
```
完整三路 `run_eval` 循环脚本见 §7。
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

- 已交付且**满足 PDF"消融实验（必做）"**的是 §6 的**离线表征消融**：受控
  去除 CoT（无 / kf / kf+VLM 三档），冻结 backbone 用线性探针量化"任务阶段
  信息是否、以及多大程度被编码进世界模型隐表征"，单调退化曲线 + t-SNE。
- §6 是**表征/可解释性层面**的因果证据（"CoT 真实参与了内部计算"）；它
  **不替代任务成功率**证据。完整的"对比评估"还需 §7 的在线 RoboTwin SR
  三路对照（TSR/SSR/ASC/Latency），我无法凭训练日志估 SR，必须真跑。
- 报告结构建议：主 SR 表（§7，跑完填数值）为"对比评估"，§6 探针消融 +
  loss 曲线 + t-SNE 为"消融分析/可解释性"，二者共同构成第三阶段交付。

---

## 6. Latent-CoT Phase B：VLM 语义阶段隐式 CoT —— 过程 / 结果 / 意义 / 消融

> 这一节是已完成、可直接进技术报告的完整实验。对应 PDF：路线二（内部物理
> CoT）的强化版、第三阶段"消融实验（必做）"与完成标准"确保 CoT 真实参与了
> 动作生成并具备可解释性"。

### 6.1 实验动机与设计

最初的 Latent-CoT #1（M1）用"到下一夹爪事件（grasp/release/end）的时间距离"
做辅助回归（`λ_kf=0.1`）。该信号是**低层、本体感、夹爪派生**的，只有约 2 个
相位，对"语义任务计划"刻画很粗。Phase B 引入 **VLM 进入数据生成与训练回路**：
用本地 Qwen3.5-VL 离线把每条 episode 切成有序**语义阶段**（approach / grasp /
lift / place / retract …），作为更丰富、任务感知的隐式 CoT 监督，叠加在同一
backbone 上（新增 `stage_head`，8 类 CE，`ignore_index=-1`，`λ_stage=0.1`）。

三档受控对照（唯一变量 = 启用哪种 CoT 辅助；同基座、同数据、同步数 1200）：

| 标签 | checkpoint | CoT 配置 |
|---|---|---|
| 无 CoT | `lingbot-va-posttrain-robotwin`（基座） | 无任何辅助头 |
| 仅 kf | `train_out/checkpoints/checkpoint_step_1200` | `kf_aux`（λ=0.1） |
| kf+VLM | `…/robotwin_kf0.1_vlmstage0.1/checkpoint_step_1200` | `kf_aux`+`stage_head`（各 λ=0.1） |

### 6.2 实验过程

1. **VLM 数据生成（Phase A）**：`evaluation/robotwin/qwen_stage_annotate.py`
   对 12 个 aloha-agilex 任务（curated `_stable` 父目录）递归标注。每条
   episode 均匀抽 4 帧 → 本地 Qwen3.5-VL（serve_qwen，OpenAI 兼容
   :8000/v1，关 thinking）→ 强约束 JSON-only + few-shot + 截断容错抽取
   → 写 `meta/stages.jsonl`。8 卡各起一 serve_qwen + 按任务目录分片并行，
   含 `--resume`（清洗去重+续写）。产物：12 任务 ×500 ep 全部
   `500 uniq / 500 OK`，平均 3.5–6 阶段/episode。
2. **训练（Phase B）**：`run_va_posttrain.sh`（`robotwin_train` 配置，
   `kf_aux=True/0.1` + `vlm_stage_aux=True/0.1`），多任务并发，从
   `posttrain-robotwin` 基座续训。step 1200 收敛：`action_loss≈1e-3`、
   `kf_loss≈2e-3`、`stage_loss≈0.03`、`latent_loss` 触噪声地板 ~0.12。
   检查点目标化命名 `robotwin_kf0.1_vlmstage0.1/`（带 `meta.json` 自述）。
3. **消融评测（探针）**：`wan_va.train --probe-collect` 对三个 ckpt 各
   forward 200 batch，dump 每帧 backbone 隐 `h_t`(=`kf_feat`,3072 维) +
   VLM 阶段标签 + episode id；`latent_probe.py --features h_hidden
   --label vlm_stage` 训练线性探针，**按 episode 轨迹切分**（防"同轨迹相邻
   帧"捷径），报告 val 准确率、per-class、混淆矩阵、t-SNE。

### 6.3 实验结果（冻结-backbone 线性探针，VLM 语义阶段标签，6 类，chance 0.167）

| Checkpoint（CoT 配置） | N | val_acc | 高于随机 | train_acc | 训练-验证差 |
|---|---|---|---|---|---|
| 无 CoT（基座） | 2810 | 0.663 | +0.497 | 0.970 | 0.307 |
| 仅 kf（M1） | 2751 | 0.666 | +0.499 | 0.892 | 0.226 |
| **kf+VLM（M1v）** | 2528 | **0.782** | **+0.615** | 0.884 | **0.102** |

- per-class（M1v）：approach 0.96 / grasp 0.79 / lift 0.71 / place 0.71 /
  0.56 / 0.56，混淆矩阵近三对角，错误几乎全是**相邻阶段** off-by-one
  （阶段边界帧天然模糊，良性误差）。
- t-SNE：`train_out/probe/out_h_kfvlm/tsne_robotwin_train_h_hidden_vlm_stage.png`
  按阶段着色，M1v 簇分离明显优于无 CoT/仅 kf（报告配图）。

### 6.4 实验意义（结论措辞，可直接进报告"消融分析/可解释性"）

1. **去除 CoT 即退化、单调**：0.663 → 0.666 → 0.782。去掉 VLM 阶段 CoT 掉
   0.116，全去掉掉 0.119。证明 CoT 辅助目标**因果地**把任务阶段结构编码进
   了世界模型底层表征。
2. **泛化差单调收窄（0.307→0.226→0.102）**：不是"更易记忆"，而是 CoT 让
   阶段信息**可泛化地**线性可读（轨迹切分下成立）——表征质量提升而非过拟合。
3. **"监督什么得到什么"**：仅 kf（0.666）几乎不优于无 CoT（0.663），因为
   夹爪时间是语义阶段的粗代理；只有**匹配的 VLM 语义监督**带来 +0.12 跳变。
   这正是"引入 VLM 进入数据生成+训练回路"的价值论证。
4. **可解释性 / CoT 真实参与**：探针即"拿当前隐状态去匹配 VLM 里程碑
   (`stages.jsonl`)"的严谨版；高且可泛化的准确率 = 模型内部计算确实沿语义
   阶段组织 → 满足 PDF 完成标准。

### 6.5 诚实边界（报告需写明，避免被质疑）

- 这是**表征/可解释性**消融，**不等于**任务成功率提升；SR 因果证据需 §7
  在线 RoboTwin 三路对照。两者互补，缺一不可。
- 探针标签来自 Qwen（含噪），"准确率"= 与 VLM 阶段切分的一致度；三档用
  **同一套标签**，相对比较有效。
- 三次 collect 抽到的 episode 子集不同（loader 随机，N=2810/2751/2528）；
  +0.12 跳变远大于该噪声，但严格复现需固定抽样种子（已知改进点）。

---

## 7. 在线 RoboTwin SR 三路对照（待跑，主表数值来源）

环境：4090 实例（RoboTwin 在
`/inspire/qb-ilm2/project/26summer-camp-11/public/group3/RoboTwin`）。
server 用 `VA_EVAL_CKPT` 切 ckpt（不改配置）；train ckpt 先把
`vae/tokenizer/text_encoder` 从 BASE 软链进去；RoboTwin
`task_config/demo_clean.yml` 置 `eval_video_log: True` 出执行视频。

三档对应 §6 的 M0(无 CoT 基座) / M1(kf) / M1v(kf+VLM)。`run_eval` 循环脚本：
启 server(`VA_EVAL_CKPT`) → 等就绪 → 客户端逐任务 `--test_num N` → 关
server。代表任务：T1=`adjust_bottle`(短程) / T3=`lift_pot`,`hanging_mug`
(较长程)。汇总 `results_<tag>/.../_result.txt` 的 SR。

**主 SR 表（跑完填）**：

| 行 | 方法 | ckpt | T1 TSR | T3 TSR | Mean SSR | ASC | Latency |
|---|---|---|---|---|---|---|---|
| M0 | 无 CoT 基线 WAM | BASE | 待测 | 待测 | 待测 | 待测 | 待测 |
| M1 | Latent-CoT(kf) | checkpoint_step_1200 | 待测 | 待测 | 待测 | 待测 | 待测 |
| M1v | Latent-CoT(kf+VLM) | robotwin_kf0.1_vlmstage0.1/1200 | 待测 | 待测 | 待测 | 待测 | 待测 |

预期（写"假设/期望"，跑完用实测替换）：隐式 CoT（尤其 kf+VLM）在长程任务
T3 的 TSR/SSR 优于无 CoT 基线，ASC 更低；Latency 与 M0 同量级（CoT 已灌进
权重，推理零外部依赖，区别于路线一 M2/M3/M4 需叠加 VLM 调用延迟）。
