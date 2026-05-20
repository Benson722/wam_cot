# 团队分工

## 项目概览

26 夏令营 WAM-CoT(路线二·**隐式物理 CoT**):在 LingBot-VA 世界模型基础上,
引入"关键帧距离"+"VLM 语义阶段"双辅助监督,把思维链灌进权重,推理零外部依赖。
交付 3 个对照 ckpt(M1 / M1v / M1v_WRONG)+ 4 项消融证据 + 探针可复现包。

---

## 个人分工(本人)

负责路线二全栈:数据生成 → 模型改造 → 训练 → 评测 → 消融 → 可解释性 → 文档。

| 模块 | 具体产出 |
|---|---|
| **模型改造** | `wan_va/modules/model.py` 新增 `kf_aux_head`(Latent-CoT #1)与 `stage_head`(Phase B);`forward_train` 5 元组改造;meta-tensor materialize 修复 |
| **数据生成** | `keyframe_annotate.py`(纯离线夹爪开合 → 关键帧)+ `qwen_stage_annotate.py`(本地 Qwen3.5-VL,8 GPU 并行任务分片 + 断点续跑去重,12 任务 × 500 ep,1 h 完成全 6000 集 VLM 阶段标注) |
| **训练** | 训练 3 个 ckpt:**M1**(kf-only)/ **M1v**(kf+VLM,主交付)/ **M1v_WRONG**(Ablation-3,错误置乱标签);训练配置(`va_robotwin_train_cfg.py` / `va_robotwin_train_wrongstage_cfg.py`)+ FSDP + cuDNN/wandb/磁盘配额/Pool fork 等工程修复 |
| **评测** | RoboTwin 仿真对接;两套 server/client wrapper(常规 + latent 出 dream_video,EVAL_ENV 源码不动);6 任务 SR 对照自动汇总;推理日志自动打印模型参数量 + 计算开销(`print_model_params.py`) |
| **消融(PDF 必做)** | **Ablation-1** 丢弃 CoT、**Ablation-2** 打乱 CoT 顺序(`script/run_ablation_explicit.sh`,3 档对照)+ **Ablation-3** 错误标记(`vlm_stage_corrupt='shuffle'` 数据腐化重训,`script/run_ablation_implicit.sh` 三阶段:train/probe/eval) |
| **可解释性** | Latent-CoT #4 冻结-backbone 线性探针 + t-SNE,4 ckpt 对照(stock 0.652 / kf 0.648 / **kf+VLM 0.778** / WRONG 0.638,chance 0.167);VLM 过程性评判(`judge_completion.py`,Qwen3-VL 看视频逐子目标打分) |
| **可复现性** | 探针 canonical 包(sha256 + expected val_acc),`script/reproduce_probe.sh` 无 GPU、~1 分钟字节级再现(4/4 PASS Δ=+0.000);完整 archive tarball(62 MB)+ README |
| **文档** | 主 `WAM_COT_README.md`(1800+ 行)/ `H200_TRAINING.md` / `EXPERIMENT_RESULTS.md` / `MODEL_AND_DATA.md` / `latent_plan*.md` |

---

## 其他成员(请按实际情况补充)

| 成员 | 主要职责(示例,填实际)|
|---|---|
| 成员 B | _______________(如:Robocasa 适配 / 路线一 External Semantic CoT / 演示视频)|
| 成员 C | _______________(如:VLM 服务搭建与维护 / 评测环境 eval_env / 数据预处理)|
| 成员 D | _______________(如:报告撰写 / 实验记录 / 失败案例归因)|

---

## 协作方式

- **代码协作**:本地 IDE 开发 → 手动同步到 SII 服务器
- **资源协作**:**训练**在 H200(8 卡,无外网),**数据生成 + 评测**在 4090(1 卡,有外网),数据集与产物统一存 qb-ilm2 共享盘(跨实例无缝)
- **流程**:数据生成 → 训练 → 离线探针自检 → 在线 RoboTwin 评测 → 消融对照 → 文档与交付
- **交付物**:3 个 ckpt + 探针 archive(62 MB tgz + sha256 README)+ 全部脚本/配置/文档(repo) + 报告
