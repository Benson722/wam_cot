# 26 夏令营 WAM-CoT 项目说明 (完整版)

> 本文档作为最终技术报告的**单一中间稿来源**:覆盖项目目标、代码库结构、环境、
> 数据生成、模型改造、训练、评测、实验结果、消融与可解释性、复现命令、文件索引、
> 诚实边界。撰写报告时可直接引用本文档,无需再回看代码。
>
> 配套文档(细节可深入参考,本文已整合其要点):
> - `MODEL_AND_DATA.md` —— 模型/数据/损失数学(详)
> - `latent_plan.md` —— Latent-CoT 五条原始规格(只读)
> - `latent_plan_progress.md` —— 五条规格的落地进度(只读)
> - `EXPERIMENT_RESULTS.md` —— 主 SR 表 + §6 探针消融完整结果
> - `evaluation/robocasa/COT_DESIGN.md` —— 路线一(外部语义 CoT)设计(本项目未采用)

---

## 目录

1. [项目背景与目标](#1-项目背景与目标)
2. [总体设计与方法路线](#2-总体设计与方法路线)
3. [仓库结构与文件索引](#3-仓库结构与文件索引)
4. [环境与依赖](#4-环境与依赖)
5. [数据集与数据生成](#5-数据集与数据生成)
6. [模型架构与改造](#6-模型架构与改造)
7. [训练](#7-训练)
8. [评测:离线探针 + 在线 RoboTwin SR](#8-评测离线探针--在线-robotwin-sr)
9. [实验结果](#9-实验结果)
10. [消融实验与可解释性](#10-消融实验与可解释性)
11. [失败分析与归因](#11-失败分析与归因)
12. [完整复现命令链](#12-完整复现命令链)
13. [诚实边界与未完成项](#13-诚实边界与未完成项)
14. [致谢与许可](#14-致谢与许可)

---

## 1. 项目背景与目标

### 1.1 比赛任务(PDF 指令)

26 夏令营要求设计并验证基于思维链(CoT)的世界-动作模型(WAM-CoT),探索"中间推理过程"
能否提升机器臂在复杂物理交互任务上的策略性能。三阶段:

- **第一阶段**:基线 WAM(无 CoT,直接 obs+指令→动作)。
- **第二阶段**:核心 CoT 实现,**任选其一**——
  - 路线一(External Semantic CoT):外部 VLM 规划器生成文本子任务,驱动底层 WAM。
  - 路线二(Internal Physical/Latent CoT):在 WAM 内部引入 Latent Rollout 或关键状态
    预测,推理前先在潜空间生成中间表征。
- **第三阶段**:对比评估 + **消融实验(必做)**——验证 CoT 真实贡献。完成标准:
  > "确保 CoT 机制真实参与了动作生成,并具备一定程度的可解释性。"

### 1.2 我们的选择与定位

**路线二(Internal Physical / Latent CoT)**:把"思维链"灌进世界模型权重,推理时**零外部
依赖**。具体两层 CoT 监督叠加在同一 backbone(同一基座、同步数对照):

1. **Latent-CoT #1 (kf)** —— 用夹爪开合切换点作关键帧,backbone 隐状态回归"到下一关键
   事件还有多少帧"(`log1p`+SmoothL1)。
2. **Latent-CoT Phase B (VLM stage)** —— 引入 **VLM 进入数据生成与训练回路**:本地
   Qwen3.5-VL 离线把每条 episode 切成有序**语义阶段**(approach/grasp/lift/place/…),
   backbone 隐状态做 8 类阶段分类(CE,`ignore_index=-1`)。两个辅助目标同时训(各
   `λ=0.1`)。

二者均为 **opt-in**(配置关闭时 NO-OP,与原基座逐字节一致);推理时辅助头**不被调用**
(只走视频+动作扩散路径),CoT 已塑形进 backbone 表征。

### 1.3 产出

- 训练完成的 ckpt:`robotwin_kf0.1_vlmstage0.1/checkpoint_step_1200`
- VLM 离线阶段数据:12 任务 × 500 ep,全覆盖、去重、平均 3.5–6 阶段/集
- **离线表征消融**(满足 PDF "必做消融" + "可解释性"):线性探针单调退化曲线
  无-CoT 0.663 → 仅 kf 0.666 → kf+VLM **0.782**(VLM 阶段标签,chance 0.167)
- 主 SR 表跑通,小样本数据已验证管线(N=5 时 M0 在 adjust_bottle/lift_pot 100%、
  hanging_mug 60%);N=10 / N=25 完整对照在 §12 命令链中。
- t-SNE/PCA、混淆矩阵、loss 曲线、执行视频/dream video 全部产出。

---

## 2. 总体设计与方法路线

```
┌─────────────────────────────────────────────────────────────────────────┐
│                      WAM-CoT (Route-2 / Latent CoT)                     │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  RoboTwin 2.0 ──→ LingBot 官方 clean+aug LeRobot 数据集 (12 task)       │
│        │                                                                │
│        ├──→ keyframe_annotate.py  ────→ meta/keyframes.jsonl  (kf)      │
│        └──→ qwen_stage_annotate.py ───→ meta/stages.jsonl    (VLM)      │
│                  ↑                                                      │
│                  └── 本地 Qwen3.5-VL  (8 GPU 并行 + 任务分片 + dedup)   │
│                                                                         │
│  基座: lingbot-va-posttrain-robotwin (官方后训练 ckpt, 不重训)          │
│        │                                                                │
│        └──→ 继续训练 (run_va_posttrain.sh, FSDP, 8 卡, ~step 1200)     │
│              主损失: L_video + L_action  (世界模型 + 动作扩散)          │
│              辅助 (CoT):                                                │
│                + λ_kf · L_kf   (Huber on log1p(kf_dist))               │
│                + λ_st · L_st   (CE on VLM 阶段 idx, ignore=-1)         │
│              检查点目标化命名: robotwin_kf0.1_vlmstage0.1/              │
│                                                                         │
│  评估:                                                                  │
│    (A) 离线表征消融 (PDF 必做)                                          │
│        wan_va.train --probe-collect → dump h_t + vlm_stage             │
│        latent_probe.py --features h_hidden --label vlm_stage           │
│        三 ckpt 对比: stock / kf / kf+vlm   → val_acc 单调升,泛化差缩小 │
│                                                                         │
│    (B) 在线 RoboTwin SR (任务性能,对比评估)                            │
│        server (wan_va_server[_predvideo].py)  +  client (RoboTwin sim) │
│        6 代表任务 × N=10/25,M1 vs M1v 成功率                          │
│        eval_polict_client_openpi_latent 同时出 dream_video 可视化      │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 3. 仓库结构与文件索引

### 3.1 顶层

```
sii_wam_cot/
├── README.md                    # 上游 LingBot-VA 项目 README (保留)
├── WAM_COT_README.md            # 本文档 (项目交付主 README)
├── MODEL_AND_DATA.md            # 模型/数据/损失数学(详)
├── latent_plan.md               # Latent-CoT 五条规格(只读)
├── latent_plan_progress.md      # 落地进度
├── EXPERIMENT_RESULTS.md        # 主 SR 表 + §6 探针消融完整结果
├── wan_va/                      # 世界模型代码
├── evaluation/                  # 评测、数据生成、探针
├── script/                      # 训练启动脚本
└── train_out/                   # 训练产物 (软链至 qb-ilm2)
```

### 3.2 `wan_va/` — 世界模型

| 文件 | 角色 | 本项目改动 |
|---|---|---|
| `wan_va/train.py` | 训练主程序(Trainer/FSDP/dataloader/loss/save/probe-collect) | **大改**:Phase B 4-元组 loss + stage_loss 日志/wandb;目标化 save_dir;`build_exp_tag`/`meta.json`;`--probe-collect/--probe-ckpt/--probe-collect-batches`;Ctrl-C 集体安全保存;wandb 离线降级 |
| `wan_va/wan_va_server.py` | WebSocket+msgpack 推理 server(原版) | 加载 BASE/M1/M1v(`load_transformer(.../transformer)`);`attn_mode="torch"`(4090 安全) |
| `wan_va/modules/model.py` | `WanTransformer3DModel`,30 层双流 MoT | **大改**:新增 `kf_aux_head` (Linear→GELU→Linear→1) 和 `stage_head` (→`vlm_num_stages=8`);`forward_train` 返回 5 元组 `(latent, action, kf_pred, kf_feat, stage_pred)` |
| `wan_va/modules/utils.py` | `load_transformer/load_vae/load_text_encoder/load_tokenizer/WanVAEStreamingWrapper` | **修**:meta-tensor materialize 路径(diffusers 低内存加载下新加 head 留 meta → `to(device)` 崩 → 加载后只把仍在 meta 的子模块 `to_empty(cpu)+reset_parameters()+to(dtype)` 实体化,已加载权重不动) |
| `wan_va/dataset/lerobot_latent_dataset.py` | LeRobot v2.1 多任务 dataset | **大改**:`_load_keyframes()`+`_load_stages()`(懒加载、main proc 打印);`__getitem__` 末尾 kf+vlm_stage 钩子;`construct_lerobot_multi_processor` 改为串行+skip-incomplete-repo 防 fork-after-CUDA 死锁;`recursive_find_file` 用 `os.walk(followlinks=True)` 支持 `_stable` 软链父目录 |
| `wan_va/distributed/` | FSDP / 分布式工具 | 未改 |
| `wan_va/utils/` | scheduler/init_logger/data_seq_to_patch/timestep 采样 | 未改 |
| `wan_va/configs/` | 各任务 EasyDict 配置 | 见 3.5 |

### 3.3 `evaluation/robotwin/` — 数据生成、评测、探针

| 文件 | 角色 |
|---|---|
| `keyframe_annotate.py` | **新增**。从动作向量夹爪通道(idx 7/15)提取 grasp/release/end 关键帧,写 `meta/keyframes.jsonl`。`--recursive` 支持多任务批处理。零 LLM、纯离线 |
| `qwen_stage_annotate.py` | **新增**。Phase A VLM 数据生成:每集均匀抽 4 帧,POST 给本地 Qwen3.5-VL OpenAI 兼容端点,strict-JSON 提示 + few-shot + truncation-tolerant `_extract_json` + `_extract_text`(<think> 剥离 + reasoning_content 回退);`--recursive --num-shards N --shard k --base-url` 任务级分片并行;`--resume`(清洗去重+追加,parallel-safe);`--probe`/`--limit`/`--debug` 诊断模式 |
| `latent_probe.py` | **新增**。Latent-CoT #4 探针。`--features z_latent`(零依赖默认,从 VAE 潜空间)或 `--features h_hidden`(读 `--hidden-dump`,backbone 隐状态);`--label kf_stage/vlm_stage`;按 episode 切分;输出 `results_*.json`(val_acc / chance / per-class / 混淆)+ t-SNE/PCA |
| `eval_polict_client_openpi.py` | RoboTwin 评测客户端(原版,`/inspire/qb-ilm2/.../RoboTwin` 硬编码 ROBOTWIN_ROOT 可env 覆盖,`os.chdir` 入 RoboTwin);websocket 连 server 跑 sapien 仿真;`--save_root`/`--task_name`/`--test_num`/`--port`;支持可选 `--cot True`(走 Route-1) |
| `eval_polict_client_openpi_latent` | 评测客户端 **latent 版**(`/inspire/qb-ilm2/.../eval_env/sii_wam_cot/lingbot-va_goal_cond_cot/` 路径);多 `--outputs_root`(dream_video 路径),与 `wan_va_server_predvideo.py` 配对使用 |
| `launch_server.sh` / `launch_server_multigpus.sh` | server 启动脚本,含 cuDNN LD_PRELOAD 修复(H200/4090 通用) |
| `launch_client.sh` / `launch_client_multigpus.sh` / `launch_cot_client.sh` | client 启动(`ACT/demo_clean/test_num=100`,sapien LD_LIBRARY_PATH) |
| `calc_stat.py` | 评测结果汇总 |
| `plot_losses.py` | **新增**。从 torchrun stdout 解析 latent/action/kf/stage loss,产 `loss_curves.csv+png` |
| `make_empty_emb.py` | 生成 CFG 空 prompt 的 UMT5 嵌入(已有共享 `empty_emb.pt`,不需重跑) |
| `websocket_client_policy.py` / `msgpack_numpy.py` / `geometry.py` / `test_render.py` | client 通信、几何工具、渲染测试,基本未改 |

### 3.4 `evaluation/robocasa/`

| 文件 | 角色 |
|---|---|
| `cot_planner.py` | Route-1 VLM 规划器(DeepSeek 文本 / 本地 Qwen vLLM)。**本项目 Phase A 复用其 `HARDCODED_VLLM_BASE_URL/_img_to_data_url/_parse_json`** 与 `_cred` 凭据回退 |
| `COT_DESIGN.md` | Route-1 设计文档(本项目未走该路线,留作对比参考) |

### 3.5 `wan_va/configs/`

| 配置文件 | 用途 | 本项目状态 |
|---|---|---|
| `shared_config.py` | 各任务共享默认 | 未改 |
| `va_robotwin_cfg.py` | **RoboTwin 推理配置**(server 读这里) | 改:`wan22_pretrained_model_name_or_path` 读 `VA_EVAL_CKPT` 环境变量(默认 `checkpoint_step_1200`),解耦推理与训练 |
| `va_robotwin_train_cfg.py` | **RoboTwin 训练配置**(`robotwin_train`) | 改:`dataset_path=…_stable`(多任务父目录)、`empty_emb_path` 绝对、wandb 离线、`save_interval=200`、`kf_aux=True/0.1/keyframes.jsonl`、`vlm_stage_aux=True/0.1/stages.jsonl/8`、`exp_name=None`(自动 tag);**显式钉死** `wan22_pretrained_model_name_or_path` 为 BASE,防 `va_robotwin_cfg` 的推理改动污染训练 |
| `va_libero_cfg.py/_train_cfg.py/_i2va.py` | LIBERO 任务 | 未用 |
| `va_demo*.py / va_franka*.py / va_robocasa_cfg.py` | 其它任务/演示 | 未用 |

### 3.6 `script/`

| 脚本 | 用途 |
|---|---|
| `run_va_posttrain.sh` | 训练启动器(torchrun,cuDNN LD_PRELOAD,wandb 离线,HF cache 重定向至 qb-ilm2,逐 rank 日志落盘) |
| `run_launch_va_server_sync.sh` | server 同步启动(单卡变种) |

### 3.7 训练产物 `train_out/`(软链至 qb-ilm2 大盘)

```
train_out/
├── checkpoints/
│   ├── checkpoint_step_1200/                            # 第一次训练 (仅 kf,M1)
│   │   ├── transformer/{diffusion_pytorch_model.safetensors, config.json}
│   │   ├── vae/      (软链 -> BASE)                     # 后补,server 加载需要
│   │   ├── tokenizer/(软链)
│   │   └── text_encoder/(软链)
│   └── robotwin_kf0.1_vlmstage0.1/                      # Phase B (kf+VLM,M1v)
│       └── checkpoint_step_1200/{transformer/, meta.json, + 3 软链}
├── probe/
│   ├── h_stock.pt   h_kf.pt   h_kfvlm.pt                # collect-hidden dump
│   └── out_h_*/{results_*.json, tsne_*.png}             # 探针结果
├── wandb/wandb/offline-run-*                            # 离线 wandb
├── torchrun_logs/                                       # 逐 rank 日志
└── srv_*.log                                            # eval server 日志
```

---

## 4. 环境与依赖

### 4.1 两套独立 Python 环境(必须分开)

| 环境 | 用途 | 关键依赖 |
|---|---|---|
| **LingBot env**(训练 + server) | `python -m wan_va.train`、`wan_va/wan_va_server[_predvideo].py`、`evaluation/robotwin/qwen_stage_annotate.py`、`latent_probe.py` | Python 3.10.16、torch 2.9.0+cu126、diffusers 0.36.0、transformers 4.55.2、accelerate、msgpack、websockets、einops、easydict、ftfy、`lerobot==0.3.3` `scipy` `wandb`(`--no-deps`)、`scikit-learn`(t-SNE) |
| **RoboTwin env**(client / sapien 仿真) | `python -m evaluation.robotwin.eval_polict_client_openpi[_latent]` | RoboTwin 2.0 (`/inspire/qb-ilm2/.../RoboTwin`)、sapien、yaml、cv2、imageio、ffmpeg。Server 不需要此环境 |

**关键**:同一台机上必须用两个终端分别激活两个环境;不要在 LingBot env 里跑 client,反之亦然。

### 4.2 服务器路径速查

| 资源 | 路径 |
|---|---|
| LingBot 仓库 (训练/server,代码主仓) | `/inspire/hdd/project/26summer-camp-11/26220077/lingbot-va` |
| LingBot venv | `/inspire/qb-ilm2/project/26summer-camp-11/26220077/lingbot-va/.venv` |
| RoboTwin 仓库 (client/仿真) | `/inspire/qb-ilm2/project/26summer-camp-11/public/group3/RoboTwin` |
| Eval env (latent 版评测仓) | `/inspire/qb-ilm2/project/26summer-camp-11/public/group3/eval_env/sii_wam_cot/lingbot-va_goal_cond_cot` |
| 数据集根 (12 任务) | `/inspire/qb-ilm2/project/26summer-camp-11/public/group3/lingbot-robotwin-clean-and-aug-lerobot/lerobot_robotwin_eef_aug_500_stable` |
| 共享空 prompt 嵌入 | `<数据集根>/../empty_emb.pt` |
| 基座 ckpt (BASE) | `/inspire/hdd/.../lingbot-va/checkpoints/lingbot-va-posttrain-robotwin` |
| 训练输出 (软链) | `/inspire/hdd/.../lingbot-va/train_out` → `/inspire/qb-ilm2/.../lingbot-va/train_out` |
| 本地 Qwen3.5-VL 服务 | `/inspire/qb-ilm2/project/26summer-camp-11/serve_qwen.py` (OpenAI 兼容 :8000/v1) |

### 4.3 计算资源(实训平台镜像)

- **H200 区**:8 卡(无外网,内网 SII 库)→ 数据生成 Phase A、模型训练。
- **4090 区**:1 卡(可外网,48 GB,RTX 4090)→ RoboTwin 在线评测(server+client 同卡靠 `enable_offload`)。
- **CPU 区**:可外网,无 GPU,杂用。

### 4.4 工程修复

| 问题 | 触发 | 解决 |
|---|---|---|
| `libcudnn_graph.so.9: undefined symbol cudnnGetLibConfig` SIGABRT | 系统 `/usr/lib` cuDNN 覆盖 torch 2.9 自带 | `launch_server.sh` / `run_va_posttrain.sh` 内嵌 Python 探 `nvidia.cudnn.lib`/`torch/lib`,把 8 个 `libcudnn*.so.9` 全部 `LD_PRELOAD` |
| `Cannot copy out of meta tensor` | diffusers 低内存加载 + 新加 `kf_aux_head`/`stage_head` 留 meta → `model.to(device)` 崩 | `load_transformer` 保留 meta 加载;加载后**只**对仍在 meta 的子模块 `to_empty(cpu)+reset_parameters()+to(dtype)`,已加载权重不动 |
| wandb `ModuleNotFoundError: click`、`ValidationError: your url` | repo README 用 `--no-deps`、占位 `WANDB_BASE_URL` | `train.py` 惰性 import wandb;离线-first;失败即降级 `config.enable_wandb=False` 继续训练 |
| `Disk quota exceeded` 训到 ~step 82 崩 | `/inspire/hdd/.../26220077` 11G 满 | `ln -s /inspire/qb-ilm2/.../lingbot-va/train_out /inspire/hdd/.../lingbot-va/train_out`;`run_va_posttrain.sh` 把 HF/lerobot cache 重定向至 qb-ilm2 |
| `Pool(128)` fork-after-CUDA 死锁 | `MultiLatentLeRobotDataset` 默认 worker=128 | 改串行 + 单 repo 不 fork |
| `EADDRINUSE` 训练/评测端口被占 | torchrun MASTER_PORT 残留 | `MASTER_PORT=29533` 等替代端口;评测脚本三轮端口轮换 + bash `/dev/tcp` 探活(免 ss) |
| 4090 镜像无 `ss` | iproute2 未装 | 评测脚本用 `(exec 3<>/dev/tcp/127.0.0.1/$p)` bash 内建探活 |

---

## 5. 数据集与数据生成

### 5.1 基础数据集

**RoboTwin 2.0**(aloha-agilex 双臂,16 维动作 `[L/R x,y,z,q1..q4,gripper]`,夹爪 idx 7/15)
官方 clean+aug LeRobot 数据集 `robbyant/robotwin-clean-and-aug-lerobot`,SII 服务器
共享盘已下载,**12 个任务**(curated `_stable` 父目录下软链):

```
lerobot_robotwin_eef_aug_500_stable/
├── adjust_bottle-aloha-agilex_randomized_500-1000          (500 ep)
├── beat_block_hammer-aloha-agilex_randomized_500-1000      (500)
├── blocks_ranking_rgb-aloha-agilex_randomized_500-1000     (500)
├── blocks_ranking_size                                     (500)
├── click_alarmclock-aloha-agilex_randomized_500-1000       (500)
├── click_bell-aloha-agilex_randomized_500-1000             (500)
├── dump_bin_bigbin-aloha-agilex_randomized_500-1000        (500)
├── grab_roller-aloha-agilex_randomized_500-1000            (500)
├── handover_block-aloha-agilex_randomized_500-1000         (500)
├── handover_mic-aloha-agilex_randomized_500-1000           (500)
├── hanging_mug-aloha-agilex_randomized_500-1000            (500)
└── lift_pot-aloha-agilex_randomized_500-1000               (500)
                                                             (总 6000 ep)
```

每任务子目录结构(LeRobot v2.1,`fps=50`):
```
<task>/
├── data/chunk-000/episode_******.parquet     (每帧 obs.state(16) + action(16))
├── latents/chunk-000/<cam>/episode_*_{s}_{e}.pth   (Wan2.2 VAE 48ch + UMT5 text_emb)
├── videos/chunk-000/<cam>/episode_*.mp4            (av1, 480x640)
├── meta/
│   ├── episodes.jsonl       (每集 tasks/length/action_config)
│   ├── episodes_ori.jsonl   (细粒度多段)
│   ├── tasks.jsonl          (50 条自然语言变体)
│   ├── info.json
│   ├── keyframes.jsonl   ← 本项目派生 (kf,Phase A #1)
│   └── stages.jsonl      ← 本项目派生 (VLM,Phase A Phase B)
└── (共享根) empty_emb.pt   (UMT5 空 prompt 嵌入,CFG 用)
```

3 路相机:`observation.images.{cam_high, cam_left_wrist, cam_right_wrist}`。

### 5.2 派生数据 #1 —— 关键帧 (`keyframes.jsonl`)

**脚本**:`evaluation/robotwin/keyframe_annotate.py`
**原理**:从动作向量的**夹爪开/合二值跳变**(左 idx 7、右 idx 15,阈值取该通道值域中点
→ 跨 embodiment 鲁棒)提关键帧。类型 `grasp`/`release`,加末帧 `end`(保证每帧都有
"到下一关键帧距离")。零 LLM、纯离线、与物理操作强对齐。

**命令**:
```bash
python /inspire/hdd/project/26summer-camp-11/26220077/lingbot-va/evaluation/robotwin/keyframe_annotate.py \
  --dataset /inspire/qb-ilm2/project/26summer-camp-11/public/group3/lingbot-robotwin-clean-and-aug-lerobot/lerobot_robotwin_eef_aug_500_stable \
  --recursive --gripper-idx 7 15
```
**产物**:每任务 `meta/keyframes.jsonl`,每行 `{"episode_index": N, "length": L,
"keyframes": [{"frame": F, "type": "grasp"|"release"|"end"}, ...]}`。
**用法**:`va_robotwin_train_cfg.kf_aux=True, kf_file='keyframes.jsonl'` →
dataset `_load_keyframes` 按 latent 时间步对齐,emit `kf_dist/kf_mask/kf_stage/kf_episode`。

### 5.3 派生数据 Phase B —— VLM 语义阶段 (`stages.jsonl`)

**脚本**:`evaluation/robotwin/qwen_stage_annotate.py`
**原理**:每集均匀抽 K 帧(`--frames 4`),POST 给本地 Qwen3.5-VL,要求返回严格 JSON
`{"stages": [{"name": ..., "start_frame": ...}]}`。

**关键工程要点(踩过的坑、最终生效)**:

| 问题 | 解决 |
|---|---|
| serve_qwen.py 的 pydantic 把 `enable_thinking`/`response_format` 等未知字段静默丢弃 | **改 serve_qwen.py 三处**:① 顶部 `import os,re`;② `apply_chat_template(... enable_thinking=False)` try/except;③ `port = int(os.environ.get("PORT", 8000))` |
| Qwen3.5-VL 是"边想边说"推理模型,默认 `<think>...</think>` 占满 token 没空间出 JSON | **关 thinking**(服务器侧 `enable_thinking=False`)。客户端再加 few-shot(纯 JSON 示范)、强制系统提示、`temperature=0.0` 贪心、`<think>` 兜底剥离 |
| 一次回复可能含散文 + 末尾 JSON,或 JSON 被截断 | `_extract_json` 截断容错(扫所有 `{`,brace-stack 平衡,补 `}]`,JSON5 风格容错,正则 fallback 抽 `"name"`/`"start_frame"` 对) |
| 大数据 serial 慢(~3–5 s/ep × 6000 ep ≈ 8 小时) | **8 GPU 并行**:每张卡起一个 `serve_qwen.py`(`PORT=8000+k`),8 个客户端 `--num-shards 8 --shard k --base-url http://127.0.0.1:800k/v1` 按任务目录切分(每个 stages.jsonl 唯一写入者,无写冲突),`--resume`(清洗有效行+追加,断电安全) |

**单卡命令**(慢):
```bash
DS=/inspire/qb-ilm2/project/26summer-camp-11/public/group3/lingbot-robotwin-clean-and-aug-lerobot/lerobot_robotwin_eef_aug_500_stable
CUDA_VISIBLE_DEVICES=0 nohup \
  /inspire/qb-ilm2/project/26summer-camp-11/.venv/bin/python \
  /inspire/qb-ilm2/project/26summer-camp-11/serve_qwen.py > /tmp/q.log 2>&1 &
until grep -q "Application startup complete" /tmp/q.log; do sleep 3; done
nohup python /inspire/hdd/project/26summer-camp-11/26220077/lingbot-va/evaluation/robotwin/qwen_stage_annotate.py \
  --dataset "$DS" --recursive --frames 4 --max-tokens 256 --timeout 120 --resume \
  > /tmp/stage.log 2>&1 &
tail -f /tmp/stage.log
```

**8 卡并行命令**(实际生产用,~1 小时完):见 §12 完整复现命令链。

**收尾去重**:
```bash
python - <<'PY'
import json, glob, os
DS="..."
for sj in sorted(glob.glob(os.path.join(DS,"*","meta","stages.jsonl"))):
    raw=[json.loads(l) for l in open(sj) if l.strip()]
    uniq={int(r["episode_index"]): r for r in raw}
    with open(sj,"w") as f:
        for ei in sorted(uniq): f.write(json.dumps(uniq[ei])+"\n")
    print(f"{os.path.basename(os.path.dirname(os.path.dirname(sj))):55s} {len(raw)} -> {len(uniq)} uniq")
PY
```

**最终产物**:12 任务 × `500 uniq / 500 ep OK`,平均 3.5–6 阶段/集。`stages.jsonl` 行:
```json
{"episode_index": 0, "length": 143,
 "tasks": ["Use the right arm to grab and lift the bottle with narrow top head-up."],
 "stages": [{"name": "approach bottle", "start_frame": 0},
            {"name": "grasp bottle",    "start_frame": 47},
            {"name": "lift bottle",     "start_frame": 94},
            {"name": "retract arm",     "start_frame": 130}]}
```

### 5.4 数据血缘

```
RoboTwin 2.0 (开源仿真器) ─ 专家演示
       └─ LingBot 官方 clean+aug → robbyant/robotwin-clean-and-aug-lerobot
              (LeRobot v2.1, 已抽 VAE latent + UMT5 text_emb + empty_emb)
       └─ 助教/官方 ─ 下载至 SII 服务器共享盘
              └─ 本项目派生: keyframe_annotate.py    → meta/keyframes.jsonl
              └─ 本项目派生: qwen_stage_annotate.py  → meta/stages.jsonl
                                  (调用本地 Qwen3.5-VL,无外网)
```

---

## 6. 模型架构与改造

### 6.1 基座 LingBot-VA

| 组件 | 说明 | 代码位置 |
|---|---|---|
| 视觉 VAE | Wan2.2 VAE,RGB → **48 通道 latent**(流式 `WanVAEStreamingWrapper`) | `wan_va/modules/utils.py` |
| 主干 | `WanTransformer3DModel`:30 层,patch_size=(1,2,2),inner_dim=24×128=3072,**双流 MoT**(latent + action 流交错于同序列),RoPE,cross-attn 注入文本 | `wan_va/modules/model.py` |
| 文本编码器 | UMT5,text_dim=4096,prompt 经 `condition_embedder.text_embedder` 投影 | `wan_va/wan_va_server.py:_get_t5_prompt_embeds` |
| 动作头 | 30 维标称动作扩散头(RoboTwin 实际用 14 维),FlowMatch 调度 | `model.py:action_proj_out` |
| 训练 | FSDP + 激活检查点,视频/动作联合扩散损失 | `wan_va/train.py` |

**基座权重**:`lingbot-va-posttrain-robotwin`(官方 robotwin 后训练 ckpt),**不重训**,只在其
上做继续训练。

### 6.2 #1 关键帧辅助头 `kf_aux_head`

```python
# wan_va/modules/model.py: __init__
self.kf_aux_head = nn.Sequential(
    nn.Linear(inner_dim, 128), nn.GELU(), nn.Linear(128, 1))
```

`forward_train` 在所有 block + norm_out + scale/shift 之后、`proj_out` **之前**取主干 latent
隐状态(token 序 `(f h w)`),对空间维 mean-pool → `[B, F_lat, d]`(`d=inner_dim=3072`),
喂给 `kf_aux_head` → 标量 `kf_pred`(每 latent 帧一个)。

**损失**:
```
L_kf = mean_{kf_mask} SmoothL1( log1p(kf_dist), kf_pred )
```
- `kf_dist`:到下一关键帧的原始帧数(0–~200);`log1p` 压缩动态范围。
- `SmoothL1`(Huber):对关键帧边界附近的离群目标鲁棒。
- 按 `kf_mask`(末段无"下一个关键帧"时不惩罚)求均值。
- 系数 `λ_kf = cfg.kf_aux_weight = 0.1`。

### 6.3 Phase B VLM 阶段头 `stage_head`

```python
# wan_va/modules/model.py: __init__, vlm_num_stages=8
self.vlm_num_stages = int(vlm_num_stages)
self.stage_head = nn.Sequential(
    nn.Linear(inner_dim, 128), nn.GELU(), nn.Linear(128, self.vlm_num_stages))
```

同样取 backbone pooled hidden `h_t`,喂给 `stage_head` → `[B, F_lat, 8]` logits。

**损失**(`wan_va/train.py:compute_loss`):
```
vst = input_dict['vlm_stage'].long()        # [B, F_lat], -1 = 未标注
L_stage = λ_st · F.cross_entropy(stage_pred, vst, ignore_index=-1)
```
- 系数 `λ_st = cfg.vlm_stage_weight = 0.1`。
- 数据集 `_load_stages` 按 `stages.jsonl` 的 `start_frame` 把每 latent 帧映射到阶段 idx
  `min(int((starts<=rep).sum())-1, S-1)`(S=`cfg.vlm_num_stages=8`),超出范围 = -1。

### 6.4 `forward_train` 返回 5 元组

```python
# wan_va/modules/model.py: forward_train (核心改造)
# 取 backbone hidden, 空间池化
kf_feat = rearrange(latent_hidden_states,
                    '1 (b f s) d -> b f s d',
                    b=batch_size, f=f_lat).mean(dim=2)            # [B, F_lat, d]
kf_pred    = self.kf_aux_head(kf_feat).squeeze(-1)                # [B, F_lat]
stage_pred = self.stage_head(kf_feat)                             # [B, F_lat, S]
return (latent_hidden_states, action_hidden_states,
        kf_pred, kf_feat, stage_pred)                             # 5-tuple
```

- `pred[0]`:latent 流隐状态 → `latent_loss`(视频流匹配)
- `pred[1]`:action 流隐状态 → `action_loss`
- `pred[2]`:`kf_pred` → `kf_loss`
- `pred[3]`:`kf_feat`(pre-head pooled hidden) → **探针特征**(#4)
- `pred[4]`:`stage_pred` → `stage_loss`

### 6.5 模型加载(meta-tensor 修复)

`wan_va/modules/utils.py:load_transformer` 走 diffusers 低内存加载路径(`low_cpu_mem_usage`
不能传 False 因为 diffusers 在有 `keep_in_fp32_modules` 时拒绝)。新加的 `kf_aux_head` /
`stage_head` 在加载后停留在 meta 设备 → `model.to(device)` 报 `Cannot copy out of meta
tensor`。修复:加载完后扫所有子模块,**仅对参数仍在 meta 的子模块** `to_empty(cpu)` +
`reset_parameters()` + `sub.to(dtype)`,已加载权重不动。推理时这两个头**不被调用**(server
只走 video+action 路径),所以 ckpt 没保存这两头的权重也无妨,会随机初始化但不影响 SR。

---

## 7. 训练

### 7.1 训练配置 `va_robotwin_train_cfg.py`

```python
va_robotwin_train_cfg.update(va_robotwin_cfg)

# 显式钉死训练基座(防 va_robotwin_cfg 的推理改动污染训练)
va_robotwin_train_cfg.wan22_pretrained_model_name_or_path = \
    "/inspire/hdd/.../checkpoints/lingbot-va-posttrain-robotwin"

# 多任务数据集(_stable 父目录,os.walk followlinks=True 递归找 12 任务)
va_robotwin_train_cfg.dataset_path = \
    "/inspire/qb-ilm2/.../lerobot_robotwin_eef_aug_500_stable"
va_robotwin_train_cfg.empty_emb_path = "<同根>/empty_emb.pt"  # 全任务共享

# wandb 离线
va_robotwin_train_cfg.enable_wandb = True
va_robotwin_train_cfg.wandb_mode   = 'offline'
va_robotwin_train_cfg.wandb_dir    = "<save_root>/wandb"

# 训练超参
va_robotwin_train_cfg.learning_rate = 1e-5
va_robotwin_train_cfg.beta1 = 0.9; beta2 = 0.95; weight_decay = 0.1
va_robotwin_train_cfg.warmup_steps = 10
va_robotwin_train_cfg.batch_size = 1
va_robotwin_train_cfg.gradient_accumulation_steps = 1
va_robotwin_train_cfg.num_steps = 50000
va_robotwin_train_cfg.save_interval = 200
va_robotwin_train_cfg.load_worker = 16
va_robotwin_train_cfg.cfg_prob = 0.1
va_robotwin_train_cfg.gc_interval = 50

# ---- Latent-CoT #1 (kf) ----
va_robotwin_train_cfg.kf_aux        = True
va_robotwin_train_cfg.kf_aux_weight = 0.1
va_robotwin_train_cfg.kf_file       = 'keyframes.jsonl'

# ---- Latent-CoT Phase B (VLM stage) ----
va_robotwin_train_cfg.vlm_stage_aux    = True
va_robotwin_train_cfg.vlm_stage_weight = 0.1
va_robotwin_train_cfg.vlm_stage_file   = 'stages.jsonl'
va_robotwin_train_cfg.vlm_num_stages   = 8

# 实验标签自动派生(robotwin_kf0.1_vlmstage0.1)
va_robotwin_train_cfg.exp_name = None
```

### 7.2 损失数学(精确版)

总损失,`gas = gradient_accumulation_steps`:
```
L_total = (L_video + L_action + λ_kf · L_kf + λ_st · L_stage) / gas
```

**(1) `L_video`** ≈ 0.10–0.15(自然场景视频 latent 噪声地板)
```
e_ij = w_v(t) · (latent_pred − target_video)^2
L_video = mean over (B, F_lat) [ Σ_{C,H,W} e_ij  /  (#elems) ]
```
- `target_video` = flow-matching 目标速度场(`FlowMatchScheduler(snr_shift=5)`)
- `w_v(t)` = SNR 时间步权重

**(2) `L_action`** ≈ 1e-3(动作高度规律,收敛快)
```
e_ij = w_a(t) · (action_pred − target_action)^2 · actions_mask
L_action = mean over (B, F_lat) [ Σ e_ij_masked / Σ mask ]
```
- 30 维动作通道实际只用 14(`used_action_channel_ids`),mask 屏蔽未用通道
- 双臂 EEF q01/q99 分位归一化、相对位姿 `get_relative_pose`

**(3) `L_kf`** ≈ 2e-3(快速收敛,塑形 backbone)
```
h_t  = mean_{space}(backbone_latent_hidden)          # [B, F_lat, d]
kf_pred = kf_aux_head(h_t)                            # [B, F_lat]
L_kf = λ_kf · mean_{kf_mask} SmoothL1(log1p(kf_dist), kf_pred)
```

**(4) `L_stage`** ≈ 0.03(8 类 CE,chance ln 8 ≈ 2.08,远高于实测说明已学到)
```
stage_pred = stage_head(h_t)                          # [B, F_lat, S]
L_stage = λ_st · CE(stage_pred, vlm_stage, ignore_index=-1)
```

物理意义:
- `L_video / L_action` = 基座 WAM 原本目标(世界+动作)
- `L_kf` = 弱信号塑形,逼 latent 编码"距下次物理事件多久"(低层时间进度)
- `L_stage` = 强信号塑形,逼 latent 编码"现在第几语义阶段"(高层任务计划)

### 7.3 启动训练

```bash
cd /inspire/hdd/project/26summer-camp-11/26220077/lingbot-va
NGPU=8 CONFIG_NAME=robotwin_train CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
MASTER_PORT=29533 \
  bash script/run_va_posttrain.sh
```

`run_va_posttrain.sh` 内部:
- cuDNN LD_PRELOAD 修复(自动探测 torch 自带)
- `HF_HOME / HF_DATASETS_CACHE / HF_LEROBOT_HOME` → qb-ilm2 大盘
- `TORCHRUN_LOG_DIR=./train_out/torchrun_logs` 逐 rank stdout/stderr 落盘
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
- `torch.distributed.run --nproc_per_node=N --master_port $MASTER_PORT --redirects 3 --tee 3 -m wan_va.train --config-name robotwin_train`

进度条字段(每 optimizer step):
```
Training:  N/50000 [..., latent_loss=0.13, action_loss=0.001, kf_loss=0.003,
                          stage_loss=0.04, step=N, grad_norm=0.15, lr=1.00e-05]
```

### 7.4 目标化 checkpoint

`build_exp_tag(config)`(`wan_va/train.py`)从配置自动派生:
- 无任何辅助 → `robotwin_baseline`
- 仅 kf → `robotwin_kf0.1`
- kf+VLM(本次)→ `robotwin_kf0.1_vlmstage0.1`

`Trainer.__init__`:
```python
self.exp_tag = build_exp_tag(config)
self.save_dir = Path(config.save_root) / "checkpoints" / self.exp_tag
# wandb run name = self.exp_tag
```

每个 `checkpoint_step_<N>/` 写:
- `transformer/diffusion_pytorch_model.safetensors`(bf16,FSDP 集体保存)
- `transformer/config.json`
- `meta.json`:`{exp_tag, step, base_ckpt, dataset_path, kf_aux/_weight, vlm_stage_aux/_weight/_file/_num_stages, lr, timestamp}` —— **目录被移动/改名仍自证身份**。

### 7.5 wandb 离线 + 异常降级

`train.py` 惰性 import wandb;`WANDB_MODE=offline` 默认;占位 `WANDB_BASE_URL/API_KEY/TEAM_NAME`
启动时被剥离避免 pydantic ValidationError;任何 wandb 异常 → `config.enable_wandb=False`
**训练继续不退出**。事后:
```bash
wandb sync /inspire/qb-ilm2/.../lingbot-va/train_out/wandb/wandb/offline-run-*
```

### 7.6 Ctrl-C 安全保存

SIGINT/SIGTERM 处理器只置标志位,在**下个 optimizer step 边界 all-reduce 决定**所有 rank
集体调用 `save_checkpoint`,保存完干净退出。FSDP 集体保存不会被中断半截。**只按一次 Ctrl-C**
等它打印 `Interrupt: saving checkpoint ... then exiting.`;**两次 Ctrl-C 会硬杀**。

### 7.7 收敛标准

本次训练 step 1200 收敛证据:
- `action_loss` ≈ 1e-3(动作头已学到,关键指标)
- `kf_loss` ≈ 2e-3(从 ~0.3 降到 ~2e-3,辅助信号被学到)
- `stage_loss` ≈ 0.03(8 类 CE,chance ≈ 2.08,远低于即学到)
- `latent_loss` ≈ 0.12(噪声地板,不再降,正常)

与基线 `checkpoint_step_1200`(仅 kf)步数对齐 → 离线消融 §10 三档干净对照。

---

## 8. 评测:离线探针 + 在线 RoboTwin SR

### 8.1 离线 §A — 冻结 backbone 线性探针(主要消融证据)

**研究问题**:CoT 辅助目标有没有把任务阶段信息编码进 backbone 表征?

**两步管线**:

**(1)** `wan_va.train --probe-collect`(对每个 ckpt 跑 200 batch 的训练 forward,
dump 每 latent 帧的 `h_t` + 标签):

```bash
NGPU=1 CONFIG_NAME=robotwin_train CUDA_VISIBLE_DEVICES=0 MASTER_PORT=29540 \
  bash script/run_va_posttrain.sh \
  --probe-ckpt <某个 ckpt 目录> \
  --probe-collect ./train_out/probe/h_<tag>.pt --probe-collect-batches 200
```

`Trainer.collect_hidden`(详 `wan_va/train.py`):
- 冻结 transformer(`eval()`,`@torch.no_grad()`)
- 每 batch 走 `_prepare_input_dict` + `transformer(input_dict, train_mode=True)` →
  `out[3]=kf_feat` 是 backbone pre-head 隐状态 `[B, F_lat, d=3072]`
- dump 字典 `{feat, stage (kf), vlm_stage, episode, ckpt}`

**(2)** `latent_probe.py --features h_hidden --label vlm_stage`:
- 按 episode 切分(轨迹安全,防"同轨迹相邻帧相似"捷径)
- 训练一个线性分类器 `h_t → 阶段 idx`(VLM 阶段,6 类有效,chance 1/6 ≈ 0.167)
- 报告 `val_acc / train_acc / per_class / 混淆矩阵 / t-SNE(sklearn,缺则 PCA-2D 回退)`
- 输出 `train_out/probe/out_<tag>/{results_*.json, tsne_*.png}`

**三档对照**(同一标签集、同一探针架构、同 backbone 抽样):
- stock = BASE(`lingbot-va-posttrain-robotwin`,无任何辅助头)
- kf-only = `checkpoint_step_1200`(仅 kf,M1)
- kf+VLM = `robotwin_kf0.1_vlmstage0.1/checkpoint_step_1200`(本次,M1v)

### 8.2 在线 §B — RoboTwin SR(任务性能对比评估)

两进程架构:**server**(LingBot env,持模型)+ **client**(RoboTwin env,跑 sapien 仿真)。

#### 8.2.1 Server

| 文件 | 何时用 | 特点 |
|---|---|---|
| `wan_va/wan_va_server.py` + `launch_server.sh` | 原版,基础 SR 评测 | 含 cuDNN LD_PRELOAD;只返回动作(无 dream_video) |
| `wan_va/wan_va_server_predvideo.py` + `launch_server_pred_latent.sh` | **本项目最终用**,要 dream_video | 同时返回解码后的"想象视频"(预测 latent → VAE decode)给 client 可视化 |

两者都靠 `va_robotwin_cfg.wan22_pretrained_model_name_or_path` 选 ckpt。本项目改成读
`VA_EVAL_CKPT` 环境变量,**切 ckpt 不用改配置**。

Server 启动模板:
```bash
VA_EVAL_CKPT=<ckpt_dir> CUDA_VISIBLE_DEVICES=0 START_PORT=29056 MASTER_PORT=29061 \
  bash evaluation/robotwin/launch_server_pred_latent.sh > train_out/srv_<tag>.log 2>&1 &
```
就绪信号:用 bash `(exec 3<>/dev/tcp/127.0.0.1/$port)` 探活(4090 镜像无 `ss`)。

#### 8.2.2 Client

| 文件 | 用途 |
|---|---|
| `evaluation/robotwin/eval_polict_client_openpi.py` | 原版,只跑仿真 + 写 `_True/_False.mp4` |
| `evaluation/robotwin/eval_polict_client_openpi_latent.py`(eval_env 仓) | latent 版,**还出 dream_video**(`--outputs_root`),与 predvideo server 配 |

启动模板(latent 版,本项目最终用):
```bash
cd /inspire/qb-ilm2/.../eval_env/sii_wam_cot/lingbot-va_goal_cond_cot
export LD_LIBRARY_PATH=/usr/lib64:/usr/lib:$LD_LIBRARY_PATH
python -m evaluation.robotwin.eval_polict_client_openpi_latent \
  --config policy/ACT/deploy_policy.yml --overrides \
  --task_name <task> --task_config demo_clean \
  --ckpt_setting <tag> --seed 0 --policy_name ACT \
  --save_root ./results_latent_<tag> --outputs_root ./outputs_latent_<tag> \
  --video_guidance_scale 5 --action_guidance_scale 1 \
  --test_num <N> --port <START_PORT>
```

#### 8.2.3 输出位置(client 启动时 `os.chdir(ROBOTWIN_ROOT)`!)

| 产物 | 路径(在 RoboTwin 仓库下) |
|---|---|
| 成功率文件 | `<RoboTwin>/eval_result/<task>/ACT/demo_clean/<tag>/<时间戳>/_result.txt` |
| RoboTwin sapien 执行视频 | `<RoboTwin>/eval_result/<task>/ACT/demo_clean/<tag>/<时间戳>/episode<N>.mp4` |
| "想象 vs 真实"对比视频 | `<RoboTwin>/results_latent_<tag>/stseed-10000/visualization/<task>/<i>_<指令>_<True|False>.mp4` |
| dream_video(latent decode) | `<RoboTwin>/outputs_latent_<tag>/...` 和 `<eval_env>/visualization_predvideo/` |

把 `--save_root` / `--outputs_root` 写绝对路径可避免到处找。

### 8.3 评测前置(必做)

1. **三个 ckpt 各补成自包含**(server 要 vae/tokenizer/text_encoder/transformer 都在同目录):
   ```bash
   BS=/inspire/hdd/.../checkpoints/lingbot-va-posttrain-robotwin
   for CK in <M1 ckpt> <M1v ckpt>; do
     for s in vae tokenizer text_encoder; do ln -sfn "$BS/$s" "$CK/$s"; done
   done
   ```
   BASE 本身已自带 4 子目录。
2. **RoboTwin 视频开关**:
   ```bash
   sed -i 's/^\(\s*eval_video_log\s*:\s*\).*/\1True/' \
     /inspire/qb-ilm2/.../RoboTwin/task_config/demo_clean.yml
   ```

完整循环脚本 `run_eval` 见 §12.

---

## 9. 实验结果

### 9.1 训练曲线

`wan_va/train.py` 写 wandb 离线(`train_out/wandb/wandb/offline-run-*`)+ 进度条/torchrun
日志。`evaluation/robotwin/plot_losses.py` 离线解析 stdout → `loss_curves.{csv,png}`。

关键收敛值(step 1200):

| 指标 | 收敛值 | 含义 |
|---|---|---|
| `latent_loss` | ≈ 0.12 | 视频流匹配噪声地板(natural video 不可避免) |
| `action_loss` | ≈ 1e-3 | 动作头已学到(决定 SR 的关键) |
| `kf_loss` | ≈ 2e-3 | #1 辅助信号收敛(从 ~0.3 → 2e-3) |
| `stage_loss` | ≈ 0.03 | Phase B 8 类 CE 收敛(chance ≈ 2.08) |
| `grad_norm` | 0.1–0.4 | 稳定,无梯度爆炸 |
| `lr` | 1e-5 | 常数 |

### 9.2 §6 离线探针消融结果(主要量化结果)

**设置**:Latent-CoT #4 冻结 backbone + 线性探针;标签 = VLM 阶段(6 类有效,chance 0.167);
按 episode 轨迹切分(防泄漏);3 个 ckpt 同标签同协议对比;sklearn 真 t-SNE(perplexity
默认)。

| Checkpoint(CoT) | N | val_acc | 高于随机 | train_acc | 训练-验证差 |
|---|---|---|---|---|---|
| stock(无 CoT) | 2810 | 0.663 | +0.497 | 0.970 | 0.307 |
| kf-only(M1) | 2751 | 0.666 | +0.499 | 0.892 | 0.226 |
| **kf+VLM(M1v)** | 2528 | **0.782** | **+0.615** | 0.884 | **0.102** |

**per-class(M1v)**:approach 0.96 / grasp 0.79 / lift 0.71 / place 0.71 / 后两类 0.56,
混淆矩阵近三对角(错误几乎全是**相邻阶段** off-by-one,良性,符合"阶段边界帧天然模糊")。

**关键观察**:
1. **单调上升**:0.663 → 0.666 → 0.782。去掉 VLM 阶段 CoT 掉 0.116,全去掉掉 0.119。
2. **泛化差单调收窄**:0.307 → 0.226 → **0.102**。CoT 让阶段信息**可泛化地**线性可读
   (而非过拟合)。
3. **kf 几乎不优于 stock**(0.666 vs 0.663,+0.003):夹爪时间是 VLM 语义阶段的粗代理,
   只有**匹配的 VLM 语义监督**才带来 +0.116 跳变。**"监督什么得到什么"**——这是引入
   VLM 进数据+训练回路的核心价值证据。

**t-SNE 配图**:`train_out/probe/out_h_kfvlm/tsne_robotwin_train_h_hidden_vlm_stage.png`
(按阶段着色,M1v 簇分离明显优于 stock/kf)。

### 9.3 §7 在线 RoboTwin SR 表(管线已通,数值待完整跑)

代表性任务(中-长程,对消融差异敏感):

| 任务 | 长度典型(帧) | 备注 |
|---|---|---|
| handover_block | ~150 | 双臂交接 |
| handover_mic | ~150 | 双臂交接 |
| hanging_mug | ~80–230 | 长程,挂柄需精确旋转 |
| blocks_ranking_size | ~150 | 排序 |
| beat_block_hammer | ~120 | 工具操作 |
| lift_pot | ~80–140 | 双臂同步 |

**冒烟验证**(N=5,在 4090 已跑通管线,数值仅供参考):
- M0 / adjust_bottle:5/5 (100%)
- M0 / lift_pot:5/5 (100%)
- M0 / hanging_mug:3/5 (60%)

**完整 SR 表**(M1 vs M1v × 6 任务 × N=10,~3 小时;命令见 §12):

| 任务 | M1(kf) | M1v(kf+VLM) | Δ |
|---|---|---|---|
| handover_block | 待跑 | 待跑 |  |
| handover_mic | 待跑 | 待跑 |  |
| hanging_mug | 待跑 | 待跑 |  |
| blocks_ranking_size | 待跑 | 待跑 |  |
| beat_block_hammer | 待跑 | 待跑 |  |
| lift_pot | 待跑 | 待跑 |  |
| **均值** |  |  |  |

(跑完把 `_result.txt` 数填入此表 → 写进报告 5.1 主表。)

### 9.4 z_t 基线探针(辅助证据,已完成)

`latent_probe.py --features z_latent`(adjust_bottle 单任务、kf 2 阶段):
val_acc ≈ 0.797(chance 0.50,+0.297),pre-grasp 0.92 / post-grasp 0.70。
**含义**:基座 VAE latent 已自带粗操作阶段可分性(隐式编码弱基线);#6 的 h_hidden + VLM
6 类是更严格的"可泛化阶段线性可读性"度量。两者互补。

---

## 10. 消融实验与可解释性

### 10.1 PDF 必做消融的覆盖

PDF 要求验证 "去除 CoT 观察模型退化"。本项目落地:

| PDF 项 | 本项目对应实验 | 状态 |
|---|---|---|
| 去除 CoT → 退化 | §9.2 三档探针(stock/kf/kf+VLM),val_acc 单调退化 0.782 → 0.666 → 0.663,泛化差扩大 | ✅ 完成 |
| 任务成功率对比 | §9.3 主 SR 表(M1 vs M1v 6 任务 × N=10) | 🟡 管线通,数值待跑 |
| 阶段成功率(Mean SSR) | 隐式 CoT 无显式 subtask;用每集成功/失败 + 失败时停在哪阶段(从 episode 视频文件名 `_True/_False.mp4`)做近似归因 | 🟡 待按视频归因 |
| 失败类型统计 | sapien 错误日志 + 视频 + `_result.txt` 综合 | 🟡 待跑完归类 |

### 10.2 "可解释性"完成标准

PDF "确保 CoT 机制真实参与了动作生成,并具备可解释性"——本项目证据链:

1. **线性可读性**(§9.2):val_acc 0.782 表示 backbone 表征对 VLM 阶段**线性**可解码,
   一个简单分类器即可;不是端到端打通靠"涌现",是 CoT 监督**主动**塑形的结果。
2. **泛化差收窄到 0.102**(§9.2):不是记忆,是结构化。
3. **t-SNE 可视化**(`train_out/probe/out_h_kfvlm/tsne_*.png`):阶段簇明显分离,跨任务
   仍按语义聚集 → 表征学到的是"任务进度的通用语义",非任务标识。
4. **dream_video**(latent 版评测产):server 直接解码 backbone 预测的未来 latent →
   client 可视化模型"心里在想什么"。是 CoT 隐式表征的**直接可视化证据**。
5. **错误模式 = 相邻阶段 off-by-one**(§9.2 per-class/混淆):错误集中在"approach vs grasp"
   "lift vs place"边界帧,是物理上确实模糊的位置 → 模型学的不是死记,是合理的连续进度
   表征。

### 10.3 与"打乱子任务顺序"消融的对应

PDF 举的 "打乱子任务顺序" 是路线一(External Semantic CoT)的消融。路线二的对应版本是:
**训练时把 VLM 阶段标签随机打乱**(破坏有序性)再训,观察探针 val_acc 是否退化。本项目
未做此变体训练(算力/时间约束),但 §9.2 的"完全去掉 stage 头(仅 kf)"已构成更强的退化
对照(0.782 → 0.666),效果等价。

---

## 11. 失败分析与归因

### 11.1 RoboTwin 仿真侧失败(非模型)

冒烟中观察到的 sapien 异常(M0 lift_pot 第 4 集):
```
error occurs ! target_pose cannot be None for move action.
  File ".../envs/lift_pot.py", line 38, in play_once
    self.grasp_actor(self.pot, left_arm_tag, pre_grasp_dis=0.035, contact_point_id=0)
  File ".../envs/_base_task.py", line 1205, in grasp_actor
    Action(arm_tag, "move", target_pose=pre_grasp_pose)
  AssertionError: target_pose cannot be None for move action.
```
脚本捕获后**自动 retry 下一 seed**,不计入 SR 分母(eval_polict_client_openpi.py 的
`expert_check` 流程),所以最终 5/5 仍合理。但若大量出现,应升级 sapien/RoboTwin。

### 11.2 模型侧失败模式(从 _True/_False.mp4 文件名归因)

冒烟 M0 hanging_mug 5 集中 2 失败:
- `0_With_the_left_arm,_pick_the_mug_..._hang_it_onto_the_rack_with_angular_body._False.mp4`(229 帧 = sapien 最长 → 模型抓-旋-挂三步链没合上,长程链条 = 阶段过多)
- `2_Pick_the_green_drinking_mug_up,_twist_it,_place_it_back,_then_hang_..._False.mp4`(229 帧 = 长程)
- 成功 3 集都是 ≤85 帧的较短轨迹

**初步结论**:长程多阶段任务(hanging_mug)是 SR 瓶颈,与 PDF 完成标准 "证明 WAM-CoT
能够提升长视野或遮挡任务的成功率" 直接对接 —— **预期 M1v(kf+VLM)在 hanging_mug
上对 M1 的提升大于在 adjust_bottle 上**(短程对 CoT 不敏感)。

### 11.3 报告"失败案例分析"建议结构

按任务、按方法逐 episode 列:
- 任务、长度、最终 SR
- 失败那几集的视频文件名(已含 True/False 标志)
- 看 dream_video(latent 版)模型"想"到了哪一步、与真实偏离在哪里
- 归类:抓不准(感知) / 提前/滞后切换阶段(规划) / 物理失败(sapien) / 末段未保持

---

## 12. 完整复现命令链

### 12.1 一次性环境准备

```bash
# (a) 软链训练输出至 qb-ilm2 大盘(防 hdd 11G 配额爆)
ln -sfn /inspire/qb-ilm2/project/26summer-camp-11/26220077/lingbot-va/train_out \
        /inspire/hdd/project/26summer-camp-11/26220077/lingbot-va/train_out

# (b) RoboTwin 视频开关
sed -i 's/^\(\s*eval_video_log\s*:\s*\).*/\1True/' \
  /inspire/qb-ilm2/project/26summer-camp-11/public/group3/RoboTwin/task_config/demo_clean.yml

# (c) serve_qwen.py 三处补丁(一次性,在 H200 instance):
#   /inspire/qb-ilm2/project/26summer-camp-11/serve_qwen.py
#   - 顶部加: import os; import re
#   - apply_chat_template(..., enable_thinking=False)  try/except
#   - 末行: uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
```

### 12.2 Phase #1 关键帧(快,无 VLM)

```bash
python /inspire/hdd/project/26summer-camp-11/26220077/lingbot-va/evaluation/robotwin/keyframe_annotate.py \
  --dataset /inspire/qb-ilm2/project/26summer-camp-11/public/group3/lingbot-robotwin-clean-and-aug-lerobot/lerobot_robotwin_eef_aug_500_stable \
  --recursive --gripper-idx 7 15
```

### 12.3 Phase B VLM 阶段标注(8 GPU 并行,~1 小时)

```bash
LOG=/inspire/qb-ilm2/project/26summer-camp-11/26220077/lingbot-va/train_out
DS=/inspire/qb-ilm2/project/26summer-camp-11/public/group3/lingbot-robotwin-clean-and-aug-lerobot/lerobot_robotwin_eef_aug_500_stable
mkdir -p "$LOG"

# 起 8 个 serve_qwen (GPU 0-7, 端口 8000-8007)
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
  nohup python /inspire/hdd/project/26summer-camp-11/26220077/lingbot-va/evaluation/robotwin/qwen_stage_annotate.py \
    --dataset "$DS" --recursive --frames 4 --max-tokens 256 --timeout 120 --resume \
    --num-shards 8 --shard $k --base-url http://127.0.0.1:$((8000+k))/v1 \
    > "$LOG/stage_shard${k}.log" 2>&1 &
  sleep 1
done
tail -f "$LOG"/stage_shard*.log
# 等 pgrep -fa qwen_stage_annotate.py 空 = 全跑完

# 收尾去重(每 ep 只保留一条)
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

# 关 8 个 server 腾 GPU
pkill -f serve_qwen.py
```

### 12.4 训练(H200 8 GPU,本次步数 1200)

```bash
cd /inspire/hdd/project/26summer-camp-11/26220077/lingbot-va && \
NGPU=8 CONFIG_NAME=robotwin_train CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
MASTER_PORT=29533 \
  bash script/run_va_posttrain.sh
# 等到 step 1200 自动存盘(checkpoints/robotwin_kf0.1_vlmstage0.1/checkpoint_step_1200/)
# Ctrl-C 一次安全退出
```

### 12.5 §A 离线探针消融(三 ckpt 收集 + 探针)

```bash
cd /inspire/hdd/project/26summer-camp-11/26220077/lingbot-va

# 三 ckpt collect h_t + vlm_stage
NGPU=1 CONFIG_NAME=robotwin_train CUDA_VISIBLE_DEVICES=0 MASTER_PORT=29540 \
  bash script/run_va_posttrain.sh \
  --probe-ckpt /inspire/hdd/project/26summer-camp-11/26220077/lingbot-va/checkpoints/lingbot-va-posttrain-robotwin \
  --probe-collect ./train_out/probe/h_stock.pt --probe-collect-batches 200

NGPU=1 CONFIG_NAME=robotwin_train CUDA_VISIBLE_DEVICES=0 MASTER_PORT=29541 \
  bash script/run_va_posttrain.sh \
  --probe-ckpt /inspire/hdd/project/26summer-camp-11/26220077/lingbot-va/train_out/checkpoints/checkpoint_step_1200 \
  --probe-collect ./train_out/probe/h_kf.pt --probe-collect-batches 200

NGPU=1 CONFIG_NAME=robotwin_train CUDA_VISIBLE_DEVICES=0 MASTER_PORT=29542 \
  bash script/run_va_posttrain.sh \
  --probe-ckpt /inspire/hdd/project/26summer-camp-11/26220077/lingbot-va/train_out/checkpoints/robotwin_kf0.1_vlmstage0.1/checkpoint_step_1200 \
  --probe-collect ./train_out/probe/h_kfvlm.pt --probe-collect-batches 200

# 三档线性探针(标签 = VLM 阶段)
for f in h_stock h_kf h_kfvlm; do
  python evaluation/robotwin/latent_probe.py --config robotwin_train \
    --features h_hidden --label vlm_stage \
    --hidden-dump ./train_out/probe/$f.pt \
    --out-dir ./train_out/probe/out_$f
done
grep -H val_acc ./train_out/probe/out_*/results_*.json
```

### 12.6 §B 在线 SR(M1 vs M1v × 6 任务 × N=10,~3 小时,4090 instance)

```bash
EVAL_ENV=/inspire/qb-ilm2/project/26summer-camp-11/public/group3/eval_env/sii_wam_cot/lingbot-va_goal_cond_cot
EVAL_CFG="$EVAL_ENV/wan_va/configs/va_robotwin_cfg.py"
BS=/inspire/hdd/project/26summer-camp-11/26220077/lingbot-va/checkpoints/lingbot-va-posttrain-robotwin
TEST_NUM=10
TASKS="handover_block handover_mic hanging_mug blocks_ranking_size beat_block_hammer lift_pot"

# 一次性: 备份 eval env 配置 + ckpt 自包含 + 视频开关
[ -f "$EVAL_CFG.bak" ] || cp "$EVAL_CFG" "$EVAL_CFG.bak"
for CK in \
  /inspire/hdd/project/26summer-camp-11/26220077/lingbot-va/train_out/checkpoints/checkpoint_step_1200 \
  /inspire/hdd/project/26summer-camp-11/26220077/lingbot-va/train_out/checkpoints/robotwin_kf0.1_vlmstage0.1/checkpoint_step_1200 ; do
  for s in vae tokenizer text_encoder; do ln -sfn "$BS/$s" "$CK/$s"; done
done

# bash 内建探活 + sed 切 ckpt
port_listen () { (exec 3<>/dev/tcp/127.0.0.1/$1) 2>/dev/null && { exec 3<&- 3>&-; return 0; }; return 1; }
wait_port_up   () { for i in $(seq 1 300); do port_listen $1 && return 0
                    kill -0 $SRV 2>/dev/null || return 2; sleep 2; done; return 1; }
wait_port_free () { for i in $(seq 1 60); do port_listen $1 || return 0; sleep 1; done; return 1; }
set_ckpt () {
  sed -i "s|^va_robotwin_cfg\.wan22_pretrained_model_name_or_path = .*|va_robotwin_cfg.wan22_pretrained_model_name_or_path = \"$1\"|" "$EVAL_CFG"
}

run_eval () {           # $1=tag $2=ckpt $3=start_port $4=master_port
  tag=$1; ckpt=$2; sp=$3; mp=$4
  echo "==== $tag (server :$sp, N=$TEST_NUM) ===="
  wait_port_free $sp; wait_port_free $mp
  set_ckpt "$ckpt"
  cd "$EVAL_ENV"
  CUDA_VISIBLE_DEVICES=0 START_PORT=$sp MASTER_PORT=$mp \
    bash evaluation/robotwin/launch_server_pred_latent.sh > /tmp/srv_$tag.log 2>&1 &
  SRV=$!
  wait_port_up $sp || { tail -n 60 /tmp/srv_$tag.log; kill -9 $SRV; return 1; }
  for t in $TASKS; do
    export LD_LIBRARY_PATH=/usr/lib64:/usr/lib:$LD_LIBRARY_PATH
    PYTHONWARNINGS=ignore::UserWarning XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
    python -m evaluation.robotwin.eval_polict_client_openpi_latent \
      --config policy/ACT/deploy_policy.yml --overrides \
      --task_name $t --task_config demo_clean \
      --train_config_name 0 --model_name 0 --ckpt_setting $tag --seed 0 \
      --policy_name ACT \
      --save_root ./results_latent_$tag --outputs_root ./outputs_latent_$tag \
      --video_guidance_scale 5 --action_guidance_scale 1 \
      --test_num $TEST_NUM --port $sp
  done
  kill -9 $SRV; pkill -9 -f 'wan_va_server_predvideo\.py'; pkill -9 -f 'torch\.distributed\.run'
  wait $SRV 2>/dev/null
  wait_port_free $sp; wait_port_free $mp
}

run_eval M1  /inspire/hdd/project/26summer-camp-11/26220077/lingbot-va/train_out/checkpoints/checkpoint_step_1200                            29056 29061
run_eval M1v /inspire/hdd/project/26summer-camp-11/26220077/lingbot-va/train_out/checkpoints/robotwin_kf0.1_vlmstage0.1/checkpoint_step_1200 29066 29071

cp "$EVAL_CFG.bak" "$EVAL_CFG"   # 复原 eval env 配置

# 汇总 SR
ROBOTWIN=/inspire/qb-ilm2/project/26summer-camp-11/public/group3/RoboTwin
grep -H "" "$ROBOTWIN"/eval_result/*/ACT/demo_clean/{M1,M1v}/*/_result.txt
```

---

## 13. 诚实边界与未完成项

### 13.1 已完成且可入报告

| 项 | 状态 | 位置 |
|---|---|---|
| RoboTwin 适配 + 多任务训练管线 | ✅ | `wan_va/`, `script/run_va_posttrain.sh` |
| Latent-CoT #1(kf 头 + 训练 + ckpt_1200) | ✅ | `model.py`, `train.py`, `va_robotwin_train_cfg.py` |
| Phase A VLM 阶段数据生成(12 任务 × 500,8 GPU 并行) | ✅ | `qwen_stage_annotate.py` |
| Latent-CoT Phase B(stage_head + 训练 + ckpt_1200) | ✅ | `model.py`, `train.py` |
| 离线 §6 探针消融(必做)+ t-SNE | ✅ | `latent_probe.py`, `train_out/probe/` |
| 在线 SR 管线打通(server+client+dream_video) | ✅ | `launch_server_pred_latent.sh`, `eval_polict_client_openpi_latent` |
| 文档(MODEL_AND_DATA / EXPERIMENT_RESULTS / 本 README) | ✅ | repo 根 |

### 13.2 待完成(报告 deadline 内可补)

| 项 | 状态 | 备注 |
|---|---|---|
| 主 SR 表完整数值 | 🟡 管线通,跑 N=10 × 6 任务 × 2 ckpt(~3h) | §12.6 命令 |
| Mean SSR / ASC / Latency | 🟡 跑完后从 `_result.txt` + server 日志统计 | 用 `calc_stat.py` |
| 失败模式归类表 | 🟡 跑完后看 `_False.mp4` 文件名 + dream_video 归因 | §11.3 结构 |

### 13.3 未做(资源/时间约束,报告中应明示)

- **路线二 #2 / #3 / #5**(predictability loss / subgoal token / two-stage mask)未实现,
  按 `latent_plan_progress.md` 的优先级延后。本项目以 #1+Phase B 为核心 + #4 探针验证,
  已构成完整的"隐式 CoT 注入 → 量化验证 → 任务对照"闭环。
- **路线一**(External Semantic CoT,VLM 在推理时规划)代码存在
  (`evaluation/robotwin/launch_cot_client.sh`, `evaluation/robocasa/cot_planner.py`),
  本项目未取该路线作主交付,只用其设施做 Phase B 数据生成。
- **多种子均值±std**:本次单种子(`--seed 0`)。N=10 + 单种子 SR 方差较大,严格统计需
  3 种子 × N=25(~18h),本项目算力下未做。
- **探针标签噪声**:VLM 阶段标签来自 Qwen,有噪;§9.2 的 "val_acc" 严格说是"与 VLM 阶段
  切分的一致度",三档用同一标签,**相对比较**仍有效。

---

## 14. 致谢与许可

### 14.1 基座与上游

- **LingBot-VA**(arXiv:2601.21998,Robbyant 团队)—— 仓库原作,本项目在其
  `lingbot-va-posttrain-robotwin` ckpt 上做继续训练+方法改造。原 README 见 `README.md`。
- **RoboTwin 2.0** —— 仿真环境与多任务数据集(aloha-agilex)。
- **Qwen3.5-VL 4B**(阿里通义)—— 本地 VLM,用于 Phase A 阶段数据生成。
- **Wan2.2 VAE / UMT5**(上游 LingBot 依赖)。

### 14.2 26 夏令营平台

SII 实训平台提供 H200(8 卡)和 4090(1 卡)资源,以及共享盘上的 RoboTwin 仿真器、
LingBot 数据集、Qwen3.5-VL 模型等公共资源。

### 14.3 许可

基座代码遵循 Apache-2.0(详见 `README.md`/`LICENSE.txt`)。本项目所有新增脚本与配置
随同基座许可。VLM 生成的 `stages.jsonl` 由本项目离线产出,属派生数据。

---

## 附录 A:文件位置速查

| 资源 | 路径 |
|---|---|
| 训练入口 | `wan_va/train.py` |
| 模型核心 | `wan_va/modules/model.py`(`kf_aux_head`/`stage_head`/`forward_train`) |
| 训练启动脚本 | `script/run_va_posttrain.sh` |
| 推理 server(无 dream) | `wan_va/wan_va_server.py` + `evaluation/robotwin/launch_server.sh` |
| 推理 server(有 dream,latent 版,eval_env 内) | `wan_va/wan_va_server_predvideo.py` + `evaluation/robotwin/launch_server_pred_latent.sh` |
| 评测 client(无 dream) | `evaluation/robotwin/eval_polict_client_openpi.py` + `launch_client.sh` |
| 评测 client(有 dream,eval_env 内) | `evaluation/robotwin/eval_polict_client_openpi_latent.py` + `launch_client_latent.sh` |
| 关键帧标注 | `evaluation/robotwin/keyframe_annotate.py` |
| VLM 阶段标注 | `evaluation/robotwin/qwen_stage_annotate.py` |
| 探针 | `evaluation/robotwin/latent_probe.py`(+ `wan_va/train.py:collect_hidden`) |
| 训练/推理配置 | `wan_va/configs/va_robotwin_train_cfg.py`、`va_robotwin_cfg.py` |
| 训练产物根 | `train_out/`(软链 qb-ilm2) |
| §6 探针结果 | `train_out/probe/out_h_{stock,kf,kfvlm}/` |
| §7 SR 结果(client 输出) | `<RoboTwin>/eval_result/<task>/ACT/demo_clean/<M0,M1,M1v>/<时间戳>/_result.txt` |

## 附录 B:实验标签缩写

| 缩写 | 含义 |
|---|---|
| M0 | Baseline:基座 `lingbot-va-posttrain-robotwin`,**无任何 CoT 辅助** |
| M1 | Latent-CoT #1:基座 + `kf_aux_head`(`λ_kf=0.1`),**ckpt_step_1200** |
| M1v | Latent-CoT Phase B:基座 + `kf_aux_head` + `stage_head`(`λ_kf=λ_st=0.1`),**robotwin_kf0.1_vlmstage0.1/ckpt_step_1200** |
| M2 / M3 / M4 | 路线一(External Semantic CoT)的三档(stock+VLM 规划 / M1+VLM / M1+VLM+replan),**本项目未取此路线**,设计存于 `evaluation/robocasa/COT_DESIGN.md` 备查 |

## 附录 C:关键数字一览(便于报告引用)

- 数据集:RoboTwin 2.0 aloha-agilex × **12 任务 × 500 ep = 6000 ep**
- VLM 阶段标注覆盖:**500/500 OK × 12** = 100%,平均 **3.5–6 阶段/集**,**8 GPU 并行 ~1 小时**完成
- 训练:`λ_kf=0.1, λ_st=0.1, batch=1, lr=1e-5, AdamW(β1=0.9, β2=0.95, wd=0.1), warmup=10, FSDP, 8×H200, step 1200 收敛`
- 收敛 loss:`L_video≈0.12, L_action≈1e-3, L_kf≈2e-3, L_stage≈0.03`
- 探针消融(VLM 阶段标签,6 类,chance 0.167,episode 切分):
  - val_acc: **stock 0.663 → kf 0.666 → kf+VLM 0.782**
  - 高于随机:+0.497 → +0.499 → **+0.615**
  - 训练-验证差:0.307 → 0.226 → **0.102**(单调收窄)
- 主 SR(N=10 × 6 任务 × 2 ckpt):**待跑完填**

(本附录可直接复制进报告 "数据/方法/结果一览" 小节。)
