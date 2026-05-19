# WAM-CoT（路线二·隐式物理 CoT）— 模型、改动与数据说明

> 面向 26 夏令营技术报告。记录：用了什么基座模型、在其上改了什么/为什么改、
> 用了什么数据集、数据来源。配套文档：只读规格 `latent_plan.md`、进度
> `latent_plan_progress.md`、CoT 路线一 `evaluation/robocasa/COT_DESIGN.md`。

---

## 1. 基座模型（未自研，复用开源 + 官方后训练权重）

**LingBot-VA**（仓库 `wan_va/`，论文 arXiv:2601.21998）——自回归"视频-动作"世界
模型（Video-Action World Model）：

| 组件 | 说明 | 代码位置 |
|---|---|---|
| 视觉 VAE | Wan2.2 VAE，把 RGB 视频编码为 **48 通道 latent**（流式 `WanVAEStreamingWrapper`） | `wan_va/modules/utils.py`, `wan_va_server.py:_encode_obs` |
| 主干 | `WanTransformer3DModel`：30 层 Transformer，patch_size=(1,2,2)，inner_dim=24×128，**双流 MoT**（latent 流 + action 流交错于同一序列），RoPE，cross-attn 注入文本条件 | `wan_va/modules/model.py` |
| 文本编码器 | UMT5（`UMT5EncoderModel`）+ T5 tokenizer，text_dim=4096，prompt 经 `condition_embedder.text_embedder` 投影 | `wan_va/wan_va_server.py:_get_t5_prompt_embeds` |
| 动作头 | 动作扩散头，30 维动作空间（RoboTwin 实际用 14 通道：左/右臂 EEF 7+7、夹爪经 `used_action_channel_ids` 重排），FlowMatch 调度 | `model.py:action_proj_out`, `va_robotwin_cfg.py` |
| 训练 | FSDP + 激活检查点，视频/动作联合扩散损失 | `wan_va/train.py` |

**使用的权重（不是自己从头训）**：官方后训练 checkpoint
`lingbot-va-posttrain-robotwin`，服务器路径
`/inspire/hdd/project/26summer-camp-11/26220077/lingbot-va/checkpoints/lingbot-va-posttrain-robotwin`
（`va_robotwin_cfg.py` 指向它）。我们在它之上做**继续后训练（post-train）+
方法改造**，而非重训基座。

> 注：路线一（外部语义 CoT，DeepSeek/本地 Qwen3.5-27B 作 VLM 规划器）是另一条
> 线，代码在 `evaluation/robotwin/` 与 `evaluation/robocasa/`，本文件聚焦
> **当前在训的路线二（隐式/Latent CoT）**。

---

## 2. 在基座上的改动与原因

分三类：**(A) 让它在本环境跑起来的工程修复**、**(B) 路线二方法改造（核心
贡献）**、**(C) 探针评估**。所有方法改造均 *opt-in*，关闭时与原模型逐字节一致。

### A. 工程适配 / 运行期修复（不改方法，仅为可运行）

| 改动 | 文件 | 为什么 |
|---|---|---|
| RoboTwin 安装路径、ckpt 路径硬编码 | `eval_polict_client_openpi.py`, `va_robotwin_cfg.py` | 原仓库是 `/path/to/...` 占位符 |
| cuDNN `LD_PRELOAD` 修复 | `launch_server.sh`, `script/run_va_posttrain.sh` | 系统 `/usr/lib` 的 cuDNN 覆盖 torch2.9(cu126) 自带 cuDNN → `libcudnn_graph.so.9: undefined symbol` → 首次 GPU 前向 SIGABRT。强制优先加载 torch 自带匹配版本 |
| wandb 惰性导入 + **离线模式** + 失败降级 | `wan_va/train.py` | README 用 `--no-deps` 装 wandb 缺 `click`；占位 `WANDB_BASE_URL="your url"` 触发 pydantic 崩；遥测绝不能拖垮训练 → 离线本地记录、异常即降级继续 |
| `low_cpu_mem_usage` meta 张量修复 | `wan_va/modules/utils.py:load_transformer` | 新增的 `kf_aux_head`（任何旧 ckpt 都没有）在 diffusers 低内存加载下停留在 **meta 设备**，`model.to(device)` 崩。改为：保留 meta 加载，加载后**仅把仍在 meta 的子模块实体化+随机初始化**，已加载权重不动 |
| 多进程 Pool fork 死锁修复 | `wan_va/dataset/lerobot_latent_dataset.py` | `MultiLatentLeRobotDataset` 默认 `num_init_worker=128`，无论几个 repo 都 `Pool(128)`；在 CUDA/NCCL 初始化后 fork → 非 fork-safe 死锁。改为 worker≤repo 数，**单 repo 直接串行不 fork** |
| `empty_emb.pt` 路径 | `va_robotwin_train_cfg.py` | 它是 CFG 空 prompt 的文本嵌入（非模型权重），官方放在**数据集树根目录共享**，不在每个任务目录；原 join(dataset_path) 找不到 |
| `kf_loss` 可视化 + 逐 rank 日志 | `wan_va/train.py`, `run_va_posttrain.sh` | 原日志只显示 latent/action，看不到 #1 的 kf_loss 是否在训；`--redirects/--log-dir` 让非 0 rank 的真实 traceback 落盘 |

### B. 路线二方法改造：隐式物理 CoT（核心，对应 `latent_plan.md` #1）

**目标**：把"思维链"灌进世界模型权重——强迫主干 latent 编码"距下一个物理
关键事件还有多久"，从而隐式表征任务阶段，提升长程成功率，推理零外部依赖。

| 改动 | 文件 | 内容与原因 |
|---|---|---|
| 关键帧离线标注 | `evaluation/robotwin/keyframe_annotate.py`（新增） | 从动作向量的**夹爪开/合跳变**（左 idx7、右 idx15，阈值取通道值域中点，跨 embodiment 鲁棒、只认跳变）提关键帧 `grasp/release`，加末帧 `end`；**零 LLM、纯离线**，比计划里假设但不存在的 deepseek 阶段脚本更鲁棒。输出 `meta/keyframes.jsonl` |
| 数据集钩子（opt-in） | `wan_va/dataset/lerobot_latent_dataset.py` | `cfg.kf_aux` 开启时，`__getitem__` 按**与模型一致的 latent 时间步**（`frame_ids` 每 ~4× 取代表帧）算每 latent 帧 `kf_dist`（到下一关键帧距离）、`kf_mask`、`kf_stage`（阶段idx，给 #4 探针）、`kf_episode`（轨迹切分用）。关闭则不加任何键 |
| 关键帧辅助头 | `wan_va/modules/model.py` | `forward_train` 在所有 block + norm 之后、`proj_out` **之前**取主干 latent 隐状态（token 序 `(f h w)`，对空间 mean-pool → 每 latent 帧 d 维）→ `kf_aux_head: Linear(d,128)-GELU-Linear(128,1)` 预测 `log1p(距离)`。head 恒构建（极小），推理不调用 |
| 辅助损失 | `wan_va/train.py:compute_loss/_train_step` | `L = L_video + L_action + λ_kf · SmoothL1(log1p(kf_dist), kf_pred)`（按 `kf_mask` 求均值，× `kf_aux_weight`）。仅 `kf_aux & weight>0 & 有标注` 时生效；否则常数 0、不回传梯度 |
| 训练配置 | `va_robotwin_train_cfg.py` | `kf_aux=True, kf_aux_weight=0.1, kf_file='keyframes.jsonl'`（**本实验默认开启**；置 False 即纯基线）。dataset 指向 RoboTwin adjust_bottle 任务；wandb 离线 |

> **设计依据**：用夹爪跳变这种"与物理操作强对齐的二值事件"作监督信号，逼
> backbone 在 latent 里编码"时间维进度"（隐式 CoT）。`λ_kf=0.1` 起步、训稳可
> 降 0.05（计划建议）。基座权重不被破坏（aux 头从零学，主损失不变）。

### C. 评估：探针 + t-SNE（对应 `latent_plan.md` #4）

`evaluation/robotwin/latent_probe.py`（新增）：冻结表征 + 线性探针，按
**episode 轨迹切分**（防泄漏），量化 latent 是否线性可分任务阶段；输出
acc/混淆矩阵 + t-SNE/PCA 图。`z_latent` 模式离线零依赖（已得 baseline：
VAE latent 对 adjust_bottle 抓取前/后 val_acc≈0.80，chance 0.50）；`h_hidden`
模式（取 `forward_train` 暴露的 `kf_feat`）待 #1 训出 ckpt 后做
**stock vs +#1** 对比，是计划 #4 的核心结论。

---

## 2.5 损失函数定义与物理意义（精确版）

训练总损失（`wan_va/train.py:compute_loss/_train_step`，gas =
`gradient_accumulation_steps`）：

```
L_total = ( L_video + L_action + λ_kf · L_kf ) / gas
```

记号：模型按 **flow-matching** 训练——对干净 latent `z0` 在随机扩散时间步
`t` 加噪得 `z_t`，模型预测目标 `target`（flow 速度场），
`w(t)=scheduler.training_weight(t)` 是 flow-matching 的 SNR 时间步权重。

### (1) `L_video`（视频/世界模型损失 ≈ 0.10–0.15）

```
latent_pred = unpatchify(transformer_video_stream)            # (B,48,F,H,W)
e = w_v(t) · ( latent_pred − target_video )^2                  # 逐元素 MSE×权重
L_video = mean_over(B·F) [ sum_{C,H,W} e  /  (#elems) ]
```
- **是什么**：Wan2.2 VAE **48 通道视频 latent** 的流匹配去噪误差（逐 latent
  帧归一化后对 B×F 取均值）。
- **物理意义**：世界模型"**想象未来视觉动力学**"的准确度——给定历史观测，
  模型在潜空间预测接下来若干帧场景会如何演变。这是 WAM 的"世界"部分。
- 不降到 0 属正常：自然场景视频 latent 的随机性下限较高，~0.1 量级是收敛态。

### (2) `L_action`（动作损失 ≈ 1e-3，很快降很低）

```
action_pred 重排为 (B,30,F,n,1)
e = w_a(t) · (action_pred − target_action)^2 · actions_mask
L_action = mean_over(B·F) [ sum e_masked / (#masked elems) ]
```
- **是什么**：动作扩散头输出的流匹配误差。动作空间标称 30 维，RoboTwin 经
  `used_action_channel_ids` 实际只用 **14 维**（左臂 EEF 6 + 左夹爪 1 + 右臂
  EEF 6 + 右夹爪 1，已做分位 q01/q99 归一化、相对位姿 `get_relative_pose`）；
  `actions_mask` 把未用/补零通道屏蔽，只对真实通道计损失。
- **物理意义**：模型"**该输出什么底层电机指令**"（双臂末端位姿增量 + 夹爪）
  的准确度。这是 WAM 的"动作"部分，也是最终决定 SR 的输出。
- 收敛快且很低（~1e-3）：动作在专家数据上高度规律、可预测。

### (3) `L_kf`（关键帧距离辅助损失 = 隐式物理 CoT，本项目新增）

```
h_t  = mean_{空间}( backbone_latent_hidden )      # 每 latent 帧 d 维(proj_out 之前)
kf_pred = kf_aux_head(h_t)                         # (B, F_lat)
tgt = log1p( kf_dist )                             # kf_dist=到下一关键帧的原始帧数
L_kf = λ_kf · mean_{kf_mask} SmoothL1( kf_pred, tgt )
```
- **是什么**：每个 latent 时间步，回归"**距离下一个物理关键事件（夹爪
  grasp/release / 轨迹末端 end）还有多少帧**"，对数压缩 + Huber，按
  `kf_mask`（有定义的帧）求均值，× `λ_kf = kf_aux_weight = 0.1`。
- **物理意义（核心）**：强迫主干 latent 在每一时刻**显式编码"任务进度 /
  距下一阶段切换的时间"** —— 这就是把"思维链"灌进权重的**隐式物理 CoT**：
  模型不靠外部 VLM，而是其世界模型表征本身携带阶段/时序结构，从而利于
  长程任务的稳定与成功率。
- **设计细节与依据**：① `log1p`：帧距动态范围大（0–~200），对数压缩使回归
  稳定；② `SmoothL1`(Huber)：对关键帧边界附近的离群目标鲁棒；③ `kf_mask`：
  末段无"下一个关键帧"时不惩罚；④ `λ_kf=0.1` 小权重：只**塑形** backbone，
  不喧宾夺主压过 video/action 主目标（故显示值 ~1e-3 = 原始 Huber ~1e-2 ×
  0.1）；⑤ 收敛快（几百步内 ~1e-3）属**预期**：对强 backbone 而言"到下一
  关键帧的帧数"是近单调、易学的信号，**不是 bug**。关闭 `kf_aux` 时
  `L_kf≡0` 且 aux 头不接收梯度，训练与原模型逐字节一致。

> 三者关系：`L_video`/`L_action` 是 LingBot 原本的世界-动作联合目标（保持
> 基座能力）；`L_kf` 是叠加的弱监督，专门让 latent 习得物理阶段结构。报告做
> 消融时：`kf_aux=False`（纯基线）vs `kf_aux=True`（+隐式 CoT），再用
> `latent_probe.py` 比较两者 latent 的阶段线性可分性（#4 核心结论）。

## 2.6 训练产出与中途停止

- **检查点**：`save_checkpoint()` 每 `save_interval` 步写一次到
  `<save_root>/.../checkpoint_step_<N>/transformer/`（diffusers 格式
  safetensors + config.json，可直接 `from_pretrained` / 我们的
  `load_transformer` 复用，也供 `latent_probe.py --features h_hidden`）。
- **中途 Ctrl-C 保存**（已实现）：训练循环注册 SIGINT/SIGTERM 处理器，**只
  置标志位**，在下一个优化步边界**集体**（all-reduce 决定，避免 FSDP 保存
  collective 死锁）调用 `save_checkpoint()` 后干净退出。因此收敛快时可随时
  Ctrl-C，会在 `checkpoint_step_<当前步>` 留下可用权重（**再按一次 Ctrl-C
  会硬杀，可能损坏保存**——只按一次、等它打印 "Interrupt: saving
  checkpoint ... then exiting." 即可）。建议把 `save_interval` 调小（如 200）
  作双保险。
- **磁盘配额（重要）**：训练所在 `/inspire/hdd/.../26220077` 仅 ~11G 且基座
  ckpt 已占满，checkpoint/wandb/日志/HF arrow 缓存写满即
  `OSError: Disk quota exceeded` 崩溃（曾在 ~step 82 触发）。解决（**不改
  Python 代码**）：① 把 `train_out/` 软链到 qb-ilm2 大盘
  （`ln -s /inspire/qb-ilm2/.../lingbot-va/train_out
  /inspire/hdd/.../lingbot-va/train_out`）——一举把 checkpoint+wandb+
  torchrun 日志都导向大盘；② `run_va_posttrain.sh` 已设
  `HF_HOME/HF_DATASETS_CACHE/HF_LEROBOT_HOME → CACHE_ROOT`
  （qb-ilm2），避免数据集 arrow 缓存（多任务更大）再写满小盘。

## 3. 使用的数据集

**任务**：RoboTwin 2.0 `adjust_bottle`（aloha-agilex 双臂，单瓶抓取并保持
直立），500 条随机化轨迹。

**服务器路径**
```
/inspire/qb-ilm2/project/26summer-camp-11/public/group3/
  lingbot-robotwin-clean-and-aug-lerobot/lerobot_robotwin_eef_aug_500/
  adjust_bottle-aloha-agilex_randomized_500-1000/
```
**格式**：LeRobot v2.1（`meta/info.json` 自报），`fps=50`，`total_episodes=500`，
`total_frames≈71223`。目录：

| 子目录/文件 | 内容 |
|---|---|
| `data/chunk-000/episode_******.parquet` | 每帧 `observation.state(16)`、`action(16)`：`[L/R]_x,y,z,q1..q4,gripper` |
| `latents/chunk-000/<cam>/episode_*_{s}_{e}.pth` | 每段经 **Wan2.2 VAE** 预编码的视频 latent + **UMT5 text_emb** + `frame_ids` 等（3 路相机：cam_high / cam_left_wrist / cam_right_wrist） |
| `videos/chunk-000/<cam>/episode_*.mp4` | 原始 RGB（av1，480×640） |
| `meta/episodes.jsonl` / `episodes_ori.jsonl` | 每集 `tasks/length/action_config`（含 `action_text`；ori 为细粒度多段） |
| `meta/tasks.jsonl` | 50 条自然语言指令变体 |
| `<数据集树根>/empty_emb.pt` | CFG 空 prompt 的 UMT5 嵌入（**全任务共享**，~4.2MB） |

**我们派生的数据**：`meta/keyframes.jsonl` —— 由本项目
`keyframe_annotate.py` 从该数据集**自身的动作向量夹爪通道**提取（500 集，
平均 2 个关键帧 = 1 grasp + 1 end，符合原子任务语义）。**无外部来源、纯派生**。

---

## 4. 数据来源 / 血缘

```
RoboTwin 2.0 仿真器（aloha-agilex 双臂）专家演示
   └─(LingBot 官方 clean & augment)→ robbyant/robotwin-clean-and-aug-lerobot
        （HF/ModelScope；论文配套后训练数据集，LeRobot 格式）
   └─(官方预处理)→ 每段抽帧→Wan2.2 VAE 编 latent + UMT5 编 text_emb，
        连同 empty_emb.pt 一起放到 SII 服务器 public/group3 共享盘
   └─(本项目离线派生)→ keyframe_annotate.py 按夹爪跳变生成 keyframes.jsonl
        → dataset loader 在训练时算 kf_dist/kf_stage（与 latent 时间步对齐）
```
- 仿真/演示来源：**RoboTwin 2.0**（开源双臂操作基准；LingBot README 指定
  commit 2eeec322）。
- 训练数据集来源：**LingBot 官方** `robotwin-clean-and-aug-lerobot`
  （论文配套，已 clean+aug 并转 LeRobot + 预抽 VAE/文本嵌入），由助教/官方
  下载至服务器共享盘，未自行下载（避免拖垮学院网络）。
- 关键帧标注来源：**本项目自产**，仅依赖数据集已有的 action 向量，不引入
  任何外部标注或额外模型。

---

## 5. 当前状态（截至 2026-05-19）

- 基座加载 / FSDP / 数据 / wandb 离线 / 多进程 —— 全部打通，单卡训练可跑
  （~3s/it on H200），多卡命令 `NGPU=5 CUDA_VISIBLE_DEVICES=1,2,3,4,5
  CONFIG_NAME=robotwin_train bash script/run_va_posttrain.sh`。
- 已修：之前 `kf_loss=0.0000`（kf_aux 被同步覆盖回 False）→ 仓库默认改为
  `kf_aux=True, kf_aux_weight=0.1`，重同步配置后 `kf_loss` 应非 0 并下降。
- 待办：确认 `kf_loss` 收敛 → 训出带 `kf_aux_head` 的 ckpt → 跑
  `latent_probe.py --features h_hidden` 做 stock-vs-#1 对比（#4 核心结论）；
  路线二其余组件 #2/#3/#5 仍按计划延后（见 `latent_plan_progress.md`）。
