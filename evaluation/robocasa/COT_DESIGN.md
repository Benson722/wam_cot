# WAM-CoT 路线一（External Semantic CoT）设计与现状说明

> 面向 26 夏令营报告。术语保持英文，行文中文。对应 PDF 第二阶段"路线一：
> 外部语义思维链"。

---

## 0. TL;DR

- **DeepSeek V4 Pro 是纯文本 LLM**，在本方案里**不看图像**。它读取由
  Robocasa/robosuite 仿真器导出的**结构化文本场景状态**（任务指令 + 物体 /
  夹爪 / 容器 / 干扰物的符号化坐标与关系），输出"物理约束推理 + 有序原子子
  任务列表"。这仍然是 PDF 要求的路线一（外部语义规划器 → 子任务 → 底层 WAM
  执行），只是"语义来源"从"VLM 看图"换成"仿真符号状态 + LLM 推理"。
- 子任务文本经新增的 `switch_prompt` 服务端消息**软切换**进底层 WAM
  （LingBot-VA，零样本 LIBERO-Long 权重），**保留 KV / 时序上下文**。
- "子任务完成 / 重规划"信号也走文本：每 N 步把更新后的符号状态（含仿真
  ground-truth 的 `_check_success()` 与几何谓词）喂给 DeepSeek 判定。
- 代码中的图像路径（`multimodal=True` 开关 + `image` 形参）**保留为未来接入
  真正 VLM 的入口**，默认关闭。

---

## 1. 为什么纯文本 LLM 也能做"语义 CoT"（回答问题 1）

PDF 路线一的本质是：**高层规划器**产出"包含物理约束分析的文本推理（如：先
移开重物，再抓取杯子）"，再解析为子任务交由底层 WAM 执行。规划器需要的是
**语义信息**，不必然是像素。

Robocasa 基于 robosuite，probe 已确认 `env` 暴露的低维状态包括：

| 来源 | 字段 | 用途 |
|---|---|---|
| `get_ep_meta()` | `lang`, `object_cfgs`, `fixtures`, `layout_id` | 任务语言、目标/容器/家具名称 |
| obs (low-dim) | `obj_pos`, `obj_quat`, `obj_to_robot0_eef_pos` | 目标物体位姿与相对夹爪偏移 |
| obs (low-dim) | `distr_counter_*`, `distr_cab_*` | 干扰物 / 容器相对位姿（遮挡线索） |
| obs (low-dim) | `robot0_eef_pos`, `robot0_gripper_qpos` | 末端位置、夹爪开合 |
| env | `_check_success()` | 仿真 ground-truth 成功谓词 |

把这些序列化成一段紧凑文本（示意）：

```
TASK: Pick the cake from the counter and place it in the cabinet.
TARGET: cake @ counter; offset to gripper (dx,dy,dz)=(0.31,-0.05,0.12) m
RECEPTACLE: cabinet (closed) ; gripper: OPEN
DISTRACTORS: bottle, can on counter (near target)
SUBGOAL CHECK: object_in_cabinet=False
```

DeepSeek 据此即可做物理约束推理与分解，例如：
`open the cabinet door → pick up the cake from the counter → place the cake
inside the cabinet → close the cabinet door`，并给出每个子任务的 `max_steps`
安全预算。这正是合法的 Route-1：**外部语义规划器 + 文本接口**。

> ⚠️ 关于遮挡：PDF 强调"遮挡场景"。纯文本规划器看不到像素里的视觉遮挡，
> 因此遮挡必须**显式写进 scene-text**（用几何关系推断，如"target 在闭合
> cabinet 内 / 被 distractor 包围 / 不可直接抓取"）。这是文本路线的关键工程点，
> 也是与"真 VLM"最大的差别（见 §4）。

监控（子任务完成 / 重规划）同理：纯文本模型不能看帧，所以每 N 步喂给它的是
**更新后的符号状态 + 仿真 `_check_success()` + 几何谓词**，由它判定
`subtask_done / task_success / need_replan`；并配合 per-subtask 的固定步数预算
做兜底（即用户原选的"VLM-monitored periodic check"在文本模型下落地为
"LLM-monitored，状态文本驱动"）。

---

## 2. 端到端架构（当前方案）

```
Robocasa env ──get_scene_text()──► DeepSeek.plan(task, scene_text)
                                        │  reasoning + [subtask_i, max_steps_i]
                                        ▼
                            ┌─ for each subtask_i ─────────────────────────┐
                            │ WamSession.switch_prompt(subtask_i)          │ ← 软切换，保留 KV
                            │ 跑若干 action chunk（LingBot 服务端推理）     │
                            │ 每 N 步: scene_text' + _check_success()       │
                            │   → DeepSeek.monitor() → done/advance/replan │
                            └──────────────────────────────────────────────┘
                                        ▼
              env._check_success() ─► 指标(SR/阶段SR/步数/失败类型) + 视频 + CoT trace(.plan.json)
```

组件 ↔ 文件：

| 角色 | 文件 | 关键点 |
|---|---|---|
| 底层 WAM 服务端 | [wan_va/wan_va_server.py](../../wan_va/wan_va_server.py) | 新增 `switch_prompt` 软切换；`_encode_obs` 现做 bytes→str key 归一化 + 缺键诊断 |
| 推理配置（零样本） | [wan_va/configs/va_robocasa_cfg.py](../../wan_va/configs/va_robocasa_cfg.py) | 模型侧接口与 LIBERO **完全一致**（相机键 / 128² / 7 维 OSC / 分位归一化） |
| 环境适配 | [robocasa_env.py](robocasa_env.py) | 相机白名单；12 维 HYBRID_MOBILE_BASE 动作映射；任务语言；success；**待加 `get_scene_text()`** |
| 高层规划器 | [cot_planner.py](cot_planner.py) | DeepSeek OpenAI 兼容 API；`plan`/`monitor`；硬编码 key；**待切到纯文本输入** |
| 编排 / 基线 / CoT | [eval_common.py](eval_common.py) · [client.py](client.py) · [client_cot.py](client_cot.py) | 动作 chunk 消费与 LIBERO 协议一致 |
| 消融 / 统计 | [run_ablations.sh](run_ablations.sh) · [calc_stat.py](calc_stat.py) | 必做消融 + SR/开销统计 |
| 服务端启动 | [launch_server.sh](launch_server.sh) | 修复 cuDNN/LD_LIBRARY_PATH 致 SIGABRT |

设计铁律：**零样本复用 LIBERO-Long 权重，模型侧字节级不变**；所有 Robocasa
适配都在 `evaluation/robocasa/` 客户端侧完成。

---

## 3. 当前进度与已知问题

**已验证（probe，2026-05-19）**
- robosuite 1.5.2 / robocasa 1.0.1；env 可建、可 reset、可渲染。
- 相机：`robot0_agentview_left` + `robot0_eye_in_hand`（256² uint8，flip 正确）。
- 动作：`action_dim=12`，HYBRID_MOBILE_BASE，arm `[0:6]` + gripper `[6]`，其余
  （base/torso/mode 索引 11）置 0 = 机械臂操作模式、底盘冻结。
- 任务语言 `get_ep_meta()['lang']`、`_check_success()` 均可用。

**运行期已修复的两处阻塞**
- (A) `_encode_obs` `KeyError: observation.images.agentview_rgb`：跨 conda 环境
  msgpack key 类型隐患 → 服务端已做 bytes→str 归一化 + 缺键时打印"实际 vs 期望
  键"的明确诊断（本地 msgpack 往返实测：现代 msgpack 键为 str，协议本身正确）。
- (B) `libcudnn_graph.so.9: undefined symbol: cudnnGetLibConfig` → SIGABRT：系统
  cuDNN 覆盖了 torch 2.9(cu126) 自带 cuDNN → `launch_server.sh` 现把 torch 自带
  `nvidia/*/lib` 前置到 `LD_LIBRARY_PATH`。

**已实现（纯文本 DeepSeek 方案落地，2026-05-19）**
1. ✅ `robocasa_env.get_scene_text()`：从低维 obs + `get_ep_meta()` +
   `_check_success()` 导出紧凑符号场景文本（含遮挡用的相对偏移/距离）。
2. ✅ `cot_planner.py`：`multimodal` 默认 `False`；`plan/monitor` 以
   `scene_text` 为主输入；系统提示词改为"依据结构化场景状态推理"；新增
   `backend` 预设（`deepseek` 文本 / `vllm` Qwen VLM）与 `for_backend()`。
3. ✅ `client_cot.py` / launch 脚本：规划/监控传 `env.get_scene_text()`；
   监控把 `SUCCESS_PREDICATE` 一并喂给 LLM；`.plan.json` 存 scene-text 快照；
   新增 `--planner` 开关；`blind_planner` 消融 = 同时去掉 scene-text 与图像。

**待办**
- 端到端联调（依赖前述 msgpack/cuDNN 修复后的 server 重跑）。
- 零样本动作尺度调参（`arm_action_scale` / `gripper_sign`）。

---

## 4. 纯文本 DeepSeek V4 Pro vs 真·VLM —— 区别与未来路线（回答问题 3）

| 维度 | DeepSeek V4 Pro（纯文本） | 真·VLM（多模态，未来） |
|---|---|---|
| 规划器输入 | 仿真符号状态序列化文本 | 相机帧（agentview）+ 任务文本 |
| 遮挡 / 杂乱感知 | **间接**：靠几何关系写进 scene-text | **直接**：从像素感知，最贴 PDF"遮挡场景"意图 |
| Grounding 来源 | 仿真 ground-truth（精确、确定） | 视觉理解（受仿真图像域偏移影响） |
| 对仿真状态的依赖 | 强（需 env 暴露好状态；真机不一定有） | 弱（看图即可，迁移性更好） |
| 监控信号 | 符号状态 + `_check_success()` | 帧 + （可选）符号状态 |
| API 成本 / 延迟 | 低（纯文本 token） | 高（图像 token、延迟大） |
| 工程量 | 需写 `get_scene_text()` | 复用已有图像路径，几乎零改 |
| 与 PDF 契合 | 路线一合法实现；遮挡需显式编码 | 更"原汁原味"的视觉语义 CoT |

**核心区别一句话**：DeepSeek 路线把"看场景"的活交给仿真器的真值状态 +
几何谓词，LLM 只负责推理；真·VLM 路线让模型自己从像素里"看"，更通用、更贴合
遮挡主题，但更贵且引入视觉域偏移。

**未来接入 VLM 的路线（代码已就绪约 90%）**
- `cot_planner.py` 已有：`_img_to_data_url()`、`multimodal` 开关、`plan/monitor`
  的 `image` 形参（OpenAI 兼容 `image_url` 消息）。
- 客户端已能拿到 agentview 帧（即送给 WAM 的 lingbot obs）。
- 接 VLM 只需三步：① `multimodal=True`；② `--vlm-model` / `--vlm-base-url`
  指向支持图像的多模态 endpoint（GPT-4o 类 / Qwen-VL / DeepSeek 多模态版若可用）；
  ③ 把 agentview 帧传入 `plan/monitor`（编排已透传 frame 到 monitor hook）。
- **可直接做"感知模态消融"**（报告加分项，验证 CoT 机制贡献 + 感知来源影响）：
  - `text-only`（DeepSeek，符号状态）
  - `vlm`（多模态，看图）
  - `text+image` 混合（最强规划器）
  - 与 `no_cot`（=Baseline）、`shuffle_subtasks`、`no_monitor` 等必做消融并列。

---

## 5. 为落地纯文本方案需要的具体代码改动

1. **`robocasa_env.py` 新增 `get_scene_text() -> str`**
   从 `self._last_obs` 的低维键（`obj_pos`、`obj_to_robot0_eef_pos`、`distr_*`、
   `robot0_eef_pos`、`robot0_gripper_qpos`）+ `get_ep_meta()`（`lang`、
   `object_cfgs` 名称、`fixtures`）+ `_check_success()` 组装紧凑文本；遮挡用
   几何关系显式标注（如 target 在闭合容器内 / 被干扰物包围）。
2. **`cot_planner.py`**
   `PlannerConfig.multimodal` 默认改 `False`；`plan(task, scene_text)`、
   `monitor(task, subtask, remaining, scene_text)` 以文本为主输入；保留
   `image` 形参与 `multimodal=True` 分支供未来 VLM。系统提示词去掉"看图"措辞，
   改为"依据给定结构化场景状态推理"。
3. **`client_cot.py` / `eval_common.py`**
   规划 / 监控调用处改为传 `env.get_scene_text()`；监控把
   `env.check_success()` 与每个子任务的几何完成谓词一并喂给 DeepSeek 兜底；
   `.plan.json` trace 同时存 scene-text 快照，便于报告的可解释性分析。

> 这三处改完即可端到端跑通"纯文本 DeepSeek 版 WAM-CoT"。改动局限在
> `evaluation/robocasa/` 客户端侧，**不触碰零样本模型接口**。

---

## 6. 如何在 纯文本 DeepSeek ↔ 本地 Qwen3.5-27B VLM 之间切换（回答问题 3）

切换是**一个开关**。两个后端走同一套 OpenAI 兼容 `_chat`，区别只在
`base_url / model / api_key / multimodal`，由 `PlannerConfig.for_backend()`
预设；端点与 key **硬编码**在 `cot_planner.py`（`HARDCODED_DEEPSEEK_*` /
`HARDCODED_VLLM_*`），无需 `export`。

### 6.1 默认：纯文本 DeepSeek V4 Pro（当前阶段）

```bash
# 服务端已起；robocasa conda env：
bash evaluation/robocasa/launch_client_cot.sh          # PLANNER 默认 deepseek
# 或显式：
PLANNER=deepseek ABLATION=none bash evaluation/robocasa/launch_client_cot.sh
```
- `multimodal=False` → **从不发图**，只发 `scene_text`（符号状态）。
- 子任务监控 = DeepSeek 读更新后的 `scene_text` + `SUCCESS_PREDICATE`
  + per-subtask 步数预算兜底。

### 6.2 之后：本地 Qwen3.5-27B 作为 VLM（用 vLLM 驱动）

**第 1 步——起 vLLM（OpenAI 兼容、多模态）服务**，在能跑该模型的 GPU 上：

```bash
# 模型已在: /inspire/qb-ilm2/project/26summer-camp-11/public/group3/models/Qwen3.5-27B
vllm serve /inspire/qb-ilm2/project/26summer-camp-11/public/group3/models/Qwen3.5-27B \
    --served-model-name Qwen3.5-27B \
    --port 8000 --trust-remote-code \
    --limit-mm-per-prompt image=1
# 默认对外: http://127.0.0.1:8000/v1  (与 HARDCODED_VLLM_BASE_URL 一致)
```
> 若端口/机器不同：`export VLLM_BASE_URL=http://<host>:<port>/v1`
> （或传 `--vlm-base-url`）。`--served-model-name` 要与
> `HARDCODED_VLLM_MODEL`（`Qwen3.5-27B`）或 `VLLM_MODEL` 一致。

**第 2 步——客户端切到 vllm 后端**（其余命令不变）：

```bash
PLANNER=vllm ABLATION=none bash evaluation/robocasa/launch_client_cot.sh
# 整个消融矩阵都用 VLM 重跑：
PLANNER=vllm bash evaluation/robocasa/run_ablations.sh
```
- `vllm` 预设 `multimodal=True` → 规划/监控**同时**发 agentview 图像 **和**
  `scene_text`（图像为主、符号状态作额外 grounding）。
- 代码无需任何改动；客户端已透传 agentview 帧到 `plan/monitor`。

### 6.3 开关速查

| 需求 | 命令 / 变量 |
|---|---|
| 纯文本 DeepSeek（默认） | `PLANNER=deepseek` |
| 本地 Qwen VLM（vLLM） | 先 `vllm serve …`，再 `PLANNER=vllm` |
| 改 vLLM 端点 | `export VLLM_BASE_URL=http://host:port/v1` 或 `--vlm-base-url` |
| 改模型 id | `export VLLM_MODEL=…` / `DEEPSEEK_MODEL=…` 或 `--vlm-model` |
| 强制某后端走纯文本（消融） | `VLM_TEXT_ONLY=1`（或 `--vlm-text-only`） |
| Python 内构造 | `PlannerConfig.for_backend("vllm", multimodal=True)` |

### 6.4 感知模态消融（报告加分项）

同一套任务跑三种规划器并对比 SR / 阶段 SR / 失败类型 / 推理开销：

| 配置 | 命令 |
|---|---|
| 文本 DeepSeek | `PLANNER=deepseek` |
| Qwen VLM（图像） | `PLANNER=vllm` |
| Qwen 纯文本（去图，控感知变量） | `PLANNER=vllm VLM_TEXT_ONLY=1` |

与 `no_cot / shuffle_subtasks / no_monitor / blind_planner / hard_reset`
等 PDF 必做消融并列，可论证"CoT 机制贡献"与"感知来源(符号 vs 像素)影响"。
