# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
"""Ablation-3 训练配置:错误阶段数据(隐式 CoT)

与 robotwin_train 完全一致,只把 VLM 阶段标签做**每集 deterministic 置换**
(`vlm_stage_corrupt='shuffle'`)。即每个 episode 的 `vstage[i]` 被 ε-permuted
(per-episode `np.random.RandomState(episode_index)` 派生置换),保持"每帧仍有
标签"但与可视内容解相关。其它训练超参全继承(同基座、同步数、同 λ_kf/λ_st)。

期望观察(报告里写):
- `stage_loss` 收敛到更高水平(信号本身无意义,模型只能记到 marginal 分布)
- 探针 val_acc 在**真实** VLM 阶段标签上掉到接近 stock(0.66 附近)
- 在线 SR 不会高于 stock M0(若高于 = 单纯增加任何 aux head 都涨,M1v 提升
  非 VLM 信号特有)
- ↑↑↑ 若上述退化成立 → 直接证明 M1v 的提升来自**正确的** VLM 监督,而非
  "加 aux head"这一形式

跑法:
  NGPU=8 CONFIG_NAME=robotwin_train_wrongstage CUDA_VISIBLE_DEVICES=0..7 \
    MASTER_PORT=29550 bash script/run_va_posttrain.sh
产出:train_out/checkpoints/robotwin_kf0.1_vlmstage0.1_WRONG/checkpoint_step_<N>/
"""
from easydict import EasyDict

from .va_robotwin_train_cfg import va_robotwin_train_cfg

va_robotwin_train_wrongstage_cfg = EasyDict(
    __name__='Config: VA robotwin train WRONGSTAGE (Ablation-3)')
va_robotwin_train_wrongstage_cfg.update(va_robotwin_train_cfg)

# 关键开关:把 VLM 阶段标签做 per-episode 置换。
# 'shuffle' = 每集随机 permute {0..S-1};'random' = 完全随机标签;'none' = 关
va_robotwin_train_wrongstage_cfg.vlm_stage_corrupt = 'shuffle'

# 明确目标标签,checkpoint 落到独立目录,与正常 M1v 互不污染
va_robotwin_train_wrongstage_cfg.exp_name = (
    'robotwin_kf0.1_vlmstage0.1_WRONG')
