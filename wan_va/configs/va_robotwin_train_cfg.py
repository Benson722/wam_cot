# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
from easydict import EasyDict
from .va_robotwin_cfg import va_robotwin_cfg
import os

va_robotwin_train_cfg = EasyDict(__name__='Config: VA robotwin train')
va_robotwin_train_cfg.update(va_robotwin_cfg)

# H200 server: a single RoboTwin task in LeRobot format (already downloaded).
# `dataset_path` is recursively scanned for `meta/info.json`, so point it at
# the task dir (or a parent dir holding several such task dirs).
va_robotwin_train_cfg.dataset_path = '/inspire/qb-ilm2/project/26summer-camp-11/public/group3/lingbot-robotwin-clean-and-aug-lerobot/lerobot_robotwin_eef_aug_500/adjust_bottle-aloha-agilex_randomized_500-1000'
# empty_emb.pt is expected next to the dataset; override here if it lives
# elsewhere (e.g. a shared path) for your install.
va_robotwin_train_cfg.empty_emb_path = os.path.join(va_robotwin_train_cfg.dataset_path, 'empty_emb.pt')
va_robotwin_train_cfg.enable_wandb = True
va_robotwin_train_cfg.load_worker = 16
va_robotwin_train_cfg.save_interval = 1000
va_robotwin_train_cfg.gc_interval = 50
va_robotwin_train_cfg.cfg_prob = 0.1

# Training parameters
va_robotwin_train_cfg.learning_rate = 1e-5
va_robotwin_train_cfg.beta1 = 0.9
va_robotwin_train_cfg.beta2 = 0.95
va_robotwin_train_cfg.weight_decay = 0.1
va_robotwin_train_cfg.warmup_steps = 10
va_robotwin_train_cfg.batch_size = 1 
va_robotwin_train_cfg.gradient_accumulation_steps = 1
va_robotwin_train_cfg.num_steps = 50000

# ---- Latent-CoT #1: keyframe auxiliary head (see latent_plan.md) --------
# Defaults are a strict NO-OP: kf_aux=False -> dataset adds no keys, no head,
# no loss term -> baseline training byte-identical. To enable:
#   1) python evaluation/robotwin/keyframe_annotate.py --dataset <ds>
#      -> writes <ds>/meta/keyframes.jsonl
#   2) set kf_aux=True and kf_aux_weight>0 here (lambda_kf; plan suggests
#      0.1 ramping down to 0.05).
# kf_file: annotation filename under <dataset>/meta/.
va_robotwin_train_cfg.kf_aux = False
va_robotwin_train_cfg.kf_aux_weight = 0.0
va_robotwin_train_cfg.kf_file = 'keyframes.jsonl'
