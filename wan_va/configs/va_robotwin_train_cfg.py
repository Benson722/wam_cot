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
# empty_emb.pt (CFG null-prompt text embedding) is SHARED at the dataset
# tree ROOT (one file for all task dirs), not inside each task dir. Point
# at the official shared file. (If absent, regenerate with
# evaluation/robotwin/make_empty_emb.py.)
va_robotwin_train_cfg.empty_emb_path = '/inspire/qb-ilm2/project/26summer-camp-11/public/group3/lingbot-robotwin-clean-and-aug-lerobot/empty_emb.pt'
# WandB: ON but OFFLINE by default — logs are written to disk only (no
# login, no network). Sync later with:  wandb sync <wandb_dir>/wandb/offline-*
# Set wandb_mode='online' (and real WANDB_* env) to stream to the cloud,
# or enable_wandb=False to turn it off entirely.
va_robotwin_train_cfg.enable_wandb = True
va_robotwin_train_cfg.wandb_mode = 'offline'
va_robotwin_train_cfg.wandb_dir = os.path.join(
    va_robotwin_train_cfg.save_root, 'wandb')
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
# ENABLED here because robotwin_train IS the Latent-CoT #1 experiment. The
# keyframe annotation must exist (we generated it):
#   python evaluation/robotwin/keyframe_annotate.py --dataset <ds>
#   -> writes <ds>/meta/<kf_file>  (kf_file is looked up under <repo>/meta/)
# lambda_kf = kf_aux_weight (plan: start 0.1, may ramp down to 0.05).
# For a pure baseline (no implicit CoT), set kf_aux=False / kf_aux_weight=0
# -> dataset adds no keys, no head loss, training byte-identical to stock.
va_robotwin_train_cfg.kf_aux = True
va_robotwin_train_cfg.kf_aux_weight = 0.1
va_robotwin_train_cfg.kf_file = 'keyframes.jsonl'
