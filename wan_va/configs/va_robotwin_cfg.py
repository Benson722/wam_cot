# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
from easydict import EasyDict

from .shared_config import va_shared_cfg

va_robotwin_cfg = EasyDict(__name__='Config: VA robotwin')
va_robotwin_cfg.update(va_shared_cfg)

# The server loads vae/ tokenizer/ text_encoder/ transformer/ ALL from
# subdirs of this one path. A train.py checkpoint dir only has transformer/
# -> symlink the other 3 from the BASE model into it (see below) before
# pointing here. attn_mode is forced to "torch" by the server (4090-safe),
# no config.json edit needed.
#
# BASE (full model, has all 4 subdirs; also the train base):
#   /inspire/hdd/project/26summer-camp-11/26220077/lingbot-va/checkpoints/lingbot-va-posttrain-robotwin
# On the server, make checkpoint_step_1200 self-contained:
#   CK=/inspire/hdd/project/26summer-camp-11/26220077/lingbot-va/train_out/checkpoints/checkpoint_step_1200
#   BS=/inspire/hdd/project/26summer-camp-11/26220077/lingbot-va/checkpoints/lingbot-va-posttrain-robotwin
#   ln -sfn $BS/vae $CK/vae
#   ln -sfn $BS/tokenizer $CK/tokenizer
#   ln -sfn $BS/text_encoder $CK/text_encoder
#
# NOTE: robotwin_train inherits this (va_robotwin_train_cfg .update()s this
# cfg). To RESUME TRAINING from the original base, restore the BASE path.
va_robotwin_cfg.wan22_pretrained_model_name_or_path = "/inspire/hdd/project/26summer-camp-11/26220077/lingbot-va/train_out/checkpoints/checkpoint_step_1200"

va_robotwin_cfg.attn_window = 72
va_robotwin_cfg.frame_chunk_size = 2
va_robotwin_cfg.env_type = 'robotwin_tshape'

va_robotwin_cfg.height = 256
va_robotwin_cfg.width = 320
va_robotwin_cfg.action_dim = 30
va_robotwin_cfg.action_per_frame = 16
va_robotwin_cfg.obs_cam_keys = [
    'observation.images.cam_high', 'observation.images.cam_left_wrist',
    'observation.images.cam_right_wrist'
]
va_robotwin_cfg.guidance_scale = 5
va_robotwin_cfg.action_guidance_scale = 1

va_robotwin_cfg.num_inference_steps = 25
va_robotwin_cfg.video_exec_step = -1
va_robotwin_cfg.action_num_inference_steps = 50

va_robotwin_cfg.snr_shift = 5.0
va_robotwin_cfg.action_snr_shift = 1.0

va_robotwin_cfg.used_action_channel_ids = list(range(0, 7)) + list(
    range(28, 29)) + list(range(7, 14)) + list(range(29, 30))
inverse_used_action_channel_ids = [
    len(va_robotwin_cfg.used_action_channel_ids)
] * va_robotwin_cfg.action_dim
for i, j in enumerate(va_robotwin_cfg.used_action_channel_ids):
    inverse_used_action_channel_ids[j] = i
va_robotwin_cfg.inverse_used_action_channel_ids = inverse_used_action_channel_ids

va_robotwin_cfg.action_norm_method = 'quantiles'
va_robotwin_cfg.norm_stat = {
    "q01": [
        -0.06172713458538055, -3.6716461181640625e-05, -0.08783501386642456,
        -1, -1, -1, -1, -0.3547105032205582, -1.3113021850585938e-06,
        -0.11975435614585876, -1, -1, -1, -1
    ] + [0.] * 16,
    "q99": [
        0.3462600058317184, 0.39966784834861746, 0.14745532035827624, 1, 1, 1,
        1, 0.034201726913452024, 0.39142737388610793, 0.1792279863357542, 1, 1,
        1, 1
    ] + [0.] * 14 + [1.0, 1.0],
}
