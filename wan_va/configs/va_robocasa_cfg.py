# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
#
# Robocasa inference config for the 26-summer WAM-CoT project.
#
# Design note (IMPORTANT):
#   We evaluate ZERO-SHOT using the released `lingbot-va-posttrain-libero-long`
#   checkpoint. Therefore the *model-facing* interface MUST stay byte-identical
#   to the LIBERO config: same camera keys, same 128x128 input, same 7-dim
#   OSC action channels, same quantile normalization stats. Every Robocasa
#   specific adaptation (camera-name remap, 7-dim -> composite-controller
#   action expansion, success detection) is done CLIENT-SIDE in
#   evaluation/robocasa/. Do not change the values below unless you also
#   retrain / use a Robocasa-specific checkpoint.
from easydict import EasyDict

from .shared_config import va_shared_cfg

va_robocasa_cfg = EasyDict(__name__='Config: VA robocasa (zero-shot from LIBERO-Long)')
va_robocasa_cfg.update(va_shared_cfg)
va_shared_cfg.infer_mode = 'server'

# ---------------------------------------------------------------------------
# Checkpoint path. MUST point at the LIBERO-Long post-trained weights on the
# server, e.g. a sub-dir of the storage pointer in repo-root `checkpoints`:
#   /inspire/qb-ilm2/project/26summer-camp-11/public/group3/26220077/\
#       lingbot-va-storage/checkpoints/lingbot-va-posttrain-libero-long
# Remember to set transformer/config.json -> "attn_mode": "torch" for inference.
# ---------------------------------------------------------------------------
va_robocasa_cfg.wan22_pretrained_model_name_or_path = "/path/to/lingbot-va-posttrain-libero-long"

# --- Model-facing interface: keep identical to LIBERO ----------------------
va_robocasa_cfg.attn_window = 30
va_robocasa_cfg.frame_chunk_size = 4
va_robocasa_cfg.env_type = 'none'

va_robocasa_cfg.height = 128
va_robocasa_cfg.width = 128
va_robocasa_cfg.action_dim = 30
va_robocasa_cfg.action_per_frame = 4
va_robocasa_cfg.obs_cam_keys = [
    'observation.images.agentview_rgb', 'observation.images.eye_in_hand_rgb'
]
va_robocasa_cfg.guidance_scale = 5
va_robocasa_cfg.action_guidance_scale = 1

va_robocasa_cfg.num_inference_steps = 20
va_robocasa_cfg.video_exec_step = -1
va_robocasa_cfg.action_num_inference_steps = 50

va_robocasa_cfg.snr_shift = 5.0
va_robocasa_cfg.action_snr_shift = 0.05

va_robocasa_cfg.used_action_channel_ids = list(range(0, 7))
inverse_used_action_channel_ids = [len(va_robocasa_cfg.used_action_channel_ids)
                                   ] * va_robocasa_cfg.action_dim
for i, j in enumerate(va_robocasa_cfg.used_action_channel_ids):
    inverse_used_action_channel_ids[j] = i
va_robocasa_cfg.inverse_used_action_channel_ids = inverse_used_action_channel_ids

# LIBERO quantile normalization stats (unchanged - the checkpoint expects these).
va_robocasa_cfg.action_norm_method = 'quantiles'
va_robocasa_cfg.norm_stat = {
    "q01": [
        -0.6589285731315613,
        -0.84375,
        -0.9375,
        -0.12107142806053162,
        -0.15964286029338837,
        -0.26571428775787354,
        -1.0
    ] + [0.] * 23,
    "q99": [
        0.8999999761581421,
        0.8544642925262451,
        0.9375,
        0.17142857611179352,
        0.1842857152223587,
        0.34392857551574707,
        1.0
    ] + [0.] * 23,
}
