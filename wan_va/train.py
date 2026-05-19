# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
import argparse
import os
import signal
import sys
from pathlib import Path
# NOTE: wandb is imported LAZILY inside Trainer.__init__ only when
# config.enable_wandb is True. The repo's README installs wandb with
# `--no-deps`, so a top-level `import wandb` crashes on missing `click`
# even when wandb is disabled. Keep it lazy so enable_wandb=False needs
# no wandb/click at all.

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm
from torch.distributed.checkpoint.state_dict import (
    get_model_state_dict,
    get_optimizer_state_dict,
    set_optimizer_state_dict,
    StateDictOptions,
)
from safetensors.torch import save_file, load_file
import json

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from configs import VA_CONFIGS
from distributed.fsdp import shard_model, apply_ac
from distributed.util import (
    _configure_model, 
    init_distributed, 
    dist_mean, 
    dist_max
)
from einops import rearrange
from modules.utils import (
    load_transformer,
)
from utils import (
    init_logger, 
    logger, 
    get_mesh_id, 
    sample_timestep_id,
    data_seq_to_patch,
    warmup_constant_lambda,
    FlowMatchScheduler
)

from dataset import MultiLatentLeRobotDataset
import gc
import time


def build_exp_tag(config):
    """Derive a checkpoint/wandb tag that NAMES the training objective so
    multiple runs (baseline vs Latent-CoT #1 vs +VLM-stage) stay
    distinguishable on disk. Honor an explicit cfg.exp_name; else auto-build
    from the active aux objectives:
      robotwin_baseline                      (no aux)
      robotwin_kf0.1                         (#1 only)
      robotwin_kf0.1_vlmstage0.1             (#1 + Phase B)
    """
    explicit = getattr(config, 'exp_name', None)
    if explicit:
        return str(explicit)
    parts = []
    if (getattr(config, 'kf_aux', False)
            and float(getattr(config, 'kf_aux_weight', 0.0)) > 0.0):
        parts.append(f"kf{float(config.kf_aux_weight):g}")
    if (getattr(config, 'vlm_stage_aux', False)
            and float(getattr(config, 'vlm_stage_weight', 0.0)) > 0.0):
        parts.append(f"vlmstage{float(config.vlm_stage_weight):g}")
    return "robotwin_" + ("_".join(parts) if parts else "baseline")


class Trainer:
    def __init__(self, config):
        # Objective tag: names checkpoint dir + wandb run so training
        # targets are never confused later (see build_exp_tag).
        self.exp_tag = build_exp_tag(config)
        if config.enable_wandb and config.rank == 0:
            # OFFLINE-FIRST wandb: log to local disk, NO login / NO network.
            # `wandb sync <wandb_dir>/wandb/offline-run-*` to upload later.
            # Telemetry must NEVER crash training -> any failure degrades to
            # disabled and the run continues.
            wb_mode = (os.environ.get('WANDB_MODE')
                       or getattr(config, 'wandb_mode', 'offline'))

            def _clean(name, placeholder):
                v = os.environ.get(name, '').strip()
                if not v or v == placeholder:
                    os.environ.pop(name, None)
                    return None
                return v

            # run_va_posttrain.sh ships placeholders; strip them so wandb's
            # pydantic Settings parsing doesn't reject WANDB_BASE_URL.
            _u = _clean('WANDB_BASE_URL', 'your url')
            if _u and not (_u.startswith('http://')
                           or _u.startswith('https://')):
                os.environ.pop('WANDB_BASE_URL', None)
                _u = None
            _k = _clean('WANDB_API_KEY', 'your key')
            _team = _clean('WANDB_TEAM_NAME', 'your team name')
            _proj = os.environ.get('WANDB_PROJECT', '').strip()
            if not _proj or _proj == 'your project':
                _proj = 'va_robotwin'
            wb_dir = (getattr(config, 'wandb_dir', None)
                      or os.path.join(getattr(config, 'save_root',
                                              './train_out'), 'wandb'))
            try:
                os.makedirs(wb_dir, exist_ok=True)
                os.environ['WANDB_MODE'] = wb_mode
                import wandb  # lazy: only when wandb is actually used
                if wb_mode == 'online' and _k:
                    wandb.login(host=_u, key=_k)
                self.wandb = wandb
                self.wandb.init(
                    entity=_team,
                    project=_proj,
                    config=config,
                    mode=wb_mode,
                    dir=wb_dir,
                    name=self.exp_tag,
                )
                logger.info(
                    f"WandB enabled (mode={wb_mode}, dir={wb_dir}). "
                    f"Offline runs: `wandb sync {wb_dir}/wandb/offline-run-*`")
            except Exception as e:  # noqa: BLE001
                logger.warning(
                    "WandB disabled, training continues "
                    f"({type(e).__name__}: {e})")
                config.enable_wandb = False  # -> later self.wandb.log skipped
        self.step = 0
        self.config = config
        self.device = torch.device(f"cuda:{config.local_rank}")
        self.dtype = config.param_dtype
        self.patch_size = config.patch_size

        # Load models
        logger.info("Loading models...")

        # Load and shard transformer with FSDP
        logger.info("Loading transformer...")

        if hasattr(config, 'resume_from') and config.resume_from:
            transformer_path = os.path.join(config.resume_from, 'transformer')
            if config.rank == 0:
                logger.info(f"Resuming from checkpoint: {transformer_path}")
        else:
            transformer_path = os.path.join(config.wan22_pretrained_model_name_or_path, 'transformer')

        self.transformer = load_transformer(
            transformer_path,
            torch_dtype=torch.float32,
            torch_device='cpu',
            attn_mode="flex"
        )

        logger.info("Setting up activation checkpointing ...")
        apply_ac(self.transformer)

        logger.info("Setting up FSDP...")
        shard_fn = shard_model
        self.transformer = _configure_model(
            model=self.transformer,
            shard_fn=shard_fn,
            param_dtype=self.dtype,
            device=self.device,
            eval_mode=False,
        )
        self.transformer.train()
        self.transformer.requires_grad_(True)

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            [p for p in self.transformer.parameters() if p.requires_grad],
            lr=config.learning_rate,
            betas=(config.beta1, config.beta2),
            eps=1e-8,
            weight_decay=config.weight_decay,
            fused=True,
            foreach=False,
        )

        self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, 
            lr_lambda=lambda step: warmup_constant_lambda(step, warmup_steps=config.warmup_steps))

        # Setup dataloaders
        logger.info("Setting up datasets...")
        train_dataset = MultiLatentLeRobotDataset(config=config)
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=config.world_size,
            rank=config.rank,
            shuffle=True,
            seed=42
        ) if config.world_size > 1 else None
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=(train_sampler is None), 
            num_workers=config.load_worker,
            sampler=train_sampler,
        )

        self.train_scheduler_latent = FlowMatchScheduler(shift=self.config.snr_shift, sigma_min=0.0, extra_one_step=True)
        self.train_scheduler_latent.set_timesteps(1000, training=True)
        self.train_scheduler_action = FlowMatchScheduler(shift=self.config.action_snr_shift, sigma_min=0.0, extra_one_step=True)
        self.train_scheduler_action.set_timesteps(1000, training=True)

        # checkpoints/<exp_tag>/ -> the folder name states the objective
        # (e.g. robotwin_kf0.1_vlmstage0.1) so runs are never confused.
        self.save_dir = Path(config.save_root) / "checkpoints" / self.exp_tag
        self.save_dir.mkdir(parents=True, exist_ok=True)
        if config.rank == 0:
            logger.info(f"Checkpoints -> {self.save_dir} "
                        f"(objective tag: {self.exp_tag})")

        self.gradient_accumulation_steps = getattr(config, 'gradient_accumulation_steps', 1)
        self.train_loader_iter = None
        # if hasattr(config, 'resume_from') and config.resume_from:
        #     self._load_training_state(config.resume_from)
    
    def _get_next_batch(self):
        """Get next batch from iterator, reset if epoch is finished."""
        if self.train_loader_iter is None:
            self.train_loader_iter = iter(self.train_loader)
        
        try:
            batch = next(self.train_loader_iter)
        except StopIteration:
            # Reset sampler and iterator when epoch finishes
            if hasattr(self.train_loader.sampler, 'set_epoch'):
                self.train_loader.sampler.set_epoch(self.train_loader.sampler.epoch + 1)
            self.train_loader_iter = iter(self.train_loader)
            batch = next(self.train_loader_iter)
        
        return batch

    @torch.no_grad()
    def _add_noise(self, latent, train_scheduler, action_mask=False, action_mode=False, noisy_cond_prob=0.):
        B, C, F, H, W = latent.shape

        timestep_ids = sample_timestep_id(batch_size=F, num_train_timesteps=train_scheduler.num_train_timesteps)
        noise = torch.zeros_like(latent).normal_()
        timesteps = train_scheduler.timesteps[timestep_ids].to(device=self.device)
        noisy_latents =train_scheduler.add_noise(latent, noise, timesteps, t_dim=2)
        targets =train_scheduler.training_target(latent, noise, timesteps)

        patch_f, patch_h, patch_w = self.patch_size
        if action_mode:
            patch_f = patch_h = patch_w = 1
        
        latent_grid_id = get_mesh_id(
            latent.shape[-3] // patch_f,  # F
            latent.shape[-2] // patch_h,  # H
            latent.shape[-1] // patch_w,  # W
            t=1 if action_mode else 0,  # 1 for action mode (0 for latent), not used
            f_w=1,
            f_shift=0,
            action=action_mode
        ).to(self.device)  # shape: [4, seq_len]
        latent_grid_id = latent_grid_id[None].repeat(B, 1, 1)

        if torch.rand(1).item() < noisy_cond_prob:
            cond_timestep_ids = sample_timestep_id(
                    batch_size=F,
                    min_timestep_bd=0.5, 
                    max_timestep_bd=1.0, 
                    num_train_timesteps=train_scheduler.num_train_timesteps,
                )
            noise = torch.zeros_like(latent).normal_()
            cond_timesteps = train_scheduler.timesteps[cond_timestep_ids].to(device=self.device)
            latent = train_scheduler.add_noise(latent, noise, cond_timesteps, t_dim=2)
        else:
            cond_timesteps = torch.zeros_like(timesteps)

        if action_mask is not None:
            noisy_latents *= action_mask.float()
            targets *= action_mask.float()
            latent *= action_mask.float()

        return dict(
            timesteps=timesteps[None].repeat(B, 1),
            noisy_latents=noisy_latents,
            targets=targets,
            latent=latent,
            cond_timesteps=cond_timesteps[None].repeat(B, 1),
            grid_id=latent_grid_id,
        )

    @torch.no_grad()
    def _prepare_input_dict(self, batch_dict):
        """Prepare input dict following infer code pattern from wan_va_server.py."""
        # Generate grid_id following infer code (no batch dimension yet)
        # For action mode: get_mesh_id(shape[-3], shape[-2], shape[-1], t=1, f_w=1, f_shift, action=True)
        latent_dict = self._add_noise(
            latent=batch_dict['latents'], 
            train_scheduler=self.train_scheduler_latent, 
            action_mask=None, 
            action_mode=False,
            noisy_cond_prob=0.5)
        
        action_dict = self._add_noise(
            latent=batch_dict['actions'], 
            train_scheduler=self.train_scheduler_action, 
            action_mask=batch_dict['actions_mask'], 
            action_mode=True,
            noisy_cond_prob=0.0)

        latent_dict['text_emb'] = batch_dict['text_emb']
        action_dict['text_emb'] = batch_dict['text_emb']
        action_dict['actions_mask'] = batch_dict['actions_mask']

        input_dict = {
            'latent_dict': latent_dict,
            'action_dict': action_dict,
            'chunk_size': torch.randint(1, 5, (1,)).item(),
            'window_size': torch.randint(4, 65, (1,)).item(),
        }
        # Latent-CoT #1: pass keyframe targets through (None when the dataset
        # didn't add them, i.e. cfg.kf_aux off -> loss term is skipped).
        input_dict['kf_dist'] = batch_dict.get('kf_dist')
        input_dict['kf_mask'] = batch_dict.get('kf_mask')
        input_dict['vlm_stage'] = batch_dict.get('vlm_stage')  # Phase B
        return input_dict

    def convert_input_format(self, input_dict):
        """Convert input dict to match transformer input format if needed."""
        for key, value in input_dict.items():
            input_dict[key] = value.to(self.device)#.to(self.dtype)
        return input_dict

    def compute_loss(self,
        input_dict,
        pred
    ):
        latent_pred, action_pred = pred[0], pred[1]
        kf_pred = pred[2] if len(pred) > 2 else None     # Latent-CoT #1
        stage_pred = pred[4] if len(pred) > 4 else None  # Phase B (VLM)
        action_pred = rearrange(action_pred, 'b (f n) c -> b c f n 1', f=input_dict['action_dict']['targets'].shape[-3])
        latent_pred = data_seq_to_patch(
                        self.patch_size, latent_pred,
                        input_dict['latent_dict']['targets'].shape[-3], input_dict['latent_dict']['targets'].shape[-2],
                        input_dict['latent_dict']['targets'].shape[-1], batch_size=latent_pred.shape[0])
        Bn, Fn = input_dict['latent_dict']['timesteps'].shape
        latent_loss_weight = self.train_scheduler_latent.training_weight(input_dict['latent_dict']['timesteps'].flatten()).reshape(Bn, Fn)
        action_loss_weight = self.train_scheduler_action.training_weight(input_dict['action_dict']['timesteps'].flatten()).reshape(Bn, Fn)

        # Frame-wise video loss calculation
        latent_loss = F.mse_loss(latent_pred.float(), input_dict['latent_dict']['targets'].float().detach(), reduction='none')
        latent_loss = latent_loss * latent_loss_weight[:, None, :, None, None]
        # Permute to (B, F, H, W, C) and flatten to (B*F, H*W*C)
        latent_loss = latent_loss.permute(0, 2, 3, 4, 1)  # (B, C, F, H, W) -> (B, F, H, W, C)
        latent_loss = latent_loss.flatten(0, 1).flatten(1)  # (B, F, H, W, C) -> (B*F, H*W*C)
        # Sum per frame and compute mask per frame
        latent_loss_per_frame = latent_loss.sum(dim=1)  # (B*F,)
        latent_mask_per_frame = torch.ones_like(latent_loss).sum(dim=1)  # (B*F,)
        latent_loss = (latent_loss_per_frame / (latent_mask_per_frame + 1e-6)).mean()

        # Frame-wise action loss calculation
        action_loss = F.mse_loss(action_pred.float(), input_dict['action_dict']['targets'].float().detach(), reduction='none')
        action_loss = action_loss * action_loss_weight[:, None, :, None, None]
        action_loss = action_loss * input_dict['action_dict']['actions_mask'].float()
        # Permute to (B, F, H, W, C) and flatten to (B*F, H*W*C)
        action_loss = action_loss.permute(0, 2, 3, 4, 1)  # (B, C, F, H, W) -> (B, F, H, W, C)
        action_mask = input_dict['action_dict']['actions_mask'].float().permute(0, 2, 3, 4, 1)  # (B, C, F, H, W) -> (B, F, H, W, C)
        action_loss = action_loss.flatten(0, 1).flatten(1)  # (B, F, H, W, C) -> (B*F, H*W*C)
        action_mask = action_mask.flatten(0, 1).flatten(1)  # (B, F, H, W, C) -> (B*F, H*W*C)
        # Sum per frame and normalize by mask per frame
        action_loss_per_frame = action_loss.sum(dim=1)  # (B*F,)
        action_mask_per_frame = action_mask.sum(dim=1)  # (B*F,)
        action_loss = (action_loss_per_frame / (action_mask_per_frame + 1e-6)).mean()

        # ---- Latent-CoT #1: keyframe-distance auxiliary loss ------------
        # Strict no-op unless cfg.kf_aux & kf_aux_weight>0 AND the dataset
        # supplied kf targets. Huber on log1p(distance), masked-mean.
        kf_loss = torch.zeros((), device=action_pred.device,
                              dtype=torch.float32)
        cfg = self.config
        if (getattr(cfg, 'kf_aux', False)
                and float(getattr(cfg, 'kf_aux_weight', 0.0)) > 0.0
                and kf_pred is not None
                and input_dict.get('kf_mask') is not None
                and input_dict.get('kf_dist') is not None):
            kf_dist = input_dict['kf_dist'].to(kf_pred.device).float()
            kf_mask = input_dict['kf_mask'].to(kf_pred.device).float()
            # defensive length align (latent_frame_num rounding safety)
            n = min(kf_pred.shape[-1], kf_dist.shape[-1], kf_mask.shape[-1])
            kp = kf_pred[..., :n].float()
            tgt = torch.log1p(kf_dist[..., :n].clamp(min=0.0))
            m = kf_mask[..., :n]
            per = F.smooth_l1_loss(kp, tgt, reduction='none') * m
            kf_loss = per.sum() / (m.sum() + 1e-6)
            kf_loss = kf_loss * float(cfg.kf_aux_weight)

        # ---- Phase B: VLM semantic-stage CE loss ------------------------
        # Strict no-op unless cfg.vlm_stage_aux & weight>0 AND the dataset
        # supplied vlm_stage. CE over [B,F_lat,S], ignore_index=-1.
        stage_loss = torch.zeros((), device=action_pred.device,
                                 dtype=torch.float32)
        if (getattr(cfg, 'vlm_stage_aux', False)
                and float(getattr(cfg, 'vlm_stage_weight', 0.0)) > 0.0
                and stage_pred is not None
                and input_dict.get('vlm_stage') is not None):
            vst = input_dict['vlm_stage'].to(stage_pred.device).long()
            n = min(stage_pred.shape[1], vst.shape[-1])
            sp = stage_pred[:, :n].reshape(-1, stage_pred.shape[-1]).float()
            st = vst[..., :n].reshape(-1)
            if (st >= 0).any():
                stage_loss = F.cross_entropy(sp, st, ignore_index=-1)
                stage_loss = stage_loss * float(cfg.vlm_stage_weight)

        gas = self.gradient_accumulation_steps
        return (latent_loss / gas, action_loss / gas, kf_loss / gas,
                stage_loss / gas)

    def _train_step(self, batch, batch_idx):
        """Train a single batch, returns losses for logging."""
        batch = self.convert_input_format(batch)
        input_dict = self._prepare_input_dict(batch)
        
        should_sync = (batch_idx + 1) % self.gradient_accumulation_steps == 0
        
        if not should_sync:
            self.transformer.set_requires_gradient_sync(False)
        else:
            self.transformer.set_requires_gradient_sync(True)

        output = self.transformer(input_dict, train_mode=True)
        latent_loss, action_loss, kf_loss, stage_loss = self.compute_loss(
            input_dict, output)
        loss = latent_loss + action_loss + kf_loss + stage_loss

        loss.backward()

        losses = {'latent_loss': latent_loss.detach(),
                  'action_loss': action_loss.detach(),
                  'kf_loss': kf_loss.detach(),
                  'stage_loss': stage_loss.detach()}
        
        # Only update weights after accumulating gradients
        if should_sync:
            total_norm = torch.nn.utils.clip_grad_norm_(self.transformer.parameters(), 2.0)
            self.optimizer.step()
            self.lr_scheduler.step()
            self.optimizer.zero_grad()
            
            losses['total_norm'] = total_norm
            losses['should_log'] = True
        else:
            losses['should_log'] = False

        return losses

    def save_checkpoint(self,):
        """Save model checkpoint in the same format as pretrained model."""
        try:
            state_dict = get_model_state_dict(
                self.transformer,
                options=StateDictOptions(full_state_dict=True, cpu_offload=True),
            )
            state_dict_bf16 = {k: v.to(torch.bfloat16) for k, v in state_dict.items()}
            # optim_state = get_optimizer_state_dict(
            #         self.transformer, self.optimizer,
            #         options=StateDictOptions(full_state_dict=True, cpu_offload=True),
            #     )

            # Only rank 0 saves the checkpoint
            if self.config.rank == 0:
                checkpoint_dir = self.save_dir / f"checkpoint_step_{self.step}"
                checkpoint_dir.mkdir(parents=True, exist_ok=True)

                # Save transformer in the same format as pretrained model
                transformer_dir = checkpoint_dir / "transformer"
                transformer_dir.mkdir(parents=True, exist_ok=True)

                logger.info(f"Saving transformer to {transformer_dir}")

                # Manually save in diffusers format (outside FSDP context to avoid deadlock)
                # Save model weights
                model_file = transformer_dir / "diffusion_pytorch_model.safetensors"
                save_file(state_dict_bf16, model_file)

                # Save config (copy from original transformer config and update _name_or_path)
                config_file = transformer_dir / "config.json"
                config_dict = dict(self.transformer.config)
                config_dict.pop('_name_or_path', None)
                with open(config_file, 'w') as f:
                    json.dump(config_dict, f, indent=2)

                # Objective manifest: makes the run self-describing even if
                # the folder is moved/renamed (which aux heads, weights,
                # base ckpt, dataset, step).
                cfg = self.config
                meta = {
                    'exp_tag': self.exp_tag,
                    'step': self.step,
                    'num_steps': getattr(cfg, 'num_steps', None),
                    'base_ckpt': getattr(
                        cfg, 'wan22_pretrained_model_name_or_path', None),
                    'dataset_path': getattr(cfg, 'dataset_path', None),
                    'kf_aux': bool(getattr(cfg, 'kf_aux', False)),
                    'kf_aux_weight': float(getattr(cfg, 'kf_aux_weight', 0.0)),
                    'kf_file': getattr(cfg, 'kf_file', None),
                    'vlm_stage_aux': bool(
                        getattr(cfg, 'vlm_stage_aux', False)),
                    'vlm_stage_weight': float(
                        getattr(cfg, 'vlm_stage_weight', 0.0)),
                    'vlm_stage_file': getattr(cfg, 'vlm_stage_file', None),
                    'vlm_num_stages': int(
                        getattr(cfg, 'vlm_num_stages', 0)),
                    'learning_rate': getattr(cfg, 'learning_rate', None),
                    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                }
                with open(checkpoint_dir / "meta.json", 'w') as f:
                    json.dump(meta, f, indent=2)

                # # Save optimizer state and training metadata in PyTorch format
                # training_state_path = checkpoint_dir / "training_state.pt"
                # logger.info(f"Saving training state to {training_state_path}")
                # torch.save({
                #     'step': self.step,
                #     'optimizer_state_dict': optim_state,
                #     'config': vars(self.config),
                # }, training_state_path)

                logger.info(f"Checkpoint saved successfully at step {self.step}")

            # Synchronize all processes after saving
            if dist.is_initialized():
                dist.barrier()

        except Exception as e:
            if self.config.rank == 0:
                logger.error(f"Failed to save checkpoint: {e}")
                import traceback
                logger.error(traceback.format_exc())
            # Ensure all processes stay synchronized even on error
            if dist.is_initialized():
                dist.barrier()

    def _load_training_state(self, checkpoint_path):
        """Load training state (optimizer + step) after FSDP and optimizer creation."""
        checkpoint_dir = Path(checkpoint_path)
        training_state_path = checkpoint_dir / "training_state.pt"

        if not training_state_path.exists():
            if self.config.rank == 0:
                logger.warning(f"Training state not found: {training_state_path}, starting from step 0")
            return

        if self.config.rank == 0:
            logger.info(f"Loading training state from {training_state_path}")

        # All ranks load the training state directly
        training_state = torch.load(training_state_path, map_location='cpu', weights_only=False)

        # All ranks load optimizer state (required for FSDP)
        set_optimizer_state_dict(
            self.transformer, self.optimizer,
            optim_state_dict=training_state['optimizer_state_dict'],
            options=StateDictOptions(full_state_dict=True, strict=False)
        )
        self.step = training_state.get('step', 0)

        if self.config.rank == 0:
            logger.info(f"Training state loaded, resuming from step {self.step}")

        # Synchronize all ranks
        if dist.is_initialized():
            dist.barrier()

    @torch.no_grad()
    def collect_hidden(self, out_path, num_batches):
        """Latent-CoT #4: dump per-latent-frame BACKBONE hidden h_t (=kf_feat,
        the pre-head pooled hidden that forward_train returns as pred[3])
        plus the stage label, for the stock-vs-#1 linear probe. Reuses the
        EXACT training forward (model + _prepare_input_dict + _add_noise) so
        the probed features are precisely what the model computes — no
        re-implementation risk. kf_feat is pre-`kf_aux_head`, so it is the
        backbone representation whether or not #1 was trained (stock ckpt =>
        stock backbone hidden; #1 ckpt => #1-shaped hidden)."""
        self.transformer.eval()
        feats, stages, vstages, eps = [], [], [], []
        n = 0
        for bi in range(int(num_batches)):
            batch = self.convert_input_format(self._get_next_batch())
            if 'kf_stage' not in batch:
                raise RuntimeError(
                    "batch has no 'kf_stage' -> set cfg.kf_aux=True and run "
                    "keyframe_annotate.py so the loader emits it.")
            input_dict = self._prepare_input_dict(batch)
            out = self.transformer(input_dict, train_mode=True)
            if not (isinstance(out, (tuple, list)) and len(out) > 3):
                raise RuntimeError(
                    "forward_train didn't return kf_feat (pred[3]); re-sync "
                    "wan_va/modules/model.py.")
            kf_feat = out[3].float().cpu()          # [B, F_lat, d]
            st = batch['kf_stage'].cpu()            # [B, F_lat]
            ep = batch['kf_episode'].cpu()          # [B]
            # VLM semantic-stage label (stages.jsonl) if the loader emits it
            # (cfg.vlm_stage_aux=True). -1 = unlabeled frame (probe drops it).
            vst = batch.get('vlm_stage')
            vst = vst.cpu() if vst is not None else None
            B, F = kf_feat.shape[0], kf_feat.shape[1]
            m = min(F, st.shape[1])
            if vst is not None:
                m = min(m, vst.shape[1])
            for b in range(B):
                feats.append(kf_feat[b, :m])
                stages.append(st[b, :m])
                if vst is not None:
                    vstages.append(vst[b, :m])
                eps.append(st[b, :m].clone().fill_(int(ep[b])))
            n += B
            if self.config.rank == 0 and (bi + 1) % 10 == 0:
                logger.info(f"[probe-collect] {bi+1}/{num_batches} batches "
                            f"({n} samples)")
        import torch as _t
        X = _t.cat(feats, 0)                        # [N, d]
        y = _t.cat(stages, 0).long()                # [N]  kf-derived stage
        g = _t.cat(eps, 0).long()                   # [N]  episode id
        vy = _t.cat(vstages, 0).long() if vstages else None  # [N] VLM stage
        if self.config.rank == 0:
            os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".",
                        exist_ok=True)
            dump = {"feat": X, "stage": y, "episode": g,
                    "ckpt": getattr(self.config,
                                    "wan22_pretrained_model_name_or_path",
                                    "?")}
            if vy is not None:
                dump["vlm_stage"] = vy
            _t.save(dump, out_path)
            vinfo = (f" vlm_stages={int(vy[vy >= 0].max())+1}"
                     f" (labeled {int((vy >= 0).sum())}/{vy.numel()})"
                     if vy is not None and (vy >= 0).any()
                     else " vlm_stage=ABSENT")
            logger.info(f"[probe-collect] wrote {out_path} "
                        f"feat={tuple(X.shape)} N={X.shape[0]} "
                        f"kf_stages={int(y.max())+1}{vinfo}")
        if dist.is_initialized():
            dist.barrier()

    def train(self):
        """Main training loop - train by steps instead of epochs."""
        logger.info(f"Starting training for {self.config.num_steps} steps...")
        self.transformer.train()

        # Graceful Ctrl-C: SIGINT/SIGTERM only sets a flag; we checkpoint and
        # exit cleanly at the next optimizer-step boundary (so the FSDP
        # collective save isn't interrupted mid-write). Useful since the model
        # converges fast and one rarely needs all num_steps.
        self._stop_requested = False

        def _request_stop(signum, _frame):
            self._stop_requested = True
            logger.warning(
                f"Signal {signum} received -> will save a checkpoint and "
                "stop at the next step boundary (press again to hard-kill).")

        for _s in (signal.SIGINT, signal.SIGTERM):
            try:
                signal.signal(_s, _request_stop)
            except Exception:  # noqa: BLE001
                pass

        progress_bar = tqdm(
            total=self.config.num_steps,
            desc="Training",
            disable=(self.config.rank != 0),
            leave=True,
            dynamic_ncols=True,
            initial=self.step
        )

        self.optimizer.zero_grad()
        accumulated_latent_losses = []
        accumulated_action_losses = []
        accumulated_kf_losses = []          # Latent-CoT #1 visibility
        accumulated_stage_losses = []       # Phase B (VLM stage) visibility
        step_in_accumulation = 0

        while self.step < self.config.num_steps:
            # Get next batch (handles epoch reset automatically)
            batch = self._get_next_batch()
            
            losses = self._train_step(batch, step_in_accumulation)
            
            # Accumulate losses for logging
            accumulated_latent_losses.append(losses['latent_loss'])
            accumulated_action_losses.append(losses['action_loss'])
            accumulated_kf_losses.append(losses['kf_loss'])
            accumulated_stage_losses.append(losses['stage_loss'])
            step_in_accumulation += 1

            # Log and checkpoint when optimizer steps
            if losses['should_log']:
                lr = self.lr_scheduler.get_last_lr()[0]

                # Average accumulated losses
                latent_loss_show = dist_mean(torch.stack(accumulated_latent_losses).sum()).detach().cpu().item()
                action_loss_show = dist_mean(torch.stack(accumulated_action_losses).sum()).detach().cpu().item()
                max_latent_loss_show = dist_max(torch.stack(accumulated_latent_losses).sum()).detach().cpu().item()
                max_action_loss_show = dist_max(torch.stack(accumulated_action_losses).sum()).detach().cpu().item()
                kf_loss_show = dist_mean(torch.stack(accumulated_kf_losses).sum()).detach().cpu().item()
                stage_loss_show = dist_mean(torch.stack(accumulated_stage_losses).sum()).detach().cpu().item()

                # Clear accumulated losses
                accumulated_latent_losses = []
                accumulated_action_losses = []
                accumulated_kf_losses = []
                accumulated_stage_losses = []
                step_in_accumulation = 0

                torch.cuda.synchronize()
                if self.step % self.config.gc_interval == 0:
                    torch.cuda.empty_cache()
                    gc.collect()

                if self.config.rank == 0:
                    total_norm = losses['total_norm']
                    progress_bar.n += 1
                    progress_bar.set_postfix({
                        'latent_loss': f'{latent_loss_show:.4f}',
                        'action_loss': f'{action_loss_show:.4f}',
                        'kf_loss': f'{kf_loss_show:.4f}',
                        'stage_loss': f'{stage_loss_show:.4f}',
                        'step': self.step,
                        'grad_norm': f'{total_norm.item():.2f}',
                        'lr': f'{lr:.2e}'
                    })
                    if self.config.enable_wandb:
                        self.wandb.log({
                            'loss_metrics/global_avg_video_loss': latent_loss_show,
                            'loss_metrics/global_avg_action_loss': action_loss_show,
                            'loss_metrics/global_max_video_loss': max_latent_loss_show,
                            'loss_metrics/global_max_action_loss': max_action_loss_show,
                            'loss_metrics/global_avg_kf_loss': kf_loss_show,
                            'loss_metrics/global_avg_stage_loss': stage_loss_show,
                            'grad_norm': total_norm.item(),
                            'lr': lr,
                        }, step=self.step)
                
                self.step += 1
                
                if self.step % self.config.save_interval == 0:
                    if self.config.rank == 0:
                        logger.info(f"Starting save model at step {self.step}")
                    self.save_checkpoint()

                # Cooperative early stop (Ctrl-C). Decide COLLECTIVELY so all
                # ranks enter the (collective) save together -> no deadlock.
                stop_t = torch.tensor(
                    [1 if getattr(self, '_stop_requested', False) else 0],
                    device=self.device)
                if dist.is_initialized():
                    dist.all_reduce(stop_t, op=dist.ReduceOp.MAX)
                if stop_t.item() > 0:
                    if self.config.rank == 0:
                        logger.info(
                            f"Interrupt: saving checkpoint at step "
                            f"{self.step} then exiting.")
                    self.save_checkpoint()
                    if dist.is_initialized():
                        dist.barrier()
                    break

            if dist.is_initialized():
                dist.barrier()

        progress_bar.close()
        logger.info("Training completed!")


def run(args):
    """Main entry point."""
    config = VA_CONFIGS[args.config_name]

    rank = int(os.getenv("RANK", 0))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    init_distributed(world_size, local_rank, rank)

    config.rank = rank
    config.local_rank = local_rank
    config.world_size = world_size

    if args.save_root is not None:
        config.save_root = args.save_root

    # Latent-CoT #4: override the ckpt to probe (stock vs #1) without
    # editing the config file each time.
    if getattr(args, "probe_ckpt", None):
        config.wan22_pretrained_model_name_or_path = args.probe_ckpt

    if rank == 0:
        logger.info(f"Using config: {args.config_name}")
        logger.info(f"World size: {world_size}, Local rank: {local_rank}")

    trainer = Trainer(config)
    if getattr(args, "probe_collect", None):
        if rank == 0:
            logger.info(
                f"[probe-collect] ckpt="
                f"{config.wan22_pretrained_model_name_or_path} -> "
                f"{args.probe_collect} ({args.probe_collect_batches} batches)")
        trainer.collect_hidden(args.probe_collect,
                               args.probe_collect_batches)
        return
    trainer.train()


def main():
    """Parse arguments and run training."""
    parser = argparse.ArgumentParser(description="Train WAN model for robotics")
    parser.add_argument(
        "--config-name",
        type=str,
        default='robotwin_train',
        help="Config name",
    )
    parser.add_argument(
        "--save-root",
        type=str,
        default=None,
        help="Root directory for saving checkpoints",
    )
    # Latent-CoT #4: collect backbone hidden h_t instead of training.
    parser.add_argument(
        "--probe-collect", type=str, default=None,
        help="Path to write the h_t feature dump (.pt). When set, runs "
             "feature collection instead of training.",
    )
    parser.add_argument(
        "--probe-collect-batches", type=int, default=64,
        help="#batches to collect for the probe (default 64).",
    )
    parser.add_argument(
        "--probe-ckpt", type=str, default=None,
        help="Override transformer ckpt dir to probe (stock vs #1), e.g. "
             "train_out/checkpoints/checkpoint_step_1000",
    )

    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    init_logger()
    main()