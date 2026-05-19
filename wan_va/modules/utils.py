# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
import torch
from diffusers import AutoencoderKLWan
from transformers import (
    T5TokenizerFast,
    UMT5EncoderModel,
)

from .model import WanTransformer3DModel


def load_vae(
    vae_path,
    torch_dtype,
    torch_device,
):
    vae = AutoencoderKLWan.from_pretrained(
        vae_path,
        torch_dtype=torch_dtype,
    )
    return vae.to(torch_device)


def load_text_encoder(
    text_encoder_path,
    torch_dtype,
    torch_device,
):
    text_encoder = UMT5EncoderModel.from_pretrained(
        text_encoder_path,
        torch_dtype=torch_dtype,
    )
    return text_encoder.to(torch_device)


def load_tokenizer(tokenizer_path, ):
    tokenizer = T5TokenizerFast.from_pretrained(tokenizer_path, )
    return tokenizer


def load_transformer(
    transformer_path,
    torch_dtype,
    torch_device,
    **kwargs
):
    # NOTE: this diffusers build forbids low_cpu_mem_usage=False when the
    # model has keep_in_fp32_modules, so we MUST keep the accelerate
    # meta-device load path. Consequence: modules added after the checkpoint
    # was saved (the Latent-CoT #1 `kf_aux_head`, absent from every released
    # ckpt) come back as **meta** tensors and would crash the subsequent
    # `model.to(device)` ("Cannot copy out of meta tensor; no data").
    # Fix: detect the still-meta submodules and materialize + (re)initialize
    # ONLY those (a fresh from-scratch head -> random init is exactly right);
    # loaded pretrained weights are never touched.
    model = WanTransformer3DModel.from_pretrained(
        transformer_path,
        torch_dtype=torch_dtype,
        **kwargs
    )
    meta_mods = set()
    for n, p in list(model.named_parameters()):
        if getattr(p, "is_meta", False):
            meta_mods.add(n.rsplit(".", 1)[0])
    for n, b in list(model.named_buffers()):
        if getattr(b, "is_meta", False):
            meta_mods.add(n.rsplit(".", 1)[0])
    for mod_name in sorted(meta_mods):
        sub = model.get_submodule(mod_name)
        sub.to_empty(device="cpu")          # meta -> real (uninitialized)
        for m in sub.modules():
            if hasattr(m, "reset_parameters"):
                m.reset_parameters()        # proper random init
        sub.to(torch_dtype)                 # match model compute dtype only
    return model.to(torch_device)


def patchify(x, patch_size):
    if patch_size is None or patch_size == 1:
        return x
    batch_size, channels, frames, height, width = x.shape
    x = x.view(batch_size, channels, frames, height // patch_size, patch_size,
               width // patch_size, patch_size)
    x = x.permute(0, 1, 6, 4, 2, 3, 5).contiguous()
    x = x.view(batch_size, channels * patch_size * patch_size, frames,
               height // patch_size, width // patch_size)
    return x


class WanVAEStreamingWrapper:

    def __init__(self, vae_model):
        self.vae = vae_model
        self.encoder = vae_model.encoder
        self.quant_conv = vae_model.quant_conv

        if hasattr(self.vae, "_cached_conv_counts"):
            self.enc_conv_num = self.vae._cached_conv_counts["encoder"]
        else:
            count = 0
            for m in self.encoder.modules():
                if m.__class__.__name__ == "WanCausalConv3d":
                    count += 1
            self.enc_conv_num = count

        self.clear_cache()

    def clear_cache(self):
        self.feat_cache = [None] * self.enc_conv_num

    def encode_chunk(self, x_chunk):
        if hasattr(self.vae.config,
                   "patch_size") and self.vae.config.patch_size is not None:
            x_chunk = patchify(x_chunk, self.vae.config.patch_size)
        feat_idx = [0]
        out = self.encoder(x_chunk,
                           feat_cache=self.feat_cache,
                           feat_idx=feat_idx)
        enc = self.quant_conv(out)
        return enc
