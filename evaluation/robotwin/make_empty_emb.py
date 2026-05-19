# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
"""Generate the missing `empty_emb.pt` (CFG null-prompt text embedding).

`empty_emb.pt` is NOT a model weight — it is the empty prompt ("") encoded by
the SAME Wan2.2 text encoder used to produce each segment's `text_emb`. The
training dataset loader substitutes it for the real text embedding with
probability `cfg_prob` (classifier-free-guidance dropout):

    text_emb = data_dict[f"{cam}.text_emb"]
    if torch.rand(1) < cfg_prob: text_emb = self.empty_emb

USE THE OFFICIAL FILE IF IT EXISTS (exact shape guaranteed):
    find <dataset_root> -name empty_emb.pt
This generator is the fallback when it is genuinely absent. It reproduces the
server's `_get_t5_prompt_embeds("")` exactly and CONFORMS the result (rank /
seq-len / dtype) to a real segment `text_emb` read from the dataset, so it is
a perfect drop-in.

Run (lingbot venv, repo root):
    python evaluation/robotwin/make_empty_emb.py --config robotwin_train
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import torch

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

try:
    from diffusers.pipelines.wan.pipeline_wan import prompt_clean
except Exception:  # noqa: BLE001
    def prompt_clean(s):  # minimal fallback
        return s.strip()


def _encode_empty(model_path, max_seq_len, device, dtype):
    """Faithful copy of wan_va_server._get_t5_prompt_embeds(prompt='')."""
    from wan_va.modules.utils import load_text_encoder, load_tokenizer

    tok = load_tokenizer(os.path.join(model_path, "tokenizer"))
    enc = load_text_encoder(os.path.join(model_path, "text_encoder"),
                            torch_dtype=dtype, torch_device=device)
    prompt = [prompt_clean("")]
    ti = tok(prompt, padding="max_length", max_length=max_seq_len,
             truncation=True, add_special_tokens=True,
             return_attention_mask=True, return_tensors="pt")
    ids, mask = ti.input_ids, ti.attention_mask
    seq_lens = mask.gt(0).sum(dim=1).long()
    with torch.no_grad():
        emb = enc(ids.to(device), mask.to(device)).last_hidden_state
    emb = emb.to(dtype=dtype, device="cpu")
    emb = [u[:v] for u, v in zip(emb, seq_lens)]
    emb = torch.stack([
        torch.cat([u, u.new_zeros(max_seq_len - u.size(0), u.size(1))])
        for u in emb
    ], dim=0)                       # [1, max_seq_len, D]
    return emb


def _ref_text_emb(dataset_root, cam_key):
    """Read one segment .pth to learn the exact text_emb shape/dtype."""
    pat = os.path.join(dataset_root, "latents", "chunk-*", cam_key,
                       "episode_*.pth")
    files = sorted(glob.glob(pat))
    if not files:
        return None
    rec = torch.load(files[0], weights_only=False, map_location="cpu")
    te = rec.get("text_emb")
    return te


def main():
    ap = argparse.ArgumentParser(description="Generate empty_emb.pt (CFG null)")
    ap.add_argument("--config", default="robotwin_train",
                    help="VA_CONFIGS key (model path / out path / cams)")
    ap.add_argument("--dataset", default=None,
                    help="dataset root (else cfg.dataset_path)")
    ap.add_argument("--out", default=None,
                    help="output path (else cfg.empty_emb_path)")
    ap.add_argument("--model-path", default=None,
                    help="Wan2.2 model dir w/ tokenizer/ + text_encoder/ "
                         "(else cfg.wan22_pretrained_model_name_or_path)")
    ap.add_argument("--max-seq-len", type=int, default=512)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    from wan_va.configs import VA_CONFIGS

    cfg = VA_CONFIGS[args.config]
    ds = args.dataset or cfg.dataset_path
    out = args.out or cfg.empty_emb_path
    model_path = args.model_path or cfg.wan22_pretrained_model_name_or_path
    cam0 = cfg.obs_cam_keys[0]

    ref = _ref_text_emb(ds, cam0)
    ref_dtype = ref.dtype if ref is not None else torch.bfloat16
    if ref is not None:
        print(f"[empty_emb] reference text_emb: shape={tuple(ref.shape)} "
              f"dtype={ref.dtype}")
    else:
        print("[empty_emb] WARNING: no reference .pth found; using "
              f"[1,{args.max_seq_len},D] bf16 (server convention)")

    emb = _encode_empty(model_path, args.max_seq_len, args.device, ref_dtype)
    print(f"[empty_emb] encoded '' -> {tuple(emb.shape)} {emb.dtype}")

    # Conform to the reference text_emb's rank / seq-len so it is a perfect
    # drop-in substitute in the loader.
    if ref is not None:
        if ref.dim() == 2:                       # [L, D]
            e = emb[0]                           # [max_seq_len, D]
            L = ref.shape[0]
            if e.shape[0] >= L:
                e = e[:L]
            else:
                e = torch.cat(
                    [e, e.new_zeros(L - e.shape[0], e.shape[1])], dim=0)
            emb = e
        elif ref.dim() == 3:                     # [B, L, D]
            L = ref.shape[1]
            e = emb                              # [1, max_seq_len, D]
            if e.shape[1] >= L:
                e = e[:, :L]
            else:
                e = torch.cat(
                    [e, e.new_zeros(e.shape[0], L - e.shape[1],
                                    e.shape[2])], dim=1)
            emb = e
        if ref.shape[-1] != emb.shape[-1]:
            raise SystemExit(
                f"D mismatch: ref {ref.shape[-1]} vs encoded "
                f"{emb.shape[-1]} — wrong text encoder/model path?")
        emb = emb.to(ref.dtype)

    os.makedirs(os.path.dirname(out), exist_ok=True)
    torch.save(emb, out)
    print(f"[empty_emb] wrote {out}  shape={tuple(emb.shape)} {emb.dtype}")
    if ref is not None and tuple(emb.shape) != tuple(ref.shape):
        print(f"[empty_emb] NOTE shape {tuple(emb.shape)} != ref "
              f"{tuple(ref.shape)} (rank/L differ) — usually fine since the "
              "loader substitutes wholesale, but verify a train step.")


if __name__ == "__main__":
    main()
