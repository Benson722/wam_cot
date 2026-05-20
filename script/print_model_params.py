#!/usr/bin/env python
# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
"""轻量级模型参数量统计 —— 只读 safetensors 文件 metadata,不加载权重。

~100ms 完成,**不需要 torch / CUDA / 模型加载**(仅依赖 `safetensors` 包,
LingBot venv 默认有)。设计目的:在我们自己的 server wrapper(如
`script/launch_server_pred_latent.sh`)里**启动 server 之前**打印一份
"模型参数规模" 报表,供日志归档 / 报告引用 —— **完全不动 EVAL_ENV 的
`wan_va_server_predvideo.py`**(避免破坏评测环境)。

用法:
  python script/print_model_params.py --ckpt <ckpt_dir>
  python script/print_model_params.py --ckpt <ckpt_dir> --tag M1v
其中 <ckpt_dir> 含 transformer/ vae/ text_encoder/(可符号链接)子目录,
每个子目录有 `*.safetensors`。
"""
import argparse
import sys
from pathlib import Path

# kf_aux_head / stage_head 是本项目改造引入的两个辅助头,推理不调用
KF_PREFIXES = ("kf_aux_head", "stage_head")


def _count_one_file(fp):
    """返回 (total_params, {prefix: params}) — 只读 metadata,不加载张量。"""
    try:
        from safetensors import safe_open
    except ImportError as e:
        raise SystemExit(
            f"需要 safetensors: pip install safetensors ({e})")
    total = 0
    by_pref = {p: 0 for p in KF_PREFIXES}
    with safe_open(str(fp), framework="pt") as f:
        for key in f.keys():
            shape = list(f.get_slice(key).get_shape())
            n = 1
            for d in shape:
                n *= d
            total += n
            for p in KF_PREFIXES:
                if key.startswith(p + ".") or key == p:
                    by_pref[p] += n
    return total, by_pref


def count_component(d: Path):
    """目录 d 下所有 `*.safetensors` 之和。找不到返回 (None, {}, 0)。"""
    if not d.is_dir():
        return None, {}, 0
    files = sorted(d.glob("*.safetensors"))
    if not files:
        return None, {}, 0
    total = 0
    by_pref = {p: 0 for p in KF_PREFIXES}
    for fp in files:
        t, bp = _count_one_file(fp)
        total += t
        for p in KF_PREFIXES:
            by_pref[p] += bp.get(p, 0)
    return total, by_pref, len(files)


def fmt(n):
    if n is None:
        return "    --   "
    if n >= 1e9:
        return f"{n / 1e9:6.2f} B"
    if n >= 1e6:
        return f"{n / 1e6:6.1f} M"
    return f"{n / 1e3:6.1f} K"


def main():
    ap = argparse.ArgumentParser(
        description="Print model parameter counts via safetensors metadata.")
    ap.add_argument("--ckpt", required=True,
                    help="ckpt 根目录(含 transformer/ vae/ text_encoder/)")
    ap.add_argument("--tag", default="",
                    help="可选标签,如 M1 / M1v,只用于日志注释")
    args = ap.parse_args()

    ck = Path(args.ckpt)
    if not ck.is_dir():
        sys.exit(f"ERROR: not a directory: {ck}")

    p_vae, _, n_vae = count_component(ck / "vae")
    p_te,  _, n_te  = count_component(ck / "text_encoder")
    p_xf, aux, n_xf = count_component(ck / "transformer")

    p_vae = p_vae or 0
    p_te  = p_te  or 0
    p_xf  = p_xf  or 0
    p_kf  = aux.get("kf_aux_head", 0)
    p_st  = aux.get("stage_head", 0)
    p_xf_main = p_xf - p_kf - p_st
    p_total   = p_vae + p_te + p_xf

    print()
    print("================  Model Parameter Counts  ================")
    if args.tag:
        print(f"  TAG  : {args.tag}")
    print(f"  ckpt : {ck}")
    print(f"  Wan2.2 VAE                : {fmt(p_vae)}  ({p_vae:>14,d})"
          + (f"   [{n_vae} file(s)]" if n_vae else "   [N/A]"))
    print(f"  UMT5 Text Encoder         : {fmt(p_te)}  ({p_te:>14,d})"
          + (f"   [{n_te} file(s)]" if n_te else "   [N/A]"))
    print(f"  Transformer backbone      : {fmt(p_xf_main)}  ({p_xf_main:>14,d})")
    print(f"  + kf_aux_head (Latent #1) : {fmt(p_kf)}  ({p_kf:>14,d})")
    print(f"  + stage_head  (Phase B)   : {fmt(p_st)}  ({p_st:>14,d})")
    print( "  ──────────────────────────────────────────────────────────")
    print(f"  Transformer (subtotal)    : {fmt(p_xf)}  ({p_xf:>14,d})"
          + (f"   [{n_xf} file(s)]" if n_xf else "   [N/A]"))
    print(f"  TOTAL (VAE + UMT5 + Xfmr) : {fmt(p_total)}  ({p_total:>14,d})")
    print( "  (kf_aux_head / stage_head 推理时不调用;仅训练时用作辅助监督)")
    print("==========================================================")


if __name__ == "__main__":
    main()
