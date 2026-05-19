# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
"""Latent-CoT #4: probing + t-SNE (see latent_plan.md / latent_plan_progress.md).

Question: does the world-model latent encode the task *stage* (implicit
physical CoT)? Freeze the representation, train a tiny linear probe on the
keyframe-derived per-latent-frame stage label, report accuracy vs chance,
and visualize with t-SNE/PCA. Trajectory-safe split (plan risk #2: split by
episode, NOT by frame, so high acc can't come from per-episode shortcuts).

Two feature sources:
  * ``z_latent`` (DEFAULT, offline, zero GPU/model dependency, low-risk):
    the Wan-VAE latent z_t that *feeds* the world model. Spatial-mean-pooled
    per latent frame -> [C]. Runnable right now.
  * ``h_hidden`` (reserved): the transformer backbone hidden h_t
    (``forward_train`` already exposes it as ``pred[3]`` / ``kf_feat``).
    Meaningful for the plan's stock-vs-#1 comparison once a #1-trained ckpt
    exists; not implemented here on purpose (needs a model forward).

Stage labels come from the dataset's ``kf_stage`` (enabled by cfg.kf_aux;
run evaluation/robotwin/keyframe_annotate.py first). For an atomic task like
adjust_bottle there are 2 stages (pre-/post-grasp) -> chance = 50%; richer
probes need multi-stage / long-horizon tasks.

Run (lingbot env, from the LingBot repo root):
    python evaluation/robotwin/latent_probe.py \
        --config robotwin_train --num-samples 400 \
        --out-dir experiments/probing
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))


def _collect(dset, num_samples: int):
    """Iterate the dataset and pull (z_latent feature, stage, episode) per
    latent frame. Feature = spatial-mean-pooled VAE latent -> [C]."""
    feats, labels, eps = [], [], []
    n = min(len(dset), num_samples) if num_samples > 0 else len(dset)
    for i in range(n):
        item = dset[i]
        if "kf_stage" not in item:
            raise RuntimeError(
                "Dataset has no 'kf_stage' — set cfg.kf_aux=True AND run "
                "keyframe_annotate.py so meta/keyframes.jsonl exists.")
        lat = item["latents"]                    # [C, F, H, W]
        lat = np.asarray(lat.float().cpu())
        C, F = lat.shape[0], lat.shape[1]
        z = lat.reshape(C, F, -1).mean(axis=2).T  # [F, C]  spatial mean-pool
        st = np.asarray(item["kf_stage"].cpu()).reshape(-1)
        ep = int(item["kf_episode"]) if "kf_episode" in item else i
        m = min(F, st.shape[0])
        feats.append(z[:m])
        labels.append(st[:m])
        eps.append(np.full(m, ep, dtype=np.int64))
        if (i + 1) % 50 == 0:
            print(f"[probe] collected {i + 1}/{n} episodes")
    X = np.concatenate(feats, 0).astype(np.float32)
    y = np.concatenate(labels, 0).astype(np.int64)
    g = np.concatenate(eps, 0).astype(np.int64)
    return X, y, g


def _traj_split(g: np.ndarray, seed: int, val_frac: float = 0.2):
    rng = np.random.default_rng(seed)
    uniq = np.unique(g)
    rng.shuffle(uniq)
    n_val = max(1, int(round(len(uniq) * val_frac)))
    val_eps = set(uniq[:n_val].tolist())
    is_val = np.array([e in val_eps for e in g])
    return ~is_val, is_val


def _train_probe(Xtr, ytr, Xva, yva, S, epochs, device):
    import torch
    import torch.nn as nn

    mu = Xtr.mean(0, keepdims=True)
    sd = Xtr.std(0, keepdims=True) + 1e-6
    Xtr = (Xtr - mu) / sd
    Xva = (Xva - mu) / sd
    Xtr_t = torch.tensor(Xtr, device=device)
    ytr_t = torch.tensor(ytr, device=device)
    Xva_t = torch.tensor(Xva, device=device)
    probe = nn.Linear(Xtr.shape[1], S).to(device)
    opt = torch.optim.AdamW(probe.parameters(), lr=1e-3, weight_decay=1e-4)
    lossf = nn.CrossEntropyLoss()
    for ep in range(epochs):
        probe.train()
        opt.zero_grad()
        loss = lossf(probe(Xtr_t), ytr_t)
        loss.backward()
        opt.step()
    probe.eval()
    with torch.no_grad():
        pr_tr = probe(Xtr_t).argmax(1).cpu().numpy()
        pr_va = probe(Xva_t).argmax(1).cpu().numpy()
    return probe, pr_tr, pr_va, (mu, sd)


def _per_class_acc(y, pred, S):
    out = {}
    for c in range(S):
        m = y == c
        out[int(c)] = (float((pred[m] == c).mean()) if m.any() else None)
    return out


def _confusion(y, pred, S):
    cm = np.zeros((S, S), dtype=np.int64)
    for t, p in zip(y, pred):
        cm[t, p] += 1
    return cm.tolist()


def _tsne_png(X, y, path, seed):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # noqa: BLE001
        print(f"[probe] matplotlib unavailable, skip figure ({e})")
        return
    idx = np.arange(len(X))
    if len(idx) > 3000:
        idx = np.random.default_rng(seed).choice(idx, 3000, replace=False)
    Xs = X[idx]
    Xs = (Xs - Xs.mean(0)) / (Xs.std(0) + 1e-6)
    emb, method = None, ""
    try:
        from sklearn.manifold import TSNE
        emb = TSNE(n_components=2, perplexity=30,
                   init="pca", random_state=seed).fit_transform(Xs)
        method = "t-SNE"
    except Exception as e:  # noqa: BLE001
        print(f"[probe] sklearn TSNE unavailable ({e}); PCA-2D fallback")
        import torch
        t = torch.tensor(Xs, dtype=torch.float32)
        t = t - t.mean(0, keepdim=True)
        _, _, V = torch.pca_lowrank(t, q=2)
        emb = (t @ V[:, :2]).numpy()
        method = "PCA"
    plt.figure(figsize=(7, 6))
    sc = plt.scatter(emb[:, 0], emb[:, 1], c=y[idx], s=6,
                     cmap="tab10", alpha=0.6)
    plt.colorbar(sc, label="stage_idx")
    plt.title(f"latent {method} colored by stage  (n={len(idx)})")
    plt.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=140)
    plt.close()
    print(f"[probe] wrote {path}")


def main():
    ap = argparse.ArgumentParser(description="Latent-CoT #4 probing + t-SNE")
    ap.add_argument("--config", default="robotwin_train",
                    help="VA_CONFIGS key (dataset + kf settings)")
    ap.add_argument("--features", choices=["z_latent", "h_hidden"],
                    default="z_latent")
    ap.add_argument("--num-samples", type=int, default=400,
                    help="cap on dataset episodes scanned (0 = all)")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out-dir", default="experiments/probing")
    args = ap.parse_args()

    if args.features == "h_hidden":
        raise NotImplementedError(
            "h_hidden probe needs a model forward; forward_train already "
            "exposes it as pred[3] (kf_feat). Implement once a #1-trained "
            "ckpt exists (stock-vs-#1 comparison). Use --features z_latent "
            "for the offline baseline now.")

    from wan_va.configs import VA_CONFIGS
    from wan_va.dataset.lerobot_latent_dataset import MultiLatentLeRobotDataset

    cfg = VA_CONFIGS[args.config]
    cfg.kf_aux = True  # force the loader to emit kf_stage/kf_episode
    print(f"[probe] config={args.config} dataset={cfg.dataset_path}")
    dset = MultiLatentLeRobotDataset(config=cfg)
    print(f"[probe] dataset size = {len(dset)}")

    X, y, g = _collect(dset, args.num_samples)
    S = int(y.max()) + 1
    chance = 1.0 / S
    tr, va = _traj_split(g, args.seed)
    print(f"[probe] N={len(X)} feat_dim={X.shape[1]} stages={S} "
          f"chance={chance:.3f} | train={tr.sum()} val={va.sum()} "
          f"(episodes train/val = "
          f"{len(np.unique(g[tr]))}/{len(np.unique(g[va]))})")

    probe, pr_tr, pr_va, _ = _train_probe(
        X[tr], y[tr], X[va], y[va], S, args.epochs, args.device)
    tr_acc = float((pr_tr == y[tr]).mean())
    va_acc = float((pr_va == y[va]).mean())
    res = {
        "config": args.config,
        "feature_type": args.features,
        "dataset_path": str(cfg.dataset_path),
        "n_samples": int(len(X)),
        "n_episodes": int(len(np.unique(g))),
        "feature_dim": int(X.shape[1]),
        "num_stages": S,
        "chance_acc": chance,
        "train_acc": tr_acc,
        "val_acc": va_acc,
        "val_acc_above_chance": va_acc - chance,
        "val_per_class_acc": _per_class_acc(y[va], pr_va, S),
        "val_confusion": _confusion(y[va], pr_va, S),
    }
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tag = f"{args.config}_{args.features}"
    with open(out / f"results_{tag}.json", "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    try:
        import torch
        torch.save(probe.state_dict(), out / f"probe_{tag}.pt")
    except Exception:  # noqa: BLE001
        pass
    _tsne_png(X[va], y[va], str(out / f"tsne_{tag}.png"), args.seed)

    print(json.dumps(res, indent=2, ensure_ascii=False))
    print(f"[probe] val_acc={va_acc:.3f}  chance={chance:.3f}  "
          f"(+{va_acc - chance:+.3f})  -> wrote {out}/results_{tag}.json")


if __name__ == "__main__":
    main()
