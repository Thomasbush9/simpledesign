import math

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import LogNorm

from simpledesign.models.loss import AA_IDS

ESM2_AAS = "LAGVSERTIDPKQNFYMHWC"  # ESM2 token ids 4..23 (AA_IDS)


def plot_history(history, smooth=1, t_bins=(0.0, 0.3, 0.7, 1.0)):
    """Plot a training `history` (list of dicts from the training loop).

    Row 1: sequence CE, structure MSE (log scale), masked-residue accuracy, grad norm.
    Row 2 (only if the noise levels vary between steps, i.e. stage 2): per-step losses binned by
    the batch-mean t (sequence) and t' (structure), showing where the model still struggles.
    smooth: moving-average window in steps (1 = raw).
    """
    steps = np.array([h["step"] for h in history])

    def series(key):
        y = np.array([h[key] for h in history], dtype=float)
        if smooth > 1 and len(y) >= smooth:
            y = np.convolve(y, np.ones(smooth) / smooth, mode="valid")
            return steps[smooth - 1 :], y
        return steps, y

    t_mean = np.array([np.mean(h["t"]) for h in history])
    tp_mean = np.array([np.mean(h["t_prime"]) for h in history])
    varying = np.ptp(tp_mean) > 0 or np.ptp(t_mean) > 0

    fig, axes = plt.subplots(
        2 if varying else 1, 4, figsize=(18, 7.5 if varying else 3.8), squeeze=False
    )
    panels = [
        ("seq", "sequence CE (masked)", True),
        ("struct", "structure MSE (nm²)", True),
        ("acc", "masked-residue accuracy", False),
        ("grad_norm", "grad norm (pre-clip)", True),
    ]
    for ax, (key, title, log) in zip(axes[0], panels):
        x, y = series(key)
        ax.plot(x, y, lw=1.2)
        ax.set_title(title)
        ax.set_xlabel("step")
        if log:
            ax.set_yscale("log")
        if key == "acc":
            ax.set_ylim(0, 1.02)
        ax.grid(alpha=0.3)

    if varying:
        edges = np.array(t_bins)
        for ax, key, tv, name in (
            (axes[1][0], "seq", t_mean, "t"),
            (axes[1][1], "struct", tp_mean, "t'"),
        ):
            y = np.array([h[key] for h in history])
            for lo, hi in zip(edges[:-1], edges[1:]):
                sel = (tv >= lo) & (tv < hi) if hi < 1 else (tv >= lo) & (tv <= hi)
                if sel.any():
                    ax.scatter(
                        steps[sel],
                        y[sel],
                        s=6,
                        alpha=0.5,
                        label=f"{name} ∈ [{lo:.1f}, {hi:.1f})  n={sel.sum()}",
                    )
            ax.set_yscale("log")
            ax.set_title(f"{key} loss per step, colored by batch-mean {name}")
            ax.set_xlabel("step")
            ax.legend(fontsize=8)
            ax.grid(alpha=0.3)
        for ax, key, tv, name in (
            (axes[1][2], "seq", t_mean, "t"),
            (axes[1][3], "struct", tp_mean, "t'"),
        ):
            last = slice(len(history) // 2, None)  # second half of training: the plateau
            ax.scatter(tv[last], np.array([h[key] for h in history])[last], s=8, alpha=0.6)
            ax.set_yscale("log")
            ax.set_title(f"{key} loss vs {name} (second half of training)")
            ax.set_xlabel(f"batch-mean {name}  (1 = clean)")
            ax.grid(alpha=0.3)

    fig.tight_layout()
    return fig


def plot_attention(attn, batch, b=0, heads=None, log_scale=True, drop_special=False, ncols=5):
    """Heatmaps of one block's joint attention for protein `b`, one panel per head.

    attn: (B, H, 2L, 2L), e.g. store[0]. Rows are queries, columns keys, laid out as
    [sequence tokens | structure residues]; padding is removed using batch["seq_mask"] and
    batch["struct_mask"]. drop_special=True also removes <cls>/<eos> from the sequence part.
    log_scale: log colors (attention is very peaked), floored at 1e-4.
    """
    a_all = attn[b].float().cpu()
    L = a_all.shape[-1] // 2
    seq_mask, struct_mask = batch["seq_mask"][b].cpu(), batch["struct_mask"][b].cpu()
    seq_keep = seq_mask.nonzero().squeeze(-1)
    if drop_special:
        seq_keep = seq_keep[struct_mask[seq_keep]]  # residues only
    struct_keep = struct_mask.nonzero().squeeze(-1) + L  # structure slots start at L
    keep = torch.cat([seq_keep, struct_keep])
    n_seq, n_struct = len(seq_keep), len(struct_keep)

    heads = list(range(a_all.shape[0])) if heads is None else list(heads)
    nrows = math.ceil(len(heads) / ncols)
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(3.0 * ncols, 3.0 * nrows), squeeze=False, constrained_layout=True
    )
    norm = LogNorm(vmin=1e-4, vmax=1.0) if log_scale else None
    for ax, h in zip(axes.flat, heads):
        a = a_all[h][keep][:, keep]
        im = ax.imshow(
            a.clamp(min=1e-4) if log_scale else a,
            cmap="viridis",
            norm=norm,
            interpolation="nearest",
        )
        ax.axhline(n_seq - 0.5, color="white", lw=0.8)  # modality boundary
        ax.axvline(n_seq - 0.5, color="white", lw=0.8)
        ticks, labels = [n_seq / 2, n_seq + n_struct / 2], ["seq", "struct"]
        ax.set_xticks(ticks, labels, fontsize=8)
        ax.set_yticks(ticks, labels, fontsize=8, rotation=90, va="center")
        ax.set_title(f"head {h}", fontsize=9)
    for ax in axes.flat[len(heads) :]:
        ax.axis("off")
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.6, label="attention weight")
    fig.suptitle(f"{batch['ids'][b]}: rows = queries, cols = keys  [seq | struct]")
    return fig


def plot_velocity(coords_t, v_pred, v_true, t_prime, struct_mask, dims=((0, 1), (0, 2), (1, 2))):
    """Predicted vs target velocity field, one row per protein view (e.g. one protein at several
    t'), one column per 2D projection of the coordinates (nm).

    Gray: input coordinates x_t'. Arrows are velocity * (1 - t'): each one ends where a single
    Euler step to t' = 1 moves the residue, green for the target, red for the prediction; the
    endpoints are joined into CA traces (the target / predicted protein).
    coords_t, v_pred, v_true: (B, L, 3); t_prime: (B,); struct_mask: (B, L) bool.
    """
    n = coords_t.shape[0]
    fig, axes = plt.subplots(
        n, len(dims), figsize=(4.5 * len(dims), 4.5 * n), squeeze=False, constrained_layout=True
    )
    for i in range(n):
        m = struct_mask[i].cpu()
        x = coords_t[i].float().cpu()[m]
        vp, vt = v_pred[i].float().cpu()[m], v_true[i].float().cpu()[m]
        dt = 1 - float(t_prime[i])
        rmse = ((vp - vt) ** 2).sum(-1).mean().sqrt()
        cos = torch.nn.functional.cosine_similarity(vp, vt, dim=-1).mean()
        for ax, (a, c) in zip(axes[i], dims, strict=True):
            ax.scatter(x[:, a], x[:, c], s=8, c="gray", alpha=0.6, label="input $x_{t'}$")
            for v, color, name in ((vt, "tab:green", "target"), (vp, "tab:red", "predicted")):
                d = v * dt
                ax.quiver(
                    x[:, a], x[:, c], d[:, a], d[:, c], color=color, alpha=0.4,
                    angles="xy", scale_units="xy", scale=1, width=0.003,
                )  # fmt: skip
                end = x + d
                ax.plot(end[:, a], end[:, c], "-o", color=color, ms=2, lw=0.8, label=name)
            ax.set_aspect("equal", adjustable="datalim")
            ax.set_xlabel(f"{'xyz'[a]} (nm)")
            ax.set_ylabel(f"{'xyz'[c]} (nm)")
            ax.grid(alpha=0.3)
        axes[i][0].set_title(f"t'={float(t_prime[i]):.2f} | v RMSE {rmse:.3f} nm | cos {cos:.2f}")
    axes[0][-1].legend(fontsize=8, loc="upper right")
    return fig


def plot_sequence(logits, seq, noise_mask, t, struct_mask):
    """Predicted amino-acid distribution per residue, one row per protein view (e.g. one protein
    at several t). Heatmap: softmax over the 20 amino acids (as in the loss). Red circles: true
    residue at masked (scored) positions, white dots: true residue at visible positions.
    logits: (B, L, 33); seq: (B, L) clean tokens; noise_mask, struct_mask: (B, L) bool; t: (B,).
    """
    n = logits.shape[0]
    fig, axes = plt.subplots(n, 1, figsize=(16, 2.8 * n), squeeze=False, constrained_layout=True)
    for i, ax in enumerate(axes[:, 0]):
        m = struct_mask[i].cpu()
        p = logits[i].float().cpu()[m][:, AA_IDS].softmax(-1).T  # (20, L)
        true = seq[i].cpu()[m] - AA_IDS.start
        masked = noise_mask[i].cpu()[m]
        valid = (true >= 0) & (true < len(ESM2_AAS))
        pos = torch.arange(len(true))
        im = ax.imshow(p, aspect="auto", cmap="viridis", vmin=0, vmax=1, interpolation="nearest")
        sel = masked & valid
        ax.scatter(pos[sel], true[sel], s=14, facecolors="none", edgecolors="red", lw=0.9)
        ax.scatter(pos[~masked & valid], true[~masked & valid], s=3, c="white")
        acc = (p.argmax(0) == true)[sel].float().mean()
        ax.set_yticks(range(len(ESM2_AAS)), list(ESM2_AAS), fontsize=6)
        ax.set_title(f"t={float(t[i]):.2f} | {int(sel.sum())} masked | masked acc {acc:.2f}")
    axes[-1, 0].set_xlabel("residue")
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.8, label="p(aa)")
    return fig
