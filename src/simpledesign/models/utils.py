import math

import torch
import torch.nn as nn
import numpy as np


def sinusoidal(x: torch.Tensor, dim: int, max_period: float = 10_000.0) -> torch.Tensor:
    """Sinusoidal features of a scalar per element: x (...) -> (..., dim), computed in float32.

    Frequencies are geometric from 1 down to 1/max_period; the first half of the output is
    cos, the second half sin (odd dim: last channel is zero). Used for the absolute positional
    encoding of residue indices and for time features (scale t in [0, 1] up first, e.g. by
    1000, otherwise most frequencies barely change across t).
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(half, device=x.device, dtype=torch.float32) / half
    )
    args = x.float()[..., None] * freqs  # (..., half)
    emb = torch.cat((args.cos(), args.sin()), dim=-1)
    if dim % 2:
        emb = torch.cat((emb, torch.zeros_like(emb[..., :1])), dim=-1)
    return emb


def rope_cos_sin(
    pos: torch.Tensor, d_k: int, base: float = 10_000.0, inv_freq: torch.Tensor | None = None
):
    """pos: (B, N) positions -> cos, sin: (B, 1, N, d_k) in float32, broadcast over
    heads. inv_freq: optional (d_k/2,) frequencies, e.g. the fp16-rounded ones stored in
    ESM2 checkpoints; defaults to base^(-2i/d_k)."""
    assert d_k % 2 == 0, "RoPE needs an even head dim"
    if inv_freq is not None:
        assert inv_freq.shape == (d_k // 2,), f"inv_freq must have shape ({d_k // 2},)"
        inv_freq = inv_freq.to(device=pos.device, dtype=torch.float32)
    else:
        inv_freq = base ** (
            -torch.arange(0, d_k, 2, device=pos.device, dtype=torch.float32) / d_k
        )  # (d_k/2,)
    angles = pos.float()[..., None] * inv_freq  # (B, N, d_k/2)
    angles = torch.cat((angles, angles), dim=-1)  # (B, N, d_k), half-split layout
    return angles.cos()[:, None], angles.sin()[:, None]


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """x: (B, H, N, d_k). Rotates pairs (i, i + d_k/2)."""
    xf = x.float()
    return (xf * cos + rotate_half(xf) * sin).type_as(x)
