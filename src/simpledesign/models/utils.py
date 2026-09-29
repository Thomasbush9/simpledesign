import torch
import torch.nn as nn
import numpy as np


def rope_cos_sin(pos: torch.Tensor, d_k: int, base: float = 10_000.0):
    """pos: (B, N) positions -> cos, sin: (B, 1, N, d_k) in float32, broadcast over
heads."""
    assert d_k % 2 == 0, "RoPE needs an even head dim"
    inv_freq = base ** (-torch.arange(0, d_k, 2, device=pos.device, dtype=torch.float32) /
d_k)  # (d_k/2,)
    angles = pos.float()[..., None] * inv_freq          # (B, N, d_k/2)
    angles = torch.cat((angles, angles), dim=-1)         # (B, N, d_k), half-split layout
    return angles.cos()[:, None], angles.sin()[:, None]

def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)

def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """x: (B, H, N, d_k). Rotates pairs (i, i + d_k/2)."""
    xf = x.float()
    return (xf * cos + rotate_half(xf) * sin).type_as(x)
