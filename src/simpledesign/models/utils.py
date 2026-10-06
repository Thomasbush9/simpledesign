import math

import torch
import torch.nn as nn
import numpy as np

SIGMA_DATA = 10.0  # Angstrom per model unit: coordinates enter the model in nm
MASK_ID = 32  # ESM2 <mask> token


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


def sample_timesteps(B: int, device: str = "cpu"):
    """Sample t and t_prime for a given batch.

    Args:
        B: int Batch size
        device: cpu, mps or cuda
    Returns:
        t: torch.Tensor [0,1] uniform sampling for sequence noise levels 0: full noise
        t_prime : torch.Tensor [0, 1] structure noise level sampled from combination of Beta (1.9, 1.0) and uniform -> to ensure more data close to 1.0
    """
    # sequence
    t = torch.rand(B, device=device)
    # structure
    use_beta = torch.rand(B, device=device) < 0.98
    beta = torch.distributions.Beta(1.9, 1.0).sample((B,)).to(device)
    uniform = torch.rand(B, device=device)
    t_prime = torch.where(use_beta, beta, uniform)
    return t, t_prime


def random_rotations(B: int, device: str | torch.device = "cpu") -> torch.Tensor:
    """(B, 3, 3) rotation matrices, uniform over SO(3) (normalized Gaussian quaternions)."""
    q = torch.randn(B, 4, device=device)
    w, x, y, z = (q / q.norm(dim=-1, keepdim=True)).unbind(-1)
    return torch.stack(
        (
            1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
            2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
            2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y),
        ),
        dim=-1,
    ).view(B, 3, 3)  # fmt: skip


def augment(coords: torch.Tensor, struct_mask: torch.Tensor, translation_std: float = 0.0):
    """Random rigid motion per protein: center on the real residues, rotate uniformly, shift by
    N(0, translation_std^2) (same unit as coords). coords (B, L, 3) -> (B, L, 3), 0 outside mask.
    """
    B = coords.shape[0]
    m = struct_mask[..., None].to(coords.dtype)
    mu = (coords * m).sum(dim=1, keepdim=True) / m.sum(dim=1, keepdim=True)
    R = random_rotations(B, coords.device)
    shift = translation_std * torch.randn(B, 1, 3, device=coords.device)
    return ((coords - mu) @ R.transpose(-1, -2) + shift) * m


def centered_noise(struct_mask: torch.Tensor) -> torch.Tensor:
    """(B, L) bool -> (B, L, 3): N(0, I) at real residues, zero mean over them, 0 elsewhere."""
    m = struct_mask[..., None].float()
    eps = torch.randn(*struct_mask.shape, 3, device=struct_mask.device) * m
    return (eps - eps.sum(dim=1, keepdim=True) / m.sum(dim=1, keepdim=True)) * m


def corrupt(batch: dict, t: torch.Tensor, t_prime: torch.Tensor):
    """
    It corrupts a single batch using the noise levels t, t_prime for sequence and structure.

    Args:
        batch: dict = ProteinCollator output (coords in Angstrom)
        t: torch.Tensor= sequence noise level (B,)
        t_prime: torch.Tensor = structure noise level (B,)
    Returns: seq_t, x_t, seq_noise_mask, v (unaligned target), epsilon, x_clean (nm)
    """
    B = batch["seq"].shape[0]
    L = batch["seq"].shape[-1]
    dev = batch["seq"].device
    seq_noise_mask = (torch.rand(B, L, device=dev) < (1 - t)[:, None]) & batch["struct_mask"]
    seq_t = batch["seq"].masked_fill(seq_noise_mask, MASK_ID)
    x_clean = batch["coords"] / SIGMA_DATA
    epsilon = centered_noise(batch["struct_mask"])
    # interpolate
    x_prime = (1 - t_prime)[:, None, None] * epsilon + t_prime[:, None, None] * x_clean
    # get velocity target:
    v = x_clean - epsilon
    return seq_t, x_prime, seq_noise_mask, v, epsilon, x_clean


def structure_rigid_alignment(coords: torch.Tensor, ref_coords: torch.Tensor, mask: torch.Tensor):
    """Kabsch: rotate + translate `coords` onto `ref_coords`, per protein.

    coords, ref_coords: (B, L, 3); mask: (B, L) bool, True at real residues.
    Returns the aligned coords (B, L, 3), zero outside the mask. Proper rotations only
    (no reflections). Wrap the call in torch.no_grad() when used inside the loss.
    """
    m = mask[..., None].to(coords.dtype)  # (B, L, 1)
    n = m.sum(dim=1, keepdim=True)  # (B, 1, 1)
    # centroids over real residues only
    mu = (coords * m).sum(dim=1, keepdim=True) / n  # (B, 1, 3)
    mu_ref = (ref_coords * m).sum(dim=1, keepdim=True) / n
    x = (coords - mu) * m
    x_ref = (ref_coords - mu_ref) * m
    # cross-covariance H = sum_l x_ref_l (outer) x_l, then SVD per protein
    H = torch.einsum("bli,blj->bij", x_ref, x)  # (B, 3, 3)
    U, S, Vh = torch.linalg.svd(H)
    # reflection fix, per protein: flip the last axis where det(U Vh) = -1
    d = torch.sign(torch.linalg.det(U @ Vh))  # (B,)
    D = torch.diag_embed(torch.stack((torch.ones_like(d), torch.ones_like(d), d), dim=-1))
    R = U @ D @ Vh  # (B, 3, 3),rotates x onto x_ref
    # coordinates are row vectors: x_aligned = x R^T, then move to the reference centroid
    return (x @ R.transpose(-1, -2) + mu_ref) * m


def ca_rmsd(coords: torch.Tensor, ref_coords: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """RMSD after Kabsch alignment of `coords` onto `ref_coords`, per protein: (B, L, 3) x2,
    mask (B, L) bool -> (B,), in the unit of the inputs."""
    aligned = structure_rigid_alignment(coords.float(), ref_coords.float(), mask)
    m = mask.float()
    se = ((aligned - ref_coords.float()) ** 2).sum(-1) * m
    return (se.sum(1) / m.sum(1)).sqrt()


def aligned_velocity_target(x_clean, eps, coords_t, v_pred, t_prime, struct_mask):
    """Rotate/translate the ground truth onto the model's predicted structure, rebuild the target.
    Alignment runs in fp32 with autocast off (SVD has no bf16 kernels)."""
    with torch.no_grad(), torch.autocast(x_clean.device.type, enabled=False):
        x1_hat = coords_t.float() + (1 - t_prime.float())[:, None, None] * v_pred.float()
        x1_aligned = structure_rigid_alignment(x_clean.float(), x1_hat, struct_mask)
    m = struct_mask[..., None].to(x1_aligned.dtype)
    return (x1_aligned - eps.float()) * m


def make_frames(
    pos_N: torch.Tensor,
    pos_Ca: torch.Tensor,
    pos_C: torch.Tensor,
):
    """
    Constructs right-handed local coordinate frames from backbone atoms.
    Inputs are tensors of shape [N, 3]
    """

    v1 = pos_N - pos_Ca  # vector from Ca to N
    v2 = pos_C - pos_Ca  # vector from Ca to C

    # normalize v1
    e1 = v1 / torch.norm(v1, dim=-1, keepdim=True)

    # create orthogonal base from e1 to v2
    c = torch.cross(e1, v2, dim=-1)
    # make it orthonormal
    e2 = c / torch.norm(c, dim=-1, keepdim=True)

    # to get the third: cross between two orthonormal -> orthonormal
    e3 = torch.cross(
        e1,
        e2,
        dim=-1,
    )

    # form N , 3, 3
    R = torch.stack([e1, e2, e3], dim=-1)
    x = pos_Ca  # translation vector
    return R, x
