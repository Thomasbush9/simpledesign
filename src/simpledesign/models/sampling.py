"""Building blocks shared by the SimpleDesign samplers (sequence, structure, joint): start tensor,
time schedules and one update step per modality. Pure tensor functions, no model calls: each
sampler loop calls the model once per step and hands its outputs to the step function(s)."""

import torch

from simpledesign.models.loss import AA_IDS
from simpledesign.models.utils import MASK_ID, centered_noise, random_rotations

CLS_ID, PAD_ID, EOS_ID = 0, 1, 2  # ESM2 special tokens


def masked_sequence(seq_mask: torch.Tensor, struct_mask: torch.Tensor) -> torch.Tensor:
    """Fully masked tokens (B, L) long: <cls> <mask> ... <mask> <eos> <pad> ...
    seq_mask & ~struct_mask is exactly <cls> (column 0) and <eos>."""
    pad = torch.full(seq_mask.shape, PAD_ID, dtype=torch.long, device=seq_mask.device)
    seq = torch.where(struct_mask, MASK_ID, torch.where(seq_mask, EOS_ID, pad))
    seq[:, 0] = CLS_ID
    return seq


def observed_fraction(seq: torch.Tensor, struct_mask: torch.Tensor) -> torch.Tensor:
    """Sequence time fed to the model (B,): fraction of residues that are not <mask>, as in
    training (each residue masked with p = 1 - t). 1 = clean sequence, 0 = fully masked."""
    return ((seq != MASK_ID) & struct_mask).sum(1) / struct_mask.sum(1).clamp(min=1)


def joint_start(
    lengths: list[int], device: str | torch.device | None = None
) -> dict[str, torch.Tensor]:
    """Build unconditional joint inputs for B proteins, L = max(lengths) + 2.

    Positive residue lengths exclude <cls>/<eos>. Returns `SimpleDesign.forward` kwargs:
    seq (B, L) long, seq_mask/struct_mask (B, L) bool, idx (B, L) long,
    coords (B, L, 3) centered Gaussian noise in nm, and t/t_prime (B,) zeros.
    Layout matches ProteinCollator, but coords are model inputs, NOT clean Angstrom data.
    """
    if not lengths or any(type(n) is not int or n <= 0 for n in lengths):
        raise ValueError("lengths must be a non-empty list of positive integers")
    B, L = len(lengths), max(lengths) + 2
    n = torch.tensor(lengths, dtype=torch.long, device=device)[:, None]
    idx = torch.arange(L, device=device).expand(B, L)
    seq_mask = idx < n + 2
    struct_mask = (idx > 0) & (idx <= n)
    return {
        "seq": masked_sequence(seq_mask, struct_mask),
        "coords": centered_noise(struct_mask),
        "seq_mask": seq_mask,
        "struct_mask": struct_mask,
        "idx": idx,
        "t": torch.zeros(B, device=device),
        "t_prime": torch.zeros(B, device=device),
    }


def struct_schedule(
    n_steps: int,
    kind: str = "linear",
    t_start: float = 0.0,
    device: str | torch.device | None = None,
) -> torch.Tensor:
    """(n_steps + 1,) structure times t' from t_start (0 = noise) towards 1 (data).

    linear: uniform steps, ends at 1.
    log (paper): s = Flip(LogSpace(-2, 0)), min-max normalized and clamped to [1e-4, 1], then
    t' = 1 - s (s is the paper's noise-side time): large steps near noise, small ones near the
    data, ends at 1 - 1e-4. With t_start > 0 the same grid is squeezed into [t_start, 1].
    """
    if kind == "linear":
        return torch.linspace(t_start, 1.0, n_steps + 1, device=device)
    if kind == "log":
        # built on CPU (no MPS kernel for logspace), n_steps + 1 values
        s = torch.logspace(-2, 0, n_steps + 1).flip(0)
        s = ((s - s.min()) / (s.max() - s.min())).clamp(1e-4, 1.0)
        return (t_start + (1 - t_start) * (1 - s)).to(device)
    raise ValueError(f"unknown schedule {kind!r}: expected 'linear' or 'log'")


def seq_step(
    logits: torch.Tensor,
    seq: torch.Tensor,
    struct_mask: torch.Tensor,
    n_unmasked: torch.Tensor,
    temperature: float | torch.Tensor,
    *,
    gumbel_sigma: float = 0.5,
    remask: bool = True,
) -> torch.Tensor:
    """One sequence update (DPLM-style decoding).

    logits: (B, L, 33) model output for `seq`; seq: (B, L) current tokens; n_unmasked: (B,)
    long, residues unmasked after this step (floor(N t_next) for the linear schedule);
    temperature: T at this step.
    Proposes a token at every residue, Cat(softmax((logits + gumbel_sigma * g) / T)) over the
    20 amino acids, g ~ Gumbel(0, 1), and ranks residues by the log-probability (under that
    softmax) of the token they would hold:
    remask=True (paper): ranking over all residues; inside the top n_unmasked, masked residues
    take their proposal and decoded ones keep their token, every other residue goes (back) to
    <mask>, so early tokens can be revised.
    remask=False: only masked residues compete, n_unmasked - (already decoded) of them take
    their proposal; decoded tokens are final.
    Special tokens and padding are never touched. -> seq (B, L)
    """
    aa = logits.float()[..., AA_IDS]
    g = -torch.empty_like(aa).exponential_().log()  # Gumbel(0, 1), no log(0)
    aa = (aa + gumbel_sigma * g) / temperature
    # proposal: choice indexes the 20 amino acids
    choice = torch.distributions.Categorical(logits=aa).sample()  # (B, L)
    proposal = choice + AA_IDS.start
    masked = (seq == MASK_ID) & struct_mask
    # token each residue would hold if selected: proposal if masked, else its current token
    new = torch.where(masked, proposal, seq) if remask else proposal
    candidates = struct_mask if remask else masked
    # its confidence; clamp only touches non-residue slots, dropped by the -inf below
    new_idx = (new - AA_IDS.start).clamp(0, aa.shape[-1] - 1)
    confidence = aa.log_softmax(-1).gather(-1, new_idx[..., None]).squeeze(-1)  # (B, L)
    confidence = confidence.masked_fill(~candidates, -torch.inf)
    k = n_unmasked if remask else n_unmasked - (struct_mask.sum(1) - masked.sum(1))
    # rank within each protein, rank 0 is the most confident
    rank = confidence.argsort(1, descending=True).argsort(1)
    selected = (rank < k[:, None]) & candidates
    if remask:
        return torch.where(struct_mask, torch.where(selected, new, MASK_ID), seq)
    return torch.where(selected, new, seq)


def struct_step(
    v: torch.Tensor,
    x: torch.Tensor,
    t: torch.Tensor,
    t_next: torch.Tensor,
    struct_mask: torch.Tensor,
    *,
    sde: bool = True,
    tau: float = 0.5,
    eta: float = 0.01,
    rotate: bool = True,
) -> torch.Tensor:
    """One structure update from t' = t to t_next. v, x: (B, L, 3) in nm, v the model velocity
    at (x, t); t, t_next: scalar tensors (any spacing, dt = t_next - t).

    sde=False: Euler ODE, x += v dt.
    sde=True: Euler-Maruyama on the Langevin-corrected SDE
        dx = [v + 1/2 w s] dt + sqrt(tau w) dW,   s = (t v - x) / (1 - t),
        w(t) = 2 (1 - t) / (t + eta)   =>   1/2 w s = (t v - x) / (t + eta).
    tau: low -> refined structures, 1 -> diverse ones.
    Always re-centered on the real residues; rotate: random global rotation (the model was
    trained on random orientations). -> x (B, L, 3), 0 outside struct_mask.
    """
    dt = t_next - t
    m = struct_mask[..., None].float()
    if sde:
        w = 2 * (1 - t) / (t + eta)
        drift = v + (t * v - x) / (t + eta)
        x = x + drift * dt + (tau * w * dt).sqrt() * centered_noise(struct_mask)
    else:
        x = x + v * dt
    x = (x - (x * m).sum(dim=1, keepdim=True) / m.sum(dim=1, keepdim=True)) * m
    if rotate:
        x = x @ random_rotations(x.shape[0], x.device).transpose(-1, -2)
    return x
