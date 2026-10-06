import torch
import torch.nn.functional as F
from simpledesign.models.utils import make_frames

AA_IDS = slice(4, 24)


def joint_loss(
    logits,  # (B, L, 33) sequence head output
    seq,  # (B, L) long, CLEAN tokens (batch["seq"])
    noise_mask,  # (B, L) bool, residues replaced by <mask>
    v_pred,  # (B, L, 3) predicted velocity (nm)
    v_true,  # (B, L, 3) target velocity x1 - x0 (nm)
    struct_mask,  # (B, L) bool, real residues
    t,  # (B,) sequence noise level; beta(t) = t downweights heavily masked sequences
    lambda_seq=1.0,
    lambda_struct=1.0,
    aa_only=True,  # restrict the softmax to the 20 amino acids
):
    """lambda_seq * L_CE + lambda_struct * L_MSE. Returns (total, {"seq", "struct"}
    detached)."""
    # --- sequence: beta(t) * mean CE over masked tokens, per protein, then batch mean
    target = seq
    if aa_only:
        # slice instead of -inf filling the other 13 logits: same softmax, no (B, L, 33) copy;
        # masked residues that are not one of the 20 (e.g. X) are left out of the loss
        logits = logits[..., AA_IDS]
        target = seq - AA_IDS.start
        noise_mask = noise_mask & (target >= 0) & (target < logits.shape[-1])
    target = target.masked_fill(~noise_mask, -100)
    # ignore_index (-100) gives 0 CE at unmasked positions
    ce = F.cross_entropy(logits.float().transpose(1, 2), target, reduction="none")  # (B,L)
    seq_per_protein = ce.sum(1) / noise_mask.sum(1).clamp(min=1)  # (B,)
    seq_obj = (t * seq_per_protein).mean()

    # --- structure: squared L2 per residue, mean over real residues, per protein, then batch mean
    se = ((v_pred.float() - v_true.float()) ** 2).sum(-1)  # (B,L)
    struct_obj = ((se * struct_mask).sum(1) / struct_mask.sum(1).clamp(min=1)).mean()

    total = lambda_seq * seq_obj + lambda_struct * struct_obj
    return total, {"seq": seq_obj.detach(), "struct": struct_obj.detach()}


def masked_accuracy(logits, seq, noise_mask):
    pred = logits[..., AA_IDS].argmax(-1) + AA_IDS.start
    return (pred == seq)[noise_mask].float().mean()


# FAPE loss if needed:


def fape(
    x_pred: torch.Tensor,
    R_pred: torch.Tensor,
    p_pred: torch.Tensor,
    x_true: torch.Tensor,
    R_true: torch.Tensor,
    p_true: torch.Tensor,
    Z=10.0,
    D_c=10.0,
    eps=1e-8,
):
    """
    Computes the Frame-Aligned Point Error (FAPE) loss.

    Args:
        x_pred: [N, 3] Predicted frame translations (e.g., Ca positions)
        R_pred: [N, 3, 3] Predicted frame rotation matrices
        p_pred: [N, 3] Predicted atom positions to evaluate (can be same as x_pred)

        x_true: [N, 3] Ground-truth frame translations
        R_true: [N, 3, 3] Ground-truth frame rotation matrices
        p_true: [N, 3] Ground-truth atom positions to evaluate

        Z: Clamping threshold (Angstroms)
        D_c: Normalizing scale factor (Angstroms)
    """
    # 1. Project predicted points into predicted local frames
    # Equation: R^T * (p - x)
    # [N, 1, 3] - [1, N, 3] -> [N, N, 3] (rel_pos[i, j] is vector from frame i to point j)
    rel_p_pred = p_pred.unsqueeze(0) - x_pred.unsqueeze(1)
    # Rotate using predicted R (transpose of R is its inverse)
    # [N, N, 3] @ [N, 3, 3] -> [N, N, 3]
    d_pred = torch.einsum("nij,nki->nki", R_pred, rel_p_pred)

    # 2. Project true points into true local frames
    rel_p_true = p_true.unsqueeze(0) - x_true.unsqueeze(1)
    d_true = torch.einsum("nij,nki->nki", R_true, rel_p_true)

    # 3. Compute Euclidean distance between local positions
    # Added eps for numerical stability during backpropagationsqrt
    dist = torch.sqrt(torch.sum((d_pred - d_true) ** 2, dim=-1) + eps)

    # 4. Clamp the loss at threshold Z
    clamped_dist = torch.clamp(dist, max=Z)

    # 5. Average and normalize
    fape = torch.mean(clamped_dist) / D_c
    return fape
