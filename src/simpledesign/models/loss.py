import torch
import torch.nn.functional as F

AA_IDS = slice(4, 24)


def joint_loss(
    logits,  # (B, L, 33) sequence head output
    seq,  # (B, L) long, CLEAN tokens (batch["seq"])
    noise_mask,  # (B, L) bool, residues replaced by <mask>
    v_pred,  # (B, L, 3) predicted velocity (nm)
    v_true,  # (B, L, 3) target velocity x1 - x0 (nm)
    struct_mask,  # (B, L) bool, real residues
    t,  # (B,) sequence noise level; sequence weight t downweights heavily masked sequences
    lambda_seq=1.0,
    lambda_struct=1.0,
    aa_only=True,  # restrict the softmax to the 20 amino acids
    *,
    beta_fape=0.0,
    x_pred=None,  # (B, L, 3) predicted clean CA coordinates (nm)
    x_true=None,  # (B, L, 3) clean target CA coordinates (nm)
):
    """Return weighted CE + velocity MSE + CA FAPE and detached component scalars."""
    if beta_fape < 0:
        raise ValueError("beta_fape must be nonnegative")
    if beta_fape > 0 and (x_pred is None or x_true is None):
        raise ValueError("x_pred and x_true are required when beta_fape > 0")

    # --- sequence: t * mean CE over masked tokens, per protein, then batch mean
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

    fape_obj = fape(x_pred, x_true, struct_mask) if beta_fape > 0 else struct_obj.new_zeros(())
    total = lambda_seq * seq_obj + lambda_struct * struct_obj + beta_fape * fape_obj
    return total, {
        "seq": seq_obj.detach(), "struct": struct_obj.detach(), "fape": fape_obj.detach()
    }


def masked_accuracy(logits, seq, noise_mask):
    pred = logits[..., AA_IDS].argmax(-1) + AA_IDS.start
    return (pred == seq)[noise_mask].float().mean()


def fape(
    x_pred: torch.Tensor,
    x_true: torch.Tensor,
    struct_mask: torch.Tensor,
    *,
    clamp_distance: float = 1.0,
    length_scale: float = 1.0,
) -> torch.Tensor:
    """CA frame-aligned point error in nm, averaged per protein then across the batch.

    Inputs are (B, L, 3) coordinates and a (B, L) residue mask. Each consecutive
    triple defines a right-handed frame centered on its middle CA. Frames crossing
    masked residues or degenerate in the target are excluded; degenerate predicted
    frames remain in the loss. Every valid CA is compared in every valid frame.
    Proteins with no valid frames contribute differentiable zero to the batch mean.

    Independent proper rigid motions leave the loss unchanged; reflections do not.
    Distances are capped at clamp_distance and divided by length_scale (both nm).
    """
    if clamp_distance <= 0 or length_scale <= 0:
        raise ValueError("clamp_distance and length_scale must be positive")

    with torch.autocast(device_type=x_pred.device.type, enabled=False):
        # Sanitize padding before frame construction/projection: even NaN padding
        # must neither contaminate valid points nor their backward gradients.
        pred = x_pred.float().masked_fill(~struct_mask[..., None], 0.0)
        true = x_true.float().masked_fill(~struct_mask[..., None], 0.0)
        if pred.shape[1] < 3:
            return pred.sum() * 0.0

        eps = 1e-4  # nm; finite derivatives even for collapsed predicted frames

        def frames(coords):
            origin = coords[:, 1:-1]
            first = coords[:, :-2] - origin
            second = coords[:, 2:] - origin
            first_norm = torch.linalg.vector_norm(first, dim=-1, keepdim=True)
            e1 = first / first_norm.clamp_min(eps)
            normal = torch.cross(e1, second, dim=-1)
            normal_norm = torch.linalg.vector_norm(normal, dim=-1, keepdim=True)
            e3 = normal / normal_norm.clamp_min(eps)
            e2 = torch.cross(e3, e1, dim=-1)
            rotation = torch.stack((e1, e2, e3), dim=-1)
            return rotation, origin, first_norm[..., 0], normal_norm[..., 0]

        pred_frames, pred_origins, _, _ = frames(pred)
        true_frames, true_origins, first_norm, normal_norm = frames(true)
        target_valid = (first_norm >= eps) & (normal_norm >= eps)
        frame_mask = struct_mask[:, :-2] & struct_mask[:, 1:-1] & struct_mask[:, 2:]
        frame_mask = frame_mask & target_valid

        # Frame axes are columns: local_j = sum_i R_ij * (point_i - origin_i).
        # Project points/origins separately to avoid a second (B, F, L, 3)
        # displacement tensor for each structure.
        pred_local = torch.einsum("bfij,bpi->bfpj", pred_frames, pred)
        pred_local = pred_local - torch.einsum(
            "bfij,bfi->bfj", pred_frames, pred_origins
        )[:, :, None]
        true_local = torch.einsum("bfij,bpi->bfpj", true_frames, true)
        true_local = true_local - torch.einsum(
            "bfij,bfi->bfj", true_frames, true_origins
        )[:, :, None]
        error = torch.linalg.vector_norm(pred_local - true_local, dim=-1)
        error = error.clamp_max(clamp_distance) / length_scale
        error = error.masked_fill(~frame_mask[:, :, None], 0.0)
        error = error.masked_fill(~struct_mask[:, None, :], 0.0)
        pair_count = frame_mask.sum(1) * struct_mask.sum(1)
        return (error.sum(dim=(1, 2)) / pair_count.clamp_min(1)).mean()
