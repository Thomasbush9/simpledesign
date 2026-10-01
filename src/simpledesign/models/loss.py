import torch.nn.functional as F

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
