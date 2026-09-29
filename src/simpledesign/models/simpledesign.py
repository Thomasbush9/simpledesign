import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from einops import rearrange
from typing import Any

from simpledesign.models.utils import apply_rope, rope_cos_sin, sinusoidal


# build the Mixture-of-Transformer (MoT) Trunk
class MLP(nn.Module):
    def __init__(self, d_in, d):
        super().__init__()
        self.norm = nn.LayerNorm(d_in)
        self.in_ = nn.Linear(d_in, d)
        self.out = nn.Linear(d, d_in)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor):
        return self.out(self.act(self.in_(self.norm(x))))


class TimeEmbedding(nn.Module):
    def __init__(self, model_d: int, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.model_d = model_d

        self.layers = nn.Sequential(
            *[
                nn.Linear(256, self.model_d),
                nn.SiLU(),
                nn.Linear(self.model_d, self.model_d),
            ]
        )

        # zero-init last layer: output is exactly 0 at init (keeps ESM2 equivalence), while the
        # last layer still gets gradients from step 1 because its input is nonzero
        nn.init.zeros_(self.layers[-1].weight)
        nn.init.zeros_(self.layers[-1].bias)

    def forward(self, t: torch.Tensor):
        """t: (B,) noise level in [0, 1] -> (B, model_d)."""
        assert t.dim() == 1, f"t must be (B,), got {tuple(t.shape)}"
        f = sinusoidal(t * 1000, 256)  # (B,)-> (B, 256)
        return self.layers(f.to(self.layers[0].weight.dtype))


class SequenceEmbedding(nn.Module):
    # ESM2 alphabet: 33 tokens, <pad> = 1, <mask> = 32
    ESM2_KEY = "esm.embeddings.word_embeddings.weight"
    MASK_ID = 32
    # ESM2 pretraining masked 15% of tokens, 80% of those with <mask>
    ESM2_MASK_RATIO_TRAIN = 0.15 * 0.8

    def __init__(
        self,
        model_d: int,
    ):
        super().__init__()
        self.embed = nn.Embedding(33, model_d, padding_idx=1)

    @torch.no_grad()
    def load_esm2(self, esm_state_dict: dict[str, torch.Tensor]) -> None:
        """Copy ESM2's token embedding from a checkpoint state dict
        (e.g. safetensors.torch.load_file("checkpoints/<esm2>/model.safetensors"))."""
        weight = esm_state_dict[self.ESM2_KEY]
        assert weight.shape == self.embed.weight.shape, (
            f"ESM2 embedding {tuple(weight.shape)} != {tuple(self.embed.weight.shape)}: "
            "model_d must match the checkpoint's hidden_size"
        )
        self.embed.weight.copy_(weight)

    def forward(self, x: torch.Tensor, seq_mask: torch.Tensor):
        """x: (B, L) token ids, seq_mask: (B, L) bool, True for real tokens (incl. <cls>/<eos>).

        ESM2 token_dropout, exactly as in pretraining (and DPLM): <mask> embeddings are zeroed
        and all embeddings are rescaled by (1 - 0.12) / (1 - r), r = fraction of real tokens
        that are <mask>. Padding is zeroed.
        """
        emb = self.embed(x)
        is_mask = x == self.MASK_ID
        emb = emb.masked_fill(is_mask[..., None], 0.0)
        ratio = is_mask.sum(-1).float() / seq_mask.sum(-1).float()  # (B,)
        scale = (1 - self.ESM2_MASK_RATIO_TRAIN) / (1 - ratio)
        return emb * scale[:, None, None].to(emb.dtype) * seq_mask[..., None].to(emb.dtype)


class StructureEmbedding(nn.Module):
    """z_x = LayerNorm(Linear(gamma(x))) with Gaussian random Fourier features
    gamma(x) = [cos(2*pi x B^T), sin(2*pi x B^T)], B ~ N(0, sigma^2) of shape (n_fourier, 3).

    sigma is in cycles per coordinate unit, so it depends on how coordinates are scaled.
    Feature wavelengths 1/|B_i| fall in a narrow band around ~0.65/sigma (5-95%: ~0.36/sigma to
    ~1.7/sigma), e.g. sigma=0.1 with coordinates in A -> ~3.6-17 A.
    """

    def __init__(self, model_d: int, *, sigma: float, n_fourier: int = 128, seed: int = 0):
        super().__init__()
        # Own generator: deterministic B without consuming the global RNG. B is a persistent
        # buffer, so checkpoints restore it exactly regardless of seed.
        gen = torch.Generator().manual_seed(seed)
        self.register_buffer("B", torch.randn(n_fourier, 3, generator=gen) * sigma)
        self.proj = nn.Linear(2 * n_fourier, model_d)
        self.norm = nn.LayerNorm(model_d)

    def forward(self, x: torch.Tensor):
        """x: (..., 3) coordinates -> (..., model_d)."""
        angles = 2 * math.pi * x.float() @ self.B.float().T  # (..., n_fourier)
        feats = torch.cat((angles.cos(), angles.sin()), dim=-1).to(self.proj.weight.dtype)
        return self.norm(self.proj(feats))


class InputEmbeddings(nn.Module):
    def __init__(self, model_d: int, esm_state_dict, sigma):
        super().__init__()
        self.model_d = model_d
        # initialize each module
        self.seq_emb = SequenceEmbedding(model_d)
        self.seq_emb.load_esm2(esm_state_dict=esm_state_dict)
        self.struct_emb = StructureEmbedding(model_d, sigma=sigma)
        # ESM2 never saw an absolute PE: zero-init gate keeps the sequence stream == ESM2 at init
        self.seq_pe_gate = nn.Parameter(torch.zeros(()))
        # noise-level conditioning, one per modality (zero-init output)
        self.time_seq = TimeEmbedding(model_d)
        self.time_struct = TimeEmbedding(model_d)

    def forward(
        self,
        seq,
        struct,
        seq_mask,
        idx,
        t,
        t_prime,
    ):
        """seq: (B, L) token ids, struct: (B, L, 3) coords, seq_mask: (B, L) bool real tokens,
        idx: (B, L) residue index (the same tensor passed to StackMoT for RoPE),
        t / t_prime: (B,) sequence / structure noise levels -> x_seq, x_struct: (B, L, model_d).
        """
        # embed sequence and structure:
        seq_embedded = self.seq_emb(seq, seq_mask)
        struct_embedded = self.struct_emb(struct)
        # now apply abs PE: one encoding of idx, shared by both streams
        pe = sinusoidal(idx, self.model_d).to(seq_embedded.dtype)  # (B, L, model_d)
        seq_embedded = seq_embedded + self.seq_pe_gate * pe
        struct_embedded = struct_embedded + pe
        # time embeddings: (B, D) broadcast over every position of the stream
        seq_embedded = seq_embedded + self.time_seq(t)[:, None, :]
        struct_embedded = struct_embedded + self.time_struct(t_prime)[:, None, :]
        return seq_embedded, struct_embedded


class MoT(nn.Module):
    def __init__(self, model_d: int, mlp_d: int, num_heads: int = 4):
        # sequence modality specific QKV
        # TODO: remove the d_seq, d_struct -> single dim for stackabl modules
        super().__init__()
        self.num_heads = num_heads
        assert model_d % num_heads == 0, (
            f"Model dimension must be divisible by number of heads {num_heads}"
        )
        self.d = model_d // num_heads

        self.seq_Q = nn.Linear(model_d, model_d)
        self.seq_K = nn.Linear(model_d, model_d)
        self.seq_V = nn.Linear(model_d, model_d)
        # structure modality specific QKV
        self.str_Q = nn.Linear(model_d, model_d)
        self.str_K = nn.Linear(model_d, model_d)
        self.str_V = nn.Linear(model_d, model_d)
        self.norm_seq = nn.LayerNorm(model_d)
        self.norm_struct = nn.LayerNorm(model_d)

        self.seq_out = nn.Linear(model_d, model_d)
        self.struct_out = nn.Linear(model_d, model_d)

        self.seq_ffn = MLP(model_d, mlp_d)
        self.struct_ffn = MLP(model_d, mlp_d)

    def forward(
        self,
        x_seq: torch.Tensor,
        x_struct: torch.Tensor,
        seq_mask: torch.Tensor,
        struct_mask: torch.Tensor,
        rope,
    ):
        assert x_seq.shape[1] == x_struct.shape[1], "Inputs must have the same length"
        # TODO: accept a single mask if the other is not specified use single one
        assert seq_mask.shape == struct_mask.shape, "Mask must have same dimensions"
        seq_len = x_seq.shape[1]
        x_seq_norm = self.norm_seq(x_seq)
        x_struct_norm = self.norm_struct(x_struct)
        seq_Q, seq_K, seq_V = self.seq_Q(x_seq_norm), self.seq_K(x_seq_norm), self.seq_V(x_seq_norm)
        str_Q, str_K, str_V = (
            self.str_Q(x_struct_norm),
            self.str_K(x_struct_norm),
            self.str_V(x_struct_norm),
        )
        # we now concatenate the Q, K and V for both modalities:
        # B, L+L, D
        joint_Q = rearrange(
            torch.cat((seq_Q, str_Q), dim=1), "b l (h d_k) -> b h l d_k", h=self.num_heads
        )
        joint_K = rearrange(
            torch.cat((seq_K, str_K), dim=1), "b l (h d_k) -> b h l d_k", h=self.num_heads
        )
        joint_V = rearrange(
            torch.cat((seq_V, str_V), dim=1), "b l (h d_k) -> b h l d_k", h=self.num_heads
        )
        cos, sin = rope
        joint_Q = apply_rope(joint_Q, cos, sin)
        joint_K = apply_rope(joint_K, cos, sin)
        # apply the mask
        attn_scores = torch.einsum("...ld, ...zd->...lz", joint_Q, joint_K) / math.sqrt(self.d)
        joint_mask = torch.cat((seq_mask, struct_mask), dim=1)
        key_mask = rearrange(joint_mask, "b j -> b 1 1 j")
        attn_scores = attn_scores.masked_fill(~key_mask, -torch.inf)
        probs = F.softmax(attn_scores, dim=-1)
        out = torch.einsum("...ij, ...jd->...id", probs, joint_V)
        # divide the modalities
        out = rearrange(out, "b h l d_k -> b l (h d_k)", h=self.num_heads)
        seq_out, struct_out = out[:, :seq_len], out[:, seq_len:]
        seq_out = self.seq_out(seq_out)
        # residual + norm after mha
        x_seq = x_seq + seq_out
        struct_out = self.struct_out(struct_out)
        x_struct = x_struct + struct_out

        # MLP -> norm is inside the mlp
        seq_out = self.seq_ffn(x_seq)
        x_seq = x_seq + seq_out
        struct_out = self.struct_ffn(x_struct)
        x_struct = x_struct + struct_out
        return x_seq, x_struct


class StackMoT(nn.Module):
    def __init__(self, n_blocks, model_d, mlp_d, num_heads, rope_base: float = 10_000.0):
        super().__init__()
        self.d_k = model_d // num_heads
        self.layers = nn.ModuleList([MoT(model_d, mlp_d, num_heads) for _ in range(n_blocks)])
        # RoPE frequencies: a buffer so they follow .to(device) and are saved in the state_dict.
        # Loading an ESM2 checkpoint overwrites them with its fp16-rounded values.
        inv_freq = rope_base ** (-torch.arange(0, self.d_k, 2, dtype=torch.float32) / self.d_k)
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, x_seq, x_struct, seq_mask, struct_mask, idx):
        pos = torch.cat((idx, idx), dim=1)  # (B, 2L): seq i and struct i share a position
        # computed once, reused by every block
        rope = rope_cos_sin(pos, self.d_k, inv_freq=self.inv_freq)
        for layer in self.layers:
            x_seq, x_struct = layer(x_seq, x_struct, seq_mask, struct_mask, rope)
        return x_seq, x_struct
