import json
import math
import numbers
import re
from pathlib import Path
from typing import Any, List, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from einops import rearrange
from safetensors.torch import load_file

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
    def __init__(self, model_d: int, *args: Any, zero_init: bool = True, **kwargs: Any) -> None:
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
        # last layer still gets gradients from step 1 because its input is nonzero.
        # zero_init=False when the output feeds a zero-init layer itself (e.g. AdaLN modulation),
        # otherwise both start at zero and neither gets a gradient through that path at step 0.
        if zero_init:
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
    def __init__(self, model_d: int, sigma: float):
        super().__init__()
        self.model_d = model_d
        # initialize each module
        self.seq_emb = SequenceEmbedding(model_d)
        self.struct_emb = StructureEmbedding(model_d, sigma=sigma)
        # ESM2 never saw an absolute PE: zero-init gate keeps the sequence stream == ESM2 at init
        self.seq_pe_gate = nn.Parameter(torch.zeros(()))
        # noise-level conditioning, one per modality (zero-init output)
        self.time_seq = TimeEmbedding(model_d)
        self.time_struct = TimeEmbedding(model_d)

    def load_esm2(self, esm_state_dict: dict[str, torch.Tensor]) -> None:
        """Only the token embedding is pretrained; structure, PE gate and time stay as init."""
        self.seq_emb.load_esm2(esm_state_dict)

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


class SequenceOutputHead(nn.Module):
    """ESM2 LM-head layout: dense -> GELU -> LayerNorm -> decoder + bias -> (B, L, 33) logits.

    The decoder weight is *tied* to the input token embedding: `self.decoder.weight` is the very
    same nn.Parameter object as `SequenceEmbedding.embed.weight` (shape (33, model_d)), so there
    is one tensor, updated by gradients from both uses. The bias stays a separate parameter, as
    in ESM2. Logits cover all 33 ESM2 tokens; the 20 amino acids are ids 4-23, so mask the rest
    in the loss/sampler. Softmax is left to the loss (F.cross_entropy).
    """

    ESM2_PREFIX = "lm_head."

    def __init__(self, model_d: int, tied_weight: nn.Parameter):
        super().__init__()
        vocab_size, d = tied_weight.shape
        assert d == model_d, f"tied weight has width {d}, head has model_d={model_d}"
        self.dense = nn.Linear(model_d, model_d)
        self.act = nn.GELU()
        self.norm = nn.LayerNorm(model_d)
        self.decoder = nn.Linear(model_d, vocab_size, bias=False)
        self.decoder.weight = tied_weight  # tie: share the Parameter, don't copy it
        self.bias = nn.Parameter(torch.zeros(vocab_size))

    @torch.no_grad()
    def load_esm2(self, esm_state_dict: dict[str, torch.Tensor]) -> None:
        """Copy ESM2's LM head (dense, layer_norm, bias). decoder.weight is not in the
        checkpoint: ESM2 ties it to the token embedding, as here."""
        p = self.ESM2_PREFIX
        self.dense.weight.copy_(esm_state_dict[p + "dense.weight"])
        self.dense.bias.copy_(esm_state_dict[p + "dense.bias"])
        self.norm.weight.copy_(esm_state_dict[p + "layer_norm.weight"])
        self.norm.bias.copy_(esm_state_dict[p + "layer_norm.bias"])
        self.bias.copy_(esm_state_dict[p + "bias"])

    def forward(self, x_seq: torch.Tensor):
        return self.decoder(self.norm(self.act(self.dense(x_seq)))) + self.bias


class AdaLN(nn.Module):
    """Adaptive LayerNorm (DiT): LayerNorm without learned affine, whose scale and shift are
    predicted per sample from a conditioning vector c (e.g. an embedding of t'):
        shift, scale = Linear(SiLU(c)).chunk(2)
        out = LN(x) * (1 + scale) + shift
    The modulation Linear is zero-initialized, so at init this is a plain LN (no affine).
    """

    def __init__(self, model_d: int, cond_d: int, eps: float = 1e-5):
        super().__init__()
        self.norm = nn.LayerNorm(model_d, elementwise_affine=False, eps=eps)
        self.modulation = nn.Sequential(nn.SiLU(), nn.Linear(cond_d, 2 * model_d))
        nn.init.zeros_(self.modulation[-1].weight)
        nn.init.zeros_(self.modulation[-1].bias)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """x: (B, L, model_d), c: (B, cond_d) -> (B, L, model_d)."""
        shift, scale = self.modulation(c).chunk(2, dim=-1)  # (B, model_d) each
        return self.norm(x) * (1 + scale[:, None, :]) + shift[:, None, :]


class AdaNorm(nn.Module):
    def __init__(
        self,
        normalized_shape: Union[int, List[int,]],
        t: float = 0.1,
        eps: float = 1e-5,
        bias: bool = False,
    ) -> None:
        super(AdaNorm, self).__init__()

        # this handle if the shape is a single dim
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.t = t
        self.eps = eps
        self.weight = nn.Parameter(torch.empty(self.normalized_shape))
        if bias:
            self.bias = nn.Parameter(torch.empty(self.normalized_shape))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(
        self,
    ) -> None:
        init.ones_(self.weight)
        if self.bias is not None:
            init.zeros_(self.bias)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        mean = torch.mean(input, dim=-1, keepdim=True)
        var = (input - mean).pow(2).mean(dim=-1, keepdim=True) + self.eps

        input_norm = (input - mean) * torch.rsqrt(var)
        adanorm = self.weight * (1 - self.t * input_norm) * input_norm

        if self.bias is not None:
            adanorm = adanorm + self.bias
        return adanorm


class StructureOutputHead(nn.Module):
    """Predicts the structure velocity field per position: (B, L, model_in) -> (B, L, 3).

    norm="adaln": LN modulated by t' (per sample) via AdaLN, conditioned on the head's own time
    embedding of t'. norm="adanorm": your AdaNorm (constant k, ignores t'), for comparison.
    The output layer is zero-initialized: predicted velocity is 0 at init.
    """

    def __init__(self, model_in: int, model_d: int, norm: str = "adaln"):
        super().__init__()
        assert norm in ("adaln", "adanorm"), f"unknown norm {norm!r}"
        self.norm_type = norm
        self.input = nn.Linear(model_in, model_d)
        self.act = nn.ReLU()
        if norm == "adaln":
            # normal init (not zero): AdaLN's modulation is already zero-init
            self.time = TimeEmbedding(model_d, zero_init=False)
            self.norm = AdaLN(model_d, cond_d=model_d)
        else:
            self.norm = AdaNorm(model_d)
        self.output = nn.Linear(model_d, 3)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, x_struct: torch.Tensor, t_prime: torch.Tensor):
        """x_struct: (B, L, model_in), t_prime: (B,) -> velocity (B, L, 3)."""
        h = self.input(x_struct)
        if self.norm_type == "adaln":
            h = self.norm(h, self.time(t_prime))
        else:
            h = self.norm(h)
        return self.output(self.act(h))


class StackMoT(nn.Module):
    def __init__(self, n_blocks, model_d, mlp_d, num_heads, rope_base: float = 10_000.0):
        super().__init__()
        self.d_k = model_d // num_heads
        self.layers = nn.ModuleList([MoT(model_d, mlp_d, num_heads) for _ in range(n_blocks)])
        # RoPE frequencies: a buffer so they follow .to(device) and are saved in the state_dict.
        # Loading an ESM2 checkpoint overwrites them with its fp16-rounded values.
        inv_freq = rope_base ** (-torch.arange(0, self.d_k, 2, dtype=torch.float32) / self.d_k)
        self.register_buffer("inv_freq", inv_freq)

    # ESM2 layer submodule -> sequence-side submodule of MoT block i
    ESM2_LAYER_MAP = {
        "attention.LayerNorm": "norm_seq",
        "attention.self.query": "seq_Q",
        "attention.self.key": "seq_K",
        "attention.self.value": "seq_V",
        "attention.output.dense": "seq_out",
        "LayerNorm": "seq_ffn.norm",
        "intermediate.dense": "seq_ffn.in_",
        "output.dense": "seq_ffn.out",
    }
    STRUCT_SIDE = re.compile(r"\.(norm_struct|str_[QKV]|struct_out|struct_ffn)\.")

    @torch.no_grad()
    def load_esm2(self, esm_state_dict: dict[str, torch.Tensor]) -> None:
        """Load ESM2's encoder layers into the sequence side of every block, plus its RoPE
        frequencies (fp16-rounded in the checkpoint). Structure-side parameters keep their init.
        Fails on shape mismatch (e.g. mlp_d), extra checkpoint layers, or any sequence-side
        parameter left unloaded (checkpoint with fewer layers than n_blocks)."""
        layer_key = re.compile(r"esm\.encoder\.layer\.(\d+)\.(.+)\.(weight|bias)")
        mapped = {}
        for k, v in esm_state_dict.items():
            r = layer_key.fullmatch(k)
            if r and r[2] in self.ESM2_LAYER_MAP:
                mapped[f"layers.{r[1]}.{self.ESM2_LAYER_MAP[r[2]]}.{r[3]}"] = v
        inv_freq = [v for k, v in esm_state_dict.items() if k.endswith("rotary_embeddings.inv_freq")]
        assert inv_freq, "no rotary inv_freq in the ESM2 checkpoint"
        mapped["inv_freq"] = inv_freq[0]
        missing, unexpected = self.load_state_dict(mapped, strict=False)
        assert not unexpected, f"ESM2 checkpoint has layers this trunk lacks: {unexpected[:3]}"
        unloaded = [k for k in missing if not self.STRUCT_SIDE.search(k)]
        assert not unloaded, f"sequence-side parameters not in the checkpoint: {unloaded[:3]}"

    def forward(self, x_seq, x_struct, seq_mask, struct_mask, idx):
        pos = torch.cat((idx, idx), dim=1)  # (B, 2L): seq i and struct i share a position
        # computed once, reused by every block
        rope = rope_cos_sin(pos, self.d_k, inv_freq=self.inv_freq)
        for layer in self.layers:
            x_seq, x_struct = layer(x_seq, x_struct, seq_mask, struct_mask, rope)
        return x_seq, x_struct


class SimpleDesign(nn.Module):
    """InputEmbeddings -> StackMoT trunk -> final LN per modality -> heads.

    Returns sequence logits (B, L, 33) over the ESM2 alphabet and the structure velocity field
    (B, L, 3). Build from scratch with the constructor, or ESM2-initialized with `from_esm2`.
    """

    ESM2_FINAL_LN = "esm.encoder.emb_layer_norm_after."

    def __init__(
        self,
        model_d: int,
        n_blocks: int,
        num_heads: int,
        *,
        sigma: float,
        mlp_d: int | None = None,
        struct_norm: str = "adaln",
    ):
        super().__init__()
        if mlp_d is None:
            mlp_d = 4 * model_d  # ESM2 uses 4 * hidden_size
        self.input_embeddings = InputEmbeddings(model_d, sigma)
        self.trunk = StackMoT(n_blocks, model_d, mlp_d, num_heads)
        # pre-norm trunk: the residual stream is never normalized, so each modality gets a final
        # LN before its head (the sequence one is ESM2's emb_layer_norm_after)
        self.final_ln_seq = nn.LayerNorm(model_d)
        self.final_ln_struct = nn.LayerNorm(model_d)
        self.seq_head = SequenceOutputHead(model_d, self.input_embeddings.seq_emb.embed.weight)
        self.struct_head = StructureOutputHead(model_d, model_d, norm=struct_norm)

    @torch.no_grad()
    def load_esm2(self, esm_state_dict: dict[str, torch.Tensor]) -> None:
        """Initialize every sequence-side module from an ESM2 checkpoint state dict: token
        embedding (tied to the LM-head decoder), trunk layers + RoPE frequencies, final LN,
        LM head. Structure-side modules keep their init."""
        self.input_embeddings.load_esm2(esm_state_dict)
        self.trunk.load_esm2(esm_state_dict)
        self.final_ln_seq.weight.copy_(esm_state_dict[self.ESM2_FINAL_LN + "weight"])
        self.final_ln_seq.bias.copy_(esm_state_dict[self.ESM2_FINAL_LN + "bias"])
        self.seq_head.load_esm2(esm_state_dict)

    @classmethod
    def from_esm2(cls, ckpt_dir: str | Path, *, sigma: float, struct_norm: str = "adaln"):
        """Size the model from the checkpoint's config.json and load its weights, e.g.
        SimpleDesign.from_esm2("checkpoints/esm2_t6_8M_UR50D", sigma=0.1)."""
        ckpt_dir = Path(ckpt_dir)
        cfg = json.loads((ckpt_dir / "config.json").read_text())
        assert cfg["position_embedding_type"] == "rotary" and cfg["token_dropout"], (
            "SimpleDesign assumes an ESM2 checkpoint (rotary positions, token_dropout)"
        )
        model = cls(
            model_d=cfg["hidden_size"],
            n_blocks=cfg["num_hidden_layers"],
            num_heads=cfg["num_attention_heads"],
            mlp_d=cfg["intermediate_size"],
            sigma=sigma,
            struct_norm=struct_norm,
        )
        model.load_esm2(load_file(ckpt_dir / "model.safetensors"))
        return model

    def forward(
        self,
        seq: torch.Tensor,
        coords: torch.Tensor,
        seq_mask: torch.Tensor,
        struct_mask: torch.Tensor,
        idx: torch.Tensor,
        t: torch.Tensor,
        t_prime: torch.Tensor,
    ):
        """
        seq:         (B, L) long   ESM2 token ids incl. <cls>/<eos>, <mask> where corrupted
        coords:      (B, L, 3)     noisy coordinates at t' (any value at <cls>/<eos>/pad slots)
        seq_mask:    (B, L) bool   real tokens, incl. <cls>/<eos>
        struct_mask: (B, L) bool   real residues only (no <cls>/<eos>/pad)
        idx:         (B, L) long   residue index, shared by both modalities (PE + RoPE)
        t, t_prime:  (B,) float    sequence / structure noise levels
        -> logits (B, L, 33), velocity (B, L, 3)
        """
        x_seq, x_struct = self.input_embeddings(seq, coords, seq_mask, idx, t, t_prime)
        x_seq, x_struct = self.trunk(x_seq, x_struct, seq_mask, struct_mask, idx)
        logits = self.seq_head(self.final_ln_seq(x_seq))
        velocity = self.struct_head(self.final_ln_struct(x_struct), t_prime)
        return logits, velocity
