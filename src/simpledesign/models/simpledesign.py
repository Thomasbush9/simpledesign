import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from einops import rearrange

from simpledesign.models.utils import apply_rope, rope_cos_sin


# build the Mixture-of-Transformer (MoT) Trunk
class MLP(nn.Module):
    def __init__(self, d_in, d):
        super().__init__()
        self.norm = nn.LayerNorm(d_in)
        self.in_ = nn.Linear(d_in, d)
        self.out = nn.Linear(d, d_in)
        self.act = nn.GELU()
    def forward(self, x:torch.Tensor):
        return self.out(self.act(self.in_(self.norm(x))))      
        
class MoT(nn.Module):
    def __init__(self, model_d:int, mlp_d:int, num_heads:int=4):
        # sequence modality specific QKV
        #TODO: remove the d_seq, d_struct -> single dim for stackabl modules
        super().__init__()
        self.num_heads=num_heads
        assert model_d % num_heads==0, f"Model dimension must be divisible by number of heads {num_heads}"
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

    def forward(self, x_seq:torch.Tensor, x_struct:torch.Tensor, mask:torch.Tensor, rope):
        assert x_seq.shape[1] == x_struct.shape[1], "Inputs must have the same length"
        seq_len = x_seq.shape[1]
        #TODO: fix this to not overwrite the residual stream
        x_seq_norm = self.norm_seq(x_seq)
        x_struct_norm = self.norm_struct(x_struct)
        seq_Q, seq_K, seq_V = self.seq_Q(x_seq_norm), self.seq_K(x_seq_norm), self.seq_V(x_seq_norm)
        str_Q, str_K, str_V = self.str_Q(x_struct_norm), self.str_K(x_struct_norm), self.str_V(x_struct_norm)
        # we now concatenate the Q, K and V for both modalities: 
        # B, L+L, D
        joint_Q = rearrange(torch.cat((seq_Q, str_Q), dim=1), "b l (h d_k) -> b h l d_k", h=self.num_heads)
        joint_K = rearrange(torch.cat((seq_K, str_K), dim=1), "b l (h d_k) -> b h l d_k", h=self.num_heads)
        joint_V = rearrange(torch.cat((seq_V, str_V), dim=1), "b l (h d_k) -> b h l d_k", h=self.num_heads)
        cos, sin = rope
        joint_Q = apply_rope(joint_Q, cos, sin)
        joint_K = apply_rope(joint_K, cos, sin)
        # apply the mask 
        attn_scores = torch.einsum("...ld, ...zd->...lz", joint_Q, joint_K) / math.sqrt(self.d)
        joint_mask = torch.cat((mask, mask), dim=1) # B, 2L
        key_mask = rearrange(joint_mask, "b j -> b 1 1 j")
        attn_scores = attn_scores.masked_fill(~key_mask, -torch.inf)
        probs = F.softmax(attn_scores, dim=-1)
        out = torch.einsum("...ij, ...jd->...id", probs, joint_V)
        # divide the modalities 
        out = rearrange(out, "b h l d_k -> b l (h d_k)", h=self.num_heads)
        seq_out, struct_out = out[:, :seq_len], out[:, seq_len:]
        seq_out = self.seq_out(seq_out)
        #residual + norm after mha
        x_seq = x_seq + seq_out
        struct_out = self.struct_out(struct_out)
        x_struct = x_struct + struct_out
        
        #MLP -> norm is inside the mlp
        seq_out = self.seq_ffn(x_seq)
        x_seq = x_seq + seq_out
        struct_out = self.struct_ffn(x_struct)
        x_struct = x_struct + struct_out
        return x_seq, x_struct


class StackMoT(nn.Module):
    def __init__(self, n_blocks, model_d, mlp_d, num_heads):
        super().__init__()
        self.d_k = model_d // num_heads
        self.layers = nn.ModuleList([MoT(model_d, mlp_d, num_heads) for _ in range(n_blocks)])

    def forward(self, x_seq, x_struct, mask, idx):
        pos = torch.cat((idx, idx), dim=1)            # (B, 2L): seq i and struct i share a position
        rope = rope_cos_sin(pos, self.d_k)            # computed once, reused by every block
        for layer in self.layers:
            x_seq, x_struct = layer(x_seq, x_struct, mask, rope)
        return x_seq, x_struct
