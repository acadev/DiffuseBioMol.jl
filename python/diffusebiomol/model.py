"""Batch-first Pairformer-lite + DiT. Dense pair storage is still quadratic."""
import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class ModelConfig:
    d_single: int = 16
    d_pair: int = 8
    n_heads: int = 4
    n_pairformer_layers: int = 1
    n_dit_layers: int = 1
    d_time: int = 16
    d_hidden_mult: int = 2

    def __post_init__(self):
        if min(vars(self).values()) <= 0 or self.d_single % self.n_heads or self.d_time % 2:
            raise ValueError("Positive dimensions, divisible head width and even time width required")


class Attention(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.heads = c.n_heads
        self.q = nn.Linear(c.d_single, c.d_single)
        self.k = nn.Linear(c.d_single, c.d_single)
        self.v = nn.Linear(c.d_single, c.d_single)
        self.out = nn.Linear(c.d_single, c.d_single)
        self.bias = nn.Linear(c.d_pair, c.n_heads)

    def forward(self, s, z, valid):
        b, n, d = s.shape
        def heads(layer):
            return layer(s).reshape(b, n, self.heads, d // self.heads).transpose(1, 2)
        bias = self.bias(z).permute(0, 3, 1, 2)
        bias = bias.masked_fill(~valid[:, None, None, :], float("-inf"))
        y = F.scaled_dot_product_attention(heads(self.q), heads(self.k), heads(self.v), attn_mask=bias)
        return self.out(y.transpose(1, 2).reshape(b, n, d))


def mlp(c):
    return nn.Sequential(nn.Linear(c.d_single, c.d_single * c.d_hidden_mult),
                         nn.GELU(approximate="tanh"),
                         nn.Linear(c.d_single * c.d_hidden_mult, c.d_single))


class Pairformer(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.attn = Attention(c)
        self.norm1 = nn.LayerNorm(c.d_single)
        self.norm2 = nn.LayerNorm(c.d_single)
        self.mlp = mlp(c)
        self.a = nn.Linear(c.d_single, c.d_pair)
        self.b = nn.Linear(c.d_single, c.d_pair)
        self.mix = nn.Linear(c.d_pair, c.d_pair)
        self.pair_norm = nn.LayerNorm(c.d_pair)

    def forward(self, s, z, valid):
        s = self.norm1(s + self.attn(s, z, valid))
        s = self.norm2(s + self.mlp(s))
        update = F.gelu(self.mix(self.a(s)[:, :, None] + self.b(s)[:, None, :]), approximate="tanh")
        return s, self.pair_norm(z + update)


class DiT(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.attn = Attention(c)
        self.norm1 = nn.LayerNorm(c.d_single, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(c.d_single, elementwise_affine=False)
        self.ada1 = nn.Linear(c.d_time, 2 * c.d_single)
        self.ada2 = nn.Linear(c.d_time, 2 * c.d_single)
        self.mlp = mlp(c)

    @staticmethod
    def modulate(x, params):
        scale, shift = params[:, None].chunk(2, dim=-1)
        return (1 + scale) * x + shift

    def forward(self, s, z, time, valid):
        h = self.modulate(self.norm1(s), self.ada1(time))
        s = s + self.attn(h, z, valid)
        return s + self.mlp(self.modulate(self.norm2(s), self.ada2(time)))


class FlowModel(nn.Module):
    def __init__(self, config, vocab_sizes):
        super().__init__()
        self.config = config
        c = config
        self.element = nn.Embedding(vocab_sizes[0], c.d_single)
        self.modality = nn.Embedding(vocab_sizes[1], c.d_single)
        self.polymer = nn.Embedding(vocab_sizes[2], c.d_single)
        self.relpos = nn.Embedding(vocab_sizes[3], c.d_pair)
        self.condition = nn.Linear(4, c.d_single)
        self.encoder = nn.ModuleList([Pairformer(c) for _ in range(c.n_pairformer_layers)])
        self.decoder = nn.ModuleList([DiT(c) for _ in range(c.n_dit_layers)])
        self.coord = nn.Linear(3, c.d_single)
        self.time_mlp = nn.Sequential(nn.Linear(c.d_time, c.d_time),
                                      nn.GELU(approximate="tanh"), nn.Linear(c.d_time, c.d_time))
        self.head = nn.Linear(c.d_single, 3)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)
        self.register_buffer("freq", torch.exp(-math.log(10000) * torch.arange(c.d_time // 2) / (c.d_time // 2)))

    def forward(self, batch, x, t, cond=None):
        if cond is None:
            cond = x.new_zeros(*x.shape[:2], 4)
        s = self.element(batch["element"]) + self.modality(batch["modality"]) + self.polymer(batch["polymer"]) + self.condition(cond)
        z = self.relpos(batch["relpos"])
        for layer in self.encoder:
            s, z = layer(s, z, batch["valid"])
        angles = t[:, None] * self.freq[None]
        time = self.time_mlp(torch.cat((angles.sin(), angles.cos()), dim=-1))
        s = s + self.coord(x)
        for layer in self.decoder:
            s = layer(s, z, time, batch["valid"])
        return self.head(s)
