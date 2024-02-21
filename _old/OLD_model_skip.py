from __future__ import annotations

import os
from pathlib import Path
import json
import yaml
import argparse
import torch
import math
import functools
from torch import nn
from torch.utils.data import DataLoader
from data_loader import TokenizedDataset
import torch.nn.functional as F
from transformers import LlamaForCausalLM
from collections import OrderedDict
from dataclasses import dataclass, field

from typing import Optional

from model import *


class CausalSelfAttentionMemory(nn.Module):

    def __init__(self, context_length, model_dim, n_heads, dropout_prob = 0.1):
        super().__init__()

        assert model_dim % n_heads == 0, 'model_dim needs to be a multiple of n_heads'

        self.context_length = context_length
        self.model_dim = model_dim
        self.n_heads = n_heads
        self.dropout_prob = dropout_prob

        # self.attn_proj_qkv = nn.Linear(self.model_dim, 3 * self.model_dim, bias=False)
        self.attn_proj_q = nn.Linear(self.model_dim, self.model_dim, bias=False)
        self.attn_proj_k = nn.Linear(self.model_dim, self.model_dim, bias=False)
        self.attn_proj_v = nn.Linear(self.model_dim, self.model_dim, bias=False)
        self.attn_proj_out = nn.Linear(self.model_dim, self.model_dim, bias=False)
        self.attn_dropout_out = nn.Dropout(dropout_prob)

        self.input_gate = nn.Sequential(nn.Linear())
        self.register_buffer("memory", torch.zeros((self.model_dim,)))


    def forward(self, x, freqs_cos, freqs_sin):
        B, C, D = x.shape

        # q, k, v = self.attn_proj_qkv(x).split(self.model_dim, dim=-1)
        q = self.attn_proj_q(x)
        k = self.attn_proj_k(x)
        v = self.attn_proj_v(x)

        q = q.reshape(B, C, self.n_heads, self.model_dim // self.n_heads).transpose(1, 2)
        k = k.reshape(B, C, self.n_heads, self.model_dim // self.n_heads).transpose(1, 2)
        v = v.reshape(B, C, self.n_heads, self.model_dim // self.n_heads).transpose(1, 2)

        # RoPE
        q = apply_rotary_emb(q, freqs_cos, freqs_sin)
        k = apply_rotary_emb(k, freqs_cos, freqs_sin)

        attn_out = F.scaled_dot_product_attention(q, k, v, dropout_p = self.dropout_prob, is_causal = True)
        attn_cat = attn_out.transpose(1,2).contiguous().view(B, C, D)
        return self.attn_dropout_out(self.attn_proj_out(attn_cat))



class Transformer(nn.Module):

    def precompute_freqs_rope(self):
        theta = 10000.0
        head_dim = self.cfg.model_dim // self.cfg.n_attn_heads

        pos = torch.arange(self.cfg.context_length).unsqueeze(1)
        dim = torch.arange(0, head_dim, step=2).unsqueeze(0)

        freqs = pos * torch.exp(dim * -math.log(theta) / head_dim)
        freqs_cos = torch.cos(freqs)  # real part
        freqs_sin = torch.sin(freqs)  # imaginary part
        freqs_cos = freqs_cos.view(1, 1, *freqs_cos.shape)
        freqs_sin = freqs_sin.view(1, 1, *freqs_sin.shape)
        return freqs_cos, freqs_sin

    def __init__(
            self,
            cfg: TransformerConfig
        ):
        super().__init__()

        self.cfg = cfg

        self.embedding = nn.Embedding(cfg.vocab_size, cfg.model_dim)
        self.embedding_dropout = nn.Dropout(cfg.dropout_prob)
        
        self.blocks = nn.ModuleList()
        for layer_id in range(cfg.n_blocks):
            self.blocks.append(TransformerBlock(
                cfg.context_length,
                cfg.model_dim,
                cfg.n_attn_heads,
                cfg.ffn_hidden_dim,
                cfg.ffn_hidden_dim_multiple_of,
                cfg.k_config,
                cfg.v_config,
                cfg.kv_groups,
                cfg.dropout_prob,
                cfg.bias,
                cfg.norm_eps,
                layer_id
            ))

        self.controller_block = TransformerBlock(
            cfg.context_length,
            cfg.model_dim,
            cfg.n_attn_heads,
            cfg.ffn_hidden_dim,
            cfg.ffn_hidden_dim_multiple_of,
            cfg.k_config,
            cfg.v_config,
            cfg.kv_groups,
            cfg.dropout_prob,
            cfg.bias,
            cfg.norm_eps,
            layer_id = -1  
        )
        self.controller_head = nn.Linear(cfg.model_dim, cfg.n_blocks)

        # some useful precompute for the RoPE relative positional embeddings
        freqs_cos, freqs_sin = self.precompute_freqs_rope()
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('ffn_linear2.weight') or pn.endswith('attn_proj_out.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * cfg.n_blocks))

    def forward(self, x, y=None):
        _, seq_len = x.shape
        
        x = self.embedding(x)
        x = self.embedding_dropout(x)

        freqs_cos = self.freqs_cos[:, :, :seq_len, :]
        freqs_sin = self.freqs_sin[:, :, :seq_len, :]

        import IPython; IPython.embed(); exit(1)

        c = self.controller_head(self.controller_block(x, freqs_cos, freqs_sin))

        for block in self.blocks:
            x = block(x, freqs_cos, freqs_sin)
        
        return x
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        else:
            for p in module.parameters():
                torch.nn.init.normal_(p, mean=0.0, std=0.02)



if __name__ == '__main__':

    with open("cfg.yml", "r") as f:
        cfg = yaml.safe_load(f)
    cfg = TransformerConfig(**cfg['model_cfg'])

    setattr(cfg, 'norm_eps', float(cfg.norm_eps))

    ds = TokenizedDataset(filepaths='data/train.bin', context_length=cfg.context_length)
    dl = DataLoader(ds, batch_size=4)

    x, y = next(iter(dl))

    model = Transformer(cfg)

    logits, loss = model(x, y)

    import IPython; IPython.embed(); exit(1)



