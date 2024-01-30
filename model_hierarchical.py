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

from model import *


class Conv1dDynamic(nn.Module):

    def __init__(self, bias=None, stride=4, padding=6, dilation=1):
        super().__init__()
        self.bias = bias
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        #c = nn.Conv1d(128,128,kernel_size=16,stride=4, padding=6, groups=128)
        
        self.weight = nn.Parameter(torch.randn(128, 1, 16) * 0.02)
        

    def forward(self, x):
        B, T, D = x.shape
        return F.conv1d(x, self.weight[:T], bias=self.bias, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=T)
    

class TokenGroupEncoderSlow(nn.Module):

    def __init__(self, model_dim, token_group_size, bias=True):
        super().__init__()

        self.model_dim = model_dim
        self.token_group_size = token_group_size
        self.hidden_dim = model_dim // token_group_size
        self.bias = bias

        self.projections = nn.ParameterList()
        for _ in range(token_group_size):
            self.projections.append(nn.Parameter(torch.randn(model_dim * token_group_size, model_dim // token_group_size) * 0.02))
            # self.projections.append(nn.Parameter(torch.randn(self.model_dim * self.token_group_size, model_dim // self.token_group_size) * 0.02))

    def forward(self, x, return_loss=False, ignore_first_group=True):
        emb = self.encode(x, ignore_first_group=ignore_first_group)
        y = self.decode(emb)
        out = {'emb': emb, 'y': y}
        if return_loss:
            out['loss'] = F.mse_loss(x[:,ignore_first_group:], out['y'])
        return out
        
    def encode(self, x, ignore_first_group=True):
        emb = []
        for proj in self.projections:
            emb.append(x[:,ignore_first_group:]@proj)  
        return torch.cat(emb, dim=2)
    
    def decode(self, emb):
        y = torch.zeros(emb.shape[0], emb.shape[1], self.model_dim * self.token_group_size, device=emb.device)
        for i, proj in enumerate(self.projections):
            y += emb[...,self.hidden_dim*i:self.hidden_dim*(i+1)]@proj.T
        return y
    

class TokenGroupEncoder(nn.Module):

    def __init__(self, model_dim, token_group_size, bias=True):
        super().__init__()

        self.model_dim = model_dim
        self.token_group_size = token_group_size
        self.hidden_dim = model_dim // token_group_size
        self.bias = bias

        self.projection = nn.Parameter(torch.randn(model_dim * token_group_size, model_dim) * 0.02)

    def forward(self, x, return_loss=False):
        emb = self.encode(x)
        y = self.decode(emb)
        out = {'emb': emb, 'y': y}
        if return_loss:
            out['loss'] = F.mse_loss(x, y)
        return out
        
    def encode(self, x):
        return x@self.projection
    
    def decode(self, emb):
        return emb@self.projection.T



class Transformer(nn.Module):

    def precompute_freqs_rope(self):
        theta = 10000.0
        head_dim = self.model_dim // self.n_attn_heads

        pos = torch.arange(self.context_length).unsqueeze(1)
        dim = torch.arange(0, head_dim, step=2).unsqueeze(0)

        freqs = pos * torch.exp(dim * -math.log(theta) / head_dim)
        freqs_cos = torch.cos(freqs)  # real part
        freqs_sin = torch.sin(freqs)  # imaginary part
        freqs_cos = freqs_cos.view(1, 1, *freqs_cos.shape)
        freqs_sin = freqs_sin.view(1, 1, *freqs_sin.shape)
        return freqs_cos, freqs_sin

    def __init__(self, cfg):
        super().__init__()

        self.n_blocks = int(cfg.n_blocks)
        self.vocab_size = int(cfg.vocab_size)
        self.context_length = int(cfg.context_length)
        self.model_dim = int(cfg.model_dim)
        self.n_attn_heads = int(cfg.n_attn_heads)
        self.ffn_hidden_dim = int(cfg.ffn_hidden_dim) if cfg.ffn_hidden_dim is not None else None
        self.ffn_hidden_dim_multiple_of = int(cfg.ffn_hidden_dim_multiple_of) if cfg.ffn_hidden_dim_multiple_of is not None else None
        self.k_config = str(cfg.k_config)
        self.v_config = str(cfg.v_config)
        self.kv_groups = int(cfg.kv_groups) if cfg.kv_groups is not None else None
        self.dropout_prob = float(cfg.dropout_prob)
        self.use_bias = bool(cfg.bias)
        self.norm_eps = float(cfg.norm_eps)
        # new params
        self.token_group_size = 8


        self.embedding = nn.Embedding(self.vocab_size, self.model_dim)
        self.embedding_dropout = nn.Dropout(self.dropout_prob)
        
        self.blocks = nn.ModuleList()
        for layer_id in range(self.n_blocks):
            self.blocks.append(TransformerBlock(
                self.context_length,
                self.model_dim,
                self.n_attn_heads,
                self.ffn_hidden_dim,
                self.ffn_hidden_dim_multiple_of,
                self.k_config,
                self.v_config,
                self.kv_groups,
                self.dropout_prob,
                self.use_bias,
                self.norm_eps,
                layer_id
            ))

        self.output_norm = RMSNorm(self.model_dim, eps=self.norm_eps)
        self.output_proj = nn.Linear(self.model_dim*2, self.vocab_size, bias=False)
        
        self.token_group_encoder = TokenGroupEncoder(self.model_dim, self.token_group_size)
        self.token_group_proj = nn.Linear(self.model_dim, self.model_dim, bias=False)

        # some useful precompute for the RoPE relative positional embeddings
        freqs_cos, freqs_sin = self.precompute_freqs_rope()
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)

        # for autoencoder
        token_group_mask = torch.tril(torch.ones(self.token_group_size, self.token_group_size, dtype=bool), diagonal=-1).flip(0)
        self.register_buffer('token_group_mask', token_group_mask, persistent=False)

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('ffn_linear2.weight') or pn.endswith('attn_proj_out.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * self.n_blocks))

    def forward(self, x, y=None):
        B, seq_len = x.shape
        x = self.embedding(x)
        x = self.embedding_dropout(x)

        freqs_cos = self.freqs_cos[:, :, :seq_len, :]
        freqs_sin = self.freqs_sin[:, :, :seq_len, :]

        for block in self.blocks:
            x = block(x, freqs_cos, freqs_sin)
        
        x = self.output_norm(x)

        if y is not None:

            # build indices with n=token_group_size shifts
            indices = torch.arange(seq_len, device=x.device)
            indices = (indices.view(-1, 1).repeat((1, seq_len)) - indices.view(1, -1))
            indices = (indices[:, :self.token_group_size] % seq_len).flip(1)

            token_groups = torch.gather(x.detach(), dim=1, index=indices.view(1, -1, 1).expand(x.shape[0], -1, x.shape[2])).view(B, seq_len, -1)
            encoder_out = self.token_group_encoder(token_groups[:,self.token_group_size-1:], return_loss=True)

            # predict embedding of next token group
            # TODO detach or not?
            group_proj = self.token_group_proj(x.detach())
            group_proj_loss = F.mse_loss(group_proj[:,self.token_group_size-1:], encoder_out['emb'])

            # output projection
            logits = self.output_proj(torch.cat([x, group_proj], dim=2))
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1), ignore_index=-1)

            return {'logits': logits, 'loss': loss, 'group_proj_loss': group_proj_loss, 'encoder_loss': encoder_out['loss']}
        else:
            group_proj = self.token_group_proj(x[:, [-1], :].detach())
            return {'logits': self.output_proj(torch.cat([x[:, [-1], :], group_proj], dim=2))}

        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)


    @torch.inference_mode()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        Also note this is a super inefficient version of sampling with no key/value cache.
        """
        training = self.training
        self.eval()
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = idx if idx.size(1) <= self.context_length else idx[:, -self.context_length:]
            # forward the model to get the logits for the index in the sequence
            logits = self(idx_cond)['logits']
            logits = logits[:, -1, :] # crop to just the final time step
            if temperature == 0.0:
                # "sample" the single most likely index
                _, idx_next = torch.topk(logits, k=1, dim=-1)
            else:
                # pluck the logits at the final step and scale by desired temperature
                logits = logits / temperature
                # optionally crop the logits to only the top k options
                if top_k is not None:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float('Inf')
                # apply softmax to convert logits to (normalized) probabilities
                probs = F.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)
        self.train(training)

        return idx
  


if __name__ == '__main__':

    # test_act()

    with open("cfg.yml", "r") as f:
        cfg = yaml.safe_load(f)
    cfg = argparse.Namespace(**cfg['model_cfg'])

    setattr(cfg, 'vocab_size', TokenizedDataset.TOKENIZER.n_words)

    ds = TokenizedDataset(filepaths='data/train.bin', context_length=cfg.context_length)

    dl = DataLoader(ds, batch_size=4)

    x, y = next(iter(dl))

    model = Transformer(cfg)

    # import IPython; IPython.embed(); exit(1)

    # x = x[:, :x.shape[1]//2]

    out = model(x, y)

    # loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1), ignore_index=-1)

    import IPython; IPython.embed(); exit(1)



