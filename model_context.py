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
        self.future_context_size = 8
        # self.register_parameter('context_conv_scale', nn.Parameter(torch.Tensor(1)))
        self.register_parameter('context_conv_weight', nn.Parameter(torch.Tensor(1)))
        self.context_conv_scale = 1
        # self.context_conv_weight = 0.5

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
        self.output_proj = nn.Linear(self.model_dim, self.vocab_size, bias=False)
        self.context_proj = nn.Linear(self.model_dim, self.vocab_size, bias=False)

        # some useful precompute for the RoPE relative positional embeddings
        freqs_cos, freqs_sin = self.precompute_freqs_rope()
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('ffn_linear2.weight') or pn.endswith('attn_proj_out.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * self.n_blocks))
            elif pn.endswith('context_conv_weight'):
                torch.nn.init.constant_(p, 0.5)

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

            logits = self.output_proj(x)
            c = self.context_proj(x)

            # convolution along time dimension
            # TODO conv weight as output of linear projection of last token?
            # TODO detach c or not?
            context_conv_kernel = self.context_conv_scale * torch.exp(-self.context_conv_weight * torch.arange(self.future_context_size))
            context = F.conv1d(
                F.pad(c[:,:-1].transpose(1,-1), (self.future_context_size,0), value=0).view(B*self.vocab_size, 1, -1),
                context_conv_kernel.view(1,1,-1)
            ).view(B, self.vocab_size, seq_len).transpose(1,-1)

            # build indices for context target
            indices = torch.arange(seq_len, device=x.device)
            indices = (indices.view(-1, 1).repeat((1, seq_len)) - indices.view(1, -1))
            indices = (indices[:, :self.future_context_size] % seq_len).flip(1)

            # get context targets
            y_context = torch.gather(y[:, self.future_context_size-1:], dim=1, index=indices.view(1, -1).expand(x.shape[0], -1)).view(B, seq_len, -1)
            y_context = F.one_hot(y_context, num_classes=self.vocab_size).sum(-2)
            context_loss_alpha = 0.5
            context_loss_weights = 1 + context_loss_alpha * (y_context > 1).any(dim=0)
            context_loss = F.binary_cross_entropy_with_logits(c, y_context.bool().float(), weight = context_loss_weights)

            import IPython; IPython.embed(); exit(1)
            logits += context
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y[:,:-self.future_context_size+1:].view(-1), ignore_index=-1)
            # import IPython ;IPython.embed(); exit(1)
            return {'logits': logits, 'loss': loss, 'context_loss': context_loss}
        else:
            logits = self.output_proj(x[:, [-1]])
            c = self.context_proj(x[:, -self.future_context_size-1:-1])
            context_conv_kernel = self.context_conv_scale * torch.exp(-self.context_conv_weight * torch.arange(self.future_context_size))
            context = F.conv1d(
                F.pad(c.transpose(1,-1), (self.future_context_size-c.shape[1],0), value=0).contiguous().view(B*self.vocab_size, 1, -1),
                context_conv_kernel.view(1,1,-1)
            ).view(B, self.vocab_size, -1).transpose(1,-1)
            logits += context
            return {'logits': logits}
        
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

    ds = TokenizedDataset(filepaths='data/train.bin', context_length=cfg.context_length, predict_n_tokens=8)

    dl = DataLoader(ds, batch_size=4)

    x, y = next(iter(dl))

    model = Transformer(cfg)

    # import IPython; IPython.embed(); exit(1)

    # x = x[:, :x.shape[1]//2]

    out = model(x, y)

    # y2 = TokenizedDataset.TOKENIZER.encode(':', bos=True)

    # loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1), ignore_index=-1)

    import IPython; IPython.embed(); exit(1)



