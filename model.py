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
from dataclasses import dataclass

from typing import Optional


def get_submodule_from_name(
    module: torch.nn.Module,
    submodule_name: str
):
    return functools.reduce(lambda a,b: getattr(a,b), [module] + submodule_name.split("."))


def apply_rotary_emb(
    x: torch.Tensor,
    freqs_cos: torch.Tensor,
    freqs_sin: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:

    # reshape q and k to match the complex representation
    r, i = x.float().reshape(x.shape[:-1] + (-1, 2)).unbind(-1)

    # apply rotation using real numbers
    x_a = r * freqs_cos - i * freqs_sin
    x_b = i * freqs_cos + r * freqs_sin

    # flatten last two dimensions
    out = torch.stack([x_a, x_b], dim=-1).flatten(3)

    return out.type_as(x)


@dataclass
class TransformerConfig:
    vocab_size: int
    context_length: int
    model_dim: int
    n_blocks: int
    n_attn_heads: int
    dropout_prob: float
    bias: bool
    norm_eps: float
    tie_word_embeddings: bool = False
    ffn_hidden_dim: Optional[int] = None
    ffn_hidden_dim_multiple_of: Optional[int] = None
    k_config: Optional[str] = 'mha'
    v_config: Optional[str] = 'mha'
    kv_groups: Optional[int] = None


class AbsPosEmbedding(nn.Module):

    def __init__(self, vocab_size, context_length, model_dim, dropout_prob = 0.1):
        super().__init__()

        self.vocab_size = vocab_size
        self.context_length = context_length
        self.model_dim = model_dim
        self.dropout_prob = dropout_prob

        self.token_emb = nn.Embedding(vocab_size, model_dim)

        pos_enc = self.precompute_pos_enc()
        self.register_buffer("pos_enc", pos_enc)

        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x):
        x = self.token_emb(x)
        x = x * math.sqrt(self.model_dim)
        x = x + self.pos_enc
        return self.dropout(x)
    
    def precompute_pos_enc(self):
        pos = torch.arange(self.context_length).unsqueeze(1)
        dim = torch.arange(0, self.model_dim, step=2).unsqueeze(0)
        val = pos * torch.exp(dim * -math.log(10000) / self.model_dim)

        pos_enc = torch.empty((self.context_length, self.model_dim))
        pos_enc[:,0::2] = torch.sin(val)
        pos_enc[:,1::2] = torch.cos(val)
        return pos_enc.unsqueeze(0)
    

class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.rms_eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.rms_eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


class CausalSelfAttention(nn.Module):

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
    

class CausalSelfAttention2(nn.Module):

    def __init__(
            self,
            context_length,
            model_dim,
            n_heads,
            dropout_prob=0.1,
            k_config = 'mha',
            v_config = 'mha',
            kv_groups = None,
            flash=True
        ):
        super().__init__()

        assert model_dim % n_heads == 0, 'model_dim needs to be a multiple of n_heads'

        self.context_length = context_length
        self.model_dim = model_dim
        self.n_heads = n_heads
        self.dropout_prob = dropout_prob
        self.flash = flash
        self.k_config = k_config
        self.v_config = v_config
        self.kv_groups = kv_groups

        self.attn_proj_q = nn.Linear(self.model_dim, self.model_dim, bias=False)

        self.attn_proj_k = self._get_proj(k_config)
        self.attn_proj_v = self._get_proj(v_config)

        self.attn_proj_out = nn.Linear(self.model_dim, self.model_dim, bias=False)
        self.attn_dropout_out = nn.Dropout(dropout_prob)

        if not self.flash:
            causal_mask = torch.tril(torch.ones((context_length, context_length), dtype=bool)) \
                .view(1, 1, context_length, context_length)
            self.register_buffer('causal_mask', causal_mask)
            self.attn_dropout_probs = nn.Dropout(dropout_prob)

        self.dummy_module = nn.Identity()


    def _get_proj(self, config):
        if config == 'conv':
            return nn.Linear(self.n_heads, self.kv_groups, bias=False)
        elif config == 'gqa':
            return nn.Linear(self.model_dim, self.model_dim // self.n_heads * self.kv_groups, bias=False)
        elif config == 'gqqa' or config == 'gqqa_det': # gqa is gqa based on queries as input instead of x
            return nn.Linear(self.model_dim, self.model_dim // self.n_heads * self.kv_groups, bias=False)
        else:
            return nn.Linear(self.model_dim, self.model_dim, bias=False)
        

    def _forward_kv(self, x, q, proj_layer, config):
        if config == 'conv':
            k_v = proj_layer(q.transpose(1, -1)).transpose(1, -1)
        elif config == 'gqa':
            k_v = proj_layer(x)
            k_v = k_v.reshape(*x.shape[:2], self.kv_groups, self.model_dim // self.n_heads).transpose(1, 2)
        elif config == 'gqqa':
            k_v = proj_layer(q.transpose(1, 2).view(q.shape[0], q.shape[2], -1))
            k_v = k_v.view(*x.shape[:2], self.kv_groups, -1).transpose(1, 2)
        elif config == 'gqqa_det':
            q = q.detach()
            k_v = proj_layer(q.transpose(1, 2).view(q.shape[0], q.shape[2], -1))
            k_v = k_v.view(*x.shape[:2], self.kv_groups, -1).transpose(1, 2) 
        else:
            k_v = proj_layer(x)
            k_v = k_v.reshape(*x.shape[:2], self.n_heads, self.model_dim // self.n_heads).transpose(1, 2)

        if 1 < k_v.shape[1] < self.n_heads:
            # probably not very efficient
            k_v = torch.repeat_interleave(k_v, k_v.shape[1], dim=1, output_size=self.n_heads)
        
        return k_v


    def forward(self, x, freqs_cos, freqs_sin):
        B, C, D = x.shape

        q = self.attn_proj_q(x)
        q = q.reshape(B, C, self.n_heads, self.model_dim // self.n_heads).transpose(1, 2)

        k = self._forward_kv(x, q, self.attn_proj_k, self.k_config)
        v = self._forward_kv(x, q, self.attn_proj_v, self.v_config)

        # RoPE
        q = apply_rotary_emb(q, freqs_cos, freqs_sin)
        k = apply_rotary_emb(k, freqs_cos, freqs_sin)

        if self.flash:
            attn_out = F.scaled_dot_product_attention(q, k, v, dropout_p = self.dropout_prob, is_causal = True)
        else:
            attn = q @ k.transpose(-1, -2)
            attn = self.dummy_module(attn)
            attn_scaled = attn * (1.0 / math.sqrt(self.model_dim))
            attn_masked = attn_scaled.masked_fill(~self.causal_mask[:, :, :C,:C], -math.inf)
            attn_probs = F.softmax(attn_masked, dim=-1)
            attn_drop = self.attn_dropout_probs(attn_probs)
            attn_out = attn_drop @ v

        attn_cat = attn_out.transpose(1,2).contiguous().view(B, C, D)

        return self.attn_dropout_out(self.attn_proj_out(attn_cat))


class FFN(nn.Module):

    def __init__(self, model_dim, hidden_dim = None, dropout_prob = 0.1, bias = False, multiple_of = 256):
        super().__init__()

        self.model_dim = model_dim

        if hidden_dim is None:
            hidden_dim = 4 * model_dim
            hidden_dim = int(2 * hidden_dim / 3)
            hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.ffn_linear1 = nn.Linear(model_dim, hidden_dim, bias=bias)
        self.ffn_linear2 = nn.Linear(model_dim, hidden_dim, bias=bias)
        self.ffn_linear3 = nn.Linear(hidden_dim, model_dim, bias=bias)
        self.ffn_dropout = nn.Dropout(dropout_prob)

    def forward(self, x):
        x = F.silu(self.ffn_linear1(x)) * self.ffn_linear2(x) # SwiGLU activation
        x = self.ffn_linear3(x)
        return self.ffn_dropout(x)
    

class TransformerBlock(nn.Module):

    def __init__(
        self,
        context_length,
        model_dim,
        n_attn_heads,
        ffn_hidden_dim,
        ffn_hidden_dim_multiple_of,
        k_config,
        v_config,
        kv_groups,
        dropout_prob = 0.1,
        bias = False,
        norm_eps = 1e-5,
        layer_id = None
    ):
        super().__init__()

        self.context_length = context_length
        self.model_dim = model_dim
        self.n_attn_heads = n_attn_heads
        self.ffn_hidden_dim = ffn_hidden_dim
        self.ffn_hidden_dim_multiple_of = ffn_hidden_dim_multiple_of
        self.k_config = k_config
        self.v_config = v_config
        self.kv_groups = kv_groups
        self.dropout_prob = dropout_prob
        self.bias = bias
        self.norm_eps = norm_eps
        self.layer_id = layer_id

        self.attn_norm = RMSNorm(model_dim, eps=norm_eps)
        self.attn_layer = CausalSelfAttention(context_length, model_dim, n_attn_heads, dropout_prob)
        self.ffn_norm = RMSNorm(model_dim, eps=norm_eps)
        self.ffn_layer = FFN(model_dim, ffn_hidden_dim, dropout_prob, bias, ffn_hidden_dim_multiple_of)

    def forward(self, x, freqs_cos, freqs_sin):
        x = x + self.attn_layer(self.attn_norm(x), freqs_cos, freqs_sin)
        x = x + self.ffn_layer(self.ffn_norm(x))
        return x
    

class LMHead(nn.Module):

    def __init__(self, model_dim, vocab_size, norm_eps):
        super().__init__()

        self.model_dim = model_dim
        self.vocab_size = vocab_size
        self.norm_eps = norm_eps

        self.output_norm = RMSNorm(self.model_dim, eps=self.norm_eps)
        self.output_proj = nn.Linear(self.model_dim, self.vocab_size, bias=False)

        self.apply(self._init_weights)

    def forward(self, x):
        return self.output_proj(self.output_norm(x))
    
    def _init_weights(self, module):
        for p in module.parameters():
            torch.nn.init.normal_(p, mean=0.0, std=0.02)


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

        for block in self.blocks:
            x = block(x, freqs_cos, freqs_sin)
        
        return {'last_hidden_state': x}
        
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

    @classmethod
    @torch.no_grad()
    def load_hf_llama2(cls, model_path, return_lm_head: bool = False):

        # huggingface permutes WQ and WK, this function reverses it
        def permute_reverse(w, n_heads, dim1, dim2):
            return w.view(n_heads, 2, dim1 // n_heads // 2, dim2).transpose(1, 2).reshape(dim1, dim2)
        
        layer_map = [
            ('embedding', 'model.embed_tokens'),
            ('blocks.{}.attn_norm', 'model.layers.{}.input_layernorm'),
            ('blocks.{}.ffn_norm', 'model.layers.{}.post_attention_layernorm'),
            ('blocks.{}.attn_layer.attn_proj_q', 'model.layers.{}.self_attn.q_proj'),
            ('blocks.{}.attn_layer.attn_proj_k', 'model.layers.{}.self_attn.k_proj'),
            ('blocks.{}.attn_layer.attn_proj_v', 'model.layers.{}.self_attn.v_proj'),
            ('blocks.{}.attn_layer.attn_proj_out', 'model.layers.{}.self_attn.o_proj'),
            ('blocks.{}.ffn_layer.ffn_linear1', 'model.layers.{}.mlp.gate_proj'),
            ('blocks.{}.ffn_layer.ffn_linear2', 'model.layers.{}.mlp.up_proj'),
            ('blocks.{}.ffn_layer.ffn_linear3', 'model.layers.{}.mlp.down_proj')
        ]

        src_model = LlamaForCausalLM.from_pretrained(model_path)

        dst_cfg = TransformerConfig(
            vocab_size = getattr(src_model.config, 'vocab_size'),
            context_length = getattr(src_model.config, 'max_position_embeddings'),
            model_dim = getattr(src_model.config, 'hidden_size'),
            n_blocks = getattr(src_model.config, 'num_hidden_layers'),
            n_attn_heads = getattr(src_model.config, 'num_attention_heads'),
            dropout_prob = getattr(src_model.config, 'attention_dropout'),
            bias = getattr(src_model.config, 'attention_bias'),
            norm_eps = getattr(src_model.config, 'rms_norm_eps'),
            ffn_hidden_dim = getattr(src_model.config, 'intermediate_size')
        )
        dst_model = cls(dst_cfg)

        permute_reverse = functools.partial(
            permute_reverse,
            n_heads=dst_cfg.n_attn_heads,
            dim1=dst_cfg.model_dim,
            dim2=dst_cfg.model_dim
        )

        layer_map = [(dst.format(i), src.format(i)) for i in range(dst_cfg.n_blocks) for dst, src in layer_map if 'layers' in src] \
            + [(dst, src) for dst, src in layer_map if 'layers' not in src]

        for (dst, src) in layer_map:
            l_dst = get_submodule_from_name(dst_model, dst)
            l_src = get_submodule_from_name(src_model, src)
            if 'self_attn.q_proj' in src or 'self_attn.k_proj' in src:
                l_dst.weight.data.copy_(permute_reverse(l_src.weight.data))
            else:
                l_dst.load_state_dict(l_src.state_dict())

        if return_lm_head:
            lm_head = LMHead(dst_cfg.model_dim, dst_cfg.vocab_size, dst_cfg.norm_eps)
            lm_head.output_norm.weight.data.copy_(src_model.model.norm.weight)
            lm_head.output_proj.weight.data.copy_(src_model.lm_head.weight)
            if getattr(src_model.config, 'tie_word_embeddings'):
                lm_head.output_proj.weight = dst_model.embedding.weight
            return dst_model, lm_head
        else:
            return dst_model
        
    @classmethod
    @torch.no_grad()
    def load_meta_llama2(cls, model_path, return_lm_head: bool = False):

        def concat_weights(models):
            state_dict = {}
            for name in list(models[0]):
                tensors = [model[name] for model in models]
                if len(tensors) == 1 or len(tensors[0].shape) == 1:
                    state_dict[name] = tensors[0]
                    continue
                is_axis_1 = (
                    name.startswith('tok_embeddings.')
                    or name.endswith('.attention.wo.weight')
                    or name.endswith('.feed_forward.w2.weight')
                )
                axis = 1 if is_axis_1 else 0
                state_dict[name] = torch.cat(tensors, dim=axis)
                for model in models:
                    del model[name]
            return state_dict
        
        layer_map = [
            ('embedding', 'tok_embeddings'),
            ('blocks.{}.attn_norm', 'layers.{}.attention_norm'),
            ('blocks.{}.ffn_norm', 'layers.{}.ffn_norm'),
            ('blocks.{}.attn_layer.attn_proj_q', 'layers.{}.attention.wq'),
            ('blocks.{}.attn_layer.attn_proj_k', 'layers.{}.attention.wk'),
            ('blocks.{}.attn_layer.attn_proj_v', 'layers.{}.attention.wv'),
            ('blocks.{}.attn_layer.attn_proj_out', 'layers.{}.attention.wo'),
            ('blocks.{}.ffn_layer.ffn_linear1', 'layers.{}.feed_forward.w1'),
            ('blocks.{}.ffn_layer.ffn_linear2', 'layers.{}.feed_forward.w3'),
            ('blocks.{}.ffn_layer.ffn_linear3', 'layers.{}.feed_forward.w2')
        ]

        params_path = os.path.join(model_path, 'params.json')
        with open(params_path) as f:
            params = json.load(f)

        model_paths = sorted(list(Path(model_path).glob('consolidated.*.pth')))
        models = [torch.load(p, map_location='cpu') for p in model_paths]

        state_dict = concat_weights(models)
        del models

        dst_cfg = TransformerConfig(
            vocab_size = state_dict['tok_embeddings.weight'].shape[0],
            context_length = params['dim'],
            model_dim = params['dim'],
            n_blocks = params['n_layers'],
            n_attn_heads = params['n_heads'],
            dropout_prob = 0.0,
            bias = 'bias' in set([k.split('.')[-1] for k in state_dict.keys()]),
            norm_eps = params['norm_eps'],
            ffn_hidden_dim_multiple_of = params['multiple_of']
        )
        dst_model = cls(dst_cfg)

        layer_map = [(dst.format(i), src.format(i)) for i in range(dst_cfg.n_blocks) for dst, src in layer_map if 'layers' in src] \
            + [(dst, src) for dst, src in layer_map if 'layers' not in src]

        for (dst, src) in layer_map:
            layer_sd = OrderedDict([(k.split('.')[-1], v) for k,v in state_dict.items() if src in k])
            l_dst = get_submodule_from_name(dst_model, dst)
            l_dst.load_state_dict(layer_sd)

        if return_lm_head:
            lm_head = LMHead(dst_cfg.model_dim, dst_cfg.vocab_size, dst_cfg.norm_eps)
            lm_head.output_norm.weight.data.copy_(state_dict['norm.weight'])
            lm_head.output_proj.weight.data.copy_(state_dict['output.weight'])
            tie_word_embeddings = torch.eq(state_dict['tok_embeddings.weight'], state_dict['output.weight']).all().item()
            if tie_word_embeddings:
                lm_head.output_proj.weight = dst_model.embedding.weight
            return dst_model, lm_head
        else:
            return dst_model


class TransformerLM(nn.Module):

    def __init__(self, cfg = None, model = None, lm_head = None):
        super().__init__()

        if model is None:
            self.model = Transformer(cfg)
            self.cfg = cfg
        else:
            self.model = model
            self.cfg = model.cfg

        if lm_head is None:
            self.lm_head = LMHead(self.cfg.model_dim, self.cfg.vocab_size, self.cfg.norm_eps)
            if self.cfg.tie_word_embeddings:
                self.lm_head.output_proj.weight = self.model.embedding.weight
        else:
            self.lm_head = lm_head


    def forward(self, x, y=None):
        x = self.model(x)['last_hidden_state']

        if y is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1), ignore_index=-1)
            return {'logits': logits, 'loss': loss}
        else:
            return {'logits': self.lm_head(x[:, [-1], :])}
        
    @classmethod
    @torch.no_grad()
    def load_hf_llama2(cls, model_path):
        base_model, lm_head = Transformer.load_hf_llama2(model_path, return_lm_head = True)
        return cls(model=base_model, lm_head=lm_head)

    
    @classmethod
    @torch.no_grad()
    def load_meta_llama2(cls, model_path):
        base_model, lm_head = Transformer.load_meta_llama2(model_path, return_lm_head = True)
        return cls(model=base_model, lm_head=lm_head)
    
    @torch.no_grad()
    def save_checkpoint(self, output_dir, checkpoint_name, *args, **kwargs):
        info_dict = {
            'state_dict': self.state_dict(),
            'cfg': self.cfg
        }
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        filepath = output_dir / checkpoint_name
        torch.save(info_dict, filepath, *args, **kwargs)
        return filepath
    
    @classmethod
    def load_checkpoint(cls, filepath, map_location = 'cpu'):
        info_dict = torch.load(filepath, map_location=map_location)
        model = cls(info_dict['cfg'])
        model.load_state_dict(info_dict['state_dict'])
        return model
    
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
    

class ContextHead(nn.Module):

    def __init__(self, model_dim, vocab_size, norm_eps = 1e-5):
        super().__init__()
        
        self.context_norm = RMSNorm(model_dim, eps=norm_eps)
        self.proj_in = nn.Linear(model_dim, model_dim, bias=False)
        self.proj_out = nn.Linear(model_dim, vocab_size, bias=False)

        self.apply(self._init_weights)

    def forward(self, x):
        x = self.context_norm(x)
        return self.proj_out(x + F.silu(self.proj_in(x)))

    def _init_weights(self, module):
        for p in module.parameters():
            torch.nn.init.normal_(p, mean=0.0, std=0.02)
        

class TransformerContext(TransformerLM):

    def __init__(self, cfg = None, model = None, lm_head = None, context_head = None):
        super().__init__(cfg, model, lm_head)

        # new params
        # TODO integrate into cfg
        self.ctx_future_context_size = 16
        self.ctx_topk_logits = None
        self.ctx_conv_strength = 0.5

        # exponential decay context
        exp_decay = torch.exp(-self.ctx_conv_strength * torch.arange(self.ctx_future_context_size).flip(0))
        self.register_buffer('exp_decay', exp_decay / exp_decay.sum(), persistent=False)

        # context projection
        if context_head is None:
            self.context_head = ContextHead(self.cfg.model_dim, self.cfg.vocab_size)
        else:
            self.context_head = context_head

        self.context_head.proj_out.weight.data.copy_(self.lm_head.output_proj.weight.data)
        self.context_head.context_norm.weight.data.copy_(self.lm_head.output_norm.weight.data)

        self.frozen_backbone = False
        self.frozen_head = False


    def freeze_base_model(self, freeze = True):
        if freeze != self.frozen_backbone:
            for p in self.model.parameters():
                try:
                    p.requires_grad = (not freeze)
                except:
                    pass
            self.frozen_backbone = freeze


    def freeze_head(self, freeze = True):
        if freeze != self.frozen_head:
            for p in self.lm_head.parameters():
                try:
                    p.requires_grad = (not freeze)
                except:
                    pass
            self.frozen_head = freeze


    def freeze_model(self, freeze = True):
        self.freeze_base_model(freeze)
        self.freeze_head(freeze)


    def forward(self, x, y=None, y_context=None, context_loss_device = None):

        head_device = self.lm_head.output_proj.weight.device

        if context_loss_device is None:
            context_loss_device = head_device

        B, seq_len = x.shape
        
        emb = self.model(x)['last_hidden_state']

        emb_norm = self.lm_head.output_norm(emb.to(head_device))

        if y is not None:

            logits = self.lm_head.output_proj(emb_norm)
            c = self.context_head(emb_norm)

            if y_context is None:
                # build indices for context target
                indices = torch.arange(y.shape[1], device=context_loss_device)
                indices = (indices.view(-1, 1).repeat((1, y.shape[1])) - indices.view(1, -1))
                indices = (indices[self.ctx_future_context_size-1:, :self.ctx_future_context_size] % y.shape[1]).flip(1)

                # get context targets
                y_context = torch.gather(y.to(context_loss_device), dim=1, index=indices.view(1, -1).expand(B, -1)).view(B, seq_len, -1)
                y_context = F.one_hot(y_context, num_classes=self.cfg.vocab_size).sum(-2)

            # context loss
            context_loss_alpha = 0.5
            context_loss_weights = 1. + context_loss_alpha * (y_context > 1).any(dim=0)
            context_loss = F.binary_cross_entropy_with_logits(c.to(context_loss_device), y_context.bool().float(), weight = context_loss_weights)

            # convolution along time dimension
            # TODO conv weight as output of linear projection of last token?
            # TODO detach c or not?
            context = F.conv1d(
                F.pad(c.transpose(1,2), (self.ctx_future_context_size-1,0), value=0).view(B*self.cfg.vocab_size, 1, -1),
                self.exp_decay.view(1,1,-1).type_as(c)
            ).view(B, self.cfg.vocab_size, seq_len).transpose(1,2)

            if self.ctx_topk_logits is not None:
                idx = torch.argsort(logits, dim=-1, descending=True)[...,self.ctx_topk_logits:]
                context = context.scatter(dim=-1, index=idx, value=-math.inf)
            # not sure rescaling both logits is needed
                
            import IPython; IPython.embed(); exit(1)
            logits = logits + context
            
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y[:,:-self.ctx_future_context_size+1:].contiguous().view(-1), ignore_index=-1)

            return {'logits': logits, 'loss': loss, 'context_loss': context_loss.to(loss.device)}
        else:
            logits = self.lm_head.output_proj(emb_norm[:, [-1]])
            c = self.context_head(emb_norm[:, -self.ctx_future_context_size:])
            
            context = (c.transpose(1,2) * self.exp_decay[-c.shape[1]:]).sum(-1).unsqueeze(1)

            if self.ctx_topk_logits is not None:
                idx = torch.argsort(logits, dim=-1, descending=True)[...,self.ctx_topk_logits:]
                context = context.scatter(dim=-1, index=idx, value=0.)
            logits = logits + context

            return {'logits': logits}



if __name__ == '__main__':

    with open("cfg.yml", "r") as f:
        cfg = yaml.safe_load(f)
    cfg = TransformerConfig(**cfg['model_cfg_llama7b'])
    
    setattr(cfg, 'norm_eps', float(cfg.norm_eps))

    ds = TokenizedDataset(filepaths='data/train.bin', context_length=cfg.context_length, predict_n_tokens=16)
    dl = DataLoader(ds, batch_size=4)

    x, y = next(iter(dl))

    # model = TransformerLM.load_hf_llama2("meta-llama/Llama-2-7b-hf")
    # model = TransformerContext.load_meta_llama2("../llama/llama-2-7b")
    
    # qunatization config
    from transformers import BitsAndBytesConfig, AutoModel
    quantization_config = BitsAndBytesConfig(
        load_in_4bit = True, # enable 4-bit quantization
        bnb_4bit_quant_type = 'nf4', # information theoretically optimal dtype for normally distributed weights
        bnb_4bit_use_double_quant = True, # quantize quantized weights //insert xzibit meme
        bnb_4bit_compute_dtype = torch.bfloat16 # optimized fp format for ML
    )
    base_model = AutoModel.from_pretrained("meta-llama/Llama-2-7b-hf", quantization_config=quantization_config, device_map='auto')
    setattr(base_model, 'cfg', cfg)
    model = TransformerContext(cfg, model = base_model)

    # import IPython; IPython.embed(); exit(1)

    # model = Transformer(cfg)

    # import IPython; IPython.embed(); exit(1)

    # x = x[:, :x.shape[1]//2]

    with torch.no_grad():
        out = model(x, y)

    # loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1), ignore_index=-1)

    import IPython; IPython.embed(); exit(1)



