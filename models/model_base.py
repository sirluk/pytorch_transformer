from __future__ import annotations

import os
from pathlib import Path
import json
import torch
import math
import functools
from torch import nn
import torch.nn.functional as F
from transformers import AutoModel, AutoModelForCausalLM, LlamaForCausalLM
from collections import OrderedDict
from dataclasses import dataclass

from .model_elements import *

from typing import Optional


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


class TransformerBase(nn.Module):

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

    def forward(self, x):
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
        

class Transformer(nn.Module):

    def __init__(self, lm: bool, cfg = None, model = None, lm_head = None):
        super().__init__()

        self.lm = lm

        if model is None:
            self.model = TransformerBase(cfg)
            self.cfg = cfg
        else:
            self.model = model
            self.cfg = model.cfg

        if lm:
            if lm_head is None:
                self.lm_head = LMHead(self.cfg.model_dim, self.cfg.vocab_size, self.cfg.norm_eps)
                if self.cfg.tie_word_embeddings:
                    self.lm_head.output_proj.weight = self.model.embedding.weight
            else:
                self.lm_head = lm_head


    def forward(self, x, y=None):
        emb = self.model(x)['last_hidden_state']
        logits = None
        loss = None
        if self.lm:
            logits = self.lm_head(emb)
            if y is not None:
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.contiguous().view(-1), ignore_index=-1)
        return {
            'last_hidden_state': emb,
            'logits': logits,
            'loss': loss
        }
        
    @classmethod
    @torch.no_grad()
    def load_hf_llama2(cls, model_path):
        base_model, lm_head = TransformerBase.load_hf_llama2(model_path, return_lm_head = True)
        return cls(model=base_model, lm_head=lm_head)

    
    @classmethod
    @torch.no_grad()
    def load_meta_llama2(cls, model_path):
        base_model, lm_head = TransformerBase.load_meta_llama2(model_path, return_lm_head = True)
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
        

class HFTransformer(nn.Module):

    MODEL_FACTORY = {
        'auto_model': AutoModel,
        'auto_model_lm': AutoModelForCausalLM
    }

    @property
    def cfg(self):
        return self.model.config.__dict__

    def _init_lm(self, hf_model):
        self.model = hf_model.model
        self.lm_head = hf_model.lm_head

    def _init_base(self, hf_model):
        self.model = hf_model

    def __init__(self, model_name_or_path = None, config = None, lm = True):
        super().__init__()

        self.lm = lm

        if lm:
            model_cls = self.MODEL_FACTORY['auto_model_lm']
            sub_init = self._init_lm
        else:
            model_cls = self.MODEL_FACTORY['auto_model']
            sub_init = self._init_base

        if model_name_or_path is not None:
            hf_model = model_cls.from_pretrained(model_name_or_path)
        else:
            hf_model = model_cls.from_config(config)
        
        sub_init(hf_model)


    def forward(self, x, y=None):
        emb = self.model(input_ids = x)["last_hidden_state"]
        logits = None
        loss = None
        if self.lm:
            logits = self.lm_head(emb)
            if y is not None:
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.contiguous().view(-1), ignore_index=-1)
        return {
            'last_hidden_state': emb,
            'logits': logits,
            'loss': loss
        }
    
    @torch.no_grad()
    def save_checkpoint(self, output_dir, checkpoint_name, *args, **kwargs):
        info_dict = {
            'state_dict': self.state_dict(),
            'cfg': self.model.config,
            'lm': self.lm
        }
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        filepath = output_dir / checkpoint_name
        torch.save(info_dict, filepath, *args, **kwargs)
        return filepath
    
    @classmethod
    def load_checkpoint(cls, filepath, map_location = 'cpu'):
        info_dict = torch.load(filepath, map_location=map_location)
        model = cls(config = info_dict['cfg'], lm = info_dict['lm'])
        model.load_state_dict(info_dict['state_dict'])
        return model


def transformer_factory(hf: bool, *args, **kwargs):
    if hf:
        return HFTransformer(*args, **kwargs)
    else:
        return Transformer(*args, **kwargs)



"""
        if y is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1), ignore_index=-1)
            return {'logits': logits, 'loss': loss}
        else:
            return {'logits': self.lm_head(x[:, [-1], :])}
"""