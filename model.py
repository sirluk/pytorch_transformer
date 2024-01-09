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


def get_submodule_from_name(
    module: torch.nn.Module,
    submodule_name: str
):
    return functools.reduce(lambda a,b: getattr(a,b), [module] + submodule_name.split("."))


def apply_rotary_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    freqs_cos: torch.Tensor,
    freqs_sin: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:

    # reshape q and k to match the complex representation
    q_r, q_i = q.float().reshape(q.shape[:-1] + (-1, 2)).unbind(-1)
    k_r, k_i = k.float().reshape(k.shape[:-1] + (-1, 2)).unbind(-1)

    # apply rotation using real numbers
    q_out_a = q_r * freqs_cos - q_i * freqs_sin
    q_out_b = q_i * freqs_cos + q_r * freqs_sin
    k_out_a = k_r * freqs_cos - k_i * freqs_sin
    k_out_b = k_i * freqs_cos + k_r * freqs_sin

    # flatten last two dimensions
    q_out = torch.stack([q_out_a, q_out_b], dim=-1).flatten(3)
    k_out = torch.stack([k_out_a, k_out_b], dim=-1).flatten(3)

    return q_out.type_as(q), k_out.type_as(k)


class LegacyEmbedding(nn.Module):

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

    def __init__(self, context_length, model_dim, n_heads, dropout_prob = 0.1, flash=True):
        super().__init__()

        assert model_dim % n_heads == 0, 'model_dim needs to be a multiple of n_heads'

        self.context_length = context_length
        self.model_dim = model_dim
        self.n_heads = n_heads
        self.dropout_prob = dropout_prob
        self.flash = flash

        # self.attn_proj_qkv = nn.Linear(self.model_dim, 3 * self.model_dim, bias=False)
        self.attn_proj_q = nn.Linear(self.model_dim, self.model_dim, bias=False)
        self.attn_proj_k = nn.Linear(self.model_dim, self.model_dim, bias=False)
        self.attn_proj_v = nn.Linear(self.model_dim, self.model_dim, bias=False)
        self.attn_proj_out = nn.Linear(self.model_dim, self.model_dim, bias=False)
        self.attn_dropout_out = nn.Dropout(dropout_prob)

        if not self.flash:
            causal_mask = torch.tril(torch.ones((context_length, context_length), dtype=bool)) \
                .view(1, 1, context_length, context_length)
            self.register_buffer('causal_mask', causal_mask)
            self.attn_dropout_probs = nn.Dropout(dropout_prob)

        

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
        q, k = apply_rotary_emb(q, k, freqs_cos, freqs_sin)

        if self.flash:
            attn_out = F.scaled_dot_product_attention(q, k, v, dropout_p = self.dropout_prob, is_causal = True)
        else:
            attn = q @ k.transpose(-1, -2)
            attn_scaled = attn * (1.0 / math.sqrt(self.model_dim))
            attn_masked = attn_scaled.masked_fill(~self.causal_mask[:, :, :C,:C], -math.inf)
            attn_probs = F.softmax(attn_masked, dim=-1)
            attn_drop = self.attn_dropout_probs(attn_probs)
            attn_out = attn_drop @ v
        
        attn_cat = attn_out.transpose(1,2).contiguous().view(B, C, D)
        return self.attn_dropout_out(self.attn_proj_out(attn_cat))
    

class ControllerLayer(nn.Module):

    def __init__(self, input_dim, output_dim, k=1):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.k = k

        self.proj = nn.Linear(input_dim, output_dim, bias=False)

    def forward(self, x):
        probs = F.softmax(self.proj(x), dim=-1)
        max_probs, max_idx = probs.max(dim=-1, keepdims=True)
        static_dims = (len(x.shape) - 1) * [-1]
        return max_probs * torch.gather(x, dim=1, index=max_idx.expand(*static_dims, head_dim))


class QQAttention(nn.Module):

    def __init__(self, context_length, model_dim, n_heads, dropout_prob=0.1, flash=True, k=1):
        super().__init__()

        assert model_dim % n_heads == 0, 'model_dim needs to be a multiple of n_heads'

        self.context_length = context_length
        self.model_dim = model_dim
        self.n_heads = n_heads
        self.dropout_prob = dropout_prob
        self.flash = flash
        self.k = k

        # self.attn_proj_qkv = nn.Linear(self.model_dim, 3 * self.model_dim, bias=False)
        self.attn_proj_q = nn.Linear(self.model_dim, self.model_dim, bias=False)
        self.attn_k_controller = ControllerLayer(self.model_dim // self.n_heads, self.n_heads, k=k)
        self.attn_v_controller = ControllerLayer(self.model_dim // self.n_heads, self.n_heads, k=k)
        self.attn_proj_out = nn.Linear(self.model_dim, self.model_dim, bias=False)
        self.attn_dropout_out = nn.Dropout(dropout_prob)

        if not self.flash:
            causal_mask = torch.tril(torch.ones((context_length, context_length), dtype=bool)) \
                .view(1, 1, context_length, context_length)
            self.register_buffer('causal_mask', causal_mask)
            self.attn_dropout_probs = nn.Dropout(dropout_prob)

        

    def forward(self, x, freqs_cos, freqs_sin):
        B, C, D = x.shape
        head_dim = self.model_dim // self.n_heads

        q = self.attn_proj_q(x)
        q = q.reshape(B, C, self.n_heads, head_dim).transpose(1, 2)
        k = self.attn_k_controller(q)
        v = self.attn_v_controller(q)

        # RoPE
        q, k = apply_rotary_emb(q, k, freqs_cos, freqs_sin)

        if self.flash:
            attn_out = F.scaled_dot_product_attention(q, k, v, dropout_p = self.dropout_prob, is_causal = True)
        else:
            attn = q @ k.transpose(-1, -2)
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
        self.dropout_prob = dropout_prob
        self.use_bias = bias
        self.norm_eps = norm_eps
        self.layer_id = layer_id

        self.attn_norm = RMSNorm(model_dim, eps=norm_eps)
        self.attn_layer = CausalSelfAttention(context_length, model_dim, n_attn_heads, dropout_prob)
        self.ffn_norm = RMSNorm(model_dim, eps=norm_eps)
        self.ffn_layer = FFN(model_dim, ffn_hidden_dim, dropout_prob, bias)

    def forward(self, x, freqs_cos, freqs_sin):
        x = x + self.attn_layer(self.attn_norm(x), freqs_cos, freqs_sin)
        x = x + self.ffn_layer(self.ffn_norm(x))
        return x
    

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
        self.dropout_prob = float(cfg.dropout_prob)
        self.use_bias = bool(cfg.bias)
        self.norm_eps = float(cfg.norm_eps)
        self.tie_word_embeddings = bool(cfg.tie_word_embeddings)


        self.embedding = nn.Embedding(self.vocab_size, self.model_dim)
        self.embedding_dropout = nn.Dropout(self.dropout_prob)
        
        self.blocks = nn.ModuleList()
        for layer_id in range(self.n_blocks):
            self.blocks.append(TransformerBlock(
                self.context_length,
                self.model_dim,
                self.n_attn_heads,
                self.ffn_hidden_dim,
                self.dropout_prob,
                self.use_bias,
                self.norm_eps,
                layer_id
            ))

        self.output_norm = RMSNorm(self.model_dim, eps=self.norm_eps)
        self.output_proj = nn.Linear(self.model_dim, self.vocab_size, bias=False)
        
        # weight tying
        if self.tie_word_embeddings:
            self.output_proj.weight = self.embedding.weight

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

    def forward(self, x, inference=False):
        _, seq_len = x.shape
        x = self.embedding(x)
        x = self.embedding_dropout(x)

        freqs_cos = self.freqs_cos[:, :, :seq_len, :]
        freqs_sin = self.freqs_sin[:, :, :seq_len, :]

        for block in self.blocks:
            x = block(x, freqs_cos, freqs_sin)
        
        x = self.output_norm(x)

        if inference:
            return self.output_proj(x[:, [-1], :])
        else:
            return self.output_proj(x)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)


    @classmethod
    @torch.no_grad()
    def load_hf_llama2(cls, llama2_name):

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
            ('blocks.{}.ffn_layer.ffn_linear3', 'model.layers.{}.mlp.down_proj'),
            ('output_norm', 'model.norm'),
            ('output_proj', 'lm_head')
        ]

        src_model = LlamaForCausalLM.from_pretrained(llama2_name)

        dst_cfg = argparse.Namespace(**{
            'context_length': getattr(src_model.config, 'max_position_embeddings'),
            'model_dim': getattr(src_model.config, 'hidden_size'),
            'n_blocks': getattr(src_model.config, 'num_hidden_layers'),
            'n_attn_heads': getattr(src_model.config, 'num_attention_heads'),
            'ffn_hidden_dim': getattr(src_model.config, 'intermediate_size'),
            'dropout_prob': getattr(src_model.config, 'attention_dropout'),
            'bias': getattr(src_model.config, 'attention_bias'),
            'norm_eps': getattr(src_model.config, 'rms_norm_eps'),
            'vocab_size': getattr(src_model.config, 'vocab_size'),
            'tie_word_embeddings': getattr(src_model.config, 'tie_word_embeddings'),
            'ffn_hidden_dim_multiple_of': None
        })
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

        return dst_model
    
    @classmethod
    @torch.no_grad()
    def load_meta_llama2(cls, model_path):

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
            ('blocks.{}.ffn_layer.ffn_linear3', 'layers.{}.feed_forward.w2'),
            ('output_norm', 'norm'),
            ('output_proj', 'output')
        ]

        params_path = os.path.join(model_path, 'params.json')
        with open(params_path) as f:
            params = json.load(f)

        model_paths = sorted(list(Path(model_path).glob('consolidated.*.pth')))
        models = [torch.load(p, map_location='cpu') for p in model_paths]

        state_dict = concat_weights(models)
        del models

        dst_cfg = argparse.Namespace(**{
            'context_length': params['dim'],
            'model_dim': params['dim'],
            'n_blocks': params['n_layers'],
            'n_attn_heads': params['n_heads'],
            'ffn_hidden_dim': None,
            'dropout_prob':  0.0,
            'bias': 'bias' in set([k.split('.')[-1] for k in state_dict.keys()]),
            'norm_eps': params['norm_eps'],
            'vocab_size': state_dict['tok_embeddings.weight'].shape[0],
            'tie_word_embeddings': torch.eq(state_dict['tok_embeddings.weight'], state_dict['output.weight']).all().item(),
            'ffn_hidden_dim_multiple_of': params['multiple_of']
        })
        dst_model = cls(dst_cfg)

        layer_map = [(dst.format(i), src.format(i)) for i in range(dst_cfg.n_blocks) for dst, src in layer_map if 'layers' in src] \
            + [(dst, src) for dst, src in layer_map if 'layers' not in src]

        for (dst, src) in layer_map:
            layer_sd = OrderedDict([(k.split('.')[-1], v) for k,v in state_dict.items() if src in k])
            l_dst = get_submodule_from_name(dst_model, dst)
            l_dst.load_state_dict(layer_sd)

        return dst_model

        


if __name__ == '__main__':

    with open("cfg.yml", "r") as f:
        cfg = yaml.safe_load(f)
    cfg = argparse.Namespace(**cfg['model_cfg'])

    setattr(cfg, 'vocab_size', TokenizedDataset.TOKENIZER.n_words)

    ds = TokenizedDataset(filepaths='data/train.bin', context_length=cfg.context_length)

    dl = DataLoader(ds, batch_size=4)

    x, y = next(iter(dl))

    """     emb = Embedding(
        vocab_size = TokenizedDataset.TOKENIZER.n_words,
        context_length=cfg.context_length,
        model_dim=cfg.model_dim
    )
    x = emb(x) """

    qq = QQAttention(
        context_length=cfg.context_length,
        model_dim=cfg.model_dim,
        n_heads=cfg.n_attn_heads,
        dropout_prob=cfg.dropout_prob,
        flash=True
    )

    attn = CausalSelfAttention(
        context_length=cfg.context_length,
        model_dim=cfg.model_dim,
        n_heads=cfg.n_attn_heads,
        dropout_prob=cfg.dropout_prob,
        flash=True
    )

    theta = 10000.0
    head_dim = cfg.model_dim // cfg.n_attn_heads

    pos = torch.arange(cfg.context_length).unsqueeze(1)
    dim = torch.arange(0, head_dim, step=2).unsqueeze(0)

    freqs = pos * torch.exp(dim * -math.log(theta) / head_dim)
    freqs_cos = torch.cos(freqs)  # real part
    freqs_sin = torch.sin(freqs)  # imaginary part
    freqs_cos = freqs_cos.view(1, 1, *freqs_cos.shape)
    freqs_sin = freqs_sin.view(1, 1, *freqs_sin.shape)

    x = torch.randn((4, cfg.context_length, cfg.model_dim))

    y = qq(x, freqs_cos, freqs_sin)

    y.sum().backward()

    # model = Transformer(cfg)

    import IPython; IPython.embed(); exit(1)

    x = x[:, :x.shape[1]//2]

    logits = model(x)

    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1), ignore_index=-1)

    import IPython; IPython.embed(); exit(1)

