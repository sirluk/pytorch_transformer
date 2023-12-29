from __future__ import annotations

import yaml
import argparse
import torch
import math
from torch import nn
from torch.utils.data import DataLoader
from data_loader import TokenizedDataset
import torch.nn.functional as F


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
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


class CausalSelfAttention(nn.Module):

    @classmethod
    def apply_rotary_emb(
        cls,
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
        q_out_b = q_r * freqs_sin + q_i * freqs_cos
        k_out_a = k_r * freqs_cos - k_i * freqs_sin
        k_out_b = k_r * freqs_sin + k_i * freqs_cos

        # flatten last two dimensions
        q_out = torch.stack([q_out_a, q_out_b], dim=-1).flatten(3)
        k_out = torch.stack([k_out_a, k_out_b], dim=-1).flatten(3)

        return q_out.type_as(q), k_out.type_as(k)

    def __init__(self, context_length, model_dim, n_heads, dropout_prob = 0.1, flash=True):
        super().__init__()

        assert model_dim % n_heads == 0, 'model_dim needs to be a multiple of n_heads'

        self.context_length = context_length
        self.model_dim = model_dim
        self.n_heads = n_heads
        self.dropout_prob = dropout_prob
        self.flash = flash

        self.proj_qkv = nn.Linear(self.model_dim, 3 * self.model_dim, bias=False)
        self.proj_final = nn.Linear(self.model_dim, self.model_dim, bias=False)
        self.dropout = nn.Dropout(dropout_prob)

        if not self.flash:
            causal_mask = torch.tril(torch.ones((context_length, context_length), dtype=bool)) \
                .view(1, 1, context_length, context_length)
            self.register_buffer('causal_mask', causal_mask)
            self.attn_dropout = nn.Dropout(dropout_prob)

        

    def forward(self, x, freqs_cos, freqs_sin):
        B, C, D = x.shape

        q, k, v = self.proj_qkv(x).split(self.model_dim, dim=-1)

        q = q.reshape(B, C, self.n_heads, self.model_dim // self.n_heads).transpose(1, 2)
        k = k.reshape(B, C, self.n_heads, self.model_dim // self.n_heads).transpose(1, 2)
        v = v.reshape(B, C, self.n_heads, self.model_dim // self.n_heads).transpose(1, 2)

        # RoPE
        q, k = self.apply_rotary_emb(q, k, freqs_cos, freqs_sin)

        if self.flash:
            attn_out = F.scaled_dot_product_attention(q, k, v, dropout_p = self.dropout_prob, is_causal = True)
        else:
            attn = q @ k.transpose(-1, -2)
            attn_scaled = attn * (1.0 / math.sqrt(self.model_dim))
            attn_masked = attn_scaled.masked_fill(~self.causal_mask[:, :, :C,:C], -math.inf)
            attn_probs = F.softmax(attn_masked, dim=-1)
            attn_drop = self.attn_dropout(attn_probs)
            attn_out = attn_drop @ v
        
        attn_cat = attn_out.transpose(1,2).contiguous().view(B, C, D)
        return self.dropout(self.proj_final(attn_cat))
    

class FFN(nn.Module):

    multiple_of = 256 # force hidden_dim to be a multiple of this number

    def __init__(self, model_dim, hidden_dim = None, dropout_prob = 0.1, bias = False):
        super().__init__()

        self.model_dim = model_dim

        if hidden_dim is None:
            hidden_dim = model_dim * 4
            hidden_dim = int(2 * hidden_dim / 3)
            hidden_dim = (hidden_dim - 1 // self.multiple_of + 1) * self.multiple_of

        self.linear1 = nn.Linear(model_dim, hidden_dim, bias=bias)
        self.linear2 = nn.Linear(model_dim, hidden_dim, bias=bias)
        self.linear3 = nn.Linear(hidden_dim, model_dim, bias=bias)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x):
        x = F.silu(self.linear1(x)) * self.linear2(x) # SwiGLU activation
        x = self.linear3(x)
        return self.dropout(x)
    

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
        self.attn = CausalSelfAttention(context_length, model_dim, n_attn_heads, dropout_prob)
        self.ffn_norm = RMSNorm(model_dim, eps=norm_eps)
        self.ffn = FFN(model_dim, ffn_hidden_dim, dropout_prob, bias)

    def forward(self, x, freqs_cos, freqs_sin):
        x = x + self.attn(self.attn_norm(x), freqs_cos, freqs_sin)
        x = x + self.ffn(self.ffn_norm(x))
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
        self.dropout_prob = float(cfg.dropout_prob)
        self.use_bias = bool(cfg.bias)
        self.norm_eps = float(cfg.norm_eps)

        self.embedding = nn.Embedding(self.vocab_size, self.model_dim)
        self.embedding_dropout = nn.Dropout(self.dropout_prob)
        
        self.blocks = nn.ModuleList([TransformerBlock(
            self.context_length,
            self.model_dim,
            self.n_attn_heads,
            self.ffn_hidden_dim,
            self.dropout_prob,
            self.use_bias,
            layer_id
        ) for layer_id in range(self.n_blocks)])

        self.output_norm = RMSNorm(self.model_dim, eps=self.norm_eps)
        self.output_proj = nn.Linear(self.model_dim, self.vocab_size, bias=False)
        
        # weight tying
        self.output_proj.weight = self.embedding.weight

        # some useful precompute for the RoPE relative positional embeddings
        freqs_cos, freqs_sin = self.precompute_freqs_rope()
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)

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
        


if __name__ == '__main__':

    with open("cfg.yml", "r") as f:
        cfg = yaml.safe_load(f)
    cfg = argparse.Namespace(**cfg['model_cfg'])

    setattr(cfg, 'vocab_size', TokenizedDataset.TOKENIZER.n_words)

    ds = TokenizedDataset(filenames='data/train.bin', context_length=cfg.context_length)

    dl = DataLoader(ds, batch_size=4)

    x, y = next(iter(dl))

    """     emb = Embedding(
        vocab_size = TokenizedDataset.TOKENIZER.n_words,
        context_length=cfg.context_length,
        model_dim=cfg.model_dim
    )
    x = emb(x) """

    model = Transformer(cfg)

    import IPython; IPython.embed(); exit(1)

    x = x[:, :x.shape[1]//2]

    logits = model(x)

    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1), ignore_index=-1)

    import IPython; IPython.embed(); exit(1)

