from __future__ import annotations

import torch
import math
import functools
from torch import nn
import torch.nn.functional as F

from typing import Union


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


class ContextHead(nn.Module):

    def __init__(self, dim_in, dim_out, norm_eps = 1e-5):
        super().__init__()
        
        self.context_norm = RMSNorm(dim_in, eps=norm_eps)
        self.proj_in = nn.Linear(dim_in, dim_in, bias=False)
        self.proj_out = nn.Linear(dim_in, dim_out, bias=False)

        self.apply(self._init_weights)

    def forward(self, x):
        x = self.context_norm(x)
        return self.proj_out(x + F.silu(self.proj_in(x)))

    def _init_weights(self, module):
        for p in module.parameters():
            torch.nn.init.normal_(p, mean=0.0, std=0.02)


@torch.inference_mode()
def generate(model, logits_fn, idx, max_new_tokens, temperature=1.0, top_k=None):
    """
    Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
    the sequence max_new_tokens times, feeding the predictions back into the model each time.
    Most likely you'll want to make sure to be in model.eval() mode of operation for this.
    Also note this is a super inefficient version of sampling with no key/value cache.
    """
    training = model.training
    model.eval()
    for _ in range(max_new_tokens):
        # if the sequence context is growing too long we must crop it at block_size
        # idx_cond = idx if idx.size(1) <= model.context_length else idx[:, -model.context_length:]
        # forward the model to get the logits for the index in the sequence
        # logits = self(idx_cond)['logits']
        logits = logits_fn(model)
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
    model.train(training)

    return idx


def concrete_stretched(
    alpha: torch.Tensor,
    l: Union[float, int] = -1.5,
    r: Union[float, int] = 1.5,
    deterministic: bool = False
) -> torch.Tensor:
    if not deterministic:
        u = torch.zeros_like(alpha).uniform_().clamp_(0.0001, 0.9999)
        u_term = u.log() - (1-u).log()
        ua = u_term + alpha
    else:
        ua = alpha
    s = torch.sigmoid(ua)
    s_stretched = s*(r-l) + l
    z = s_stretched.clamp(0, 1000).clamp(-1000, 1)
    return z