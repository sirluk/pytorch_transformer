from __future__ import annotations

import yaml
import argparse
import torch
import math
import functools
from torch import nn
from torch.utils.data import DataLoader
from data_loader import TokenizedDataset
import torch.nn.functional as F

from model import apply_rotary_emb, RMSNorm, FFN


class ControllerLayer(nn.Module):

    def __init__(self, input_dim, output_dim, alpha = 1e-2, k=1, temperature=1.0):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.alpha = alpha
        self.k = k
        self.temperature = temperature
        self.use_temperature = temperature != 1.0

        self.proj = nn.Linear(input_dim, output_dim, bias=False)

    def forward(self, x, return_loss=True, return_moe_metrics=False):
        logits = self.proj(x)
        if self.use_temperature:
            logits = logits / self.temperature
        probs = F.softmax(logits, dim=-1)
        topk_probs, topk_idx = torch.topk(probs, k=self.k, dim=-1)
        import IPython; IPython.embed(); exit(1)
        static_dims = (len(x.shape)) * [-1]
        router_selections = torch.gather(
            x.unsqueeze(-2).expand(*x.shape[:-1], self.k, self.input_dim),
            dim=1,
            index=topk_idx.unsqueeze(-1).expand(*static_dims, self.input_dim)
        )
        return_dict = {} 
        return_dict['y'] = (topk_probs.unsqueeze(-1) * router_selections).sum(dim=-2)
        if return_loss:
            return_dict['aux_loss'] = self._get_auxiliary_loss(probs, topk_idx)
        if return_moe_metrics:
            return_dict['topk_probs'] = topk_probs
            return_dict['topk_idx'] = topk_idx
        return return_dict
    
    def _gather_router_probs(self, probs):
        batch_dims = torch.tensor(probs.shape[:-1])
        router_probs = probs.sum(dim=list(range(len(batch_dims)))) / batch_dims.prod()
        return router_probs

    def _gather_router_fraction(self, topk_idx):
        counts = torch.bincount(topk_idx.view(-1), minlength=self.output_dim)
        return counts / counts.sum()
    
    def _get_auxiliary_loss(self, probs, topk_idx):
        router_probs = self._gather_router_probs(probs)
        router_fraction = self._gather_router_fraction(topk_idx)
        return self.alpha * self.output_dim * (router_probs * router_fraction).sum()


class ControllerLayer2(nn.Module):

    def __init__(self, input_dim, output_dim, alpha = 1e-2, k=1, temperature=1.0):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.alpha = alpha
        self.k = k
        self.temperature = temperature
        self.use_temperature = temperature != 1.0

        self.proj = nn.Linear(input_dim, output_dim, bias=False)

    def forward(self, x, return_loss=True, return_moe_metrics=False):
        logits = self.proj(x)
        if self.use_temperature:
            logits = logits / self.temperature
        probs = F.softmax(logits, dim=-1)
        topk_probs, topk_idx = torch.topk(probs, k=self.k, dim=-1)
        static_dims = (len(x.shape)) * [-1]
        router_selections = torch.gather(
            x.unsqueeze(-2).expand(*x.shape[:-1], self.k, self.input_dim),
            dim=1,
            index=topk_idx.unsqueeze(-1).expand(*static_dims, self.input_dim)
        )
        return_dict = {} 
        return_dict['y'] = (topk_probs.unsqueeze(-1) * router_selections).sum(dim=-2)
        if return_loss:
            return_dict['aux_loss'] = self._get_auxiliary_loss(probs, topk_idx)
        if return_moe_metrics:
            return_dict['topk_probs'] = topk_probs
            return_dict['topk_idx'] = topk_idx
        return return_dict
    
    def _gather_router_probs(self, probs):
        batch_dims = torch.tensor(probs.shape[:-1])
        router_probs = probs.sum(dim=list(range(len(batch_dims)))) / batch_dims.prod()
        return router_probs

    def _gather_router_fraction(self, topk_idx):
        counts = torch.bincount(topk_idx.view(-1), minlength=self.output_dim)
        return counts / counts.sum()
    
    def _get_auxiliary_loss(self, probs, topk_idx):
        router_probs = self._gather_router_probs(probs)
        router_fraction = self._gather_router_fraction(topk_idx)
        return self.alpha * self.output_dim * (router_probs * router_fraction).sum()
        

class CausalSelfAttention(nn.Module):

    def __init__(
            self,
            context_length,
            model_dim,
            n_heads,
            dropout_prob=0.1,
            flash=True,
            k_config = 'qq',
            v_config = 'qq',
            gqa_groups = None,
            **controller_kwargs
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
        self.gqa_groups = gqa_groups

        self.attn_proj_q = nn.Linear(self.model_dim, self.model_dim, bias=False)

        self.attn_proj_k = self._get_proj(k_config, **controller_kwargs)
        self.attn_proj_v = self._get_proj(v_config, **controller_kwargs)

        self.attn_proj_out = nn.Linear(self.model_dim, self.model_dim, bias=False)
        self.attn_dropout_out = nn.Dropout(dropout_prob)

        if not self.flash:
            causal_mask = torch.tril(torch.ones((context_length, context_length), dtype=bool)) \
                .view(1, 1, context_length, context_length)
            self.register_buffer('causal_mask', causal_mask)
            self.attn_dropout_probs = nn.Dropout(dropout_prob)


    def _get_proj(self, config, **kwargs):
        if config == 'qq':
            return ControllerLayer(self.model_dim // self.n_heads, self.n_heads, **kwargs)
        elif config == 'mqa':
            return nn.Linear(self.model_dim, self.model_dim // self.n_heads, bias=False)
        elif config == 'gqa':
            return nn.Linear(self.model_dim, self.model_dim // self.n_heads * self.gqa_groups, bias=False)
        else:
            return nn.Linear(self.model_dim, self.model_dim, bias=False)
        

    def _forward_kv(self, x, proj_layer, config):
        if config == 'qq':
            return proj_layer(x)
        elif config == 'mha':
            k_v = proj_layer(x)
            k_v = k_v.reshape(*x.shape[:2], self.n_heads, self.model_dim // self.n_heads).transpose(1, 2)
            return {'y': k_v}
        elif config == 'gqa':
            k_v = proj_layer(x)
            k_v = k_v.reshape(*x.shape[:2], self.gqa_groups, self.model_dim // self.n_heads).transpose(1, 2)
            k_v =  torch.repeat_interleave(k_v, self.n_heads // self.gqa_groups, dim=1) # probably not very efficient
            return {'y': k_v}
        else: # mqa
            k_v = proj_layer(x)
            return {'y': k_v.unsqueeze(1)}  


    def forward(self, x, freqs_cos, freqs_sin, **controller_kwargs):
        B, C, D = x.shape

        q = self.attn_proj_q(x)
        q = q.reshape(B, C, self.n_heads, self.model_dim // self.n_heads).transpose(1, 2)

        proj_k = functools.partial(self.attn_proj_k, **controller_kwargs)
        proj_v = functools.partial(self.attn_proj_v, **controller_kwargs)
        k_return_dict = self._forward_kv(q if self.k_config == 'qq' else x, proj_k, self.k_config)
        v_return_dict = self._forward_kv(x if self.v_config != 'qq' else q, proj_v, self.v_config)
        k = k_return_dict.pop('y')
        v = v_return_dict.pop('y')

        # RoPE
        q = apply_rotary_emb(q, freqs_cos, freqs_sin)
        k = apply_rotary_emb(k, freqs_cos, freqs_sin)

        if self.flash:
            attn_out = F.scaled_dot_product_attention(q, k, v_return_dict, dropout_p = self.dropout_prob, is_causal = True)
        else:
            attn = q @ k.transpose(-1, -2)
            attn_scaled = attn * (1.0 / math.sqrt(self.model_dim))
            attn_masked = attn_scaled.masked_fill(~self.causal_mask[:, :, :C,:C], -math.inf)
            attn_probs = F.softmax(attn_masked, dim=-1)
            attn_drop = self.attn_dropout_probs(attn_probs)
            attn_out = attn_drop @ v
        
        attn_cat = attn_out.transpose(1,2).contiguous().view(B, C, D)

        return {'y': self.attn_dropout_out(self.attn_proj_out(attn_cat)), **k_return_dict, **v_return_dict}


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
        layer_id = None,
        k_config = 'mha',
        v_config = 'mha',
        gqa_groups = None,
        **controller_kwargs
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
        self.attn_layer = CausalSelfAttention(context_length, model_dim, n_attn_heads, dropout_prob, True, k_config, v_config, gqa_groups, **controller_kwargs)
        self.ffn_norm = RMSNorm(model_dim, eps=norm_eps)
        self.ffn_layer = FFN(model_dim, ffn_hidden_dim, dropout_prob, bias)

    def forward(self, x, freqs_cos, freqs_sin, **controller_kwargs):
        attn_out = self.attn_layer(self.attn_norm(x), freqs_cos, freqs_sin, **controller_kwargs)
        x = x + attn_out.pop('y')
        x = x + self.ffn_layer(self.ffn_norm(x))
        return {'y': x, **attn_out}
    

class TransformerQQ(nn.Module):

    @functools.cached_property
    def qq_active(self):
        return self.k_config == 'qq' or self.v_config == 'qq'

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
        self.k_config = cfg.k_config
        self.v_config = cfg.v_config
        self.gqa_groups = int(cfg.gqa_groups) if cfg.gqa_groups is not None else None
        self.controller_kwargs = {
            'alpha': float(cfg.controller_alpha),
            'k': int(cfg.controller_k),
            'temperature': float(cfg.controller_temperature)
        }

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
                layer_id,
                self.k_config,
                self.v_config,
                self.gqa_groups,
                **self.controller_kwargs
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

    def forward(self, x, y=None, return_aux_loss=False, return_moe_metrics=False):
        inference_mode = y is None
        _, seq_len = x.shape
        x = self.embedding(x)
        x = self.embedding_dropout(x)

        freqs_cos = self.freqs_cos[:, :, :seq_len, :]
        freqs_sin = self.freqs_sin[:, :, :seq_len, :]

        return_dict = {}
        for i, block in enumerate(self.blocks):
            block_return_dict = block(x, freqs_cos, freqs_sin, return_aux_loss=True, return_moe_metrics=False)
            if return_aux_loss:
                try:
                    return_dict['aux_loss'] += block_return_dict.pop('aux_loss', 0)
                except KeyError:
                    return_dict['aux_loss'] = block_return_dict.pop('loss_aux', 0)
            if return_moe_metrics:
                return_dict[f'block_{i}_topk_probs'] = block_return_dict.pop('topk_probs')
                return_dict[f'block_{i}_topk_idx'] = block_return_dict.pop('topk_idx')
            x = block_return_dict.pop('y')
        
        x = self.output_norm(x)

        if inference_mode:
            return {'logits': self.output_proj(x[:, [-1], :])}
        else:
            logits = self.output_proj(x)
            return {
                'logits': logits,
                'loss': F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1), ignore_index=-1),
                **return_dict
            }
            
    
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

    with open("cfg.yml", "r") as f:
        cfg = yaml.safe_load(f)
    cfg = argparse.Namespace(**cfg['qq_model_cfg'])

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


    qq = CausalSelfAttention(
        context_length=cfg.context_length,
        model_dim=cfg.model_dim,
        n_heads=cfg.n_attn_heads,
        dropout_prob=cfg.dropout_prob,
        flash=True,
        k_config='gqa',
        v_config='qq',
        gqa_groups=4,
        temperature=0.5,
        k=2
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

    import IPython; IPython.embed(); exit(1)

    # y = qq(x, freqs_cos, freqs_sin)

    # y.sum().backward()

    model = TransformerQQ(cfg)

    model.generate(x, 10, temperature=0.5, top_k=5)

    # x = x[:, :x.shape[1]//2]

    logits, aux_loss = model(x, y)

    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1), ignore_index=-1)

    import IPython; IPython.embed(); exit(1)

