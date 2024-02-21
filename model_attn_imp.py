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

    

class TransformerBlock(nn.Module):

    def __init__(
        self,
        context_length,
        model_dim,
        n_attn_heads,
        ffn_hidden_dim,
        ffn_hidden_dim_multiple_of,
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
        self.dropout_prob = dropout_prob
        self.use_bias = bias
        self.norm_eps = norm_eps
        self.layer_id = layer_id

        self.attn_norm = RMSNorm(model_dim, eps=norm_eps)
        self.attn_layer = CausalSelfAttention(context_length, model_dim, n_attn_heads, dropout_prob, flash=False)
        self.ffn_norm = RMSNorm(model_dim, eps=norm_eps)
        self.ffn_layer = FFN(model_dim, ffn_hidden_dim, dropout_prob, bias, ffn_hidden_dim_multiple_of)

    def forward(self, x, freqs_cos, freqs_sin, attn_gate = 1., ffn_gate = 1.):
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
                self.ffn_hidden_dim_multiple_of,
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

    def forward(self, x, y=None):
        _, seq_len = x.shape
        x = self.embedding(x)
        x = self.embedding_dropout(x)

        freqs_cos = self.freqs_cos[:, :, :seq_len, :]
        freqs_sin = self.freqs_sin[:, :, :seq_len, :]

        for block in self.blocks:
            x = block(x, freqs_cos, freqs_sin)
        
        x = self.output_norm(x)

        if y is not None:
            logits = self.output_proj(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1), ignore_index=-1)
            return {'logits': logits, 'loss': loss}
        else:
            return {'logits': self.output_proj(x[:, [-1], :])}
        
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

    # test_act()

    with open("cfg.yml", "r") as f:
        cfg = yaml.safe_load(f)
    cfg = argparse.Namespace(**cfg['model_cfg'])

    setattr(cfg, 'vocab_size', TokenizedDataset.TOKENIZER.n_words)

    ds = TokenizedDataset(filepaths='data/train.bin', context_length=cfg.context_length)

    dl = DataLoader(ds, batch_size=4)

    x, y = next(iter(dl))

    # model = Transformer(cfg)
    model = Transformer.load_meta_llama2("../llama/llama-2-7b")

    # import IPython; IPython.embed(); exit(1)

    # x = x[:, :x.shape[1]//2]

    logits, loss = model(x, y)

    # loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1), ignore_index=-1)

    import IPython; IPython.embed(); exit(1)



