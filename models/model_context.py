from __future__ import annotations

import yaml
import torch
import math
from torch import nn
from torch.utils.data import DataLoader
import torch.nn.functional as F

from model_base import *
from model_elements import *

from typing import Optional


class TransformerContext(nn.Module):

    def __init__(self, model_dim: int, vocab_size: int, hf: bool, context_head: Optional[nn.Module] = None, *args, **kwargs):
        super().__init__()

        self.model_dim = model_dim
        self.vocab_size = vocab_size
        self.hf = hf
        self.transformer = transformer_factory(hf, *args, **kwargs)

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
            self.context_head = ContextHead(self.model_dim, self.vocab_size)
        else:
            self.context_head = context_head

        self.context_head.proj_out.weight.data.copy_(self.transformer.lm_head.output_proj.weight.data)
        self.context_head.context_norm.weight.data.copy_(self.transformer.lm_head.output_norm.weight.data)

        self.frozen_backbone = False
        self.frozen_head = False


    def freeze_base_model(self, freeze = True):
        if freeze != self.frozen_backbone:
            for p in self.transformer.model.parameters():
                try:
                    p.requires_grad = (not freeze)
                except:
                    pass
            self.frozen_backbone = freeze


    def freeze_head(self, freeze = True):
        if freeze != self.frozen_head:
            for p in self.transformer.lm_head.parameters():
                try:
                    p.requires_grad = (not freeze)
                except:
                    pass
            self.frozen_head = freeze


    def freeze_model(self, freeze = True):
        self.freeze_base_model(freeze)
        self.freeze_head(freeze)


    def forward(self, x, y=None, y_context=None, context_loss_device = None):

        head_device = self.transformer.lm_head.output_proj.weight.device

        if context_loss_device is None:
            context_loss_device = head_device

        B, seq_len = x.shape
        
        emb = self.transformer.model(x)['last_hidden_state']

        emb_norm = self.transformer.lm_head.output_norm(emb.to(head_device))

        if y is not None:

            logits = self.transformer.lm_head.output_proj(emb_norm)
            c = self.context_head(emb_norm)

            if y_context is None:
                # build indices for context target
                indices = torch.arange(y.shape[1], device=context_loss_device)
                indices = (indices.view(-1, 1).repeat((1, y.shape[1])) - indices.view(1, -1))
                indices = (indices[self.ctx_future_context_size-1:, :self.ctx_future_context_size] % y.shape[1]).flip(1)

                # get context targets
                y_context = torch.gather(y.to(context_loss_device), dim=1, index=indices.view(1, -1).expand(B, -1)).view(B, seq_len, -1)
                y_context = F.one_hot(y_context, num_classes=self.vocab_size).sum(-2)

            # context loss
            context_loss_alpha = 0.5
            context_loss_weights = 1. + context_loss_alpha * (y_context > 1).any(dim=0)
            context_loss = F.binary_cross_entropy_with_logits(c.to(context_loss_device), y_context.bool().float(), weight = context_loss_weights)

            # convolution along time dimension
            # TODO conv weight as output of linear projection of last token?
            # TODO detach c or not?
            context = F.conv1d(
                F.pad(c.transpose(1,2), (self.ctx_future_context_size-1,0), value=0).view(B*self.vocab_size, 1, -1),
                self.exp_decay.view(1,1,-1).type_as(c)
            ).view(B, self.vocab_size, seq_len).transpose(1,2)

            if self.ctx_topk_logits is not None:
                idx = torch.argsort(logits, dim=-1, descending=True)[...,self.ctx_topk_logits:]
                context = context.scatter(dim=-1, index=idx, value=-math.inf)
            # not sure rescaling both logits is needed
                
            logits = logits + context
            
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y[:,:-self.ctx_future_context_size+1:].contiguous().view(-1), ignore_index=-1)

            return {'logits': logits, 'loss': loss, 'context_loss': context_loss.to(loss.device)}
        else:
            logits = self.transformer.lm_head.output_proj(emb_norm[:, [-1]])
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



