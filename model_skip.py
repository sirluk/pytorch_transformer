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
import torch.nn.functional as F
from transformers import AutoConfig, AutoTokenizer
from transformers.utils import add_start_docstrings_to_model_forward, replace_return_docstrings
from transformers.modeling_outputs import CausalLMOutputWithPast, BaseModelOutputWithPast
from transformers.modeling_utils import ModuleUtilsMixin
from transformers.generation_utils import GenerationMixin
from transformers.utils.hub import PushToHubMixin
from transformers.integrations.peft import PeftAdapterMixin
from transformers.cache_utils import Cache, DynamicCache
from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask
from transformers.dynamic_module_utils import get_class_from_dynamic_module
from torch.utils.data import DataLoader
from data_loader import TokenizedDataset

from models.model_base import *
from models.model_elements import *
from models.modeling_phi import (
    PhiConfig,
    PhiDecoderLayer,
    PhiModel,
    PhiForCausalLM,
    PhiPreTrainedModel,
    logger,
    PHI_INPUTS_DOCSTRING,
    _CONFIG_FOR_DOC
)

from typing import Optional, Tuple, Union, List


class CustomPhiDecoderLayer(PhiDecoderLayer):

    def __init__(self, config: PhiConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        self.layer_idx = layer_idx

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        route_probs: Optional[torch.Tensor] = None
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`):
                input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            position_ids (`torch.LongTensor` of shape `({0})`, *optional*):
                Indices of positions of each input sequence tokens in the position embeddings. Selected in the range
                `[0, config.n_positions - 1]`. [What are position IDs?](../glossary#position-ids)
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
        """

        residual = hidden_states

        # TEMP - FOR TESTING ONLY
        #routes = torch.randn_like(routes)
        #routes = F.sigmoid(routes)
        # TEMP TEMP TEMP

        # import IPython; IPython.embed(); exit(1)

        hidden_states = hidden_states * route_probs

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        attn_outputs, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )
        attn_outputs = self.resid_dropout(attn_outputs)

        feed_forward_hidden_states = self.resid_dropout(self.mlp(hidden_states))
        
        layer_hidden_state = attn_outputs + feed_forward_hidden_states

        hidden_states = layer_hidden_state + residual
        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs
    

class CustomPhiModel(PhiPreTrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`PhiDecoderLayer`]

    Args:
        config: PhiConfig
    """

    def __init__(self):
        config = AutoConfig.from_pretrained("microsoft/phi-2")
        ref = AutoModelForCausalLM.from_pretrained("microsoft/phi-2", trust_remote_code=True)
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.embed_dropout = nn.Dropout(config.embd_pdrop)
        self.layers = nn.ModuleList(
            [CustomPhiDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.final_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self._use_flash_attention_2 = config._attn_implementation == "flash_attention_2"

        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=True)
        self.router = ContextHead(config.hidden_size, config.hidden_size * (config.num_hidden_layers - 1))

        self.concrete_lower = -1.5
        self.concrete_upper = 1.5
        self.n_samples = 3

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        # self.init_weights()
        for n, p_src in ref.model.named_parameters():
            p_dst = get_submodule_from_name(self, n)
            p_dst.data.copy_(p_src.data)
        self.lm_head.load_state_dict(ref.lm_head.state_dict())

        self._backward_compatibility_gradient_checkpointing()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def get_l0_norm_term(self, alpha: torch.Tensor) -> torch.Tensor:
        log_ratio = 0 if (self.concrete_lower == 0) else math.log(-self.concrete_lower / self.concrete_upper)
        return torch.sigmoid(alpha - log_ratio).sum()

    @add_start_docstrings_to_model_forward(PHI_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        labels: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape[:2]
        elif inputs_embeds is not None:
            batch_size, seq_length = inputs_embeds.shape[:2]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        past_key_values_length = 0

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        if use_cache:
            use_legacy_cache = not isinstance(past_key_values, Cache)
            if use_legacy_cache:
                past_key_values = DynamicCache.from_legacy_cache(past_key_values)
            past_key_values_length = past_key_values.get_usable_length(seq_length)

        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
            )
            position_ids = position_ids.unsqueeze(0)

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        inputs_embeds = self.embed_dropout(inputs_embeds)

        # Attention mask.
        if self._use_flash_attention_2:
            # 2d mask is passed through the layers
            attention_mask = attention_mask if (attention_mask is not None and 0 in attention_mask) else None
        else:
            # 4d mask is passed through the layers
            attention_mask = _prepare_4d_causal_attention_mask(
                attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length
            )

        hidden_states = inputs_embeds

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None
        route_probs = torch.ones(*input_ids.shape, self.config.hidden_size, 1, dtype=bool)

        for i, decoder_layer in enumerate(self.layers):
            
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    attention_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    route_probs = route_probs[...,i-1],
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    route_probs = route_probs[...,i-1],
                )

            hidden_states = layer_outputs[0]

            if i == 0:
                routes = self.router(hidden_states.detach())
                route_probs = concrete_stretched(routes.view(1, *routes.shape).expand(3, *[-1 for _ in range(self.n_samples)])).mean(dim=0)
                route_probs = route_probs.view(*routes.shape[:2], self.config.hidden_size, -1)
                l0_loss = self.get_l0_norm_term(routes)

            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.final_layernorm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = None
        if use_cache:
            next_cache = next_decoder_cache.to_legacy_cache() if use_legacy_cache else next_decoder_cache
    
        logits = self.lm_head(hidden_states)
        logits = logits.float()

        loss = None
        if labels is not None:
            logits_ = logits.view(-1, logits.size(-1))
            labels_ = labels.contiguous().view(-1).to(logits_.device)
            loss = F.cross_entropy(logits_, labels_, ignore_index=-100)

        return {
            "logits": logits,
            "loss": loss,
            "l0_loss": l0_loss,
            "route_probs": route_probs
        }


if __name__ == '__main__':

    tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2", trust_remote_code=True)
    model = CustomPhiModel()

    ds = TokenizedDataset(filepaths='/system/user/publicdata/slimpajama_sampled/pretokenized_phi2/train/', context_length=model.config.max_position_embeddings)
    dl = DataLoader(ds, batch_size=4)

    x, y = next(iter(dl))

    o = model(input_ids=x, labels=y)

    import IPython; IPython.embed(); exit(1)