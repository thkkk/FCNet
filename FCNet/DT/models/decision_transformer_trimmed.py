import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import warnings
from typing import Optional, Tuple, Union

# from transformers.models.gpt2.modeling_gpt2 import GPT2Model, GPT2Config, GPT2Block, GPT2PreTrainedModel, \
from .transformer_hf.modeling_gpt2 import GPT2Model, GPT2Config, GPT2Block, GPT2PreTrainedModel, \
    PARALLELIZE_DOCSTRING, DEPARALLELIZE_DOCSTRING, GPT2_INPUTS_DOCSTRING, _CHECKPOINT_FOR_DOC, \
    _CONFIG_FOR_DOC, logger, GPT2Attention, GPT2Block, GPT2MLP
from transformers.utils.doc import add_start_docstrings, add_start_docstrings_to_model_forward, add_code_sample_docstrings
from transformers.utils.model_parallel_utils import get_device_map, assert_device_map
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions
from transformers.pytorch_utils import Conv1D

# from FCNet.DT.models.backup.trajectory_gpt2 import GPT2Model
from torch.cuda.amp import autocast

from .decision_offline_model import DecisionOfflineModel

class GPT2Attention_mk(GPT2Attention):
    def __init__(self, config, is_cross_attention=False, layer_idx=None):
        nn.Module.__init__(self)

        max_positions = config.max_position_embeddings
        self.register_buffer(
            "bias",
            torch.tril(torch.ones((max_positions, max_positions), dtype=torch.bool)).view(
                1, 1, max_positions, max_positions
            ),
            persistent=False,
        )
        self.register_buffer("masked_bias", torch.tensor(-1e4), persistent=False)

        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.split_size = self.embed_dim
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"`embed_dim` must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`:"
                f" {self.num_heads})."
            )

        self.scale_attn_weights = config.scale_attn_weights
        self.is_cross_attention = is_cross_attention

        # Layer-wise attention scaling, reordering, and upcasting
        self.scale_attn_by_inverse_layer_idx = config.scale_attn_by_inverse_layer_idx
        self.layer_idx = layer_idx
        self.reorder_and_upcast_attn = config.reorder_and_upcast_attn

        if self.is_cross_attention:
            self.c_attn = Conv1D(2 * self.embed_dim, self.embed_dim)
            self.q_attn = Conv1D(self.embed_dim, self.embed_dim)
        else:
            self.c_attn = Conv1D(3 * self.embed_dim, self.embed_dim)
        self.c_proj = Conv1D(self.embed_dim, self.embed_dim)

        # replace by flash attention
        # self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.attn_pdrop = config.attn_pdrop
        self.n_head = config.n_head
        self.is_causal = config.is_causal

        self.resid_dropout = nn.Dropout(config.resid_pdrop)

        self.pruned_heads = set()

    def _split_heads(self, x:torch.Tensor):
        new_x_shape = x.size()[:-1] + (self.n_head, x.size(-1) // self.n_head)
        x = x.view(*new_x_shape)  # in Tensorflow implem: fct split_states
        return x.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)

    def _merge_heads(self, x:torch.Tensor):
        x = x.permute(0, 2, 1, 3).contiguous()
        new_x_shape = x.size()[:-2] + (x.size(-2) * x.size(-1),)
        return x.view(*new_x_shape)  # in Tensorflow implem: fct merge_states

    def forward(
        self,
        hidden_states: Optional[Tuple[torch.FloatTensor]],
        layer_past: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor]], ...]:
        q, k, v = self.c_attn(hidden_states).split(self.split_size, dim=2)
        q, k, v = list(map(self._split_heads, [q, k, v]))
        attn_output = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_pdrop if self.training else 0.0,
                                           is_causal=self.is_causal)
        attn_output = self._merge_heads(attn_output)
        attn_output = self.c_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)

        present = None # k,v cache
        attn_weights = None
        outputs = (attn_output, present, attn_weights)
        return outputs

class GPT2Block_mk(GPT2Block):
    def __init__(self, config, layer_idx=None):
        nn.Module.__init__(self)
        hidden_size = config.hidden_size
        inner_dim = config.n_inner if config.n_inner is not None else 4 * hidden_size

        self.ln_1 = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)
        if config.use_flash_attn:
            # print('Transformer uses flash attention.')
            self.attn = GPT2Attention_mk(config, layer_idx=layer_idx)
        else:
            # print('Transformer uses normal attention.')
            self.attn = GPT2Attention(config, layer_idx=layer_idx)
        self.ln_2 = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)

        if config.add_cross_attention:
            if config.use_flash_attn:
                self.crossattention = GPT2Attention_mk(config, is_cross_attention=True, layer_idx=layer_idx)
            else:
                self.crossattention = GPT2Attention(config, is_cross_attention=True, layer_idx=layer_idx)
            self.ln_cross_attn = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)

        self.mlp = GPT2MLP(inner_dim, config)
        return

class GPT2Model_mk(GPT2Model):
    def __init__(self, config):
        GPT2PreTrainedModel.__init__(self, config)

        self.embed_dim = config.hidden_size

        # self.wte = nn.Embedding(config.vocab_size, self.embed_dim)
        # self.wpe = nn.Embedding(config.max_position_embeddings, self.embed_dim)

        self.drop = nn.Dropout(config.embd_pdrop)
        # self.h = nn.ModuleList([GPT2Block_mk(config, layer_idx=i) for i in range(config.num_hidden_layers)])
        self.h = nn.ModuleList([GPT2Block(config, layer_idx=i) for i in range(config.num_hidden_layers)])
        self.ln_f = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_epsilon)

        # Model parallel
        self.model_parallel = False
        self.device_map = None
        self.gradient_checkpointing = False

        # Initialize weights and apply final processing
        self.post_init()

    @add_start_docstrings(PARALLELIZE_DOCSTRING)
    def parallelize(self, device_map=None):
        # Check validity of device_map
        warnings.warn(
            "`GPT2Model.parallelize` is deprecated and will be removed in v5 of Transformers, you should load your"
            " model with `device_map='balanced'` in the call to `from_pretrained`. You can also provide your own"
            " `device_map` but it needs to be a dictionary module_name to device, so for instance {'h.0': 0, 'h.1': 1,"
            " ...}",
            FutureWarning,
        )
        self.device_map = (
            get_device_map(len(self.h), range(torch.cuda.device_count())) if device_map is None else device_map
        )
        assert_device_map(self.device_map, len(self.h))
        self.model_parallel = True
        self.first_device = "cpu" if "cpu" in self.device_map.keys() else "cuda:" + str(min(self.device_map.keys()))
        self.last_device = "cuda:" + str(max(self.device_map.keys()))
        # self.wte = self.wte.to(self.first_device)
        # self.wpe = self.wpe.to(self.first_device)
        # Load onto devices
        for k, v in self.device_map.items():
            for block in v:
                cuda_device = "cuda:" + str(k)
                self.h[block] = self.h[block].to(cuda_device)
        # ln_f to last
        self.ln_f = self.ln_f.to(self.last_device)

    @add_start_docstrings(DEPARALLELIZE_DOCSTRING)
    def deparallelize(self):
        warnings.warn(
            "Like `parallelize`, `deparallelize` is deprecated and will be removed in v5 of Transformers.",
            FutureWarning,
        )
        self.model_parallel = False
        self.device_map = None
        self.first_device = "cpu"
        self.last_device = "cpu"
        # self.wte = self.wte.to("cpu")
        # self.wpe = self.wpe.to("cpu")
        for index in range(len(self.h)):
            self.h[index] = self.h[index].to("cpu")
        self.ln_f = self.ln_f.to("cpu")
        torch.cuda.empty_cache()

    def get_input_embeddings(self):
        raise NotImplementedError("mk version of GPT has no input embeddings.")
        # return self.wte

    def set_input_embeddings(self, new_embeddings):
        raise NotImplementedError("mk version of GPT has no input embeddings.")
        # self.wte = new_embeddings

    @add_start_docstrings_to_model_forward(GPT2_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=BaseModelOutputWithPastAndCrossAttentions,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        # input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        # token_type_ids: Optional[torch.LongTensor] = None,
        # position_ids: Optional[torch.LongTensor] = None,
        # head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        # encoder_hidden_states: Optional[torch.Tensor] = None,
        # encoder_attention_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        # output_attentions: Optional[bool] = None,
        # output_hidden_states: Optional[bool] = None,
        # return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPastAndCrossAttentions]:
        # output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        # output_hidden_states = (
        #     output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        # )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        # return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
            batch_size = inputs_embeds.shape[0]
        else:
            raise ValueError("You have to specify either inputs_embeds")

        device = inputs_embeds.device

        # if token_type_ids is not None:
        #     token_type_ids = token_type_ids.view(-1, input_shape[-1])
        # if position_ids is not None:
        #     position_ids = position_ids.view(-1, input_shape[-1])

        if past_key_values is None:
            # past_length = 0
            past_key_values = tuple([None] * len(self.h))
        # else:
        #     past_length = past_key_values[0][0].size(-2)
        # if position_ids is None:
        #     position_ids = torch.arange(past_length, input_shape[-1] + past_length, dtype=torch.long, device=device)
        #     position_ids = position_ids.unsqueeze(0).view(-1, input_shape[-1])

        # GPT2Attention mask.
        # if attention_mask is not None:
        #     if batch_size <= 0:
        #         raise ValueError("batch_size has to be defined and > 0")
        #     attention_mask = attention_mask.view(batch_size, -1)
        #     # We create a 3D attention mask from a 2D tensor mask.
        #     # Sizes are [batch_size, 1, 1, to_seq_length]
        #     # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
        #     # this attention mask is more simple than the triangular masking of causal attention
        #     # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
        #     attention_mask = attention_mask[:, None, None, :]

        #     # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        #     # masked positions, this operation will create a tensor which is 0.0 for
        #     # positions we want to attend and the dtype's smallest value for masked positions.
        #     # Since we are adding it to the raw scores before the softmax, this is
        #     # effectively the same as removing these entirely.
        #     attention_mask = attention_mask.to(dtype=self.dtype)  # fp16 compatibility
        #     attention_mask = (1.0 - attention_mask) * torch.finfo(self.dtype).min

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        # if self.config.add_cross_attention and encoder_hidden_states is not None:
        #     encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
        #     encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
        #     if encoder_attention_mask is None:
        #         encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
        #     encoder_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        # else:
        #     encoder_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # head_mask has shape n_layer x batch x n_heads x N x N
        # head_mask = self.get_head_mask(head_mask, self.config.n_layer)
        # head_mask = [None] * self.config.n_layer

        # mk: inputs_embeds is definitely not None
        # mk: self.wte is discarded
        # if inputs_embeds is None:
        #     inputs_embeds = self.wte(input_ids)

        # mk: self.wpe is discarded, position_embeds are not need
        # position_embeds = self.wpe(position_ids)
        # hidden_states = inputs_embeds + position_embeds
        hidden_states = inputs_embeds

        # mk: self.wte is discarded
        # if token_type_ids is not None:
        #     token_type_embeds = self.wte(token_type_ids)
        #     hidden_states = hidden_states + token_type_embeds

        hidden_states = self.drop(hidden_states)

        output_shape = (-1,) + input_shape[1:] + (hidden_states.size(-1),)

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        presents = () if use_cache else None
        # all_self_attentions = () if output_attentions else None
        # all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None
        # all_hidden_states = () if output_hidden_states else None
        # all_self_attentions = None
        # all_cross_attentions = None
        # all_hidden_states = None
        for i, (block, layer_past) in enumerate(zip(self.h, past_key_values)):
            # Model parallel
            if self.model_parallel:
                torch.cuda.set_device(hidden_states.device)
                # Ensure layer_past is on same device as hidden_states (might not be correct)
                if layer_past is not None:
                    layer_past = tuple(past_state.to(hidden_states.device) for past_state in layer_past)
                # Ensure that attention_mask is always on the same device as hidden_states
                # if attention_mask is not None:
                #     attention_mask = attention_mask.to(hidden_states.device)
                # if isinstance(head_mask, torch.Tensor):
                #     head_mask = head_mask.to(hidden_states.device)
            # if output_hidden_states:
            #     all_hidden_states = all_hidden_states + (hidden_states,)

            # if self.gradient_checkpointing and self.training:

            #     def create_custom_forward(module):
            #         def custom_forward(*inputs):
            #             # None for past_key_value
            #             return module(*inputs, use_cache, output_attentions)

            #         return custom_forward

            #     outputs = torch.utils.checkpoint.checkpoint(
            #         create_custom_forward(block),
            #         hidden_states,
            #         None,
            #         attention_mask,
            #         head_mask[i],
            #         encoder_hidden_states,
            #         encoder_attention_mask,
            #     )
            # else:
            #     outputs = block(
            #         hidden_states,
            #         layer_past=layer_past,
            #         attention_mask=attention_mask,
            #         head_mask=head_mask[i],
            #         encoder_hidden_states=encoder_hidden_states,
            #         encoder_attention_mask=encoder_attention_mask,
            #         use_cache=use_cache,
            #         output_attentions=output_attentions,
            #     )
            outputs = block(
                hidden_states,
                layer_past=layer_past,
                attention_mask=attention_mask,
                # head_mask=None,
                # encoder_hidden_states=None,
                # encoder_attention_mask=None,
                use_cache=use_cache,
                # output_attentions=None,
            )

            hidden_states = outputs[0]
            if use_cache is True:
                presents = presents + (outputs[1],)

            # if output_attentions:
            #     all_self_attentions = all_self_attentions + (outputs[2 if use_cache else 1],)
            #     if self.config.add_cross_attention:
            #         all_cross_attentions = all_cross_attentions + (outputs[3 if use_cache else 2],)

            # Model Parallel: If it's the last layer for that device, put things on the next device
            # if self.model_parallel:
            #     for k, v in self.device_map.items():
            #         if i == v[-1] and "cuda:" + str(k) != self.last_device:
            #             hidden_states = hidden_states.to("cuda:" + str(k + 1))

        hidden_states = self.ln_f(hidden_states)

        hidden_states = hidden_states.view(output_shape)
        # Add last hidden state
        # if output_hidden_states:
        #     all_hidden_states = all_hidden_states + (hidden_states,)

        # if not return_dict:
        #     return tuple(
        #         v
        #         for v in [hidden_states, presents, all_hidden_states, all_self_attentions, all_cross_attentions]
        #         if v is not None
        #     )

        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=presents,
            # hidden_states=all_hidden_states,
            # attentions=all_self_attentions,
            # cross_attentions=all_cross_attentions,
        )

class DecisionTransformer(DecisionOfflineModel):
    def __init__(self, config: dict):
        '''
            @data_mode in [mdp, nonmdp]
        '''
        super().__init__(config)

        config = dict(
            vocab_size=1,  # doesn't matter -- we don't use the vocab
            n_embd=self.hidden_size,
            n_ctx=self.seq_len, # decoder's max_len
            n_layer=self.n_layer,
            n_head=self.n_head,
            input_data_mode=self.input_data_mode,
        )
        if self.ffn_coef != 4:
            config['n_inner'] = int(self.hidden_size * self.ffn_coef)
        config = GPT2Config(**config)
        
        self.transformer = GPT2Model_mk(config)

        self.embed_state = torch.nn.Linear(self.src_dim, self.hidden_size)
        self.embed_ln = nn.LayerNorm(self.hidden_size)
        if self.data_mode == 'nonmdp':
            self.predict_tgt = nn.Sequential(
                nn.Linear(self.hidden_size, self.tgt_dim),
            )
        elif self.data_mode == 'mdp':
            self.predict_tgt = nn.Sequential(
                nn.Linear(self.seq_len * self.hidden_size, self.tgt_dim),
            )
        return

    def forward(
        self, 
        srcs:torch.Tensor, 
        past_key_values=None,
        use_cache=None,
    ):
        '''
            @param srcs: (bz, seq_len, <=src_dim)
        '''
        # with autocast(enabled=self.use_fp16):
        srcs = self._complete_input_tensor_feature(srcs)
        outputs = self.transformer(
            inputs_embeds=self.embed_ln(self.embed_state(srcs)),
            past_key_values=past_key_values,
            use_cache=use_cache,
        )
        if self.data_mode == 'nonmdp':
            tgt_preds = self.predict_tgt(outputs.last_hidden_state)
        elif self.data_mode == 'mdp':
            tgt_preds = self.predict_tgt(
                torch.flatten(outputs.last_hidden_state, start_dim=-2, end_dim=-1)
            )
            tgt_preds = torch.unsqueeze(tgt_preds, -1)
        if use_cache:
            return tgt_preds, outputs.past_key_values
        return tgt_preds
