# SPDX-License-Identifier: Apache-2.0

# Adapted from
# https://github.com/huggingface/transformers/blob/v4.28.0/src/transformers/models/llama/modeling_llama.py
# Copyright 2025 The vLLM team.
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Inference-only IBM Granite speeech model."""
import math
from typing import Iterable, Mapping, Optional, Set, Tuple, TypedDict, Union

import torch
import torch.nn.functional as F
from torch import einsum, nn
from transformers import BatchFeature

from vllm.config import VllmConfig
from vllm.model_executor.layers.sampler import SamplerOutput, get_sampler
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.multimodal.inputs import MultiModalFieldConfig, MultiModalKwargs
from vllm.multimodal.parse import MultiModalDataItems
from vllm.multimodal.processing import (BaseMultiModalProcessor,
                                        BaseProcessingInfo, PromptReplacement,
                                        PromptUpdate)
from vllm.multimodal.profiling import BaseDummyInputsBuilder, ProcessorInputs
from vllm.sequence import IntermediateTensors

from .blip2 import Blip2QFormerModel
from .interfaces import SupportsPP
from .utils import AutoWeightsLoader, init_vllm_registered_model, maybe_prefix

###########################################################
# Below is a direct copy of the non-lang model components from transformers


class GraniteSpeechEncoderProjectorQFormer(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.ds_rate = config.downsample_rate
        self.window_size = config.window_size
        self.num_queries = self.window_size // self.ds_rate
        self.query = nn.Parameter(
            torch.zeros(1, self.num_queries, config.hidden_size))
        self.query.data.normal_(mean=0.0, std=1.0)
        # It would be nice if we could do this generically...
        self.qformer = Blip2QFormerModel(
            config,
            quant_config=None,
            cache_config=None,
        )  #### HACK HACK HACK - TODO need to add configs
        self.linear = nn.Linear(config.hidden_size, config.llm_dim)

    def forward(self, x, atts):
        batch_size, seq_len, dim = x.size()
        nblocks = math.ceil(seq_len / self.window_size)
        pad = nblocks * self.window_size - seq_len
        x = nn.functional.pad(x, (0, 0, 0, pad), "constant", 0)
        x = x.view(batch_size * nblocks, self.window_size, dim)

        query_output = self.qformer(
            query_embeds=self.query.data,
            encoder_hidden_states=x,
            encoder_attention_mask=atts,
            return_dict=True,
        )
        query_proj = self.linear(
            query_output.last_hidden_state.view(
                batch_size, nblocks * self.window_size // self.ds_rate, -1))
        return query_proj


### Encoder
class GraniteSpeechCTCModel(nn.Module):

    def __init__(self, config):
        super().__init__()

        self.rnn_tr = nn.ModuleList(
            [nn.Linear(config.input_dim, config.hidden_dim, bias=True)] + [
                GraniteSpeechConformerBlock(
                    dim=config.hidden_dim,
                    dim_head=config.dim_head,
                    heads=config.num_heads,
                    ff_mult=config.feedforward_mult,
                    conv_expansion_factor=config.conv_expansion_factor,
                    conv_kernel_size=config.conv_kernel_size,
                    context_size=config.context_size,  # attention context size
                    attn_dropout=config.dropout,
                    ff_dropout=config.dropout,
                    conv_dropout=config.dropout,
                ) for layer_idx in range(config.num_layers)
            ])

        self.out = nn.Linear(config.hidden_dim, config.output_dim, bias=True)
        self.out_mid = nn.Linear(config.output_dim,
                                 config.hidden_dim,
                                 bias=True)
        self.context_size = config.context_size
        self.input_dim = config.input_dim
        self.num_layers = config.num_layers
        self.hidden_dim = config.hidden_dim
        self.output_dim = config.output_dim

    def forward(self, x: torch.Tensor):
        x = self.rnn_tr[0](x)
        for idx, layer in enumerate(self.rnn_tr[1:], start=1):
            x = layer(x, self.context_size)
            if idx == self.num_layers // 2:
                x_mid = x.clone()
                x_mid = self.out(x_mid)
                x += self.out_mid(nn.Softmax(dim=-1)(x_mid))
        return x


# NOTE: Conformer adapted from: https://github.com/lucidrains/conformer.git
class GraniteSpeechConformerPermute(nn.Module):

    def __init__(self, dims):
        super().__init__()
        self.dims = dims

    def forward(self, x):
        x = x.permute(self.dims)
        return x


class GraniteSpeechConformerDepthWiseConv1d(nn.Module):

    def __init__(self, chan_in, chan_out, kernel_size, padding):
        super().__init__()
        self.padding = padding
        self.conv = nn.Conv1d(chan_in,
                              chan_out,
                              kernel_size,
                              groups=chan_in,
                              bias=False)

    def forward(self, x):
        x = F.pad(x, self.padding)
        return self.conv(x)


class GraniteSpeechConformerScale(nn.Module):

    def __init__(self, scale, fn):
        super().__init__()
        self.fn = fn
        self.scale = scale

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) * self.scale


class GraniteSpeechConformerPreNorm(nn.Module):

    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, **kwargs):
        x = self.norm(x)
        return self.fn(x, **kwargs)


class GraniteSpeechConformerPreNormAttn(nn.Module):

    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, context_size, **kwargs):
        x = self.norm(x)
        return self.fn(x, context_size, **kwargs)


class GraniteSpeechConformerAttention(nn.Module):

    def __init__(
        self,
        dim,
        heads=8,
        dim_head=64,
        dropout=0.0,
        context_size=200,
        max_pos_emb=512,
    ):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.dim_head = dim_head
        self.scale = dim_head**-0.5
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim)

        self.max_pos_emb = max_pos_emb
        self.rel_pos_emb = nn.Embedding(2 * max_pos_emb + 1, dim_head)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, context_size):
        device, h, max_pos_emb = x.device, self.heads, self.max_pos_emb
        bs, n, d = x.shape
        assert context_size > 0 and context_size <= max_pos_emb

        nb = math.ceil(n / context_size)
        nr = n % context_size
        if nr > 0:
            # right padding to reach block size
            x = torch.nn.functional.pad(x, (0, 0, 0, context_size - nr))

        q, k, v = (self.to_q(x), *self.to_kv(x).chunk(2, dim=-1))
        q, k, v = [
            t.reshape(bs, nb, context_size, h, -1).transpose(2, 3)
            for t in (q, k, v)
        ]

        dots = einsum("b m h i d, b m h j d -> b m h i j", q, k) * self.scale

        # shaw's relative positional embedding
        seq = torch.arange(context_size, device=device)
        dist = seq.view(-1, 1) - seq.view(1, -1)
        dist = torch.clamp(dist, -context_size, context_size) + max_pos_emb
        rel_pos_emb = self.rel_pos_emb(dist).to(q)
        pos_attn = einsum("b m h c d, c r d -> b m h c r", q,
                          rel_pos_emb) * self.scale
        dots = dots + pos_attn

        if nr > 0:
            # masked attention in the extended block
            mask = torch.ones(context_size,
                              context_size,
                              dtype=bool,
                              device=device)
            mask[:nr, :nr] = 0
            mask_value = -torch.finfo(dots.dtype).max
            dots[:, -1, :].masked_fill_(mask, mask_value)

        attn = dots.softmax(dim=-1)

        out = einsum("b m h i j, b m h j d -> b m h i d", attn, v)
        out = out.transpose(2, 3).reshape(bs, x.shape[1], -1)
        out = self.to_out(out[:, :n, :])
        return self.dropout(out)


class GraniteSpeechConformerFeedForward(nn.Module):

    def __init__(self, dim, mult=4, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, dim * mult), nn.SiLU(),
                                 nn.Dropout(dropout),
                                 nn.Linear(dim * mult,
                                           dim), nn.Dropout(dropout))

    def forward(self, x):
        return self.net(x)


class GraniteSpeechConformerConvModule(nn.Module):

    def __init__(self,
                 dim,
                 causal=False,
                 expansion_factor=2,
                 kernel_size=31,
                 dropout=0.0):
        super().__init__()

        inner_dim = dim * expansion_factor
        padding = self.calc_same_padding(kernel_size) if not causal else (
            kernel_size - 1, 0)

        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            GraniteSpeechConformerPermute(dims=(0, 2, 1)),
            nn.Conv1d(dim, inner_dim * 2, 1),
            nn.GLU(dim=1),
            GraniteSpeechConformerDepthWiseConv1d(inner_dim,
                                                  inner_dim,
                                                  kernel_size=kernel_size,
                                                  padding=padding),
            nn.BatchNorm1d(inner_dim) if not causal else nn.Identity(),
            nn.SiLU(),
            nn.Conv1d(inner_dim, dim, 1),
            GraniteSpeechConformerPermute(dims=(0, 2, 1)),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

    @staticmethod
    def calc_same_padding(kernel_size: int):
        pad = kernel_size // 2
        return (pad, pad - (kernel_size + 1) % 2)


class GraniteSpeechConformerBlock(nn.Module):

    def __init__(
        self,
        *,
        dim,
        dim_head=64,
        heads=8,
        ff_mult=2,
        conv_expansion_factor=2,
        conv_kernel_size=31,
        context_size=-1,
        attn_dropout=0.0,
        ff_dropout=0.0,
        conv_dropout=0.0,
    ):
        super().__init__()
        self.ff1 = GraniteSpeechConformerFeedForward(dim=dim,
                                                     mult=ff_mult,
                                                     dropout=ff_dropout)
        self.attn = GraniteSpeechConformerAttention(
            dim=dim,
            dim_head=dim_head,
            heads=heads,
            dropout=attn_dropout,
            context_size=context_size,
        )
        self.conv = GraniteSpeechConformerConvModule(
            dim=dim,
            causal=False,
            expansion_factor=conv_expansion_factor,
            kernel_size=conv_kernel_size,
            dropout=conv_dropout,
        )
        self.ff2 = GraniteSpeechConformerFeedForward(dim=dim,
                                                     mult=ff_mult,
                                                     dropout=ff_dropout)

        self.attn = GraniteSpeechConformerPreNormAttn(dim, self.attn)
        self.ff1 = GraniteSpeechConformerScale(
            0.5, GraniteSpeechConformerPreNorm(dim, self.ff1))
        self.ff2 = GraniteSpeechConformerScale(
            0.5, GraniteSpeechConformerPreNorm(dim, self.ff2))

        self.post_norm = nn.LayerNorm(dim)

    def forward(self, x, context_size):
        x = self.ff1(x) + x
        x = self.attn(x, context_size) + x
        x = self.conv(x) + x
        x = self.ff2(x) + x
        x = self.post_norm(x)
        return x


###########################################################


# === Audio Inputs === #
class GraniteSpeechAudioInputs(TypedDict):
    input_features: torch.Tensor
    """Shape: `TODO`"""


class GraniteSpeechMultiModalProcessingInfo(BaseProcessingInfo):

    def get_supported_mm_limits(self) -> Mapping[str, Optional[int]]:
        # TODO - check if multi-audio is supported.
        return {"audio": None}

    def get_mm_max_tokens_per_item(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
    ) -> Mapping[str, int]:
        # hf_config = self.get_hf_config()
        max_num_audio_tokens = 1000  # TODO calculate me
        return {"audio": max_num_audio_tokens}


class GraniteSpeechMultiModalProcessor(
        BaseMultiModalProcessor[GraniteSpeechMultiModalProcessingInfo]):

    def _get_mm_fields_config(
        self,
        hf_inputs: BatchFeature,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        return dict(input_features=MultiModalFieldConfig.batched("audio"), )

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        out_mm_kwargs: MultiModalKwargs,
    ) -> list[PromptUpdate]:
        # TODO - implement this properly, need to add feature calc
        def get_replacement(item_idx: int):
            return [49155] * 1

        return [
            PromptReplacement(
                modality="audio",
                target=[49155],
                replacement=get_replacement,
            ),
        ]


class GraniteSpeechDummyInputsBuilder(
        BaseDummyInputsBuilder[GraniteSpeechMultiModalProcessingInfo]):

    def get_dummy_processor_inputs(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
    ) -> ProcessorInputs:
        # TODO - calculate this correctly off the feature extractor
        audio_len = 1
        num_audios = mm_counts.get("audio", 0)

        mm_data = {
            "audio":
            self._get_dummy_audios(length=audio_len, num_audios=num_audios)
        }

        return ProcessorInputs(
            # TODO - can we pull this from tokenizer?
            prompt_text="<|audio|>" * num_audios,
            mm_data=mm_data,
        )


# @MULTIMODAL_REGISTRY.register_processor(
#     GraniteSpeechMultiModalProcessor,
#     info=GraniteSpeechMultiModalProcessingInfo,
#     dummy_inputs=GraniteSpeechDummyInputsBuilder)
class GraniteSpeechForConditionalGeneration(
        nn.Module,
        #SupportsMultiModal,
        SupportsPP,
):

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        lora_config = vllm_config.lora_config

        self.config = config
        self.lora_config = lora_config
        self.quant_config = quant_config
        self.sampler = get_sampler()

        # this should be a granite causal LM, but written generically for now
        self.language_model = init_vllm_registered_model(
            vllm_config=vllm_config,
            hf_config=config.text_config,
            prefix=maybe_prefix(prefix, "language_model"),
        )
        hf_config = vllm_config.model_config.hf_config
        self.encoder = GraniteSpeechCTCModel(config=hf_config.encoder_config)
        self.projector = GraniteSpeechEncoderProjectorQFormer(
            config=hf_config.projector_config)

        self.make_empty_intermediate_tensors = (
            self.language_model.make_empty_intermediate_tensors)

    def _parse_and_validate_audio_input(
            self, **kwargs: object) -> Optional[GraniteSpeechAudioInputs]:
        raise NotImplementedError(
            "Audio input parsing and validation not implemented")

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, IntermediateTensors]:
        model_output = self.language_model(input_ids, positions,
                                           intermediate_tensors, inputs_embeds)
        return model_output

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[torch.Tensor]:
        return self.language_model.compute_logits(hidden_states,
                                                  sampling_metadata)

    def sample(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[SamplerOutput]:
        return self.language_model.sample(logits, sampling_metadata)

    def _get_ignore_hacked_ignore_weight_dict(self):
        ## FIXME - there are issues with loading BN, needs to be resolved;
        # the problem is that batch norm puts the tensors in the state
        # dict, but not in the named parameters.
        #
        # For example, in keys:
        # >>> dict(self.encoder.rnn_tr[1].conv.net[5].named_parameters()).keys()
        # we do not have things like num_batches_tracked, which breaks when
        # we export the model and try to reload them.
        print("🚨 Currently we are not loading batch norm stats correctly! 🚨")
        ignore_suffixes = [
            "bias",
            "num_batches_tracked",
            "running_mean",
            "running_var",
        ]
        ignore_layers = []
        for idx1 in range(1, 11):
            for idx2 in range(1, 11):
                for suffix in ignore_suffixes:
                    ignore_layers.append(
                        f"encoder.rnn_tr.{idx1}.conv.net.{idx2}.{suffix}")
        return {"ignore_unexpected_prefixes": ignore_layers}

    def load_weights(self, weights: Iterable[Tuple[str,
                                                   torch.Tensor]]) -> Set[str]:
        extra_kwargs = self._get_ignore_hacked_ignore_weight_dict()
        loader = AutoWeightsLoader(self, **extra_kwargs)
        return loader.load_weights(weights)
