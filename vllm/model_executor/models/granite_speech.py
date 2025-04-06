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
from torch import nn
from transformers import BatchFeature

from vllm.config import VllmConfig
from vllm.model_executor.layers.sampler import SamplerOutput, get_sampler
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import MultiModalFieldConfig, MultiModalKwargs
from vllm.multimodal.parse import (AudioProcessorItems, MultiModalDataItems,
                                   MultiModalDataParser)
from vllm.multimodal.processing import (BaseMultiModalProcessor,
                                        BaseProcessingInfo, PromptReplacement,
                                        PromptUpdate)
from vllm.multimodal.profiling import BaseDummyInputsBuilder, ProcessorInputs
from vllm.sequence import IntermediateTensors

from .blip2 import Blip2QFormerModel
from .interfaces import (MultiModalEmbeddings, SupportsLoRA,
                         SupportsMultiModal, SupportsPP)
from .utils import (AutoWeightsLoader, embed_multimodal,
                    init_vllm_registered_model, maybe_prefix)


class GraniteSpeechEncoderProjector(nn.Module):

    def __init__(self, config, quant_config, cache_config):
        super().__init__()
        self.hidden_size = config.projector_config.hidden_size
        self.downsample_rate = config.downsample_rate
        self.window_size = config.window_size
        self.num_queries = config.window_size // config.downsample_rate

        self.query = nn.Parameter(
            torch.zeros(1, self.num_queries,
                        config.projector_config.hidden_size))
        self.query.data.normal_(mean=0.0, std=1.0)

        self.qformer = Blip2QFormerModel(
            config.projector_config,
            quant_config=quant_config,
            cache_config=cache_config,
        )
        self.linear = nn.Linear(config.projector_config.hidden_size,
                                config.text_config.hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, dim = hidden_states.size()
        nblocks = math.ceil(seq_len / self.window_size)
        pad = nblocks * self.window_size - seq_len
        hidden_states = nn.functional.pad(hidden_states, (0, 0, 0, pad),
                                          "constant", 0)
        hidden_states = hidden_states.view(batch_size * nblocks,
                                           self.window_size, dim)

        query_output = self.qformer(
            query_embeds=self.query.data,
            encoder_hidden_states=hidden_states,
            # encoder_attention_mask=None,
            # return_dict=True,
        )
        last_hidden_state = query_output
        query_proj = self.linear(
            last_hidden_state.view(
                batch_size, nblocks * self.window_size // self.downsample_rate,
                -1))
        return query_proj


### Encoder - conformer is adapted from: https://github.com/lucidrains/conformer.git
class GraniteSpeechCTCEncoder(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.input_linear = nn.Linear(config.input_dim,
                                      config.hidden_dim,
                                      bias=True)
        self.layers = nn.ModuleList([
            GraniteSpeechConformerBlock(config)
            for _ in range(config.num_layers)
        ])

        self.out = nn.Linear(config.hidden_dim, config.output_dim, bias=True)
        self.out_mid = nn.Linear(config.output_dim,
                                 config.hidden_dim,
                                 bias=True)
        self.num_layers = config.num_layers

    def forward(self, hidden_states: torch.Tensor):
        hidden_states = self.input_linear(hidden_states)
        for idx, layer in enumerate(self.layers, start=1):
            hidden_states = layer(hidden_states)
            if idx == self.num_layers // 2:
                hidden_states_mid = hidden_states.clone()
                hidden_states_mid = self.out(hidden_states_mid)
                hidden_states += self.out_mid(
                    nn.Softmax(dim=-1)(hidden_states_mid))
        return hidden_states


class GraniteSpeechConformerBlock(nn.Module):
    """Conformer block, consisting largely of linear layers,
    attention, and convolutional layers."""

    def __init__(self, config):
        super().__init__()
        self.ff1 = GraniteSpeechConformerFeedForward(config)
        self.attn = GraniteSpeechConformerAttention(config)
        self.conv = GraniteSpeechConformerConvModule(config)
        self.ff2 = GraniteSpeechConformerFeedForward(config)
        self.post_norm = nn.LayerNorm(config.hidden_dim)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = 0.5 * self.ff1(hidden_states) + hidden_states
        hidden_states = self.attn(hidden_states) + hidden_states
        hidden_states = self.conv(hidden_states) + hidden_states
        hidden_states = 0.5 * self.ff2(hidden_states) + hidden_states
        hidden_states = self.post_norm(hidden_states)
        return hidden_states


class GraniteSpeechConformerFeedForward(nn.Module):
    """Feedforward module for conformer encoder blocks."""

    def __init__(self, config):
        super().__init__()
        self.pre_norm = nn.LayerNorm(config.hidden_dim)
        self.up_proj = nn.Linear(config.hidden_dim,
                                 config.hidden_dim * config.feedforward_mult)
        self.silu = nn.SiLU()
        self.dropout = nn.Dropout(config.dropout)
        self.down_proj = nn.Linear(config.hidden_dim * config.feedforward_mult,
                                   config.hidden_dim)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.pre_norm(hidden_states)
        hidden_states = self.up_proj(hidden_states)
        hidden_states = self.dropout(self.silu(hidden_states))
        hidden_states = self.down_proj(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states


class GraniteSpeechConformerAttention(nn.Module):
    """Attention for conformer blocks with shaw's relpos embeddings."""

    def __init__(self, config):
        super().__init__()

        inner_dim = config.dim_head * config.num_heads
        self.max_pos_emb = config.max_pos_emb
        self.context_size = config.context_size
        self.num_heads = config.num_heads
        self.dim_head = config.dim_head
        self.scale = self.dim_head**-0.5
        self.pre_norm = nn.LayerNorm(config.hidden_dim)
        self.to_q = nn.Linear(config.hidden_dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(config.hidden_dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, config.hidden_dim)
        self.rel_pos_emb = nn.Embedding(2 * self.max_pos_emb + 1,
                                        self.dim_head)
        self.dropout = nn.Dropout(config.dropout)

        if self.context_size <= 0 or self.context_size > self.max_pos_emb:
            raise ValueError(
                "Context size is either less than 0 or exceeds the max_pos_emb"
            )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.pre_norm(hidden_states)
        bsz, num_features, _ = hidden_states.shape

        num_blocks = math.ceil(num_features / self.context_size)
        remainder = num_features % self.context_size
        if remainder > 0:
            # right padding to reach block size
            hidden_states = torch.nn.functional.pad(
                hidden_states, (0, 0, 0, self.context_size - remainder))

        query_states = self.to_q(hidden_states)
        key_states, value_states = self.to_kv(hidden_states).chunk(2, dim=-1)
        query_states, key_states, value_states = [
            t.reshape(bsz, num_blocks, self.context_size, self.num_heads,
                      -1).transpose(2, 3)
            for t in (query_states, key_states, value_states)
        ]

        # shaw's relative positional embedding
        seq = torch.arange(self.context_size, device=hidden_states.device)
        dist = seq.view(-1, 1) - seq.view(1, -1)
        dist = torch.clamp(dist, -self.context_size,
                           self.context_size) + self.max_pos_emb
        rel_pos_emb = self.rel_pos_emb(dist).to(query_states)
        rel_pos_emb_expanded = rel_pos_emb.view([1, 1, 1] +
                                                list(rel_pos_emb.shape))
        pos_attn = torch.sum(query_states.unsqueeze(-2) * rel_pos_emb_expanded,
                             dim=-1) * self.scale

        if remainder > 0:
            # masked attention in the extended block
            mask = torch.ones(self.context_size,
                              self.context_size,
                              dtype=bool,
                              device=hidden_states.device)
            mask[:remainder, :remainder] = 0
            mask_value = -torch.finfo(pos_attn.dtype).max
            pos_attn[:, -1, :].masked_fill_(mask, mask_value)

        with torch.nn.attention.sdpa_kernel(
                torch.nn.attention.SDPBackend.MATH):
            out = F.scaled_dot_product_attention(query_states,
                                                 key_states,
                                                 value_states,
                                                 attn_mask=pos_attn,
                                                 scale=self.scale)
        out = out.transpose(2, 3).reshape(bsz, hidden_states.shape[1], -1)
        out = self.to_out(out[:, :num_features, :])
        return self.dropout(out)


class GraniteSpeechConformerConvModule(nn.Module):
    """Conformer conv module consisting of several 1D/depthwise 1D
    convolutional layers."""

    def __init__(self, config):
        super().__init__()
        inner_dim = config.hidden_dim * config.conv_expansion_factor
        padding = self.calc_same_padding(config.conv_kernel_size)

        self.norm = nn.LayerNorm(config.hidden_dim)
        self.up_conv = nn.Conv1d(config.hidden_dim, inner_dim * 2, 1)
        self.glu = nn.GLU(dim=1)
        self.depth_conv = GraniteSpeechConformerDepthWiseConv1d(
            inner_dim,
            inner_dim,
            kernel_size=config.conv_kernel_size,
            padding=padding)
        self.silu = nn.SiLU()
        self.batch_norm = nn.BatchNorm1d(inner_dim)
        self.down_conv = nn.Conv1d(inner_dim, config.hidden_dim, 1)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.norm(hidden_states)
        hidden_states = self.up_conv(hidden_states.permute(0, 2, 1))
        hidden_states = self.glu(hidden_states)
        hidden_states = self.depth_conv(hidden_states)
        hidden_states = self.silu(self.batch_norm(hidden_states))
        hidden_states = self.down_conv(hidden_states).permute(0, 2, 1)
        hidden_states = self.dropout(hidden_states)
        return hidden_states

    @staticmethod
    def calc_same_padding(kernel_size: int) -> Tuple[int, int]:
        """Calculates symmetric padding for the depthwise 1D convolution."""
        pad = kernel_size // 2
        return (pad, pad - (kernel_size + 1) % 2)


class GraniteSpeechConformerDepthWiseConv1d(nn.Module):
    """Wrapper for padded 1D pointwise convolution."""

    def __init__(self, chan_in: int, chan_out: int, kernel_size: int,
                 padding: Tuple[int, int]):
        super().__init__()
        self.padding = padding
        self.conv = nn.Conv1d(chan_in,
                              chan_out,
                              kernel_size,
                              groups=chan_in,
                              bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = F.pad(hidden_states, self.padding)
        return self.conv(hidden_states)


# === Audio Inputs === #
class GraniteSpeechAudioInputs(TypedDict):
    input_features: torch.Tensor
    input_features_mask: torch.Tensor
    """Shape: `TODO`"""


class GraniteSpeechMultiModalProcessingInfo(BaseProcessingInfo):

    def get_supported_mm_limits(self) -> Mapping[str, Optional[int]]:
        # For now, we only allow one audio per input for granite speech models.
        return {"audio": 1}

    def get_mm_max_tokens_per_item(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
    ) -> Mapping[str, int]:
        return {"audio": self.get_max_audio_tokens()}

    # There is no limit to the maximum number of audio tokens that can be
    # encoded as features; we pick ~5000 as a number that is probably higher
    # than we would expect to encounter. The sequence of length
    # get_max_audio_len() produces get_max_audio_tokens().
    def get_max_audio_tokens(self):
        return 5001

    def get_max_audio_len(self):
        return 8000000


class GraniteSpeechMultiModalProcessor(
        BaseMultiModalProcessor[GraniteSpeechMultiModalProcessingInfo]):

    def _get_data_parser(self) -> MultiModalDataParser:
        feature_extractor = self.info.get_hf_processor().audio_processor
        sampling_rate = feature_extractor.melspec_kwargs["sample_rate"]
        return MultiModalDataParser(target_sr=sampling_rate)

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
        processor = self.info.get_hf_processor(**hf_processor_mm_kwargs)
        tokenizer = self.info.get_tokenizer()
        feature_extractor = processor.audio_processor
        vocab = tokenizer.get_vocab()

        # Use getattr with default to be compatible with transformers<4.48
        audio_token = getattr(processor, "audio_token", "<|audio|>")
        audio_token_id = vocab[audio_token]

        def get_replacement(item_idx: int):
            audios = mm_items.get_items("audio", AudioProcessorItems)
            audio = audios.get(item_idx)
            audio_length = audio.shape[-1]
            num_projector_features = feature_extractor._get_num_audio_features(
                [audio_length])[0]
            return [audio_token_id] * num_projector_features

        return [
            PromptReplacement(
                modality="audio",
                target=[audio_token_id],
                replacement=get_replacement,
            )
        ]

    def _call_hf_processor(
        self,
        prompt: str,
        mm_data: Mapping[str, object],
        mm_kwargs: Mapping[str, object],
    ) -> BatchFeature:
        res = super()._call_hf_processor(
            prompt=prompt,
            mm_data=mm_data,
            mm_kwargs=mm_kwargs,
        )
        return res


class GraniteSpeechDummyInputsBuilder(
        BaseDummyInputsBuilder[GraniteSpeechMultiModalProcessingInfo]):

    def get_dummy_processor_inputs(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
    ) -> ProcessorInputs:
        num_audios = mm_counts.get("audio", 0)
        mm_data = {
            "audio":
            self._get_dummy_audios(
                length=self.info.get_max_audio_len(),
                num_audios=num_audios,
            )
        }

        hf_processor = self.info.get_hf_processor()
        audio_token = getattr(hf_processor, "audio_token", "<|audio|>")

        return ProcessorInputs(
            prompt_text=audio_token * num_audios,
            mm_data=mm_data,
        )


@MULTIMODAL_REGISTRY.register_processor(
    GraniteSpeechMultiModalProcessor,
    info=GraniteSpeechMultiModalProcessingInfo,
    dummy_inputs=GraniteSpeechDummyInputsBuilder)
class GraniteSpeechForConditionalGeneration(
        nn.Module,
        SupportsMultiModal,
        SupportsPP,
        SupportsLoRA,
):

    packed_modules_mapping = {
        "qkv_proj": [
            "q_proj",
            "k_proj",
            "v_proj",
        ],
        "gate_up_proj": [
            "gate_proj",
            "up_proj",
        ],
    }

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        cache_config = vllm_config.cache_config

        self.config = config
        self.quant_config = quant_config
        self.cache_config = cache_config
        self.sampler = get_sampler()

        # this should be a granite causal LM, but written generically for now
        self.language_model = init_vllm_registered_model(
            vllm_config=vllm_config,
            hf_config=config.text_config,
            prefix=maybe_prefix(prefix, "language_model"),
        )
        hf_config = vllm_config.model_config.hf_config
        self.encoder = GraniteSpeechCTCEncoder(config=hf_config.encoder_config)
        self.projector = GraniteSpeechEncoderProjector(
            config=hf_config,
            quant_config=quant_config,
            cache_config=cache_config,
        )

        self.make_empty_intermediate_tensors = (
            self.language_model.make_empty_intermediate_tensors)

    def _parse_and_validate_audio_input(
            self, **kwargs: object) -> Optional[GraniteSpeechAudioInputs]:
        input_features = kwargs.pop('input_features', None)
        input_features_mask = kwargs.pop('input_features_mask', None)
        if input_features is None:
            return None

        if not isinstance(input_features, (torch.Tensor, list)):
            raise ValueError("Incorrect type of audio input features. "
                             f"Got type: {type(input_features)}")

        if input_features_mask and not isinstance(input_features_mask,
                                                  (torch.Tensor, list)):
            raise ValueError("Incorrect type of audio input features mask. "
                             f"Got type: {type(input_features_mask)}")

        return GraniteSpeechAudioInputs(
            input_features=input_features,
            input_features_mask=input_features_mask,
        )

    def _process_audio_input(self, audio_input: GraniteSpeechAudioInputs):
        # TODO - probably should handle audio embeddings
        # in addition to raw audio data, but for now we don't
        # TODO - handle the features mask
        # TODO - fix dtype hacking here
        # TODO - fix squeezed dim here - seems like something somewhere
        # may not be stackin properly / is creating an extra dimension
        # unnecessarily (probably the view() in the processor)
        #           should be 1, 50000, 160
        input_features = audio_input["input_features"].to(
            torch.bfloat16).squeeze(0)
        encoder_embeds = self.encoder(input_features)
        projected_embeds = self.projector(encoder_embeds)
        return projected_embeds

    def get_multimodal_embeddings(
            self, **kwargs: object) -> Optional[MultiModalEmbeddings]:
        audio_input = self._parse_and_validate_audio_input(**kwargs)
        if audio_input is None:
            return None
        audio_features = self._process_audio_input(audio_input)
        return audio_features

    def get_input_embeddings(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: Optional[MultiModalEmbeddings] = None,
    ) -> torch.Tensor:
        if multimodal_embeddings is None:
            return self.language_model.get_input_embeddings(input_ids)

        inputs_embeds = embed_multimodal(
            input_ids,
            self.config.audio_token_index,
            self.language_model.model.get_input_embeddings,
            multimodal_embeddings,
        )
        return inputs_embeds

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        **kwargs: object,
    ) -> Union[torch.Tensor, IntermediateTensors]:
        if intermediate_tensors is not None:
            inputs_embeds = None

        # NOTE: In v1, inputs_embeds is always generated at model runner, this
        # condition is for v0 compatibility.
        elif inputs_embeds is None:
            audio_embeds = self.get_multimodal_embeddings(**kwargs)
            inputs_embeds = self.get_input_embeddings(input_ids, audio_embeds)
            input_ids = None

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

    def load_weights(self, weights: Iterable[Tuple[str,
                                                   torch.Tensor]]) -> Set[str]:
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights)
