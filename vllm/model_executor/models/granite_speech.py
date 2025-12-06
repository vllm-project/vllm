# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

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
"""Inference-only IBM Granite speech model."""

import math
from collections.abc import Iterable, Mapping
from typing import Annotated, Literal, cast

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from transformers import BatchFeature, PretrainedConfig

from vllm.config import CacheConfig, RendererConfig, SpeechToTextConfig, VllmConfig
from vllm.config.multimodal import BaseDummyOptions
from vllm.inputs.data import PromptType
from vllm.model_executor.layers.linear import ColumnParallelLinear, RowParallelLinear
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.models.module_mapping import MultiModelKeys
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import (
    MultiModalDataDict,
    MultiModalFieldConfig,
    MultiModalKwargsItems,
)
from vllm.multimodal.parse import (
    AudioProcessorItems,
    MultiModalDataItems,
    MultiModalDataParser,
)
from vllm.multimodal.processing import (
    BaseMultiModalProcessor,
    BaseProcessingInfo,
    PromptReplacement,
    PromptUpdate,
)
from vllm.multimodal.profiling import BaseDummyInputsBuilder
from vllm.sequence import IntermediateTensors
from vllm.tokenizers import cached_tokenizer_from_config
from vllm.transformers_utils.processor import cached_processor_from_config
from vllm.utils.tensor_schema import TensorSchema, TensorShape

from .blip2 import Blip2QFormerModel
from .interfaces import (
    MultiModalEmbeddings,
    SupportsLoRA,
    SupportsMultiModal,
    SupportsPP,
    SupportsTranscription,
)
from .utils import AutoWeightsLoader, init_vllm_registered_model, maybe_prefix

# NOTE lang support is based on what is written here:
# https://huggingface.co/ibm-granite/granite-speech-3.3-2b
# Though this may vary from model to model, and also many langs
# work pretty well with zero shot.
ISO639_1_SUPPORTED_LANGS = {
    "en": "English",
    "fr": "French",
    "de": "German",
    "pt": "Portuguese",
    "es": "Spanish",
}


### Audio Input
class GraniteSpeechAudioInputs(TensorSchema):
    """
    Audio input features for Granite Speech model.

    Dimensions:
        - b: Batch size
        - fi: Number of input features from the Mel spectrogram.
        - fo: Number of output features, i.e. the embedding size.
        - 160: Fixed feature dimension for Mel spectrogram features
    """

    input_features: Annotated[torch.Tensor, TensorShape("b", "fi", 160)]
    """Audio input features."""

    input_features_mask: Annotated[torch.Tensor, TensorShape("b", "fo")]
    """Mask for variable length audio features."""

    audio_embed_sizes: Annotated[list[int], TensorShape("b")]
    """List of audio embedding sizes for each item in batch."""


class GraniteSpeechMultiModalProcessingInfo(BaseProcessingInfo):
    def get_supported_mm_limits(self) -> Mapping[str, int | None]:
        return {"audio": 1}

    # There is no limit to the maximum number of audio tokens that can be
    # encoded as features; we pick ~5000 as a number that is probably higher
    # than we would expect to encounter. The sequence of length
    # get_max_audio_len() produces get_max_audio_tokens().
    def get_max_audio_tokens(self):
        return 5001

    def get_max_audio_len(self):
        return 8000000


### Input Processing  & Multimodal utils
class GraniteSpeechMultiModalProcessor(
    BaseMultiModalProcessor[GraniteSpeechMultiModalProcessingInfo]
):
    def _get_data_parser(self) -> MultiModalDataParser:
        feature_extractor = self.info.get_hf_processor().audio_processor
        sampling_rate = feature_extractor.melspec_kwargs["sample_rate"]
        return MultiModalDataParser(target_sr=sampling_rate)

    def _get_mm_fields_config(
        self,
        hf_inputs: BatchFeature,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        return dict(
            input_features=MultiModalFieldConfig.batched("audio"),
            audio_embed_sizes=MultiModalFieldConfig.batched("audio"),
        )

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        out_mm_kwargs: MultiModalKwargsItems,
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
                [audio_length]
            )[0]
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
        tok_kwargs: Mapping[str, object],
    ) -> BatchFeature:
        mm_data = dict(mm_data)
        audios = mm_data.pop("audios", [])

        if audios:
            # GraniteSpeechFeatureExtractor accepts "audio"
            mm_data["audio"] = audios

        processed_outputs = super()._call_hf_processor(
            prompt=prompt,
            mm_data=mm_data,
            mm_kwargs=mm_kwargs,
            tok_kwargs=tok_kwargs,
        )

        if "audio" in mm_data:
            # Calculate the number of audio tokens per entry in the batch;
            # This is used to split the batch back out after padding.
            audio_token_index = self.info.get_hf_config().audio_token_index
            processed_outputs["audio_embed_sizes"] = (
                processed_outputs["input_ids"] == audio_token_index
            ).sum(-1)

        return processed_outputs


class GraniteSpeechDummyInputsBuilder(
    BaseDummyInputsBuilder[GraniteSpeechMultiModalProcessingInfo]
):
    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        mm_options: Mapping[str, BaseDummyOptions] | None = None,
    ) -> MultiModalDataDict:
        num_audios = mm_counts.get("audio", 0)
        audio_overrides = mm_options.get("audio") if mm_options else None

        return {
            "audio": self._get_dummy_audios(
                length=self.info.get_max_audio_len(),
                num_audios=num_audios,
                overrides=audio_overrides,
            )
        }

    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        num_audios = mm_counts.get("audio", 0)
        hf_processor = self.info.get_hf_processor()
        audio_token = getattr(hf_processor, "audio_token", "<|audio|>")
        return audio_token * num_audios


### QFormer Projector
class GraniteSpeechEncoderProjector(nn.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        cache_config: CacheConfig,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        super().__init__()
        self.hidden_size = config.projector_config.hidden_size
        self.downsample_rate = config.downsample_rate
        self.window_size = config.window_size
        self.num_queries = config.window_size // config.downsample_rate

        self.query = nn.Parameter(
            torch.zeros(1, self.num_queries, config.projector_config.hidden_size)
        )

        # NOTE - this is implemented generically in transformers,
        # but for now we create the QFormer model directly since
        # all existing models use this for the projector.
        self.qformer = Blip2QFormerModel(
            config.projector_config,
            quant_config=quant_config,
            cache_config=cache_config,
            prefix=f"{prefix}.qformer",
        )
        self.linear = nn.Linear(
            config.projector_config.hidden_size, config.text_config.hidden_size
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, dim = hidden_states.size()
        nblocks = math.ceil(seq_len / self.window_size)
        pad = nblocks * self.window_size - seq_len
        hidden_states = nn.functional.pad(hidden_states, (0, 0, 0, pad), "constant", 0)
        hidden_states = hidden_states.view(batch_size * nblocks, self.window_size, dim)

        last_hidden_state = self.qformer(
            query_embeds=self.query.data,
            encoder_hidden_states=hidden_states,
        )

        query_proj = self.linear(
            last_hidden_state.view(
                batch_size,
                nblocks * self.window_size // self.downsample_rate,
                -1,
            )
        )
        return query_proj


# Encoder - conformer is adapted from: https://github.com/lucidrains/conformer.git
# NOTE - it would be nice to see if we can align this with other models using
# conformer in vLLM, e.g., phi4mm audio.
class GraniteSpeechConformerFeedForward(nn.Module):
    """Feedforward module for conformer encoder blocks."""

    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        super().__init__()
        self.pre_norm = nn.LayerNorm(config.hidden_dim)

        self.up_proj = ColumnParallelLinear(
            input_size=config.hidden_dim,
            output_size=config.hidden_dim * config.feedforward_mult,
            quant_config=quant_config,
            prefix=f"{prefix}.up_proj",
        )
        self.silu = nn.SiLU()

        self.down_proj = RowParallelLinear(
            input_size=config.hidden_dim * config.feedforward_mult,
            output_size=config.hidden_dim,
            quant_config=quant_config,
            prefix=f"{prefix}.down_proj",
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.pre_norm(hidden_states)
        hidden_states, _ = self.up_proj(hidden_states)
        hidden_states = self.silu(hidden_states)
        hidden_states, _ = self.down_proj(hidden_states)
        return hidden_states


class GraniteSpeechConformerAttention(nn.Module):
    """Attention for conformer blocks using Shaw's relative positional
    embeddings. See the following [paper](https://arxiv.org/pdf/1803.02155)
    for more details.
    """

    def __init__(self, config: PretrainedConfig, prefix: str = ""):
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
        self.rel_pos_emb = nn.Embedding(2 * self.max_pos_emb + 1, self.dim_head)

        if self.context_size <= 0 or self.context_size > self.max_pos_emb:
            raise ValueError(
                "Context size is either less than 0 or exceeds the max_pos_emb"
            )

    def forward(
        self, hidden_states: torch.Tensor, attention_dists: torch.Tensor
    ) -> torch.Tensor:
        hidden_states = self.pre_norm(hidden_states)
        bsz, num_features, _ = hidden_states.shape

        num_blocks = math.ceil(num_features / self.context_size)
        remainder = num_features % self.context_size
        if remainder > 0:
            # right padding to reach block size
            hidden_states = torch.nn.functional.pad(
                hidden_states, (0, 0, 0, self.context_size - remainder)
            )

        # NOTE: would be nice to try to use qkvparallellinear
        # here for this block attention implementation if possible
        query_states = self.to_q(hidden_states)
        key_states, value_states = self.to_kv(hidden_states).chunk(2, dim=-1)

        query_states = query_states.reshape(
            bsz, num_blocks, self.context_size, self.num_heads, -1
        ).transpose(2, 3)
        key_states = key_states.reshape(
            bsz, num_blocks, self.context_size, self.num_heads, -1
        ).transpose(2, 3)
        value_states = value_states.reshape(
            bsz, num_blocks, self.context_size, self.num_heads, -1
        ).transpose(2, 3)

        # shaw's relative positional embedding
        dist = attention_dists.to(hidden_states.device)
        rel_pos_emb = self.rel_pos_emb(dist)
        rel_pos_emb_expanded = rel_pos_emb.view([1, 1, 1] + list(rel_pos_emb.shape))
        pos_attn = (
            torch.sum(query_states.unsqueeze(-2) * rel_pos_emb_expanded, dim=-1)
            * self.scale
        )

        if remainder > 0:
            # masked attention in the extended block
            mask = torch.ones(
                self.context_size,
                self.context_size,
                dtype=bool,
                device=hidden_states.device,
            )
            mask[:remainder, :remainder] = 0
            mask_value = -torch.finfo(pos_attn.dtype).max
            pos_attn[:, -1, :].masked_fill_(mask, mask_value)

        with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
            out = F.scaled_dot_product_attention(
                query_states,
                key_states,
                value_states,
                attn_mask=pos_attn,
                scale=self.scale,
            )
        out = out.transpose(2, 3).reshape(bsz, hidden_states.shape[1], -1)
        return self.to_out(out[:, :num_features, :])


class GraniteSpeechConformerDepthWiseConv1d(nn.Module):
    """Wrapper for padded 1D pointwise convolution."""

    def __init__(self, chan_in: int, chan_out: int, kernel_size: int, prefix: str = ""):
        super().__init__()
        # Padding for the 1D conv is symmetric or close (i.e., offset by one).
        pad = kernel_size // 2
        pad_offset = (kernel_size + 1) % 2
        self.padding = (pad, pad - pad_offset)

        self.conv = nn.Conv1d(
            chan_in, chan_out, kernel_size, groups=chan_in, bias=False
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = F.pad(hidden_states, self.padding)
        return self.conv(hidden_states)


class GraniteSpeechConformerConvModule(nn.Module):
    """Conformer conv module consisting of several 1D/depthwise 1D
    convolutional layers.
    """

    def __init__(self, config: PretrainedConfig, prefix: str = ""):
        super().__init__()
        inner_dim = config.hidden_dim * config.conv_expansion_factor

        self.norm = nn.LayerNorm(config.hidden_dim)
        self.up_conv = nn.Conv1d(config.hidden_dim, inner_dim * 2, 1)
        self.glu = nn.GLU(dim=1)
        self.depth_conv = GraniteSpeechConformerDepthWiseConv1d(
            inner_dim,
            inner_dim,
            kernel_size=config.conv_kernel_size,
            prefix=f"{prefix}.depth_conv",
        )
        self.silu = nn.SiLU()
        self.batch_norm = nn.BatchNorm1d(inner_dim)
        self.down_conv = nn.Conv1d(inner_dim, config.hidden_dim, 1)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.norm(hidden_states)
        hidden_states = self.up_conv(hidden_states.permute(0, 2, 1))
        hidden_states = self.glu(hidden_states)
        hidden_states = self.depth_conv(hidden_states)
        hidden_states = self.silu(self.batch_norm(hidden_states))
        hidden_states = self.down_conv(hidden_states).permute(0, 2, 1)
        return hidden_states


class GraniteSpeechConformerBlock(nn.Module):
    """Conformer block, consisting largely of linear layers,
    attention, and convolutional layers."""

    def __init__(self, config: PretrainedConfig, prefix: str = ""):
        super().__init__()
        self.ff1 = GraniteSpeechConformerFeedForward(config, prefix=f"{prefix}.ff1")
        self.attn = GraniteSpeechConformerAttention(config, prefix=f"{prefix}.attn")
        self.conv = GraniteSpeechConformerConvModule(config, prefix=f"{prefix}.conv")
        self.ff2 = GraniteSpeechConformerFeedForward(config, prefix=f"{prefix}.ff2")
        self.post_norm = nn.LayerNorm(config.hidden_dim)

    def forward(
        self, hidden_states: torch.Tensor, attention_dists: torch.Tensor
    ) -> torch.Tensor:
        hidden_states = 0.5 * self.ff1(hidden_states) + hidden_states
        hidden_states = (
            self.attn(hidden_states, attention_dists=attention_dists) + hidden_states
        )
        hidden_states = self.conv(hidden_states) + hidden_states
        hidden_states = 0.5 * self.ff2(hidden_states) + hidden_states
        hidden_states = self.post_norm(hidden_states)
        return hidden_states


class GraniteSpeechCTCEncoder(nn.Module):
    """CTC Encoder comprising conformer blocks and additional linear layers."""

    def __init__(
        self,
        config: PretrainedConfig,
        prefix: str,
        quant_config: QuantizationConfig | None = None,
    ):
        super().__init__()
        self.config = config

        # Precompute clamped relative positional encoding distances
        seq = torch.arange(config.context_size)
        relpos_dist = seq.view(-1, 1) - seq.view(1, -1)
        self.attention_dists = (
            torch.clamp(relpos_dist, -config.context_size, config.context_size)
            + config.max_pos_emb
        )

        self.input_linear = nn.Linear(config.input_dim, config.hidden_dim, bias=True)
        self.layers = nn.ModuleList(
            [
                GraniteSpeechConformerBlock(
                    config,
                    prefix=f"{prefix}.layers.{idx}",
                )
                for idx in range(config.num_layers)
            ]
        )

        self.out = ColumnParallelLinear(
            input_size=config.hidden_dim,
            output_size=config.output_dim,
            bias=True,
            quant_config=quant_config,
            prefix=f"{prefix}.out",
        )

        self.out_mid = RowParallelLinear(
            input_size=config.output_dim,
            output_size=config.hidden_dim,
            bias=True,
            quant_config=quant_config,
            prefix=f"{prefix}.out_mid",
        )
        self.softmax = nn.Softmax(dim=-1)
        self.num_layers = config.num_layers

    def forward(self, hidden_states: torch.Tensor):
        hidden_states = self.input_linear(hidden_states)
        for idx, layer in enumerate(self.layers, start=1):
            hidden_states = layer(hidden_states, attention_dists=self.attention_dists)

            if idx == self.num_layers // 2:
                hidden_states_mid = hidden_states.clone()
                hidden_states_mid, _ = self.out(hidden_states_mid)
                hidden_states_mid = self.softmax(hidden_states_mid)
                hidden_states_mid, _ = self.out_mid(hidden_states_mid)
                hidden_states += hidden_states_mid
        return hidden_states


@MULTIMODAL_REGISTRY.register_processor(
    GraniteSpeechMultiModalProcessor,
    info=GraniteSpeechMultiModalProcessingInfo,
    dummy_inputs=GraniteSpeechDummyInputsBuilder,
)
class GraniteSpeechForConditionalGeneration(
    nn.Module,
    SupportsMultiModal,
    SupportsPP,
    SupportsLoRA,
    SupportsTranscription,
):
    supported_languages = ISO639_1_SUPPORTED_LANGS

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

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None:
        if modality.startswith("audio"):
            return "<|audio|>"

        raise ValueError("Only audio modality is supported")

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        cache_config = vllm_config.cache_config

        self.config = config
        self.quant_config = quant_config
        self.cache_config = cache_config

        # The language model is typically a Granite LLM
        self.language_model = init_vllm_registered_model(
            vllm_config=vllm_config,
            hf_config=config.text_config,
            prefix=maybe_prefix(prefix, "language_model"),
        )

        # Conformer encoder
        self.encoder = GraniteSpeechCTCEncoder(
            config=config.encoder_config,
            quant_config=quant_config,
            prefix=f"{prefix}.encoder",
        )

        # Blip2 QFormer
        self.projector = GraniteSpeechEncoderProjector(
            config=config,
            quant_config=quant_config,
            cache_config=cache_config,
            prefix=f"{prefix}.projector",
        )

        self.make_empty_intermediate_tensors = (
            self.language_model.make_empty_intermediate_tensors
        )

    def _parse_and_validate_audio_input(
        self,
        **kwargs: object,
    ) -> GraniteSpeechAudioInputs | None:
        input_features = kwargs.pop("input_features", None)
        input_features_mask = kwargs.pop("input_features_mask", None)
        audio_embed_sizes = kwargs.pop("audio_embed_sizes", None)

        if input_features is None:
            return None

        # If we have a batch of variable feature length audio clips, we need
        # to mask the features; usually we would get an input_features_mask
        # from the processor, but we handle rebuilding it here since
        # vLLM generally processes everything independently + batches.
        if input_features_mask is None:
            input_features_mask = self._build_input_features_mask(audio_embed_sizes)

        if not isinstance(input_features, (torch.Tensor, list)):
            raise ValueError(
                "Incorrect type of audio input features. "
                f"Got type: {type(input_features)}"
            )

        if input_features_mask is not None and not isinstance(
            input_features_mask, torch.Tensor
        ):
            raise ValueError(
                "Incorrect type of audio input features mask. "
                f"Got type: {type(input_features_mask)}"
            )

        if isinstance(input_features, torch.Tensor):
            # Granite speech currently only allows one audio token per instance
            # and features are already unsqueezed in the processor, so one
            # instance will have shape [1, {num_features}, 160]. As such,
            # input features will usually be of shape
            # [bsz, 1, num_features, 160], which we squeeze to be 3D here.
            if len(input_features.shape) == 4:
                input_features = input_features.squeeze(1)
            if len(input_features.shape) != 3:
                raise ValueError(
                    "Squeezed input features should be 3D but are of shape "
                    f"{input_features.shape}"
                )
            input_features = input_features.to(self.encoder.input_linear.weight.dtype)

        else:
            # Otherwise we have a list of tensors, which are almost certainly
            # differing in their respective numbers of audio features;
            # stack them into a 3D tensor of size [bsz, most_num_features, 160].
            input_features = self._pad_and_stack_input_features(
                input_features,
            ).to(self.encoder.input_linear.weight.dtype)

        return GraniteSpeechAudioInputs(
            input_features=input_features,
            input_features_mask=input_features_mask,
            audio_embed_sizes=audio_embed_sizes.flatten().tolist(),
        )

    def _build_input_features_mask(
        self,
        audio_embed_sizes: torch.Tensor,
    ) -> torch.Tensor:
        """Calculate the input features mask, which will generally be used
        to mask the padded features for all entries in the batch except
        for those with the most audio features.

        Args:
            audio_embed_sizes: torch.Tensor
                Tensor of num features in each seq in the batch.
        Returns:
            torch.Tensor: Mask of shape (bsz, num_features) to be applied to
            the audio features prior to splitting the audio embeddings.
        """
        most_audio_features = torch.max(audio_embed_sizes).item()
        mask_indices = torch.arange(
            most_audio_features,
            device=audio_embed_sizes.device,
        ).view(1, -1)
        input_features_mask = mask_indices < audio_embed_sizes.view(-1, 1)
        return input_features_mask

    def _pad_and_stack_input_features(
        self,
        input_features: list[torch.Tensor],
    ) -> torch.Tensor:
        """Given a list of input features of varying length, pad them to the
        same length and stack them into a torch.Tensor.

        NOTE: Usually, padding is done in the input processor/feature extractor
        and zero padded prior to the computation of the Mel features; the
        resulting values are only constant within a batch and generally nonzero
        (i.e., slightly negative nums); we should validate that this is okay
        since we don't use a feature attention mask, but the more important
        thing is that we apply the input_features_mask with variable len
        batches.

        Args:
            input_features: list[torch.Tensor]
                Input features to be coerced into a tensor.
        Returns:
            torch.Tensor: Tensor of shape [bsz, num_features, 160], where
            num_features is the max number of features of any entry in the
            batch.
        """
        # Input features are of shape [bsz, num_features, 160]
        feat_lens = [feats.shape[1] for feats in input_features]
        padding = [max(feat_lens) - length for length in feat_lens]
        # TODO (Alex) - Validate that it's okay to zero pad like this;
        # in transformers we zero pad prior to calculating the speech features,
        # so the value is not zero and is dependent on the batched features.
        padded = [
            torch.nn.functional.pad(feats, (0, 0, 0, pad, 0, 0))
            for feats, pad in zip(input_features, padding)
        ]
        stacked_features = torch.cat(padded, dim=0).to(input_features[0])
        return stacked_features

    def _process_audio_input(
        self,
        audio_input: GraniteSpeechAudioInputs,
    ) -> tuple[torch.Tensor]:
        """Compute the audio features to be merged into the LLM embeddings.

        Args:
            audio_input: GraniteSpeechAudioInputs
                Audio inputs object containing Mel features, an input features
                mask, and the (flattened) number of audio tokens per instance.
        Returns:
            tuple[torch.Tensor]: List of length bsz.
        """
        # TODO (Alex) - support embedding inputs
        encoder_embeds = self.encoder(audio_input["input_features"])
        # [bsz, <max feature size>, 4096]
        projected_embeds = self.projector(encoder_embeds)
        # Apply mask on variable length audio features
        masked_embeds = projected_embeds[audio_input["input_features_mask"]]
        # Split variable length features into a tuple
        return torch.split(masked_embeds, audio_input["audio_embed_sizes"])

    def get_language_model(self) -> torch.nn.Module:
        return self.language_model

    def embed_multimodal(
        self,
        **kwargs: object,
    ) -> MultiModalEmbeddings:
        """Compute the audio embeddings if audio inputs are present."""
        audio_input = self._parse_and_validate_audio_input(**kwargs)
        if audio_input is None:
            return []

        audio_features = self._process_audio_input(audio_input)
        return audio_features

    def embed_input_ids(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: MultiModalEmbeddings | None = None,
        *,
        is_multimodal: torch.Tensor | None = None,
        # Multi-modal token ID may exceed vocab size
        handle_oov_mm_token: bool = True,
    ) -> torch.Tensor:
        # This is to satisfy the type checker for each overload
        if multimodal_embeddings is None or is_multimodal is None:
            return super().embed_input_ids(input_ids)

        return super().embed_input_ids(
            input_ids,
            multimodal_embeddings=multimodal_embeddings,
            is_multimodal=is_multimodal,
            handle_oov_mm_token=handle_oov_mm_token,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: object,
    ) -> torch.Tensor | IntermediateTensors:
        if intermediate_tensors is not None:
            inputs_embeds = None

        model_output = self.language_model(
            input_ids, positions, intermediate_tensors, inputs_embeds
        )
        return model_output

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor | None:
        return self.language_model.compute_logits(hidden_states)

    def load_weights(
        self,
        weights: Iterable[tuple[str, torch.Tensor]],
    ) -> set[str]:
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights)

    def get_mm_mapping(self) -> MultiModelKeys:
        """Get the module prefix in multimodal models."""
        return MultiModelKeys.from_string_field(
            language_model="language_model",
            connector="projector",
            tower_model="encoder",
        )

    ### Support for speech-to-text Transcription
    @classmethod
    def get_generation_prompt(
        cls,
        audio: np.ndarray,
        renderer_config: RendererConfig,
        stt_config: SpeechToTextConfig,
        language: str | None,
        task_type: Literal["transcribe", "translate"],
        request_prompt: str,
        to_language: str | None,
    ) -> PromptType:
        """Get the generation prompt to be used for transcription requests."""
        # Audio placeholders don't use an index, so value doesn't matter
        audio_tok = cls.get_placeholder_str("audio", 0)

        if task_type == "translate":
            full_lang_name_to = cls.supported_languages.get(to_language, to_language)
            user_prompt = f"{audio_tok}translate the speech to {full_lang_name_to}"  # noqa: E501
        elif task_type == "transcribe":
            user_prompt = (
                f"{audio_tok}can you transcribe the speech into a written format?"  # noqa: E501
            )
        else:
            raise ValueError(f"Unsupported task type {task_type}")

        tokenizer = cached_tokenizer_from_config(renderer_config)
        chat = [dict(role="user", content=user_prompt)]
        prompt = tokenizer.apply_chat_template(
            chat,
            tokenize=False,
            add_generation_prompt=True,
        )

        prompt_token_ids = tokenizer.encode(prompt)
        prompt = {
            "prompt_token_ids": prompt_token_ids,
            "multi_modal_data": {"audio": audio},
        }
        return cast(PromptType, prompt)

    # Adapted from https://github.com/huggingface/transformers/blob/v4.56.0/src/transformers/models/granite_speech/feature_extraction_granite_speech.py#L122 # noqa: E501
    @classmethod
    def get_num_audio_tokens(
        cls,
        audio_duration_s: float,
        stt_config: SpeechToTextConfig,
        renderer_config: RendererConfig,
    ) -> int | None:
        """Get the number of audio tokens for an audio duration in sec."""
        processor = cached_processor_from_config(renderer_config)
        hop_length = processor.audio_processor.melspec_kwargs["hop_length"]
        proj_win_size = processor.audio_processor.projector_window_size
        ds_rate = processor.audio_processor.projector_downsample_rate
        effective_window_size = proj_win_size // ds_rate

        raw_length = audio_duration_s * stt_config.sample_rate

        # mel sequence length computation
        mel_length = raw_length // hop_length + 1
        # encoder frame takes two mel features
        encoder_length = mel_length // 2
        nblocks = math.ceil(encoder_length / proj_win_size)
        # projector output length
        return nblocks * effective_window_size

    @classmethod
    def get_speech_to_text_config(
        cls,
        renderer_config: RendererConfig,
        task_type: str,
    ) -> SpeechToTextConfig:
        """Get the stt config for this model."""
        # Default settings are reasonable for this model and we don't currently
        # expose this information in the model configs, but this may change in
        # the future
        return SpeechToTextConfig()
