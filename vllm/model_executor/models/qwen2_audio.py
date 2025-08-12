# Copyright 2024 The Qwen team.
# Copyright 2023 The vLLM team.
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
"""Inference-only Qwen2-Audio model compatible with HuggingFace weights."""
from functools import lru_cache
from typing import Iterable, List, Mapping, Optional, Tuple, TypedDict, Union

import librosa
import numpy as np
import torch
import torch.nn as nn
from transformers import Qwen2AudioEncoder

from vllm.attention import AttentionMetadata
from vllm.config import VllmConfig
from vllm.inputs import (INPUT_REGISTRY, DecoderOnlyInputs, DummyData,
                         InputContext, token_inputs)
from vllm.logger import init_logger
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.sampler import SamplerOutput, get_sampler
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader, maybe_remap_kv_scale_name)
from vllm.model_executor.models.qwen2 import Qwen2Model
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.multimodal import MULTIMODAL_REGISTRY, MultiModalKwargs
from vllm.multimodal.utils import consecutive_placeholder_ranges
from vllm.sequence import IntermediateTensors, SequenceData

from .interfaces import SupportsMultiModal, SupportsPP

logger = init_logger(__name__)

_KEYS_TO_MODIFY_MAPPING = {
    "language_model.lm_head": "lm_head",
    "language_model.model": "language_model",
}


# # === Audio Inputs === #
class Qwen2AudioInputs(TypedDict):
    input_features: torch.Tensor
    """Shape: 
    `(num_audios, num_mel_bins, 3000)`
    """

    feature_attention_mask: torch.Tensor
    """Shape: `(num_audios, 3000)`
    """


# === Audio Encoder === #


class Qwen2AudioMultiModalProjector(nn.Module):

    def __init__(self, audio_hidden_size: int, text_hidden_size: int):
        super().__init__()
        self.linear = nn.Linear(audio_hidden_size, text_hidden_size, bias=True)

    def forward(self, audio_features):
        hidden_states = self.linear(audio_features)
        return hidden_states


def dummy_data_for_qwen2_audio(ctx: InputContext, seq_len: int,
                               mm_counts: Mapping[str, int]):
    num_audios = mm_counts["audio"]
    max_tokens_per_audio = get_max_qwen2_audio_audio_tokens(ctx)
    max_llm_audio_tokens = max_tokens_per_audio * num_audios
    if seq_len - max_llm_audio_tokens - 2 < 0:
        raise RuntimeError(
            f"Qwen2-Audio cannot process {num_audios} audios in a prompt, "
            "please increase max_model_len or reduce audio limit by "
            "--limit-mm-per-prompt.")

    audio_token_index = ctx.model_config.hf_config.audio_token_index

    dummy_seqdata = SequenceData.from_prompt_token_counts(
        (audio_token_index, max_llm_audio_tokens),
        (0, seq_len - max_llm_audio_tokens),
    )
    dummy_audio = np.full((max_llm_audio_tokens * 2 * 2 * 160, ), 0.)
    return DummyData(
        dummy_seqdata, {"audio": [(dummy_audio, 16000)] * num_audios}, {
            "audio":
            consecutive_placeholder_ranges(num_items=num_audios,
                                           item_size=max_tokens_per_audio)
        })


def get_processor(
    processor_name: str,
    *args,
    trust_remote_code: bool = False,
    **kwargs,
):
    """Gets a processor for the given model name via HuggingFace.

    Derived from `vllm.transformers_utils.image_processor.get_image_processor`.
    """
    # don't put this import at the top level
    # it will call torch.cuda.device_count()
    from transformers import AutoProcessor

    try:
        processor = AutoProcessor.from_pretrained(
            processor_name,
            *args,
            trust_remote_code=trust_remote_code,
            **kwargs)
    except ValueError as e:
        # If the error pertains to the processor class not existing or not
        # currently being imported, suggest using the --trust-remote-code flag.
        # Unlike AutoTokenizer, AutoProcessor does not separate such errors
        if not trust_remote_code:
            err_msg = (
                "Failed to load the processor. If the processor is "
                "a custom processor not yet available in the HuggingFace "
                "transformers library, consider setting "
                "`trust_remote_code=True` in LLM or using the "
                "`--trust-remote-code` flag in the CLI.")
            raise RuntimeError(err_msg) from e
        else:
            raise e

    return processor


cached_get_processor = lru_cache(get_processor)


def _get_feat_extract_output_lengths(input_lengths: torch.LongTensor):
    """
    Computes the output length of the convolutional layers
    and the output length of the audio encoder
    """
    input_lengths = (input_lengths - 1) // 2 + 1
    output_lengths = (input_lengths - 2) // 2 + 1
    return input_lengths, output_lengths


def get_max_qwen2_audio_audio_tokens(ctx: InputContext) -> int:
    max_source_position = (
        ctx.model_config.hf_config.audio_config.max_source_positions)
    output_lengths = (max_source_position - 2) // 2 + 1
    return output_lengths


def input_processor_for_qwen2_audio(
        ctx: InputContext, inputs: DecoderOnlyInputs) -> DecoderOnlyInputs:
    multi_modal_data = inputs.get("multi_modal_data")
    if multi_modal_data is None or "audio" not in multi_modal_data:
        return inputs

    audios = multi_modal_data["audio"]
    if not isinstance(audios, list):
        audios = [audios]

    if len(audios) == 0:
        return inputs

    processor = cached_get_processor(ctx.model_config.model)
    resampled_audios = [
        librosa.resample(audio,
                         orig_sr=sampling_rate,
                         target_sr=processor.feature_extractor.sampling_rate)
        for audio, sampling_rate in audios
    ]
    audio_input_lengths = np.array(
        [min(3000, _.shape[0] // 160 + 1) for _ in resampled_audios])

    audio_feat_lengths, audio_output_lengths = _get_feat_extract_output_lengths(
        audio_input_lengths)

    audio_token_index = ctx.model_config.hf_config.audio_token_index

    input_ids = inputs['prompt_token_ids']

    new_input_ids = []
    audio_num = input_ids.count(audio_token_index)
    assert len(audio_input_lengths) == audio_num, \
        (f'The text input contains {audio_num} audio tokens, '
         f'but {len(audio_input_lengths)} audios provided')
    start = 0
    for audio_idx in range(audio_num):
        end = input_ids.index(audio_token_index, start)
        new_input_ids.extend(input_ids[start:end])  # text part

        new_input_ids.extend([audio_token_index] *
                             audio_output_lengths[audio_idx])
        start = end + 1
    new_input_ids.extend(input_ids[start:])

    return token_inputs(
        prompt_token_ids=new_input_ids,
        prompt=inputs['prompt'],
        multi_modal_data=multi_modal_data,
    )


def input_mapper_for_qwen2_audio(
    ctx: InputContext,
    multi_modal_data: Union[np.ndarray, List[np.ndarray]],
) -> MultiModalKwargs:
    """Input mapper for Qwen2-Audio."""
    if not isinstance(multi_modal_data, list):
        multi_modal_data = [multi_modal_data]

    if len(multi_modal_data) == 0:
        return MultiModalKwargs()

    processor = cached_get_processor(ctx.model_config.model)
    audio_feature_extractor = processor.feature_extractor
    if audio_feature_extractor is None:
        raise RuntimeError(
            "No HuggingFace audio_feature_extractor is available "
            "to process the audio object")

    try:
        resampled_audios = [
            librosa.resample(
                audio,
                orig_sr=sampling_rate,
                target_sr=processor.feature_extractor.sampling_rate)
            for audio, sampling_rate in multi_modal_data
        ]
        batch_data = audio_feature_extractor(resampled_audios,
                                             sampling_rate=16000,
                                             return_attention_mask=True,
                                             padding="max_length",
                                             return_tensors="pt").data
        batch_data["feature_attention_mask"] = batch_data.pop("attention_mask")
    except Exception:
        logger.error("Failed to process audio (%s)", multi_modal_data)
        raise

    return MultiModalKwargs(batch_data)


@INPUT_REGISTRY.register_dummy_data(dummy_data_for_qwen2_audio)
@INPUT_REGISTRY.register_input_processor(input_processor_for_qwen2_audio)
@MULTIMODAL_REGISTRY.register_input_mapper("audio",
                                           input_mapper_for_qwen2_audio)
@MULTIMODAL_REGISTRY.register_max_multimodal_tokens(
    "audio", get_max_qwen2_audio_audio_tokens)
class Qwen2AudioForConditionalGeneration(nn.Module, SupportsMultiModal,
                                         SupportsPP):

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        multimodal_config = vllm_config.model_config.multimodal_config
        self.config = config
        self.multimodal_config = multimodal_config

        self.audio_tower = Qwen2AudioEncoder(config.audio_config)
        self.multi_modal_projector = Qwen2AudioMultiModalProjector(
            config.audio_config.d_model, config.text_config.hidden_size)

        self.quant_config = quant_config

        self.language_model = Qwen2Model(
            vllm_config=vllm_config.with_hf_config(config.text_config),
            prefix=prefix)
        self.unpadded_vocab_size = config.text_config.vocab_size
        if config.text_config.tie_word_embeddings:
            self.lm_head = self.language_model.embed_tokens
        else:
            self.lm_head = ParallelLMHead(config.text_config.vocab_size,
                                          config.text_config.hidden_size,
                                          quant_config=quant_config)
        logit_scale = getattr(config, "logit_scale", 1.0)
        self.logits_processor = LogitsProcessor(self.unpadded_vocab_size,
                                                config.text_config.vocab_size,
                                                logit_scale)
        self.sampler = get_sampler()

        self.make_empty_intermediate_tensors = (
            self.language_model.make_empty_intermediate_tensors)

    def _validate_and_reshape_mm_tensor(self,
                                        mm_input: Union[torch.Tensor,
                                                        List[torch.Tensor]],
                                        name: str) -> torch.Tensor:
        if not isinstance(mm_input, (torch.Tensor, list)):
            raise ValueError(f"Incorrect type of {name}. "
                             f"Got type: {type(mm_input)}")
        if isinstance(mm_input, torch.Tensor):
            return torch.concat(list(mm_input))
        else:
            return torch.concat(mm_input)

    def _parse_and_validate_audio_input(
            self, **kwargs: object) -> Optional[Qwen2AudioInputs]:
        input_features = kwargs.pop('input_features', None)
        feature_attention_mask = kwargs.pop('feature_attention_mask', None)
        if input_features is None:
            return None
        input_features = self._validate_and_reshape_mm_tensor(
            input_features, 'input_features')
        feature_attention_mask = self._validate_and_reshape_mm_tensor(
            feature_attention_mask, 'feature_attention_mask')
        if not isinstance(input_features, (torch.Tensor, list)):
            raise ValueError("Incorrect type of audio input features. "
                             f"Got type: {type(input_features)}")
        return Qwen2AudioInputs(input_features=input_features,
                                feature_attention_mask=feature_attention_mask)

    def _process_audio_input(self,
                             audio_input: Qwen2AudioInputs) -> torch.Tensor:

        input_features = audio_input["input_features"]
        feature_attention_mask = audio_input["feature_attention_mask"]

        audio_feat_lengths, audio_output_lengths = (
            self.audio_tower._get_feat_extract_output_lengths(
                feature_attention_mask.sum(-1)))

        batch_size, _, max_mel_seq_len = input_features.shape
        max_seq_len = (max_mel_seq_len - 2) // 2 + 1
        # Create a sequence tensor of shape (batch_size, max_seq_len)
        seq_range = (torch.arange(
            0,
            max_seq_len,
            dtype=audio_feat_lengths.dtype,
            device=audio_feat_lengths.device).unsqueeze(0).expand(
                batch_size, max_seq_len))
        lengths_expand = audio_feat_lengths.unsqueeze(-1).expand(
            batch_size, max_seq_len)
        # Create mask
        padding_mask = seq_range >= lengths_expand

        audio_attention_mask_ = padding_mask.view(
            batch_size, 1, 1, max_seq_len).expand(batch_size, 1, max_seq_len,
                                                  max_seq_len)
        audio_attention_mask = audio_attention_mask_.to(
            dtype=self.audio_tower.conv1.weight.dtype,
            device=self.audio_tower.conv1.weight.device)
        audio_attention_mask[audio_attention_mask_] = float("-inf")

        audio_outputs = self.audio_tower(input_features,
                                         attention_mask=audio_attention_mask)
        selected_audio_feature = audio_outputs.last_hidden_state
        audio_features = self.multi_modal_projector(selected_audio_feature)
        num_audios, max_audio_tokens, embed_dim = audio_features.shape
        audio_features_mask = torch.arange(max_audio_tokens).expand(
            num_audios, max_audio_tokens
        ).to(audio_output_lengths.device) < audio_output_lengths.unsqueeze(1)
        masked_audio_features = audio_features[audio_features_mask].view(
            -1, embed_dim)

        return masked_audio_features

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[torch.Tensor],
        attn_metadata: AttentionMetadata,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        **kwargs: object,
    ) -> Union[torch.Tensor, IntermediateTensors]:
        if intermediate_tensors is not None:
            input_ids = None
            inputs_embeds = None
        else:
            audio_input = self._parse_and_validate_audio_input(**kwargs)

            if audio_input is None:
                inputs_embeds = None
            else:
                inputs_embeds = self.language_model.embed_tokens(input_ids)
                masked_audio_features = self._process_audio_input(audio_input)
                # merge llm embeddings and audio features
                mask = (input_ids == self.config.audio_token_index)
                inputs_embeds[mask, :] = masked_audio_features

                input_ids = None

        hidden_states = self.language_model(
            input_ids=input_ids,
            positions=positions,
            kv_caches=kv_caches,
            attn_metadata=attn_metadata,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
        )
        return hidden_states

    def compute_logits(self, hidden_states: torch.Tensor,
                       sampling_metadata: SamplingMetadata) -> torch.Tensor:
        logits = self.logits_processor(self.lm_head, hidden_states,
                                       sampling_metadata)
        return logits

    def sample(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[SamplerOutput]:
        next_tokens = self.sampler(logits, sampling_metadata)
        return next_tokens

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]
        params_dict = dict(self.named_parameters(remove_duplicate=False))
        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq" in name:
                continue
            if (self.config.text_config.tie_word_embeddings
                    and "lm_head.weight" in name):
                continue
            for key_to_modify, new_key in _KEYS_TO_MODIFY_MAPPING.items():
                if key_to_modify in name:
                    name = name.replace(key_to_modify, new_key)
            for (param_name, weight_name, shard_id) in stacked_params_mapping:
                if weight_name not in name or 'audio' in name:
                    continue
                name = name.replace(weight_name, param_name)
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                # Remapping the name of FP8 kv-scale.
                name = maybe_remap_kv_scale_name(name, params_dict)
                if name is None:
                    continue

                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader",
                                        default_weight_loader)
                weight_loader(param, loaded_weight)
