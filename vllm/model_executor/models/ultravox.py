# Adapted from https://github.com/fixie-ai/ultravox/blob/ecd58c4041030bae2ad15aa6bcf04ab43199ea02/ultravox/model/ultravox_model.py
"""PyTorch Ultravox model."""

import itertools
import math
from array import array
from typing import Iterable, List, Mapping, Optional, Tuple, Union, cast

import librosa
import numpy as np
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import functional as F
from transformers.modeling_outputs import BaseModelOutput
from transformers.models.whisper import WhisperFeatureExtractor
from transformers.models.whisper.modeling_whisper import WhisperEncoder

from vllm.attention import AttentionMetadata
from vllm.config import CacheConfig, MultiModalConfig
from vllm.inputs import INPUT_REGISTRY
from vllm.inputs.data import LLMInputs
from vllm.inputs.registry import InputContext
from vllm.logger import init_logger
from vllm.model_executor.layers.activation import get_act_fn
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.interfaces import SupportsMultiModal
from vllm.model_executor.models.utils import (filter_weights,
                                              init_vllm_registered_model,
                                              merge_multimodal_embeddings)
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.base import BatchedTensors, MultiModalInputs
from vllm.multimodal.utils import (cached_get_processor, cached_get_tokenizer,
                                   repeat_and_pad_placeholder_tokens)
from vllm.sequence import VLLM_TOKEN_ID_ARRAY_TYPE, SamplerOutput, SequenceData
from vllm.transformers_utils.configs.ultravox import UltravoxConfig

_AUDIO_PLACEHOLDER_TOKEN = 128002
_AUDIO_TOKENS_PER_SECOND = 6.25

logger = init_logger(__name__)


def whisper_feature_extractor(ctx: InputContext) -> WhisperFeatureExtractor:
    whisper_processor = cached_get_processor(
        ctx.get_hf_config(UltravoxConfig).audio_model_id)
    return whisper_processor.feature_extractor


def get_ultravox_max_audio_tokens(ctx: InputContext):
    feature_extractor = whisper_feature_extractor(ctx)
    return math.ceil(feature_extractor.chunk_length * _AUDIO_TOKENS_PER_SECOND)


def dummy_data_for_ultravox(
    ctx: InputContext,
    seq_len: int,
    mm_counts: Mapping[str, int],
):
    feature_extractor = whisper_feature_extractor(ctx)

    audio_count = mm_counts["audio"]

    audio_token_ids = array(VLLM_TOKEN_ID_ARRAY_TYPE, [
        _AUDIO_PLACEHOLDER_TOKEN
    ]) * get_ultravox_max_audio_tokens(ctx) * audio_count
    other_token_ids = array(VLLM_TOKEN_ID_ARRAY_TYPE,
                            [0]) * (seq_len - len(audio_token_ids))

    audio_and_sr = (np.array([0.0] * feature_extractor.chunk_length), 1)
    mm_dict = {
        "audio":
        audio_and_sr if audio_count == 1 else [audio_and_sr] * audio_count
    }

    return (SequenceData(audio_token_ids + other_token_ids), mm_dict)


def input_mapper_for_ultravox(ctx: InputContext, data: object):
    if isinstance(data, tuple):
        (audio, sr) = cast(Tuple[np.ndarray, Union[float, int]], data)
        feature_extractor = whisper_feature_extractor(ctx)

        if sr != feature_extractor.sampling_rate:
            audio = librosa.resample(audio,
                                     orig_sr=sr,
                                     target_sr=feature_extractor.sampling_rate)
            sr = feature_extractor.sampling_rate

        minimum_audio_length = feature_extractor.n_fft // 2 + 1
        if len(audio) < minimum_audio_length:
            # Not enough audio; pad it.
            audio = np.pad(audio, (0, minimum_audio_length - len(audio)))

        return MultiModalInputs(
            feature_extractor(audio,
                              sampling_rate=sr,
                              padding="longest",
                              return_tensors="pt"))

    raise NotImplementedError(f"Unsupported data type: {type(data)}")


def input_processor_for_ultravox(ctx: InputContext, llm_inputs: LLMInputs):
    multi_modal_data = llm_inputs.get("multi_modal_data")
    if multi_modal_data is None or "audio" not in multi_modal_data:
        return llm_inputs

    feature_extractor = whisper_feature_extractor(ctx)
    audio_data, sample_rate = multi_modal_data["audio"]

    audio_length = audio_data.shape[0]
    if sample_rate != feature_extractor.sampling_rate:
        # Account for resampling.
        adjustment = feature_extractor.sampling_rate / sample_rate
        audio_length = math.ceil(adjustment * audio_length)

    feature_extractor_output_length = math.ceil(
        (audio_length -
         (feature_extractor.hop_length - 1)) / feature_extractor.hop_length)

    uv_config = ctx.get_hf_config(UltravoxConfig)
    audio_num_tokens = min(
        max(
            1,
            math.ceil(feature_extractor_output_length /
                      (uv_config.stack_factor * 2))),
        get_ultravox_max_audio_tokens(ctx))
    tokenizer = cached_get_tokenizer(ctx.model_config.tokenizer)

    new_prompt, new_token_ids = repeat_and_pad_placeholder_tokens(
        tokenizer,
        llm_inputs.get("prompt"),
        llm_inputs["prompt_token_ids"],
        placeholder_token_id=_AUDIO_PLACEHOLDER_TOKEN,
        repeat_count=audio_num_tokens,
    )

    # NOTE: Create a defensive copy of the original inputs
    return LLMInputs(prompt_token_ids=new_token_ids,
                     prompt=new_prompt,
                     multi_modal_data=multi_modal_data)


class StackAudioFrames(nn.Module):
    """
    Stack the audio embedding frames to reduce the sequence length by a factor
    of `stack_factor`.
    """

    def __init__(self, stack_factor: int = 8):
        super().__init__()
        self.stack_factor = stack_factor

    def forward(self, audio_embeds: torch.Tensor) -> torch.Tensor:
        B, T, C = audio_embeds.shape
        T_pad = (T + self.stack_factor -
                 1) // self.stack_factor * self.stack_factor
        audio_embeds = F.pad(audio_embeds, (0, 0, 0, T_pad - T))
        B, T, C = audio_embeds.shape
        audio_embeds = audio_embeds.view(B, T // self.stack_factor,
                                         C * self.stack_factor)
        return audio_embeds


class SwiGLU(nn.Module):

    def forward(self, x):
        x, gate = x.chunk(2, dim=-1)
        return F.silu(gate) * x


class UltravoxProjector(nn.Sequential):

    def __init__(self, config: UltravoxConfig):
        super().__init__()
        self.hidden_dim = config.hidden_size
        self._pad_and_stack = StackAudioFrames(config.stack_factor)
        dim = config.audio_config.hidden_size * config.stack_factor
        self.ln_pre = RMSNorm(dim)
        self.linear_1 = nn.Linear(dim, self.hidden_dim, bias=False)
        dim = self.hidden_dim
        self.act = SwiGLU(
        ) if config.projector_act == "swiglu" else get_act_fn(
            config.projector_act)
        dim = dim // 2 if config.projector_act == "swiglu" else dim
        self.linear_2 = nn.Linear(dim,
                                  config.text_config.hidden_size,
                                  bias=False)
        self.ln_post = RMSNorm(config.text_config.hidden_size)

    def forward(self, audio_features: torch.Tensor) -> torch.Tensor:
        audio_features = self._pad_and_stack(audio_features)
        audio_features = self.ln_pre(audio_features)
        hidden_states = self.linear_1(audio_features)
        hidden_states = self.act(hidden_states)
        hidden_states = self.linear_2(hidden_states)
        hidden_states = self.ln_post(hidden_states)
        return hidden_states


class ModifiedWhisperEncoder(WhisperEncoder):
    """
    Encoder portion of OpenAI's Whisper model.

    This implementation is a slightly modified version of HF Transformers'
    Whisper Encoder, with only a few fixes:
    1. base_model_prefix updated to allow for doing `.from_pretrained`
       directly on the encoder
    2. allow less than 30 second of audio padding to be passed in:
        - relaxed ValueError check for `input_features` length to be less
           than or equal to `expected_seq_length` instead of strictly equal
        - embed_pos is now sliced to match the length of `inputs_embeds`

    Original: https://github.com/huggingface/transformers/blob/main/src/transformers/models/whisper/modeling_whisper.py
    See commentary: https://github.com/huggingface/transformers/issues/25744
    """

    base_model_prefix = "model.encoder"

    def forward(
        self,
        input_features,
        attention_mask=None,
        head_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        expected_seq_length = (self.config.max_source_positions *
                               self.conv1.stride[0] * self.conv2.stride[0])
        if input_features.shape[-1] > expected_seq_length:
            raise ValueError(
                f"Whisper expects the mel input features to be of length "
                f"{expected_seq_length} or less, but found "
                f"{input_features.shape[-1]}. Make sure to pad the input mel "
                f"features to {expected_seq_length}.")

        output_attentions = (output_attentions if output_attentions is not None
                             else self.config.output_attentions)
        output_hidden_states = (output_hidden_states
                                if output_hidden_states is not None else
                                self.config.output_hidden_states)
        return_dict = (return_dict if return_dict is not None else
                       self.config.use_return_dict)
        inputs_embeds = nn.functional.gelu(self.conv1(input_features))
        inputs_embeds = nn.functional.gelu(self.conv2(inputs_embeds))

        inputs_embeds = inputs_embeds.permute(0, 2, 1)
        embed_pos = self.embed_positions.weight[:inputs_embeds.size(-2)]

        hidden_states = inputs_embeds + embed_pos
        hidden_states = nn.functional.dropout(hidden_states,
                                              p=self.dropout,
                                              training=self.training)

        encoder_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        # check if head_mask has a correct number of layers specified if desired
        if head_mask is not None:
            assert head_mask.size()[0] == (len(self.layers)), (
                f"The head_mask should be specified for {len(self.layers)} "
                f"layers, but it is for {head_mask.size()[0]}.")

        for idx, encoder_layer in enumerate(self.layers):
            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states, )
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description) # noqa: E501
            to_drop = False
            if self.training:
                dropout_probability = torch.rand([])
                if dropout_probability < self.layerdrop:  # skip the layer
                    to_drop = True

            if to_drop:
                layer_outputs = (None, None)
            else:
                if self.gradient_checkpointing and self.training:
                    layer_outputs = self._gradient_checkpointing_func(
                        encoder_layer.__call__,
                        hidden_states,
                        None,
                        (head_mask[idx] if head_mask is not None else None),
                        output_attentions,
                    )
                else:
                    layer_outputs = encoder_layer(
                        hidden_states,
                        None,
                        layer_head_mask=(head_mask[idx]
                                         if head_mask is not None else None),
                        output_attentions=output_attentions,
                    )

                hidden_states = layer_outputs[0]

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1], )

        hidden_states = self.layer_norm(hidden_states)
        if output_hidden_states:
            encoder_states = encoder_states + (hidden_states, )

        if not return_dict:
            return tuple(
                v for v in [hidden_states, encoder_states, all_attentions]
                if v is not None)
        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=encoder_states,
            attentions=all_attentions,
        )


@MULTIMODAL_REGISTRY.register_input_mapper("audio", input_mapper_for_ultravox)
@MULTIMODAL_REGISTRY.register_max_multimodal_tokens(
    "audio", get_ultravox_max_audio_tokens)
@INPUT_REGISTRY.register_dummy_data(dummy_data_for_ultravox)
@INPUT_REGISTRY.register_input_processor(input_processor_for_ultravox)
class UltravoxModel(nn.Module, SupportsMultiModal):

    def __init__(self,
                 config: UltravoxConfig,
                 multimodal_config: MultiModalConfig,
                 cache_config: Optional[CacheConfig] = None,
                 quant_config: Optional["QuantizationConfig"] = None):
        super().__init__()
        self.config = config
        self.multi_modal_config = multimodal_config
        assert self.multi_modal_config

        if config.audio_model_id is not None:
            self.audio_tower = ModifiedWhisperEncoder.from_pretrained(
                config.audio_model_id)
        else:
            self.audio_tower = ModifiedWhisperEncoder(config.audio_config)
        self.multi_modal_projector = UltravoxProjector(config)
        self.language_model = init_vllm_registered_model(
            config.text_config, cache_config, quant_config)

    def _audio_features_to_embeddings(self, input_features: torch.Tensor,
                                      dtype: torch.dtype) -> torch.Tensor:
        audio_input = input_features.to(self.audio_tower.dtype)
        audio_features = self.audio_tower(audio_input).last_hidden_state
        audio_features = audio_features.to(self.audio_tower.dtype)
        audio_embeddings = self.multi_modal_projector(audio_features).to(dtype)
        return audio_embeddings

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[torch.Tensor],
        attn_metadata: AttentionMetadata,
        intermediate_tensors: Optional[torch.Tensor],
        *,
        input_features: Optional[BatchedTensors] = None,
    ) -> SamplerOutput:
        """Run forward pass for Ultravox

        One key thing to understand is the `input_ids` already accounts for the
        positions of the to-be-inserted audio embeddings. The to-be-inserted
        audio has a size that is essentially 6.25 tokens per second of audio.

        This way, the `positions` and `attn_metadata` are consistent
        with the `input_ids`.

        Args:
            input_features: A batch of audio inputs, [1, 80, M].
        """
        if input_features is not None:
            inputs_embeds = self.language_model.model.get_input_embeddings(
                input_ids)
            if isinstance(input_features, list):
                # TODO: Batch these through the encoder/projector instead of
                # serializing them.
                audio_embeddings = [
                    self._audio_features_to_embeddings(
                        single_features.unsqueeze(0),
                        inputs_embeds.dtype).squeeze(0)
                    for single_features in input_features
                ]
            elif isinstance(input_features, torch.Tensor):
                audio_embeddings = self._audio_features_to_embeddings(
                    input_features, inputs_embeds.dtype)
            else:
                raise ValueError(
                    "The input audio features should be a tensor or a list "
                    f"of tensors, not {type(input_features)}")

            merge_multimodal_embeddings(input_ids, inputs_embeds,
                                        audio_embeddings,
                                        _AUDIO_PLACEHOLDER_TOKEN)
            input_ids = None
        else:
            inputs_embeds = None

        hidden_states = self.language_model.model(
            input_ids=input_ids,
            positions=positions,
            kv_caches=kv_caches,
            attn_metadata=attn_metadata,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds)
        return hidden_states

    def compute_logits(self, hidden_states: torch.Tensor,
                       sampling_metadata: SamplingMetadata) -> torch.Tensor:
        return self.language_model.compute_logits(hidden_states,
                                                  sampling_metadata)

    def sample(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[SamplerOutput]:
        return self.language_model.sample(logits, sampling_metadata)

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        # prepare weight iterators for components
        projector_weights, llm_weights = itertools.tee(weights, 2)

        # load projector weights
        projector_weights = filter_weights(projector_weights,
                                           "multi_modal_projector")
        projector_params_dict = dict(
            self.multi_modal_projector.named_parameters())
        for name, loaded_weight in projector_weights:
            param = projector_params_dict[name]
            weight_loader = getattr(param, "weight_loader",
                                    default_weight_loader)
            weight_loader(param, loaded_weight)

        # load llm backbone
        llm_weights = filter_weights(llm_weights, "language_model")
        self.language_model.load_weights(llm_weights)
