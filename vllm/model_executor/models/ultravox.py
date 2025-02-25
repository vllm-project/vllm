# SPDX-License-Identifier: Apache-2.0

# Adapted from https://github.com/fixie-ai/ultravox/blob/ecd58c4041030bae2ad15aa6bcf04ab43199ea02/ultravox/model/ultravox_model.py
"""PyTorch Ultravox model."""
import math
from functools import cached_property
from typing import (Any, Iterable, Literal, Mapping, Optional, Set, Tuple,
                    TypedDict, Union)

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import functional as F
from transformers import BatchFeature, ProcessorMixin
from transformers.models.whisper import WhisperFeatureExtractor
from transformers.models.whisper.modeling_whisper import WhisperEncoder

from vllm import envs
from vllm.config import VllmConfig
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.activation import MulAndSilu, get_act_fn
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.sampler import SamplerOutput, get_sampler
from vllm.model_executor.model_loader.loader import DefaultModelLoader
from vllm.model_executor.models.module_mapping import MultiModelKeys
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import (MultiModalFieldConfig, MultiModalKwargs,
                                    NestedTensors)
from vllm.multimodal.parse import MultiModalDataItems, MultiModalDataParser
from vllm.multimodal.processing import (BaseMultiModalProcessor,
                                        BaseProcessingInfo, PromptReplacement)
from vllm.multimodal.profiling import BaseDummyInputsBuilder, ProcessorInputs
from vllm.sequence import IntermediateTensors
from vllm.transformers_utils.configs.ultravox import UltravoxConfig

from .interfaces import SupportsLoRA, SupportsMultiModal, SupportsPP
from .utils import (AutoWeightsLoader, WeightsMapper, flatten_bn,
                    init_vllm_registered_model, maybe_prefix,
                    merge_multimodal_embeddings,
                    merge_multimodal_embeddings_from_map)

_AUDIO_PLACEHOLDER_OVERRIDE = "<|reserved_special_token_0|>"
_AUDIO_PLACEHOLDER_TOKEN = 128002
_AUDIO_TOKENS_PER_SECOND = 6.25


class UltravoxAudioFeatureInputs(TypedDict):
    type: Literal["audio_features"]
    data: NestedTensors
    """Shape: `(batch_size, num_audios, 80, M)`"""


class UltravoxAudioEmbeddingInputs(TypedDict):
    type: Literal["audio_embeds"]
    data: NestedTensors
    """Shape: `(batch_size, num_audios, audio_feature_size, hidden_size)`"""


UltravoxAudioInputs = Union[UltravoxAudioFeatureInputs,
                            UltravoxAudioEmbeddingInputs]


class UltravoxProcessingInfo(BaseProcessingInfo):

    def get_hf_processor(
        self,
        *,
        # Ignored in initialization
        sampling_rate: Optional[int] = None,
        **kwargs: object,
    ) -> ProcessorMixin:
        hf_processor = self.ctx.get_hf_processor(**kwargs)

        # NOTE: Ultravox processing definition uses '<|eot_id|>' as the
        # placeholder that will cause confusion with the actual end of turn
        # token, thus we override placeholder with a reserved special
        # token.
        hf_processor.audio_token_replacement = _AUDIO_PLACEHOLDER_OVERRIDE
        return hf_processor

    def get_feature_extractor(
        self,
        *,
        # Ignored in initialization
        sampling_rate: Optional[int] = None,
    ) -> WhisperFeatureExtractor:
        hf_processor = self.get_hf_processor(sampling_rate=sampling_rate)
        audio_processor = hf_processor.audio_processor  # type: ignore
        feature_extractor = audio_processor.feature_extractor  # type: ignore
        assert isinstance(feature_extractor, WhisperFeatureExtractor)
        return feature_extractor

    def get_supported_mm_limits(self) -> Mapping[str, Optional[int]]:
        return {"audio": None}

    def get_mm_max_tokens_per_item(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
    ) -> Mapping[str, int]:
        feature_extractor = self.get_feature_extractor()
        max_audio_tokens = math.ceil(feature_extractor.chunk_length *
                                     _AUDIO_TOKENS_PER_SECOND)

        return {"audio": max_audio_tokens}


class UltravoxDummyInputsBuilder(BaseDummyInputsBuilder[UltravoxProcessingInfo]
                                 ):

    def get_dummy_processor_inputs(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
    ) -> ProcessorInputs:
        feature_extractor = self.info.get_feature_extractor()

        sampling_rate = feature_extractor.sampling_rate
        audio_len = feature_extractor.chunk_length * sampling_rate
        num_audios = mm_counts.get("audio", 0)

        mm_data = {
            "audio":
            self._get_dummy_audios(length=audio_len, num_audios=num_audios)
        }

        return ProcessorInputs(
            prompt_text="<|audio|>" * num_audios,
            mm_data=mm_data,
        )


class UltravoxMultiModalProcessor(
        BaseMultiModalProcessor[UltravoxProcessingInfo]):

    def _get_data_parser(self) -> MultiModalDataParser:
        feature_extractor = self.info.get_feature_extractor()
        return MultiModalDataParser(target_sr=feature_extractor.sampling_rate)

    def _call_hf_processor(
        self,
        prompt: str,
        mm_data: Mapping[str, object],
        mm_kwargs: Mapping[str, object],
    ) -> BatchFeature:
        # Text-only input not supported in composite processor
        if not mm_data or not mm_data.get("audios", []):
            prompt_ids = self.info.get_tokenizer().encode(prompt)
            prompt_ids = self._apply_hf_processor_tokens_only(prompt_ids)
            return BatchFeature(dict(input_ids=[prompt_ids]), tensor_type="pt")

        mm_data = dict(mm_data)
        audios = mm_data.pop("audios", [])
        assert isinstance(audios, list)

        feature_extractor = self.info.get_feature_extractor()
        mm_kwargs = dict(
            **mm_kwargs,
            sampling_rate=feature_extractor.sampling_rate,
        )

        # Ultravox processor doesn't support multiple inputs,
        # therefore we need to input text and audio one by one
        audio_features, audio_token_len = [], []
        shared_outputs = {}
        for audio in audios:
            # NOTE: Ultravox processor accepts "audio" instead of "audios"
            item_processor_data = dict(**mm_data, audio=audio)

            item_outputs = super()._call_hf_processor(
                prompt=prompt,
                mm_data=item_processor_data,
                mm_kwargs=mm_kwargs,
            )

            audio_features.append(item_outputs.pop("audio_values")[0])
            audio_token_len.append(item_outputs.pop("audio_token_len").item())
            shared_outputs = item_outputs

        combined_outputs = dict(
            **shared_outputs,
            audio_features=audio_features,
            audio_token_len=audio_token_len,
        )
        return BatchFeature(combined_outputs)

    def _apply_hf_processor_tokens_only(
        self,
        prompt_tokens: list[int],
    ) -> list[int]:
        # HF processor omits bos_token_id by setting add_special_tokens=False
        tokenizer = self.info.get_tokenizer()
        assert prompt_tokens[0] == tokenizer.bos_token_id

        return prompt_tokens[1:]

    def _get_mm_fields_config(
        self,
        hf_inputs: BatchFeature,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        return dict(
            audio_features=MultiModalFieldConfig.batched("audio"),
            audio_token_len=MultiModalFieldConfig.batched("audio"),
            audio_embeds=MultiModalFieldConfig.batched("audio"),
        )

    def _get_prompt_replacements(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, Any],
        out_mm_kwargs: MultiModalKwargs,
    ) -> list[PromptReplacement]:
        hf_processor = self.info.get_hf_processor(**hf_processor_mm_kwargs)
        tokenizer = self.info.get_tokenizer()
        vocab = tokenizer.get_vocab()

        replacement_id = vocab[
            hf_processor.audio_token_replacement]  # type: ignore

        def get_replacement_ultravox(item_idx: int):
            audio_token_len = out_mm_kwargs["audio_token_len"][item_idx]
            return [replacement_id] * int(audio_token_len)  # type: ignore

        return [
            PromptReplacement(
                modality="audio",
                target="<|audio|>",
                replacement=get_replacement_ultravox,
            )
        ]


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


class UltravoxProjector(nn.Module):

    def __init__(self, config: UltravoxConfig):
        super().__init__()
        self.hidden_dim = config.hidden_size
        self._pad_and_stack = StackAudioFrames(config.stack_factor)
        dim_in = config.audio_config.hidden_size * config.stack_factor
        self.ln_pre = RMSNorm(dim_in)
        self.linear_1 = nn.Linear(dim_in, self.hidden_dim, bias=False)
        dim_mid = self.hidden_dim

        if config.projector_act == "swiglu":
            self.act = MulAndSilu()
            dim_mid = dim_mid // 2
        else:
            self.act = get_act_fn(config.projector_act)

        dim_out = config.text_config.hidden_size
        self.linear_2 = nn.Linear(dim_mid, dim_out, bias=False)

        # Ultravox v0.4.1 and below use layer_norm after the second linear layer
        # while v0.5.0 and above uses layer_norm after the first linear layer.
        if config.projector_ln_mid:
            self.ln_mid: nn.Module = RMSNorm(dim_mid)
            self.ln_post = nn.Identity()
        else:
            self.ln_mid = nn.Identity()
            self.ln_post = RMSNorm(dim_out)

    def forward(self, audio_features: torch.Tensor) -> torch.Tensor:
        audio_features = self._pad_and_stack(audio_features)
        audio_features = self.ln_pre(audio_features)
        hidden_states = self.linear_1(audio_features)
        hidden_states = self.act(hidden_states)
        hidden_states = self.ln_mid(hidden_states)
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
    ):
        expected_seq_length = (self.config.max_source_positions *
                               self.conv1.stride[0] * self.conv2.stride[0])
        if input_features.shape[-1] > expected_seq_length:
            raise ValueError(
                f"Whisper expects the mel input features to be of length "
                f"{expected_seq_length} or less, but found "
                f"{input_features.shape[-1]}. Make sure to pad the input mel "
                f"features to {expected_seq_length}.")

        inputs_embeds = nn.functional.gelu(self.conv1(input_features))
        inputs_embeds = nn.functional.gelu(self.conv2(inputs_embeds))

        inputs_embeds = inputs_embeds.permute(0, 2, 1)
        embed_pos = self.embed_positions.weight[:inputs_embeds.size(-2)]

        hidden_states = inputs_embeds + embed_pos
        hidden_states = nn.functional.dropout(hidden_states,
                                              p=self.dropout,
                                              training=self.training)

        for encoder_layer in self.layers:
            layer_outputs = encoder_layer(
                hidden_states,
                None,
                layer_head_mask=None,
            )

            hidden_states = layer_outputs[0]

        hidden_states = self.layer_norm(hidden_states)
        return hidden_states


@MULTIMODAL_REGISTRY.register_processor(
    UltravoxMultiModalProcessor,
    info=UltravoxProcessingInfo,
    dummy_inputs=UltravoxDummyInputsBuilder)
class UltravoxModel(nn.Module, SupportsMultiModal, SupportsPP, SupportsLoRA):

    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"]
    }

    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_prefix={"audio_tower.model.encoder.": "audio_tower."})

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        multimodal_config = vllm_config.model_config.multimodal_config
        self.config = config
        self.multi_modal_config = multimodal_config
        assert self.multi_modal_config

        self.secondary_weights = []
        self.audio_tower = ModifiedWhisperEncoder(config.audio_config)
        if config.audio_model_id is not None:
            # this prefix is not for initialization, but for loading weights
            # note the trailing dot
            self.secondary_weights.append(
                DefaultModelLoader.Source(
                    model_or_path=config.audio_model_id,
                    revision=None,
                    prefix="audio_tower.",
                ))
        self.multi_modal_projector = UltravoxProjector(config)
        self.language_model = init_vllm_registered_model(
            vllm_config=vllm_config,
            hf_config=config.text_config,
            prefix=maybe_prefix(prefix, "language_model"),
        )
        if config.text_model_id is not None:
            # this prefix is not for initialization, but for loading weights
            # note the trailing dot
            self.secondary_weights.append(
                DefaultModelLoader.Source(model_or_path=config.text_model_id,
                                          revision=None,
                                          prefix="language_model."))

        self.make_empty_intermediate_tensors = (
            self.language_model.make_empty_intermediate_tensors)

    @cached_property
    def sampler(self):
        if hasattr(self.language_model, "sampler"):
            return self.language_model.sampler

        return get_sampler()

    def get_mm_mapping(self) -> MultiModelKeys:
        """
        Get the module prefix in multimodal models
        """
        return MultiModelKeys.from_string_field(
            language_model="language_model.",
            connector="multi_modal_projector.",
            tower_model="audio_tower.",
        )

    def _audio_features_to_embeddings(
            self, input_features: torch.Tensor) -> torch.Tensor:
        audio_input = input_features.to(self.audio_tower.dtype)
        audio_features = self.audio_tower(audio_input)
        audio_features = audio_features.to(self.audio_tower.dtype)
        audio_embeddings = self.multi_modal_projector(audio_features)
        return audio_embeddings

    def _parse_and_validate_audio_input(
            self, **kwargs: object) -> Optional[UltravoxAudioInputs]:
        audio_features = kwargs.pop("audio_features", None)
        audio_embeds = kwargs.pop("audio_embeds", None)

        if audio_features is None and audio_embeds is None:
            return None

        if audio_features is not None:
            if not isinstance(audio_features, (torch.Tensor, list)):
                raise ValueError("Incorrect type of audio features. "
                                 f"Got type: {type(audio_features)}")

            return UltravoxAudioFeatureInputs(type="audio_features",
                                              data=audio_features)

        if audio_embeds is not None:
            if not isinstance(audio_embeds, (torch.Tensor, list)):
                raise ValueError("Incorrect type of audio embeds. "
                                 f"Got type: {type(audio_embeds)}")

            return UltravoxAudioEmbeddingInputs(type="audio_embeds",
                                                data=audio_embeds)

        raise AssertionError("This line should be unreachable.")

    def _process_audio_input(
            self, audio_input: UltravoxAudioInputs) -> NestedTensors:
        if audio_input["type"] == "audio_embeds":
            return audio_input["data"]

        audio_features = audio_input["data"]
        if isinstance(audio_features, torch.Tensor):
            # Combine the B and N dimensions for the encoder/projector
            flattened = flatten_bn(audio_features)
            flattened_embeddings = self._audio_features_to_embeddings(
                flattened)

            # Restore the original dimensions
            embeddings = flattened_embeddings.unflatten(
                0, audio_features.shape[:2])
            return embeddings

        result = []
        # TODO: Batch heterogeneous tensors through the encoder/projector
        for audio_features_item in audio_features:
            if isinstance(audio_features_item, torch.Tensor):
                result.append(
                    self._audio_features_to_embeddings(audio_features_item))
            else:
                embeddings = [
                    # Add a batch dimension to embed it, then remove it.
                    self._audio_features_to_embeddings(tensor.unsqueeze(0)
                                                       ).squeeze(0)
                    for tensor in audio_features_item
                ]
                result.append(embeddings)

        return result

    def get_multimodal_embeddings(self, **kwargs) -> Optional[NestedTensors]:
        audio_input = self._parse_and_validate_audio_input(**kwargs)
        if audio_input is None:
            return None
        audio_embeddings = self._process_audio_input(audio_input)
        return audio_embeddings

    def get_input_embeddings(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: Optional[NestedTensors] = None,
    ) -> torch.Tensor:
        inputs_embeds = self.language_model.get_input_embeddings(input_ids)
        if multimodal_embeddings is not None:

            # TODO(ywang96): remove this block after v0 is deprecated.
            if not envs.VLLM_USE_V1:
                attn_metadata = get_forward_context().attn_metadata
                merge_multimodal_embeddings_from_map(
                    inputs_embeds, multimodal_embeddings,
                    attn_metadata.multi_modal_placeholder_index_maps["audio"])
            else:
                inputs_embeds = merge_multimodal_embeddings(
                    input_ids, inputs_embeds, multimodal_embeddings,
                    _AUDIO_PLACEHOLDER_TOKEN)
        return inputs_embeds

    def forward(self,
                input_ids: torch.Tensor,
                positions: torch.Tensor,
                intermediate_tensors: Optional[torch.Tensor] = None,
                inputs_embeds: Optional[torch.Tensor] = None,
                **kwargs) -> Union[torch.Tensor, IntermediateTensors]:
        """Run forward pass for Ultravox

        One key thing to understand is the `input_ids` already accounts for the
        positions of the to-be-inserted audio embeddings. The to-be-inserted
        audio has a size that is essentially 6.25 tokens per second of audio.

        This way, the `positions` and `attn_metadata` are consistent
        with the `input_ids`.

        Args:
            audio_features: A batch of audio inputs [B, N, 80, M].
        """

        if intermediate_tensors is not None:
            inputs_embeds = None

        # NOTE: In v1, inputs_embeds is always generated at model runner, this
        # condition is for v0 compatibility.
        elif inputs_embeds is None:
            multimodal_embeddings = self.get_multimodal_embeddings(**kwargs)

            inputs_embeds = self.get_input_embeddings(input_ids,
                                                      multimodal_embeddings)
            input_ids = None

        hidden_states = self.language_model.model(input_ids,
                                                  positions,
                                                  intermediate_tensors,
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

    def load_weights(self, weights: Iterable[Tuple[str,
                                                   torch.Tensor]]) -> Set[str]:

        loader = AutoWeightsLoader(self,
                                   ignore_unexpected_prefixes=["audio_tower."])
        return loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)
