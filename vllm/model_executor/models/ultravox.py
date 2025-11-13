# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Adapted from https://github.com/fixie-ai/ultravox/blob/ecd58c4041030bae2ad15aa6bcf04ab43199ea02/ultravox/model/ultravox_model.py
"""PyTorch Ultravox model."""

from collections.abc import Iterable, Mapping, Sequence
from typing import Annotated, Any, Literal, TypeAlias

import torch
from torch import nn
from torch.nn import functional as F
from transformers import BatchFeature, ProcessorMixin
from transformers.models.whisper import WhisperFeatureExtractor
from transformers.models.whisper.modeling_whisper import WhisperEncoder

from vllm.config import VllmConfig
from vllm.config.multimodal import BaseDummyOptions
from vllm.model_executor.layers.activation import MulAndSilu, get_act_fn
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.model_loader import DefaultModelLoader
from vllm.model_executor.models.module_mapping import MultiModelKeys
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import (
    MultiModalDataDict,
    MultiModalFieldConfig,
    MultiModalKwargsItems,
    NestedTensors,
)
from vllm.multimodal.parse import MultiModalDataItems, MultiModalDataParser
from vllm.multimodal.processing import (
    BaseMultiModalProcessor,
    BaseProcessingInfo,
    PromptReplacement,
    PromptUpdate,
)
from vllm.multimodal.profiling import BaseDummyInputsBuilder
from vllm.sequence import IntermediateTensors
from vllm.transformers_utils.configs.ultravox import UltravoxConfig
from vllm.utils.tensor_schema import TensorSchema, TensorShape

from .interfaces import (
    MultiModalEmbeddings,
    SupportsLoRA,
    SupportsMultiModal,
    SupportsPP,
)
from .utils import (
    AutoWeightsLoader,
    WeightsMapper,
    flatten_bn,
    init_vllm_registered_model,
    maybe_prefix,
)

_AUDIO_PLACEHOLDER_OVERRIDE = "<|audio|>"
_MAX_ENCODER_BATCH_SIZE = 16


class UltravoxAudioFeatureInputs(TensorSchema):
    """
    Dimensions:
    - b: batch size
    - n: number of chunks
    - t: Time frames (M)
    - nmb: Number of mel bins
    """

    type: Literal["audio_features"]
    data: Annotated[
        torch.Tensor | list[torch.Tensor] | list[list[torch.Tensor]],
        TensorShape("bn", "nmb", "t"),
    ]
    lens: Annotated[torch.Tensor, TensorShape("bn")]
    """
    Length of the audio frames per chunk. Used for attention mask in WhisperEncoder.
    """
    token_len: Annotated[torch.Tensor, TensorShape("bn")]
    """Length of the audio tokens per chunk. Used for flattening the audio features."""
    num_chunks: Annotated[torch.Tensor, TensorShape("n")]
    """Number of chunks per audio. Used for flattening the audio features."""


class UltravoxAudioEmbeddingInputs(TensorSchema):
    """
    Dimensions:
    - b: batch size
    - na: number of audios
    - afs: audio feature size
    - hs: hidden size
    """

    type: Literal["audio_embeds"]
    data: Annotated[
        torch.Tensor | list[torch.Tensor], TensorShape("b", "na", "afs", "hs")
    ]


UltravoxAudioInputs: TypeAlias = (
    UltravoxAudioFeatureInputs | UltravoxAudioEmbeddingInputs
)


class UltravoxProcessingInfo(BaseProcessingInfo):
    def get_hf_processor(self, **kwargs: object) -> ProcessorMixin:
        config = self.ctx.model_config.hf_config
        hf_processor = self.ctx.get_hf_processor(**kwargs)

        # NOTE: Ultravox processing definition uses '<|eot_id|>' as the
        # placeholder that will cause confusion with the actual end of turn
        # token, thus we override placeholder with a reserved token.
        hf_processor.audio_token_replacement = _AUDIO_PLACEHOLDER_OVERRIDE
        hf_processor.audio_replacement_token_id = config.audio_token_index

        return hf_processor

    def get_feature_extractor(self, **kwargs: object) -> WhisperFeatureExtractor:
        hf_processor = self.get_hf_processor(**kwargs)
        audio_processor = hf_processor.audio_processor  # type: ignore
        feature_extractor = audio_processor.feature_extractor  # type: ignore
        assert isinstance(feature_extractor, WhisperFeatureExtractor)
        return feature_extractor

    def get_supported_mm_limits(self) -> Mapping[str, int | None]:
        return {"audio": None}


class UltravoxDummyInputsBuilder(BaseDummyInputsBuilder[UltravoxProcessingInfo]):
    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        num_audios = mm_counts.get("audio", 0)

        return "<|audio|>" * num_audios

    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        mm_options: Mapping[str, BaseDummyOptions] | None = None,
    ) -> MultiModalDataDict:
        feature_extractor = self.info.get_feature_extractor()

        sampling_rate = feature_extractor.sampling_rate
        audio_len = (
            feature_extractor.chunk_length * sampling_rate * _MAX_ENCODER_BATCH_SIZE
        )
        num_audios = mm_counts.get("audio", 0)

        audio_overrides = mm_options.get("audio") if mm_options else None

        return {
            "audio": self._get_dummy_audios(
                length=audio_len, num_audios=num_audios, overrides=audio_overrides
            )
        }


class UltravoxMultiModalProcessor(BaseMultiModalProcessor[UltravoxProcessingInfo]):
    def _get_data_parser(self) -> MultiModalDataParser:
        feature_extractor = self.info.get_feature_extractor()
        return MultiModalDataParser(target_sr=feature_extractor.sampling_rate)

    def _call_hf_processor(
        self,
        prompt: str,
        mm_data: Mapping[str, object],
        mm_kwargs: Mapping[str, object],
        tok_kwargs: Mapping[str, object],
    ) -> BatchFeature:
        # Text-only input not supported in composite processor
        if not mm_data.get("audios", []):
            prompt_ids = self.info.get_tokenizer().encode(
                prompt, add_special_tokens=False
            )
            prompt_ids = self._apply_hf_processor_tokens_only(prompt_ids)
            return BatchFeature(dict(input_ids=[prompt_ids]), tensor_type="pt")

        mm_data = dict(mm_data)
        audios = mm_data.pop("audios", [])
        assert isinstance(audios, list)

        feature_extractor = self.info.get_feature_extractor(**mm_kwargs)
        mm_kwargs = dict(
            **mm_kwargs,
            sampling_rate=feature_extractor.sampling_rate,
            include_audio_num_chunks=True,
        )

        item_processor_data = dict(**mm_data, audios=audios)

        # some tokenizer kwargs are incompatible with UltravoxProcessor
        tok_kwargs.pop("padding", None)
        tok_kwargs.pop("truncation", None)

        output = super()._call_hf_processor(
            prompt=prompt,
            mm_data=item_processor_data,
            mm_kwargs=mm_kwargs,
            tok_kwargs=tok_kwargs,
        )
        output["audio_features"] = output.pop("audio_values")

        return output

    def _get_mm_fields_config(
        self,
        hf_inputs: BatchFeature,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        num_chunks = hf_inputs.get("audio_num_chunks", torch.zeros(0))
        return dict(
            # to handle longer than 30s audio, each audio might be split
            # into multiple chunks as such, their batch dimension can be
            # higher than the number of audio samples
            audio_features=MultiModalFieldConfig.flat_from_sizes("audio", num_chunks),
            audio_token_len=MultiModalFieldConfig.flat_from_sizes("audio", num_chunks),
            audio_lens=MultiModalFieldConfig.flat_from_sizes("audio", num_chunks),
            # num_chunks can convert audio_chunked to audio batch dimension
            audio_num_chunks=MultiModalFieldConfig.batched("audio"),
            audio_embeds=MultiModalFieldConfig.batched("audio"),
        )

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, Any],
        out_mm_kwargs: MultiModalKwargsItems,
    ) -> Sequence[PromptUpdate]:
        hf_processor = self.info.get_hf_processor(**hf_processor_mm_kwargs)

        replacement_id = hf_processor.audio_replacement_token_id  # type: ignore

        # Each audio can be split into multiple chunks.
        # chunks_start_idx[i] indicates the start index of the chunks
        # belonging to the i-th audio.
        out_mm_data = out_mm_kwargs.get_data()
        num_chunks = out_mm_data.get("audio_num_chunks", torch.zeros(0))
        chunks_start_idx: torch.Tensor = torch.cumsum(
            num_chunks, dim=0, dtype=torch.int32
        )
        chunks_start_idx = torch.cat(
            [torch.tensor([0], dtype=torch.int32), chunks_start_idx]
        )

        def get_replacement_ultravox(item_idx: int):
            start = chunks_start_idx[item_idx]
            end = chunks_start_idx[item_idx + 1]
            audio_token_len = out_mm_data["audio_token_len"][start:end].sum()
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
        T_pad = (T + self.stack_factor - 1) // self.stack_factor * self.stack_factor
        audio_embeds = F.pad(audio_embeds, (0, 0, 0, T_pad - T))
        B, T, C = audio_embeds.shape
        audio_embeds = audio_embeds.view(
            B, T // self.stack_factor, C * self.stack_factor
        )
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

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config.is_decoder = False

    @property
    def max_context_length(self):
        return (
            self.config.max_source_positions
            * self.conv1.stride[0]
            * self.conv2.stride[0]
        )

    def get_attention_mask_by_audio_len(
        self, audio_lens: torch.Tensor | None, hidden_states: torch.Tensor
    ):
        """
        Create attention mask based on audio lengths to mask out padding tokens
        For each sample in batch:
        - Convert raw audio length to feature length after convolutions
        - Create bool mask: True for valid positions and False for padding
        - Convert to attention mask format expected by transformer layers
        (1.0 for positions to attend to, large negative for positions to ignore)
        This masking ensures consistent behavior between training and inference
        by preventing the model from attending to padding tokens in both cases
        """
        if audio_lens is None:
            return None

        audio_feature_len = self._get_feat_extract_output_lengths(audio_lens)
        max_seq_len = hidden_states.shape[1]
        attention_mask = torch.arange(max_seq_len, device=hidden_states.device)[
            None, :
        ].lt(audio_feature_len.view(-1, 1))
        attention_mask = self.get_extended_attention_mask(
            attention_mask,
            None,
            dtype=hidden_states.dtype,
        )
        return attention_mask

    def forward(
        self,
        input_features: torch.Tensor,
        audio_lens: torch.Tensor | None = None,
    ):
        expected_seq_length = self.max_context_length
        if input_features.shape[-1] > expected_seq_length:
            raise ValueError(
                f"Whisper expects the mel input features to be of length "
                f"{expected_seq_length} or less, but found "
                f"{input_features.shape[-1]}. Make sure to pad the input mel "
                f"features to {expected_seq_length}."
            )

        inputs_embeds = nn.functional.gelu(self.conv1(input_features))
        inputs_embeds = nn.functional.gelu(self.conv2(inputs_embeds))

        inputs_embeds = inputs_embeds.permute(0, 2, 1)
        embed_pos = self.embed_positions.weight[: inputs_embeds.size(-2)]

        hidden_states = inputs_embeds + embed_pos
        hidden_states = nn.functional.dropout(
            hidden_states, p=self.dropout, training=self.training
        )

        attention_mask = self.get_attention_mask_by_audio_len(audio_lens, hidden_states)

        for encoder_layer in self.layers:
            layer_outputs = encoder_layer(
                hidden_states,
                attention_mask,
                layer_head_mask=None,
            )

            hidden_states = layer_outputs[0]

        hidden_states = self.layer_norm(hidden_states)
        return hidden_states


@MULTIMODAL_REGISTRY.register_processor(
    UltravoxMultiModalProcessor,
    info=UltravoxProcessingInfo,
    dummy_inputs=UltravoxDummyInputsBuilder,
)
class UltravoxModel(nn.Module, SupportsMultiModal, SupportsPP, SupportsLoRA):
    merge_by_field_config = True

    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
    }

    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_prefix={"audio_tower.model.encoder.": "audio_tower."}
    )

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None:
        if modality.startswith("audio"):
            return "<|audio|>"

        raise ValueError("Only audio modality is supported")

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config: UltravoxConfig = vllm_config.model_config.hf_config
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
                )
            )
        self.multi_modal_projector = UltravoxProjector(config)
        self.language_model = init_vllm_registered_model(
            vllm_config=vllm_config,
            hf_config=config.wrapped_model_config,
            prefix=maybe_prefix(prefix, "language_model"),
        )
        if config.text_model_id is not None:
            # this prefix is not for initialization, but for loading weights
            # note the trailing dot
            self.secondary_weights.append(
                DefaultModelLoader.Source(
                    model_or_path=config.text_model_id,
                    revision=None,
                    prefix="language_model.",
                )
            )

        self.make_empty_intermediate_tensors = (
            self.language_model.make_empty_intermediate_tensors
        )

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
        self, input_features: torch.Tensor, audio_lens: torch.Tensor
    ) -> torch.Tensor:
        audio_features = input_features.to(self.audio_tower.dtype)
        batch_size = audio_features.size(0)
        audio_embeddings = []

        # Process audio features in batches to keep memory usage predictable
        for start in range(0, batch_size, _MAX_ENCODER_BATCH_SIZE):
            end = min(start + _MAX_ENCODER_BATCH_SIZE, batch_size)
            # Process through audio tower
            batch_features = self.audio_tower(
                audio_features[start:end], audio_lens[start:end]
            )
            batch_features = batch_features.to(self.audio_tower.dtype)

            # Process through projector
            batch_embeddings = self.multi_modal_projector(batch_features)
            audio_embeddings.append(batch_embeddings)

        # Concatenate results
        audio_embeddings = torch.cat(audio_embeddings, dim=0)
        return audio_embeddings

    def _parse_and_validate_audio_input(
        self, **kwargs: object
    ) -> UltravoxAudioInputs | None:
        audio_features = kwargs.pop("audio_features", None)
        audio_embeds = kwargs.pop("audio_embeds", None)
        audio_lens = kwargs.pop("audio_lens", None)
        audio_token_len = kwargs.pop("audio_token_len", None)
        audio_num_chunks = kwargs.pop("audio_num_chunks", None)

        if audio_features is None and audio_embeds is None:
            return None

        if audio_features is not None:
            return UltravoxAudioFeatureInputs(
                type="audio_features",
                data=audio_features,
                lens=audio_lens,
                token_len=audio_token_len,
                num_chunks=audio_num_chunks,
            )

        if audio_embeds is not None:
            return UltravoxAudioEmbeddingInputs(type="audio_embeds", data=audio_embeds)

        raise AssertionError("This line should be unreachable.")

    def _process_audio_input(
        self,
        audio_input: UltravoxAudioInputs,
    ) -> NestedTensors | tuple[torch.Tensor, ...]:
        if audio_input["type"] == "audio_embeds":
            return audio_input["data"]

        # Pad and concatenate audio features
        # [[B1, 80, M1], [B2, 80, M2]] -> [B1+B2, 80, max(M1, M2)]
        audio_features = pad_and_concat_to_dim3(audio_input["data"])

        audio_lens = audio_input["lens"]
        audio_token_len = audio_input["token_len"]

        embeddings = self._audio_features_to_embeddings(audio_features, audio_lens)

        # We should flatten and concatenate embeddings based on token lengths
        # For example, with token_len = [4, 2, 3], flattened_embeddings will be
        # concat(embeddings[0][:4], embeddings[1][:2], embeddings[2][:3])

        # Create a mask of valid indices based on token lengths
        max_len = embeddings.shape[1]
        indices = torch.arange(max_len, device=embeddings.device).expand(
            embeddings.shape[0], -1
        )
        mask = indices < audio_token_len[:, None]
        # Apply mask and flatten
        flattened_embeddings = embeddings[mask]

        # Return one tensor per input audio
        embed_lens = [
            chunk_lens.sum().item()
            for chunk_lens in audio_token_len.split(audio_input["num_chunks"].tolist())
        ]
        return flattened_embeddings.split(embed_lens)

    def get_language_model(self) -> torch.nn.Module:
        return self.language_model

    def embed_multimodal(self, **kwargs: object) -> MultiModalEmbeddings:
        audio_input = self._parse_and_validate_audio_input(**kwargs)
        if audio_input is None:
            return []
        audio_embeddings = self._process_audio_input(audio_input)
        return audio_embeddings

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
        intermediate_tensors: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor | IntermediateTensors:
        """Run forward pass for Ultravox

        One key thing to understand is the `input_ids` already accounts for the
        positions of the to-be-inserted audio embeddings. The to-be-inserted
        audio has a size that is essentially 6.25 tokens per second of audio.

        This way, the `positions` and `attn_metadata` are consistent
        with the `input_ids`.

        Args:
            input_ids: Flattened (concatenated) input_ids corresponding to a
                batch.
            positions: Position indices for the input tokens.
            intermediate_tensors: Intermediate tensors from prior forward pass.
            inputs_embeds: Optional tensor of input embeddings.

        """

        if intermediate_tensors is not None:
            inputs_embeds = None

        language_model = self.language_model
        if hasattr(language_model, "language_model"):
            language_model = language_model.language_model

        hidden_states = language_model.model(
            input_ids, positions, intermediate_tensors, inputs_embeds=inputs_embeds
        )
        return hidden_states

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.language_model.compute_logits(hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(self, ignore_unexpected_prefixes=["audio_tower."])
        return loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)


def pad_and_concat_to_dim3(
    features: torch.Tensor | list[torch.Tensor] | list[list[torch.Tensor]],
) -> torch.Tensor:
    """
    Pad and concatenate a list of tensors.

    output:
        Tensor of shape [B, C, M] where M is the maximum length of the input
        tensors, B is the sum of the batch sizes of the input tensors.
        C must be the same for all input tensors.
    """
    if isinstance(features, torch.Tensor):
        if features.ndim > 3:
            # Flatten [B, N, 80, M] -> [B * N, 80, M]
            features = flatten_bn(features)

        return features

    features = [pad_and_concat_to_dim3(f) for f in features]

    max_len = max(f.shape[-1] for f in features)
    # Ensure all features have dim=3
    features = [f.view(-1, *f.shape[-2:]) for f in features]
    # Pad and concatenate:
    # [[B1, 80, M1], [B2, 80, M2]] -> [B1+B2, 80, max(M1, M2)]
    features = [F.pad(f, (0, max_len - f.shape[-1])) for f in features]
    return torch.cat(features)
