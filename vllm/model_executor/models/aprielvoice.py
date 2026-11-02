# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""PyTorch AprielVoice model."""

import math
from collections.abc import Iterable, Mapping, Sequence
from functools import cached_property
from typing import Annotated, Any, Literal, TypeAlias

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from transformers import BatchFeature, Gemma4ImageProcessor, ProcessorMixin
from transformers.masking_utils import create_bidirectional_mask
from transformers.models.whisper import WhisperFeatureExtractor
from transformers.models.whisper.modeling_whisper import (
    WhisperEncoder as _HFWhisperEncoder,
)

from vllm.config import VllmConfig
from vllm.config.multimodal import (
    AudioDummyOptions,
    BaseDummyOptions,
    ImageDummyOptions,
)
from vllm.inputs import MultiModalDataDict
from vllm.model_executor.layers.activation import MulAndSilu, get_act_fn
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.model_loader import DefaultModelLoader
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import (
    MultiModalFieldConfig,
    MultiModalKwargsItems,
    NestedTensors,
)
from vllm.multimodal.parse import MultiModalDataItems, MultiModalDataParser
from vllm.multimodal.processing import (
    BaseDummyInputsBuilder,
    BaseMultiModalProcessor,
    BaseProcessingInfo,
    PromptReplacement,
    PromptUpdate,
    PromptUpdateDetails,
)
from vllm.renderers import TokenizeParams
from vllm.sequence import IntermediateTensors
from vllm.transformers_utils.configs.aprielvoice import AprielVoiceConfig
from vllm.utils.gpu_sync_debug import gpu_sync_allowed
from vllm.utils.tensor_schema import TensorSchema, TensorShape

from .gemma4_mm import _SUPPORTED_SOFT_TOKENS, _get_max_soft_tokens
from .interfaces import (
    MultiModalEmbeddings,
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
_IMAGE_PLACEHOLDER = "<|image|>"
_MAX_ENCODER_BATCH_SIZE = 16
_WHISPER_MEL_CONTEXT_SIZE = 3000  # 30s at 16kHz with hop_length=160
_SHORT_AUDIO_THRESHOLD_SECONDS = 2.0
_SHORT_AUDIO_SILENCE_SECONDS = 3.0


def _prepend_silence_to_short_audios(
    audios: list[np.ndarray], sampling_rate: int
) -> list[np.ndarray]:
    threshold_samples = int(_SHORT_AUDIO_THRESHOLD_SECONDS * sampling_rate)
    silence_samples = int(_SHORT_AUDIO_SILENCE_SECONDS * sampling_rate)

    return [
        np.pad(audio, (silence_samples, 0), mode="constant")
        if len(audio) < threshold_samples
        else audio
        for audio in audios
    ]


class AprielVoiceAudioFeatureInputs(TensorSchema):
    """Dimensions:
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
    token_len: Annotated[torch.Tensor, TensorShape("bn")]
    num_chunks: Annotated[torch.Tensor, TensorShape("n")]


class AprielVoiceAudioEmbeddingInputs(TensorSchema):
    """Dimensions:
    - b: batch size
    - na: number of audios
    - afs: audio feature size
    - hs: hidden size
    """

    type: Literal["audio_embeds"]
    data: Annotated[
        torch.Tensor | list[torch.Tensor], TensorShape("b", "na", "afs", "hs")
    ]


AprielVoiceAudioInputs: TypeAlias = (
    AprielVoiceAudioFeatureInputs | AprielVoiceAudioEmbeddingInputs
)


class AprielVoiceProcessingInfo(BaseProcessingInfo):
    def get_hf_config(self) -> AprielVoiceConfig:
        return self.ctx.get_hf_config(AprielVoiceConfig)

    def get_hf_processor(self, **kwargs: object) -> ProcessorMixin:
        config = self.get_hf_config()
        hf_processor = self.ctx.get_hf_processor(**kwargs)
        hf_processor.audio_token_replacement = _AUDIO_PLACEHOLDER_OVERRIDE
        hf_processor.audio_replacement_token_id = config.audio_token_index
        if hf_processor.audio_context_size is None:
            hf_processor.audio_context_size = _WHISPER_MEL_CONTEXT_SIZE
        return hf_processor

    def get_feature_extractor(self, **kwargs: object) -> WhisperFeatureExtractor:
        hf_processor = self.get_hf_processor(**kwargs)
        audio_processor = hf_processor.audio_processor  # type: ignore
        if isinstance(audio_processor, WhisperFeatureExtractor):
            return audio_processor
        feature_extractor = audio_processor.feature_extractor  # type: ignore
        assert isinstance(feature_extractor, WhisperFeatureExtractor)
        return feature_extractor

    @cached_property
    def image_processor(self) -> Gemma4ImageProcessor:
        vision_config = self.get_hf_config().vision_config
        if vision_config is None:
            raise ValueError("This AprielVoice checkpoint has no Gemma4 vision tower.")
        return Gemma4ImageProcessor(
            patch_size=vision_config.patch_size,
            pooling_kernel_size=vision_config.pooling_kernel_size,
            max_soft_tokens=getattr(
                vision_config,
                "default_output_length",
                Gemma4ImageProcessor.max_soft_tokens,
            ),
        )

    def get_image_processor_kwargs(
        self, mm_kwargs: Mapping[str, object]
    ) -> dict[str, object]:
        merged_kwargs = self.ctx.get_merged_mm_kwargs(mm_kwargs)
        nested_kwargs = merged_kwargs.get("images_kwargs")
        image_kwargs = dict(nested_kwargs) if isinstance(nested_kwargs, Mapping) else {}
        image_kwargs.update(
            (key, value)
            for key, value in merged_kwargs.items()
            if key in Gemma4ImageProcessor.valid_kwargs.__annotations__
        )
        max_soft_tokens, _ = _get_max_soft_tokens(merged_kwargs)
        if max_soft_tokens is not None:
            if (
                not isinstance(max_soft_tokens, int)
                or max_soft_tokens not in _SUPPORTED_SOFT_TOKENS
            ):
                raise ValueError(
                    f"Unsupported max_soft_tokens value: {max_soft_tokens}. "
                    f"Valid values are {_SUPPORTED_SOFT_TOKENS}."
                )
            image_kwargs["max_soft_tokens"] = max_soft_tokens
        image_kwargs["return_tensors"] = "pt"
        return image_kwargs

    def get_default_tok_params(self) -> TokenizeParams:
        return super().get_default_tok_params().with_kwargs(add_special_tokens=False)

    def get_data_parser(self):
        feature_extractor = self.get_feature_extractor()
        return MultiModalDataParser(
            target_sr=feature_extractor.sampling_rate,
            target_channels=self.get_target_channels(),
            expected_hidden_size=self._get_expected_hidden_size(),
        )

    def get_target_channels(self) -> int:
        return 1

    def get_supported_mm_limits(self) -> Mapping[str, int | None]:
        limits: dict[str, int | None] = {"audio": None}
        if self.get_hf_config().vision_config is not None:
            limits["image"] = None
        return limits


class AprielVoiceDummyInputsBuilder(
    BaseDummyInputsBuilder[AprielVoiceProcessingInfo],
):
    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        num_images = mm_counts.get("image", 0)
        num_audios = mm_counts.get("audio", 0)
        return (
            _IMAGE_PLACEHOLDER * num_images + _AUDIO_PLACEHOLDER_OVERRIDE * num_audios
        )

    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        mm_options: Mapping[str, BaseDummyOptions],
    ) -> MultiModalDataDict:
        data: dict[str, Any] = {}
        num_images = mm_counts.get("image", 0)
        if num_images:
            image_processor = self.info.image_processor
            image_kwargs = self.info.get_image_processor_kwargs({})
            max_tokens = image_kwargs.get(
                "max_soft_tokens", image_processor.max_soft_tokens
            )
            assert isinstance(max_tokens, int)
            # An exact patch grid makes dummy profiling cover the full budget.
            grid_height = math.isqrt(max_tokens)
            while max_tokens % grid_height:
                grid_height -= 1
            unit = image_processor.patch_size * image_processor.pooling_kernel_size
            image_overrides = mm_options.get("image")
            assert image_overrides is None or isinstance(
                image_overrides, ImageDummyOptions
            )
            data["image"] = self._get_dummy_images(
                width=max_tokens // grid_height * unit,
                height=grid_height * unit,
                num_images=num_images,
                overrides=image_overrides,
            )

        num_audios = mm_counts.get("audio", 0)
        if num_audios:
            feature_extractor = self.info.get_feature_extractor()
            audio_len = (
                feature_extractor.chunk_length
                * feature_extractor.sampling_rate
                * _MAX_ENCODER_BATCH_SIZE
            )
            audio_overrides = mm_options.get("audio")
            assert audio_overrides is None or isinstance(
                audio_overrides, AudioDummyOptions
            )
            data["audio"] = self._get_dummy_audios(
                length=audio_len,
                num_audios=num_audios,
                overrides=audio_overrides,
            )
        return data


class AprielVoiceMultiModalProcessor(
    BaseMultiModalProcessor[AprielVoiceProcessingInfo],
):
    def _process_mm_features(
        self,
        images: list | None = None,
        audios: list[np.ndarray] | None = None,
        **kwargs: object,
    ) -> BatchFeature:
        output = BatchFeature()
        if images:
            image_output = self.info.image_processor(
                images=images,
                **self.info.get_image_processor_kwargs(kwargs),
            )
            image_output["pixel_position_ids"] = image_output.pop("image_position_ids")
            output.update(image_output)

        if audios:
            image_keys = Gemma4ImageProcessor.valid_kwargs.__annotations__
            audio_kwargs = {
                key: value
                for key, value in kwargs.items()
                if key not in image_keys
                and key not in ("images_kwargs", "audio_kwargs", "text_kwargs")
            }
            nested_audio_kwargs = kwargs.get("audio_kwargs")
            if isinstance(nested_audio_kwargs, Mapping):
                audio_kwargs.update(nested_audio_kwargs)
            for key in ("add_special_tokens", "padding", "truncation"):
                audio_kwargs.pop(key, None)
            hf_processor = self.info.get_hf_processor(**audio_kwargs)
            feature_extractor = self.info.get_feature_extractor(**audio_kwargs)
            audios = _prepend_silence_to_short_audios(
                audios, feature_extractor.sampling_rate
            )
            audio_kwargs.update(
                sampling_rate=feature_extractor.sampling_rate,
                include_audio_num_chunks=True,
                return_tensors="pt",
            )
            audio_output = hf_processor(text=None, audios=audios, **audio_kwargs)
            audio_output["audio_features"] = audio_output.pop("audio_values")
            output.update(audio_output)
        return output

    def _call_hf_processor(
        self,
        hf_data: Mapping[str, object],
        hf_kwargs: Mapping[str, object],
    ) -> BatchFeature:
        # vLLM 0.32 renders/tokenizes text before multimodal processing and
        # normalizes the audio input key; our checkpoint uses "audios".
        data = dict(hf_data)
        if "audio" in data:
            data["audios"] = data.pop("audio")
        return self.info.ctx.call_hf_processor(
            self._process_mm_features, data, hf_kwargs
        )

    def _get_mm_fields_config(
        self,
        hf_inputs: BatchFeature,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        num_chunks = hf_inputs.get("audio_num_chunks", torch.zeros(0))
        return dict(
            pixel_values=MultiModalFieldConfig.batched("image"),
            pixel_position_ids=MultiModalFieldConfig.batched("image"),
            num_soft_tokens_per_image=MultiModalFieldConfig.batched(
                "image", keep_on_cpu=True
            ),
            audio_features=MultiModalFieldConfig.flat_from_sizes("audio", num_chunks),
            audio_token_len=MultiModalFieldConfig.flat_from_sizes("audio", num_chunks),
            audio_lens=MultiModalFieldConfig.flat_from_sizes("audio", num_chunks),
            audio_num_chunks=MultiModalFieldConfig.batched("audio", keep_on_cpu=True),
            audio_embeds=MultiModalFieldConfig.batched("audio"),
        )

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, Any],
        out_mm_kwargs: MultiModalKwargsItems,
    ) -> Sequence[PromptUpdate]:
        config = self.info.get_hf_config()
        updates: list[PromptUpdate] = []
        if "image" in mm_items:

            def get_replacement_image(item_idx: int):
                item = out_mm_kwargs["image"][item_idx]
                num_soft_tokens = item["num_soft_tokens_per_image"].data
                assert isinstance(num_soft_tokens, torch.Tensor)
                num_tokens = int(num_soft_tokens)
                gemma_config = config.wrapped_model_config
                token_ids = (
                    [gemma_config.boi_token_id]
                    + [config.image_token_index] * num_tokens
                    + [gemma_config.eoi_token_id]
                )
                return PromptUpdateDetails.select_token_id(
                    token_ids, config.image_token_index
                )

            updates.append(
                PromptReplacement(
                    modality="image",
                    target=self.info.get_tokenizer().encode(
                        _IMAGE_PLACEHOLDER, add_special_tokens=False
                    ),
                    replacement=get_replacement_image,
                )
            )

        if "audio" in mm_items:

            def get_replacement_audio(item_idx: int):
                item = out_mm_kwargs["audio"][item_idx]
                if "audio_embeds" in item:
                    audio_embeds = item["audio_embeds"].data
                    assert isinstance(audio_embeds, torch.Tensor)
                    num_tokens = audio_embeds.shape[-2]
                else:
                    audio_token_len = item["audio_token_len"].data
                    assert isinstance(audio_token_len, torch.Tensor)
                    num_tokens = int(audio_token_len.sum())
                return [config.audio_token_index] * num_tokens

            updates.append(
                PromptReplacement(
                    modality="audio",
                    target=self.info.get_tokenizer().encode(
                        _AUDIO_PLACEHOLDER_OVERRIDE, add_special_tokens=False
                    ),
                    replacement=get_replacement_audio,
                )
            )
        return updates


class StackAudioFrames(nn.Module):
    """Stack the audio embedding frames to reduce the sequence length by a factor
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


class AprielVoiceProjector(nn.Module):
    def __init__(self, config: AprielVoiceConfig):
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

        if config.projector_ln_mid:
            self.ln_mid: nn.Module = RMSNorm(dim_mid)
            self.ln_post = nn.Identity()
        else:
            self.ln_mid = nn.Identity()
            self.ln_post = RMSNorm(dim_out)

    def forward(
        self, audio_features: torch.Tensor, audio_token_len: torch.Tensor
    ) -> torch.Tensor:
        audio_features = self._pad_and_stack(audio_features)
        audio_features = self.ln_pre(audio_features)
        hidden_states = self.linear_1(audio_features)
        hidden_states = self.act(hidden_states)
        hidden_states = self.ln_mid(hidden_states)
        hidden_states = self.linear_2(hidden_states)
        hidden_states = self.ln_post(hidden_states)
        return hidden_states


class WhisperEncoder(_HFWhisperEncoder):
    """Whisper encoder with variable-length audio support.

    Differences from stock HF WhisperEncoder:
    1. base_model_prefix set for direct .from_pretrained on the encoder.
    2. Accepts audio shorter than 30s (sliced positional embeddings).
    3. Accepts optional audio_lens to mask padding.
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
        if audio_lens is None:
            return None

        audio_feature_len = self._get_feat_extract_output_lengths(audio_lens)
        audio_feature_len = audio_feature_len.to(hidden_states.device)
        max_seq_len = hidden_states.shape[1]
        attention_mask = torch.arange(max_seq_len, device=hidden_states.device)[
            None, :
        ].lt(audio_feature_len.view(-1, 1))
        # Transformers may inspect the padding mask when choosing the SDPA path.
        with gpu_sync_allowed():
            attention_mask = create_bidirectional_mask(
                config=self.config,
                inputs_embeds=hidden_states,
                attention_mask=attention_mask,
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
            hidden_states = encoder_layer(
                hidden_states,
                attention_mask,
            )
            if isinstance(hidden_states, tuple):
                hidden_states = hidden_states[0]

        hidden_states = self.layer_norm(hidden_states)
        return hidden_states


@MULTIMODAL_REGISTRY.register_processor(
    AprielVoiceMultiModalProcessor,
    info=AprielVoiceProcessingInfo,
    dummy_inputs=AprielVoiceDummyInputsBuilder,
)
class AprielVoiceModel(nn.Module, SupportsMultiModal, SupportsPP):
    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_prefix={"audio_tower.model.encoder.": "audio_tower."}
    )

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None:
        if modality.startswith("audio"):
            return _AUDIO_PLACEHOLDER_OVERRIDE
        if modality == "image":
            return _IMAGE_PLACEHOLDER
        raise ValueError(f"Unsupported modality: {modality}")

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config: AprielVoiceConfig = vllm_config.model_config.hf_config
        multimodal_config = vllm_config.model_config.multimodal_config
        self.config = config
        self.multi_modal_config = multimodal_config
        assert self.multi_modal_config

        mm_token_ids = [config.audio_token_index]
        if config.vision_config is not None:
            mm_token_ids.append(config.image_token_index)
        self.configure_mm_token_handling(config.vocab_size, mm_token_ids)

        self.secondary_weights = []
        if config.audio_model_id is not None:
            self.secondary_weights.append(
                DefaultModelLoader.Source(
                    model_or_path=config.audio_model_id,
                    revision=vllm_config.model_config.revision,
                    prefix="audio_tower.",
                )
            )
        if config.text_model_id is not None:
            self.secondary_weights.append(
                DefaultModelLoader.Source(
                    model_or_path=config.text_model_id,
                    revision=vllm_config.model_config.revision,
                    prefix="language_model.",
                )
            )

        with self._mark_tower_model(vllm_config, "audio"):
            self.audio_tower = WhisperEncoder(config.audio_config)
            self.multi_modal_projector = AprielVoiceProjector(config)

        with self._mark_language_model(vllm_config):
            self.language_model = init_vllm_registered_model(
                vllm_config=vllm_config,
                hf_config=config.wrapped_model_config,
                prefix=maybe_prefix(prefix, "language_model"),
            )

        self.mm_prefix_clamp_sliding_window = getattr(
            self.language_model, "mm_prefix_clamp_sliding_window", False
        )
        if config.vision_config is not None:
            self.language_model.configure_mm_token_handling(
                config.vocab_size, mm_token_ids
            )

        self.make_empty_intermediate_tensors = (
            self.language_model.make_empty_intermediate_tensors
        )

    def _audio_features_to_embeddings(
        self,
        input_features: torch.Tensor,
        audio_lens: torch.Tensor,
        audio_token_len: torch.Tensor,
    ) -> torch.Tensor:
        audio_features = input_features.to(self.audio_tower.dtype)
        batch_size = audio_features.size(0)
        audio_embeddings = []

        for start in range(0, batch_size, _MAX_ENCODER_BATCH_SIZE):
            end = min(start + _MAX_ENCODER_BATCH_SIZE, batch_size)
            batch_features = self.audio_tower(
                audio_features[start:end], audio_lens[start:end]
            )
            batch_features = batch_features.to(self.audio_tower.dtype)

            batch_embeddings = self.multi_modal_projector(
                batch_features, audio_token_len[start:end]
            )
            audio_embeddings.append(batch_embeddings)

        audio_embeddings = torch.cat(audio_embeddings, dim=0)
        return audio_embeddings

    def _parse_and_validate_audio_input(
        self, **kwargs: object
    ) -> AprielVoiceAudioInputs | None:
        audio_features = kwargs.pop("audio_features", None)
        audio_embeds = kwargs.pop("audio_embeds", None)
        audio_lens = kwargs.pop("audio_lens", None)
        audio_token_len = kwargs.pop("audio_token_len", None)
        audio_num_chunks = kwargs.pop("audio_num_chunks", None)

        if audio_features is None and audio_embeds is None:
            return None

        if audio_features is not None:
            return AprielVoiceAudioFeatureInputs(
                type="audio_features",
                data=audio_features,
                lens=audio_lens,
                token_len=audio_token_len,
                num_chunks=audio_num_chunks,
            )

        if audio_embeds is not None:
            return AprielVoiceAudioEmbeddingInputs(
                type="audio_embeds", data=audio_embeds
            )

        raise AssertionError("This line should be unreachable.")

    def _process_audio_input(
        self,
        audio_input: AprielVoiceAudioInputs,
    ) -> NestedTensors | tuple[torch.Tensor, ...]:
        if audio_input["type"] == "audio_embeds":
            return audio_input["data"]

        audio_features = pad_and_concat_to_dim3(audio_input["data"])

        audio_lens = audio_input["lens"]
        audio_token_len = audio_input["token_len"]

        embeddings = self._audio_features_to_embeddings(
            audio_features, audio_lens, audio_token_len
        )

        max_len = embeddings.shape[1]
        indices = torch.arange(max_len, device=embeddings.device).expand(
            embeddings.shape[0], -1
        )
        mask = indices < audio_token_len[:, None]
        # Variable-length audio compaction reads the output sizes on the host.
        with gpu_sync_allowed():
            flattened_embeddings = embeddings[mask]
            embed_lens = [
                chunk_lens.sum().item()
                for chunk_lens in audio_token_len.split(
                    audio_input["num_chunks"].tolist()
                )
            ]
            return flattened_embeddings.split(embed_lens)

    def embed_multimodal(self, **kwargs: object) -> MultiModalEmbeddings:
        if kwargs.get("image_embeds") is not None:
            raise ValueError(
                "AprielVoice does not support precomputed image embeddings."
            )

        embeddings: list[torch.Tensor] = []
        processed_modalities: set[str] = set()
        # Match Gemma4's dispatch order to vLLM's batched modality kwargs.
        for key in kwargs:
            if key == "pixel_values" and "image" not in processed_modalities:
                processed_modalities.add("image")
                if self.config.vision_config is None:
                    raise ValueError("This AprielVoice checkpoint has no vision tower.")
                embeddings.extend(
                    self.language_model.embed_multimodal(
                        pixel_values=kwargs["pixel_values"],
                        pixel_position_ids=kwargs.get("pixel_position_ids"),
                    )
                )
            elif (
                key in ("audio_features", "audio_embeds")
                and "audio" not in processed_modalities
            ):
                processed_modalities.add("audio")
                audio_input = self._parse_and_validate_audio_input(**kwargs)
                if audio_input is not None:
                    embeddings.extend(self._process_audio_input(audio_input))
        return embeddings

    def embed_input_ids(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: MultiModalEmbeddings | None = None,
        *,
        is_multimodal: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.config.vision_config is not None:
            return self.language_model.embed_input_ids(
                input_ids,
                multimodal_embeddings=multimodal_embeddings,
                is_multimodal=is_multimodal,
            )
        if multimodal_embeddings is None or is_multimodal is None:
            return super().embed_input_ids(input_ids)

        return super().embed_input_ids(
            input_ids,
            multimodal_embeddings=multimodal_embeddings,
            is_multimodal=is_multimodal,
        )

    def get_language_model(self) -> nn.Module:
        """Return the innermost causal LM (the Gemma4 text model).

        ``self.language_model`` is a ``Gemma4ForConditionalGeneration``, which
        wraps the actual ``Gemma4ForCausalLM`` under its own ``language_model``
        attribute. The default ``SupportsMultiModal.get_language_model()``
        stops at the first child with ``embed_input_ids`` and so returns the
        outer wrapper, which has no ``.model``. Speculative decoding walks
        ``get_language_model().model.embed_tokens`` when wiring the drafter
        (``SpecDecodeBaseProposer._maybe_share_embeddings``), so unwrap both
        levels here.
        """
        language_model = self.language_model
        if hasattr(language_model, "language_model"):
            language_model = language_model.language_model
        return language_model

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor | IntermediateTensors:
        if intermediate_tensors is not None:
            inputs_embeds = None

        if self.config.vision_config is not None:
            return self.language_model(
                input_ids=input_ids,
                positions=positions,
                intermediate_tensors=intermediate_tensors,
                inputs_embeds=inputs_embeds,
                **kwargs,
            )

        hidden_states = self.get_language_model().model(
            input_ids, positions, intermediate_tensors, inputs_embeds=inputs_embeds
        )
        return hidden_states

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.language_model.compute_logits(hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        mapper = self.hf_to_vllm_mapper
        if self.config.wrapped_model_config.model_type == "gemma4":
            # Checkpoint shards can interleave projector and backbone weights.
            # Normalize the nested Gemma4 names before the outer loader checks
            # tied embeddings, so its check spans every backbone group.
            backbone_prefixes = self.language_model.hf_to_vllm_mapper.orig_to_new_prefix
            mapper = WeightsMapper(
                orig_to_new_prefix={
                    **mapper.orig_to_new_prefix,
                    **{
                        "language_model." + source: (
                            "language_model." + target if target is not None else None
                        )
                        for source, target in backbone_prefixes.items()
                    },
                }
            )
        loader = AutoWeightsLoader(self, ignore_unexpected_prefixes=["audio_tower."])
        return loader.load_weights(weights, mapper=mapper)


def pad_and_concat_to_dim3(
    features: torch.Tensor | list[torch.Tensor] | list[list[torch.Tensor]],
) -> torch.Tensor:
    if isinstance(features, torch.Tensor):
        if features.ndim > 3:
            features = flatten_bn(features)

        return features

    tensors = [pad_and_concat_to_dim3(f) for f in features]
    max_len = max(f.shape[-1] for f in tensors)
    tensors = [f.view(-1, *f.shape[-2:]) for f in tensors]
    tensors = [F.pad(f, (0, max_len - f.shape[-1])) for f in tensors]
    return torch.cat(tensors)
