# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Apertus 1.5 multimodal model."""

from collections.abc import Iterable, Mapping, Sequence
from math import isqrt
from typing import Annotated, Any, Literal

import torch
import torch.nn.functional as F
from transformers import (
    Apertus1p5VisionTokenizerModel,
    AutoConfig,
    AutoModel,
    PretrainedConfig,
)

from vllm.config import VllmConfig
from vllm.config.multimodal import BaseDummyOptions
from vllm.distributed import get_pp_group
from vllm.inputs import MultiModalDataDict, MultiModalInput, mm_input
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import (
    MultiModalFieldConfig,
    MultiModalKwargsItems,
    PlaceholderRange,
)
from vllm.multimodal.parse import (
    AudioProcessorItems,
    ImageProcessorItems,
    MultiModalDataParser,
)
from vllm.multimodal.processing import (
    BaseDummyInputsBuilder,
    BaseMultiModalProcessor,
    BaseProcessingInfo,
    ProcessorInputs,
    PromptUpdate,
    TimingContext,
)
from vllm.sequence import IntermediateTensors
from vllm.utils.tensor_schema import TensorSchema, TensorShape
from vllm.utils.torch_utils import set_default_torch_dtype

from .apertus import ApertusForCausalLM, ApertusModel
from .interfaces import (
    MultiModalEmbeddings,
    SupportsEagle,
    SupportsEagle3,
    SupportsLoRA,
    SupportsMultiModal,
    SupportsPP,
)
from .utils import (
    AutoWeightsLoader,
    PPMissingLayer,
    WeightsMapper,
    maybe_prefix,
)

_IMAGE_TOKEN_BUDGET_OVERHEAD = 512
_MAX_AUDIO_SECONDS = 300
_AUDIO_TOKEN_BUDGET_OVERHEAD = 4

_DEFAULT_IMAGE_PLACEHOLDER = "<|image|>"
_DEFAULT_BOI_TOKEN = "<|img_start|>"
_DEFAULT_EOI_TOKEN = "<|img_end|>"
_DEFAULT_AUDIO_PLACEHOLDER = "<|audio|>"
_DEFAULT_AUDIO_START_TOKEN = "<|audio_start|>"
_DEFAULT_AUDIO_END_TOKEN = "<|audio_end|>"

_DEFAULT_IMAGE_TOKEN_ID = 131079
_DEFAULT_AUDIO_TOKEN_ID = 131085
_DEFAULT_IMAGE_TOKEN_OFFSET = 131272
_DEFAULT_AUDIO_TOKEN_OFFSET = 262344
_DEFAULT_IMAGE_START_TOKEN_ID = 131073
_DEFAULT_IMAGE_END_TOKEN_ID = 131074
_DEFAULT_AUDIO_START_TOKEN_ID = 131080
_DEFAULT_AUDIO_END_TOKEN_ID = 131081


class Apertus1p5ImageInputs(TensorSchema):
    """Processed image inputs for the Apertus vision tokenizer."""

    type: Literal["pixel_values"]

    pixel_values: Annotated[
        torch.Tensor | list[torch.Tensor],
        TensorShape("ni", 3, "h", "w", dynamic_dims={"h", "w"}),
    ]


class Apertus1p5AudioInputs(TensorSchema):
    """Processed audio inputs for the Apertus audio tokenizer."""

    type: Literal["audio_values"]

    audio_values: Annotated[
        torch.Tensor | list[torch.Tensor],
        TensorShape("na", "t", dynamic_dims={"t"}),
    ]


def _pad_logits_to_input_vocab(
    logits: torch.Tensor, input_vocab_size: int
) -> torch.Tensor:
    # Keep input-only token IDs unsampleable while preserving the expected shape.
    return F.pad(
        logits,
        (0, input_vocab_size - logits.shape[-1]),
        value=float("-inf"),
    )


def _init_component_model(
    component_config: PretrainedConfig,
    model_cls: type[torch.nn.Module] | None = None,
) -> torch.nn.Module:
    config_dict = component_config.to_dict()
    config = AutoConfig.for_model(config_dict.pop("model_type"), **config_dict)
    return AutoModel.from_config(config) if model_cls is None else model_cls(config)


class Apertus1p5ProcessingInfo(BaseProcessingInfo):
    def get_data_parser(self) -> MultiModalDataParser:
        audio_feature_extractor = self.get_hf_processor().feature_extractor
        return MultiModalDataParser(
            target_sr=audio_feature_extractor.sampling_rate,
            target_channels=1,
            expected_hidden_size=self._get_expected_hidden_size(),
        )

    def get_default_tok_params(self):
        """Avoid duplicate BOS when a chat template renders it."""
        tokenizer = self.ctx.get_tokenizer()
        has_chat_template = getattr(tokenizer, "chat_template", None) is not None

        params = super().get_default_tok_params()
        if has_chat_template:
            # The template emits BOS itself; suppress tokenizer-added BOS to
            # avoid sending two BOS tokens to the model.
            params = params.with_kwargs(add_special_tokens=False)
        return params

    def get_supported_mm_limits(self) -> Mapping[str, int | None]:
        return {"image": None, "audio": None}

    def get_mm_max_tokens_per_item(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
    ) -> Mapping[str, int] | None:
        del mm_counts
        processor = self.get_hf_processor()
        image_processor = processor.image_processor
        feature_extractor = processor.feature_extractor
        return {
            # Maximum image codes plus room for the image layout's wrapper
            # and row-separator tokens.
            "image": min(
                (image_processor.max_pixels // (image_processor.spatial_factor**2))
                + _IMAGE_TOKEN_BUDGET_OVERHEAD,
                seq_len,
            ),
            # Maximum audio codes for the configured duration plus special tokens.
            "audio": min(
                (
                    feature_extractor.get_num_audio_codes(
                        feature_extractor.sampling_rate
                    )
                    * _MAX_AUDIO_SECONDS
                )
                + _AUDIO_TOKEN_BUDGET_OVERHEAD,
                seq_len,
            ),
        }


class Apertus1p5DummyInputsBuilder(BaseDummyInputsBuilder[Apertus1p5ProcessingInfo]):
    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        tokenizer = self.info.get_tokenizer()
        image_placeholder = getattr(
            tokenizer, "image_token", _DEFAULT_IMAGE_PLACEHOLDER
        )
        audio_placeholder = getattr(
            tokenizer, "audio_token", _DEFAULT_AUDIO_PLACEHOLDER
        )
        return image_placeholder * mm_counts.get(
            "image", 0
        ) + audio_placeholder * mm_counts.get("audio", 0)

    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        mm_options: Mapping[str, BaseDummyOptions],
    ) -> MultiModalDataDict:
        image_overrides = mm_options.get("image")
        audio_overrides = mm_options.get("audio")
        processor = self.info.get_hf_processor()
        max_image_side = isqrt(processor.image_processor.max_pixels)

        return {
            "image": self._get_dummy_images(
                width=max_image_side,
                height=max_image_side,
                num_images=mm_counts.get("image", 0),
                overrides=image_overrides,
            ),
            "audio": self._get_dummy_audios(
                length=processor.feature_extractor.sampling_rate * _MAX_AUDIO_SECONDS,
                num_audios=mm_counts.get("audio", 0),
                overrides=audio_overrides,
            ),
        }


class Apertus1p5MultiModalProcessor(BaseMultiModalProcessor[Apertus1p5ProcessingInfo]):
    """Process Apertus multimodal inputs on the CPU."""

    def __init__(
        self,
        info: Apertus1p5ProcessingInfo,
        dummy_inputs: BaseDummyInputsBuilder,
        *,
        cache: object | None = None,
    ) -> None:
        super().__init__(info, dummy_inputs, cache=cache)
        tokenizer = info.get_tokenizer()
        self.hf_processor = info.get_hf_processor()
        config = info.get_hf_config()
        self.image_token_id = getattr(config, "image_token_id", _DEFAULT_IMAGE_TOKEN_ID)
        self.audio_token_id = getattr(config, "audio_token_id", _DEFAULT_AUDIO_TOKEN_ID)
        self.image_start_token = getattr(tokenizer, "boi_token", _DEFAULT_BOI_TOKEN)
        self.image_end_token = getattr(tokenizer, "eoi_token", _DEFAULT_EOI_TOKEN)
        self.audio_start_token = getattr(
            tokenizer, "boa_token", _DEFAULT_AUDIO_START_TOKEN
        )
        self.audio_end_token = getattr(tokenizer, "eoa_token", _DEFAULT_AUDIO_END_TOKEN)
        self.image_start_token_id = getattr(
            tokenizer, "boi_token_id", _DEFAULT_IMAGE_START_TOKEN_ID
        )
        self.image_end_token_id = getattr(
            tokenizer, "eoi_token_id", _DEFAULT_IMAGE_END_TOKEN_ID
        )
        self.audio_start_token_id = getattr(
            tokenizer, "boa_token_id", _DEFAULT_AUDIO_START_TOKEN_ID
        )
        self.audio_end_token_id = getattr(
            tokenizer, "eoa_token_id", _DEFAULT_AUDIO_END_TOKEN_ID
        )

    def _get_mm_fields_config(
        self,
        hf_inputs: object,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        """Routes per-item tensors to the GPU Worker's embed_multimodal kwargs."""
        return {
            "pixel_values": MultiModalFieldConfig.batched("image"),
            "audio_values": MultiModalFieldConfig.batched("audio"),
        }

    def _get_prompt_updates(self, *args: Any, **kwargs: Any) -> Sequence[PromptUpdate]:
        return []

    def apply(
        self, inputs: ProcessorInputs, timing_ctx: TimingContext
    ) -> MultiModalInput:
        tokenizer = self.info.get_tokenizer()
        prompt_text = (
            inputs.prompt
            if isinstance(inputs.prompt, str)
            else tokenizer.decode(inputs.prompt)
        )

        tokenization_kwargs = dict(inputs.tokenization_kwargs)
        if not isinstance(inputs.prompt, str):
            # A template-rendered prompt already starts with BOS. Otherwise,
            # restore the BOS added before token IDs were decoded to text.
            tokenization_kwargs.setdefault(
                "add_special_tokens",
                not (
                    bool(inputs.prompt) and inputs.prompt[0] == tokenizer.bos_token_id
                ),
            )

        num_images = inputs.mm_data_items.get_count("image", strict=False)
        num_audios = inputs.mm_data_items.get_count("audio", strict=False)

        images = (
            inputs.mm_data_items.get_items("image", ImageProcessorItems).get_all()
            if num_images > 0
            else None
        )
        audios = (
            inputs.mm_data_items.get_items("audio", AudioProcessorItems).get_all()
            if num_audios > 0
            else None
        )

        mm_kwargs: dict[str, torch.Tensor | list[torch.Tensor]] = {}

        with timing_ctx.record("preprocess_apertus"):
            hf_outputs = self.hf_processor(
                text=prompt_text,
                images=images,
                audio=audios,
                padding=True,
                return_tensors="pt",
                **tokenization_kwargs,
            )
            prompt_token_ids = hf_outputs["input_ids"][0].tolist()

        if num_images > 0:
            # Hugging Face pads images in a multimodal batch to a common size.
            # Crop each item back to its reported dimensions before encoding it.
            pixel_values = [
                image[:, : int(height), : int(width)].contiguous()
                for image, (height, width) in zip(
                    hf_outputs["pixel_values"],
                    hf_outputs["image_sizes"],
                )
            ]
            mm_kwargs["pixel_values"] = pixel_values

        if num_audios > 0:
            # Apertus1p5 input-processor pads audio features in a multimodal
            # batch to a common length. Remove that padding with each item's
            # attention mask.
            audio_values = [
                audio[0, : int(mask.sum())].contiguous()
                for audio, mask in zip(
                    hf_outputs["input_features"],
                    hf_outputs["feature_attention_mask"],
                )
            ]
            mm_kwargs["audio_values"] = audio_values

        # Each placeholder range marks the token positions replaced by embeddings.
        with timing_ctx.record("get_mm_hashes"):
            mm_hashes = inputs.get_mm_hashes(self.info.model_id)

        def _span_ranges(
            start_token: str,
            start_id: int,
            end_token: str,
            end_id: int,
            embed_token_id: int,
            count: int,
        ) -> list[PlaceholderRange]:
            """Locate processor-created multimodal spans and embedding slots.

            The Apertus1p5 input-processor from transformers expands images into
            ``<|img_start|>H*W<|img_token_start|><|image|>...<|img_end|>``
            and audio into ``<|audio_start|><|audio|>...<|audio_end|>``.
            For each item, find the next complete range after the previous
            one and mark its ``<|image|>`` or ``<|audio|>`` positions for
            replacement with the corresponding encoded embeddings.
            """
            ranges: list[PlaceholderRange] = []
            pos = 0
            for _ in range(count):
                try:
                    s = prompt_token_ids.index(start_id, pos)
                    e = prompt_token_ids.index(end_id, s)
                except ValueError as exc:
                    raise ValueError(
                        f"Apertus MM: {start_token!r}/{end_token!r} pair not "
                        f"found in prompt (search from {pos})"
                    ) from exc
                span = prompt_token_ids[s : e + 1]
                is_embed = torch.tensor(
                    [tok == embed_token_id for tok in span], dtype=torch.bool
                )
                ranges.append(
                    PlaceholderRange(offset=s, length=e - s + 1, is_embed=is_embed)
                )
                pos = e + 1
            return ranges

        mm_placeholders: dict[str, list[PlaceholderRange]] = {}
        if num_images > 0:
            mm_placeholders["image"] = _span_ranges(
                self.image_start_token,
                self.image_start_token_id,
                self.image_end_token,
                self.image_end_token_id,
                self.image_token_id,
                num_images,
            )
        if num_audios > 0:
            mm_placeholders["audio"] = _span_ranges(
                self.audio_start_token,
                self.audio_start_token_id,
                self.audio_end_token,
                self.audio_end_token_id,
                self.audio_token_id,
                num_audios,
            )

        return mm_input(
            prompt_token_ids=prompt_token_ids,
            mm_kwargs=MultiModalKwargsItems.from_hf_inputs(
                mm_kwargs, self._get_mm_fields_config(mm_kwargs, {})
            ),
            mm_hashes=mm_hashes,
            mm_placeholders=mm_placeholders,
        )


@MULTIMODAL_REGISTRY.register_processor(
    Apertus1p5MultiModalProcessor,
    info=Apertus1p5ProcessingInfo,
    dummy_inputs=Apertus1p5DummyInputsBuilder,
)
class Apertus1p5ForConditionalGeneration(
    torch.nn.Module,
    SupportsMultiModal,
    SupportsLoRA,
    SupportsPP,
    SupportsEagle,
    SupportsEagle3,
):
    hf_to_vllm_mapper = ApertusForCausalLM.hf_to_vllm_mapper | WeightsMapper(
        orig_to_new_prefix={
            "model.language_model.": "language_model.",
            "model.vision_tokenizer.": "vision_tower.",
            "model.audio_tokenizer.": "audio_tower.",
        }
    )

    packed_modules_mapping = ApertusForCausalLM.packed_modules_mapping
    embedding_modules = ApertusForCausalLM.embedding_modules
    allow_patterns_overrides = None

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None:
        if modality.startswith("image"):
            return _DEFAULT_IMAGE_PLACEHOLDER
        if modality.startswith("audio"):
            return _DEFAULT_AUDIO_PLACEHOLDER
        raise ValueError(f"Unsupported modality: {modality}")

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        config = vllm_config.model_config.hf_config
        text_config = vllm_config.model_config.hf_text_config
        self.config = config

        with self._mark_language_model(vllm_config):
            self.language_model = ApertusModel(
                vllm_config=vllm_config.with_hf_config(text_config),
                prefix=maybe_prefix(prefix, "language_model"),
            )
        self.make_empty_intermediate_tensors = (
            self.language_model.make_empty_intermediate_tensors
        )

        # `output_vocab_size` is the number of rows in the output LM head. The
        # pruned head contains text tokens only; image, audio, and omni special
        # token IDs remain input-only embeddings and cannot be generated.
        output_vocab_size = (
            getattr(text_config, "output_vocab_size", None) or text_config.vocab_size
        )
        if output_vocab_size > text_config.vocab_size:
            raise ValueError("Output vocabulary cannot exceed input vocabulary.")
        self._input_vocab_size = text_config.vocab_size
        self._should_pad_logits_to_input_vocab = (
            output_vocab_size != text_config.vocab_size
        )
        if get_pp_group().is_last_rank:
            self.lm_head = ParallelLMHead(
                output_vocab_size,
                text_config.hidden_size,
                quant_config=vllm_config.quant_config,
                prefix=maybe_prefix(prefix, "lm_head"),
            )
            if config.tie_word_embeddings:
                raise ValueError(
                    "Apertus 1.5 does not support tied input and output embeddings."
                )
            logit_scale = getattr(config, "logit_scale", 1.0)
            self.logits_processor = LogitsProcessor(
                output_vocab_size, scale=logit_scale
            )
        else:
            self.lm_head = PPMissingLayer()

        # A single primary source now yields the vision/audio tensors too (routed by
        # hf_to_vllm_mapper)
        self.secondary_weights = []
        if get_pp_group().is_first_rank:
            with set_default_torch_dtype(torch.float32):
                with self._mark_tower_model(vllm_config, "image"):
                    self.vision_tower = _init_component_model(
                        config.vision_tokenizer_config,
                        model_cls=Apertus1p5VisionTokenizerModel,
                    )
                with self._mark_tower_model(vllm_config, "audio"):
                    self.audio_tower = _init_component_model(
                        config.audio_tokenizer_config,
                    )

        self.image_token_offset = getattr(
            config, "image_token_offset", _DEFAULT_IMAGE_TOKEN_OFFSET
        )
        self.audio_token_offset = getattr(
            config, "audio_token_offset", _DEFAULT_AUDIO_TOKEN_OFFSET
        )

    def get_language_model(self):
        return self.language_model

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        logits = self.logits_processor(self.lm_head, hidden_states)
        if logits is None or not self._should_pad_logits_to_input_vocab:
            return logits
        # The text-only LM head is narrower than the input vocabulary used by
        # repetition penalties and other sampler logic.
        return _pad_logits_to_input_vocab(logits, self._input_vocab_size)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: object,
    ) -> torch.Tensor | IntermediateTensors:
        return self.language_model(
            input_ids,
            positions,
            intermediate_tensors,
            inputs_embeds,
        )

    def load_weights(
        self,
        weights: Iterable[tuple[str, torch.Tensor]],
    ) -> set[str]:
        skip_prefixes = ["lm_head."] if self.config.tie_word_embeddings else []
        loader = AutoWeightsLoader(self, skip_prefixes=skip_prefixes)
        return loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)

    def _get_module_device_dtype(
        self,
        module: torch.nn.Module,
    ) -> tuple[torch.device, torch.dtype]:
        parameter = next(module.parameters())
        return parameter.device, parameter.dtype

    def _encode_image_to_llm(
        self,
        image_input: Apertus1p5ImageInputs,
    ) -> list[torch.Tensor]:
        vision_tower = self.vision_tower
        assert vision_tower is not None
        target_device, target_dtype = self._get_module_device_dtype(vision_tower)

        # `list()` expands tensors along dimension 0 while preserving list inputs.
        images = list(image_input["pixel_values"])
        if not images:
            return []

        # The vision tower expects a single image with a batch dimension;
        # per-item encoding avoids quality degradation from tokenizer batching.
        with torch.inference_mode():
            ids_per_image = []
            for image in images:
                image_codes = vision_tower.encode(
                    image.unsqueeze(0).to(device=target_device, dtype=target_dtype)
                )
                ids_per_image.append(image_codes.flatten())
        ids_per_image = [
            ids.to(torch.long) + self.image_token_offset for ids in ids_per_image
        ]
        lengths = [ids.shape[0] for ids in ids_per_image]
        all_embeds = self.language_model.embed_input_ids(torch.cat(ids_per_image))
        return list(all_embeds.split(lengths))

    def _encode_audio_to_llm(
        self,
        audio_input: Apertus1p5AudioInputs,
    ) -> list[torch.Tensor]:
        audio_tower = self.audio_tower
        assert audio_tower is not None
        target_device, target_dtype = self._get_module_device_dtype(audio_tower)

        # `list()` expands tensors along dimension 0 while preserving list inputs.
        audios = list(audio_input["audio_values"])
        if not audios:
            return []

        # The audio tower expects a single clip with batch/channel dimensions;
        # per-item encoding avoids quality degradation from tokenizer batching.
        with torch.inference_mode():
            ids_per_audio = []
            for audio in audios:
                audio_codes = audio_tower.encode(
                    audio.unsqueeze(0)
                    .unsqueeze(0)
                    .to(device=target_device, dtype=target_dtype)
                ).audio_codes
                ids_per_audio.append(audio_codes.squeeze(0).squeeze(0))
        ids_per_audio = [
            ids.to(torch.long) + self.audio_token_offset for ids in ids_per_audio
        ]
        lengths = [ids.shape[0] for ids in ids_per_audio]
        all_embeds = self.language_model.embed_input_ids(torch.cat(ids_per_audio))
        return list(all_embeds.split(lengths))

    def _parse_and_validate_image_input(
        self, **kwargs: object
    ) -> Apertus1p5ImageInputs | None:
        pixel_values = kwargs.pop("pixel_values", None)
        if pixel_values is None:
            return None

        return Apertus1p5ImageInputs(
            type="pixel_values",
            pixel_values=pixel_values,
        )

    def _parse_and_validate_audio_input(
        self, **kwargs: object
    ) -> Apertus1p5AudioInputs | None:
        audio_values = kwargs.pop("audio_values", None)
        if audio_values is None:
            return None

        return Apertus1p5AudioInputs(
            type="audio_values",
            audio_values=audio_values,
        )

    def _parse_and_validate_multimodal_inputs(self, **kwargs: object) -> dict:
        modalities: dict[str, Any] = {}

        # Preserve the order of modalities when images and audio are interleaved.
        for input_key in kwargs:
            if input_key == "pixel_values" and "images" not in modalities:
                image_input = self._parse_and_validate_image_input(**kwargs)
                assert image_input is not None
                modalities["images"] = image_input
            if input_key == "audio_values" and "audios" not in modalities:
                audio_input = self._parse_and_validate_audio_input(**kwargs)
                assert audio_input is not None
                modalities["audios"] = audio_input

        return modalities

    def embed_multimodal(
        self,
        **kwargs: object,
    ) -> MultiModalEmbeddings:
        """Encode all modality batches into language embeddings."""
        modalities = self._parse_and_validate_multimodal_inputs(**kwargs)
        multimodal_embeddings: list[torch.Tensor] = []

        for modality in modalities:
            if modality == "images" and self.vision_tower is not None:
                image_input = modalities["images"]
                image_embeds = self._encode_image_to_llm(image_input)
                multimodal_embeddings.extend(image_embeds)
            if modality == "audios" and self.audio_tower is not None:
                audio_input = modalities["audios"]
                audio_embeds = self._encode_audio_to_llm(audio_input)
                multimodal_embeddings.extend(audio_embeds)

        return multimodal_embeddings

    def embed_input_ids(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: MultiModalEmbeddings | None = None,
        *,
        is_multimodal: torch.Tensor | None = None,
        **kwargs: object,
    ) -> torch.Tensor:
        return SupportsMultiModal.embed_input_ids(
            self,
            input_ids,
            multimodal_embeddings=multimodal_embeddings,
            is_multimodal=is_multimodal,
        )
