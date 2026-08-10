# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Copyright 2024 The vLLM team.
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
"""Transformers modeling backend mixin for multi-modal models."""

from collections import defaultdict
from collections.abc import Mapping, Sequence
from contextlib import ExitStack, contextmanager
from typing import TYPE_CHECKING, Any

import torch

from vllm.compilation.decorators import should_torch_compile_mm_encoder
from vllm.config.utils import getattr_iter
from vllm.inputs import MultiModalDataDict, MultiModalInput, mm_input
from vllm.logger import init_logger
from vllm.model_executor.models.interfaces import (
    MultiModalEmbeddings,
    SupportsMRoPE,
    SupportsMultiModal,
)
from vllm.model_executor.models.module_mapping import MultiModelKeys
from vllm.multimodal import MULTIMODAL_REGISTRY, MultiModalKwargsItems
from vllm.multimodal.inputs import (
    MultiModalFeatureSpec,
    MultiModalFieldConfig,
    PlaceholderRange,
)
from vllm.multimodal.parse import (
    ImageProcessorItems,
    ImageSize,
    MultiModalDataItems,
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

if TYPE_CHECKING:
    from transformers import BatchFeature, PreTrainedModel

    from vllm.config import VllmConfig
    from vllm.config.multimodal import BaseDummyOptions

logger = init_logger(__name__)

_MODALITY_TO_TOKEN_TYPE_ID = {"image": 1, "video": 2, "audio": 3}


class MultiModalProcessingInfo(BaseProcessingInfo):
    def _get_audio_processor(self) -> Any:
        # TODO: drop feature_extractor branch once huggingface/transformers#44394 lands.
        return getattr_iter(
            self.get_hf_processor(), ("audio_processor", "feature_extractor")
        )

    def _is_audio_model(self) -> bool:
        return self._get_audio_processor() is not None

    def _is_image_model(self) -> bool:
        return hasattr(self.get_hf_processor(), "image_processor")

    def _get_supported_modalities(self) -> list[str]:
        modalities = []
        if self._is_audio_model():
            modalities.append("audio")
        if self._is_image_model():
            modalities.append("image")
        if not modalities:
            raise ValueError(
                f"{type(self.get_hf_processor()).__name__} exposes neither an image "
                "processor nor an audio processor, so the Transformers modeling "
                "backend cannot serve this model as multi-modal."
            )
        return modalities

    def _get_audio_token_id(self) -> int:
        processor = self.get_hf_processor()
        if hasattr(processor, "audio_token_id"):
            return processor.audio_token_id
        config = self.get_hf_config()
        val = getattr_iter(config, ("audio_token_id", "audio_token_index"))
        if val is not None:
            return val
        if hasattr(processor, "audio_token"):
            tokenizer = self.get_tokenizer()
            vocab = tokenizer.get_vocab()
            if processor.audio_token in vocab:
                return vocab[processor.audio_token]
        raise ValueError("Cannot find audio_token_id on processor or model config")

    def _get_audio_sampling_rate(self) -> float:
        sub = self._get_audio_processor()
        if sub is not None and hasattr(sub, "sampling_rate"):
            return sub.sampling_rate
        return 16000.0

    def get_data_parser(self) -> MultiModalDataParser:
        if self._is_audio_model():
            return MultiModalDataParser(
                target_sr=self._get_audio_sampling_rate(),
                expected_hidden_size=self._get_expected_hidden_size(),
            )
        return super().get_data_parser()

    def get_supported_mm_limits(self):
        return dict.fromkeys(self._get_supported_modalities())

    def get_mm_max_tokens_per_item(self, seq_len, mm_counts):
        modalities = self._get_supported_modalities()
        max_tokens = {}
        if "audio" in modalities:
            max_tokens["audio"] = self.get_max_audio_tokens()
        if "image" in modalities:
            max_tokens["image"] = self.get_max_image_tokens()
        return max_tokens

    def get_max_audio_tokens(self) -> int:
        config = self.get_hf_config()
        audio_config_names = ("audio_config", "encoder_config")
        names = ("max_source_positions", "max_position_embeddings", "max_pos_emb")
        audio_config = getattr_iter(config, audio_config_names, default=config)
        val = getattr_iter(audio_config, names)
        if val is not None:
            return int(val)
        raise ValueError(
            f"Unable to get max input length from {type(audio_config).__name__}. "
            f"The following attribute names were checked: {names}."
        )

    def get_max_image_tokens(self) -> int:
        width, height = self.get_image_size_with_most_features()
        processor = self.get_hf_processor()
        multimodal_config = self.ctx.model_config.multimodal_config
        mm_processor_kwargs = multimodal_config.mm_processor_kwargs or {}
        mm_tokens = processor._get_num_multimodal_tokens(
            image_sizes=([height, width],), **mm_processor_kwargs
        )
        image_tokens = mm_tokens["num_image_tokens"][0]
        return image_tokens

    def get_image_size_with_most_features(self) -> ImageSize:
        return ImageSize(width=10_000, height=10_000)  # arbitrary very large size


class MultiModalDummyInputsBuilder(BaseDummyInputsBuilder[MultiModalProcessingInfo]):
    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        text = ""
        if self.info._is_audio_model() and (num_audios := mm_counts.get("audio", 0)):
            processor = self.info.get_hf_processor()
            audio_token = getattr(processor, "audio_token", "")
            # Separated so that `_apply_audio` can tell the placeholders apart
            text += " ".join([audio_token] * num_audios)
        if self.info._is_image_model() and (num_images := mm_counts.get("image", 0)):
            processor = self.info.get_hf_processor()
            if "gemma3" in processor.__class__.__name__.lower():
                image_token = processor.boi_token
            else:
                image_token = getattr(processor, "image_token", "")
            text += image_token * num_images
        return text

    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        mm_options: Mapping[str, "BaseDummyOptions"],
    ) -> MultiModalDataDict:
        data: MultiModalDataDict = {}
        if self.info._is_audio_model() and (num_audios := mm_counts.get("audio", 0)):
            sampling_rate = self.info._get_audio_sampling_rate()
            sub = self.info._get_audio_processor()
            chunk_length = getattr(sub, "chunk_length", None) if sub else None
            if chunk_length is None:
                chunk_length = 30
            audio_len = int(chunk_length * sampling_rate)
            data["audio"] = self._get_dummy_audios(
                length=audio_len,
                num_audios=num_audios,
                overrides=mm_options.get("audio"),
            )
        if self.info._is_image_model() and (num_images := mm_counts.get("image", 0)):
            target_width, target_height = self.info.get_image_size_with_most_features()
            data["image"] = self._get_dummy_images(
                width=target_width,
                height=target_height,
                num_images=num_images,
                overrides=mm_options.get("image"),
            )
        return data


class MultiModalProcessor(BaseMultiModalProcessor[MultiModalProcessingInfo]):
    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        out_mm_kwargs: MultiModalKwargsItems,
    ) -> Sequence[PromptUpdate]:
        """No updates: `apply` locates placeholders via `mm_token_type_ids` instead.

        HF processors have no generic contract for the token sequence they insert,
        so it cannot be expressed as a `PromptUpdate`. Returning nothing is only
        safe because `apply` is overridden; the base class would reject it.
        """
        return []

    def _get_modality_field_names(self, modality: str) -> set[str]:
        """Field names the sub-processor for `modality` produces."""
        # TODO: use else branch only once huggingface/transformers#44394 lands.
        if modality == "audio":
            sub_processor = self.info._get_audio_processor()
        else:
            processor = self.info.get_hf_processor()
            sub_processor = getattr(processor, f"{modality}_processor", None)

        # Pre-computed embeddings bypass the sub-processor entirely
        names = {f"{modality}_embeds"}
        for name in getattr(sub_processor, "model_input_names", None) or ():
            # Companion masks are emitted but not always declared
            names.update((name, f"{name}_mask"))
        return names

    def _partition_keys_by_modality(
        self,
        keys: list[str],
        modalities: list[str],
    ) -> dict[str, list[str]]:
        """Attribute each HF processor output key to the modality that produced it."""
        if len(modalities) == 1:
            return {modalities[0]: keys}

        claimed = {m: self._get_modality_field_names(m) for m in modalities}

        owned: dict[str, list[str]] = {modality: [] for modality in modalities}
        unclaimed = []
        for key in keys:
            for modality in modalities:
                if key in claimed[modality]:
                    owned[modality].append(key)
                    break
            else:
                unclaimed.append(key)

        if unclaimed:
            logger.warning_once(
                "Unable to attribute %s to any of the modalities %s, so they "
                "will not be passed to the model. Add them to the relevant "
                "sub-processor's `model_input_names` to fix this.",
                tuple(unclaimed),
                tuple(modalities),
            )

        return owned

    def _get_mm_fields_config(
        self,
        hf_inputs: "BatchFeature",
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        # HF Processors always return a mask but vLLM doesn't need it
        hf_inputs.pop("attention_mask", None)

        # Written by `_apply_audio`/`_apply_vision`; absent if the modality had no items
        sizes = {
            "audio": hf_inputs.get("num_audio_tokens"),
            "image": hf_inputs.get("num_image_patches"),
        }
        modalities = [m for m, size in sizes.items() if size is not None]

        size_keys = {"num_audio_tokens", "num_image_patches"}
        keys = [key for key in hf_inputs if key not in size_keys]
        owned = self._partition_keys_by_modality(keys, modalities)

        mm_fields: dict[str, MultiModalFieldConfig] = {
            key: MultiModalFieldConfig.flat_from_sizes(modality, sizes[modality])
            for modality in modalities
            for key in owned[modality]
        }

        # Keep these as batched, as they always have batch size as first dim
        if "audio" in modalities:
            mm_fields["num_audio_tokens"] = MultiModalFieldConfig.batched("audio")
        if "image" in modalities:
            mm_fields["image_grid_thw"] = MultiModalFieldConfig.batched("image")
            # TODO: route to "video" once the video modality is supported
            mm_fields["video_grid_thw"] = MultiModalFieldConfig.batched("image")
            mm_fields["num_image_patches"] = MultiModalFieldConfig.batched(
                "image", keep_on_cpu=True
            )
        return mm_fields

    def _get_hf_mm_data(
        self,
        mm_items: MultiModalDataItems,
    ) -> tuple[Mapping[str, object], Mapping[str, object]]:
        """
        In contrast to the base class, this method requests
        `return_mm_token_type_ids` and remaps the `audios` key to `audio` for
        audio models.
        """
        processor_data, passthrough_data = super()._get_hf_mm_data(mm_items)
        if self.info._is_audio_model() and "audios" in processor_data:
            processor_data["audio"] = processor_data.pop("audios")
        processor_data["return_mm_token_type_ids"] = True
        return processor_data, passthrough_data

    def _apply_audio(
        self,
        prompt_ids: list[int],
        processed_data: "BatchFeature",
        num_audios: int,
    ) -> dict[str, list[PlaceholderRange]]:
        audio_token_id = self.info._get_audio_token_id()
        prompt_tensor = torch.tensor(prompt_ids)
        is_audio = prompt_tensor == audio_token_id

        if not is_audio.any():
            return {}

        padded = torch.cat([torch.tensor([False]), is_audio, torch.tensor([False])])
        transitions = padded.int().diff()
        offsets = torch.where(transitions == 1)[0]
        lengths = torch.where(transitions == -1)[0] - offsets

        if len(offsets) != num_audios:
            raise ValueError(
                f"Found {len(offsets)} run(s) of the audio token in the prompt but "
                f"{num_audios} audio item(s) were passed. The Transformers backend "
                "locates audio placeholders by finding contiguous runs of the audio "
                "token, so placeholders with no text between them cannot yet be told "
                "apart. Separate them in the prompt to work around this."
            )

        ranges = [
            PlaceholderRange(offset=offset.item(), length=length.item())
            for offset, length in zip(offsets, lengths)
        ]
        processed_data["num_audio_tokens"] = lengths
        return {"audio": ranges}

    def _apply_vision(
        self,
        prompt_ids: list[int],
        processed_data: "BatchFeature",
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        mm_token_type_ids: torch.Tensor | None,
    ) -> dict[str, list[PlaceholderRange]]:
        # Placeholders can't be located without them, so give up rather than guess
        if mm_token_type_ids is None:
            return {}

        hf_processor = self.info.get_hf_processor(**hf_processor_mm_kwargs)

        # We can infer vLLM style placeholder from token type ids, if we split
        # it for each input `mm_data`.
        mm_positions = torch.where(mm_token_type_ids == 1)[1]
        images = mm_items.get_items("image", ImageProcessorItems)
        image_sizes = []
        for item_idx in range(len(images)):
            image_size = images.get_image_size(item_idx)
            image_sizes.append((image_size.height, image_size.width))

        mm_tokens_per_modality = hf_processor._get_num_multimodal_tokens(
            image_sizes=image_sizes,
            **self.info.ctx.get_merged_mm_kwargs({}),
        )

        mm_placeholders: dict[str, list[PlaceholderRange]] = {}
        split_sizes = mm_tokens_per_modality["num_image_tokens"]
        if split_sizes:
            chunked_mm_positions = torch.split(mm_positions, split_sizes)
            mm_tokens = torch.tensor(prompt_ids)[mm_token_type_ids[0].bool()]
            chunked_mm_tokens = torch.split(mm_tokens, split_sizes)
            ranges = [
                PlaceholderRange(
                    offset=positions[0].item(),
                    length=positions.shape[0],
                    is_embed=(mm_tokens == hf_processor.image_token_id).bool(),
                )
                for positions, mm_tokens in zip(chunked_mm_positions, chunked_mm_tokens)
            ]
            mm_placeholders = {"image": ranges}

        processed_data["num_image_patches"] = torch.tensor(
            mm_tokens_per_modality["num_image_patches"]
        )
        return mm_placeholders

    def apply(
        self,
        inputs: ProcessorInputs,
        timing_ctx: TimingContext,
    ) -> MultiModalInput:
        """
        Process multi-modal inputs to be used in vLLM.

        Apply HF Processor on prompt text and multi-modal data together,
        outputting token IDs and processed tensors.
        """
        prompt = inputs.prompt
        mm_items = inputs.mm_data_items
        hf_processor_mm_kwargs = inputs.hf_processor_mm_kwargs
        tokenization_kwargs = inputs.tokenization_kwargs

        with timing_ctx.record("apply_hf_processor"):
            hf_processor = self.info.get_hf_processor(**hf_processor_mm_kwargs)
            if not isinstance(prompt, str):
                # HF processors only accept text, and the decoded string already
                # contains any special tokens, so don't let them be added again
                prompt = hf_processor.decode(prompt)
                tokenization_kwargs = {
                    **tokenization_kwargs,
                    "add_special_tokens": False,
                }

            # Bypass cached processor and always apply to the full set of mm inputs
            # NOTE: we can't just set caching=False because base class method
            # transforms outputs to `MultiModalKwargs` which is not going to
            # work for Transformers. The vision path has logic tied to
            # `mm_tokens_per_modality` in _apply_vision()
            prompt_ids, processed_data, _ = self._apply_hf_processor_text_mm(
                prompt_text=prompt,
                mm_items=mm_items,
                hf_processor_mm_kwargs=hf_processor_mm_kwargs,
                tokenization_kwargs=tokenization_kwargs,
            )

        # Use overrides if provided; fallback to data-dependent hashing.
        with timing_ctx.record("get_mm_hashes"):
            mm_hashes = inputs.get_mm_hashes(
                self.info.model_id,
                self.info.ctx.get_mm_config().mm_hasher_algorithm,
            )

        # For gemma3 we check `token_type_ids` as the key
        mm_token_type_ids = processed_data.pop("token_type_ids", None)
        mm_token_type_ids = processed_data.pop("mm_token_type_ids", mm_token_type_ids)

        mm_placeholders: dict[str, list[PlaceholderRange]] = {}
        if num_audios := mm_items.get_count("audio", strict=False):
            mm_placeholders.update(
                self._apply_audio(prompt_ids, processed_data, num_audios)
            )
        if mm_items.get_count("image", strict=False):
            mm_placeholders.update(
                self._apply_vision(
                    prompt_ids,
                    processed_data,
                    mm_items,
                    hf_processor_mm_kwargs,
                    mm_token_type_ids,
                )
            )

        mm_kwargs = MultiModalKwargsItems.from_hf_inputs(
            processed_data,
            self._get_mm_fields_config(processed_data, hf_processor_mm_kwargs),
        )

        # Bypassing `_maybe_apply_prompt_updates` also bypasses its validation.
        # `_validate_mm_placeholders` can't be reused because it is typed for the
        # `PlaceholderFeaturesInfo` the prompt update machinery produces.
        mm_item_counts = mm_items.get_all_counts()
        self._validate_mm_kwargs(mm_kwargs, mm_item_counts)
        for modality, item_count in mm_item_counts.items():
            num_placeholders = len(mm_placeholders.get(modality, []))
            if num_placeholders != item_count:
                raise RuntimeError(
                    f"Expected there to be {item_count} prompt placeholders "
                    f"corresponding to {item_count} {modality} items, but instead "
                    f"found {num_placeholders} prompt placeholders! Make sure the "
                    "prompt contains a placeholder token for each item."
                )

        return mm_input(
            prompt_token_ids=prompt_ids,
            mm_kwargs=mm_kwargs,
            mm_hashes=mm_hashes,
            mm_placeholders=mm_placeholders,
        )


class MultiModalMixin(SupportsMultiModal, SupportsMRoPE):
    def __init__(self, *, vllm_config: "VllmConfig", prefix: str = ""):
        # Skip SupportsMRoPE.__init__ and call the next class in MRO
        super(SupportsMRoPE, self).__init__(vllm_config=vllm_config, prefix=prefix)

    def _find_encoder_classes(
        self, model: "PreTrainedModel"
    ) -> dict[str, type["PreTrainedModel"]]:
        """Modalities whose encoder cannot be told apart from the model itself are
        omitted, as are those `get_encoder` rejects."""
        encoder_classes: dict[str, type[PreTrainedModel]] = {}
        for modality in _MODALITY_TO_TOKEN_TYPE_ID:
            try:
                encoder_cls = type(model.get_encoder(modality=modality))
            except (TypeError, ValueError):
                continue
            if encoder_cls is not type(model):
                encoder_classes[modality] = encoder_cls
        return encoder_classes

    @contextmanager
    def _mark_model_components(self, vllm_config: "VllmConfig"):
        model_config = vllm_config.model_config
        encoder_classes = self._pre_trained_model_classes.encoders
        if not encoder_classes:
            logger.debug("No encoders identified, so no components will be marked")
            yield
            return

        if model_config.skip_tokenizer_init:
            # Determining the supported modalities needs the HF processor, which in
            # turn needs a tokenizer
            mm_config = model_config.multimodal_config
            if mm_config.mm_encoder_only or any(
                mm_config.get_limit_per_prompt(modality) == 0
                for modality in encoder_classes
            ):
                logger.warning_once(
                    "Unable to determine the supported modalities without a "
                    "tokenizer, so no model components will be skipped."
                )
            yield
            return

        # Modalities we don't serve report a limit of 999, which would stop their
        # encoder ever being skipped
        supported_modalities = MULTIMODAL_REGISTRY.get_processing_info(
            model_config
        ).supported_mm_limits

        # One encoder often serves several modalities, and may only be skipped when
        # all of them are disabled, so mark it once for the whole set
        modalities_by_encoder = defaultdict(set)
        for modality, encoder_cls in encoder_classes.items():
            if modality in supported_modalities:
                modalities_by_encoder[encoder_cls].add(modality)

        with ExitStack() as stack:
            stack.enter_context(
                self._mark_language_model(
                    vllm_config, targets=self._pre_trained_model_classes.decoder
                )
            )
            for encoder_cls, modalities in modalities_by_encoder.items():
                stack.enter_context(
                    self._mark_tower_model(vllm_config, modalities, targets=encoder_cls)
                )
            yield

    def _decorate_for_torch_compile(self):
        """
        Decorate the model's decoder and encoder classes to indicate to vLLM that they
        support torch compile if `can_enable_torch_compile` and
        `should_torch_compile_mm_encoder` are True respectively.
        """
        super()._decorate_for_torch_compile()
        # Decorate the encoder model classes to support torch compile if needed
        if self.compilation_config.compile_mm_encoder:
            self.check_version("5.0.0", "multimodal encoder compilation support")
            encoder_classes = self._pre_trained_model_classes.encoders
            if not encoder_classes:
                raise ValueError(
                    "Unable to infer any encoder classes from the model. "
                    "You must either: update the model so that "
                    "https://huggingface.co/docs/transformers/en/main_classes/model#transformers.PreTrainedModel.get_encoder"
                    " can detect the encoders correctly, or remove "
                    "'compile_mm_encoder'."
                )
            logger.warning_once(
                "Multimodal encoder compilation with the Transformers modeling backend "
                "is an experimental feature. It relies on:\n"
                "- The encoder being torch compilable.\n"
                "- All encoder tensor inputs must be type hinted as either "
                "`torch.Tensor` or `torch.FloatTensor`.\n"
                "- The 0-th dimension of all tensor inputs to the encoder being the "
                "dynamic dimension (e.g. sequence length, number of patches).\n"
                "Please report any issues you encounter to help us improve it."
            )
            # One encoder can serve several modalities, and must only be decorated once
            for encoder_cls in dict.fromkeys(encoder_classes.values()):
                self._decorate_cls_for_torch_compile(
                    cls=encoder_cls,
                    # TODO: properly infer dynamic_arg_dims based on the encoder's
                    # forward method signature. We assume dim 0 for all tensor inputs.
                    dynamic_arg_dims=None,
                    enable_if=should_torch_compile_mm_encoder,
                    is_encoder=True,
                )

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: object,
    ) -> torch.Tensor | IntermediateTensors:
        # Positions shape handling for MRoPE models
        if self.model_config.uses_mrope:
            # [3, seq_len] -> [3, 1, seq_len]
            positions = positions[:, None].contiguous()
        model_output = super().forward(
            input_ids, positions, intermediate_tensors, inputs_embeds
        )
        return model_output

    def get_language_model(self) -> torch.nn.Module:
        """Transformers modeling backend multimodal classes do not contain a separate
        vLLM language model class. Therefore, in order to return a language model vLLM
        class, we use a wrapper to give `self` the same interface as a text model."""

        # Exclude self and object
        bases = self.__class__.mro()[1:-1]
        # Keep only classes defined in `vllm.model_executor.models.transformers`
        bases = [b for b in bases if ".transformers." in b.__module__]
        # Exclude MultiModalMixin itself
        bases = [b for b in bases if b is not MultiModalMixin]

        class LanguageModel(*bases):
            def __init__(self, multimodal_model):
                # Don't call super().__init__() to avoid re-initialization
                self.__dict__.update(multimodal_model.__dict__)

            model = getattr_iter(self.model, ("language_model", "text_model"), None)

        return LanguageModel(self)

    def get_mm_mapping(self) -> MultiModelKeys:
        """
        Get the module prefix in multimodal models
        """
        for name in ("language_model", "text_model"):
            if getattr(self.model, name, None) is not None:
                return MultiModelKeys.from_string_field(language_model=f"model.{name}")
        raise ValueError(
            "Could not locate the language model submodule for LoRA support"
        )

    def _split_embeddings(
        self, embeddings: torch.Tensor, split_sizes: list[int]
    ) -> list[torch.Tensor]:
        total_expected = sum(split_sizes)

        # Flatten to 2D: [total_tokens, hidden_dim]
        if embeddings.ndim > 2:
            embeddings = embeddings.reshape(-1, embeddings.shape[-1])

        total_tokens = embeddings.shape[0]
        if total_tokens == total_expected:
            # Direct match: split_sizes are actual token counts
            token_split_sizes = split_sizes
        elif total_expected > 0 and total_tokens % total_expected == 0:
            # Uniform expansion: each item expands to N tokens
            tokens_per_item = total_tokens // total_expected
            token_split_sizes = [s * tokens_per_item for s in split_sizes]
        elif total_expected > 0:
            # TODO: make this an error once we know profiling never relies on it
            if total_tokens == 0:
                raise ValueError(
                    "Encoder returned empty embeddings. "
                    f"Expected {total_expected} tokens from "
                    f"split_sizes={split_sizes}"
                )
            # Keep the counts out of the message: `warning_once` keys its cache on
            # the args, so varying them would log on every new pair
            logger.warning_once(
                "Encoder returned a different number of tokens than expected; "
                "padding or truncating to fit. The embeddings are not trustworthy "
                "outside of memory profiling."
            )
            logger.debug(
                "Encoder returned %s tokens but %s were expected",
                total_tokens,
                total_expected,
            )
            if total_tokens < total_expected:
                repeat_factor = (total_expected + total_tokens - 1) // total_tokens
                embeddings = embeddings.repeat(repeat_factor, 1)
            embeddings = embeddings[:total_expected]
            token_split_sizes = split_sizes
        else:
            return []

        return list(torch.split(embeddings, token_split_sizes, dim=0))

    def _process_audio_input(self, **kwargs) -> list[torch.Tensor] | None:
        input_features: torch.Tensor | None = kwargs.pop("input_features", None)
        if input_features is None:
            input_features = kwargs.pop("input_values", None)
        if input_features is None:
            return None

        self.check_version("5.13.0", "audio models support")
        num_audio_tokens = kwargs.pop("num_audio_tokens")
        kwargs.pop("token_type_ids", None)
        kwargs.pop("mm_token_type_ids", None)

        audio_output = self.model.get_audio_features(
            input_features, return_dict=True, **kwargs
        )
        audio_embeddings = audio_output.pooler_output

        split_sizes = num_audio_tokens.flatten().tolist()
        return self._split_embeddings(audio_embeddings, split_sizes)

    def _process_image_input(self, **kwargs) -> list[torch.Tensor] | None:
        pixel_values: torch.Tensor | None = kwargs.pop("pixel_values", None)
        image_embeds: torch.Tensor | None = kwargs.pop("image_embeds", None)
        # Model might use `image_patches` instead of `pixel_values`
        if pixel_values is None:
            pixel_values = kwargs.pop("image_patches", None)

        if image_embeds is not None:
            return [image_embeds]

        if pixel_values is None:
            return None

        num_image_patches = kwargs.pop("num_image_patches")

        vision_embeddings = self.model.get_image_features(pixel_values, **kwargs)

        # Transformers `v5`, `self.get_image_features` returns a tuple
        # containing the features and optionally attentions/hidden_states
        # After v5 is settled, we can enable qwen3-vl with several outputs
        # from `self.get_image_features`
        if isinstance(vision_embeddings, tuple):
            vision_embeddings = vision_embeddings[0]
        elif isinstance(vision_embeddings, dict):
            vision_embeddings = vision_embeddings.pooler_output

        if isinstance(vision_embeddings, torch.Tensor):
            split_sizes = num_image_patches.flatten().tolist()
            return self._split_embeddings(vision_embeddings, split_sizes)

        return list(vision_embeddings)

    def embed_multimodal(self, **kwargs) -> MultiModalEmbeddings:
        # Each helper detects its own inputs. We are called once per modality, so the
        # leftovers a helper forwards to the HF model can't belong to the other one.
        embeddings: list[torch.Tensor] = []
        for process_input in (self._process_audio_input, self._process_image_input):
            embeddings.extend(process_input(**kwargs) or [])
        return embeddings

    def get_mrope_input_positions(
        self,
        input_tokens: list[int],
        mm_features: list[MultiModalFeatureSpec],
    ) -> tuple[torch.Tensor, int]:
        kwargs = MultiModalFeatureSpec.gather_kwargs(
            mm_features,
            {
                "image_grid_thw",
                "video_grid_thw",
                "second_per_grid_ts",
                "audio_feature_lengths",
                "use_audio_in_video",
            },
        )
        if any(v for k, v in kwargs.items() if k not in {"image_grid_thw"}):
            raise NotImplementedError(
                "Transformers modeling backend only supports images."
            )

        image_grid_thw = kwargs.get("image_grid_thw", [])
        video_grid_thw = kwargs.get("video_grid_thw", [])

        image_grid_thw = torch.stack(image_grid_thw) if image_grid_thw else None
        video_grid_thw = torch.stack(video_grid_thw) if video_grid_thw else None

        # `get_rope_index` doesn't always accept arbitrary `kwargs`
        kwargs = {}
        if not hasattr(self, "_get_rope_index_accepts_mm_token_type_ids"):
            import inspect

            sig = inspect.signature(self.model.get_rope_index)
            params = sig.parameters
            self._get_rope_index_accepts_mm_token_type_ids = (
                "mm_token_type_ids" in params
                or any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())
            )
        if self._get_rope_index_accepts_mm_token_type_ids:
            mm_token_type_ids = torch.zeros(len(input_tokens), dtype=torch.int)
            for feature in mm_features:
                position = feature.mm_position
                offset, length = position.offset, position.length
                mm_token_type_id = _MODALITY_TO_TOKEN_TYPE_ID[feature.modality]
                mm_token_type_ids[offset : offset + length] = mm_token_type_id
            kwargs["mm_token_type_ids"] = mm_token_type_ids.unsqueeze(0)

        mrope_positions, mrope_position_delta = self.model.get_rope_index(
            input_ids=torch.tensor(input_tokens).unsqueeze(0),
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            **kwargs,
        )

        mrope_positions = mrope_positions[:, 0]
        mrope_position_delta = mrope_position_delta[0].item()

        return mrope_positions, mrope_position_delta
