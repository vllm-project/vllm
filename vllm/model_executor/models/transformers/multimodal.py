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

from collections import Counter
from collections.abc import Mapping, Sequence
from contextlib import nullcontext
from functools import partial
from typing import TYPE_CHECKING, Any

import torch
from transformers import AutoModel, BatchFeature

from vllm.compilation.decorators import should_torch_compile_mm_encoder
from vllm.config.utils import getattr_iter
from vllm.inputs import MultiModalDataDict
from vllm.logger import init_logger
from vllm.model_executor.models.interfaces import SupportsMRoPE, SupportsMultiModal
from vllm.model_executor.models.module_mapping import MultiModelKeys
from vllm.multimodal import MultiModalKwargsItems
from vllm.multimodal.inputs import (
    MultiModalFeatureSpec,
    MultiModalFieldConfig,
)
from vllm.multimodal.parse import (
    ImageProcessorItems,
    MultiModalDataItems,
    MultiModalDataParser,
)
from vllm.multimodal.processing import (
    BaseDummyInputsBuilder,
    BaseMultiModalProcessor,
    BaseProcessingInfo,
    PromptReplacement,
    PromptUpdate,
    PromptUpdateDetails,
)
from vllm.platforms import current_platform
from vllm.sequence import IntermediateTensors

if TYPE_CHECKING:
    from transformers import PreTrainedModel

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

    def _is_video_model(self) -> bool:
        return hasattr(self.get_hf_processor(), "video_processor")

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
        limits = {}
        if self._is_audio_model():
            limits["audio"] = None
        if self._is_image_model():
            limits["image"] = None
        if not limits:
            raise ValueError(
                f"Unable to detect a supported modality on "
                f"{type(self.get_hf_processor()).__name__}. "
                "Checked `_is_audio_model` and `_is_image_model`."
            )
        return limits

    def get_mm_max_tokens_per_item(self, seq_len, mm_counts):
        result = {}
        if self._is_audio_model():
            result["audio"] = self.get_max_audio_tokens()
        if self._is_image_model():
            result["image"] = self.get_max_image_tokens()
        if not result:
            raise ValueError(
                f"Unable to detect a supported modality on "
                f"{type(self.get_hf_processor()).__name__}. "
                "Checked `_is_audio_model` and `_is_image_model`."
            )
        return result

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
        width, height = self.get_max_image_size()
        processor = self.get_hf_processor()
        multimodal_config = self.ctx.model_config.multimodal_config
        mm_processor_kwargs = multimodal_config.mm_processor_kwargs or {}
        mm_tokens = processor._get_num_multimodal_tokens(
            image_sizes=([height, width],), **mm_processor_kwargs
        )
        image_tokens = mm_tokens["num_image_tokens"][0]
        return image_tokens

    def get_max_image_size(self):
        return 10_000, 10_000  # hardcode for arbitrary very large size


class MultiModalDummyInputsBuilder(BaseDummyInputsBuilder[MultiModalProcessingInfo]):
    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        text = ""
        if self.info._is_audio_model() and (num_audios := mm_counts.get("audio", 0)):
            processor = self.info.get_hf_processor()
            audio_token = getattr(processor, "audio_token", "")
            text += audio_token * num_audios
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
            target_width, target_height = self.info.get_max_image_size()
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
        """
        Given the original multi-modal items for this modality
        and HF-processed data, output the updates to perform.

        The information returned by this method is used to update token inputs
        which bypass the HF processor. It is also used to update the output of
        HF processor if the HF process does not apply prompt updates to text
        inputs.

        Moreover, this information is critical to determine the token positions
        in order to construct  :class:`~vllm-multimodal.input.PlaceholderRange`
        for each multi-modal item.
        """

        def get_target(item_idx: int, modality: str):
            out_item = out_mm_kwargs[modality][item_idx]
            return out_item[f"{modality}_target_ids"].data.tolist()

        def get_replacement(item_idx: int, modality: str):
            out_item = out_mm_kwargs[modality][item_idx]
            repl_ids = out_item[f"{modality}_placeholder_ids"].data.tolist()
            embed_id = Counter(repl_ids).most_common(1)[0][0]
            return PromptUpdateDetails.select_token_id(repl_ids, embed_id)

        modalities: list[str] = []
        if self.info._is_audio_model():
            modalities.append("audio")
        if self.info._is_image_model():
            modalities.append("image")
        return [
            PromptReplacement(
                modality=modality,
                target=partial(get_target, modality=modality),
                replacement=partial(get_replacement, modality=modality),
            )
            for modality in modalities
        ]

    def _get_mm_fields_config(
        self,
        hf_inputs: "BatchFeature",
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        # HF Processors always return a mask but vLLM doesn't need it
        hf_inputs.pop("attention_mask", None)
        # Drop token_type_ids (gemma3) so it isn't an mm field
        hf_inputs.pop("token_type_ids", None)

        mm_fields: dict[str, MultiModalFieldConfig] = {}
        if self.info._is_audio_model():
            num_audio_tokens = hf_inputs.get("num_audio_tokens")
            if num_audio_tokens is not None:
                mm_fields.update(
                    {
                        key: MultiModalFieldConfig.flat_from_sizes(
                            "audio", num_audio_tokens
                        )
                        for key in hf_inputs
                    }
                )
                mm_fields["num_audio_tokens"] = MultiModalFieldConfig.batched("audio")
            else:
                mm_fields["audio_embeds"] = MultiModalFieldConfig.batched("audio")
            self._add_placeholder_fields("audio", hf_inputs, mm_fields)
        if self.info._is_image_model():
            num_image_patches = hf_inputs.get("num_image_patches")
            if num_image_patches is not None:
                mm_fields.update(
                    {
                        key: MultiModalFieldConfig.flat_from_sizes(
                            "image", num_image_patches
                        )
                        for key in hf_inputs
                    }
                )
                mm_fields["image_embeds"] = MultiModalFieldConfig.flat_from_sizes(
                    "image", num_image_patches
                )
                mm_fields["num_image_patches"] = MultiModalFieldConfig.batched(
                    "image", keep_on_cpu=True
                )
            else:
                mm_fields["image_embeds"] = MultiModalFieldConfig.batched("image")

            # Keep these as batched, as they always have batch size as first dim
            mm_fields["image_grid_thw"] = MultiModalFieldConfig.batched("image")
            mm_fields["video_grid_thw"] = MultiModalFieldConfig.batched("image")
            self._add_placeholder_fields("image", hf_inputs, mm_fields)
        return mm_fields

    def _add_placeholder_fields(
        self,
        modality: str,
        hf_inputs: "BatchFeature",
        mm_fields: dict[str, MultiModalFieldConfig],
    ) -> None:
        sizes = hf_inputs.get(f"num_{modality}_placeholders")
        if sizes is None:
            return
        mm_fields[f"num_{modality}_placeholders"] = MultiModalFieldConfig.batched(
            modality, keep_on_cpu=True
        )
        mm_fields[f"{modality}_placeholder_ids"] = (
            MultiModalFieldConfig.flat_from_sizes(modality, sizes, keep_on_cpu=True)
        )
        mm_fields[f"{modality}_target_ids"] = MultiModalFieldConfig.batched(
            modality, keep_on_cpu=True
        )

    def _get_hf_mm_data(
        self,
        mm_items: MultiModalDataItems,
    ) -> tuple[Mapping[str, object], Mapping[str, object]]:
        """
        In contrast to the base class, this method requests
        `return_text_replacement_offsets` and remaps the `audios` key to `audio`
        for audio models.
        """
        processor_data, passthrough_data = super()._get_hf_mm_data(mm_items)
        if self.info._is_audio_model() and "audios" in processor_data:
            processor_data["audio"] = processor_data.pop("audios")
        processor_data["return_text_replacement_offsets"] = True
        return processor_data, passthrough_data

    def _call_hf_processor(
        self,
        prompt: str,
        mm_data: Mapping[str, object],
        mm_kwargs: Mapping[str, object],
        tok_kwargs: Mapping[str, object],
    ) -> "BatchFeature":
        # Text-only input not supported in composite processor
        if not any(mm_data.get(key) for key in ("images", "audio")):
            prompt_ids = self.info.get_tokenizer().encode(prompt)
            prompt_ids = self._apply_hf_processor_tokens_only(prompt_ids)
            return BatchFeature(dict(input_ids=[prompt_ids]), tensor_type="pt")

        processed_data = super()._call_hf_processor(
            prompt=prompt,
            mm_data=mm_data,
            mm_kwargs=mm_kwargs,
            tok_kwargs=tok_kwargs,
        )
        self._add_placeholder_ids(processed_data)
        images = mm_data.get("images")
        if images is not None and self.info._is_image_model():
            hf_processor = self.info.get_hf_processor(**mm_kwargs)
            image_items = ImageProcessorItems(images)
            image_sizes = []
            for item_idx in range(len(image_items)):
                image_size = image_items.get_image_size(item_idx)
                image_sizes.append((image_size.height, image_size.width))
            mm_tokens = hf_processor._get_num_multimodal_tokens(
                image_sizes=image_sizes,
                **self.info.ctx.get_merged_mm_kwargs({}),
            )
            processed_data["num_image_patches"] = torch.tensor(
                mm_tokens["num_image_patches"]
            )
        return processed_data

    def _add_placeholder_ids(self, processed_data: "BatchFeature") -> None:
        offsets = processed_data.pop("text_replacement_offsets", None)
        if not offsets:
            raise RuntimeError(
                "Expected the processor to return prompt replacement offsets, "
                "but instead found none! Make sure the implementation of "
                f"{type(self.info.get_hf_processor()).__name__} supports "
                "`return_text_replacement_offsets`."
            )
        tokenizer = self.info.get_tokenizer()
        repl_ids: dict[str, list[list[int]]] = {}
        target_ids: dict[str, list[list[int]]] = {}
        for off in offsets[0]:
            ids = tokenizer.encode(off["replacement"], add_special_tokens=False)
            repl_ids.setdefault(off["type"], []).append(ids)
            target = tokenizer.encode(off["text"], add_special_tokens=False)
            target_ids.setdefault(off["type"], []).append(target)
        for modality, per_item in repl_ids.items():
            processed_data[f"{modality}_placeholder_ids"] = torch.tensor(
                [i for ids in per_item for i in ids]
            )
            processed_data[f"num_{modality}_placeholders"] = torch.tensor(
                [len(ids) for ids in per_item]
            )
            processed_data[f"{modality}_target_ids"] = torch.tensor(
                target_ids[modality]
            )
        if "audio" in repl_ids:
            processed_data["num_audio_tokens"] = torch.tensor(
                [Counter(ids).most_common(1)[0][1] for ids in repl_ids["audio"]]
            )


class MultiModalMixin(SupportsMultiModal, SupportsMRoPE):
    def __init__(self, *, vllm_config: "VllmConfig", prefix: str = ""):
        # Skip SupportsMRoPE.__init__ and call the next class in MRO
        super(SupportsMRoPE, self).__init__(vllm_config=vllm_config, prefix=prefix)

    def _get_encoder_cls(
        self, modality: str = "image", **kwargs: dict
    ) -> type["PreTrainedModel"]:
        """
        Get the encoder class from the model.

        Args:
            kwargs: The kwargs to create the model.

        Returns:
            The encoder class.
        """
        with torch.device("meta"):
            model: PreTrainedModel = AutoModel.from_config(**kwargs)
        encoder_cls = type(model.get_encoder(modality=modality))
        logger.debug("Identified encoder class as: %s", encoder_cls)
        if type(model) is encoder_cls:
            raise ValueError(
                "Unable to infer vision encoder class from the model. "
                "You must either: update the model so that "
                "https://huggingface.co/docs/transformers/en/main_classes/model#transformers.PreTrainedModel.get_encoder"
                " can detect the vision encoder correctly, or remove "
                "'compile_mm_encoder'."
            )
        del model
        return encoder_cls

    def _decorate_for_torch_compile(self, **kwargs: dict):
        """
        Decorate the model's decoder and encoder classes to indicate to vLLM that they
        support torch compile if `can_enable_torch_compile` and
        `should_torch_compile_mm_encoder` are True respectively.

        Args:
            kwargs: The kwargs to create the model, which are needed to get the decoder
                and encoder classes.
        """
        super()._decorate_for_torch_compile(**kwargs)
        # Decorate the vision encoder model class to support torch compile if needed
        if self.compilation_config.compile_mm_encoder:
            self.check_version("5.0.0", "multimodal encoder compilation support")
            logger.warning_once(
                "Multimodal encoder compilation with the Transformers modeling backend "
                "is an experimental feature. It relies on:\n"
                "- The vision encoder being torch compilable.\n"
                "- All vision encoder tensor inputs must be type hinted as either "
                "`torch.Tensor` or `torch.FloatTensor`.\n"
                "- The 0-th dimension of all tensor inputs to the vision encoder being "
                "the dynamic dimension (i.e., sequence length or number of patches).\n"
                "Please report any issues you encounter to help us improve it."
            )
            self._decorate_cls_for_torch_compile(
                cls=self._get_encoder_cls(**kwargs),
                # TODO: properly infer dynamic_arg_dims based on the encoder's forward
                # method signature. Currently we assume dim 0 for all tensor inputs.
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
        if embeddings.ndim == 3:
            embeddings = embeddings.view(-1, embeddings.shape[-1])

        total_tokens = embeddings.shape[0]
        if total_tokens == total_expected:
            # Direct match: split_sizes are actual token counts
            token_split_sizes = split_sizes
        elif total_expected > 0 and total_tokens % total_expected == 0:
            # Uniform expansion: each item expands to N tokens
            tokens_per_item = total_tokens // total_expected
            token_split_sizes = [s * tokens_per_item for s in split_sizes]
        elif total_expected > 0:
            # Mismatch (profiling with dummy data) - pad/truncate
            if total_tokens == 0:
                raise ValueError(
                    "Encoder returned empty embeddings. "
                    f"Expected {total_expected} tokens from "
                    f"split_sizes={split_sizes}"
                )
            if total_tokens < total_expected:
                repeat_factor = (total_expected + total_tokens - 1) // total_tokens
                embeddings = embeddings.repeat(repeat_factor, 1)
            embeddings = embeddings[:total_expected]
            token_split_sizes = split_sizes
        else:
            return []

        return list(torch.split(embeddings, token_split_sizes, dim=0))

    def _embed_audio(self, **kwargs) -> list[torch.Tensor] | None:
        self.check_version("5.13.0", "audio models support")
        input_features: torch.Tensor | None = kwargs.pop("input_features", None)
        if input_features is None:
            input_features = kwargs.pop("input_values", None)
        if input_features is None:
            return None

        num_audio_tokens = kwargs.pop("num_audio_tokens")
        kwargs.pop("token_type_ids", None)
        kwargs.pop("mm_token_type_ids", None)

        context = nullcontext()
        if current_platform.is_rocm():
            context = torch.nn.attention.sdpa_kernel(
                backends=[torch.nn.attention.SDPBackend.MATH]
            )
        with context:
            audio_output = self.model.get_audio_features(
                input_features, return_dict=True, **kwargs
            )
        audio_embeddings = audio_output.pooler_output

        split_sizes = num_audio_tokens.flatten().tolist()
        return self._split_embeddings(audio_embeddings, split_sizes)

    def _embed_vision(self, **kwargs) -> list[torch.Tensor] | torch.Tensor | None:
        pixel_values: torch.Tensor | None = kwargs.pop("pixel_values", None)
        image_embeds: torch.Tensor | None = kwargs.pop("image_embeds", None)
        # Model might use `image_patches` instead of `pixel_values`
        if pixel_values is None:
            pixel_values = kwargs.pop("image_patches", None)

        if image_embeds is not None:
            return image_embeds

        if pixel_values is None:
            return None

        num_image_patches = kwargs.pop("num_image_patches")

        context = nullcontext()
        if current_platform.is_rocm():
            # ROCm: Force math SDP backend for vision encoder to avoid accuracy issues
            # with flash_sdp and mem_efficient_sdp
            # TODO: [ROCm] Fix accuracy issues with flash backend
            logger.debug(
                "ROCm platform detected. Forcing math SDP backend "
                "for vision encoder. Currently ROCm platform has "
                "accuracy issues with `flash_sdp` and"
                "`mem_efficient_sdp` backends. See issue: "
                "https://github.com/vllm-project/vllm/issues/30167"
            )
            context = torch.nn.attention.sdpa_kernel(
                backends=[torch.nn.attention.SDPBackend.MATH]
            )
        with context:
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

        return vision_embeddings

    def embed_multimodal(self, **kwargs):
        for modality in ("image", "audio"):
            kwargs.pop(f"{modality}_placeholder_ids", None)
            kwargs.pop(f"num_{modality}_placeholders", None)
            kwargs.pop(f"{modality}_target_ids", None)

        embeddings: tuple[torch.Tensor, ...] = ()
        if "input_features" in kwargs or "input_values" in kwargs:
            audio_embeddings = self._embed_audio(**kwargs)
            if audio_embeddings is not None:
                embeddings += tuple(audio_embeddings)
        if (
            "pixel_values" in kwargs
            or "image_embeds" in kwargs
            or "image_patches" in kwargs
        ):
            vision_embeddings = self._embed_vision(**kwargs)
            if vision_embeddings is not None:
                if isinstance(vision_embeddings, torch.Tensor):
                    embeddings += (vision_embeddings,)
                else:
                    embeddings += tuple(vision_embeddings)
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
