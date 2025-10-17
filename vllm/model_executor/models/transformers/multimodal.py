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
"""Transformers backend mixin for multi-modal models."""

from collections.abc import Mapping
from typing import TYPE_CHECKING

import torch

from vllm.config.utils import getattr_iter
from vllm.model_executor.models.interfaces import SupportsMRoPE, SupportsMultiModal
from vllm.model_executor.models.utils import WeightsMapper
from vllm.multimodal import MultiModalKwargsItems
from vllm.multimodal.inputs import (
    MultiModalDataDict,
    MultiModalFieldConfig,
    MultiModalInputs,
    MultiModalUUIDDict,
    PlaceholderRange,
)
from vllm.multimodal.parse import ImageProcessorItems, MultiModalDataItems
from vllm.multimodal.processing import BaseMultiModalProcessor, BaseProcessingInfo
from vllm.multimodal.profiling import BaseDummyInputsBuilder
from vllm.sequence import IntermediateTensors

if TYPE_CHECKING:
    from transformers import BatchFeature, PretrainedConfig

    from vllm.config import VllmConfig
    from vllm.config.multimodal import BaseDummyOptions

DYNAMIC_ARG_DIMS = {
    "input_ids": 0,
    # set `positions` to last dim to support Qwen-mrope
    "positions": -1,
    "intermediate_tensors": 0,
    "inputs_embeds": 0,
}


class MultiModalProcessingInfo(BaseProcessingInfo):
    def get_supported_mm_limits(self):
        return {"image": None}

    def get_mm_max_tokens_per_item(self, seq_len, mm_counts):
        return {"image": self.get_max_image_tokens()}

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
        num_images = mm_counts.get("image", 0)

        processor = self.info.get_hf_processor()
        if "gemma3" in processor.__class__.__name__.lower():
            image_token = processor.boi_token
        else:
            image_token = getattr(processor, "image_token", "")
        return image_token * num_images

    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        mm_options: Mapping[str, "BaseDummyOptions"] | None = None,
    ) -> MultiModalDataDict:
        num_images = mm_counts.get("image", 0)

        target_width, target_height = self.info.get_max_image_size()

        image_overrides = mm_options.get("image") if mm_options else None

        return {
            "image": self._get_dummy_images(
                width=target_width,
                height=target_height,
                num_images=num_images,
                overrides=image_overrides,
            ),
        }


class MultiModalProcessor(BaseMultiModalProcessor[MultiModalProcessingInfo]):
    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        out_mm_kwargs: MultiModalKwargsItems,
    ):
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
        return None

    def _get_mm_fields_config(
        self,
        hf_inputs: "BatchFeature",
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        # HF Processors always return a mask but vLLM doesn't need it
        hf_inputs.pop("attention_mask", None)
        num_image_patches = hf_inputs.get("num_image_patches")
        mm_fields = {
            key: MultiModalFieldConfig.flat_from_sizes("image", num_image_patches)
            for key in hf_inputs
        }
        mm_fields["image_embeds"] = MultiModalFieldConfig.flat_from_sizes(
            "image", num_image_patches
        )

        # Keep these as batched, as they always have batch size as first dim
        mm_fields["image_grid_thw"] = MultiModalFieldConfig.batched("image")
        mm_fields["video_grid_thw"] = MultiModalFieldConfig.batched("image")
        mm_fields["num_image_patches"] = MultiModalFieldConfig.batched("image")
        return mm_fields

    def _get_hf_mm_data(
        self,
        mm_items: MultiModalDataItems,
    ) -> tuple[Mapping[str, object], Mapping[str, object]]:
        """
        In contrast to the base class, this method always adds
        `return_mm_token_type_ids` to the processor data
        """
        processor_data, passthrough_data = super()._get_hf_mm_data(mm_items)
        processor_data["return_mm_token_type_ids"] = True
        return processor_data, passthrough_data

    def apply(
        self,
        prompt: str | list[int],
        mm_data: MultiModalDataDict,
        hf_processor_mm_kwargs: Mapping[str, object],
        tokenization_kwargs: Mapping[str, object] | None = None,
        mm_uuids: MultiModalUUIDDict | None = None,
    ) -> MultiModalInputs:
        """
        Process multi-modal inputs to be used in vLLM.

        Apply HF Processor on prompt text and multi-modal data together,
        outputting token IDs and processed tensors.
        """
        if tokenization_kwargs is None:
            tokenization_kwargs = {}

        mm_items = self._to_mm_items(mm_data)
        hf_processor = self.info.get_hf_processor(**hf_processor_mm_kwargs)
        if not isinstance(prompt, str):
            # the prompt is the tokenized ids which is not supported
            # by the hf_processor, which is why we would need to decode the ids
            # into string
            prompt = hf_processor.decode(prompt)

        # Bypass cached processor and always apply to the full set of mm inputs
        # NOTE: we can't just set caching=False because base class method
        # transforms outputs to `MultiModalKwargs` which is not going to
        # work for Transformers. We have a lot of logic tied to
        # `mm_tokens_per_modality` below
        prompt_ids, processed_data, _ = self._apply_hf_processor_text_mm(
            prompt_text=prompt,
            mm_items=mm_items,
            hf_processor_mm_kwargs=hf_processor_mm_kwargs,
            tokenization_kwargs=tokenization_kwargs,
        )

        # For gemma3 we check `token_type_ids` as the key
        token_type_key = (
            "mm_token_type_ids"
            if "mm_token_type_ids" in processed_data
            else "token_type_ids"
        )
        mm_token_type_ids = processed_data.pop(token_type_key)

        # We can infer vLLM style placeholder from token type ids, if we split
        # it for each input `mm_data`.
        mm_positions = torch.where(mm_token_type_ids == 1)[1]
        images = mm_items.get_items("image", ImageProcessorItems)
        multimodal_config = self.info.ctx.model_config.multimodal_config
        mm_processor_kwargs = multimodal_config.mm_processor_kwargs or {}
        image_sizes = []
        for item_idx in range(len(images)):
            image_size = images.get_image_size(item_idx)
            image_sizes.append((image_size.height, image_size.width))

        mm_tokens_per_modality = hf_processor._get_num_multimodal_tokens(
            image_sizes=image_sizes, **mm_processor_kwargs
        )

        mm_placeholders = {}
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
        mm_kwargs = MultiModalKwargsItems.from_hf_inputs(
            processed_data,
            self._get_mm_fields_config(processed_data, hf_processor_mm_kwargs),
        )

        # Use overrides if provided; fallback to data-dependent hashing.
        mm_hashes = self._hash_mm_items(
            mm_items, hf_processor_mm_kwargs, tokenization_kwargs, mm_uuids=mm_uuids
        )

        return MultiModalInputs(
            type="multimodal",
            prompt_token_ids=prompt_ids,
            mm_kwargs=mm_kwargs,
            mm_hashes=mm_hashes,
            mm_placeholders=mm_placeholders,
        )


class MultiModalMixin(SupportsMultiModal, SupportsMRoPE):
    supports_multimodal_raw_input_only = True
    merge_by_field_config = True
    # Backwards compatibility for prev released models. State dicts back then
    # had different formats and cannot be loaded with `AutoModel` mapping as is
    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_prefix={
            "language_model.model": "model.language_model",
            "text_model.model": "model.text_model",
            "vision_tower": "model.vision_tower",
            "vqmodel": "model.vqmodel",
            "visual": "model.visual",
            "vision_model": "model.vision_model",
            "vision_embed_tokens": "model.vision_embed_tokens",
            "image_newline": "model.image_newline",
            "multi_modal_projector": "model.multi_modal_projector",
            "text_model.lm_head": "lm_head",
            "language_model.lm_head": "lm_head",
            # Qwen models used "model" as the name for the language model.
            # Therefore, we must map each of submodule explicitly to avoid
            # conflicts with newer models that use "model.language_model".
            "model.embed_tokens": "model.language_model.embed_tokens",
            "model.layers": "model.language_model.layers",
            "model.norm": "model.language_model.norm",
        }
    )

    def __init__(self, *, vllm_config: "VllmConfig", prefix: str = ""):
        # Skip SupportsMRoPE.__init__ and call the next class in MRO
        super(SupportsMRoPE, self).__init__(vllm_config=vllm_config, prefix=prefix)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: object,
    ) -> torch.Tensor | IntermediateTensors:
        # Gemma3 and PaliGemma needs `token_type_ids` to work correctly
        # Other models will not have `token_type_ids` in kwargs
        kwargs = {k: v for k, v in kwargs.items() if k == "token_type_ids"}
        model_output = super().forward(
            input_ids, positions, intermediate_tensors, inputs_embeds, **kwargs
        )
        return model_output

    def get_language_model(self) -> torch.nn.Module:
        """Transformers backend multimodal classes do not contain a separate vLLM
        language model class. Therefore, in order to return a language model vLLM class,
        we use a wrapper to give `self` the same interface as a text model."""

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

    def get_multimodal_embeddings(self, **kwargs):
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
        kwargs.pop("token_type_ids", None)  # used only in `forward`
        if pixel_values is not None:
            vision_embeddings = self.model.get_image_features(pixel_values, **kwargs)

            if isinstance(vision_embeddings, torch.Tensor):
                if vision_embeddings.ndim == 2:
                    vision_embeddings = vision_embeddings.unsqueeze(0)

                # Embeddings have to be 2D tensors of length `num_images`
                # but transformers returns concat tensors if each patch
                # is of different size. We split it back to make vLLM happy
                vision_embeddings = torch.split(
                    vision_embeddings, num_image_patches.flatten().tolist()
                )
                vision_embeddings = [
                    embed.flatten(start_dim=0, end_dim=-2)
                    for embed in vision_embeddings
                ]

            return vision_embeddings

    def get_mrope_input_positions(
        self,
        input_tokens: list[int],
        hf_config: "PretrainedConfig",
        image_grid_thw: list[list[int]] | torch.Tensor | None,
        video_grid_thw: list[list[int]] | torch.Tensor | None,
        second_per_grid_ts: list[float] | None = None,
        context_len: int = 0,
        seq_len: int | None = None,
        audio_feature_lengths: torch.Tensor | None = None,
        use_audio_in_video: bool = False,
    ) -> tuple[torch.Tensor, int]:
        if any((second_per_grid_ts, audio_feature_lengths, use_audio_in_video)):
            raise NotImplementedError("Transformers backend only supports images.")

        if isinstance(image_grid_thw, list):
            image_grid_thw = torch.tensor(image_grid_thw)
        if isinstance(video_grid_thw, list):
            video_grid_thw = torch.tensor(video_grid_thw)

        mrope_positions, mrope_position_delta = self.model.get_rope_index(
            input_ids=torch.tensor(input_tokens).unsqueeze(0),
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
        )

        mrope_positions = mrope_positions[:, 0, context_len:seq_len]
        mrope_position_delta = mrope_position_delta[0].item()

        return mrope_positions, mrope_position_delta
