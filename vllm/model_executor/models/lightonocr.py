# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Iterable, Mapping, Sequence
from typing import TypeVar

import torch
import torch.nn as nn
from transformers import (
    BatchFeature,
    PixtralVisionConfig,
)

from vllm.config import VllmConfig
from vllm.model_executor.models.mistral3 import (
    Mistral3DummyInputsBuilder,
    Mistral3ForConditionalGeneration,
    Mistral3MultiModalProjector,
    Mistral3ProcessingInfo,
    _build_mistral3_info,
    init_vision_tower_for_llava,
)
from vllm.model_executor.models.pixtral import PixtralHFEncoderInfo
from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    WeightsMapper,
    init_vllm_registered_model,
    maybe_prefix,
)
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.cache import BaseMultiModalProcessorCache
from vllm.multimodal.inputs import MultiModalFieldConfig, MultiModalKwargsItems
from vllm.multimodal.parse import ImageProcessorItems, MultiModalDataItems
from vllm.multimodal.processing import (
    BaseMultiModalProcessor,
    PromptReplacement,
    PromptUpdate,
    PromptUpdateDetails,
)
from vllm.multimodal.profiling import BaseDummyInputsBuilder

_I = TypeVar("_I", bound=Mistral3ProcessingInfo)


class LightOnOCRMultiModalProcessor(BaseMultiModalProcessor[Mistral3ProcessingInfo]):
    def _call_hf_processor(
        self,
        prompt: str,
        mm_data: Mapping[str, object],
        mm_kwargs: Mapping[str, object],
        tok_kwargs: Mapping[str, object],
    ) -> BatchFeature:
        processed_outputs = super()._call_hf_processor(
            prompt=prompt,
            mm_data=mm_data,
            mm_kwargs=mm_kwargs,
            tok_kwargs=tok_kwargs,
        )

        # NOTE: LightOnOCR does not use break/end tokens, so we remove them here.
        input_ids = processed_outputs.get("input_ids")
        if input_ids is not None:
            processor = self.info.get_hf_processor()
            tokenizer = self.info.get_tokenizer()
            vocab = tokenizer.get_vocab()

            break_id = vocab.get(processor.image_break_token)
            end_id = vocab.get(processor.image_end_token)

            # create mask to remove break/end tokens
            keep_mask = ~torch.isin(
                input_ids,
                torch.tensor([break_id, end_id]),
            )

            processed_outputs["input_ids"] = input_ids[keep_mask].unsqueeze(0)
            if "attention_mask" in processed_outputs:
                processed_outputs["attention_mask"] = processed_outputs[
                    "attention_mask"
                ][keep_mask].unsqueeze(0)

        # un-pad pixel_values per-image so caches remain independent.
        pixel_values = processed_outputs.get("pixel_values")
        if pixel_values is not None:
            image_sizes = processed_outputs["image_sizes"]
            assert len(pixel_values) == len(image_sizes)
            processed_outputs["pixel_values"] = [
                p[:, :h, :w] for p, (h, w) in zip(pixel_values, image_sizes)
            ]

        return processed_outputs

    def _get_mm_fields_config(
        self,
        hf_inputs: BatchFeature,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        return dict(
            pixel_values=MultiModalFieldConfig.batched("image"),
            image_embeds=MultiModalFieldConfig.batched("image"),
        )

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        out_mm_kwargs: MultiModalKwargsItems,
    ) -> Sequence[PromptUpdate]:
        hf_config = self.info.get_hf_config()
        image_token_id = hf_config.image_token_index

        assert isinstance(hf_config.vision_config, PixtralVisionConfig)
        encoder_info = PixtralHFEncoderInfo(hf_config)

        def replace(item_idx: int):
            images = mm_items.get_items("image", ImageProcessorItems)
            size = images.get_image_size(item_idx)
            ncols, nrows = encoder_info.get_patch_grid_size(
                image_width=size.width, image_height=size.height
            )
            # break/end tokens are not used in LightOnOCR
            tokens = [image_token_id] * (ncols * nrows)
            return PromptUpdateDetails.select_token_id(tokens, image_token_id)

        return [
            PromptReplacement(
                modality="image", target=[image_token_id], replacement=replace
            )
        ]


def _build_LightOnOCR_processor(
    info: _I,
    dummy_inputs: BaseDummyInputsBuilder[_I],
    *,
    cache: BaseMultiModalProcessorCache | None = None,
):
    assert isinstance(info, Mistral3ProcessingInfo)
    return LightOnOCRMultiModalProcessor(info, dummy_inputs, cache=cache)


@MULTIMODAL_REGISTRY.register_processor(
    _build_LightOnOCR_processor,
    info=_build_mistral3_info,
    dummy_inputs=Mistral3DummyInputsBuilder,
)
class LightOnOCRForConditionalGeneration(Mistral3ForConditionalGeneration):
    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_prefix={
            "model.vision_encoder.": "vision_tower.",
            "model.vision_projection.": "multi_modal_projector.",
            "lm_head.": "language_model.lm_head.",
            "model.language_model.": "language_model.model.",
        }
    )

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        nn.Module.__init__(self)

        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        multimodal_config = vllm_config.model_config.multimodal_config

        self.config = config
        self.multimodal_config = multimodal_config

        self.vision_tower = init_vision_tower_for_llava(
            config,
            quant_config=quant_config,
            multimodal_config=multimodal_config,
            require_post_norm=False,
            prefix=maybe_prefix(prefix, "vision_tower"),
        )

        self.multi_modal_projector = Mistral3MultiModalProjector(
            vision_hidden_size=config.vision_config.hidden_size,
            text_hidden_size=config.text_config.hidden_size,
            projector_hidden_act=config.projector_hidden_act,
            spatial_merge_size=config.spatial_merge_size,
            patch_size=config.vision_config.patch_size,
            multimodal_projector_bias=config.multimodal_projector_bias,
            quant_config=quant_config,
            prefix=maybe_prefix(prefix, "multi_modal_projector"),
        )

        self.language_model = init_vllm_registered_model(
            vllm_config=vllm_config,
            hf_config=config.text_config,
            prefix=maybe_prefix(prefix, "language_model"),
        )

        self.make_empty_intermediate_tensors = (
            self.language_model.make_empty_intermediate_tensors
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)
