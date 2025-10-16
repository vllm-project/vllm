# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Iterable, Mapping, Sequence
from typing import Annotated, Literal, TypeVar

import torch
import torch.nn as nn
from transformers import (
    BatchFeature,
    Mistral3Config,
    PixtralVisionConfig,
)
from transformers.models.pixtral import PixtralProcessor

from vllm.config import VllmConfig
from vllm.config.multimodal import BaseDummyOptions
from vllm.model_executor.models.interfaces import (
    MultiModalEmbeddings,
    SupportsLoRA,
    SupportsMultiModal,
    SupportsPP,
)
from vllm.model_executor.models.mistral3 import (
    BaseLlavaProcessingInfo,
    Mistral3DummyInputsBuilder,
    Mistral3MultiModalProjector,
    init_vision_tower_for_llava,
)
from vllm.model_executor.models.module_mapping import MultiModelKeys
from vllm.model_executor.models.pixtral import PixtralHFEncoderInfo
from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    WeightsMapper,
    init_vllm_registered_model,
    maybe_prefix,
)
from vllm.model_executor.models.vision import get_vision_encoder_info
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.cache import BaseMultiModalProcessorCache
from vllm.multimodal.inputs import MultiModalFieldConfig, MultiModalKwargs
from vllm.multimodal.parse import ImageProcessorItems, MultiModalDataItems
from vllm.multimodal.processing import (
    BaseMultiModalProcessor,
    InputProcessingContext,
    PromptReplacement,
    PromptUpdate,
    PromptUpdateDetails,
)
from vllm.multimodal.profiling import BaseDummyInputsBuilder
from vllm.sequence import IntermediateTensors
from vllm.utils.tensor_schema import TensorSchema, TensorShape


class LightOnOCRImagePixelInputs(TensorSchema):
    """
    Dimensions:
        - bn: Batch size * number of images
        - c: Number of channels (3)
        - h: Height of each image
        - w: Width of each image
    """

    type: Literal["pixel_values_pixtral"] = "pixel_values_pixtral"

    # Note that `height` or `width` may be different per batch and image,
    # in which case the data is passed as a list instead of a batched tensor.
    pixel_values: Annotated[
        torch.Tensor | list[torch.Tensor],
        TensorShape("bn", 3, "h", "w", dynamic_dims={"h", "w"}),
    ]


_I = TypeVar("_I", bound="LightOnOCRProcessingInfo")


class LightOnOCRProcessingInfo(BaseLlavaProcessingInfo):
    def get_hf_config(self):
        return self.ctx.get_hf_config(Mistral3Config)

    def get_vision_encoder_info(self):
        return get_vision_encoder_info(self.get_hf_config())

    def get_hf_processor(self, **kwargs: object):
        return self.ctx.get_hf_processor(PixtralProcessor, **kwargs)


class LightOnOCRMultiModalProcessor(BaseMultiModalProcessor[LightOnOCRProcessingInfo]):
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
        out_mm_kwargs: MultiModalKwargs,
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


def _build_LightOnOCR_info(ctx: InputProcessingContext) -> LightOnOCRProcessingInfo:
    hf_config = ctx.get_hf_config(Mistral3Config)
    assert isinstance(hf_config.vision_config, PixtralVisionConfig)
    return LightOnOCRProcessingInfo(ctx)


def _build_LightOnOCR_processor(
    info: _I,
    dummy_inputs: BaseDummyInputsBuilder[_I],
    *,
    cache: BaseMultiModalProcessorCache | None = None,
):
    assert isinstance(info, LightOnOCRProcessingInfo)
    return LightOnOCRMultiModalProcessor(info, dummy_inputs, cache=cache)


class LightOnOCRDummyInputsBuilder(Mistral3DummyInputsBuilder):
    def get_dummy_text(self, mm_counts):
        n = mm_counts.get("image", 0)
        processor = self.info.get_hf_processor()
        image_token = processor.image_token
        return image_token * n

    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        mm_options: Mapping[str, BaseDummyOptions] | None = None,
    ):
        n = mm_counts.get("image", 0)
        w, h = self.info.get_image_size_with_most_features()
        image_overrides = mm_options.get("image") if mm_options else None
        return {
            "image": self._get_dummy_images(
                width=w, height=h, num_images=n, overrides=image_overrides
            )
        }


class LightOnOCRMultiModalProjector(Mistral3MultiModalProjector):
    pass


@MULTIMODAL_REGISTRY.register_processor(
    _build_LightOnOCR_processor,
    info=_build_LightOnOCR_info,
    dummy_inputs=LightOnOCRDummyInputsBuilder,
)
class LightOnOCRForConditionalGeneration(
    nn.Module, SupportsLoRA, SupportsMultiModal, SupportsPP
):
    merge_by_field_config = True

    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
    }

    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_prefix={
            "vision_encoder.": "vision_tower.",
            "vision_projection.": "multi_modal_projector.",
        }
    )

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None:
        if modality.startswith("image"):
            return None
        raise ValueError("Only image modality is supported")

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        multimodal_config = vllm_config.model_config.multimodal_config

        self.config = config
        self.multimodal_config = multimodal_config

        self.vision_tower = init_vision_tower_for_llava(
            config,
            quant_config,
            require_post_norm=False,
            prefix=maybe_prefix(prefix, "vision_tower"),
        )

        self.multi_modal_projector = LightOnOCRMultiModalProjector(
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

    def _parse_and_validate_image_input(
        self, **kwargs: object
    ) -> LightOnOCRImagePixelInputs | None:
        pixel_values = kwargs.pop("pixel_values", None)
        image_embeds = kwargs.pop("image_embeds", None)

        if pixel_values is None and image_embeds is None:
            return None

        return LightOnOCRImagePixelInputs(
            type="pixel_values_pixtral", pixel_values=pixel_values
        )

    def _process_image_input(self, image_input: LightOnOCRImagePixelInputs):
        image_sizes = [(t.shape[-2], t.shape[-1]) for t in image_input["pixel_values"]]
        feats = self.vision_tower(image_input["pixel_values"])
        if isinstance(feats, torch.Tensor):
            return self.multi_modal_projector(feats, image_sizes)
        feature_sizes = [f.shape[0] // self.config.spatial_merge_size**2 for f in feats]
        embeds = self.multi_modal_projector(torch.cat(feats), image_sizes)
        return (
            tuple(torch.split(embeds, feature_sizes))
            if len(feature_sizes) > 1
            else (embeds,)
        )

    def get_language_model(self) -> torch.nn.Module:
        return self.language_model

    def get_multimodal_embeddings(self, **kwargs: object) -> MultiModalEmbeddings:
        image_input = self._parse_and_validate_image_input(**kwargs)
        if image_input is None:
            return []

        vision_embeddings = self._process_image_input(image_input)

        return vision_embeddings

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: object,
    ) -> torch.Tensor | IntermediateTensors:
        if intermediate_tensors is not None:
            inputs_embeds = None
        hidden_states = self.language_model.model(
            input_ids, positions, intermediate_tensors, inputs_embeds=inputs_embeds
        )
        return hidden_states

    def compute_logits(self, hidden_states: torch.Tensor):
        return self.language_model.compute_logits(hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)

    def get_mm_mapping(self) -> MultiModelKeys:
        return MultiModelKeys.from_string_field(
            language_model="language_model",
            connector="multi_modal_projector",
            tower_model="vision_tower",
        )
