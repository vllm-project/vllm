# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inference-only Deepseek-OCR model compatible with HuggingFace weights."""

import math
from collections.abc import Iterable, Mapping, Sequence
from typing import Annotated, Literal

import torch
import torch.nn as nn
from transformers import BatchFeature, CLIPVisionConfig

from vllm.config import VllmConfig
from vllm.config.multimodal import BaseDummyOptions
from vllm.model_executor.models.interfaces import (
    MultiModalEmbeddings,
    SupportsLoRA,
    SupportsMultiModal,
    SupportsPP,
)
from vllm.model_executor.models.module_mapping import MultiModelKeys
from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    WeightsMapper,
    init_vllm_registered_model,
    maybe_prefix,
)
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import (
    MultiModalDataDict,
    MultiModalFieldConfig,
    MultiModalKwargsItems,
    NestedTensors,
)
from vllm.multimodal.parse import (
    ImageEmbeddingItems,
    ImageProcessorItems,
    ImageSize,
    MultiModalDataItems,
)
from vllm.multimodal.processing import (
    BaseDummyInputsBuilder,
    BaseMultiModalProcessor,
    BaseProcessingInfo,
    PromptReplacement,
    PromptUpdate,
)
from vllm.sampling_params import SamplingParams
from vllm.sequence import IntermediateTensors
from vllm.tokenizers import cached_tokenizer_from_config
from vllm.transformers_utils.configs.deepseek_vl2 import DeepseekVLV2Config
from vllm.transformers_utils.processors.deepseek_ocr2 import (
    BASE_SIZE,
    CROP_MODE,
    IMAGE_SIZE,
    DeepseekOCR2Processor,
    count_tiles,
)
from vllm.utils.tensor_schema import TensorSchema, TensorShape
from vllm.v1.sample.logits_processor import (
    AdapterLogitsProcessor,
    RequestLogitsProcessor,
)

from .deepencoder2 import build_sam_vit_b, build_qwen2_decoder_as_encoder
from .deepseek_vl2 import MlpProjector

# The image token id may be various
_IMAGE_TOKEN = "<image>"


class DeepseekOCRImagePixelInputs(TensorSchema):
    """
    Dimensions:
        - b: Batch size
        - n: Number of images
        - p: Number of patches
        - base_size: Base size of the processor
        - image_size: Image size of the processor
    """

    type: Literal["pixel_values"]
    data: Annotated[
        torch.Tensor,
        TensorShape("bn", 3, "base_size", "base_size", dynamic_dims={"bnp"}),
    ]
    images_crop: Annotated[
        torch.Tensor,
        TensorShape("bnp", 3, "image_size", "image_size", dynamic_dims={"bnp"}),
    ]
    images_spatial_crop: Annotated[torch.Tensor, TensorShape("bn", 2)]


class NoRepeatNGramLogitsProcessor:
    def __init__(
            self,
            ngram_size: int,
            window_size: int,
            whitelist_token_ids: set[int] | None = None,
    ):
        self.ngram_size = ngram_size
        self.window_size = window_size
        self.whitelist_token_ids = whitelist_token_ids or set()

    def __call__(
            self,
            output_ids: list[int],
            logits: torch.Tensor,
    ) -> torch.Tensor:
        if len(output_ids) < self.ngram_size:
            return logits

        current_prefix = tuple(output_ids[-(self.ngram_size - 1):])

        search_start = max(0, len(output_ids) - self.window_size)
        search_end = len(output_ids) - self.ngram_size + 1

        banned_tokens = set()
        for i in range(search_start, search_end):
            ngram = tuple(output_ids[i: i + self.ngram_size])
            if ngram[:-1] == current_prefix:
                banned_tokens.add(ngram[-1])

        banned_tokens = banned_tokens - self.whitelist_token_ids

        if banned_tokens:
            logits[list(banned_tokens)] = -float("inf")

        return logits


class NGramPerReqLogitsProcessor(AdapterLogitsProcessor):
    """Example of overriding the wrapper class `__init__()` in order to utilize
    info about the device type"""

    @classmethod
    def validate_params(cls, params: SamplingParams):
        ngram_size = params.extra_args and params.extra_args.get("ngram_size")
        window_size = params.extra_args and params.extra_args.get("window_size", 100)
        whitelist_token_ids = params.extra_args and params.extra_args.get(
            "whitelist_token_ids", None
        )
        # if ngram_size is not provided, skip validation because the processor
        # will not be used.
        if ngram_size is None:
            return None

        if not isinstance(ngram_size, int) or ngram_size <= 0:
            raise ValueError(
                f"`ngram_size` has to be a strictly positive integer, got {ngram_size}."
            )
        if not isinstance(window_size, int) or window_size <= 0:
            raise ValueError(
                "`window_size` has to be a strictly positive integer, "
                f"got {window_size}."
            )
        if whitelist_token_ids is not None and not isinstance(
                whitelist_token_ids, Iterable
        ):
            raise ValueError(
                "`whitelist_token_ids` has to be a sequence of integers, "
                f"got {whitelist_token_ids}."
            )

    def is_argmax_invariant(self) -> bool:
        return False

    def new_req_logits_processor(
            self,
            params: SamplingParams,
    ) -> RequestLogitsProcessor | None:
        ngram_size = params.extra_args and params.extra_args.get("ngram_size")
        window_size = params.extra_args and params.extra_args.get("window_size", 100)
        whitelist_token_ids = params.extra_args and params.extra_args.get(
            "whitelist_token_ids", None
        )
        if ngram_size is None:
            return None

        whitelist_token_ids = set(whitelist_token_ids) if whitelist_token_ids else None
        return NoRepeatNGramLogitsProcessor(
            ngram_size=ngram_size,
            window_size=window_size,
            whitelist_token_ids=whitelist_token_ids,
        )


class DeepseekOCR2ProcessingInfo(BaseProcessingInfo):

    def get_hf_config(self):
        return self.ctx.get_hf_config(DeepseekVLV2Config)

    def get_hf_processor(self, **kwargs: object):
        return self.ctx.get_hf_processor(DeepseekOCR2Processor, **kwargs)

    def get_supported_mm_limits(self) -> Mapping[str, int | None]:
        return {"image": None}

    def get_num_image_tokens(self,
                             *,
                             image_width: int,
                             image_height: int,
                             cropping: bool = True) -> int:
        hf_processor = self.get_hf_processor()

        # image_size = hf_processor.image_size
        # patch_size = hf_processor.patch_size
        # downsample_ratio = hf_processor.downsample_ratio

        image_size = IMAGE_SIZE
        base_size = BASE_SIZE
        patch_size = 16
        downsample_ratio = 4

        if CROP_MODE:
            if image_width <= 768 and image_height <= 768:
                crop_ratio = [1, 1]
            else:
                # images_crop_raw, crop_ratio = hf_processor.dynamic_preprocess(image)

                # find the closest aspect ratio to the target
                crop_ratio = count_tiles(image_width, image_height,
                                         image_size=IMAGE_SIZE)

                # print('===========')
                # print('crop_ratio ', crop_ratio)
                # print('============')

            num_width_tiles, num_height_tiles = crop_ratio
        else:
            num_width_tiles = num_height_tiles = 1

        h = w = math.ceil((base_size // patch_size) / downsample_ratio)

        h2 = w2 = math.ceil((image_size // patch_size) / downsample_ratio)

        global_views_tokens = h * (w)
        if num_width_tiles > 1 or num_height_tiles > 1:
            local_views_tokens = (num_height_tiles * h2) * (num_width_tiles * w2)
        else:
            local_views_tokens = 0

        return global_views_tokens + local_views_tokens + 1

    def get_image_size_with_most_features(self) -> ImageSize:

        if IMAGE_SIZE == 1024 and BASE_SIZE == 1280:
            return ImageSize(width=1024 * 2, height=1024 * 2)
        return ImageSize(width=768 * 2, height=768 * 2)


class DeepseekOCR2DummyInputsBuilder(
    BaseDummyInputsBuilder[DeepseekOCR2ProcessingInfo]):
    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        num_images = mm_counts.get("image", 0)

        processor = self.info.get_hf_processor()
        image_token = processor.image_token

        return image_token * num_images

    def get_dummy_mm_data(
            self,
            seq_len: int,
            mm_counts: Mapping[str, int],
            mm_options: Mapping[str, BaseDummyOptions] | None = None,
    ) -> MultiModalDataDict:
        num_images = mm_counts.get("image", 0)

        max_image_size = self.info.get_image_size_with_most_features()

        return {
            "image": self._get_dummy_images(
                width=max_image_size.width,
                height=max_image_size.height,
                num_images=num_images,
            )
        }


class DeepseekOCRMultiModalProcessor(
    BaseMultiModalProcessor[DeepseekOCR2ProcessingInfo]
):
    def _call_hf_processor(
            self,
            prompt: str,
            mm_data: Mapping[str, object],
            mm_kwargs: Mapping[str, object],
            tok_kwargs: Mapping[str, object],
    ) -> BatchFeature:
        if mm_data:
            processed_outputs = self.info.ctx.call_hf_processor(
                self.info.get_hf_processor(**mm_kwargs),
                dict(prompt=prompt, **mm_data),
                mm_kwargs,
            )

        else:
            tokenizer = self.info.get_tokenizer()
            processed_outputs = tokenizer(
                prompt, add_special_tokens=True, return_tensors="pt"
            )

        return processed_outputs

    def _get_mm_fields_config(
            self,
            hf_inputs: BatchFeature,
            hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        images_spatial_crop = hf_inputs.get("images_spatial_crop", torch.empty((0, 2)))
        is_tiled = (images_spatial_crop[:, 0] > 1) | (images_spatial_crop[:, 1] > 1)
        patches_per_image = torch.where(is_tiled, images_spatial_crop.prod(dim=-1), 0)
        return dict(
            pixel_values=MultiModalFieldConfig.batched("image"),
            images_spatial_crop=MultiModalFieldConfig.batched("image"),
            images_crop=MultiModalFieldConfig.flat_from_sizes(
                "image", patches_per_image
            ),
        )

    def _get_prompt_updates(
            self,
            mm_items: MultiModalDataItems,
            hf_processor_mm_kwargs: Mapping[str, object],
            out_mm_kwargs: MultiModalKwargsItems,
    ) -> Sequence[PromptUpdate]:
        hf_processor = self.info.get_hf_processor(**hf_processor_mm_kwargs)

        image_token_id = hf_processor.image_token_id
        assert isinstance(image_token_id, int)

        def get_replacement_deepseek_vl2(item_idx: int):
            images = mm_items.get_items(
                "image", (ImageEmbeddingItems, ImageProcessorItems)
            )

            if isinstance(images, ImageEmbeddingItems):
                num_image_tokens = images.get_feature_size(item_idx)
            else:
                size = images.get_image_size(item_idx)

                num_image_tokens = self.info.get_num_image_tokens(
                    image_width=size.width,
                    image_height=size.height,
                    cropping=CROP_MODE,
                )
            return [image_token_id] * num_image_tokens

        return [
            PromptReplacement(
                modality="image",
                target=[image_token_id],
                replacement=get_replacement_deepseek_vl2,
            )
        ]


@MULTIMODAL_REGISTRY.register_processor(
    DeepseekOCRMultiModalProcessor,
    info=DeepseekOCR2ProcessingInfo,
    dummy_inputs=DeepseekOCR2DummyInputsBuilder,
)
class DeepseekOCR2ForCausalLM(nn.Module, SupportsMultiModal, SupportsPP, SupportsLoRA):
    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_prefix={
            # map prefix for language backbone
            "model.embed_tokens.": "language_model.model.embed_tokens.",
            "model.layers.": "language_model.model.layers.",
            "model.norm.": "language_model.model.norm.",
            "lm_head.": "language_model.lm_head.",
            "model.qwen2_model.": "vision_model.",
            # remove "model." prefix for other components
            "model.": "",
        }
    )

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None:
        if modality.startswith("image"):
            return "<image>"

        raise ValueError("Only image modality is supported")

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()

        config: DeepseekVLV2Config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        multimodal_config = vllm_config.model_config.multimodal_config

        self.config = config
        self.multimodal_config = multimodal_config

        self.vision_config = config.vision_config
        self.projector_config = config.projector_config
        self.text_config = config.text_config

        model_config = vllm_config.model_config
        tokenizer = cached_tokenizer_from_config(model_config)
        self.image_token_id = tokenizer.vocab[_IMAGE_TOKEN]

        with self._mark_tower_model(vllm_config, "image"):
            self.sam_model = build_sam_vit_b()
            self.vision_model = build_qwen2_decoder_as_encoder()

            self.projector = MlpProjector(self.projector_config)
            self.tile_tag = config.tile_tag
            self.global_view_pos = config.global_view_pos

            # special token for image token sequence format
            n_embed = self.projector_config.n_embed
            embed_std = 1 / torch.sqrt(torch.tensor(n_embed, dtype=torch.float32))
            if self.tile_tag == "2D":
                # This is a typo in original implementation
                self.view_seperator = nn.Parameter(torch.randn(n_embed) * embed_std)
            else:
                raise ValueError(
                    f"Only 2D tile_tag is supported currently, got: {self.tile_tag}"
                )

        with self._mark_language_model(vllm_config):
            self.language_model = init_vllm_registered_model(
                vllm_config=vllm_config,
                hf_config=self.text_config,
                prefix=maybe_prefix(prefix, "language_model"),
            )

        self.make_empty_intermediate_tensors = (
            self.language_model.make_empty_intermediate_tensors
        )

    def _parse_and_validate_image_input(
            self, **kwargs: object
    ) -> DeepseekOCRImagePixelInputs | None:
        pixel_values = kwargs.pop("pixel_values", None)
        images_spatial_crop = kwargs.pop("images_spatial_crop", None)
        images_crop = kwargs.pop("images_crop", None)

        if pixel_values is None or torch.sum(pixel_values).item() == 0:
            return None

        base_size = self.vision_config.image_size
        return DeepseekOCRImagePixelInputs(
            type="pixel_values",
            data=pixel_values,
            images_crop=images_crop,
            images_spatial_crop=images_spatial_crop,
            resolve_bindings={
                "base_size": base_size,
            },
        )

    def _encode_global_features(self, image_tensor: torch.Tensor) -> torch.Tensor:
        global_features_1 = self.sam_model(image_tensor)
        global_features_2 = self.vision_model(global_features_1)

        features = self.projector(global_features_2)

        _, hw, dim = features.shape

        return features.view(-1, dim)

    def _encode_local_features(
            self, patches: torch.Tensor
    ) -> torch.Tensor | None:
        if torch.sum(patches).item() == 0:
            return None

        local_features = self.sam_model(patches)
        local_features = self.vision_model(local_features)

        features = self.projector(local_features)

        _, _, dim = features.shape

        return features.view(-1, dim)

    def _pixel_values_to_embedding(
            self,
            pixel_values: torch.Tensor,
            images_crop: torch.Tensor,
            images_spatial_crop: torch.Tensor,
    ) -> NestedTensors:
        images_in_this_batch = []

        is_tiled = (images_spatial_crop[:, 0] > 1) | (images_spatial_crop[:, 1] > 1)
        patches_per_image = torch.where(is_tiled, images_spatial_crop.prod(dim=-1), 0)
        images_crop = images_crop.split(patches_per_image.tolist())
        for jdx in range(images_spatial_crop.size(0)):
            patches = images_crop[jdx]
            image_ori = pixel_values[[jdx]]

            global_features = self._encode_global_features(image_ori)
            local_features = self._encode_local_features(patches)

            if local_features is not None:
                combined = torch.cat(
                    [local_features, global_features, self.view_seperator[None, :]],
                    dim=0,
                )
            else:
                combined = torch.cat(
                    [global_features, self.view_seperator[None, :]], dim=0
                )

            images_in_this_batch.append(combined)

        return images_in_this_batch

    def _process_image_input(
            self, image_input: DeepseekOCRImagePixelInputs
    ) -> torch.Tensor:
        pixel_values = image_input.data
        images_crop = image_input.images_crop
        images_spatial_crop = image_input.images_spatial_crop.to(dtype=torch.long)

        vision_features = self._pixel_values_to_embedding(
            pixel_values=pixel_values,
            images_crop=images_crop,
            images_spatial_crop=images_spatial_crop,
        )

        return vision_features

    def embed_multimodal(self, **kwargs: object) -> MultiModalEmbeddings | None:
        image_input = self._parse_and_validate_image_input(**kwargs)
        if image_input is None:
            return None
        vision_embeddings = self._process_image_input(image_input)
        return vision_embeddings

    def forward(
            self,
            input_ids: torch.Tensor,
            positions: torch.Tensor,
            intermediate_tensors: IntermediateTensors | None = None,
            inputs_embeds: torch.Tensor | None = None,
            **kwargs: object,
    ):
        if intermediate_tensors is not None:
            inputs_embeds = None

        hidden_states = self.language_model(
            input_ids, positions, intermediate_tensors, inputs_embeds=inputs_embeds
        )

        return hidden_states

    def compute_logits(
            self,
            hidden_states: torch.Tensor,
    ) -> torch.Tensor | None:
        return self.language_model.compute_logits(hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(self)
        autoloaded_weights = loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)
        return autoloaded_weights

    def get_mm_mapping(self) -> MultiModelKeys:
        """
        Get the module prefix in multimodal models
        """
        return MultiModelKeys.from_string_field(
            language_model="language_model",
            connector="projector",
            tower_model=["sam_model", "vision_model"],
        )
