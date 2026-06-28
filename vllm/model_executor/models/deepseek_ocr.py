# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inference-only Deepseek-OCR model compatible with HuggingFace weights."""

import math
from collections.abc import Hashable, Iterable, Mapping, Sequence
from typing import Annotated, Any, Literal

import torch
import torch.nn as nn
from transformers import BatchFeature, CLIPVisionConfig

from vllm.config import VllmConfig
from vllm.config.multimodal import BaseDummyOptions
from vllm.inputs import MultiModalDataDict
from vllm.model_executor.models.interfaces import (
    MultiModalEmbeddings,
    SupportsEncoderCudaGraph,
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
from vllm.transformers_utils.processors.deepseek_ocr import (
    BASE_SIZE,
    CROP_MODE,
    IMAGE_SIZE,
    DeepseekOCRProcessor,
    count_tiles,
)
from vllm.utils.tensor_schema import TensorSchema, TensorShape
from vllm.v1.sample.logits_processor import (
    AdapterLogitsProcessor,
    RequestLogitsProcessor,
)
from vllm.v1.worker.encoder_cudagraph_defs import (
    EncoderCudaGraphCaptureInputs,
    EncoderCudaGraphConfig,
    EncoderCudaGraphReplayBuffers,
    EncoderItemSpec,
)

from .deepencoder import DeepCLIPVisionTransformer, build_sam_vit_b
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

        current_prefix = tuple(output_ids[-(self.ngram_size - 1) :])

        search_start = max(0, len(output_ids) - self.window_size)
        search_end = len(output_ids) - self.ngram_size + 1

        banned_tokens = set()
        for i in range(search_start, search_end):
            ngram = tuple(output_ids[i : i + self.ngram_size])
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


class DeepseekOCRProcessingInfo(BaseProcessingInfo):
    def get_hf_config(self):
        return self.ctx.get_hf_config(DeepseekVLV2Config)

    def get_hf_processor(self, **kwargs: object):
        v1_processor_config = dict(
            image_size=IMAGE_SIZE,
            base_size=BASE_SIZE,
            crop_mode=CROP_MODE,
            strategy="v1",
        )

        return self.ctx.get_hf_processor(
            DeepseekOCRProcessor,
            **{**v1_processor_config, **kwargs},
        )

    def get_supported_mm_limits(self) -> Mapping[str, int | None]:
        return {"image": None}

    def get_num_image_tokens(
        self, *, image_width: int, image_height: int, cropping: bool = True
    ) -> int:
        image_size = IMAGE_SIZE
        base_size = BASE_SIZE
        patch_size = 16
        downsample_ratio = 4

        # Use the caller-supplied `cropping` flag so that callers that disable
        # crop mode for multi-image requests get a consistent token count.
        if cropping:
            if image_width <= IMAGE_SIZE and image_height <= IMAGE_SIZE:
                crop_ratio = [1, 1]
            else:
                # find the closest aspect ratio to the target
                crop_ratio = count_tiles(
                    image_width, image_height, image_size=IMAGE_SIZE
                )

            num_width_tiles, num_height_tiles = crop_ratio
        else:
            num_width_tiles = num_height_tiles = 1

        h = w = math.ceil((base_size // patch_size) / downsample_ratio)

        h2 = w2 = math.ceil((image_size // patch_size) / downsample_ratio)

        global_views_tokens = h * (w + 1)
        if num_width_tiles > 1 or num_height_tiles > 1:
            local_views_tokens = (num_height_tiles * h2) * (num_width_tiles * w2 + 1)
        else:
            local_views_tokens = 0

        return global_views_tokens + local_views_tokens + 1

    def get_image_size_with_most_features(self) -> ImageSize:
        if IMAGE_SIZE == 1024 and BASE_SIZE == 1280:
            return ImageSize(width=1024 * 2, height=1024 * 2)
        return ImageSize(width=640 * 2, height=640 * 2)


class DeepseekOCRDummyInputsBuilder(BaseDummyInputsBuilder[DeepseekOCRProcessingInfo]):
    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        num_images = mm_counts.get("image", 0)

        processor = self.info.get_hf_processor()
        image_token = processor.image_token

        return image_token * num_images

    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        mm_options: Mapping[str, BaseDummyOptions],
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
    BaseMultiModalProcessor[DeepseekOCRProcessingInfo]
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
    info=DeepseekOCRProcessingInfo,
    dummy_inputs=DeepseekOCRDummyInputsBuilder,
)
class DeepseekOCRForCausalLM(
    nn.Module, SupportsMultiModal, SupportsPP, SupportsLoRA, SupportsEncoderCudaGraph
):
    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_prefix={
            # map prefix for language backbone
            "model.embed_tokens.": "language_model.model.embed_tokens.",
            "model.layers.": "language_model.model.layers.",
            "model.norm.": "language_model.model.norm.",
            "lm_head.": "language_model.lm_head.",
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
        self.model_config = vllm_config.model_config
        self.multimodal_config = multimodal_config

        self.vision_config = config.vision_config
        self.projector_config = config.projector_config
        self.text_config = config.text_config

        model_config = vllm_config.model_config
        tokenizer = cached_tokenizer_from_config(model_config)
        self.image_token_id = tokenizer.vocab[_IMAGE_TOKEN]

        with self._mark_tower_model(vllm_config, "image"):
            self.sam_model = build_sam_vit_b()
            clip_vision_config = CLIPVisionConfig(
                hidden_size=1024,
                intermediate_size=4096,
                num_attention_heads=16,
                num_hidden_layers=24,
                image_size=224,
                patch_size=14,
                projection_dim=512,
                layer_norm_eps=1e-5,
            )
            self.vision_model = DeepCLIPVisionTransformer(
                config=clip_vision_config,
                quant_config=quant_config,
                prefix=maybe_prefix(prefix, "vision_model"),
            )

            self.projector = MlpProjector(self.projector_config)
            self.tile_tag = config.tile_tag
            self.global_view_pos = config.global_view_pos

            # special token for image token sequence format
            n_embed = self.projector_config.n_embed
            embed_std = 1 / torch.sqrt(torch.tensor(n_embed, dtype=torch.float32))
            if self.tile_tag == "2D":
                # <|view_separator|>, <|\n|>
                self.image_newline = nn.Parameter(torch.randn(n_embed) * embed_std)
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

        # Use actual tensor spatial dim instead of hardcoded
        # vision_config.image_size (1024). The vision encoders (SAM & CLIP)
        # support arbitrary resolutions via pos-encoding interpolation,
        # so Tiny/Small/Base/Large variants all work with the same weights.
        base_size = pixel_values.shape[-1]
        image_size = images_crop.shape[-1] if images_crop is not None else base_size

        return DeepseekOCRImagePixelInputs(
            type="pixel_values",
            data=pixel_values,
            images_crop=images_crop,
            images_spatial_crop=images_spatial_crop,
            resolve_bindings={
                "base_size": base_size,
                "image_size": image_size,
            },
        )

    def _encode_global_features(self, image_tensor: torch.Tensor) -> torch.Tensor:
        global_features_1 = self.sam_model(image_tensor)
        global_features_2 = self.vision_model(image_tensor, global_features_1)
        features = torch.cat(
            (
                global_features_2[:, 1:],
                global_features_1.flatten(2).permute(0, 2, 1),
            ),
            dim=-1,
        )
        features = self.projector(features)

        _, hw, dim = features.shape
        side = int(hw**0.5)

        features = features.view(side, side, dim)
        newline = self.image_newline[None, None, :].expand(side, 1, dim)
        features = torch.cat([features, newline], dim=1)
        return features.view(-1, dim)

    def _encode_local_features(
        self, patches: torch.Tensor, crop_shape: torch.Tensor
    ) -> torch.Tensor | None:
        if torch.sum(patches).item() == 0:
            return None

        local_features_1 = self.sam_model(patches)
        local_features_2 = self.vision_model(patches, local_features_1)
        features = torch.cat(
            (
                local_features_2[:, 1:],
                local_features_1.flatten(2).permute(0, 2, 1),
            ),
            dim=-1,
        )
        features = self.projector(features)

        return self._assemble_patch_grid(features, crop_shape)

    def _assemble_patch_grid(
        self, features: torch.Tensor, crop_shape: torch.Tensor
    ) -> torch.Tensor:
        """Assemble projected patches into a 2-D tile grid with newline columns."""
        _, hw, dim = features.shape
        patch_side = int(hw**0.5)
        width_tiles = int(crop_shape[0].item())
        height_tiles = int(crop_shape[1].item())

        features = (
            features.view(height_tiles, width_tiles, patch_side, patch_side, dim)
            .permute(0, 2, 1, 3, 4)
            .reshape(height_tiles * patch_side, width_tiles * patch_side, dim)
        )
        newline = self.image_newline[None, None, :].expand(
            height_tiles * patch_side, 1, dim
        )
        features = torch.cat([features, newline], dim=1)
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
            crop_shape = images_spatial_crop[jdx]

            global_features = self._encode_global_features(image_ori)
            local_features = self._encode_local_features(patches, crop_shape)

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
        input_ids: torch.Tensor | None,
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

    # -- Fixed spatial constants (computed from BASE_SIZE / IMAGE_SIZE) --

    @property
    def image_side(self) -> int:
        """Number of output grid cells per spatial dim for a global image."""
        return math.ceil((BASE_SIZE // 16) / 4)  # 16

    @property
    def global_image_output_token(self) -> int:
        """Tokens per global image (grid + one newline per row)."""
        return self.image_side * (self.image_side + 1)  # 272

    @property
    def patch_side(self) -> int:
        """Number of output grid cells per spatial dim for a local patch."""
        return math.ceil((IMAGE_SIZE // 16) / 4)  # 10

    @property
    def single_patch_output_token(self) -> int:
        """Tokens per local patch (square grid, no newlines)."""
        return self.patch_side * self.patch_side  # 100

    # -- SupportsEncoderCudaGraph protocol methods --

    def _get_num_input_output_tokens(
        self,
        image_spatial_crop: torch.Tensor | None = None,
    ) -> tuple[int, int, int, int]:
        """
        Return (num_input_tokens, num_output_tokens, global_output_token,
        local_output_token) for a single image described by
        ``image_spatial_crop``.
        """
        is_tiled = False
        if image_spatial_crop is not None:
            is_tiled = image_spatial_crop[0] > 1 or image_spatial_crop[1] > 1

        # Compute input size:
        global_input_side = BASE_SIZE // 16  # 64
        local_input_side = IMAGE_SIZE // 16  # 40
        num_input_tokens = global_input_side**2

        if is_tiled:
            num_patches = image_spatial_crop.prod(dim=-1)
            num_input_tokens += num_patches * (local_input_side**2)

        global_output_token = self.global_image_output_token
        num_output_tokens = global_output_token

        local_output_token = 0
        if is_tiled:
            local_output_token = num_patches * self.single_patch_output_token
            num_output_tokens += local_output_token

        return (
            num_input_tokens,
            num_output_tokens,
            global_output_token,
            local_output_token,
        )

    def get_encoder_cudagraph_config(self):
        return EncoderCudaGraphConfig(
            modalities=["image"],
            buffer_keys=["pixel_values"],
            out_hidden_size=self.projector_config.n_embed,
            enable_dual_path_graph=True,
            global_token_per_image=self.global_image_output_token,
            local_token_per_patch=self.single_patch_output_token,
        )

    def get_encoder_cudagraph_budget_range(
        self,
        vllm_config,
    ) -> tuple[int, int]:
        # Min budget: at least one global image with newline tokens (without patches).
        min_budget = self.global_image_output_token
        max_budget = min(
            vllm_config.scheduler_config.max_num_batched_tokens,
            self.model_config.max_model_len,
        )
        return (min_budget, max_budget)

    def get_encoder_cudagraph_item_specs(
        self,
        mm_kwargs: dict[str, Any],
    ) -> list[EncoderItemSpec]:
        item_specs = []
        for image_spatial_crop in mm_kwargs["images_spatial_crop"]:
            (
                num_input_tokens,
                num_output_tokens,
                global_output_token,
                local_output_token,
            ) = self._get_num_input_output_tokens(image_spatial_crop)
            item_specs.append(
                EncoderItemSpec(
                    input_size=num_input_tokens,
                    output_tokens=num_output_tokens,
                    global_output_tokens=global_output_token,
                    local_output_tokens=local_output_token,
                )
            )
        return item_specs

    def select_encoder_cudagraph_items(
        self,
        mm_kwargs: dict[str, Any],
        indices: list[int],
        secondary_capture_axis_key: Hashable | None = None,
    ) -> dict[str, Any]:
        pixel_values = mm_kwargs["pixel_values"]
        images_crop = mm_kwargs["images_crop"]
        images_spatial_crop = mm_kwargs["images_spatial_crop"]

        if len(indices) == 0:
            return {
                "pixel_values": pixel_values[:0],
                "images_crop": images_crop[:0],
                "images_spatial_crop": images_spatial_crop[:0],
            }

        is_tiled = (images_spatial_crop[:, 0] > 1) | (images_spatial_crop[:, 1] > 1)
        patches_per_image = torch.where(is_tiled, images_spatial_crop.prod(dim=-1), 0)
        cum_patches = [0]
        for num_patches in patches_per_image:
            cum_patches.append(cum_patches[-1] + int(num_patches))

        selected_pv = pixel_values[indices]
        selected_ic = torch.cat(
            [images_crop[cum_patches[i] : cum_patches[i + 1]] for i in indices]
        )
        selected_sp = images_spatial_crop[indices]

        return {
            "pixel_values": selected_pv,
            "images_crop": selected_ic,
            "images_spatial_crop": selected_sp,
        }

    def prepare_encoder_cudagraph_capture_inputs(
        self,
        token_budget: int,
        max_batch_size: int,
        max_frames_per_batch: int,
        device: torch.device,
        dtype: torch.dtype,
        path: str = "default",
        secondary_capture_axis_key: Hashable | None = None,
    ):
        assert path in ("global", "local")

        if path == "global":
            max_num_images = token_budget // self.global_image_output_token
            max_batch_size = min(max_batch_size, max_num_images)
            dummy_pixel_values = torch.randn(
                max_batch_size,
                3,
                BASE_SIZE,
                BASE_SIZE,
                device=device,
                dtype=dtype,
            )
            values = {"pixel_values": dummy_pixel_values}
        else:
            max_num_patches = token_budget // self.single_patch_output_token
            dummy_images_crop = torch.randn(
                max_num_patches,
                3,
                IMAGE_SIZE,
                IMAGE_SIZE,
                device=device,
                dtype=dtype,
            )
            values = {"images_crop": dummy_images_crop}

        return EncoderCudaGraphCaptureInputs(values=values)

    def prepare_encoder_cudagraph_replay_buffers(
        self,
        mm_kwargs: dict[str, Any],
        max_batch_size: int,
        max_frames_per_batch: int,
        path: str = "default",
    ):
        assert path in ("global", "local")

        if path == "global":
            values = {"pixel_values": mm_kwargs["pixel_values"]}
        else:
            values = {"images_crop": mm_kwargs["images_crop"]}

        return EncoderCudaGraphReplayBuffers(values=values)

    def _batched_encoder_forward_global_path(
        self,
        pixel_values: torch.Tensor,
    ) -> torch.Tensor:
        """
        Encode batched global images with newline tokens inserted.
        Output shape: ``[B * 272, n_embed]``.
        """
        bsz = pixel_values.shape[0]
        global_features_1 = self.sam_model(pixel_values)
        global_features_2 = self.vision_model(pixel_values, global_features_1)
        features = torch.cat(
            (
                global_features_2[:, 1:],
                global_features_1.flatten(2).permute(0, 2, 1),
            ),
            dim=-1,
        )
        features = self.projector(features)
        side = self.image_side
        dim = features.shape[-1]
        features = features.view(bsz, side, side, dim)
        newline = self.image_newline.view(1, 1, 1, dim).expand(bsz, side, 1, dim)
        features = torch.cat([features, newline], dim=2)
        return features.view(-1, dim)

    def _batched_encoder_forward_local_path(
        self,
        images_crop: torch.Tensor,
    ) -> torch.Tensor:
        """
        Encode local patches without newline insertion (newlines are added later
        in ``postprocess_encoder_output`` via ``_assemble_patch_grid``).
        Output shape: ``[P * 100, n_embed]``.
        """
        features_1 = self.sam_model(images_crop)
        features_2 = self.vision_model(images_crop, features_1)
        features = torch.cat(
            (
                features_2[:, 1:],
                features_1.flatten(2).permute(0, 2, 1),
            ),
            dim=-1,
        )
        features = self.projector(features)
        return features.view(-1, features.shape[-1])

    def encoder_cudagraph_forward(
        self,
        values: dict[str, torch.Tensor],
        path: str = "default",
    ) -> torch.Tensor:
        assert path in ("global", "local")

        if path == "global":
            pixel_values = values["pixel_values"]
            return self._batched_encoder_forward_global_path(pixel_values)
        else:
            images_crop = values["images_crop"]
            return self._batched_encoder_forward_local_path(images_crop)

    def encoder_eager_forward(
        self,
        mm_kwargs: dict[str, Any],
        path: str = "default",
    ) -> torch.Tensor:
        """Eager encoder forward with optional per-path execution.

        ``path="default"``: full forward (global + local + assembly).
        ``path="global"``: global-only batched forward with newlines.
        ``path="local"``: local-only batched forward without newlines.
        """
        if path == "default":
            # Original eager implementation: process each image one by one
            # (with both global and local paths) and concatenate results.
            image_input = DeepseekOCRImagePixelInputs(
                type="pixel_values",
                data=mm_kwargs["pixel_values"],
                images_crop=mm_kwargs["images_crop"],
                images_spatial_crop=mm_kwargs["images_spatial_crop"],
            )
            vision_embeddings = self._process_image_input(image_input)
            return torch.cat(vision_embeddings, dim=0)

        assert path in ("global", "local")
        if path == "global":
            pixel_values = mm_kwargs["pixel_values"]
            return self._batched_encoder_forward_global_path(pixel_values)
        else:
            images_crop = mm_kwargs["images_crop"]
            return self._batched_encoder_forward_local_path(images_crop)

    def postprocess_encoder_output(
        self,
        output: torch.Tensor,
        indices: list[int],
        per_item_out_tokens: list[int],
        dest: dict[int, torch.Tensor] | list[torch.Tensor | None],
        clone: bool = False,
        batch_mm_kwargs: dict[str, Any] | None = None,
        local_output: torch.Tensor | None = None,
    ) -> None:
        """
        Assemble per-image embeddings from global and local encoder outputs.

        ``output`` contains global-image features with newlines already
        inserted (from CUDA graph replay or eager fallback):
        ``[B * 272, n_embed]``.

        ``local_output`` contains local-patch features without
        newlines (from CUDA graph replay or eager fallback):
        ``[P * 100, n_embed]``. May be ``None`` if no patches in batch.

        This method:
        1. Splits ``output`` into per-image global portions.
        2. Splits ``local_output`` into per-image patch groups.
        3. For each image: assembles patch grid with newlines via
           ``_assemble_patch_grid``, then concatenates
           ``[local_tiled, global, view_seperator]``.
        """
        bsz = len(indices)
        n_embed = output.shape[-1]

        images_spatial_crop = batch_mm_kwargs["images_spatial_crop"]
        is_tiled = (images_spatial_crop[:, 0] > 1) | (images_spatial_crop[:, 1] > 1)
        num_patches = [
            int(np) for np in torch.where(is_tiled, images_spatial_crop.prod(dim=-1), 0)
        ]
        total_patches = sum(num_patches)

        global_part = output[: bsz * self.global_image_output_token].reshape(
            bsz, self.global_image_output_token, n_embed
        )

        # Split local output into per-patch groups.
        local_flat = None
        if total_patches > 0 and local_output is not None:
            local_flat = local_output[: total_patches * self.single_patch_output_token]
            local_flat = local_flat.reshape(
                total_patches, self.single_patch_output_token, n_embed
            )

        cur_patch = 0
        for i, idx in enumerate(indices):
            num_patch = num_patches[i]
            single_image_output: list[torch.Tensor] = []

            # 1. Process local patches: assemble tile grid, add 1 newline per row.
            if num_patch > 0 and local_flat is not None:
                patches = local_flat[cur_patch : cur_patch + num_patch]
                cur_patch += num_patch
                single_image_output.append(
                    self._assemble_patch_grid(patches, images_spatial_crop[i])
                )

            # 2. Global image: newlines already inserted.
            single_image_output.append(global_part[i])

            # 3. Add view separator for each image.
            single_image_output.append(self.view_seperator[None, :])

            # 4. Save final outputs for each image.
            dest[idx] = torch.cat(single_image_output, dim=0)
