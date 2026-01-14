# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import itertools
import math
from collections.abc import Iterable, Mapping, Sequence
from typing import Annotated, Literal

import torch
import torch.nn as nn
from transformers import BatchFeature
from transformers.activations import ACT2FN
from transformers.models.lfm2_vl import Lfm2VlProcessor
from transformers.models.lfm2_vl.configuration_lfm2_vl import Lfm2VlConfig
from transformers.models.lfm2_vl.image_processing_lfm2_vl_fast import (
    Lfm2VlImageProcessorFast,
    find_closest_aspect_ratio,
    round_by_factor,
)

from vllm.config import VllmConfig
from vllm.config.multimodal import BaseDummyOptions
from vllm.forward_context import set_forward_context
from vllm.model_executor.layers.mamba.mamba_utils import (
    MambaStateDtypeCalculator,
    MambaStateShapeCalculator,
)
from vllm.model_executor.models.module_mapping import MultiModelKeys
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import (
    MultiModalDataDict,
    MultiModalFieldConfig,
    MultiModalKwargsItems,
)
from vllm.multimodal.parse import ImageProcessorItems, ImageSize, MultiModalDataItems
from vllm.multimodal.processing import (
    BaseDummyInputsBuilder,
    BaseMultiModalProcessor,
    BaseProcessingInfo,
    PromptReplacement,
    PromptUpdateDetails,
)
from vllm.sequence import IntermediateTensors
from vllm.utils.tensor_schema import TensorSchema, TensorShape

from .interfaces import (
    IsHybrid,
    MultiModalEmbeddings,
    SupportsLoRA,
    SupportsMultiModal,
    SupportsPP,
)
from .siglip2 import Siglip2Model
from .utils import (
    AutoWeightsLoader,
    WeightsMapper,
    init_vllm_registered_model,
    maybe_prefix,
)


class Lfm2VLImagePixelInputs(TensorSchema):
    """
    Dimensions:
        - b: Number of images in the prompt
        - bn: Batch size * number of images
        - d: Number of dimensions
        - fd: Number of features per dimension
    """

    type: Literal["pixel_values"] = "pixel_values"
    pixel_values: Annotated[torch.Tensor, TensorShape("bn", "d", "fd")]
    spatial_shapes: Annotated[torch.Tensor, TensorShape("bn", 2)]
    num_patches: Annotated[torch.Tensor, TensorShape("b")]


LFM2VLImageInputs = Lfm2VLImagePixelInputs


class Lfm2VLProcessingInfo(BaseProcessingInfo):
    def get_hf_config(self):
        return self.ctx.get_hf_config(Lfm2VlConfig)

    def get_hf_processor(self, **kwargs):
        return self.ctx.get_hf_processor(Lfm2VlProcessor, **kwargs)

    def get_image_processor(self, **kwargs: object) -> Lfm2VlImageProcessorFast:
        return self.get_hf_processor(**kwargs).image_processor

    def get_supported_mm_limits(self) -> Mapping[str, int | None]:
        return {"image": None}

    def get_image_size_with_most_features(self) -> ImageSize:
        processor = self.get_image_processor()
        max_image_tokens = processor.max_image_tokens
        encoder_patch_size = processor.encoder_patch_size
        downsample_factor = processor.downsample_factor
        max_pixels = max_image_tokens * (encoder_patch_size**2) * (downsample_factor**2)
        side = int(math.sqrt(max_pixels))
        return ImageSize(width=side, height=side)

    def _is_image_too_large(
        self,
        height: int,
        width: int,
        max_image_tokens: int,
        encoder_patch_size: int,
        downsample_factor: int,
        max_pixels_tolerance: float,
    ) -> bool:
        """Check if the image is too large to be processed as one tile."""
        total_factor = encoder_patch_size * downsample_factor

        h_bar = max(encoder_patch_size, round_by_factor(height, total_factor))
        w_bar = max(encoder_patch_size, round_by_factor(width, total_factor))
        return (
            h_bar * w_bar
            > max_image_tokens
            * encoder_patch_size**2
            * downsample_factor**2
            * max_pixels_tolerance
        )

    def smart_resize(
        self,
        height: int,
        width: int,
        downsample_factor: int,
        min_image_tokens: int,
        max_image_tokens: int,
        encoder_patch_size: int,
    ) -> tuple[int, int]:
        total_factor = encoder_patch_size * downsample_factor
        smart_resize_min_pixels = (
            min_image_tokens * encoder_patch_size**2 * downsample_factor**2
        )
        smart_resize_max_pixels = (
            max_image_tokens * encoder_patch_size**2 * downsample_factor**2
        )

        h_bar = max(total_factor, round_by_factor(height, total_factor))
        w_bar = max(total_factor, round_by_factor(width, total_factor))

        if h_bar * w_bar > smart_resize_max_pixels:
            beta = math.sqrt((height * width) / smart_resize_max_pixels)
            h_bar = max(
                total_factor, math.floor(height / beta / total_factor) * total_factor
            )
            w_bar = max(
                total_factor, math.floor(width / beta / total_factor) * total_factor
            )
        elif h_bar * w_bar < smart_resize_min_pixels:
            beta = math.sqrt(smart_resize_min_pixels / (height * width))
            h_bar = math.ceil(height * beta / total_factor) * total_factor
            w_bar = math.ceil(width * beta / total_factor) * total_factor

        return w_bar, h_bar

    def _target_ratios(self, min_tiles: int, max_tiles: int) -> list[tuple[int, int]]:
        ratios = [
            (w, h)
            for n in range(min_tiles, max_tiles + 1)
            for w in range(1, n + 1)
            for h in range(1, n + 1)
            if min_tiles <= w * h <= max_tiles
        ]
        return sorted(set(ratios), key=lambda x: x[0] * x[1])

    def _get_grid_layout(
        self,
        height: int,
        width: int,
        min_tiles: int,
        max_tiles: int,
        tile_size: int,
    ) -> tuple[int, int]:
        aspect_ratio = width / height
        target_ratios = self._target_ratios(min_tiles, max_tiles)
        # find best matching grid configuration
        grid_width, grid_height = find_closest_aspect_ratio(
            aspect_ratio, target_ratios, width, height, tile_size
        )
        total_patches = grid_width * grid_height
        return grid_width, grid_height, total_patches

    def _get_image_feature_grid_size(
        self,
        image_width: int,
        image_height: int,
        processor: Lfm2VlProcessor | None,
    ) -> tuple[int, int]:
        if processor is None:
            processor = self.get_image_processor()

        downsample_factor = processor.image_processor.downsample_factor
        encoder_patch_size = processor.image_processor.encoder_patch_size
        max_pixels_tolerance = processor.image_processor.max_pixels_tolerance
        min_tiles = processor.image_processor.min_tiles
        max_tiles = processor.image_processor.max_tiles
        max_image_tokens = processor.image_processor.max_image_tokens
        tile_size = processor.image_processor.tile_size

        do_image_splitting = not min_tiles == max_tiles == 1
        is_image_large = self._is_image_too_large(
            height=image_height,
            width=image_width,
            max_image_tokens=max_image_tokens,
            encoder_patch_size=encoder_patch_size,
            downsample_factor=downsample_factor,
            max_pixels_tolerance=max_pixels_tolerance,
        )

        # Big image will be cropped into patches and small images are just resized
        if is_image_large and do_image_splitting:
            grid_width, grid_height, total_patches = self._get_grid_layout(
                image_height,
                image_width,
                min_tiles=min_tiles,
                max_tiles=max_tiles,
                tile_size=tile_size,
            )
        else:
            grid_width = grid_height = total_patches = 1

        if grid_width * grid_height != 1:  # Thumbnail
            total_patches += 1

        return grid_width, grid_height, total_patches

    def get_num_patches(
        self,
        *,
        image_width: int,
        image_height: int,
        processor: Lfm2VlProcessor | None,
    ) -> int:
        _, _, total_patches = self._get_image_feature_grid_size(
            image_width=image_width,
            image_height=image_height,
            processor=processor,
        )
        return total_patches

    def get_image_repl(
        self,
        image_width: int,
        image_height: int,
        spatial_shapes: torch.Tensor,
        processor: Lfm2VlProcessor | None,
    ) -> str:
        if processor is None:
            processor = self.get_hf_processor()

        grid_placeholder = "<|img_row_{n_h}_col_{n_w}|>"
        image_token = processor.image_token
        image_start_token = processor.image_start_token
        image_end_token = processor.image_end_token
        image_thumbnail_token = processor.image_thumbnail_token

        num_thumbnail_tokens, num_tokens_per_tile = self.get_num_image_tokens(
            spatial_shapes=spatial_shapes,
            processor=processor,
        )
        tile_img_placeholder = grid_placeholder + (image_token * num_tokens_per_tile)

        grid_w, grid_h, _ = self._get_image_feature_grid_size(
            image_width=image_width,
            image_height=image_height,
            processor=processor,
        )

        if grid_w > 1 or grid_h > 1:
            tiles_placeholder: list[str] = [
                tile_img_placeholder.format(n_h=i + 1, n_w=j + 1)
                for i in range(grid_h)
                for j in range(grid_w)
            ]

            if num_thumbnail_tokens > 0:
                tiles_placeholder.append(
                    image_thumbnail_token + (image_token * num_thumbnail_tokens)
                )
        else:
            tiles_placeholder = [image_token * num_thumbnail_tokens]

        placeholder = "".join(
            itertools.chain([image_start_token], tiles_placeholder, [image_end_token])
        )
        return placeholder

    def get_num_image_tokens(
        self,
        *,
        spatial_shapes: torch.Tensor,
        processor: Lfm2VlProcessor | None,
    ) -> tuple[int, int]:
        tile_size = processor.image_processor.tile_size
        downsample_factor = processor.image_processor.downsample_factor
        encoder_patch_size = processor.image_processor.encoder_patch_size
        num_thumbnail_tokens = spatial_shapes[-1].prod() // (downsample_factor**2)
        num_patches_tile = tile_size // encoder_patch_size
        dwn_num_patches_tile = math.ceil(num_patches_tile / downsample_factor)
        num_tiles_tokens = dwn_num_patches_tile * dwn_num_patches_tile
        return num_thumbnail_tokens, num_tiles_tokens


class Lfm2VLDummyInputsBuilder(BaseDummyInputsBuilder[Lfm2VLProcessingInfo]):
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

        target_width, target_height = self.info.get_image_size_with_most_features()

        image_overrides = mm_options.get("image") if mm_options else None

        return {
            "image": self._get_dummy_images(
                width=target_width,
                height=target_height,
                num_images=num_images,
                overrides=image_overrides,
            ),
        }


class Lfm2VLMultiModalProcessor(BaseMultiModalProcessor[Lfm2VLProcessingInfo]):
    def _call_hf_processor(
        self,
        prompt: str,
        mm_data: Mapping[str, object],
        mm_kwargs: Mapping[str, object],
        tok_kwargs: Mapping[str, object],
    ) -> BatchFeature:
        # Text-only input not supported in composite processor
        if not (images := mm_data.get("images", [])):
            prompt_ids = self.info.get_tokenizer().encode(prompt)
            prompt_ids = self._apply_hf_processor_tokens_only(prompt_ids)
            return BatchFeature(dict(input_ids=[prompt_ids]), tensor_type="pt")

        processed_outputs = super()._call_hf_processor(
            prompt,
            mm_data,
            mm_kwargs,
            tok_kwargs,
        )

        parsed_images = (
            self._get_data_parser()
            .parse_mm_data({"image": images})
            .get_items("image", ImageProcessorItems)
        )
        image_sizes = [
            parsed_images.get_image_size(i) for i in range(len(parsed_images))
        ]
        hf_processor = self.info.get_hf_processor(**mm_kwargs)

        num_patches = [
            self.info.get_num_patches(
                image_width=size.width,
                image_height=size.height,
                processor=hf_processor,
            )
            for size in image_sizes
        ]
        processed_outputs["num_patches"] = torch.tensor(num_patches)

        return processed_outputs

    def _get_mm_fields_config(
        self,
        hf_inputs: BatchFeature,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        num_patches = hf_inputs.get("num_patches", torch.empty(0))

        return dict[str, MultiModalFieldConfig](
            pixel_values=MultiModalFieldConfig.flat_from_sizes("image", num_patches),
            spatial_shapes=MultiModalFieldConfig.flat_from_sizes(
                "image", num_patches, keep_on_cpu=True
            ),
            num_patches=MultiModalFieldConfig.batched("image", keep_on_cpu=True),
        )

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        out_mm_kwargs: MultiModalKwargsItems,
    ) -> Sequence[PromptReplacement]:
        hf_processor = self.info.get_hf_processor(**hf_processor_mm_kwargs)
        image_token = hf_processor.image_token

        def get_image_replacement_lfm2vl(item_idx: int):
            images = mm_items.get_items("image", ImageProcessorItems)
            image_size = images.get_image_size(item_idx)
            out_item = out_mm_kwargs["image"][item_idx]
            spatial_shapes = out_item["spatial_shapes"].data
            assert isinstance(spatial_shapes, torch.Tensor)
            image_repl = self.info.get_image_repl(
                image_width=image_size.width,
                image_height=image_size.height,
                spatial_shapes=spatial_shapes,
                processor=hf_processor,
            )
            return PromptUpdateDetails.select_text(
                image_repl,
                embed_text=image_token,
            )

        return [
            PromptReplacement(
                modality="image",
                target=image_token,
                replacement=get_image_replacement_lfm2vl,
            )
        ]


class Lfm2VLMultiModalProjector(nn.Module):
    def __init__(
        self, config: Lfm2VlConfig, use_data_parallel: bool = False, prefix: str = ""
    ):
        super().__init__()
        self.use_data_parallel = use_data_parallel

        in_channels = config.vision_config.hidden_size * (config.downsample_factor**2)
        self.factor = config.downsample_factor
        self.projector_use_layernorm = config.projector_use_layernorm
        if self.projector_use_layernorm:
            self.layer_norm = nn.LayerNorm(in_channels)
        self.linear_1 = nn.Linear(
            in_channels,
            config.projector_hidden_size,
            bias=config.projector_bias,
        )
        self.act = ACT2FN[config.projector_hidden_act]
        self.linear_2 = nn.Linear(
            config.projector_hidden_size,
            config.text_config.hidden_size,
            bias=config.projector_bias,
        )

    def forward(self, image_features: torch.Tensor):
        image_features = self.pixel_unshuffle(image_features)
        if self.projector_use_layernorm:
            image_features = self.layer_norm(image_features)
        hidden_states = self.linear_1(image_features)
        hidden_states = self.act(hidden_states)
        hidden_states = self.linear_2(hidden_states)
        return hidden_states

    def pixel_unshuffle(self, hidden_states: torch.Tensor):
        batch_size, width, height, channels = hidden_states.size()
        hidden_states = hidden_states.reshape(
            batch_size, width, height // self.factor, channels * self.factor
        )
        hidden_states = hidden_states.permute(0, 2, 1, 3)
        hidden_states = hidden_states.reshape(
            batch_size,
            height // self.factor,
            width // self.factor,
            channels * self.factor**2,
        )
        hidden_states = hidden_states.permute(0, 2, 1, 3)
        return hidden_states


@MULTIMODAL_REGISTRY.register_processor(
    Lfm2VLMultiModalProcessor,
    info=Lfm2VLProcessingInfo,
    dummy_inputs=Lfm2VLDummyInputsBuilder,
)
class Lfm2VLForConditionalGeneration(
    nn.Module, SupportsMultiModal, SupportsLoRA, SupportsPP, IsHybrid
):
    merge_by_field_config = True

    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_prefix={
            "lm_head.": "language_model.lm_head.",
            "model.language_model.": "language_model.model.",
            "model.vision_tower.": "vision_tower.",
            "model.multi_modal_projector.": "multi_modal_projector.",
        }
    )

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None:
        if modality.startswith("image"):
            return "<image>"

        raise ValueError("Only image modality is supported")

    @classmethod
    def get_mamba_state_dtype_from_config(
        cls,
        vllm_config: "VllmConfig",
    ) -> tuple[torch.dtype, ...]:
        return MambaStateDtypeCalculator.short_conv_state_dtype(
            vllm_config.model_config.dtype,
            vllm_config.cache_config.mamba_cache_dtype,
        )

    @classmethod
    def get_mamba_state_shape_from_config(
        cls,
        vllm_config: "VllmConfig",
    ) -> tuple[tuple[int, int]]:
        """Calculate shapes for LFM2's convolutional cache.

        Args:
            vllm_config: vLLM config

        Returns:
            Tuple containing:
            - conv_state_shape: Shape for convolutional state cache
        """
        parallel_config = vllm_config.parallel_config
        hf_language_config = vllm_config.model_config.hf_config.text_config

        return MambaStateShapeCalculator.short_conv_state_shape(
            tp_world_size=parallel_config.tensor_parallel_size,
            intermediate_size=hf_language_config.hidden_size,
            conv_kernel=hf_language_config.conv_L_cache,
        )

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "model"):
        super().__init__()
        config: Lfm2VlConfig = vllm_config.model_config.hf_config
        multimodal_config = vllm_config.model_config.multimodal_config
        vision_config = config.vision_config
        quant_config = vllm_config.quant_config

        self.config = config
        self.vllm_config = vllm_config
        self.multimodal_config = multimodal_config
        self.use_data_parallel = multimodal_config.mm_encoder_tp_mode == "data"

        if vision_config.model_type == "siglip2_vision_model":
            self.vision_tower = Siglip2Model(
                config=vision_config,
                quant_config=quant_config,
                multimodal_config=multimodal_config,
                prefix=maybe_prefix(prefix, "vision_tower"),
            )
        else:
            raise ValueError(
                f"Unsupported visual tokenizer model_type: {vision_config.model_type}"
            )

        self.multi_modal_projector = Lfm2VLMultiModalProjector(
            config=config,
            use_data_parallel=self.use_data_parallel,
            prefix=f"{prefix}.multi_modal_projector",
        )

        self.language_model = init_vllm_registered_model(
            vllm_config=vllm_config,
            hf_config=config.text_config,
            prefix=maybe_prefix(prefix, "language"),
            architectures=config.text_config.architectures,
        )

        self.make_empty_intermediate_tensors = (
            self.language_model.make_empty_intermediate_tensors
        )

    def get_language_model(self) -> torch.nn.Module:
        return self.language_model

    def _parse_and_validate_image_input(
        self, **kwargs: object
    ) -> LFM2VLImageInputs | None:
        pixel_values = kwargs.pop("pixel_values", None)
        spatial_shapes = kwargs.pop("spatial_shapes", None)
        num_patches = kwargs.pop("num_patches", None)
        if pixel_values is None:
            return None

        return LFM2VLImageInputs(
            type="pixel_values",
            pixel_values=pixel_values,
            spatial_shapes=spatial_shapes,
            num_patches=num_patches,
        )

    def image_pixels_to_features(
        self,
        pixel_values: torch.FloatTensor,
        spatial_shapes: torch.Tensor,
    ) -> torch.Tensor:
        pixel_values = pixel_values.to(
            dtype=self.vision_tower.vision_model.embeddings.patch_embedding.weight.dtype
        )  # fp16 compatibility

        # LFM2-VL's HF processor pads patch sequences with trailing zeros.
        # Derive the valid-patch mask from spatial_shapes instead of carrying
        # pixel_attention_mask through the vLLM multimodal pipeline.
        max_seq_len = pixel_values.shape[1]
        lengths_cpu = (spatial_shapes[:, 0] * spatial_shapes[:, 1]).to(
            dtype=torch.int32
        )
        max_seqlen = (
            lengths_cpu.max().reshape(1).to(device=pixel_values.device)
            if lengths_cpu.numel()
            else torch.tensor([0], dtype=torch.int32, device=pixel_values.device)
        )
        lengths = lengths_cpu.to(device=pixel_values.device)
        packed_mask = (
            torch.arange(max_seq_len, device=pixel_values.device)[None, :]
            < lengths[:, None]
        )
        cu_seqlens = torch.zeros(
            lengths.shape[0] + 1,
            dtype=torch.int32,
            device=lengths.device,
        )
        cu_seqlens[1:] = torch.cumsum(lengths, dim=0)

        with set_forward_context(None, self.vllm_config):
            vision_outputs = self.vision_tower(
                pixel_values=pixel_values,
                spatial_shapes=spatial_shapes,
                packed_mask=packed_mask,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
            )
        image_outputs = getattr(vision_outputs, "last_hidden_state", vision_outputs)

        image_features = []

        # spatial_shapes is on CPU (keep_on_cpu=True), so .tolist() is instant
        spatial_shapes_list = spatial_shapes.tolist()
        for img_idx, (feature_org_h, feature_org_w) in enumerate(spatial_shapes_list):
            feature_len = feature_org_h * feature_org_w
            feature = image_outputs[img_idx, :feature_len]

            # reshape to original height and width
            feature = feature.reshape(1, feature_org_h, feature_org_w, -1)

            # project the image representation
            img_embedding = self.multi_modal_projector(feature)

            # flatten here to handle variable length in naflex
            img_embedding = img_embedding.reshape(-1, img_embedding.size(-1))
            image_features.append(img_embedding)

        return image_features

    def _process_image_input(
        self,
        image_input: LFM2VLImageInputs,
    ) -> torch.Tensor | list[torch.Tensor]:
        pixel_values = image_input["pixel_values"]
        spatial_shapes = image_input["spatial_shapes"]
        num_patches = image_input["num_patches"]

        image_features = self.image_pixels_to_features(
            pixel_values,
            spatial_shapes=spatial_shapes,
        )

        # Group patches by image - num_patches is on CPU (keep_on_cpu=True)
        # so .tolist() is instant with no DtoH sync
        num_patches_list = num_patches.tolist()
        batched_features: list[torch.Tensor] = []
        patch_idx = 0
        for count in num_patches_list:
            # Slice the list of patch tensors for this image
            image_patches = image_features[patch_idx : patch_idx + count]
            # Concatenate patches for this image
            batched_features.append(torch.cat(image_patches, dim=0))
            patch_idx += count

        return batched_features

    def embed_multimodal(self, **kwargs: object) -> MultiModalEmbeddings:
        image_input = self._parse_and_validate_image_input(**kwargs)
        if image_input is None:
            return []

        return self._process_image_input(image_input)

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

        hidden_states = self.language_model(
            input_ids=input_ids,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
        )
        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor | None:
        logits = self.language_model.compute_logits(hidden_states)
        return logits

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)

    def get_mm_mapping(self) -> MultiModelKeys:
        """
        Get the module prefix in multimodal models
        """
        return MultiModelKeys.from_string_field(
            language_model="language_model",
            connector="multi_modal_projector",
            tower_model="vision_tower",
        )
