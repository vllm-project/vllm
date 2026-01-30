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
    MambaStateCopyFunc,
    MambaStateCopyFuncCalculator,
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
from .lfm2_siglip2 import Siglip2Model
from .utils import (
    AutoWeightsLoader,
    WeightsMapper,
    init_vllm_registered_model,
    maybe_prefix,
)
from .vision import is_vit_use_data_parallel


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

        parsed_images = self.data_parser.parse_mm_data({"image": images}).get_items(
            "image", ImageProcessorItems
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
        self,
        config: Lfm2VlConfig,
        prefix: str = "",
    ):
        super().__init__()
        self.use_data_parallel = is_vit_use_data_parallel()

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

    def forward(
        self,
        vision_features_packed: torch.Tensor,
        spatial_shapes: torch.Tensor,
    ) -> torch.Tensor:
        """Project packed vision features without materializing padded tensors.

        Args:
            vision_features_packed: (total_tokens, hidden_size) packed in tile order.
            spatial_shapes: (num_tiles, 2) on CPU (height, width) per tile.

        Returns:
            projected_packed: (total_projected_tokens, text_hidden_size)
        """
        assert spatial_shapes.device.type == "cpu", (
            "Expected `spatial_shapes` on CPU to avoid device-to-host sync in "
            "variable-length packing."
        )
        factor = self.factor
        device = vision_features_packed.device
        hidden_size = vision_features_packed.shape[-1]

        spatial_shapes_list: list[list[int]] = spatial_shapes.tolist()
        lengths_list = [h * w for h, w in spatial_shapes_list]

        gather_idx_parts: list[torch.Tensor] = []
        offset = 0

        dh = torch.arange(factor, dtype=torch.int64)
        dw = torch.arange(factor, dtype=torch.int64)
        dh_grid, dw_grid = torch.meshgrid(dh, dw, indexing="ij")
        dh_flat = dh_grid.reshape(-1)
        dw_flat = dw_grid.reshape(-1)

        for (height, width), length in zip(spatial_shapes_list, lengths_list):
            if length <= 0:
                continue
            if height % factor != 0 or width % factor != 0:
                raise ValueError(
                    "spatial_shapes must be divisible by downsample_factor: "
                    f"got ({height}, {width}) with factor={factor}."
                )
            height_out = height // factor
            width_out = width // factor

            rows_out = torch.arange(height_out, dtype=torch.int64)
            cols_out = torch.arange(width_out, dtype=torch.int64)
            rr, cc = torch.meshgrid(rows_out, cols_out, indexing="ij")
            rr = rr.reshape(-1)
            cc = cc.reshape(-1)

            token_idx = (rr[:, None] * factor + dh_flat[None, :]) * width + (
                cc[:, None] * factor + dw_flat[None, :]
            )
            gather_idx_parts.append(token_idx.reshape(-1) + offset)
            offset += length

        if gather_idx_parts:
            gather_idx = torch.cat(gather_idx_parts).to(device=device)
            gathered = vision_features_packed.index_select(0, gather_idx)
            unshuffled = gathered.reshape(-1, factor * factor * hidden_size)
        else:
            unshuffled = vision_features_packed.new_empty(
                (0, factor * factor * hidden_size)
            )

        if self.projector_use_layernorm:
            unshuffled = self.layer_norm(unshuffled)
        hidden_states = self.linear_1(unshuffled)
        hidden_states = self.act(hidden_states)
        projected_packed = self.linear_2(hidden_states)
        return projected_packed


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

    @classmethod
    def get_mamba_state_copy_func(cls) -> tuple[MambaStateCopyFunc]:
        return MambaStateCopyFuncCalculator.short_conv_state_copy_func()

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

        with self._mark_tower_model(vllm_config, "image"):
            if vision_config.model_type == "siglip2_vision_model":
                self.vision_tower = Siglip2Model(
                    config=vision_config,
                    quant_config=quant_config,
                    prefix=maybe_prefix(prefix, "vision_tower"),
                )
            else:
                raise ValueError(
                    f"Unsupported visual tokenizer type: {vision_config.model_type}"
                )

            self.multi_modal_projector = Lfm2VLMultiModalProjector(
                config=config,
                prefix=maybe_prefix(prefix, "multi_modal_projector"),
            )

        with self._mark_language_model(vllm_config):
            self.language_model = init_vllm_registered_model(
                vllm_config=vllm_config,
                hf_config=config.text_config,
                prefix=maybe_prefix(prefix, "language"),
                architectures=config.text_config.architectures,
            )

        self.make_empty_intermediate_tensors = (
            self.language_model.make_empty_intermediate_tensors
        )

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
        assert spatial_shapes.device.type == "cpu", (
            "Expected `spatial_shapes` on CPU to avoid device-to-host sync in "
            "variable-length packing."
        )

        pixel_values = pixel_values.to(
            dtype=self.vision_tower.vision_model.embeddings.patch_embedding.weight.dtype
        )  # fp16 compatibility

        # LFM2-VL's HF processor pads patch sequences with trailing zeros.
        # Pack patch tokens upfront so the vision tower runs entirely unpadded.
        spatial_shapes_list: list[list[int]] = spatial_shapes.tolist()
        lengths_list = [h * w for h, w in spatial_shapes_list]
        total_tokens = int(sum(lengths_list))
        lengths_cpu = (spatial_shapes[:, 0] * spatial_shapes[:, 1]).to(
            dtype=torch.int32
        )
        max_seqlen = (
            lengths_cpu.max().reshape(1)
            if lengths_cpu.numel()
            else torch.tensor([0], dtype=torch.int32)
        )

        if total_tokens == 0:
            return []

        packed_pixel_values = pixel_values.new_empty(
            (total_tokens, pixel_values.shape[-1])
        )
        offset = 0
        for i, length in enumerate(lengths_list):
            if length <= 0:
                continue
            packed_pixel_values[offset : offset + length].copy_(
                pixel_values[i, :length]
            )
            offset += length
        packed_pixel_values = packed_pixel_values.unsqueeze(0)

        lengths = torch.tensor(
            lengths_list, dtype=torch.int32, device=pixel_values.device
        )
        cu_seqlens = torch.zeros(
            lengths.shape[0] + 1,
            dtype=torch.int32,
            device=pixel_values.device,
        )
        cu_seqlens[1:] = torch.cumsum(lengths, dim=0)

        with set_forward_context(None, self.vllm_config):
            vision_outputs = self.vision_tower(
                pixel_values_packed=packed_pixel_values,
                spatial_shapes=spatial_shapes,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
            )
        image_outputs_packed = getattr(
            vision_outputs, "last_hidden_state", vision_outputs
        )
        vision_features_packed = image_outputs_packed[0]

        factor = self.multi_modal_projector.factor
        projected_lengths_list: list[int] = []
        for (height, width), length in zip(spatial_shapes_list, lengths_list):
            if length <= 0:
                projected_lengths_list.append(0)
                continue
            if height % factor != 0 or width % factor != 0:
                raise ValueError(
                    "spatial_shapes must be divisible by downsample_factor: "
                    f"got ({height}, {width}) with factor={factor}."
                )
            projected_lengths_list.append((height // factor) * (width // factor))

        projected_packed = self.multi_modal_projector(
            vision_features_packed=vision_features_packed,
            spatial_shapes=spatial_shapes,
        )

        image_features: list[torch.Tensor] = []
        offset = 0
        for out_len in projected_lengths_list:
            image_features.append(projected_packed[offset : offset + out_len])
            offset += out_len

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
        input_ids: torch.Tensor | None,
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
        return self.language_model.compute_logits(hidden_states)

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
