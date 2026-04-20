# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""vLLM implementation of Granite 4 Vision.

Uses GraniteForCausalLM as the language backbone with SigLIP vision encoder
and deepstack feature injection via WindowQFormer projectors.

LoRA support:
- Full merge (--hf-overrides '{"adapter_path": "..."}') merges LM-only LoRA
  deltas into base weights at load time.
- Native LoRA (--enable-lora --default-mm-loras) lets vLLM runtime serve
  LM LoRA deltas per-request.
Both modes expect a LM-only adapter (no modules_to_save).
"""

import json
import math
import os
from collections.abc import Iterable, Mapping
from fractions import Fraction

import torch
import torch.nn as nn
from safetensors.torch import load_file
from transformers import BatchFeature
from transformers.models.blip_2.configuration_blip_2 import Blip2QFormerConfig

from .blip2 import Blip2QFormerModel
from transformers.models.llava_next.modeling_llava_next import (
    get_anyres_image_grid_shape,
    image_size_to_num_patches,
    unpad_image,
)

from vllm.config import CacheConfig, VllmConfig
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.distributed.parallel_state import (
    get_pp_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from vllm.logger import init_logger
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import MultiModalFieldConfig
from vllm.sequence import IntermediateTensors

from vllm.model_executor.models.interfaces import (
    MultiModalEmbeddings,
    SupportsLoRA,
    SupportsMultiModal,
    SupportsPP,
)
from vllm.model_executor.models.llava import (
    LlavaDummyInputsBuilder,
    init_vision_tower_for_llava,
)
from vllm.model_executor.models.siglip import SiglipVisionModel
from vllm.model_executor.models.llava_next import (
    BaseLlavaNextMultiModalProcessor,
    LlavaNextProcessingInfo,
    LlavaNextImagePixelInputs,
    LlavaNextImageEmbeddingInputs,
    LlavaNextImageInputs,
)
from vllm.model_executor.models.module_mapping import MultiModelKeys
from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    WeightsMapper,
    _merge_multimodal_embeddings,
    init_vllm_registered_model,
    maybe_prefix,
)

logger = init_logger(__name__)


# ---------------------------------------------------------------------------
# Downsampler modules (translated from HF downsampling.py)
# ---------------------------------------------------------------------------

class InterpolateDownsampler:
    """Spatial downsampling via area interpolation."""

    def __init__(self, config, mode="area"):
        self.orig_image_side = (
            config.vision_config.image_size // config.vision_config.patch_size
        )
        self.new_image_side = int(
            self.orig_image_side * Fraction(config.downsample_rate)
        )
        self.mode = mode

    def __call__(self, image_features: torch.Tensor) -> torch.Tensor:
        batch_size, _, dim = image_features.size()
        up_shape = [batch_size, self.orig_image_side, self.orig_image_side, dim]
        large = image_features.view(up_shape).permute(0, 3, 1, 2)
        small = torch.nn.functional.interpolate(
            large,
            size=(self.new_image_side, self.new_image_side),
            mode=self.mode,
        )
        return small.permute(0, 2, 3, 1).flatten(1, 2)


class SpatialOffsetDownsampler:
    """Sample one position from each 2x2 block (offset 0-3 = TL/TR/BL/BR)."""

    def __init__(self, config, offset: int = 0):
        self.orig_image_side = (
            config.vision_config.image_size // config.vision_config.patch_size
        )
        self.new_image_side = self.orig_image_side // 2
        offsets = [(0, 0), (0, 1), (1, 0), (1, 1)]
        self.offset_h, self.offset_w = offsets[offset]

    def __call__(self, image_features: torch.Tensor) -> torch.Tensor:
        B, _, C = image_features.shape
        features_2d = image_features.reshape(
            B, self.orig_image_side, self.orig_image_side, C
        )
        n = self.new_image_side
        blocks = features_2d.reshape(B, n, 2, n, 2, C)
        sampled = blocks[:, :, self.offset_h, :, self.offset_w, :]
        return sampled.reshape(B, -1, C)


class WindowQFormerDownsampler(nn.Module):
    """Window-based QFormer downsampler (matches HF downsampling.py exactly)."""

    def __init__(
        self,
        config,
        quant_config: QuantizationConfig | None = None,
        cache_config: CacheConfig | None = None,
        spatial_offset: int | None = None,
        prefix: str = "",
    ):
        super().__init__()
        llm_hidden_size = config.text_config.hidden_size
        vision_hidden_size = config.vision_config.hidden_size

        self.dropout = nn.Dropout(config.projector_dropout)

        if spatial_offset is not None:
            self.downsampler = SpatialOffsetDownsampler(config, offset=spatial_offset)
        else:
            self.downsampler = InterpolateDownsampler(config)

        qformer_config = Blip2QFormerConfig(
            hidden_size=vision_hidden_size,
            num_attention_heads=vision_hidden_size // 64,
            intermediate_size=3072,
            num_hidden_layers=1,
            encoder_hidden_size=vision_hidden_size,
            cross_attention_frequency=1,
            max_position_embeddings=2048,
            use_qformer_text_input=False,
        )
        self.qformer = Blip2QFormerModel(
            qformer_config,
            quant_config=quant_config,
            cache_config=cache_config,
            prefix=f"{prefix}.qformer",
        )

        self.image_side = (
            config.vision_config.image_size // config.vision_config.patch_size
        )
        q, w = config.downsample_rate.split("/")
        self.query_side, self.window_side = int(q), int(w)
        self.query_length = self.query_side**2

        embed_std = 1 / math.sqrt(vision_hidden_size)
        self.norm = nn.LayerNorm(vision_hidden_size, eps=1e-6)
        self.query = nn.Parameter(
            torch.randn(1, self.query_length, vision_hidden_size) * embed_std
        )
        self.image_positions = nn.Parameter(
            torch.randn(1, self.window_side**2, vision_hidden_size) * embed_std
        )
        self.out_linear = nn.Linear(vision_hidden_size, llm_hidden_size, bias=True)

    def _win(self, x: torch.Tensor, side: int, win: int) -> torch.Tensor:
        """(B, side*side, C) → (B*n*n, win*win, C) where n=side//win."""
        B, _, C = x.shape
        n = side // win
        return (
            x.view(B, side, side, C)
            .view(B, n, win, n, win, C)
            .transpose(2, 3)
            .flatten(0, 2)
            .flatten(1, 2)
        )

    def _unwin(self, xw: torch.Tensor, n: int, win: int) -> torch.Tensor:
        """(B*n*n, win*win, C) → (B, (n*win)^2, C)."""
        Bnn, _, C = xw.shape
        B = Bnn // (n * n)
        side = n * win
        return (
            xw.view(B, n, n, win, win, C)
            .transpose(2, 3)
            .contiguous()
            .view(B, side, side, C)
            .flatten(1, 2)
        )

    def forward(self, image_features: torch.Tensor) -> torch.Tensor:
        B, HW, C = image_features.shape
        assert HW == self.image_side * self.image_side
        n = self.image_side // self.window_side

        image_features = self.norm(image_features)
        enc = self._win(image_features, self.image_side, self.window_side)

        downsampled = self.downsampler(image_features)
        new_side = n * self.query_side
        downsampled_w = self._win(downsampled, new_side, self.query_side)

        query_embeds = self.query + downsampled_w
        encoder_embeds = self.dropout(enc + self.image_positions)
        out_w = self.qformer(
            query_embeds=query_embeds,
            encoder_hidden_states=encoder_embeds,
        )

        out = self._unwin(out_w, n=n, win=self.query_side)
        out = self.dropout(out)
        return self.out_linear(out)


# ---------------------------------------------------------------------------
# Processing info / processor (reuses LlavaNext patterns)
# ---------------------------------------------------------------------------

class Granite4VisionProcessingInfo(LlavaNextProcessingInfo):

    def get_hf_config(self):
        return self.ctx.get_hf_config()

    def get_hf_processor(self, **kwargs):
        return self.ctx.get_hf_processor(**kwargs)

    def get_num_image_tokens(
        self,
        *,
        image_width: int,
        image_height: int,
    ) -> int:
        hf_config = self.get_hf_config()
        vision_encoder_info = self.get_vision_encoder_info()

        # After QFormer downsampling, patch grid is scaled by downsample_rate
        ds_rate = Fraction(hf_config.downsample_rate)
        patch_grid = vision_encoder_info.get_patch_grid_length()  # 24 for 384/16
        downsampled_grid = int(patch_grid * ds_rate)  # 12 for rate 4/8

        # Base feature: downsampled_grid^2
        base_feature_size = downsampled_grid * downsampled_grid

        num_patch_height, num_patch_width = get_anyres_image_grid_shape(
            image_size=(image_height, image_width),
            grid_pinpoints=hf_config.image_grid_pinpoints,
            patch_size=vision_encoder_info.get_image_size(),
        )

        (
            unpadded_feature_size,
            newline_feature_size,
        ) = self._get_num_unpadded_features(
            original_height=image_height,
            original_width=image_width,
            npatches=downsampled_grid,
            num_patch_height=num_patch_height,
            num_patch_width=num_patch_width,
        )

        return unpadded_feature_size + newline_feature_size + base_feature_size


class Granite4VisionMultiModalProcessor(
    BaseLlavaNextMultiModalProcessor[Granite4VisionProcessingInfo]
):
    def _get_mm_fields_config(
        self,
        hf_inputs: BatchFeature,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        return dict(
            pixel_values=MultiModalFieldConfig.batched("image"),
            image_sizes=MultiModalFieldConfig.batched("image"),
        )


# ---------------------------------------------------------------------------
# Top-level model
# ---------------------------------------------------------------------------

@MULTIMODAL_REGISTRY.register_processor(
    Granite4VisionMultiModalProcessor,
    info=Granite4VisionProcessingInfo,
    dummy_inputs=LlavaDummyInputsBuilder,
)
class Granite4VisionForConditionalGeneration(
    nn.Module, SupportsLoRA, SupportsMultiModal, SupportsPP
):
    """vLLM implementation of Granite 4 Vision.

    Architecture:
    - SigLIP vision tower -> WindowQFormerDownsampler projectors
    - Deepstack: 4 vision layers projected and injected at 4 LLM layers
    - Spatial: 4 offset groups from last vision layer injected at 4 more LLM layers
    - Granite language backbone with embedding_multiplier
    - logits_scaling via LogitsProcessor

    The outer model runs the LLM layer loop directly (like HF does) to inject
    deepstack features. This avoids wrapping the inner model and keeps weight
    loading simple.

    LoRA support:
    - Full merge: --hf-overrides '{"adapter_path": "path/to/lora"}' merges
      LM-only LoRA deltas at load time (W += scaling * B @ A).
    - Native LoRA: --enable-lora --default-mm-loras '{"image": "path/to/lora"}'
      lets vLLM runtime serve LM LoRA per-request.
    Both modes expect a LM-only adapter (no modules_to_save).
    """

    # LoRA class attributes (matches GraniteForCausalLM)
    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
    }
    embedding_modules = {}

    # Weight mapping: HF checkpoint -> vLLM parameter names
    # HF: model.language_model.layers.0...
    # vLLM: language_model.model.layers.0...
    # (because GraniteForCausalLM.model = GraniteModel)
    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_prefix={
            "model.language_model.": "language_model.model.",
            "model.layerwise_projectors.": "layerwise_projectors.",
            "model.spatial_projectors.": "spatial_projectors.",
            "model.image_newline": "image_newline",
            "model.vision_tower.": "vision_tower.vision_model.",
            "lm_head.": "language_model.lm_head.",
        }
    )

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None:
        if modality.startswith("image"):
            return "<image>"
        raise ValueError(f"Only image modality is supported, got {modality}")

    def get_mm_mapping(self) -> MultiModelKeys:
        return MultiModelKeys.from_string_field(
            language_model="language_model",
            connector=["layerwise_projectors", "spatial_projectors"],
            tower_model="vision_tower",
        )

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()

        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        self.config = config
        self.vllm_config = vllm_config

        # ----- Vision tower + projectors (marked as tower) -----
        with self._mark_tower_model(vllm_config, "image"):
            # Do NOT use init_vision_tower_for_llava here — it truncates the
            # encoder to vision_feature_layer depth. Deepstack needs ALL hidden
            # states (deepstack_layer_map uses negative indices into the full
            # encoder output list).
            self.vision_tower = SiglipVisionModel(
                config.vision_config,
                quant_config=quant_config,
                require_post_norm=False,
                prefix=maybe_prefix(prefix, "vision_tower"),
            )

            # image_newline parameter
            if config.use_image_newline_parameter:
                self.image_newline = nn.Parameter(
                    torch.empty(config.text_config.hidden_size)
                )
            else:
                self.image_newline = None

            cache_config = vllm_config.cache_config

            # Deepstack projectors: one per (vision_layer, llm_layer) pair
            self.layerwise_projectors = nn.ModuleList([
                WindowQFormerDownsampler(
                    config,
                    quant_config=quant_config,
                    cache_config=cache_config,
                    prefix=maybe_prefix(prefix, f"layerwise_projectors.{i}"),
                )
                for i in range(len(config.deepstack_layer_map))
            ])

            # Spatial projectors: 4 offset groups
            self.spatial_projectors = None
            if config.use_spatial_sampling:
                self.spatial_projectors = nn.ModuleList([
                    WindowQFormerDownsampler(
                        config,
                        quant_config=quant_config,
                        cache_config=cache_config,
                        spatial_offset=i,
                        prefix=maybe_prefix(prefix, f"spatial_projectors.{i}"),
                    )
                    for i in range(4)
                ])

        # ----- Language model (marked as LM) -----
        with self._mark_language_model(vllm_config):
            self.language_model = init_vllm_registered_model(
                vllm_config=vllm_config,
                hf_config=config.text_config,
                prefix=maybe_prefix(prefix, "language_model"),
            )

        self.make_empty_intermediate_tensors = (
            self.language_model.make_empty_intermediate_tensors
        )

        # Store config values we need
        self._deepstack_layer_map = config.deepstack_layer_map  # [[-19, 9], ...]
        self._use_spatial_sampling = getattr(config, "use_spatial_sampling", False)
        self._spatial_vision_layer = getattr(config, "spatial_vision_layer", -1)
        self._spatial_target_layers = getattr(config, "spatial_target_layers", [])
        self._vision_feature_select_strategy = getattr(
            config, "vision_feature_select_strategy", "full"
        )
        self._downsample_rate = Fraction(config.downsample_rate)

        # Deepstack state — set during embed_input_ids, consumed during forward
        # list of (llm_layer_idx, features_buffer) where buffer is (N, hidden_size)
        self._ds_features: list[tuple[int, torch.Tensor]] = []
        self._ds_vision_mask: torch.Tensor | None = None

    # ----- Vision feature extraction -----

    def _get_vision_hidden_states(
        self, pixel_values: torch.Tensor
    ) -> list[torch.Tensor]:
        """Run vision tower and return all hidden states (including input embeddings).

        Uses SiglipEncoder's built-in return_all_hidden_states support.
        Returns list[Tensor] where index 0 = embeddings, index i = after layer i-1.
        """
        vt = self.vision_tower
        vm = vt.vision_model if hasattr(vt, "vision_model") else vt

        hidden_states = vm.embeddings(pixel_values)
        all_hidden_states = vm.encoder(
            inputs_embeds=hidden_states,
            return_all_hidden_states=True,
        )
        return all_hidden_states

    def _pack_and_unpad_image_features(
        self,
        image_features: list[torch.Tensor] | tuple[torch.Tensor, ...],
        image_sizes: torch.Tensor,
    ) -> list[torch.Tensor]:
        """Reshape, unpad, and pack image features.

        Matches HF Granite4VisionModel.pack_and_unpad_image_features exactly.
        """
        config = self.config
        ds_rate = self._downsample_rate
        new_image_features = []

        for image_idx, image_feature in enumerate(image_features):
            if image_feature.shape[0] > 1:
                # Multi-patch: first is base, rest are high-res
                base_image_feature = image_feature[0]
                image_feature = image_feature[1:]

                height = width = (
                    config.vision_config.image_size
                    // config.vision_config.patch_size
                )
                # After QFormer downsampling
                height = int(height * ds_rate)
                width = int(width * ds_rate)

                num_patch_height, num_patch_width = get_anyres_image_grid_shape(
                    image_sizes[image_idx],
                    config.image_grid_pinpoints,
                    config.vision_config.image_size,
                )

                image_feature = image_feature.view(
                    num_patch_height, num_patch_width, height, width, -1
                )
                image_feature = (
                    image_feature.permute(4, 0, 2, 1, 3).contiguous()
                    .flatten(1, 2)
                    .flatten(2, 3)
                )
                image_feature = unpad_image(
                    image_feature, image_sizes[image_idx]
                )

                if self.image_newline is not None:
                    image_feature = torch.cat(
                        (
                            image_feature,
                            self.image_newline[:, None, None]
                            .expand(*image_feature.shape[:-1], 1)
                            .to(image_feature.device, image_feature.dtype),
                        ),
                        dim=-1,
                    )

                image_feature = image_feature.flatten(1, 2).transpose(0, 1)
                image_feature = torch.cat(
                    (base_image_feature, image_feature), dim=0
                )
            else:
                image_feature = image_feature[0]
                if self.image_newline is not None:
                    image_feature = torch.cat(
                        (image_feature, self.image_newline[None].to(image_feature)),
                        dim=0,
                    )

            new_image_features.append(image_feature)

        return new_image_features

    def _get_all_layer_features(
        self,
        pixel_values: torch.Tensor,
        image_sizes: torch.Tensor,
    ) -> list[tuple[int, list[torch.Tensor]]]:
        """Extract deepstack + spatial features.

        Returns list of (llm_layer_idx, [per_image_features, ...]) tuples.
        This is the vLLM equivalent of HF's get_image_features.
        """
        select_strategy = self._vision_feature_select_strategy

        # Count patches per image for splitting
        image_num_patches = [
            image_size_to_num_patches(
                image_size=imsize,
                grid_pinpoints=self.config.image_grid_pinpoints,
                patch_size=self.config.vision_config.image_size,
            )
            for imsize in image_sizes
        ]

        # Flatten 5D → 4D if needed
        if pixel_values.dim() == 5:
            _pv_list = [
                pv[:np_]
                for pv, np_ in zip(pixel_values, image_num_patches)
            ]
            pixel_values = torch.cat(_pv_list, dim=0)

        # Run vision tower once, get all hidden states
        all_hidden_states = self._get_vision_hidden_states(pixel_values)

        all_features = []

        # ----- Deepstack features -----
        for proj_idx, (vision_layer, llm_layer) in enumerate(
            self._deepstack_layer_map
        ):
            selected = all_hidden_states[vision_layer]

            if select_strategy == "default":
                selected = selected[:, 1:]  # remove CLS

            projected = self.layerwise_projectors[proj_idx](selected)

            projected_split = torch.split(projected, image_num_patches, dim=0)

            packed = self._pack_and_unpad_image_features(
                projected_split, image_sizes
            )

            all_features.append((llm_layer, packed))

        # ----- Spatial features -----
        if self._use_spatial_sampling and self.spatial_projectors is not None:
            spatial_hidden = all_hidden_states[self._spatial_vision_layer]

            if select_strategy == "default":
                spatial_hidden = spatial_hidden[:, 1:]

            for group_idx, llm_layer in enumerate(self._spatial_target_layers):
                projected = self.spatial_projectors[group_idx](spatial_hidden)
                projected_split = torch.split(
                    projected, image_num_patches, dim=0
                )
                packed = self._pack_and_unpad_image_features(
                    projected_split, image_sizes
                )
                all_features.append((llm_layer, packed))

        return all_features

    # ----- Multimodal interface -----

    def _parse_and_validate_image_input(
        self, **kwargs: object
    ) -> LlavaNextImageInputs | None:
        pixel_values = kwargs.pop("pixel_values", None)
        image_sizes = kwargs.pop("image_sizes", None)
        image_embeds = kwargs.pop("image_embeds", None)

        if pixel_values is None and image_embeds is None:
            return None

        if pixel_values is not None:
            expected_h = expected_w = self.config.vision_config.image_size
            return LlavaNextImagePixelInputs(
                type="pixel_values",
                pixel_values=pixel_values,
                image_sizes=image_sizes,
                resolve_bindings={"h": expected_h, "w": expected_w},
            )

        if image_embeds is not None:
            return LlavaNextImageEmbeddingInputs(
                type="image_embeds",
                data=image_embeds,
            )

        raise AssertionError("Unreachable")

    def embed_multimodal(self, **kwargs: object) -> MultiModalEmbeddings:
        """Convert pixel values → per-image placeholder tensors.

        The actual vision features are stored in self._ds_level_features and
        injected during the forward loop (like HF does). We return zero-
        tensors with the right shape so that _merge_multimodal_embeddings
        fills image positions with zeros (matching HF's masked_fill(mask, 0)).
        """
        image_input = self._parse_and_validate_image_input(**kwargs)
        if image_input is None:
            return []

        if image_input["type"] == "image_embeds":
            return [image_input["data"]]

        pixel_values = image_input["pixel_values"]
        image_sizes = image_input.get("image_sizes")

        if isinstance(pixel_values, list):
            pixel_values = torch.cat(pixel_values, dim=0)

        # Get all (llm_layer, [per_image_features]) pairs
        all_features = self._get_all_layer_features(pixel_values, image_sizes)

        # Store ALL level features for deepstack injection in forward()
        self._ds_level_features = all_features

        # Return zero-tensors matching the shape of level-0 features.
        # This makes _merge_multimodal_embeddings write zeros at image
        # positions (equivalent to HF's inputs_embeds.masked_fill(mask, 0)).
        # All real features are injected during the layer loop.
        if all_features:
            return [
                torch.zeros_like(feat)
                for feat in all_features[0][1]
            ]
        return []

    def embed_input_ids(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: MultiModalEmbeddings | None = None,
        *,
        is_multimodal: torch.Tensor | None = None,
        handle_oov_mm_token: bool = True,
    ) -> torch.Tensor:
        """Merge text and vision embeddings, apply embedding_multiplier.

        HF flow:
        1. inputs_embeds = embed_tokens(input_ids)
        2. inputs_embeds.masked_fill(vision_mask, 0.0)
        3. hidden_states = inputs_embeds * embedding_multiplier
        4. layer loop with deepstack injection via masked_scatter

        vLLM's GraniteModel.forward:
        - if inputs_embeds given: hidden_states = inputs_embeds (NO multiplier)
        - if input_ids given: hidden_states = embed(input_ids) * multiplier

        So we apply embedding_multiplier here and pass inputs_embeds to forward.
        """
        # Access the inner GraniteModel
        lm_inner = self.language_model.model  # GraniteModel

        has_vision = (
            multimodal_embeddings is not None
            and is_multimodal is not None
            and len(multimodal_embeddings) > 0
            and is_multimodal.any()
        )

        if not has_vision:
            # Text-only or decode: clear deepstack state
            self._ds_features = []
            self._ds_vision_mask = None
            self._ds_level_features = []
            # Apply embedding_multiplier here because forward() always receives
            # inputs_embeds and skips the inner model's embed+multiply path.
            embeds = lm_inner.embed_input_ids(input_ids)
            return embeds * lm_inner.config.embedding_multiplier

        # --- Vision path ---
        # HF flow: embed -> masked_fill(0) -> multiply
        # by embedding_multiplier. Layer loop adds vision
        # features via masked_scatter (never in inputs_embeds).

        # 1. Get text embeddings
        text_embeds = lm_inner.embed_input_ids(input_ids)

        # 2. Zero out image positions (HF: inputs_embeds.masked_fill(vision_mask, 0.0))
        #    _merge_multimodal_embeddings writes our zero-tensors here, same effect.
        _merge_multimodal_embeddings(
            inputs_embeds=text_embeds,
            multimodal_embeddings=multimodal_embeddings,
            is_multimodal=is_multimodal,
        )

        # 3. Apply embedding_multiplier to ALL positions (text + vision=0)
        embedding_multiplier = lm_inner.config.embedding_multiplier
        inputs_embeds = text_embeds * embedding_multiplier

        # 5. Prepare deepstack feature buffers for the layer loop
        N = inputs_embeds.size(0)
        hidden_size = inputs_embeds.size(1)

        prepared = []
        for llm_layer, per_image_features in getattr(self, "_ds_level_features", []):
            concat_features = torch.cat(per_image_features, dim=0)
            buf = torch.zeros(
                N, hidden_size,
                dtype=inputs_embeds.dtype,
                device=inputs_embeds.device,
            )
            buf[is_multimodal] = concat_features.to(dtype=inputs_embeds.dtype)
            prepared.append((llm_layer, buf))

        self._ds_features = prepared
        self._ds_vision_mask = is_multimodal
        self._ds_level_features = []  # consumed

        return inputs_embeds

    # ----- Forward -----

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: object,
    ) -> torch.Tensor | IntermediateTensors:
        """Forward pass with deepstack injection.

        Runs the LLM layer loop directly (like HF Granite4VisionModel.forward)
        to inject vision features at target layers via masked addition.
        """
        if intermediate_tensors is not None:
            inputs_embeds = None

        # Access the inner GraniteModel
        lm_inner = self.language_model.model

        if get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                # embed_input_ids already applied embedding_multiplier
                hidden_states = inputs_embeds
            else:
                # Text-only path: embed + multiply
                hidden_states = lm_inner.embed_input_ids(input_ids)
                hidden_states = hidden_states * lm_inner.config.embedding_multiplier
        else:
            if intermediate_tensors is None:
                raise RuntimeError("Intermediate tensors may not be None!")
            hidden_states = intermediate_tensors["hidden_states"]

        num_tokens = hidden_states.size(0)

        # Build O(1) lookup for deepstack features
        ds_map: dict[int, torch.Tensor] = {}
        for llm_layer_idx, features in self._ds_features:
            ds_map[llm_layer_idx] = features

        vision_mask = self._ds_vision_mask
        if vision_mask is not None:
            vision_mask = vision_mask[:num_tokens]

        # Run through decoder layers with deepstack injection
        for i in range(lm_inner.start_layer, lm_inner.end_layer):
            layer = lm_inner.layers[i]

            # Inject deepstack features at target layers (before layer forward)
            if i in ds_map and vision_mask is not None and vision_mask.any():
                features = ds_map[i][:num_tokens]
                hidden_states[vision_mask] += features[vision_mask]

            hidden_states = layer(positions, hidden_states)

        if not get_pp_group().is_last_rank:
            return IntermediateTensors({"hidden_states": hidden_states})

        hidden_states = lm_inner.norm(hidden_states)
        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor | None:
        # GraniteForCausalLM.compute_logits uses
        # LogitsProcessor(scale=1/logits_scaling)
        return self.language_model.compute_logits(hidden_states)

    # ----- Full-merge LoRA support -----

    # HF→vLLM key prefix mapping (same transforms as hf_to_vllm_mapper)
    _ADAPTER_PREFIX_MAP = [
        ("model.language_model.", "language_model.model."),
    ]

    # vLLM fuses q/k/v_proj into qkv_proj and gate/up_proj into gate_up_proj.
    _STACKED_PARAMS_MAPPING = [
        (".qkv_proj", ".q_proj", "q"),
        (".qkv_proj", ".k_proj", "k"),
        (".qkv_proj", ".v_proj", "v"),
        (".gate_up_proj", ".gate_proj", 0),
        (".gate_up_proj", ".up_proj", 1),
    ]

    @staticmethod
    def _peft_to_vllm(peft_key: str) -> str:
        """Strip 'base_model.model.' and apply HF→vLLM prefix mapping."""
        name = peft_key
        if name.startswith("base_model.model."):
            name = name[len("base_model.model."):]
        for old_pfx, new_pfx in (
            Granite4VisionForConditionalGeneration._ADAPTER_PREFIX_MAP
        ):
            if name.startswith(old_pfx):
                name = new_pfx + name[len(old_pfx):]
                break
        return name

    @staticmethod
    def _load_adapter(adapter_path: str) -> tuple[dict, dict[str, torch.Tensor]]:
        """Load adapter config and safetensors from a directory or HF hub ID."""
        # Resolve HF hub IDs to local cache path
        if not os.path.isdir(adapter_path):
            from huggingface_hub import snapshot_download
            adapter_path = snapshot_download(adapter_path)
        config_path = os.path.join(adapter_path, "adapter_config.json")
        weights_path = os.path.join(adapter_path, "adapter_model.safetensors")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"No adapter_config.json in {adapter_path}")
        if not os.path.exists(weights_path):
            raise FileNotFoundError(
                f"No adapter_model.safetensors in {adapter_path}")
        with open(config_path) as f:
            config = json.load(f)
        weights = load_file(weights_path)
        return config, weights

    def _merge_lora_deltas(
        self,
        adapter_config: dict,
        adapter_weights: dict[str, torch.Tensor],
    ) -> int:
        """Merge LM-only LoRA deltas into model weights: W += scaling * B @ A.

        Uses _STACKED_PARAMS_MAPPING + module._get_shard_offset_mapping()
        to handle packed QKV correctly (works with GQA automatically).
        """
        lora_alpha = adapter_config.get("lora_alpha", 1)
        lora_r = adapter_config.get("r", 1)
        scaling = lora_alpha / lora_r

        # Collect lora_A / lora_B by vLLM module key
        lora_a: dict[str, torch.Tensor] = {}
        lora_b: dict[str, torch.Tensor] = {}
        for peft_key, tensor in adapter_weights.items():
            if ".lora_A." in peft_key:
                module_key = self._peft_to_vllm(
                    peft_key.replace(".lora_A.weight", ""))
                lora_a[module_key] = tensor
            elif ".lora_B." in peft_key:
                module_key = self._peft_to_vllm(
                    peft_key.replace(".lora_B.weight", ""))
                lora_b[module_key] = tensor

        params_dict = dict(self.named_parameters())
        modules_dict = dict(self.named_modules())

        def _add_delta(name: str, delta: torch.Tensor) -> bool:
            # Try stacked/fused params first (qkv_proj, gate_up_proj)
            for fused_name, orig_name, shard_id in self._STACKED_PARAMS_MAPPING:
                if orig_name not in name:
                    continue
                fused_param_name = name.replace(orig_name, fused_name)
                if fused_param_name not in params_dict:
                    continue
                param = params_dict[fused_param_name]
                module_path = fused_param_name.rsplit(".weight", 1)[0]
                module = modules_dict.get(module_path)
                if module is None:
                    continue

                tp_rank = get_tensor_model_parallel_rank()
                tp_size = get_tensor_model_parallel_world_size()

                if hasattr(module, "_get_shard_offset_mapping"):
                    # QKVParallelLinear: string shard_id ("q", "k", "v")
                    shard_offset = module._get_shard_offset_mapping(shard_id)
                    if shard_offset is not None:
                        shard_size = delta.shape[0] // tp_size
                        tp_delta = delta.narrow(
                            0, tp_rank * shard_size, shard_size)
                        shard = param.data[shard_offset:shard_offset + shard_size]
                        param.data[shard_offset:shard_offset + shard_size] = (
                            shard.float() + tp_delta.to(shard.device)
                        ).to(shard.dtype)
                        return True
                elif hasattr(module, "output_sizes") and isinstance(shard_id, int):
                    # MergedColumnParallelLinear: integer shard_id (0, 1)
                    shard_size = module.output_sizes[shard_id] // tp_size
                    shard_offset = sum(
                        s // tp_size for s in module.output_sizes[:shard_id]
                    )
                    tp_delta = delta.narrow(
                        0, tp_rank * (delta.shape[0] // tp_size),
                        delta.shape[0] // tp_size)
                    shard = param.data[shard_offset:shard_offset + shard_size]
                    param.data[shard_offset:shard_offset + shard_size] = (
                        shard.float() + tp_delta.to(shard.device)
                    ).to(shard.dtype)
                    return True
            # Direct param (o_proj, down_proj)
            if name in params_dict:
                param = params_dict[name]
                # Under TP, param is already sharded but delta is full-size.
                # Slice delta to match: dim 0 for column-parallel, dim 1 for
                # row-parallel.
                if delta.shape != param.data.shape:
                    tp_rank = get_tensor_model_parallel_rank()
                    for dim in range(delta.dim()):
                        if delta.shape[dim] != param.data.shape[dim]:
                            shard_size = param.data.shape[dim]
                            offset = tp_rank * shard_size
                            delta = delta.narrow(dim, offset, shard_size)
                            break
                merged = param.data.float() + delta.to(param.device)
                param.data = merged.to(param.dtype)
                return True
            return False

        merge_device = next(self.parameters()).device
        merged = 0
        for module_key in sorted(lora_a):
            if module_key not in lora_b:
                logger.warning("LoRA B missing for %s, skipping", module_key)
                continue
            A = lora_a[module_key].to(merge_device).float()
            B = lora_b[module_key].to(merge_device).float()
            delta = scaling * (B @ A)
            if _add_delta(module_key + ".weight", delta):
                merged += 1
            else:
                logger.warning("LoRA target not found: %s", module_key)

        return merged

    def _apply_adapter(self) -> None:
        """Full-merge entry point: called when config.adapter_path is set."""
        adapter_path = getattr(self.config, "adapter_path", None)
        if not adapter_path:
            return
        logger.info("Full-merge LoRA from %s", adapter_path)
        adapter_config, adapter_weights = self._load_adapter(adapter_path)

        if adapter_config.get("modules_to_save"):
            raise ValueError(
                "Adapter has modules_to_save — only LM-only adapters "
                "(no modules_to_save) are supported."
            )

        n = self._merge_lora_deltas(adapter_config, adapter_weights)
        logger.info("Merged %d LoRA pairs into base weights", n)

    def load_weights(
        self, weights: Iterable[tuple[str, torch.Tensor]]
    ) -> set[str]:
        loader = AutoWeightsLoader(self)
        loaded = loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)
        self._apply_adapter()
        return loaded
