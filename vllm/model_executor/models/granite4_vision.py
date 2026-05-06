# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""vLLM implementation of Granite 4 Vision.

Uses GraniteForCausalLM as the language backbone with SigLIP vision encoder
and deepstack feature injection via WindowQFormer projectors.

LoRA support: use --enable-lora --default-mm-loras for LM-only LoRA adapters.
"""

import math
from collections.abc import Iterable, Mapping
from fractions import Fraction
from itertools import islice

import torch
import torch.nn as nn
from transformers import BatchFeature
from transformers.models.blip_2.configuration_blip_2 import Blip2QFormerConfig
from transformers.models.llava_next.modeling_llava_next import (
    get_anyres_image_grid_shape,
    image_size_to_num_patches,
    unpad_image,
)

from vllm.compilation.decorators import support_torch_compile
from vllm.config import CacheConfig, VllmConfig
from vllm.distributed.parallel_state import get_pp_group
from vllm.logger import init_logger
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
from vllm.model_executor.models.granite import GraniteForCausalLM, GraniteModel
from vllm.model_executor.models.interfaces import (
    MultiModalEmbeddings,
    SupportsLoRA,
    SupportsMultiModal,
    SupportsPP,
)
from vllm.model_executor.models.llava import LlavaDummyInputsBuilder
from vllm.model_executor.models.llava_next import (
    BaseLlavaNextMultiModalProcessor,
    LlavaNextImageEmbeddingInputs,
    LlavaNextImageInputs,
    LlavaNextImagePixelInputs,
    LlavaNextProcessingInfo,
)
from vllm.model_executor.models.module_mapping import MultiModelKeys
from vllm.model_executor.models.siglip import SiglipVisionModel
from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    PPMissingLayer,
    WeightsMapper,
    maybe_prefix,
)
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import MultiModalFieldConfig
from vllm.sequence import IntermediateTensors

from .blip2 import Blip2QFormerModel

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
        assert self.image_side * self.image_side == HW
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
# LLM subclasses with deepstack injection in the layer loop
# ---------------------------------------------------------------------------


@support_torch_compile(
    dynamic_arg_dims={
        "input_ids": 0,
        "positions": 0,
        "intermediate_tensors": 0,
        "inputs_embeds": 0,
        "deepstack_input_embeds": 0,
    }
)
class Granite4VisionLLMModel(GraniteModel):
    """GraniteModel with deepstack feature injection in the layer loop."""

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        deepstack_input_embeds: IntermediateTensors | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        if get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                hidden_states = self.embed_input_ids(input_ids)
                hidden_states = hidden_states * self.config.embedding_multiplier
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            # Recover deepstack features forwarded from the previous PP rank.
            if deepstack_input_embeds is None:
                ds_keys = [
                    k for k in intermediate_tensors.tensors if k.startswith("ds_")
                ]
                if ds_keys:
                    deepstack_input_embeds = IntermediateTensors(
                        {k: intermediate_tensors[k] for k in ds_keys}
                    )

        for layer_idx, layer in islice(
            enumerate(self.layers), self.start_layer, self.end_layer
        ):
            if deepstack_input_embeds is not None:
                key = f"ds_{layer_idx}"
                if key in deepstack_input_embeds.tensors:
                    feat = deepstack_input_embeds[key]
                    # Resize to match hidden_states in case of CUDA graph padding
                    num_tokens = hidden_states.size(0)
                    buf_len = feat.shape[0]
                    if buf_len != num_tokens:
                        feat = torch.nn.functional.pad(
                            feat[:num_tokens],
                            (0, 0, 0, max(0, num_tokens - buf_len)),
                        )
                    hidden_states = hidden_states + feat
            hidden_states = layer(positions, hidden_states)

        if not get_pp_group().is_last_rank:
            # Forward hidden_states and any deepstack features for later ranks.
            it = {"hidden_states": hidden_states}
            if deepstack_input_embeds is not None:
                remaining = {
                    k: v
                    for k, v in deepstack_input_embeds.tensors.items()
                    if int(k.split("_")[1]) >= self.end_layer
                }
                it.update(remaining)
            return IntermediateTensors(it)

        hidden_states = self.norm(hidden_states)
        return hidden_states


class Granite4VisionLLMForCausalLM(GraniteForCausalLM):
    """GraniteForCausalLM backed by Granite4VisionLLMModel."""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        nn.Module.__init__(self)
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        self.config = config
        self.quant_config = quant_config
        self.model = Granite4VisionLLMModel(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model")
        )
        if get_pp_group().is_last_rank:
            self.lm_head = ParallelLMHead(
                config.vocab_size,
                config.hidden_size,
                quant_config=quant_config,
                prefix=maybe_prefix(prefix, "lm_head"),
            )
            if config.tie_word_embeddings:
                self.lm_head.weight = self.model.embed_tokens.weight
            logit_scale = getattr(config, "logit_scale", 1.0)
            if hasattr(config, "logits_scaling"):
                logit_scale /= config.logits_scaling
            self.logits_processor = LogitsProcessor(
                config.vocab_size, scale=logit_scale
            )
        else:
            self.lm_head = PPMissingLayer()

    def make_empty_intermediate_tensors(
        self, batch_size: int, dtype: torch.dtype, device: torch.device
    ) -> IntermediateTensors:
        tensors = super().make_empty_intermediate_tensors(batch_size, dtype, device)
        # Include deepstack buffers so non-first PP ranks receive them.
        # _ds_layer_indices is set directly on this instance by the outer model.
        for llm_layer in getattr(self, "_ds_layer_indices", []):
            tensors.tensors[f"ds_{llm_layer}"] = torch.zeros(
                (batch_size, self.config.hidden_size), dtype=dtype, device=device
            )
        return tensors


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
            "model.vision_tower.": "vision_tower.",
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
            self.layerwise_projectors = nn.ModuleList(
                [
                    WindowQFormerDownsampler(
                        config,
                        quant_config=quant_config,
                        cache_config=cache_config,
                        prefix=maybe_prefix(prefix, f"layerwise_projectors.{i}"),
                    )
                    for i in range(len(config.deepstack_layer_map))
                ]
            )

            # Spatial projectors: 4 offset groups
            self.spatial_projectors = None
            if config.use_spatial_sampling:
                self.spatial_projectors = nn.ModuleList(
                    [
                        WindowQFormerDownsampler(
                            config,
                            quant_config=quant_config,
                            cache_config=cache_config,
                            spatial_offset=i,
                            prefix=maybe_prefix(prefix, f"spatial_projectors.{i}"),
                        )
                        for i in range(4)
                    ]
                )

        # ----- Language model (marked as LM) -----
        with self._mark_language_model(vllm_config):
            self.language_model = Granite4VisionLLMForCausalLM(
                vllm_config=vllm_config.with_hf_config(config.text_config),
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

        # Ordered list of LLM layer indices for each deepstack level.
        # Pre-populated from config so it's available during CUDA graph capture
        # (before any embed_multimodal call).
        self._ds_layer_indices: list[int] = [
            llm_layer for _, llm_layer in config.deepstack_layer_map
        ] + list(getattr(config, "spatial_target_layers", []))

        # Share ds_layer_indices with the LLM causal model so
        # make_empty_intermediate_tensors includes the correct keys
        # (its self.config is text_config, no deepstack_layer_map).
        self.language_model._ds_layer_indices = self._ds_layer_indices

        # Pre-allocated persistent GPU buffers for deepstack features.
        # Written via .copy_() in embed_input_ids(), read by forward() via a
        # slice. Because the buffer address is fixed, CUDA graph replay sees
        # the updated values written just before each prefill.
        # Shape: (max_num_batched_tokens, lm_hidden_size) per level.
        n_layerwise = len(config.deepstack_layer_map)
        n_spatial = len(getattr(config, "spatial_target_layers", []))
        num_ds_levels = n_layerwise + n_spatial
        lm_hidden = config.text_config.hidden_size
        max_tokens = vllm_config.scheduler_config.max_num_batched_tokens
        # Allocated on CPU first; moved to GPU in embed_input_ids on first use.
        self._ds_buffers: list[torch.Tensor] = [
            torch.zeros(max_tokens, lm_hidden) for _ in range(num_ds_levels)
        ]
        self._ds_num_tokens: int = 0  # tokens written in last embed_input_ids call

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
                    config.vision_config.image_size // config.vision_config.patch_size
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
                    image_feature.permute(4, 0, 2, 1, 3)
                    .contiguous()
                    .flatten(1, 2)
                    .flatten(2, 3)
                )
                image_feature = unpad_image(image_feature, image_sizes[image_idx])

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
                image_feature = torch.cat((base_image_feature, image_feature), dim=0)
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
    ) -> tuple[list[int], list[torch.Tensor]]:
        """Extract deepstack + spatial features for all levels.

        Returns:
          llm_layer_indices: ordered list of target LLM layer indices
          per_image_packed:  one tensor per image, shape
                             (num_tokens_i, lm_hidden_size * num_levels),
                             all levels packed on dim=-1.

        Packing on dim=-1 means the framework's token-level slicing for
        chunked prefill preserves all levels intact.
        """
        select_strategy = self._vision_feature_select_strategy

        image_num_patches = [
            image_size_to_num_patches(
                image_size=imsize,
                grid_pinpoints=self.config.image_grid_pinpoints,
                patch_size=self.config.vision_config.image_size,
            )
            for imsize in image_sizes
        ]

        if pixel_values.dim() == 5:
            pixel_values = torch.cat(
                [pv[:np_] for pv, np_ in zip(pixel_values, image_num_patches)],
                dim=0,
            )

        all_hidden_states = self._get_vision_hidden_states(pixel_values)

        # Collect per-level: (llm_layer, [per_image_tensor, ...])
        levels: list[tuple[int, list[torch.Tensor]]] = []

        for proj_idx, (vision_layer, llm_layer) in enumerate(self._deepstack_layer_map):
            selected = all_hidden_states[vision_layer]
            if select_strategy == "default":
                selected = selected[:, 1:]
            projected = self.layerwise_projectors[proj_idx](selected)
            per_image = self._pack_and_unpad_image_features(
                torch.split(projected, image_num_patches, dim=0), image_sizes
            )
            levels.append((llm_layer, per_image))

        if self._use_spatial_sampling and self.spatial_projectors is not None:
            spatial_hidden = all_hidden_states[self._spatial_vision_layer]
            if select_strategy == "default":
                spatial_hidden = spatial_hidden[:, 1:]
            for group_idx, llm_layer in enumerate(self._spatial_target_layers):
                projected = self.spatial_projectors[group_idx](spatial_hidden)
                per_image = self._pack_and_unpad_image_features(
                    torch.split(projected, image_num_patches, dim=0), image_sizes
                )
                levels.append((llm_layer, per_image))

        llm_layer_indices = [llm_layer for llm_layer, _ in levels]
        num_images = len(image_sizes)
        per_image_packed = [
            torch.cat([levels[lvl][1][img] for lvl in range(len(levels))], dim=-1)
            for img in range(num_images)
        ]

        return llm_layer_indices, per_image_packed

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
        """Run vision tower and return per-image packed feature tensors.

        Each returned tensor has shape (num_tokens_i, lm_hidden_size * num_levels)
        with all deepstack levels packed on dim=-1. The framework caches these
        tensors and slices along dim=0 for chunked prefill — all levels survive
        intact because slicing is token-wise, not feature-wise.

        embed_input_ids() splits the packed tensor back into per-level buffers.
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

        llm_layer_indices, per_image_packed = self._get_all_layer_features(
            pixel_values, image_sizes
        )
        self._ds_layer_indices = llm_layer_indices
        return per_image_packed

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
        4. layer loop injects deepstack features at target layers

        multimodal_embeddings contains packed tensors from embed_multimodal():
        shape (num_tokens_i, lm_hidden_size * num_levels). We split on dim=-1
        to get per-level features, build batch-sized buffers (zero at text
        positions), and store in self._ds_features for forward().
        """
        lm_inner = self.language_model.model

        has_vision = (
            multimodal_embeddings is not None
            and is_multimodal is not None
            and len(multimodal_embeddings) > 0
            and is_multimodal.any()
        )

        if not has_vision:
            self._ds_num_tokens = 0
            embeds = lm_inner.embed_input_ids(input_ids)
            return embeds * lm_inner.config.embedding_multiplier

        # 1. Text embeddings
        text_embeds = lm_inner.embed_input_ids(input_ids)

        # 2. Zero image positions (matches HF masked_fill(vision_mask, 0.0))
        text_embeds[is_multimodal] = 0.0

        # 3. Apply embedding_multiplier
        inputs_embeds = text_embeds * lm_inner.config.embedding_multiplier

        # 4. Split packed tensors into per-level features and build buffers.
        #    multimodal_embeddings is a list of per-image packed tensors
        #    (possibly a chunk slice from the framework's encoder cache).
        #    Concatenate along token dim → (total_mm_tokens, lm_h * num_levels).
        N, lm_h = inputs_embeds.shape
        all_packed = torch.cat(
            [t.to(dtype=inputs_embeds.dtype) for t in multimodal_embeddings],
            dim=0,
        )
        level_features = all_packed.split(lm_h, dim=-1)  # num_levels tensors

        # Ensure persistent buffers are on the right device/dtype (first call).
        buf0 = self._ds_buffers[0]
        if buf0.device != inputs_embeds.device or buf0.dtype != inputs_embeds.dtype:
            self._ds_buffers = [
                b.to(device=inputs_embeds.device, dtype=inputs_embeds.dtype)
                for b in self._ds_buffers
            ]

        for level_idx in range(len(self._ds_layer_indices)):
            target = self._ds_buffers[level_idx][:N]
            target.zero_()
            target[is_multimodal] = level_features[level_idx]

        self._ds_num_tokens = N
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
        if intermediate_tensors is not None:
            inputs_embeds = None

        # Build IntermediateTensors from pre-allocated persistent buffers.
        # Always pass deepstack when inputs_embeds is non-None (prefill path),
        # including during CUDA graph capture (buffers are zero → no-op injection).
        # This ensures the graph captures the injection code path.
        if (
            inputs_embeds is not None
            and get_pp_group().is_first_rank
            and self._ds_layer_indices
        ):
            n = inputs_embeds.size(0)
            ds: IntermediateTensors | None = IntermediateTensors(
                {
                    f"ds_{llm_layer}": self._ds_buffers[lvl][:n]
                    for lvl, llm_layer in enumerate(self._ds_layer_indices)
                }
            )
        else:
            ds = None

        hidden_states = self.language_model.model(
            input_ids=input_ids,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
            deepstack_input_embeds=ds,
        )

        # Clear buffers after use so stale features don't leak into the next request.
        if (
            inputs_embeds is not None
            and get_pp_group().is_first_rank
            and self._ds_num_tokens > 0
        ):
            n = self._ds_num_tokens
            for buf in self._ds_buffers:
                buf[:n].zero_()
            self._ds_num_tokens = 0

        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor | None:
        # GraniteForCausalLM.compute_logits uses
        # LogitsProcessor(scale=1/logits_scaling)
        return self.language_model.compute_logits(hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)
