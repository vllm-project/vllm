# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Adapted from Qwen2.5-VL implementation
# OpenCUA-7B uses 1D-RoPE instead of M-RoPE (Multimodal RoPE)
# Copyright 2025 The vLLM team.
# Copyright 2025 XLANG Lab, The University of Hong Kong

"""Inference-only OpenCUA-7B model compatible with HuggingFace weights."""

from collections.abc import Iterable, Mapping, Sequence
from functools import lru_cache, partial
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BatchFeature

from vllm.attention.backends.registry import AttentionBackendEnum
from vllm.attention.layer import maybe_get_vit_flash_attn_backend
from vllm.config import VllmConfig
from vllm.forward_context import set_forward_context
from vllm.logger import init_logger
from vllm.model_executor.layers.activation import get_act_and_mul_fn
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.module_mapping import MultiModelKeys
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import (
    MultiModalFeatureSpec,
    MultiModalFieldConfig,
    MultiModalKwargs,
)
from vllm.multimodal.parse import MultiModalDataItems, MultiModalDataParser
from vllm.multimodal.processing import (
    BaseMultiModalProcessor,
    PromptReplacement,
    PromptUpdate,
)
from vllm.sequence import IntermediateTensors
from vllm.utils.platform_utils import is_pin_memory_available

from .qwen2_vl import (
    Qwen2VLMultiModalDataParser,
    Qwen2VLProcessingInfo,
    _create_qwen2vl_field_factory,
)
from .qwen2_vl import Qwen2VLDummyInputsBuilder
from vllm.multimodal.profiling import BaseDummyInputsBuilder
from transformers.models.qwen2_vl import (
    Qwen2VLImageProcessor,
    Qwen2VLProcessor,
    Qwen2VLVideoProcessor,
)
from vllm.transformers_utils.tokenizer import AnyTokenizer
from .interfaces import (
    MultiModalEmbeddings,
    SupportsEagle3,
    SupportsLoRA,
    SupportsMRoPE,
    SupportsMultiModal,
    SupportsPP,
    SupportsQuant,
)
from .qwen2_5_vl import (
    Qwen2_5_VLImageEmbeddingInputs,
    Qwen2_5_VLImageInputs,
    Qwen2_5_VLImagePixelInputs,
    Qwen2_5_VLVideoEmbeddingInputs,
    Qwen2_5_VLVideoInputs,
    Qwen2_5_VLVideoPixelInputs,
    Qwen2_5_VisionTransformer,
)
from .utils import (
    AutoWeightsLoader,
    WeightsMapper,
    cast_overflow_tensors,
    init_vllm_registered_model,
    maybe_prefix,
)
from .vision import (
    get_vit_attn_backend,
    run_dp_sharded_mrope_vision_model,
)

logger = init_logger(__name__)


class OpenCUAVisionTransformer(nn.Module):
    """Vision Transformer for OpenCUA with 1D-RoPE instead of M-RoPE."""

    def __init__(
        self,
        vision_config: Any,  # OpenCUAConfig.vision_config
        norm_eps: float = 1e-6,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        use_data_parallel: bool = False,
        attn_backend_override: AttentionBackendEnum | None = None,
    ) -> None:
        super().__init__()

        patch_size = vision_config.patch_size
        temporal_patch_size = vision_config.temporal_patch_size
        in_channels = vision_config.in_channels
        depth = vision_config.depth
        self.hidden_size = vision_config.hidden_size
        self.num_heads = vision_config.num_heads
        self.use_data_parallel = use_data_parallel
        self.out_hidden_size = vision_config.out_hidden_size

        # args for get_window_index_thw
        self.window_size = vision_config.window_size
        self.patch_size = vision_config.patch_size
        self.spatial_merge_size = vision_config.spatial_merge_size
        self.fullatt_block_indexes = vision_config.fullatt_block_indexes
        self.spatial_merge_unit = self.spatial_merge_size**2

        from vllm.compilation.backends import set_model_tag

        with set_model_tag("OpenCUAVisionPatchEmbed"):
            self.patch_embed = Qwen2_5_VisionPatchEmbed(
                patch_size=patch_size,
                temporal_patch_size=temporal_patch_size,
                in_channels=in_channels,
                hidden_size=self.hidden_size,
            )

        norm_layer = partial(RMSNorm, eps=norm_eps)
        head_dim = self.hidden_size // self.num_heads
        # Use 1D-RoPE instead of M-RoPE
        self.rotary_pos_emb = Qwen2_5_VisionRotaryEmbedding(head_dim // 2)

        # Flash Attention: vLLM's flash_attn requires head_dim to be a multiple of 32
        # but upstream flash_attn (flash-attn library) supports head_dim=80
        # OpenCUA uses head_dim=80 (1280/16), same as Qwen2.5-VL
        # Use upstream flash_attn if available, otherwise fallback to XFORMERS
        use_upstream_fa = False
        from vllm.attention.layer import check_upstream_fa_availability
        from vllm.platforms import current_platform
        
        if head_dim % 32 != 0:
            # If head_dim is not a multiple of 32, we need upstream flash_attn
            # which supports arbitrary head_dim values
            if current_platform.is_cuda() and check_upstream_fa_availability(torch.get_default_dtype()):
                use_upstream_fa = True
            else:
                # Fallback to XFORMERS if upstream flash_attn is not available
                attn_backend_override = AttentionBackendEnum.XFORMERS
        
        self.attn_backend = get_vit_attn_backend(
            head_size=head_dim,
            dtype=torch.get_default_dtype(),
            attn_backend_override=attn_backend_override,
        )

        self.attn_backend, self.flash_attn_varlen_func = (
            maybe_get_vit_flash_attn_backend(
                self.attn_backend,
                use_upstream_fa,
                attn_backend_override=attn_backend_override,
            )
        )

        if self.attn_backend not in {
            AttentionBackendEnum.FLASH_ATTN,
            AttentionBackendEnum.TORCH_SDPA,
            AttentionBackendEnum.XFORMERS,
            AttentionBackendEnum.ROCM_AITER_FA,
        }:
            raise RuntimeError(
                f"OpenCUA does not support {self.attn_backend} backend now."
            )

        with set_model_tag("OpenCUAVisionBlock"):
            self.blocks = nn.ModuleList(
                [
                    Qwen2_5_VisionBlock(
                        dim=self.hidden_size,
                        num_heads=self.num_heads,
                        mlp_hidden_dim=vision_config.intermediate_size,
                        act_fn=get_act_and_mul_fn(vision_config.hidden_act),
                        norm_layer=norm_layer,
                        quant_config=quant_config,
                        prefix=f"{prefix}.blocks.{layer_idx}",
                        use_data_parallel=use_data_parallel,
                        attn_backend=self.attn_backend,
                        use_upstream_fa=use_upstream_fa,
                        attn_backend_override=attn_backend_override,
                    )
                    for layer_idx in range(depth)
                ]
            )

        with set_model_tag("OpenCUAVisionPatchMerger"):
            self.merger = Qwen2_5_VisionPatchMerger(
                d_model=vision_config.out_hidden_size,
                context_dim=self.hidden_size,
                norm_layer=norm_layer,
                spatial_merge_size=self.spatial_merge_size,
                quant_config=quant_config,
                prefix=f"{prefix}.merger",
                use_data_parallel=use_data_parallel,
            )

    @property
    def dtype(self) -> torch.dtype:
        return self.patch_embed.proj.weight.dtype

    @property
    def device(self) -> torch.device:
        return self.patch_embed.proj.weight.device

    def rotary_pos_emb_1d(self, seq_len: int) -> torch.Tensor:
        """Generate 1D-RoPE embeddings based on sequence length only.
        
        This is equivalent to Qwen2.5-VL's rotary_pos_emb but for 1D positions.
        Returns shape: (seq_len, rotary_dim)
        """
        rotary_pos_emb = self.rotary_pos_emb(seq_len)
        # rotary_pos_emb shape: (seq_len, rotary_dim)
        return rotary_pos_emb

    def get_window_index_thw(self, grid_t, grid_h, grid_w):
        vit_merger_window_size = (
            self.window_size // self.spatial_merge_size // self.patch_size
        )

        llm_grid_h = grid_h // self.spatial_merge_size
        llm_grid_w = grid_w // self.spatial_merge_size
        index = torch.arange(grid_t * llm_grid_h * llm_grid_w).reshape(
            grid_t, llm_grid_h, llm_grid_w
        )
        pad_h = vit_merger_window_size - llm_grid_h % vit_merger_window_size
        pad_w = vit_merger_window_size - llm_grid_w % vit_merger_window_size
        num_windows_h = (llm_grid_h + pad_h) // vit_merger_window_size
        num_windows_w = (llm_grid_w + pad_w) // vit_merger_window_size
        index_padded = F.pad(index, (0, pad_w, 0, pad_h), "constant", -100)
        index_padded = index_padded.reshape(
            grid_t,
            num_windows_h,
            vit_merger_window_size,
            num_windows_w,
            vit_merger_window_size,
        )
        index_padded = index_padded.permute(0, 1, 3, 2, 4).reshape(
            grid_t,
            num_windows_h * num_windows_w,
            vit_merger_window_size,
            vit_merger_window_size,
        )
        seqlens = (index_padded != -100).sum([2, 3]).reshape(-1)
        index_padded = index_padded.reshape(-1)
        index_new = index_padded[index_padded != -100]
        cu_seqlens_tmp = seqlens.cumsum(0) * self.spatial_merge_unit
        cu_seqlens_tmp = cu_seqlens_tmp.to(dtype=torch.int32)
        cu_seqlens_tmp = torch.unique_consecutive(cu_seqlens_tmp)

        return index_new, cu_seqlens_tmp

    def rotary_pos_emb_1d_thw(self, t, h, w):
        """Generate 1D-RoPE for spatial structure, preserving spatial ordering.
        
        Returns shape: (total_seq_len, spatial_merge_unit, rotary_dim)
        
        OpenCUA uses 1D-RoPE but must preserve spatial ordering like Qwen2.5-VL.
        This follows Qwen2.5-VL's rotary_pos_emb_thw pattern exactly, but uses
        1D sequential positions instead of 2D (H, W) positions.
        
        Key difference from Qwen2.5-VL:
        - Qwen2.5-VL: Uses 2D positions (hpos_ids, wpos_ids) -> rotary_pos_emb_full[pos_ids]
        - OpenCUA: Uses 1D sequential positions -> rotary_pos_emb_full[pos_1d]
        
        The spatial structure is preserved through row-major ordering:
        pos = t * (llm_h * llm_w) + h * llm_w + w
        
        CRITICAL: The first dimension must be total_seq_len (t * llm_h * llm_w) to match
        window_index_thw size, NOT total_seq_len // spatial_merge_unit.
        This matches Qwen2.5-VL's rotary_pos_emb_thw exactly.
        """
        logger.info(f"[OpenCUA DEBUG] rotary_pos_emb_1d_thw called: t={t}, h={h}, w={w}")
        llm_h = h // self.spatial_merge_size
        llm_w = w // self.spatial_merge_size
        total_seq_len = t * llm_h * llm_w
        logger.info(f"[OpenCUA DEBUG] rotary_pos_emb_1d_thw: llm_h={llm_h}, llm_w={llm_w}, total_seq_len={total_seq_len}")
        
        # Create 1D positions matching Qwen2.5-VL's spatial merge ordering exactly
        # Qwen2.5-VL uses hpos_ids and wpos_ids with specific reshape/permute pattern
        # We need to match this ordering for 1D-RoPE to preserve spatial relationships
        # 
        # Qwen2.5-VL pattern (using original h, w):
        # 1. Create hpos_ids and wpos_ids as 2D grids of size (h, w)
        # 2. Reshape to [h // spatial_merge_size, spatial_merge_size, w // spatial_merge_size, spatial_merge_size]
        # 3. Permute to [h // spatial_merge_size, w // spatial_merge_size, spatial_merge_size, spatial_merge_size]
        # 4. Flatten to get the spatial merge order
        #
        # For 1D-RoPE, we create positions that match this spatial merge order
        # Use original h, w like Qwen2.5-VL does
        hpos_ids = torch.arange(h).unsqueeze(1).expand(-1, w)
        wpos_ids = torch.arange(w).unsqueeze(0).expand(h, -1)
        
        # Match Qwen2.5-VL's reshape/permute pattern exactly
        hpos_ids_reshaped = (
            hpos_ids.reshape(
                h // self.spatial_merge_size,
                self.spatial_merge_size,
                w // self.spatial_merge_size,
                self.spatial_merge_size,
            )
            .permute(0, 2, 1, 3)
            .flatten()
        )
        wpos_ids_reshaped = (
            wpos_ids.reshape(
                h // self.spatial_merge_size,
                self.spatial_merge_size,
                w // self.spatial_merge_size,
                self.spatial_merge_size,
            )
            .permute(0, 2, 1, 3)
            .flatten()
        )
        
        # For 1D-RoPE, we use a single position value that preserves spatial ordering
        # We combine h and w positions into a single 1D position
        # CRITICAL: hpos_ids_reshaped and wpos_ids_reshaped are already in Qwen2.5-VL's
        # spatial merge order. We need to map them to llm_h and llm_w ranges.
        # 
        # The issue: hpos_ids_reshaped has values like [0, 0, 1, 1, ...] which when divided
        # by spatial_merge_size=2 gives [0, 0, 0, 0, ...], losing information.
        # 
        # Solution: Use hpos_ids_reshaped and wpos_ids_reshaped directly, but map them
        # to llm_h and llm_w ranges by taking unique values from each spatial_merge_unit group.
        # Each group of spatial_merge_unit elements has the same (hpos, wpos) values,
        # so we can take the first element of each group.
        
        # Reshape to group by spatial_merge_unit: [llm_h * llm_w, spatial_merge_unit]
        # CRITICAL: hpos_ids_reshaped and wpos_ids_reshaped are in spatial merge order,
        # but each spatial_merge_unit group may not have identical values.
        # We need to understand the actual pattern.
        hpos_ids_grouped = hpos_ids_reshaped.reshape(-1, self.spatial_merge_unit)
        wpos_ids_grouped = wpos_ids_reshaped.reshape(-1, self.spatial_merge_unit)
        
        # Debug: log the grouped values to understand the pattern
        if len(hpos_ids_grouped) > 0:
            logger.info(f"[OpenCUA DEBUG] rotary_pos_emb_1d_thw: hpos_ids_grouped[:4]={hpos_ids_grouped[:4].tolist()}, wpos_ids_grouped[:4]={wpos_ids_grouped[:4].tolist()}")
            # Also log more groups to see if pattern changes
            if len(hpos_ids_grouped) >= 8:
                logger.info(f"[OpenCUA DEBUG] rotary_pos_emb_1d_thw: hpos_ids_grouped[4:8]={hpos_ids_grouped[4:8].tolist()}, wpos_ids_grouped[4:8]={wpos_ids_grouped[4:8].tolist()}")
            # Log the original hpos_ids_reshaped pattern - check more values to see h variation
            logger.info(f"[OpenCUA DEBUG] rotary_pos_emb_1d_thw: hpos_ids_reshaped[:32]={hpos_ids_reshaped[:32].tolist()}, wpos_ids_reshaped[:32]={wpos_ids_reshaped[:32].tolist()}")
            # Check if h values change later - check at different llm_w positions
            if len(hpos_ids_reshaped) >= 256:
                logger.info(f"[OpenCUA DEBUG] rotary_pos_emb_1d_thw: hpos_ids_reshaped[128:160]={hpos_ids_reshaped[128:160].tolist()}, unique_h_count={len(torch.unique(hpos_ids_reshaped))}")
                # Check hpos_ids_grouped at different llm_w positions to see if h changes
                if len(hpos_ids_grouped) >= llm_w * 2:
                    logger.info(f"[OpenCUA DEBUG] rotary_pos_emb_1d_thw: hpos_ids_grouped[llm_w:llm_w+4]={hpos_ids_grouped[llm_w:llm_w+4].tolist()}, hpos_ids_grouped[llm_w*2:llm_w*2+4]={hpos_ids_grouped[llm_w*2:llm_w*2+4].tolist()}")
        
        # The pattern shows that within each group, values vary:
        # hpos_ids_grouped: [[0, 0, 1, 1], [0, 0, 1, 1], ...] for llm_h=0
        #                   [[2, 2, 3, 3], [2, 2, 3, 3], ...] for llm_h=1
        # wpos_ids_grouped: [[0, 1, 0, 1], [2, 3, 2, 3], ...]
        # 
        # For 1D-RoPE, we need to create a single position value per group.
        # The key insight: each group represents a spatial merge unit, and within
        # each group, the h and w values represent the original patch positions
        # before spatial merge. We need to map these to llm_h and llm_w ranges.
        # 
        # CRITICAL: Using min() loses information because [0, 0, 1, 1].min() = 0,
        # which when divided by spatial_merge_size=2 gives 0, losing the distinction
        # between different llm_h positions.
        # 
        # Solution: Use mean() to preserve information, but divide by spatial_merge_size
        # BEFORE converting to long to avoid information loss:
        # - [0, 0, 1, 1].mean() = 0.5 -> 0.5 / 2 = 0.25 -> long() = 0 (llm_h=0)
        # - [2, 2, 3, 3].mean() = 2.5 -> 2.5 / 2 = 1.25 -> long() = 1 (llm_h=1)
        # This correctly maps to llm_h and llm_w ranges.
        hpos_mean = hpos_ids_grouped.float().mean(dim=1)  # Keep as float
        wpos_mean = wpos_ids_grouped.float().mean(dim=1)  # Keep as float
        
        # Map to llm_h and llm_w ranges by dividing first, then converting to long
        hpos_mapped = (hpos_mean / self.spatial_merge_size).long()
        wpos_mapped = (wpos_mean / self.spatial_merge_size).long()
        
        # For debugging, also compute hpos_unique and wpos_unique (original values before mapping)
        hpos_unique = hpos_mean.long()
        wpos_unique = wpos_mean.long()
        
        # Calculate 1D positions: pos = hpos * llm_w + wpos
        pos_1d_per_frame = hpos_mapped * llm_w + wpos_mapped
        # pos_1d_per_frame shape: (llm_h * llm_w,)
        
        # Debug: log values to verify
        if len(pos_1d_per_frame) > 0:
            logger.info(f"[OpenCUA DEBUG] rotary_pos_emb_1d_thw: hpos_unique[:10]={hpos_unique[:10].tolist() if len(hpos_unique) >= 10 else hpos_unique.tolist()}, wpos_unique[:10]={wpos_unique[:10].tolist() if len(wpos_unique) >= 10 else wpos_unique.tolist()}")
            logger.info(f"[OpenCUA DEBUG] rotary_pos_emb_1d_thw: hpos_mapped[:10]={hpos_mapped[:10].tolist() if len(hpos_mapped) >= 10 else hpos_mapped.tolist()}, wpos_mapped[:10]={wpos_mapped[:10].tolist() if len(wpos_mapped) >= 10 else wpos_mapped.tolist()}")
            logger.info(f"[OpenCUA DEBUG] rotary_pos_emb_1d_thw: pos_1d_per_frame[:10]={pos_1d_per_frame[:10].tolist() if len(pos_1d_per_frame) >= 10 else pos_1d_per_frame.tolist()}, max={pos_1d_per_frame.max().item()}")
            # Check hpos_unique and hpos_mapped at different llm_w positions to verify h changes
            if len(hpos_unique) >= llm_w * 2:
                logger.info(f"[OpenCUA DEBUG] rotary_pos_emb_1d_thw: hpos_unique[llm_w:llm_w+10]={hpos_unique[llm_w:llm_w+10].tolist() if len(hpos_unique) >= llm_w+10 else 'N/A'}, hpos_unique[llm_w*2:llm_w*2+10]={hpos_unique[llm_w*2:llm_w*2+10].tolist() if len(hpos_unique) >= llm_w*2+10 else 'N/A'}")
                logger.info(f"[OpenCUA DEBUG] rotary_pos_emb_1d_thw: hpos_mapped[llm_w:llm_w+10]={hpos_mapped[llm_w:llm_w+10].tolist() if len(hpos_mapped) >= llm_w+10 else 'N/A'}, hpos_mapped[llm_w*2:llm_w*2+10]={hpos_mapped[llm_w*2:llm_w*2+10].tolist() if len(hpos_mapped) >= llm_w*2+10 else 'N/A'}")
                logger.info(f"[OpenCUA DEBUG] rotary_pos_emb_1d_thw: pos_1d_per_frame[llm_w:llm_w+10]={pos_1d_per_frame[llm_w:llm_w+10].tolist() if len(pos_1d_per_frame) >= llm_w+10 else 'N/A'}, pos_1d_per_frame[llm_w*2:llm_w*2+10]={pos_1d_per_frame[llm_w*2:llm_w*2+10].tolist() if len(pos_1d_per_frame) >= llm_w*2+10 else 'N/A'}")
        
        # CRITICAL: Qwen2.5-VL calculates positions from h * w (before spatial merge),
        # then groups by spatial_merge_unit. We need to do the same.
        # 
        # Qwen2.5-VL approach:
        # 1. Calculate positions for h * w patches (before merge) -> h * w positions
        # 2. Repeat for t frames -> t * h * w positions
        # 3. Group by spatial_merge_unit -> (t * h * w // spatial_merge_unit, spatial_merge_unit) = (t * llm_h * llm_w, spatial_merge_unit)
        #
        # Our approach should match:
        # 1. Calculate 1D positions for h * w patches (before merge) -> h * w positions
        # 2. Repeat for t frames -> t * h * w positions
        # 3. Group by spatial_merge_unit -> (t * h * w // spatial_merge_unit, spatial_merge_unit) = (t * llm_h * llm_w, spatial_merge_unit)
        #
        # The key insight: we need to calculate positions for the ORIGINAL h * w grid,
        # not the merged llm_h * llm_w grid. The positions should reflect the original
        # spatial structure before merging.
        #
        # hpos_ids_reshaped and wpos_ids_reshaped are already in Qwen2.5-VL's spatial merge order
        # and have length h * w. We can use them directly to create 1D positions.
        # For 1D-RoPE, we combine h and w into a single position: pos = hpos * w + wpos
        # This preserves the spatial ordering while using a single dimension.
        
        # Calculate 1D positions directly from hpos_ids_reshaped and wpos_ids_reshaped
        # This matches Qwen2.5-VL's approach but uses 1D positions instead of 2D
        # 
        # Qwen2.5-VL: Uses 2D positions (hpos_ids_reshaped, wpos_ids_reshaped) directly
        # with max(h, w) sized RoPE cache. For 1D-RoPE, we combine them into a single
        # 1D position: pos = hpos * max_size + wpos, where max_size = max(h, w)
        # This preserves the spatial ordering while using 1D positions.
        #
        # CRITICAL: Use max(h, w) like Qwen2.5-VL does, not the mapped llm_h/llm_w ranges.
        # This ensures the position values match Qwen2.5-VL's range exactly.
        max_size = max(h, w)
        pos_1d_per_frame_full = hpos_ids_reshaped * max_size + wpos_ids_reshaped
        # pos_1d_per_frame_full shape: (h * w,)
        
        # Debug: log pos_1d_per_frame_full values
        if len(pos_1d_per_frame_full) > 0:
            logger.info(f"[OpenCUA DEBUG] rotary_pos_emb_1d_thw: pos_1d_per_frame_full[:10]={pos_1d_per_frame_full[:10].tolist()}, max={pos_1d_per_frame_full.max().item()}, min={pos_1d_per_frame_full.min().item()}")
            if len(pos_1d_per_frame_full) >= 32:
                logger.info(f"[OpenCUA DEBUG] rotary_pos_emb_1d_thw: pos_1d_per_frame_full[128:160]={pos_1d_per_frame_full[128:160].tolist()}")
        
        # Repeat for each frame t
        pos_1d = pos_1d_per_frame_full.repeat(t)
        # pos_1d shape: (t * h * w,)
        
        # Generate RoPE embeddings for these positions
        # CRITICAL: For 1D-RoPE, pos_1d = hpos * max_size + wpos, so the maximum value
        # is (h-1) * max_size + (w-1), which is much larger than max_size.
        # We need to use the actual maximum position value, not max_size.
        max_pos = pos_1d.max().item() + 1 if len(pos_1d) > 0 else max_size
        logger.info(f"[OpenCUA DEBUG] rotary_pos_emb_1d_thw: max_size={max_size}, max_pos={max_pos}, pos_1d.max()={pos_1d.max().item() if len(pos_1d) > 0 else 'N/A'}")
        rotary_pos_emb_full = self.rotary_pos_emb_1d(max_pos)
        # rotary_pos_emb_full shape: (max_pos, rotary_dim)
        
        # Index into the full RoPE cache using our 1D positions
        rotary_pos_emb_1d = rotary_pos_emb_full[pos_1d]
        # rotary_pos_emb_1d shape: (t * h * w, rotary_dim)
        
        # Match Qwen2.5-VL's exact structure:
        # Qwen2.5-VL: [t * h * w, 2*rotary_dim] -> reshape to [t * h * w // spatial_merge_unit, spatial_merge_unit, 2*rotary_dim]
        # OpenCUA: [t * h * w, rotary_dim] -> reshape to [t * h * w // spatial_merge_unit, spatial_merge_unit, rotary_dim]
        # where t * h * w // spatial_merge_unit = t * llm_h * llm_w
        rotary_pos_emb_reshaped = rotary_pos_emb_1d.reshape(
            rotary_pos_emb_1d.shape[0] // self.spatial_merge_unit,
            self.spatial_merge_unit,
            -1,
        )  # (t * h * w // spatial_merge_unit, spatial_merge_unit, rotary_dim) = (t * llm_h * llm_w, spatial_merge_unit, rotary_dim)
        
        return rotary_pos_emb_reshaped

    @lru_cache(maxsize=1024)  # noqa: B019
    def get_rope_by_thw(self, t, h, w):
        """Generate 1D-RoPE based on total sequence length.
        
        Follows Qwen2.5-VL's get_rope_by_thw pattern exactly, but uses 1D-RoPE
        instead of 2D (H, W) RoPE. The key is to match the structure:
        1. Generate rotary_pos_emb_thw with shape (num_positions, spatial_merge_unit, rotary_dim)
        2. Apply window indexing: rotary_pos_emb_thw[window_index_thw, :, :]
        3. Flatten: flatten(start_dim=0, end_dim=1)
        
        IMPORTANT: Window indexing reorders tokens, so we must ensure RoPE positions
        match the reordered sequence to preserve spatial relationships.
        """
        window_index_thw, cu_seqlens_window_thw = self.get_window_index_thw(t, h, w)
        
        # Generate 1D-RoPE with spatial_merge_unit expansion
        # This matches Qwen2.5-VL's rotary_pos_emb_thw structure
        rotary_pos_emb_thw = self.rotary_pos_emb_1d_thw(t, h, w)
        # rotary_pos_emb_thw shape: (t * llm_h * llm_w, spatial_merge_unit, rotary_dim)
        
        # CRITICAL: window_index_thw has values in range [0, grid_t * llm_h * llm_w - 1],
        # and rotary_pos_emb_thw has first dimension = t * llm_h * llm_w (after our fix).
        # So they should match exactly now. Use window_index_thw directly like Qwen2.5-VL.
        
        # Debug logging to verify shapes match
        logger.info(f"[OpenCUA DEBUG] get_rope_by_thw: rotary_pos_emb_thw.shape={rotary_pos_emb_thw.shape}, window_index_thw.shape={window_index_thw.shape}, window_index_thw.max()={window_index_thw.max() if len(window_index_thw) > 0 else 'N/A'}, rotary_pos_emb_thw.shape[0]={rotary_pos_emb_thw.shape[0]}")
        
        # CRITICAL: Apply window indexing to reorder RoPE positions
        # window_index_thw contains the reordered indices after window partitioning.
        # We must index into rotary_pos_emb_thw using these indices to match the
        # reordered hidden states. This preserves spatial relationships after windowing.
        # This matches Qwen2.5-VL's behavior exactly.
        rotary_pos_emb_thw = rotary_pos_emb_thw[window_index_thw, :, :]
        # rotary_pos_emb_thw shape: (len(window_index_thw), spatial_merge_unit, rotary_dim)
        
        # Flatten: exactly like Qwen2.5-VL
        rotary_pos_emb_thw = rotary_pos_emb_thw.flatten(start_dim=0, end_dim=1)
        # rotary_pos_emb_thw shape: (len(window_index_thw) * spatial_merge_unit, rotary_dim)
        
        # cu_seqlens_thw represents the number of patches (before spatial merge)
        # This matches Qwen2.5-VL's implementation exactly
        # Note: h * w is the patch count, not the token count after spatial merge
        cu_seqlens_thw = torch.repeat_interleave(
            torch.tensor([h * w], dtype=torch.int32), t
        )
        return (
            rotary_pos_emb_thw,
            window_index_thw,
            cu_seqlens_window_thw,
            cu_seqlens_thw,
        )

    def compute_attn_mask_seqlen(
        self,
        cu_seqlens: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.attn_backend in {
            AttentionBackendEnum.FLASH_ATTN,
            AttentionBackendEnum.ROCM_AITER_FA,
        }:
            # Calculate max_seqlen and ensure it's int32 tensor
            # cu_seqlens is int32, so difference is int32, but max() may return float
            seq_diffs = cu_seqlens[1:] - cu_seqlens[:-1]
            if len(seq_diffs) > 0:
                max_seqlen = seq_diffs.max().to(torch.int32)
            else:
                max_seqlen = torch.tensor(0, device=cu_seqlens.device, dtype=torch.int32)
            seqlens = torch.zeros(1, device=cu_seqlens.device, dtype=torch.int32)
        elif self.attn_backend == AttentionBackendEnum.XFORMERS:
            seqlens = (cu_seqlens[1:] - cu_seqlens[:-1]).to(torch.int32)
            max_seqlen = torch.tensor(0, device=cu_seqlens.device, dtype=torch.int32)
        else:
            max_seqlen = torch.tensor(0, device=cu_seqlens.device, dtype=torch.int32)
            seqlens = torch.zeros(1, device=cu_seqlens.device, dtype=torch.int32)
        return max_seqlen, seqlens

    @staticmethod
    def invert_permutation(perm: torch.Tensor) -> torch.Tensor:
        inv = torch.empty_like(perm, pin_memory=is_pin_memory_available())
        inv[perm] = torch.arange(perm.numel(), device=perm.device, dtype=perm.dtype)
        return inv

    def forward(
        self,
        x: torch.Tensor,
        grid_thw: list[list[int]],
    ) -> torch.Tensor:
        logger.info(f"[OpenCUA DEBUG] VisionEncoder.forward called: x.shape={x.shape}, grid_thw={grid_thw}")
        seq_len, _ = x.size()
        rotary_pos_emb = []
        window_index: list = []
        cu_window_seqlens: list = [torch.tensor([0], dtype=torch.int32)]
        cu_seqlens: list = []

        hidden_states = x.to(device=self.device, dtype=self.dtype)
        hidden_states = self.patch_embed(hidden_states)

        window_index_id = 0
        cu_window_seqlens_last = 0
        for t, h, w in grid_thw:
            t, h, w = int(t), int(h), int(w)
            llm_h = h // self.spatial_merge_size
            llm_w = w // self.spatial_merge_size

            (
                rotary_pos_emb_thw,
                window_index_thw,
                cu_seqlens_window_thw,
                cu_seqlens_thw,
            ) = self.get_rope_by_thw(t, h, w)

            window_index.append(window_index_thw + window_index_id)
            window_index_id += t * llm_h * llm_w

            cu_seqlens_window_thw = cu_seqlens_window_thw + cu_window_seqlens_last
            cu_window_seqlens_last = cu_seqlens_window_thw[-1]
            cu_window_seqlens.append(cu_seqlens_window_thw)

            rotary_pos_emb.append(rotary_pos_emb_thw)

            cu_seqlens.append(cu_seqlens_thw)

        rotary_pos_emb = torch.cat(rotary_pos_emb)
        window_index = torch.cat(window_index)
        reverse_indices = self.invert_permutation(window_index)
        cu_window_seqlens = torch.cat(cu_window_seqlens)
        cu_window_seqlens = torch.unique_consecutive(cu_window_seqlens)
        cu_seqlens = torch.cat(cu_seqlens)
        cu_seqlens = torch.cumsum(cu_seqlens, dim=0, dtype=torch.int32)
        cu_seqlens = F.pad(cu_seqlens, (1, 0), "constant", 0)

        max_seqlen_full, seqlens_full = self.compute_attn_mask_seqlen(cu_seqlens)
        max_seqlen_window, seqlens_window = self.compute_attn_mask_seqlen(
            cu_window_seqlens
        )

        cu_seqlens = cu_seqlens.to(device=self.device, non_blocking=True)
        cu_window_seqlens = cu_window_seqlens.to(device=self.device, non_blocking=True)
        rotary_pos_emb = rotary_pos_emb.to(device=self.device, non_blocking=True)
        window_index = window_index.to(device=hidden_states.device, non_blocking=True)
        reverse_indices = reverse_indices.to(
            device=hidden_states.device, non_blocking=True
        )

        hidden_states = hidden_states.reshape(
            seq_len // self.spatial_merge_unit, self.spatial_merge_unit, -1
        )
        # CRITICAL: window_index has values in range [0, grid_t * llm_h * llm_w - 1],
        # which matches hidden_states' first dimension = seq_len // spatial_merge_unit.
        # This matches Qwen2.5-VL's behavior exactly - use window_index directly.
        hidden_states = hidden_states[window_index, :, :]
        hidden_states = hidden_states.reshape(seq_len, -1)

        hidden_states = hidden_states.unsqueeze(1)

        for layer_num, blk in enumerate(self.blocks):
            if layer_num in self.fullatt_block_indexes:
                cu_seqlens_now = cu_seqlens
                max_seqlen_now = max_seqlen_full
                seqlens_now = seqlens_full
            else:
                cu_seqlens_now = cu_window_seqlens
                max_seqlen_now = max_seqlen_window
                seqlens_now = seqlens_window

            hidden_states = blk(
                hidden_states,
                cu_seqlens=cu_seqlens_now,
                rotary_pos_emb=rotary_pos_emb,
                max_seqlen=max_seqlen_now,
                seqlens=seqlens_now,
            )

        if hidden_states.dtype == torch.float16:
            hidden_states = cast_overflow_tensors(hidden_states)

        hidden_states = self.merger(hidden_states)
        hidden_states = hidden_states[reverse_indices, :]
        logger.info(f"[OpenCUA DEBUG] VisionEncoder.forward output: hidden_states.shape={hidden_states.shape}")
        return hidden_states

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        stacked_params_mapping = [
            ("attn.qkv.", "attn.q.", "q"),
            ("attn.qkv.", "attn.k.", "k"),
            ("attn.qkv.", "attn.v.", "v"),
            ("mlp.gate_up_proj.", "mlp.gate_proj.", 0),
            ("mlp.gate_up_proj.", "mlp.up_proj.", 1),
        ]
        params_dict = dict(self.named_parameters(remove_duplicate=False))
        loaded_params: set[str] = set()

        for name, loaded_weight in weights:
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
            loaded_params.add(name)
        return loaded_params


class OpenCUAProcessingInfo(Qwen2VLProcessingInfo):
    """Processing info for OpenCUA models."""

    def get_hf_config(self):
        """Load OpenCUAConfig.
        
        When trust_remote_code=True, HuggingFace loads the config class
        from the model repository, which may differ from vLLM's OpenCUAConfig.
        We skip type validation to allow both cases.
        """
        # Skip type validation to support trust_remote_code configs
        return self.ctx.get_hf_config(None)

    def get_hf_processor(self, **kwargs: object):
        """Load OpenCUA processor.
        
        OpenCUA uses TikTokenV3 tokenizer, which is incompatible with
        Qwen2VLProcessor's expected Qwen2Tokenizer. We construct a custom processor
        similar to Tarsier2Processor.
        """
        # Get OpenCUA's actual tokenizer
        tokenizer = self.get_tokenizer()
        
        # Get image processor config and create Qwen2-VL processors
        vision_config = self.ctx.get_hf_image_processor_config()
        
        # Use custom processor class that accepts any tokenizer type
        return OpenCUAProcessor(
            vision_config=vision_config,
            tokenizer=tokenizer,
            **kwargs,
        )


class OpenCUAProcessor(Qwen2VLProcessor):
    """Custom processor for OpenCUA that accepts TikTokenV3 tokenizer."""
    
    def check_argument_for_proper_class(self, attribute_name: str, arg: object) -> None:
        """Override to bypass type validation for tokenizer."""
        # Skip type checking for tokenizer to allow TikTokenV3
        if attribute_name == "tokenizer":
            return
        # Call parent's check for other attributes
        return super().check_argument_for_proper_class(attribute_name, arg)
    
    def __init__(
        self,
        vision_config: dict,
        tokenizer: AnyTokenizer,
        **kwargs,
    ):
        # Create processors
        image_processor = Qwen2VLImageProcessor(**vision_config)
        video_processor = Qwen2VLVideoProcessor(**vision_config)
        chat_template = kwargs.pop("chat_template", None)
        
        # Call super().__init__ - our check_argument_for_proper_class will bypass tokenizer validation
        super().__init__(
            image_processor=image_processor,
            tokenizer=tokenizer,
            video_processor=video_processor,
            chat_template=chat_template,
            **kwargs,
        )
        
        # Set image_token and video_token attributes required by Qwen2VLProcessor
        # These are the default tokens used by Qwen2-VL models
        self.image_token = "<|image_pad|>"
        self.video_token = "<|video_pad|>"
    
    def __call__(
        self,
        text=None,
        images=None,
        videos=None,
        return_tensors=None,
        **kwargs,
    ):
        """Override __call__ to ensure compatibility with TikTokenV3 tokenizer.
        
        This method directly calls tokenizer and image/video processors
        to avoid any compatibility issues with TikTokenV3.
        """
        # Process text with tokenizer
        if text is not None:
            # Ensure text is a list
            if not isinstance(text, list):
                text = [text]
            # Call tokenizer directly - TikTokenV3 should support standard interface
            text_inputs = self.tokenizer(text, **kwargs)
        else:
            text_inputs = {}
        
        # Process images with image processor
        image_inputs = {}
        if images is not None:
            if not isinstance(images, list):
                images = [images]
            if len(images) > 0:
                image_inputs = self.image_processor(images, return_tensors=return_tensors or "pt")
        
        # Process videos with video processor
        video_inputs = {}
        if videos is not None:
            if not isinstance(videos, list):
                videos = [videos]
            if len(videos) > 0:
                video_inputs = self.video_processor(videos, return_tensors=return_tensors or "pt")
        
        # Combine all inputs
        combined_inputs = {**text_inputs, **image_inputs, **video_inputs}
        
        return BatchFeature(combined_inputs, tensor_type=return_tensors)


class OpenCUAMultiModalProcessor(BaseMultiModalProcessor[OpenCUAProcessingInfo]):
    """Multi-modal processor for OpenCUA using transformers processor."""

    def _get_data_parser(self) -> MultiModalDataParser:
        return Qwen2VLMultiModalDataParser(
            self.info.get_hf_config().vision_config.spatial_merge_size
        )

    def _get_mm_fields_config(
        self,
        hf_inputs: BatchFeature,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        return _create_qwen2vl_field_factory(
            self.info.get_hf_config().vision_config.spatial_merge_size
        )(hf_inputs)

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, Any],
        out_mm_kwargs: MultiModalKwargs,
    ) -> Sequence[PromptUpdate]:
        """Get prompt updates for OpenCUA.
        
        Follows Qwen2.5-VL's pattern exactly. The key is to match the placeholder
        token that appears in the tokenized prompt after chat_template processing.
        OpenCUA uses <|media_placeholder|> which gets tokenized to media_placeholder_token_id.
        """
        logger.info(f"[OpenCUA DEBUG] _get_prompt_updates called: mm_items={len(mm_items) if mm_items else 0} items")
        hf_processor = self.info.get_hf_processor(**hf_processor_mm_kwargs)
        image_processor = self.info.get_image_processor(**hf_processor_mm_kwargs)
        tokenizer = self.info.get_tokenizer()
        vocab = tokenizer.get_vocab()
        hf_config = self.info.get_hf_config()

        # OpenCUA's chat_template uses <|media_placeholder|> for images
        # This is what actually appears in the tokenized prompt
        # Get the token ID from vocab first, then fallback to config
        media_placeholder_str = "<|media_placeholder|>"
        media_placeholder_token_id = vocab.get(
            media_placeholder_str,
            getattr(hf_config, "media_placeholder_token_id", 151664)
        )
        
        # For video, use the processor's video_token or fallback
        video_token_str = getattr(hf_processor, "video_token", "<|video_pad|>")
        video_token_id = vocab.get(
            video_token_str,
            getattr(hf_config, "video_token_id", 151656)
        )
        
        placeholder = {
            "image": media_placeholder_token_id,
            "video": video_token_id,
        }

        merge_length = image_processor.merge_size**2

        def get_replacement_opencua(item_idx: int, modality: str):
            """Calculate replacement tokens for a multimodal item.
            
            This must match Qwen2.5-VL's calculation exactly:
            num_tokens = (T * H * W) // (spatial_merge_size^2)
            """
            out_item = out_mm_kwargs[modality][item_idx]
            grid_thw = out_item[f"{modality}_grid_thw"].data
            assert isinstance(grid_thw, torch.Tensor)

            # Calculate number of tokens: total patches divided by merge_length
            # This matches Qwen2.5-VL's calculation
            num_tokens = int(grid_thw.prod()) // merge_length
            return [placeholder[modality]] * num_tokens

        return [
            PromptReplacement(
                modality=modality,
                target=[placeholder[modality]],
                replacement=partial(get_replacement_opencua, modality=modality),
            )
            for modality in ("image", "video")
        ]


class OpenCUADummyInputsBuilder(Qwen2VLDummyInputsBuilder):
    """Dummy inputs builder for OpenCUA that uses <|media_placeholder|> instead of <|image_pad|>."""
    
    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        num_images = mm_counts.get("image", 0)
        num_videos = mm_counts.get("video", 0)
        
        # OpenCUA's chat_template uses <|media_placeholder|> for images
        # This must match what actually appears in the prompt after chat_template processing
        image_token = "<|media_placeholder|>"
        video_token = "<|video_pad|>"  # Keep video token as is for now
        
        return image_token * num_images + video_token * num_videos


@MULTIMODAL_REGISTRY.register_processor(
    OpenCUAMultiModalProcessor,
    info=OpenCUAProcessingInfo,
    dummy_inputs=OpenCUADummyInputsBuilder,
)
class OpenCUAForConditionalGeneration(
    nn.Module,
    SupportsMultiModal,
    SupportsLoRA,
    SupportsPP,
    SupportsQuant,
    SupportsEagle3,
    SupportsMRoPE,
):
    """OpenCUA-7B model with 1D-RoPE for vision encoder."""

    merge_by_field_config = True
    multimodal_cpu_fields = {"image_grid_thw", "video_grid_thw"}

    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
    }

    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_prefix={
            "model.language_model.": "language_model.model.",
            "model.visual.": "visual.",
            "vision_tower.": "visual.",  # Map vision_tower to visual (for some checkpoints)
            "lm_head.": "language_model.lm_head.",
            "model.": "language_model.model.",
        }
    )

    supports_encoder_tp_data = True

    def get_mrope_input_positions(
        self,
        input_tokens: list[int],
        mm_features: list[MultiModalFeatureSpec],
    ) -> tuple[torch.Tensor, int]:
        """Get M-RoPE input positions for OpenCUA.
        
        Follows Qwen2.5-VL's pattern exactly for consistency.
        OpenCUA uses 1D-RoPE in vision encoder, but LLM still needs 3D positions.
        The key is to match Qwen2.5-VL's token detection and position calculation.
        """
        logger.info(f"[OpenCUA DEBUG] ===== get_mrope_input_positions CALLED ===== input_tokens len={len(input_tokens)}, mm_features len={len(mm_features)}")
        kwargs = MultiModalFeatureSpec.gather_kwargs(
            mm_features,
            {"image_grid_thw", "video_grid_thw"},
        )
        image_grid_thw = [item.tolist() for item in kwargs.get("image_grid_thw", [])]
        video_grid_thw = [item.tolist() for item in kwargs.get("video_grid_thw", [])]
        logger.info(f"[OpenCUA DEBUG] get_mrope_input_positions: image_grid_thw={image_grid_thw}, video_grid_thw={video_grid_thw}")

        hf_config = self.config
        # OpenCUA uses media_placeholder_token_id, but we need to handle it like Qwen2.5-VL
        media_placeholder_token_id = getattr(
            hf_config, "media_placeholder_token_id", 151664
        )
        image_token_id = getattr(hf_config, "image_token_id", media_placeholder_token_id)
        video_token_id = getattr(hf_config, "video_token_id", media_placeholder_token_id)
        vision_start_token_id = getattr(
            hf_config, "vision_start_token_id", media_placeholder_token_id
        )
        spatial_merge_size = hf_config.vision_config.spatial_merge_size

        input_tokens_tensor = torch.tensor(input_tokens)
        # Follow Qwen2.5-VL pattern: find vision_start_token_id first
        # OpenCUA may use media_placeholder_token_id directly, so check both
        vision_start_indices = torch.argwhere(
            (input_tokens_tensor == vision_start_token_id)
            | (input_tokens_tensor == media_placeholder_token_id)
        ).squeeze(1)
        
        # Get the token after vision_start (or the media token itself)
        if len(vision_start_indices) > 0:
            # If we found vision_start, next token is the media token
            vision_tokens = input_tokens_tensor[
                torch.clamp(vision_start_indices + 1, max=len(input_tokens_tensor) - 1)
            ]
            # Also check if the token itself is media_placeholder
            vision_tokens_alt = input_tokens_tensor[vision_start_indices]
            # Combine both checks
            image_nums = ((vision_tokens == image_token_id) | (vision_tokens_alt == media_placeholder_token_id)).sum().item()
            video_nums = (vision_tokens == video_token_id).sum().item()
        else:
            # Fallback: count media_placeholder_token_id directly
            image_nums = (input_tokens_tensor == media_placeholder_token_id).sum().item()
            video_nums = 0
        
        # Ensure counts match grid_thw
        image_nums = min(image_nums, len(image_grid_thw))
        video_nums = min(video_nums, len(video_grid_thw))
        
        llm_pos_ids_list: list = []
        st = 0
        remain_images, remain_videos = image_nums, video_nums
        image_index, video_index = 0, 0
        last_vision_position = -1  # Track last vision position for next text segment

        for _ in range(image_nums + video_nums):
            # Find next image or video token, following Qwen2.5-VL pattern
            ed_image = len(input_tokens) + 1
            ed_video = len(input_tokens) + 1
            
            if remain_images > 0 and image_index < len(image_grid_thw):
                # Try to find media_placeholder_token_id or image_token_id
                try:
                    ed_media = input_tokens.index(media_placeholder_token_id, st)
                except ValueError:
                    ed_media = len(input_tokens) + 1
                try:
                    ed_img = input_tokens.index(image_token_id, st)
                except ValueError:
                    ed_img = len(input_tokens) + 1
                ed_image = min(ed_media, ed_img)
            
            if remain_videos > 0 and video_index < len(video_grid_thw):
                try:
                    ed_video = input_tokens.index(video_token_id, st)
                except ValueError:
                    ed_video = len(input_tokens) + 1
            
            # Both are invalid, break early
            if ed_image > len(input_tokens) and ed_video > len(input_tokens):
                logger.warning(f"[OpenCUA DEBUG] No more tokens found at st={st}, breaking loop")
                break
            
            if ed_image <= ed_video and image_index < len(image_grid_thw):
                t, h, w = image_grid_thw[image_index]
                image_index += 1
                remain_images -= 1
                ed = ed_image
            elif video_index < len(video_grid_thw):
                t, h, w = video_grid_thw[video_index]
                video_index += 1
                remain_videos -= 1
                ed = ed_video
            else:
                # No valid token found, break
                logger.warning(f"[OpenCUA DEBUG] No valid token found, breaking loop")
                break

            llm_grid_t, llm_grid_h, llm_grid_w = (
                t,
                h // spatial_merge_size,
                w // spatial_merge_size,
            )
            text_len = ed - st
            logger.info(f"[OpenCUA DEBUG] get_mrope_input_positions: processing media at ed={ed}, st={st}, text_len={text_len}, grid_thw=({t},{h},{w}) -> llm_grid=({llm_grid_t},{llm_grid_h},{llm_grid_w})")

            st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
            logger.info(f"[OpenCUA DEBUG] get_mrope_input_positions: text segment st_idx={st_idx}, text_len={text_len}")
            llm_pos_ids_list.append(
                torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx
            )

            # For OpenCUA: LLM still needs 3D positions for M-RoPE
            # Even though vision encoder uses 1D-RoPE, the LLM's M-RoPE requires (T, H, W) positions
            # Follow Qwen2.5-VL's pattern exactly for 3D position calculation
            t_index = (
                torch.arange(llm_grid_t)
                .view(-1, 1)
                .expand(-1, llm_grid_h * llm_grid_w)
                .flatten()
            )
            h_index = (
                torch.arange(llm_grid_h)
                .view(1, -1, 1)
                .expand(llm_grid_t, -1, llm_grid_w)
                .flatten()
            )
            w_index = (
                torch.arange(llm_grid_w)
                .view(1, 1, -1)
                .expand(llm_grid_t, llm_grid_h, -1)
                .flatten()
            )
            vision_st_idx = st_idx + text_len
            num_vision_tokens = llm_grid_t * llm_grid_h * llm_grid_w
            vision_positions = torch.stack([t_index, h_index, w_index]) + vision_st_idx
            logger.info(f"[OpenCUA DEBUG] get_mrope_input_positions: vision segment st_idx={vision_st_idx}, num_vision_tokens={num_vision_tokens}, vision_positions.shape={vision_positions.shape}, vision_positions.max()={vision_positions.max().item()}")
            llm_pos_ids_list.append(vision_positions)
            
            # Update st: ed is the position of placeholder token in input_tokens
            # After placeholder, we have num_vision_tokens virtual positions
            # So next text starts after ed + num_vision_tokens
            logger.info(f"[OpenCUA DEBUG] get_mrope_input_positions: added {num_vision_tokens} vision tokens, next st={ed + num_vision_tokens}")
            st = ed + num_vision_tokens
            # Track the last position after vision tokens for next text segment
            # Vision positions are 3D (T, H, W), so we need to calculate the sequential position
            # The last vision token position is vision_st_idx + num_vision_tokens - 1
            last_vision_position = vision_st_idx + num_vision_tokens - 1

        if st < len(input_tokens):
            # Calculate st_idx from the last vision position
            # For 3D positions, we need to use the sequential position, not the max of 3D coordinates
            if len(llm_pos_ids_list) > 0:
                # Use the last vision position we tracked, or fallback to max if not tracked
                st_idx = last_vision_position + 1
                logger.info(f"[OpenCUA DEBUG] get_mrope_input_positions: last_vision_position={last_vision_position}, calculated st_idx={st_idx}")
            else:
                st_idx = 0
            text_len = len(input_tokens) - st
            logger.info(f"[OpenCUA DEBUG] get_mrope_input_positions: final text segment st={st}, text_len={text_len}, st_idx={st_idx}")
            llm_pos_ids_list.append(
                torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx
            )

        if len(llm_pos_ids_list) == 0:
            # No multimodal tokens found, return empty positions
            llm_positions = torch.zeros((3, len(input_tokens)), dtype=torch.long)
            mrope_position_delta = 0
            logger.warning(f"[OpenCUA DEBUG] get_mrope_input_positions: No multimodal tokens found!")
        else:
            llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
            mrope_position_delta = (llm_positions.max() + 1 - len(input_tokens)).item()
            logger.info(f"[OpenCUA DEBUG] get_mrope_input_positions: final llm_positions.shape={llm_positions.shape}, mrope_position_delta={mrope_position_delta}, max_pos={llm_positions.max().item()}, input_tokens_len={len(input_tokens)}")

        return llm_positions, mrope_position_delta

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None:
        """Get placeholder string for OpenCUA.
        
        OpenCUA uses <|media_placeholder|> for images (matching HF repo).
        This is what appears in the chat_template after processing.
        """
        if modality.startswith("image"):
            # OpenCUA uses <|media_placeholder|> instead of <|image_pad|>
            return "<|media_placeholder|>"
        if modality.startswith("video"):
            # Keep video token as is for now
            return "<|video_pad|>"
        raise ValueError("Only image or video modality is supported")

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        logger.info(f"[OpenCUA DEBUG] OpenCUAForConditionalGeneration.__init__ called: prefix={prefix}")
        super().__init__()
        # Use hf_config from vllm_config (already loaded with trust_remote_code)
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        multimodal_config = vllm_config.model_config.multimodal_config

        self.use_data_parallel = multimodal_config.mm_encoder_tp_mode == "data"
        self.config = config
        self.vllm_config = vllm_config
        self.multimodal_config = multimodal_config
        self.quant_config = quant_config

        if multimodal_config.get_limit_per_prompt(
            "image"
        ) or multimodal_config.get_limit_per_prompt("video"):
            attn_backend_override = (
                multimodal_config.mm_encoder_attn_backend
                if multimodal_config is not None
                else None
            )
            # Use Qwen2_5_VisionTransformer directly, matching original OpenCUA
            self.visual = Qwen2_5_VisionTransformer(
                vision_config=config.vision_config,
                norm_eps=getattr(config, "rms_norm_eps", 1e-6),
                quant_config=self.quant_config,
                prefix=maybe_prefix(prefix, "visual"),
                use_data_parallel=self.use_data_parallel,
                attn_backend_override=attn_backend_override,
            )
            logger.info(f"[OpenCUA DEBUG] OpenCUAForConditionalGeneration.__init__: Created Qwen2_5_VisionTransformer, type={type(self.visual).__name__}")
        else:
            self.visual = None

        self.language_model = init_vllm_registered_model(
            vllm_config=vllm_config,
            hf_config=config.text_config,  # Use text_config from OpenCUA's main config
            prefix=maybe_prefix(prefix, "language_model"),
            architectures=["Qwen2ForCausalLM"],
        )

        self.make_empty_intermediate_tensors = (
            self.language_model.make_empty_intermediate_tensors
        )

    def set_aux_hidden_state_layers(self, layers: tuple[int, ...]) -> None:
        self.language_model.model.aux_hidden_state_layers = layers

    def get_eagle3_aux_hidden_state_layers(self) -> tuple[int, ...]:
        num_layers = len(self.language_model.model.layers)
        return (2, num_layers // 2, num_layers - 3)

    def _parse_and_validate_image_input(
        self, **kwargs: object
    ) -> Qwen2_5_VLImageInputs | None:
        pixel_values = kwargs.pop("pixel_values", None)
        image_embeds = kwargs.pop("image_embeds", None)
        image_grid_thw = kwargs.pop("image_grid_thw", None)

        if pixel_values is None and image_embeds is None:
            return None

        if pixel_values is not None:
            return Qwen2_5_VLImagePixelInputs(
                type="pixel_values",
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
            )

        if image_embeds is not None:
            return Qwen2_5_VLImageEmbeddingInputs(
                type="image_embeds",
                image_embeds=image_embeds,
                image_grid_thw=image_grid_thw,
            )

    def _parse_and_validate_video_input(
        self, **kwargs: object
    ) -> Qwen2_5_VLVideoInputs | None:
        pixel_values_videos = kwargs.pop("pixel_values_videos", None)
        video_embeds = kwargs.pop("video_embeds", None)
        video_grid_thw = kwargs.pop("video_grid_thw", None)

        if pixel_values_videos is None and video_embeds is None:
            return None

        if pixel_values_videos is not None:
            return Qwen2_5_VLVideoPixelInputs(
                type="pixel_values_videos",
                pixel_values_videos=pixel_values_videos,
                video_grid_thw=video_grid_thw,
            )

        if video_embeds is not None:
            return Qwen2_5_VLVideoEmbeddingInputs(
                type="video_embeds",
                video_embeds=video_embeds,
                video_grid_thw=video_grid_thw,
            )

    def _process_image_input(
        self, image_input: Qwen2_5_VLImageInputs
    ) -> tuple[torch.Tensor, ...]:
        grid_thw = image_input["image_grid_thw"]
        if grid_thw is None:
            raise ValueError("image_grid_thw is required for image input")
        assert grid_thw.ndim == 2
        grid_thw_list = grid_thw.tolist()

        if self.visual is None:
            raise ValueError("Visual encoder is not initialized")
        
        if image_input["type"] == "image_embeds":
            image_embeds = image_input["image_embeds"].type(self.visual.dtype)
        else:
            pixel_values = image_input["pixel_values"]
            with set_forward_context(None, self.vllm_config):
                if self.use_data_parallel:
                    # OpenCUA uses 1D-RoPE internally, but run_dp_sharded_mrope_vision_model
                    # only supports "rope_3d" or "rope_2d". Since OpenCUA's vision model
                    # handles 1D-RoPE internally via get_rope_by_thw, we use "rope_3d"
                    # which matches the structure (T, H, W) even though RoPE is 1D.
                    # The actual RoPE generation happens inside OpenCUAVisionTransformer.
                    return run_dp_sharded_mrope_vision_model(
                        self.visual, pixel_values, grid_thw_list, rope_type="rope_3d"
                    )
                else:
                    image_embeds = self.visual(pixel_values, grid_thw=grid_thw_list)

        merge_size = self.visual.spatial_merge_size
        sizes = (grid_thw.prod(-1) // merge_size // merge_size).tolist()
        image_embeds_split = image_embeds.split(sizes)
        logger.info(f"[OpenCUA DEBUG] _process_image_input: image_embeds split into {len(image_embeds_split)} parts, sizes={sizes}, total tokens={sum(sizes)}")
        return image_embeds_split

    def _process_video_input(
        self, video_input: Qwen2_5_VLVideoInputs
    ) -> tuple[torch.Tensor, ...]:
        grid_thw = video_input["video_grid_thw"]
        if grid_thw is None:
            raise ValueError("video_grid_thw is required for video input")
        assert grid_thw.ndim == 2
        grid_thw_list = grid_thw.tolist()

        if self.visual is None:
            raise ValueError("Visual encoder is not initialized")
        
        if video_input["type"] == "video_embeds":
            video_embeds = video_input["video_embeds"].type(self.visual.dtype)
        else:
            pixel_values_videos = video_input["pixel_values_videos"]
            with set_forward_context(None, self.vllm_config):
                if self.use_data_parallel:
                    # OpenCUA uses 1D-RoPE internally, but run_dp_sharded_mrope_vision_model
                    # only supports "rope_3d" or "rope_2d". Since OpenCUA's vision model
                    # handles 1D-RoPE internally via get_rope_by_thw, we use "rope_3d"
                    # which matches the structure (T, H, W) even though RoPE is 1D.
                    # The actual RoPE generation happens inside OpenCUAVisionTransformer.
                    return run_dp_sharded_mrope_vision_model(
                        self.visual,
                        pixel_values_videos,
                        grid_thw_list,
                        rope_type="rope_3d",
                    )
                else:
                    video_embeds = self.visual(
                        pixel_values_videos, grid_thw=grid_thw_list
                    )

        merge_size = self.visual.spatial_merge_size
        sizes = (grid_thw.prod(-1) // merge_size // merge_size).tolist()
        return video_embeds.split(sizes)

    def _parse_and_validate_multimodal_inputs(self, **kwargs: object) -> dict:
        mm_input_by_modality = {}

        for input_key in kwargs:
            if (
                input_key in ("pixel_values", "image_embeds")
                and "image" not in mm_input_by_modality
            ):
                mm_input_by_modality["image"] = self._parse_and_validate_image_input(
                    **kwargs
                )
            if (
                input_key in ("pixel_values_videos", "video_embeds")
                and "video" not in mm_input_by_modality
            ):
                mm_input_by_modality["video"] = self._parse_and_validate_video_input(
                    **kwargs
                )
        return mm_input_by_modality

    def get_language_model(self) -> torch.nn.Module:
        return self.language_model

    def embed_multimodal(self, **kwargs: object) -> MultiModalEmbeddings:
        mm_input_by_modality = self._parse_and_validate_multimodal_inputs(**kwargs)
        if not mm_input_by_modality:
            return []

        multimodal_embeddings: tuple[torch.Tensor, ...] = ()

        for modality in mm_input_by_modality:
            multimodal_input = mm_input_by_modality[modality]
            if modality == "image":
                image_embeddings = self._process_image_input(multimodal_input)
                multimodal_embeddings += tuple(image_embeddings)
            if modality == "video":
                video_embeddings = self._process_video_input(multimodal_input)
                multimodal_embeddings += tuple(video_embeddings)
        return multimodal_embeddings

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: object,
    ) -> torch.Tensor | IntermediateTensors:
        """Run forward pass for OpenCUA."""
        # Only log first forward call (prompt processing), not every generation step
        if positions is not None and positions.shape[0] > 10:
            logger.info(f"[OpenCUA DEBUG] OpenCUAForConditionalGeneration.forward (prompt): input_ids.shape={input_ids.shape if input_ids is not None else None}, positions.shape={positions.shape if positions is not None else None}")

        if intermediate_tensors is not None:
            inputs_embeds = None

        hidden_states = self.language_model.model(
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
        skip_prefixes = []
        if self.visual is None:
            skip_prefixes.extend(["visual."])
        loader = AutoWeightsLoader(self, skip_prefixes=skip_prefixes)
        return loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)

    def get_mm_mapping(self) -> MultiModelKeys:
        """Get the module prefix in multimodal models."""
        return MultiModelKeys.from_string_field(
            language_model="language_model",
            connector="visual.merger.",
            tower_model="visual.",
        )

