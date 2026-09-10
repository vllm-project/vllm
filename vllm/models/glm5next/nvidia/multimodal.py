# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GLM-5.3-Flash vision tower and multimodal processor."""

from collections.abc import Mapping
from functools import cached_property, partial

import numpy as np
import torch
import torch.nn as nn
from einops import rearrange

from vllm.distributed import (
    get_tensor_model_parallel_world_size,
    parallel_state,
)
from vllm.distributed import utils as dist_utils
from vllm.model_executor.layers.activation import SiluAndMulWithClamp
from vllm.model_executor.layers.attention import MMEncoderAttention
from vllm.model_executor.layers.conv import Conv2dLayer, Conv3dLayer
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.rotary_embedding.common import ApplyRotaryEmb
from vllm.model_executor.models.glm4_1v import (
    Glm4vMultiModalProcessor,
    Glm4vProcessingInfo,
)
from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    WeightsMapper,
)
from vllm.model_executor.models.vision import (
    get_vit_attn_backend,
    is_vit_use_data_parallel,
)
from vllm.models.common.ops import fused_q_kv_rmsnorm
from vllm.multimodal.parse import ImageSize, MultiModalDataItems
from vllm.v1.attention.backends.registry import AttentionBackendEnum


class Glm5NextVisionPatchEmbed(nn.Module):
    def __init__(
        self,
        patch_size: int = 14,
        temporal_patch_size: int = 1,
        in_channels: int = 3,
        hidden_size: int = 1536,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.temporal_patch_size = temporal_patch_size
        self.hidden_size = hidden_size

        kernel_size = (temporal_patch_size, patch_size, patch_size)
        self.proj = Conv3dLayer(
            in_channels,
            hidden_size,
            kernel_size=kernel_size,
            stride=kernel_size,
            bias=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        L, C = x.shape
        x = x.view(L, -1, self.temporal_patch_size, self.patch_size, self.patch_size)
        x = self.proj(x).view(L, self.hidden_size)
        return x


class Glm5NextVisionMLP(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        swiglu_limit: float,
        bias: bool = True,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        super().__init__()
        use_data_parallel = is_vit_use_data_parallel()
        self.gate_up_proj = MergedColumnParallelLinear(
            input_size=in_features,
            output_sizes=[hidden_features] * 2,
            bias=bias,
            quant_config=quant_config,
            prefix=f"{prefix}.gate_up_proj",
            disable_tp=use_data_parallel,
        )
        self.down_proj = RowParallelLinear(
            hidden_features,
            in_features,
            bias=bias,
            quant_config=quant_config,
            prefix=f"{prefix}.down_proj",
            disable_tp=use_data_parallel,
        )
        # GLM-5.3-Flash clamps the vision SwiGLU gate/up unlike GLM-OCR/GLM-4V.
        self.act_fn = SiluAndMulWithClamp(swiglu_limit=swiglu_limit)

    def forward(self, x: torch.Tensor):
        x, _ = self.gate_up_proj(x)
        x = self.act_fn(x)
        x, _ = self.down_proj(x)
        return x


class Glm5NextVisionAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        projection_size: int,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        use_data_parallel = is_vit_use_data_parallel()
        self.tp_size = (
            1 if use_data_parallel else get_tensor_model_parallel_world_size()
        )
        self.tp_rank = (
            0 if use_data_parallel else parallel_state.get_tensor_model_parallel_rank()
        )
        self.hidden_size_per_attention_head = dist_utils.divide(
            projection_size, num_heads
        )
        self.num_attention_heads_per_partition = dist_utils.divide(
            num_heads, self.tp_size
        )

        self.head_dim = embed_dim // num_heads

        # q/k norm eps hard-coded 1e-5 — distinct from block/post norm eps.
        self.q_norm = RMSNorm(self.head_dim, eps=1e-5)
        self.k_norm = RMSNorm(self.head_dim, eps=1e-5)

        self.qkv = QKVParallelLinear(
            hidden_size=embed_dim,
            head_size=self.hidden_size_per_attention_head,
            total_num_heads=num_heads,
            total_num_kv_heads=num_heads,
            bias=True,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj" if quant_config else f"{prefix}.qkv",
            disable_tp=use_data_parallel,
        )
        self.proj = RowParallelLinear(
            input_size=projection_size,
            output_size=embed_dim,
            quant_config=quant_config,
            prefix=f"{prefix}.proj",
            bias=True,
            disable_tp=use_data_parallel,
        )

        self.attn = MMEncoderAttention(
            num_heads=self.num_attention_heads_per_partition,
            head_size=self.hidden_size_per_attention_head,
            scale=self.hidden_size_per_attention_head**-0.5,
            prefix=f"{prefix}.attn",
        )
        self.apply_rotary_emb = ApplyRotaryEmb(enforce_enable=True)

    def split_qkv(self, qkv: torch.Tensor) -> tuple[torch.Tensor, ...]:
        seq_len, bs, _ = qkv.shape
        q, k, v = qkv.chunk(3, dim=2)
        new_shape = (
            seq_len,
            bs,
            self.num_attention_heads_per_partition,
            self.hidden_size_per_attention_head,
        )
        q, k, v = (x.view(*new_shape) for x in (q, k, v))
        return q, k, v

    def forward(
        self,
        x: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rotary_pos_emb_cos: torch.Tensor,
        rotary_pos_emb_sin: torch.Tensor,
        max_seqlen: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x, _ = self.qkv(x)
        q, k, v = self.split_qkv(x)

        # P1: fused q/k RMSNorm (two distinct weights, one launch; fp32, bit-identical).
        q_shape, k_shape = q.shape, k.shape
        q_flat = q.reshape(-1, self.head_dim)
        k_flat = k.reshape(-1, self.head_dim)
        q, k = fused_q_kv_rmsnorm(
            q_flat,
            k_flat,
            self.q_norm.weight,
            self.k_norm.weight,
            self.q_norm.variance_epsilon,
        )
        q = q.view(q_shape)
        k = k.view(k_shape)

        q, k, v = (rearrange(t, "s b ... -> b s ...").contiguous() for t in (q, k, v))
        if rotary_pos_emb_cos is not None and rotary_pos_emb_sin is not None:
            qk_concat = torch.cat([q, k], dim=0)
            qk_rotated = self.apply_rotary_emb(
                qk_concat,
                rotary_pos_emb_cos,
                rotary_pos_emb_sin,
            )
            q, k = torch.chunk(qk_rotated, 2, dim=0)

        context_layer = self.attn(
            query=q,
            key=k,
            value=v,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
        )
        context_layer = rearrange(context_layer, "b s h d -> s b (h d)").contiguous()

        output, _ = self.proj(context_layer)
        return output


class Glm5NextVisionBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_hidden_dim: int,
        swiglu_limit: float,
        norm_layer: partial[nn.Module] | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = partial(nn.LayerNorm, eps=1e-6)
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)
        self.attn = Glm5NextVisionAttention(
            embed_dim=dim,
            num_heads=num_heads,
            projection_size=dim,
            quant_config=quant_config,
            prefix=f"{prefix}.attn",
        )
        self.mlp = Glm5NextVisionMLP(
            dim,
            mlp_hidden_dim,
            swiglu_limit=swiglu_limit,
            bias=True,
            quant_config=quant_config,
            prefix=f"{prefix}.mlp",
        )

    def forward(
        self,
        x: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rotary_pos_emb_cos: torch.Tensor,
        rotary_pos_emb_sin: torch.Tensor,
        max_seqlen: int | None = None,
    ) -> torch.Tensor:
        x_attn = self.attn(
            self.norm1(x),
            cu_seqlens=cu_seqlens,
            rotary_pos_emb_cos=rotary_pos_emb_cos,
            rotary_pos_emb_sin=rotary_pos_emb_sin,
            max_seqlen=max_seqlen,
        )
        x_fused_norm, residual = self.norm2(x, residual=x_attn)
        x = residual + self.mlp(x_fused_norm)
        return x


class Glm5NextPatchMerger(nn.Module):
    def __init__(
        self,
        d_model: int,
        context_dim: int,
        swiglu_limit: float,
        quant_config: QuantizationConfig | None = None,
        bias: bool = False,
        prefix: str = "",
    ) -> None:
        super().__init__()
        use_data_parallel = is_vit_use_data_parallel()
        self.hidden_size = d_model
        self.proj = ColumnParallelLinear(
            self.hidden_size,
            self.hidden_size,
            bias=bias,
            gather_output=True,
            quant_config=quant_config,
            prefix=f"{prefix}.proj",
            disable_tp=use_data_parallel,
        )
        self.post_projection_norm = nn.LayerNorm(self.hidden_size)
        self.gate_up_proj = MergedColumnParallelLinear(
            input_size=self.hidden_size,
            output_sizes=[context_dim] * 2,
            bias=bias,
            quant_config=quant_config,
            prefix=f"{prefix}.gate_up_proj",
            disable_tp=use_data_parallel,
        )
        self.down_proj = RowParallelLinear(
            context_dim,
            self.hidden_size,
            bias=bias,
            quant_config=quant_config,
            prefix=f"{prefix}.down_proj",
            disable_tp=use_data_parallel,
        )
        # GLM-5.3-Flash also clamps the merger SwiGLU.
        self.act_fn = SiluAndMulWithClamp(swiglu_limit=swiglu_limit)
        self.extra_activation_func = nn.GELU()

    def forward(self, x: torch.Tensor):
        x, _ = self.proj(x)
        x = self.extra_activation_func(self.post_projection_norm(x))
        gate_up, _ = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x, _ = self.down_proj(x)
        return x


class Glm5NextVisionTransformer(nn.Module):
    # Stacked-weight remap for the GLM-OCR/GLM-4V vision checkpoint layout.
    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_stacked={
            ".attn.q.": (".attn.qkv.", "q"),
            ".attn.k.": (".attn.qkv.", "k"),
            ".attn.v.": (".attn.qkv.", "v"),
            ".gate_proj": (".gate_up_proj", 0),
            ".up_proj": (".gate_up_proj", 1),
        }
    )

    def __init__(
        self,
        text_config,  # noqa: ANN001
        vision_config,
        norm_eps: float = 1e-6,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        use_data_parallel = is_vit_use_data_parallel()
        self.tp_size = (
            1 if use_data_parallel else get_tensor_model_parallel_world_size()
        )

        patch_size = vision_config.patch_size
        temporal_patch_size = vision_config.temporal_patch_size
        in_channels = vision_config.in_channels
        depth = vision_config.depth
        self.hidden_size = vision_config.hidden_size
        self.num_heads = vision_config.num_heads

        self.patch_size = vision_config.patch_size
        self.spatial_merge_size = vision_config.spatial_merge_size
        self.out_hidden_size = vision_config.out_hidden_size

        swiglu_limit = vision_config.swiglu_limit
        if swiglu_limit is None:
            swiglu_limit = text_config.swiglu_limit
        assert swiglu_limit is not None, (
            "GLM-5.3-Flash vision requires swiglu_limit (vision_config or text_config)"
        )

        # Single construction pass — no abs-pos embeddings / post-conv norm (OCR delta).
        self.patch_embed = Glm5NextVisionPatchEmbed(
            patch_size=patch_size,
            temporal_patch_size=temporal_patch_size,
            in_channels=in_channels,
            hidden_size=self.hidden_size,
        )

        norm_layer = partial(RMSNorm, eps=norm_eps)
        head_dim = self.hidden_size // self.num_heads
        self.rotary_pos_emb = get_rope(
            head_size=head_dim,
            max_position=8192,
            is_neox_style=True,
            rope_parameters={"partial_rotary_factor": 0.5},
        )
        self.blocks = nn.ModuleList(
            [
                Glm5NextVisionBlock(
                    dim=self.hidden_size,
                    num_heads=self.num_heads,
                    mlp_hidden_dim=vision_config.intermediate_size,
                    swiglu_limit=swiglu_limit,
                    norm_layer=norm_layer,
                    quant_config=quant_config,
                    prefix=f"{prefix}.blocks.{layer_idx}",
                )
                for layer_idx in range(depth)
            ]
        )
        # GLM-5.3-Flash merger bottleneck width.
        self.merger = Glm5NextPatchMerger(
            d_model=vision_config.out_hidden_size,
            context_dim=vision_config.projection_intermediate_size,
            swiglu_limit=swiglu_limit,
            quant_config=quant_config,
            bias=False,
            prefix=f"{prefix}.merger",
        )

        self.downsample = Conv2dLayer(
            in_channels=vision_config.hidden_size,
            out_channels=vision_config.out_hidden_size,
            kernel_size=vision_config.spatial_merge_size,
            stride=vision_config.spatial_merge_size,
        )
        self.post_layernorm = RMSNorm(
            vision_config.hidden_size, eps=vision_config.rms_norm_eps
        )

        self.attn_backend = get_vit_attn_backend(
            head_size=head_dim,
            dtype=torch.get_default_dtype(),
        )

    @property
    def dtype(self) -> torch.dtype:
        return self.patch_embed.proj.weight.dtype

    @property
    def device(self) -> torch.device:
        return self.patch_embed.proj.weight.device

    def rot_pos_emb(
        self, grid_thw: list[list[int]]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pos_ids = []
        for t, h, w in grid_thw:
            hpos_ids = torch.arange(h).unsqueeze(1).expand(-1, w)
            wpos_ids = torch.arange(w).unsqueeze(0).expand(h, -1)
            hpos_ids = (
                hpos_ids.reshape(
                    h // self.spatial_merge_size,
                    self.spatial_merge_size,
                    w // self.spatial_merge_size,
                    self.spatial_merge_size,
                )
                .permute(0, 2, 1, 3)
                .flatten()
            )
            wpos_ids = (
                wpos_ids.reshape(
                    h // self.spatial_merge_size,
                    self.spatial_merge_size,
                    w // self.spatial_merge_size,
                    self.spatial_merge_size,
                )
                .permute(0, 2, 1, 3)
                .flatten()
            )
            pos_ids.append(torch.stack([hpos_ids, wpos_ids], dim=-1).repeat(t, 1))
        pos_ids = torch.cat(pos_ids, dim=0)
        max_grid_size = max(max(h, w) for _, h, w in grid_thw)

        cos, sin = self.rotary_pos_emb.get_cos_sin(max_grid_size)

        pos_ids = pos_ids.to(cos.device, non_blocking=True)
        cos_combined = cos[pos_ids].flatten(1)
        sin_combined = sin[pos_ids].flatten(1)
        return cos_combined, sin_combined, pos_ids

    def compute_attn_mask_seqlen(
        self,
        cu_seqlens: torch.Tensor,
    ) -> torch.Tensor | None:
        max_seqlen = None
        if self.attn_backend in {
            AttentionBackendEnum.FLASH_ATTN,
            AttentionBackendEnum.ROCM_AITER_FA,
            AttentionBackendEnum.TRITON_ATTN,
        }:
            max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max()
        return max_seqlen

    def prepare_encoder_metadata(
        self,
        grid_thw_list: list[list[int]],
        *,
        max_batch_size: int | None = None,
        max_frames_per_batch: int | None = None,
        max_seqlen_override: int | None = None,
        device: torch.device | None = None,
    ) -> dict[str, torch.Tensor | None]:
        """Compute encoder metadata for eager and CUDA graph execution."""
        if device is None:
            device = self.device

        metadata: dict[str, torch.Tensor | None] = {}

        rotary_cos, rotary_sin, _ = self.rot_pos_emb(grid_thw_list)
        metadata["rotary_pos_emb_cos"] = rotary_cos
        metadata["rotary_pos_emb_sin"] = rotary_sin

        grid_thw_np = np.array(grid_thw_list, dtype=np.int32)
        patches_per_frame = grid_thw_np[:, 1] * grid_thw_np[:, 2]
        cu_seqlens = np.repeat(patches_per_frame, grid_thw_np[:, 0]).cumsum(
            dtype=np.int32
        )
        cu_seqlens = np.concatenate([np.zeros(1, dtype=np.int32), cu_seqlens])

        pad_to = (
            max_frames_per_batch if max_frames_per_batch is not None else max_batch_size
        )
        if pad_to is not None:
            num_seqs = len(cu_seqlens) - 1
            if num_seqs < pad_to:
                cu_seqlens = np.concatenate(
                    [
                        cu_seqlens,
                        np.full(
                            pad_to - num_seqs,
                            cu_seqlens[-1],
                            dtype=np.int32,
                        ),
                    ]
                )

        metadata["sequence_lengths"] = MMEncoderAttention.maybe_compute_seq_lens(
            self.attn_backend, cu_seqlens, device
        )

        if max_seqlen_override is not None:
            max_seqlen_val = max_seqlen_override
        else:
            max_seqlen_val = MMEncoderAttention.compute_max_seqlen(
                self.attn_backend, cu_seqlens
            )
        metadata["max_seqlen"] = torch.tensor(max_seqlen_val, dtype=torch.int32)

        metadata["cu_seqlens"] = MMEncoderAttention.maybe_recompute_cu_seqlens(
            self.attn_backend,
            cu_seqlens,
            self.hidden_size,
            self.tp_size,
            device,
        )

        return metadata

    def forward(
        self,
        x: torch.Tensor,
        grid_thw: torch.Tensor | list[list[int]],
        *,
        encoder_metadata: dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        # patchify
        x = x.to(device=self.device, dtype=self.dtype)
        x = self.patch_embed(x)

        if encoder_metadata is not None:
            # Encoder CUDA-graph path (PR #49852): rotary/cu_seqlens/max_seqlen are
            # precomputed by prepare_encoder_metadata (which uses rot_pos_emb exactly
            # as the eager rebuild does), so reuse them and skip the per-call CPU
            # rebuild (the low-GPU-util culprit on multimodal workloads).
            rotary_pos_emb_cos = encoder_metadata["rotary_pos_emb_cos"]
            rotary_pos_emb_sin = encoder_metadata["rotary_pos_emb_sin"]
            cu_seqlens = encoder_metadata["cu_seqlens"]
            max_seqlen = encoder_metadata["max_seqlen"]
        else:
            if isinstance(grid_thw, list):
                grid_thw = torch.tensor(grid_thw, dtype=torch.int32)
            rotary_pos_emb_cos, rotary_pos_emb_sin, _ = self.rot_pos_emb(grid_thw)
            cu_seqlens = torch.repeat_interleave(
                grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]
            ).cumsum(dim=0, dtype=torch.int32)
            cu_seqlens = torch.cat([cu_seqlens.new_zeros(1), cu_seqlens])
            cu_seqlens = cu_seqlens.to(self.device, non_blocking=True)
            max_seqlen = self.compute_attn_mask_seqlen(cu_seqlens)

        # transformers
        x = x.unsqueeze(1)
        for blk in self.blocks:
            x = blk(
                x,
                cu_seqlens=cu_seqlens,
                rotary_pos_emb_cos=rotary_pos_emb_cos,
                rotary_pos_emb_sin=rotary_pos_emb_sin,
                max_seqlen=max_seqlen,
            )

        # adapter
        x = self.post_layernorm(x)
        x = x.view(-1, self.spatial_merge_size, self.spatial_merge_size, x.shape[-1])
        x = x.permute(0, 3, 1, 2)
        x = self.downsample(x).view(-1, self.out_hidden_size)
        x = self.merger(x)
        return x

    def load_weights(self, weights) -> set[str]:
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)


class Glm5NextProcessingInfo(Glm4vProcessingInfo):
    """Wires up the vLLM-native processor for the multimodal checkpoint.

    The checkpoint's ``processor_config.json`` declares a custom ``processor_class``
    and stores its image/video processor configs inline (no standalone
    ``preprocessor_config.json``), so ``AutoProcessor`` cannot resolve the
    config. We bypass it and build our own ``Glm5NextProcessor``
    (``vllm/transformers_utils/processors/glm5next.py``), a port of the
    training-side pipeline that no longer imports transformers' GLM processor
    classes. The port applies ``patch_expand_factor`` (checkpoint ships 1)
    inside ``smart_resize``'s spatial factor.
    """

    @cached_property
    def _glm5_hf_processor(self):
        from vllm.transformers_utils.processors.glm5next import Glm5NextProcessor

        return Glm5NextProcessor.from_pretrained(self.ctx.model_config.model)

    def get_hf_processor(self, **kwargs: object):
        return self._glm5_hf_processor

    def _processor_pixel_budget(self, proc) -> tuple[int, int]:
        from vllm.transformers_utils.processors.glm5next import _pixel_budget

        return _pixel_budget(
            proc.min_image_tokens,
            proc.max_image_tokens,
            proc.patch_size,
            proc.merge_size,
            proc.temporal_patch_size,
        )

    def _get_image_max_pixels(self) -> int:
        mm_kwargs = self.ctx.get_merged_mm_kwargs({})
        if (override := mm_kwargs.get("max_pixels")) is not None:
            return int(override)
        return self._processor_pixel_budget(self.get_hf_processor().image_processor)[1]

    def _get_video_max_pixels(self) -> int:
        mm_kwargs = self.ctx.get_merged_mm_kwargs({})
        if (override := mm_kwargs.get("max_pixels")) is not None:
            return int(override)
        return self._processor_pixel_budget(self.get_hf_processor().video_processor)[1]

    def _get_vision_info(
        self,
        *,
        image_width: int,
        image_height: int,
        num_frames: int = 16,
        do_resize: bool = True,
        max_image_pixels: int = 28 * 28 * 2 * 30000,
    ) -> tuple[ImageSize, int]:
        """GLM-5.3-Flash canvas geometry for token budgeting and dummy inputs.

        The inherited Glm4v path resolves the pixel budget from
        ``size.longest_edge`` and resizes with GLM-4V's ``smart_resize``. This
        checkpoint's ``processor_config.json`` ships the token-budget style
        (``min_image_tokens`` / ``max_image_tokens``) with no ``size`` key, and
        the alignment factor carries ``patch_expand_factor`` — resolve both
        from the vLLM-native processor so profiling matches runtime geometry.
        """
        from vllm.transformers_utils.processors.glm5next import smart_resize

        vision_config = self.get_hf_config().vision_config
        patch_size = vision_config.patch_size
        merge_size = vision_config.spatial_merge_size
        temporal_patch_size = vision_config.temporal_patch_size

        image_processor = self.get_hf_processor().image_processor
        factor = patch_size * merge_size * image_processor.patch_expand_factor
        # Keep the profiling search viable when the caller's budget is below
        # one aligned canvas of the requested duration.
        max_image_pixels = max(max_image_pixels, temporal_patch_size * factor * factor)

        if do_resize:
            t = num_frames if num_frames > temporal_patch_size else temporal_patch_size
            resized_height, resized_width = smart_resize(
                t=t,
                h=image_height,
                w=image_width,
                t_factor=temporal_patch_size,
                h_factor=factor,
                w_factor=factor,
                min_pixels=1,
                max_pixels=max_image_pixels,
            )
            preprocessed_size = ImageSize(width=resized_width, height=resized_height)
        else:
            preprocessed_size = ImageSize(width=image_width, height=image_height)

        padded_num_frames = num_frames + (-num_frames % temporal_patch_size)
        grid_t = max(padded_num_frames // temporal_patch_size, 1)
        grid_h = preprocessed_size.height // patch_size
        grid_w = preprocessed_size.width // patch_size

        num_patches = grid_t * grid_h * grid_w
        num_vision_tokens = num_patches // (merge_size**2)

        return preprocessed_size, num_vision_tokens


class Glm5NextMultiModalProcessor(Glm4vMultiModalProcessor):
    """The vLLM-native ``Glm5NextProcessor`` extracts image/video features
    only and passes the prompt text through unchanged, so prompt expansion
    (image token repeat, video frame/timestamp structure) is owned by vLLM's
    prompt-update machinery — the inherited ``_get_prompt_updates`` builds
    the replacement content and the placeholder scan validates against
    exactly that."""

    def _hf_processor_applies_updates(
        self,
        prompt_text: str,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        tokenization_kwargs: Mapping[str, object],
    ) -> bool:
        return False
