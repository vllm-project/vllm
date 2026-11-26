# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inference-only LFM3-VL model: images and videos in, text out."""

from collections.abc import Hashable, Iterable, Mapping, Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from vllm.config import VllmConfig
from vllm.distributed import divide, get_tensor_model_parallel_world_size
from vllm.model_executor.layers.activation import get_act_fn
from vllm.model_executor.layers.attention import MMEncoderAttention
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    QKVParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.mamba.mamba_utils import (
    MambaStateCopyFunc,
    MambaStateCopyFuncCalculator,
)
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.rotary_embedding.common import ApplyRotaryEmb
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.sequence import IntermediateTensors
from vllm.transformers_utils.configs.lfm3_vl import Lfm3VlConfig, Lfm3VlVisionConfig
from vllm.transformers_utils.processors.lfm3_vl import Lfm3VlProcessor
from vllm.triton_utils import HAS_TRITON, tl, triton
from vllm.utils.cache import LRUCache
from vllm.utils.torch_utils import async_tensor_h2d
from vllm.v1.attention.backends.registry import AttentionBackendEnum

from .interfaces import (
    HasInnerState,
    IsHybrid,
    MultiModalEmbeddings,
    SupportsEncoderCudaGraph,
    SupportsMultiModal,
    SupportsPP,
)
from .lfm3 import Lfm3MLP, _short_conv_state_dtype, _short_conv_state_shape
from .qwen3_vl import (
    Qwen3VLDummyInputsBuilder,
    Qwen3VLMultiModalProcessor,
    Qwen3VLProcessingInfo,
)
from .utils import (
    AutoWeightsLoader,
    StageMissingLayer,
    WeightsMapper,
    init_vllm_registered_model,
    maybe_prefix,
)
from .vision import get_fp8_padded_hidden_size, get_vit_attn_backend

if TYPE_CHECKING:
    from vllm.v1.worker.encoder_cudagraph_defs import (
        EncoderCudaGraphCaptureInputs,
        EncoderCudaGraphConfig,
        EncoderCudaGraphReplayBuffers,
        EncoderItemSpec,
    )


if HAS_TRITON:

    @triton.jit(do_not_specialize=["num_patches"])
    def _conv2d_gather_kernel(
        gated_ptr,
        weight_ptr,
        conv_indices_ptr,
        output_ptr,
        num_patches,
        hidden_dim,
        stride_gn,
        stride_gd,
        stride_wd,
        stride_wk,
        stride_in,
        stride_ik,
        stride_on,
        stride_od,
        KERNEL_AREA: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        """Fused 1-based depthwise 2D gather-convolution in FP32 accumulation."""
        pid_n = tl.program_id(0)
        pid_d = tl.program_id(1)

        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
        mask_n = offs_n < num_patches
        mask_d = offs_d < hidden_dim

        offs_n_i64 = offs_n.to(tl.int64)
        offs_d_i64 = offs_d.to(tl.int64)

        acc = tl.zeros((BLOCK_N, BLOCK_D), dtype=tl.float32)
        for k_idx in tl.static_range(KERNEL_AREA):
            idx = tl.load(
                conv_indices_ptr + offs_n_i64 * stride_in + k_idx * stride_ik,
                mask=mask_n,
                other=0,
            )
            valid = mask_n & (idx > 0)
            # 1-based indices use 0 as the zero-padding sentinel.
            src_row = tl.where(valid, idx - 1, 0)
            g = tl.load(
                gated_ptr
                + src_row[:, None] * stride_gn
                + offs_d_i64[None, :] * stride_gd,
                mask=valid[:, None] & mask_d[None, :],
                other=0.0,
            ).to(tl.float32)
            w = tl.load(
                weight_ptr + offs_d_i64 * stride_wd + k_idx * stride_wk,
                mask=mask_d,
                other=0.0,
            ).to(tl.float32)
            acc += g * w[None, :]

        tl.store(
            output_ptr
            + offs_n_i64[:, None] * stride_on
            + offs_d_i64[None, :] * stride_od,
            acc.to(output_ptr.dtype.element_ty),
            mask=mask_n[:, None] & mask_d[None, :],
        )


def _triton_conv2d_gather(
    gated: torch.Tensor,
    weight: torch.Tensor,
    conv_indices: torch.Tensor,
) -> torch.Tensor:
    """Compute depthwise 2D conv over 1-based `conv_indices` via Triton."""
    num_patches, hidden_dim = gated.shape
    if num_patches == 0:
        return gated.new_empty(gated.shape)
    gated = gated.contiguous()
    conv_indices = conv_indices.contiguous()
    weight_2d = weight.reshape(hidden_dim, -1).contiguous()
    kernel_area = weight_2d.shape[1]
    output = torch.empty_like(gated)

    block_n = 8
    block_d = min(128, triton.next_power_of_2(hidden_dim))
    grid = (triton.cdiv(num_patches, block_n), triton.cdiv(hidden_dim, block_d))
    _conv2d_gather_kernel[grid](
        gated,
        weight_2d,
        conv_indices,
        output,
        num_patches,
        hidden_dim,
        gated.stride(0),
        gated.stride(1),
        weight_2d.stride(0),
        weight_2d.stride(1),
        conv_indices.stride(0),
        conv_indices.stride(1),
        output.stride(0),
        output.stride(1),
        KERNEL_AREA=kernel_area,
        BLOCK_N=block_n,
        BLOCK_D=block_d,
    )
    return output


def _native_conv2d_gather(
    gated: torch.Tensor,
    weight: torch.Tensor,
    conv_indices: torch.Tensor,
) -> torch.Tensor:
    """Native PyTorch fallback for `conv_indices` depthwise 2D conv."""
    if gated.shape[0] == 0:
        return gated.new_empty(gated.shape)
    # Prepend row 0 so 0 indices read zeros without materializing [N, k^2, D].
    gated_padded = F.pad(gated.float(), (0, 0, 1, 0))
    weight_t = weight.reshape(gated.shape[1], -1).float().t()
    acc = gated.new_zeros(gated.shape, dtype=torch.float32)
    for k_idx in range(conv_indices.shape[1]):
        acc.addcmul_(
            F.embedding(conv_indices[:, k_idx], gated_padded),
            weight_t[k_idx],
        )
    return acc.to(gated.dtype)


class Lfm3VlVisionAttention(nn.Module):
    """Bidirectional grouped-query attention over the patches of one frame, with
    RMSNorm on the query and key heads and axial 2D RoPE."""

    def __init__(
        self,
        config: Lfm3VlVisionConfig,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.qkv_proj = QKVParallelLinear(
            config.hidden_size,
            self.head_dim,
            config.num_attention_heads,
            config.num_key_value_heads,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
        )
        self.num_heads = self.qkv_proj.num_heads
        self.num_kv_heads = self.qkv_proj.num_kv_heads
        self.out_proj = RowParallelLinear(
            config.num_attention_heads * self.head_dim,
            config.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.out_proj",
        )
        self.q_layernorm = RMSNorm(self.head_dim, eps=config.norm_eps)
        self.k_layernorm = RMSNorm(self.head_dim, eps=config.norm_eps)
        self.apply_rotary_emb = ApplyRotaryEmb(enforce_enable=True)
        self.attn = MMEncoderAttention(
            num_heads=self.num_heads,
            head_size=self.head_dim,
            scale=self.head_dim**-0.5,
            num_kv_heads=self.num_kv_heads,
            prefix=f"{prefix}.attn",
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        sequence_lengths: torch.Tensor | None,
    ) -> torch.Tensor:
        seq_len = hidden_states.shape[0]
        qkv, _ = self.qkv_proj(hidden_states)
        q_size = self.num_heads * self.head_dim
        kv_size = self.num_kv_heads * self.head_dim
        q, k, v = qkv.split([q_size, kv_size, kv_size], dim=-1)
        q = self.q_layernorm(q.reshape(seq_len, self.num_heads, self.head_dim))
        k = self.k_layernorm(k.reshape(seq_len, self.num_kv_heads, self.head_dim))
        v = v.reshape(1, seq_len, self.num_kv_heads, self.head_dim)
        q = self.apply_rotary_emb(q[None], cos, sin)
        k = self.apply_rotary_emb(k[None], cos, sin)
        if self.attn.attn_backend == AttentionBackendEnum.FLASHINFER:
            # The cuDNN wrapper shares physical offsets between Q, K and O, so
            # K and V take the query head count and V the same contiguous layout.
            if self.num_kv_heads != self.num_heads:
                repeats = self.num_heads // self.num_kv_heads
                k = k.repeat_interleave(repeats, dim=2)
                v = v.repeat_interleave(repeats, dim=2)
            v = v.contiguous()
        output = self.attn(
            query=q,
            key=k,
            value=v,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            sequence_lengths=sequence_lengths,
        )
        output, _ = self.out_proj(output.reshape(seq_len, -1))
        return output


class Lfm3VlVisionConv2d(nn.Module):
    """Gated depthwise 2D convolution over the patch grid of each frame."""

    def __init__(
        self,
        config: Lfm3VlVisionConfig,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        dim = config.hidden_size
        self.spatial_merge_size = config.spatial_merge_size
        self.kernel_size = config.conv_kernel_size
        self.in_proj = MergedColumnParallelLinear(
            dim,
            [dim] * 3,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.in_proj",
        )
        # One (k, k) filter per channel, sharded over channels with the
        # projections; the parameter keeps the Conv2d weight shape for loading.
        self.spatial_conv = ColumnParallelLinear(
            self.kernel_size**2, dim, bias=False, prefix=f"{prefix}.spatial_conv"
        )
        self.spatial_conv.weight.data = self.spatial_conv.weight.data.view(
            -1, 1, self.kernel_size, self.kernel_size
        )
        self.out_proj = RowParallelLinear(
            dim, dim, bias=False, quant_config=quant_config, prefix=f"{prefix}.out_proj"
        )

    def _convolve(self, frames: torch.Tensor) -> torch.Tensor:
        return F.conv2d(
            frames,
            self.spatial_conv.weight,
            padding=self.kernel_size // 2,
            groups=frames.shape[1],
        )

    def _convolve_with_indices(
        self, gated: torch.Tensor, conv_indices: torch.Tensor
    ) -> torch.Tensor:
        if HAS_TRITON and gated.is_cuda:
            return _triton_conv2d_gather(gated, self.spatial_conv.weight, conv_indices)
        return _native_conv2d_gather(gated, self.spatial_conv.weight, conv_indices)

    def forward(
        self,
        hidden_states: torch.Tensor,
        frame_shapes: list[tuple[int, int]] | None = None,
        conv_indices: torch.Tensor | None = None,
    ) -> torch.Tensor:
        BCx, _ = self.in_proj(hidden_states)
        B, C, x = BCx.chunk(3, dim=-1)
        gated = B * x
        if conv_indices is not None:
            conv_output = self._convolve_with_indices(gated, conv_indices)
        else:
            assert frame_shapes is not None
            merge = self.spatial_merge_size
            if len(set(frame_shapes)) == 1:
                # frames of one shape convolve as one batch: merge-block order
                # -> raster grid, convolve, back
                height, width = frame_shapes[0]
                frames = gated.view(
                    len(frame_shapes),
                    height // merge,
                    width // merge,
                    merge,
                    merge,
                    -1,
                )
                frames = frames.permute(0, 5, 1, 3, 2, 4).reshape(
                    len(frame_shapes), -1, height, width
                )
                frames = self._convolve(frames)
                frames = frames.view(
                    len(frame_shapes),
                    -1,
                    height // merge,
                    merge,
                    width // merge,
                    merge,
                )
                conv_output = frames.permute(0, 2, 4, 3, 5, 1).reshape(
                    gated.shape[0], -1
                )
            else:
                outputs = []
                lengths = [height * width for height, width in frame_shapes]
                for frame, (height, width) in zip(gated.split(lengths), frame_shapes):
                    frame = frame.view(
                        height // merge, width // merge, merge, merge, -1
                    )
                    frame = frame.permute(4, 0, 2, 1, 3).reshape(1, -1, height, width)
                    frame = self._convolve(frame)
                    frame = frame.view(
                        -1, height // merge, merge, width // merge, merge
                    )
                    outputs.append(
                        frame.permute(1, 3, 2, 4, 0).reshape(height * width, -1)
                    )
                conv_output = torch.cat(outputs)
        output, _ = self.out_proj(C * conv_output)
        return output


class Lfm3VlVisionLayer(nn.Module):
    def __init__(
        self,
        config: Lfm3VlVisionConfig,
        layer_idx: int,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.is_attention_layer = config.layer_types[layer_idx] == "full_attention"
        if self.is_attention_layer:
            self.self_attn = Lfm3VlVisionAttention(
                config, quant_config=quant_config, prefix=f"{prefix}.self_attn"
            )
        else:
            self.conv = Lfm3VlVisionConv2d(
                config, quant_config=quant_config, prefix=f"{prefix}.conv"
            )
        self.feed_forward = Lfm3MLP(
            config, quant_config=quant_config, prefix=f"{prefix}.feed_forward"
        )
        self.operator_norm = RMSNorm(config.hidden_size, eps=config.norm_eps)
        self.ffn_norm = RMSNorm(config.hidden_size, eps=config.norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        cu_seqlens: torch.Tensor,
        max_seqlen: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        sequence_lengths: torch.Tensor | None,
        frame_shapes: list[tuple[int, int]] | None = None,
        conv_indices: torch.Tensor | None = None,
    ) -> torch.Tensor:
        normed = self.operator_norm(hidden_states)
        if self.is_attention_layer:
            operator_output = self.self_attn(
                normed, cu_seqlens, max_seqlen, cos, sin, sequence_lengths
            )
        else:
            operator_output = self.conv(
                normed, frame_shapes=frame_shapes, conv_indices=conv_indices
            )
        hidden_states = hidden_states + operator_output
        return hidden_states + self.feed_forward(self.ffn_norm(hidden_states))


class Lfm3VlVisionModel(nn.Module):
    """The LFM3-VL vision encoder: gated convolution and bidirectional attention
    layers over the patches of each image or video frame, embedded by a linear
    patchifier."""

    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_stacked={
            ".q_proj.": (".qkv_proj.", "q"),
            ".k_proj.": (".qkv_proj.", "k"),
            ".v_proj.": (".qkv_proj.", "v"),
            ".w1.": (".w13.", 0),
            ".w3.": (".w13.", 1),
        }
    )

    def __init__(
        self,
        config: Lfm3VlVisionConfig,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.spatial_merge_size = config.spatial_merge_size
        self.conv_kernel_size = config.conv_kernel_size
        self._conv_rel_cache: LRUCache[tuple[int, int], torch.Tensor] = LRUCache(
            capacity=256
        )
        head_dim = config.hidden_size // config.num_attention_heads
        patch_dim = (
            config.in_channels * config.temporal_patch_size * config.patch_size**2
        )
        self.patchifier = ReplicatedLinear(
            patch_dim,
            config.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.patchifier",
        )
        self.embedding_norm = RMSNorm(config.hidden_size, eps=config.norm_eps)
        # Axial 2D RoPE: half the head rotates with the row, half with the column.
        self.rotary_pos_emb = get_rope(
            head_size=head_dim,
            max_position=8192,
            is_neox_style=True,
            rope_parameters={
                "partial_rotary_factor": 0.5,
                "rope_theta": config.rope_parameters["rope_theta"],
            },
        )
        self.layers = nn.ModuleList(
            Lfm3VlVisionLayer(
                config,
                layer_idx,
                quant_config=quant_config,
                prefix=f"{prefix}.layers.{layer_idx}",
            )
            for layer_idx in range(config.num_hidden_layers)
        )
        self.attn_backend = get_vit_attn_backend(
            head_size=head_dim, dtype=torch.get_default_dtype()
        )
        # FlashInfer expands KV to the TP-local query head count before attention,
        # so Q/K/V share one contiguous stride.
        num_heads = divide(
            config.num_attention_heads, get_tensor_model_parallel_world_size()
        )
        self.flashinfer_hidden_size = (
            get_fp8_padded_hidden_size(num_heads, head_dim) or num_heads * head_dim
        )

    @property
    def dtype(self) -> torch.dtype:
        # Read from unquantized embedding_norm so quantized patchifier keeps
        # activation dtype intact.
        return self.embedding_norm.weight.dtype

    @property
    def device(self) -> torch.device:
        return self.embedding_norm.weight.device

    def position_embeddings(
        self, grid_thw: Sequence[Sequence[int]]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Cos and sin of every patch's (row, column), in merge-block order."""
        if not grid_thw:
            rotary_dim = self.rotary_pos_emb.rotary_dim
            cache_dtype = self.rotary_pos_emb.cos_sin_cache.dtype
            empty = torch.empty((0, rotary_dim), device=self.device, dtype=cache_dtype)
            return empty, empty.clone()
        merge = self.spatial_merge_size
        positions = []
        for t, h, w in grid_thw:
            rows = torch.arange(h, device=self.device)[:, None].expand(h, w)
            cols = torch.arange(w, device=self.device)[None, :].expand(h, w)
            rows = rows.reshape(h // merge, merge, w // merge, merge).transpose(1, 2)
            cols = cols.reshape(h // merge, merge, w // merge, merge).transpose(1, 2)
            positions.append(
                torch.stack([rows.flatten(), cols.flatten()], -1).repeat(t, 1)
            )
        positions_t = torch.cat(positions)
        cos, sin = self.rotary_pos_emb.get_cos_sin(
            max(max(h, w) for _, h, w in grid_thw)
        )
        return cos[positions_t].flatten(1), sin[positions_t].flatten(1)

    def _build_frame_conv_indices(
        self, height: int, width: int, *, use_cache: bool = True
    ) -> torch.Tensor:
        """Build 1-based merge-block neighbor indices `[H*W, k*k]` for one frame."""
        cache_key = (height, width)
        if use_cache and cache_key in self._conv_rel_cache:
            return self._conv_rel_cache[cache_key]

        merge = self.spatial_merge_size
        k = self.conv_kernel_size
        if k % 2 == 0:
            raise ValueError(f"conv_kernel_size must be odd, got {k}.")
        if height % merge != 0 or width % merge != 0:
            raise ValueError(
                f"Frame shape ({height}, {width}) must be divisible by "
                f"spatial_merge_size={merge}."
            )
        pad = k // 2
        n = height * width
        grid = (
            torch.arange(1, n + 1, dtype=torch.int64)
            .view(height // merge, width // merge, merge, merge)
            .permute(0, 2, 1, 3)
            .reshape(height, width)
        )
        padded = F.pad(grid, (pad, pad, pad, pad), value=0)
        windows = padded.unfold(0, k, 1).unfold(1, k, 1)
        rel = (
            windows.reshape(height // merge, merge, width // merge, merge, k * k)
            .permute(0, 2, 1, 3, 4)
            .reshape(n, k * k)
            .contiguous()
        )
        if use_cache:
            self._conv_rel_cache[cache_key] = rel
        return rel

    def _build_conv_indices(
        self,
        grid_thw_list: Sequence[Sequence[int]],
        *,
        device: torch.device,
        use_cache: bool = True,
    ) -> torch.Tensor:
        """Build global 1-based `conv_indices` `[N, k*k]` isolated across frames."""
        k2 = self.conv_kernel_size * self.conv_kernel_size
        if not grid_thw_list:
            return torch.empty((0, k2), dtype=torch.int64, device=device)

        parts: list[torch.Tensor] = []
        offset = 0
        for t, h, w in grid_thw_list:
            rel = self._build_frame_conv_indices(h, w, use_cache=use_cache)
            frame_size = h * w
            if t == 1:
                idx = rel if offset == 0 else torch.where(rel > 0, rel + offset, 0)
            else:
                frame_offsets = (
                    offset
                    + torch.arange(t, dtype=torch.int64).view(t, 1, 1) * frame_size
                )
                idx = torch.where(
                    rel.unsqueeze(0) > 0,
                    rel.unsqueeze(0) + frame_offsets,
                    0,
                ).reshape(t * frame_size, k2)
            parts.append(idx)
            offset += t * frame_size

        conv_indices_cpu = parts[0] if len(parts) == 1 else torch.cat(parts, dim=0)
        return async_tensor_h2d(conv_indices_cpu, device=device)

    def _prepare_attention_metadata(
        self,
        grid_thw_list: Sequence[Sequence[int]],
        *,
        pad_to_num_frames: int | None = None,
        max_seqlen_override: int | None = None,
        device: torch.device | None = None,
    ) -> dict[str, torch.Tensor | None]:
        if device is None:
            device = self.device

        frame_lengths = [h * w for t, h, w in grid_thw_list for _ in range(t)]
        cu_seqlens_np = np.cumsum([0] + frame_lengths, dtype=np.int32)

        if pad_to_num_frames is not None:
            num_seqs = len(cu_seqlens_np) - 1
            if num_seqs < pad_to_num_frames:
                cu_seqlens_np = np.concatenate(
                    [
                        cu_seqlens_np,
                        np.full(
                            pad_to_num_frames - num_seqs,
                            cu_seqlens_np[-1],
                            dtype=np.int32,
                        ),
                    ]
                )

        sequence_lengths = MMEncoderAttention.maybe_compute_seq_lens(
            self.attn_backend, cu_seqlens_np, device
        )
        if max_seqlen_override is not None:
            max_seqlen_val = MMEncoderAttention.compute_max_seqlen(
                self.attn_backend,
                np.array([0, max_seqlen_override], dtype=np.int32),
            )
        else:
            max_seqlen_val = MMEncoderAttention.compute_max_seqlen(
                self.attn_backend, cu_seqlens_np
            )
        max_seqlen = torch.tensor(max_seqlen_val, dtype=torch.int32)
        cu_seqlens = MMEncoderAttention.maybe_recompute_cu_seqlens(
            self.attn_backend,
            cu_seqlens_np,
            hidden_size=self.hidden_size,
            tp_size=1,  # flashinfer_hidden_size is already TP-local
            device=device,
            fp8_padded_hidden_size=self.flashinfer_hidden_size,
        )
        return {
            "cu_seqlens": cu_seqlens,
            "max_seqlen": max_seqlen,
            "sequence_lengths": sequence_lengths,
        }

    def prepare_capture_metadata(
        self,
        *,
        num_patches: int,
        pad_to_num_frames: int,
        device: torch.device,
    ) -> dict[str, torch.Tensor | None]:
        """Build dummy capture-time buffers without indexing `rotary_pos_emb`."""
        m = self.spatial_merge_size
        k2 = self.conv_kernel_size * self.conv_kernel_size
        rotary_dim = self.rotary_pos_emb.rotary_dim
        rotary_dtype = self.rotary_pos_emb.cos_sin_cache.dtype
        metadata = self._prepare_attention_metadata(
            [[1, m, num_patches // m]],
            pad_to_num_frames=pad_to_num_frames,
            max_seqlen_override=num_patches,
            device=device,
        )
        metadata["rotary_pos_emb_cos"] = torch.zeros(
            (num_patches, rotary_dim), device=device, dtype=rotary_dtype
        )
        metadata["rotary_pos_emb_sin"] = torch.zeros(
            (num_patches, rotary_dim), device=device, dtype=rotary_dtype
        )
        metadata["conv_indices"] = torch.zeros(
            (num_patches, k2), device=device, dtype=torch.int64
        )
        return metadata

    def prepare_encoder_metadata(
        self,
        grid_thw_list: Sequence[Sequence[int]],
        *,
        pad_to_num_frames: int | None = None,
        max_seqlen_override: int | None = None,
        device: torch.device | None = None,
    ) -> dict[str, torch.Tensor | None]:
        """Precompute RoPE, `conv_indices`, and attention metadata on `device`."""
        if device is None:
            device = self.device
        metadata = self._prepare_attention_metadata(
            grid_thw_list,
            pad_to_num_frames=pad_to_num_frames,
            max_seqlen_override=max_seqlen_override,
            device=device,
        )
        cos, sin = self.position_embeddings(grid_thw_list)
        metadata["rotary_pos_emb_cos"] = cos
        metadata["rotary_pos_emb_sin"] = sin
        metadata["conv_indices"] = self._build_conv_indices(
            grid_thw_list, device=device
        )
        return metadata

    def forward(
        self,
        pixel_values: torch.Tensor,
        grid_thw: list[list[int]] | None = None,
        *,
        encoder_metadata: Mapping[str, torch.Tensor | None] | None = None,
    ) -> torch.Tensor:
        hidden_states, _ = self.patchifier(pixel_values.to(self.device, self.dtype))
        hidden_states = self.embedding_norm(hidden_states)
        if encoder_metadata is not None:
            cos = encoder_metadata["rotary_pos_emb_cos"]
            sin = encoder_metadata["rotary_pos_emb_sin"]
            cu_seqlens = encoder_metadata["cu_seqlens"]
            max_seqlen = encoder_metadata["max_seqlen"]
            sequence_lengths = encoder_metadata.get("sequence_lengths")
            conv_indices = encoder_metadata["conv_indices"]
            assert cos is not None
            assert sin is not None
            assert cu_seqlens is not None
            assert max_seqlen is not None
            assert conv_indices is not None
            frame_shapes = None
        else:
            assert grid_thw is not None
            cos, sin = self.position_embeddings(grid_thw)
            frame_shapes = [(h, w) for t, h, w in grid_thw for _ in range(t)]
            attn_metadata = self._prepare_attention_metadata(grid_thw)
            cu_seqlens = attn_metadata["cu_seqlens"]
            max_seqlen = attn_metadata["max_seqlen"]
            sequence_lengths = attn_metadata["sequence_lengths"]
            assert cu_seqlens is not None
            assert max_seqlen is not None
            conv_indices = None

        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
                cos=cos,
                sin=sin,
                sequence_lengths=sequence_lengths,
                frame_shapes=frame_shapes,
                conv_indices=conv_indices,
            )
        return hidden_states

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        return AutoWeightsLoader(self).load_weights(
            weights, mapper=self.hf_to_vllm_mapper
        )


class Lfm3VlMultiModalProjector(nn.Module):
    """Merges each block of `spatial_merge_size**2` vision tokens, normalized
    before the merge, into one language token."""

    def __init__(
        self,
        config: Lfm3VlConfig,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        vision_config = config.vision_config
        self.layer_norm = nn.LayerNorm(vision_config.hidden_size, eps=1e-6)
        self.linear_1 = ColumnParallelLinear(
            vision_config.hidden_size * vision_config.spatial_merge_size**2,
            config.projector_hidden_size,
            bias=config.projector_bias,
            quant_config=quant_config,
            prefix=f"{prefix}.linear_1",
        )
        self.act = get_act_fn(config.projector_hidden_act)
        self.linear_2 = RowParallelLinear(
            config.projector_hidden_size,
            config.text_config.hidden_size,
            bias=config.projector_bias,
            quant_config=quant_config,
            prefix=f"{prefix}.linear_2",
        )

    def forward(self, image_features: torch.Tensor) -> torch.Tensor:
        hidden_states = self.layer_norm(image_features)
        hidden_states = hidden_states.reshape(-1, self.linear_1.input_size)
        hidden_states, _ = self.linear_1(hidden_states)
        hidden_states, _ = self.linear_2(self.act(hidden_states))
        return hidden_states


class Lfm3VlProcessingInfo(Qwen3VLProcessingInfo):
    def get_hf_config(self) -> Lfm3VlConfig:
        return self.ctx.get_hf_config(Lfm3VlConfig)

    def get_hf_processor(self, **kwargs: object) -> Lfm3VlProcessor:
        return self.ctx.get_hf_processor(Lfm3VlProcessor, **kwargs)


class Lfm3VlDummyInputsBuilder(Qwen3VLDummyInputsBuilder):
    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        image = "<|vision_start|><|image|><|vision_end|>"
        video = "<|vision_start|><|video|><|vision_end|>"
        return image * mm_counts.get("image", 0) + video * mm_counts.get("video", 0)


@MULTIMODAL_REGISTRY.register_processor(
    Qwen3VLMultiModalProcessor,
    info=Lfm3VlProcessingInfo,
    dummy_inputs=Lfm3VlDummyInputsBuilder,
)
class Lfm3VlForConditionalGeneration(
    nn.Module,
    HasInnerState,
    IsHybrid,
    SupportsMultiModal,
    SupportsEncoderCudaGraph,
    SupportsPP,
):
    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_prefix={
            "model.language_model.": "language_model.model.",
            "model.vision_tower.": "vision_tower.",
            "model.multi_modal_projector.": "multi_modal_projector.",
            "lm_head.": "language_model.lm_head.",
        }
    )

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None:
        if modality.startswith("image"):
            return "<|vision_start|><|image|><|vision_end|>"
        if modality.startswith("video"):
            return "<|vision_start|><|video|><|vision_end|>"
        raise ValueError("Only image or video modality is supported")

    @classmethod
    def get_mamba_state_dtype_from_config(
        cls, vllm_config: VllmConfig
    ) -> tuple[torch.dtype, ...]:
        return _short_conv_state_dtype(vllm_config)

    @classmethod
    def get_mamba_state_shape_from_config(
        cls, vllm_config: VllmConfig
    ) -> tuple[tuple[int, int]]:
        return _short_conv_state_shape(
            vllm_config, vllm_config.model_config.hf_config.text_config
        )

    @classmethod
    def get_mamba_state_copy_func(cls) -> tuple[MambaStateCopyFunc]:
        return MambaStateCopyFuncCalculator.short_conv_state_copy_func()

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        config: Lfm3VlConfig = vllm_config.model_config.hf_config
        self.config = config
        self.model_config = vllm_config.model_config
        self.multimodal_config = vllm_config.model_config.multimodal_config
        with self._mark_tower_model(vllm_config, {"image", "video"}):
            self.vision_tower = Lfm3VlVisionModel(
                config.vision_config,
                quant_config=vllm_config.quant_config,
                prefix=maybe_prefix(prefix, "vision_tower"),
            )
            self.multi_modal_projector = Lfm3VlMultiModalProjector(
                config,
                quant_config=vllm_config.quant_config,
                prefix=maybe_prefix(prefix, "multi_modal_projector"),
            )
        with self._mark_language_model(vllm_config):
            self.language_model = init_vllm_registered_model(
                vllm_config=vllm_config,
                hf_config=config.text_config,
                prefix=maybe_prefix(prefix, "language_model"),
                architectures=["Lfm3ForCausalLM"],
            )
        self.make_empty_intermediate_tensors = (
            self.language_model.make_empty_intermediate_tensors
        )

    # -- SupportsEncoderCudaGraph protocol methods --

    def get_encoder_cudagraph_config(self) -> "EncoderCudaGraphConfig":
        from vllm.v1.worker.encoder_cudagraph_defs import EncoderCudaGraphConfig

        if isinstance(self.vision_tower, StageMissingLayer):
            return EncoderCudaGraphConfig(
                modalities=[],
                buffer_keys=[],
                out_hidden_size=self.config.text_config.hidden_size,
            )

        attn_backend = self.vision_tower.attn_backend
        has_dynamic_fp8 = any(
            layer.is_attention_layer
            and getattr(layer.self_attn.attn, "_fp8_dynamic_scale", False)
            for layer in self.vision_tower.layers
        )
        # TORCH_SDPA syncs cu_seqlens on host; dynamic FP8 scales mutate state
        # and reduce amax across padded tail tokens.
        graph_safe = (
            attn_backend != AttentionBackendEnum.TORCH_SDPA and not has_dynamic_fp8
        )
        modalities = ["image", "video"] if graph_safe else []
        max_frames = self.get_max_frames_per_video() if "video" in modalities else 1
        padding_logics = MMEncoderAttention.get_cu_seqlens_padding_logics(
            attn_backend, "cu_seqlens"
        )
        return EncoderCudaGraphConfig(
            modalities=modalities,
            buffer_keys=[
                "pixel_values",
                "conv_indices",
                "rotary_pos_emb_cos",
                "rotary_pos_emb_sin",
                "cu_seqlens",
                "max_seqlen",
                "sequence_lengths",
            ],
            padding_logics=padding_logics,
            out_hidden_size=self.config.text_config.hidden_size,
            max_frames_per_video=max_frames,
        )

    def get_input_modality(self, mm_kwargs: dict[str, Any]) -> str:
        if "image_grid_thw" in mm_kwargs or "pixel_values" in mm_kwargs:
            return "image"
        if "video_grid_thw" in mm_kwargs or "pixel_values_videos" in mm_kwargs:
            return "video"
        raise ValueError(
            "Expected image or video multimodal inputs in mm_kwargs, "
            f"got {list(mm_kwargs.keys())}."
        )

    def get_max_frames_per_video(self) -> int:
        if (
            self.multimodal_config is not None
            and self.multimodal_config.get_limit_per_prompt("video") == 0
        ):
            return 0
        info = MULTIMODAL_REGISTRY.get_processing_info(self.model_config)
        assert isinstance(info, Lfm3VlProcessingInfo)
        num_videos = (
            self.multimodal_config.get_limit_per_prompt("video")
            if self.multimodal_config is not None
            else 1
        )
        return info.get_num_frames_with_most_features(
            seq_len=self.model_config.max_model_len,
            num_videos=num_videos,
        )

    def get_encoder_cudagraph_budget_range(
        self,
        vllm_config: VllmConfig,
    ) -> tuple[int, int]:
        min_budget = 64
        max_budget = min(
            vllm_config.scheduler_config.max_num_batched_tokens,
            self.model_config.max_model_len,
        )
        return (min_budget, max_budget)

    def _get_pixel_values_by_modality(self, mm_kwargs: dict[str, Any]) -> torch.Tensor:
        modality = self.get_input_modality(mm_kwargs)
        key = "pixel_values" if modality == "image" else "pixel_values_videos"
        pixel_values = mm_kwargs[key]
        assert isinstance(pixel_values, torch.Tensor)
        return pixel_values

    def _get_grid_thw_by_modality(self, mm_kwargs: dict[str, Any]) -> list[list[int]]:
        modality = self.get_input_modality(mm_kwargs)
        grid_thw = mm_kwargs[f"{modality}_grid_thw"]
        if isinstance(grid_thw, torch.Tensor):
            return grid_thw.tolist()
        return [list(row) for row in grid_thw]

    def get_encoder_cudagraph_item_specs(
        self, mm_kwargs: dict[str, Any]
    ) -> list["EncoderItemSpec"]:
        from vllm.v1.worker.encoder_cudagraph_defs import EncoderItemSpec

        m = self.vision_tower.spatial_merge_size
        grid_thw = self._get_grid_thw_by_modality(mm_kwargs)
        specs: list[EncoderItemSpec] = []
        for t, h, w in grid_thw:
            if h % m != 0 or w % m != 0:
                raise ValueError(
                    f"Grid shape ({t}, {h}, {w}) must have h and w divisible "
                    f"by spatial_merge_size={m}."
                )
            specs.append(
                EncoderItemSpec(
                    input_size=t * h * w,
                    output_tokens=t * (h // m) * (w // m),
                )
            )
        return specs

    def select_encoder_cudagraph_items(
        self, mm_kwargs: dict[str, Any], indices: list[int]
    ) -> dict[str, Any]:
        modality = self.get_input_modality(mm_kwargs)
        pv_key = "pixel_values" if modality == "image" else "pixel_values_videos"
        grid_key = f"{modality}_grid_thw"

        pixel_values = self._get_pixel_values_by_modality(mm_kwargs)
        raw_grid = mm_kwargs[grid_key]
        grid_thw = self._get_grid_thw_by_modality(mm_kwargs)

        if not indices:
            empty_grid: torch.Tensor | list[list[int]] = (
                torch.empty((0, 3), dtype=raw_grid.dtype, device="cpu")
                if isinstance(raw_grid, torch.Tensor)
                else []
            )
            return {pv_key: pixel_values[:0], grid_key: empty_grid}

        cum_patches = [0]
        for t, h, w in grid_thw:
            cum_patches.append(cum_patches[-1] + t * h * w)

        if len(indices) == 1:
            idx = indices[0]
            selected_pv = pixel_values[cum_patches[idx] : cum_patches[idx + 1]]
        else:
            selected_pv = torch.cat(
                [pixel_values[cum_patches[i] : cum_patches[i + 1]] for i in indices],
                dim=0,
            )

        selected_rows = [grid_thw[i] for i in indices]
        # Keep selected grid_thw on CPU so replay preparation avoids D2H syncs.
        selected_grid: torch.Tensor | list[list[int]] = (
            torch.tensor(selected_rows, dtype=raw_grid.dtype, device="cpu")
            if isinstance(raw_grid, torch.Tensor)
            else selected_rows
        )
        return {pv_key: selected_pv, grid_key: selected_grid}

    def prepare_encoder_cudagraph_capture_inputs(
        self,
        token_budget: int,
        max_batch_size: int,
        max_frames_per_batch: int,
        device: torch.device,
        dtype: torch.dtype,
        path: str = "default",
        axis_keys: tuple[Hashable, ...] | None = None,
    ) -> "EncoderCudaGraphCaptureInputs":
        from vllm.v1.worker.encoder_cudagraph_defs import (
            EncoderCudaGraphCaptureInputs,
        )

        m = self.vision_tower.spatial_merge_size
        num_patches = token_budget * (m**2)
        dummy_pixel_values = torch.zeros(
            num_patches,
            self.vision_tower.patchifier.input_size,
            device=device,
            dtype=dtype,
        )
        metadata = self.vision_tower.prepare_capture_metadata(
            num_patches=num_patches,
            pad_to_num_frames=max(max_batch_size, max_frames_per_batch),
            device=device,
        )
        values = {
            k: v
            for k, v in (metadata | {"pixel_values": dummy_pixel_values}).items()
            if v is not None
        }
        return EncoderCudaGraphCaptureInputs(values=values)

    def prepare_encoder_cudagraph_replay_buffers(
        self,
        mm_kwargs: dict[str, Any],
        max_batch_size: int,
        max_frames_per_batch: int,
        path: str = "default",
    ) -> "EncoderCudaGraphReplayBuffers":
        from vllm.v1.worker.encoder_cudagraph_defs import (
            EncoderCudaGraphReplayBuffers,
        )

        grid_thw_list = self._get_grid_thw_by_modality(mm_kwargs)
        metadata = self.vision_tower.prepare_encoder_metadata(
            grid_thw_list,
            device=self.vision_tower.device,
        )
        metadata.pop("max_seqlen", None)
        pixel_values = self._get_pixel_values_by_modality(mm_kwargs)
        if pixel_values.device != self.vision_tower.device:
            pixel_values = pixel_values.to(self.vision_tower.device, non_blocking=True)
        values: dict[str, torch.Tensor | None] = metadata | {
            "pixel_values": pixel_values,
        }
        return EncoderCudaGraphReplayBuffers(values=values)

    def encoder_cudagraph_forward(
        self,
        values: dict[str, torch.Tensor],
        path: str = "default",
    ) -> torch.Tensor:
        pixel_values = values["pixel_values"]
        vision_out = self.vision_tower(pixel_values, encoder_metadata=values)
        return self.multi_modal_projector(vision_out)

    def encoder_eager_forward(
        self,
        mm_kwargs: dict[str, Any],
        path: str = "default",
    ) -> torch.Tensor:
        pixel_values = self._get_pixel_values_by_modality(mm_kwargs)
        grid_thw = self._get_grid_thw_by_modality(mm_kwargs)
        return self.multi_modal_projector(self.vision_tower(pixel_values, grid_thw))

    def embed_multimodal(self, **kwargs: object) -> MultiModalEmbeddings:
        merge = self.vision_tower.spatial_merge_size**2
        embeddings: list[torch.Tensor] = []
        for pixels_key, grid_key, embeds_key in (
            ("pixel_values", "image_grid_thw", "image_embeds"),
            ("pixel_values_videos", "video_grid_thw", "video_embeds"),
        ):
            grid = kwargs.get(grid_key)
            if (pixels := kwargs.get(pixels_key)) is not None:
                assert isinstance(pixels, torch.Tensor)
                assert isinstance(grid, torch.Tensor)
                grid_thw = grid.tolist()
                features = self.multi_modal_projector(
                    self.vision_tower(pixels, grid_thw)
                )
                embeddings += features.split(
                    [t * h * w // merge for t, h, w in grid_thw]
                )
            elif (embeds := kwargs.get(embeds_key)) is not None:
                assert isinstance(embeds, torch.Tensor)
                assert isinstance(grid, torch.Tensor)
                embeddings += embeds.split((grid.prod(-1) // merge).tolist())
        return embeddings

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
        return self.language_model(
            input_ids, positions, intermediate_tensors, inputs_embeds
        )

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        return self.language_model.compute_logits(hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)
