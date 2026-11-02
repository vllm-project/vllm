# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Direct FP8 projection and cached-context attention for ROCm MLA."""

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from vllm.model_executor.kernels.linear.scaled_mm.aiter import (
    AiterHipbMMPerTokenFp8ScaledMMLinearKernel,
    AiterPerTokenFp8ScaledMMLinearKernel,
    AiterPreshuffledPerTokenFp8ScaledMMLinearKernel,
)
from vllm.model_executor.kernels.linear.scaled_mm.pytorch import (
    RowWiseTorchFP8ScaledMMLinearKernel,
)
from vllm.model_executor.layers.attention.mla_attention import (
    accumulate_mla_context_chunk,
    init_mla_context_partial,
)
from vllm.model_executor.layers.quantization.online.fp8 import Fp8PtpcOnlineLinearMethod
from vllm.v1.attention.ops.rocm_aiter_mla_fp8_quant import fused_qkv_per_tensor_quant
from vllm.v1.attention.ops.rocm_aiter_mla_prefill import gather_compressed_context

if TYPE_CHECKING:
    from vllm.model_executor.layers.attention.mla_attention import (
        MLACommonPrefillMetadata,
    )
    from vllm.v1.attention.backends.mla.rocm_aiter_mla import (
        AiterMLAImpl,
        AiterMLAMetadata,
    )


@dataclass(frozen=True)
class ProjectionWeight:
    weight: torch.Tensor
    scale: torch.Tensor
    preshuffled: bool


def projection_weight(layer: torch.nn.Module, heads: int) -> ProjectionWeight:
    """Return views only. Never re-quantize, repack, or alter the native layer."""
    method = layer.quant_method
    if type(method) is not Fp8PtpcOnlineLinearMethod:
        raise ValueError(
            "FP8 projection requires native fp8_per_channel online quantization"
        )
    kernel = type(method.fp8_linear)
    weight, scale = layer.weight, layer.weight_scale
    if kernel is AiterHipbMMPerTokenFp8ScaledMMLinearKernel:
        # Native hipb_mm stores a transposed VIEW of shuffled [N,K].
        # Transpose back before any contiguous operation.
        weight = weight.t()
        preshuffled = True
    elif kernel is AiterPreshuffledPerTokenFp8ScaledMMLinearKernel:
        preshuffled = True
    elif kernel is AiterPerTokenFp8ScaledMMLinearKernel:
        preshuffled = False
    elif kernel is RowWiseTorchFP8ScaledMMLinearKernel:
        weight = weight.t()
        preshuffled = False
    else:
        raise ValueError(f"Unsupported FP8 projection weight layout: {kernel.__name__}")
    if (
        weight.dtype != torch.float8_e4m3fn
        or weight.shape != (heads * 256, 512)
        or not weight.is_contiguous()
    ):
        raise ValueError(
            "Expected contiguous OCP E4M3 [heads*256,512] projection storage"
        )
    if (
        scale.dtype != torch.float32
        or scale.numel() != heads * 256
        or not scale.is_contiguous()
        or scale.device != weight.device
    ):
        raise ValueError("Expected contiguous FP32 per-output-channel descales")
    if not weight.is_cuda or getattr(layer, "bias", None) is not None:
        raise ValueError("FP8 projection requires GPU weights without bias")
    return ProjectionWeight(weight, scale.view(-1), preshuffled)


def project(gathered, cache_scale, indices, params, heads, output_scales=None):
    """FP8 cache x FP8 weight, FP32 accumulation, direct FP8 or BF16 output."""
    from aiter.ops.flydsl.gather_kv_b_proj import gather_kv_b_proj_flydsl

    if (
        gathered.ndim != 2
        or gathered.shape[1] != 576
        or gathered.dtype != torch.float8_e4m3fn
        or not gathered.is_contiguous()
    ):
        raise ValueError("Expected contiguous OCP E4M3 cache [rows,576]")
    if indices.ndim != 1 or indices.dtype != torch.int32 or not indices.is_contiguous():
        raise ValueError("Expected contiguous int32 cache row indices")
    if (
        cache_scale.dtype != torch.float32
        or cache_scale.numel() != 1
        or not cache_scale.is_contiguous()
    ):
        raise ValueError("Expected scalar FP32 cache descale")
    tensors = [cache_scale, indices, params.weight, params.scale]
    if output_scales is not None:
        if (
            output_scales.dtype != torch.float32
            or output_scales.shape != (2,)
            or not output_scales.is_contiguous()
        ):
            raise ValueError("Expected contiguous FP32 [K,V] output descales")
        tensors.append(output_scales)
    if not gathered.is_cuda or any(t.device != gathered.device for t in tensors):
        raise ValueError("All projection tensors must share a GPU")
    dtype = torch.bfloat16 if output_scales is None else torch.float8_e4m3fn
    with torch.accelerator.device_index(gathered.device.index):
        k = torch.empty(
            (indices.numel(), heads, 192), device=gathered.device, dtype=dtype
        )
        v = torch.empty(
            (indices.numel(), heads, 128), device=gathered.device, dtype=dtype
        )
        kwargs = (
            {}
            if output_scales is None
            else {
                "k_out_scale": output_scales[:1],
                "v_out_scale": output_scales[1:],
            }
        )
        if indices.numel():
            gather_kv_b_proj_flydsl(
                gathered.view(-1, 1, 576),
                cache_scale,
                None,
                indices,
                None,
                params.weight,
                params.scale,
                k,
                v,
                weight_preshuffle=params.preshuffled,
                **kwargs,
            )
    if output_scales is None:
        return k, v
    return k, v, output_scales[:1], output_scales[1:]


def use_fp8_context(max_query_len: int, max_seq_len: int) -> bool:
    # Short calls do not amortize Q quantization and FP8 FMHA dispatch.
    return max_query_len >= 512 and max_seq_len >= 8192


class FP8PrefillContext:
    """Q and projection descales owned by one layer's prefill forward."""

    def __init__(self) -> None:
        from aiter.ops.flydsl import flydsl_flash_attn_fp8_func

        self.attention = flydsl_flash_attn_fp8_func
        self.queries: dict[tuple, tuple[torch.Tensor, torch.Tensor]] = {}
        self.gather_scales: torch.Tensor | None = None

    def prepare(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> None:
        with torch.accelerator.device_index(q.device.index):
            quant = fused_qkv_per_tensor_quant(q, k, v)
            key = (q.data_ptr(), tuple(q.shape), tuple(q.stride()))
            self.queries[key] = (quant[0], quant[3])
            self.gather_scales = torch.cat((quant[6], quant[7]))

    def run(
        self,
        chunk: "MLACommonPrefillMetadata.ContextChunk",
        q: torch.Tensor,
        kv8: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        scale: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        with torch.accelerator.device_index(q.device.index):
            key = (q.data_ptr(), tuple(q.shape), tuple(q.stride()))
            cached = self.queries.get(key)
            if cached is None:
                # Quantize each distinct slice independently, as in the prototype.
                quant = fused_qkv_per_tensor_quant(q, q[:0], q[:0, :, :128])
                cached = quant[0], quant[3]
                self.queries[key] = cached
            q8, qs = cached
            k8, v8, ks, vs = kv8
            return self.attention(
                q8,
                k8,
                v8,
                softmax_scale=scale,
                cu_seqlens_q=chunk.query_start_loc,
                cu_seqlens_kv=chunk.cu_seq_lens,
                max_seqlen_q=chunk.max_query_len,
                max_seqlen_kv=chunk.max_seq_len,
                cross_seqlen=True,
                causal=False,
                return_lse=True,
                q_descale=qs,
                k_descale=ks,
                v_descale=vs,
                out=None,
                stream=None,
            )


def compute_context(
    impl: "AiterMLAImpl",
    q: torch.Tensor,
    cache: torch.Tensor,
    metadata: "AiterMLAMetadata",
    k_scale: torch.Tensor,
    state: FP8PrefillContext,
) -> tuple[torch.Tensor, torch.Tensor]:
    assert metadata.prefill is not None
    prefill = metadata.prefill
    backend = prefill.prefill_backend
    chunked = prefill.chunked_context
    row_groups = metadata.dcp_context_row_indices
    params = impl._flydsl_fp8_projection
    assert backend is not None and chunked is not None
    assert chunked.dcp_manager is not None
    assert params is not None and row_groups is not None
    assert state.gather_scales is not None
    assert len(chunked.chunks) == len(row_groups)
    output = output_lse = None
    for chunk, row_indices in zip(chunked.chunks, row_groups):
        gathered = gather_compressed_context(
            cache,
            chunked.workspace,
            prefill.block_table[chunk.request_slice],
            chunk,
            chunked.dcp_manager.kv_gather,
            torch.float8_e4m3fn,
        )
        query = q[chunk.token_slice]
        if use_fp8_context(chunk.max_query_len, chunk.max_seq_len):
            kv8 = project(
                gathered,
                k_scale,
                row_indices,
                params,
                impl.num_heads,
                state.gather_scales,
            )
            attn_output, attn_lse = state.run(chunk, query, kv8, impl.scale)
        else:
            k, v = project(gathered, k_scale, row_indices, params, impl.num_heads)
            attn_output, attn_lse = backend.run_prefill_context_chunk(
                chunk,
                query,
                k,
                v,
            )
        if output is None:
            if len(chunked.chunks) == 1 and not chunked.empty_token_slices:
                return attn_output, attn_lse
            output, output_lse = init_mla_context_partial(
                chunked,
                attn_output,
                attn_lse,
                num_tokens=q.shape[0],
            )
        accumulate_mla_context_chunk(
            chunk,
            attn_output,
            attn_lse,
            output,
            output_lse,
        )
    assert output is not None and output_lse is not None
    return output, output_lse
