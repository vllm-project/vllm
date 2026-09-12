# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import functools
from dataclasses import dataclass
from typing import cast

import torch

from vllm.distributed import (
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_reduce,
)
from vllm.forward_context import get_forward_context
from vllm.logger import init_logger
from vllm.models.deepseek_v4_1.attention import AttentionOutput, DeepseekV4Attention
from vllm.models.deepseek_v4_1.common.ops import dequantize_and_gather_k_cache
from vllm.models.deepseek_v4_1.sparse_mla import (
    DeepseekV4FlashMLAMetadata,
    DeepseekV4SparseMLABackend,
    DeepseekV4SparseMLAMetadataBuilder,
    DeepseekV41SparseSWAMetadataBuilder,
)
from vllm.platforms import current_platform
from vllm.platforms.rocm import _ON_GFX950
from vllm.triton_utils import tl, triton
from vllm.v1.attention.backend import (
    CommonAttentionMetadata,
)
from vllm.v1.attention.backends.mla.sparse_swa import (
    DeepseekSparseSWABackend,
    DeepseekSparseSWAMetadata,
)
from vllm.v1.attention.ops.rocm_aiter_mla_sparse import (
    build_ragged_indices_from_dense,
    rocm_inv_rope_einsum,
    rocm_sparse_attn_decode,
    rocm_sparse_attn_prefill,
)
from vllm.v1.worker.workspace import current_workspace_manager

logger = init_logger(__name__)


def _trust_dsv4_extra_cache_nan_free(
    kv_cache_dtype: str,
    has_kv_transfer: bool,
    has_extra_cache: bool,
) -> bool:
    return (
        _ON_GFX950
        and kv_cache_dtype == "fp8_ds_mla"
        and not has_kv_transfer
        and has_extra_cache
    )


def _build_indptr_from_lengths(lengths: torch.Tensor) -> torch.Tensor:
    lengths = lengths.to(dtype=torch.int32).contiguous()
    indptr = torch.zeros(lengths.shape[0] + 1, dtype=torch.int32, device=lengths.device)
    torch.cumsum(lengths, dim=0, out=indptr[1:])
    return indptr


def apply_pre_quantized_block_scaled_mm(
    linear: torch.nn.Module,
    x_fp8: torch.Tensor,
    x_scale: torch.Tensor,
) -> torch.Tensor:
    """Block-scaled fp8 GEMM on pre-quantized activations.

    The fused q/kv norm kernel writes fp8 qr + per-1x128 scales; this
    drives the linear's block-scaled GEMM directly with them, bypassing
    apply_weights which would re-quantize the fp8 input. Only valid for
    the wq_b-style column/replicated linears: their output is the local
    TP shard, so no all-reduce is needed.
    """
    from vllm.model_executor.kernels.linear.scaled_mm.BlockScaledMMLinearKernel import (
        FP8BlockParams,
    )

    params = FP8BlockParams.from_layer(linear)
    weight_scale = (
        params.weight_scale
        if params.weight_scale_inv is None
        else params.weight_scale_inv
    )
    kernel = linear.quant_method.fp8_linear
    out = kernel.apply_block_scaled_mm(
        A=x_fp8, B=params.weight, As=x_scale, Bs=weight_scale
    )
    return out.to(dtype=kernel.config.out_dtype)


# ROCm sparse prefill keeps this dense combine local so AMD-specific SWA changes
# do not touch the shared DeepSeek V4 cache utilities.
_SPARSE_PREFILL_TOPK_ALIGNMENT = 128


@triton.jit
def _combine_topk_swa_indices_kernel(
    combined_indices_ptr,
    combined_indices_stride,
    combined_lens_ptr,
    topk_indices_ptr,
    topk_indices_stride,
    query_start_loc_ptr,
    seq_lens_ptr,
    gather_lens_ptr,
    M,
    N,
    TOP_K: tl.constexpr,
    COMPRESS_RATIO: tl.constexpr,
    WINDOW_SIZE: tl.constexpr,
    TOPK_WIDTH: tl.constexpr,
    PADDED_TOP_K: tl.constexpr,
):
    batch_idx = tl.program_id(0)
    worker_id = tl.program_id(1)
    num_workers = tl.num_programs(1)

    base = tl.load(query_start_loc_ptr)
    query_start = tl.load(query_start_loc_ptr + batch_idx) - base
    query_end = tl.load(query_start_loc_ptr + batch_idx + 1) - base
    query_len = query_end - query_start
    seq_len = tl.load(seq_lens_ptr + batch_idx)
    gather_len = tl.load(gather_lens_ptr + batch_idx)
    start_pos = seq_len - query_len
    gather_start = seq_len - gather_len

    for token_idx in range(query_start + worker_id, query_end, num_workers):
        token_idx_in_query = token_idx - query_start
        pos = start_pos + token_idx_in_query
        topk_len = tl.minimum((pos + 1) // COMPRESS_RATIO, TOP_K)
        swa_len = tl.minimum(pos + 1, WINDOW_SIZE)

        topk_offset = tl.arange(0, PADDED_TOP_K)
        topk_mask = topk_offset < topk_len
        safe_topk_offset = tl.where(topk_offset < TOPK_WIDTH, topk_offset, 0)
        topk_indices = tl.load(
            topk_indices_ptr + token_idx * topk_indices_stride + safe_topk_offset,
            mask=topk_mask,
            other=-1,
        )
        valid_topk = (topk_indices >= 0) & (topk_indices < N)
        topk_indices = tl.where(valid_topk, topk_indices + M * batch_idx, -1)
        tl.store(
            combined_indices_ptr + token_idx * combined_indices_stride + topk_offset,
            topk_indices,
            mask=topk_mask,
        )

        swa_offset = tl.arange(0, WINDOW_SIZE)
        tl.store(
            combined_indices_ptr
            + token_idx * combined_indices_stride
            + topk_len
            + swa_offset,
            M * batch_idx + N + swa_offset + pos - swa_len + 1 - gather_start,
            mask=swa_offset < swa_len,
        )

        tl.store(combined_lens_ptr + token_idx, topk_len + swa_len)


def combine_topk_swa_indices(
    topk_indices: torch.Tensor,
    query_start_loc: torch.Tensor,
    seq_lens: torch.Tensor,
    gather_lens: torch.Tensor,
    window_size: int,
    compress_ratio: int,
    topk: int,
    M: int,
    N: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Combine compressed-attention and sliding-window indices with Torch.

    The Triton implementation inherited from DeepSeek V4 launches a
    two-dimensional grid with 128 workers per request.  On gfx950 it can issue
    an out-of-bounds access for V4.1's mixed prefill metadata (including the
    synthetic mixed-token warmup).  This path is prefill-only and the tensors
    are small, so use ordinary Torch indexing until a gfx950-safe fused kernel
    is available.
    """
    topk_indices = topk_indices.reshape(topk_indices.shape[0], -1).contiguous()
    num_tokens = topk_indices.shape[0]
    combined_topk = (
        (topk + window_size + _SPARSE_PREFILL_TOPK_ALIGNMENT - 1)
        // _SPARSE_PREFILL_TOPK_ALIGNMENT
        * _SPARSE_PREFILL_TOPK_ALIGNMENT
    )
    combined_indices = torch.full(
        (num_tokens, combined_topk),
        fill_value=-1,
        dtype=torch.int32,
        device=topk_indices.device,
    )
    combined_lens = torch.empty(
        num_tokens, dtype=torch.int32, device=topk_indices.device
    )

    # query_start_loc may have a non-zero base for a narrowed mixed batch.
    query_lens = query_start_loc[1:] - query_start_loc[:-1]
    req_ids = torch.repeat_interleave(
        torch.arange(seq_lens.shape[0], device=seq_lens.device), query_lens
    )
    query_starts = query_start_loc[:-1] - query_start_loc[0]
    token_offsets = torch.arange(num_tokens, device=seq_lens.device) - (
        torch.repeat_interleave(query_starts, query_lens)
    )
    positions = seq_lens[req_ids] - query_lens[req_ids] + token_offsets

    logical_topk_width = min(topk, topk_indices.shape[1])
    topk_lens = torch.minimum(
        (positions + 1) // compress_ratio,
        torch.full_like(positions, logical_topk_width),
    ).clamp_min(0)
    topk_offsets = torch.arange(logical_topk_width, device=seq_lens.device)
    topk_mask = topk_offsets[None, :] < topk_lens[:, None]
    topk_values = topk_indices[:, :logical_topk_width].to(torch.int32)
    topk_valid = topk_mask & (topk_values >= 0) & (topk_values < N)
    combined_indices[:, :logical_topk_width] = torch.where(
        topk_valid,
        topk_values + (M * req_ids).to(torch.int32)[:, None],
        -1,
    )

    swa_lens = torch.minimum(
        positions + 1, torch.full_like(positions, window_size)
    ).clamp_min(0)
    swa_offsets = torch.arange(window_size, device=seq_lens.device)
    swa_mask = swa_offsets[None, :] < swa_lens[:, None]
    swa_columns = topk_lens[:, None] + swa_offsets[None, :]
    gather_starts = seq_lens - gather_lens
    swa_values = (
        M * req_ids[:, None]
        + N
        + swa_offsets[None, :]
        + positions[:, None]
        - swa_lens[:, None]
        + 1
        - gather_starts[req_ids, None]
    ).to(torch.int32)
    rows = torch.arange(num_tokens, device=seq_lens.device)[:, None].expand_as(
        swa_columns
    )
    flat_dst = rows[swa_mask] * combined_topk + swa_columns[swa_mask]
    combined_indices.view(-1).index_copy_(0, flat_dst, swa_values[swa_mask])
    combined_lens.copy_((topk_lens + swa_lens).to(torch.int32))
    return combined_indices, combined_lens


@triton.jit
def _compute_topk_lens_kernel(
    topk_lens_ptr,
    topk_indices_ptr,
    topk_indices_stride,
    topk,
    is_valid_token_ptr,
    TRITON_BLOCK_SIZE: tl.constexpr,
):
    token_idx = tl.program_id(0)
    is_valid_token = tl.load(is_valid_token_ptr + token_idx)

    count = tl.zeros((), dtype=tl.int32)
    for i in range(0, topk, TRITON_BLOCK_SIZE):
        offset = i + tl.arange(0, TRITON_BLOCK_SIZE)
        mask = offset < topk
        local_idx = tl.load(
            topk_indices_ptr + token_idx * topk_indices_stride + offset,
            mask=mask,
            other=-1,
        )
        count += tl.sum((local_idx >= 0).to(tl.int32), axis=0)

    tl.store(topk_lens_ptr + token_idx, tl.where(is_valid_token, count, 0))


@triton.jit
def _pack_global_topk_ragged_kernel(
    global_topk_ragged_ptr,
    topk_indptr_ptr,
    topk_indices_ptr,
    topk_indices_stride,
    token_to_req_indices_ptr,
    block_table_ptr,
    block_table_stride,
    block_size,
    topk,
    BLOCK_SIZE: tl.constexpr,
):
    token_idx = tl.program_id(0)
    block_idx = tl.program_id(1)
    offset = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    out_start = tl.load(topk_indptr_ptr + token_idx)
    out_end = tl.load(topk_indptr_ptr + token_idx + 1)
    out_len = out_end - out_start
    if block_idx * BLOCK_SIZE >= out_len:
        return

    req_idx = tl.load(token_to_req_indices_ptr + token_idx)
    mask = (offset < out_len) & (offset < topk)
    local_idx = tl.load(
        topk_indices_ptr + token_idx * topk_indices_stride + offset,
        mask=mask,
        other=-1,
    )
    valid = mask & (local_idx >= 0)
    block_indices = local_idx // block_size
    block_numbers = tl.load(
        block_table_ptr + req_idx * block_table_stride + block_indices,
        mask=valid,
        other=0,
    )
    block_offsets = local_idx % block_size
    slot_ids = tl.where(valid, block_numbers * block_size + block_offsets, -1)
    tl.store(global_topk_ragged_ptr + out_start + offset, slot_ids, mask=mask)


def compute_global_topk_ragged_indices_and_indptr(
    topk_indices: torch.Tensor,
    token_to_req_indices: torch.Tensor,
    block_table: torch.Tensor,
    block_size: int,
    is_valid_token: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    topk_indices = topk_indices.reshape(topk_indices.shape[0], -1).contiguous()
    num_tokens = topk_indices.shape[0]
    topk = topk_indices.shape[1]

    topk_lens = torch.empty(num_tokens, dtype=torch.int32, device=topk_indices.device)
    _compute_topk_lens_kernel[(num_tokens,)](
        topk_lens,
        topk_indices,
        topk_indices.stride(0),
        topk,
        is_valid_token,
        TRITON_BLOCK_SIZE=1024,
    )

    topk_indptr = _build_indptr_from_lengths(topk_lens)
    global_topk_ragged = torch.empty(
        num_tokens * topk,
        dtype=torch.int32,
        device=topk_indices.device,
    )
    if global_topk_ragged.numel() > 0:
        block = 128
        _pack_global_topk_ragged_kernel[(num_tokens, triton.cdiv(topk, block))](
            global_topk_ragged,
            topk_indptr,
            topk_indices,
            topk_indices.stride(0),
            token_to_req_indices,
            block_table,
            block_table.stride(0),
            block_size,
            topk,
            BLOCK_SIZE=block,
        )
    return global_topk_ragged, topk_indptr, topk_lens


def _copy_ragged_to_graph_buffers(
    ragged_indices: torch.Tensor,
    ragged_indptr: torch.Tensor,
    ragged_indices_buffer: torch.Tensor,
    ragged_indptr_buffer: torch.Tensor,
    num_rows: int,
    max_entries_per_row: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Copy dynamic ragged metadata into persistent CUDA graph buffers.

    FULL decode graphs capture kernel argument addresses. Keep the returned
    tensors backed by stable storage, while indptr continues to bound reads.
    """
    indptr_out = ragged_indptr_buffer[: num_rows + 1]
    indptr_out.copy_(ragged_indptr, non_blocking=True)

    max_entries = max(num_rows * max_entries_per_row, 1)
    ragged_out = ragged_indices_buffer[:max_entries]
    source_entries = ragged_indices.numel()
    if source_entries > 0:
        ragged_out[:source_entries].copy_(ragged_indices, non_blocking=True)
    if _ON_GFX950:
        # Preserve the graph-stable base pointer while exposing source capacity
        # to the sync-free split selector; indptr still carries the true NNZ.
        ragged_out = ragged_out[: max(source_entries, 1)]
    return ragged_out, indptr_out


@dataclass
class DeepseekV4ROCMAiterSparseSWAMetadata(DeepseekSparseSWAMetadata):
    decode_swa_ragged_indices: torch.Tensor | None = None
    decode_swa_ragged_indptr: torch.Tensor | None = None


class DeepseekV4ROCMAiterSparseSWAMetadataBuilder(DeepseekV41SparseSWAMetadataBuilder):
    # Keep fused multi-step decode disabled until update_draft_decode_metadata()
    # also refreshes the ROCm-specific ragged SWA indices and indptrs.
    supports_draft_decode_metadata_update = False

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        max_tokens = self.vllm_config.scheduler_config.max_num_batched_tokens
        # The non-causal (DSpark draft) path widens each token's SWA index list
        # to ``noncausal_index_width`` (>= window_size), so size the persistent
        # ragged buffer to the wider bound to cover both causal and non-causal.
        swa_index_width = max(self.window_size, self.noncausal_index_width)
        self.decode_swa_ragged_indices_buffer = torch.empty(
            max_tokens * swa_index_width,
            dtype=torch.int32,
            device=self.device,
        )
        self.decode_swa_ragged_indptr_buffer = torch.empty(
            max_tokens + 1,
            dtype=torch.int32,
            device=self.device,
        )

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> DeepseekV4ROCMAiterSparseSWAMetadata:
        base = super().build(
            common_prefix_len=common_prefix_len,
            common_attn_metadata=common_attn_metadata,
            fast_build=fast_build,
        )

        ragged_indices = None
        ragged_indptr = None
        if (
            base.num_decode_tokens > 0
            and base.decode_swa_indices is not None
            and base.decode_swa_lens is not None
        ):
            ragged_indices, ragged_indptr = build_ragged_indices_from_dense(
                base.decode_swa_indices.reshape(
                    base.num_decode_tokens, base.decode_swa_width
                ),
                base.decode_swa_lens,
            )
            ragged_indices, ragged_indptr = _copy_ragged_to_graph_buffers(
                ragged_indices,
                ragged_indptr,
                self.decode_swa_ragged_indices_buffer,
                self.decode_swa_ragged_indptr_buffer,
                base.num_decode_tokens,
                base.decode_swa_width,
            )

        return DeepseekV4ROCMAiterSparseSWAMetadata(
            **vars(base),
            decode_swa_ragged_indices=ragged_indices,
            decode_swa_ragged_indptr=ragged_indptr,
        )


class DeepseekV4ROCMAiterMLASparseBackend(DeepseekV4SparseMLABackend):
    @staticmethod
    def get_name() -> str:
        return "ROCM_FLASHMLA_SPARSE_DSV4"

    @staticmethod
    def get_builder_cls() -> type[DeepseekV4SparseMLAMetadataBuilder]:
        return DeepseekV4SparseMLAMetadataBuilder


class DeepseekV41ROCMAiterSparseSWABackend(DeepseekSparseSWABackend):
    @staticmethod
    def get_builder_cls() -> type["DeepseekV4ROCMAiterSparseSWAMetadataBuilder"]:
        return DeepseekV4ROCMAiterSparseSWAMetadataBuilder


class DeepseekV41ROCMAiterMLAAttention(DeepseekV4Attention):
    """ROCm sparse MLA attention layer for DeepSeek V4.1."""

    backend_cls = DeepseekV4ROCMAiterMLASparseBackend
    swa_backend_cls = DeepseekV41ROCMAiterSparseSWABackend

    def __init__(self, *args, **kwargs):
        vllm_config = args[0] if args else kwargs["vllm_config"]
        super().__init__(*args, **kwargs)
        # CUDA executes WO_A with a quantized grouped-BMM kernel.  ROCm's
        # correctness path below dequantizes WO_A once and uses torch.einsum,
        # so retain the ordinary MXFP8 linear kernel for post-load processing
        # instead of asking for the CUDA-only BMM kernel.
        self.wo_a.is_bmm = False
        self._has_kv_transfer = vllm_config.kv_transfer_config is not None
        # Block scale for the preshuffled weight; None = not preshuffled.
        self._wqa_wkv_scale: torch.Tensor | None = None
        self._wo_b_scale: torch.Tensor | None = None
        self._fused_compressor_weight: torch.Tensor | None
        self.register_buffer("_fused_compressor_weight", None, persistent=False)
        self._fused_compressor_split_sizes: tuple[int, int] | None = None

    @classmethod
    def get_padded_num_q_heads(cls, num_heads: int) -> int:
        return num_heads

    def prepare_attn_preshuffle(self) -> None:
        from vllm._aiter_ops import rocm_aiter_ops

        if not rocm_aiter_ops.is_enabled():
            return
        from vllm.model_executor.layers.quantization.utils.fp8_utils import (
            _upcast_e8m0_to_fp32,
        )
        from vllm.model_executor.utils import replace_parameter

        def _prep(linear) -> torch.Tensor | None:
            w = getattr(linear, "weight", None)
            if w is None or w.dim() != 2:
                return None
            # K % 128 (group-128 quant) and N % 16 (shuffle_weight) must hold.
            if w.shape[-1] % 128 != 0 or w.shape[0] % 16 != 0:
                return None
            ws = getattr(linear, "weight_scale_inv", None)  # per-block scale
            if ws is None:
                return None
            if ws.dtype == torch.float8_e8m0fnu:
                ws = _upcast_e8m0_to_fp32(ws).contiguous()
            # Shuffle the weight in place (single weight, no unshuffled copy).
            replace_parameter(
                linear,
                "weight",
                rocm_aiter_ops.shuffle_weight(w.data, layout=(16, 16)),
            )
            return ws

        self._wqa_wkv_scale = _prep(self.fused_wqa_wkv)
        self._wo_b_scale = _prep(self.wo_b)

    def prepare_compressor_gemm_fusion(self) -> bool:
        # V4.1 derives index keys from the source compressor's emitted latent
        # and has no nested ``indexer.compressor``.  Keep the projections
        # separate and use the shared linear/PyTorch correctness path.
        return False

    def _bpre_attn_gemm(
        self,
        weight: torch.Tensor,
        scale: torch.Tensor,
        x: torch.Tensor,
        reduce_tp: bool,
    ) -> torch.Tensor:
        from vllm._aiter_ops import rocm_aiter_ops

        x_fp8, x_scale = rocm_aiter_ops.group_fp8_quant(x, transpose_scale=True)
        out = rocm_aiter_ops.gemm_a8w8_blockscale_bpreshuffle(
            x_fp8, weight, x_scale, scale, output_dtype=x.dtype
        )
        if reduce_tp and get_tensor_model_parallel_world_size() > 1:
            out = tensor_model_parallel_all_reduce(out)
        return out

    def _fused_wqa_wkv_gemm(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self._wqa_wkv_scale is not None and hidden_states.dim() == 2:
            return self._bpre_attn_gemm(
                self.fused_wqa_wkv.weight, self._wqa_wkv_scale, hidden_states, False
            )
        return super()._fused_wqa_wkv_gemm(hidden_states)

    def _run_parallel_input_projections(
        self, hidden_states: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        return super()._run_parallel_input_projections(hidden_states)

    @functools.cached_property
    def _wq_b_uses_aiter_block_scaled(self) -> bool:
        """True when both wq_b GEMMs run the aiter block-scaled fp8 kernel.

        Cached: the linear kernels and the aiter env gates are fixed once
        the model is built, so this is evaluated at the first forward
        only.

        The fused norm+quant path is only valid if the quant and GEMM it
        replaces are exactly the aiter ones; otherwise fall back to the
        shared path.
        """
        from vllm._aiter_ops import rocm_aiter_ops
        from vllm.model_executor.kernels.linear.scaled_mm import (
            Fp8BlockScaledMMLinearKernel,
        )

        if not rocm_aiter_ops.is_linear_fp8_enabled():
            return False

        linears = [self.wq_b]
        if self.indexer is not None:
            linears.append(self.indexer.wq_b)
        for linear in linears:
            kernel = getattr(getattr(linear, "quant_method", None), "fp8_linear", None)
            if not isinstance(kernel, Fp8BlockScaledMMLinearKernel):
                return False
        return True

    def _split_qkv_and_norm(
        self, qr_kv: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
        """Fuse q/kv RMSNorm + per-1x128 fp8 q quant into one aiter kernel.

        The shared path norms q and kv in one triton kernel and the wq_b
        linears then re-read the bf16 qr to quantize it. The aiter kernel
        computes both RMSNorms (fp32 accumulate) and the fp8 group quant
        in a single pass, writing fp8 qr + group scales directly; both
        wq_b GEMMs (attention and indexer) then consume that pair and
        skip their own input quant. kv stays bf16: the fused insert
        kernel RoPE/quantizes it itself. Falls back to the shared path
        when the aiter linear path is not active.
        """
        qr, kv = qr_kv.split([self.q_lora_rank, self.head_dim], dim=-1)
        if not (
            qr.dim() == 2
            and qr.shape[0] > 0
            and self.q_lora_rank % 128 == 0
            and self._wq_b_uses_aiter_block_scaled
        ):
            return super()._split_qkv_and_norm(qr_kv)

        from vllm._aiter_ops import rocm_aiter_ops

        return rocm_aiter_ops.fused_qk_rmsnorm_group_quant(
            q=qr,
            q_weight=self.q_norm.weight.data,
            q_epsilon=self.eps,
            kv=kv,
            kv_weight=self.kv_norm.weight.data,
            kv_epsilon=self.eps,
            group_size=128,
            transpose_scale=False,
        )

    def _o_proj(self, o: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        # ROCm BF16 reference wo_a path (inverse RoPE + einsum) + wo_b.
        z = rocm_inv_rope_einsum(
            self.rotary_emb,
            o,
            positions,
            self.rope_head_dim,
            self.n_local_groups,
            self.o_lora_rank,
            self.wo_a,
        )
        zf = z.flatten(1)
        if self._wo_b_scale is not None and zf.dim() == 2:
            result = self._bpre_attn_gemm(self.wo_b.weight, self._wo_b_scale, zf, True)
        else:
            result = self.wo_b(zf)
        return result

    def forward_mqa(
        self,
        q: torch.Tensor,
        kv: torch.Tensor,
        positions: torch.Tensor,
        output: AttentionOutput,
    ) -> None:
        assert isinstance(output, torch.Tensor)
        assert output.shape == q.shape, (
            f"output buffer shape {output.shape} must match q shape {q.shape}"
        )
        assert output.dtype == q.dtype, (
            f"output buffer dtype {output.dtype} must match q dtype {q.dtype}"
        )

        forward_context = get_forward_context()
        attn_metadata = forward_context.attn_metadata

        if attn_metadata is None:
            # Warmup dummy run: no real metadata. Reserve the same bf16
            # gather workspace _forward_prefill would; the dequantize / topk
            # / sparse_fwd kernels are skipped this step.
            swa_only = self.compress_ratio == 0
            N = (
                0
                if swa_only
                else (self.max_model_len + self.compress_ratio - 1)
                // self.compress_ratio
            )
            M = N + self.window_size + self.max_num_batched_tokens
            current_workspace_manager().get_simultaneous(
                ((self.PREFILL_CHUNK_SIZE, M, q.shape[-1]), torch.bfloat16),
            )
            output.zero_()
            return

        assert isinstance(attn_metadata, dict)
        rocm_metadata = cast(
            DeepseekV4FlashMLAMetadata | None,
            attn_metadata.get(self.compressed_cache_prefix)
            if self.compressed_cache_prefix is not None
            else None,
        )
        swa_metadata = cast(
            DeepseekV4ROCMAiterSparseSWAMetadata | None,
            attn_metadata.get(self.swa_cache_layer.prefix),
        )
        assert swa_metadata is not None

        swa_only = self.compress_ratio == 0
        self_kv_cache = None if swa_only else self._compressed_kv_cache()
        swa_kv_cache = self.swa_cache_layer.kv_cache

        num_decodes = swa_metadata.num_decodes
        num_prefills = swa_metadata.num_prefills
        num_decode_tokens = swa_metadata.num_decode_tokens

        if num_prefills > 0:
            self._forward_prefill(
                q=q[num_decode_tokens:],
                positions=positions[num_decode_tokens:],
                compressed_k_cache=self_kv_cache,
                swa_k_cache=swa_kv_cache,
                output=output[num_decode_tokens:],
                attn_metadata=rocm_metadata,
                swa_metadata=swa_metadata,
            )
        if num_decodes > 0:
            self._forward_decode(
                q=q[:num_decode_tokens],
                kv_cache=self_kv_cache,
                swa_metadata=swa_metadata,
                attn_metadata=rocm_metadata,
                swa_only=swa_only,
                output=output[:num_decode_tokens],
            )

    def _forward_decode(
        self,
        q: torch.Tensor,
        kv_cache: torch.Tensor | None,
        swa_metadata: DeepseekV4ROCMAiterSparseSWAMetadata,
        attn_metadata: DeepseekV4FlashMLAMetadata | None,
        swa_only: bool,
        output: torch.Tensor,
    ) -> None:
        num_decodes = swa_metadata.num_decodes
        num_decode_tokens = swa_metadata.num_decode_tokens

        topk_lens = None
        topk_ragged_indices = None
        topk_ragged_indptr = None
        if not swa_only:
            # Local indices filled by the index-source layer's indexer.
            assert attn_metadata is not None
            assert swa_metadata.is_valid_token is not None
            assert self.topk_indices_buffer is not None
            block_size = attn_metadata.block_size // self.compress_ratio
            is_valid = swa_metadata.is_valid_token[:num_decode_tokens]
            (
                topk_ragged_indices,
                topk_ragged_indptr,
                topk_lens,
            ) = compute_global_topk_ragged_indices_and_indptr(
                self.topk_indices_buffer[:num_decode_tokens],
                swa_metadata.token_to_req_indices,
                attn_metadata.block_table[:num_decodes],
                block_size,
                is_valid,
            )

        rocm_sparse_attn_decode(
            q=q,
            kv_cache=kv_cache,
            swa_k_cache=self.swa_cache_layer.kv_cache,
            swa_only=swa_only,
            topk_indices=None,
            topk_lens=topk_lens,
            swa_indices=swa_metadata.decode_swa_indices,
            swa_lens=swa_metadata.decode_swa_lens,
            swa_ragged_indices=swa_metadata.decode_swa_ragged_indices,
            swa_ragged_indptr=swa_metadata.decode_swa_ragged_indptr,
            topk_ragged_indices=topk_ragged_indices,
            topk_ragged_indptr=topk_ragged_indptr,
            attn_sink=self.attn_sink,
            scale=self.scale,
            head_dim=self.head_dim,
            nope_head_dim=self.nope_head_dim,
            rope_head_dim=self.rope_head_dim,
            output=output,
            extra_cache_nan_free=_trust_dsv4_extra_cache_nan_free(
                self.kv_cache_dtype,
                self._has_kv_transfer,
                not swa_only and kv_cache is not None,
            ),
        )

    def _forward_prefill(
        self,
        q: torch.Tensor,
        positions: torch.Tensor,
        compressed_k_cache: torch.Tensor | None,
        swa_k_cache: torch.Tensor,
        output: torch.Tensor,
        attn_metadata: DeepseekV4FlashMLAMetadata | None,
        swa_metadata: DeepseekV4ROCMAiterSparseSWAMetadata,
    ) -> None:
        swa_only = attn_metadata is None

        num_prefills = swa_metadata.num_prefills
        num_prefill_tokens = swa_metadata.num_prefill_tokens
        num_decodes = swa_metadata.num_decodes
        num_decode_tokens = swa_metadata.num_decode_tokens

        seq_lens = swa_metadata.prefill_seq_lens
        gather_lens = swa_metadata.prefill_gather_lens
        assert seq_lens is not None
        assert gather_lens is not None

        query_start_loc_cpu = swa_metadata.query_start_loc_cpu
        query_start_loc = swa_metadata.query_start_loc
        assert query_start_loc_cpu is not None
        assert query_start_loc is not None
        prefill_token_base = query_start_loc_cpu[num_decodes]

        # Local indices filled by the index source; SWA-only layers pass
        # top_k=0 and never read them.
        assert self.topk_indices_buffer is not None
        topk_indices = self.topk_indices_buffer[num_decode_tokens:]
        topk_indices = topk_indices[:num_prefill_tokens]
        if not swa_only:
            top_k = topk_indices.shape[-1]
            N = (self.max_model_len + self.compress_ratio - 1) // self.compress_ratio
        else:
            top_k = 0
            N = 0

        M = N + self.window_size + self.max_num_batched_tokens
        num_chunks = (num_prefills + self.PREFILL_CHUNK_SIZE - 1) // (
            self.PREFILL_CHUNK_SIZE
        )

        workspace_manager = current_workspace_manager()
        kv = workspace_manager.get_simultaneous(
            ((self.PREFILL_CHUNK_SIZE, M, q.shape[-1]), torch.bfloat16),
        )[0]
        for chunk_idx in range(num_chunks):
            chunk_start = chunk_idx * self.PREFILL_CHUNK_SIZE
            chunk_end = min(chunk_start + self.PREFILL_CHUNK_SIZE, num_prefills)
            chunk_size = chunk_end - chunk_start
            if not swa_only:
                assert attn_metadata is not None
                assert compressed_k_cache is not None
                block_table = attn_metadata.block_table[num_decodes:]
                # compressed_k_cache is OCP on every platform (Triton encoder).
                dequantize_and_gather_k_cache(
                    kv[:chunk_size],
                    compressed_k_cache,
                    seq_lens=seq_lens[chunk_start:chunk_end] // self.compress_ratio,
                    gather_lens=None,
                    block_table=block_table[chunk_start:chunk_end],
                    block_size=attn_metadata.block_size // self.compress_ratio,
                    offset=0,
                    use_fnuz=False,
                )

            swa_block_table = swa_metadata.block_table[num_decodes:]
            dequantize_and_gather_k_cache(
                kv[:chunk_size],
                swa_k_cache,
                seq_lens=seq_lens[chunk_start:chunk_end],
                gather_lens=gather_lens[chunk_start:chunk_end],
                block_table=swa_block_table[chunk_start:chunk_end],
                block_size=swa_metadata.block_size,
                offset=N,
                use_fnuz=current_platform.is_fp8_fnuz(),
            )

            query_start = (
                query_start_loc_cpu[num_decodes + chunk_start] - prefill_token_base
            )
            query_end = (
                query_start_loc_cpu[num_decodes + chunk_end] - prefill_token_base
            )

            combined_indices, combined_lens = combine_topk_swa_indices(
                topk_indices[query_start:query_end],
                query_start_loc[
                    num_decodes + chunk_start : num_decodes + chunk_end + 1
                ],
                seq_lens[chunk_start:chunk_end],
                gather_lens[chunk_start:chunk_end],
                self.window_size,
                self.compress_ratio,
                top_k,
                M,
                N,
            )
            rocm_sparse_attn_prefill(
                q=q[query_start:query_end],
                kv=kv.view(-1, 1, q.shape[-1]),
                indices=combined_indices,
                topk_length=combined_lens,
                scale=self.scale,
                head_dim=self.head_dim,
                nope_head_dim=self.nope_head_dim,
                rope_head_dim=self.rope_head_dim,
                attn_sink=self.attn_sink,
                output=output[query_start:query_end],
            )
