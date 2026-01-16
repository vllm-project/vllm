# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Custom Sparse Attention Indexer layers."""

import torch

from vllm._aiter_ops import rocm_aiter_ops
from vllm.forward_context import get_forward_context
from vllm.logger import init_logger
from vllm.model_executor.custom_op import CustomOp
from vllm.platforms import current_platform
from vllm.utils.deep_gemm import fp8_mqa_logits, fp8_paged_mqa_logits
from vllm.utils.torch_utils import direct_register_custom_op
from vllm.v1.attention.backends.mla.indexer import (
    DeepseekV32IndexerMetadata,
)
from vllm.v1.attention.ops.common import pack_seq_triton, unpack_seq_triton
from vllm.v1.worker.workspace import current_workspace_manager

if current_platform.is_cuda_alike():
    from vllm import _custom_ops as ops
elif current_platform.is_xpu():
    from vllm._ipex_ops import ipex_ops as ops

logger = init_logger(__name__)


def sparse_attn_indexer(
    hidden_states: torch.Tensor,
    k_cache_prefix: str,
    kv_cache: torch.Tensor,
    q_fp8: torch.Tensor,
    k: torch.Tensor,
    weights: torch.Tensor,
    quant_block_size: int,
    scale_fmt: str | None,
    topk_tokens: int,
    head_dim: int,
    max_model_len: int,
    total_seq_lens: int,
    topk_indices_buffer: torch.Tensor,
) -> torch.Tensor:
    # careful! this will be None in dummy run
    attn_metadata = get_forward_context().attn_metadata
    fp8_dtype = current_platform.fp8_dtype()

    # assert isinstance(attn_metadata, dict)
    if not isinstance(attn_metadata, dict):
        # Reserve workspace for indexer during profiling run
        current_workspace_manager().get_simultaneous(
            ((total_seq_lens, head_dim), torch.float8_e4m3fn),
            ((total_seq_lens, 4), torch.uint8),
        )
        return sparse_attn_indexer_fake(
            hidden_states,
            k_cache_prefix,
            kv_cache,
            q_fp8,
            k,
            weights,
            quant_block_size,
            scale_fmt,
            topk_tokens,
            head_dim,
            max_model_len,
            total_seq_lens,
            topk_indices_buffer,
        )
    attn_metadata = attn_metadata[k_cache_prefix]
    assert isinstance(attn_metadata, DeepseekV32IndexerMetadata)
    slot_mapping = attn_metadata.slot_mapping
    has_decode = attn_metadata.num_decodes > 0
    has_prefill = attn_metadata.num_prefills > 0
    num_decode_tokens = attn_metadata.num_decode_tokens

    ops.indexer_k_quant_and_cache(
        k,
        kv_cache,
        slot_mapping,
        quant_block_size,
        scale_fmt,
    )

    topk_indices_buffer[: hidden_states.shape[0]] = -1
    if has_prefill:
        prefill_metadata = attn_metadata.prefill

        # Get the full shared workspace buffers once (will allocate on first use)
        workspace_manager = current_workspace_manager()
        k_fp8_full, k_scale_full = workspace_manager.get_simultaneous(
            ((total_seq_lens, head_dim), fp8_dtype),
            ((total_seq_lens, 4), torch.uint8),
        )
        for chunk in prefill_metadata.chunks:
            k_fp8 = k_fp8_full[: chunk.total_seq_lens]
            k_scale = k_scale_full[: chunk.total_seq_lens]
            ops.cp_gather_indexer_k_quant_cache(
                kv_cache,
                k_fp8,
                k_scale,
                chunk.block_table,
                chunk.cu_seq_lens,
            )

            logits = fp8_mqa_logits(
                q_fp8[chunk.token_start : chunk.token_end],
                (k_fp8, k_scale.view(torch.float32).flatten()),
                weights[chunk.token_start : chunk.token_end],
                chunk.cu_seqlen_ks,
                chunk.cu_seqlen_ke,
            )
            num_rows = logits.shape[0]

            topk_indices = topk_indices_buffer[
                chunk.token_start : chunk.token_end, :topk_tokens
            ]
            torch.ops._C.top_k_per_row_prefill(
                logits,
                chunk.cu_seqlen_ks,
                chunk.cu_seqlen_ke,
                topk_indices,
                num_rows,
                logits.stride(0),
                logits.stride(1),
                topk_tokens,
            )

    if has_decode:
        decode_metadata = attn_metadata.decode
        # kv_cache size requirement [num_block, block_size, n_head, head_dim],
        # we only have [num_block, block_size, head_dim],
        kv_cache = kv_cache.unsqueeze(-2)
        decode_lens = decode_metadata.decode_lens
        if decode_metadata.requires_padding:
            # pad in edge case where we have short chunked prefill length <
            # decode_threshold since we unstrictly split
            # prefill and decode by decode_threshold
            # (currently set to 1 + speculative tokens)
            padded_q_fp8_decode_tokens = pack_seq_triton(
                q_fp8[:num_decode_tokens], decode_lens
            )
        else:
            padded_q_fp8_decode_tokens = q_fp8[:num_decode_tokens].reshape(
                decode_lens.shape[0], -1, *q_fp8.shape[1:]
            )
        # TODO: move and optimize below logic with triton kernels
        batch_size = padded_q_fp8_decode_tokens.shape[0]
        next_n = padded_q_fp8_decode_tokens.shape[1]
        assert batch_size == decode_metadata.seq_lens.shape[0]
        num_padded_tokens = batch_size * next_n

        logits = fp8_paged_mqa_logits(
            padded_q_fp8_decode_tokens,
            kv_cache,
            weights[:num_padded_tokens],
            decode_metadata.seq_lens,
            decode_metadata.block_table,
            decode_metadata.schedule_metadata,
            max_model_len=max_model_len,
        )

        num_rows = logits.shape[0]

        topk_indices = topk_indices_buffer[:num_padded_tokens, :topk_tokens]
        torch.ops._C.top_k_per_row_decode(
            logits,
            next_n,
            decode_metadata.seq_lens,
            topk_indices,
            num_rows,
            logits.stride(0),
            logits.stride(1),
            topk_tokens,
        )

        if decode_metadata.requires_padding:
            # if padded, we need to unpack
            # the topk indices removing padded tokens
            topk_indices = unpack_seq_triton(
                topk_indices.reshape(batch_size, -1, topk_indices.shape[-1]),
                decode_lens,
            )
            topk_indices_buffer[:num_decode_tokens, : topk_indices.shape[-1]] = (
                topk_indices
            )

    return topk_indices_buffer


def sparse_attn_indexer_fake(
    hidden_states: torch.Tensor,
    k_cache_prefix: str,
    kv_cache: torch.Tensor,
    q_fp8: torch.Tensor,
    k: torch.Tensor,
    weights: torch.Tensor,
    quant_block_size: int,
    scale_fmt: str | None,
    topk_tokens: int,
    head_dim: int,
    max_model_len: int,
    total_seq_lens: int,
    topk_indices_buffer: torch.Tensor | None,
) -> torch.Tensor:
    return topk_indices_buffer


direct_register_custom_op(
    op_name="sparse_attn_indexer",
    op_func=sparse_attn_indexer,
    mutates_args=["topk_indices_buffer"],
    fake_impl=sparse_attn_indexer_fake,
    dispatch_key=current_platform.dispatch_key,
)


@CustomOp.register("sparse_attn_indexer")
class SparseAttnIndexer(CustomOp):
    """Sparse Attention Indexer Custom Op Layer. This layer is extracted as a
    separate custom op since it involves heavy custom kernels like `mqa_logits`,
    `paged_mqa_logits` and `top_k_per_row`, etc. Those kernels maybe requires
    specific memory layout or implementation for different hardware backends to
    achieve optimal performance.

    For now, the default native path will use CUDA backend path. Other platform
    may requires add the corresponding Custom Op name `sparse_attn_indexer` to
    `custom_ops` in `CompilationConfig` to enable the platform specific path.
    """

    def __init__(
        self,
        k_cache,
        quant_block_size: int,
        scale_fmt: str,
        topk_tokens: int,
        head_dim: int,
        max_model_len: int,
        max_total_seq_len: int,
        topk_indices_buffer: torch.Tensor,
    ):
        super().__init__()
        self.k_cache = k_cache
        self.quant_block_size = quant_block_size
        self.scale_fmt = scale_fmt
        self.topk_tokens = topk_tokens
        self.head_dim = head_dim
        self.max_model_len = max_model_len
        self.max_total_seq_len = max_total_seq_len
        self.topk_indices_buffer = topk_indices_buffer

    def forward_native(
        self,
        hidden_states: torch.Tensor,
        q_fp8: torch.Tensor,
        k: torch.Tensor,
        weights: torch.Tensor,
    ):
        if current_platform.is_cuda():
            return self.forward_cuda(hidden_states, q_fp8, k, weights)
        elif current_platform.is_rocm():
            return self.forward_hip(hidden_states, q_fp8, k, weights)
        else:
            raise NotImplementedError(
                "SparseAttnIndexer native forward is only implemented for "
                "CUDA and ROCm platform."
            )

    def forward_cuda(
        self,
        hidden_states: torch.Tensor,
        q_fp8: torch.Tensor,
        k: torch.Tensor,
        weights: torch.Tensor,
    ):
        return torch.ops.vllm.sparse_attn_indexer(
            hidden_states,
            self.k_cache.prefix,
            self.k_cache.kv_cache[0],
            q_fp8,
            k,
            weights,
            self.quant_block_size,
            self.scale_fmt,
            self.topk_tokens,
            self.head_dim,
            self.max_model_len,
            self.max_total_seq_len,
            self.topk_indices_buffer,
        )

    def forward_hip(
        self,
        hidden_states: torch.Tensor,
        q_fp8: torch.Tensor,
        k: torch.Tensor,
        weights: torch.Tensor,
    ):
        if rocm_aiter_ops.is_enabled():
            return torch.ops.vllm.rocm_aiter_sparse_attn_indexer(
                hidden_states,
                self.k_cache.prefix,
                self.k_cache.kv_cache[0],
                q_fp8,
                k,
                weights,
                self.quant_block_size,
                self.scale_fmt,
                self.topk_tokens,
                self.head_dim,
                self.max_model_len,
                self.max_total_seq_len,
                self.topk_indices_buffer,
            )
        else:
            raise RuntimeError(
                "Sparse attention indexer ROCm custom op requires ROCm "
                "Aiter ops to be enabled."
            )
