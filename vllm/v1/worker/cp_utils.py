# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast

import torch

from vllm.config import VllmConfig, get_layers_from_vllm_config
from vllm.distributed import get_dcp_group, get_pcp_group
from vllm.logger import init_logger
from vllm.v1.attention.backend import CommonAttentionMetadata
from vllm.v1.attention.backends.utils import split_decodes_prefills_and_extends

if TYPE_CHECKING:
    from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
else:
    AttentionLayerBase = object

logger = init_logger(__name__)


def check_attention_cp_compatibility(vllm_config: VllmConfig) -> None:
    pcp_size = vllm_config.parallel_config.prefill_context_parallel_size
    dcp_size = vllm_config.parallel_config.decode_context_parallel_size
    interleave_size = vllm_config.parallel_config.cp_kv_cache_interleave_size
    if pcp_size * dcp_size > 1:
        layer_type = cast(type[Any], AttentionLayerBase)
        layers = get_layers_from_vllm_config(vllm_config, layer_type)
        for layer in layers.values():
            layer_impl = getattr(layer, "impl", None)
            if layer_impl is None:
                continue
            if vllm_config.speculative_config is not None and interleave_size > 1:
                assert layer_impl.supports_mtp_with_cp_non_trivial_interleave_size, (
                    "MTP with cp_kv_cache_interleave_size > 1 is not "
                    f"supported in {layer_impl.__class__.__name__}."
                )
            if dcp_size > 1:
                assert layer_impl.need_to_return_lse_for_decode, (
                    "Decode Context Parallelism (DCP) requires attention "
                    "implementations to return the softmax LSE during decode, "
                    f"but {layer_impl.__class__.__name__} does not. "
                    "Try a different backend by setting "
                    "--attention-backend or disable DCP."
                )

            if pcp_size > 1:
                assert layer_impl.supports_pcp, (
                    "PCP requires attention impls' support, "
                    f"but the impl {layer_impl.__class__.__name__} "
                    "does not support PCP."
                )


def get_total_cp_world_size():
    try:
        pcp_world_size = get_pcp_group().world_size
    except AssertionError:
        # PCP might not be initialized in testing
        pcp_world_size = 1
    try:
        dcp_world_size = get_dcp_group().world_size
    except AssertionError:
        # DCP might not be initialized in testing
        dcp_world_size = 1
    return dcp_world_size * pcp_world_size


def get_dcp_dummy_context_len(
    dcp_world_size: int,
    cp_kv_cache_interleave_size: int,
    has_kv_cache_config: bool,
    create_mixed_batch: bool,
    is_graph_capturing: bool,
    uniform_decode: bool,
) -> int:
    if (
        dcp_world_size <= 1
        or not has_kv_cache_config
        or not (create_mixed_batch or (is_graph_capturing and uniform_decode))
    ):
        return 0
    return dcp_world_size * cp_kv_cache_interleave_size


def prepare_dcp_dummy_context_metadata(
    *,
    input_batch: Any,
    kv_cache_config: Any,
    query_pos: Any,
    positions: torch.Tensor,
    query_start_loc: Any,
    num_reqs: int,
    num_tokens_unpadded: int,
    dcp_dummy_context_len: int,
) -> None:
    """Populate valid fake KV metadata for DCP CUDA graph warmup/capture."""
    if dcp_dummy_context_len == 0:
        return

    # DCP graph warmup may exercise context attention, so block-table entries
    # must point at allocated KV blocks.
    assert kv_cache_config is not None
    max_valid_block_id = kv_cache_config.num_blocks - 1
    assert max_valid_block_id > 0
    for blk_table in input_batch.block_table.block_tables:
        max_row_blocks = (
            blk_table.max_num_blocks_per_req // blk_table.blocks_per_kv_block
        )
        block_ids = [
            (block_idx % max_valid_block_id) + 1 for block_idx in range(max_row_blocks)
        ]
        for req_idx in range(num_reqs):
            blk_table.add_row(block_ids, req_idx)
        blk_table.commit_block_table(num_reqs)

    query_pos.copy_to_gpu(num_tokens_unpadded)
    positions[:num_tokens_unpadded] = (
        query_pos.gpu[:num_tokens_unpadded] + dcp_dummy_context_len
    )
    input_batch.block_table.compute_slot_mapping(
        num_reqs,
        query_start_loc.gpu[: num_reqs + 1],
        positions[:num_tokens_unpadded],
    )


def should_skip_dcp_context_attention(context_kv_lens_cpu: torch.Tensor) -> bool:
    """Whether DCP context attention can be skipped for this batch.

    Must be computed from rank-invariant inputs only (the global context
    lengths, NOT this rank's local share from get_dcp_local_seq_lens): the
    non-skip path in _forward_with_dcp issues DCP collectives (query
    all-gather + LSE combine), so every DCP rank must take the same branch.
    A rank can hold zero local context tokens while other ranks still hold
    context for the same batch.
    """
    return int(context_kv_lens_cpu.max().item()) == 0


def split_dcp_context_queries(
    query_start_loc: torch.Tensor,
    seq_lens_cpu_upper_bound: torch.Tensor | None,
    max_query_len: int,
    num_actual_tokens: int,
) -> tuple[int, int, int, int]:
    """Split reordered DCP context queries into decode and extend regions."""
    num_reqs = query_start_loc.shape[0] - 1
    if max_query_len <= 1:
        return num_reqs, 0, num_actual_tokens, 0
    if seq_lens_cpu_upper_bound is None:
        return 0, num_reqs, 0, num_actual_tokens

    common_attn_metadata = cast(
        CommonAttentionMetadata,
        SimpleNamespace(
            max_query_len=max_query_len,
            num_reqs=num_reqs,
            num_actual_tokens=num_actual_tokens,
            query_start_loc_cpu=query_start_loc,
            seq_lens_cpu_upper_bound=seq_lens_cpu_upper_bound,
            is_prefilling=None,
        ),
    )
    (
        num_decodes,
        num_extends,
        _num_prefills,
        num_decode_tokens,
        num_extend_tokens,
        _num_prefill_tokens,
    ) = split_decodes_prefills_and_extends(common_attn_metadata)
    return num_decodes, num_extends, num_decode_tokens, num_extend_tokens


def should_split_fa2_dcp_context_attention(
    fa_version: int | None,
    max_query_len: int,
    num_reqs: int,
    num_decode_reqs: int,
    num_context_prefill_reqs: int,
) -> bool:
    num_prefills = num_reqs - num_decode_reqs
    # TODO: Remove this FA2-only DCP compatibility path once FA4 supports
    # the Qwen3.5 head_size=256 shape on Blackwell and can be used here.
    # FA2 paged-varlen context attention can fail for DCP mixed batches when
    # decode rows, context-bearing extend rows, and zero-context pure prefill
    # rows are submitted together.
    return (
        fa_version == 2
        and max_query_len > 1
        and num_prefills > 0
        and (num_decode_reqs > 0 or num_context_prefill_reqs < num_prefills)
    )


def run_split_fa2_dcp_context_attention(
    flash_attn_varlen_func: Any,
    query_across_dcp: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    dcp_context_out: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    max_seqlen_q: int,
    dcp_context_kv_lens: torch.Tensor,
    max_dcp_context_kv_len: int,
    softmax_scale: float,
    alibi_slopes: torch.Tensor | None,
    sliding_window_size: list[int] | None,
    block_table: torch.Tensor,
    softcap: float,
    fa_version: int,
    q_descale: torch.Tensor | None,
    k_descale: torch.Tensor | None,
    v_descale: torch.Tensor | None,
    max_num_splits: int,
    num_heads: int,
    dcp_world_size: int,
    num_decode_reqs: int,
    num_context_prefill_reqs: int,
    num_decode_tokens: int,
    num_context_prefill_tokens: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    dcp_context_out.zero_()
    context_lse = torch.full(
        (num_heads * dcp_world_size, query_across_dcp.shape[0]),
        -torch.inf,
        dtype=torch.float32,
        device=query_across_dcp.device,
    )

    if num_decode_tokens > 0:
        _, decode_context_lse = flash_attn_varlen_func(
            q=query_across_dcp[:num_decode_tokens],
            k=key_cache,
            v=value_cache,
            out=dcp_context_out[:num_decode_tokens],
            cu_seqlens_q=cu_seqlens_q[: num_decode_reqs + 1],
            max_seqlen_q=1,
            seqused_k=dcp_context_kv_lens[:num_decode_reqs],
            max_seqlen_k=max_dcp_context_kv_len,
            softmax_scale=softmax_scale,
            causal=False,
            alibi_slopes=alibi_slopes,
            window_size=sliding_window_size,
            block_table=block_table[:num_decode_reqs],
            softcap=softcap,
            return_softmax_lse=True,
            scheduler_metadata=None,
            fa_version=fa_version,
            q_descale=q_descale[:num_decode_reqs] if q_descale is not None else None,
            k_descale=k_descale[:num_decode_reqs] if k_descale is not None else None,
            v_descale=v_descale[:num_decode_reqs] if v_descale is not None else None,
            num_splits=max_num_splits,
        )
        context_lse[:, :num_decode_tokens] = decode_context_lse

    if num_context_prefill_tokens > 0:
        prefill_start = num_decode_tokens
        prefill_end = prefill_start + num_context_prefill_tokens
        prefill_query_start_loc = (
            cu_seqlens_q[
                num_decode_reqs : num_decode_reqs + num_context_prefill_reqs + 1
            ]
            - num_decode_tokens
        )
        prefill_req_slice = slice(
            num_decode_reqs, num_decode_reqs + num_context_prefill_reqs
        )
        _, prefill_context_lse = flash_attn_varlen_func(
            q=query_across_dcp[prefill_start:prefill_end],
            k=key_cache,
            v=value_cache,
            out=dcp_context_out[prefill_start:prefill_end],
            cu_seqlens_q=prefill_query_start_loc,
            max_seqlen_q=max_seqlen_q,
            seqused_k=dcp_context_kv_lens[prefill_req_slice],
            max_seqlen_k=max_dcp_context_kv_len,
            softmax_scale=softmax_scale,
            causal=False,
            alibi_slopes=alibi_slopes,
            window_size=sliding_window_size,
            block_table=block_table[prefill_req_slice],
            softcap=softcap,
            return_softmax_lse=True,
            scheduler_metadata=None,
            fa_version=fa_version,
            q_descale=q_descale[prefill_req_slice] if q_descale is not None else None,
            k_descale=k_descale[prefill_req_slice] if k_descale is not None else None,
            v_descale=v_descale[prefill_req_slice] if v_descale is not None else None,
            num_splits=max_num_splits,
        )
        context_lse[:, prefill_start:prefill_end] = prefill_context_lse

    return dcp_context_out, context_lse
