# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import TYPE_CHECKING, cast

import torch

from vllm.distributed.parallel_state import (
    get_dcp_group,
    get_pcp_group,
    get_tp_group,
)
from vllm.v1.attention.ops.common import cp_lse_ag_out_ar
from vllm.v1.attention.ops.dcp_alltoall import dcp_a2a_lse_reduce

if TYPE_CHECKING:
    from vllm.config import ParallelConfig


def get_dcp_tp_size(parallel_config: "ParallelConfig") -> int:
    return (
        parallel_config.decode_context_parallel_size
        // parallel_config.prefill_context_parallel_size
    )


def allgather_padded_token_tensors(
    tensors: tuple[torch.Tensor, ...],
) -> tuple[torch.Tensor, ...]:
    pcp_group = get_pcp_group()
    return tuple(
        pcp_group.all_gather(
            tensor if tensor.is_contiguous() else tensor.contiguous(),
            dim=0,
        )
        for tensor in tensors
    )


def maybe_gather_mla_latent_cache_inputs(
    attn_metadata: object | None,
    kv_c_normed: torch.Tensor,
    k_pe: torch.Tensor,
    slot_mapping: torch.Tensor | None,
    use_pcp: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    if not use_pcp or attn_metadata is None:
        return kv_c_normed, k_pe, slot_mapping
    assert slot_mapping is not None
    num_tokens = kv_c_normed.shape[0]
    k_pe_flat = k_pe.reshape(num_tokens, -1)
    gathered_kv_c, gathered_k_pe_flat = allgather_padded_token_tensors(
        (kv_c_normed, k_pe_flat)
    )
    gathered_k_pe = gathered_k_pe_flat.view(-1, *k_pe.shape[1:])
    return gathered_kv_c, gathered_k_pe, slot_mapping[: gathered_kv_c.shape[0]]


def maybe_gather_indexer_cache_inputs(
    k: torch.Tensor,
    slot_mapping: torch.Tensor,
    use_pcp: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not use_pcp:
        return k, slot_mapping
    (gathered_k,) = allgather_padded_token_tensors((k,))
    return gathered_k, slot_mapping[: gathered_k.shape[0]]


def pcp_dcp_a2a_lse_reduce(
    cp_attn_out: torch.Tensor,
    cp_attn_lse: torch.Tensor,
    return_lse: bool = False,
    is_lse_base_on_e: bool = True,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    combined = dcp_a2a_lse_reduce(
        cp_attn_out,
        cp_attn_lse,
        get_dcp_group(),
        return_lse=return_lse,
        is_lse_base_on_e=is_lse_base_on_e,
    )
    pcp_group = get_pcp_group()
    if return_lse:
        attn_out, attn_lse = cast(tuple[torch.Tensor, torch.Tensor], combined)
        return (
            pcp_group.all_gather(attn_out, dim=1),
            pcp_group.all_gather(attn_lse, dim=1),
        )
    return pcp_group.all_gather(cast(torch.Tensor, combined), dim=1)


def prepare_mla_pcp_decode_query(
    q: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
    dcp_tp_size: int,
    fp8_attention: bool,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    if dcp_tp_size <= 1:
        return q
    if fp8_attention:
        raise NotImplementedError("DCP does not support FP8 KV cache yet.")
    if isinstance(q, tuple):
        q = torch.cat(q, dim=-1)
    return get_tp_group().all_gather(q, dim=1)


def finalize_mla_pcp_decode(
    output: torch.Tensor,
    lse: torch.Tensor,
    dcp_tp_size: int,
    use_dcp_a2a: bool,
    num_heads: int,
    is_lse_base_on_e: bool,
) -> torch.Tensor:
    if use_dcp_a2a:
        return cast(
            torch.Tensor,
            pcp_dcp_a2a_lse_reduce(output, lse, is_lse_base_on_e=is_lse_base_on_e),
        )

    output = cp_lse_ag_out_ar(
        output,
        lse,
        get_dcp_group(),
        is_lse_base_on_e=is_lse_base_on_e,
    )
    if dcp_tp_size > 1:
        head_start = get_tp_group().rank_in_group * num_heads
        output = output[:, head_start : head_start + num_heads]
    return output
