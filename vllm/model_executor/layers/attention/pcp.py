# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import TYPE_CHECKING, NamedTuple, TypeVar

import torch

from vllm.distributed.parallel_state import (
    GroupCoordinator,
    get_dcp_group,
    get_pcp_group,
    get_tp_group,
)
from vllm.v1.attention.ops.common import cp_lse_ag_out_ar, cp_lse_ag_out_rs
from vllm.v1.attention.ops.dcp_alltoall import dcp_a2a_lse_reduce

if TYPE_CHECKING:
    from vllm.config import VllmConfig

# Query as handed to the DCP context attention: a tensor, or MLA's split
# (nope, pe) pair which is only concatenated if a gather happens.
_QueryT = TypeVar("_QueryT", torch.Tensor, tuple[torch.Tensor, ...])


def _gather_prefill_cache_inputs(
    tensors: tuple[torch.Tensor, ...],
    slot_mapping: torch.Tensor,
    num_decode_tokens: int,
) -> tuple[tuple[torch.Tensor, ...], torch.Tensor]:
    """Keep replicated decode writes local and gather partitioned prefills."""
    local_num_tokens = tensors[0].shape[0]
    assert all(tensor.shape[0] == local_num_tokens for tensor in tensors)
    assert 0 <= num_decode_tokens <= local_num_tokens

    if num_decode_tokens == local_num_tokens:
        return tensors, slot_mapping[:num_decode_tokens]

    pcp_group = get_pcp_group()
    gathered_prefills = tuple(
        pcp_group.all_gather(tensor[num_decode_tokens:].contiguous(), dim=0)
        for tensor in tensors
    )
    pcp_size = pcp_group.world_size
    gathered_slot_mapping = slot_mapping[: pcp_size * local_num_tokens]
    if num_decode_tokens == 0:
        return gathered_prefills, gathered_slot_mapping

    cache_inputs = tuple(
        torch.cat((tensor[:num_decode_tokens], gathered_prefill), dim=0)
        for tensor, gathered_prefill in zip(tensors, gathered_prefills)
    )
    rank_slot_mappings = gathered_slot_mapping.view(pcp_size, local_num_tokens)
    cache_slot_mapping = torch.cat(
        (
            rank_slot_mappings[0, :num_decode_tokens],
            rank_slot_mappings[:, num_decode_tokens:].flatten(),
        )
    )
    return cache_inputs, cache_slot_mapping


def maybe_gather_mla_latent_cache_inputs(
    kv_c_normed: torch.Tensor,
    k_pe: torch.Tensor,
    slot_mapping: torch.Tensor | None,
    num_decode_tokens: int | None,
    use_pcp: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    if not use_pcp or num_decode_tokens is None:
        return kv_c_normed, k_pe, slot_mapping
    assert slot_mapping is not None
    num_tokens = kv_c_normed.shape[0]
    k_pe_flat = k_pe.reshape(num_tokens, -1)
    (cache_kv_c, cache_k_pe_flat), cache_slot_mapping = _gather_prefill_cache_inputs(
        (kv_c_normed, k_pe_flat),
        slot_mapping,
        num_decode_tokens,
    )
    cache_k_pe = cache_k_pe_flat.view(-1, *k_pe.shape[1:])
    return cache_kv_c, cache_k_pe, cache_slot_mapping


def maybe_gather_kv_cache_inputs(
    key: torch.Tensor,
    value: torch.Tensor,
    slot_mapping: torch.Tensor | None,
    num_decode_tokens: int | None,
    use_pcp: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """GQA/MHA PCP KV-cache gather.

    All-gather the prefill portion of K/V across PCP ranks so every rank can
    write the full prefill KV to its (replicated, ``dcp=1``) cache, while
    keeping decode writes local. Returns contiguous K, V and the gathered
    cache slot mapping ready for ``reshape_and_cache_flash``. No-op when PCP
    is off.

    Gathering K and V separately (rather than ``cat``-then-``split``) keeps
    each output contiguous, so the cache kernel's ``head stride == head_size``
    assumption holds.
    """
    if not use_pcp or num_decode_tokens is None:
        return key, value, slot_mapping
    assert slot_mapping is not None
    (cache_key, cache_value), cache_slot_mapping = _gather_prefill_cache_inputs(
        (key, value),
        slot_mapping,
        num_decode_tokens,
    )
    return cache_key, cache_value, cache_slot_mapping


def maybe_gather_indexer_k(
    k: torch.Tensor,
    slot_mapping: torch.Tensor,
    num_decode_tokens: int,
    use_pcp: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not use_pcp:
        return k, slot_mapping
    (cache_k,), cache_slot_mapping = _gather_prefill_cache_inputs(
        (k,), slot_mapping, num_decode_tokens
    )
    return cache_k, cache_slot_mapping


def gather_prefill_qkv_global(
    pcp_prefill_gather: tuple[torch.Tensor, torch.Tensor, torch.Tensor, int],
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """All-gather this rank's DualChunkSwap chunk Q/K/V across PCP and reorder
    to global position order (real tokens only).

    ``pcp_prefill_gather`` is the (restore_idx, gather_idx, global_cu_seqlens,
    padded_n) tuple from ``PCPManager.prefill_gather_indices()``.
    """
    restore_idx, _, _, padded_n = pcp_prefill_gather
    pcp_group = get_pcp_group()

    def _pad(t: torch.Tensor) -> torch.Tensor:
        if t.shape[0] < padded_n:
            pad = t.new_zeros((padded_n - t.shape[0],) + tuple(t.shape[1:]))
            return torch.cat([t, pad], dim=0)
        return t

    q_g = pcp_group.all_gather(_pad(query), dim=0)[restore_idx]
    k_g = pcp_group.all_gather(_pad(key), dim=0)[restore_idx]
    v_g = pcp_group.all_gather(_pad(value), dim=0)[restore_idx]
    return q_g, k_g, v_g


def slice_prefill_output_local(
    pcp_prefill_gather: tuple[torch.Tensor, torch.Tensor, torch.Tensor, int],
    out_g: torch.Tensor,
    num_actual: int,
) -> torch.Tensor:
    """Slice a global-order output tensor back to this PCP rank's local chunk."""
    _, gather_idx, _, padded_n = pcp_prefill_gather
    pcp_rank = get_pcp_group().rank_in_group
    out_gathered = out_g[gather_idx]
    local = out_gathered[pcp_rank * padded_n : (pcp_rank + 1) * padded_n]
    return local[:num_actual]


class DecodeSubset(NamedTuple):
    """Decode-token subset of a mixed prefill+decode batch."""

    token_mask: torch.Tensor
    q: torch.Tensor
    k: torch.Tensor
    v: torch.Tensor
    cu_seqlens: torch.Tensor
    max_seqlen: int
    ctx_kv_lens: torch.Tensor
    block_table: torch.Tensor


def build_mixed_decode_subset(
    is_prefilling: torch.Tensor | None,
    query_start_loc: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    dcp_context_kv_lens: torch.Tensor,
    block_table: torch.Tensor,
) -> DecodeSubset | None:
    """Extract the decode-token subset of a mixed prefill+decode batch.

    DualChunkSwap gives ranks with no prefill chunks a 0-token dummy decode
    segment mirroring global req 0; such segments are excluded (counting them
    would desync the DCP LSE-combine). Returns None for a pure-prefill batch.
    """
    if is_prefilling is None:
        return None
    is_pre = is_prefilling.to(device=query_start_loc.device)
    seg_lens = query_start_loc[1:] - query_start_loc[:-1]
    decode_seg = (~is_pre.bool()) & (seg_lens > 0)
    num_decode = int(decode_seg.sum().item())
    if num_decode == 0:
        return None
    token_mask = torch.repeat_interleave(decode_seg, seg_lens)[: query.shape[0]]
    dec_lens = seg_lens[decode_seg]
    dec_cu = torch.zeros(
        num_decode + 1,
        dtype=query_start_loc.dtype,
        device=query_start_loc.device,
    )
    torch.cumsum(dec_lens, dim=0, out=dec_cu[1:])
    return DecodeSubset(
        token_mask=token_mask,
        q=query[token_mask].contiguous(),
        k=key[token_mask].contiguous(),
        v=value[token_mask].contiguous(),
        cu_seqlens=dec_cu,
        max_seqlen=int(dec_lens.max().item()),
        ctx_kv_lens=dcp_context_kv_lens[decode_seg],
        block_table=block_table[decode_seg],
    )


def _dcp_q_gather_group(
    dcp_world_size: int,
    pcp_world_size: int,
) -> GroupCoordinator | None:
    """Group holding the query-head shards this rank must gather, or None.

    DCP groups span the PCP axis before TP (see ``init_model_parallel_groups``),
    so how far the group reaches decides which ranks hold distinct query heads:

    - ``pcp == 1``: the DCP group is a TP subgroup that splits Q heads, so
      gather over the DCP group.
    - ``dcp > pcp``: the group spans TP x PCP. Q is head-sharded across TP and
      replicated across PCP, so gather over the TP group -- gathering over the
      DCP group here would duplicate the PCP-replicated heads.
    - otherwise: the DCP group is the PCP group and already holds every head.
    """
    if pcp_world_size == 1:
        return get_dcp_group() if dcp_world_size > 1 else None
    if dcp_world_size > pcp_world_size:
        return get_tp_group()
    return None


def dcp_q_gather_size(dcp_world_size: int, pcp_world_size: int) -> int:
    """Head-shard count ``maybe_all_gather_q_for_dcp`` will produce.

    For sizing work that has to be planned before the gather happens, such as
    FlashAttention's AOT scheduler metadata. 1 means no gather.
    """
    group = _dcp_q_gather_group(dcp_world_size, pcp_world_size)
    return 1 if group is None else group.world_size


def maybe_all_gather_q_for_dcp(
    q: _QueryT,
    dcp_world_size: int,
    pcp_world_size: int,
    already_replicated: bool = False,
) -> _QueryT | torch.Tensor:
    """All-gather query heads for attention against the DCP-sharded cache.

    Returns ``q`` untouched when this rank already holds every head the kernel
    needs; ``already_replicated`` lets a caller declare that up front. A split
    ``q`` -- MLA's ``(nope, pe)`` pair -- is concatenated only when a gather
    actually happens, so callers that can consume the split form keep it.

    Pairs with ``resolve_dcp_combine_fn``: changing one without the other breaks
    the head layout the combine expects.
    """
    if already_replicated:
        return q
    group = _dcp_q_gather_group(dcp_world_size, pcp_world_size)
    if group is None:
        return q
    gathered = torch.cat(q, dim=-1) if isinstance(q, tuple) else q
    return group.all_gather(gathered, dim=1)


def resolve_dcp_combine_fn(vllm_config: "VllmConfig | None"):
    """LSE-combine to fold per-rank partial attentions over the DCP group.

    Under PCP the partials cover the full head set on every rank, so they
    all-reduce and each rank takes its own heads back out with
    ``cp_reconcile_heads``. Without PCP the DCP group is a TP subgroup, so the
    reduce-scatter lands each rank's heads directly.

    Pairs with ``maybe_all_gather_q_for_dcp``. ``vllm_config`` may be None when
    there is no config in scope (unit tests), which implies no CP.
    """
    if vllm_config is None:
        return cp_lse_ag_out_rs
    parallel_config = vllm_config.parallel_config
    if (
        parallel_config.decode_context_parallel_size > 1
        and parallel_config.dcp_comm_backend == "a2a"
    ):
        return dcp_a2a_lse_reduce
    if parallel_config.prefill_context_parallel_size > 1:
        return cp_lse_ag_out_ar
    return cp_lse_ag_out_rs


def cp_reconcile_heads(
    output: torch.Tensor,
    num_heads: int,
) -> torch.Tensor:
    if output.shape[1] < num_heads:
        output = get_pcp_group().all_gather(output, dim=1)
    elif output.shape[1] > num_heads:
        head_start = get_tp_group().rank_in_group * num_heads
        output = output[:, head_start : head_start + num_heads]
    return output
