# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""EPLB replica mapping and expert-load recording shared by Triton kernels."""

from vllm.triton_utils import tl, triton


@triton.jit
def eplb_map_and_record(
    logical_ids,
    token_idx,
    mask,
    logical_to_physical,
    replica_count,
    num_logical,
    map_stride,
    expert_load,
    num_physical,
    record_enabled,
    num_unpadded_tokens,
    replica_routing_table=None,
    ROUTING_TABLE_ROWS: tl.constexpr = 0,
):
    """Map logical expert ids to physical replicas and record expert load.

    `token_idx` broadcasts against `logical_ids`; masked-off or
    out-of-range ids map to -1. An optional contiguous int32 table of shape
    `(ROUTING_TABLE_ROWS, num_logical)` selects physical replicas periodically.
    Callers own table construction and refresh after placement changes.
    """
    valid = mask & (logical_ids >= 0) & (logical_ids < num_logical)
    safe_ids = tl.where(valid, logical_ids, 0)
    if replica_routing_table is not None:
        physical_ids = tl.load(
            replica_routing_table
            + (token_idx % ROUTING_TABLE_ROWS) * num_logical
            + safe_ids,
            mask=valid,
            other=-1,
        )
    else:
        replicas = tl.load(replica_count + safe_ids, mask=valid, other=1)
        # floor(2^32 / phi), classic Knuth multiplicative hash multiplier.
        KNUTH_MULTIPLIER = 2654435769
        hashed = (token_idx.to(tl.int64) * KNUTH_MULTIPLIER) & 0xFFFFFFFF
        physical_ids = tl.load(
            logical_to_physical + safe_ids * map_stride + hashed % replicas,
            mask=valid,
            other=-1,
        )

    record = (
        (tl.load(record_enabled) != 0)
        & (physical_ids >= 0)
        & (physical_ids < num_physical)
    )
    if num_unpadded_tokens is not None:
        record &= token_idx < tl.load(num_unpadded_tokens)
    tl.atomic_add(expert_load + tl.where(record, physical_ids, 0), 1, mask=record)
    return physical_ids


@triton.jit
def _eplb_map_and_record_i32_kernel(
    topk_ids_ptr,
    logical_replica_count_ptr,
    logical_to_physical_ptr,
    out_ids_ptr,
    out_ptr,
    record_enabled_ptr,
    num_unpadded_tokens_ptr,
    num_logical_experts,
    map_slots,
    out_size,
    numel,
    num_active_experts,
    BLOCK_SIZE: tl.constexpr,
    replica_routing_table=None,
    ROUTING_TABLE_ROWS: tl.constexpr = 0,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < numel

    expert_id = tl.load(topk_ids_ptr + offs, mask=mask, other=0).to(tl.int64)

    physical_id = eplb_map_and_record(
        expert_id,
        offs // num_active_experts,
        mask,
        logical_to_physical_ptr,
        logical_replica_count_ptr,
        num_logical_experts,
        map_slots,
        out_ptr,
        out_size,
        record_enabled_ptr,
        num_unpadded_tokens_ptr,
        replica_routing_table,
        ROUTING_TABLE_ROWS,
    )
    tl.store(out_ids_ptr + offs, physical_id, mask=mask)
