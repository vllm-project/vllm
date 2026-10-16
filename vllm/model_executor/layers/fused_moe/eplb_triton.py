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
):
    """Map logical expert ids to physical replicas and record expert load.

    ``token_idx`` broadcasts against ``logical_ids``; masked-off or
    out-of-range ids map to -1.
    """
    valid = mask & (logical_ids >= 0) & (logical_ids < num_logical)
    safe_ids = tl.where(valid, logical_ids, 0)
    replicas = tl.load(replica_count + safe_ids, mask=valid, other=1)
    # Avoid invalid modulo/div by forcing at least 1.
    replicas = tl.maximum(replicas, 1)
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
