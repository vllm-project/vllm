# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""One-round DCP decode top-k merge for the DSA indexer.

The kernels live in `kernels.py` (imports CuTe DSL); this module only holds
the configuration helpers so model construction stays import-light.

Packed candidate rows (the "B'" layout), int32 `[T, N + N/32]` per rank,
N a multiple of 32, local index `l` = the token's position in this rank's
shard (global id `l * W + rank`, interleave 1), `l < 2**17`:

- word `j < N`: `(ord16(s) << 16) | (0xFFFF - (l & 0xFFFF))` as uint32,
  where `s` is the bf16 score bits and `ord16(s) = s | 0x8000` for
  `s >= 0` (sign bit clear), `~s & 0xFFFF` otherwise (order-preserving:
  a larger score gives a larger `ord16`). `0` marks an empty slot; slots
  may be in any order and empty ones anywhere.
- word `N + j // 32`, bit `j % 32`: bit 16 of `l` for candidate `j`.
"""

from typing import Any

from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.utils.import_utils import has_cutedsl

logger = init_logger(__name__)

# The packed candidate word keeps local[15:0]; a bit plane carries local[16].
LOCAL_INDEX_BITS = 17
# The saturation word holds a 16-bit rank mask.
MAX_WORLD_SIZE = 16
_WARP = 32


def dcp_merge_local_topk(topk: int, world_size: int) -> int:
    """Per-rank candidate count N: 1.5x the fair share `topk / world_size`,
    warp-aligned. 0 when it would not be smaller than `topk`."""
    fair_share = -(-topk // world_size)
    local_topk = -(-(3 * fair_share) // (2 * _WARP)) * _WARP
    return local_topk if local_topk < topk else 0


def packed_row_words(local_topk: int) -> int:
    """int32 words per packed row: N candidate words + N/32 bit-plane words."""
    return local_topk + local_topk // _WARP


def dcp_topk_merge_local_topk(vllm_config: Any, topk: int) -> int:
    """N for the one-round DCP decode top-k merge, or 0 when it is unavailable
    (no DCP, more than `MAX_WORLD_SIZE` DCP ranks, PCP, a top-k backend
    other than auto / deep_select, not SM100, no CuTe DSL / DeepSelect, or
    local token indices that do not fit `LOCAL_INDEX_BITS`)."""
    parallel_config = vllm_config.parallel_config
    world_size = parallel_config.decode_context_parallel_size
    max_local_len = -(-vllm_config.model_config.max_model_len // world_size)
    if (
        world_size <= 1
        or world_size > MAX_WORLD_SIZE
        or parallel_config.prefill_context_parallel_size > 1
        or vllm_config.kernel_config.sparse_indexer_topk_backend
        not in ("auto", "deep_select")
        or not current_platform.is_cuda()
        or not current_platform.is_device_capability_family(100)
        or not has_cutedsl()
        or max_local_len > 1 << LOCAL_INDEX_BITS
    ):
        return 0
    try:
        import vllm._deepselect_C  # noqa: F401
    except ImportError:
        return 0
    return dcp_merge_local_topk(topk, world_size)


def rescue_slots(topk: int, local_topk: int) -> int:
    """Upper bound on the saturated ranks of one row in the exact merge.

    A saturated rank sent N words, all strictly above the K-th selected key,
    so they are among the K - 1 keys above it: at most `(K - 1) // N`."""
    return (topk - 1) // local_topk


# (topk, local_topk) -> rows of the rescue workspace built by
# init_dcp_topk_merge; a decode step with more rows takes the generic merge.
_MERGE_MAX_ROWS: dict[tuple[int, int], int] = {}
# topk -> init_dcp_topk_merge's result: the layers of a model share one
# workspace, and the collectives run once (in the same order on every rank).
_MERGE_INIT: dict[int, int] = {}


def dcp_topk_merge_max_rows(topk: int, local_topk: int) -> int:
    """Rows one merge call may hold (0 when the merge was not initialized)."""
    return _MERGE_MAX_ROWS.get((topk, local_topk), 0)


def dcp_topk_merge_fits(num_rows: int, topk: int, local_topk: int) -> bool:
    """Whether a decode step of `num_rows` rows can take the one-round merge
    (identical on every DCP rank): it was initialized and its rescue
    workspace holds the rows. Otherwise the generic merge runs."""
    return local_topk > 0 and num_rows <= dcp_topk_merge_max_rows(topk, local_topk)


def init_dcp_topk_merge(vllm_config: Any, topk: int) -> int:
    """Set up the exact one-round DCP decode top-k merge at model construction
    and return its N, or 0 to keep the generic (fp32 all-gather) merge.

    Collective: every DCP rank must call it, in the same order, outside CUDA graph
    capture (only the first call per `topk` runs collectives). The merge is exact only
    with its rescue workspace (symmetric memory over the DCP group, sized by the decode
    rows of a step); without one the merge is disabled here (fail closed) rather than
    run lossy. The ranks agree on that before allocating; a failure of the (collective)
    allocation itself is fatal. Also registers the merge kernels for JIT warmup.
    """
    if topk in _MERGE_INIT:
        return _MERGE_INIT[topk]
    local_topk = dcp_topk_merge_local_topk(vllm_config, topk)
    if local_topk:
        local_topk = _init_rescue(vllm_config, topk, local_topk)
    _MERGE_INIT[topk] = local_topk
    return local_topk


def _init_rescue(vllm_config: Any, topk: int, local_topk: int) -> int:
    parallel_config = vllm_config.parallel_config
    if parallel_config.use_ubatching:
        # The rescue workspace holds one ubatch.
        logger.info_once(
            "DSA DCP top-k merge: generic merge (ubatching is not supported)."
        )
        return 0
    import torch

    from vllm.distributed import get_dcp_group
    from vllm.v1.attention.ops.cp_common import direct_cp_enabled
    from vllm.v1.attention.ops.dcp import get_dcp_workspace_max_num_tokens

    group = get_dcp_group()
    max_rows = get_dcp_workspace_max_num_tokens(vllm_config)
    # All ranks must agree before the collective allocation, or one would
    # wait on the others (in the allocation, or later on rescue flags).
    agree = torch.tensor(
        [int(direct_cp_enabled(group, torch.int32, None))], dtype=torch.int32
    )
    torch.distributed.all_reduce(
        agree, op=torch.distributed.ReduceOp.MIN, group=group.cpu_group
    )
    if not agree.item():
        logger.warning_once(
            "DSA DCP top-k merge: no symmetric memory over the DCP group; "
            "using the generic merge."
        )
        return 0
    from vllm.models.kimi_k3.nvidia.ops.dcp_topk_merge.kernels import (
        init_dcp_topk_rescue_workspace,
        register_dcp_topk_merge_warmup,
    )

    init_dcp_topk_rescue_workspace(max_rows, topk, local_topk, group)
    _MERGE_MAX_ROWS[(topk, local_topk)] = max_rows
    if vllm_config.kernel_config.enable_jit_warmup:
        register_dcp_topk_merge_warmup(
            topk, local_topk, parallel_config.decode_context_parallel_size
        )
    logger.info_once(
        "DSA DCP top-k merge: exact one-round merge of each rank's local "
        "top-%d (rescue workspace for %d rows).",
        local_topk,
        max_rows,
    )
    return local_topk
