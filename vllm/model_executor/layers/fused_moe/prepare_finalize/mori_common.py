# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Helpers shared by the MORI EP prepare/finalize implementations (no mori import)."""

from vllm.distributed import get_dp_group
from vllm.forward_context import get_forward_context, is_forward_context_available
from vllm.logger import init_logger

logger = init_logger(__name__)

# Receive-row bounds are rounded up to this many rows (as SGLang's MORI path does).
MORI_RECV_ROW_ALIGN = 32


def mori_recv_rows_bound(
    num_local_tokens: int, capacity: int, num_dispatchers: int
) -> int:
    """Host-side upper bound on the rows this rank can receive this step.

    MORI receives each source token at most once per rank, so the receive
    count is at most the tokens all dispatchers send: the sum of the DP
    batches, times the dispatchers per DP rank (each TP rank may send the
    same batch). MORI returns capacity-sized buffers (world_size *
    max_num_tokens rows); experts that do not read only the received rows
    (e.g. the Triton experts' activation and top-k sum) otherwise process the
    whole capacity every layer. The bound comes from DP metadata on the host,
    so it needs no device sync and is fixed per captured CUDA graph size. It is
    rounded up to a multiple of MORI_RECV_ROW_ALIGN to limit distinct shapes.
    """
    dp_meta = (
        get_forward_context().dp_metadata if is_forward_context_available() else None
    )
    if dp_meta is None:
        return capacity
    tokens = dp_meta.num_tokens_across_dp_cpu
    n_dp = tokens.numel()
    dp_rank = get_dp_group().rank_in_group
    if dp_rank >= n_dp or num_local_tokens > int(tokens[dp_rank]):
        # Metadata that does not cover this rank's own batch is stale; a
        # bound derived from it could cut off received rows.
        logger.debug_once(
            "MORI: DP token counts do not cover this rank's batch; "
            "using the full receive buffer."
        )
        return capacity
    if n_dp > 0 and num_dispatchers % n_dp == 0:
        bound = num_dispatchers // n_dp * int(tokens.sum())
    else:
        bound = num_dispatchers * int(tokens.max())
    bound = max(bound, 1)
    bound = -(-bound // MORI_RECV_ROW_ALIGN) * MORI_RECV_ROW_ALIGN
    return min(capacity, bound)
