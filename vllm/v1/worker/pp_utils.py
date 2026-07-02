# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pipeline-parallel (PP) cross-rank coordination helpers.

When pipeline parallelism is enabled, every PP rank independently runs the
cudagraph dispatcher (``CudagraphDispatcher.dispatch``) for the same logical
step. The dispatch decision is a pure function of the per-step batch shape, but
several of its inputs are *per-rank local* state (e.g. ``calculate_kv_scales``,
cascade-attention prefix detection from a rank's local block table, LoRA
bookkeeping, or a partially-failed graph capture). If those differ, one PP rank
can resolve a step to ``PIECEWISE`` (replay a captured graph whose baked-in
inter-stage P2P send/recv shapes are fixed) while another resolves the same step
to ``NONE`` (eager). The two stages then disagree on the cross-stage P2P
schedule, the send/recv never rendezvous, and the engine wedges forever
(vllm-project/vllm#45094: ``shm_broadcast`` "No available shared memory
broadcast block found in 60 seconds" -> RPC timeout -> EngineCore crash).

This mirrors the existing data-parallel consensus in ``dp_utils.py``
(``_post_process_cudagraph_mode`` takes the ``min`` across DP ranks so that if
any rank picks ``NONE`` they all do). DP already pays this cost; PP did not have
an equivalent, so a pure-PP (DP=1) deployment had no guard. This module adds the
PP equivalent: a cheap ``min`` all-reduce over the PP *CPU* process group
(gloo), which keeps the consensus off the NCCL/GPU P2P channel that is itself
the thing prone to wedging.
"""

import torch

from vllm.distributed.parallel_state import get_pp_group
from vllm.logger import init_logger

logger = init_logger(__name__)


def coordinate_cudagraph_mode_across_pp(local_cudagraph_mode: int) -> int:
    """Reach consensus on the cudagraph runtime mode across all PP ranks.

    Takes the minimum of the per-rank cudagraph mode over the pipeline-parallel
    group, where ``NONE=0 < PIECEWISE=1 < FULL=2``. If *any* PP rank decided to
    run a step eager (``NONE``), every rank runs it eager, guaranteeing all
    stages agree on the inter-stage P2P send/recv schedule for the step.

    The reduction is performed on the PP CPU (gloo) process group so it does not
    contend with — and cannot itself be wedged by — the GPU P2P channel.

    Args:
        local_cudagraph_mode: This rank's intended runtime mode as an int
            (``CUDAGraphMode`` ``.value``: 0=NONE, 1=PIECEWISE, 2=FULL).

    Returns:
        The agreed-upon runtime mode int (min across all PP ranks). On a
        single-stage pipeline (``world_size == 1``) the input is returned
        unchanged with no communication.
    """
    pp_group = get_pp_group()
    if pp_group.world_size == 1:
        # Early exit: no peers, nothing to coordinate.
        return local_cudagraph_mode

    # CPU all-reduce keeps this off the GPU P2P channel (matches the
    # disable_nccl_for_dp_synchronization path in dp_utils).
    tensor = torch.tensor([local_cudagraph_mode], dtype=torch.int32, device="cpu")
    torch.distributed.all_reduce(
        tensor,
        op=torch.distributed.ReduceOp.MIN,
        group=pp_group.cpu_group,
    )
    synced = int(tensor.item())
    if synced != local_cudagraph_mode:
        logger.debug(
            "PP cudagraph-mode consensus dropped this rank from %d to %d "
            "(a peer PP stage chose a lower mode); running the step at the "
            "agreed mode to keep inter-stage P2P consistent.",
            local_cudagraph_mode,
            synced,
        )
    return synced
