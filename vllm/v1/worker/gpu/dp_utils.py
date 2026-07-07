# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from __future__ import annotations

import torch
import torch.distributed as dist

from vllm.config.compilation import CUDAGraphMode
from vllm.distributed.parallel_state import get_dp_group, get_pp_group
from vllm.v1.worker.gpu.cudagraph_utils import (
    BatchExecutionDescriptor,
    CudaGraphManager,
)


def sync_cudagraph_and_dp_padding(
    cudagraph_manager: CudaGraphManager | None,
    desired_batch_desc: BatchExecutionDescriptor,
    num_tokens: int,
    num_reqs: int,
    uniform_token_count: int | None,
    dp_size: int,
    dp_rank: int,
    num_active_loras: int = 0,
    pp_synced_cudagraph_mode: int | None = None,
) -> tuple[BatchExecutionDescriptor, torch.Tensor | None]:
    """
    Coordinates the batch descriptor and DP padding across all ranks.

    Returns (synced_batch_desc, num_tokens_across_dp).

    ``pp_synced_cudagraph_mode`` (when not ``None``) is the PP-group MIN cudagraph
    mode for *this* DP replica, already agreed by the worker's pre-recv
    ``coordinate_cudagraph_mode_across_pp`` all-reduce. It is FOLDED INTO the value
    this rank contributes to the DP cg_mode all-reduce so the DP MIN subsumes the
    PP MIN of every replica. Without that fold the DP and PP MINs are taken over
    *orthogonal* slices of the (DP replica x PP stage) grid, and a graph-vs-eager
    split present in one replica's PP group but not another's makes the two
    replicas of the SAME PP stage end on different modes *after* the DP collective
    — a DP lockstep violation that re-wedges the pipeline (#45094 MED-2). Folding
    here makes the contributed value ``min(local, pp_synced)``, so the reduced
    ``synced_cg_mode`` is the global MIN: identical across every replica and
    stage, and the post-reduce PP reconciliation in ``dispatch_cg_and_sync_dp``
    becomes a provable no-op.
    """
    assert dp_size > 1, "DP size must be greater than 1"
    group = get_dp_group().cpu_group
    local_cg_mode = desired_batch_desc.cg_mode.value
    if pp_synced_cudagraph_mode is not None:
        # Contribute the PP-agreed value so the DP MIN is the global (DP x PP)
        # MIN. Keeps every DP replica of a PP stage on one mode post-reduce.
        local_cg_mode = min(local_cg_mode, pp_synced_cudagraph_mode)
    tensor = torch.zeros(3, dp_size, dtype=torch.int32, device="cpu")
    tensor[0][dp_rank] = num_tokens
    tensor[1][dp_rank] = local_cg_mode
    tensor[2][dp_rank] = uniform_token_count or 0  # (0 means None)
    dist.all_reduce(tensor, group=group)

    num_tokens_across_dp = tensor[0]
    cg_mode_across_dp = tensor[1]
    uniform_token_counts_across_dp = tensor[2]

    if torch.all(num_tokens_across_dp == 0).item():
        synced_desc = BatchExecutionDescriptor(
            cg_mode=CUDAGraphMode.NONE, num_tokens=0, num_reqs=0
        )
        return synced_desc, None

    synced_cg_mode = CUDAGraphMode(int(cg_mode_across_dp.min().item()))

    # If any rank wants to run eager, all ranks run eager
    if synced_cg_mode == CUDAGraphMode.NONE:
        return BatchExecutionDescriptor(
            cg_mode=CUDAGraphMode.NONE,
            num_tokens=num_tokens,
            num_reqs=num_reqs,
            num_active_loras=desired_batch_desc.num_active_loras,
        ), num_tokens_across_dp

    assert cudagraph_manager is not None, (
        "cudagraph_manager should only be None during profile run, "
        "where synced_cg_mode must be NONE across all DP ranks"
    )
    synced_num_tokens = int(num_tokens_across_dp.max().item())
    synced_uniform_token_count = uniform_token_counts_across_dp[0]
    # If ranks disagree on the uniform token count, or its 0 (means None) set to None
    if synced_uniform_token_count == 0 or not torch.all(
        uniform_token_counts_across_dp == synced_uniform_token_count
    ):
        synced_uniform_token_count = None

    # Dispatch for the final synced values, use num_reqs instead of synced_num_reqs
    # so we don't perform request padding for PIECEWISE graphs.
    # num_active_loras is per-rank and doesn't need cross-rank agreement.
    synced_desc = cudagraph_manager.dispatch(
        num_reqs,
        synced_num_tokens,
        synced_uniform_token_count,
        num_active_loras=num_active_loras,
    )

    # Update num_tokens_across_dp to reflect padded size.
    num_tokens_across_dp[:] = synced_desc.num_tokens

    return synced_desc, num_tokens_across_dp


def _reconcile_cudagraph_mode_across_pp(
    batch_desc: BatchExecutionDescriptor,
    num_reqs: int,
    num_tokens: int,
    pp_synced_mode_val: int,
) -> BatchExecutionDescriptor:
    """Reconcile this rank's V2 batch descriptor to the PP-agreed cudagraph mode.

    Mirror of the V1 fix (vllm-project/vllm#45094). Each PP rank runs
    ``CudaGraphManager.dispatch`` independently, and per-rank-local state
    (calculate_kv_scales / cascade-attention detection / LoRA bookkeeping / a
    partially-failed capture / encoder-decoder skip_compiled) can make one stage
    pick PIECEWISE/FULL (replay a graph with baked-in inter-stage P2P send/recv
    shapes) while another picks NONE (eager). The stages then disagree on the
    cross-stage P2P schedule, the send/recv never rendezvous, and the engine
    wedges (``shm_broadcast`` 60s timeout -> RPC timeout -> EngineCore crash).

    ``pp_synced_mode_val`` is the MIN of ``cg_mode`` across the PP group
    (NONE=0 < PIECEWISE=1 < FULL=2), already agreed via the CPU (gloo) all-reduce
    in ``coordinate_cudagraph_mode_across_pp``. Crucially that all-reduce is
    issued by ``Worker.execute_model`` *before* the inter-stage recv — NOT here
    — because the collective and the point-to-point recv form a deadlock cycle
    otherwise (#45094/#45610). This function only consumes the agreed value.

    Reconciliation rule: if the synced MIN differs from this rank's local mode,
    drop this rank to NONE (eager). Unlike the V1 dispatcher, the V2
    ``CudaGraphManager.dispatch`` is *shape-deterministic* — it returns the one
    best-matching captured descriptor for ``(num_reqs, num_tokens,
    uniform_token_count)`` and takes no ``valid_modes`` allow-list — so it cannot
    be coaxed to an arbitrary intermediate mode. NONE is the only target every
    rank can always honor. This is provably consistent: whenever any rank goes
    eager the MIN is NONE, so every graph rank sees ``synced(NONE) !=
    local(graph)`` and also drops to NONE; and because cudagraph capture sizes
    are config-global and identical across PP stages, a same-shape
    FULL-vs-PIECEWISE split between stages cannot arise (the realistic PP
    divergence is strictly graph-vs-NONE).
    """
    if get_pp_group().world_size == 1:
        # No peers: nothing to coordinate.
        return batch_desc

    if pp_synced_mode_val == batch_desc.cg_mode.value:
        return batch_desc

    # A peer stage chose a lower mode and this rank cannot be re-dispatched to an
    # arbitrary intermediate mode (V2 dispatch is shape-deterministic). Drop to
    # eager — the universally-honorable mode — to keep the inter-stage P2P
    # schedule consistent across all stages.
    return BatchExecutionDescriptor(
        cg_mode=CUDAGraphMode.NONE,
        num_tokens=num_tokens,
        num_reqs=num_reqs,
    )


def dispatch_cg_and_sync_dp(
    cudagraph_manager: CudaGraphManager | None,
    num_reqs: int,
    num_tokens: int,
    uniform_token_count: int | None,
    dp_size: int,
    dp_rank: int,
    need_eager: bool = False,
    num_active_loras: int = 0,
    # Pre-agreed pipeline-parallel cudagraph mode (a CUDAGraphMode ``.value``
    # int), or ``None`` to skip PP reconciliation. The PP consensus MIN
    # all-reduce is NO LONGER issued here: it must run in ``Worker.execute_model``
    # *before* the inter-stage ``irecv_tensor_dict`` (see the V2
    # ``predispatch_cudagraph_mode`` + the worker call site), or the collective
    # and the recv deadlock (#45094/#45610). The dummy/profile/capture path
    # leaves this ``None`` so no reconciliation happens there.
    pp_synced_cudagraph_mode: int | None = None,
) -> tuple[BatchExecutionDescriptor, torch.Tensor | None]:
    if need_eager:
        batch_desc = BatchExecutionDescriptor(
            cg_mode=CUDAGraphMode.NONE,
            num_tokens=num_tokens,
            num_reqs=num_reqs,
            num_active_loras=num_active_loras,
        )
    else:
        assert cudagraph_manager is not None, (
            "cudagraph_manager should only be None during profile run, "
            "where need_eager must be True"
        )
        batch_desc = cudagraph_manager.dispatch(
            num_reqs,
            num_tokens,
            uniform_token_count,
            num_active_loras=num_active_loras,
        )

    if dp_size == 1:
        # No DP consensus, but PP ranks within this replica may still have
        # diverged (the #45094 split-brain). Reconcile to the PP-agreed mode
        # (the all-reduce ran earlier, in Worker.execute_model, before the recv).
        if pp_synced_cudagraph_mode is not None:
            batch_desc = _reconcile_cudagraph_mode_across_pp(
                batch_desc,
                num_reqs,
                num_tokens,
                pp_synced_cudagraph_mode,
            )
        return batch_desc, None

    synced_desc, num_tokens_across_dp = sync_cudagraph_and_dp_padding(
        cudagraph_manager,
        batch_desc,
        num_tokens,
        num_reqs,
        uniform_token_count,
        dp_size,
        dp_rank,
        num_active_loras=num_active_loras,
        pp_synced_cudagraph_mode=pp_synced_cudagraph_mode,
    )

    # The PP-agreed mode was already FOLDED INTO the DP cg_mode all-reduce above,
    # so ``synced_desc.cg_mode`` is the global (DP x PP) MIN and is identical on
    # every DP replica of this PP stage *provided all replicas of the stage
    # contributed the same pp_synced value* (homogeneous PP consensus across
    # replicas). Under that precondition the reconcile below is a no-op for DP>1
    # (synced == pp_synced whenever pp_synced was the limiting mode). It is NOT a
    # guaranteed no-op under heterogeneous replicas (e.g. per-replica LoRA /
    # partial-capture divergence yielding different pp_synced per replica), so it
    # is kept (not skipped) as a real guard, not merely a defensive one.
    # Crucially we must NOT skip the fold and rely on this reconcile alone: that
    # was the #45094 MED-2 bug — the DP and PP MINs are taken over orthogonal
    # grid slices, so a post-reduce per-replica drop to NONE diverges the DP
    # ranks of a stage after the collective.
    if pp_synced_cudagraph_mode is not None:
        # F2 (known limitation, pre-existing, correctness-safe): when one PP
        # stage is forced eager by cascade-attention detection (cascade-attn +
        # PP>=2) while peers hold a PIECEWISE graph, the MIN drives ALL stages to
        # NONE for that step. That is a throughput cost (the peers run eager when
        # they could have replayed PIECEWISE) but is correct — every stage agrees
        # on the inter-stage P2P schedule. We do NOT try to keep the peers on
        # PIECEWISE here: that is exactly the split the consensus exists to
        # prevent. No logic change.
        synced_desc = _reconcile_cudagraph_mode_across_pp(
            synced_desc,
            num_reqs,
            num_tokens,
            pp_synced_cudagraph_mode,
        )
        # F4 (known limitation, defensive): this clobber of the DP padding tensor
        # to the (possibly lowered) mode's num_tokens is only reachable if the
        # reconcile above actually lowered the mode — i.e. only under the
        # heterogeneous-replica DP>1 case described in the comment above the
        # reconcile (homogeneous replicas make the reconcile a no-op, leaving
        # num_tokens unchanged). It is kept so the padding stays consistent with
        # the lowered mode if that case fires. No logic change.
        # Keep the DP padding tensor consistent with the (possibly lowered) mode.
        if num_tokens_across_dp is not None:
            num_tokens_across_dp[:] = synced_desc.num_tokens

    return synced_desc, num_tokens_across_dp
