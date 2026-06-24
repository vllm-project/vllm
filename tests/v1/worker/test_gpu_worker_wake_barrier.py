# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Regression tests for the cross-rank wake synchronization in Worker.wake_up.

Context (issue #45519): with ``--sleep-mode-backend=cumem_tag`` on a TP>1/PP>1
deployment, ``wake_up`` is dispatched to each worker independently. Each worker
re-maps its own VMM-backed regions at its own pace. The very next decode step
issues a cross-rank ``torch.distributed.broadcast`` on ``pp.device_group``
(``_pp_receive_prev_sampled_token_ids_to_input_batch``). If a fast rank reaches
that collective before a slow rank has finished re-mapping the regions backing
the broadcast buffers, the collective issues against memory the peer MMU still
treats as invalid -> ``cudaErrorIllegalAddress`` and the NCCL comm is corrupted
permanently (engine deadlocks, /health lies 200).

The fix gates ``Worker.wake_up`` on a cross-rank wake-success handshake over the
CPU (gloo) group *after* the local allocator wake and *before* returning to the
caller, so no rank can reach a device-group collective until every rank has
finished its local wake. Crucially the handshake is an all-reduce of a per-rank
success flag (``ReduceOp.MIN``), NOT a bare barrier: if one rank's local wake
raises, a bare barrier would strand its peers forever (re-introducing the very
full-fleet hang we are fixing). The all-reduce instead lets every rank learn
that a peer failed and raise *symmetrically* -- loud, no hang, no rank silently
proceeding into a device-group collective against a peer whose wake never
completed. The gloo group is used deliberately so the synchronization itself
never touches the not-yet-resynced NCCL device_group.

These tests are GPU-free: they drive ``Worker.wake_up`` against mocked allocator
and distributed groups, and assert the handshake is (a) performed on multi-rank
cumem configs, (b) ordered after the allocator wake and before return,
(c) skipped on single-rank configs, (d) routed through the gloo cpu_group only,
and -- the adversarial-finding regression -- (e) does NOT hang and DOES raise
symmetrically when one simulated rank's local wake fails.
"""

from types import SimpleNamespace
from unittest import mock

import pytest
import torch

from vllm.v1.worker.gpu_worker import Worker


def _make_worker(*, enable_cumem: bool = True) -> Worker:
    """Build a Worker shell sufficient to exercise wake_up without a GPU.

    We bypass __init__ entirely and set only the attributes wake_up touches.
    """
    worker = Worker.__new__(Worker)
    worker._sleep_saved_buffers = {}
    worker.model_runner = mock.MagicMock()
    worker.vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(enable_cumem_allocator=enable_cumem)
    )
    return worker


def _patch_groups(
    calls: list[str],
    *,
    tp_world_size: int,
    pp_world_size: int,
    allocator_raises: bool = False,
    # Simulated *peer* outcome folded into the all-reduce result. The real
    # all-reduce uses ReduceOp.MIN over int32 flags, so the post-reduce value is
    # 0 if THIS rank failed OR any peer failed, else 1. `peer_ok=False` models
    # "some other rank failed even though this rank's local wake succeeded".
    peer_ok: bool = True,
):
    """Patch the distributed group getters, allocator, and the gloo all-reduce
    used by wake_up, recording the order of the load-bearing events into
    ``calls``.

    The all-reduce is patched to emulate the real ReduceOp.MIN cross-rank
    reduction *in process*: the post-reduce flag becomes
    ``min(local_flag, 1 if peer_ok else 0)``. This is what lets a GPU-free test
    exercise the symmetric-failure semantics deterministically.
    """
    tp = mock.MagicMock()
    tp.world_size = tp_world_size
    pp = mock.MagicMock()
    pp.world_size = pp_world_size

    world = mock.MagicMock()
    # `barrier` must NOT be used by the fix anymore; record it if it ever is so
    # a regression to a bare barrier is caught.
    world.barrier.side_effect = lambda: calls.append("barrier")

    allocator = mock.MagicMock()

    def _wake(tags=None):
        calls.append("alloc_wake")
        if allocator_raises:
            raise RuntimeError("simulated local cumem wake failure on this rank")

    allocator.wake_up.side_effect = _wake

    platform = mock.MagicMock()
    platform.is_cuda_alike.return_value = True

    def _all_reduce(tensor, op=None, group=None):
        # Emulate ReduceOp.MIN across this rank + a simulated peer.
        calls.append("all_reduce")
        assert op == torch.distributed.ReduceOp.MIN, (
            "wake handshake must use ReduceOp.MIN so any single-rank failure "
            "(flag=0) forces the global result to 0"
        )
        # The fix must route the handshake through the gloo cpu_group, never the
        # NCCL device_group.
        assert group is world.cpu_group, (
            "wake handshake must run on get_world_group().cpu_group (gloo)"
        )
        peer_flag = 1 if peer_ok else 0
        tensor[0] = min(int(tensor[0].item()), peer_flag)
        return tensor

    patcher = mock.patch.multiple(
        "vllm.v1.worker.gpu_worker",
        get_tp_group=mock.MagicMock(return_value=tp),
        get_pp_group=mock.MagicMock(return_value=pp),
        get_world_group=mock.MagicMock(return_value=world),
        get_mem_allocator_instance=mock.MagicMock(return_value=allocator),
        current_platform=platform,
    )
    dist_patcher = mock.patch(
        "vllm.v1.worker.gpu_worker.torch.distributed.all_reduce",
        side_effect=_all_reduce,
    )
    return patcher, dist_patcher, world, allocator


@pytest.mark.parametrize(
    "tp,pp",
    [(2, 2), (2, 1), (1, 2)],
)
def test_wake_up_handshakes_on_multi_rank_cumem(tp, pp):
    """On any multi-rank (TP>1 or PP>1) cumem config the wake handshake fires."""
    worker = _make_worker()
    calls: list[str] = []
    patcher, dist_patcher, world, allocator = _patch_groups(
        calls, tp_world_size=tp, pp_world_size=pp
    )
    with patcher, dist_patcher:
        worker.wake_up()

    # Exactly one cross-rank all-reduce handshake; no bare barrier.
    assert calls.count("all_reduce") == 1
    assert "barrier" not in calls
    # The post_kv_cache_wake_up hook should still run (tags=None path).
    worker.model_runner.post_kv_cache_wake_up.assert_called_once()


@pytest.mark.parametrize(
    "tp,pp",
    [(2, 2), (2, 1), (1, 2)],
)
def test_wake_up_handshake_ordered_after_alloc_before_return(tp, pp):
    """The handshake must run AFTER the local allocator wake and BEFORE the
    method returns control to the caller (which then issues the PP broadcast).

    This is the load-bearing ordering: a handshake that ran before the local
    re-map, or one skipped entirely, would leave the #45519 race open.
    """
    worker = _make_worker()
    calls: list[str] = []
    patcher, dist_patcher, world, allocator = _patch_groups(
        calls, tp_world_size=tp, pp_world_size=pp
    )
    with patcher, dist_patcher:
        worker.wake_up()

    assert calls == ["alloc_wake", "all_reduce"], (
        f"expected local wake then cross-rank handshake, got {calls}"
    )


def test_wake_up_no_handshake_on_single_rank():
    """TP=1 PP=1 is unaffected by the PP-broadcast race; no handshake overhead."""
    worker = _make_worker()
    calls: list[str] = []
    patcher, dist_patcher, world, allocator = _patch_groups(
        calls, tp_world_size=1, pp_world_size=1
    )
    with patcher, dist_patcher:
        worker.wake_up()

    assert "all_reduce" not in calls
    assert "barrier" not in calls
    assert calls == ["alloc_wake"]


def test_wake_up_no_handshake_when_cumem_disabled():
    """Without the cumem allocator there is no VMM remap to race against."""
    worker = _make_worker(enable_cumem=False)
    calls: list[str] = []
    patcher, dist_patcher, world, allocator = _patch_groups(
        calls, tp_world_size=2, pp_world_size=2
    )
    with patcher, dist_patcher:
        worker.wake_up()

    assert "all_reduce" not in calls
    assert "barrier" not in calls
    assert calls == ["alloc_wake"]


def test_wake_up_handshake_routes_through_cpu_group_not_device_group():
    """The wake handshake MUST go through the world GroupCoordinator's
    CPU-group (gloo), NOT a device_group (NCCL) collective.

    Synchronizing the ranks via the not-yet-resynced NCCL ``device_group``
    would itself issue against the corrupt communicator. We assert ``wake_up``
    drives synchronization *only* via a ``torch.distributed.all_reduce`` on
    ``get_world_group().cpu_group`` and never reaches for any group's
    ``device_group`` or a GroupCoordinator device collective (``.broadcast``,
    ``.all_reduce``, etc., which dispatch to the NCCL device communicator).
    """
    worker = _make_worker()

    forbidden_hits: list[str] = []
    FORBIDDEN_METHODS = (
        "broadcast",
        "all_reduce",
        "all_gather",
        "send",
        "recv",
        "barrier",
    )

    def _make_group(name: str, world_size: int):
        grp = mock.MagicMock()
        grp.world_size = world_size
        for meth in FORBIDDEN_METHODS:
            getattr(grp, meth).side_effect = (
                lambda *a, n=name, m=meth, **k: forbidden_hits.append(f"{n}.{m}()")
            )
        return grp

    tp = _make_group("tp", 2)
    pp = _make_group("pp", 2)
    world = _make_group("world", 4)

    allocator = mock.MagicMock()

    cpu_group_used: list[object] = []

    def _all_reduce(tensor, op=None, group=None):
        cpu_group_used.append(group)
        tensor[0] = min(int(tensor[0].item()), 1)
        return tensor

    with mock.patch.multiple(
        "vllm.v1.worker.gpu_worker",
        get_tp_group=mock.MagicMock(return_value=tp),
        get_pp_group=mock.MagicMock(return_value=pp),
        get_world_group=mock.MagicMock(return_value=world),
        get_mem_allocator_instance=mock.MagicMock(return_value=allocator),
        current_platform=mock.MagicMock(
            is_cuda_alike=mock.MagicMock(return_value=True)
        ),
    ), mock.patch(
        "vllm.v1.worker.gpu_worker.torch.distributed.all_reduce",
        side_effect=_all_reduce,
    ):
        worker.wake_up()

    # The raw gloo all-reduce ran exactly once, on the world group's cpu_group.
    assert cpu_group_used == [world.cpu_group]
    # ...and NO GroupCoordinator device-group collective (incl. NCCL barrier)
    # was invoked on any group.
    assert forbidden_hits == [], (
        "wake handshake must synchronize via a raw torch.distributed.all_reduce "
        "on get_world_group().cpu_group (gloo) only; these GroupCoordinator "
        f"device-group collectives were invoked: {forbidden_hits}"
    )


# ---------------------------------------------------------------------------
# Adversarial-finding regression: a single-rank local wake failure must NOT
# hang peers, and must surface symmetrically (every rank raises).
# ---------------------------------------------------------------------------


def test_wake_up_failed_local_wake_still_participates_in_handshake():
    """If THIS rank's local ``allocator.wake_up`` raises, the rank must STILL
    reach the cross-rank handshake (so peers are never stranded waiting) and
    then raise loudly.

    PRE-FIX BEHAVIOR (bare barrier *after* the wake, no exception handling):
    the exception escaped ``allocator.wake_up`` before ``barrier()`` was
    reached, so the failing rank never entered the collective -> every surviving
    rank would block forever on the gloo barrier. That is the full-fleet hang
    this test guards against. Emulated here as: 'all_reduce' would be ABSENT
    from ``calls`` and the surviving peers (in a real cluster) deadlock.

    POST-FIX: the local wake is wrapped in try/except so the all-reduce
    ('all_reduce' in ``calls``) runs even on local failure; the rank then raises
    a unified abort (chaining the original cause).
    """
    worker = _make_worker()
    calls: list[str] = []
    patcher, dist_patcher, world, allocator = _patch_groups(
        calls, tp_world_size=2, pp_world_size=2, allocator_raises=True
    )
    with patcher, dist_patcher:
        with pytest.raises(
            RuntimeError, match="wake_up failed on at least one rank"
        ) as excinfo:
            worker.wake_up()

    # The original local failure is chained as the cause (debuggability) on the
    # rank that actually failed.
    assert isinstance(excinfo.value.__cause__, RuntimeError)
    assert "simulated local cumem wake failure" in str(excinfo.value.__cause__)

    # The crux: even though the local wake raised, the rank still entered the
    # cross-rank handshake. Absence here == peers hang in a real cluster.
    assert calls == ["alloc_wake", "all_reduce"], (
        "a rank whose local wake failed MUST still participate in the handshake "
        f"so peers do not hang; got {calls}"
    )
    # It must NOT have proceeded to restore buffers / post-kv-cache hooks with
    # corrupt state.
    worker.model_runner.post_kv_cache_wake_up.assert_not_called()


def test_wake_up_raises_symmetrically_when_a_peer_failed():
    """If a PEER rank failed its local wake (but THIS rank's local wake
    succeeded), this rank must learn of the failure via the all-reduce and
    raise too -- it must NOT silently proceed into a device-group collective
    against the peer whose wake never completed.

    PRE-FIX BEHAVIOR (bare barrier): a barrier conveys no success/failure
    payload, so a healthy rank sails past it and issues the PP broadcast
    against the failed peer -> CUDA_ERROR_ILLEGAL_ADDRESS / NCCL wedge (the
    #45519 wedge class, just relocated). Asymmetric: this rank proceeds, peer
    is broken.

    POST-FIX: the ReduceOp.MIN all-reduce yields 0 when any peer failed, so
    this rank raises symmetrically.
    """
    worker = _make_worker()
    calls: list[str] = []
    # This rank's local wake SUCCEEDS (allocator_raises=False) but a peer failed
    # (peer_ok=False).
    patcher, dist_patcher, world, allocator = _patch_groups(
        calls,
        tp_world_size=2,
        pp_world_size=2,
        allocator_raises=False,
        peer_ok=False,
    )
    with patcher, dist_patcher:
        with pytest.raises(RuntimeError, match="wake_up failed on at least one rank"):
            worker.wake_up()

    # This rank's local wake succeeded, the handshake ran, and it still raised
    # (because a peer's flag was 0). No silent proceed.
    assert calls == ["alloc_wake", "all_reduce"]
    worker.model_runner.post_kv_cache_wake_up.assert_not_called()


def test_wake_up_all_ranks_ok_proceeds_normally():
    """Happy path: this rank and all peers succeeded -> all-reduce yields 1,
    wake_up returns normally and runs the post-kv-cache hook."""
    worker = _make_worker()
    calls: list[str] = []
    patcher, dist_patcher, world, allocator = _patch_groups(
        calls,
        tp_world_size=2,
        pp_world_size=2,
        allocator_raises=False,
        peer_ok=True,
    )
    with patcher, dist_patcher:
        worker.wake_up()  # must not raise

    assert calls == ["alloc_wake", "all_reduce"]
    worker.model_runner.post_kv_cache_wake_up.assert_called_once()
