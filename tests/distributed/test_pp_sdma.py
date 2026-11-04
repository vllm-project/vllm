# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""PP stage hand-off on SDMA engines: byte-exact over more steps than slots, with
the receiver reading each slot late. The metadata-send reaping tests run on CPU
(gloo).

Run `pytest tests/distributed/test_pp_sdma.py`.
"""

import multiprocessing as mp
import os

import pytest
import torch

from vllm.distributed import get_pp_group
from vllm.platforms import current_platform
from vllm.utils.network_utils import get_open_port

from ..utils import init_test_distributed_environment, multi_gpu_test

MAX_TOKENS = 4096
HIDDEN = 512
STEPS = [4096, 1, 1000, 4096, 17, 4096, 2048, 3000]


def _step_tensors(step: int, n: int, device: torch.device) -> dict:
    g = torch.Generator().manual_seed(step)
    return {
        "hidden_states": torch.randn(n, HIDDEN, generator=g).to(device, torch.bfloat16),
        "residual": torch.randn(n, HIDDEN, generator=g).to(device, torch.bfloat16),
        "aux": torch.randn(n, 3 * HIDDEN, generator=g).to(device),
    }


def _on_gfx950() -> bool:
    if not current_platform.is_rocm():
        return False
    from vllm.platforms.rocm import on_gfx950

    return on_gfx950()


def pp_sdma_worker(rank: int, port: str):
    from vllm.distributed.pp_sdma import PPSdmaReceiver, PPSdmaSender, _Counters

    device = torch.device(f"cuda:{rank}")
    torch.accelerator.set_device_index(device)
    init_test_distributed_environment(1, 2, rank, port)
    pp = get_pp_group()

    if pp.is_first_rank:
        sender = PPSdmaSender(1, MAX_TOKENS)
        for step, n in enumerate(STEPS):
            sender.send(_step_tensors(step, n, device), pp)
        _Counters.wait(sender.counters.ready, sender.seq)
    else:
        receiver = PPSdmaReceiver(0)
        for step, n in enumerate(STEPS):
            receiver.release_previous()
            got = receiver.recv(pp)
            want = _step_tensors(step, n, device)
            assert got.keys() == want.keys()
            # Read the slot late: the sender must not overwrite it meanwhile.
            torch.cuda._sleep(10_000_000)
            for k in want:
                assert torch.equal(got[k], want[k]), (step, k)
    torch.distributed.barrier(group=pp.cpu_group)


@multi_gpu_test(num_gpus=2)
@pytest.mark.skipif(not _on_gfx950(), reason="SDMA engines 2 and 4: MI355X")
def test_pp_sdma_handoff():
    ctx = mp.get_context("spawn")
    port = str(get_open_port())
    procs = [ctx.Process(target=pp_sdma_worker, args=(r, port)) for r in range(2)]
    for p in procs:
        p.start()
    for p in procs:
        p.join(timeout=300)
        if p.is_alive():
            p.kill()
    assert [p.exitcode for p in procs] == [0, 0]


# The metadata isends: released once the receiver's ``freed`` covers their step,
# never by polling ``is_completed()``, which a gloo isend reports False until
# ``wait()`` is called. Polling it made ``_pending`` grow by one handle per step
# for the life of the worker and the per-step poll cost grow with uptime. CPU only.


class _GlooLikeHandle:
    """An isend handle with gloo's semantics: complete only once waited."""

    def __init__(self) -> None:
        self.waited = 0

    def is_completed(self) -> bool:
        return False

    def wait(self) -> None:
        self.waited += 1


class _FakeGroup:
    def __init__(self) -> None:
        self.handles: list[_GlooLikeHandle] = []

    def isend_object(self, obj, dst: int) -> _GlooLikeHandle:
        h = _GlooLikeHandle()
        self.handles.append(h)
        return h


def _cpu_sender():
    import ctypes
    from types import SimpleNamespace

    from vllm.distributed.pp_sdma import PPSdmaSender

    sender = PPSdmaSender(1, MAX_TOKENS)
    sender.counters = SimpleNamespace(freed=ctypes.c_uint64(0))
    return sender


def test_pp_sdma_meta_reap_gated_on_freed():
    sender, group = _cpu_sender(), _FakeGroup()
    for seq in range(1, 11):
        sender._post_meta({"seq": seq}, group)
    # Nothing confirmed by the receiver: all kept, none waited.
    assert [s for s, _ in sender._pending] == list(range(1, 11))
    assert not any(h.waited for h in group.handles)

    sender.counters.freed.value = 7
    sender._post_meta({"seq": 11}, group)
    assert [s for s, _ in sender._pending] == [8, 9, 10, 11]
    assert [h.waited for h in group.handles] == [1] * 7 + [0] * 4

    # Steady state, the receiver NUM_SLOTS steps behind: bounded, each handle
    # waited exactly once.
    from vllm.distributed.pp_sdma import NUM_SLOTS

    for seq in range(12, 5000):
        sender.counters.freed.value = seq - NUM_SLOTS
        sender._post_meta({"seq": seq}, group)
        assert len(sender._pending) <= NUM_SLOTS
    assert all(h.waited == 1 for h in group.handles[: 5000 - 1 - NUM_SLOTS])


def test_pp_sdma_meta_reap_caps_unconfirmed():
    from vllm.distributed.pp_sdma import _MAX_PENDING_META

    sender, group = _cpu_sender(), _FakeGroup()
    for seq in range(1, 3 * _MAX_PENDING_META):
        sender._post_meta({"seq": seq}, group)
        assert len(sender._pending) <= _MAX_PENDING_META
    # Past the cap the oldest are waited (blocking) and dropped, in order.
    n_waited = sum(h.waited for h in group.handles)
    assert n_waited == len(group.handles) - _MAX_PENDING_META
    assert all(h.waited == 1 for h in group.handles[:n_waited])


LAG = 3


def _meta_reap_gloo_worker(rank: int, port: str, steps: int) -> None:
    """Real gloo isends through ``GroupCoordinator.isend_object``: rank 0 posts
    one metadata send per step at most LAG steps ahead of rank 1, which
    receives them in order and bumps ``freed`` like the receiver's free loop."""
    import functools
    import os
    from types import SimpleNamespace

    import torch.distributed as dist

    from vllm.distributed.parallel_state import GroupCoordinator
    from vllm.distributed.pp_sdma import PPSdmaSender, _Counters

    dist.init_process_group(
        "gloo", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=2
    )
    group = SimpleNamespace(
        world_size=2, rank_in_group=rank, ranks=[0, 1], cpu_group=dist.group.WORLD
    )
    group.isend_object = functools.partial(GroupCoordinator.isend_object, group)
    group.recv_object = functools.partial(GroupCoordinator.recv_object, group)
    name = f"vllm_pp_sdma_meta_test_{port}"
    if rank == 0:
        counters = _Counters(name, create=True)
        dist.barrier()
        sender = PPSdmaSender(1, MAX_TOKENS)
        sender.counters = counters
        peak = 0
        for seq in range(1, steps + 1):
            # The pipeline keeps a stage at most a few steps ahead.
            _Counters.wait(counters.freed, seq - LAG)
            sender._post_meta({"seq": seq, "pad": bytes(512)}, group)
            peak = max(peak, len(sender._pending))
        _Counters.wait(counters.freed, steps)
        sender._reap_meta_sends()
        assert not sender._pending
        assert peak <= LAG + 1, peak
        dist.barrier()
        os.unlink(counters.path)
    else:
        dist.barrier()
        counters = _Counters(name, create=False)
        for seq in range(1, steps + 1):
            meta = group.recv_object(src=0)
            assert meta["seq"] == seq
            # The step that read slot ``seq`` ran: free it.
            counters.freed.value = seq
        dist.barrier()
    dist.destroy_process_group()


@pytest.mark.skipif(not os.path.isdir("/dev/shm"), reason="needs /dev/shm")
def test_pp_sdma_meta_reap_gloo():
    ctx = mp.get_context("spawn")
    port = str(get_open_port())
    procs = [
        ctx.Process(target=_meta_reap_gloo_worker, args=(r, port, 2000))
        for r in range(2)
    ]
    for p in procs:
        p.start()
    for p in procs:
        p.join(timeout=120)
        if p.is_alive():
            p.kill()
    assert [p.exitcode for p in procs] == [0, 0]
