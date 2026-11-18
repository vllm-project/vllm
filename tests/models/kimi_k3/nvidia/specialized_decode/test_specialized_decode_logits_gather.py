# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kimi K3 specialized decode: the target logits gather on a Lamport mailbox
(`runtime/logits_gather.py`).

- Policy (no GPU): the HEAD_GATHER rows per (TP, DCP) pair, and
  `install_logits_gather` leaves the LogitsProcessor untouched (and runs no
  collective) wherever the table, the head or the processor does not apply, or
  where a HeadGather is installed already.
- On real ranks (2 GPUs by default; `KIMI_K3_LOGITS_GATHER_WORLD=4` / 8 runs more):
  `HeadGather` against vLLM's `tensor_model_parallel_all_gather` byte for byte
  (the sentinel pair (+0, -0) arrives as (+0, +0), (-0, +0) and (-0, -0) arrive
  unchanged) for every served row count, random bit patterns, both parities and
  back-to-back calls with one rank late, unserved row counts (NCCL) interleaved; a
  call under graph capture takes the stock gather; one rank whose gather kernel
  does not build sends every rank to the stock gather; and a real
  `LogitsProcessor` returns the same logits, bit for bit, before and after
  `install_logits_gather`.
"""

import os
from types import SimpleNamespace

import pytest
import torch
import torch.multiprocessing as mp

from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.common.policy import (
    batch_table,
)
from vllm.models.kimi_k3.nvidia.specialized_decode.runtime import logits_gather
from vllm.platforms import current_platform
from vllm.utils.network_utils import get_open_port

WORLD = int(os.environ.get("KIMI_K3_LOGITS_GATHER_WORLD", "2"))
VOCAB = 163840  # the K3 / K3.1 padded vocabulary
SENT = -(1 << 31)  # int32 word of the BF16 pair (+0, -0): the mailbox sentinel

PAIRS = [
    (1, 1),
    (2, 1),
    (4, 1),
    (4, 4),
    (8, 1),
    (8, 8),
    (16, 1),
    (16, 2),
    (16, 4),
    (16, 8),
    (16, 16),
    (32, 32),
]


@pytest.mark.parametrize("tp,dcp", PAIRS)
def test_gather_rows_per_tp_dcp(tp, dcp):
    table = batch_table(logits_gather.HEAD_GATHER.get((tp, dcp), ""))
    rows = sorted(r for r, on in table.items() if on)
    assert rows == (list(range(1, 9)) if (tp, dcp) == (16, 16) else [])


def _model(tp, shard_rows=10240, dtype=torch.bfloat16, head_tp=None, **processor):
    lp = SimpleNamespace(logits_as_input=False, head_dtype=None, use_all_gather=True)
    lp.__dict__.update(processor)
    stock = lp._gather_logits = object()
    weight = torch.empty(shard_rows, 8, dtype=dtype)
    head = SimpleNamespace(weight=weight, tp_size=tp if head_tp is None else head_tp)
    return SimpleNamespace(logits_processor=lp, lm_head=head), stock


@pytest.mark.parametrize(
    "tp,dcp,kwargs",
    [
        (8, 1, {}),
        (8, 8, {}),
        (16, 1, {}),
        (16, 8, {}),
        (16, 16, {"logits_as_input": True}),
        (16, 16, {"head_dtype": torch.float32}),
        (16, 16, {"use_all_gather": False}),
        (16, 16, {"dtype": torch.float16}),
        (16, 16, {"head_tp": 8}),
        (16, 16, {"shard_rows": 10244}),  # not a whole number of 16 B fragments
    ],
)
def test_install_keeps_stock_gather(tp, dcp, kwargs):
    model, stock = _model(tp, **kwargs)
    # group None: any collective here would raise.
    rows = logits_gather.install_logits_gather(
        model, None, tp, 0, torch.device("cpu"), dcp
    )
    assert rows == ()
    assert model.logits_processor._gather_logits is stock
    assert not hasattr(model.logits_processor, "head_gather")


def test_install_keeps_an_installed_gather():
    """A second install (`bind_model_state` again) keeps the first HeadGather: no
    second mailbox, no wrapper around the wrapper, no collective."""
    model, stock = _model(16)
    lp = model.logits_processor
    lp.head_gather = SimpleNamespace(rows=frozenset(range(1, 9)))
    # group None: any collective here would raise.
    rows = logits_gather.install_logits_gather(
        model, None, 16, 0, torch.device("cpu"), 16
    )
    assert rows == tuple(range(1, 9))
    assert lp._gather_logits is stock


# ---------------------------------------------------------------------------------
# On real ranks.
# ---------------------------------------------------------------------------------


def _has_multicast(device) -> bool:
    from torch._C._autograd import DeviceType
    from torch._C._distributed_c10d import _SymmetricMemory

    return bool(_SymmetricMemory.has_multicast_support(DeviceType.CUDA, device.index))


def _stock(shard):
    from vllm.distributed import tensor_model_parallel_all_gather

    return tensor_model_parallel_all_gather(shard, -1)


def _bits(rows, seed):
    """[rows, VOCAB] BF16 of random bit patterns (the same on every rank), with
    sentinel pairs, (-0, +0) and (-0, -0) words sprinkled in."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    words = torch.randint(-(1 << 31), (1 << 31) - 1, (rows, VOCAB // 2), generator=g)
    words = words.to(torch.int32)
    for value in (SENT, 0x00008000, SENT | 0x8000):
        words[torch.rand(words.shape, generator=g) < 0.01] = value
    return words.view(torch.bfloat16)


def _cleared(full):
    """`full` as the mailbox delivers it: sentinel words (+0, -0) -> (+0, +0)."""
    words = full.view(torch.int32).clone()
    words[words == SENT] = 0
    return words.view(torch.bfloat16)


def _same_bytes(a, b):
    a, b = a.contiguous(), b.contiguous()
    return torch.equal(a.view(torch.int16), b.view(torch.int16))


def _check_head_gather(gather, rank, world, device):
    shard = VOCAB // world
    sequence = (1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 8, 7, 5, 1, 16, 8, 8, 4, 3, 8)
    sequence += (8, 5) * 8  # back to back on both parities
    fulls = [_bits(rows, seed) for seed, rows in enumerate(sequence)]
    shards = [
        f.view(f.shape[0], world, shard)[:, rank].contiguous().to(device) for f in fulls
    ]
    torch.accelerator.synchronize(device)
    torch.distributed.barrier()
    parity = gather.parity
    outs: list[torch.Tensor] = []
    for i, mine in enumerate(shards):
        if rank == i % world:
            torch.cuda._sleep(200_000)  # this rank pushes late
        outs.append(gather(mine, _stock))
    torch.accelerator.synchronize(device)
    served = sum(rows <= 8 for rows in sequence)
    assert gather.parity == (parity + served) % 2
    for rows, full, mine, out in zip(sequence, fulls, shards, outs):
        assert _same_bytes(_stock(mine).cpu(), full), rows  # NCCL: a byte copy
        expect = _cleared(full) if rows <= 8 else full
        assert _same_bytes(out.cpu(), expect), rows


def _check_capture(gather, world, device):
    """Under a graph capture the gather takes `stock` (the parity is untouched)."""
    mine = torch.zeros(4, gather.shard, dtype=torch.bfloat16, device=device)
    seen: list[int] = []

    def stock(t):
        seen.append(t.shape[0])
        return t.repeat(1, world)

    parity = gather.parity
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = gather(mine, stock)
    assert seen == [4] and gather.parity == parity
    assert out.shape == (4, gather.shard * world)


def _check_one_rank_fails(group, rank, world, device):
    """The last rank's HeadGather fails after the mailbox rendezvous (its gather
    kernel does not build): every rank keeps the stock gather and none waits."""
    from vllm.models.kimi_k3.nvidia.specialized_decode.drafter import comm

    def fail(*args, **kwargs):
        raise RuntimeError("no gather kernel on this rank (test)")

    model, stock = _model(world, shard_rows=VOCAB // world)
    compile_gather = comm.compile_gather
    saved = dict(logits_gather.HEAD_GATHER)
    logits_gather.HEAD_GATHER[world, 1] = logits_gather.HEAD_GATHER[16, 16]
    if rank == world - 1:
        comm.compile_gather = fail
    try:
        rows = logits_gather.install_logits_gather(model, group, world, rank, device)
    finally:
        comm.compile_gather = compile_gather
        logits_gather.HEAD_GATHER.clear()
        logits_gather.HEAD_GATHER.update(saved)
    assert rows == ()
    assert model.logits_processor._gather_logits is stock
    assert not hasattr(model.logits_processor, "head_gather")


def _check_logits_processor(group, rank, world, device):
    """A real LogitsProcessor over a vocab-parallel head: the same logits, bit for
    bit, before and after `install_logits_gather` (this world size given the
    TP16 x DCP16 rows)."""
    from vllm.model_executor.layers.logits_processor import LogitsProcessor

    shard, hidden = VOCAB // world, 256
    g = torch.Generator(device="cpu").manual_seed(7)
    weight = (torch.randn(VOCAB, hidden, generator=g) / 16).to(torch.bfloat16)
    weight = weight[rank * shard : (rank + 1) * shard].contiguous().to(device)

    class Method:
        def apply(self, layer, x, bias=None):
            return x @ layer.weight.t()

    head = SimpleNamespace(weight=weight, tp_size=world, quant_method=Method())
    lp = LogitsProcessor(VOCAB - 100)  # padded vocabulary: sliced after the gather
    model = SimpleNamespace(logits_processor=lp, lm_head=head)
    xs = {
        m: torch.randn(m, hidden, generator=g).to(torch.bfloat16).to(device)
        for m in (1, 2, 4, 5, 7, 8, 9, 12)
    }
    before = {m: lp(head, x) for m, x in xs.items()}
    saved = dict(logits_gather.HEAD_GATHER)
    logits_gather.HEAD_GATHER[world, 1] = logits_gather.HEAD_GATHER[16, 16]
    try:
        rows = logits_gather.install_logits_gather(model, group, world, rank, device)
    finally:
        logits_gather.HEAD_GATHER.clear()
        logits_gather.HEAD_GATHER.update(saved)
    assert rows == tuple(range(1, 9))
    assert isinstance(lp.head_gather, logits_gather.HeadGather)
    for _ in range(3):
        for m, x in xs.items():
            got = lp(head, x)
            assert got.shape == (m, VOCAB - 100)
            assert _same_bytes(got, before[m]), m


def _worker(rank, world, port, result):
    os.environ.update(
        RANK=str(rank),
        LOCAL_RANK=str(rank),
        WORLD_SIZE=str(world),
        MASTER_ADDR="127.0.0.1",
        MASTER_PORT=str(port),
    )
    device = torch.device("cuda", rank)
    torch.accelerator.set_device_index(rank)
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.distributed import (
        get_tp_group,
        init_distributed_environment,
        initialize_model_parallel,
    )

    with set_current_vllm_config(VllmConfig()):
        init_distributed_environment(world, rank, f"tcp://127.0.0.1:{port}", rank)
        initialize_model_parallel(tensor_model_parallel_size=world)
        if not _has_multicast(device):
            result[rank] = 2
            return
        group = get_tp_group().device_group
        rows = range(1, 9)
        gather = logits_gather.HeadGather(
            group, device, world, rank, VOCAB // world, rows
        )
        _check_head_gather(gather, rank, world, device)
        _check_capture(gather, world, device)
        _check_one_rank_fails(group, rank, world, device)
        _check_logits_processor(group, rank, world, device)
        torch.distributed.barrier()
    result[rank] = 1


@pytest.mark.skipif(
    not current_platform.is_device_capability_family(100),
    reason="the specialized decode kernels need SM100/SM103",
)
@pytest.mark.skipif(
    torch.accelerator.device_count() < WORLD, reason=f"Need {WORLD} GPUs."
)
def test_logits_gather_matches_all_gather():
    ctx = mp.get_context("spawn")
    result = ctx.Array("i", [0] * WORLD)
    port = get_open_port()
    procs = [
        ctx.Process(target=_worker, args=(r, WORLD, port, result)) for r in range(WORLD)
    ]
    for p in procs:
        p.start()
    for p in procs:
        p.join(timeout=600)
    assert all(p.exitcode == 0 for p in procs), [p.exitcode for p in procs]
    if 2 in list(result):
        pytest.skip("needs NVLink multicast")
    assert list(result) == [1] * WORLD


if __name__ == "__main__":
    test_logits_gather_matches_all_gather()
    print("ok")
