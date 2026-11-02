# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The specialized decode's TP all-gather (``LamportAllGather``) at TP4.

* ``accepts`` by size: 16-byte multiples up to the capacity.
* Eager gathers of int32, BF16 and FP32 tensors whose sizes grow and shrink
  from call to call (1 pack to the full capacity), one rank late each call,
  bitwise against NCCL.
* The Lamport sentinel word (FP32 negative zero, INT32_MIN) arrives as 0 on
  every rank, the sending rank included.
* CUDA graphs of three gathers of different sizes each, some with the ranks
  skewed, replayed in a seeded random order with new contents every replay;
  a device-side count of words that differ from the expected concatenation
  must stay 0. A word left in the mailbox by an earlier call of another size
  would show up here as a mismatch.

Needs four SM100-family GPUs with NVLink multicast; about a minute. For a
stress run, run this file as a module
(``python -m tests.models.deepseek_v41.nvidia.specialized_decode.test_all_gather``)
with ``--replays 2000000``. With ``--custom-all-gather`` it replays the same
graphs on vLLM's custom all-gather (``CustomAllreduce.custom_all_gather``, the
MNNVL Lamport kernel) instead and only counts the mismatched words: that
kernel never waits for a rank's own multicast copy, and words of an earlier
call can reach a later one.
"""

import argparse
import os
import random
import tempfile

import pytest
import torch

from vllm.platforms import current_platform

TP = 4
CAPACITY = 4 << 20  # bytes a rank, as the executor builds it
REPLAYS = 2000
LATE_CYCLES = 20_000
SENTINEL = -(2**31)
# Pack counts (16 bytes each) of the eager sequence: growing, shrinking, a
# partial last CTA, one pack and the full capacity.
EAGER_PACKS = (1, 37, 389, 128, 5000, 3, CAPACITY // 16, 2, 777, 1)
# Words a rank of each graph's three gathers.
GRAPH_WORDS = (
    (4, 2800, 148),
    (20000, 12, 2048),
    (512, 8192, 4),
    (CAPACITY // 4, 4, 64),
)


def _late(rank: int, step: int) -> None:
    """Delay one rank per step, a different one each step."""
    if rank == step % TP:
        torch.cuda._sleep(LATE_CYCLES)


def _random_words(generator, words: int) -> torch.Tensor:
    """Random int32 words, none of them the sentinel."""
    x = torch.randint(
        -(2**31),
        2**31 - 1,
        (words,),
        generator=generator,
        device="cuda",
        dtype=torch.int64,
    ).to(torch.int32)
    return torch.where(x == SENTINEL, 1, x)


def _check_accepts(all_gather):
    assert not all_gather.accepts(torch.empty(0, device="cuda"))
    assert not all_gather.accepts(torch.empty(2, device="cuda"))  # 8 bytes
    assert all_gather.accepts(torch.empty(4, device="cuda"))
    assert all_gather.accepts(torch.empty(CAPACITY // 4, device="cuda"))
    assert not all_gather.accepts(torch.empty(CAPACITY // 4 + 4, device="cuda"))
    with pytest.raises(ValueError):
        all_gather(torch.empty(2, device="cuda"))


def _check_eager(all_gather, rank):
    from vllm.distributed import get_tp_group

    group = get_tp_group()
    generator = torch.Generator(device="cuda").manual_seed(rank)
    for step, packs in enumerate(EAGER_PACKS):
        words = _random_words(generator, packs * 4)
        for dtype in (torch.int32, torch.bfloat16, torch.float32):
            # One 16-byte pack a row.
            x = words.view(dtype).view(packs, -1)
            _late(rank, step)
            got = all_gather(x)
            want = group.all_gather(x, 0)
            assert got.dtype == dtype and got.shape == want.shape, (packs, dtype)
            assert torch.equal(got.view(torch.int32), want.view(torch.int32)), (
                packs,
                dtype,
            )


def _check_sentinel(all_gather, rank):
    from vllm.distributed import get_tp_group

    generator = torch.Generator(device="cuda").manual_seed(100 + rank)
    x = _random_words(generator, 64)
    x[rank::7] = SENTINEL
    got = all_gather(x)
    want = get_tp_group().all_gather(x, 0)
    want = torch.where(want == SENTINEL, 0, want)
    assert torch.equal(got, want)
    negative_zero = torch.full((8,), -0.0, device="cuda")
    gathered = all_gather(negative_zero)
    assert torch.equal(gathered.view(torch.int32), torch.zeros_like(gathered).int())


def _pattern(rank: int, kind: int, words: int) -> torch.Tensor:
    """(rank, kind, index) as int32 bits, never the sentinel; the replay
    counter goes into bits 20-25."""
    index = torch.arange(words, dtype=torch.int32, device="cuda") & 0xFFFFF
    return (rank << 28) | (kind << 26) | index


def _check_replays(all_gather, rank, replays, stop=True):
    """Gather graphs replayed in a random order; returns the mismatch count,
    after the first 1000 replays with a mismatch if ``stop``."""
    from vllm.distributed.parallel_state import graph_capture

    counter = torch.zeros(1, dtype=torch.int32, device="cuda")
    errors = torch.zeros(1, dtype=torch.int64, device="cuda")
    graphs = []
    with graph_capture(device=torch.device("cuda")) as context:
        for sizes in GRAPH_WORDS:
            mine = [_pattern(rank, k, n) for k, n in enumerate(sizes)]
            want = [
                torch.cat([_pattern(r, k, n) for r in range(TP)])
                for k, n in enumerate(sizes)
            ]
            inputs = [torch.empty_like(m) for m in mine]
            for skew in (False, True):

                def body(mine=mine, want=want, inputs=inputs, skew=skew):
                    counter.add_(1)
                    bits = (counter & 0x3F) << 20
                    outs = []
                    for kind in range(3):
                        torch.add(mine[kind], bits, out=inputs[kind])
                        if skew:
                            torch.cuda._sleep(LATE_CYCLES * ((rank + kind) % TP))
                        outs.append(all_gather(inputs[kind]))
                    for kind in range(3):
                        errors.add_((outs[kind] != want[kind] + bits).sum())

                body()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=context.stream):
                    body()
                # The graph keeps raw pointers: keep its tensors alive too.
                graphs.append((graph, (mine, want, inputs)))
    torch.accelerator.synchronize()
    assert not stop or errors.item() == 0, "eager warm-up"
    order = random.Random(1234)
    for start in range(0, replays, 1000):
        for _ in range(min(1000, replays - start)):
            graphs[order.randrange(len(graphs))][0].replay()
        torch.accelerator.synchronize()
        if stop and errors.item():
            break
    return int(errors.item())


def _worker(rank: int, port: int, replays: int, skip_path: str, custom: bool) -> None:
    from tests.utils import init_test_distributed_environment
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.distributed import cleanup_dist_env_and_memory, get_tp_group
    from vllm.models.deepseek_v41.nvidia.ops.cute_dsl import LamportAllGather

    torch.accelerator.set_device_index(rank)
    with set_current_vllm_config(VllmConfig()), torch.inference_mode():
        init_test_distributed_environment(TP, 1, rank, str(port), local_rank=rank)
        ca_comm = getattr(get_tp_group().device_communicator, "ca_comm", None)
        supported = torch.tensor(
            [int(ca_comm is not None and bool(ca_comm.mnnvl_lamport_ag_multicast_ptr))]
        )
        torch.distributed.all_reduce(
            supported, torch.distributed.ReduceOp.MIN, get_tp_group().cpu_group
        )
        if not supported.item():
            if rank == 0:
                with open(skip_path, "w") as f:
                    f.write("no NVLink multicast")
            cleanup_dist_env_and_memory()
            return
        if custom:

            def all_gather(x):
                # It takes floating-point inputs only; the int32 words pass
                # bitwise as FP32.
                out = ca_comm.custom_all_gather(x.view(torch.float32))
                assert out is not None, x.nbytes
                return out.view(x.dtype)

        else:
            all_gather = LamportAllGather(
                rank_capacity_bytes=CAPACITY, device=torch.device("cuda", rank)
            )
            _check_accepts(all_gather)
            _check_eager(all_gather, rank)
            _check_sentinel(all_gather, rank)
        errors = _check_replays(all_gather, rank, replays, stop=not custom)
        total = torch.tensor([errors])
        torch.distributed.all_reduce(total, group=get_tp_group().cpu_group)
        print(
            f"rank {rank}: {replays} replays ({3 * replays} gathers) on the "
            f"{'custom' if custom else 'Lamport'} all-gather, {errors} mismatched "
            f"words here, {int(total.item())} on all ranks",
            flush=True,
        )
        assert custom or total.item() == 0
        torch.accelerator.synchronize()
        cleanup_dist_env_and_memory()


def run(replays: int, custom: bool = False) -> str | None:
    """Run the checks on four GPUs (only the replays on the custom all-gather
    if ``custom``); returns why they were skipped, if so."""
    from vllm.utils.network_utils import get_open_port

    with tempfile.TemporaryDirectory() as tmp:
        skip_path = os.path.join(tmp, "skipped")
        torch.multiprocessing.spawn(
            _worker, args=(get_open_port(), replays, skip_path, custom), nprocs=TP
        )
        if os.path.exists(skip_path):
            with open(skip_path) as f:
                return f.read()
    return None


@pytest.mark.distributed(num_gpus=4)
@pytest.mark.skipif(
    not current_platform.is_device_capability_family(100), reason="needs SM100"
)
def test_lamport_all_gather():
    if torch.accelerator.device_count() < TP:
        pytest.skip("needs four GPUs")
    skipped = run(REPLAYS)
    if skipped:
        pytest.skip(skipped)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--replays", type=int, default=REPLAYS)
    parser.add_argument(
        "--custom-all-gather",
        action="store_true",
        help="replay the graphs on vLLM's custom all-gather and count mismatches",
    )
    args = parser.parse_args()
    skipped = run(args.replays, args.custom_all_gather)
    print(f"skipped: {skipped}" if skipped else "done")
