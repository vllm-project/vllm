# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare original, Python-sharded, and Triton-sharded XPU collectives.

The original rounds after every addition; both sharded versions accumulate in
fp32 and round once. Validate them against a fixed-rank fp32 reference.

Run with `VLLM_BATCH_INVARIANT=1 python -m torch.distributed.run
--standalone --nproc-per-node=2
benchmarks/kernels/benchmark_xpu_batch_invariant_collectives.py`.

Times include Python dispatch, allocations, copies, and device completion.
Inputs and process-group barriers are outside timing. Each sample is the
slowest rank's mean latency; implementations rotate order between samples.
"""

import argparse
import inspect
import os
import statistics
import time
from collections.abc import Callable
from datetime import timedelta
from functools import partial

import torch
import torch.distributed as dist

from vllm import envs
from vllm.distributed.device_communicators.xpu_communicator import (
    _BATCH_INVARIANT_GATHER_MAX_BYTES,
    XpuCommunicator,
)


class _PythonShardCommunicator(XpuCommunicator):
    """Use separate PyTorch additions with the same fp32 accumulation policy."""

    def _fixed_rank_reduce_scatter(
        self, input_: torch.Tensor, sizes: list[int] | None = None
    ) -> torch.Tensor:
        if input_.numel() * input_.element_size() <= _BATCH_INVARIANT_GATHER_MAX_BYTES:
            return super()._fixed_rank_reduce_scatter(input_, sizes)
        if sizes is None:
            assert input_.shape[0] % self.world_size == 0
            sizes = [input_.shape[0] // self.world_size] * self.world_size
        rows = sizes[self.rank_in_group]
        gathered = input_.new_empty((self.world_size * rows, *input_.shape[1:]))
        dist.all_to_all_single(
            gathered,
            input_,
            input_split_sizes=sizes,
            output_split_sizes=[rows] * self.world_size,
            group=self.device_group,
        )
        gathered = gathered.view(self.world_size, rows, *input_.shape[1:])
        output = torch.zeros(
            (rows, *input_.shape[1:]), dtype=torch.float32, device=input_.device
        )
        for rank in range(self.world_size):
            output.add_(gathered[rank])
        return output.to(input_.dtype)


def _baseline(
    x: torch.Tensor,
    communicator: XpuCommunicator,
    operation: str,
    sizes: list[int],
    *,
    accumulate_fp32: bool = False,
) -> torch.Tensor:
    output = None
    if operation != "all_reduce":
        output = x.new_empty((sizes[communicator.rank_in_group], *x.shape[1:]))
    gathered = x.new_empty((communicator.world_size, x.numel()))
    dist.all_gather_single(
        gathered.view(-1), x.reshape(-1), group=communicator.device_group
    )
    if accumulate_fp32:
        reduced = torch.zeros(x.numel(), dtype=torch.float32, device=x.device)
        start_rank = 0
    else:
        reduced = gathered[0].clone()
        start_rank = 1
    for rank in range(start_rank, communicator.world_size):
        reduced.add_(gathered[rank])
    reduced = reduced.to(x.dtype).view(x.shape)
    if output is None:
        return reduced
    start = sum(sizes[: communicator.rank_in_group])
    output.copy_(reduced.narrow(0, start, output.shape[0]))
    return output


def _peak_memory(fn: Callable[[], torch.Tensor]) -> int:
    torch.xpu.synchronize()
    before = torch.xpu.memory_allocated()
    torch.xpu.reset_peak_memory_stats()
    output = fn()
    torch.xpu.synchronize()
    peak = torch.xpu.max_memory_allocated() - before
    del output
    return peak


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", nargs="+", type=int, default=[2, 16, 256, 2048, 8192])
    parser.add_argument("--hidden-size", type=int, default=4096)
    parser.add_argument(
        "--dtype", choices=["float32", "float16", "bfloat16"], default="bfloat16"
    )
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--samples", type=int, default=15)
    parser.add_argument("--iterations", type=int, default=5)
    args = parser.parse_args()
    if not envs.VLLM_BATCH_INVARIANT:
        parser.error("set VLLM_BATCH_INVARIANT=1")
    if min(*args.rows, args.hidden_size, args.samples, args.iterations) <= 0:
        parser.error("dimensions, samples, and iterations must be positive")
    if args.warmup < 0:
        parser.error("warmup must be nonnegative")
    if int(os.environ["WORLD_SIZE"]) < 2:
        parser.error("use at least two XPU ranks")

    device = torch.device("xpu", int(os.environ["LOCAL_RANK"]))
    torch.xpu.set_device(device)
    dist.init_process_group("xccl", timeout=timedelta(seconds=120))
    try:
        cpu_group = dist.new_group(backend="gloo")
        communicator = XpuCommunicator(cpu_group, device, dist.group.WORLD)
        python_communicator = _PythonShardCommunicator(
            cpu_group, device, dist.group.WORLD
        )
        rank, world_size = communicator.rank_in_group, communicator.world_size
        torch.manual_seed(42 + rank)
        if rank == 0:
            print(
                f"device={torch.xpu.get_device_name(device)}, "
                f"torch={torch.__version__}, "
                f"backend=xccl, world_size={world_size}, dtype={args.dtype}, "
                f"hidden_size={args.hidden_size}, samples={args.samples}, "
                f"iterations={args.iterations}, warmup={args.warmup}",
                flush=True,
            )
            print(f"source={inspect.getfile(XpuCommunicator)}", flush=True)
            print(
                "operation,rows,original_us,python_us,triton_us,"
                "triton_vs_python,triton_vs_original,"
                "original_iqr_us,python_iqr_us,triton_iqr_us,"
                "original_peak_MiB,python_peak_MiB,triton_peak_MiB",
                flush=True,
            )
        for operation in ("all_reduce", "reduce_scatter", "reduce_scatterv"):
            for rows in args.rows:
                if operation == "reduce_scatter":
                    rows = ((rows + world_size - 1) // world_size) * world_size
                sizes = [rows // world_size] * world_size
                sizes[-1] += rows % world_size
                if operation == "reduce_scatterv":
                    sizes = [0] + [rows // (world_size - 1)] * (world_size - 1)
                    sizes[-1] += rows - sum(sizes)
                x = torch.randn(
                    (rows, args.hidden_size),
                    dtype=getattr(torch, args.dtype),
                    device=device,
                )

                baseline = partial(_baseline, x, communicator, operation, sizes)
                functions = [baseline]
                for comm in (python_communicator, communicator):
                    if operation == "all_reduce":
                        functions.append(partial(comm.all_reduce, x))
                    elif operation == "reduce_scatter":
                        functions.append(partial(comm.reduce_scatter, x, dim=0))
                    else:
                        functions.append(
                            partial(comm.reduce_scatterv, x, dim=0, sizes=sizes)
                        )
                expected = _baseline(
                    x, communicator, operation, sizes, accumulate_fp32=True
                )
                for fn in functions[1:]:
                    torch.testing.assert_close(fn(), expected, rtol=0, atol=0)
                del expected
                for _ in range(args.warmup):
                    for fn in functions:
                        fn()
                peaks = torch.tensor([_peak_memory(fn) for fn in functions])
                dist.all_reduce(peaks, op=dist.ReduceOp.MAX, group=cpu_group)
                samples: list[list[float]] = [[], [], []]
                for sample in range(args.samples):
                    for offset in range(3):
                        index = (sample + offset) % 3
                        dist.barrier()
                        torch.xpu.synchronize()
                        start = time.perf_counter()
                        for _ in range(args.iterations):
                            functions[index]()
                        torch.xpu.synchronize()
                        elapsed = torch.tensor(
                            (time.perf_counter() - start) * 1e6 / args.iterations,
                            dtype=torch.float64,
                        )
                        dist.all_reduce(elapsed, op=dist.ReduceOp.MAX, group=cpu_group)
                        samples[index].append(elapsed.item())
                if rank == 0:
                    original, python, triton = map(statistics.median, samples)
                    iqrs = [
                        (
                            sorted(values)[3 * len(values) // 4]
                            - sorted(values)[len(values) // 4]
                        )
                        for values in samples
                    ]
                    print(
                        f"{operation},{rows},{original:.1f},{python:.1f},{triton:.1f},"
                        f"{python / triton:.2f},{original / triton:.2f},"
                        f"{iqrs[0]:.1f},{iqrs[1]:.1f},{iqrs[2]:.1f},"
                        f"{peaks[0].item() / 2**20:.3f},"
                        f"{peaks[1].item() / 2**20:.3f},"
                        f"{peaks[2].item() / 2**20:.3f}",
                        flush=True,
                    )
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
