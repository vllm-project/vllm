# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Standalone repro for an intermittent oneCCL hang on the first xccl all_reduce
(TP=2 on XPU). Mirrors what vLLM's XPUWorker.init_device does: set the CCL env,
init a 2-rank xccl process group with a file:// init method, then do one
all_reduce. Each iteration spawns two fresh processes, like each vLLM startup.

Usage:
    python xccl_repro.py [iterations] [--timeout SEC] [--dump] [--keep]

    --dump  on a hang, run `py-spy dump --native` on the stuck ranks before killing
    --keep  on a hang, keep the processes alive for manual py-spy / gdb inspection

Prints one line per iteration and a final SUMMARY line; exits 1 if any iteration
hung or failed.
"""

import argparse
import contextlib
import os
import sys
import time
import uuid

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def worker(rank: int, world_size: int, init_file: str) -> None:
    # Same as vllm/v1/worker/xpu_worker.py init_device
    os.environ["CCL_ATL_TRANSPORT"] = os.getenv("CCL_ATL_TRANSPORT", "ofi")
    os.environ["LOCAL_WORLD_SIZE"] = str(world_size)
    os.environ["LOCAL_RANK"] = str(rank)
    torch.accelerator.set_device_index(rank)
    dist.init_process_group(
        backend="xccl",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
    )
    dist.all_reduce(torch.zeros(1).xpu())
    torch.xpu.synchronize()
    dist.destroy_process_group()


def dump_stacks(pids: list[int]) -> None:
    for pid in pids:
        print(f"=== py-spy dump --native pid={pid} ===", flush=True)
        os.system(f"py-spy dump --native -p {pid} 2>&1")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("iterations", type=int, nargs="?", default=100)
    ap.add_argument(
        "--timeout",
        type=float,
        default=60.0,
        help="seconds before an iteration counts as hung",
    )
    ap.add_argument("--dump", action="store_true")
    ap.add_argument("--keep", action="store_true")
    args = ap.parse_args()

    print(
        "config: "
        f"CCL_ZE_IPC_EXCHANGE={os.getenv('CCL_ZE_IPC_EXCHANGE', '<default>')} "
        f"ZE_AFFINITY_MASK={os.getenv('ZE_AFFINITY_MASK', '<unset>')} "
        f"xpu_device_count={torch.xpu.device_count()}",
        flush=True,
    )

    ctx = mp.get_context("spawn")  # vLLM workers are spawned too
    hangs = fails = 0
    for i in range(1, args.iterations + 1):
        init_file = f"/tmp/xccl_repro_{uuid.uuid4().hex}"
        procs = [ctx.Process(target=worker, args=(r, 2, init_file)) for r in range(2)]
        t0 = time.time()
        for p in procs:
            p.start()
        deadline = t0 + args.timeout
        for p in procs:
            p.join(max(0.0, deadline - time.time()))

        alive = [p for p in procs if p.is_alive()]
        if alive:
            hangs += 1
            pids = [p.pid for p in alive]
            print(
                f"iter {i}: HANG (> {args.timeout:.0f}s), alive pids={pids}", flush=True
            )
            if args.dump:
                dump_stacks(pids)
            if args.keep:
                print(
                    "keeping processes alive for inspection; Ctrl+C to stop", flush=True
                )
                for p in procs:
                    p.join()
            for p in alive:
                p.kill()
            for p in procs:
                p.join()
        else:
            codes = [p.exitcode for p in procs]
            if any(codes):
                fails += 1
                print(f"iter {i}: FAILED exitcodes={codes}", flush=True)
            else:
                print(f"iter {i}: ok in {time.time() - t0:.1f}s", flush=True)

        with contextlib.suppress(OSError):
            os.remove(init_file)

    print(
        f"SUMMARY: iterations={args.iterations} hangs={hangs} failures={fails}",
        flush=True,
    )
    sys.exit(1 if hangs or fails else 0)


if __name__ == "__main__":
    main()
