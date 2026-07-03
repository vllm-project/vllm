# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared multiprocess test harness for CPU distributed/EP tests."""

import traceback

import pytest
import torch.multiprocessing as mp

from vllm.utils.network_utils import get_open_port


def ensure_spawn_start_method():
    if mp.get_start_method(allow_none=True) is None:
        mp.set_start_method("spawn")


def report_worker_failure(rank: int, err_q: mp.Queue, err: Exception) -> None:
    err_q.put(f"[Rank {rank}]\n{traceback.format_exc()}")
    raise SystemExit(1) from err


def collect_worker_failures(procs, err_q: mp.Queue) -> list[str]:
    exit_errors = []
    for rank, proc in enumerate(procs):
        proc.join()
        if proc.exitcode != 0:
            exit_errors.append(f"[Rank {rank}] worker exited with code {proc.exitcode}")

    errors = []
    while not err_q.empty():
        errors.append(err_q.get_nowait())
    err_q.close()
    err_q.join_thread()
    return errors + exit_errors


def spawn_workers(
    worker_fn,
    world_size,
    tp_size,
    dp_size,
    params,
    *,
    distributed_init_ports=None,
    dp_port=None,
):
    ensure_spawn_start_method()

    if distributed_init_ports is None:
        shared_init_port = get_open_port()
        distributed_init_ports = [shared_init_port] * world_size
    elif len(distributed_init_ports) != world_size:
        raise ValueError("distributed_init_ports must provide one port per worker rank")

    if dp_port is None:
        dp_port = get_open_port()

    err_q: mp.Queue = mp.Queue()
    procs = []
    for rank in range(world_size):
        proc = mp.Process(
            target=worker_fn,
            args=(
                rank,
                world_size,
                tp_size,
                dp_size,
                distributed_init_ports[rank],
                dp_port,
                params,
                err_q,
            ),
        )
        proc.start()
        procs.append(proc)

    failures = collect_worker_failures(procs, err_q)
    if failures:
        pytest.fail("Worker(s) failed:\n" + "\n---\n".join(failures))
