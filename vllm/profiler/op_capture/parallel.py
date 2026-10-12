# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Capture every tensor-parallel rank of a model, one process per rank.

A rank's forward only differs from another's in the shards it holds, but the
collectives between them are real ops of the model and only exist once the
process groups do, so each rank gets its own process and harness. On the `meta`
device they rendezvous over gloo, whose collectives have meta kernels and
exchange nothing.
"""

import multiprocessing
import queue
import traceback
from collections.abc import Sequence
from multiprocessing.queues import Queue
from typing import Any

from vllm.engine.arg_utils import EngineArgs
from vllm.profiler.op_capture.capture import BatchSpec, OpCapture, capture_batches
from vllm.utils.network_utils import get_distributed_init_method, get_open_port

_POLL_SECONDS = 1.0


def _capture_rank(results: Queue, rank: int, *args: Any, **kwargs: Any) -> None:
    try:
        results.put((rank, capture_batches(*args, rank=rank, **kwargs)))
    except BaseException:
        results.put((rank, traceback.format_exc()))
        raise


def capture_ranks(
    model: str,
    batches: Sequence[BatchSpec],
    *,
    device: str = "meta",
    engine_args: EngineArgs | None = None,
    keep_going: bool = False,
) -> list[list[OpCapture]]:
    """Capture each batch on every tensor-parallel rank `engine_args` asks for.

    Args:
        model: Model id or local path.
        batches: As for `capture_batches`.
        device: As for `capture_batches`; a real device needs one per rank.
        engine_args: Base engine args, whose `tensor_parallel_size` sets the
            number of ranks.
        keep_going: As for `capture_model_ops`.

    Returns:
        Per rank, in rank order, one capture per batch.

    Raises:
        RuntimeError: If any rank failed; the others are stopped.

    """
    engine_args = engine_args or EngineArgs(model=model)
    world_size = engine_args.tensor_parallel_size
    init_method = get_distributed_init_method("127.0.0.1", get_open_port())
    context = multiprocessing.get_context("spawn")
    results: Queue = context.Queue()
    processes = [
        context.Process(
            target=_capture_rank,
            args=(results, rank, model, batches),
            kwargs={
                "device": device,
                "engine_args": engine_args,
                "keep_going": keep_going,
                "distributed_init_method": init_method,
            },
        )
        for rank in range(world_size)
    ]
    for process in processes:
        process.start()
    captures: dict[int, list[OpCapture]] = {}
    try:
        while len(captures) < world_size:
            try:
                rank, result = results.get(timeout=_POLL_SECONDS)
            except queue.Empty:
                for rank, process in enumerate(processes):
                    if rank not in captures and process.exitcode is not None:
                        raise RuntimeError(
                            f"Rank {rank} of {model} exited with code "
                            f"{process.exitcode} before reporting"
                        ) from None
                continue
            if isinstance(result, str):
                raise RuntimeError(f"Rank {rank} of {model} failed:\n{result}")
            captures[rank] = result
    finally:
        for process in processes:
            if process.is_alive() and len(captures) < world_size:
                process.terminate()
            process.join()
    return [captures[rank] for rank in range(world_size)]
