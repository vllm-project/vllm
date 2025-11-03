# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import ctypes
import json
import os
import pickle
import subprocess
import sys
import tempfile
from collections.abc import Sequence
from itertools import product
from typing import Any

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

import vllm.envs as envs
from vllm.distributed.device_communicators.cuda_wrapper import CudaRTLibrary
from vllm.logger import init_logger
from vllm.model_executor.layers.batch_invariant import (
    vllm_is_batch_invariant,
)
from vllm.utils.system_utils import update_environment_variables
from vllm.utils.torch_utils import cuda_device_count_stateless

logger = init_logger(__name__)

MiB = 1024 * 1024
# Max size for each world size in case symmetric memory is available
# For different SM architectures
CUSTOM_ALL_REDUCE_MAX_SIZES = {
    "9.0": {
        2: 64 * MiB,  # 64 MB
        4: 32 * MiB,  # 32 MB
        6: MiB // 2,  # 512 KB
        8: MiB // 4,  # 256 KB
    },
    "10.0": {
        2: 2 * MiB,  # 2 MB
        4: 2 * MiB,  # 2 MB
        6: 1 * MiB,  # 1 MB
        8: 1 * MiB,  # 1 MB
    },
}

SYMM_MEM_ALL_REDUCE_MAX_SIZES = {
    "9.0": {
        2: 64 * MiB,  # 64 MB
        4: 32 * MiB,  # 32 MB
        6: 64 * MiB,  # 64 MB
        8: 64 * MiB,  # 64 MB
    },
    "10.0": {
        2: 8 * MiB,  # 8 MB
        4: 32 * MiB,  # 32 MB
        6: 128 * MiB,  # 128 MB
        8: 128 * MiB,  # 128 MB
    },
}

NCCL_SYMM_MEM_ALL_REDUCE_CONFIG: dict[str, Any] = {
    "min_world_size": 4,
    "thresholds": {
        4: 2 * MiB,  # 2 MB
        8: 1 * MiB,  # 1 MB
    },
    "always_use_above_world_size": 8,  # Always use symm mem for world_size > 8
}


def should_nccl_symm_mem_allreduce(world_size: int, input_tensor: torch.Tensor) -> bool:
    from vllm.distributed.device_communicators.pynccl_allocator import (
        is_symmetric_memory_enabled,
    )

    if vllm_is_batch_invariant():
        return False

    if not is_symmetric_memory_enabled():
        return False
    if world_size < NCCL_SYMM_MEM_ALL_REDUCE_CONFIG["min_world_size"]:
        return False
    threshold = NCCL_SYMM_MEM_ALL_REDUCE_CONFIG["thresholds"].get(world_size)
    if threshold is not None and input_tensor.nbytes >= threshold:
        return True
    return world_size > NCCL_SYMM_MEM_ALL_REDUCE_CONFIG["always_use_above_world_size"]


def producer(
    batch_src: Sequence[int],
    producer_queue,
    consumer_queue,
    result_queue,
    cuda_visible_devices: str | None = None,
):
    if cuda_visible_devices is not None:
        update_environment_variables({"CUDA_VISIBLE_DEVICES": cuda_visible_devices})

    lib = CudaRTLibrary()
    for i in batch_src:
        lib.cudaSetDevice(i)
        pointer = lib.cudaMalloc(1024)
        lib.cudaMemset(pointer, 1, 1024)
        lib.cudaDeviceSynchronize()
        handle = lib.cudaIpcGetMemHandle(pointer)
        producer_queue.put(handle)
        open_success = consumer_queue.get()
        if open_success:
            # use two queues to simulate barrier
            producer_queue.put(0)
            consumer_queue.get()
            # check if the memory is modified
            host_data = (ctypes.c_char * 1024)()
            lib.cudaMemcpy(host_data, pointer, 1024)  # type: ignore
            for i in range(1024):
                if ord(host_data[i]) != 2:
                    open_success = False
                    break
        result_queue.put(open_success)
        lib.cudaDeviceReset()


def consumer(
    batch_tgt: Sequence[int],
    producer_queue,
    consumer_queue,
    result_queue,
    cuda_visible_devices: str | None = None,
):
    if cuda_visible_devices is not None:
        update_environment_variables({"CUDA_VISIBLE_DEVICES": cuda_visible_devices})

    lib = CudaRTLibrary()
    for j in batch_tgt:
        lib.cudaSetDevice(j)
        handle = producer_queue.get()
        open_success = False
        try:
            pointer = lib.cudaIpcOpenMemHandle(handle)  # type: ignore
            open_success = True
        except RuntimeError:
            # cannot error out here, because the producer process
            # is still waiting for the response.
            pass
        consumer_queue.put(open_success)
        if open_success:
            # modify the memory
            lib.cudaMemset(pointer, 2, 1024)
            lib.cudaDeviceSynchronize()
            # use two queues to simulate barrier
            producer_queue.get()
            consumer_queue.put(0)
            # check if the memory is modified
            host_data = (ctypes.c_char * 1024)()
            lib.cudaMemcpy(host_data, pointer, 1024)  # type: ignore
            for i in range(1024):
                if ord(host_data[i]) != 2:
                    open_success = False
                    break
        result_queue.put(open_success)
        lib.cudaDeviceReset()


def can_actually_p2p(
    batch_src: Sequence[int],
    batch_tgt: Sequence[int],
) -> Sequence[bool]:
    """
    Usually, checking if P2P access is enabled can be done by
    `torch.cuda.can_device_access_peer(src, tgt)`. However, sometimes
    the driver might be broken, and `torch.cuda.can_device_access_peer(src, tgt)`
    returns `True` even if P2P access is not actually possible.
    See https://github.com/vllm-project/vllm/issues/2728 and
    https://forums.developer.nvidia.com/t/direct-gpu-gpu-communication-does-not-seem-to-work-properly/283264/10
    Therefore, we have to perform a real P2P access to check if it is actually
    possible.

    Note on p2p and cuda IPC:
    Usually, one process uses one GPU:
    GPU src --> cuda context src --> tensor src --> process src

    We need to combine p2p and cuda IPC, so that:
    GPU src --> cuda context src --> tensor src --> process src
                                      |shared|
    GPU tgt --> cuda context tgt --> tensor tgt --> process tgt
    That is to say, process src creates a tensor in GPU src, passes IPC handle to
    process tgt, and process tgt accesses the tensor in GPU tgt. Any operation on the
    tensor in process tgt will be reflected in the tensor in process src, because
    they are the same memory segment.
    It is important to note that process tgt accesses the tensor in GPU tgt, not
    GPU src. That's why we need p2p access.

    The most time-consuming part is the process creation. To avoid creating
    processes for every pair of GPUs, we use batched testing. We create two
    processes for testing all pairs of GPUs in batch. The trick is to reset
    the device after each test (which is not available in PyTorch).
    """  # noqa
    cuda_visible_devices = envs.CUDA_VISIBLE_DEVICES
    # pass the CUDA_VISIBLE_DEVICES to the child process
    # to make sure they see the same set of GPUs

    # make sure the processes are spawned
    smp = mp.get_context("spawn")
    producer_queue = smp.Queue()
    consumer_queue = smp.Queue()
    result_queue = smp.Queue()
    p_src = smp.Process(
        target=producer,
        args=(
            batch_src,
            producer_queue,
            consumer_queue,
            result_queue,
            cuda_visible_devices,
        ),
    )
    p_tgt = smp.Process(
        target=consumer,
        args=(
            batch_tgt,
            producer_queue,
            consumer_queue,
            result_queue,
            cuda_visible_devices,
        ),
    )
    p_src.start()
    p_tgt.start()
    p_src.join()
    p_tgt.join()
    assert p_src.exitcode == 0 and p_tgt.exitcode == 0
    result: list[bool] = []
    for src, tgt in zip(batch_src, batch_tgt):
        a = result_queue.get()
        b = result_queue.get()
        if a != b:
            logger.warning(
                "Two processes do not agree on the P2P access"
                " status on %d -> %d, treat as disabled.",
                src,
                tgt,
            )
            result.append(False)
        else:
            result.append(a)
    return result


# why do we need this cache?
# we are testing peer-to-peer (p2p) access between GPUs,across processes.
# if we test it every time, it will be very slow, because we need to create
#  N * N * 2 processes, where N is the world size. This is very slow.
# to reduce the time, we use a cache file to store the p2p access status.
# the cache file is generated by the master process if it does not exist.
# then all the processes can read the cache file to check the p2p access status.
# Note that the cache file is suffixed by the CUDA_VISIBLE_DEVICES, so that we
#  can have different cache files for different CUDA_VISIBLE_DEVICES settings,
#  e.g. used by different vllm engines. The device id in the cache file is a
#  **local** device id, i.e. from 0 to num_dev-1, where num_dev is the number
#  of visible devices in the vllm engine.
_gpu_p2p_access_cache: dict[str, bool] | None = None


def gpu_p2p_access_check(src: int, tgt: int) -> bool:
    """Check if GPU src can access GPU tgt."""

    # if the cache variable is already calculated,
    # read from the cache instead of checking it again
    global _gpu_p2p_access_cache
    if _gpu_p2p_access_cache is not None:
        return _gpu_p2p_access_cache[f"{src}->{tgt}"]

    is_distributed = dist.is_initialized()

    num_dev = cuda_device_count_stateless()
    cuda_visible_devices = envs.CUDA_VISIBLE_DEVICES
    if cuda_visible_devices is None:
        cuda_visible_devices = ",".join(str(i) for i in range(num_dev))

    path = os.path.join(
        envs.VLLM_CACHE_ROOT, f"gpu_p2p_access_cache_for_{cuda_visible_devices}.json"
    )
    os.makedirs(os.path.dirname(path), exist_ok=True)
    from vllm.distributed.parallel_state import get_world_group

    if (not is_distributed or get_world_group().local_rank == 0) and (
        not os.path.exists(path)
    ):
        # only the local master process (with local_rank == 0) can
        #  enter this block to calculate the cache
        logger.info("generating GPU P2P access cache in %s", path)
        cache: dict[str, bool] = {}
        ids = list(range(num_dev))
        # batch of all pairs of GPUs
        batch_src, batch_tgt = zip(*list(product(ids, ids)))
        # NOTE: we use `subprocess` rather than `multiprocessing` here
        # because the caller might not have `if __name__ == "__main__":`,
        # in that case we cannot use spawn method in multiprocessing.
        # However, `can_actually_p2p` requires spawn method.
        # The fix is, we use `subprocess` to call the function,
        # where we have `if __name__ == "__main__":` in this file.

        # use a temporary file to store the result
        # we don't use the output of the subprocess directly,
        # because the subprocess might produce logging output
        with tempfile.NamedTemporaryFile() as output_file:
            input_bytes = pickle.dumps((batch_src, batch_tgt, output_file.name))
            returned = subprocess.run(
                [sys.executable, __file__], input=input_bytes, capture_output=True
            )
            # check if the subprocess is successful
            try:
                returned.check_returncode()
            except Exception as e:
                # wrap raised exception to provide more information
                raise RuntimeError(
                    f"Error happened when batch testing "
                    f"peer-to-peer access from {batch_src} to {batch_tgt}:\n"
                    f"{returned.stderr.decode()}"
                ) from e
            with open(output_file.name, "rb") as f:
                result = pickle.load(f)
        for _i, _j, r in zip(batch_src, batch_tgt, result):
            cache[f"{_i}->{_j}"] = r
        with open(path, "w") as f:
            json.dump(cache, f, indent=4)
    if is_distributed:
        get_world_group().barrier()
    logger.info("reading GPU P2P access cache from %s", path)
    with open(path) as f:
        cache = json.load(f)
    _gpu_p2p_access_cache = cache
    return _gpu_p2p_access_cache[f"{src}->{tgt}"]


__all__ = ["gpu_p2p_access_check"]

if __name__ == "__main__":
    batch_src, batch_tgt, output_file = pickle.loads(sys.stdin.buffer.read())
    result = can_actually_p2p(batch_src, batch_tgt)
    with open(output_file, "wb") as f:
        f.write(pickle.dumps(result))
