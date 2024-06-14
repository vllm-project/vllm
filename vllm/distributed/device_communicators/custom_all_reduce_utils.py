import ctypes
import json
import os
import sys
import tempfile
import time
from contextlib import contextmanager
from typing import Dict, Optional

import torch.distributed as dist
import torch.multiprocessing as mp

import vllm.envs as envs
from vllm.distributed.device_communicators.cuda_wrapper import CudaRTLibrary
from vllm.logger import init_logger
from vllm.utils import cuda_device_count_stateless

logger = init_logger(__name__)


@contextmanager
def mute_output():
    with open(os.devnull, "w") as f:
        sys.stderr = f
        sys.stdout = f
        yield


def producer(i: int,
             init_method: str,
             cuda_visible_devices: Optional[str] = None):
    if cuda_visible_devices is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices
    with mute_output():
        dist.init_process_group(
            backend="gloo",
            init_method=init_method,
            world_size=2,
            rank=0,
        )
        lib = CudaRTLibrary()
        lib.cudaSetDevice(i)
        pointer = lib.cudaMalloc(1024)
        lib.cudaMemset(pointer, 1, 1024)
        lib.cudaDeviceSynchronize()
        handle = lib.cudaIpcGetMemHandle(pointer)
        dist.broadcast_object_list([handle], src=0)
        recv = [None]
        dist.broadcast_object_list(recv, src=1)
        open_success = recv[0]
        if open_success:
            dist.barrier()
            host_data = (ctypes.c_char * 1024)()
            lib.cudaMemcpy(host_data, pointer, 1024)
            for i in range(1024):
                assert ord(host_data[i]) == 2
        else:
            raise RuntimeError("Failed to open the IPC handle")


def consumer(j: int,
             init_method: str,
             cuda_visible_devices: Optional[str] = None):
    if cuda_visible_devices is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices
    with mute_output():
        dist.init_process_group(
            backend="gloo",
            init_method=init_method,
            world_size=2,
            rank=1,
        )
        lib = CudaRTLibrary()
        lib.cudaSetDevice(j)
        recv = [None]
        dist.broadcast_object_list(recv, src=0)
        handle = recv[0]
        open_success = False
        try:
            pointer = lib.cudaIpcOpenMemHandle(handle)  # type: ignore
            open_success = True
        except RuntimeError:
            # cannot error out here, because the producer process
            # is still waiting for the response.
            pass
        dist.broadcast_object_list([open_success], src=1)
        if open_success:
            lib.cudaMemset(pointer, 2, 1024)
            dist.barrier()
            host_data = (ctypes.c_char * 1024)()
            lib.cudaMemcpy(host_data, pointer, 1024)  # type: ignore
            for i in range(1024):
                assert ord(host_data[i]) == 2
        else:
            raise RuntimeError("Failed to open the IPC handle")


def can_actually_p2p(i, j):
    """
    Usually, checking if P2P access is enabled can be done by
    `torch.cuda.can_device_access_peer(i, j)`. However, sometimes
    the driver might be broken, and `torch.cuda.can_device_access_peer(i, j)`
    returns `True` even if P2P access is not actually possible.
    See https://github.com/vllm-project/vllm/issues/2728 and
    https://forums.developer.nvidia.com/t/direct-gpu-gpu-communication-does-not-seem-to-work-properly/283264/10
    Therefore, we have to perform a real P2P access to check if it is actually
    possible.

    Note on p2p and cuda IPC:
    Usually, one process uses one GPU:
    GPU i --> cuda context i --> tensor i --> process i

    We need to combine p2p and cuda IPC, so that:
    GPU i --> cuda context i --> tensor i --> process i
                                 |shared|
    GPU j --> cuda context j --> tensor j --> process j
    That is to say, process i creates a tensor in GPU i, passes IPC handle to
    process j, and process j accesses the tensor in GPU j. Any operation on the
    tensor in process j will be reflected in the tensor in process i, because
    they are the same memory segment.
    It is important to note that process j accesses the tensor in GPU j, not
    GPU i. That's why we need p2p access. # noqa
    """
    cuda_visible_devices = os.getenv('CUDA_VISIBLE_DEVICES', None)
    # pass the CUDA_VISIBLE_DEVICES to the child process
    # to make sure they see the same set of GPUs

    # make sure the temp file is not the same across different calls
    temp_path = tempfile.mktemp() + str(time.time())
    # create an empty file
    with open(temp_path, "w"):
        pass
    init_method = f"file://{temp_path}"

    # make sure the processes are spawned
    smp = mp.get_context("spawn")
    pi = smp.Process(target=producer,
                     args=(i, init_method, cuda_visible_devices))
    pj = smp.Process(target=consumer,
                     args=(j, init_method, cuda_visible_devices))
    pi.start()
    pj.start()
    pi.join()
    pj.join()
    return pi.exitcode == 0 and pj.exitcode == 0


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
_gpu_p2p_access_cache: Optional[Dict[str, bool]] = None


def gpu_p2p_access_check(i: int, j: int) -> bool:
    """Check if GPU i can access GPU j."""

    # if the cache variable is already calculated,
    # read from the cache instead of checking it again
    global _gpu_p2p_access_cache
    if _gpu_p2p_access_cache is not None:
        return _gpu_p2p_access_cache[f"{i}->{j}"]

    is_distributed = dist.is_initialized()

    num_dev = cuda_device_count_stateless()
    cuda_visible_devices = envs.CUDA_VISIBLE_DEVICES
    if cuda_visible_devices is None:
        cuda_visible_devices = ",".join(str(i) for i in range(num_dev))
    VLLM_CONFIG_ROOT = envs.VLLM_CONFIG_ROOT
    path = os.path.expanduser(
        f"{VLLM_CONFIG_ROOT}/vllm/gpu_p2p_access_cache_for_{cuda_visible_devices}.json"
    )
    os.makedirs(os.path.dirname(path), exist_ok=True)
    from vllm.distributed.parallel_state import get_world_group
    if ((not is_distributed or get_world_group().local_rank == 0)
            and (not os.path.exists(path))):
        # only the local master process (with local_rank == 0) can
        #  enter this block to calculate the cache
        logger.info("generating GPU P2P access cache in %s", path)
        cache = {}
        for _i in range(num_dev):
            for _j in range(num_dev):
                cache[f"{_i}->{_j}"] = can_actually_p2p(_i, _j)
        with open(path, "w") as f:
            json.dump(cache, f, indent=4)
    if is_distributed:
        get_world_group().barrier()
    logger.info("reading GPU P2P access cache from %s", path)
    with open(path, "r") as f:
        cache = json.load(f)
    _gpu_p2p_access_cache = cache
    return _gpu_p2p_access_cache[f"{i}->{j}"]


__all__ = ["gpu_p2p_access_check"]
