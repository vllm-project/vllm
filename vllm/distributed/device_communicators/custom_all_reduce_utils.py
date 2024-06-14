import ctypes
import json
import os
from typing import Dict, List, Optional

import torch.distributed as dist
import torch.multiprocessing as mp

import vllm.envs as envs
from vllm.distributed.device_communicators.cuda_wrapper import CudaRTLibrary
from vllm.logger import init_logger
from vllm.utils import cuda_device_count_stateless

logger = init_logger(__name__)


def producer(batch_is: List[int],
             producer_queue,
             consumer_queue,
             result_queue,
             cuda_visible_devices: Optional[str] = None):
    if cuda_visible_devices is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices

    lib = CudaRTLibrary()
    for i in batch_is:
        lib.cudaSetDevice(i)
        pointer = lib.cudaMalloc(1024)
        lib.cudaMemset(pointer, 1, 1024)
        lib.cudaDeviceSynchronize()
        handle = lib.cudaIpcGetMemHandle(pointer)
        producer_queue.put(handle)
        open_success = consumer_queue.get()
        if open_success:
            producer_queue.put(0)
            consumer_queue.get()
            host_data = (ctypes.c_char * 1024)()
            lib.cudaMemcpy(host_data, pointer, 1024)  # type: ignore
            for i in range(1024):
                if ord(host_data[i]) != 2:
                    open_success = False
                    break
        result_queue.put(open_success)
        lib.cudaDeviceReset()


def consumer(batch_js: List[int],
             producer_queue,
             consumer_queue,
             result_queue,
             cuda_visible_devices: Optional[str] = None):
    if cuda_visible_devices is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices

    lib = CudaRTLibrary()
    for j in batch_js:
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
            lib.cudaMemset(pointer, 2, 1024)
            producer_queue.get()
            consumer_queue.put(0)
            host_data = (ctypes.c_char * 1024)()
            lib.cudaMemcpy(host_data, pointer, 1024)  # type: ignore
            for i in range(1024):
                if ord(host_data[i]) != 2:
                    open_success = False
                    break
        result_queue.put(open_success)
        lib.cudaDeviceReset()


def can_actually_p2p(batch_is, batch_js):
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

    # make sure the processes are spawned
    smp = mp.get_context("spawn")
    producer_queue = smp.Queue()
    consumer_queue = smp.Queue()
    result_queue = smp.Queue()
    pi = smp.Process(target=producer,
                     args=(batch_is, producer_queue, consumer_queue,
                           result_queue, cuda_visible_devices))
    pj = smp.Process(target=consumer,
                     args=(batch_js, producer_queue, consumer_queue,
                           result_queue, cuda_visible_devices))
    pi.start()
    pj.start()
    pi.join()
    pj.join()
    result = []
    for i, j in zip(batch_is, batch_js):
        a = result_queue.get()
        b = result_queue.get()
        assert a == b
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
