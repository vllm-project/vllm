import asyncio
import datetime
import enum
import gc
import glob
import os
import socket
import subprocess
import tempfile
import threading
import uuid
import warnings
from collections import defaultdict
from functools import lru_cache, partial
from platform import uname
from typing import (Any, AsyncIterator, Awaitable, Callable, Dict, Generic,
                    Hashable, List, Optional, OrderedDict, Tuple, TypeVar,
                    Union)

import psutil
import torch

import vllm.envs as envs
from vllm.logger import enable_trace_function_call, init_logger

T = TypeVar("T")
logger = init_logger(__name__)

STR_DTYPE_TO_TORCH_DTYPE = {
    "half": torch.half,
    "bfloat16": torch.bfloat16,
    "float": torch.float,
    "fp8": torch.uint8,
}


class Device(enum.Enum):
    GPU = enum.auto()
    CPU = enum.auto()


class Counter:

    def __init__(self, start: int = 0) -> None:
        self.counter = start

    def __next__(self) -> int:
        i = self.counter
        self.counter += 1
        return i

    def reset(self) -> None:
        self.counter = 0


class LRUCache(Generic[T]):

    def __init__(self, capacity: int):
        self.cache: OrderedDict[Hashable, T] = OrderedDict()
        self.capacity = capacity

    def __contains__(self, key: Hashable) -> bool:
        return key in self.cache

    def __len__(self) -> int:
        return len(self.cache)

    def __getitem__(self, key: Hashable) -> Optional[T]:
        return self.get(key)

    def __setitem__(self, key: Hashable, value: T) -> None:
        self.put(key, value)

    def __delitem__(self, key: Hashable) -> None:
        self.pop(key)

    def touch(self, key: Hashable) -> None:
        self.cache.move_to_end(key)

    def get(self,
            key: Hashable,
            default_value: Optional[T] = None) -> Optional[T]:
        if key in self.cache:
            value: Optional[T] = self.cache[key]
            self.cache.move_to_end(key)
        else:
            value = default_value
        return value

    def put(self, key: Hashable, value: T) -> None:
        self.cache[key] = value
        self.cache.move_to_end(key)
        self._remove_old_if_needed()

    def _on_remove(self, key: Hashable, value: Optional[T]):
        pass

    def remove_oldest(self):
        if not self.cache:
            return
        key, value = self.cache.popitem(last=False)
        self._on_remove(key, value)

    def _remove_old_if_needed(self) -> None:
        while len(self.cache) > self.capacity:
            self.remove_oldest()

    def pop(self,
            key: Hashable,
            default_value: Optional[T] = None) -> Optional[T]:
        run_on_remove = key in self.cache
        value: Optional[T] = self.cache.pop(key, default_value)
        if run_on_remove:
            self._on_remove(key, value)
        return value

    def clear(self):
        while len(self.cache) > 0:
            self.remove_oldest()
        self.cache.clear()


def is_hip() -> bool:
    return torch.version.hip is not None


@lru_cache(maxsize=None)
def is_cpu() -> bool:
    from importlib.metadata import PackageNotFoundError, version
    try:
        return "cpu" in version("vllm")
    except PackageNotFoundError:
        return False


@lru_cache(maxsize=None)
def is_neuron() -> bool:
    try:
        import transformers_neuronx
    except ImportError:
        transformers_neuronx = None
    return transformers_neuronx is not None


@lru_cache(maxsize=None)
def get_max_shared_memory_bytes(gpu: int = 0) -> int:
    """Returns the maximum shared memory per thread block in bytes."""
    # NOTE: This import statement should be executed lazily since
    # the Neuron-X backend does not have the `cuda_utils` module.
    from vllm._C import cuda_utils

    max_shared_mem = (
        cuda_utils.get_max_shared_memory_per_block_device_attribute(gpu))
    # value 0 will cause MAX_SEQ_LEN become negative and test_attention.py
    # will fail
    assert max_shared_mem > 0, "max_shared_mem can not be zero"
    return int(max_shared_mem)


def get_cpu_memory() -> int:
    """Returns the total CPU memory of the node in bytes."""
    return psutil.virtual_memory().total


def random_uuid() -> str:
    return str(uuid.uuid4().hex)


@lru_cache(maxsize=None)
def get_vllm_instance_id():
    """
    If the environment variable VLLM_INSTANCE_ID is set, return it.
    Otherwise, return a random UUID.
    Instance id represents an instance of the VLLM. All processes in the same
    instance should have the same instance id.
    """
    return envs.VLLM_INSTANCE_ID or f"vllm-instance-{random_uuid()}"


@lru_cache(maxsize=None)
def in_wsl() -> bool:
    # Reference: https://github.com/microsoft/WSL/issues/4071
    return "microsoft" in " ".join(uname()).lower()


def make_async(func: Callable[..., T]) -> Callable[..., Awaitable[T]]:
    """Take a blocking function, and run it on in an executor thread.

    This function prevents the blocking function from blocking the
    asyncio event loop.
    The code in this function needs to be thread safe.
    """

    def _async_wrapper(*args, **kwargs) -> asyncio.Future:
        loop = asyncio.get_event_loop()
        p_func = partial(func, *args, **kwargs)
        return loop.run_in_executor(executor=None, func=p_func)

    return _async_wrapper


def merge_async_iterators(
        *iterators: AsyncIterator[T]) -> AsyncIterator[Tuple[int, T]]:
    """Merge multiple asynchronous iterators into a single iterator.

    This method handle the case where some iterators finish before others.
    When it yields, it yields a tuple (i, item) where i is the index of the
    iterator that yields the item.
    """
    queue: asyncio.Queue[Union[Tuple[int, T], Exception]] = asyncio.Queue()

    finished = [False] * len(iterators)

    async def producer(i: int, iterator: AsyncIterator[T]):
        try:
            async for item in iterator:
                await queue.put((i, item))
        except Exception as e:
            await queue.put(e)
        finished[i] = True

    _tasks = [
        asyncio.create_task(producer(i, iterator))
        for i, iterator in enumerate(iterators)
    ]

    async def consumer():
        try:
            while not all(finished) or not queue.empty():
                item = await queue.get()
                if isinstance(item, Exception):
                    raise item
                yield item
        except (Exception, asyncio.CancelledError) as e:
            for task in _tasks:
                # NOTE: Pass the error msg in cancel()
                # when only Python 3.9+ is supported.
                task.cancel()
            raise e
        await asyncio.gather(*_tasks)

    return consumer()


def get_ip() -> str:
    host_ip = envs.VLLM_HOST_IP
    if host_ip:
        return host_ip

    # IP is not set, try to get it from the network interface

    # try ipv4
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))  # Doesn't need to be reachable
        return s.getsockname()[0]
    except Exception:
        pass

    # try ipv6
    try:
        s = socket.socket(socket.AF_INET6, socket.SOCK_DGRAM)
        # Google's public DNS server, see
        # https://developers.google.com/speed/public-dns/docs/using#addresses
        s.connect(("2001:4860:4860::8888", 80))  # Doesn't need to be reachable
        return s.getsockname()[0]
    except Exception:
        pass

    warnings.warn(
        "Failed to get the IP address, using 0.0.0.0 by default."
        "The value can be set by the environment variable"
        " VLLM_HOST_IP or HOST_IP.",
        stacklevel=2)
    return "0.0.0.0"


def get_distributed_init_method(ip: str, port: int) -> str:
    # Brackets are not permitted in ipv4 addresses,
    # see https://github.com/python/cpython/issues/103848
    return f"tcp://[{ip}]:{port}" if ":" in ip else f"tcp://{ip}:{port}"


def get_open_port() -> int:
    # try ipv4
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("", 0))
            return s.getsockname()[1]
    except OSError:
        # try ipv6
        with socket.socket(socket.AF_INET6, socket.SOCK_STREAM) as s:
            s.bind(("", 0))
            return s.getsockname()[1]


def update_environment_variables(envs: Dict[str, str]):
    for k, v in envs.items():
        if k in os.environ and os.environ[k] != v:
            logger.warning(
                "Overwriting environment variable %s "
                "from '%s' to '%s'", k, os.environ[k], v)
        os.environ[k] = v


def chunk_list(lst, chunk_size):
    """Yield successive chunk_size chunks from lst."""
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]


def cdiv(a: int, b: int) -> int:
    """Ceiling division."""
    return -(a // -b)


def _generate_random_fp8(
    tensor: torch.tensor,
    low: float,
    high: float,
) -> None:
    # NOTE(zhaoyang): Due to NaN and Inf representation for fp8 data type,
    # it may occur Inf or NaN if we directly use torch.randint
    # to generate random data for fp8 data.
    # For example, s.11111.00 in fp8e5m2 format represents Inf.
    #     | E4M3        | E5M2
    #-----|-------------|-------------------
    # Inf | N/A         | s.11111.00
    # NaN | s.1111.111  | s.11111.{01,10,11}
    from vllm import _custom_ops as ops
    tensor_tmp = torch.empty_like(tensor, dtype=torch.float16)
    tensor_tmp.uniform_(low, high)
    ops.convert_fp8(tensor_tmp, tensor)
    del tensor_tmp


def get_kv_cache_torch_dtype(
        cache_dtype: Optional[Union[str, torch.dtype]],
        model_dtype: Optional[Union[str, torch.dtype]] = None) -> torch.dtype:
    if isinstance(cache_dtype, str):
        if cache_dtype == "auto":
            if isinstance(model_dtype, str):
                torch_dtype = STR_DTYPE_TO_TORCH_DTYPE[model_dtype]
            elif isinstance(model_dtype, torch.dtype):
                torch_dtype = model_dtype
            else:
                raise ValueError(f"Invalid model dtype: {model_dtype}")
        elif cache_dtype in ["half", "bfloat16", "float"]:
            torch_dtype = STR_DTYPE_TO_TORCH_DTYPE[cache_dtype]
        elif cache_dtype == "fp8":
            torch_dtype = torch.uint8
        else:
            raise ValueError(f"Invalid kv cache dtype: {cache_dtype}")
    elif isinstance(cache_dtype, torch.dtype):
        torch_dtype = cache_dtype
    else:
        raise ValueError(f"Invalid kv cache dtype: {cache_dtype}")
    return torch_dtype


def create_kv_caches_with_random_flash(
    num_blocks: int,
    block_size: int,
    num_layers: int,
    num_heads: int,
    head_size: int,
    cache_dtype: Optional[Union[str, torch.dtype]],
    model_dtype: Optional[Union[str, torch.dtype]] = None,
    seed: int = 0,
    device: Optional[str] = "cuda",
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    assert cache_dtype != "fp8"
    torch.random.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    torch_dtype = get_kv_cache_torch_dtype(cache_dtype, model_dtype)
    key_value_cache_shape = (num_blocks, 2, block_size, num_heads, head_size)
    scale = head_size**-0.5
    key_caches, value_caches = [], []
    for _ in range(num_layers):
        key_value_cache = torch.empty(size=key_value_cache_shape,
                                      dtype=torch_dtype,
                                      device=device)
        key_value_cache.uniform_(-scale, scale)
        key_caches.append(key_value_cache[:, 0])
        value_caches.append(key_value_cache[:, 1])
    return key_caches, value_caches


def create_kv_caches_with_random(
    num_blocks: int,
    block_size: int,
    num_layers: int,
    num_heads: int,
    head_size: int,
    cache_dtype: Optional[Union[str, torch.dtype]],
    model_dtype: Optional[Union[str, torch.dtype]] = None,
    seed: int = 0,
    device: Optional[str] = "cuda",
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    torch.random.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    torch_dtype = get_kv_cache_torch_dtype(cache_dtype, model_dtype)

    scale = head_size**-0.5
    x = 16 // torch.tensor([], dtype=torch_dtype).element_size()
    key_cache_shape = (num_blocks, num_heads, head_size // x, block_size, x)
    key_caches = []
    for _ in range(num_layers):
        key_cache = torch.empty(size=key_cache_shape,
                                dtype=torch_dtype,
                                device=device)
        if cache_dtype in ["auto", "half", "bfloat16", "float"]:
            key_cache.uniform_(-scale, scale)
        elif cache_dtype == 'fp8':
            _generate_random_fp8(key_cache, -scale, scale)
        else:
            raise ValueError(
                f"Does not support key cache of type {cache_dtype}")
        key_caches.append(key_cache)

    value_cache_shape = (num_blocks, num_heads, head_size, block_size)
    value_caches = []
    for _ in range(num_layers):
        value_cache = torch.empty(size=value_cache_shape,
                                  dtype=torch_dtype,
                                  device=device)
        if cache_dtype in ["auto", "half", "bfloat16", "float"]:
            value_cache.uniform_(-scale, scale)
        elif cache_dtype == 'fp8':
            _generate_random_fp8(value_cache, -scale, scale)
        else:
            raise ValueError(
                f"Does not support value cache of type {cache_dtype}")
        value_caches.append(value_cache)
    return key_caches, value_caches


@lru_cache
def print_warning_once(msg: str) -> None:
    logger.warning(msg)


@lru_cache(maxsize=None)
def is_pin_memory_available() -> bool:

    if in_wsl():
        # Pinning memory in WSL is not supported.
        # https://docs.nvidia.com/cuda/wsl-user-guide/index.html#known-limitations-for-linux-cuda-applications
        print_warning_once("Using 'pin_memory=False' as WSL is detected. "
                           "This may slow down the performance.")
        return False
    elif is_neuron():
        print_warning_once("Pin memory is not supported on Neuron.")
        return False
    elif is_cpu():
        return False
    return True


class CudaMemoryProfiler:

    def __init__(self, device=None):
        self.device = device

    def current_memory_usage(self) -> float:
        # Return the memory usage in bytes.
        torch.cuda.reset_peak_memory_stats(self.device)
        mem = torch.cuda.max_memory_allocated(self.device)
        return mem

    def __enter__(self):
        self.initial_memory = self.current_memory_usage()
        # This allows us to call methods of the context manager if needed
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.final_memory = self.current_memory_usage()
        self.consumed_memory = self.final_memory - self.initial_memory

        # Force garbage collection
        gc.collect()


def str_to_int_tuple(s: str) -> Tuple[int, ...]:
    """Convert a string to a tuple of integers."""
    try:
        return tuple(map(int, s.split(",")))
    except ValueError as e:
        raise ValueError(
            "String must be a series of integers separated by commas "
            f"(e.g., 1, 2, 3). Given input: {s}") from e


def pad_to_max_length(x: List[int], max_len: int, pad: int) -> List[int]:
    assert len(x) <= max_len
    return x + [pad] * (max_len - len(x))


def make_tensor_with_pad(
    x: List[List[int]],
    max_len: int,
    pad: int,
    dtype: torch.dtype,
    device: Optional[Union[str, torch.device]],
) -> torch.Tensor:
    """Make a padded tensor of a 2D inputs.

    The padding is applied to the end of each inner list until it reaches
    `max_len`.
    """
    padded_x = [pad_to_max_length(x_i, max_len, pad) for x_i in x]
    return torch.tensor(padded_x, dtype=dtype, device=device)


def async_tensor_h2d(
    data: list,
    dtype: torch.dtype,
    target_device: Union[str, torch.device],
    pin_memory: bool,
) -> torch.Tensor:
    """Asynchronously create a tensor and copy it from host to device."""
    t = torch.tensor(data, dtype=dtype, pin_memory=pin_memory, device="cpu")
    return t.to(device=target_device, non_blocking=True)


def maybe_expand_dim(tensor: torch.Tensor,
                     target_dims: int,
                     size: int = 1) -> torch.Tensor:
    """Expand the tensor to the target_dims."""
    if tensor.ndim < target_dims:
        tensor = tensor.view(-1, *([size] * (target_dims - tensor.ndim)))
    return tensor


def merge_dicts(dict1: Dict[Any, List[Any]],
                dict2: Dict[Any, List[Any]]) -> Dict[Any, List[Any]]:
    """Merge 2 dicts that have key -> List of items.

    When a key conflicts, the values in dict1 is prioritized.
    """
    merged_dict = defaultdict(list)

    for key, value in dict1.items():
        merged_dict[key].extend(value)

    for key, value in dict2.items():
        merged_dict[key].extend(value)

    return dict(merged_dict)


def init_cached_hf_modules():
    """
    Lazy initialization of the Hugging Face modules.
    """
    from transformers.dynamic_module_utils import init_hf_modules
    init_hf_modules()


def nccl_integrity_check(filepath):
    """
    when the library is corrupted, we cannot catch
    the exception in python. it will crash the process.
    instead, we use the exit code of `ldd` to check
    if the library is corrupted. if not, we will return
    the version of the library.
    """
    exit_code = os.system(f"ldd {filepath} 2>&1 > /dev/null")
    if exit_code != 0:
        raise RuntimeError(f"Failed to load NCCL library from {filepath} .")
    import ctypes

    nccl = ctypes.CDLL(filepath)
    version = ctypes.c_int()
    nccl.ncclGetVersion.restype = ctypes.c_int
    nccl.ncclGetVersion.argtypes = [ctypes.POINTER(ctypes.c_int)]
    result = nccl.ncclGetVersion(ctypes.byref(version))
    assert result == 0
    return version.value


@lru_cache(maxsize=None)
def find_library(lib_name: str) -> str:
    """
    Find the library file in the system.
    `lib_name` is full filename, with both prefix and suffix.
    This function resolves `lib_name` to the full path of the library.
    """
    # Adapted from https://github.com/openai/triton/blob/main/third_party/nvidia/backend/driver.py#L19 # noqa
    # According to https://en.wikipedia.org/wiki/Filesystem_Hierarchy_Standard
    # `/sbin/ldconfig` should exist in all Linux systems.
    # `/sbin/ldconfig` searches the library in the system
    libs = subprocess.check_output(["/sbin/ldconfig", "-p"]).decode()
    # each line looks like the following:
    # libcuda.so.1 (libc6,x86-64) => /lib/x86_64-linux-gnu/libcuda.so.1
    locs = [line.split()[-1] for line in libs.splitlines() if lib_name in line]
    # `LD_LIBRARY_PATH` searches the library in the user-defined paths
    env_ld_library_path = envs.LD_LIBRARY_PATH
    if not locs and env_ld_library_path:
        locs = [
            os.path.join(dir, lib_name)
            for dir in env_ld_library_path.split(":")
            if os.path.exists(os.path.join(dir, lib_name))
        ]
    if not locs:
        raise ValueError(f"Cannot find {lib_name} in the system.")
    return locs[0]


def find_nccl_library():
    so_file = envs.VLLM_NCCL_SO_PATH
    VLLM_CONFIG_ROOT = envs.VLLM_CONFIG_ROOT

    # check if we have vllm-managed nccl
    vllm_nccl_path = None
    if torch.version.cuda is not None:
        cuda_major = torch.version.cuda.split(".")[0]
        path = os.path.expanduser(
            f"{VLLM_CONFIG_ROOT}/vllm/nccl/cu{cuda_major}/libnccl.so.*")
        files = glob.glob(path)
        vllm_nccl_path = files[0] if files else None

    # manually load the nccl library
    if so_file:
        logger.info(
            "Found nccl from environment variable VLLM_NCCL_SO_PATH=%s",
            so_file)
    else:
        if torch.version.cuda is not None:
            so_file = vllm_nccl_path or find_library("libnccl.so.2")
        elif torch.version.hip is not None:
            so_file = find_library("librccl.so.1")
        else:
            raise ValueError("NCCL only supports CUDA and ROCm backends.")
        logger.info("Found nccl from library %s", so_file)
    return so_file


def enable_trace_function_call_for_thread() -> None:
    """Set up function tracing for the current thread,
    if enabled via the VLLM_TRACE_FUNCTION environment variable
    """

    if envs.VLLM_TRACE_FUNCTION:
        tmp_dir = tempfile.gettempdir()
        filename = (f"VLLM_TRACE_FUNCTION_for_process_{os.getpid()}"
                    f"_thread_{threading.get_ident()}_"
                    f"at_{datetime.datetime.now()}.log").replace(" ", "_")
        log_path = os.path.join(tmp_dir, "vllm", get_vllm_instance_id(),
                                filename)
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        enable_trace_function_call(log_path)
