import enum
import os
import socket
import subprocess
import uuid
from platform import uname
from typing import List, Tuple, Union
from packaging.version import parse, Version

import psutil
import torch
import asyncio
from functools import partial
from typing import (
    Awaitable,
    Callable,
    TypeVar,
)
from collections import OrderedDict
from typing import Any, Hashable, Optional

from vllm.logger import init_logger

T = TypeVar("T")
logger = init_logger(__name__)

STR_DTYPE_TO_TORCH_DTYPE = {
    "half": torch.half,
    "bfloat16": torch.bfloat16,
    "float": torch.float,
    "fp8_e5m2": torch.uint8,
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


class LRUCache:

    def __init__(self, capacity: int):
        self.cache = OrderedDict()
        self.capacity = capacity

    def __contains__(self, key: Hashable) -> bool:
        return key in self.cache

    def __len__(self) -> int:
        return len(self.cache)

    def __getitem__(self, key: Hashable) -> Any:
        return self.get(key)

    def __setitem__(self, key: Hashable, value: Any) -> None:
        self.put(key, value)

    def __delitem__(self, key: Hashable) -> None:
        self.pop(key)

    def touch(self, key: Hashable) -> None:
        self.cache.move_to_end(key)

    def get(self, key: Hashable, default_value: Optional[Any] = None) -> int:
        if key in self.cache:
            value = self.cache[key]
            self.cache.move_to_end(key)
        else:
            value = default_value
        return value

    def put(self, key: Hashable, value: Any) -> None:
        self.cache[key] = value
        self.cache.move_to_end(key)
        self._remove_old_if_needed()

    def _on_remove(self, key: Hashable, value: Any):
        pass

    def remove_oldest(self):
        if not self.cache:
            return
        key, value = self.cache.popitem(last=False)
        self._on_remove(key, value)

    def _remove_old_if_needed(self) -> None:
        while len(self.cache) > self.capacity:
            self.remove_oldest()

    def pop(self, key: int, default_value: Optional[Any] = None) -> Any:
        run_on_remove = key in self.cache
        value = self.cache.pop(key, default_value)
        if run_on_remove:
            self._on_remove(key, value)
        return value

    def clear(self):
        while len(self.cache) > 0:
            self.remove_oldest()
        self.cache.clear()


def is_hip() -> bool:
    return torch.version.hip is not None


def is_neuron() -> bool:
    try:
        import transformers_neuronx
    except ImportError:
        transformers_neuronx = None
    return transformers_neuronx is not None


def get_max_shared_memory_bytes(gpu: int = 0) -> int:
    """Returns the maximum shared memory per thread block in bytes."""
    # NOTE: This import statement should be executed lazily since
    # the Neuron-X backend does not have the `cuda_utils` module.
    from vllm._C import cuda_utils

    max_shared_mem = cuda_utils.get_max_shared_memory_per_block_device_attribute(
        gpu)
    # value 0 will cause MAX_SEQ_LEN become negative and test_attention.py will fail
    assert max_shared_mem > 0, "max_shared_mem can not be zero"
    return int(max_shared_mem)


def get_cpu_memory() -> int:
    """Returns the total CPU memory of the node in bytes."""
    return psutil.virtual_memory().total


def random_uuid() -> str:
    return str(uuid.uuid4().hex)


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


def get_ip() -> str:
    # try ipv4
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("dns.google", 80))  # Doesn't need to be reachable
        return s.getsockname()[0]
    except OSError:
        # try ipv6
        s = socket.socket(socket.AF_INET6, socket.SOCK_DGRAM)
        s.connect(("dns.google", 80))
        return s.getsockname()[0]


def get_distributed_init_method(ip: str, port: int) -> str:
    return f"tcp://{ip}:{port}"


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


def set_cuda_visible_devices(device_ids: List[int]) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, device_ids))


def get_nvcc_cuda_version() -> Optional[Version]:
    cuda_home = os.environ.get('CUDA_HOME')
    if not cuda_home:
        cuda_home = '/usr/local/cuda'
        if os.path.isfile(cuda_home + '/bin/nvcc'):
            logger.info(
                f'CUDA_HOME is not found in the environment. Using {cuda_home} as CUDA_HOME.'
            )
        else:
            logger.warning(
                f'Not found nvcc in {cuda_home}. Skip cuda version check!')
            return None
    nvcc_output = subprocess.check_output([cuda_home + "/bin/nvcc", "-V"],
                                          universal_newlines=True)
    output = nvcc_output.split()
    release_idx = output.index("release") + 1
    nvcc_cuda_version = parse(output[release_idx].split(",")[0])
    return nvcc_cuda_version


def _generate_random_fp8_e5m2(
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
    from vllm._C import cache_ops
    tensor_tmp = torch.empty_like(tensor, dtype=torch.float16)
    tensor_tmp.uniform_(low, high)
    cache_ops.convert_fp8_e5m2(tensor_tmp, tensor)
    del tensor_tmp


def create_kv_caches_with_random(
    num_blocks: int,
    block_size: int,
    num_layers: int,
    num_heads: int,
    head_size: int,
    cache_dtype: Optional[Union[str, torch.dtype]],
    model_dtype: Optional[Union[str, torch.dtype]] = None,
    seed: Optional[int] = 0,
    device: Optional[str] = "cuda",
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    torch.random.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

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
        elif cache_dtype == "fp8_e5m2":
            torch_dtype = torch.uint8
        else:
            raise ValueError(f"Invalid kv cache dtype: {cache_dtype}")
    elif isinstance(cache_dtype, torch.dtype):
        torch_dtype = cache_dtype
    else:
        raise ValueError(f"Invalid kv cache dtype: {cache_dtype}")

    scale = head_size**-0.5
    x = 16 // torch.tensor([], dtype=torch_dtype).element_size()
    key_cache_shape = (num_blocks, num_heads, head_size // x, block_size, x)
    key_caches = []
    for _ in range(num_layers):
        key_cache = torch.empty(size=key_cache_shape,
                                dtype=torch_dtype,
                                device=device)
        if cache_dtype == 'fp8_e5m2':
            _generate_random_fp8_e5m2(key_cache, -scale, scale)
        elif torch_dtype in [torch.half, torch.bfloat16, torch.float]:
            key_cache.uniform_(-scale, scale)
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
        if cache_dtype == 'fp8_e5m2':
            _generate_random_fp8_e5m2(value_cache, -scale, scale)
        elif torch_dtype in [torch.half, torch.bfloat16, torch.float]:
            value_cache.uniform_(-scale, scale)
        else:
            raise ValueError(
                f"Does not support value cache of type {cache_dtype}")
        value_caches.append(value_cache)
    return key_caches, value_caches
