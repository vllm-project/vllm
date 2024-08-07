import argparse
import asyncio
import contextlib
import datetime
import enum
import gc
import os
import socket
import subprocess
import sys
import tempfile
import threading
import uuid
import warnings
from asyncio import FIRST_COMPLETED, ensure_future
from collections import defaultdict
from functools import lru_cache, partial, wraps
from platform import uname
from typing import (Any, AsyncGenerator, Awaitable, Callable, Dict, Generic,
                    Hashable, List, Optional, OrderedDict, Set, Tuple, TypeVar,
                    Union, overload)

import numpy as np
import numpy.typing as npt
import psutil
import torch
import torch.types
from typing_extensions import ParamSpec

import vllm.envs as envs
from vllm import _custom_ops as ops
from vllm.inputs import (ExplicitEncoderDecoderPrompt, PromptInputs,
                         SingletonPromptInputs)
from vllm.logger import enable_trace_function_call, init_logger

logger = init_logger(__name__)

# Exception strings for non-implemented encoder/decoder scenarios

STR_NOT_IMPL_ENC_DEC_SWA = \
    "Sliding window attention for encoder/decoder models " + \
                    "is not currently supported."

STR_NOT_IMPL_ENC_DEC_PREFIX_CACHE = \
    "Prefix caching for encoder/decoder models " + \
                    "is not currently supported."

STR_NOT_IMPL_ENC_DEC_CHUNKED_PREFILL = \
    "Chunked prefill for encoder/decoder models " + \
                    "is not currently supported."

STR_NOT_IMPL_ENC_DEC_LOGIT_SOFTCAP = (
    "Models with logits_soft_cap "
    "require FlashInfer backend, which is "
    "currently not supported for encoder/decoder "
    "models.")

STR_NOT_IMPL_ENC_DEC_LORA = ("LoRA is currently not currently "
                             "supported with encoder/decoder "
                             "models.")

STR_NOT_IMPL_ENC_DEC_PP = ("Pipeline parallelism is not "
                           "currently supported with "
                           "encoder/decoder models.")

STR_NOT_IMPL_ENC_DEC_MM = ("Multimodal is not currently "
                           "supported with encoder/decoder "
                           "models.")

STR_NOT_IMPL_ENC_DEC_SPEC_DEC = ("Speculative decoding is not "
                                 "currently supported with encoder/"
                                 "decoder models.")

STR_NOT_IMPL_ENC_DEC_CUDAGRAPH = ("CUDAGraph is not "
                                  "currently supported with encoder/"
                                  "decoder models.")

STR_NOT_IMPL_ENC_DEC_BACKEND = ("XFormers is the only backend "
                                "currently supported with encoder/"
                                "decoder models.")

STR_NOT_IMPL_ENC_DEC_PROMPT_ADAPTER = ("Prompt adapters are not "
                                       "currently supported with encoder/"
                                       "decoder models.")

# Efficiently import all enc/dec error strings
# rather than having to import all of the above
STR_NOT_IMPL_ENC_DEC_ERR_STRS = {
    "STR_NOT_IMPL_ENC_DEC_SWA": STR_NOT_IMPL_ENC_DEC_SWA,
    "STR_NOT_IMPL_ENC_DEC_PREFIX_CACHE": STR_NOT_IMPL_ENC_DEC_PREFIX_CACHE,
    "STR_NOT_IMPL_ENC_DEC_CHUNKED_PREFILL":
    STR_NOT_IMPL_ENC_DEC_CHUNKED_PREFILL,
    "STR_NOT_IMPL_ENC_DEC_LOGIT_SOFTCAP": STR_NOT_IMPL_ENC_DEC_LOGIT_SOFTCAP,
    "STR_NOT_IMPL_ENC_DEC_LORA": STR_NOT_IMPL_ENC_DEC_LORA,
    "STR_NOT_IMPL_ENC_DEC_PP": STR_NOT_IMPL_ENC_DEC_PP,
    "STR_NOT_IMPL_ENC_DEC_MM": STR_NOT_IMPL_ENC_DEC_MM,
    "STR_NOT_IMPL_ENC_DEC_SPEC_DEC": STR_NOT_IMPL_ENC_DEC_SPEC_DEC,
    "STR_NOT_IMPL_ENC_DEC_CUDA_GRAPH": STR_NOT_IMPL_ENC_DEC_CUDAGRAPH,
    "STR_NOT_IMPL_ENC_DEC_BACKEND": STR_NOT_IMPL_ENC_DEC_BACKEND,
    "STR_NOT_IMPL_ENC_DEC_PROMPT_ADAPTER": STR_NOT_IMPL_ENC_DEC_PROMPT_ADAPTER,
}

# Constants related to forcing the attention backend selection

# String name of register which may be set in order to
# force auto-selection of attention backend by Attention
# wrapper
STR_BACKEND_ENV_VAR: str = "VLLM_ATTENTION_BACKEND"

# Possible string values of STR_BACKEND_ENV_VAR
# register, corresponding to possible backends
STR_FLASHINFER_ATTN_VAL: str = "FLASHINFER"
STR_TORCH_SDPA_ATTN_VAL: str = "TORCH_SDPA"
STR_ROCM_FLASH_ATTN_VAL: str = "ROCM_FLASH"
STR_XFORMERS_ATTN_VAL: str = "XFORMERS"
STR_FLASH_ATTN_VAL: str = "FLASH_ATTN"
STR_INVALID_VAL: str = "INVALID"

STR_DTYPE_TO_TORCH_DTYPE = {
    "half": torch.half,
    "bfloat16": torch.bfloat16,
    "float": torch.float,
    "fp8": torch.uint8,
    "fp8_e4m3": torch.uint8,
    "fp8_e5m2": torch.uint8,
}

TORCH_DTYPE_TO_NUMPY_DTYPE = {
    torch.float16: np.float16,
    torch.float32: np.float32,
    torch.float64: np.float64,
    torch.uint8: np.uint8,
    torch.int32: np.int32,
    torch.int64: np.int64,
}

P = ParamSpec('P')
K = TypeVar("K")
T = TypeVar("T")
U = TypeVar("U")


class _Sentinel:
    ...


ALL_PINNED_SENTINEL = _Sentinel()


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
        self.pinned_items: Set[Hashable] = set()
        self.capacity = capacity

    def __contains__(self, key: Hashable) -> bool:
        return key in self.cache

    def __len__(self) -> int:
        return len(self.cache)

    def __getitem__(self, key: Hashable) -> T:
        value = self.cache[key]  # Raise KeyError if not exists
        self.cache.move_to_end(key)
        return value

    def __setitem__(self, key: Hashable, value: T) -> None:
        self.put(key, value)

    def __delitem__(self, key: Hashable) -> None:
        self.pop(key)

    def touch(self, key: Hashable) -> None:
        self.cache.move_to_end(key)

    def get(self,
            key: Hashable,
            default_value: Optional[T] = None) -> Optional[T]:
        value: Optional[T]
        if key in self.cache:
            value = self.cache[key]
            self.cache.move_to_end(key)
        else:
            value = default_value
        return value

    def put(self, key: Hashable, value: T) -> None:
        self.cache[key] = value
        self.cache.move_to_end(key)
        self._remove_old_if_needed()

    def pin(self, key: Hashable) -> None:
        """
        Pins a key in the cache preventing it from being
        evicted in the LRU order.
        """
        if key not in self.cache:
            raise ValueError(f"Cannot pin key: {key} not in cache.")
        self.pinned_items.add(key)

    def _unpin(self, key: Hashable) -> None:
        self.pinned_items.remove(key)

    def _on_remove(self, key: Hashable, value: Optional[T]):
        pass

    def remove_oldest(self, remove_pinned=False):
        if not self.cache:
            return

        if not remove_pinned:
            # pop the oldest item in the cache that is not pinned
            lru_key = next(
                (key for key in self.cache if key not in self.pinned_items),
                ALL_PINNED_SENTINEL)
            if lru_key is ALL_PINNED_SENTINEL:
                raise RuntimeError("All items are pinned, "
                                   "cannot remove oldest from the cache.")
        else:
            lru_key = next(iter(self.cache))
        self.pop(lru_key)

    def _remove_old_if_needed(self) -> None:
        while len(self.cache) > self.capacity:
            self.remove_oldest()

    def pop(self,
            key: Hashable,
            default_value: Optional[T] = None) -> Optional[T]:
        run_on_remove = key in self.cache
        value: Optional[T] = self.cache.pop(key, default_value)
        # remove from pinned items
        if key in self.pinned_items:
            self._unpin(key)
        if run_on_remove:
            self._on_remove(key, value)
        return value

    def clear(self):
        while len(self.cache) > 0:
            self.remove_oldest(remove_pinned=True)
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
def is_openvino() -> bool:
    from importlib.metadata import PackageNotFoundError, version
    try:
        return "openvino" in version("vllm")
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
def is_tpu() -> bool:
    try:
        import libtpu
    except ImportError:
        libtpu = None
    return libtpu is not None


@lru_cache(maxsize=None)
def is_xpu() -> bool:
    from importlib.metadata import version
    is_xpu_flag = "xpu" in version("vllm")
    # vllm is not build with xpu
    if not is_xpu_flag:
        return False
    try:
        import intel_extension_for_pytorch as ipex  # noqa: F401
        _import_ipex = True
    except ImportError as e:
        logger.warning("Import Error for IPEX: %s", e.msg)
        _import_ipex = False
    # ipex dependency is not ready
    if not _import_ipex:
        logger.warning("not found ipex lib")
        return False
    return hasattr(torch, "xpu") and torch.xpu.is_available()


@lru_cache(maxsize=None)
def get_max_shared_memory_bytes(gpu: int = 0) -> int:
    """Returns the maximum shared memory per thread block in bytes."""
    max_shared_mem = (
        ops.get_max_shared_memory_per_block_device_attribute(gpu))
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
def get_vllm_instance_id() -> str:
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


def make_async(func: Callable[P, T]) -> Callable[P, Awaitable[T]]:
    """Take a blocking function, and run it on in an executor thread.

    This function prevents the blocking function from blocking the
    asyncio event loop.
    The code in this function needs to be thread safe.
    """

    def _async_wrapper(*args: P.args, **kwargs: P.kwargs) -> asyncio.Future:
        loop = asyncio.get_event_loop()
        p_func = partial(func, *args, **kwargs)
        return loop.run_in_executor(executor=None, func=p_func)

    return _async_wrapper


async def iterate_with_cancellation(
    iterator: AsyncGenerator[T, None],
    is_cancelled: Callable[[], Awaitable[bool]],
) -> AsyncGenerator[T, None]:
    """Convert async iterator into one that polls the provided function
    at least once per second to check for client cancellation.
    """

    # Can use anext() in python >= 3.10
    awaits = [ensure_future(iterator.__anext__())]
    while True:
        done, pending = await asyncio.wait(awaits, timeout=1)
        if await is_cancelled():
            with contextlib.suppress(BaseException):
                awaits[0].cancel()
                await iterator.aclose()
            raise asyncio.CancelledError("client cancelled")
        if done:
            try:
                item = await awaits[0]
                awaits[0] = ensure_future(iterator.__anext__())
                yield item
            except StopAsyncIteration:
                # we are done
                return


async def merge_async_iterators(
    *iterators: AsyncGenerator[T, None],
    is_cancelled: Callable[[], Awaitable[bool]],
) -> AsyncGenerator[Tuple[int, T], None]:
    """Merge multiple asynchronous iterators into a single iterator.

    This method handle the case where some iterators finish before others.
    When it yields, it yields a tuple (i, item) where i is the index of the
    iterator that yields the item.

    It also polls the provided function at least once per second to check
    for client cancellation.
    """

    # Can use anext() in python >= 3.10
    awaits = {
        ensure_future(pair[1].__anext__()): pair
        for pair in enumerate(iterators)
    }
    try:
        while awaits:
            done, pending = await asyncio.wait(awaits.keys(),
                                               return_when=FIRST_COMPLETED,
                                               timeout=1)
            if await is_cancelled():
                raise asyncio.CancelledError("client cancelled")
            for d in done:
                pair = awaits.pop(d)
                try:
                    item = await d
                    i, it = pair
                    awaits[ensure_future(it.__anext__())] = pair
                    yield i, item
                except StopAsyncIteration:
                    pass
    finally:
        # Cancel any remaining iterators
        for f, (_, it) in awaits.items():
            with contextlib.suppress(BaseException):
                f.cancel()
                await it.aclose()


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


def get_open_port(port: Optional[int] = None) -> int:
    if port is None:
        # Default behavior here is to return a port for multi-gpu communication
        port = envs.VLLM_PORT
    if port is not None:
        while True:
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.bind(("", port))
                    return port
            except OSError:
                port += 1  # Increment port number if already in use
                logger.info("Port %d is already in use, trying port %d",
                            port - 1, port)
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


def chunk_list(lst: List[T], chunk_size: int):
    """Yield successive chunk_size chunks from lst."""
    for i in range(0, len(lst), chunk_size):
        yield lst[i:i + chunk_size]


def cdiv(a: int, b: int) -> int:
    """Ceiling division."""
    return -(a // -b)


def _generate_random_fp8(
    tensor: torch.Tensor,
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
    ops.convert_fp8(tensor, tensor_tmp)
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
    torch.random.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    torch_dtype = get_kv_cache_torch_dtype(cache_dtype, model_dtype)
    key_value_cache_shape = (num_blocks, 2, block_size, num_heads, head_size)
    scale = head_size**-0.5

    key_caches: List[torch.Tensor] = []
    value_caches: List[torch.Tensor] = []

    for _ in range(num_layers):
        key_value_cache = torch.empty(size=key_value_cache_shape,
                                      dtype=torch_dtype,
                                      device=device)
        if cache_dtype in ["auto", "half", "bfloat16", "float"]:
            key_value_cache.uniform_(-scale, scale)
        elif cache_dtype == 'fp8':
            _generate_random_fp8(key_value_cache, -scale, scale)
        else:
            raise ValueError(
                f"Does not support key cache of type {cache_dtype}")
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

    if cache_dtype == "fp8" and head_size % 16:
        raise ValueError(
            f"Does not support key cache of type fp8 with head_size {head_size}"
        )

    torch.random.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    torch_dtype = get_kv_cache_torch_dtype(cache_dtype, model_dtype)

    scale = head_size**-0.5
    x = 16 // torch.tensor([], dtype=torch_dtype).element_size()
    key_cache_shape = (num_blocks, num_heads, head_size // x, block_size, x)
    key_caches: List[torch.Tensor] = []
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
    value_caches: List[torch.Tensor] = []
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
    elif is_xpu():
        print_warning_once("Pin memory is not supported on XPU.")
        return False
    elif is_neuron():
        print_warning_once("Pin memory is not supported on Neuron.")
        return False
    elif is_cpu() or is_openvino():
        return False
    return True


class CudaMemoryProfiler:

    def __init__(self, device: Optional[torch.types.Device] = None):
        self.device = device

    def current_memory_usage(self) -> float:
        # Return the memory usage in bytes.
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(self.device)
            mem = torch.cuda.max_memory_allocated(self.device)
        elif is_xpu():
            torch.xpu.reset_peak_memory_stats(self.device)  # type: ignore
            mem = torch.xpu.max_memory_allocated(self.device)  # type: ignore
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


def make_ndarray_with_pad(
    x: List[List[T]],
    pad: T,
    dtype: npt.DTypeLike,
    *,
    max_len: Optional[int] = None,
) -> npt.NDArray:
    """
    Make a padded array from 2D inputs.

    The padding is applied to the end of each inner list until it reaches
    `max_len`.
    """
    if max_len is None:
        # Unlike for most functions, map is faster than a genexpr over `len`
        max_len = max(map(len, x), default=0)

    padded_x = np.full((len(x), max_len), pad, dtype=dtype)
    for ind, blocktb in enumerate(x):
        assert len(blocktb) <= max_len
        padded_x[ind, :len(blocktb)] = blocktb

    return padded_x


def make_tensor_with_pad(
    x: List[List[T]],
    pad: T,
    dtype: torch.dtype,
    *,
    max_len: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    pin_memory: bool = False,
) -> torch.Tensor:
    """
    Make a padded tensor from 2D inputs.

    The padding is applied to the end of each inner list until it reaches
    `max_len`.
    """
    np_dtype = TORCH_DTYPE_TO_NUMPY_DTYPE[dtype]
    padded_x = make_ndarray_with_pad(x, pad, np_dtype, max_len=max_len)

    tensor = torch.from_numpy(padded_x).to(device)
    if pin_memory:
        tensor = tensor.pin_memory()

    return tensor


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


def get_dtype_size(dtype: torch.dtype) -> int:
    """Get the size of the data type in bytes."""
    return torch.tensor([], dtype=dtype).element_size()


def merge_dicts(dict1: Dict[K, List[T]],
                dict2: Dict[K, List[T]]) -> Dict[K, List[T]]:
    """Merge 2 dicts that have key -> List of items.

    When a key conflicts, the values in dict1 is prioritized.
    """
    merged_dict: Dict[K, List[T]] = defaultdict(list)

    for key, value in dict1.items():
        merged_dict[key].extend(value)

    for key, value in dict2.items():
        merged_dict[key].extend(value)

    return dict(merged_dict)


JSONTree = Union[Dict[str, "JSONTree[T]"], List["JSONTree[T]"],
                 Tuple["JSONTree[T]", ...], T]
"""A nested JSON structure where the leaves need not be JSON-serializable."""


@overload
def json_map_leaves(
    func: Callable[[T], U],
    value: Dict[str, JSONTree[T]],
) -> Dict[str, JSONTree[U]]:
    ...


@overload
def json_map_leaves(
    func: Callable[[T], U],
    value: List[JSONTree[T]],
) -> List[JSONTree[U]]:
    ...


@overload
def json_map_leaves(
    func: Callable[[T], U],
    value: Tuple[JSONTree[T], ...],
) -> Tuple[JSONTree[U], ...]:
    ...


@overload
def json_map_leaves(
    func: Callable[[T], U],
    value: JSONTree[T],
) -> JSONTree[U]:
    ...


def json_map_leaves(func: Callable[[T], U], value: JSONTree[T]) -> JSONTree[U]:
    if isinstance(value, dict):
        return {k: json_map_leaves(func, v) for k, v in value.items()}
    elif isinstance(value, list):
        return [json_map_leaves(func, v) for v in value]
    elif isinstance(value, tuple):
        return tuple(json_map_leaves(func, v) for v in value)
    else:
        return func(value)


def flatten_2d_lists(lists: List[List[T]]) -> List[T]:
    """Flatten a list of lists to a single list."""
    return [item for sublist in lists for item in sublist]


def init_cached_hf_modules() -> None:
    """
    Lazy initialization of the Hugging Face modules.
    """
    from transformers.dynamic_module_utils import init_hf_modules
    init_hf_modules()


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


def find_nccl_library() -> str:
    """
    We either use the library file specified by the `VLLM_NCCL_SO_PATH`
    environment variable, or we find the library file brought by PyTorch.
    After importing `torch`, `libnccl.so.2` or `librccl.so.1` can be
    found by `ctypes` automatically.
    """
    so_file = envs.VLLM_NCCL_SO_PATH

    # manually load the nccl library
    if so_file:
        logger.info(
            "Found nccl from environment variable VLLM_NCCL_SO_PATH=%s",
            so_file)
    else:
        if torch.version.cuda is not None:
            so_file = "libnccl.so.2"
        elif torch.version.hip is not None:
            so_file = "librccl.so.1"
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


def identity(value: T) -> T:
    return value


F = TypeVar('F', bound=Callable[..., Any])


def deprecate_kwargs(
        *kws: str,
        is_deprecated: Union[bool, Callable[[], bool]] = True,
        additional_message: Optional[str] = None) -> Callable[[F], F]:
    deprecated_kws = set(kws)

    if not callable(is_deprecated):
        is_deprecated = partial(identity, is_deprecated)

    def wrapper(fn: F) -> F:

        @wraps(fn)
        def inner(*args, **kwargs):
            if is_deprecated():
                deprecated_kwargs = kwargs.keys() & deprecated_kws
                if deprecated_kwargs:
                    msg = (
                        f"The keyword arguments {deprecated_kwargs} are "
                        "deprecated and will be removed in a future update.")
                    if additional_message is not None:
                        msg += f" {additional_message}"

                    warnings.warn(
                        DeprecationWarning(msg),
                        stacklevel=3,  # The inner function takes up one level
                    )

            return fn(*args, **kwargs)

        return inner  # type: ignore

    return wrapper


@lru_cache(maxsize=8)
def _cuda_device_count_stateless(
        cuda_visible_devices: Optional[str] = None) -> int:
    # Note: cuda_visible_devices is not used, but we keep it as an argument for
    # LRU Cache purposes.

    # Code below is based on
    # https://github.com/pytorch/pytorch/blob/
    # c1cd946818442aca8c7f812b16d187ce1586c3bc/
    # torch/cuda/__init__.py#L831C1-L831C17
    import torch.cuda
    import torch.version

    if not torch.cuda._is_compiled():
        return 0
    if is_hip():
        # ROCm uses amdsmi instead of nvml for stateless device count
        # This requires a sufficiently modern version of Torch 2.4.0
        raw_count = torch.cuda._device_count_amdsmi() if (hasattr(
            torch.cuda, "_device_count_amdsmi")) else -1
    else:
        raw_count = torch.cuda._device_count_nvml()
    r = torch._C._cuda_getDeviceCount() if raw_count < 0 else raw_count
    return r


def cuda_device_count_stateless() -> int:
    """Get number of CUDA devices, caching based on the value of
    CUDA_VISIBLE_DEVICES at the time of call.
    
    This should be used instead of torch.cuda.device_count()
    unless CUDA_VISIBLE_DEVICES has already been set to the desired
    value."""

    # This can be removed and simply replaced with torch.cuda.get_device_count
    # after https://github.com/pytorch/pytorch/pull/122815 is released.
    return _cuda_device_count_stateless(envs.CUDA_VISIBLE_DEVICES)


# NVML utils
# Note that NVML is not affected by `CUDA_VISIBLE_DEVICES`,
# all the related functions work on real physical device ids.
# the major benefit of using NVML is that it will not initialize CUDA

try:
    import pynvml
except ImportError:
    # For non-NV devices
    pynvml = None


def with_nvml_context(fn):

    @wraps(fn)
    def wrapper(*args, **kwargs):
        if pynvml is not None:
            pynvml.nvmlInit()
        try:
            return fn(*args, **kwargs)
        finally:
            if pynvml is not None:
                pynvml.nvmlShutdown()

    return wrapper


@with_nvml_context
def is_full_nvlink(device_ids: List[int]) -> bool:
    """
    query if the set of gpus are fully connected by nvlink (1 hop)
    """
    handles = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in device_ids]
    for i, handle in enumerate(handles):
        for j, peer_handle in enumerate(handles):
            if i < j:
                try:
                    p2p_status = pynvml.nvmlDeviceGetP2PStatus(
                        handle, peer_handle, pynvml.NVML_P2P_CAPS_INDEX_NVLINK)
                    if p2p_status != pynvml.NVML_P2P_STATUS_OK:
                        return False
                except pynvml.NVMLError as error:
                    logger.error(
                        "NVLink detection failed. This is normal if your"
                        " machine has no NVLink equipped.",
                        exc_info=error)
                    return False
    return True


#From: https://stackoverflow.com/a/4104188/2749989
def run_once(f):

    def wrapper(*args, **kwargs) -> Any:
        if not wrapper.has_run:  # type: ignore[attr-defined]
            wrapper.has_run = True  # type: ignore[attr-defined]
            return f(*args, **kwargs)

    wrapper.has_run = False  # type: ignore[attr-defined]
    return wrapper


class FlexibleArgumentParser(argparse.ArgumentParser):
    """ArgumentParser that allows both underscore and dash in names."""

    def parse_args(self, args=None, namespace=None):
        if args is None:
            args = sys.argv[1:]

        # Convert underscores to dashes and vice versa in argument names
        processed_args = []
        for arg in args:
            if arg.startswith('--'):
                if '=' in arg:
                    key, value = arg.split('=', 1)
                    key = '--' + key[len('--'):].replace('_', '-')
                    processed_args.append(f'{key}={value}')
                else:
                    processed_args.append('--' +
                                          arg[len('--'):].replace('_', '-'))
            else:
                processed_args.append(arg)

        return super().parse_args(processed_args, namespace)


async def _run_task_with_lock(task: Callable, lock: asyncio.Lock, *args,
                              **kwargs):
    """Utility function to run async task in a lock"""
    async with lock:
        return await task(*args, **kwargs)


def is_encoder_decoder_model_config(model_config) -> bool:
    '''
    Extract the HF encoder/decoder model flag from the ModelConfig instance.
    Return False if model_config is None.
    '''
    return model_config is not None and \
                getattr(model_config.hf_config,
                        "is_encoder_decoder",
                        False)


def is_embedding_model_config(model_config) -> bool:
    '''
    Extract the embedding model flag from the ModelConfig instance.
    Return False if model_config is None.
    '''
    return model_config is not None and \
                model_config.embedding_mode


def build_explicit_enc_dec_prompt(
    encoder_prompt: SingletonPromptInputs,
    decoder_prompt: SingletonPromptInputs,
) -> ExplicitEncoderDecoderPrompt:
    return ExplicitEncoderDecoderPrompt(encoder_prompt=encoder_prompt,
                                        decoder_prompt=decoder_prompt)


def zip_enc_dec_prompt_lists(
    enc_prompt_list: List[SingletonPromptInputs],
    dec_prompt_list: List[SingletonPromptInputs],
) -> List[ExplicitEncoderDecoderPrompt]:
    return [
        build_explicit_enc_dec_prompt(encoder_prompt, decoder_prompt)
        for (encoder_prompt,
             decoder_prompt) in zip(enc_prompt_list, dec_prompt_list)
    ]


def to_enc_dec_tuple_list(
    enc_dec_prompts: List[ExplicitEncoderDecoderPrompt],
) -> List[Tuple[PromptInputs, PromptInputs]]:
    return [(enc_dec_prompt['encoder_prompt'],
             enc_dec_prompt['decoder_prompt'])
            for enc_dec_prompt in enc_dec_prompts]
