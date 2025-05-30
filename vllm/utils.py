# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import concurrent
import contextlib
import datetime
import enum
import gc
import getpass
import hashlib
import importlib
import importlib.metadata
import importlib.util
import inspect
import ipaddress
import json
import multiprocessing
import os
import pickle
import signal
import socket
import subprocess
import sys
import tempfile
import textwrap
import threading
import time
import traceback
import types
import uuid
import warnings
import weakref
from argparse import (Action, ArgumentDefaultsHelpFormatter, ArgumentParser,
                      ArgumentTypeError, RawDescriptionHelpFormatter,
                      _ArgumentGroup)
from asyncio import FIRST_COMPLETED, AbstractEventLoop, Task
from collections import UserDict, defaultdict
from collections.abc import (AsyncGenerator, Awaitable, Generator, Hashable,
                             Iterable, Iterator, KeysView, Mapping)
from concurrent.futures.process import ProcessPoolExecutor
from dataclasses import dataclass, field
from functools import cache, lru_cache, partial, wraps
from types import MappingProxyType
from typing import (TYPE_CHECKING, Any, Callable, Generic, Literal, NamedTuple,
                    Optional, Sequence, Tuple, Type, TypeVar, Union, cast,
                    overload)
from urllib.parse import urlparse
from uuid import uuid4

import cachetools
import cloudpickle
import numpy as np
import numpy.typing as npt
import psutil
import regex as re
import torch
import torch.types
import yaml
import zmq
import zmq.asyncio
from packaging import version
from packaging.version import Version
from torch.library import Library
from typing_extensions import Never, ParamSpec, TypeIs, assert_never

import vllm.envs as envs
# NOTE: import triton_utils to make TritonPlaceholderModule work
#       if triton is unavailable
import vllm.triton_utils  # noqa: F401
from vllm.logger import enable_trace_function_call, init_logger

if TYPE_CHECKING:
    from argparse import Namespace

    from vllm.config import ModelConfig, VllmConfig

logger = init_logger(__name__)

# This value is chosen to have a balance between ITL and TTFT. Note it is
# not optimized for throughput.
DEFAULT_MAX_NUM_BATCHED_TOKENS = 2048
POOLING_MODEL_MAX_NUM_BATCHED_TOKENS = 32768
MULTIMODAL_MODEL_MAX_NUM_BATCHED_TOKENS = 5120

# Exception strings for non-implemented encoder/decoder scenarios

# Reminder: Please update docs/features/compatibility_matrix.md
# If the feature combo become valid

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

STR_NOT_IMPL_ENC_DEC_LORA = ("LoRA is not currently "
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

STR_NOT_IMPL_ENC_DEC_BACKEND = ("XFormers and Flash-Attention are the only "
                                "backends currently supported with encoder/"
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
STR_DUAL_CHUNK_FLASH_ATTN_VAL: str = "DUAL_CHUNK_FLASH_ATTN"
STR_INVALID_VAL: str = "INVALID"

GB_bytes = 1_000_000_000
"""The number of bytes in one gigabyte (GB)."""

GiB_bytes = 1 << 30
"""The number of bytes in one gibibyte (GiB)."""

STR_DTYPE_TO_TORCH_DTYPE = {
    "half": torch.half,
    "bfloat16": torch.bfloat16,
    "float": torch.float,
    "fp8": torch.uint8,
    "fp8_e4m3": torch.uint8,
    "fp8_e5m2": torch.uint8,
    "int8": torch.int8,
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
T = TypeVar("T")
U = TypeVar("U")

_K = TypeVar("_K", bound=Hashable)
_V = TypeVar("_V")
_T = TypeVar("_T")


class _Sentinel:
    ...


ALL_PINNED_SENTINEL = _Sentinel()


class Device(enum.Enum):
    GPU = enum.auto()
    CPU = enum.auto()


class LayerBlockType(enum.Enum):
    attention = "attention"
    mamba = "mamba"


class Counter:

    def __init__(self, start: int = 0) -> None:
        self.counter = start

    def __next__(self) -> int:
        i = self.counter
        self.counter += 1
        return i

    def reset(self) -> None:
        self.counter = 0


class _MappingOrderCacheView(UserDict[_K, _V]):

    def __init__(self, data: Mapping[_K, _V], ordered_keys: Mapping[_K, None]):
        super().__init__(data)
        self.ordered_keys = ordered_keys

    def __iter__(self) -> Iterator[_K]:
        return iter(self.ordered_keys)

    def keys(self) -> KeysView[_K]:
        return KeysView(self.ordered_keys)


class CacheInfo(NamedTuple):
    hits: int
    total: int

    @property
    def hit_ratio(self) -> float:
        if self.total == 0:
            return 0

        return self.hits / self.total

    def __sub__(self, other: CacheInfo):
        return CacheInfo(
            hits=self.hits - other.hits,
            total=self.total - other.total,
        )


class LRUCache(cachetools.LRUCache[_K, _V], Generic[_K, _V]):

    def __init__(self,
                 capacity: float,
                 getsizeof: Optional[Callable[[_V], float]] = None):
        super().__init__(capacity, getsizeof)

        self.pinned_items = set[_K]()

        self._hits = 0
        self._total = 0
        self._last_info = CacheInfo(hits=0, total=0)

    def __getitem__(self, key: _K, *, update_info: bool = True) -> _V:
        value = super().__getitem__(key)

        if update_info:
            self._hits += 1
            self._total += 1

        return value

    def __delitem__(self, key: _K) -> None:
        run_on_remove = key in self
        value = self.__getitem__(key,
                                 update_info=False)  # type: ignore[call-arg]
        super().__delitem__(key)
        if key in self.pinned_items:
            # Todo: add warning to inform that del pinned item
            self._unpin(key)
        if run_on_remove:
            self._on_remove(key, value)

    @property
    def cache(self) -> Mapping[_K, _V]:
        """Return the internal cache dictionary in order (read-only)."""
        return _MappingOrderCacheView(
            self._Cache__data,  # type: ignore
            self.order)

    @property
    def order(self) -> Mapping[_K, None]:
        """Return the internal order dictionary (read-only)."""
        return MappingProxyType(self._LRUCache__order)  # type: ignore

    @property
    def capacity(self) -> float:
        return self.maxsize

    @property
    def usage(self) -> float:
        if self.maxsize == 0:
            return 0

        return self.currsize / self.maxsize

    def stat(self, *, delta: bool = False) -> CacheInfo:
        """
        Gets the cumulative number of hits and queries against this cache.

        If `delta=True`, instead gets these statistics
        since the last call that also passed `delta=True`.
        """
        info = CacheInfo(hits=self._hits, total=self._total)

        if delta:
            info_delta = info - self._last_info
            self._last_info = info
            info = info_delta

        return info

    def touch(self, key: _K) -> None:
        try:
            self._LRUCache__order.move_to_end(key)  # type: ignore
        except KeyError:
            self._LRUCache__order[key] = None  # type: ignore

    @overload
    def get(self, key: _K, /) -> Optional[_V]:
        ...

    @overload
    def get(self, key: _K, /, default: Union[_V, _T]) -> Union[_V, _T]:
        ...

    def get(self,
            key: _K,
            /,
            default: Optional[Union[_V,
                                    _T]] = None) -> Optional[Union[_V, _T]]:
        value: Optional[Union[_V, _T]]
        if key in self:
            value = self.__getitem__(
                key, update_info=False)  # type: ignore[call-arg]

            self._hits += 1
        else:
            value = default

        self._total += 1
        return value

    @overload
    def pop(self, key: _K) -> _V:
        ...

    @overload
    def pop(self, key: _K, default: Union[_V, _T]) -> Union[_V, _T]:
        ...

    def pop(self,
            key: _K,
            default: Optional[Union[_V,
                                    _T]] = None) -> Optional[Union[_V, _T]]:
        value: Optional[Union[_V, _T]]
        if key not in self:
            return default

        value = self.__getitem__(key,
                                 update_info=False)  # type: ignore[call-arg]
        self.__delitem__(key)
        return value

    def put(self, key: _K, value: _V) -> None:
        self.__setitem__(key, value)

    def pin(self, key: _K) -> None:
        """
        Pins a key in the cache preventing it from being
        evicted in the LRU order.
        """
        if key not in self:
            raise ValueError(f"Cannot pin key: {key} not in cache.")
        self.pinned_items.add(key)

    def _unpin(self, key: _K) -> None:
        """
        Unpins a key in the cache allowing it to be
        evicted in the LRU order.
        """
        self.pinned_items.remove(key)

    def _on_remove(self, key: _K, value: Optional[_V]) -> None:
        pass

    def remove_oldest(self, *, remove_pinned: bool = False) -> None:
        if len(self) == 0:
            return

        self.popitem(remove_pinned=remove_pinned)

    def _remove_old_if_needed(self) -> None:
        while self.currsize > self.capacity:
            self.remove_oldest()

    def popitem(self, remove_pinned: bool = False):
        """Remove and return the `(key, value)` pair least recently used."""
        if not remove_pinned:
            # pop the oldest item in the cache that is not pinned
            lru_key = next(
                (key for key in self.order if key not in self.pinned_items),
                ALL_PINNED_SENTINEL)
            if lru_key is ALL_PINNED_SENTINEL:
                raise RuntimeError("All items are pinned, "
                                   "cannot remove oldest from the cache.")
        else:
            lru_key = next(iter(self.order))
        value = self.pop(cast(_K, lru_key))
        return (lru_key, value)

    def clear(self) -> None:
        while len(self) > 0:
            self.remove_oldest(remove_pinned=True)

        self._hits = 0
        self._total = 0
        self._last_info = CacheInfo(hits=0, total=0)


class PyObjectCache:
    """Used to cache python objects to avoid object allocations
    across scheduler iterations.
    """

    def __init__(self, obj_builder):
        self._obj_builder = obj_builder
        self._index = 0

        self._obj_cache = []
        for _ in range(128):
            self._obj_cache.append(self._obj_builder())

    def _grow_cache(self):
        # Double the size of the cache
        num_objs = len(self._obj_cache)
        for _ in range(num_objs):
            self._obj_cache.append(self._obj_builder())

    def get_object(self):
        """Returns a pre-allocated cached object. If there is not enough
        objects, then the cache size will double.
        """
        if self._index >= len(self._obj_cache):
            self._grow_cache()
            assert self._index < len(self._obj_cache)

        obj = self._obj_cache[self._index]
        self._index += 1

        return obj

    def reset(self):
        """Makes all cached-objects available for the next scheduler iteration.
        """
        self._index = 0


@cache
def get_max_shared_memory_bytes(gpu: int = 0) -> int:
    """Returns the maximum shared memory per thread block in bytes."""
    from vllm import _custom_ops as ops
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


def make_async(
    func: Callable[P, T],
    executor: Optional[concurrent.futures.Executor] = None
) -> Callable[P, Awaitable[T]]:
    """Take a blocking function, and run it on in an executor thread.

    This function prevents the blocking function from blocking the
    asyncio event loop.
    The code in this function needs to be thread safe.
    """

    def _async_wrapper(*args: P.args, **kwargs: P.kwargs) -> asyncio.Future:
        loop = asyncio.get_event_loop()
        p_func = partial(func, *args, **kwargs)
        return loop.run_in_executor(executor=executor, func=p_func)

    return _async_wrapper


def _next_task(iterator: AsyncGenerator[T, None],
               loop: AbstractEventLoop) -> Task:
    # Can use anext() in python >= 3.10
    return loop.create_task(iterator.__anext__())  # type: ignore[arg-type]


async def merge_async_iterators(
    *iterators: AsyncGenerator[T,
                               None], ) -> AsyncGenerator[tuple[int, T], None]:
    """Merge multiple asynchronous iterators into a single iterator.

    This method handle the case where some iterators finish before others.
    When it yields, it yields a tuple (i, item) where i is the index of the
    iterator that yields the item.
    """
    if len(iterators) == 1:
        # Fast-path single iterator case.
        async for item in iterators[0]:
            yield 0, item
        return

    loop = asyncio.get_running_loop()

    awaits = {_next_task(pair[1], loop): pair for pair in enumerate(iterators)}
    try:
        while awaits:
            done, _ = await asyncio.wait(awaits.keys(),
                                         return_when=FIRST_COMPLETED)
            for d in done:
                pair = awaits.pop(d)
                try:
                    item = await d
                    i, it = pair
                    awaits[_next_task(it, loop)] = pair
                    yield i, item
                except StopAsyncIteration:
                    pass
    finally:
        # Cancel any remaining iterators
        for f, (_, it) in awaits.items():
            with contextlib.suppress(BaseException):
                f.cancel()
                await it.aclose()


async def collect_from_async_generator(
        iterator: AsyncGenerator[T, None]) -> list[T]:
    """Collect all items from an async generator into a list."""
    items = []
    async for item in iterator:
        items.append(item)
    return items


def get_ip() -> str:
    host_ip = envs.VLLM_HOST_IP
    if "HOST_IP" in os.environ and "VLLM_HOST_IP" not in os.environ:
        logger.warning(
            "The environment variable HOST_IP is deprecated and ignored, as"
            " it is often used by Docker and other software to"
            " interact with the container's network stack. Please "
            "use VLLM_HOST_IP instead to set the IP address for vLLM processes"
            " to communicate with each other.")
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


def is_valid_ipv6_address(address: str) -> bool:
    try:
        ipaddress.IPv6Address(address)
        return True
    except ValueError:
        return False


def get_distributed_init_method(ip: str, port: int) -> str:
    return get_tcp_uri(ip, port)


def get_tcp_uri(ip: str, port: int) -> str:
    # Brackets are not permitted in ipv4 addresses,
    # see https://github.com/python/cpython/issues/103848
    return f"tcp://[{ip}]:{port}" if ":" in ip else f"tcp://{ip}:{port}"


def get_open_zmq_ipc_path() -> str:
    base_rpc_path = envs.VLLM_RPC_BASE_PATH
    return f"ipc://{base_rpc_path}/{uuid4()}"


def get_open_zmq_inproc_path() -> str:
    return f"inproc://{uuid4()}"


def get_open_port() -> int:
    """
    Get an open port for the vLLM process to listen on.
    An edge case to handle, is when we run data parallel,
    we need to avoid ports that are potentially used by
    the data parallel master process.
    Right now we reserve 10 ports for the data parallel master
    process. Currently it uses 2 ports.
    """
    if "VLLM_DP_MASTER_PORT" in os.environ:
        dp_master_port = envs.VLLM_DP_MASTER_PORT
        reserved_port_range = range(dp_master_port, dp_master_port + 10)
        while True:
            candidate_port = _get_open_port()
            if candidate_port not in reserved_port_range:
                return candidate_port
    return _get_open_port()


def _get_open_port() -> int:
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


def find_process_using_port(port: int) -> Optional[psutil.Process]:
    # TODO: We can not check for running processes with network
    # port on macOS. Therefore, we can not have a full graceful shutdown
    # of vLLM. For now, let's not look for processes in this case.
    # Ref: https://www.florianreinhard.de/accessdenied-in-psutil/
    if sys.platform.startswith("darwin"):
        return None

    for conn in psutil.net_connections():
        if conn.laddr.port == port:
            try:
                return psutil.Process(conn.pid)
            except psutil.NoSuchProcess:
                return None
    return None


def update_environment_variables(envs: dict[str, str]):
    for k, v in envs.items():
        if k in os.environ and os.environ[k] != v:
            logger.warning(
                "Overwriting environment variable %s "
                "from '%s' to '%s'", k, os.environ[k], v)
        os.environ[k] = v


def chunk_list(lst: list[T], chunk_size: int):
    """Yield successive chunk_size chunks from lst."""
    for i in range(0, len(lst), chunk_size):
        yield lst[i:i + chunk_size]


def cdiv(a: int, b: int) -> int:
    """Ceiling division."""
    return -(a // -b)


def next_power_of_2(n) -> int:
    """The next power of 2 (inclusive)"""
    if n < 1:
        return 1
    return 1 << (n - 1).bit_length()


def round_up(x: int, y: int) -> int:
    return ((x + y - 1) // y) * y


def round_down(x: int, y: int) -> int:
    return (x // y) * y


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
            if isinstance(model_dtype,
                          str) and model_dtype in STR_DTYPE_TO_TORCH_DTYPE:
                torch_dtype = STR_DTYPE_TO_TORCH_DTYPE[model_dtype]
            elif isinstance(model_dtype, torch.dtype):
                torch_dtype = model_dtype
            else:
                raise ValueError(f"Invalid model dtype: {model_dtype}")
        elif cache_dtype in STR_DTYPE_TO_TORCH_DTYPE:
            torch_dtype = STR_DTYPE_TO_TORCH_DTYPE[cache_dtype]
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
    seed: Optional[int] = None,
    device: Optional[str] = "cuda",
    cache_layout: Optional[str] = "NHD",
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    from vllm.platforms import current_platform
    current_platform.seed_everything(seed)

    torch_dtype = get_kv_cache_torch_dtype(cache_dtype, model_dtype)
    generic_kv_cache_shape = (num_blocks, 2, block_size, num_heads, head_size)
    assert cache_layout in ("NHD", "HND")
    stride_order = (0, 1, 2, 3, 4) if cache_layout == "NHD" else (0, 1, 3, 2,
                                                                  4)

    kv_cache_allocation_shape = tuple(generic_kv_cache_shape[i]
                                      for i in stride_order)
    scale = head_size**-0.5

    key_caches: list[torch.Tensor] = []
    value_caches: list[torch.Tensor] = []

    for _ in range(num_layers):
        key_value_cache = torch.empty(size=kv_cache_allocation_shape,
                                      dtype=torch_dtype,
                                      device=device).permute(*stride_order)
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
    seed: Optional[int] = None,
    device: Optional[str] = "cuda",
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:

    if cache_dtype == "fp8" and head_size % 16:
        raise ValueError(
            f"Does not support key cache of type fp8 with head_size {head_size}"
        )
    from vllm.platforms import current_platform
    current_platform.seed_everything(seed)

    torch_dtype = get_kv_cache_torch_dtype(cache_dtype, model_dtype)

    scale = head_size**-0.5
    x = 16 // torch.tensor([], dtype=torch_dtype).element_size()
    key_cache_shape = (num_blocks, num_heads, head_size // x, block_size, x)
    key_caches: list[torch.Tensor] = []
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
    value_caches: list[torch.Tensor] = []
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


@cache
def is_pin_memory_available() -> bool:
    from vllm.platforms import current_platform
    return current_platform.is_pin_memory_available()


@cache
def is_uva_available() -> bool:
    """Check if Unified Virtual Addressing (UVA) is available."""
    # UVA requires pinned memory.
    # TODO: Add more requirements for UVA if needed.
    return is_pin_memory_available()


class DeviceMemoryProfiler:

    def __init__(self, device: Optional[torch.types.Device] = None):
        self.device = device

    def current_memory_usage(self) -> float:
        # Return the memory usage in bytes.
        from vllm.platforms import current_platform
        return current_platform.get_current_memory_usage(self.device)

    def __enter__(self):
        self.initial_memory = self.current_memory_usage()
        # This allows us to call methods of the context manager if needed
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.final_memory = self.current_memory_usage()
        self.consumed_memory = self.final_memory - self.initial_memory

        # Force garbage collection
        gc.collect()


def make_ndarray_with_pad(
    x: list[list[T]],
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
    x: list[list[T]],
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


def get_dtype_size(dtype: torch.dtype) -> int:
    """Get the size of the data type in bytes."""
    return torch.tensor([], dtype=dtype).element_size()


# `collections` helpers
def is_list_of(
    value: object,
    typ: Union[type[T], tuple[type[T], ...]],
    *,
    check: Literal["first", "all"] = "first",
) -> TypeIs[list[T]]:
    if not isinstance(value, list):
        return False

    if check == "first":
        return len(value) == 0 or isinstance(value[0], typ)
    elif check == "all":
        return all(isinstance(v, typ) for v in value)

    assert_never(check)


def flatten_2d_lists(lists: Iterable[Iterable[T]]) -> list[T]:
    """Flatten a list of lists to a single list."""
    return [item for sublist in lists for item in sublist]


def full_groupby(values: Iterable[_V], *, key: Callable[[_V], _K]):
    """
    Unlike [`itertools.groupby`][], groups are not broken by
    non-contiguous data.
    """
    groups = defaultdict[_K, list[_V]](list)

    for value in values:
        groups[key(value)].append(value)

    return groups.items()


# TODO: This function can be removed if transformer_modules classes are
# serialized by value when communicating between processes
def init_cached_hf_modules() -> None:
    """
    Lazy initialization of the Hugging Face modules.
    """
    from transformers.dynamic_module_utils import init_hf_modules
    init_hf_modules()


@cache
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


prev_set_stream = torch.cuda.set_stream

_current_stream = None


def _patched_set_stream(stream: torch.cuda.Stream) -> None:
    global _current_stream
    _current_stream = stream
    prev_set_stream(stream)


torch.cuda.set_stream = _patched_set_stream


def current_stream() -> torch.cuda.Stream:
    """
    replace `torch.cuda.current_stream()` with `vllm.utils.current_stream()`.
    it turns out that `torch.cuda.current_stream()` is quite expensive,
    as it will construct a new stream object at each call.
    here we patch `torch.cuda.set_stream` to keep track of the current stream
    directly, so that we can avoid calling `torch.cuda.current_stream()`.

    the underlying hypothesis is that we do not call `torch._C._cuda_setStream`
    from C/C++ code.
    """
    from vllm.platforms import current_platform
    global _current_stream
    if _current_stream is None:
        # when this function is called before any stream is set,
        # we return the default stream.
        # On ROCm using the default 0 stream in combination with RCCL
        # is hurting performance. Therefore creating a dedicated stream
        # per process
        _current_stream = torch.cuda.Stream() if current_platform.is_rocm(
        ) else torch.cuda.current_stream()
    return _current_stream


def enable_trace_function_call_for_thread(vllm_config: VllmConfig) -> None:
    """Set up function tracing for the current thread,
    if enabled via the VLLM_TRACE_FUNCTION environment variable
    """

    if envs.VLLM_TRACE_FUNCTION:
        tmp_dir = tempfile.gettempdir()
        # add username to tmp_dir to avoid permission issues
        tmp_dir = os.path.join(tmp_dir, getpass.getuser())
        filename = (f"VLLM_TRACE_FUNCTION_for_process_{os.getpid()}"
                    f"_thread_{threading.get_ident()}_"
                    f"at_{datetime.datetime.now()}.log").replace(" ", "_")
        log_path = os.path.join(tmp_dir, "vllm",
                                f"vllm-instance-{vllm_config.instance_id}",
                                filename)
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        enable_trace_function_call(log_path)


# `functools` helpers
def identity(value: T, **kwargs) -> T:
    """Returns the first provided value."""
    return value


F = TypeVar('F', bound=Callable[..., Any])


def deprecate_args(
    start_index: int,
    is_deprecated: Union[bool, Callable[[], bool]] = True,
    additional_message: Optional[str] = None,
) -> Callable[[F], F]:

    if not callable(is_deprecated):
        is_deprecated = partial(identity, is_deprecated)

    def wrapper(fn: F) -> F:

        params = inspect.signature(fn).parameters
        pos_types = (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        )
        pos_kws = [
            kw for kw, param in params.items() if param.kind in pos_types
        ]

        @wraps(fn)
        def inner(*args, **kwargs):
            if is_deprecated():
                deprecated_args = pos_kws[start_index:len(args)]
                if deprecated_args:
                    msg = (
                        f"The positional arguments {deprecated_args} are "
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


def deprecate_kwargs(
    *kws: str,
    is_deprecated: Union[bool, Callable[[], bool]] = True,
    additional_message: Optional[str] = None,
) -> Callable[[F], F]:
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

    from vllm.platforms import current_platform
    if not torch.cuda._is_compiled():
        return 0
    if current_platform.is_rocm():
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


def cuda_is_initialized() -> bool:
    """Check if CUDA is initialized."""
    if not torch.cuda._is_compiled():
        return False
    return torch.cuda.is_initialized()


def cuda_get_device_properties(device,
                               names: Sequence[str],
                               init_cuda=False) -> tuple[Any, ...]:
    """Get specified CUDA device property values without initializing CUDA in
    the current process."""
    if init_cuda or cuda_is_initialized():
        props = torch.cuda.get_device_properties(device)
        return tuple(getattr(props, name) for name in names)

    # Run in subprocess to avoid initializing CUDA as a side effect.
    mp_ctx = multiprocessing.get_context("fork")
    with ProcessPoolExecutor(max_workers=1, mp_context=mp_ctx) as executor:
        return executor.submit(cuda_get_device_properties, device, names,
                               True).result()


def weak_bind(bound_method: Callable[..., Any], ) -> Callable[..., None]:
    """Make an instance method that weakly references
    its associated instance and no-ops once that
    instance is collected."""
    ref = weakref.ref(bound_method.__self__)  # type: ignore[attr-defined]
    unbound = bound_method.__func__  # type: ignore[attr-defined]

    def weak_bound(*args, **kwargs) -> None:
        if inst := ref():
            unbound(inst, *args, **kwargs)

    return weak_bound


#From: https://stackoverflow.com/a/4104188/2749989
def run_once(f: Callable[P, None]) -> Callable[P, None]:

    def wrapper(*args: P.args, **kwargs: P.kwargs) -> None:
        if not wrapper.has_run:  # type: ignore[attr-defined]
            wrapper.has_run = True  # type: ignore[attr-defined]
            return f(*args, **kwargs)

    wrapper.has_run = False  # type: ignore[attr-defined]
    return wrapper


class StoreBoolean(Action):

    def __call__(self, parser, namespace, values, option_string=None):
        if values.lower() == "true":
            setattr(namespace, self.dest, True)
        elif values.lower() == "false":
            setattr(namespace, self.dest, False)
        else:
            raise ValueError(f"Invalid boolean value: {values}. "
                             "Expected 'true' or 'false'.")


class SortedHelpFormatter(ArgumentDefaultsHelpFormatter,
                          RawDescriptionHelpFormatter):
    """SortedHelpFormatter that sorts arguments by their option strings."""

    def _split_lines(self, text, width):
        """
        1. Sentences split across lines have their single newlines removed.
        2. Paragraphs and explicit newlines are split into separate lines.
        3. Each line is wrapped to the specified width (width of terminal).
        """
        # The patterns also include whitespace after the newline
        single_newline = re.compile(r"(?<!\n)\n(?!\n)\s*")
        multiple_newlines = re.compile(r"\n{2,}\s*")
        text = single_newline.sub(' ', text)
        lines = re.split(multiple_newlines, text)
        return sum([textwrap.wrap(line, width) for line in lines], [])

    def add_arguments(self, actions):
        actions = sorted(actions, key=lambda x: x.option_strings)
        super().add_arguments(actions)


class FlexibleArgumentParser(ArgumentParser):
    """ArgumentParser that allows both underscore and dash in names."""

    _deprecated: set[Action] = set()

    def __init__(self, *args, **kwargs):
        # Set the default 'formatter_class' to SortedHelpFormatter
        if 'formatter_class' not in kwargs:
            kwargs['formatter_class'] = SortedHelpFormatter
        super().__init__(*args, **kwargs)

    if sys.version_info < (3, 13):
        # Enable the deprecated kwarg for Python 3.12 and below

        def parse_known_args(self, args=None, namespace=None):
            namespace, args = super().parse_known_args(args, namespace)
            for action in FlexibleArgumentParser._deprecated:
                if (hasattr(namespace, dest := action.dest)
                        and getattr(namespace, dest) != action.default):
                    logger.warning_once("argument '%s' is deprecated", dest)
            return namespace, args

        def add_argument(self, *args, **kwargs):
            deprecated = kwargs.pop("deprecated", False)
            action = super().add_argument(*args, **kwargs)
            if deprecated:
                FlexibleArgumentParser._deprecated.add(action)
            return action

        class _FlexibleArgumentGroup(_ArgumentGroup):

            def add_argument(self, *args, **kwargs):
                deprecated = kwargs.pop("deprecated", False)
                action = super().add_argument(*args, **kwargs)
                if deprecated:
                    FlexibleArgumentParser._deprecated.add(action)
                return action

        def add_argument_group(self, *args, **kwargs):
            group = self._FlexibleArgumentGroup(self, *args, **kwargs)
            self._action_groups.append(group)
            return group

    def parse_args(  # type: ignore[override]
        self,
        args: list[str] | None = None,
        namespace: Namespace | None = None,
    ):
        if args is None:
            args = sys.argv[1:]

        # Check for --model in command line arguments first
        if args and args[0] == "serve":
            model_in_cli_args = any(arg == '--model' for arg in args)

            if model_in_cli_args:
                raise ValueError(
                    "With `vllm serve`, you should provide the model as a "
                    "positional argument or in a config file instead of via "
                    "the `--model` option.")

        if '--config' in args:
            args = self._pull_args_from_config(args)

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
            elif arg.startswith('-O') and arg != '-O' and len(arg) == 2:
                # allow -O flag to be used without space, e.g. -O3
                processed_args.append('-O')
                processed_args.append(arg[2:])
            else:
                processed_args.append(arg)

        def create_nested_dict(keys: list[str], value: str):
            """Creates a nested dictionary from a list of keys and a value.

            For example, `keys = ["a", "b", "c"]` and `value = 1` will create:
            `{"a": {"b": {"c": 1}}}`
            """
            nested_dict: Any = value
            for key in reversed(keys):
                nested_dict = {key: nested_dict}
            return nested_dict

        def recursive_dict_update(original: dict, update: dict):
            """Recursively updates a dictionary with another dictionary."""
            for k, v in update.items():
                if isinstance(v, dict) and isinstance(original.get(k), dict):
                    recursive_dict_update(original[k], v)
                else:
                    original[k] = v

        delete = set()
        dict_args: dict[str, dict] = defaultdict(dict)
        for i, processed_arg in enumerate(processed_args):
            if processed_arg.startswith("--") and "." in processed_arg:
                if "=" in processed_arg:
                    processed_arg, value = processed_arg.split("=", 1)
                    if "." not in processed_arg:
                        # False positive, . was only in the value
                        continue
                else:
                    value = processed_args[i + 1]
                    delete.add(i + 1)
                key, *keys = processed_arg.split(".")
                # Merge all values with the same key into a single dict
                arg_dict = create_nested_dict(keys, value)
                recursive_dict_update(dict_args[key], arg_dict)
                delete.add(i)
        # Filter out the dict args we set to None
        processed_args = [
            a for i, a in enumerate(processed_args) if i not in delete
        ]
        # Add the dict args back as if they were originally passed as JSON
        for dict_arg, dict_value in dict_args.items():
            processed_args.append(dict_arg)
            processed_args.append(json.dumps(dict_value))

        return super().parse_args(processed_args, namespace)

    def check_port(self, value):
        try:
            value = int(value)
        except ValueError:
            msg = "Port must be an integer"
            raise ArgumentTypeError(msg) from None

        if not (1024 <= value <= 65535):
            raise ArgumentTypeError("Port must be between 1024 and 65535")

        return value

    def _pull_args_from_config(self, args: list[str]) -> list[str]:
        """Method to pull arguments specified in the config file
        into the command-line args variable.

        The arguments in config file will be inserted between
        the argument list.

        example:
        ```yaml
            port: 12323
            tensor-parallel-size: 4
        ```
        ```python
        $: vllm {serve,chat,complete} "facebook/opt-12B" \
            --config config.yaml -tp 2
        $: args = [
            "serve,chat,complete",
            "facebook/opt-12B",
            '--config', 'config.yaml',
            '-tp', '2'
        ]
        $: args = [
            "serve,chat,complete",
            "facebook/opt-12B",
            '--port', '12323',
            '--tensor-parallel-size', '4',
            '-tp', '2'
            ]
        ```

        Please note how the config args are inserted after the sub command.
        this way the order of priorities is maintained when these are args
        parsed by super().
        """
        assert args.count(
            '--config') <= 1, "More than one config file specified!"

        index = args.index('--config')
        if index == len(args) - 1:
            raise ValueError("No config file specified! \
                             Please check your command-line arguments.")

        file_path = args[index + 1]

        config_args = self._load_config_file(file_path)

        # 0th index is for {serve,chat,complete}
        # optionally followed by model_tag (only for serve)
        # followed by config args
        # followed by rest of cli args.
        # maintaining this order will enforce the precedence
        # of cli > config > defaults
        if args[0] == "serve":
            model_in_cli = len(args) > 1 and not args[1].startswith('-')
            model_in_config = any(arg == '--model' for arg in config_args)

            if not model_in_cli and not model_in_config:
                raise ValueError(
                    "No model specified! Please specify model either "
                    "as a positional argument or in a config file.")

            if model_in_cli:
                # Model specified as positional arg, keep CLI version
                args = [args[0]] + [
                    args[1]
                ] + config_args + args[2:index] + args[index + 2:]
            else:
                # No model in CLI, use config if available
                args = [args[0]
                        ] + config_args + args[1:index] + args[index + 2:]
        else:
            args = [args[0]] + config_args + args[1:index] + args[index + 2:]

        return args

    def _load_config_file(self, file_path: str) -> list[str]:
        """Loads a yaml file and returns the key value pairs as a
        flattened list with argparse like pattern
        ```yaml
            port: 12323
            tensor-parallel-size: 4
        ```
        returns:
            processed_args: list[str] = [
                '--port': '12323',
                '--tensor-parallel-size': '4'
            ]
        """
        extension: str = file_path.split('.')[-1]
        if extension not in ('yaml', 'yml'):
            raise ValueError(
                "Config file must be of a yaml/yml type.\
                              %s supplied", extension)

        # only expecting a flat dictionary of atomic types
        processed_args: list[str] = []

        config: dict[str, Union[int, str]] = {}
        try:
            with open(file_path) as config_file:
                config = yaml.safe_load(config_file)
        except Exception as ex:
            logger.error(
                "Unable to read the config file at %s. \
                Make sure path is correct", file_path)
            raise ex

        store_boolean_arguments = [
            action.dest for action in self._actions
            if isinstance(action, StoreBoolean)
        ]

        for key, value in config.items():
            if isinstance(value, bool) and key not in store_boolean_arguments:
                if value:
                    processed_args.append('--' + key)
            else:
                processed_args.append('--' + key)
                processed_args.append(str(value))

        return processed_args


async def _run_task_with_lock(task: Callable, lock: asyncio.Lock, *args,
                              **kwargs):
    """Utility function to run async task in a lock"""
    async with lock:
        return await task(*args, **kwargs)


def supports_kw(
    callable: Callable[..., object],
    kw_name: str,
    *,
    requires_kw_only: bool = False,
    allow_var_kwargs: bool = True,
) -> bool:
    """Check if a keyword is a valid kwarg for a callable; if requires_kw_only
    disallows kwargs names that can also be positional arguments.
    """
    params = inspect.signature(callable).parameters
    if not params:
        return False

    param_val = params.get(kw_name)

    # Types where the it may be valid, i.e., explicitly defined & nonvariadic
    passable_kw_types = set((inspect.Parameter.POSITIONAL_ONLY,
                             inspect.Parameter.POSITIONAL_OR_KEYWORD,
                             inspect.Parameter.KEYWORD_ONLY))

    if param_val:
        is_sig_param = param_val.kind in passable_kw_types
        # We want kwargs only, but this is passable as a positional arg
        if (requires_kw_only and is_sig_param
                and param_val.kind != inspect.Parameter.KEYWORD_ONLY):
            return False
        if ((requires_kw_only
             and param_val.kind == inspect.Parameter.KEYWORD_ONLY)
                or (not requires_kw_only and is_sig_param)):
            return True

    # If we're okay with var-kwargs, it's supported as long as
    # the kw_name isn't something like *args, **kwargs
    if allow_var_kwargs:
        # Get the last param; type is ignored here because params is a proxy
        # mapping, but it wraps an ordered dict, and they appear in order.
        # Ref: https://docs.python.org/3/library/inspect.html#inspect.Signature.parameters
        last_param = params[next(reversed(params))]  # type: ignore
        return (last_param.kind == inspect.Parameter.VAR_KEYWORD
                and last_param.name != kw_name)
    return False


def resolve_mm_processor_kwargs(
    init_kwargs: Optional[Mapping[str, object]],
    inference_kwargs: Optional[Mapping[str, object]],
    callable: Callable[..., object],
    *,
    requires_kw_only: bool = True,
    allow_var_kwargs: bool = False,
) -> dict[str, Any]:
    """Applies filtering to eliminate invalid mm_processor_kwargs, i.e.,
    those who are not explicit keywords to the given callable (of one is
    given; otherwise no filtering is done), then merges the kwarg dicts,
    giving priority to inference_kwargs if there are any collisions.

    In the case that no kwarg overrides are provided, returns an empty
    dict so that it can still be kwarg expanded into the callable later on.

    If allow_var_kwargs=True, allows for things that can be expanded into
    kwargs as long as they aren't naming collision for var_kwargs or potential
    positional arguments.
    """
    # Filter inference time multimodal processor kwargs provided
    runtime_mm_kwargs = get_allowed_kwarg_only_overrides(
        callable,
        overrides=inference_kwargs,
        requires_kw_only=requires_kw_only,
        allow_var_kwargs=allow_var_kwargs,
    )

    # Filter init time multimodal processor kwargs provided
    init_mm_kwargs = get_allowed_kwarg_only_overrides(
        callable,
        overrides=init_kwargs,
        requires_kw_only=requires_kw_only,
        allow_var_kwargs=allow_var_kwargs,
    )

    # Merge the final processor kwargs, prioritizing inference
    # time values over the initialization time values.
    mm_processor_kwargs = {**init_mm_kwargs, **runtime_mm_kwargs}
    return mm_processor_kwargs


def get_allowed_kwarg_only_overrides(
    callable: Callable[..., object],
    overrides: Optional[Mapping[str, object]],
    *,
    requires_kw_only: bool = True,
    allow_var_kwargs: bool = False,
) -> dict[str, Any]:
    """
    Given a callable which has one or more keyword only params and a dict
    mapping param names to values, drop values that can be not be kwarg
    expanded to overwrite one or more keyword-only args. This is used in a
    few places to handle custom processor overrides for multimodal models,
    e.g., for profiling when processor options provided by the user
    may affect the number of mm tokens per instance.

    Args:
        callable: Callable which takes 0 or more keyword only arguments.
                  If None is provided, all overrides names are allowed.
        overrides: Potential overrides to be used when invoking the callable.
        allow_var_kwargs: Allows overrides that are expandable for var kwargs.

    Returns:
        Dictionary containing the kwargs to be leveraged which may be used
        to overwrite one or more keyword only arguments when invoking the
        callable.
    """
    if not overrides:
        return {}

    # Drop any mm_processor_kwargs provided by the user that
    # are not kwargs, unless it can fit it var_kwargs param
    filtered_overrides = {
        kwarg_name: val
        for kwarg_name, val in overrides.items()
        if supports_kw(callable,
                       kwarg_name,
                       requires_kw_only=requires_kw_only,
                       allow_var_kwargs=allow_var_kwargs)
    }

    # If anything is dropped, log a warning
    dropped_keys = overrides.keys() - filtered_overrides.keys()
    if dropped_keys:
        if requires_kw_only:
            logger.warning(
                "The following intended overrides are not keyword-only args "
                "and will be dropped: %s", dropped_keys)
        else:
            logger.warning(
                "The following intended overrides are not keyword args "
                "and will be dropped: %s", dropped_keys)

    return filtered_overrides


# Using dynamo with vLLM doesn't really work well with PyTorch versions < 2.4.0.
# In particular, the FakeScalarType is not supported for earlier versions of
# PyTorch which breaks dynamo for any ops registered using ScalarType.
def supports_dynamo() -> bool:
    base_torch_version = Version(Version(torch.__version__).base_version)
    return base_torch_version >= Version("2.4.0")


# Some backends use pytorch version < 2.4.0 which doesn't
# support `torch.library.custom_op`.
def supports_custom_op() -> bool:
    return hasattr(torch.library, "custom_op")


class AtomicCounter:
    """An atomic, thread-safe counter"""

    def __init__(self, initial=0):
        """Initialize a new atomic counter to given initial value"""
        self._value = initial
        self._lock = threading.Lock()

    def inc(self, num=1):
        """Atomically increment the counter by num and return the new value"""
        with self._lock:
            self._value += num
            return self._value

    def dec(self, num=1):
        """Atomically decrement the counter by num and return the new value"""
        with self._lock:
            self._value -= num
            return self._value

    @property
    def value(self):
        return self._value


# Adapted from: https://stackoverflow.com/a/47212782/5082708
class LazyDict(Mapping[str, T], Generic[T]):

    def __init__(self, factory: dict[str, Callable[[], T]]):
        self._factory = factory
        self._dict: dict[str, T] = {}

    def __getitem__(self, key: str) -> T:
        if key not in self._dict:
            if key not in self._factory:
                raise KeyError(key)
            self._dict[key] = self._factory[key]()
        return self._dict[key]

    def __setitem__(self, key: str, value: Callable[[], T]):
        self._factory[key] = value

    def __iter__(self):
        return iter(self._factory)

    def __len__(self):
        return len(self._factory)


class ClassRegistry(UserDict[Type[T], _V]):

    def __getitem__(self, key: Type[T]) -> _V:
        for cls in key.mro():
            if cls in self.data:
                return self.data[cls]

        raise KeyError(key)

    def __contains__(self, key: object) -> bool:
        return self.contains(key)

    def contains(self, key: object, *, strict: bool = False) -> bool:
        if not isinstance(key, type):
            return False

        if strict:
            return key in self.data

        return any(cls in self.data for cls in key.mro())


def weak_ref_tensor(tensor: Any) -> Any:
    """
    Create a weak reference to a tensor.
    The new tensor will share the same data as the original tensor,
    but will not keep the original tensor alive.
    """
    if isinstance(tensor, torch.Tensor):
        return torch.ops._C.weak_ref_tensor(tensor)
    else:
        return tensor


def weak_ref_tensors(
    tensors: Union[torch.Tensor, list[torch.Tensor], tuple[torch.Tensor]]
) -> Union[torch.Tensor, list[Any], tuple[Any], Any]:
    """
    Convenience function to create weak references to tensors,
    for single tensor, list of tensors or tuple of tensors.
    """
    if isinstance(tensors, torch.Tensor):
        return weak_ref_tensor(tensors)
    if isinstance(tensors, list):
        return [weak_ref_tensor(t) for t in tensors]
    if isinstance(tensors, tuple):
        return tuple(weak_ref_tensor(t) for t in tensors)
    raise ValueError("Invalid type for tensors")


def get_cuda_view_from_cpu_tensor(cpu_tensor: torch.Tensor) -> torch.Tensor:
    """
    Get a CUDA view of a CPU tensor using Unified Virtual Addressing (UVA).
    """
    assert cpu_tensor.is_pinned(), "CPU tensor must be pinned"
    return torch.ops._C.get_cuda_view_from_cpu_tensor(cpu_tensor)


def import_from_path(module_name: str, file_path: Union[str, os.PathLike]):
    """
    Import a Python file according to its file path.

    Based on the official recipe:
    https://docs.python.org/3/library/importlib.html#importing-a-source-file-directly
    """
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None:
        raise ModuleNotFoundError(f"No module named '{module_name}'")

    assert spec.loader is not None

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


@cache
def get_vllm_optional_dependencies():
    metadata = importlib.metadata.metadata("vllm")
    requirements = metadata.get_all("Requires-Dist", [])
    extras = metadata.get_all("Provides-Extra", [])

    return {
        extra: [
            re.split(r";|>=|<=|==", req)[0] for req in requirements
            if req.endswith(f'extra == "{extra}"')
        ]
        for extra in extras
    }


class _PlaceholderBase:
    """
    Disallows downstream usage of placeholder modules.

    We need to explicitly override each dunder method because
    [`__getattr__`][vllm.utils._PlaceholderBase.__getattr__]
    is not called when they are accessed.

    Info:
        [Special method lookup](https://docs.python.org/3/reference/datamodel.html#special-lookup)
    """

    def __getattr__(self, key: str) -> Never:
        """
        The main class should implement this to throw an error
        for attribute accesses representing downstream usage.
        """
        raise NotImplementedError

    # [Basic customization]

    def __lt__(self, other: object):
        return self.__getattr__("__lt__")

    def __le__(self, other: object):
        return self.__getattr__("__le__")

    def __eq__(self, other: object):
        return self.__getattr__("__eq__")

    def __ne__(self, other: object):
        return self.__getattr__("__ne__")

    def __gt__(self, other: object):
        return self.__getattr__("__gt__")

    def __ge__(self, other: object):
        return self.__getattr__("__ge__")

    def __hash__(self):
        return self.__getattr__("__hash__")

    def __bool__(self):
        return self.__getattr__("__bool__")

    # [Callable objects]

    def __call__(self, *args: object, **kwargs: object):
        return self.__getattr__("__call__")

    # [Container types]

    def __len__(self):
        return self.__getattr__("__len__")

    def __getitem__(self, key: object):
        return self.__getattr__("__getitem__")

    def __setitem__(self, key: object, value: object):
        return self.__getattr__("__setitem__")

    def __delitem__(self, key: object):
        return self.__getattr__("__delitem__")

    # __missing__ is optional according to __getitem__ specification,
    # so it is skipped

    # __iter__ and __reversed__ have a default implementation
    # based on __len__ and __getitem__, so they are skipped.

    # [Numeric Types]

    def __add__(self, other: object):
        return self.__getattr__("__add__")

    def __sub__(self, other: object):
        return self.__getattr__("__sub__")

    def __mul__(self, other: object):
        return self.__getattr__("__mul__")

    def __matmul__(self, other: object):
        return self.__getattr__("__matmul__")

    def __truediv__(self, other: object):
        return self.__getattr__("__truediv__")

    def __floordiv__(self, other: object):
        return self.__getattr__("__floordiv__")

    def __mod__(self, other: object):
        return self.__getattr__("__mod__")

    def __divmod__(self, other: object):
        return self.__getattr__("__divmod__")

    def __pow__(self, other: object, modulo: object = ...):
        return self.__getattr__("__pow__")

    def __lshift__(self, other: object):
        return self.__getattr__("__lshift__")

    def __rshift__(self, other: object):
        return self.__getattr__("__rshift__")

    def __and__(self, other: object):
        return self.__getattr__("__and__")

    def __xor__(self, other: object):
        return self.__getattr__("__xor__")

    def __or__(self, other: object):
        return self.__getattr__("__or__")

    # r* and i* methods have lower priority than
    # the methods for left operand so they are skipped

    def __neg__(self):
        return self.__getattr__("__neg__")

    def __pos__(self):
        return self.__getattr__("__pos__")

    def __abs__(self):
        return self.__getattr__("__abs__")

    def __invert__(self):
        return self.__getattr__("__invert__")

    # __complex__, __int__ and __float__ have a default implementation
    # based on __index__, so they are skipped.

    def __index__(self):
        return self.__getattr__("__index__")

    def __round__(self, ndigits: object = ...):
        return self.__getattr__("__round__")

    def __trunc__(self):
        return self.__getattr__("__trunc__")

    def __floor__(self):
        return self.__getattr__("__floor__")

    def __ceil__(self):
        return self.__getattr__("__ceil__")

    # [Context managers]

    def __enter__(self):
        return self.__getattr__("__enter__")

    def __exit__(self, *args: object, **kwargs: object):
        return self.__getattr__("__exit__")


class PlaceholderModule(_PlaceholderBase):
    """
    A placeholder object to use when a module does not exist.

    This enables more informative errors when trying to access attributes
    of a module that does not exists.
    """

    def __init__(self, name: str) -> None:
        super().__init__()

        # Apply name mangling to avoid conflicting with module attributes
        self.__name = name

    def placeholder_attr(self, attr_path: str):
        return _PlaceholderModuleAttr(self, attr_path)

    def __getattr__(self, key: str):
        name = self.__name

        try:
            importlib.import_module(name)
        except ImportError as exc:
            for extra, names in get_vllm_optional_dependencies().items():
                if name in names:
                    msg = f"Please install vllm[{extra}] for {extra} support"
                    raise ImportError(msg) from exc

            raise exc

        raise AssertionError("PlaceholderModule should not be used "
                             "when the original module can be imported")


class _PlaceholderModuleAttr(_PlaceholderBase):

    def __init__(self, module: PlaceholderModule, attr_path: str) -> None:
        super().__init__()

        # Apply name mangling to avoid conflicting with module attributes
        self.__module = module
        self.__attr_path = attr_path

    def placeholder_attr(self, attr_path: str):
        return _PlaceholderModuleAttr(self.__module,
                                      f"{self.__attr_path}.{attr_path}")

    def __getattr__(self, key: str):
        getattr(self.__module, f"{self.__attr_path}.{key}")

        raise AssertionError("PlaceholderModule should not be used "
                             "when the original module can be imported")


# create a library to hold the custom op
vllm_lib = Library("vllm", "FRAGMENT")  # noqa


def direct_register_custom_op(
        op_name: str,
        op_func: Callable,
        mutates_args: list[str],
        fake_impl: Optional[Callable] = None,
        target_lib: Optional[Library] = None,
        dispatch_key: str = "CUDA",
        tags: Tuple[torch.Tag, ...] = (),
):
    """
    `torch.library.custom_op` can have significant overhead because it
    needs to consider complicated dispatching logic. This function
    directly registers a custom op and dispatches it to the CUDA backend.
    See https://gist.github.com/youkaichao/ecbea9ec9fc79a45d2adce1784d7a9a5
    for more details.

    By default, the custom op is registered to the vLLM library. If you
    want to register it to a different library, you can pass the library
    object to the `target_lib` argument.

    IMPORTANT: the lifetime of the operator is tied to the lifetime of the
    library object. If you want to bind the operator to a different library,
    make sure the library object is alive when the operator is used.
    """
    if not supports_custom_op():
        from vllm.platforms import current_platform
        assert not current_platform.is_cuda_alike(), (
            "cuda platform needs torch>=2.4 to support custom op, "
            "chances are you are using an old version of pytorch "
            "or a custom build of pytorch. It is recommended to "
            "use vLLM in a fresh new environment and let it install "
            "the required dependencies.")
        return

    import torch.library
    if hasattr(torch.library, "infer_schema"):
        schema_str = torch.library.infer_schema(op_func,
                                                mutates_args=mutates_args)
    else:
        # for pytorch 2.4
        import torch._custom_op.impl
        schema_str = torch._custom_op.impl.infer_schema(op_func, mutates_args)
    my_lib = target_lib or vllm_lib
    my_lib.define(op_name + schema_str, tags=tags)
    my_lib.impl(op_name, op_func, dispatch_key=dispatch_key)
    if fake_impl is not None:
        my_lib._register_fake(op_name, fake_impl)


def resolve_obj_by_qualname(qualname: str) -> Any:
    """
    Resolve an object by its fully qualified name.
    """
    module_name, obj_name = qualname.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, obj_name)


def kill_process_tree(pid: int):
    """
    Kills all descendant processes of the given pid by sending SIGKILL.

    Args:
        pid (int): Process ID of the parent process
    """
    try:
        parent = psutil.Process(pid)
    except psutil.NoSuchProcess:
        return

    # Get all children recursively
    children = parent.children(recursive=True)

    # Send SIGKILL to all children first
    for child in children:
        with contextlib.suppress(ProcessLookupError):
            os.kill(child.pid, signal.SIGKILL)

    # Finally kill the parent
    with contextlib.suppress(ProcessLookupError):
        os.kill(pid, signal.SIGKILL)


@dataclass
class MemorySnapshot:
    """Memory snapshot."""
    torch_peak: int = 0
    cuda_memory: int = 0
    torch_memory: int = 0
    non_torch_memory: int = 0
    timestamp: float = 0.0
    auto_measure: bool = True

    def __post_init__(self):
        if self.auto_measure:
            self.measure()

    def measure(self):
        # we measure the torch peak memory usage via allocated_bytes,
        # rather than `torch.cuda.memory_reserved()` .
        # After `torch.cuda.reset_peak_memory_stats()`,
        # `torch.cuda.memory_reserved()` will keep growing, and only shrink
        # when we call `torch.cuda.empty_cache()` or OOM happens.
        self.torch_peak = torch.cuda.memory_stats().get(
            "allocated_bytes.all.peak", 0)

        self.cuda_memory = torch.cuda.mem_get_info(
        )[1] - torch.cuda.mem_get_info()[0]

        # torch.cuda.memory_reserved() is how many bytes
        # PyTorch gets from cuda (by calling cudaMalloc, etc.)
        # this is used to measure the non-torch memory usage
        self.torch_memory = torch.cuda.memory_reserved()

        self.non_torch_memory = self.cuda_memory - self.torch_memory
        self.timestamp = time.time()

    def __sub__(self, other: MemorySnapshot) -> MemorySnapshot:
        return MemorySnapshot(
            torch_peak=self.torch_peak - other.torch_peak,
            cuda_memory=self.cuda_memory - other.cuda_memory,
            torch_memory=self.torch_memory - other.torch_memory,
            non_torch_memory=self.non_torch_memory - other.non_torch_memory,
            timestamp=self.timestamp - other.timestamp,
            auto_measure=False,
        )


@dataclass
class MemoryProfilingResult:
    """Memory profiling result. All numbers are in bytes.
    """
    non_kv_cache_memory: int = 0
    torch_peak_increase: int = 0
    non_torch_increase: int = 0
    weights_memory: float = 0
    before_create: MemorySnapshot = field(default_factory=MemorySnapshot)
    before_profile: MemorySnapshot = field(default_factory=MemorySnapshot)
    after_profile: MemorySnapshot = field(default_factory=MemorySnapshot)
    profile_time: float = 0.0


@contextlib.contextmanager
def memory_profiling(
        baseline_snapshot: MemorySnapshot,
        weights_memory: int) -> Generator[MemoryProfilingResult, None, None]:
    """Memory profiling context manager.
    baseline_snapshot: the memory snapshot before the current vLLM instance.
    weights_memory: memory used by PyTorch when loading the model weights.
        Note that, before loading the model weights, we also initialize the device
        and distributed environment, which may consume some memory. This part is not
        included in the weights_memory because PyTorch does not control it.

    The memory in one GPU can be classified into 3 categories:
    1. memory used by anything other than the current vLLM instance.
    2. memory used by torch in the current vLLM instance.
    3. memory used in the current vLLM instance, but not by torch.

    A quantitive example:

    Before creating the current vLLM instance:
        category 1: 1 GiB
        category 2: 0 GiB
        category 3: 0 GiB

    After creating the current vLLM instance and loading the model,
    (i.e. before profiling):
        category 1: 1 GiB
        category 2: 2 GiB (model weights take 2 GiB)
        category 3: 0.5 GiB (memory used by NCCL)

    During profiling (peak):
        category 1: 1 GiB
        category 2: 4 GiB (peak activation tensors take 2 GiB)
        category 3: 1 GiB (memory used by NCCL + buffers for some attention backends)

    After profiling:
        category 1: 1 GiB
        category 2: 3 GiB (after garbage-collecting activation tensors)
        category 3: 1 GiB (memory used by NCCL + buffers for some attention backends)

    In this case, non-kv cache takes 5 GiB in total, including:
    a. 2 GiB used by the model weights (category 2)
    b. 2 GiB reserved for the peak activation tensors (category 2)
    c. 1 GiB used by non-torch components (category 3)

    The memory used for loading weights (a.) is directly given from the argument `weights_memory`.

    The increase of `torch.cuda.memory_stats()["allocated_bytes.all.peak"]` during profiling gives (b.).

    The increase of `non_torch_memory` from creating the current vLLM instance until after profiling to get (c.).
    """ # noqa
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    result = MemoryProfilingResult()

    result.before_create = baseline_snapshot
    # the part of memory used for holding the model weights
    result.weights_memory = weights_memory

    result.before_profile.measure()

    yield result

    gc.collect()
    torch.cuda.empty_cache()

    result.after_profile.measure()

    diff_profile = result.after_profile - result.before_profile
    diff_from_create = result.after_profile - result.before_create
    result.torch_peak_increase = diff_profile.torch_peak
    result.non_torch_increase = diff_from_create.non_torch_memory
    result.profile_time = diff_profile.timestamp
    result.non_kv_cache_memory = result.non_torch_increase + result.torch_peak_increase + result.weights_memory  # noqa


# Adapted from: https://github.com/sgl-project/sglang/blob/v0.4.1/python/sglang/srt/utils.py#L630 # noqa: E501
def set_ulimit(target_soft_limit=65535):
    if sys.platform.startswith('win'):
        logger.info("Windows detected, skipping ulimit adjustment.")
        return

    import resource
    resource_type = resource.RLIMIT_NOFILE
    current_soft, current_hard = resource.getrlimit(resource_type)

    if current_soft < target_soft_limit:
        try:
            resource.setrlimit(resource_type,
                               (target_soft_limit, current_hard))
        except ValueError as e:
            logger.warning(
                "Found ulimit of %s and failed to automatically increase "
                "with error %s. This can cause fd limit errors like "
                "`OSError: [Errno 24] Too many open files`. Consider "
                "increasing with ulimit -n", current_soft, e)


# Adapted from: https://github.com/sgl-project/sglang/blob/v0.4.1/python/sglang/utils.py#L28 # noqa: E501
def get_exception_traceback():
    etype, value, tb = sys.exc_info()
    err_str = "".join(traceback.format_exception(etype, value, tb))
    return err_str


def split_zmq_path(path: str) -> Tuple[str, str, str]:
    """Split a zmq path into its parts."""
    parsed = urlparse(path)
    if not parsed.scheme:
        raise ValueError(f"Invalid zmq path: {path}")

    scheme = parsed.scheme
    host = parsed.hostname or ""
    port = str(parsed.port or "")

    if scheme == "tcp" and not all((host, port)):
        # The host and port fields are required for tcp
        raise ValueError(f"Invalid zmq path: {path}")

    if scheme != "tcp" and port:
        # port only makes sense with tcp
        raise ValueError(f"Invalid zmq path: {path}")

    return scheme, host, port


def make_zmq_path(scheme: str, host: str, port: Optional[int] = None) -> str:
    """Make a ZMQ path from its parts.

    Args:
        scheme: The ZMQ transport scheme (e.g. tcp, ipc, inproc).
        host: The host - can be an IPv4 address, IPv6 address, or hostname.
        port: Optional port number, only used for TCP sockets.

    Returns:
        A properly formatted ZMQ path string.
    """
    if not port:
        return f"{scheme}://{host}"
    if is_valid_ipv6_address(host):
        return f"{scheme}://[{host}]:{port}"
    return f"{scheme}://{host}:{port}"


# Adapted from: https://github.com/sgl-project/sglang/blob/v0.4.1/python/sglang/srt/utils.py#L783 # noqa: E501
def make_zmq_socket(
    ctx: Union[zmq.asyncio.Context, zmq.Context],  # type: ignore[name-defined]
    path: str,
    socket_type: Any,
    bind: Optional[bool] = None,
    identity: Optional[bytes] = None,
) -> Union[zmq.Socket, zmq.asyncio.Socket]:  # type: ignore[name-defined]
    """Make a ZMQ socket with the proper bind/connect semantics."""

    mem = psutil.virtual_memory()
    socket = ctx.socket(socket_type)

    # Calculate buffer size based on system memory
    total_mem = mem.total / 1024**3
    available_mem = mem.available / 1024**3
    # For systems with substantial memory (>32GB total, >16GB available):
    # - Set a large 0.5GB buffer to improve throughput
    # For systems with less memory:
    # - Use system default (-1) to avoid excessive memory consumption
    if total_mem > 32 and available_mem > 16:
        buf_size = int(0.5 * 1024**3)  # 0.5GB in bytes
    else:
        buf_size = -1  # Use system default buffer size

    if bind is None:
        bind = socket_type != zmq.PUSH

    if socket_type in (zmq.PULL, zmq.DEALER, zmq.ROUTER):
        socket.setsockopt(zmq.RCVHWM, 0)
        socket.setsockopt(zmq.RCVBUF, buf_size)

    if socket_type in (zmq.PUSH, zmq.DEALER, zmq.ROUTER):
        socket.setsockopt(zmq.SNDHWM, 0)
        socket.setsockopt(zmq.SNDBUF, buf_size)

    if identity is not None:
        socket.setsockopt(zmq.IDENTITY, identity)

    # Determine if the path is a TCP socket with an IPv6 address.
    # Enable IPv6 on the zmq socket if so.
    scheme, host, _ = split_zmq_path(path)
    if scheme == "tcp" and is_valid_ipv6_address(host):
        socket.setsockopt(zmq.IPV6, 1)

    if bind:
        socket.bind(path)
    else:
        socket.connect(path)

    return socket


@contextlib.contextmanager
def zmq_socket_ctx(
    path: str,
    socket_type: Any,
    bind: Optional[bool] = None,
    linger: int = 0,
    identity: Optional[bytes] = None,
) -> Iterator[zmq.Socket]:
    """Context manager for a ZMQ socket"""

    ctx = zmq.Context()  # type: ignore[attr-defined]
    try:
        yield make_zmq_socket(ctx,
                              path,
                              socket_type,
                              bind=bind,
                              identity=identity)
    except KeyboardInterrupt:
        logger.debug("Got Keyboard Interrupt.")

    finally:
        ctx.destroy(linger=linger)


def is_in_ray_actor():
    """Check if we are in a Ray actor."""

    try:
        import ray
        return (ray.is_initialized()
                and ray.get_runtime_context().get_actor_id() is not None)
    except ImportError:
        return False


def _maybe_force_spawn():
    """Check if we need to force the use of the `spawn` multiprocessing start
    method.
    """
    if os.environ.get("VLLM_WORKER_MULTIPROC_METHOD") == "spawn":
        return

    reason = None
    if cuda_is_initialized():
        reason = "CUDA is initialized"
    elif is_in_ray_actor():
        # even if we choose to spawn, we need to pass the ray address
        # to the subprocess so that it knows how to connect to the ray cluster.
        # env vars are inherited by subprocesses, even if we use spawn.
        import ray
        os.environ["RAY_ADDRESS"] = ray.get_runtime_context().gcs_address
        reason = "In a Ray actor and can only be spawned"

    if reason is not None:
        logger.warning(
            "We must use the `spawn` multiprocessing start method. "
            "Overriding VLLM_WORKER_MULTIPROC_METHOD to 'spawn'. "
            "See https://docs.vllm.ai/en/latest/usage/"
            "troubleshooting.html#python-multiprocessing "
            "for more information. Reason: %s", reason)
        os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"


def get_mp_context():
    """Get a multiprocessing context with a particular method (spawn or fork).
    By default we follow the value of the VLLM_WORKER_MULTIPROC_METHOD to
    determine the multiprocessing method (default is fork). However, under
    certain conditions, we may enforce spawn and override the value of
    VLLM_WORKER_MULTIPROC_METHOD.
    """
    _maybe_force_spawn()
    mp_method = envs.VLLM_WORKER_MULTIPROC_METHOD
    return multiprocessing.get_context(mp_method)


def bind_kv_cache(
        ctx: dict[str, Any],
        kv_cache: list[list[torch.Tensor]],  # [virtual_engine][layer_index]
) -> None:
    # Bind the kv_cache tensor to Attention modules, similar to
    # ctx[layer_name].kv_cache[ve]=kv_cache[ve][extract_layer_index(layer_name)]
    # Special things handled here:
    # 1. Some models have non-attention layers, e.g., Jamba
    # 2. Pipeline parallelism, each rank only has a subset of layers
    # 3. Encoder attention has no kv cache
    # 4. Encoder-decoder models, encoder-decoder attention and decoder-only
    #    attention of the same layer (e.g., bart's decoder.layers.1.self_attn
    #    and decoder.layers.1.encoder_attn) is mapped to the same kv cache
    #    tensor
    from vllm.attention import AttentionType
    from vllm.model_executor.models.utils import extract_layer_index
    layer_need_kv_cache = [
        layer_name for layer_name in ctx
        if (hasattr(ctx[layer_name], 'attn_type') and ctx[layer_name].attn_type
            in (AttentionType.DECODER, AttentionType.ENCODER_DECODER))
    ]
    layer_index_sorted = sorted(
        set(
            extract_layer_index(layer_name)
            for layer_name in layer_need_kv_cache))
    for layer_name in layer_need_kv_cache:
        kv_cache_idx = layer_index_sorted.index(
            extract_layer_index(layer_name))
        forward_ctx = ctx[layer_name]
        assert len(forward_ctx.kv_cache) == len(kv_cache)
        for ve, ve_kv_cache in enumerate(kv_cache):
            forward_ctx.kv_cache[ve] = ve_kv_cache[kv_cache_idx]


def run_method(obj: Any, method: Union[str, bytes, Callable], args: tuple[Any],
               kwargs: dict[str, Any]) -> Any:
    """
    Run a method of an object with the given arguments and keyword arguments.
    If the method is string, it will be converted to a method using getattr.
    If the method is serialized bytes and will be deserialized using
    cloudpickle.
    If the method is a callable, it will be called directly.
    """
    if isinstance(method, bytes):
        func = partial(cloudpickle.loads(method), obj)
    elif isinstance(method, str):
        try:
            func = getattr(obj, method)
        except AttributeError:
            raise NotImplementedError(f"Method {method!r} is not"
                                      " implemented.") from None
    else:
        func = partial(method, obj)  # type: ignore
    return func(*args, **kwargs)


def import_pynvml():
    """
    Historical comments:

    libnvml.so is the library behind nvidia-smi, and
    pynvml is a Python wrapper around it. We use it to get GPU
    status without initializing CUDA context in the current process.
    Historically, there are two packages that provide pynvml:
    - `nvidia-ml-py` (https://pypi.org/project/nvidia-ml-py/): The official
        wrapper. It is a dependency of vLLM, and is installed when users
        install vLLM. It provides a Python module named `pynvml`.
    - `pynvml` (https://pypi.org/project/pynvml/): An unofficial wrapper.
        Prior to version 12.0, it also provides a Python module `pynvml`,
        and therefore conflicts with the official one. What's worse,
        the module is a Python package, and has higher priority than
        the official one which is a standalone Python file.
        This causes errors when both of them are installed.
        Starting from version 12.0, it migrates to a new module
        named `pynvml_utils` to avoid the conflict.
    It is so confusing that many packages in the community use the
    unofficial one by mistake, and we have to handle this case.
    For example, `nvcr.io/nvidia/pytorch:24.12-py3` uses the unofficial
    one, and it will cause errors, see the issue
    https://github.com/vllm-project/vllm/issues/12847 for example.
    After all the troubles, we decide to copy the official `pynvml`
    module to our codebase, and use it directly.
    """
    import vllm.third_party.pynvml as pynvml
    return pynvml


def warn_for_unimplemented_methods(cls: type[T]) -> type[T]:
    """
    A replacement for `abc.ABC`.
    When we use `abc.ABC`, subclasses will fail to instantiate
    if they do not implement all abstract methods.
    Here, we only require `raise NotImplementedError` in the
    base class, and log a warning if the method is not implemented
    in the subclass.
    """

    original_init = cls.__init__

    def find_unimplemented_methods(self: object):
        unimplemented_methods = []
        for attr_name in dir(self):
            # bypass inner method
            if attr_name.startswith('_'):
                continue

            try:
                attr = getattr(self, attr_name)
                # get the func of callable method
                if callable(attr):
                    attr_func = attr.__func__
            except AttributeError:
                continue
            src = inspect.getsource(attr_func)
            if "NotImplementedError" in src:
                unimplemented_methods.append(attr_name)
        if unimplemented_methods:
            method_names = ','.join(unimplemented_methods)
            msg = (f"Methods {method_names} not implemented in {self}")
            logger.warning(msg)

    @wraps(original_init)
    def wrapped_init(self, *args, **kwargs) -> None:
        original_init(self, *args, **kwargs)
        find_unimplemented_methods(self)

    type.__setattr__(cls, '__init__', wrapped_init)
    return cls


class LazyLoader(types.ModuleType):
    """
    LazyLoader module borrowed from Tensorflow
    https://github.com/tensorflow/tensorflow/blob/main/tensorflow/python/util/lazy_loader.py
    with a addition of "module caching".

    Lazily import a module, mainly to avoid pulling in large dependencies.
    Modules such as `xgrammar` might do additional side effects, so we
    only want to use this when it is needed, delaying all eager effects
    """

    def __init__(
        self,
        local_name: str,
        parent_module_globals: dict[str, Any],
        name: str,
    ):
        self._local_name = local_name
        self._parent_module_globals = parent_module_globals
        self._module: types.ModuleType | None = None

        super().__init__(str(name))

    def _load(self) -> types.ModuleType:
        # Import the target module and insert it into the parent's namespace
        try:
            module = importlib.import_module(self.__name__)
            self._parent_module_globals[self._local_name] = module
            # The additional add to sys.modules
            # ensures library is actually loaded.
            sys.modules[self._local_name] = module
        except ModuleNotFoundError as err:
            raise err from None

        # Update this object's dict so that if someone keeps a
        # reference to the LazyLoader, lookups are efficient
        # (__getattr__ is only called on lookups that fail).
        self.__dict__.update(module.__dict__)
        return module

    def __getattr__(self, item: Any) -> Any:
        if self._module is None:
            self._module = self._load()
        return getattr(self._module, item)

    def __dir__(self) -> list[str]:
        if self._module is None:
            self._module = self._load()
        return dir(self._module)


def swap_dict_values(obj: dict[_K, _V], key1: _K, key2: _K) -> None:
    """
    Helper function to swap values for two keys
    """
    v1 = obj.get(key1)
    v2 = obj.get(key2)
    if v1 is not None:
        obj[key2] = v1
    else:
        obj.pop(key2, None)
    if v2 is not None:
        obj[key1] = v2
    else:
        obj.pop(key1, None)


@contextlib.contextmanager
def cprofile_context(save_file: Optional[str] = None):
    """Run a cprofile

    Args:
        save_file: path to save the profile result. "1" or
          None will result in printing to stdout.
    """
    import cProfile

    prof = cProfile.Profile()
    prof.enable()

    try:
        yield
    finally:
        prof.disable()
        if save_file and save_file != "1":
            prof.dump_stats(save_file)
        else:
            prof.print_stats(sort="cumtime")


def cprofile(save_file: Optional[str] = None, enabled: bool = True):
    """Decorator to profile a Python method using cProfile.

    Args:
        save_file: Path to save the profile result.
            If "1", None, or "", results will be printed to stdout.
        enabled: Set to false to turn this into a no-op
    """

    def decorator(func: Callable):

        @wraps(func)
        def wrapper(*args, **kwargs):
            if not enabled:
                # If profiling is disabled, just call the function directly.
                return func(*args, **kwargs)

            with cprofile_context(save_file):
                return func(*args, **kwargs)

        return wrapper

    return decorator


# Only relevant for models using ALiBi (e.g, MPT)
def check_use_alibi(model_config: ModelConfig) -> bool:
    cfg = model_config.hf_text_config
    return (getattr(cfg, "alibi", False)  # Falcon
            or ("BloomForCausalLM" in getattr(model_config.hf_config,
                                              "architectures", []))  # Bloom
            or getattr(cfg, "position_encoding_type", "") ==
            "alibi"  # codellm_1b_alibi
            or (hasattr(cfg, "attn_config")  # MPT
                and ((isinstance(cfg.attn_config, dict)
                      and cfg.attn_config.get("alibi", False)) or
                     (not isinstance(cfg.attn_config, dict)
                      and getattr(cfg.attn_config, "alibi", False)))))


def sha256(input) -> int:
    """Hash any picklable Python object using SHA-256.

    The input is serialized using pickle before hashing, which allows
    arbitrary Python objects to be used. Note that this function does
    not use a hash seed—if you need one, prepend it explicitly to the input.

    Args:
        input: Any picklable Python object.

    Returns:
        An integer representing the SHA-256 hash of the serialized input.
    """
    input_bytes = pickle.dumps(input, protocol=pickle.HIGHEST_PROTOCOL)
    return int.from_bytes(hashlib.sha256(input_bytes).digest(),
                          byteorder="big")


def is_torch_equal_or_newer(target: str) -> bool:
    """Check if the installed torch version is >= the target version.

    Args:
        target: a version string, like "2.6.0".

    Returns:
        Whether the condition meets.
    """
    try:
        torch_version = version.parse(str(torch.__version__))
        return torch_version >= version.parse(target)
    except Exception:
        # Fallback to PKG-INFO to load the package info, needed by the doc gen.
        return Version(importlib.metadata.version('torch')) >= Version(target)
