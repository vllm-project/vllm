# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import contextlib
import datetime
import enum
import gc
import getpass
import hashlib
import importlib
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
import uuid
import warnings
import weakref
from argparse import (
    Action,
    ArgumentDefaultsHelpFormatter,
    ArgumentParser,
    ArgumentTypeError,
    RawDescriptionHelpFormatter,
    _ArgumentGroup,
)
from collections import defaultdict
from collections.abc import (
    Callable,
    Collection,
    Generator,
    Iterator,
    Sequence,
)
from concurrent.futures.process import ProcessPoolExecutor
from dataclasses import dataclass, field
from functools import cache, lru_cache, partial, wraps
from pathlib import Path
from typing import TYPE_CHECKING, Any, TextIO, TypeVar
from urllib.parse import urlparse
from uuid import uuid4

import cbor2
import cloudpickle
import numpy as np
import numpy.typing as npt
import psutil
import regex as re
import setproctitle
import torch
import torch.types
import yaml
import zmq
import zmq.asyncio
from packaging import version
from packaging.version import Version
from torch.library import Library

import vllm.envs as envs
from vllm.logger import enable_trace_function_call, init_logger
from vllm.ray.lazy_utils import is_in_ray_actor

if TYPE_CHECKING:
    from argparse import Namespace

    from vllm.config import ModelConfig, VllmConfig
    from vllm.sequence import IntermediateTensors
else:
    Namespace = object

    ModelConfig = object
    VllmConfig = object
    IntermediateTensors = object

logger = init_logger(__name__)

# This value is chosen to have a balance between ITL and TTFT. Note it is
# not optimized for throughput.
DEFAULT_MAX_NUM_BATCHED_TOKENS = 2048
POOLING_MODEL_MAX_NUM_BATCHED_TOKENS = 32768
MULTIMODAL_MODEL_MAX_NUM_BATCHED_TOKENS = 5120

# Constants related to forcing the attention backend selection

# String name of register which may be set in order to
# force auto-selection of attention backend by Attention
# wrapper
STR_BACKEND_ENV_VAR: str = "VLLM_ATTENTION_BACKEND"

# Possible string values of STR_BACKEND_ENV_VAR
# register, corresponding to possible backends
STR_FLASHINFER_ATTN_VAL: str = "FLASHINFER"
STR_TORCH_SDPA_ATTN_VAL: str = "TORCH_SDPA"
STR_XFORMERS_ATTN_VAL: str = "XFORMERS"
STR_FLASH_ATTN_VAL: str = "FLASH_ATTN"
STR_INVALID_VAL: str = "INVALID"

MB_bytes = 1_000_000
"""The number of bytes in one megabyte (MB)."""

MiB_bytes = 1 << 20
"""The number of bytes in one mebibyte (MiB)."""

GB_bytes = 1_000_000_000
"""The number of bytes in one gigabyte (GB)."""

GiB_bytes = 1 << 30
"""The number of bytes in one gibibyte (GiB)."""

# ANSI color codes
CYAN = "\033[1;36m"
RESET = "\033[0;0m"

STR_DTYPE_TO_TORCH_DTYPE = {
    "float32": torch.float32,
    "half": torch.half,
    "bfloat16": torch.bfloat16,
    "float": torch.float,
    "fp8": torch.uint8,
    "fp8_e4m3": torch.uint8,
    "fp8_e5m2": torch.uint8,
    "int8": torch.int8,
    "fp8_inc": torch.float8_e4m3fn,
    "fp8_ds_mla": torch.uint8,
}

TORCH_DTYPE_TO_NUMPY_DTYPE = {
    torch.float16: np.float16,
    torch.float32: np.float32,
    torch.float64: np.float64,
    torch.uint8: np.uint8,
    torch.int32: np.int32,
    torch.int64: np.int64,
}


@contextlib.contextmanager
def set_default_torch_num_threads(num_threads: int):
    """Sets the default number of threads for PyTorch to the given value."""
    old_num_threads = torch.get_num_threads()
    torch.set_num_threads(num_threads)
    yield
    torch.set_num_threads(old_num_threads)


T = TypeVar("T")
U = TypeVar("U")


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


@cache
def get_max_shared_memory_bytes(gpu: int = 0) -> int:
    """Returns the maximum shared memory per thread block in bytes."""
    from vllm import _custom_ops as ops

    max_shared_mem = ops.get_max_shared_memory_per_block_device_attribute(gpu)
    # value 0 will cause MAX_SEQ_LEN become negative and test_attention.py
    # will fail
    assert max_shared_mem > 0, "max_shared_mem can not be zero"
    return int(max_shared_mem)


def get_cpu_memory() -> int:
    """Returns the total CPU memory of the node in bytes."""
    return psutil.virtual_memory().total


def random_uuid() -> str:
    return str(uuid.uuid4().hex)


def close_sockets(sockets: Sequence[zmq.Socket | zmq.asyncio.Socket]):
    for sock in sockets:
        if sock is not None:
            sock.close(linger=0)


def get_ip() -> str:
    host_ip = envs.VLLM_HOST_IP
    if "HOST_IP" in os.environ and "VLLM_HOST_IP" not in os.environ:
        logger.warning(
            "The environment variable HOST_IP is deprecated and ignored, as"
            " it is often used by Docker and other software to"
            " interact with the container's network stack. Please "
            "use VLLM_HOST_IP instead to set the IP address for vLLM processes"
            " to communicate with each other."
        )
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
        stacklevel=2,
    )
    return "0.0.0.0"


def test_loopback_bind(address, family):
    try:
        s = socket.socket(family, socket.SOCK_DGRAM)
        s.bind((address, 0))  # Port 0 = auto assign
        s.close()
        return True
    except OSError:
        return False


def get_loopback_ip() -> str:
    loopback_ip = envs.VLLM_LOOPBACK_IP
    if loopback_ip:
        return loopback_ip

    # VLLM_LOOPBACK_IP is not set, try to get it based on network interface

    if test_loopback_bind("127.0.0.1", socket.AF_INET):
        return "127.0.0.1"
    elif test_loopback_bind("::1", socket.AF_INET6):
        return "::1"
    else:
        raise RuntimeError(
            "Neither 127.0.0.1 nor ::1 are bound to a local interface. "
            "Set the VLLM_LOOPBACK_IP environment variable explicitly."
        )


def is_valid_ipv6_address(address: str) -> bool:
    try:
        ipaddress.IPv6Address(address)
        return True
    except ValueError:
        return False


def split_host_port(host_port: str) -> tuple[str, int]:
    # ipv6
    if host_port.startswith("["):
        host, port = host_port.rsplit("]", 1)
        host = host[1:]
        port = port.split(":")[1]
        return host, int(port)
    else:
        host, port = host_port.split(":")
        return host, int(port)


def join_host_port(host: str, port: int) -> str:
    if is_valid_ipv6_address(host):
        return f"[{host}]:{port}"
    else:
        return f"{host}:{port}"


def get_distributed_init_method(ip: str, port: int) -> str:
    return get_tcp_uri(ip, port)


def get_tcp_uri(ip: str, port: int) -> str:
    if is_valid_ipv6_address(ip):
        return f"tcp://[{ip}]:{port}"
    else:
        return f"tcp://{ip}:{port}"


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


def get_open_ports_list(count: int = 5) -> list[int]:
    """Get a list of open ports."""
    ports = set[int]()
    while len(ports) < count:
        ports.add(get_open_port())
    return list(ports)


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
                logger.info("Port %d is already in use, trying port %d", port - 1, port)
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


def find_process_using_port(port: int) -> psutil.Process | None:
    # TODO: We can not check for running processes with network
    # port on macOS. Therefore, we can not have a full graceful shutdown
    # of vLLM. For now, let's not look for processes in this case.
    # Ref: https://www.florianreinhard.de/accessdenied-in-psutil/
    if sys.platform.startswith("darwin"):
        return None

    our_pid = os.getpid()
    for conn in psutil.net_connections():
        if conn.laddr.port == port and (conn.pid is not None and conn.pid != our_pid):
            try:
                return psutil.Process(conn.pid)
            except psutil.NoSuchProcess:
                return None
    return None


def update_environment_variables(envs: dict[str, str]):
    for k, v in envs.items():
        if k in os.environ and os.environ[k] != v:
            logger.warning(
                "Overwriting environment variable %s from '%s' to '%s'",
                k,
                os.environ[k],
                v,
            )
        os.environ[k] = v


def cdiv(a: int, b: int) -> int:
    """Ceiling division."""
    return -(a // -b)


def next_power_of_2(n) -> int:
    """The next power of 2 (inclusive)"""
    if n < 1:
        return 1
    return 1 << (n - 1).bit_length()


def prev_power_of_2(n: int) -> int:
    """The previous power of 2 (inclusive)"""
    if n <= 0:
        return 0
    return 1 << (n.bit_length() - 1)


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
    # -----|-------------|-------------------
    # Inf | N/A         | s.11111.00
    # NaN | s.1111.111  | s.11111.{01,10,11}
    from vllm import _custom_ops as ops

    tensor_tmp = torch.empty_like(tensor, dtype=torch.float16)
    tensor_tmp.uniform_(low, high)
    ops.convert_fp8(tensor, tensor_tmp)
    del tensor_tmp


def get_kv_cache_torch_dtype(
    cache_dtype: str | torch.dtype | None,
    model_dtype: str | torch.dtype | None = None,
) -> torch.dtype:
    if isinstance(cache_dtype, str):
        if cache_dtype == "auto":
            if isinstance(model_dtype, str) and model_dtype in STR_DTYPE_TO_TORCH_DTYPE:
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
    cache_dtype: str | torch.dtype | None,
    model_dtype: str | torch.dtype | None = None,
    seed: int | None = None,
    device: str | None = "cuda",
    cache_layout: str | None = "NHD",
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    from vllm.platforms import current_platform

    current_platform.seed_everything(seed)

    dtype = get_kv_cache_torch_dtype(cache_dtype, model_dtype)
    generic_kv_cache_shape = (num_blocks, 2, block_size, num_heads, head_size)
    assert cache_layout in ("NHD", "HND")
    stride_order = (0, 1, 2, 3, 4) if cache_layout == "NHD" else (0, 1, 3, 2, 4)

    kv_cache_allocation_shape = tuple(generic_kv_cache_shape[i] for i in stride_order)
    scale = head_size**-0.5

    key_caches: list[torch.Tensor] = []
    value_caches: list[torch.Tensor] = []

    for _ in range(num_layers):
        key_value_cache = torch.empty(
            size=kv_cache_allocation_shape, dtype=dtype, device=device
        ).permute(*stride_order)
        if cache_dtype in ["auto", "half", "bfloat16", "float"]:
            key_value_cache.uniform_(-scale, scale)
        elif cache_dtype == "fp8":
            _generate_random_fp8(key_value_cache, -scale, scale)
        else:
            raise ValueError(f"Does not support key cache of type {cache_dtype}")
        key_caches.append(key_value_cache[:, 0])
        value_caches.append(key_value_cache[:, 1])
    return key_caches, value_caches


def create_kv_caches_with_random(
    num_blocks: int,
    block_size: int,
    num_layers: int,
    num_heads: int,
    head_size: int,
    cache_dtype: str | torch.dtype | None,
    model_dtype: str | torch.dtype | None = None,
    seed: int | None = None,
    device: str | None = "cuda",
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    if cache_dtype == "fp8" and head_size % 16:
        raise ValueError(
            f"Does not support key cache of type fp8 with head_size {head_size}"
        )
    from vllm.platforms import current_platform

    current_platform.seed_everything(seed)

    dtype = get_kv_cache_torch_dtype(cache_dtype, model_dtype)

    scale = head_size**-0.5
    x = 16 // torch.tensor([], dtype=dtype).element_size()
    key_cache_shape = (num_blocks, num_heads, head_size // x, block_size, x)
    key_caches: list[torch.Tensor] = []
    for _ in range(num_layers):
        key_cache = torch.empty(size=key_cache_shape, dtype=dtype, device=device)
        if cache_dtype in ["auto", "half", "bfloat16", "float"]:
            key_cache.uniform_(-scale, scale)
        elif cache_dtype == "fp8":
            _generate_random_fp8(key_cache, -scale, scale)
        else:
            raise ValueError(f"Does not support key cache of type {cache_dtype}")
        key_caches.append(key_cache)

    value_cache_shape = (num_blocks, num_heads, head_size, block_size)
    value_caches: list[torch.Tensor] = []
    for _ in range(num_layers):
        value_cache = torch.empty(size=value_cache_shape, dtype=dtype, device=device)
        if cache_dtype in ["auto", "half", "bfloat16", "float"]:
            value_cache.uniform_(-scale, scale)
        elif cache_dtype == "fp8":
            _generate_random_fp8(value_cache, -scale, scale)
        else:
            raise ValueError(f"Does not support value cache of type {cache_dtype}")
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
    def __init__(self, device: torch.types.Device | None = None):
        self.device = device

    def current_memory_usage(self) -> float:
        # Return the memory usage in bytes.
        from vllm.platforms import current_platform

        gc.collect()
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
    max_len: int | None = None,
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
        padded_x[ind, : len(blocktb)] = blocktb

    return padded_x


def make_tensor_with_pad(
    x: list[list[T]],
    pad: T,
    dtype: torch.dtype,
    *,
    max_len: int | None = None,
    device: str | torch.device | None = None,
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
    target_device: str | torch.device,
    pin_memory: bool,
) -> torch.Tensor:
    """Asynchronously create a tensor and copy it from host to device."""
    t = torch.tensor(data, dtype=dtype, pin_memory=pin_memory, device="cpu")
    return t.to(device=target_device, non_blocking=True)


def get_dtype_size(dtype: torch.dtype) -> int:
    """Get the size of the data type in bytes."""
    return torch.tensor([], dtype=dtype).element_size()


# bool = 0, int = 1, float = 2, complex = 3
def _get_precision_level(dtype: torch.dtype) -> int:
    # NOTE: Complex dtypes return `is_floating_point=False`
    return (dtype != torch.bool) + dtype.is_floating_point + dtype.is_complex * 2


def is_lossless_cast(src_dtype: torch.dtype, tgt_dtype: torch.dtype):
    """
    Test whether it is lossless to cast a tensor from
    `src_dtype` to `tgt_dtype`.
    """
    if src_dtype == tgt_dtype:
        return True

    src_level = _get_precision_level(src_dtype)
    tgt_level = _get_precision_level(tgt_dtype)

    if src_level < tgt_level:
        return True
    if src_level > tgt_level:
        return False

    # Compare integral types
    if not src_dtype.is_floating_point and not src_dtype.is_complex:
        src_info = torch.iinfo(src_dtype)
        tgt_info = torch.iinfo(tgt_dtype)
        return src_info.min >= tgt_info.min and src_info.max <= tgt_info.max

    # Compare floating-point types
    src_info = torch.finfo(src_dtype)
    tgt_info = torch.finfo(tgt_dtype)
    return (
        src_info.min >= tgt_info.min
        and src_info.max <= tgt_info.max
        and src_info.resolution >= tgt_info.resolution
    )


def common_broadcastable_dtype(dtypes: Collection[torch.dtype]):
    """
    Get the common `dtype` where all of the other `dtypes` can be
    cast to it without losing any information.
    """
    return max(
        dtypes,
        key=lambda dtype: sum(is_lossless_cast(dt, dtype) for dt in dtypes),
    )


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
            "Found nccl from environment variable VLLM_NCCL_SO_PATH=%s", so_file
        )
    else:
        if torch.version.cuda is not None:
            so_file = "libnccl.so.2"
        elif torch.version.hip is not None:
            so_file = "librccl.so.1"
        else:
            raise ValueError("NCCL only supports CUDA and ROCm backends.")
        logger.debug_once("Found nccl from library %s", so_file)
    return so_file


def find_nccl_include_paths() -> list[str] | None:
    """
    We either use the nccl.h specified by the `VLLM_NCCL_INCLUDE_PATH`
    environment variable, or we find the library file brought by
    nvidia-nccl-cuXX. load_inline by default uses
    torch.utils.cpp_extension.include_paths
    """
    paths: list[str] = []
    inc = envs.VLLM_NCCL_INCLUDE_PATH
    if inc and os.path.isdir(inc):
        paths.append(inc)

    try:
        spec = importlib.util.find_spec("nvidia.nccl")
        if spec and getattr(spec, "submodule_search_locations", None):
            for loc in spec.submodule_search_locations:
                inc_dir = os.path.join(loc, "include")
                if os.path.exists(os.path.join(inc_dir, "nccl.h")):
                    paths.append(inc_dir)
    except Exception:
        pass

    seen = set()
    out: list[str] = []
    for p in paths:
        if p and p not in seen:
            out.append(p)
            seen.add(p)
    return out or None


prev_set_stream = torch.cuda.set_stream

_current_stream_tls = threading.local()


def _patched_set_stream(stream: torch.cuda.Stream) -> None:
    _current_stream_tls.value = stream
    prev_set_stream(stream)


torch.cuda.set_stream = _patched_set_stream


class _StreamPlaceholder:
    def __init__(self):
        self.synchronize = lambda: None


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

    if not hasattr(_current_stream_tls, "value") or _current_stream_tls.value is None:
        # when this function is called before any stream is set,
        # we return the default stream.
        # On ROCm using the default 0 stream in combination with RCCL
        # is hurting performance. Therefore creating a dedicated stream
        # per process
        if current_platform.is_rocm():
            # torch.cuda.set_stream here is the alias of _pathed_set_stream
            torch.cuda.set_stream(torch.cuda.Stream())
        elif current_platform.is_cpu():
            _current_stream_tls.value = _StreamPlaceholder()
        else:
            current_stream = current_platform.current_stream
            if current_stream is not None:
                _current_stream_tls.value = current_stream()
            else:
                raise ValueError(
                    "Fail to set current stream, current platform "
                    "may not support current_stream with torch API"
                )
    return _current_stream_tls.value


def enable_trace_function_call_for_thread(vllm_config: VllmConfig) -> None:
    """Set up function tracing for the current thread,
    if enabled via the VLLM_TRACE_FUNCTION environment variable
    """

    if envs.VLLM_TRACE_FUNCTION:
        tmp_dir = tempfile.gettempdir()
        # add username to tmp_dir to avoid permission issues
        tmp_dir = os.path.join(tmp_dir, getpass.getuser())
        filename = (
            f"VLLM_TRACE_FUNCTION_for_process_{os.getpid()}"
            f"_thread_{threading.get_ident()}_"
            f"at_{datetime.datetime.now()}.log"
        ).replace(" ", "_")
        log_path = os.path.join(
            tmp_dir, "vllm", f"vllm-instance-{vllm_config.instance_id}", filename
        )
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        enable_trace_function_call(log_path)


@lru_cache(maxsize=8)
def _cuda_device_count_stateless(cuda_visible_devices: str | None = None) -> int:
    # Note: cuda_visible_devices is not used, but we keep it as an argument for
    # LRU Cache purposes.

    # Code below is based on
    # https://github.com/pytorch/pytorch/blob/
    # c1cd946818442aca8c7f812b16d187ce1586c3bc/
    # torch/cuda/__init__.py#L831C1-L831C17
    import torch.cuda

    from vllm.platforms import current_platform

    if not torch.cuda._is_compiled():
        return 0
    if current_platform.is_rocm():
        # ROCm uses amdsmi instead of nvml for stateless device count
        # This requires a sufficiently modern version of Torch 2.4.0
        raw_count = (
            torch.cuda._device_count_amdsmi()
            if (hasattr(torch.cuda, "_device_count_amdsmi"))
            else -1
        )
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


def xpu_is_initialized() -> bool:
    """Check if XPU is initialized."""
    if not torch.xpu._is_compiled():
        return False
    return torch.xpu.is_initialized()


def cuda_get_device_properties(
    device, names: Sequence[str], init_cuda=False
) -> tuple[Any, ...]:
    """Get specified CUDA device property values without initializing CUDA in
    the current process."""
    if init_cuda or cuda_is_initialized():
        props = torch.cuda.get_device_properties(device)
        return tuple(getattr(props, name) for name in names)

    # Run in subprocess to avoid initializing CUDA as a side effect.
    mp_ctx = multiprocessing.get_context("fork")
    with ProcessPoolExecutor(max_workers=1, mp_context=mp_ctx) as executor:
        return executor.submit(cuda_get_device_properties, device, names, True).result()


def weak_bind(
    bound_method: Callable[..., Any],
) -> Callable[..., None]:
    """Make an instance method that weakly references
    its associated instance and no-ops once that
    instance is collected."""
    ref = weakref.ref(bound_method.__self__)  # type: ignore[attr-defined]
    unbound = bound_method.__func__  # type: ignore[attr-defined]

    def weak_bound(*args, **kwargs) -> None:
        if inst := ref():
            unbound(inst, *args, **kwargs)

    return weak_bound


class StoreBoolean(Action):
    def __call__(self, parser, namespace, values, option_string=None):
        if values.lower() == "true":
            setattr(namespace, self.dest, True)
        elif values.lower() == "false":
            setattr(namespace, self.dest, False)
        else:
            raise ValueError(
                f"Invalid boolean value: {values}. Expected 'true' or 'false'."
            )


class SortedHelpFormatter(ArgumentDefaultsHelpFormatter, RawDescriptionHelpFormatter):
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
        text = single_newline.sub(" ", text)
        lines = re.split(multiple_newlines, text)
        return sum([textwrap.wrap(line, width) for line in lines], [])

    def add_arguments(self, actions):
        actions = sorted(actions, key=lambda x: x.option_strings)
        super().add_arguments(actions)


class FlexibleArgumentParser(ArgumentParser):
    """ArgumentParser that allows both underscore and dash in names."""

    _deprecated: set[Action] = set()
    _json_tip: str = (
        "When passing JSON CLI arguments, the following sets of arguments "
        "are equivalent:\n"
        '   --json-arg \'{"key1": "value1", "key2": {"key3": "value2"}}\'\n'
        "   --json-arg.key1 value1 --json-arg.key2.key3 value2\n\n"
        "Additionally, list elements can be passed individually using +:\n"
        '   --json-arg \'{"key4": ["value3", "value4", "value5"]}\'\n'
        "   --json-arg.key4+ value3 --json-arg.key4+='value4,value5'\n\n"
    )
    _search_keyword: str | None = None

    def __init__(self, *args, **kwargs):
        # Set the default "formatter_class" to SortedHelpFormatter
        if "formatter_class" not in kwargs:
            kwargs["formatter_class"] = SortedHelpFormatter
        # Pop kwarg "add_json_tip" to control whether to add the JSON tip
        self.add_json_tip = kwargs.pop("add_json_tip", True)
        super().__init__(*args, **kwargs)

    if sys.version_info < (3, 13):
        # Enable the deprecated kwarg for Python 3.12 and below

        def parse_known_args(self, args=None, namespace=None):
            if args is not None and "--disable-log-requests" in args:
                # Special case warning because the warning below won't trigger
                # if –-disable-log-requests because its value is default.
                logger.warning_once(
                    "argument '--disable-log-requests' is deprecated and "
                    "replaced with '--enable-log-requests'. This will be "
                    "removed in v0.12.0."
                )
            namespace, args = super().parse_known_args(args, namespace)
            for action in FlexibleArgumentParser._deprecated:
                if (
                    hasattr(namespace, dest := action.dest)
                    and getattr(namespace, dest) != action.default
                ):
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

    def format_help(self):
        # Only use custom help formatting for bottom level parsers
        if self._subparsers is not None:
            return super().format_help()

        formatter = self._get_formatter()

        # Handle keyword search of the args
        if (search_keyword := self._search_keyword) is not None:
            # Normalise the search keyword
            search_keyword = search_keyword.lower().replace("_", "-")
            # Return full help if searching for 'all'
            if search_keyword == "all":
                self.epilog = self._json_tip
                return super().format_help()

            # Return group help if searching for a group title
            for group in self._action_groups:
                if group.title and group.title.lower() == search_keyword:
                    formatter.start_section(group.title)
                    formatter.add_text(group.description)
                    formatter.add_arguments(group._group_actions)
                    formatter.end_section()
                    formatter.add_text(self._json_tip)
                    return formatter.format_help()

            # Return matched args if searching for an arg name
            matched_actions = []
            for group in self._action_groups:
                for action in group._group_actions:
                    # search option name
                    if any(
                        search_keyword in opt.lower() for opt in action.option_strings
                    ):
                        matched_actions.append(action)
            if matched_actions:
                formatter.start_section(f"Arguments matching '{search_keyword}'")
                formatter.add_arguments(matched_actions)
                formatter.end_section()
                formatter.add_text(self._json_tip)
                return formatter.format_help()

            # No match found
            formatter.add_text(
                f"No group or arguments matching '{search_keyword}'.\n"
                "Use '--help' to see available groups or "
                "'--help=all' to see all available parameters."
            )
            return formatter.format_help()

        # usage
        formatter.add_usage(self.usage, self._actions, self._mutually_exclusive_groups)

        # description
        formatter.add_text(self.description)

        # positionals, optionals and user-defined groups
        formatter.start_section("Config Groups")
        config_groups = ""
        for group in self._action_groups:
            if not group._group_actions:
                continue
            title = group.title
            description = group.description or ""
            config_groups += f"{title: <24}{description}\n"
        formatter.add_text(config_groups)
        formatter.end_section()

        # epilog
        formatter.add_text(self.epilog)

        # determine help from format above
        return formatter.format_help()

    def parse_args(  # type: ignore[override]
        self,
        args: list[str] | None = None,
        namespace: Namespace | None = None,
    ):
        if args is None:
            args = sys.argv[1:]

        # Check for --model in command line arguments first
        if args and args[0] == "serve":
            try:
                model_idx = next(
                    i
                    for i, arg in enumerate(args)
                    if arg == "--model" or arg.startswith("--model=")
                )
                logger.warning(
                    "With `vllm serve`, you should provide the model as a "
                    "positional argument or in a config file instead of via "
                    "the `--model` option. "
                    "The `--model` option will be removed in v0.13."
                )

                if args[model_idx] == "--model":
                    model_tag = args[model_idx + 1]
                    rest_start_idx = model_idx + 2
                else:
                    model_tag = args[model_idx].removeprefix("--model=")
                    rest_start_idx = model_idx + 1

                # Move <model> to the front, e,g:
                # [Before]
                # vllm serve -tp 2 --model <model> --enforce-eager --port 8001
                # [After]
                # vllm serve <model> -tp 2 --enforce-eager --port 8001
                args = [
                    "serve",
                    model_tag,
                    *args[1:model_idx],
                    *args[rest_start_idx:],
                ]
                print("args", args)
            except StopIteration:
                pass

        if "--config" in args:
            args = self._pull_args_from_config(args)

        def repl(match: re.Match) -> str:
            """Replaces underscores with dashes in the matched string."""
            return match.group(0).replace("_", "-")

        # Everything between the first -- and the first .
        pattern = re.compile(r"(?<=--)[^\.]*")

        # Convert underscores to dashes and vice versa in argument names
        processed_args = list[str]()
        for i, arg in enumerate(args):
            if arg.startswith("--help="):
                FlexibleArgumentParser._search_keyword = arg.split("=", 1)[-1].lower()
                processed_args.append("--help")
            elif arg.startswith("--"):
                if "=" in arg:
                    key, value = arg.split("=", 1)
                    key = pattern.sub(repl, key, count=1)
                    processed_args.append(f"{key}={value}")
                else:
                    key = pattern.sub(repl, arg, count=1)
                    processed_args.append(key)
            elif arg.startswith("-O") and arg != "-O" and arg[2] != ".":
                # allow -O flag to be used without space, e.g. -O3 or -Odecode
                # -O.<...> handled later
                # also handle -O=<mode> here
                mode = arg[3:] if arg[2] == "=" else arg[2:]
                processed_args.append(f"-O.mode={mode}")
            elif (
                arg == "-O"
                and i + 1 < len(args)
                and args[i + 1] in {"0", "1", "2", "3"}
            ):
                # Convert -O <n> to -O.mode <n>
                processed_args.append("-O.mode")
            else:
                processed_args.append(arg)

        def create_nested_dict(keys: list[str], value: str) -> dict[str, Any]:
            """Creates a nested dictionary from a list of keys and a value.

            For example, `keys = ["a", "b", "c"]` and `value = 1` will create:
            `{"a": {"b": {"c": 1}}}`
            """
            nested_dict: Any = value
            for key in reversed(keys):
                nested_dict = {key: nested_dict}
            return nested_dict

        def recursive_dict_update(
            original: dict[str, Any],
            update: dict[str, Any],
        ) -> set[str]:
            """Recursively updates a dictionary with another dictionary.
            Returns a set of duplicate keys that were overwritten.
            """
            duplicates = set[str]()
            for k, v in update.items():
                if isinstance(v, dict) and isinstance(original.get(k), dict):
                    nested_duplicates = recursive_dict_update(original[k], v)
                    duplicates |= {f"{k}.{d}" for d in nested_duplicates}
                elif isinstance(v, list) and isinstance(original.get(k), list):
                    original[k] += v
                else:
                    if k in original:
                        duplicates.add(k)
                    original[k] = v
            return duplicates

        delete = set[int]()
        dict_args = defaultdict[str, dict[str, Any]](dict)
        duplicates = set[str]()
        for i, processed_arg in enumerate(processed_args):
            if i in delete:  # skip if value from previous arg
                continue

            if processed_arg.startswith("-") and "." in processed_arg:
                if "=" in processed_arg:
                    processed_arg, value_str = processed_arg.split("=", 1)
                    if "." not in processed_arg:
                        # False positive, '.' was only in the value
                        continue
                else:
                    value_str = processed_args[i + 1]
                    delete.add(i + 1)

                if processed_arg.endswith("+"):
                    processed_arg = processed_arg[:-1]
                    value_str = json.dumps(list(value_str.split(",")))

                key, *keys = processed_arg.split(".")
                try:
                    value = json.loads(value_str)
                except json.decoder.JSONDecodeError:
                    value = value_str

                # Merge all values with the same key into a single dict
                arg_dict = create_nested_dict(keys, value)
                arg_duplicates = recursive_dict_update(dict_args[key], arg_dict)
                duplicates |= {f"{key}.{d}" for d in arg_duplicates}
                delete.add(i)
        # Filter out the dict args we set to None
        processed_args = [a for i, a in enumerate(processed_args) if i not in delete]
        if duplicates:
            logger.warning("Found duplicate keys %s", ", ".join(duplicates))

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
        assert args.count("--config") <= 1, "More than one config file specified!"

        index = args.index("--config")
        if index == len(args) - 1:
            raise ValueError(
                "No config file specified! \
                             Please check your command-line arguments."
            )

        file_path = args[index + 1]

        config_args = self.load_config_file(file_path)

        # 0th index might be the sub command {serve,chat,complete,...}
        # optionally followed by model_tag (only for serve)
        # followed by config args
        # followed by rest of cli args.
        # maintaining this order will enforce the precedence
        # of cli > config > defaults
        if args[0].startswith("-"):
            # No sub command (e.g., api_server entry point)
            args = config_args + args[0:index] + args[index + 2 :]
        elif args[0] == "serve":
            model_in_cli = len(args) > 1 and not args[1].startswith("-")
            model_in_config = any(arg == "--model" for arg in config_args)

            if not model_in_cli and not model_in_config:
                raise ValueError(
                    "No model specified! Please specify model either "
                    "as a positional argument or in a config file."
                )

            if model_in_cli:
                # Model specified as positional arg, keep CLI version
                args = (
                    [args[0]]
                    + [args[1]]
                    + config_args
                    + args[2:index]
                    + args[index + 2 :]
                )
            else:
                # No model in CLI, use config if available
                args = [args[0]] + config_args + args[1:index] + args[index + 2 :]
        else:
            args = [args[0]] + config_args + args[1:index] + args[index + 2 :]

        return args

    def load_config_file(self, file_path: str) -> list[str]:
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
        extension: str = file_path.split(".")[-1]
        if extension not in ("yaml", "yml"):
            raise ValueError(
                "Config file must be of a yaml/yml type.\
                              %s supplied",
                extension,
            )

        # only expecting a flat dictionary of atomic types
        processed_args: list[str] = []

        config: dict[str, int | str] = {}
        try:
            with open(file_path) as config_file:
                config = yaml.safe_load(config_file)
        except Exception as ex:
            logger.error(
                "Unable to read the config file at %s. \
                Make sure path is correct",
                file_path,
            )
            raise ex

        store_boolean_arguments = [
            action.dest for action in self._actions if isinstance(action, StoreBoolean)
        ]

        for key, value in config.items():
            if isinstance(value, bool) and key not in store_boolean_arguments:
                if value:
                    processed_args.append("--" + key)
            elif isinstance(value, list):
                if value:
                    processed_args.append("--" + key)
                    for item in value:
                        processed_args.append(str(item))
            else:
                processed_args.append("--" + key)
                processed_args.append(str(value))

        return processed_args


# Using dynamo with vLLM doesn't really work well with PyTorch versions < 2.4.0.
# In particular, the FakeScalarType is not supported for earlier versions of
# PyTorch which breaks dynamo for any ops registered using ScalarType.
def supports_dynamo() -> bool:
    base_torch_version = Version(Version(torch.__version__).base_version)
    return base_torch_version >= Version("2.4.0")


# Supports xccl with PyTorch versions >= 2.8.0.dev for XPU platform
def supports_xccl() -> bool:
    return (
        is_torch_equal_or_newer("2.8.0.dev") and torch.distributed.is_xccl_available()
    )


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
    tensors: torch.Tensor
    | list[torch.Tensor]
    | tuple[torch.Tensor]
    | IntermediateTensors,
) -> torch.Tensor | list[Any] | tuple[Any] | Any:
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

    # For IntermediateTensors used in pipeline parallelism
    from vllm.sequence import IntermediateTensors

    if isinstance(tensors, IntermediateTensors):
        ret = IntermediateTensors(
            {key: weak_ref_tensor(val) for key, val in tensors.tensors.items()}
        )
        return ret
    raise ValueError("Invalid type for tensors")


def get_cuda_view_from_cpu_tensor(cpu_tensor: torch.Tensor) -> torch.Tensor:
    """
    Get a CUDA view of a CPU tensor using Unified Virtual Addressing (UVA).
    """
    assert cpu_tensor.is_pinned(), "CPU tensor must be pinned"
    return torch.ops._C.get_cuda_view_from_cpu_tensor(cpu_tensor)


# create a library to hold the custom op
vllm_lib = Library("vllm", "FRAGMENT")  # noqa


def direct_register_custom_op(
    op_name: str,
    op_func: Callable,
    mutates_args: list[str] | None = None,
    fake_impl: Callable | None = None,
    target_lib: Library | None = None,
    dispatch_key: str | None = None,
    tags: tuple[torch.Tag, ...] = (),
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
            "the required dependencies."
        )
        return

    if mutates_args is None:
        mutates_args = []

    if dispatch_key is None:
        from vllm.platforms import current_platform

        dispatch_key = current_platform.dispatch_key

    import torch.library

    if hasattr(torch.library, "infer_schema"):
        schema_str = torch.library.infer_schema(op_func, mutates_args=mutates_args)
    else:
        # for pytorch 2.4
        import torch._custom_op.impl

        schema_str = torch._custom_op.impl.infer_schema(op_func, mutates_args)
    my_lib = target_lib or vllm_lib
    my_lib.define(op_name + schema_str, tags=tags)
    my_lib.impl(op_name, op_func, dispatch_key=dispatch_key)
    if fake_impl is not None:
        my_lib._register_fake(op_name, fake_impl)


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
    free_memory: int = 0
    total_memory: int = 0
    cuda_memory: int = 0
    torch_memory: int = 0
    non_torch_memory: int = 0
    timestamp: float = 0.0
    auto_measure: bool = True

    def __post_init__(self):
        if self.auto_measure:
            self.measure()

    def measure(self):
        from vllm.platforms import current_platform

        # we measure the torch peak memory usage via allocated_bytes,
        # rather than `torch.cuda.memory_reserved()` .
        # After `torch.cuda.reset_peak_memory_stats()`,
        # `torch.cuda.memory_reserved()` will keep growing, and only shrink
        # when we call `torch.cuda.empty_cache()` or OOM happens.
        self.torch_peak = torch.cuda.memory_stats().get("allocated_bytes.all.peak", 0)

        self.free_memory, self.total_memory = torch.cuda.mem_get_info()
        shared_sysmem_device_mem_sms = ((8, 7), (11, 0), (12, 1))  # Orin, Thor, Spark
        if (
            current_platform.is_cuda()
            and current_platform.get_device_capability() in shared_sysmem_device_mem_sms
        ):
            # On UMA (Orin, Thor and Spark) platform,
            # where both CPU and GPU rely on system memory,
            # the cudaMemGetInfo function shows the amount of free system memory
            # rather than what’s actually available.
            # In the case,
            # torch.cuda.mem_get_info() only reports "free" memory,
            # which can be lower than what is actually
            # available due to not including cache memory.
            # There’s also a comprehensive reference page
            # that explains how you can compute the proper value yourself.
            # https://docs.nvidia.com/cuda/cuda-for-tegra-appnote/#estimating-total-allocatable-device-memory-on-an-integrated-gpu-device
            self.free_memory = psutil.virtual_memory().available

        self.cuda_memory = self.total_memory - self.free_memory

        # torch.cuda.memory_reserved() is how many bytes
        # PyTorch gets from cuda (by calling cudaMalloc, etc.)
        # this is used to measure the non-torch memory usage
        self.torch_memory = torch.cuda.memory_reserved()

        self.non_torch_memory = self.cuda_memory - self.torch_memory
        self.timestamp = time.time()

    def __sub__(self, other: "MemorySnapshot") -> "MemorySnapshot":
        return MemorySnapshot(
            torch_peak=self.torch_peak - other.torch_peak,
            free_memory=self.free_memory - other.free_memory,
            total_memory=self.total_memory - other.total_memory,
            cuda_memory=self.cuda_memory - other.cuda_memory,
            torch_memory=self.torch_memory - other.torch_memory,
            non_torch_memory=self.non_torch_memory - other.non_torch_memory,
            timestamp=self.timestamp - other.timestamp,
            auto_measure=False,
        )


@dataclass
class MemoryProfilingResult:
    """Memory profiling result. All numbers are in bytes."""

    non_kv_cache_memory: int = 0
    torch_peak_increase: int = 0
    non_torch_increase: int = 0
    weights_memory: float = 0
    before_create: MemorySnapshot = field(default_factory=MemorySnapshot)
    before_profile: MemorySnapshot = field(default_factory=MemorySnapshot)
    after_profile: MemorySnapshot = field(default_factory=MemorySnapshot)
    profile_time: float = 0.0

    def __repr__(self) -> str:
        return (
            f"Memory profiling takes {self.profile_time:.2f} seconds. "
            f"Total non KV cache memory: "
            f"{(self.non_kv_cache_memory / GiB_bytes):.2f}GiB; "
            f"torch peak memory increase: "
            f"{(self.torch_peak_increase / GiB_bytes):.2f}GiB; "
            f"non-torch forward increase memory: "
            f"{(self.non_torch_increase / GiB_bytes):.2f}GiB; "
            f"weights memory: {(self.weights_memory / GiB_bytes):.2f}GiB."
        )


@contextlib.contextmanager
def memory_profiling(
    baseline_snapshot: MemorySnapshot, weights_memory: int
) -> Generator[MemoryProfilingResult, None, None]:
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
    """  # noqa
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

    non_torch_memory = result.non_torch_increase
    peak_activation_memory = result.torch_peak_increase
    result.non_kv_cache_memory = (
        non_torch_memory + peak_activation_memory + result.weights_memory
    )  # noqa


# Adapted from: https://github.com/sgl-project/sglang/blob/v0.4.1/python/sglang/srt/utils.py#L630 # noqa: E501
def set_ulimit(target_soft_limit=65535):
    if sys.platform.startswith("win"):
        logger.info("Windows detected, skipping ulimit adjustment.")
        return

    import resource

    resource_type = resource.RLIMIT_NOFILE
    current_soft, current_hard = resource.getrlimit(resource_type)

    if current_soft < target_soft_limit:
        try:
            resource.setrlimit(resource_type, (target_soft_limit, current_hard))
        except ValueError as e:
            logger.warning(
                "Found ulimit of %s and failed to automatically increase "
                "with error %s. This can cause fd limit errors like "
                "`OSError: [Errno 24] Too many open files`. Consider "
                "increasing with ulimit -n",
                current_soft,
                e,
            )


# Adapted from: https://github.com/sgl-project/sglang/blob/v0.4.1/python/sglang/utils.py#L28 # noqa: E501
def get_exception_traceback():
    etype, value, tb = sys.exc_info()
    err_str = "".join(traceback.format_exception(etype, value, tb))
    return err_str


def split_zmq_path(path: str) -> tuple[str, str, str]:
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


def make_zmq_path(scheme: str, host: str, port: int | None = None) -> str:
    """Make a ZMQ path from its parts.

    Args:
        scheme: The ZMQ transport scheme (e.g. tcp, ipc, inproc).
        host: The host - can be an IPv4 address, IPv6 address, or hostname.
        port: Optional port number, only used for TCP sockets.

    Returns:
        A properly formatted ZMQ path string.
    """
    if port is None:
        return f"{scheme}://{host}"
    if is_valid_ipv6_address(host):
        return f"{scheme}://[{host}]:{port}"
    return f"{scheme}://{host}:{port}"


# Adapted from: https://github.com/sgl-project/sglang/blob/v0.4.1/python/sglang/srt/utils.py#L783 # noqa: E501
def make_zmq_socket(
    ctx: zmq.asyncio.Context | zmq.Context,  # type: ignore[name-defined]
    path: str,
    socket_type: Any,
    bind: bool | None = None,
    identity: bytes | None = None,
    linger: int | None = None,
) -> zmq.Socket | zmq.asyncio.Socket:  # type: ignore[name-defined]
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
    buf_size = int(0.5 * 1024**3) if total_mem > 32 and available_mem > 16 else -1

    if bind is None:
        bind = socket_type not in (zmq.PUSH, zmq.SUB, zmq.XSUB)

    if socket_type in (zmq.PULL, zmq.DEALER, zmq.ROUTER):
        socket.setsockopt(zmq.RCVHWM, 0)
        socket.setsockopt(zmq.RCVBUF, buf_size)

    if socket_type in (zmq.PUSH, zmq.DEALER, zmq.ROUTER):
        socket.setsockopt(zmq.SNDHWM, 0)
        socket.setsockopt(zmq.SNDBUF, buf_size)

    if identity is not None:
        socket.setsockopt(zmq.IDENTITY, identity)

    if linger is not None:
        socket.setsockopt(zmq.LINGER, linger)

    if socket_type == zmq.XPUB:
        socket.setsockopt(zmq.XPUB_VERBOSE, True)

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
    bind: bool | None = None,
    linger: int = 0,
    identity: bytes | None = None,
) -> Iterator[zmq.Socket]:
    """Context manager for a ZMQ socket"""

    ctx = zmq.Context()  # type: ignore[attr-defined]
    try:
        yield make_zmq_socket(ctx, path, socket_type, bind=bind, identity=identity)
    except KeyboardInterrupt:
        logger.debug("Got Keyboard Interrupt.")

    finally:
        ctx.destroy(linger=linger)


def _maybe_force_spawn():
    """Check if we need to force the use of the `spawn` multiprocessing start
    method.
    """
    if os.environ.get("VLLM_WORKER_MULTIPROC_METHOD") == "spawn":
        return

    reasons = []
    if is_in_ray_actor():
        # even if we choose to spawn, we need to pass the ray address
        # to the subprocess so that it knows how to connect to the ray cluster.
        # env vars are inherited by subprocesses, even if we use spawn.
        import ray

        os.environ["RAY_ADDRESS"] = ray.get_runtime_context().gcs_address
        reasons.append("In a Ray actor and can only be spawned")

    if cuda_is_initialized():
        reasons.append("CUDA is initialized")
    elif xpu_is_initialized():
        reasons.append("XPU is initialized")

    if reasons:
        logger.warning(
            "We must use the `spawn` multiprocessing start method. "
            "Overriding VLLM_WORKER_MULTIPROC_METHOD to 'spawn'. "
            "See https://docs.vllm.ai/en/latest/usage/"
            "troubleshooting.html#python-multiprocessing "
            "for more information. Reasons: %s",
            "; ".join(reasons),
        )
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
    shared_kv_cache_layers: dict[str, str] | None = None,
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
    # 5. Some models have attention layers that share kv cache with previous
    #    layers, this is specified through shared_kv_cache_layers
    if shared_kv_cache_layers is None:
        shared_kv_cache_layers = {}
    from vllm.attention import AttentionType
    from vllm.model_executor.models.utils import extract_layer_index

    layer_need_kv_cache = [
        layer_name
        for layer_name in ctx
        if (
            hasattr(ctx[layer_name], "attn_type")
            and ctx[layer_name].attn_type
            in (AttentionType.DECODER, AttentionType.ENCODER_DECODER)
        )
        and ctx[layer_name].kv_sharing_target_layer_name is None
    ]
    layer_index_sorted = sorted(
        set(extract_layer_index(layer_name) for layer_name in layer_need_kv_cache)
    )
    for layer_name in layer_need_kv_cache:
        kv_cache_idx = layer_index_sorted.index(extract_layer_index(layer_name))
        forward_ctx = ctx[layer_name]
        assert len(forward_ctx.kv_cache) == len(kv_cache)
        for ve, ve_kv_cache in enumerate(kv_cache):
            forward_ctx.kv_cache[ve] = ve_kv_cache[kv_cache_idx]
    if shared_kv_cache_layers is not None:
        for layer_name, target_layer_name in shared_kv_cache_layers.items():
            assert extract_layer_index(target_layer_name) < extract_layer_index(
                layer_name
            ), "v0 doesn't support interleaving kv sharing"
            ctx[layer_name].kv_cache = ctx[target_layer_name].kv_cache


def run_method(
    obj: Any,
    method: str | bytes | Callable,
    args: tuple[Any],
    kwargs: dict[str, Any],
) -> Any:
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
            raise NotImplementedError(
                f"Method {method!r} is not implemented."
            ) from None
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
            if attr_name.startswith("_"):
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
            method_names = ",".join(unimplemented_methods)
            msg = f"Methods {method_names} not implemented in {self}"
            logger.debug(msg)

    @wraps(original_init)
    def wrapped_init(self, *args, **kwargs) -> None:
        original_init(self, *args, **kwargs)
        find_unimplemented_methods(self)

    type.__setattr__(cls, "__init__", wrapped_init)
    return cls


@contextlib.contextmanager
def cprofile_context(save_file: str | None = None):
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


def cprofile(save_file: str | None = None, enabled: bool = True):
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
    return (
        getattr(cfg, "alibi", False)  # Falcon
        or (
            "BloomForCausalLM" in getattr(model_config.hf_config, "architectures", [])
        )  # Bloom
        or getattr(cfg, "position_encoding_type", "") == "alibi"  # codellm_1b_alibi
        or (
            hasattr(cfg, "attn_config")  # MPT
            and (
                (
                    isinstance(cfg.attn_config, dict)
                    and cfg.attn_config.get("alibi", False)
                )
                or (
                    not isinstance(cfg.attn_config, dict)
                    and getattr(cfg.attn_config, "alibi", False)
                )
            )
        )
    )


def sha256(input: Any) -> bytes:
    """Hash any picklable Python object using SHA-256.

    The input is serialized using pickle before hashing, which allows
    arbitrary Python objects to be used. Note that this function does
    not use a hash seed—if you need one, prepend it explicitly to the input.

    Args:
        input: Any picklable Python object.

    Returns:
        Bytes representing the SHA-256 hash of the serialized input.
    """
    input_bytes = pickle.dumps(input, protocol=pickle.HIGHEST_PROTOCOL)
    return hashlib.sha256(input_bytes).digest()


def sha256_cbor(input: Any) -> bytes:
    """
    Hash objects using CBOR serialization and SHA-256.

    This option is useful for non-Python-dependent serialization and hashing.

    Args:
        input: Object to be serialized and hashed. Supported types include
            basic Python types and complex structures like lists, tuples, and
            dictionaries.
            Custom classes must implement CBOR serialization methods.

    Returns:
        Bytes representing the SHA-256 hash of the CBOR serialized input.
    """
    input_bytes = cbor2.dumps(input, canonical=True)
    return hashlib.sha256(input_bytes).digest()


def get_hash_fn_by_name(hash_fn_name: str) -> Callable[[Any], bytes]:
    """Get a hash function by name, or raise an error if
    the function is not found.
    Args:
        hash_fn_name: Name of the hash function.
    Returns:
        A hash function.
    """
    if hash_fn_name == "sha256":
        return sha256
    if hash_fn_name == "sha256_cbor":
        return sha256_cbor

    raise ValueError(f"Unsupported hash function: {hash_fn_name}")


def is_torch_equal_or_newer(target: str) -> bool:
    """Check if the installed torch version is >= the target version.

    Args:
        target: a version string, like "2.6.0".

    Returns:
        Whether the condition meets.
    """
    try:
        return _is_torch_equal_or_newer(str(torch.__version__), target)
    except Exception:
        # Fallback to PKG-INFO to load the package info, needed by the doc gen.
        return Version(importlib.metadata.version("torch")) >= Version(target)


# Helper function used in testing.
def _is_torch_equal_or_newer(torch_version: str, target: str) -> bool:
    torch_version = version.parse(torch_version)
    return torch_version >= version.parse(target)


def _is_torch_equal(target: str) -> bool:
    assert target.count(".") == 2
    torch_version = str(torch.__version__)
    torch_version = version.parse(torch_version)
    # torch version is like "2.6.0.dev20240101" or "2.6.0.dev20240101+cpu"
    # or "2.6.0+cu128" but never "2.6.0.1"
    return (
        torch_version >= version.parse(target)
        and version.parse(target + ".1") > torch_version
    )


def is_torch_equal(target: str) -> bool:
    """Check if the installed torch version is == the target version.

    Args:
        target: a version string, like "2.6.0".

    Returns:
        Whether the condition meets.
    """
    try:
        return _is_torch_equal(target)
    except Exception:
        return Version(importlib.metadata.version("torch")) == Version(target)


@cache
def _has_module(module_name: str) -> bool:
    """Return True if *module_name* can be found in the current environment.

    The result is cached so that subsequent queries for the same module incur
    no additional overhead.
    """
    return importlib.util.find_spec(module_name) is not None


def has_pplx() -> bool:
    """Whether the optional `pplx_kernels` package is available."""

    return _has_module("pplx_kernels")


def has_deep_ep() -> bool:
    """Whether the optional `deep_ep` package is available."""

    return _has_module("deep_ep")


def has_deep_gemm() -> bool:
    """Whether the optional `deep_gemm` package is available."""

    return _has_module("deep_gemm")


def has_triton_kernels() -> bool:
    """Whether the optional `triton_kernels` package is available."""

    return _has_module("triton_kernels")


def has_tilelang() -> bool:
    """Whether the optional `tilelang` package is available."""

    return _has_module("tilelang")


def set_process_title(
    name: str, suffix: str = "", prefix: str = envs.VLLM_PROCESS_NAME_PREFIX
) -> None:
    """
    Set the current process title to a specific name with an
    optional suffix.

    Args:
        name: The title to assign to the current process.
        suffix: An optional suffix to append to the base name.
        prefix: A prefix to prepend to the front separated by `::`.
    """
    if suffix:
        name = f"{name}_{suffix}"
    setproctitle.setproctitle(f"{prefix}::{name}")


def _add_prefix(file: TextIO, worker_name: str, pid: int) -> None:
    """Prepend each output line with process-specific prefix"""

    prefix = f"{CYAN}({worker_name} pid={pid}){RESET} "
    file_write = file.write

    def write_with_prefix(s: str):
        if not s:
            return
        if file.start_new_line:  # type: ignore[attr-defined]
            file_write(prefix)
        idx = 0
        while (next_idx := s.find("\n", idx)) != -1:
            next_idx += 1
            file_write(s[idx:next_idx])
            if next_idx == len(s):
                file.start_new_line = True  # type: ignore[attr-defined]
                return
            file_write(prefix)
            idx = next_idx
        file_write(s[idx:])
        file.start_new_line = False  # type: ignore[attr-defined]

    file.start_new_line = True  # type: ignore[attr-defined]
    file.write = write_with_prefix  # type: ignore[method-assign]


def decorate_logs(process_name: str | None = None) -> None:
    """
    Adds a process-specific prefix to each line of output written to stdout and
    stderr.

    This function is intended to be called before initializing the api_server,
    engine_core, or worker classes, so that all subsequent output from the
    process is prefixed with the process name and PID. This helps distinguish
    log output from different processes in multi-process environments.

    Args:
        process_name: Optional; the name of the process to use in the prefix.
            If not provided, the current process name from the multiprocessing
            context is used.
    """
    if process_name is None:
        process_name = get_mp_context().current_process().name
    pid = os.getpid()
    _add_prefix(sys.stdout, process_name, pid)
    _add_prefix(sys.stderr, process_name, pid)


def length_from_prompt_token_ids_or_embeds(
    prompt_token_ids: list[int] | None,
    prompt_embeds: torch.Tensor | None,
) -> int:
    """Calculate the request length (in number of tokens) give either
    prompt_token_ids or prompt_embeds.
    """
    prompt_token_len = None if prompt_token_ids is None else len(prompt_token_ids)
    prompt_embeds_len = None if prompt_embeds is None else len(prompt_embeds)

    if prompt_token_len is None:
        if prompt_embeds_len is None:
            raise ValueError("Neither prompt_token_ids nor prompt_embeds were defined.")
        return prompt_embeds_len
    else:
        if prompt_embeds_len is not None and prompt_embeds_len != prompt_token_len:
            raise ValueError(
                "Prompt token ids and prompt embeds had different lengths"
                f" prompt_token_ids={prompt_token_len}"
                f" prompt_embeds={prompt_embeds_len}"
            )
        return prompt_token_len


@contextlib.contextmanager
def set_env_var(key, value):
    old = os.environ.get(key)
    os.environ[key] = value
    try:
        yield
    finally:
        if old is None:
            del os.environ[key]
        else:
            os.environ[key] = old


def unique_filepath(fn: Callable[[int], Path]) -> Path:
    """
    unique_filepath returns a unique path by trying
    to include an integer in increasing order.

    fn should be a callable that returns a path that
    includes the passed int at a fixed location.

    Note: This function has a TOCTOU race condition.
    Caller should use atomic operations (e.g., open with 'x' mode)
    when creating the file to ensure thread safety.
    """
    i = 0
    while True:
        p = fn(i)
        if not p.exists():
            return p
        i += 1
