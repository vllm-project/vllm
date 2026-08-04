# SPDX-License-Identifier: Apache-2.0
# Standard
from typing import Optional
from unittest.mock import MagicMock
import asyncio
import ctypes
import functools
import inspect
import os
import random
import socket
import string
import tempfile
import threading
import uuid

# Third Party
import torch

# First Party
from lmcache.utils import CacheEngineKey
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.gpu_connector.gpu_connectors import VLLMPagedMemGPUConnectorV2
from lmcache.v1.memory_allocators.ad_hoc_memory_allocator import AdHocMemoryAllocator
from lmcache.v1.memory_management import (
    MemoryFormat,
    MemoryObj,
)
from lmcache.v1.metadata import LMCacheMetadata

# Conditional import for CUDA-only operations
if torch.cuda.is_available() or torch.xpu.is_available():
    try:
        # First Party
        import lmcache.c_ops as lmc_ops
    except ImportError:
        # If c_ops is not built, create a mock
        lmc_ops = None
else:
    # Mock c_ops when CUDA is not available
    # First Party
    lmc_ops = None

# Define mock EngineKVFormat enum if c_ops is not available
if lmc_ops is None:

    class MockEngineKVFormat:
        NL_X_TWO_NB_BS_NH_HS = 0
        NL_X_NB_TWO_BS_NH_HS = 1
        NL_X_NB_BS_HS = 2
        NL_X_TWO_NB_NH_BS_HS = 3
        NL_X_NB_TWO_NH_BS_HS = 4
        NL_X_NB_NH_BS_TWO_HS = 5

    class MockCOps:
        EngineKVFormat = MockEngineKVFormat
        GPUKVFormat = MockEngineKVFormat

    lmc_ops = MockCOps()


def _probe_cufile_register() -> bool:
    """
    Try to actually register a cuFile handle on a real file in the test
    scratch dir. Returns True iff cuFileHandleRegister succeeds.

    Importability of cufile / libcufile.so is necessary but not sufficient:
    on hosts without nvidia-fs (or on a non-GDS-capable filesystem),
    cuFileHandleRegister fails at runtime with CU_FILE_IO_NOT_SUPPORTED
    (err=5027). This probe matches the exact path tests will exercise.
    """
    try:
        # Third Party
        import cufile
    except Exception:
        return False

    probe_dir = os.environ.get("LMCACHE_TEST_TMPDIR") or tempfile.gettempdir()
    if not os.path.isdir(probe_dir):
        return False

    try:
        fd, probe_path = tempfile.mkstemp(dir=probe_dir, prefix="cufile-probe-")
    except OSError:
        return False

    try:
        try:
            os.write(fd, b"\0" * 4096)
        finally:
            os.close(fd)
        # Mirror production: GdsBackend opens with mode "r+" and
        # use_direct_io=True (see gds_backend.py:950). If the FS doesn't
        # support GDS+O_DIRECT, register fails here exactly as in tests.
        cu = cufile.CuFile(probe_path, "r+", use_direct_io=True)
        try:
            cu.open()
        except Exception:
            # Register failed. cu._handle may hold the raw fd from os.open
            # without a registered cuFile handle; close it ourselves and
            # null the state so __del__ doesn't try to deregister None.
            raw_fd = getattr(cu, "_handle", None)
            if raw_fd is not None:
                try:
                    os.close(raw_fd)
                except OSError:
                    pass
                cu._handle = None
            return False
        cu.close()
        return True
    finally:
        try:
            os.unlink(probe_path)
        except OSError:
            pass


@functools.lru_cache(maxsize=1)
def has_cufile() -> bool:
    """
    True only when NVIDIA cuFile is usable on this host's test scratch dir:
    - python package `cufile` importable
    - dynamic library `libcufile.so` loadable
    - cuFileHandleRegister succeeds on a real file in LMCACHE_TEST_TMPDIR
      (or the system tmpdir as a fallback)
    """
    try:
        # Third Party
        import cufile  # noqa: F401
    except Exception:
        return False

    try:
        ctypes.CDLL("libcufile.so")
    except OSError:
        return False

    return _probe_cufile_register()


def has_hipfile() -> bool:
    """
    True only when AMD hipFile is available:
    - python package `hipfile` importable
    - dynamic library `libhipfile.so` loadable
    """
    try:
        # Third Party
        import hipfile  # noqa: F401
    except Exception:
        return False

    try:
        ctypes.CDLL("libhipfile.so")
    except OSError:
        return False

    return True


def recover_engine_states(engine):
    engine.gpu_connector.kv_cache_pointers_on_gpu = {}


def recover_gpu_connector_states(gpu_connector):
    gpu_connector.kv_cache_pointers_on_gpu = {}


def dumb_metadata(kv_shape=(32, 2, 256, 8, 128)):
    return LMCacheMetadata(
        model_name="test_model",
        world_size=3,
        local_world_size=3,
        worker_id=1,
        local_worker_id=1,
        kv_dtype=torch.bfloat16,
        kv_shape=kv_shape,
    )


def dumb_metadata_with_model_name(model_name: str, kv_shape=(32, 2, 256, 8, 128)):
    return LMCacheMetadata(
        model_name=model_name,
        world_size=3,
        local_world_size=3,
        worker_id=1,
        local_worker_id=1,
        kv_dtype=torch.bfloat16,
        kv_shape=kv_shape,
    )


def dumb_cache_engine_key(id: int = 0) -> CacheEngineKey:
    return CacheEngineKey(
        model_name="test_model",
        world_size=3,
        worker_id=1,
        chunk_hash=id,
        dtype=torch.bfloat16,
    )


def random_string(N):
    return "".join(random.choices(string.ascii_uppercase + string.digits, k=N))


def init_asyncio_loop():
    async_loop = asyncio.new_event_loop()
    async_thread = threading.Thread(target=async_loop.run_forever)
    async_thread.start()
    return async_loop, async_thread


def close_asyncio_loop(async_loop, async_thread):
    if async_loop.is_running():
        # First, cancel all pending tasks
        try:
            # Get all tasks and cancel them
            pending = asyncio.all_tasks(async_loop)
            for task in pending:
                if not task.done():
                    task.cancel()
        except Exception as e:
            print(f"Error during close pending tasks: - {e}")

        # Then stop the loop
        async_loop.call_soon_threadsafe(async_loop.stop)

    if async_thread.is_alive():
        async_thread.join(timeout=2.0)

    # Close the loop to release resources
    if not async_loop.is_closed():
        async_loop.close()

    # Set event loop to None
    asyncio.set_event_loop(None)


def get_available_port(host: str = "127.0.0.1") -> int:
    """
    Get an available port dynamically by binding to port 0.

    Args:
        host: The host address to bind to. Default is "127.0.0.1".

    Returns:
        An available port number.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((host, 0))
        s.listen(1)
        port = s.getsockname()[1]
    return port


def get_available_ports(count: int, host: str = "127.0.0.1") -> list[int]:
    """
    Get multiple available ports dynamically.

    Args:
        count: Number of ports to get.
        host: The host address to bind to. Default is "127.0.0.1".

    Returns:
        A list of available port numbers.
    """
    ports = []
    for _ in range(count):
        ports.append(get_available_port(host))
    return ports


def generate_kv_cache(num_tokens, device):
    ret = []
    num_layers = 32
    num_heads = 8
    head_size = 128
    shape = [num_tokens, num_heads, head_size]
    dtype = torch.bfloat16

    for i in range(num_layers):
        k = torch.rand(shape, dtype=dtype, device=device)
        v = torch.rand(shape, dtype=dtype, device=device)
        ret.append((k, v))

    return tuple(ret)


def generate_kv_cache_paged_list_tensors(
    num_blocks,
    device,
    block_size=16,
    dtype=torch.bfloat16,
    num_layers=32,
    head_size=128,
    # default vllm non-MLA flash attention
    engine_kv_format=lmc_ops.EngineKVFormat.NL_X_TWO_NB_BS_NH_HS,
):
    """
    Instead of Tuple[Tuple[Tensor, Tensor]], return List[Tensor]
    where KV are in the same tensor
    """
    ret = []
    # only support vllm MLA format for now
    use_mla = engine_kv_format == lmc_ops.EngineKVFormat.NL_X_NB_BS_HS
    num_heads = 1 if use_mla else 8
    if use_mla:
        shape = [num_blocks, block_size, head_size]
    else:
        if engine_kv_format == lmc_ops.EngineKVFormat.NL_X_TWO_NB_BS_NH_HS:
            shape = [2, num_blocks, block_size, num_heads, head_size]
        elif engine_kv_format == lmc_ops.EngineKVFormat.NL_X_NB_TWO_BS_NH_HS:
            shape = [num_blocks, 2, block_size, num_heads, head_size]
        elif engine_kv_format == lmc_ops.EngineKVFormat.NL_X_TWO_NB_NH_BS_HS:
            shape = [2, num_blocks, num_heads, block_size, head_size]
        elif engine_kv_format == lmc_ops.EngineKVFormat.NL_X_NB_TWO_NH_BS_HS:
            shape = [num_blocks, 2, num_heads, block_size, head_size]
        elif engine_kv_format == lmc_ops.EngineKVFormat.NL_X_NB_NH_BS_TWO_HS:
            # blocks-first, K/V fused into the trailing dim
            shape = [num_blocks, num_heads, block_size, 2, head_size]
        else:
            raise ValueError(f"Unsupported engine_kv_format: {engine_kv_format}")

    for i in range(num_layers):
        # TODO(chunxiaozheng): support more dtypes
        if dtype == torch.uint8:
            kv = torch.randint(0, 256, shape, dtype=dtype, device=device)
        else:
            kv = torch.rand(shape, dtype=dtype, device=device)
        ret.append(kv)

    return ret


def generate_sglang_kv_cache_paged_list_tensors(
    num_layers,
    num_blocks,
    block_size,
    num_heads,
    head_size,
    use_mla=False,
    device="cuda",
    dtype=torch.bfloat16,
):
    """
    Instead of Tuple[Tuple[Tensor, Tensor]], return List[Tensor]
    where KV are in the same tensor

    For MLA: List[num_layers] of [page_buffer_size, 1, head_size]
    For MHA: List[2] -> List[num_layers] of [page_buffer_size, num_heads, head_size]
    """
    shape = (
        [num_blocks * block_size, 1, head_size]
        if use_mla
        else [num_blocks * block_size, num_heads, head_size]
    )
    if use_mla:
        kv_cache = [
            torch.rand(shape, dtype=dtype, device=device) for i in range(num_layers)
        ]
    else:
        # MHA: List[2] -> List[num_layers]
        k_cache = [
            torch.rand(shape, dtype=dtype, device=device) for i in range(num_layers)
        ]
        v_cache = [
            torch.rand(shape, dtype=dtype, device=device) for i in range(num_layers)
        ]
        kv_cache = [k_cache, v_cache]
    return kv_cache


def generate_kv_cache_paged(num_blocks, device, block_size=16, dtype=torch.bfloat16):
    ret = []
    num_layers = 32
    num_heads = 8
    head_size = 128
    shape = [num_blocks, block_size, num_heads, head_size]

    for i in range(num_layers):
        k = torch.rand(shape, dtype=dtype, device=device)
        v = torch.rand(shape, dtype=dtype, device=device)
        ret.append((k, v))

    return tuple(ret)


def generate_tokens(num_tokens, device, fixed=False):
    if fixed:
        return torch.tensor([-1] * num_tokens).to(device)
    else:
        # random tokens
        return torch.randint(0, 10000, size=[num_tokens]).to(device)


def concatenate_kv_caches(kv_chunks):
    dim = 0
    ret = []
    for kv_layer in zip(*kv_chunks, strict=False):
        klist, vlist = zip(*kv_layer, strict=False)
        klayer = torch.cat(klist, dim=dim)
        vlayer = torch.cat(vlist, dim=dim)
        ret.append((klayer, vlayer))
    return tuple(ret)


def check_mem_obj_equal(left, right, use_mla: bool = False):
    """
    check whether two memory objects are the same
    """
    for left_mem_obj, right_mem_obj in zip(left, right, strict=False):
        left_tensor_size = left_mem_obj.tensor.size()
        right_tensor_size = right_mem_obj.tensor.size()
        if use_mla:
            assert left_tensor_size[0] == 1
            assert right_tensor_size[0] == 1

            left_kv, right_kv = left_mem_obj.tensor[0], right_mem_obj.tensor[0]
            right_kv = right_kv.to(left_kv.device)

            assert len(left_kv.shape) == 3
            assert len(right_kv.shape) == 3

            assert (left_kv[:, :, :] == right_kv[:, :, :]).all()
        else:
            assert left_tensor_size[0] == 2
            assert right_tensor_size[0] == 2

            left_kv, right_kv = left_mem_obj.tensor, right_mem_obj.tensor
            left_k, left_v = left_kv[0], left_kv[1]
            right_k, right_v = right_kv[0], right_kv[1]
            right_k = right_k.to(left_k.device)
            right_v = right_v.to(left_v.device)

            assert len(left_k.shape) == 3
            assert len(left_v.shape) == 3
            assert len(right_k.shape) == 3
            assert len(right_v.shape) == 3

            assert (left_k[:, :, :] == right_k[:, :, :]).all()
            assert (left_v[:, :, :] == right_v[:, :, :]).all()


# default checks for vllm non-MLA flash attention
def check_paged_kv_cache_equal(
    left,
    right,
    slot_mapping,
    num_heads=8,
    head_size=128,
    engine_kv_format=lmc_ops.EngineKVFormat.NL_X_TWO_NB_BS_NH_HS,
):
    """
    check whether two paged kv caches are the same at slot_mapping
    """

    if engine_kv_format == lmc_ops.EngineKVFormat.NL_X_TWO_NB_BS_NH_HS:
        token_dim = 0
        num_tokens = slot_mapping.shape[0]
        for left_kv_layer, right_kv_layer in zip(left, right, strict=False):
            left_k = left_kv_layer[0].reshape(-1, num_heads, head_size)
            left_v = left_kv_layer[1].reshape(-1, num_heads, head_size)
            right_k = right_kv_layer[0].reshape(-1, num_heads, head_size)
            right_v = right_kv_layer[1].reshape(-1, num_heads, head_size)

            assert len(left_k.shape) == 3
            assert len(left_v.shape) == 3
            assert len(right_k.shape) == 3
            assert len(right_v.shape) == 3

            assert left_k.shape[token_dim] >= num_tokens
            assert left_v.shape[token_dim] >= num_tokens
            assert right_k.shape[token_dim] >= num_tokens
            assert right_v.shape[token_dim] >= num_tokens

            assert (left_k[slot_mapping, :, :] == right_k[slot_mapping, :, :]).all()
            assert (left_v[slot_mapping, :, :] == right_v[slot_mapping, :, :]).all()

    elif engine_kv_format == lmc_ops.EngineKVFormat.NL_X_NB_TWO_BS_NH_HS:
        token_dim = 0
        num_tokens = slot_mapping.shape[0]
        for left_kv_layer, right_kv_layer in zip(left, right, strict=False):
            left_k = left_kv_layer[:, 0].contiguous().reshape(-1, num_heads, head_size)
            left_v = left_kv_layer[:, 1].contiguous().reshape(-1, num_heads, head_size)
            right_k = (
                right_kv_layer[:, 0].contiguous().reshape(-1, num_heads, head_size)
            )
            right_v = (
                right_kv_layer[:, 1].contiguous().reshape(-1, num_heads, head_size)
            )

            assert len(left_k.shape) == 3
            assert len(left_v.shape) == 3
            assert len(right_k.shape) == 3
            assert len(right_v.shape) == 3

            assert left_k.shape[token_dim] >= num_tokens
            assert left_v.shape[token_dim] >= num_tokens
            assert right_k.shape[token_dim] >= num_tokens
            assert right_v.shape[token_dim] >= num_tokens

            assert (left_k[slot_mapping, :, :] == right_k[slot_mapping, :, :]).all()
            assert (left_v[slot_mapping, :, :] == right_v[slot_mapping, :, :]).all()

    elif engine_kv_format == lmc_ops.EngineKVFormat.NL_X_TWO_NB_NH_BS_HS:
        # HND flash attention: [2, num_blocks, num_heads, block_size, head_size]
        # Flatten [num_blocks, num_heads, block_size, head_size] ->
        #   swap to [num_blocks, block_size, num_heads, head_size] ->
        #   reshape to [num_blocks*block_size, num_heads, head_size]
        num_tokens = slot_mapping.shape[0]
        for left_kv_layer, right_kv_layer in zip(left, right, strict=False):
            left_k = (
                left_kv_layer[0]
                .permute(0, 2, 1, 3)
                .contiguous()
                .reshape(-1, num_heads, head_size)
            )
            left_v = (
                left_kv_layer[1]
                .permute(0, 2, 1, 3)
                .contiguous()
                .reshape(-1, num_heads, head_size)
            )
            right_k = (
                right_kv_layer[0]
                .permute(0, 2, 1, 3)
                .contiguous()
                .reshape(-1, num_heads, head_size)
            )
            right_v = (
                right_kv_layer[1]
                .permute(0, 2, 1, 3)
                .contiguous()
                .reshape(-1, num_heads, head_size)
            )

            assert left_k.shape[0] >= num_tokens
            assert (left_k[slot_mapping, :, :] == right_k[slot_mapping, :, :]).all()
            assert (left_v[slot_mapping, :, :] == right_v[slot_mapping, :, :]).all()

    elif engine_kv_format == lmc_ops.EngineKVFormat.NL_X_NB_TWO_NH_BS_HS:
        # HND flash infer: [num_blocks, 2, num_heads, block_size, head_size]
        # left_kv_layer[:, 0] -> [num_blocks, num_heads, block_size, head_size]
        num_tokens = slot_mapping.shape[0]
        for left_kv_layer, right_kv_layer in zip(left, right, strict=False):
            left_k = (
                left_kv_layer[:, 0]
                .permute(0, 2, 1, 3)
                .contiguous()
                .reshape(-1, num_heads, head_size)
            )
            left_v = (
                left_kv_layer[:, 1]
                .permute(0, 2, 1, 3)
                .contiguous()
                .reshape(-1, num_heads, head_size)
            )
            right_k = (
                right_kv_layer[:, 0]
                .permute(0, 2, 1, 3)
                .contiguous()
                .reshape(-1, num_heads, head_size)
            )
            right_v = (
                right_kv_layer[:, 1]
                .permute(0, 2, 1, 3)
                .contiguous()
                .reshape(-1, num_heads, head_size)
            )

            assert left_k.shape[0] >= num_tokens
            assert (left_k[slot_mapping, :, :] == right_k[slot_mapping, :, :]).all()
            assert (left_v[slot_mapping, :, :] == right_v[slot_mapping, :, :]).all()


def check_sglang_paged_kv_cache_equal(
    left, right, slot_mapping, num_heads=8, head_size=128
):
    """
    check whether two paged kv caches are the same at slot_mapping

    Format: List[2] -> List[num_layers] of [page_buffer_size, num_heads, head_size]
    """
    token_dim = 0
    num_tokens = slot_mapping.shape[0]

    # left and right are [k_list, v_list]
    assert len(left) == 2, "Expected [k_list, v_list]"
    assert len(right) == 2, "Expected [k_list, v_list]"

    # Check K and V separately
    for kv_idx in range(2):  # 0 for K, 1 for V
        left_kv_list = left[kv_idx]
        right_kv_list = right[kv_idx]

        for left_kv, right_kv in zip(left_kv_list, right_kv_list, strict=False):
            _left_kv = left_kv.reshape(-1, num_heads, head_size)
            _right_kv = right_kv.reshape(-1, num_heads, head_size)

            assert len(_left_kv.shape) == 3
            assert len(_right_kv.shape) == 3

            assert _left_kv.shape[token_dim] >= num_tokens
            assert _right_kv.shape[token_dim] >= num_tokens

            assert (_left_kv[slot_mapping, :, :] == _right_kv[slot_mapping, :, :]).all()


def check_paged_kv_cache_equal_with_mla(left, right, slot_mapping, head_size=128):
    """
    check whether two paged kv caches are the same at slot_mapping when use mla
    """
    token_dim = 0
    num_tokens = slot_mapping.shape[0]
    for left_kv, right_kv in zip(left, right, strict=False):
        new_left_kv = left_kv.reshape(-1, head_size)
        new_right_kv = right_kv.reshape(-1, head_size)

        assert len(new_left_kv.shape) == 2
        assert len(new_right_kv.shape) == 2

        assert new_left_kv.shape[token_dim] >= num_tokens
        assert new_right_kv.shape[token_dim] >= num_tokens

        assert (new_left_kv[slot_mapping, :] == new_right_kv[slot_mapping, :]).all()


def check_kv_cache_device(kvs, device):
    for kv in kvs:
        k, v = kv
        assert k.device == torch.device(device)
        assert v.device == torch.device(device)


def create_gpu_connector(hidden_dim, num_layers):
    return VLLMPagedMemGPUConnectorV2(hidden_dim, num_layers)


def create_xpu_connector(hidden_dim, num_layers):
    # First Party
    from lmcache.v1.gpu_connector.xpu_connectors import VLLMPagedMemXPUConnectorV2

    return VLLMPagedMemXPUConnectorV2(hidden_dim, num_layers)


def get_all_methods_from_base(base_class):
    """
    Get all public methods defined in the base class (excluding inherited from object).
    """
    methods = set()
    for name in dir(base_class):
        # Skip private and special methods
        if name.startswith("_"):
            continue
        attr = getattr(base_class, name)
        if callable(attr):
            methods.add(name)
    return methods


def get_methods_implemented_in_class(cls, base_class=None):
    """
    Get methods that are actually implemented in the class itself.
    Args:
        cls: The class to inspect
        base_class: Optional base class to stop at. If None, stops at
            abstract base classes.
    """
    implemented = set()

    # Check the class's own __dict__ for methods
    for name in cls.__dict__:
        if name.startswith("_"):
            continue
        attr = cls.__dict__[name]
        # Check if it's callable (function, method, etc.)
        if callable(attr):
            implemented.add(name)

    # Also check using getattr to catch any dynamically added methods
    for name in dir(cls):
        if name.startswith("_"):
            continue
        if name in implemented:
            continue  # Already found
        try:
            attr = getattr(cls, name)
            if callable(attr):
                # Verify it's not inherited from base class
                # by checking if it exists in the class's MRO
                for base in cls.__mro__:
                    # Stop when we hit the specified base class
                    if base_class is not None and base is base_class:
                        break
                    # Or stop when we hit an abstract base class
                    if base_class is None and inspect.isabstract(base):
                        break
                    if name in base.__dict__:
                        implemented.add(name)
                        break
        except AttributeError:
            pass

    return implemented


def get_abstract_methods(cls):
    """
    Get all abstract methods from a class.
    """
    abstract_methods = set()
    for name, method in inspect.getmembers(cls, predicate=inspect.isfunction):
        if getattr(method, "__isabstractmethod__", False):
            abstract_methods.add(name)
    return abstract_methods


def check_method_signatures(base_class, impl_class):
    """
    Check if method signatures in implementation class match the base class.
    Returns a list of mismatches.
    """
    base_methods = get_all_methods_from_base(base_class)
    signature_mismatches = []

    for method_name in base_methods:
        base_method = getattr(base_class, method_name)
        impl_method = getattr(impl_class, method_name, None)

        if impl_method is None:
            continue

        try:
            base_sig = inspect.signature(base_method)
            impl_sig = inspect.signature(impl_method)

            # Compare parameter names (excluding 'self')
            base_params = [p for p in base_sig.parameters.keys() if p != "self"]
            impl_params = [p for p in impl_sig.parameters.keys() if p != "self"]

            if base_params != impl_params:
                signature_mismatches.append(
                    {
                        "method": method_name,
                        "base_params": base_params,
                        "impl_params": impl_params,
                    }
                )
        except (ValueError, TypeError):
            # Some methods might not have inspectable signatures
            pass

    return signature_mismatches


class DummyLMCacheAsyncLookupServer:
    def __init__(self):
        pass

    def send_response_to_scheduler(
        self,
        lookup_id: str,
        retrieved_length: int,
    ) -> None:
        pass


class MockAdapter:
    """
    Mock adapter to provide config and lmcache_engine to InternalAPIServer.
    """

    def __init__(self, engine, config):
        self.lmcache_engine = engine
        self.config = config


def create_test_metadata(
    worker_id: int = 0,
    world_size: int = 1,
    kv_shape: tuple = (4, 2, 256, 8, 128),
    engine_id: Optional[str] = "test_engine",
    kv_connector_extra_config: Optional[dict] = None,
) -> LMCacheMetadata:
    """Create test metadata for LMCacheEngine."""
    return LMCacheMetadata(
        model_name="test_model",
        world_size=world_size,
        local_world_size=world_size,
        worker_id=worker_id,
        local_worker_id=worker_id,
        kv_dtype=torch.bfloat16,
        kv_shape=kv_shape,
        engine_id=engine_id,
        kv_connector_extra_config=kv_connector_extra_config,
    )


def create_test_config(
    chunk_size: int = 256,
    local_cpu: bool = True,
    max_local_cpu_size: float = 1.0,
    rpc_port: int = 0,
    extra_config: Optional[dict] = None,
    instance_id: Optional[str] = None,
) -> LMCacheEngineConfig:
    """Create test configuration for LMCacheEngine."""
    if instance_id is None:
        instance_id = f"test_instance_{uuid.uuid4().hex[:8]}"
    config = LMCacheEngineConfig.from_defaults(
        chunk_size=chunk_size,
        local_cpu=local_cpu,
        max_local_cpu_size=max_local_cpu_size,
        lmcache_instance_id=instance_id,
    )
    config.extra_config = extra_config.copy() if extra_config else {}
    config.extra_config["lmcache_rpc_port"] = rpc_port
    return config


def create_mock_vllm_config(
    rank: int = 0, world_size: int = 1, rpc_port: int = 0
) -> MagicMock:
    """Create a mock VllmConfig for testing."""
    vllm_config = MagicMock()

    # Mock model_config
    vllm_config.model_config = MagicMock()
    vllm_config.model_config.model = "test_model"
    vllm_config.model_config.dtype = torch.bfloat16
    vllm_config.model_config.get_num_layers = MagicMock(return_value=4)
    vllm_config.model_config.get_num_kv_heads = MagicMock(return_value=8)
    vllm_config.model_config.get_head_size = MagicMock(return_value=128)
    vllm_config.model_config.hf_config = MagicMock()
    vllm_config.model_config.hf_config.model_type = "llama"

    # Mock parallel_config
    vllm_config.parallel_config = MagicMock()
    vllm_config.parallel_config.rank = rank
    vllm_config.parallel_config.world_size = world_size
    vllm_config.parallel_config.tensor_parallel_size = world_size
    vllm_config.parallel_config.pipeline_parallel_size = 1

    # Mock cache_config
    vllm_config.cache_config = MagicMock()
    vllm_config.cache_config.cache_dtype = torch.bfloat16

    # Mock kv_transfer_config with engine_id
    vllm_config.kv_transfer_config = MagicMock()
    vllm_config.kv_transfer_config.engine_id = "test_engine"
    vllm_config.kv_transfer_config.get_from_extra_config = MagicMock(
        side_effect=lambda key, default: (
            rpc_port if key == "lmcache_rpc_port" else default
        )
    )

    return vllm_config


def create_test_memory_obj(shape=None, dtype=torch.bfloat16, device="cpu") -> MemoryObj:
    """Create a test MemoryObj using AdHocMemoryAllocator for testing."""
    if shape is None:
        shape = torch.Size([2, 16, 8, 128])
    allocator = AdHocMemoryAllocator(device=device)
    memory_obj = allocator.allocate([shape], [dtype], fmt=MemoryFormat.KV_T2D)
    return memory_obj
