# SPDX-License-Identifier: Apache-2.0
# Standard
from pathlib import Path
import asyncio
import contextlib
import functools
import os
import shutil
import tempfile
import threading
import uuid

# Third Party
import pytest
import torch

pytest.importorskip("nixl", reason="nixl package is required for nixl tests")

# Third Party
from nixl._api import nixl_agent as NixlAgent
from nixl._api import nixl_agent_config as NixlAgentConfig

# First Party
from lmcache.utils import CacheEngineKey
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.memory_allocators.paged_tensor_memory_allocator import (
    PagedTensorMemoryAllocator,
)
from lmcache.v1.metadata import LMCacheMetadata
from lmcache.v1.storage_backend import CreateStorageBackends
from lmcache.v1.storage_backend.nixl_storage_backend import (
    NixlStorageBackend,
    NixlStorageConfig,
)
from lmcache.v1.transfer_channel.transfer_utils import get_correct_device

# cuFile-based backends (GDS, GDS_MT) need a GDS-capable filesystem
_TEST_TMPDIR = os.environ.get("LMCACHE_TEST_TMPDIR") or None


@functools.lru_cache(maxsize=None)
def _can_register_file_with_nixl_backend(backend: str) -> bool:
    """Probe ``cuFileHandleRegister`` via NIXL on the test scratch dir."""

    probe_dir = tempfile.mkdtemp(prefix="nixl_gds_probe_", dir=_TEST_TMPDIR)
    probe_path = os.path.join(probe_dir, "probe.bin")
    fd = -1
    try:
        agent = NixlAgent(
            f"NixlGdsProbe_{uuid.uuid4().hex}",
            NixlAgentConfig(backends=[]),
        )
        agent.create_backend(backend, {})
        fd = os.open(probe_path, os.O_CREAT | os.O_RDWR, 0o600)
        os.write(fd, b"\x00" * 4096)
        agent.register_memory([(0, 4096, fd, "")], mem_type="FILE")
        return True
    except Exception:
        return False
    finally:
        if fd >= 0:
            with contextlib.suppress(OSError):
                os.close(fd)
        shutil.rmtree(probe_dir, ignore_errors=True)


_GDS_SKIP_REASON = (
    "NIXL {backend} cannot register file handles in this environment; "
    "set LMCACHE_TEST_TMPDIR to a GDS-capable mount (ext4/xfs) to enable."
)


@pytest.fixture
def nixl_tmp_path():
    """Per-test scratch dir, honoring ``LMCACHE_TEST_TMPDIR``."""
    path = tempfile.mkdtemp(prefix="nixl_test_", dir=_TEST_TMPDIR)
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


def create_key(chunk_hash: str):
    return CacheEngineKey(
        model_name="meta-llama/Llama-3.1-70B-Instruct",
        world_size=8,
        worker_id=0,
        chunk_hash=int(chunk_hash, base=16),
        dtype=torch.bfloat16,
    )


def run(config: LMCacheEngineConfig, shape, dtype):
    BACKEND_NAME = "NixlStorageBackend"
    keys = []
    objs = []
    keys.append(
        create_key("e3229141e680fb413d2c5d3ebb416c4ad300d381e309fc9e417757b91406c157")
    )
    keys.append(
        create_key("e3229141e680fb413d2c5d3ebb416c4ad300d381e309fc9e417757b91406d268")
    )
    keys.append(
        create_key("e3229141e680fb413d2c5d3ebb416c4ad300d381e309fc9e417757b91406e379")
    )
    bad_key = create_key("deadbeefdeadbeef")

    thread_loop = None
    thread = None
    try:
        thread_loop = asyncio.new_event_loop()
        thread = threading.Thread(target=thread_loop.run_forever)
        thread.start()

        metadata = LMCacheMetadata(
            model_name="Llama-3.1-70B-Instruct",
            world_size=1,
            local_world_size=1,
            worker_id=0,
            local_worker_id=0,
            kv_dtype=dtype,
            kv_shape=shape,
        )

        backends = CreateStorageBackends(
            config,
            metadata,
            thread_loop,
            dst_device=get_correct_device(
                config.nixl_buffer_device, metadata.worker_id
            ),
        )
        assert len(backends) == 2  # NixlStorageBackend + LocalCPUBackend
        assert BACKEND_NAME in backends

        nixl_backend = backends[BACKEND_NAME]
        assert isinstance(nixl_backend, NixlStorageBackend)
        assert isinstance(nixl_backend.memory_allocator, PagedTensorMemoryAllocator)
        assert nixl_backend is not None
        assert nixl_backend.memory_allocator is not None

        # Allocate via the chunk shape that LMCacheMetadata derives from
        # kv_shape (LocalCPUBackend / NIXL share the same paged pool sized
        # by metadata.get_shapes()); passing the raw 5D kv_shape would
        # produce a same-byte-count but differently-indexed tensor.
        alloc_shape = metadata.get_shapes()[0]
        for key in keys:
            assert not nixl_backend.contains(key, False)
            assert not nixl_backend.exists_in_put_tasks(key)

            obj = nixl_backend.memory_allocator.allocate(alloc_shape, dtype)
            assert obj is not None
            assert obj.tensor is not None
            objs.append(obj)

        # small tensor changes for data validation (chunk shape is 4D:
        # [kv_size, num_layers, num_tokens, num_heads * head_size] =
        # [2, 4, 256, 1024] for kv_shape (4, 2, 256, 8, 128))
        objs[0].tensor[0, 0, 100, 200] = 1e-3
        objs[0].tensor[1, 0, 200, 100] = 1e-4

        objs[1].tensor[0, 1, 150, 400] = 1e-2
        objs[1].tensor[1, 1, 100, 300] = 1e-5

        objs[2].tensor[0, 2, 50, 200] = 3e-2
        objs[2].tensor[1, 3, 200, 100] = 4e-5

        # Insert first 2 keys
        first_keys = keys[0:2]
        first_objs = objs[0:2]
        nixl_backend.batched_submit_put_task(first_keys, first_objs)

        for key, obj in zip(first_keys, first_objs, strict=False):
            returned_memory_obj = nixl_backend.get_blocking(key)
            assert returned_memory_obj is not None
            assert returned_memory_obj.get_size() == obj.get_size()
            assert returned_memory_obj.get_shape() == obj.get_shape()
            assert returned_memory_obj.get_dtype() == obj.get_dtype()
            assert returned_memory_obj.metadata.address != obj.metadata.address

            returned_tensor = returned_memory_obj.tensor
            obj_tensor = obj.tensor
            assert returned_tensor is not None
            assert obj_tensor is not None
            assert torch.equal(returned_tensor, obj_tensor)

        obj_list = asyncio.run(
            nixl_backend.batched_get_non_blocking(lookup_id="test", keys=first_keys)
        )

        for i, obj in enumerate(first_objs):
            returned_memory_obj = obj_list[i]
            assert returned_memory_obj is not None
            assert returned_memory_obj.get_size() == obj.get_size()
            assert returned_memory_obj.get_shape() == obj.get_shape()
            assert returned_memory_obj.get_dtype() == obj.get_dtype()
            assert returned_memory_obj.metadata.address != obj.metadata.address

            returned_tensor = returned_memory_obj.tensor
            obj_tensor = obj.tensor
            assert returned_tensor is not None
            assert obj_tensor is not None
            assert torch.equal(returned_tensor, obj_tensor)

        def test_eviction(new_idx, old_idx):
            nixl_backend.batched_submit_put_task([keys[new_idx]], [objs[new_idx]])

            obj = nixl_backend.get_blocking(keys[new_idx])
            assert obj is not None
            assert obj.tensor is not None
            assert torch.equal(obj.tensor, objs[new_idx].tensor)

            obj = nixl_backend.get_blocking(keys[old_idx])
            assert obj is None

        ######## Test bad key lookup #########
        obj = nixl_backend.get_blocking(bad_key)
        assert obj is None

        ######## Test eviction #########
        obj = nixl_backend.get_blocking(keys[0])
        assert obj is not None

        # At this point, key 0 & key 1 are cached. Key 1 is LRU key.
        # Submitting key 2 should evict key 1.

        test_eviction(new_idx=2, old_idx=1)

        ######## Test pin #########
        val = nixl_backend.pin(keys[2])
        assert val is True

        obj = nixl_backend.get_blocking(keys[0])
        assert obj is not None

        # At this point, key 0 & key 2 are cached.
        # Key 2 is LRU key, but is pinned.
        # Submitting key 1 should evict key 0.

        test_eviction(new_idx=1, old_idx=0)

        ######## Test unpin #########
        val = nixl_backend.unpin(keys[2])
        assert val is True

        obj = nixl_backend.get_blocking(keys[1])
        assert obj is not None

        # At this point, key 1 & key 2 are cached.
        # Key 2 is LRU key, and is now unpinned.
        # Submitting key 0 should evict key 2.

        test_eviction(new_idx=0, old_idx=2)

        for backend in backends.values():
            backend.close()

    except Exception:
        raise
    finally:
        if thread_loop and thread_loop.is_running():
            thread_loop.call_soon_threadsafe(thread_loop.stop)
        if thread and thread.is_alive():
            thread.join()


@pytest.mark.no_shared_allocator
@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
@pytest.mark.skipif(
    not _can_register_file_with_nixl_backend("GDS_MT"),
    reason=_GDS_SKIP_REASON.format(backend="GDS_MT"),
)
def test_nixl_gds_mt_cuda_backend(nixl_tmp_path):
    BASE_DIR = Path(__file__).parent
    config = LMCacheEngineConfig.from_file(BASE_DIR / "data/nixl.yaml")

    dtype = torch.bfloat16
    shape = torch.Size([4, 2, 256, 8, 128])

    config.nixl_buffer_device = "cuda"
    # data/nixl.yaml is CPU-mode (no nixl_buffer_size); CUDA mode sizes the
    # NIXL buffer via nixl_buffer_size, so restore it when flipping the device.
    config.nixl_buffer_size = 1024**3
    config.extra_config["nixl_backend"] = "GDS_MT"
    config.extra_config["enable_cuda"] = True
    config.extra_config["nixl_path"] = nixl_tmp_path

    run(config, shape, dtype)


@pytest.mark.no_shared_allocator
@pytest.mark.skipif(
    not _can_register_file_with_nixl_backend("GDS_MT"),
    reason=_GDS_SKIP_REASON.format(backend="GDS_MT"),
)
def test_nixl_gds_mt_cpu_backend(nixl_tmp_path):
    BASE_DIR = Path(__file__).parent
    config = LMCacheEngineConfig.from_file(BASE_DIR / "data/nixl.yaml")

    dtype = torch.bfloat16
    shape = torch.Size([4, 2, 256, 8, 128])

    config.nixl_buffer_device = "cpu"
    config.extra_config["nixl_backend"] = "GDS_MT"
    config.extra_config["enable_cuda"] = False
    config.extra_config["nixl_path"] = nixl_tmp_path

    run(config, shape, dtype)


@pytest.mark.no_shared_allocator
@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
@pytest.mark.skipif(
    not _can_register_file_with_nixl_backend("GDS"),
    reason=_GDS_SKIP_REASON.format(backend="GDS"),
)
def test_nixl_gds_cuda_backend(nixl_tmp_path):
    BASE_DIR = Path(__file__).parent
    config = LMCacheEngineConfig.from_file(BASE_DIR / "data/nixl.yaml")

    dtype = torch.bfloat16
    shape = torch.Size([4, 2, 256, 8, 128])

    config.nixl_buffer_device = "cuda"
    # data/nixl.yaml is CPU-mode (no nixl_buffer_size); CUDA mode sizes the
    # NIXL buffer via nixl_buffer_size, so restore it when flipping the device.
    config.nixl_buffer_size = 1024**3
    config.extra_config["nixl_backend"] = "GDS"
    config.extra_config["enable_cuda"] = True
    config.extra_config["nixl_path"] = nixl_tmp_path

    run(config, shape, dtype)


@pytest.mark.no_shared_allocator
@pytest.mark.skipif(
    not _can_register_file_with_nixl_backend("GDS"),
    reason=_GDS_SKIP_REASON.format(backend="GDS"),
)
def test_nixl_gds_cpu_backend(nixl_tmp_path):
    BASE_DIR = Path(__file__).parent
    config = LMCacheEngineConfig.from_file(BASE_DIR / "data/nixl.yaml")

    dtype = torch.bfloat16
    shape = torch.Size([4, 2, 256, 8, 128])

    config.nixl_buffer_device = "cpu"
    config.extra_config["nixl_backend"] = "GDS"
    config.extra_config["enable_cuda"] = False
    config.extra_config["nixl_path"] = nixl_tmp_path

    run(config, shape, dtype)


@pytest.mark.no_shared_allocator
def test_nixl_endpoint_list_empty_raises():
    """nixl_endpoint_list=[] should raise ValueError before any nixl ops."""
    BASE_DIR = Path(__file__).parent
    config = LMCacheEngineConfig.from_file(BASE_DIR / "data/nixl.yaml")
    config.extra_config["nixl_endpoint_list"] = []

    metadata = LMCacheMetadata(
        model_name="Llama-3.1-70B-Instruct",
        world_size=1,
        local_world_size=1,
        worker_id=0,
        local_worker_id=0,
        kv_dtype=torch.bfloat16,
        kv_shape=(4, 2, 256, 8, 128),
    )

    with pytest.raises(ValueError, match="nixl_endpoint_list is set but empty"):
        NixlStorageConfig.from_cache_engine_config(config, metadata)


@pytest.mark.no_shared_allocator
def test_nixl_endpoint_list_malformed_url_raises():
    """A non-http(s) entry in nixl_endpoint_list should raise ValueError."""
    BASE_DIR = Path(__file__).parent
    config = LMCacheEngineConfig.from_file(BASE_DIR / "data/nixl.yaml")
    config.extra_config["nixl_endpoint_list"] = ["htps://typo.example.com"]
    config.extra_config["nixl_backend"] = "OBJ"

    metadata = LMCacheMetadata(
        model_name="Llama-3.1-70B-Instruct",
        world_size=1,
        local_world_size=1,
        worker_id=0,
        local_worker_id=0,
        kv_dtype=torch.bfloat16,
        kv_shape=(4, 2, 256, 8, 128),
    )

    with pytest.raises(ValueError, match="is not a valid URL"):
        NixlStorageConfig.from_cache_engine_config(config, metadata)


def _presence_cache_metadata() -> LMCacheMetadata:
    return LMCacheMetadata(
        model_name="Llama-3.1-70B-Instruct",
        world_size=1,
        local_world_size=1,
        worker_id=0,
        local_worker_id=0,
        kv_dtype=torch.bfloat16,
        kv_shape=(4, 2, 256, 8, 128),
    )


@pytest.mark.no_shared_allocator
def test_nixl_presence_cache_only_defaults_false():
    """presence_cache_only defaults to False when the config key is absent."""
    BASE_DIR = Path(__file__).parent
    config = LMCacheEngineConfig.from_file(BASE_DIR / "data/nixl.yaml")

    nixl_config = NixlStorageConfig.from_cache_engine_config(
        config, _presence_cache_metadata()
    )

    assert nixl_config.presence_cache_only is False


@pytest.mark.no_shared_allocator
def test_nixl_presence_cache_only_parsed():
    """nixl_presence_cache_only=True is parsed onto NixlStorageConfig."""
    BASE_DIR = Path(__file__).parent
    config = LMCacheEngineConfig.from_file(BASE_DIR / "data/nixl.yaml")
    config.extra_config["nixl_presence_cache"] = True
    config.extra_config["nixl_presence_cache_only"] = True

    nixl_config = NixlStorageConfig.from_cache_engine_config(
        config, _presence_cache_metadata()
    )

    assert nixl_config.presence_cache_only is True


@pytest.mark.no_shared_allocator
def test_nixl_presence_cache_only_requires_presence_cache():
    """nixl_presence_cache_only=True without nixl_presence_cache raises ValueError."""
    BASE_DIR = Path(__file__).parent
    config = LMCacheEngineConfig.from_file(BASE_DIR / "data/nixl.yaml")
    config.extra_config["nixl_presence_cache_only"] = True

    with pytest.raises(ValueError, match="nixl_presence_cache must be true"):
        config.validate()


@pytest.mark.no_shared_allocator
def test_nixl_posix_backend(nixl_tmp_path):
    BASE_DIR = Path(__file__).parent
    config = LMCacheEngineConfig.from_file(BASE_DIR / "data/nixl.yaml")

    dtype = torch.bfloat16
    shape = torch.Size([4, 2, 256, 8, 128])

    config.nixl_buffer_device = "cpu"
    config.extra_config["nixl_backend"] = "POSIX"
    config.extra_config["enable_cuda"] = False
    config.extra_config["nixl_path"] = nixl_tmp_path

    run(config, shape, dtype)


@pytest.mark.no_shared_allocator
def test_nixl_posix_backend_multipath():
    """Test NIXL backend with multipath support and path sharding."""
    BASE_DIR = Path(__file__).parent
    config = LMCacheEngineConfig.from_file(BASE_DIR / "data/nixl_multipath.yaml")

    dtype = torch.bfloat16
    shape = torch.Size([4, 2, 256, 8, 128])

    config.nixl_buffer_device = "cpu"
    config.extra_config["nixl_backend"] = "POSIX"
    config.extra_config["enable_cuda"] = False

    # Test that multipath configuration is properly handled
    assert isinstance(config.extra_config["nixl_path"], list)
    assert len(config.extra_config["nixl_path"]) == 3
    assert config.extra_config["nixl_path_sharding"] == "by_gpu"

    run(config, shape, dtype)
