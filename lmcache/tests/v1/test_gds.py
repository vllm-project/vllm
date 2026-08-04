# SPDX-License-Identifier: Apache-2.0
# Standard
from pathlib import Path
import asyncio
import os
import shutil
import tempfile
import threading

# Third Party
import pytest
import safetensors
import torch

# First Party
from lmcache.utils import CacheEngineKey
from lmcache.v1.cache_engine import LMCacheEngineBuilder
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.memory_allocators.cu_file_memory_allocator import CuFileMemoryAllocator
from lmcache.v1.memory_allocators.hip_file_memory_allocator import (
    HipFileMemoryAllocator,
)
from lmcache.v1.memory_management import MemoryFormat
from lmcache.v1.storage_backend import CreateStorageBackends
from lmcache.v1.storage_backend.gds_backend import pack_metadata, unpack_metadata


def test_gds_backend_metadata():
    # This is a sanity check that packing and unpacking works. We can add
    # more tensor types to be sure.
    for [tensor, expected_nbytes] in [(torch.randn(3, 10), 120)]:
        r = pack_metadata(tensor, fmt=MemoryFormat.KV_2LTD, version="test")
        size, dtype, nbytes, fmt, meta = unpack_metadata(r)
        assert size == tensor.size()
        assert dtype == tensor.dtype
        assert expected_nbytes == nbytes
        assert fmt == MemoryFormat.KV_2LTD
        assert meta["version"] == "test"

        # Make sure that safetensors can load this
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_file_path = os.path.join(temp_dir, "test.safetensors")
            with open(temp_file_path, "wb") as f:
                f.write(r)
                f.write(b" " * nbytes)

            with safetensors.safe_open(temp_file_path, framework="pt") as f:
                tensor = f.get_tensor("kvcache")
                assert size == tensor.size()
                assert dtype == tensor.dtype
                assert expected_nbytes == nbytes


@pytest.mark.skip(reason="We need to add this test back after implementing prefetch")
def test_gds_backend_sanity():
    BASE_DIR = Path(__file__).parent
    GDS_DIR = "/tmp/gds/test-cache"
    TEST_KEY = CacheEngineKey(
        model_name="meta-llama/Llama-3.1-70B-Instruct",
        world_size=8,
        worker_id=0,
        chunk_hash="e3229141e680fb413d2c5d3ebb416c4ad300d381e309fc9e417757b91406c157",
        dtype=torch.uint8,
    )
    BACKEND_NAME = "GdsBackend"

    try:
        os.makedirs(GDS_DIR, exist_ok=True)
        config_gds = LMCacheEngineConfig.from_file(BASE_DIR / "data/gds.yaml")
        assert config_gds.gds_buffer_size == 128

        thread_loop = asyncio.new_event_loop()
        thread = threading.Thread(target=thread_loop.run_forever)
        thread.start()

        backends = CreateStorageBackends(
            config_gds,
            None,
            thread_loop,
            LMCacheEngineBuilder._Create_memory_allocator(config_gds, None),
        )
        assert len(backends) == 2
        assert BACKEND_NAME in backends

        gds_backend = backends[BACKEND_NAME]
        assert gds_backend is not None
        assert gds_backend.memory_allocator is not None
        assert isinstance(
            gds_backend.memory_allocator,
            (CuFileMemoryAllocator, HipFileMemoryAllocator),
        )

        assert not gds_backend.contains(TEST_KEY, False)
        assert not gds_backend.exists_in_put_tasks(TEST_KEY)

        memory_obj = gds_backend.memory_allocator.allocate(
            [2048, 2048], dtype=torch.uint8
        )
        future = gds_backend.submit_put_task(TEST_KEY, memory_obj)
        assert future is not None
        assert gds_backend.exists_in_put_tasks(TEST_KEY)
        assert not gds_backend.contains(TEST_KEY, False)
        future.result()
        assert gds_backend.contains(TEST_KEY, False)
        assert not gds_backend.exists_in_put_tasks(TEST_KEY)

        returned_memory_obj = gds_backend.get_blocking(TEST_KEY)
        assert returned_memory_obj is not None
        assert returned_memory_obj.get_size() == memory_obj.get_size()
        assert returned_memory_obj.get_shape() == memory_obj.get_shape()
        assert returned_memory_obj.get_dtype() == memory_obj.get_dtype()

        future = gds_backend.get_non_blocking(TEST_KEY)
        assert future is not None
        returned_memory_obj = future.result()
        assert returned_memory_obj is not None
        assert returned_memory_obj.get_size() == memory_obj.get_size()
        assert returned_memory_obj.get_shape() == memory_obj.get_shape()
        assert returned_memory_obj.get_dtype() == memory_obj.get_dtype()
    finally:
        if thread_loop.is_running():
            thread_loop.call_soon_threadsafe(thread_loop.stop)
        if thread.is_alive():
            thread.join()
        # We rmtree AFTER we ensure that the thread loop is done.
        # This way we don't hit any race conditions in rmtree()
        # where temp files are renamed while we try to unlink them.
        # We also take care of any other errors with ignore_errors=True
        # so if we want to run tests in parallel in the future they
        # don't make each other fail.
        if os.path.exists(GDS_DIR):
            shutil.rmtree(GDS_DIR)
