# SPDX-License-Identifier: Apache-2.0

# Standard
from unittest.mock import patch
import tempfile
import time

# Third Party
import pytest
import torch

pytest.importorskip("vllm", reason="EC connector adapter imports vLLM at module top")

# First Party
from lmcache.integration.vllm.vllm_ec_adapter import (  # noqa: E402
    LMCacheECConnectorImpl,
    LMCacheECConnectorMetadata,
    MMMeta,
)


class _FakeTransferConfig:
    def __init__(self, storage_path: str):
        self._storage_path = storage_path

    def get_from_extra_config(self, key: str, default=None):
        if key == "shared_storage_path":
            return self._storage_path
        return default


class _FakeVllmConfig:
    def __init__(self, storage_path: str):
        self.ec_transfer_config = _FakeTransferConfig(storage_path)
        self.model_config = _FakeModelConfig()
        self.parallel_config = _FakeParallelConfig()
        self.cache_config = _FakeCacheConfig()


class _FakeModelConfig:
    model = "fake-model"
    served_model_name = "fake-model"
    dtype = torch.float16

    def get_num_layers(self, parallel_config):
        return 1

    def get_num_kv_heads(self, parallel_config):
        return 1

    def get_head_size(self):
        return 1


class _FakeParallelConfig:
    world_size = 1
    rank = 0


class _FakeCacheConfig:
    cache_dtype = "auto"


class _FakeRole:
    name = "WORKER"


class _FakeParent:
    def __init__(self):
        self.is_producer = True
        self._meta = None

    def _get_connector_metadata(self):
        return self._meta


def test_ec_roundtrip_save_then_load():
    with tempfile.TemporaryDirectory() as td:
        vllm_config = _FakeVllmConfig(td)
        parent = _FakeParent()
        conn = LMCacheECConnectorImpl(
            vllm_config=vllm_config,
            role=_FakeRole(),
            parent=parent,
        )

        try:
            mm_hash = "hash_abc"
            x = torch.randn(7, 13, dtype=torch.float16)
            encoder_cache = {mm_hash: x}

            conn.save_caches(encoder_cache, mm_hash)

            # EC storage is asynchronous across tiers; wait briefly for visibility.
            deadline = time.time() + 2.0
            while not conn.has_cache_item(mm_hash) and time.time() < deadline:
                time.sleep(0.05)

            assert conn.has_cache_item(mm_hash)

            # Minimal fake metadata plumbing
            meta = LMCacheECConnectorMetadata()
            meta.add_mm_data(MMMeta.make_meta(mm_hash))
            parent._meta = meta

            # Pin device to "cpu" for the load path: on CPU-only CI runners
            # vLLM's UnspecifiedPlatform returns an empty device_type, which
            # would make tensor.to(device="") raise. The test exercises the
            # round-trip itself, not vLLM's device discovery.
            encoder_cache2 = {}
            with patch(
                "lmcache.integration.vllm.vllm_ec_adapter.torch_device_type",
                new="cpu",
            ):
                conn.start_load_caches(encoder_cache2)

            assert mm_hash in encoder_cache2
            assert encoder_cache2[mm_hash].shape == x.shape
            assert encoder_cache2[mm_hash].dtype == x.dtype
        finally:
            conn.close()
