# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import contextlib

import pytest

from vllm.platforms import current_platform


@pytest.fixture
def registered_nixl_worker(monkeypatch, dist_init):
    """Construct and register real workers, replacing only the NIXL transport."""
    from vllm.config import set_current_vllm_config
    from vllm.distributed.kv_transfer.kv_connector.v1.nixl import base_worker
    from vllm.distributed.kv_transfer.kv_connector.v1.nixl.worker import (
        NixlConnectorWorker,
    )
    from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheGroupSpec

    from .test_nixl_connector import FakeNixlWrapper
    from .utils import create_vllm_config

    monkeypatch.setattr(base_worker, "NixlWrapper", FakeNixlWrapper)
    with contextlib.ExitStack() as stack:

        def make_worker(spec, caches, layout="LBHNC", enable_permute_local_kv=False):
            config = create_vllm_config(
                block_size=spec.block_size,
                enable_permute_local_kv=enable_permute_local_kv,
                hf_overrides={
                    "num_attention_heads": spec.num_kv_heads,
                    "hidden_size": spec.num_kv_heads * spec.head_size,
                    "num_hidden_layers": len(caches),
                },
            )
            config.cache_config.kv_cache_layout = layout
            config.kv_transfer_config.kv_buffer_device = "cpu"
            kv_config = KVCacheConfig(
                num_blocks=next(iter(caches.values())).shape[0],
                kv_cache_tensors=[],
                kv_cache_groups=[KVCacheGroupSpec(list(caches), spec)],
            )
            stack.enter_context(set_current_vllm_config(config))
            worker = NixlConnectorWorker(
                config, config.kv_transfer_config.engine_id, kv_config
            )
            stack.callback(worker.shutdown)
            worker.register_kv_caches(caches)
            return worker

        yield make_worker


@pytest.fixture(autouse=True)
def _mock_cpu_backend_block_sizes(monkeypatch):
    """Mock the CPU attention backend's kernel block-size declaration.

    Connector unit tests exercise transfer logic, never the attention
    kernel, and hard-code block_size=16. On CPU-only machines
    `CPUAttentionBackend` declares `MultipleOf(32)` (#54042), so every
    worker construction would fail in `select_common_block_size` before
    reaching the code under test.
    """
    if not current_platform.is_cpu():
        return
    from vllm.v1.attention.backend import MultipleOf
    from vllm.v1.attention.backends.cpu_attn import CPUAttentionBackend

    monkeypatch.setattr(
        CPUAttentionBackend,
        "get_supported_kernel_block_sizes",
        staticmethod(lambda kv_cache_spec=None: [MultipleOf(16)]),
    )
