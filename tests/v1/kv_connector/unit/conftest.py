# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import os
from unittest.mock import MagicMock, patch

import pytest

from vllm.platforms import current_platform


@pytest.fixture
def nixl_worker_factory(dist_init):
    """Construct real workers; only the external NIXL transport is mocked."""
    from vllm.config import set_current_vllm_config
    from vllm.distributed.kv_transfer.kv_connector.v1.nixl import base_worker

    from .utils import create_vllm_config, make_kv_cache_config

    workers = []
    affinity = os.sched_getaffinity(0)

    def create(worker_type, engine_id):
        config = create_vllm_config(block_size=16)
        caches = make_kv_cache_config(16, num_blocks=512)
        with set_current_vllm_config(config):
            worker = worker_type(config, engine_id, caches)
        workers.append(worker)
        worker.nixl_wrapper.get_agent_metadata.return_value = engine_id.encode()
        return worker

    with patch.object(base_worker, "NixlWrapper") as transport:
        # Each worker owns a distinct external transport agent.
        transport.side_effect = lambda *args, **kwargs: MagicMock()
        try:
            yield create
        finally:
            for worker in reversed(workers):
                worker.shutdown()
            os.sched_setaffinity(0, affinity)


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
