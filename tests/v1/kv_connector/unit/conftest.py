# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest

from vllm.platforms import current_platform


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


@pytest.fixture(autouse=True)
def _fake_remote_layer_meta_tensors(monkeypatch):
    """Remote metadata under test advertises no views, so stand in for them
    (see `fake_remote_layer_meta_tensors`)."""
    from tests.v1.kv_connector.unit.utils import fake_remote_layer_meta_tensors
    from vllm.distributed.kv_transfer.kv_connector.v1.nixl.base_worker import (
        NixlBaseConnectorWorker,
    )

    tp_byte_ranges = NixlBaseConnectorWorker._tp_byte_ranges

    def with_remote_views(self, meta, regions, remote_tp_size, *args, **kwargs):
        fake_remote_layer_meta_tensors(self, meta, regions, remote_tp_size)
        return tp_byte_ranges(self, meta, regions, remote_tp_size, *args, **kwargs)

    monkeypatch.setattr(NixlBaseConnectorWorker, "_tp_byte_ranges", with_remote_views)
