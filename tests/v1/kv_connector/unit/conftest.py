# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest
import torch

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


@pytest.fixture
def register_mooncake_store_caches():
    """Keep CPU tensors alive while Store layouts retain their raw addresses."""
    buffers: list[torch.Tensor] = []

    def register(layout, block_lens: list[int], num_blocks: int = 32):
        tensors = [
            torch.zeros((num_blocks, length), dtype=torch.uint8, device="cpu")
            for length in block_lens
        ]
        layout.register_kv_caches(tensors, num_blocks)
        buffers.extend(tensors)
        return tensors

    return register
