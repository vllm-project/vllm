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
        staticmethod(lambda: [MultipleOf(16)]),
    )
