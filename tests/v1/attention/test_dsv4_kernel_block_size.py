# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.models.deepseek_v4.nvidia.flashinfer_sparse import (
    DeepseekV4FlashInferMLASparseBackend,
)
from vllm.models.deepseek_v4.sparse_mla import (
    DeepseekV4SparseMLABackend,
    dsv4_supported_kernel_block_sizes,
)
from vllm.platforms import current_platform
from vllm.v1.attention.backends.mla.indexer import DeepseekV4IndexerBackend


def test_dsv4_kernel_block_size_sm12x(monkeypatch):
    monkeypatch.setattr(
        current_platform, "is_device_capability_family", lambda family: family == 120
    )
    assert dsv4_supported_kernel_block_sizes() == [64]
    assert DeepseekV4SparseMLABackend.get_supported_kernel_block_sizes() == [64]
    assert DeepseekV4FlashInferMLASparseBackend.get_supported_kernel_block_sizes() == [
        64
    ]
    assert DeepseekV4IndexerBackend.get_supported_kernel_block_sizes() == [64]


def test_dsv4_kernel_block_size_not_sm12x(monkeypatch):
    monkeypatch.setattr(
        current_platform, "is_device_capability_family", lambda family: False
    )
    assert dsv4_supported_kernel_block_sizes() == [256]
    assert DeepseekV4SparseMLABackend.get_supported_kernel_block_sizes() == [256]
    assert DeepseekV4FlashInferMLASparseBackend.get_supported_kernel_block_sizes() == [
        256
    ]
    assert DeepseekV4IndexerBackend.get_supported_kernel_block_sizes() == [256]
