# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DSv4.1 SM12x kernel block-size declarations and resolved sizes.

Pins the declarations touched by the SM12x uniform-64 change and the block
size resolved through AttentionBackend.get_preferred_block_size for the
DSv4.1 backend set (indexer, FlashInfer sparse, sparse SWA, compressor),
for the default launch and with --block-size 64 / 128.
"""

import pytest

from vllm.platforms import current_platform
from vllm.v1.attention.backend import MultipleOf
from vllm.v1.attention.backends.mla.indexer import DeepseekV41IndexerBackend
from vllm.models.deepseek_v41.sparse_mla import DeepseekV4SparseMLABackend
from vllm.models.deepseek_v41.nvidia.flashinfer_sparse import (
    DeepseekV4FlashInferMLASparseBackend,
    DeepseekSparseSWAFlashInferBackend,
)
from vllm.models.deepseek_v41.compressor import CompressorBackend

# Expected SM12x declarations. NOTE: if the PR adopts [64, 128] (keeps
# native 128-token kernel blocks for --block-size 128), update UNIFORM
# accordingly — the resolution tests below are written to pass for both
# [64] and [64, 128] since min() and the modulo checks agree.
UNIFORM = [64]

# The backends actually touched by the SM12x change. The sparse_mla base
# declaration is deliberately NOT touched: on SM12x it is unreachable
# (FLASHMLA_SPARSE_DSV41 gates supports_compute_capability to major 9/10,
# and the ROCm backend overrides it).
V41_BACKENDS = [
    DeepseekV41IndexerBackend,
    DeepseekV4FlashInferMLASparseBackend,
]


@pytest.fixture
def sm12x(monkeypatch):
    monkeypatch.setattr(current_platform, "is_rocm", lambda: False)
    monkeypatch.setattr(
        current_platform,
        "is_device_capability_family",
        lambda family: family == 120,
    )


@pytest.mark.parametrize("backend", V41_BACKENDS)
def test_sm12x_v41_declarations(sm12x, backend):
    assert backend.get_supported_kernel_block_sizes() == UNIFORM


def test_sm12x_sparse_mla_base_declaration_unchanged(sm12x):
    # Unreachable on SM12x at runtime (see comment above), so it keeps the
    # stock declaration — pin that so the change stays scoped.
    assert DeepseekV4SparseMLABackend.get_supported_kernel_block_sizes() == [128]


def test_sm12x_swa_and_compressor_unchanged(sm12x):
    # The SWA path stays on its own page geometry (multiples of 32,
    # preferred 256) and the compressor accepts any alignment.
    swa_sizes = DeepseekSparseSWAFlashInferBackend.get_supported_kernel_block_sizes()
    assert len(swa_sizes) == 1
    assert isinstance(swa_sizes[0], MultipleOf) and swa_sizes[0].base == 32
    assert DeepseekSparseSWAFlashInferBackend.get_preferred_block_size(16) == 256
    comp = CompressorBackend.get_supported_kernel_block_sizes()
    assert len(comp) == 1
    assert isinstance(comp[0], MultipleOf) and comp[0].base == 1


@pytest.mark.parametrize("default,expected", [(16, 64), (64, 64), (128, 128)])
@pytest.mark.parametrize("backend", V41_BACKENDS)
def test_sm12x_resolved_block_size(sm12x, backend, default, expected):
    # vLLM default block size (16) is not a multiple of any declared size,
    # so the resolved size is min(declared) = 64; explicit 64 and 128 are
    # multiples of a declared size and are kept.
    assert backend.get_preferred_block_size(default) == expected


def test_non_sm12x_declarations_unchanged(monkeypatch):
    monkeypatch.setattr(current_platform, "is_rocm", lambda: False)
    monkeypatch.setattr(
        current_platform, "is_device_capability_family", lambda family: False
    )
    # Hopper keeps the 64/128 split, FlashInfer sparse keeps 128.
    assert DeepseekV4SparseMLABackend.get_supported_kernel_block_sizes() == [128]
    assert DeepseekV4FlashInferMLASparseBackend.get_supported_kernel_block_sizes() == [128]
