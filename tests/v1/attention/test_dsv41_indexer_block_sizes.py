# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DSv4.1 SM12x kernel block-size declarations and resolved sizes.

Pins the SM12x declarations touched by the uniform-64 change (indexer,
FlashInfer sparse), the deliberately-unchanged sparse_mla base declaration,
the SWA/compressor geometry, and the block size resolved through
Platform._preferred_block_size_for_backends for the full DSv4.1 backend set.
"""

from types import SimpleNamespace

import pytest

from vllm.platforms import current_platform
from vllm.platforms.interface import Platform
from vllm.v1.attention.backend import MultipleOf
from vllm.v1.attention.backends.mla.indexer import DeepseekV41IndexerBackend
from vllm.models.deepseek_v41.sparse_mla import DeepseekV4SparseMLABackend
from vllm.models.deepseek_v41.nvidia.flashinfer_sparse import (
    DeepseekV4FlashInferMLASparseBackend,
    DeepseekSparseSWAFlashInferBackend,
)
from vllm.models.deepseek_v41.compressor import CompressorBackend

# SM12x declarations changed by this PR: 64-token kernel blocks.
UNIFORM = [64]

# Backends actually touched. The sparse_mla base declaration is deliberately
# NOT touched: on SM12x it is unreachable (FLASHMLA_SPARSE_DSV41 gates
# supports_compute_capability to major 9/10, and the ROCm backend overrides it).
V41_BACKENDS = [
    DeepseekV41IndexerBackend,
    DeepseekV4FlashInferMLASparseBackend,
]

# The full DSv4.1 backend set the platform resolves the default launch over.
V41_SET = V41_BACKENDS + [
    DeepseekSparseSWAFlashInferBackend,
    CompressorBackend,
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


def test_sm12x_default_launch_resolves_64_for_full_set(sm12x):
    # What this PR actually changes: the default launch resolves through
    # the platform over the whole DSv4.1 backend set. With the indexer and
    # FlashInfer sparse backends declaring [64] on SM12x (and the SWA path
    # a multiple of 32), the smallest size every backend accepts is 64 —
    # instead of the stock 128 that the FlashInfer SM120 kernels reject.
    resolved = Platform._preferred_block_size_for_backends(
        V41_SET, 16, SimpleNamespace()
    )
    assert resolved == 64


@pytest.mark.parametrize("default,expected", [(16, 64), (64, 64), (128, 128)])
@pytest.mark.parametrize("backend", V41_BACKENDS)
def test_sm12x_resolved_block_size(sm12x, backend, default, expected):
    # vLLM default block size (16) is not a multiple of any declared size,
    # so the resolved size is min(declared) = 64; explicit 64 and 128 are
    # multiples of a declared size and are kept.
    assert backend.get_preferred_block_size(default) == expected


@pytest.mark.parametrize("family,base_expected", [(90, [64]), (100, [128])])
def test_non_sm12x_sparse_mla_base_declaration(monkeypatch, family, base_expected):
    monkeypatch.setattr(current_platform, "is_rocm", lambda: False)
    monkeypatch.setattr(
        current_platform, "is_device_capability_family", lambda f: f == family
    )
    # Off SM12x the sparse_mla base keeps its Hopper/Blackwell split and the
    # FlashInfer sparse backend keeps 128 — the change is SM12x-only.
    assert DeepseekV4SparseMLABackend.get_supported_kernel_block_sizes() == base_expected
    assert DeepseekV4FlashInferMLASparseBackend.get_supported_kernel_block_sizes() == [128]
