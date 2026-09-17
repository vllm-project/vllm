# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kernel block size selection for the DeepSeek-V4.1 sparse-MLA backends.

FlashInfer's SM120 sparse-MLA decode kernels only ship 64-token pages
(``flashinfer/mla/_sparse_mla_sm120.py``:
``_DECODE_DSV4_PAGE_BLOCK_SIZE = 64``), and the DeepGEMM paged-MQA logits
kernel accepts ``block_kv`` only in ``{32, 64}``. V4.1 mixes ratio-1 and
ratio-2 indexer layers, and the indexer feeds
``num_states = kernel_block_size // compress_ratio`` to that kernel, so a
128-token kernel block lands on 128 for ratio 1 and trips the assert.

These tests patch the platform capability, so they do not need an SM120 GPU.
"""

from unittest.mock import patch

import pytest

from vllm.models.deepseek_v41.attention import _swa_cache_block_size
from vllm.models.deepseek_v41.nvidia.flashinfer_sparse import (
    DeepseekV4FlashInferMLASparseBackend,
)
from vllm.models.deepseek_v41.sparse_mla import (
    DeepseekV4FlashMLABackend,
    DeepseekV4SparseMLABackend,
    FlashMLAMegaAttnBackend,
)
from vllm.platforms import current_platform
from vllm.platforms.interface import DeviceCapability
from vllm.v1.attention.backends.mla.indexer import DeepseekV41IndexerBackend

pytestmark = pytest.mark.skipif(
    not current_platform.is_cuda(),
    reason="DeepSeek V4.1 sparse MLA backends require a CUDA build of vLLM",
)

# (major, minor, expected sparse-MLA kernel block size)
# SM90/SM120 page the cache at 64 tokens, SM100 at 128.
SPARSE_MLA_BLOCK_SIZES = [
    pytest.param(9, 0, 64, id="sm90-hopper"),
    pytest.param(10, 0, 128, id="sm100-blackwell-dc"),
    pytest.param(12, 0, 64, id="sm120-blackwell-client"),
    pytest.param(12, 1, 64, id="sm121-gb10"),
]


def _capability(major: int, minor: int):
    """Patch the platform's reported device capability for one test."""
    return patch.object(
        type(current_platform),
        "get_device_capability",
        return_value=DeviceCapability(major, minor),
    )


@pytest.mark.parametrize(("major", "minor", "expected"), SPARSE_MLA_BLOCK_SIZES)
def test_sparse_mla_kernel_block_size(major: int, minor: int, expected: int):
    with _capability(major, minor):
        assert DeepseekV4SparseMLABackend.get_supported_kernel_block_sizes() == [
            expected
        ]
        assert DeepseekV4FlashMLABackend.get_supported_kernel_block_sizes() == [
            expected
        ]
        assert (
            DeepseekV4FlashInferMLASparseBackend.get_supported_kernel_block_sizes()
            == [expected]
        )
        assert DeepseekV41IndexerBackend.get_supported_kernel_block_sizes() == [
            expected
        ]


@pytest.mark.parametrize(("major", "minor", "expected"), SPARSE_MLA_BLOCK_SIZES)
def test_mega_attn_backend_stays_sm100_only(major: int, minor: int, expected: int):
    # FlashMLAMegaAttnBackend has no SM90/SM120 kernel instantiation, so it
    # keeps declaring 128 regardless of the capability.
    with _capability(major, minor):
        assert FlashMLAMegaAttnBackend.get_supported_kernel_block_sizes() == [128]


@pytest.mark.parametrize(
    ("major", "minor", "expected"),
    [
        pytest.param(9, 0, 32, id="sm90-hopper"),
        pytest.param(10, 0, 32, id="sm100-blackwell-dc"),
        pytest.param(12, 0, 64, id="sm120-blackwell-client"),
        pytest.param(12, 1, 64, id="sm121-gb10"),
    ],
)
def test_swa_cache_block_size(major: int, minor: int, expected: int):
    with _capability(major, minor):
        assert _swa_cache_block_size() == expected
