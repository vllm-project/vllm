# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Variant metadata tests for ROCm AITER MLA backends.

This file owns the small backend-contract checks that are still useful after
splitting the other MLA coverage by responsibility:
- ``test_rocm_aiter_mla.py`` covers the dense ROCm AITER MLA kernel path
- ``test_rocm_aiter_mla_sparse.py`` covers the sparse ROCm AITER MLA kernels
- ``tests/v1/attention/test_rocm_attention_backends_selection.py`` covers enum
  wiring and selector behavior

What remains here is the variant-specific class wiring for:
- ``AiterTritonMLABackend``
- ``ROCMAiterMLASparseBackend``
"""

import pytest
import torch

from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(
    not current_platform.is_rocm(), reason="ROCm-specific tests"
)


# Triton MLA variant tests ------------------------------------------------


def test_aiter_triton_mla_variant_identity():
    """The Triton MLA variant should expose the impl and capability markers
    that make it distinct from dense ROCm MLA."""
    from vllm.v1.attention.backends.mla.aiter_triton_mla import (
        AiterTritonMLABackend,
        AiterTritonMLAImpl,
    )

    assert AiterTritonMLABackend.is_mla() is True
    assert AiterTritonMLABackend.get_impl_cls() is AiterTritonMLAImpl


def test_aiter_triton_mla_inherits_dense_rocm_mla_contract():
    """The Triton variant should keep the same backend metadata contract as
    dense ROCm MLA for dtypes, block sizes, and metadata builder wiring."""
    from vllm.v1.attention.backends.mla.aiter_triton_mla import (
        AiterTritonMLABackend,
    )
    from vllm.v1.attention.backends.mla.rocm_aiter_mla import AiterMLABackend

    assert AiterTritonMLABackend.supported_dtypes == AiterMLABackend.supported_dtypes
    assert (
        AiterTritonMLABackend.supported_kv_cache_dtypes
        == AiterMLABackend.supported_kv_cache_dtypes
    )
    assert (
        AiterTritonMLABackend.get_supported_kernel_block_sizes()
        == AiterMLABackend.get_supported_kernel_block_sizes()
    )
    assert (
        AiterTritonMLABackend.get_supported_head_sizes()
        == AiterMLABackend.get_supported_head_sizes()
    )
    assert AiterTritonMLABackend.get_builder_cls() is AiterMLABackend.get_builder_cls()


# Sparse MLA variant tests ------------------------------------------------


def test_rocm_sparse_mla_backend_identity_and_class_wiring():
    """The sparse ROCm MLA backend should publish the classes vLLM relies on
    for metadata building and kernel dispatch."""
    from vllm.v1.attention.backends.mla.rocm_aiter_mla_sparse import (
        ROCMAiterMLASparseBackend,
        ROCMAiterMLASparseImpl,
        ROCMAiterMLASparseMetadata,
        ROCMAiterMLASparseMetadataBuilder,
    )

    assert ROCMAiterMLASparseBackend.is_mla() is True
    assert ROCMAiterMLASparseBackend.is_sparse() is True
    assert ROCMAiterMLASparseBackend.accept_output_buffer is True
    assert ROCMAiterMLASparseBackend.get_impl_cls() is ROCMAiterMLASparseImpl
    assert ROCMAiterMLASparseBackend.get_metadata_cls() is ROCMAiterMLASparseMetadata
    assert (
        ROCMAiterMLASparseBackend.get_builder_cls() is ROCMAiterMLASparseMetadataBuilder
    )


def test_rocm_sparse_mla_backend_static_contract():
    """The sparse ROCm MLA backend should keep its advertised dtype and
    block-size contract."""
    from vllm.v1.attention.backends.mla.rocm_aiter_mla_sparse import (
        ROCMAiterMLASparseBackend,
    )

    assert ROCMAiterMLASparseBackend.supported_dtypes == [
        torch.float16,
        torch.bfloat16,
    ]
    assert ROCMAiterMLASparseBackend.supported_kv_cache_dtypes == [
        "auto",
        "float16",
        "bfloat16",
    ]
    assert ROCMAiterMLASparseBackend.get_supported_kernel_block_sizes() == [1]
    assert ROCMAiterMLASparseBackend.get_supported_head_sizes() == []


def test_rocm_sparse_mla_kv_cache_shape_contract():
    """Sparse ROCm MLA should describe a token-level KV cache shape."""
    from vllm.v1.attention.backends.mla.rocm_aiter_mla_sparse import (
        ROCMAiterMLASparseBackend,
    )

    assert ROCMAiterMLASparseBackend.get_kv_cache_shape(
        num_blocks=11,
        block_size=1,
        num_kv_heads=1,
        head_size=576,
        cache_dtype_str="bfloat16",
    ) == (11, 1, 576)
