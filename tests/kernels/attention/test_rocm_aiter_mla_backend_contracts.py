# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Backend contract tests for the ROCm AITER MLA attention backends.

These are pure class-metadata checks with no kernel execution, so they pin the
contracts that the attention registry, the KV cache spec and the metadata
builders rely on without needing MI3xx hardware:

* the class accessors (``get_impl_cls`` / ``get_builder_cls``) stay wired to
  the matching concrete classes,
* ``AiterTritonMLABackend`` keeps sharing the dense AITER builder and
  dtype/block-size contract while overriding only the implementation, and
* ``ROCMAiterMLASparseBackend`` declares the MLA/sparse flags and the stricter
  per-token KV cache block-size requirement that the sparse path depends on.

See the ROCm AITER MLA kernel coverage tracker for the wider matrix.
"""

from typing import get_args

import pytest
import torch

from vllm.platforms import current_platform

if not current_platform.is_rocm():
    pytest.skip(
        "ROCm AITER MLA backend contract tests require ROCm.",
        allow_module_level=True,
    )

from vllm.v1.attention.backend import AttentionMetadataBuilder, MultipleOf
from vllm.v1.attention.backends.mla.aiter_triton_mla import AiterTritonMLABackend
from vllm.v1.attention.backends.mla.rocm_aiter_mla import AiterMLABackend
from vllm.v1.attention.backends.mla.rocm_aiter_mla_sparse import (
    ROCMAiterMLASparseBackend,
    ROCMAiterMLASparseImpl,
    ROCMAiterMLASparseMetadata,
    ROCMAiterMLASparseMetadataBuilder,
)


def _block_size_bases(backend_cls) -> list[int]:
    """Concrete bases of the advertised kernel block sizes.

    ``MultipleOf`` defines no ``__eq__``, so comparing the objects themselves
    would never match; compare the bases instead.
    """
    return [
        size.base if isinstance(size, MultipleOf) else size
        for size in backend_cls.get_supported_kernel_block_sizes()
    ]


def test_triton_mla_shares_dense_builder_and_impl_base() -> None:
    """AITER_TRITON_MLA must reuse the dense builder and metadata contract.

    The Triton variant only swaps the attention implementation, so it has to
    keep consuming the dense builder. If it drifted to its own builder, the two
    backends could desync from the shared kernel metadata layout.
    """
    assert AiterTritonMLABackend.get_builder_cls() is AiterMLABackend.get_builder_cls()
    assert issubclass(
        AiterTritonMLABackend.get_impl_cls(), AiterMLABackend.get_impl_cls()
    )


def test_triton_mla_overrides_only_the_name_and_impl() -> None:
    """The Triton backend must change the name and impl, and nothing else."""
    assert AiterTritonMLABackend.get_name() != AiterMLABackend.get_name()
    assert AiterTritonMLABackend.get_impl_cls() is not AiterMLABackend.get_impl_cls()
    assert AiterTritonMLABackend.supported_dtypes == AiterMLABackend.supported_dtypes
    assert (
        AiterTritonMLABackend.supported_kv_cache_dtypes
        == AiterMLABackend.supported_kv_cache_dtypes
    )
    assert _block_size_bases(AiterTritonMLABackend) == _block_size_bases(
        AiterMLABackend
    )


def test_dense_mla_declares_no_head_size_restriction() -> None:
    """AITER MLA handles every MLA head size, so it filters none out."""
    assert AiterMLABackend.get_supported_head_sizes() == []
    # An empty list means "no restriction", not "nothing supported".
    assert AiterMLABackend.supports_head_size(576)
    assert AiterMLABackend.supports_head_size(320)


def test_dense_mla_supports_the_full_fp8_kv_cache_set() -> None:
    """The dense AITER MLA decode kernel accepts every advertised cache dtype."""
    for dtype in (torch.float16, torch.bfloat16):
        assert AiterMLABackend.supports_dtype(dtype)
    assert not AiterMLABackend.supports_dtype(torch.float32)

    for cache_dtype in (
        "auto",
        "float16",
        "bfloat16",
        "fp8",
        "fp8_e4m3",
        "fp8_e5m2",
    ):
        assert AiterMLABackend.supports_kv_cache_dtype(cache_dtype)


def test_dense_mla_accepts_any_kernel_block_size() -> None:
    """The dense builder expands page indices, so every block size works."""
    # MultipleOf(1) is the permissive declaration: any block size divides it.
    assert _block_size_bases(AiterMLABackend) == [1]
    for block_size in (1, 16, 32, 64, 128):
        assert AiterMLABackend.supports_block_size(block_size)


def test_sparse_backend_class_accessors_are_wired() -> None:
    """ROCM_AITER_MLA_SPARSE must return its dedicated builder and impl.

    The sparse backend is standalone and does not inherit the dense one, so
    nothing else guarantees these accessors track the sparse declarations.
    """
    assert ROCMAiterMLASparseBackend.get_builder_cls() is (
        ROCMAiterMLASparseMetadataBuilder
    )
    assert issubclass(ROCMAiterMLASparseMetadataBuilder, AttentionMetadataBuilder)
    # The builder is declared as AttentionMetadataBuilder[<metadata>], so the
    # generic parameter is the metadata class it is required to build.
    assert get_args(ROCMAiterMLASparseMetadataBuilder.__orig_bases__[0]) == (
        ROCMAiterMLASparseMetadata,
    )
    assert ROCMAiterMLASparseBackend.get_impl_cls() is ROCMAiterMLASparseImpl


def test_sparse_backend_is_mla_and_sparse() -> None:
    """The sparse backend must advertise both flags the selector keys on."""
    assert ROCMAiterMLASparseBackend.is_mla() is True
    assert ROCMAiterMLASparseBackend.is_sparse() is True
    # The dense AITER backend is MLA but deliberately not sparse.
    assert AiterMLABackend.is_mla() is True
    assert AiterMLABackend.is_sparse() is False


def test_sparse_backend_restricts_kernel_block_size() -> None:
    """The sparse path declares block sizes 1 and multiples of 16.

    The dense backend accepts any block size; the sparse one is stricter, and
    ``supports_block_size`` has to keep agreeing with what
    ``get_supported_kernel_block_sizes`` advertises.
    """
    assert _block_size_bases(ROCMAiterMLASparseBackend) == [1, 16]
    for block_size in (1, 16, 32, 64, 128):
        assert ROCMAiterMLASparseBackend.supports_block_size(block_size)
    # Not a multiple of 16, so the sparse kernel cannot consume it.
    assert not ROCMAiterMLASparseBackend.supports_block_size(24)


def test_sparse_backend_supports_fp8_but_not_e5m2() -> None:
    """Pin the sparse fp8 cache set, which is narrower than the dense one.

    The sparse metadata builder normalizes ``fp8_e5m2`` to an fp8 cache dtype,
    but the backend does not advertise it, so selection rejects it. Locking the
    current contract keeps an accidental widening or narrowing visible.
    """
    for cache_dtype in ("auto", "float16", "bfloat16", "fp8", "fp8_e4m3"):
        assert ROCMAiterMLASparseBackend.supports_kv_cache_dtype(cache_dtype)

    assert not ROCMAiterMLASparseBackend.supports_kv_cache_dtype("fp8_e5m2")
    assert AiterMLABackend.supports_kv_cache_dtype("fp8_e5m2")


def test_sparse_backend_supported_dtypes() -> None:
    """Sparse MLA kernels are fp16/bf16 only."""
    for dtype in (torch.float16, torch.bfloat16):
        assert ROCMAiterMLASparseBackend.supports_dtype(dtype)
    assert not ROCMAiterMLASparseBackend.supports_dtype(torch.float32)
