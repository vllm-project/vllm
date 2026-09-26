# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""A KV-cache backend must not advertise an fp8 format its writer cannot store.

``triton_reshape_and_cache_flash`` stores ``current_platform.fp8_dtype()``, an e4m3
variant on every platform, while ``chunked_prefill_paged_decode`` and ``prefix_prefill``
decode an ``fp8_e5m2`` cache as ``torch.float8_e5m2``. Writing e4m3 bits under an e5m2
label reinterprets them rather than converting, and the request still succeeds, so the
failure is silent.
"""

import torch

from vllm.platforms import current_platform
from vllm.v1.attention.backends.rocm_attn import RocmAttentionBackend
from vllm.v1.attention.backends.triton_attn import TritonAttentionBackend
from vllm.v1.attention.ops.triton_reshape_and_cache_flash import (
    _is_supported_kv_cache_dtype,
)

# Both backends route KV writes through triton_reshape_and_cache_flash:
# TritonAttentionBackend unconditionally, RocmAttentionBackend for any block size that
# is not 16 or 32 -- and get_supported_kernel_block_sizes permits MultipleOf(16).
_TRITON_WRITER_BACKENDS = (TritonAttentionBackend, RocmAttentionBackend)


def test_the_kernel_stores_an_e4m3_variant():
    """The premise. If this ever changes, the rest of the file needs rewriting."""
    assert current_platform.fp8_dtype() in (
        torch.float8_e4m3fn,
        torch.float8_e4m3fnuz,
    )


def test_e5m2_is_refused_by_the_writer():
    """Refused for a format reason, so it must hold on every device."""
    assert not _is_supported_kv_cache_dtype("fp8_e5m2")


def test_unquantized_dtypes_are_untouched():
    """A control: the refusal is about the fp8 format, not about caching in general."""
    assert _is_supported_kv_cache_dtype("auto")
    assert _is_supported_kv_cache_dtype("float32")


def test_backends_do_not_advertise_a_format_the_writer_refuses():
    for backend in _TRITON_WRITER_BACKENDS:
        advertised = set(backend.supported_kv_cache_dtypes)
        assert "fp8_e5m2" not in advertised, (
            f"{backend.__name__} advertises fp8_e5m2, but its KV writer stores "
            f"{current_platform.fp8_dtype()}; the cache would be decoded as e5m2"
        )


def test_backends_still_advertise_the_format_the_writer_does_store():
    """The other control: e4m3 must not have been swept away with e5m2."""
    for backend in _TRITON_WRITER_BACKENDS:
        advertised = set(backend.supported_kv_cache_dtypes)
        assert {"fp8", "fp8_e4m3"} <= advertised, (
            f"{backend.__name__} no longer advertises the fp8 format it can store"
        )
