# SPDX-License-Identifier: Apache-2.0
"""
Unit tests for the asymmetric K16/V8 split-tier (V-only)
multi-output serde in ``lmcache/v1/distributed/serde/asym_k16_v8.py``.

Validates the V-only-write path:

* ``serialize`` accepts ``src = (None, V)``: K is not written to
  the byte buffer (the K slot must be ``None``); only V_fp8 +
  scales hit L2.
* ``deserialize`` accepts ``src = blob``, ``dst = (None|K_skip,
  V_out)``: V is restored from the blob's FP8 + scales; the K slot
  is a no-op regardless of whether the caller provides one.

The byte-ratio claim under test: the V-only blob is
``V_8 / (K_16 + V_16) = 1/4`` of FP16 KV -- equivalently ``1/3``
of the corresponding storage-only-dequant blob (which carries both
K and V).

Cross-mode cases are also covered: a V-only deserializer must
refuse a storage-only-dequant blob (K is present), and the
storage-only-dequant deserializer would mis-decode a V-only blob
(only the V-only deserializer recognizes ``k_payload_len = 0``).
"""

# Future
from __future__ import annotations

# Standard
from dataclasses import dataclass
from typing import cast

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.distributed.api import MemoryLayoutDesc, ObjectKey
from lmcache.v1.distributed.serde.asym_k16_v8 import (
    AsymK16V8MultiSerializer,
    AsymK16V8VOnlyMultiDeserializer,
    AsymK16V8VOnlyMultiSerializer,
)
from lmcache.v1.distributed.serde.multi import MemoryObjGroup
from lmcache.v1.memory_management import MemoryObj

_TEST_KEY = ObjectKey(chunk_hash=b"\x00" * 32, model_name="test", kv_rank=0)

# Mirror the _FakeMemoryObj used elsewhere; lets the test stay GPU-free
# and L1Manager-free.


@dataclass
class _FakeMemoryObj:
    tensor: torch.Tensor


def _bf16_tensor(*shape: int, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randn(*shape, dtype=torch.bfloat16, generator=g).contiguous()


def _byte_buffer(num_bytes: int) -> MemoryObj:
    return cast(
        MemoryObj, _FakeMemoryObj(tensor=torch.zeros(num_bytes, dtype=torch.uint8))
    )


def _grp(*objs: object) -> MemoryObjGroup:
    """Cast a tuple of test fakes to the production MemoryObjGroup type."""
    return cast(MemoryObjGroup, objs)


# Llama-3.1-8B-Instruct-shaped chunk; same dimensions as the
# storage-only-dequant tests for direct ratio comparability.
_LLAMA_KV_SHAPE = (32, 64, 8, 128)


# =============================================================================
# group_size and contract surface
# =============================================================================


def test_group_size_is_two_on_both_endpoints() -> None:
    s = AsymK16V8VOnlyMultiSerializer()
    d = AsymK16V8VOnlyMultiDeserializer()
    assert s.group_size == 2
    assert d.group_size == 2


def test_serialize_rejects_k_present() -> None:
    """K slot must be None — providing K is a contract error."""
    s = AsymK16V8VOnlyMultiSerializer()
    k = _FakeMemoryObj(tensor=_bf16_tensor(2, 4, seed=0))
    v = _FakeMemoryObj(tensor=_bf16_tensor(2, 4, seed=1))
    buf = _byte_buffer(64)
    with pytest.raises(ValueError, match="K slot must be None"):
        s.serialize(_grp(k, v), buf, _TEST_KEY)


def test_serialize_requires_v() -> None:
    s = AsymK16V8VOnlyMultiSerializer()
    buf = _byte_buffer(64)
    with pytest.raises(ValueError, match="V slot is required"):
        s.serialize(_grp(None, None), buf, _TEST_KEY)
    v_no_tensor = _FakeMemoryObj(tensor=None)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="V slot is required"):
        s.serialize(_grp(None, v_no_tensor), buf, _TEST_KEY)


def test_estimate_serialized_size_rejects_k_layout_present() -> None:
    s = AsymK16V8VOnlyMultiSerializer()
    v_layout = MemoryLayoutDesc(shapes=[torch.Size([2, 4])], dtypes=[torch.bfloat16])
    k_layout = MemoryLayoutDesc(shapes=[torch.Size([2, 4])], dtypes=[torch.bfloat16])
    with pytest.raises(ValueError, match="K layout must be None"):
        s.estimate_serialized_size((k_layout, v_layout))
    with pytest.raises(ValueError, match="V layout is required"):
        s.estimate_serialized_size((None, None))


# =============================================================================
# Round-trip: V-only write, V-only read
# =============================================================================


def test_v_only_round_trip_v_within_fp8_noise() -> None:
    s = AsymK16V8VOnlyMultiSerializer()
    d = AsymK16V8VOnlyMultiDeserializer()

    v = _bf16_tensor(*_LLAMA_KV_SHAPE, seed=2)

    layout = (
        None,
        MemoryLayoutDesc(shapes=[v.shape], dtypes=[v.dtype]),
    )
    capacity = s.estimate_serialized_size(layout)
    buf = _byte_buffer(capacity)
    n = s.serialize(_grp(None, _FakeMemoryObj(tensor=v)), buf, _TEST_KEY)
    assert 0 < n <= capacity

    v_out = _FakeMemoryObj(tensor=torch.zeros_like(v))
    d.deserialize(buf, _grp(None, v_out), _TEST_KEY)

    v_diff = (v_out.tensor.float() - v.float()).abs()
    rel = v_diff / (v.float().abs() + 1e-6)
    assert rel.median().item() < 0.075, (
        f"V-only relative error median {rel.median().item():.4f} "
        f"exceeds FP8 noise threshold 0.075"
    )


def test_v_only_deserialize_ignores_k_slot() -> None:
    """A non-None K dst slot is a no-op for the V-only deserializer."""
    s = AsymK16V8VOnlyMultiSerializer()
    d = AsymK16V8VOnlyMultiDeserializer()

    v = _bf16_tensor(2, 4, 8, 64, seed=3)
    layout = (None, MemoryLayoutDesc(shapes=[v.shape], dtypes=[v.dtype]))
    buf = _byte_buffer(s.estimate_serialized_size(layout))
    s.serialize(_grp(None, _FakeMemoryObj(tensor=v)), buf, _TEST_KEY)

    sentinel = torch.full((2, 4, 8, 64), fill_value=42.0, dtype=torch.bfloat16)
    k_unused = _FakeMemoryObj(tensor=sentinel.clone())
    v_out = _FakeMemoryObj(tensor=torch.zeros_like(v))
    d.deserialize(buf, _grp(k_unused, v_out), _TEST_KEY)

    # K must be untouched.
    assert torch.equal(k_unused.tensor, sentinel)


# =============================================================================
# Byte ratio: V-only blob == 1/3 of storage-only-dequant blob
# =============================================================================


def test_v_only_blob_is_one_third_of_storage_only_blob() -> None:
    """V-only blob bytes = V_8 + scales + small header.

    Storage-only-dequant blob bytes = K_16 + V_8 + scales + small
    header.  The layout-invariant ratio V-only / storage-only =
    V_8 / (K_16 + V_8) = 1/3.  Test on a chunk size where the
    headers are negligible relative to the payload so the ratio
    resolves cleanly.
    """
    storage_only_s = AsymK16V8MultiSerializer()
    v_only_s = AsymK16V8VOnlyMultiSerializer()

    # Big-enough chunk that header overhead is < 0.5%.
    k = _bf16_tensor(*_LLAMA_KV_SHAPE, seed=4)
    v = _bf16_tensor(*_LLAMA_KV_SHAPE, seed=5)

    storage_only_layout = (
        MemoryLayoutDesc(shapes=[k.shape], dtypes=[k.dtype]),
        MemoryLayoutDesc(shapes=[v.shape], dtypes=[v.dtype]),
    )
    v_only_layout = (None, MemoryLayoutDesc(shapes=[v.shape], dtypes=[v.dtype]))

    storage_only_buf = _byte_buffer(
        storage_only_s.estimate_serialized_size(storage_only_layout)
    )
    v_only_buf = _byte_buffer(v_only_s.estimate_serialized_size(v_only_layout))
    storage_only_n = storage_only_s.serialize(
        _grp(_FakeMemoryObj(tensor=k), _FakeMemoryObj(tensor=v)),
        storage_only_buf,
        _TEST_KEY,
    )
    v_only_n = v_only_s.serialize(
        _grp(None, _FakeMemoryObj(tensor=v)), v_only_buf, _TEST_KEY
    )

    ratio = v_only_n / storage_only_n
    # Tolerance for header overhead.
    assert abs(ratio - 1.0 / 3.0) < 0.005, (
        f"v_only/storage_only byte ratio {ratio:.4f} should be ~0.3333; "
        f"storage_only_bytes={storage_only_n}, v_only_bytes={v_only_n}"
    )


# =============================================================================
# Cross-mode: V-only deserializer must refuse a storage-only-dequant blob
# =============================================================================


def test_v_only_deserializer_refuses_storage_only_blob() -> None:
    """A storage-only-dequant blob has k_payload_len > 0; the V-only
    deserializer must reject it rather than silently mis-decode."""
    storage_only_s = AsymK16V8MultiSerializer()
    v_only_d = AsymK16V8VOnlyMultiDeserializer()

    k = _bf16_tensor(2, 4, 8, 64, seed=6)
    v = _bf16_tensor(2, 4, 8, 64, seed=7)
    layout = (
        MemoryLayoutDesc(shapes=[k.shape], dtypes=[k.dtype]),
        MemoryLayoutDesc(shapes=[v.shape], dtypes=[v.dtype]),
    )
    buf = _byte_buffer(storage_only_s.estimate_serialized_size(layout))
    storage_only_s.serialize(
        _grp(_FakeMemoryObj(tensor=k), _FakeMemoryObj(tensor=v)), buf, _TEST_KEY
    )

    v_out = _FakeMemoryObj(tensor=torch.zeros_like(v))
    with pytest.raises(ValueError, match="storage-only-dequant"):
        v_only_d.deserialize(buf, _grp(None, v_out), _TEST_KEY)
