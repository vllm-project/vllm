# SPDX-License-Identifier: Apache-2.0
"""
Unit tests for the asymmetric K16/V8 multi-output serde in
``lmcache/v1/distributed/serde/asym_k16_v8.py``.

Validates the storage-only-dequant path: ``(K, V)`` group on
serialize, ``(K_out, V_out)`` group on deserialize, K bit-exact
and V within FP8 noise.
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
    AsymK16V8MultiDeserializer,
    AsymK16V8MultiSerializer,
)
from lmcache.v1.distributed.serde.multi import MemoryObjGroup
from lmcache.v1.memory_management import MemoryObj

_TEST_KEY = ObjectKey(chunk_hash=b"\x00" * 32, model_name="test", kv_rank=0)

# =============================================================================
# Test scaffolding (mirrors the _FakeMemoryObj in test_multi.py / test_fp8.py)
# =============================================================================


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


# Llama-3.1-8B-Instruct-shaped chunk: (n_layers=32, seq=64, n_kv_heads=8,
# head_dim=128) is large enough to exercise per-tensor V scaling without
# wasting test runtime.
_LLAMA_KV_SHAPE = (32, 64, 8, 128)


# =============================================================================
# group_size and contract surface
# =============================================================================


def test_group_size_is_two_on_both_endpoints() -> None:
    s = AsymK16V8MultiSerializer()
    d = AsymK16V8MultiDeserializer()
    assert s.group_size == 2
    assert d.group_size == 2


def test_validate_group_size_rejects_wrong_arity() -> None:
    s = AsymK16V8MultiSerializer()
    d = AsymK16V8MultiDeserializer()
    buf = _byte_buffer(64)

    with pytest.raises(ValueError, match="src group length 1"):
        s.serialize(_grp(_FakeMemoryObj(tensor=torch.zeros(8))), buf, _TEST_KEY)
    with pytest.raises(ValueError, match="dst group length 3"):
        d.deserialize(
            buf,
            _grp(
                _FakeMemoryObj(tensor=torch.zeros(8)),
                _FakeMemoryObj(tensor=torch.zeros(8)),
                _FakeMemoryObj(tensor=torch.zeros(8)),
            ),
            _TEST_KEY,
        )


def test_serialize_rejects_none_slot_in_storage_only_mode() -> None:
    """Storage-only-dequant mode requires both K and V."""
    s = AsymK16V8MultiSerializer()
    buf = _byte_buffer(64)
    v = _FakeMemoryObj(tensor=_bf16_tensor(2, 4, seed=0))
    with pytest.raises(ValueError, match="both K and V must be provided"):
        s.serialize(_grp(None, v), buf, _TEST_KEY)


# =============================================================================
# Round-trip: K bit-exact, V within FP8 noise
# =============================================================================


def test_storage_only_round_trip_k_bit_exact_v_within_fp8_noise() -> None:
    s = AsymK16V8MultiSerializer()
    d = AsymK16V8MultiDeserializer()

    n_layers, seq, n_heads, head_dim = _LLAMA_KV_SHAPE
    k = _bf16_tensor(n_layers, seq, n_heads, head_dim, seed=1)
    v = _bf16_tensor(n_layers, seq, n_heads, head_dim, seed=2)
    src = _grp(_FakeMemoryObj(tensor=k), _FakeMemoryObj(tensor=v))

    layout = (
        MemoryLayoutDesc(shapes=[k.shape], dtypes=[k.dtype]),
        MemoryLayoutDesc(shapes=[v.shape], dtypes=[v.dtype]),
    )
    capacity = s.estimate_serialized_size(layout)
    buf = _byte_buffer(capacity)
    n = s.serialize(src, buf, _TEST_KEY)
    assert 0 < n <= capacity

    k_out = _FakeMemoryObj(tensor=torch.zeros_like(k))
    v_out = _FakeMemoryObj(tensor=torch.zeros_like(v))
    d.deserialize(buf, _grp(k_out, v_out), _TEST_KEY)

    # K must be bit-exact.
    assert torch.equal(k_out.tensor, k), "K must round-trip bit-exact"

    # V is dequantized from FP8 e4m3 — bounded but non-zero error.
    v_diff = (v_out.tensor.float() - v.float()).abs()
    rel = v_diff / (v.float().abs() + 1e-6)
    assert rel.median().item() < 0.075, (
        f"V relative error median {rel.median().item():.4f} exceeds "
        f"FP8 noise threshold 0.075"
    )


def test_storage_only_round_trip_with_skipped_v_dst_leaves_buf_untouched() -> None:
    """Pass (K_out, None) to load K only; the V slot is a deliberate skip."""
    s = AsymK16V8MultiSerializer()
    d = AsymK16V8MultiDeserializer()

    k = _bf16_tensor(4, 8, 8, 64, seed=3)
    v = _bf16_tensor(4, 8, 8, 64, seed=4)
    layout = (
        MemoryLayoutDesc(shapes=[k.shape], dtypes=[k.dtype]),
        MemoryLayoutDesc(shapes=[v.shape], dtypes=[v.dtype]),
    )
    buf = _byte_buffer(s.estimate_serialized_size(layout))
    s.serialize(
        _grp(_FakeMemoryObj(tensor=k), _FakeMemoryObj(tensor=v)), buf, _TEST_KEY
    )

    sentinel = torch.full_like(v, fill_value=42.0)
    v_unused = _FakeMemoryObj(tensor=sentinel.clone())
    k_out = _FakeMemoryObj(tensor=torch.zeros_like(k))
    d.deserialize(buf, _grp(k_out, None), _TEST_KEY)

    assert torch.equal(k_out.tensor, k)
    assert torch.equal(v_unused.tensor, sentinel), (
        "deserialize must not touch a None dst slot"
    )


def test_estimate_serialized_size_is_non_decreasing_in_chunk_size() -> None:
    """Sanity: larger chunks predict larger byte budgets."""
    s = AsymK16V8MultiSerializer()
    small = (
        MemoryLayoutDesc(shapes=[torch.Size([2, 4, 8, 64])], dtypes=[torch.bfloat16]),
        MemoryLayoutDesc(shapes=[torch.Size([2, 4, 8, 64])], dtypes=[torch.bfloat16]),
    )
    large = (
        MemoryLayoutDesc(shapes=[torch.Size([2, 4, 8, 256])], dtypes=[torch.bfloat16]),
        MemoryLayoutDesc(shapes=[torch.Size([2, 4, 8, 256])], dtypes=[torch.bfloat16]),
    )
    assert s.estimate_serialized_size(small) < s.estimate_serialized_size(large)


def test_estimate_serialized_size_meets_actual_blob_length() -> None:
    """estimate_serialized_size must be an upper bound on the produced blob."""
    s = AsymK16V8MultiSerializer()
    k = _bf16_tensor(2, 4, 8, 64, seed=5)
    v = _bf16_tensor(2, 4, 8, 64, seed=6)
    layout = (
        MemoryLayoutDesc(shapes=[k.shape], dtypes=[k.dtype]),
        MemoryLayoutDesc(shapes=[v.shape], dtypes=[v.dtype]),
    )
    capacity = s.estimate_serialized_size(layout)
    buf = _byte_buffer(capacity)
    n = s.serialize(
        _grp(_FakeMemoryObj(tensor=k), _FakeMemoryObj(tensor=v)), buf, _TEST_KEY
    )
    assert n <= capacity, (
        f"actual blob {n} bytes exceeded estimate {capacity}; "
        f"estimate must be a true upper bound"
    )
