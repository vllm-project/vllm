# SPDX-License-Identifier: Apache-2.0
"""
Unit tests for the multi-output serde extensions in
``lmcache/v1/distributed/serde/multi.py``.

These tests exercise the additive contract:

* Fixed-length :class:`MemoryObjGroup` semantics, including the
  ``None`` slot meaning "absent on serialize input" or "skip on
  deserialize output".
* The single-to-multi adapters preserve exact bytes vs the
  underlying single-tensor :class:`Serializer` /
  :class:`Deserializer` (so existing serdes opt into the group
  call site without changing their on-the-wire format).
* :func:`validate_group_size` rejects mismatched group lengths
  with messages that name the offending side.

The test deliberately uses a toy "concat" multi-serde defined
in-file (rather than importing a production multi-serde) so that
the tests pin down the API contract without depending on any
specific concrete implementation. The format is documented inline.
"""

# Future
from __future__ import annotations

# Standard
from dataclasses import dataclass
from typing import cast
import struct

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.distributed.api import MemoryLayoutDesc, ObjectKey
from lmcache.v1.distributed.serde.base import Deserializer, Serializer
from lmcache.v1.distributed.serde.multi import (
    LayoutDescGroup,
    MemoryObjGroup,
    MultiDeserializer,
    MultiSerializer,
    single_to_multi_deserializer,
    single_to_multi_serializer,
    validate_group_size,
)

_TEST_KEY = ObjectKey(chunk_hash=b"\x00" * 32, model_name="test", kv_rank=0)

# =============================================================================
# Test scaffolding: a minimal MemoryObj stand-in mirroring test_fp8.py.
# =============================================================================


@dataclass
class _FakeMemoryObj:
    """Minimal stand-in exposing the ``.tensor`` attribute used by serdes.

    Mirrors the ``_FakeMemoryObj`` in ``test_fp8.py`` so the multi-serde
    tests stay GPU-free and L1Manager-free.
    """

    tensor: torch.Tensor


def _byte_buffer(num_bytes: int) -> _FakeMemoryObj:
    return _FakeMemoryObj(tensor=torch.zeros(num_bytes, dtype=torch.uint8))


def _bf16_tensor_obj(*shape: int, seed: int = 0) -> _FakeMemoryObj:
    g = torch.Generator().manual_seed(seed)
    t = torch.randn(*shape, dtype=torch.bfloat16, generator=g).contiguous()
    return _FakeMemoryObj(tensor=t)


# =============================================================================
# Toy reference multi-serde used to validate the API contract.
#
# Wire format (group of fixed length N):
#   header: N bytes of present-mask (0 or 1 per slot)
#         + N * uint32 little-endian payload-length (0 when absent)
#   body:   concatenation of the present slots' raw tensor bytes,
#           in slot order. Absent slots contribute zero bytes.
#
# Header byte size = N + 4*N = 5*N. Payload size is the sum of present
# slots' tensor byte sizes. Total = 5*N + sum(present payloads).
# =============================================================================


_MASK_FMT = struct.Struct("<B")  # one byte per present-mask entry
_LEN_FMT = struct.Struct("<I")  # uint32 little-endian per length


def _header_size(group_size: int) -> int:
    return group_size * (_MASK_FMT.size + _LEN_FMT.size)


def _tensor_bytes(t: torch.Tensor) -> bytes:
    # Reinterpret as uint8 to avoid Python bytes() per-byte iteration on
    # storage. Mirrors the trick used elsewhere in the tree but kept
    # local so this test file does not depend on production helpers.
    return t.contiguous().view(torch.uint8).numpy().tobytes()


class ConcatMultiSerializer(MultiSerializer):
    """Toy multi-serializer that concatenates present slots verbatim."""

    def __init__(self, group_size: int) -> None:
        if group_size <= 0:
            raise ValueError(f"group_size must be positive, got {group_size}")
        self._group_size = group_size

    @property
    def group_size(self) -> int:
        return self._group_size

    def serialize(self, src: MemoryObjGroup, dst, key) -> int:
        validate_group_size(src, self._group_size, role="src")
        # Build header and payload separately so we can write into dst
        # in two contiguous moves.
        masks = bytearray()
        lens = bytearray()
        payload = bytearray()
        for slot in src:
            if slot is None:
                masks += _MASK_FMT.pack(0)
                lens += _LEN_FMT.pack(0)
                continue
            if slot.tensor is None:
                raise ValueError(
                    "ConcatMultiSerializer: a non-None group slot must "
                    "have a tensor attribute set"
                )
            blob = _tensor_bytes(slot.tensor)
            masks += _MASK_FMT.pack(1)
            lens += _LEN_FMT.pack(len(blob))
            payload += blob
        header = bytes(masks) + bytes(lens)
        total = len(header) + len(payload)

        if dst.tensor is None:
            raise ValueError("ConcatMultiSerializer: dst.tensor is None")
        if dst.tensor.numel() < total:
            raise ValueError(
                f"ConcatMultiSerializer: dst capacity {dst.tensor.numel()} "
                f"is below required {total}"
            )

        dst_view = dst.tensor.view(torch.uint8)
        dst_view[: len(header)].copy_(
            torch.frombuffer(bytearray(header), dtype=torch.uint8)
        )
        if payload:
            dst_view[len(header) : total].copy_(
                torch.frombuffer(bytearray(payload), dtype=torch.uint8)
            )
        return total

    def estimate_serialized_size(
        self,
        layout_descs: LayoutDescGroup,
    ) -> int:
        validate_group_size(layout_descs, self._group_size, role="layout")
        total = _header_size(self._group_size)
        for desc in layout_descs:
            if desc is None:
                continue
            for shape, dtype in zip(desc.shapes, desc.dtypes, strict=True):
                numel = 1
                for dim in shape:
                    numel *= int(dim)
                total += numel * dtype.itemsize
        return total


class ConcatMultiDeserializer(MultiDeserializer):
    """Inverse of :class:`ConcatMultiSerializer`."""

    def __init__(self, group_size: int) -> None:
        if group_size <= 0:
            raise ValueError(f"group_size must be positive, got {group_size}")
        self._group_size = group_size

    @property
    def group_size(self) -> int:
        return self._group_size

    def deserialize(self, src, dst: MemoryObjGroup, key) -> None:
        validate_group_size(dst, self._group_size, role="dst")
        if src.tensor is None:
            raise ValueError("ConcatMultiDeserializer: src.tensor is None")

        src_view = src.tensor.view(torch.uint8)
        n = self._group_size
        present = [bool(src_view[i].item()) for i in range(n)]
        lens_off = n
        lens = [
            int(
                _LEN_FMT.unpack_from(
                    src_view[
                        lens_off + i * _LEN_FMT.size : lens_off
                        + (i + 1) * _LEN_FMT.size
                    ]
                    .numpy()
                    .tobytes()
                )[0]
            )
            for i in range(n)
        ]
        cursor = _header_size(n)
        for i, slot in enumerate(dst):
            this_len = lens[i]
            if slot is None:
                cursor += this_len
                continue
            if not present[i]:
                # Caller asked for a slot the producer did not write.
                # Leave dst untouched; this mirrors the wrapper-side
                # handling for "absent on serialize" cases.
                continue
            if slot.tensor is None:
                raise ValueError(
                    "ConcatMultiDeserializer: a non-None dst slot must "
                    "have a tensor attribute set"
                )
            payload = src_view[cursor : cursor + this_len]
            slot_view = slot.tensor.view(torch.uint8).flatten()
            if slot_view.numel() < this_len:
                raise ValueError(
                    f"ConcatMultiDeserializer: dst slot {i} capacity "
                    f"{slot_view.numel()} below payload {this_len}"
                )
            slot_view[:this_len].copy_(payload)
            cursor += this_len


# =============================================================================
# group_size invariants
# =============================================================================


def test_group_size_property_is_fixed() -> None:
    s = ConcatMultiSerializer(group_size=2)
    d = ConcatMultiDeserializer(group_size=2)
    assert s.group_size == 2
    assert d.group_size == 2


def test_group_size_must_be_positive() -> None:
    with pytest.raises(ValueError):
        ConcatMultiSerializer(group_size=0)
    with pytest.raises(ValueError):
        ConcatMultiDeserializer(group_size=-1)


def test_validate_group_size_rejects_mismatch() -> None:
    with pytest.raises(ValueError, match="src"):
        validate_group_size((None,), expected=2, role="src")
    with pytest.raises(ValueError, match="dst"):
        validate_group_size((None, None, None), expected=2, role="dst")


# =============================================================================
# Round-trip with all slots present
# =============================================================================


def test_two_slot_roundtrip_all_present() -> None:
    s = ConcatMultiSerializer(group_size=2)
    d = ConcatMultiDeserializer(group_size=2)

    k = _bf16_tensor_obj(2, 4, 8, seed=1)
    v = _bf16_tensor_obj(2, 4, 8, seed=2)
    src: MemoryObjGroup = cast(MemoryObjGroup, (k, v))

    layout = (
        MemoryLayoutDesc(shapes=[k.tensor.shape], dtypes=[k.tensor.dtype]),
        MemoryLayoutDesc(shapes=[v.tensor.shape], dtypes=[v.tensor.dtype]),
    )
    capacity = s.estimate_serialized_size(layout)
    buf = _byte_buffer(capacity)
    n = s.serialize(src, buf, _TEST_KEY)
    assert n <= capacity

    k_out = _FakeMemoryObj(tensor=torch.zeros_like(k.tensor))
    v_out = _FakeMemoryObj(tensor=torch.zeros_like(v.tensor))
    d.deserialize(buf, (k_out, v_out), _TEST_KEY)  # type: ignore[arg-type]

    assert torch.equal(k_out.tensor, k.tensor)
    assert torch.equal(v_out.tensor, v.tensor)


# =============================================================================
# None on serialize input: absent K slot
# =============================================================================


def test_serialize_with_none_slot_skips_payload() -> None:
    s = ConcatMultiSerializer(group_size=2)
    d = ConcatMultiDeserializer(group_size=2)

    v = _bf16_tensor_obj(2, 4, 8, seed=3)
    src: MemoryObjGroup = cast(MemoryObjGroup, (None, v))

    layout: LayoutDescGroup = (
        None,
        MemoryLayoutDesc(shapes=[v.tensor.shape], dtypes=[v.tensor.dtype]),
    )
    capacity = s.estimate_serialized_size(layout)
    buf = _byte_buffer(capacity)
    n = s.serialize(src, buf, _TEST_KEY)

    # Capacity must accommodate the full payload exactly when absences
    # are accounted for; the toy header is 5*group_size so the absent
    # K slot only saves the K payload, not the header bookkeeping.
    expected = 2 * 5 + v.tensor.numel() * v.tensor.dtype.itemsize
    assert n == expected

    # Round-trip into matching dst group: K slot left None to mirror.
    v_out = _FakeMemoryObj(tensor=torch.zeros_like(v.tensor))
    d.deserialize(buf, (None, v_out), _TEST_KEY)  # type: ignore[arg-type]

    assert torch.equal(v_out.tensor, v.tensor)


# =============================================================================
# None on deserialize output: skip K materialization on read
# =============================================================================


def test_deserialize_with_none_slot_leaves_caller_buffer_untouched() -> None:
    s = ConcatMultiSerializer(group_size=2)
    d = ConcatMultiDeserializer(group_size=2)

    k = _bf16_tensor_obj(1, 2, 4, seed=4)
    v = _bf16_tensor_obj(1, 2, 4, seed=5)
    layout = (
        MemoryLayoutDesc(shapes=[k.tensor.shape], dtypes=[k.tensor.dtype]),
        MemoryLayoutDesc(shapes=[v.tensor.shape], dtypes=[v.tensor.dtype]),
    )
    capacity = s.estimate_serialized_size(layout)
    buf = _byte_buffer(capacity)
    s.serialize((k, v), buf, _TEST_KEY)  # type: ignore[arg-type]

    # Deserialize, but skip the K slot.
    sentinel = torch.full_like(k.tensor, fill_value=42.0)
    k_out_unused = _FakeMemoryObj(tensor=sentinel.clone())
    v_out = _FakeMemoryObj(tensor=torch.zeros_like(v.tensor))
    d.deserialize(buf, (None, v_out), _TEST_KEY)  # type: ignore[arg-type]

    # k_out_unused must still equal the sentinel: deserialize did not
    # touch a None dst slot.
    assert torch.equal(k_out_unused.tensor, sentinel)
    assert torch.equal(v_out.tensor, v.tensor)


# =============================================================================
# Single-tensor adapter: equivalent bytes vs the underlying serde
# =============================================================================


class _IdentitySerializer(Serializer):
    """Trivial single-tensor serializer copying tensor bytes verbatim."""

    def serialize(self, src, dst, key) -> int:
        if src.tensor is None or dst.tensor is None:
            raise ValueError("identity serde requires tensors on both sides")
        blob = _tensor_bytes(src.tensor)
        dst_view = dst.tensor.view(torch.uint8)
        if dst_view.numel() < len(blob):
            raise ValueError("identity serde: dst capacity too small")
        dst_view[: len(blob)].copy_(
            torch.frombuffer(bytearray(blob), dtype=torch.uint8)
        )
        return len(blob)

    def estimate_serialized_size(self, layout_desc: MemoryLayoutDesc) -> int:
        total = 0
        for shape, dtype in zip(layout_desc.shapes, layout_desc.dtypes, strict=True):
            numel = 1
            for dim in shape:
                numel *= int(dim)
            total += numel * dtype.itemsize
        return total


class _IdentityDeserializer(Deserializer):
    """Inverse of :class:`_IdentitySerializer`."""

    def deserialize(self, src, dst, key) -> None:
        if src.tensor is None or dst.tensor is None:
            raise ValueError("identity serde requires tensors on both sides")
        n = dst.tensor.numel() * dst.tensor.dtype.itemsize
        src_view = src.tensor.view(torch.uint8)
        dst_view = dst.tensor.view(torch.uint8).flatten()
        dst_view[:n].copy_(src_view[:n])


def test_single_to_multi_serializer_round_trip_equivalence() -> None:
    """A length-1 group MUST produce the same bytes as direct invocation."""
    inner_s = _IdentitySerializer()
    inner_d = _IdentityDeserializer()
    multi_s = single_to_multi_serializer(inner_s)
    multi_d = single_to_multi_deserializer(inner_d)
    assert multi_s.group_size == 1
    assert multi_d.group_size == 1

    src = _bf16_tensor_obj(2, 4, 8, seed=6)
    layout = MemoryLayoutDesc(shapes=[src.tensor.shape], dtypes=[src.tensor.dtype])

    direct_buf = _byte_buffer(inner_s.estimate_serialized_size(layout))
    direct_n = inner_s.serialize(src, direct_buf, _TEST_KEY)

    multi_buf = _byte_buffer(multi_s.estimate_serialized_size((layout,)))
    multi_n = multi_s.serialize((src,), multi_buf, _TEST_KEY)  # type: ignore[arg-type]

    assert direct_n == multi_n
    assert torch.equal(direct_buf.tensor, multi_buf.tensor)

    direct_out = _FakeMemoryObj(tensor=torch.zeros_like(src.tensor))
    inner_d.deserialize(direct_buf, direct_out, _TEST_KEY)

    multi_out = _FakeMemoryObj(tensor=torch.zeros_like(src.tensor))
    multi_d.deserialize(multi_buf, (multi_out,), _TEST_KEY)  # type: ignore[arg-type]

    assert torch.equal(direct_out.tensor, multi_out.tensor)


def test_single_to_multi_serializer_rejects_non_unit_group() -> None:
    multi_s = single_to_multi_serializer(_IdentitySerializer())
    src_a = _bf16_tensor_obj(2, 2, seed=7)
    src_b = _bf16_tensor_obj(2, 2, seed=8)
    buf = _byte_buffer(64)
    with pytest.raises(ValueError, match="size 1"):
        multi_s.serialize((src_a, src_b), buf, _TEST_KEY)  # type: ignore[arg-type]


def test_single_to_multi_serializer_rejects_none_slot() -> None:
    multi_s = single_to_multi_serializer(_IdentitySerializer())
    buf = _byte_buffer(64)
    with pytest.raises(ValueError, match="None src"):
        multi_s.serialize((None,), buf, _TEST_KEY)  # type: ignore[arg-type]


def test_single_to_multi_deserializer_treats_none_slot_as_skip() -> None:
    """A length-1 group with None dst is a deliberate no-op, not an error."""
    multi_d = single_to_multi_deserializer(_IdentityDeserializer())
    src = _byte_buffer(8)
    # Deliberately skip the only output: must not raise.
    multi_d.deserialize(src, (None,), _TEST_KEY)  # type: ignore[arg-type]
