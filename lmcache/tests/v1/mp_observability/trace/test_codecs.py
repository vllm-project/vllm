# SPDX-License-Identifier: Apache-2.0

"""Round-trip tests for the trace codec registry."""

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.distributed.api import (
    AttnWindowDesc,
    MemoryLayoutDesc,
    ObjectKey,
    PrefetchHandle,
    TrimPolicy,
)
from lmcache.v1.mp_observability.trace import codecs


def _roundtrip(v):
    """Encode then decode through msgpack to mirror the recorder path."""
    # Third Party
    import msgspec

    encoded = codecs.encode_value(v)
    blob = msgspec.msgpack.encode(encoded)
    decoded_msgpack = msgspec.msgpack.decode(blob)
    return codecs.decode_value(decoded_msgpack)


class TestPrimitives:
    @pytest.mark.parametrize("v", [0, 1, -1, 3.14, "hi", b"bytes", True, False, None])
    def test_passthrough(self, v):
        assert _roundtrip(v) == v

    def test_list(self):
        assert _roundtrip([1, "a", None]) == [1, "a", None]

    def test_dict(self):
        assert _roundtrip({"a": 1, "b": [2, 3]}) == {"a": 1, "b": [2, 3]}

    def test_tuple_preserved(self):
        out = _roundtrip((1, 2, "x"))
        assert out == (1, 2, "x")
        assert isinstance(out, tuple)


class TestObjectKey:
    def test_roundtrip(self):
        k = ObjectKey(chunk_hash=b"\x00\x01\x02", model_name="m", kv_rank=42)
        out = _roundtrip(k)
        assert out == k

    def test_object_group_id_roundtrip(self):
        k = ObjectKey(
            chunk_hash=b"\x00\x01\x02",
            model_name="m",
            kv_rank=42,
            object_group_id=7,
        )
        out = _roundtrip(k)
        assert out == k
        assert out.object_group_id == 7

    def test_inside_list(self):
        keys = [
            ObjectKey(chunk_hash=b"a", model_name="m", kv_rank=1),
            ObjectKey(chunk_hash=b"b", model_name="m", kv_rank=2),
        ]
        assert _roundtrip(keys) == keys


class TestMemoryLayoutDesc:
    def test_roundtrip(self):
        d = MemoryLayoutDesc(
            shapes=[torch.Size([2, 3]), torch.Size([4])],
            dtypes=[torch.float16, torch.bfloat16],
        )
        out = _roundtrip(d)
        assert out == d


class TestPrefetchHandle:
    def test_roundtrip(self):
        h = PrefetchHandle(
            prefetch_request_id=7,
            external_request_id="req-1",
            l1_found_indices=(0, 1, 2),
            total_requested_keys=10,
            submit_time=12345.6,
            l2_orig_indices=(3, 4, 5),
        )
        out = _roundtrip(h)
        assert out == h


class TestTrimPolicy:
    def test_roundtrip(self):
        for p in TrimPolicy:
            assert _roundtrip(p) is p


class TestAttnWindowDesc:
    def test_full_roundtrip(self):
        out = _roundtrip(AttnWindowDesc(num_chunks_in_sw=[-1]))
        assert out.num_chunks_in_sw == [-1]
        assert out.is_full_attention(0)

    def test_mixed_roundtrip(self):
        out = _roundtrip(AttnWindowDesc(num_chunks_in_sw=[-1, 4, 1]))
        assert out.num_chunks_in_sw == [-1, 4, 1]
        assert out.is_full_attention(0)
        assert not out.is_full_attention(1)
        assert out.num_object_groups == 3

    def test_empty_roundtrip(self):
        out = _roundtrip(AttnWindowDesc(num_chunks_in_sw=[]))
        assert out.num_chunks_in_sw == []
        assert out.num_object_groups == 0


class TestSet:
    def test_roundtrip(self):
        keys = {
            ObjectKey(chunk_hash=b"a", model_name="m", kv_rank=1),
            ObjectKey(chunk_hash=b"b", model_name="m", kv_rank=2),
        }
        out = _roundtrip(keys)
        assert out == keys
        assert isinstance(out, set)


class TestTorchTypes:
    def test_torch_size(self):
        s = torch.Size([1, 2, 3])
        out = _roundtrip(s)
        assert isinstance(out, torch.Size)
        assert tuple(out) == (1, 2, 3)

    def test_torch_dtype(self):
        for dt in [torch.float16, torch.bfloat16, torch.float32, torch.uint8]:
            assert _roundtrip(dt) is dt


class TestErrors:
    def test_unknown_type_raises(self):
        class Custom:
            pass

        with pytest.raises(TypeError, match="no codec"):
            codecs.encode_value(Custom())

    def test_unknown_tag_raises(self):
        with pytest.raises(ValueError, match="unknown tag"):
            codecs.decode_value({"__t__": "no-such-tag", "v": None})


class TestEncodeArgs:
    def test_args_dict(self):
        args = {
            "keys": [ObjectKey(chunk_hash=b"x", model_name="m", kv_rank=0)],
            "mode": "new",
            "extra_count": 0,
        }
        encoded = codecs.encode_args(args)
        # mode and extra_count pass through; keys is wrapped.
        assert encoded["mode"] == "new"
        assert encoded["extra_count"] == 0
        assert isinstance(encoded["keys"], list)
        decoded = codecs.decode_args(encoded)
        assert decoded == args
