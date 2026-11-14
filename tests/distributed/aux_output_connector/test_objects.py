# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import threading

import numpy as np
import pytest
from safetensors.numpy import save

from vllm.distributed.aux_output_connector.object_store import (
    AuxObjectStoreError,
    LocalAuxObjectStore,
)
from vllm.distributed.aux_output_connector.objects import (
    decode_aux_object,
    encode_aux_object,
    read_aux_rows,
)


def _rows(start, n, dtype=np.uint8):
    return (np.arange(start, start + n)[:, None, None] + np.zeros((1, 2, 3))).astype(
        dtype
    )


class TestAuxObjectFormat:
    @pytest.mark.parametrize(
        "dtype", [np.uint8, np.uint16, np.int32], ids=["u8", "u16", "i32"]
    )
    def test_round_trip(self, dtype):
        fields = {"routed_experts": _rows(0, 4, dtype), "extra": np.ones((4, 5))}
        decoded = decode_aux_object(encode_aux_object(fields))
        assert decoded.keys() == fields.keys()
        for name, rows in fields.items():
            assert decoded[name].dtype == rows.dtype
            np.testing.assert_array_equal(decoded[name], rows)

    def test_fields_must_cover_the_same_rows(self):
        with pytest.raises(ValueError, match="row counts"):
            encode_aux_object({"a": np.zeros((4, 1)), "b": np.zeros((3, 1))})

    def test_rejects_foreign_safetensors(self):
        with pytest.raises(AuxObjectStoreError, match="format"):
            decode_aux_object(save({"routed_experts": _rows(0, 4)}))


class TestReadAuxRows:
    def _store(self, blocks):
        store = LocalAuxObjectStore()
        keys = [f"s/{i}" for i in range(len(blocks))]
        store.put_batch(
            keys, [encode_aux_object({"routed_experts": b}) for b in blocks]
        )
        return store, keys

    @pytest.mark.parametrize(
        "num_blocks,tail_len",
        [(3, 5), (3, 0), (0, 5)],
        ids=["blocks-and-tail", "blocks-only", "tail-only"],
    )
    def test_concatenates_in_key_order(self, num_blocks, tail_len):
        store, keys = self._store([_rows(4 * i, 4) for i in range(num_blocks)])
        tail = _rows(4 * num_blocks, tail_len) if tail_len else None
        rows = read_aux_rows(store, keys, "routed_experts", tail)
        np.testing.assert_array_equal(rows, _rows(0, 4 * num_blocks + tail_len))

    def test_waits_for_object_being_written(self):
        store, keys = self._store([_rows(0, 4)])
        late = encode_aux_object({"routed_experts": _rows(4, 4)})
        timer = threading.Timer(0.2, store.put_batch, (["s/late"], [late]))
        timer.start()
        rows = read_aux_rows(
            store,
            keys + ["s/late"],
            "routed_experts",
            timeout_s=10,
            poll_interval_s=0.01,
        )
        timer.join()
        np.testing.assert_array_equal(rows, _rows(0, 8))

    def test_missing_object_times_out(self):
        store, keys = self._store([_rows(0, 4)])
        with pytest.raises(AuxObjectStoreError, match="1 of 2 .*first 's/gone'"):
            read_aux_rows(store, keys + ["s/gone"], "routed_experts", timeout_s=0.05)

    @pytest.mark.parametrize(
        "block,tail,match",
        [
            (_rows(0, 4), _rows(4, 2, np.uint16), "dtypes or row shapes"),
            (_rows(0, 4), np.zeros((2, 2, 4), np.uint8), "dtypes or row shapes"),
        ],
        ids=["dtype", "row-shape"],
    )
    def test_rejects_mismatched_tail(self, block, tail, match):
        store, keys = self._store([block])
        with pytest.raises(AuxObjectStoreError, match=match):
            read_aux_rows(store, keys, "routed_experts", tail)

    def test_rejects_missing_field(self):
        store, keys = self._store([_rows(0, 4)])
        with pytest.raises(AuxObjectStoreError, match="no field 'indexer'"):
            read_aux_rows(store, keys, "indexer")
