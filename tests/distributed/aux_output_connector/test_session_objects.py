# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import numpy as np
import pytest

from vllm.distributed.aux_output_connector.object_store import (
    AuxObjectStoreError,
    LocalAuxObjectStore,
)
from vllm.distributed.aux_output_connector.objects import read_aux_rows
from vllm.distributed.aux_output_connector.session_objects import (
    SessionObjectWriter,
)

HASHES = [f"h{i}" for i in range(8)]
IDS = {0: "a", 1: "b", 2: "c", 3: "d"}


def _block(start, size=4, offset=0):
    rows = np.arange(start, start + size, dtype=np.uint8) + offset
    return start, rows.reshape(-1, 1, 1)


class CountingStore(LocalAuxObjectStore):
    def __init__(self, fail=False):
        super().__init__()
        self.fail = fail
        self.writes = []

    def put_batch(self, keys, values):
        self.writes.append(list(keys))
        if self.fail:
            self.fail = False
            raise AuxObjectStoreError("pool is full")
        super().put_batch(keys, values)


def _writer(store):
    return SessionObjectWriter(
        store,
        key_prefix="job",
        object_size=8,
        hash_block_size=4,
        max_pending_batches=4,
    )


class TestSessionObjectWriter:
    def test_writes_complete_objects(self):
        store = CountingStore()
        writer = _writer(store)
        writer.add_blocks("r", "s", HASHES, [_block(0), _block(4), _block(8)], IDS)
        writer.add_blocks("r", "s", HASHES, [_block(12)], IDS)
        writer.flush()
        assert store.writes == [["job/s/h1@a"], ["job/s/h3@b"]]
        rows = read_aux_rows(store, ["job/s/h1@a", "job/s/h3@b"], "routed_experts")
        assert rows[:, 0, 0].tolist() == list(range(16))

    def test_discarded_and_unnamed_objects_are_not_written(self):
        store = CountingStore()
        writer = _writer(store)
        writer.add_blocks("r", "s", HASHES, [_block(4)], IDS)
        writer.discard("r")
        writer.add_blocks("r", "s", HASHES, [_block(0)], IDS)
        writer.add_blocks("q", "s", HASHES, [_block(0), _block(4)], {1: "b"})
        writer.flush()
        assert store.writes == []

    @pytest.mark.parametrize(
        "second_session,expected_writes",
        [("s", [["job/s/h1@a"]]), ("t", [["job/s/h1@a"], ["job/t/h1@a"]])],
        ids=["same-session", "other-session"],
    )
    def test_each_object_is_written_once_per_session(
        self, second_session, expected_writes
    ):
        store = CountingStore()
        writer = _writer(store)
        writer.add_blocks("a", "s", HASHES, [_block(0), _block(4)], IDS)
        writer.add_blocks("b", second_session, HASHES, [_block(0), _block(4)], IDS)
        writer.flush()
        assert store.writes == expected_writes

    def test_recomputed_object_is_read_back(self):
        store = CountingStore()
        writer = _writer(store)
        writer.add_blocks("a", "s", HASHES, [_block(0), _block(4)], {0: "x"})
        writer.add_blocks(
            "b", "s", HASHES, [_block(0, offset=100), _block(4, offset=100)], {0: "y"}
        )
        writer.flush()
        rows = read_aux_rows(store, ["job/s/h1@y"], "routed_experts")
        assert rows[:, 0, 0].tolist() == list(range(100, 108))

    def test_failed_write_does_not_stop_later_writes(self):
        store = CountingStore(fail=True)
        writer = _writer(store)
        writer.add_blocks("r", "s", HASHES, [_block(0), _block(4)], IDS)
        writer.flush()
        writer.add_blocks("r", "s", HASHES, [_block(8), _block(12)], IDS)
        writer.flush()
        assert store.writes == [["job/s/h1@a"], ["job/s/h3@b"]]
        assert store.exists(["job/s/h1@a", "job/s/h3@b"]) == [False, True]

    def test_forgotten_session_writes_again(self):
        store = CountingStore()
        writer = _writer(store)
        writer.add_blocks("a", "s", HASHES, [_block(0), _block(4)], IDS)
        writer.forget_session("s")
        writer.add_blocks("b", "s", HASHES, [_block(0), _block(4)], IDS)
        writer.flush()
        assert store.writes == [["job/s/h1@a"], ["job/s/h1@a"]]
