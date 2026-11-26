# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
import sys
import threading
import time
import types
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import regex as re
from safetensors.numpy import save

from tests.distributed.aux_output_connector.utils import (
    DictObjectStore,
    FakeMooncake,
    fake_store,
    wait_for_writer,
)
from vllm.distributed.aux_output_connector import object_store
from vllm.distributed.aux_output_connector.object_store import (
    AuxObjectWriter,
    MooncakeAuxObjectStore,
    aux_namespace_prefix,
    aux_prefix,
    create_aux_object_store,
    read_aux_rows,
)
from vllm.distributed.aux_output_connector.store import BlockObjectStoreError

pytestmark = pytest.mark.cpu_test


def _rows(start, n, dtype=np.uint8):
    return (np.arange(start, start + n)[:, None, None] + np.zeros((1, 2, 3))).astype(
        dtype
    )


def _stored(blocks):
    _, store = fake_store()
    keys = [f"job/g/{i}" for i in range(len(blocks))]
    store.put_batch(keys, [save({"routed_experts": block}) for block in blocks])
    return store, keys


@pytest.mark.parametrize(
    "num_objects,dtype",
    [(3, np.uint8), (3, np.uint16), (0, np.uint8)],
    ids=["uint8", "uint16", "no-keys"],
)
def test_read_aux_rows_concatenates_objects_in_key_order(num_objects, dtype):
    store, keys = _stored([_rows(4 * i, 4, dtype) for i in range(num_objects)])
    rows = read_aux_rows(store, keys[::-1], "routed_experts")
    assert len(rows) == 4 * num_objects
    if num_objects:
        assert rows.dtype == dtype
        np.testing.assert_array_equal(
            rows, np.concatenate([_rows(8, 4), _rows(4, 4), _rows(0, 4)])
        )


def test_read_aux_rows_waits_for_an_object_through_a_write_stall(monkeypatch):
    store, keys = _stored([_rows(0, 4)])
    late = save({"routed_experts": _rows(4, 4)})
    now = [0.0]

    def sleep(seconds):
        now[0] += seconds
        if now[0] >= 599.0 and store.get("job/g/late") is None:
            store.put_batch(["job/g/late"], [late])

    monkeypatch.setattr(
        object_store, "time", SimpleNamespace(monotonic=lambda: now[0], sleep=sleep)
    )
    rows = read_aux_rows(store, [*keys, "job/g/late"], "routed_experts")
    np.testing.assert_array_equal(rows, _rows(0, 8))
    assert now[0] >= 599.0


def test_read_aux_rows_raises_for_a_missing_object():
    store, keys = _stored([_rows(0, 4)])
    with pytest.raises(BlockObjectStoreError, match="'job/g/gone' is missing"):
        read_aux_rows(store, [*keys, "job/g/gone"], "routed_experts", timeout_s=0.05)


@pytest.fixture
def mooncake(tmp_path, monkeypatch):
    paths = {}
    for name, mode, segment_size in (
        ("kv", "standalone-store", 0),
        ("objects", "standalone-store", 0),
        ("embedded", "embedded", 1 << 20),
    ):
        path = tmp_path / f"{name}.json"
        path.write_text(
            json.dumps(
                {
                    "master_server_address": f"{name}-master:50051",
                    "mode": mode,
                    "global_segment_size": segment_size,
                }
            )
        )
        paths[name] = str(path)
    monkeypatch.setenv("MOONCAKE_CONFIG_PATH", paths["kv"])
    clients = []

    def make_client():
        clients.append(FakeMooncake())
        return clients[-1]

    store_module = types.ModuleType("mooncake.store")
    store_module.MooncakeDistributedStore = make_client  # type: ignore[attr-defined]
    store_module.ReplicateConfig = SimpleNamespace  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "mooncake", types.ModuleType("mooncake"))
    monkeypatch.setitem(sys.modules, "mooncake.store", store_module)
    connected = []
    monkeypatch.setattr(
        "vllm.distributed.mooncake_store.setup_mooncake_store",
        lambda store, config, hostname: connected.append(config.master_server_address),
    )
    return paths, connected, clients


@pytest.mark.parametrize("name", [None, "objects"], ids=["environment", "argument"])
def test_store_connects_to_the_master_of_its_config(mooncake, name):
    paths, connected, _ = mooncake
    MooncakeAuxObjectStore.create(None if name is None else paths[name])
    assert connected == [f"{name or 'kv'}-master:50051"]


@pytest.mark.parametrize("hard_pin", [None, True], ids=["default", "hard-pin"])
def test_store_writes_objects_pinned_as_configured(mooncake, hard_pin):
    _, _, clients = mooncake
    store = MooncakeAuxObjectStore.create(
        **({} if hard_pin is None else {"hard_pin": hard_pin})
    )
    store.put_batch(["job/g/0"], [b"rows"])
    assert [config.with_hard_pin for config in clients[0].configs] == [bool(hard_pin)]
    assert store.get("job/g/0") == b"rows"


def test_store_refuses_a_mode_other_than_standalone_store(mooncake):
    paths, connected, clients = mooncake
    with pytest.raises(ValueError, match="standalone-store"):
        MooncakeAuxObjectStore.create(paths["embedded"])
    assert connected == [] and clients == []


def test_store_is_created_by_name_with_its_extra_config(mooncake):
    paths, connected, _ = mooncake
    store = create_aux_object_store("mooncake", {"config_path": paths["objects"]})
    assert isinstance(store, MooncakeAuxObjectStore)
    assert connected == ["objects-master:50051"]
    other = create_aux_object_store(
        f"{DictObjectStore.__module__}.DictObjectStore", {"bucket": "b"}
    )
    assert isinstance(other, DictObjectStore) and other.extra_config == {"bucket": "b"}
    for name in ("Mooncake", ".Store", "pkg."):
        with pytest.raises(ValueError, match=f"object_store {name!r}"):
            create_aux_object_store(name, {})


@pytest.mark.parametrize(
    "code,error",
    [(-200, BlockObjectStoreError), (-900, RuntimeError), (-600, RuntimeError)],
    ids=["no-room", "rpc-failed", "invalid-params"],
)
def test_store_retries_only_puts_mooncake_has_no_room_for(code, error):
    client, store = fake_store(code)
    with pytest.raises(error, match=f"from 'g/0': {code}") as exc_info:
        store.put_batch(["g/0"], [b"rows"])
    if error is RuntimeError:
        assert not isinstance(exc_info.value, BlockObjectStoreError)
        # The writer drops the batch at once instead of retrying it.
        writer = AuxObjectWriter(store, timeout_s=60)
        writer.put(_objects("g/0"))
        wait_for_writer(writer)
        assert client.writes == [["g/0"]] * 2 and not client.objects
        writer.close()


def test_store_raises_when_mooncake_fails_to_remove_a_prefix():
    client, store = fake_store()
    client.remove_by_regex = lambda pattern, force: -900  # type: ignore[method-assign]
    with pytest.raises(RuntimeError, match="under 'job/g/': -900"):
        store.remove_prefix(aux_namespace_prefix("job", "g"))


def test_objects_round_trip_through_a_store_other_than_mooncake():
    store = DictObjectStore()
    writer = AuxObjectWriter(store, timeout_s=5)
    rows = _rows(0, 4)
    writer.put([("job/g/0", {"routed_experts": rows}), ("job/h/0", {"x": rows})])
    wait_for_writer(writer)
    np.testing.assert_array_equal(
        read_aux_rows(store, ["job/g/0"], "routed_experts"), rows
    )
    store.remove_prefix(aux_namespace_prefix("job", "g"))
    assert list(store.objects) == ["job/h/0"]
    writer.close()


_ECMASCRIPT_SYNTAX_CHARACTERS = set("^$\\.*+?()[]{}|")


def _released(prefix, keys, kind="mooncake"):
    if kind == "dict":
        store = client = DictObjectStore()
    else:
        client, store = fake_store()
    store.put_batch(keys, [b"rows"] * len(keys))
    store.remove_prefix(prefix)
    for pattern in getattr(client, "patterns", []):
        assert set(re.findall(r"\\(.)", pattern)) <= _ECMASCRIPT_SYNTAX_CHARACTERS
    return [key for key in keys if key not in client.objects]


@pytest.mark.parametrize(
    "prefix,namespace",
    [
        ("job", "g"),
        ("run-1", "g_1/s-0"),
        ("p+q*r?", "g"),
        ("(p)[q]{r}", "g"),
        ("p|q^r$", "g"),
        ("p#1 &~", "g"),
        ("p\\q", "g"),
    ],
    ids=[
        "plain",
        "nested",
        "quantifiers",
        "groups",
        "anchors",
        "punctuation",
        "backslash",
    ],
)
@pytest.mark.parametrize("kind", ["mooncake", "dict"])
def test_release_prefixes_remove_exactly_their_keys(prefix, namespace, kind):
    prefixes = [aux_prefix(prefix), aux_namespace_prefix(prefix, namespace)]
    keys = [
        f"{prefix}/{namespace}/ab",
        f"{prefix}/{namespace}/tail-cd",
        f"{prefix}/{namespace}x/ab",
        f"{prefix}x/{namespace}/ab",
        f"x{prefix}/{namespace}/ab",
    ]
    assert [_released(p, keys, kind) for p in prefixes] == [keys[:3], keys[:2]]


def test_release_prefixes_match_dots_literally():
    keys = ["a.b/g.0/ab", "axb/g.0/ab", "a.b/gx0/ab"]
    assert _released(aux_prefix("a.b"), keys) == [keys[0], keys[2]]
    assert _released(aux_namespace_prefix("a.b", "g.0"), keys) == keys[:1]


@pytest.mark.parametrize(
    "namespace,released",
    [("g", [0, 1, 2, 3, 4]), ("g/0", [1, 2, 3])],
    ids=["parent", "nested"],
)
def test_namespace_prefix_covers_the_namespaces_nested_in_it(namespace, released):
    keys = ["p/g/ab", "p/g/0/ab", "p/g/0/tail-cd", "p/g/0/1/ab", "p/g/01/ab", "p/g0/ab"]
    assert _released(aux_namespace_prefix("p", namespace), keys) == [
        keys[i] for i in released
    ]


def _objects(*keys):
    return [(key, {"routed_experts": np.zeros((1, 1), np.uint8)}) for key in keys]


def _writer(put, timeout_s=10.0):
    client, store = fake_store()
    put_batch = client.put_batch
    client.put_batch = lambda keys, values, config: (
        put(keys) or put_batch(keys, values, config)
    )
    return client, store, AuxObjectWriter(store, timeout_s=timeout_s)


def test_writer_writes_a_batch_a_store_refused_once():
    refusals: list[int] = []
    release = threading.Event()

    def put(keys):
        if refusals:
            return None
        refusals.append(1)
        release.wait(10)
        return -200

    client, _, writer = _writer(put)
    writer.put(_objects("g/0"))
    writer.put(_objects("h/0"))
    # Queuing never waits for the store.
    assert not release.is_set() and not client.objects
    release.set()
    wait_for_writer(writer)
    assert list(client.objects) == ["g/0", "h/0"] and refusals == [1]
    writer.close()
    writer._thread.join(10)
    assert client.closed


@pytest.mark.parametrize(
    "failure,reason",
    [("refused", BlockObjectStoreError), ("raises", RuntimeError)],
    ids=["refused-to-deadline", "put-raises"],
)
def test_writer_drops_only_the_batch_it_cannot_write(monkeypatch, failure, reason):
    logger = Mock()
    monkeypatch.setattr(object_store, "logger", logger)
    refusals = []
    queued = threading.Event()

    def put(keys):
        if keys != ["g/0"]:
            return None
        if failure == "raises":
            queued.wait(5)
            raise RuntimeError("put failed")
        refusals.append(time.monotonic())
        return -200

    client, store, writer = _writer(put, 2.0)
    start = time.monotonic()
    writer.put(_objects("g/0"))
    if failure == "refused":
        time.sleep(1.0)
    # Queued behind the failing batch, with time left to be written.
    writer.put(_objects("g/1", "h/0"))
    queued.set()
    wait_for_writer(writer)
    # Its keys are missing; later objects of its namespace are written.
    assert list(client.objects) == ["g/1", "h/0"]
    assert client.writes[-1] == ["g/1", "h/0"]
    assert len(read_aux_rows(store, ["g/1"], "routed_experts", timeout_s=0)) == 1
    ((_, _, key, logged), _) = logger.error.call_args
    assert key == "g/0" and isinstance(logged, reason)
    assert logger.error.call_count == 1
    if refusals:
        # Refusals are retried until the deadline, the last one less than a
        # retry interval before it.
        assert refusals[-1] - start >= 1.7
    # A batch queued after the failure is written as well.
    writer.put(_objects("g/2", "h/1"))
    wait_for_writer(writer)
    assert list(client.objects) == ["g/1", "h/0", "g/2", "h/1"]
    assert logger.error.call_count == 1
    writer.close()
    writer._thread.join(10)


def test_writer_deadline_counts_from_queueing():
    # A batch queued behind a slow one keeps the deadline it was queued with.
    refusals: list[float] = []

    def put(keys):
        if keys == ["g/0"]:
            time.sleep(1.5)
            return None
        refusals.append(time.monotonic())
        return -200

    client, _, writer = _writer(put, 2.0)
    writer.put(_objects("g/0"))
    time.sleep(0.3)
    queued = time.monotonic()
    writer.put(_objects("h/0"))
    wait_for_writer(writer)
    assert list(client.objects) == ["g/0"]
    # Counted from the start of its write, the last refusal would come more
    # than 3 s after queueing.
    assert 1.5 <= refusals[-1] - queued < 2.5
    writer.close()


def test_writer_close_writes_the_queued_batches():
    client, _, writer = _writer(lambda keys: time.sleep(0.1))
    writer.put(_objects("g/0", "h/0"))
    writer.put(_objects("k/0"))
    writer.close()
    assert list(client.objects) == ["g/0", "h/0", "k/0"] and client.closed
    # One store call per batch.
    assert client.writes == [["g/0", "h/0"], ["k/0"]]


def test_writer_drops_the_batches_queued_behind_a_put_that_does_not_return(
    monkeypatch,
):
    logger = Mock()
    monkeypatch.setattr(object_store, "logger", logger)
    release = threading.Event()
    calls = []

    def put(keys):
        calls.append(keys)
        return release.wait(30) and None

    client, _, writer = _writer(put, 1.0)
    writer.put(_objects("g/0"))
    writer.put(_objects("h/0"))
    writer.put(_objects("h/1"))
    time.sleep(1.2)
    # Past their deadline the batches are dropped, also those queued behind
    # the put that does not return, and the writer no longer holds their
    # objects. While the put has not returned, queuing drops without waiting.
    start = time.monotonic()
    writer.put(_objects("k/0"))
    assert time.monotonic() - start < 0.5
    assert [batch.objects for batch in writer._pending] == [None] * 3
    assert logger.error.call_count == 4
    # Every deadline has passed: closing does not wait for the put.
    start = time.monotonic()
    writer.close()
    assert time.monotonic() - start < 0.5
    # The writer thread closes the store once the put returns, and writes
    # none of the dropped batches.
    assert not client.closed
    release.set()
    writer._thread.join(10)
    assert client.closed
    assert calls == [["g/0"]]


@pytest.mark.parametrize(
    "stuck,written",
    [(False, ["g/0", "h/0", "k/0"]), (True, ["g/0"])],
    ids=["write-ends", "store-stuck"],
)
def test_writer_blocks_a_full_queue_until_a_write_ends_or_misses_its_deadline(
    monkeypatch, stuck, written
):
    monkeypatch.setattr(object_store, "_MAX_QUEUED_BATCHES", 1)
    release = threading.Event()
    client, _, writer = _writer(
        lambda keys: (release.wait(30) and None) if keys == ["g/0"] else None, 2.0
    )
    writer.put(_objects("g/0"))
    if not stuck:
        threading.Timer(0.2, release.set).start()
    else:
        time.sleep(0.5)
    start = time.monotonic()
    writer.put(_objects("h/0"))
    waited = time.monotonic() - start
    if stuck:
        # The wait ends at the deadline of the head, not one timeout later.
        assert 1.0 <= waited < 2.0
    else:
        assert release.is_set() and waited < 1.5
    # Behind a stuck store the next put does not wait another deadline.
    start = time.monotonic()
    writer.put(_objects("k/0"))
    assert time.monotonic() - start < (0.5 if stuck else 1.5)
    # Behind a stuck store only the put that did not return lands, late.
    release.set()
    wait_for_writer(writer)
    writer.close()
    writer._thread.join(10)
    assert list(client.objects) == written


def test_writer_does_not_block_while_the_store_refuses_every_write(monkeypatch):
    monkeypatch.setattr(object_store, "_MAX_QUEUED_BATCHES", 2)
    accept = threading.Event()
    client, _, writer = _writer(lambda keys: None if accept.is_set() else -200, 1.0)
    writer.put(_objects("g/0"))
    writer.put(_objects("g/1"))
    # The first full queue waits until the head misses its deadline.
    start = time.monotonic()
    writer.put(_objects("g/2"))
    assert time.monotonic() - start >= 0.5
    # From then on queuing never waits, over several deadlines, while one
    # batch at a time keeps trying the store.
    longest, start = 0.0, time.monotonic()
    while time.monotonic() - start < 3.5:
        before = time.monotonic()
        writer.put(_objects(f"h/{before}"))
        longest = max(longest, time.monotonic() - before)
        assert len(writer._pending) <= 1
        time.sleep(0.01)
    assert longest < 0.5
    # A put that lands ends it: later batches are queued and written again.
    accept.set()
    for key in ("k/0", "k/1", "k/2"):
        wait_for_writer(writer)
        writer.put(_objects(key))
    writer.put(_objects("k/3"))
    wait_for_writer(writer)
    assert {"k/1", "k/2", "k/3"} <= set(client.objects)
    writer.close()
