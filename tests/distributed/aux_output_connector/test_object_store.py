# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
import re
import sys
import threading
import types
from types import SimpleNamespace

import numpy as np
import pytest
from safetensors.numpy import save

from tests.distributed.aux_output_connector.utils import FakeMooncake, fake_store
from vllm.config import AuxOutputConfig
from vllm.distributed.aux_output_connector import object_store
from vllm.distributed.aux_output_connector.object_store import (
    AuxSampleKeys,
    MooncakeAuxObjectStore,
    aux_prefix_pattern,
    aux_session_pattern,
    read_aux_rows,
    release_aux_keys,
)
from vllm.distributed.aux_output_connector.store import BlockObjectStoreError


def _rows(start, n, dtype=np.uint8):
    return (np.arange(start, start + n)[:, None, None] + np.zeros((1, 2, 3))).astype(
        dtype
    )


class TestReadAuxRows:
    def _store(self, blocks):
        _, store = fake_store()
        keys = [f"s/{i}" for i in range(len(blocks))]
        store.put_batch(keys, [save({"routed_experts": b}) for b in blocks])
        return store, keys

    @pytest.mark.parametrize(
        "num_blocks,dtype",
        [(3, np.uint8), (3, np.uint16), (0, np.uint8)],
        ids=["u8", "u16", "no-keys"],
    )
    def test_concatenates_in_key_order(self, num_blocks, dtype):
        store, keys = self._store([_rows(4 * i, 4, dtype) for i in range(num_blocks)])
        rows = read_aux_rows(store, keys, "routed_experts")
        assert len(rows) == 4 * num_blocks
        if num_blocks:
            assert rows.dtype == dtype
            np.testing.assert_array_equal(rows, _rows(0, 12))

    def test_waits_for_object_being_written(self):
        store, keys = self._store([_rows(0, 4)])
        late = save({"routed_experts": _rows(4, 4)})
        timer = threading.Timer(0.2, store.put_batch, (["s/late"], [late]))
        timer.start()
        rows = read_aux_rows(store, keys + ["s/late"], "routed_experts", timeout_s=10)
        timer.join()
        np.testing.assert_array_equal(rows, _rows(0, 8))

    @pytest.mark.parametrize(
        "write_delay_s", [61.0, 599.0], ids=["past-60s", "below-stall-deadline"]
    )
    def test_default_timeout_outlasts_a_write_stall(self, monkeypatch, write_delay_s):
        store, keys = self._store([_rows(0, 4)])
        late = save({"routed_experts": _rows(4, 4)})
        now = [0.0]

        def sleep(seconds):
            now[0] += seconds
            if now[0] >= write_delay_s and store.get("s/late") is None:
                store.put_batch(["s/late"], [late])

        monkeypatch.setattr(
            object_store, "time", SimpleNamespace(monotonic=lambda: now[0], sleep=sleep)
        )
        rows = read_aux_rows(store, keys + ["s/late"], "routed_experts")
        np.testing.assert_array_equal(rows, _rows(0, 8))
        assert now[0] >= write_delay_s

    def test_missing_object_times_out(self):
        store, keys = self._store([_rows(0, 4)])
        with pytest.raises(BlockObjectStoreError, match="'s/gone' is missing"):
            read_aux_rows(store, keys + ["s/gone"], "routed_experts", timeout_s=0.05)

    def test_missing_field_is_named(self):
        store, keys = self._store([_rows(0, 4), _rows(4, 4)])
        with pytest.raises(BlockObjectStoreError, match="'s/0' has no field 'hidden"):
            read_aux_rows(store, keys, "hidden_states")


@pytest.fixture
def mooncake(tmp_path, monkeypatch):
    """Config files of a KV, an object and an embedded master, with
    `MOONCAKE_CONFIG_PATH` naming the KV one; the masters clients connect to;
    the clients; the replicate configs created."""
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
    clients: list[FakeMooncake] = []

    def make_client():
        clients.append(FakeMooncake())
        return clients[-1]

    store_module = types.ModuleType("mooncake.store")
    store_module.MooncakeDistributedStore = make_client
    replicate_configs: list[SimpleNamespace] = []

    def make_replicate_config():
        replicate_configs.append(SimpleNamespace())
        return replicate_configs[-1]

    store_module.ReplicateConfig = make_replicate_config
    monkeypatch.setitem(sys.modules, "mooncake", types.ModuleType("mooncake"))
    monkeypatch.setitem(sys.modules, "mooncake.store", store_module)
    connected = []
    monkeypatch.setattr(
        "vllm.distributed.mooncake_store.setup_mooncake_store",
        lambda store, config, hostname: connected.append(config.master_server_address),
    )
    return paths, connected, clients, replicate_configs


@pytest.mark.parametrize("name", [None, "objects"], ids=["environment", "argument"])
def test_store_connects_to_the_master_of_its_config(mooncake, name):
    paths, connected, _, _ = mooncake
    MooncakeAuxObjectStore.create(None if name is None else paths[name])
    assert connected == [f"{name or 'kv'}-master:50051"]


@pytest.mark.parametrize(
    "hard_pin",
    [None, True, False],
    ids=["default", "memory-only", "ssd-offload-on-eviction"],
)
def test_store_writes_objects_pinned_as_configured(mooncake, hard_pin):
    _, _, clients, _ = mooncake
    store = MooncakeAuxObjectStore.create(
        **({} if hard_pin is None else {"hard_pin": hard_pin})
    )
    store.put_batch(["job/s/0"], [b"rows"])
    assert [config.with_hard_pin for config in clients[0].configs] == [bool(hard_pin)]
    assert store.get("job/s/0") == b"rows"


def test_store_refuses_an_embedded_client(mooncake):
    paths, connected, _, _ = mooncake
    with pytest.raises(ValueError, match="standalone-store"):
        MooncakeAuxObjectStore.create(paths["embedded"])
    assert connected == []


class _Capturer:
    shape_per_token = (2, 2)
    output_dtype_name = "int32"

    def __init__(self, **kwargs):
        pass

    def close(self):
        pass


@pytest.mark.parametrize(
    "first_tp_rank,last_stage,masters",
    [(True, True, ["objects-master:50051"]), (False, True, []), (True, False, [])],
    ids=["output-rank", "other-tp-rank", "earlier-stage"],
)
@pytest.mark.parametrize(
    "hard_pin", [None, True, False], ids=["default", "hard-pin", "no-pin"]
)
@pytest.mark.parametrize(
    "execute_timeout_s,put_stall_timeout_s",
    [("20", 7.0), ("10", 5.0)],
    ids=["configured", "half-the-step-timeout"],
)
def test_output_rank_writes_objects_to_the_configured_master(
    mooncake,
    monkeypatch,
    first_tp_rank,
    last_stage,
    masters,
    hard_pin,
    execute_timeout_s,
    put_stall_timeout_s,
):
    monkeypatch.setenv("VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS", execute_timeout_s)
    import vllm.distributed.aux_output_connector.connector as aux_connector
    import vllm.distributed.aux_output_connector.worker as aux_worker

    paths, connected, _, replicate_configs = mooncake
    writers = []
    writer_class = aux_worker.SessionObjectWriter

    def recording_writer(*args, **kwargs):
        writers.append(kwargs)
        return writer_class(*args, **kwargs)

    monkeypatch.setattr(aux_worker, "SessionObjectWriter", recording_writer)
    group = SimpleNamespace(
        is_first_rank=first_tp_rank,
        is_last_rank=last_stage,
        world_size=1,
        rank_in_group=0,
    )
    monkeypatch.setattr(aux_worker, "get_tp_group", lambda: group)
    monkeypatch.setattr(aux_worker, "get_pp_group", lambda: group)
    monkeypatch.setattr(aux_worker, "RoutedExpertsCapturer", _Capturer)
    monkeypatch.setattr(
        aux_worker, "bind_routed_experts_capturer", lambda model, capturer: None
    )
    monkeypatch.setattr(
        aux_connector, "resolve_kv_cache_block_sizes", lambda *_: (16, 16)
    )
    config = SimpleNamespace(
        aux_output_config=AuxOutputConfig(
            enable_return_routed_experts=True,
            object_key_prefix="job",
            mooncake_config_path=paths["objects"],
            **({} if hard_pin is None else {"object_hard_pin": hard_pin}),
            object_put_stall_timeout_s=7.0,
        ),
        kv_transfer_config=None,
        scheduler_config=SimpleNamespace(max_num_seqs=2, max_num_batched_tokens=16),
        max_concurrent_batches=2,
    )
    connector = aux_worker.AuxOutputWorkerConnector(vllm_config=config, model=None)
    connector.initialize_output_storage(SimpleNamespace(num_blocks=4))
    connector.close()
    assert connected == masters
    pins = [config.with_hard_pin for config in replicate_configs]
    assert pins == [bool(hard_pin)] * len(masters)
    assert [kwargs["put_stall_timeout_s"] for kwargs in writers] == [
        put_stall_timeout_s
    ] * len(masters)


_ECMASCRIPT_SYNTAX_CHARACTERS = set("^$\\.*+?()[]{}|")


@pytest.mark.parametrize(
    "prefix,session",
    [
        ("job", "s"),
        ("run-1", "turn-2"),
        ("a.b", "x+y*z?"),
        ("(p)", "[t]{u}"),
        ("p|q", "a^b$c"),
        ("p#1", "s p&a~ce"),
        ("p\\q", "s\\t"),
    ],
    ids=[
        "plain",
        "hyphens",
        "quantifiers",
        "groups",
        "anchors",
        "punctuation",
        "backslash",
    ],
)
def test_session_patterns_match_their_keys_in_ecmascript_syntax(prefix, session):
    patterns = [aux_prefix_pattern(prefix), aux_session_pattern(prefix, session)]
    for pattern in patterns:
        assert set(re.findall(r"\\(.)", pattern)) <= _ECMASCRIPT_SYNTAX_CHARACTERS
    keys = [
        f"{prefix}/{session}/ab",
        f"{prefix}/{session}/tail/cd",
        f"{prefix}/{session}x/ab",
        f"{prefix}x/{session}/ab",
        f"x{prefix}/{session}/ab",
    ]
    matched = [[key for key in keys if re.search(pattern, key)] for pattern in patterns]
    assert matched == [keys[:3], keys[:2]]


class LateMooncake(FakeMooncake):
    def __init__(self, late, codes=()):
        super().__init__()
        self.late = late
        self.codes = list(codes)

    def batch_remove(self, keys, force):
        code = self.codes.pop(0) if self.codes else 0
        codes = [code] * len(keys) if code else super().batch_remove(keys, force)
        if self.late:
            self.objects[self.late.pop(0)] = b"x"
        return codes


class TestReleaseAuxKeys:
    @pytest.mark.parametrize(
        "late,codes,left",
        [
            (["b"], (), []),
            ([], (), ["b"]),
            (["b"], (-703,), []),
            ([], (-600,), ["a", "b"]),
        ],
        ids=["written-after-the-first-try", "never-written", "being-written", "error"],
    )
    def test_retries_keys_not_written_yet(self, late, codes, left):
        client = LateMooncake(late, codes)
        store = MooncakeAuxObjectStore(client, None)
        client.objects["a"] = b"x"
        assert release_aux_keys(store, ["a", "b"], timeout_s=0.3) == left
        assert not client.objects or left

    def test_sample_owns_only_keys_it_did_not_hold(self):
        parent = AuxSampleKeys()
        parent.add(["p0", "p1", "pt"])
        child = AuxSampleKeys(parent)
        child.add(["p0", "c1", "ct"])
        child.add(["p0", "c1", "c2", "ct2"])
        assert parent.owned == ["p0", "p1", "pt"]
        assert child.owned == ["c1", "ct", "c2", "ct2"]
