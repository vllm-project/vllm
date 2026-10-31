# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MoRIIO WRITE from a pipeline-parallel producer (e.g. P TP1 x PP4 -> D TP4)."""

import itertools
import threading
from types import SimpleNamespace
from unittest.mock import patch

import msgspec
import pytest

from vllm.distributed.kv_transfer.kv_connector.v1.moriio import (
    moriio_common,
    moriio_engine,
)
from vllm.distributed.kv_transfer.kv_connector.v1.moriio.moriio_common import (
    ROLE,
    MoRIIOAgentMetadata,
    MoRIIOConfig,
    MoRIIOError,
    MoRIIOMode,
    RemoteAllocInfo,
    get_peer_notify_ports,
    get_port_offset,
)
from vllm.distributed.kv_transfer.kv_connector.v1.moriio.moriio_connector import (
    MoRIIOConnectorScheduler,
    MoRIIOConnectorWorker,
)
from vllm.distributed.kv_transfer.kv_connector.v1.moriio.moriio_engine import (
    MoRIIOWrapper,
)

from .utils import make_moriio_writer

pytestmark = pytest.mark.cpu_test

NOTIFY_PORT = 61005
HANDSHAKE_PORT = 6301


def _moriio_config(pp_rank, pp_size, tp_rank, tp_size, dp_rank=0, dp_size=1):
    vllm_config = SimpleNamespace(
        kv_transfer_config=SimpleNamespace(
            kv_connector_extra_config={
                "host_ip": "127.0.0.1",
                "http_port": 8000,
                "handshake_port": HANDSHAKE_PORT,
                "notify_port": NOTIFY_PORT,
            }
        ),
        parallel_config=SimpleNamespace(
            data_parallel_rank=dp_rank,
            data_parallel_size=dp_size,
            data_parallel_size_local=dp_size,
            pipeline_parallel_size=pp_size,
        ),
    )
    with patch.multiple(
        moriio_common,
        get_tensor_model_parallel_rank=lambda: tp_rank,
        get_tensor_model_parallel_world_size=lambda: tp_size,
        get_pipeline_model_parallel_rank=lambda: pp_rank,
    ):
        return MoRIIOConfig.from_vllm_config(vllm_config)


@pytest.mark.parametrize(
    ("pp_size", "tp_size", "dp_size"),
    [(1, 4, 1), (4, 1, 1), (2, 2, 1), (2, 1, 2)],
)
def test_every_stage_worker_binds_its_own_ports(pp_size, tp_size, dp_size):
    configs = [
        _moriio_config(pp, pp_size, tp, tp_size, dp, dp_size)
        for pp, dp, tp in itertools.product(
            range(pp_size), range(dp_size), range(tp_size)
        )
    ]

    notify_ports = [config.notify_port for config in configs]
    assert len(set(notify_ports)) == len(configs)
    for config in configs:
        assert config.notify_port == NOTIFY_PORT + config.port_offset
        if config.pp_rank == 0:
            # Stage 0 keeps the ports of a pipeline-free instance.
            assert config.port_offset == get_port_offset(config.dp_rank, config.tp_rank)

    for dp_rank in range(dp_size):
        assert sorted(
            get_peer_notify_ports(NOTIFY_PORT, dp_rank, tp_size, pp_size, dp_size)
        ) == sorted(
            config.notify_port for config in configs if config.dp_rank == dp_rank
        )


def _decode_scheduler() -> MoRIIOConnectorScheduler:
    scheduler = MoRIIOConnectorScheduler.__new__(MoRIIOConnectorScheduler)
    scheduler.mode = MoRIIOMode.WRITE
    scheduler.is_producer = False
    scheduler.tp_size = 4
    scheduler._has_mamba = False
    scheduler._is_hma_required = False
    scheduler._is_kv_master = True
    scheduler._global_dp_rank = 0
    scheduler.transfer_id_to_request_id = {}
    scheduler.request_id_to_transfer_id = {}
    scheduler.notified = []
    scheduler.released = []
    scheduler.send_notify_block = (  # type: ignore[method-assign]
        lambda **kwargs: scheduler.notified.append(kwargs)
    )
    scheduler._send_transfer_release = (  # type: ignore[method-assign]
        lambda transfer_id, host, port: scheduler.released.append(port)
    )
    return scheduler


def _prefill_stage_ports(pp_size: int, tp_size: int) -> list[int]:
    return [
        _moriio_config(pp, pp_size, tp, tp_size).notify_port
        for pp, tp in itertools.product(range(pp_size), range(tp_size))
    ]


@pytest.mark.parametrize(("pp_size", "tp_size"), [(4, 1), (2, 2)])
def test_decode_sends_its_allocation_to_every_prefill_stage(pp_size, tp_size):
    scheduler = _decode_scheduler()
    request = SimpleNamespace(
        request_id="req",
        kv_transfer_params={
            "do_remote_prefill": True,
            "transfer_id": "tA",
            "remote_host": "10.0.0.1",
            "remote_notify_port": NOTIFY_PORT,
            "remote_tp_size": tp_size,
            "remote_pp_size": pp_size,
            "remote_dp_size": 1,
        },
    )
    blocks = SimpleNamespace(get_block_ids=lambda: ([5, 6, 7],))

    scheduler.update_state_after_alloc(request, blocks, num_external_tokens=40)

    assert sorted(n["port"] for n in scheduler.notified) == sorted(
        _prefill_stage_ports(pp_size, tp_size)
    )
    assert all(n["block_notify_list"] == [5, 6, 7] for n in scheduler.notified)
    assert all(n["transfer_id"] == "tA" for n in scheduler.notified)


@pytest.mark.parametrize(
    ("params", "pp_size", "tp_size"),
    [
        ({"remote_tp_size": 1, "remote_pp_size": 4}, 4, 1),
        ({"remote_tp_size": 2}, 1, 2),
    ],
)
def test_decode_releases_every_prefill_stage(params, pp_size, tp_size):
    scheduler = _decode_scheduler()

    scheduler._release_write_prefill_blocks(
        "req",
        {
            "transfer_id": "tA",
            "remote_dp_rank": 0,
            "remote_host": "10.0.0.1",
            "remote_notify_port": NOTIFY_PORT,
            **params,
        },
    )

    assert sorted(scheduler.released) == sorted(_prefill_stage_ports(pp_size, tp_size))


MR_BASE = 0x7F0000000000
LAYER_BYTES = 1 << 20
LAYERS = [f"model.layers.{i}.self_attn.attn" for i in range(4)]


def _remote_meta(*, advertise_index: bool = True) -> MoRIIOAgentMetadata:
    meta = MoRIIOAgentMetadata(
        engine_id="decode",
        agent_metadata=b"",
        kv_caches_base_addr=[MR_BASE + 4096 + i * LAYER_BYTES for i in range(4)],
        num_blocks=8,
        block_len=64,
        attn_backend_name="test",
        layer_base_addr_index=(
            {name: i for i, name in enumerate(LAYERS)} if advertise_index else {}
        ),
    )
    # The handshake carries the metadata as msgpack.
    return msgspec.msgpack.decode(
        msgspec.msgpack.encode(meta), type=MoRIIOAgentMetadata
    )


def _stage_worker(layers: list[str]) -> MoRIIOConnectorWorker:
    """A worker holding ``layers`` in one shared memory region."""
    worker = MoRIIOConnectorWorker.__new__(MoRIIOConnectorWorker)
    worker.kv_layer_mr_offset = {name: i * LAYER_BYTES for i, name in enumerate(layers)}
    worker.layer_base_addr_index = {name: i for i, name in enumerate(layers)}
    worker.num_regions = len(layers)
    worker.layer_name_to_remote_kv_cache_metadata = {
        "decode_dp0": {name: [b"shared-mr"] for name in LAYERS}
    }
    worker.moriio_wrapper = SimpleNamespace(
        get_unpack_memory_metadata=lambda packed: SimpleNamespace(data=MR_BASE),
        shutdown=lambda: None,
    )
    return worker


def test_stage_writes_each_layer_at_the_peer_address_of_that_layer():
    # Stage 1 of 2 holds layers 2 and 3; locally they are regions 0 and 1.
    worker = _stage_worker(LAYERS[2:])
    meta = _remote_meta()

    for i, name in enumerate(LAYERS[2:], start=2):
        assert worker._remote_layer_mr_offset(name, meta, "decode_dp0") == (
            4096 + i * LAYER_BYTES
        )
    local, remote, sizes = worker._apply_layer_mr_offsets(
        LAYERS[3], ([0, 64], [128, 192], [64, 64]), meta, "decode_dp0"
    )
    assert local == [LAYER_BYTES, LAYER_BYTES + 64]
    assert remote == [4096 + 3 * LAYER_BYTES + 128, 4096 + 3 * LAYER_BYTES + 192]
    assert sizes == [64, 64]


def test_peer_with_an_extra_draft_layer_is_addressed_by_name():
    """Without PP: a decode with MTP registers its draft latent after the
    target latents and before every indexer cache (one spec bucket per cache
    kind), so a producer without MTP sees each indexer one position early."""
    latents = ["model.layers.0.self_attn.attn", "model.layers.1.self_attn.attn"]
    indexers = [
        "model.layers.0.self_attn.indexer.k_cache",
        "model.layers.1.self_attn.indexer.k_cache",
    ]
    draft = "model.layers.2.self_attn.attn"
    decode_layers = [*latents, draft, *indexers]
    producer = _stage_worker([*latents, *indexers])
    producer.layer_name_to_remote_kv_cache_metadata = {
        "decode_dp0": {name: [b"shared-mr"] for name in decode_layers}
    }
    meta = MoRIIOAgentMetadata(
        engine_id="decode",
        agent_metadata=b"",
        kv_caches_base_addr=[MR_BASE + i * LAYER_BYTES for i in range(5)],
        num_blocks=8,
        block_len=64,
        attn_backend_name="test",
        layer_base_addr_index={name: i for i, name in enumerate(decode_layers)},
    )

    assert producer._remote_layer_mr_offset(indexers[0], meta, "decode_dp0") == (
        3 * LAYER_BYTES
    )
    # Positional lookup would have written it over the decode's draft latent.
    assert producer.layer_base_addr_index[indexers[0]] == decode_layers.index(draft)


def test_peer_without_layer_index_is_used_only_with_the_same_layers():
    meta = _remote_meta(advertise_index=False)

    full = _stage_worker(LAYERS)
    assert full._remote_layer_mr_offset(LAYERS[2], meta, "decode_dp0") == (
        4096 + 2 * LAYER_BYTES
    )

    # A stage holds fewer layers; its positions say nothing about the peer.
    stage = _stage_worker(LAYERS[2:])
    with pytest.raises(MoRIIOError, match="does not advertise"):
        stage._remote_layer_mr_offset(LAYERS[2], meta, "decode_dp0")


def test_layer_missing_on_the_peer_is_an_error():
    worker = _stage_worker(["model.layers.9.self_attn.attn"])

    with pytest.raises(MoRIIOError, match="no KV cache for layer"):
        worker._remote_layer_mr_offset(
            "model.layers.9.self_attn.attn", _remote_meta(), "decode_dp0"
        )


@pytest.fixture
def consumer_role(monkeypatch):
    monkeypatch.setattr(moriio_engine, "get_role", lambda: ROLE.CONSUMER)


def _write_done(transfer_id, pp_rank, tp_rank=0, num_writers=4):
    return {
        "type": "write_done",
        "transfer_id": transfer_id,
        "num_writers": num_writers,
        "sender": [pp_rank, tp_rank],
    }


@pytest.mark.usefixtures("consumer_role")
def test_decode_rank_waits_for_every_distinct_stage():
    wrapper = MoRIIOWrapper()

    for pp_rank in (0, 1, 2):
        wrapper._handle_structured_message(_write_done("t", pp_rank))
        assert wrapper.pop_finished_write_req_ids() == set()
    # A repeated sender is still one writer.
    wrapper._handle_structured_message(_write_done("t", 2))
    assert wrapper.pop_finished_write_req_ids() == set()

    wrapper._handle_structured_message(_write_done("t", 3))
    assert wrapper.pop_finished_write_req_ids() == {"t"}
    assert wrapper._write_done_senders == {}

    # A straggler for a finished transfer does not open a new count.
    wrapper._handle_structured_message(_write_done("t", 3))
    assert wrapper._write_done_senders == {}
    assert wrapper.pop_finished_write_req_ids() == set()


@pytest.mark.usefixtures("consumer_role")
def test_first_stage_failure_fails_the_transfer_once():
    wrapper = MoRIIOWrapper()

    wrapper._handle_structured_message(_write_done("t", 0))
    wrapper._handle_structured_message({"type": "write_failed", "transfer_id": "t"})
    assert wrapper.pop_failed_write_req_ids() == {"t"}

    # The other stages report later; none of it reaches the scheduler again.
    wrapper._handle_structured_message(_write_done("t", 2))
    wrapper._handle_structured_message(_write_done("t", 3))
    wrapper._handle_structured_message({"type": "write_failed", "transfer_id": "t"})
    assert wrapper.pop_failed_write_req_ids() == set()
    assert wrapper.pop_finished_write_req_ids() == set()
    assert wrapper._write_done_senders == {}


def _write_done_fields(pp_rank: int, pp_size: int) -> list:
    sent = []
    wrapper = SimpleNamespace(
        lock=threading.Lock(),
        done_req_ids=[],
        done_remote_allocate_req_dict={},
        waiting_for_transfer_complete=lambda statuses: None,
        _mark_transfer_terminal_locked=lambda transfer_id: None,
        send_notify=lambda *args, message_type, message_fields=None: sent.append(
            message_fields
        ),
    )
    writer = make_moriio_writer(SimpleNamespace(moriio_wrapper=wrapper, tp_rank=0))
    writer._pp_rank, writer._pp_size = pp_rank, pp_size
    info = RemoteAllocInfo(
        block_ids=[[7]],
        writes_done=1,
        writes_expected=1,
        completion_request_id="req",
        completion_notify_port=NOTIFY_PORT,
        completion_remote_ip="127.0.0.1",
    )

    writer._finalize_if_complete("t", info)
    return sent


def test_pipeline_free_single_writer_message_is_unchanged():
    assert _write_done_fields(pp_rank=0, pp_size=1) == [None]


def test_stage_reports_every_stage_as_a_writer_and_names_itself():
    assert _write_done_fields(pp_rank=2, pp_size=4) == [
        {"num_writers": 4, "sender": [2, 0]}
    ]
