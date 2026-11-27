# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MoRIIO WRITE from a pipeline-parallel producer (e.g. P TP1 x PP4 -> D TP4)."""

import itertools
import socket
import threading
import time
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
    get_port_offset,
)
from vllm.distributed.kv_transfer.kv_connector.v1.moriio.moriio_connector import (
    MoRIIOConnectorScheduler,
    MoRIIOConnectorWorker,
)
from vllm.distributed.kv_transfer.kv_connector.v1.moriio.moriio_engine import (
    MoRIIOWrapper,
)
from vllm.v1.request import RequestStatus

from ..utils import make_moriio_writer

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

        # The first stage forwards to the same worker of every later stage.
        later = [
            other.notify_port
            for other in configs
            if other.pp_rank > 0
            and (other.dp_rank, other.tp_rank) == (config.dp_rank, config.tp_rank)
        ]
        assert config.later_stage_notify_ports() == (
            later if config.pp_rank == 0 else []
        )


def _decode_scheduler(*, send: bool = False) -> MoRIIOConnectorScheduler:
    """A decode scheduler that records its notifications, or sends them."""
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
    scheduler.paths = {}
    scheduler.notified = []
    scheduler.released = []
    if not send:
        scheduler.send_notify_block = (  # type: ignore[method-assign]
            lambda **kwargs: scheduler.notified.append(kwargs)
        )
        scheduler._send_transfer_release = (  # type: ignore[method-assign]
            lambda transfer_id, host, port: scheduler.released.append(port)
        )
    return scheduler


def _prefill_scheduler(pp_size: int, tp_size: int, host: str, notify_port: int):
    """The prefill-side state request_finished reads."""
    scheduler = SimpleNamespace(
        is_producer=True,
        _is_hma_required=False,
        _has_mamba=False,
        blocks_per_sw=[0],
        _global_dp_rank=0,
        engine_id="prefill",
        host_ip=host,
        handshake_port=HANDSHAKE_PORT,
        side_notify_port=notify_port,
        vllm_config=SimpleNamespace(
            parallel_config=SimpleNamespace(
                data_parallel_size=1,
                data_parallel_size_local=1,
                tensor_parallel_size=tp_size,
                pipeline_parallel_size=pp_size,
            )
        ),
    )
    scheduler.get_exchange_clipped_blocks = (
        MoRIIOConnectorScheduler.get_exchange_clipped_blocks.__get__(scheduler)
    )
    return scheduler


# The shapes of the decode request's kv_transfer_params a router can send. None
# carries a pipeline size.
PARAMS_SHAPES = ["peer sizes", "returned by prefill", "tp_size only"]


def _decode_params(shape, pp_size, tp_size, host="10.0.0.1", notify_port=NOTIFY_PORT):
    """The decode request's kv_transfer_params in one of PARAMS_SHAPES."""
    if shape == "returned by prefill":
        # What the prefill's request_finished returns, forwarded as is.
        _, params = MoRIIOConnectorScheduler.request_finished(
            _prefill_scheduler(pp_size, tp_size, host, notify_port),  # type: ignore[arg-type]
            SimpleNamespace(  # type: ignore[arg-type]
                request_id="req",
                kv_transfer_params={"do_remote_decode": True, "transfer_id": "tA"},
                status=RequestStatus.FINISHED_LENGTH_CAPPED,
            ),
            block_ids=[],
        )
        assert params is not None
        return params
    params = {
        "do_remote_prefill": True,
        "transfer_id": "tA",
        "remote_host": host,
        "remote_notify_port": notify_port,
        "remote_dp_size": 1,
    }
    if shape == "tp_size only":
        # One TP size and a pinned DP rank, set by the router itself.
        return {**params, "tp_size": tp_size, "remote_dp_rank": 0}
    # The prefill's TP and DP sizes, as a router learns them from registration.
    return {**params, "remote_tp_size": tp_size}


@pytest.mark.parametrize("shape", PARAMS_SHAPES)
@pytest.mark.parametrize(("pp_size", "tp_size"), [(4, 1), (2, 2)])
def test_decode_addresses_only_the_first_stage(shape, pp_size, tp_size):
    """The decode notifies a pipeline-parallel prefill as a pipeline-free one,
    whatever params it gets; the first stage forwards to the others."""
    params = _decode_params(shape, pp_size, tp_size)
    first_stage = sorted(
        _moriio_config(0, pp_size, tp, tp_size).notify_port for tp in range(tp_size)
    )
    scheduler = _decode_scheduler()

    scheduler.update_state_after_alloc(
        SimpleNamespace(request_id="req", kv_transfer_params=params),
        SimpleNamespace(
            get_block_ids=lambda: ([5, 6, 7],),
            get_unhashed_block_ids_all_groups=lambda: [[5, 6, 7]],
        ),
        num_external_tokens=40,
    )
    scheduler._release_write_prefill_blocks("req", {**params, "remote_dp_rank": 0})

    assert sorted(n["port"] for n in scheduler.notified) == first_stage
    assert all(n["block_notify_list"] == [5, 6, 7] for n in scheduler.notified)
    assert sorted(scheduler.released) == first_stage


def _free_consecutive_ports(n: int) -> list[int]:
    """N consecutive free ports, laid out like one TP1 worker per stage."""
    for base in range(20000, 30000, 97):
        socks = []
        try:
            for port in range(base, base + n):
                sock = socket.socket()
                socks.append(sock)
                sock.bind(("127.0.0.1", port))
        except OSError:
            continue
        finally:
            for sock in socks:
                sock.close()
        return list(range(base, base + n))
    raise RuntimeError("no free port range")


def _wait_for(condition, timeout: float = 10.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if condition():
            return True
        time.sleep(0.01)
    return condition()


@pytest.mark.parametrize("shape", PARAMS_SHAPES)
def test_every_stage_gets_the_allocation_and_release(shape, monkeypatch):
    """Decode scheduler -> first stage's notify listener -> later stages, over
    real sockets, with each params shape (P TP1 x PP4)."""
    monkeypatch.setattr(moriio_engine, "get_role", lambda: ROLE.PRODUCER)
    ports = _free_consecutive_ports(4)
    stages = []
    for port in ports:
        wrapper = MoRIIOWrapper()
        wrapper.notify_port = port
        stages.append(wrapper)
    stages[0].stage_notify_ports = ports[1:]
    for wrapper in stages:
        wrapper.async_wait_reqid()
    params = _decode_params(shape, 4, 1, host="127.0.0.1", notify_port=ports[0])
    scheduler = _decode_scheduler(send=True)

    scheduler.update_state_after_alloc(
        SimpleNamespace(request_id="req", kv_transfer_params=params),
        SimpleNamespace(
            get_block_ids=lambda: ([5, 6, 7],),
            get_unhashed_block_ids_all_groups=lambda: [[5, 6, 7]],
        ),
        num_external_tokens=40,
    )
    assert _wait_for(
        lambda: all("tA" in w.done_remote_allocate_req_dict for w in stages)
    )
    assert all(
        w.done_remote_allocate_req_dict["tA"].block_ids == [[5, 6, 7]] for w in stages
    )

    scheduler._release_write_prefill_blocks("req", {**params, "remote_dp_rank": 0})
    assert _wait_for(lambda: all(len(w.done_req_ids) == 1 for w in stages))
    assert all(w.done_req_ids[0].transfer_id == "tA" for w in stages)
    for sock in scheduler.paths.values():
        sock.close(linger=0)


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
    """A worker holding `layers` in one shared memory region."""
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
    writer = make_moriio_writer(
        SimpleNamespace(
            moriio_wrapper=wrapper, tp_rank=0, pp_rank=pp_rank, pp_size=pp_size
        )
    )
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
