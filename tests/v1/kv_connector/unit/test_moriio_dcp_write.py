# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MoRIIO WRITE into a decode instance with decode context parallelism (DCP)."""

import queue
import threading
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import msgspec
import pytest
import torch

from vllm.distributed.kv_transfer.kv_connector.v1.moriio import moriio_engine
from vllm.distributed.kv_transfer.kv_connector.v1.moriio.moriio_common import (
    ROLE,
    HandshakeError,
    HandshakeRefusedError,
    MoRIIOAgentMetadata,
    MoRIIOConnectorMetadata,
    MoRIIOMode,
    RemoteAllocInfo,
    ReqMeta,
    WriteTask,
    get_port_offset,
)
from vllm.distributed.kv_transfer.kv_connector.v1.moriio.moriio_connector import (
    MoRIIOConnectorWorker,
)
from vllm.distributed.kv_transfer.kv_connector.v1.moriio.moriio_engine import (
    MoRIIOWrapper,
)
from vllm.distributed.kv_transfer.kv_connector.v1.moriio.moriio_tp_mapping import (
    validate_peer_dcp_layout,
)
from vllm.utils.math_utils import cdiv
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    MambaSpec,
    MLAAttentionSpec,
    SparseCacheRole,
)

from .utils import make_moriio_writer

pytestmark = pytest.mark.cpu_test

BLOCK_SIZE = 64
NUM_BLOCKS = 24
NOTIFY_PORT = 20000


def _spec(head_size: int, **kwargs) -> MLAAttentionSpec:
    return MLAAttentionSpec(
        block_size=BLOCK_SIZE,
        num_kv_heads=1,
        head_size=head_size,
        dtype=torch.uint8,
        **kwargs,
    )


# The fp8 MLA latent of DeepSeek-V3.2 and GLM-5 (kv_lora_rank 512 +
# qk_rope_head_dim 64).
SPECS = {
    "latent": _spec(576),
}


def _cache(spec: MLAAttentionSpec) -> torch.Tensor:
    return torch.zeros(
        NUM_BLOCKS,
        spec.num_heads,
        spec.num_states,
        spec.state_content_size_bytes,
        dtype=torch.uint8,
    )


def _decode_position(token: int, dcp_size: int, interleave: int) -> tuple[int, int]:
    """DCP rank and position of a token, as the decode slot mapping places it
    (vllm/v1/worker/block_table.py)."""
    rank = token // interleave % dcp_size
    return rank, token // (interleave * dcp_size) * interleave + token % interleave


def _token_bytes(spec: MLAAttentionSpec, slot: int) -> torch.Tensor:
    """Offsets of one token's bytes in a block, as the cache writers lay them
    out."""
    entry = spec.state_content_size_bytes
    return slot * entry + torch.arange(entry)


def _validate(local_dcp, local_interleave, peer_dcp, peer_interleave, **kwargs):
    validate_peer_dcp_layout(
        local_dcp,
        local_interleave,
        peer_dcp,
        peer_interleave,
        block_size=BLOCK_SIZE,
        **{"writes_into_peer": True, **kwargs},
    )


@pytest.mark.parametrize(
    ("layout", "kwargs"),
    [
        ((1, 1, 4, 64), {}),
        ((1, 1, 4, 1), {}),
        ((1, 1, 1, 64), {}),
        ((8, 1, 8, 1), {"non_mla_layer": "attn0"}),
        ((8, 1, None, None), {"writes_into_peer": False}),
    ],
    ids=[
        "write-into-dcp",
        "write-into-dcp-unit-interleave",
        "no-dcp",
        "same-dcp-gqa",
        "older-peer",
    ],
)
def test_peer_dcp_layout_accepted(layout, kwargs):
    _validate(*layout, **kwargs)


@pytest.mark.parametrize(
    ("layout", "kwargs", "match"),
    [
        ((1, 1, 4, 48), {}, "divides the block size 64; it uses 48"),
        (
            (1, 1, 4, 1),
            {"whole_block_layer": "layer0"},
            "layer layer0 in whole blocks only.*needs --cp-kv-cache-interleave-size 64",
        ),
        ((1, 1, 4, 64), {"non_mla_layer": "attn0"}, "layer attn0 is not MLA"),
        ((2, 1, 4, 1), {}, "between decode_context_parallel_size=2 here and 4"),
        ((1, 1, 4, 64), {"writes_into_peer": False}, "=1 here and 4 on the peer"),
        ((8, 64, 8, 1), {"writes_into_peer": False}, "64 here, 1 on the peer"),
    ],
    ids=[
        "write-into-dcp-interleave-not-dividing-block",
        "write-into-dcp-whole-block-layer",
        "write-into-dcp-gqa",
        "dcp2-into-dcp4",
        "read-no-dcp-from-dcp",
        "same-dcp-interleave",
    ],
)
def test_peer_dcp_layout_refused(layout, kwargs, match):
    with pytest.raises(ValueError, match=match):
        _validate(*layout, **kwargs)


def _worker(
    tp_rank: int, tp_size: int, specs: dict[str, MLAAttentionSpec], dcp_size: int = 1
) -> MoRIIOConnectorWorker:
    caches = {name: _cache(spec) for name, spec in specs.items()}
    worker = MoRIIOConnectorWorker.__new__(MoRIIOConnectorWorker)
    worker.tp_rank = tp_rank
    worker.world_size = tp_size
    worker.dcp_size = dcp_size
    worker.block_size = BLOCK_SIZE
    worker.use_mla = True
    worker._has_mamba = False
    worker.model_config = SimpleNamespace(
        get_total_num_kv_heads=lambda: 1, get_head_size=lambda: 576
    )
    worker.kv_caches = caches
    worker.layer_to_spec = dict(specs)
    worker.layer_to_group = dict.fromkeys(caches, 0)
    worker.kv_layer_mr_offset = {}
    worker._region_session_index = None
    worker.layer_name_to_local_kv_cache_metadata = {name: [b""] for name in caches}
    worker._record_slice_layouts(caches)
    worker.shutdown = lambda: None
    return worker


@pytest.mark.parametrize(
    ("specs", "whole_block_layer"),
    [
        (SPECS, None),
        (
            {**SPECS, "indexer": _spec(132, cache_role=SparseCacheRole.INDEXER)},
            "indexer",
        ),
        ({"latent": _spec(576, tokens_per_state=4)}, "latent"),
    ],
    ids=["latent", "indexer", "compressed-states"],
)
def test_only_known_token_layouts_are_written_in_runs(specs, whole_block_layer):
    """Runs of tokens are written only for an MLA latent of the model's latent
    width; any other layer, such as the indexer K cache of DeepSeek-V3.2 and
    GLM-5, needs whole blocks, i.e. an interleave equal to the block size."""
    worker = _worker(0, 1, specs)
    for interleave in (1, 16):
        assert worker._whole_block_dcp_layer(interleave) == whole_block_layer
    assert worker._whole_block_dcp_layer(BLOCK_SIZE) is None


KDA = MambaSpec(
    block_size=BLOCK_SIZE, shapes=((4, 3), (2, 8, 8)), dtypes=(torch.uint8,) * 2
)
GQA = FullAttentionSpec(
    block_size=BLOCK_SIZE, num_kv_heads=8, head_size=64, dtype=torch.uint8
)


@pytest.mark.parametrize(
    ("mla_specs", "other_specs", "refused"),
    [
        (
            {**SPECS, "indexer": _spec(132, cache_role=SparseCacheRole.INDEXER)},
            {},
            {1: "layer indexer in whole blocks only.*interleave-size 64"},
        ),
        (
            {"latent": _spec(576, tokens_per_state=4)},
            {},
            {1: "layer latent in whole blocks only.*interleave-size 64"},
        ),
        (SPECS, {"kda": KDA}, {}),
        (
            {**SPECS, "draft": _spec(576, non_causal_multi_token_decode=True)},
            {"kda": KDA},
            {},
        ),
        (
            SPECS,
            {"kda": KDA, "draft": GQA},
            dict.fromkeys((1, BLOCK_SIZE), "layer draft is not MLA"),
        ),
        ({}, {"attn": GQA}, dict.fromkeys((1, BLOCK_SIZE), "layer attn is not MLA")),
    ],
    ids=[
        "glm5-deepseek-v32",
        "compressed-states",
        "kimi-linear",
        "kimi-k3-dspark",
        "non-mla-draft",
        "gqa",
    ],
)
def test_handshake_accepts_each_model_at_its_interleaves(
    mla_specs, other_specs, refused
):
    """A prefill instance without DCP writes a DCP decode instance's MLA
    layers. DSpark and DFlash run a draft without MLA without DCP, so the
    decode instance keeps its cache whole, which the prefill instance cannot
    see: such a draft is refused."""
    worker = _worker(0, 1, mla_specs)
    worker.use_mla = bool(mla_specs)
    for name, spec in other_specs.items():
        worker.kv_caches[name] = None
        worker.layer_to_spec[name] = spec
    # A layer whose KV is not transferred is never written.
    worker.layer_to_spec["not_transferred"] = GQA
    for interleave in (1, BLOCK_SIZE):
        kwargs = {
            "non_mla_layer": worker._non_mla_dcp_layer(),
            "whole_block_layer": worker._whole_block_dcp_layer(interleave),
        }
        if interleave in refused:
            with pytest.raises(ValueError, match=refused[interleave]):
                _validate(1, 1, 4, interleave, **kwargs)
        else:
            _validate(1, 1, 4, interleave, **kwargs)


def _wrapper() -> MoRIIOWrapper:
    wrapper = MoRIIOWrapper.__new__(MoRIIOWrapper)
    wrapper.lock = threading.Lock()
    wrapper.done_req_ids = []
    wrapper.done_remote_allocate_req_dict = {}
    wrapper.done_write_cache_req_ids = []
    wrapper.failed_write_cache_req_ids = []
    wrapper._write_done_counts = {}
    wrapper._terminal_transfer_ids = OrderedDict()
    return wrapper


def _decode_metadata(dcp_size: int, interleave: int) -> MoRIIOAgentMetadata:
    metadata = MoRIIOAgentMetadata(
        engine_id="decode",
        agent_metadata=b"",
        kv_caches_base_addr=[],
        num_blocks=NUM_BLOCKS,
        block_len=0,
        attn_backend_name="test",
        attention_block_sizes=dict.fromkeys(SPECS, BLOCK_SIZE),
        dcp_size=dcp_size,
        cp_kv_cache_interleave_size=interleave,
    )
    return msgspec.msgpack.decode(
        msgspec.msgpack.encode(metadata), type=MoRIIOAgentMetadata
    )


def _write_task(name: str, prefill_blocks: list[int], mapping) -> WriteTask:
    return WriteTask(
        request_id="req",
        transfer_id="t",
        dst_engine_id="decode",
        local_block_ids=[prefill_blocks],
        remote_block_ids_hint=None,
        layer_name=name,
        event=SimpleNamespace(synchronize=lambda: None),
        remote_notify_port=NOTIFY_PORT,
        remote_ip="decode",
        remote_tp_size=mapping.consumer_tp_size,
        tp_mapping=mapping,
    )


def _run_write(
    producer_tp: int,
    consumer_tp: int,
    dcp_size: int,
    num_prefill_blocks: int,
    decode_blocks: list[int],
    interleave: int,
    producer_dcp_size: int = 1,
):
    """Run every prefill rank's writer for all layers into fake decode ranks;
    return their caches and write_done results, the prefill block contents and
    the bytes each prefill rank wrote."""
    gen = torch.Generator().manual_seed(0)
    content = {
        name: torch.randint(
            1, 255, (num_prefill_blocks, *_cache(spec).shape[1:]), generator=gen
        ).to(torch.uint8)
        for name, spec in SPECS.items()
    }
    # Prefill block ids are not in order, so a pairing by position shows up.
    prefill_blocks = [NUM_BLOCKS - 1 - j for j in range(num_prefill_blocks)]
    consumer_caches = [
        {name: _cache(spec) for name, spec in SPECS.items()} for _ in range(consumer_tp)
    ]
    consumer_wrappers = [_wrapper() for _ in range(consumer_tp)]
    metadata = _decode_metadata(dcp_size, interleave)
    bytes_written = []
    for p in range(producer_tp):
        worker = _worker(p, producer_tp, SPECS, producer_dcp_size)
        for name, cache in worker.kv_caches.items():
            cache[prefill_blocks] = content[name]
        mapping = worker._write_tp_mapping(consumer_tp)

        def get_built_session(engine_id, worker=worker, mapping=mapping):
            if engine_id == "decode_dp0":
                (target,) = mapping.targets
                tp_rank = target.tp_rank
            else:
                tp_rank = int(engine_id.rsplit("_tp", 1)[1])
            sessions = [
                SimpleNamespace(
                    src=worker.kv_caches[name].view(-1),
                    dst=consumer_caches[tp_rank][name].view(-1),
                )
                for name in worker.kv_caches
            ]
            return sessions, metadata

        def write_remote_data(sizes, local_offsets, remote_offsets, session):
            for size, local, remote in zip(sizes, local_offsets, remote_offsets):
                session.dst[remote : remote + size] = session.src[local : local + size]
            return "status"

        def send_notify(transfer_id, ip, port, message_type, message_fields=None):
            message = {"type": message_type, "transfer_id": transfer_id}
            consumer_wrappers[port - NOTIFY_PORT]._handle_structured_message(
                {**message, **(message_fields or {})}
            )

        worker._get_built_session = get_built_session
        info = RemoteAllocInfo(block_ids=[list(decode_blocks)])
        worker.moriio_wrapper = SimpleNamespace(
            done_remote_allocate_req_dict={"t": info},
            lock=threading.Lock(),
            done_req_ids=[],
            write_remote_data=write_remote_data,
            send_notify=send_notify,
            waiting_for_transfer_complete=lambda statuses: None,
            _mark_transfer_terminal_locked=lambda transfer_id: None,
            _is_transfer_terminal_locked=lambda transfer_id: False,
        )
        writer = make_moriio_writer(worker)
        for name in worker.kv_caches:
            writer._execute_write_task(_write_task(name, prefill_blocks, mapping))
        info.writes_expected = len(worker.kv_caches)
        writer._finalize_if_complete("t", info)
        # Prefill frees its blocks whether or not it had anything to write.
        assert [ack.transfer_id for ack in worker.moriio_wrapper.done_req_ids] == ["t"]
        bytes_written.append(info.bytes_written)
    done = [wrapper.pop_finished_write_req_ids() for wrapper in consumer_wrappers]
    return consumer_caches, done, content, bytes_written


def _expected_decode_cache(
    spec: MLAAttentionSpec,
    content: torch.Tensor,
    decode_blocks: list[int],
    dcp_size: int,
    dcp_rank: int,
    interleave: int,
) -> torch.Tensor:
    """A decode DCP rank's cache once every token it holds has arrived."""
    expected = torch.zeros_like(_cache(spec)).view(NUM_BLOCKS, -1)
    source = content.view(len(content), -1)
    for token in range(len(content) * BLOCK_SIZE):
        rank, position = _decode_position(token, dcp_size, interleave)
        if rank == dcp_rank:
            expected[
                decode_blocks[position // BLOCK_SIZE],
                _token_bytes(spec, position % BLOCK_SIZE),
            ] = source[token // BLOCK_SIZE, _token_bytes(spec, token % BLOCK_SIZE)]
    return expected.view_as(_cache(spec))


@pytest.mark.parametrize("interleave", [1, 16, BLOCK_SIZE])
@pytest.mark.parametrize("num_prefill_blocks", [1, 5])
@pytest.mark.parametrize(
    ("producer_tp", "consumer_tp", "dcp_size"), [(1, 4, 4), (8, 4, 2)]
)
def test_write_lands_each_token_on_its_dcp_rank(
    monkeypatch, producer_tp, consumer_tp, dcp_size, num_prefill_blocks, interleave
):
    """An MLA model writes its latent cache token by token at any interleave,
    from one prefill rank into several decode ranks and from several prefill
    ranks into one."""
    monkeypatch.setattr(moriio_engine, "get_role", lambda: ROLE.CONSUMER)
    # One spare decode block, as when decode reserves lookahead slots.
    num_decode_blocks = cdiv(num_prefill_blocks, dcp_size) + 1
    decode_blocks = [2 * i + 1 for i in range(num_decode_blocks)]
    caches, done, content, bytes_written = _run_write(
        producer_tp,
        consumer_tp,
        dcp_size,
        num_prefill_blocks,
        decode_blocks,
        interleave,
    )

    # Every decode rank completes, including ranks that hold no token of a
    # short prompt.
    assert done == [{"t"}] * consumer_tp
    for t in range(consumer_tp):
        for name, spec in SPECS.items():
            expected = _expected_decode_cache(
                spec, content[name], decode_blocks, dcp_size, t % dcp_size, interleave
            )
            assert torch.equal(caches[t][name], expected), (t, name)
    # Each token is written once per DCP group (consumer_tp // dcp_size
    # replicas), 1 / dcp_size of what the same TP without DCP writes.
    block_len = sum(_cache(spec)[0].numel() for spec in SPECS.values())
    total = num_prefill_blocks * block_len
    assert sum(bytes_written) == total * consumer_tp // dcp_size


def test_write_refuses_decode_allocation_short_of_its_blocks(monkeypatch):
    monkeypatch.setattr(moriio_engine, "get_role", lambda: ROLE.CONSUMER)
    with pytest.raises(ValueError, match="allocated 1"):
        _run_write(1, 4, 4, num_prefill_blocks=5, decode_blocks=[3], interleave=1)


def test_write_between_equal_dcp_sizes_is_rank_to_rank(monkeypatch):
    """Prefill and decode with the same DCP size hold the same shard on the same
    rank, so each rank writes its own blocks unchanged to its peer."""
    monkeypatch.setattr(moriio_engine, "get_role", lambda: ROLE.CONSUMER)
    decode_blocks = [2 * i + 1 for i in range(5)]
    caches, done, content, _ = _run_write(
        4, 4, 4, 5, decode_blocks, interleave=1, producer_dcp_size=4
    )

    assert done == [{"t"}] * 4
    for t in range(4):
        for name in SPECS:
            expected = torch.zeros_like(caches[t][name])
            expected[decode_blocks] = content[name]
            assert torch.equal(caches[t][name], expected), (t, name)


def _handshake_worker(is_producer: bool, handshake) -> MoRIIOConnectorWorker:
    """Rank 0 of a prefill instance with TP 1 in WRITE mode or of a decode
    instance with TP 2 in READ mode; ``handshake`` stands in for the peer."""
    worker = MoRIIOConnectorWorker.__new__(MoRIIOConnectorWorker)
    worker.is_producer = is_producer
    worker.mode = MoRIIOMode.WRITE if is_producer else MoRIIOMode.READ
    worker.tp_rank = 0
    worker.world_size = 1 if is_producer else 2
    worker.use_mla = True
    worker._has_mamba = False
    worker.layer_to_spec = {}
    worker.moriio_config = SimpleNamespace(backend="xgmi", transfer_timeout=30.0)
    worker._moriio_handshake = handshake
    worker._handshake_lock = threading.RLock()
    worker._handshake_initiation_executor = ThreadPoolExecutor(1)
    worker._handshake_futures = {}
    worker._eager_handshaked_engines = set()
    worker._remote_agents = {}
    worker.remote_moriio_metadata = {}
    worker.write_ready_flags = {}
    worker.load_ready_flag = {}
    worker._ready_requests = queue.Queue()
    worker._invalid_block_ids = set()
    worker._reqs_to_send = {}
    worker.tp_group = SimpleNamespace(cpu_group=None)
    worker.shutdown = lambda: None
    return worker


def _req_meta(remote_host: str, tp_size: int) -> ReqMeta:
    return ReqMeta(
        transfer_id="t",
        local_block_ids=[[5, 6]],
        remote_block_ids=[[7, 8]],
        remote_host=remote_host,
        remote_port=0,
        remote_handshake_port=0,
        remote_notify_port=NOTIFY_PORT,
        remote_engine_id=f"{remote_host}:0",
        tp_size=tp_size,
        remote_dp_size=1,
    )


def _refuse(host, port, tp_size, engine_id, dp_rank, tp_rank):
    raise HandshakeRefusedError(f"Cannot transfer KV with {engine_id}")


@pytest.mark.parametrize("targets_connected", [False, True])
def test_refused_write_handshake_fails_the_request_on_every_decode_rank(
    targets_connected,
):
    """A refused handshake is tried once per request, not once per layer. The
    request's first write task frees the prefill blocks and tells each decode
    rank it would have written that the transfer failed, unless those ranks
    are connected through another handshake."""
    handshakes = []

    def refuse(*args):
        handshakes.append(args[3])
        _refuse(*args)

    worker = _handshake_worker(True, refuse)
    wrapper = _wrapper()
    sent = []
    wrapper.send_notify = lambda transfer_id, ip, port, message_type: sent.append(
        (transfer_id, port, message_type)
    )
    worker.moriio_wrapper = wrapper
    writer = worker._writer = make_moriio_writer(worker)
    tasks = []

    def write_blocks(req_id, meta, layer_name, kv_layer):
        task = _write_task(layer_name, [5, 6], worker._write_tp_mapping(meta.tp_size))
        task.dst_engine_id = meta.remote_engine_id
        if writer.schedule_write(task):
            tasks.append(task)

    worker._write_blocks_for_req = write_blocks
    metadata = MoRIIOConnectorMetadata()
    metadata.reqs_to_save["req"] = _req_meta("decode", tp_size=4)

    for layer in ("mla0", "mla1", "mla2"):
        worker.save_kv_layer(metadata, layer, kv_layer=None, attn_metadata=None)
    worker._handshake_initiation_executor.shutdown()

    # One handshake per decode rank of the MLA fan-out, for the first layer.
    assert sorted(handshakes) == [f"decode:0_dp0_tp{t}" for t in range(4)]
    assert {task.layer_name for task in tasks} >= {"mla1", "mla2"}

    wrapper.done_remote_allocate_req_dict["t"] = RemoteAllocInfo(
        block_ids=[[7, 8]], decode_dp_rank=0
    )
    written = []
    if targets_connected:
        worker.remote_moriio_metadata = dict.fromkeys(handshakes)
        worker._get_built_session = lambda engine_id: ([engine_id], None)
        writer._prepare_transfer_plan = lambda *args, **kwargs: SimpleNamespace(
            transfer_sizes=[64]
        )

        def do_layer_write(plan, sessions):
            written.append(sessions)
            return []

        writer._do_layer_write = do_layer_write
    writer._execute_write_task(tasks[0])

    if targets_connected:
        assert written == [[f"decode:0_dp0_tp{t}"] for t in range(4)]
        assert sent == [] and wrapper.done_req_ids == []
        return
    assert [ack.transfer_id for ack in wrapper.done_req_ids] == ["t"]
    assert sent == [
        ("t", NOTIFY_PORT + get_port_offset(0, t), "write_failed") for t in range(4)
    ]
    # The request's other write tasks are dropped.
    assert writer._is_transfer_terminal("t")
    assert not writer.is_unconnected("t")


VOTES = {"connected": 2, "refused": 1, "failed": 0}


@pytest.mark.parametrize(
    ("here", "peer", "has_mamba", "outcome"),
    [
        ("connected", "connected", False, "read"),
        ("refused", "connected", False, "fail"),
        ("connected", "refused", False, "fail"),
        ("failed", "connected", False, "raise"),
        ("refused", "refused", True, "raise"),
    ],
)
def test_read_refusal_fails_the_requests_on_every_tp_rank(
    monkeypatch, here, peer, has_mamba, outcome
):
    """A decode instance in READ mode fails the requests from a prefill
    instance whose layout any TP rank refused, the same way on every rank,
    and keeps running; any other handshake failure still fails the step."""

    def handshake(*args):
        if here == "refused":
            _refuse(*args)
        if here == "failed":
            raise HandshakeError("connection reset")
        return {"agent"}

    def all_reduce(vote, group, op):
        vote[0] = min(int(vote[0]), VOTES[peer])

    monkeypatch.setattr(torch.distributed, "all_reduce", all_reduce)
    worker = _handshake_worker(False, handshake)
    worker._has_mamba = has_mamba
    reads = []
    worker._read_blocks_for_req = lambda req_id, meta: reads.append(
        (req_id, meta.local_block_ids)
    )
    worker._await_reads_issued_this_step = lambda: None
    metadata = MoRIIOConnectorMetadata()
    metadata.reqs_to_recv["req"] = _req_meta("prefill", tp_size=2)

    if outcome == "raise":
        with pytest.raises(HandshakeError, match="failing this step fast"):
            worker.start_load_kv(metadata)
    else:
        worker.start_load_kv(metadata)
    worker._handshake_initiation_executor.shutdown()

    # A refused peer gets a read into no blocks, which only releases its blocks.
    expected_reads = {"read": [("req", [[5, 6]])], "fail": [("req", [[]])]}
    assert reads == expected_reads.get(outcome, [])
    assert worker.get_block_ids_with_load_errors() == (
        {5, 6} if outcome == "fail" else set()
    )
    # A refused peer is handshaked again for the next requests.
    assert worker._eager_handshaked_engines == (
        {"prefill:0"} if outcome == "read" else set()
    )
