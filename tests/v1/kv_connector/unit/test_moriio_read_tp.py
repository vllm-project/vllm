# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Mock end-to-end MoRIIO READ: what lands on decode, and who is released.

Every decode rank runs the real read path (source resolution, offsets,
completion polling) on real cache views; a fake MoRI transport copies the
planned bytes out of the prefill ranks' caches. Checks that every decode rank
ends up with exactly its shard, and that every prefill rank holding the
request is released as many times as it waits for.
"""

import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import msgspec
import pytest
import torch

from vllm.distributed.kv_transfer.kv_connector.v1 import ssm_conv_transfer_utils
from vllm.distributed.kv_transfer.kv_connector.v1.moriio.moriio_common import (
    MoRIIOAgentMetadata,
    MoRIIOConnectorMetadata,
    MoRIIOError,
    MoRIIOMode,
    MoRIIOTransferAck,
    ReqMeta,
)
from vllm.distributed.kv_transfer.kv_connector.v1.moriio.moriio_connector import (
    resolve_moriio_transfer_ack,
)
from vllm.distributed.kv_transfer.kv_connector.v1.moriio.moriio_engine import (
    MoRIIOWrapper,
)
from vllm.distributed.kv_transfer.kv_connector.v1.moriio.moriio_layout import (
    HeadLayout,
)
from vllm.v1.kv_cache_layout import KVCacheLayout

from .test_moriio_hetero_tp_write import (
    DECODE_BLOCKS,
    MIXED_HEAD_SCENARIOS,
    MODELS,
    NUM_BLOCKS,
    PREFILL_BLOCKS,
    SCENARIOS,
    Rank,
    _build_rank,
    _expected,
    _global_state,
    _groups,
)

pytestmark = pytest.mark.cpu_test

NOTIFY_PORT = 30000
LAYOUTS = [
    (KVCacheLayout.LBNHC, KVCacheLayout.LBNHC),
    (KVCacheLayout.LBHNC, KVCacheLayout.LBHNC),
    (KVCacheLayout.LBNHC, KVCacheLayout.LBHNC),
]
LAYOUT_IDS = ["token-major", "head-major", "mixed"]


class _Status:
    def __init__(self, ok: bool = True):
        self.ok = ok

    def Succeeded(self):
        return self.ok

    def Failed(self):
        return not self.ok

    def Message(self):
        return "" if self.ok else "failed"

    def Code(self):
        return 0 if self.ok else 1


class _Session:
    def __init__(self, local: torch.Tensor, remote: torch.Tensor, tp_rank: int):
        self.local, self.remote, self.tp_rank = local, remote, tp_rank


def _metadata(rank: Rank) -> MoRIIOAgentMetadata:
    worker = rank.worker
    metadata = MoRIIOAgentMetadata(
        engine_id="prefill",
        agent_metadata=b"",
        kv_caches_base_addr=[],
        num_blocks=NUM_BLOCKS,
        block_len=0,
        attn_backend_name="test",
        kv_head_layouts={
            name: layout.to_wire() for name, layout in worker.kv_head_layouts.items()
        },
        mamba_state_layouts={
            name: layout.to_wire()
            for name, layout in worker.mamba_state_layouts.items()
        },
        attention_block_sizes=worker.attention_block_sizes,
    )
    # The handshake carries the metadata as msgpack.
    return msgspec.msgpack.decode(
        msgspec.msgpack.encode(metadata), type=MoRIIOAgentMetadata
    )


def _make_reader(
    consumer: Rank,
    producers: list[Rank],
    backend: str,
    fail: bool = False,
    prewarmed: dict[str, int] | None = None,
) -> list:
    """Route the decode rank's reads to the prefill ranks; return its releases.

    With ``prewarmed`` (session key -> dialed prefill rank, filled by
    ``_prewarm``), a read may only use a session the eager handshake opened.
    """
    worker = consumer.worker
    released: list = []
    read_from: list[int] = []

    def get_built_session(engine_id: str):
        if prewarmed is not None:
            assert engine_id in prewarmed, (engine_id, sorted(prewarmed))
            tp_rank = prewarmed[engine_id]
        elif "_tp" in engine_id:
            tp_rank = int(engine_id.rsplit("_tp", 1)[1])
        else:
            tp_rank = worker._remote_tp_rank(len(producers))
        remote = producers[tp_rank]
        sessions = [
            _Session(local, remote_region, tp_rank)
            for local, remote_region in zip(
                consumer.regions, remote.regions, strict=True
            )
        ]
        return sessions, _metadata(remote)

    def read_remote_data(sizes, local_offsets, remote_offsets, session):
        read_from.append(session.tp_rank)
        for size, local, remote in zip(sizes, local_offsets, remote_offsets):
            session.local[local : local + size] = session.remote[remote : remote + size]
        return _Status(not fail)

    def send_notify(transfer_id, host, port, message_type=None, message_fields=None):
        released.append((int(port) - NOTIFY_PORT, message_type, message_fields))

    worker.mode = MoRIIOMode.READ
    worker.is_producer = False
    worker.dp_rank = 0
    worker._num_mamba_transfer_groups = 1 if worker._has_mamba else 0
    worker.moriio_config = SimpleNamespace(
        backend=backend, transfer_timeout=30.0, recv_abort_timeout=600.0
    )
    worker._get_built_session = get_built_session
    worker.moriio_wrapper = SimpleNamespace(
        lock=threading.Lock(),
        read_remote_data=read_remote_data,
        send_notify=send_notify,
        poll_transfer_batch=lambda statuses: MoRIIOWrapper.poll_transfer_batch(
            None, statuses
        ),
    )
    worker._recving_transfers = defaultdict(dict)
    worker._recving_transfers_start = {}
    worker._recving_transfers_callback_addr = {}
    worker._recving_local_blocks = {}
    worker._reads_issued_this_step = []
    worker._mamba_reads_this_step = []
    worker._invalid_block_ids = set()
    consumer.sent = read_from
    return released


def _meta(groups: list[str], producer_tp: int) -> ReqMeta:
    return ReqMeta(
        transfer_id="t",
        local_block_ids=[DECODE_BLOCKS[g] for g in groups],
        remote_block_ids=[PREFILL_BLOCKS[g] for g in groups],
        remote_host="prefill",
        remote_port=0,
        remote_handshake_port=0,
        remote_notify_port=NOTIFY_PORT,
        # start_load_kv keys the remote engine as host:handshake_port.
        remote_engine_id="prefill:0",
        tp_size=producer_tp,
        remote_dp_size=1,
    )


def _prewarm(consumer: Rank, meta: ReqMeta, monkeypatch) -> dict[str, int]:
    """Run the eager handshake; return session key -> dialed prefill rank."""
    worker = consumer.worker
    prewarmed: dict[str, int] = {}

    def handshake(host, port, tp_size, engine_id, dp_rank, tp_rank):
        prewarmed[engine_id] = (
            worker._remote_tp_rank(tp_size) if tp_rank is None else tp_rank
        )
        return set()

    worker._moriio_handshake = handshake
    worker._handshake_initiation_executor = ThreadPoolExecutor(1)
    worker._handshake_lock = threading.Lock()
    worker._eager_handshaked_engines = set()
    worker._remote_agents = {}
    worker.layer_name_to_remote_kv_cache_metadata = {}
    worker.tp_group = SimpleNamespace(cpu_group=None)
    monkeypatch.setattr(torch.distributed, "all_reduce", lambda *a, **k: None)
    metadata = MoRIIOConnectorMetadata()
    metadata.reqs_to_recv["req"] = meta
    worker._eager_handshake_all_dp_ranks(metadata)
    worker._handshake_initiation_executor.shutdown()
    assert prewarmed
    return prewarmed


def _ranks(model, producer_tp, consumer_tp, hma, layouts, state):
    producers = [
        _build_rank(model, p, producer_tp, layouts[0], hma, state, True)
        for p in range(producer_tp)
    ]
    consumers = [
        _build_rank(model, d, consumer_tp, layouts[1], hma, state, False)
        for d in range(consumer_tp)
    ]
    return producers, consumers


def _read_all(
    model, producer_tp, consumer_tp, hma, layouts, backend="xgmi", advertised_tp=None
):
    """Run every decode rank's READ; return the releases per prefill rank.

    ``advertised_tp`` is the prefill TP size the request carries, which is 0
    when the proxy does not forward it.
    """
    if advertised_tp is None:
        advertised_tp = producer_tp
    heads, layers = MODELS[model]
    state = _global_state(layers, heads)
    groups = _groups(model, hma)
    producers, consumers = _ranks(model, producer_tp, consumer_tp, hma, layouts, state)
    releases: dict[int, list] = defaultdict(list)
    for consumer in consumers:
        released = _make_reader(consumer, producers, backend)
        consumer.worker._read_blocks_for_req("req", _meta(groups, advertised_tp))
        assert consumer.worker._pop_done_transfers() == {"t"}
        for p, message_type, fields in released:
            releases[p].append((message_type, fields))
        for layer in layers:
            cache = consumer.caches[layer.name]
            received = cache.view(NUM_BLOCKS, -1) if layer.kind == "mamba" else cache
            expected = _expected(
                layer,
                layer.group if hma else "full",
                state,
                heads,
                consumer.tp_rank,
                consumer_tp,
                received,
            )
            assert torch.equal(received, expected), (layer.name, consumer.tp_rank)
    return releases


def _assert_released(releases: dict[int, list], producer_tp: int) -> None:
    """Every prefill rank frees the request exactly once, after its last ACK."""
    for p in range(producer_tp):
        counts: dict = {}
        completed: set = set()
        resolved = [
            resolve_moriio_transfer_ack(
                MoRIIOTransferAck("t", fields["consumer_tp_size"]),
                producer_tp_size=producer_tp,
                live_transfer_ids={"t"},
                notification_counts=counts,
                completed_transfer_ids=completed,
            )
            for message_type, fields in releases[p]
            if message_type == "release"
        ]
        assert resolved and resolved[-1] == "t", (p, releases[p])
        assert resolved.count("t") == 1, (p, releases[p])


@pytest.fixture(autouse=True)
def _conv_dim_first(monkeypatch):
    monkeypatch.setattr(
        ssm_conv_transfer_utils, "is_conv_state_dim_first", lambda: True
    )


@pytest.mark.parametrize("layouts", LAYOUTS, ids=LAYOUT_IDS)
@pytest.mark.parametrize("tp", [1, 2])
@pytest.mark.parametrize(
    ("model", "hma"),
    [
        (model, hma)
        for model in MODELS
        for hma in (True, False)
        # Recurrent state needs the hybrid KV cache manager.
        if hma or "hybrid" not in model
    ],
)
def test_equal_tp_read_lands_every_byte(model, hma, tp, layouts):
    """Prefill and decode may lay heads and tokens out differently."""
    _assert_released(_read_all(model, tp, tp, hma, layouts), tp)


@pytest.mark.parametrize("layouts", LAYOUTS, ids=LAYOUT_IDS)
@pytest.mark.parametrize(
    ("model", "producer_tp", "consumer_tp", "hma"),
    [
        scenario
        for scenario in SCENARIOS + MIXED_HEAD_SCENARIOS
        if scenario[1] != scenario[2]
    ],
)
def test_hetero_tp_read_lands_every_shard(
    model, producer_tp, consumer_tp, hma, layouts
):
    """Each decode rank reads its head and state slices from every prefill
    rank that holds them, and releases every holder."""
    releases = _read_all(model, producer_tp, consumer_tp, hma, layouts)
    _assert_released(releases, producer_tp)


@pytest.mark.parametrize("layouts", LAYOUTS, ids=LAYOUT_IDS)
@pytest.mark.parametrize("tp", [1, 2])
@pytest.mark.parametrize("model", list(MODELS))
def test_read_without_advertised_prefill_tp_is_homogeneous(model, tp, layouts):
    """A request without the prefill TP size (0) reads as equal TP."""
    _assert_released(_read_all(model, tp, tp, True, layouts, advertised_tp=0), tp)


def test_flexible_mirror_spreads_reads_and_releases_every_holder(monkeypatch):
    """MLA decode TP1 under prefill TP4 over RDMA: the round-robin picks a
    different prefill rank per request, and all four are released each time."""
    heads, layers = MODELS["mla"]
    state = _global_state(layers, heads)
    producers, (consumer,) = _ranks("mla", 4, 1, True, LAYOUTS[0], state)
    consumer.worker.moriio_config = SimpleNamespace(backend="rdma")
    prewarmed = _prewarm(consumer, _meta(["full"], 4), monkeypatch)
    assert sorted(prewarmed.values()) == [0, 1, 2, 3]
    released = _make_reader(consumer, producers, "rdma", prewarmed=prewarmed)
    picked = []
    for _ in range(4):
        consumer.sent.clear()
        released.clear()
        consumer.worker._read_blocks_for_req("req", _meta(["full"], 4))
        assert consumer.worker._pop_done_transfers() == {"t"}
        picked.append(set(consumer.sent))
        assert sorted(p for p, _, _ in released) == [0, 1, 2, 3]
    assert picked == [{0}, {1}, {2}, {3}]


def test_xgmi_mirror_reads_the_fixed_source(monkeypatch):
    """Over xGMI there is no NIC to balance, and MoRI cannot open sessions to
    every peer GPU, so the replicated MLA copy is read from one fixed rank,
    over the one session the eager handshake opened."""
    heads, layers = MODELS["mla"]
    state = _global_state(layers, heads)
    producers, (consumer,) = _ranks("mla", 4, 1, True, LAYOUTS[0], state)
    consumer.worker.moriio_config = SimpleNamespace(backend="xgmi")
    prewarmed = _prewarm(consumer, _meta(["full"], 4), monkeypatch)
    assert len(prewarmed) == 1
    released = _make_reader(consumer, producers, "xgmi", prewarmed=prewarmed)
    for _ in range(3):
        consumer.sent.clear()
        released.clear()
        consumer.worker._read_blocks_for_req("req", _meta(["full"], 4))
        assert consumer.worker._pop_done_transfers() == {"t"}
        assert set(consumer.sent) == set(prewarmed.values())
        assert sorted(p for p, _, _ in released) == [0, 1, 2, 3]


@pytest.mark.parametrize(
    ("producer_tp", "consumer_tp", "advertised_tp"),
    [(4, 2, 4), (2, 4, 2), (2, 2, 0)],
)
def test_aborted_read_releases_every_holder(producer_tp, consumer_tp, advertised_tp):
    heads, layers = MODELS["mla"]
    state = _global_state(layers, heads)
    producers, consumers = _ranks(
        "mla", producer_tp, consumer_tp, True, LAYOUTS[0], state
    )
    for consumer in consumers:
        released = _make_reader(consumer, producers, "xgmi")
        consumer.worker._read_blocks(
            local_block_ids=[],
            remote_block_ids=[PREFILL_BLOCKS["full"]],
            dst_engine_id="prefill",
            request_id="req",
            transfer_id="t",
            remote_host="prefill",
            remote_notify_port=NOTIFY_PORT,
            remote_tp_size=advertised_tp,
        )
        holders = consumer.worker._read_plan(producer_tp, None)[1]
        assert sorted(p for p, _, _ in released) == sorted(holders)
        assert consumer.sent == []


def test_failed_read_releases_every_holder():
    heads, layers = MODELS["mla"]
    state = _global_state(layers, heads)
    producers, (consumer,) = _ranks("mla", 2, 1, True, LAYOUTS[0], state)
    released = _make_reader(consumer, producers, "xgmi", fail=True)
    consumer.worker._read_blocks_for_req("req", _meta(["full"], 2))
    # A failed read is a load error, not a completion.
    assert consumer.worker._pop_done_transfers() == set()
    assert sorted(p for p, _, _ in released) == [0, 1]
    assert consumer.worker._invalid_block_ids == set(DECODE_BLOCKS["full"])


def test_read_refuses_peer_with_different_block_size():
    """Hybrid MLA models get TP-dependent attention block sizes; reading blocks
    between them would misplace tokens, so the read must refuse."""
    heads, layers = MODELS["mla"]
    state = _global_state(layers, heads)
    producers, (consumer,) = _ranks("mla", 2, 1, True, LAYOUTS[0], state)
    for layer in layers:
        producers[0].worker.attention_block_sizes[layer.name] *= 2
    _make_reader(consumer, producers, "xgmi")
    with pytest.raises(MoRIIOError, match="same --block-size"):
        consumer.worker._read_blocks_for_req("req", _meta(["full"], 2))


def test_sharded_read_needs_head_layouts():
    """A peer that does not advertise head layouts (an older vLLM) cannot be
    read by head slice."""
    heads, layers = MODELS["gqa"]
    state = _global_state(layers, heads)
    producers, (consumer,) = _ranks("gqa", 2, 1, True, LAYOUTS[0], state)
    for producer in producers:
        producer.worker.kv_head_layouts = {}
    _make_reader(consumer, producers, "xgmi")
    with pytest.raises(MoRIIOError, match="Heterogeneous-TP READ"):
        consumer.worker._read_blocks_for_req("req", _meta(["full"], 2))


def test_equal_tp_read_keeps_the_dp_session_and_one_release():
    """Equal TP is unchanged: one source, the DP-keyed session, one release."""
    heads, layers = MODELS["gqa"]
    state = _global_state(layers, heads)
    producers, consumers = _ranks("gqa", 2, 2, True, LAYOUTS[0], state)
    keys: list[str] = []
    for consumer in consumers:
        released = _make_reader(consumer, producers, "xgmi")
        built = consumer.worker._get_built_session

        def record(engine_id, built=built):
            keys.append(engine_id)
            return built(engine_id)

        consumer.worker._get_built_session = record
        consumer.worker._read_blocks_for_req("req", _meta(["full"], 2))
        consumer.worker._pop_done_transfers()
        assert [p for p, _, _ in released] == [consumer.tp_rank]
    assert keys == ["prefill:0_dp0", "prefill:0_dp0"]


def test_background_handshake_dials_every_read_source():
    """A heterogeneous-TP READ keys sessions per (dp, tp) source, so the lazy
    handshake must open exactly those."""
    heads, layers = MODELS["gqa"]
    state = _global_state(layers, heads)
    _, (consumer,) = _ranks("gqa", 4, 1, True, LAYOUTS[0], state)
    worker = consumer.worker
    worker.mode = MoRIIOMode.READ
    worker.is_producer = False
    submitted = []

    class Executor:
        def submit(self, fn, *args):
            submitted.append(args)
            return SimpleNamespace(
                result=lambda: True, add_done_callback=lambda callback: None
            )

    worker._handshake_initiation_executor = Executor()
    worker._handshake_futures = {}
    worker._handshake_lock = threading.Lock()
    worker._background_moriio_handshake("req", "prefill", _meta(["full"], 4))
    dialed = [(args[3], args[5]) for args in submitted if len(args) == 6]
    assert dialed == [(f"prefill_dp0_tp{p}", p) for p in range(4)]


def _mixed_layout_pair():
    """Equal-TP GQA ranks whose layouts differ, so reads go head by head."""
    heads, layers = MODELS["gqa"]
    state = _global_state(layers, heads)
    (producer,), (consumer,) = _ranks("gqa", 1, 1, True, LAYOUTS[2], state)
    return producer, consumer


def test_head_read_rejects_peer_with_other_head_count():
    producer, consumer = _mixed_layout_pair()
    metadata = _metadata(producer)
    layout = HeadLayout.from_wire(metadata.kv_head_layouts["attn0"])
    metadata.kv_head_layouts["attn0"] = layout._replace(
        num_heads=layout.num_heads // 2
    ).to_wire()
    with pytest.raises(MoRIIOError, match="KV heads per rank"):
        consumer.worker._compute_read_attention_offsets(
            "attn0",
            DECODE_BLOCKS["full"],
            PREFILL_BLOCKS["full"],
            metadata,
            remote_tp_size=1,
            remote_engine_id="prefill",
            source=None,
        )


@pytest.mark.parametrize(
    ("local", "remote", "match"),
    [
        ([NUM_BLOCKS], [0], f"local block id {NUM_BLOCKS} is outside"),
        ([0], [NUM_BLOCKS], f"remote block id {NUM_BLOCKS} is outside"),
        ([0], [-1], "remote block id -1 is outside"),
    ],
)
def test_head_read_rejects_block_ids_outside_the_cache(local, remote, match):
    producer, consumer = _mixed_layout_pair()
    with pytest.raises(ValueError, match=match):
        consumer.worker._compute_read_attention_offsets(
            "attn0",
            local,
            remote,
            _metadata(producer),
            remote_tp_size=1,
            remote_engine_id="prefill",
            source=None,
        )


def test_head_read_rejects_more_local_than_remote_blocks():
    """Equal TP between different layouts reads head by head, and refuses a
    local block table longer than the peer's."""
    producer, consumer = _mixed_layout_pair()
    remote = PREFILL_BLOCKS["full"]
    with pytest.raises(ValueError, match="longer than remote_block_ids"):
        consumer.worker._compute_read_attention_offsets(
            "attn0",
            DECODE_BLOCKS["full"][: len(remote)] + [0],
            remote,
            _metadata(producer),
            remote_tp_size=1,
            remote_engine_id="prefill",
            source=None,
        )
