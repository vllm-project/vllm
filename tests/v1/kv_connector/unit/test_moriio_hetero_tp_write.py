# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Mock end-to-end MoRIIO WRITE between prefill and decode of any TP sizes.

Every prefill rank runs the real writer and connector offset code on real
cache views; a fake MoRI transport copies the planned bytes into the decode
ranks' caches, and each decode rank counts write_done with the real wrapper.
Checks that every decode rank ends up with exactly its shard of every block
and recurrent-state slot, nothing else is touched, every prefill rank frees
its blocks, and a decode rank completes only after all its writers did.
"""

import threading
from collections import OrderedDict
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

import msgspec
import pytest
import torch

from vllm.distributed.kv_transfer.kv_connector.v1 import ssm_conv_transfer_utils
from vllm.distributed.kv_transfer.kv_connector.v1.moriio import moriio_engine
from vllm.distributed.kv_transfer.kv_connector.v1.moriio.moriio_common import (
    ROLE,
    MoRIIOAgentMetadata,
    MoRIIOError,
    RemoteAllocInfo,
    WriteTask,
)
from vllm.distributed.kv_transfer.kv_connector.v1.moriio.moriio_connector import (
    MoRIIOConnectorWorker,
)
from vllm.distributed.kv_transfer.kv_connector.v1.moriio.moriio_engine import (
    MoRIIOWrapper,
)
from vllm.distributed.kv_transfer.kv_connector.v1.moriio.moriio_layout import (
    kda_conv_ssm,
)
from vllm.distributed.kv_transfer.kv_connector.v1.moriio.moriio_tp_mapping import (
    _rank_heads,
)
from vllm.distributed.kv_transfer.kv_connector.v1.ssm_conv_transfer_utils import (
    derive_mamba_conv_split,
)
from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    MambaSpec,
    MLAAttentionSpec,
    SlidingWindowSpec,
    compute_layout_strides,
)
from vllm.v1.kv_cache_layout import KVCacheLayout

from .utils import make_moriio_writer

pytestmark = pytest.mark.cpu_test

NUM_BLOCKS = 6
BLOCK_SIZE = 4
NOTIFY_PORT = 20000
# Prefill blocks and the decode blocks they land in, per transfer group.
PREFILL_BLOCKS = {"full": [1, 4], "swa": [3], "mamba": [2]}
DECODE_BLOCKS = {"full": [5, 2], "swa": [0], "mamba": [3]}
# GDN state: Q and K are KEY_DIM channels, V is V_HEADS * V_DIM channels.
KEY_DIM, V_HEADS, V_DIM, K_DIM, CONV_ROWS = 4, 4, 2, 2, 2


@dataclass(frozen=True)
class Layer:
    name: str
    kind: str  # "full", "swa", "mla" or "mamba"
    group: str  # transfer group the layer's blocks come from


MODELS = {
    "gqa": (4, [Layer("attn0", "full", "full"), Layer("attn1", "full", "full")]),
    "gqa_replicated": (2, [Layer("attn0", "full", "full")]),
    "mla": (1, [Layer("mla0", "mla", "full"), Layer("mla1", "mla", "full")]),
    "sliding": (4, [Layer("attn0", "full", "full"), Layer("swa0", "swa", "swa")]),
    "gdn_hybrid": (
        4,
        [Layer("attn0", "full", "full"), Layer("gdn0", "mamba", "mamba")],
    ),
    "mla_hybrid": (1, [Layer("mla0", "mla", "full"), Layer("kda0", "mamba", "mamba")]),
}


def _groups(model: str, hma: bool) -> list[str]:
    _, layers = MODELS[model]
    groups = sorted({layer.group for layer in layers}, key=list(DECODE_BLOCKS).index)
    return groups if hma else ["full"]


def _spec(layer: Layer, num_heads: int, tp_size: int):
    if layer.kind == "full":
        return FullAttentionSpec(
            block_size=BLOCK_SIZE,
            num_kv_heads=num_heads,
            head_size=2,
            dtype=torch.bfloat16,
        )
    if layer.kind == "swa":
        return SlidingWindowSpec(
            block_size=BLOCK_SIZE,
            num_kv_heads=num_heads,
            head_size=2,
            dtype=torch.bfloat16,
            sliding_window=8,
        )
    if layer.kind == "mla":
        return MLAAttentionSpec(
            block_size=BLOCK_SIZE, num_kv_heads=1, head_size=8, dtype=torch.bfloat16
        )
    return MambaSpec(
        block_size=BLOCK_SIZE,
        shapes=(
            ((2 * KEY_DIM + V_HEADS * V_DIM) // tp_size, CONV_ROWS),
            (V_HEADS // tp_size, V_DIM, K_DIM),
        ),
        dtypes=(torch.uint8, torch.uint8),
        mamba_type=MambaAttentionBackendEnum.GDN_ATTN,
    )


def _global_state(layers: list[Layer], heads: int) -> dict[str, torch.Tensor]:
    """The unsharded content of every layer, per block / state slot."""
    gen = torch.Generator().manual_seed(0)
    state = {}
    for layer in layers:
        if layer.kind == "mamba":
            n = (2 * KEY_DIM + V_HEADS * V_DIM) * CONV_ROWS + V_HEADS * V_DIM * K_DIM
            shape: tuple[int, ...] = (NUM_BLOCKS, n)
        else:
            spec = _spec(layer, 1 if layer.kind == "mla" else heads, 1)
            shape = (
                NUM_BLOCKS,
                spec.num_heads,
                spec.num_states,
                spec.state_content_size_bytes,
            )
        state[layer.name] = torch.randint(
            1, 255, shape, dtype=torch.uint8, generator=gen
        )
    return state


def _mamba_shard(slot: torch.Tensor, tp_rank: int, tp_size: int) -> tuple:
    """(conv, ssm) bytes of ``tp_rank`` for one unsharded GDN state slot."""
    parts, offset = [], 0
    for channels in (KEY_DIM, KEY_DIM, V_HEADS * V_DIM):
        size = channels * CONV_ROWS
        shard = size // tp_size
        parts.append(slot[offset + tp_rank * shard : offset + (tp_rank + 1) * shard])
        offset += size
    ssm = slot[offset:].view(V_HEADS, -1)
    heads = V_HEADS // tp_size
    return torch.cat(parts), ssm[tp_rank * heads : (tp_rank + 1) * heads].reshape(-1)


@dataclass
class Rank:
    tp_rank: int
    tp_size: int
    caches: dict[str, torch.Tensor]  # registered tensor per layer
    regions: list[torch.Tensor]  # registered byte regions, in session order
    worker: MoRIIOConnectorWorker
    wrapper: Any = None
    sent: list = field(default_factory=list)


def _build_rank(model, tp_rank, tp_size, layout, hma, state, is_producer) -> Rank:
    heads, layers = MODELS[model]
    groups = _groups(model, hma)
    caches, regions, specs, meta_counts = {}, [], {}, {}
    for layer in layers:
        layer_heads = len(_rank_heads(tp_rank, tp_size, heads))
        spec = _spec(layer, layer_heads, tp_size)
        specs[layer.name] = spec
        buffer = torch.zeros(spec.page_size_bytes * NUM_BLOCKS, dtype=torch.uint8)
        if layer.kind == "mamba":
            cache = buffer.view(NUM_BLOCKS, 1, 1, spec.page_size_bytes)
            conv, ssm = kda_conv_ssm(cache, spec)
            if is_producer:
                for slot in range(NUM_BLOCKS):
                    c, s = _mamba_shard(state[layer.name][slot], tp_rank, tp_size)
                    conv[slot].view(-1).copy_(c)
                    ssm[slot].view(-1).copy_(s)
            regions += [
                MoRIIOConnectorWorker._contiguous_byte_alias(conv),
                MoRIIOConnectorWorker._contiguous_byte_alias(ssm),
            ]
            meta_counts[layer.name] = 2
        else:
            strides = compute_layout_strides(spec, NUM_BLOCKS, 1, layout)[1:]
            shape = (
                NUM_BLOCKS,
                spec.num_heads,
                spec.num_states,
                spec.state_content_size_bytes,
            )
            cache = torch.as_strided(buffer, shape, strides)
            if is_producer:
                held = (
                    slice(None)
                    if layer.kind == "mla"
                    else _rank_heads(tp_rank, tp_size, heads)
                )
                cache.copy_(state[layer.name][:, held])
            regions.append(buffer)
            meta_counts[layer.name] = 1
        caches[layer.name] = cache

    worker = MoRIIOConnectorWorker.__new__(MoRIIOConnectorWorker)
    worker.tp_rank = tp_rank
    worker.world_size = tp_size
    worker.use_mla = layers[0].kind == "mla"
    worker._has_mamba = any(layer.kind == "mamba" for layer in layers)
    worker.model_config = SimpleNamespace(get_total_num_kv_heads=lambda: heads)
    worker.kv_caches = caches
    worker.layer_to_spec = specs
    worker.layer_to_group = {
        layer.name: groups.index(layer.group if hma else "full") for layer in layers
    }
    worker._hybrid_payload_index_by_layer = {
        layer.name: groups.index("mamba") for layer in layers if layer.kind == "mamba"
    }
    worker._conv_decomp = next(
        (
            derive_mamba_conv_split(spec, tp_size)
            for spec in specs.values()
            if isinstance(spec, MambaSpec)
        ),
        None,
    )
    worker.kv_layer_mr_offset = {}
    worker._mamba_offset_templates = {}
    worker._region_session_index = None
    worker.layer_name_to_local_kv_cache_metadata = {
        layer.name: [b""] * meta_counts[layer.name] for layer in layers
    }
    worker._record_slice_layouts(caches)
    # Built without __init__: there is no engine to shut down.
    worker.shutdown = lambda: None
    return Rank(tp_rank, tp_size, caches, regions, worker)


class _Session:
    def __init__(self, src: torch.Tensor, dst: torch.Tensor):
        self.src, self.dst = src, dst


def _connect(producer: Rank, consumers: list[Rank], dp_engine: str) -> None:
    """Route the producer's sessions and notifications to the decode ranks."""
    consumer = consumers[0].worker
    metadata = MoRIIOAgentMetadata(
        engine_id="decode",
        agent_metadata=b"",
        kv_caches_base_addr=[],
        num_blocks=NUM_BLOCKS,
        block_len=0,
        attn_backend_name="test",
        kv_head_layouts={
            name: layout.to_wire() for name, layout in consumer.kv_head_layouts.items()
        },
        mamba_state_layouts={
            name: layout.to_wire()
            for name, layout in consumer.mamba_state_layouts.items()
        },
        attention_block_sizes=consumer.attention_block_sizes,
    )
    # The handshake carries the metadata as msgpack.
    metadata = msgspec.msgpack.decode(
        msgspec.msgpack.encode(metadata), type=MoRIIOAgentMetadata
    )

    def get_built_session(engine_id: str):
        if engine_id == f"{dp_engine}_dp0":
            (target,) = producer.worker._write_tp_mapping(len(consumers)).targets
            tp_rank = target.tp_rank
        else:
            tp_rank = int(engine_id.rsplit("_tp", 1)[1])
        remote = consumers[tp_rank]
        sessions = [
            _Session(src, dst)
            for src, dst in zip(producer.regions, remote.regions, strict=True)
        ]
        return sessions, metadata

    def write_remote_data(sizes, local_offsets, remote_offsets, session):
        for size, local, remote in zip(sizes, local_offsets, remote_offsets):
            session.dst[remote : remote + size] = session.src[local : local + size]
        return "status"

    def send_notify(transfer_id, ip, port, message_type, message_fields=None):
        producer.sent.append((port - NOTIFY_PORT, message_type))
        consumers[port - NOTIFY_PORT].wrapper._handle_structured_message(
            {"type": message_type, "transfer_id": transfer_id, **(message_fields or {})}
        )

    producer.worker._get_built_session = get_built_session
    producer.wrapper = SimpleNamespace(
        done_remote_allocate_req_dict={},
        lock=threading.Lock(),
        done_req_ids=[],
        write_remote_data=write_remote_data,
        send_notify=send_notify,
        waiting_for_transfer_complete=lambda statuses: None,
        _mark_transfer_terminal_locked=lambda transfer_id: None,
        _is_transfer_terminal_locked=lambda transfer_id: False,
    )
    producer.worker.moriio_wrapper = producer.wrapper


def _consumer_wrapper() -> MoRIIOWrapper:
    wrapper = MoRIIOWrapper.__new__(MoRIIOWrapper)
    wrapper.lock = threading.Lock()
    wrapper.done_write_cache_req_ids = []
    wrapper.failed_write_cache_req_ids = []
    wrapper._write_done_counts = {}
    wrapper._terminal_transfer_ids = OrderedDict()
    return wrapper


def _expected(
    layer: Layer, group: str, state, heads, tp_rank, tp_size, cache
) -> torch.Tensor:
    """What the decode cache must hold after the transfer."""
    expected = torch.zeros_like(cache)
    pairs = zip(PREFILL_BLOCKS[group], DECODE_BLOCKS[group])
    if layer.kind == "mamba":
        expected = torch.zeros(NUM_BLOCKS, cache.shape[-1], dtype=torch.uint8)
        for src, dst in pairs:
            conv, ssm = _mamba_shard(state[layer.name][src], tp_rank, tp_size)
            expected[dst, : conv.numel() + ssm.numel()] = torch.cat([conv, ssm])
        return expected
    held = slice(None) if layer.kind == "mla" else _rank_heads(tp_rank, tp_size, heads)
    for src, dst in pairs:
        expected[dst] = state[layer.name][src, held]
    return expected


SCENARIOS = [
    (model, producer_tp, consumer_tp, hma)
    for model in MODELS
    for producer_tp, consumer_tp in [(1, 2), (2, 4), (2, 1), (4, 2), (2, 2), (1, 4)]
    for hma in (True, False)
    # Recurrent state needs the hybrid KV cache manager.
    if hma or "hybrid" not in model
]


@pytest.mark.parametrize(
    ("layouts"),
    [
        (KVCacheLayout.LBNHC, KVCacheLayout.LBNHC),
        (KVCacheLayout.LBHNC, KVCacheLayout.LBHNC),
        (KVCacheLayout.LBNHC, KVCacheLayout.LBHNC),
    ],
    ids=["token-major", "head-major", "mixed"],
)
@pytest.mark.parametrize(("model", "producer_tp", "consumer_tp", "hma"), SCENARIOS)
def test_write_lands_every_shard_byte_exact(
    monkeypatch, model, producer_tp, consumer_tp, hma, layouts
):
    monkeypatch.setattr(
        ssm_conv_transfer_utils, "is_conv_state_dim_first", lambda: True
    )
    monkeypatch.setattr(moriio_engine, "get_role", lambda: ROLE.CONSUMER)
    heads, layers = MODELS[model]
    state = _global_state(layers, heads)
    groups = _groups(model, hma)
    consumers = [
        _build_rank(model, d, consumer_tp, layouts[1], hma, state, False)
        for d in range(consumer_tp)
    ]
    for consumer in consumers:
        consumer.wrapper = _consumer_wrapper()
    producers = [
        _build_rank(model, p, producer_tp, layouts[0], hma, state, True)
        for p in range(producer_tp)
    ]

    writers: dict[int, set[int]] = {d: set() for d in range(consumer_tp)}
    pending = []
    for producer in producers:
        _connect(producer, consumers, "decode")
        mapping = producer.worker._write_tp_mapping(consumer_tp)
        for target in mapping.targets:
            writers[target.tp_rank].add(producer.tp_rank)
        info = RemoteAllocInfo(block_ids=[DECODE_BLOCKS[g] for g in groups])
        producer.wrapper.done_remote_allocate_req_dict["t"] = info
        writer = make_moriio_writer(producer.worker)
        for layer in layers:
            writer._execute_write_task(
                WriteTask(
                    request_id="req",
                    transfer_id="t",
                    dst_engine_id="decode",
                    local_block_ids=[PREFILL_BLOCKS[g] for g in groups],
                    remote_block_ids_hint=None,
                    layer_name=layer.name,
                    event=SimpleNamespace(synchronize=lambda: None),
                    remote_notify_port=NOTIFY_PORT,
                    remote_ip="decode",
                    remote_tp_size=consumer_tp,
                    tp_mapping=mapping,
                )
            )
        info.writes_expected = len(layers)
        pending.append((producer, writer, info))

    finished: set[int] = set()
    for producer, writer, info in pending:
        writer._finalize_if_complete("t", info)
        finished.add(producer.tp_rank)
        # Prefill frees its blocks whether or not it had anything to write.
        assert [ack.transfer_id for ack in producer.wrapper.done_req_ids] == ["t"]
        for d, consumer in enumerate(consumers):
            done = consumer.wrapper.pop_finished_write_req_ids()
            assert done == (
                {"t"}
                if writers[d] <= finished and (producer.tp_rank in writers[d])
                else set()
            ), (d, finished)

    for consumer in consumers:
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


def test_write_refuses_peer_with_different_block_size(monkeypatch):
    """Hybrid MLA models get TP-dependent attention block sizes; copying blocks
    between them would misplace tokens, so the write must refuse."""
    monkeypatch.setattr(
        ssm_conv_transfer_utils, "is_conv_state_dim_first", lambda: True
    )
    monkeypatch.setattr(moriio_engine, "get_role", lambda: ROLE.CONSUMER)
    heads, layers = MODELS["mla"]
    state = _global_state(layers, heads)
    consumers = [_build_rank("mla", 0, 1, KVCacheLayout.LBNHC, True, state, False)]
    consumers[0].wrapper = _consumer_wrapper()
    for layer in layers:
        consumers[0].worker.attention_block_sizes[layer.name] = 2 * BLOCK_SIZE
    producer = _build_rank("mla", 0, 2, KVCacheLayout.LBNHC, True, state, True)
    _connect(producer, consumers, "decode")
    info = RemoteAllocInfo(block_ids=[DECODE_BLOCKS["full"]])
    producer.wrapper.done_remote_allocate_req_dict["t"] = info
    writer = make_moriio_writer(producer.worker)
    with pytest.raises(MoRIIOError, match="same --block-size"):
        writer._execute_write_task(
            WriteTask(
                request_id="req",
                transfer_id="t",
                dst_engine_id="decode",
                local_block_ids=[PREFILL_BLOCKS["full"]],
                remote_block_ids_hint=None,
                layer_name=layers[0].name,
                event=SimpleNamespace(synchronize=lambda: None),
                remote_notify_port=NOTIFY_PORT,
                remote_ip="decode",
                remote_tp_size=1,
                tp_mapping=producer.worker._write_tp_mapping(1),
            )
        )
