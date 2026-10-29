# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MoRIIO WRITE into a decode instance with decode context parallelism (DCP)."""

import threading
from collections import OrderedDict
from types import SimpleNamespace

import msgspec
import pytest
import torch

from vllm.distributed.kv_transfer.kv_connector.v1.moriio import moriio_engine
from vllm.distributed.kv_transfer.kv_connector.v1.moriio.moriio_common import (
    ROLE,
    MoRIIOAgentMetadata,
    RemoteAllocInfo,
    WriteTask,
)
from vllm.distributed.kv_transfer.kv_connector.v1.moriio.moriio_connector import (
    MoRIIOConnectorWorker,
)
from vllm.distributed.kv_transfer.kv_connector.v1.moriio.moriio_engine import (
    MoRIIOWrapper,
)
from vllm.distributed.kv_transfer.kv_connector.v1.moriio.moriio_tp_mapping import (
    dcp_write_block_pairs,
    validate_write_dcp_topology,
)
from vllm.utils.math_utils import cdiv
from vllm.v1.kv_cache_interface import MLAAttentionSpec

from .utils import make_moriio_writer

pytestmark = pytest.mark.cpu_test

BLOCK_SIZE = 4
NUM_BLOCKS = 24
NOTIFY_PORT = 20000
LAYERS = ("mla0", "mla1")


@pytest.mark.parametrize("num_tokens", [1, 65, 256, 257, 1000])
@pytest.mark.parametrize("dcp_size", [2, 4])
@pytest.mark.parametrize("extra_remote_blocks", [0, 1])
def test_dcp_pairs_cover_every_block_once(num_tokens, dcp_size, extra_remote_blocks):
    """Prompt lengths around the 64-token block and 256-token DCP4 boundaries."""
    block_size = 64
    num_local = cdiv(num_tokens, block_size)
    num_remote = cdiv(num_tokens, block_size * dcp_size) + extra_remote_blocks
    local = list(range(100, 100 + num_local))
    remote = list(range(500, 500 + num_remote))

    written = []
    for dcp_rank in range(dcp_size):
        pairs = dcp_write_block_pairs(local, remote, dcp_size, dcp_rank)
        written += [(lb, dcp_rank, rb) for lb, rb in zip(*pairs)]
    assert sorted(lb for lb, _, _ in written) == local
    for lb, dcp_rank, rb in written:
        j = local.index(lb)
        assert (dcp_rank, rb) == (j % dcp_size, remote[j // dcp_size])


def test_validate_write_dcp_topology():
    validate_write_dcp_topology(1, 1, 16, is_mla=False, has_mamba=True)
    validate_write_dcp_topology(4, 64, 64, is_mla=True, has_mamba=False)
    with pytest.raises(ValueError, match="MLA models without recurrent state"):
        validate_write_dcp_topology(4, 64, 64, is_mla=False, has_mamba=False)
    with pytest.raises(ValueError, match="MLA models without recurrent state"):
        validate_write_dcp_topology(4, 64, 64, is_mla=True, has_mamba=True)
    with pytest.raises(
        ValueError, match="cp_kv_cache_interleave_size=1, block_size=64"
    ):
        validate_write_dcp_topology(4, 1, 64, is_mla=True, has_mamba=False)


def test_handshake_metadata_carries_dcp_and_defaults_for_older_peers():
    fields = dict(
        engine_id="decode",
        agent_metadata=b"",
        kv_caches_base_addr=[],
        num_blocks=NUM_BLOCKS,
        block_len=0,
        attn_backend_name="test",
    )
    meta = MoRIIOAgentMetadata(**fields, dcp_size=4, cp_kv_cache_interleave_size=64)
    decoded = msgspec.msgpack.decode(
        msgspec.msgpack.encode(meta), type=MoRIIOAgentMetadata
    )
    assert (decoded.dcp_size, decoded.cp_kv_cache_interleave_size) == (4, 64)
    older = msgspec.msgpack.decode(
        msgspec.msgpack.encode(fields), type=MoRIIOAgentMetadata
    )
    assert (older.dcp_size, older.cp_kv_cache_interleave_size) == (1, 1)


def _spec() -> MLAAttentionSpec:
    return MLAAttentionSpec(
        block_size=BLOCK_SIZE, num_kv_heads=1, head_size=8, dtype=torch.bfloat16
    )


def _cache() -> torch.Tensor:
    spec = _spec()
    return torch.zeros(
        NUM_BLOCKS,
        spec.num_heads,
        spec.num_states,
        spec.state_content_size_bytes,
        dtype=torch.uint8,
    )


def _worker(tp_rank: int, tp_size: int, caches: dict) -> MoRIIOConnectorWorker:
    worker = MoRIIOConnectorWorker.__new__(MoRIIOConnectorWorker)
    worker.tp_rank = tp_rank
    worker.world_size = tp_size
    worker.use_mla = True
    worker._has_mamba = False
    worker.model_config = SimpleNamespace(get_total_num_kv_heads=lambda: 1)
    worker.kv_caches = caches
    worker.layer_to_spec = {name: _spec() for name in caches}
    worker.layer_to_group = {name: 0 for name in caches}
    worker.kv_layer_mr_offset = {}
    worker._region_session_index = None
    worker.layer_name_to_local_kv_cache_metadata = {name: [b""] for name in caches}
    worker._record_slice_layouts(caches)
    worker.shutdown = lambda: None
    return worker


class _Session:
    def __init__(self, src: torch.Tensor, dst: torch.Tensor):
        self.src, self.dst = src.view(-1), dst.view(-1)


def _consumer_wrapper() -> MoRIIOWrapper:
    wrapper = MoRIIOWrapper.__new__(MoRIIOWrapper)
    wrapper.lock = threading.Lock()
    wrapper.done_write_cache_req_ids = []
    wrapper.failed_write_cache_req_ids = []
    wrapper._write_done_counts = {}
    wrapper._terminal_transfer_ids = OrderedDict()
    return wrapper


def _run_write(
    producer_tp: int,
    consumer_tp: int,
    dcp_size: int,
    num_prefill_blocks: int,
    decode_blocks: list[int],
):
    """Run every prefill rank's writer for all layers into fake decode ranks.

    Returns the decode caches, the per-rank write_done results, the prefill
    block ids, the global block contents and the bytes each prefill rank wrote.
    """
    gen = torch.Generator().manual_seed(0)
    content = {
        name: torch.randint(
            1, 255, (num_prefill_blocks, *_cache().shape[1:]), generator=gen
        ).to(torch.uint8)
        for name in LAYERS
    }
    # Prefill block ids are not in order, so a pairing by position shows up.
    prefill_blocks = [NUM_BLOCKS - 1 - j for j in range(num_prefill_blocks)]
    consumer_caches = [{name: _cache() for name in LAYERS} for _ in range(consumer_tp)]
    consumer_wrappers = [_consumer_wrapper() for _ in range(consumer_tp)]
    metadata = msgspec.msgpack.decode(
        msgspec.msgpack.encode(
            MoRIIOAgentMetadata(
                engine_id="decode",
                agent_metadata=b"",
                kv_caches_base_addr=[],
                num_blocks=NUM_BLOCKS,
                block_len=0,
                attn_backend_name="test",
                attention_block_sizes={name: BLOCK_SIZE for name in LAYERS},
                dcp_size=dcp_size,
                cp_kv_cache_interleave_size=BLOCK_SIZE,
            )
        ),
        type=MoRIIOAgentMetadata,
    )
    bytes_written = []
    done: list[set[str]] = [set() for _ in range(consumer_tp)]
    for p in range(producer_tp):
        caches = {name: _cache() for name in LAYERS}
        for name in LAYERS:
            caches[name][prefill_blocks] = content[name]
        worker = _worker(p, producer_tp, caches)
        mapping = worker._write_tp_mapping(consumer_tp)

        def get_built_session(engine_id, worker=worker, mapping=mapping):
            if engine_id == "decode_dp0":
                (target,) = mapping.targets
                tp_rank = target.tp_rank
            else:
                tp_rank = int(engine_id.rsplit("_tp", 1)[1])
            sessions = [
                _Session(worker.kv_caches[name], consumer_caches[tp_rank][name])
                for name in worker.kv_caches
            ]
            return sessions, metadata

        def write_remote_data(sizes, local_offsets, remote_offsets, session):
            for size, local, remote in zip(sizes, local_offsets, remote_offsets):
                session.dst[remote : remote + size] = session.src[local : local + size]
            return "status"

        def send_notify(transfer_id, ip, port, message_type, message_fields=None):
            consumer_wrappers[port - NOTIFY_PORT]._handle_structured_message(
                {
                    "type": message_type,
                    "transfer_id": transfer_id,
                    **(message_fields or {}),
                }
            )

        worker._get_built_session = get_built_session
        worker.moriio_wrapper = SimpleNamespace(
            done_remote_allocate_req_dict={},
            lock=threading.Lock(),
            done_req_ids=[],
            write_remote_data=write_remote_data,
            send_notify=send_notify,
            waiting_for_transfer_complete=lambda statuses: None,
            _mark_transfer_terminal_locked=lambda transfer_id: None,
            _is_transfer_terminal_locked=lambda transfer_id: False,
        )
        info = RemoteAllocInfo(block_ids=[list(decode_blocks)])
        worker.moriio_wrapper.done_remote_allocate_req_dict["t"] = info
        writer = make_moriio_writer(worker)
        for name in LAYERS:
            writer._execute_write_task(
                WriteTask(
                    request_id="req",
                    transfer_id="t",
                    dst_engine_id="decode",
                    local_block_ids=[prefill_blocks],
                    remote_block_ids_hint=None,
                    layer_name=name,
                    event=SimpleNamespace(synchronize=lambda: None),
                    remote_notify_port=NOTIFY_PORT,
                    remote_ip="decode",
                    remote_tp_size=consumer_tp,
                    tp_mapping=mapping,
                )
            )
        info.writes_expected = len(LAYERS)
        writer._finalize_if_complete("t", info)
        # Prefill frees its blocks whether or not it had anything to write.
        assert [ack.transfer_id for ack in worker.moriio_wrapper.done_req_ids] == ["t"]
        bytes_written.append(info.bytes_written)
    for rank_done, wrapper in zip(done, consumer_wrappers):
        rank_done |= wrapper.pop_finished_write_req_ids()
    return consumer_caches, done, content, bytes_written


@pytest.mark.parametrize("num_prefill_blocks", [1, 5, 9])
@pytest.mark.parametrize(
    ("producer_tp", "consumer_tp", "dcp_size"),
    [(4, 4, 4), (1, 4, 4), (8, 4, 4), (4, 4, 2), (1, 1, 1)],
)
def test_write_lands_each_block_on_its_dcp_rank(
    monkeypatch, producer_tp, consumer_tp, dcp_size, num_prefill_blocks
):
    monkeypatch.setattr(moriio_engine, "get_role", lambda: ROLE.CONSUMER)
    # One spare decode block, as when decode reserves lookahead slots.
    num_decode_blocks = cdiv(num_prefill_blocks, dcp_size) + 1
    decode_blocks = [2 * i + 1 for i in range(num_decode_blocks)]
    caches, done, content, bytes_written = _run_write(
        producer_tp, consumer_tp, dcp_size, num_prefill_blocks, decode_blocks
    )

    # Every decode rank completes, including ranks that own no block of a
    # short prompt.
    assert done == [{"t"}] * consumer_tp
    for t in range(consumer_tp):
        dcp_rank = t % dcp_size
        for name in LAYERS:
            expected = torch.zeros_like(caches[t][name])
            for i, block in enumerate(decode_blocks):
                j = i * dcp_size + dcp_rank
                if j < num_prefill_blocks:
                    expected[block] = content[name][j]
            assert torch.equal(caches[t][name], expected), (t, name)

    # Each block is written once per DCP group (consumer_tp // dcp_size
    # replicas), 1 / dcp_size of what the same TP without DCP writes.
    block_len = _cache()[0].numel()
    total = num_prefill_blocks * block_len * len(LAYERS)
    assert sum(bytes_written) == total * consumer_tp // dcp_size


def test_write_refuses_decode_allocation_short_of_its_blocks(monkeypatch):
    monkeypatch.setattr(moriio_engine, "get_role", lambda: ROLE.CONSUMER)
    with pytest.raises(ValueError, match="allocated 1"):
        _run_write(4, 4, 4, num_prefill_blocks=5, decode_blocks=[3])
