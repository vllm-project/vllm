# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exercise layerwise scheduling, executor handoff, and completion without RDMA."""

import asyncio
import time
from collections import OrderedDict
from concurrent.futures import Future, ThreadPoolExecutor
from threading import Barrier
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, PropertyMock, patch

import msgspec
import pytest
import torch

from vllm.distributed.kv_transfer.kv_connector.v1.mooncake import (
    mooncake_layerwise_connector as lw,
)
from vllm.v1.core.sched.output import KVConnectorBlockState, SchedulerOutput
from vllm.v1.kv_cache_interface import (
    KVCacheConfig,
    KVCacheGroupSpec,
    UniformTypeKVCacheSpecs,
)


def test_partial_pp_alignment_preserves_matching_cache_names():
    first = lw.TransferRegion("model.layers.0.attn", 0, 1024, 64, 64)
    second = lw.TransferRegion("model.layers.1.attn", 1, 2048, 64, 64)
    remote = lw.TransferRegion("model.layers.1.attn", 1, 4096, 64, 64)
    local, target, error = lw._align_transfer_regions(
        [first, second], [remote], allow_partial_layers=True
    )
    assert error is None
    assert local == [second]
    assert target == [remote]


def _metadata(reqs=None, *, rank=0, tp=1, width=32):
    requests = reqs or {"d": ("transfer", [[9, 10]])}
    return lw.MooncakeLayerwiseXferMetadata(
        remote_engine_id="decode-engine",
        layout_id=1,
        remote_hostname="decode",
        remote_port=9000,
        remote_tp_size=tp,
        remote_tp_rank=rank,
        req_blocks=requests,
        kv_caches_base_addr=[10000, 20000],
        block_lens=[width, width],
        kv_block_lens=[width, width],
        registered_layer_names=["layer0", "layer1"],
        registered_layer_indices=[0, 1],
        registered_group_indices=[0, 0],
        logical_block_size=16,
        kernel_block_size=16,
        recv_timeout_s=3600,
    )


@pytest.fixture
def worker():
    w = lw.MooncakeLayerwiseWorker.__new__(lw.MooncakeLayerwiseWorker)
    w.shutdown = MagicMock()
    w.is_kv_producer = True
    w.requests_to_send = {}
    w.prefilling_transfer_ids = ()
    w._pending_sends = []
    w.send_timeout_margin_s = 10
    w._ready_event = None
    w.retired_transfer_expiry_times = {}
    w.logical_block_size = w.block_size = 16
    w._physical_blocks_per_logical_kv_block = 1
    w._kernel_block_ratios = (1,)
    w.tp_size = 1
    w.tp_rank = 0
    w.use_mla = False
    spec = lw.FullAttentionSpec(
        block_size=16, num_kv_heads=1, head_size=8, dtype=torch.float16
    )
    w.kv_cache_config = KVCacheConfig(
        num_blocks=100,
        kv_cache_tensors=[],
        kv_cache_groups=[KVCacheGroupSpec(["layer0", "layer1"], spec)],
    )
    w._layer_specs = dict.fromkeys(["layer0", "layer1"], spec)
    w.vllm_config = SimpleNamespace(
        parallel_config=SimpleNamespace(prefill_context_parallel_size=1)
    )
    w.registered_layer_names = ["layer0", "layer1"]
    w.registered_layer_indices = [0, 1]
    w._layer_index_by_name = dict(
        zip(w.registered_layer_names, w.registered_layer_indices)
    )
    w.transfer_topo = SimpleNamespace(
        handshake_target_ranks=lambda size: range(size),
        total_num_kv_heads=8,
        local_replicates_kv_cache=False,
    )
    w._layout_by_peer = OrderedDict()
    w._encoder = msgspec.msgpack.Encoder()
    w._control_channel = lw.LayerwiseController(w)
    w._control_channel.schedule_completion_ack = MagicMock()
    w._send_blocks = MagicMock(return_value=0)
    with ThreadPoolExecutor(max_workers=1) as executor:
        w._sender_executor = executor
        yield w


def _regions(meta, local_width=32):
    return (
        [
            lw.TransferRegion(f"layer{i}", i, (i + 1) * 1000, local_width, local_width)
            for i in range(2)
        ],
        [
            lw.TransferRegion(
                f"layer{i}",
                i,
                (i + 1) * 10000,
                meta.block_lens[i],
                meta.kv_block_lens[i],
            )
            for i in range(2)
        ],
    )


def _ready(worker, meta):
    async def receive():
        with patch.object(
            worker._control_channel,
            "_prepare_transfer_regions",
            return_value=_regions(meta),
        ):
            await worker._control_channel.receive_decode_ready(
                b"identity", AsyncMock(), meta
            )

    asyncio.run(receive())


def _state(worker, *, tokens=17, final=True):
    state = lw.RequestState(
        "transfer",
        req_id="p",
        block_ids=((3, 4),),
        num_prompt_tokens=17,
        scheduled_end_tokens=tokens if final else min(tokens, 16),
    )
    worker.requests_to_send[state.transfer_id] = state
    worker.prefilling_transfer_ids = (state.transfer_id,)
    return state


@pytest.mark.parametrize("kernel_block_size", [8, 16])
def test_ready_during_bootstrap_publication_uses_initialized_layout(
    worker, kernel_block_size
):
    worker.block_size = kernel_block_size
    worker.kv_caches_base_addr = [1000, 2000]
    worker.block_len_per_layer = worker.kv_block_len_per_layer = [32, 32]
    worker.registered_group_indices = [0, 0]
    worker.region_shared_groups = [(0,), (0,)]
    worker.region_row_offsets = [-1, -1]
    worker.opaque_packed_storages = set()
    worker._layer_index_by_name = {}
    worker._cache_specs = {}
    worker._local_transfer_regions = []
    meta = msgspec.structs.replace(
        _metadata(),
        kernel_block_size=kernel_block_size,
        cache_specs={
            name: worker._spec_signature(name) for name in worker._layer_specs
        },
    )

    async def publish(_worker):
        # A peer can send ready as soon as bootstrap exposes this worker.
        await worker._control_channel.receive_decode_ready(b"ready", AsyncMock(), meta)

    with patch.object(
        lw.MooncakeConnectorWorker, "register_worker_with_bootstrap", publish
    ):
        asyncio.run(worker.register_worker_with_bootstrap())
    state = _state(worker)
    state.available_tokens = {0: 17, 1: 17}
    worker._ready_event = MagicMock()
    # Registration fixes the ratio; sends must not inspect specs to derive it again.
    with patch.object(worker, "_physical_blocks_for_spec", side_effect=AssertionError):
        worker._process_pending_transfers(wait=True)
    assert state.error is None
    assert state.sent_tokens == {
        (meta.worker_key, name): 32 for name in ("layer0", "layer1")
    }
    ratio = 16 // kernel_block_size
    worker._send_blocks.assert_called_once_with(
        "decode:9000",
        [1000 + 96 * ratio, 2000 + 96 * ratio],
        [10000 + 288 * ratio, 20000 + 288 * ratio],
        [64 * ratio, 64 * ratio],
    )


def test_scheduler_snapshots_updated_blocks():
    scheduler = lw.MooncakeLayerwiseScheduler.__new__(lw.MooncakeLayerwiseScheduler)
    scheduler.is_kv_producer = True
    scheduler._is_hma_required = False
    scheduler.kv_cache_config = KVCacheConfig(
        num_blocks=100,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(
                ["layer0"],
                lw.FullAttentionSpec(
                    block_size=16, num_kv_heads=1, head_size=8, dtype=torch.float16
                ),
            )
        ],
    )
    request = SimpleNamespace(
        num_computed_tokens=0,
        num_preemptions=0,
        num_prompt_tokens=17,
        kv_transfer_params={"transfer_id": "transfer"},
    )
    scheduler._prefill_requests = {"p": request}
    output = SchedulerOutput.make_empty()
    output.num_scheduled_tokens = {"p": 16}
    blocks = ([3],)
    output.kv_connector_block_state = KVConnectorBlockState({"p"}, lambda _: blocks, {})
    with patch.object(
        lw.MooncakeConnectorScheduler,
        "build_connector_meta",
        return_value=lw.MooncakeConnectorMetadata(),
    ):
        first = scheduler.build_connector_meta(output)
        request.num_computed_tokens = 16
        output.num_scheduled_tokens = {"p": 1}
        blocks[0].append(4)
        second = scheduler.build_connector_meta(output)
    assert first.requests[0].block_ids == ((3,),)
    assert first.requests[0].scheduled_end_tokens == 16
    assert second.requests[0].block_ids == ((3, 4),)
    assert second.requests[0].scheduled_end_tokens == 17
    assert output.has_sync_kv_loads


def test_start_load_updates_state_in_place_without_clearing_send_progress(worker):
    state = _state(worker, tokens=16, final=False)
    state.sent_tokens[_metadata().worker_key, "layer0"] = 16
    meta = lw.MooncakeLayerwiseMetadata()
    meta.requests = (lw.LayerwisePushReqMeta("p", "transfer", ((3, 4),), 17, 17),)
    worker.start_load_kv(meta)
    assert worker.requests_to_send["transfer"] is state
    assert state.scheduled_end_tokens == 17
    assert state.is_final_chunk
    assert state.block_ids == ((3, 4),)
    assert state.sent_tokens[_metadata().worker_key, "layer0"] == 16


def test_submission_copies_ranges_and_retains_destination_layout(worker):
    state = _state(worker)
    _ready(worker, _metadata())
    worker.process_decode_ready_requests()
    state.available_tokens[0] = 17
    worker._ready_event = MagicMock()
    pending = Future()
    worker._sender_executor = MagicMock()
    worker._sender_executor.submit.return_value = pending
    worker._submit_available_layers()
    layer_transfer_task = worker._sender_executor.submit.call_args.args[2][0]
    assert layer_transfer_task.destination is state.destinations[_metadata().worker_key]
    state.block_ids = ((99,),)
    state.available_tokens.clear()
    assert layer_transfer_task.block_ids == ((3, 4),)
    assert layer_transfer_task.kv_cache_name == "layer0"
    assert layer_transfer_task.remote_block_ids == ((9, 10),)
    assert layer_transfer_task.end_tokens == 32
    worker._submit_available_layers()
    worker._sender_executor.submit.assert_called_once()


@pytest.mark.parametrize("with_healthy_request", [False, True])
def test_planning_failure_discards_earlier_layers_but_keeps_other_requests(
    worker, with_healthy_request
):
    failed = _state(worker)
    failed.block_ids = ((3,),)
    # Layer 0 can send one block; layer 1 needs a missing second source block.
    failed.available_tokens = {0: 16, 1: 17}
    requests = {"d": ("transfer", [[9, 10]])}
    if with_healthy_request:
        healthy = lw.RequestState(
            "healthy",
            req_id="healthy-p",
            block_ids=((5, 6),),
            num_prompt_tokens=17,
            scheduled_end_tokens=17,
            available_tokens={0: 17, 1: 17},
        )
        worker.requests_to_send[healthy.transfer_id] = healthy
        requests["healthy-d"] = ("healthy", [[11, 12]])
    meta = _metadata(requests)
    _ready(worker, meta)
    worker._ready_event = MagicMock()
    worker._process_pending_transfers(wait=True)

    assert failed.error == "Incomplete source/destination block mapping"
    assert failed.sent_tokens == {}
    if with_healthy_request:
        assert healthy.error is None
        assert healthy.sent_tokens == {
            (meta.worker_key, name): 32 for name in ("layer0", "layer1")
        }
        worker._send_blocks.assert_called_once_with(
            "decode:9000", [1160, 2160], [10352, 20352], [64, 64]
        )
    else:
        worker._send_blocks.assert_not_called()
    assert not worker._pending_sends


@pytest.mark.parametrize("final, expected", [(False, 16), (True, 32)])
def test_layer_send_waits_device_event_and_handles_partial_block(
    worker, final, expected
):
    state = _state(worker, final=final)
    _ready(worker, _metadata())
    event = MagicMock()
    order = []
    event.synchronize.side_effect = lambda: order.append("event")
    worker._send_blocks.side_effect = lambda *args: order.append("write") or 0
    with patch.object(worker, "_record_event", return_value=event):
        worker.save_kv_layer("layer0")
    worker._finish_pending_send(wait=True)
    assert state.sent_tokens == {(_metadata().worker_key, "layer0"): expected}
    assert order == ["event", "write"]
    worker._control_channel.schedule_completion_ack.assert_not_called()


def test_late_ready_flushes_all_layers_without_another_forward(worker):
    state = _state(worker)
    event = MagicMock()
    with patch.object(worker, "_record_event", return_value=event):
        worker.wait_for_save()
    worker._send_blocks.assert_not_called()
    state.inference_finished = True
    _ready(worker, _metadata())
    # Completion polling must not start work after this step's send barrier.
    assert worker.get_finished() == (None, None)
    worker._send_blocks.assert_not_called()
    worker.start_load_kv(lw.MooncakeLayerwiseMetadata())
    # Empty steps finish transfers in start_load_kv; wait_for_save is a no-op.
    with patch.object(worker, "_process_pending_transfers") as process:
        worker.wait_for_save()
    process.assert_not_called()
    assert worker.get_finished() == ({"p"}, None)
    worker._send_blocks.assert_called_once()
    worker._control_channel.schedule_completion_ack.assert_called_once_with(
        "transfer", _metadata().worker_key, None
    )
    assert "transfer" not in worker.requests_to_send


def test_wait_for_save_drains_hook_sends_and_remaining_layers(worker):
    state = _state(worker)
    state.inference_finished = True
    meta = _metadata()
    _ready(worker, meta)
    with patch.object(worker, "_record_event", return_value=MagicMock()):
        worker.save_kv_layer("layer0")
        assert worker._pending_sends
        # Speculative runners can poll before the final send barrier.
        assert worker.get_finished() == (None, None)
        worker._control_channel.schedule_completion_ack.assert_not_called()
        worker.wait_for_save()
    assert not worker._pending_sends
    assert worker.prefilling_transfer_ids == ()
    assert state.sent_tokens == {
        (meta.worker_key, name): 32 for name in ("layer0", "layer1")
    }
    assert [call.args[1] for call in worker._send_blocks.call_args_list] == [
        [1096],
        [2096],
    ]
    assert worker.get_finished() == ({"p"}, None)


def test_current_lower_layer_does_not_delay_previously_ready_layers(worker):
    state = _state(worker)
    state.sealed = True
    state.available_tokens[1] = 17
    worker.requests_to_send["new"] = lw.RequestState(
        "new", req_id="new-p", scheduled_end_tokens=16, num_prompt_tokens=17
    )
    worker.prefilling_transfer_ids = ("new",)
    meta = _metadata()
    _ready(worker, meta)
    with patch.object(worker, "_record_event", return_value=MagicMock()):
        worker.save_kv_layer("layer0")
    worker._finish_pending_send(wait=True)
    assert state.sent_tokens == {(meta.worker_key, "layer1"): 32}
    worker._send_blocks.assert_called_once_with("decode:9000", [2096], [20288], [64])


@pytest.mark.parametrize("pcp_rank", [0, 1])
def test_only_sending_pcp_replica_retains_abort_tombstones(worker, pcp_rank):
    _state(worker)
    metadata = lw.MooncakeLayerwiseMetadata()
    metadata.reqs_not_processed = {"transfer"}
    with patch.object(
        lw.MooncakeLayerwiseWorker,
        "pcp_rank",
        new_callable=PropertyMock,
        return_value=pcp_rank,
    ):
        worker.start_load_kv(metadata)
        assert "transfer" not in worker.requests_to_send
        assert bool(worker.retired_transfer_expiry_times) == (pcp_rank == 0)
        assert worker.get_finished() == (None, None)


@pytest.mark.parametrize("empty_blocks", [False, True])
def test_destination_without_data_needs_no_final_layer_or_send_window(
    worker, empty_blocks
):
    state = _state(worker, tokens=16, final=False)
    state.available_tokens = {0: 16, 1: 16}
    meta = msgspec.structs.replace(
        _metadata({"d": ("transfer", [])} if empty_blocks else None), recv_timeout_s=0
    )
    with patch.object(
        worker._control_channel,
        "_prepare_transfer_regions",
        return_value=_regions(meta) if empty_blocks else ([], []),
    ):
        asyncio.run(
            worker._control_channel.receive_decode_ready(b"id", AsyncMock(), meta)
        )
    worker.start_load_kv(lw.MooncakeLayerwiseMetadata())
    worker.wait_for_save()
    assert worker.get_finished() == (None, None)
    assert not state.sealed
    assert state.error is None
    assert state.ack_scheduled_workers == {meta.worker_key}
    worker._control_channel.schedule_completion_ack.assert_called_once_with(
        "transfer", meta.worker_key, None
    )
    worker._send_blocks.assert_not_called()


def test_executor_sends_pre_grouped_requests_and_layers(worker):
    meta = _metadata({"d1": ("a", [[9]]), "d2": ("b", [[10]])})
    _ready(worker, meta)
    worker.process_decode_ready_requests()
    event = MagicMock()
    tasks = tuple(
        lw.LayerwiseSendTask(
            tid,
            f"p{tid}",
            worker.requests_to_send[tid].destinations[meta.worker_key],
            f"layer{layer}",
            ((block,),),
            ((remote,),),
            16,
        )
        for tid, block, remote in [("a", 3, 9), ("b", 4, 10)]
        for layer in range(2)
    )
    result = worker._sender_executor.submit(
        worker._send_batch, "decode:9000", tasks, event
    ).result()
    assert not result.errors
    assert len(result.progress) == 4
    worker._send_blocks.assert_called_once()
    assert worker._send_blocks.call_args.args == (
        "decode:9000",
        [1096, 2096, 1128, 2128],
        [10288, 20288, 10320, 20320],
        [32, 32, 32, 32],
    )
    event.synchronize.assert_called_once()


@pytest.mark.parametrize(
    "p_tp,d_tp,p_rank,d_rank,src,dst",
    [
        (1, 2, 0, 0, 1096, 10144),
        (1, 2, 0, 1, 1112, 10144),
        (2, 1, 0, 0, 1048, 10288),
        (2, 1, 1, 0, 1048, 10304),
    ],
)
def test_executor_reuses_asymmetric_tp_address_plan(
    worker, p_tp, d_tp, p_rank, d_rank, src, dst
):
    worker.tp_size, worker.tp_rank = p_tp, p_rank
    meta = _metadata({"d": ("transfer", [[9]])}, rank=d_rank, tp=d_tp, width=32 // d_tp)
    layout = lw.LayerwisePeerLayout(
        {p.layer_name: ([p], [d]) for p, d in zip(*_regions(meta, 32 // p_tp))},
        {"layer0": 0, "layer1": 0},
        {0: ["layer0"], 1: ["layer1"]},
        ["layer0", "layer1"],
        frozenset({meta.worker_key}),
    )
    layer_transfer_task = lw.LayerwiseSendTask(
        "transfer",
        "p",
        lw.Destination(
            "d",
            meta,
            layout,
            time.monotonic() + meta.recv_timeout_s,
            ("layer0", "layer1"),
        ),
        "layer0",
        ((3,),),
        ((9,),),
        16,
    )
    result = worker._sender_executor.submit(
        worker._send_batch,
        "decode:9000",
        (layer_transfer_task,),
        MagicMock(),
    ).result()
    assert not result.errors
    worker._send_blocks.assert_called_once_with("decode:9000", [src], [dst], [16])


def test_write_failure_does_not_advance_sent_tokens(worker):
    state = _state(worker)
    _ready(worker, _metadata())
    worker._send_blocks.return_value = -1
    with patch.object(worker, "_record_event", return_value=MagicMock()):
        worker.save_kv_layer("layer0")
    worker._finish_pending_send(wait=True)
    assert state.sent_tokens == {}
    assert "WRITE returned -1" in state.error
    worker._send_blocks.reset_mock()
    worker._process_pending_transfers(wait=True)
    worker._send_blocks.assert_not_called()


def test_ack_batch_shares_pending_and_finishes_only_after_last_reply(worker):
    async def run():
        meta = _metadata({"d1": ("a", [[9]]), "d2": ("b", [])})
        sock = AsyncMock()
        with patch.object(
            worker._control_channel,
            "_prepare_transfer_regions",
            return_value=_regions(meta),
        ):
            await worker._control_channel.receive_decode_ready(b"id", sock, meta)
        channel = worker._control_channel
        batch = channel._ack_contexts["a", meta.worker_key]
        assert channel._ack_contexts["b", meta.worker_key] is batch
        destinations = list(channel.get_decode_ready_requests())
        assert [destination.remote_req_id for destination in destinations] == [
            "d1",
            "d2",
        ]
        assert all(
            destination.meta is meta and destination.layout is batch.layout
            for destination in destinations
        )
        assert [destination.required_layers for destination in destinations] == [
            ("layer0", "layer1"),
            (),
        ]
        await channel._send_completion_ack("a", meta.worker_key, None)
        assert batch.pending == {"b": "d2"}
        await channel._send_completion_ack("b", meta.worker_key, None)
        decoder = msgspec.msgpack.Decoder(lw.MooncakeXferResponse)
        replies = [
            decoder.decode(call.args[0][1])
            for call in sock.send_multipart.call_args_list
        ]
        assert [reply.status for reply in replies] == [
            lw.MooncakeXferResponseStatus.CONTINUE,
            lw.MooncakeXferResponseStatus.FINISH,
        ]
        assert [reply.ok_reqs for reply in replies] == [["d1"], ["d2"]]
        assert channel._ack_contexts == {}

    asyncio.run(run())


def test_missing_layout_requests_resend_without_publishing_ready(worker):
    async def run():
        meta = msgspec.structs.replace(
            _metadata(), layout_id=77, kv_caches_base_addr=[]
        )
        sock = AsyncMock()
        await worker._control_channel.receive_decode_ready(b"id", sock, meta)
        response = msgspec.msgpack.decode(
            sock.send_multipart.call_args.args[0][1], type=lw.MooncakeXferResponse
        )
        assert response.status == lw.MooncakeXferResponseStatus.LAYOUT_MISS
        assert worker._control_channel._ack_contexts == {}
        assert list(worker._control_channel.get_decode_ready_requests()) == []

    asyncio.run(run())


def test_cached_completion_requires_all_workers_after_pcp_size_changes(worker):
    first = msgspec.structs.replace(
        _metadata({"first": ("first-transfer", [])}, tp=2), remote_pp_size=2
    )
    _ready(worker, first)
    meta = msgspec.structs.replace(
        first, remote_pcp_size=2, req_blocks={"d": ("transfer", [])}
    )
    _ready(worker, meta)
    worker.process_decode_ready_requests()
    old_layout = (
        worker.requests_to_send["first-transfer"].destinations[first.worker_key].layout
    )
    state = worker.requests_to_send["transfer"]
    layout = state.destinations[meta.worker_key].layout
    assert layout is not old_layout
    assert len(old_layout.expected_workers) == 4
    assert len(layout.expected_workers) == 8
    assert not worker._finish_destinations(state)
    for _, tp, pp, pcp in sorted(layout.expected_workers - {meta.worker_key}):
        _ready(
            worker,
            msgspec.structs.replace(
                meta, remote_tp_rank=tp, remote_pp_rank=pp, remote_pcp_rank=pcp
            ),
        )
    worker.process_decode_ready_requests()
    assert worker._finish_destinations(state)
    assert state.ack_scheduled_workers == layout.expected_workers


@pytest.mark.parametrize("remote_group", [0, 1])
def test_blocks_only_ready_reuses_layout_and_survives_eviction(worker, remote_group):
    meta = _metadata({"first": ("first-transfer", [[]] * remote_group + [[7, 8]])})
    meta = msgspec.structs.replace(
        meta, registered_group_indices=[remote_group, remote_group]
    )
    _ready(worker, meta)
    channel = worker._control_channel
    worker.process_decode_ready_requests()
    first_layout = (
        worker.requests_to_send["first-transfer"].destinations[meta.worker_key].layout
    )
    blocks_only = msgspec.structs.replace(
        meta,
        req_blocks={"d": ("transfer", [[]] * remote_group + [[9, 10]])},
        kv_caches_base_addr=[],
        block_lens=[],
        kv_block_lens=[],
        registered_layer_names=[],
        registered_layer_indices=[],
        registered_group_indices=[],
    )
    asyncio.run(channel.receive_decode_ready(b"second", AsyncMock(), blocks_only))
    channel._layouts.clear()
    state = _state(worker)
    worker.process_decode_ready_requests()
    assert state.destinations[meta.worker_key].layout is first_layout
    state.available_tokens[0] = 17
    layer_transfer_task = worker._build_layer_transfer_task(
        state, state.destinations[meta.worker_key], "layer0"
    )
    assert layer_transfer_task.block_ids == ((3, 4),)
    assert layer_transfer_task.remote_block_ids == ((9, 10),)
    assert layer_transfer_task.end_tokens == 32
    assert layer_transfer_task.destination.layout is first_layout
    # Execution must use the task's layout, independent of the controller lifetime.
    channel._ack_contexts.clear()
    result = worker._send_batch("decode:9000", (layer_transfer_task,), MagicMock())
    assert not result.errors
    worker._send_blocks.assert_called_once_with("decode:9000", [1096], [10288], [64])


def test_timeout_releases_finished_request_but_keeps_active_request(
    worker, monkeypatch
):
    state = _state(worker)
    state.deadline = 1
    monkeypatch.setattr(lw.time, "monotonic", lambda: 2)
    with patch.object(worker, "_record_event", return_value=MagicMock()):
        worker.wait_for_save()
    assert worker.get_finished() == (None, None)
    assert "transfer" in worker.requests_to_send
    state.inference_finished = True
    assert worker.get_finished() == ({"p"}, None)
    assert "transfer" not in worker.requests_to_send


def test_ready_advertises_d_receive_duration_without_a_timestamp(worker, monkeypatch):
    worker.hostname, worker.rpc_port = "decode", 9000
    worker.engine_id = "decode-engine"
    worker.pp_size, worker.pp_rank = 1, 0
    worker._kv_layout_id = 1
    monkeypatch.setattr(lw.envs, "VLLM_MOONCAKE_ABORT_REQUEST_TIMEOUT", 30)
    for wall_clock in (-1e10, 1e10):
        monkeypatch.setattr(lw.time, "time", lambda value=wall_clock: value)
        meta = worker._build_pull_metadata({}, include_layout=False)
        decoded = msgspec.msgpack.decode(
            msgspec.msgpack.encode(meta), type=lw.MooncakeLayerwiseXferMetadata
        )
        assert decoded.recv_timeout_s == 90


@pytest.mark.parametrize(
    "margin, remaining, expired",
    [(10, 11, False), (10, 10, True), (10, -1, True), (20, 15, True)],
)
def test_d_receive_deadline_controls_submission(
    worker, monkeypatch, margin, remaining, expired
):
    now = 1000
    monkeypatch.setattr(lw.time, "monotonic", lambda: now)
    monkeypatch.setattr(lw.time, "time", lambda: -1e10)
    worker.send_timeout_margin_s = margin
    state = _state(worker)
    state.sealed = state.inference_finished = True
    state.available_tokens = {0: 17, 1: 17}
    worker._ready_event = MagicMock()
    meta = msgspec.structs.replace(_metadata(), recv_timeout_s=30)
    _ready(worker, meta)
    # Time spent in the model-thread queue must consume the same deadline.
    now = 1030 - remaining

    worker.start_load_kv(lw.MooncakeLayerwiseMetadata())
    worker.wait_for_save()
    assert worker.get_finished() == ({"p"}, None)
    assert state.destinations[meta.worker_key].send_deadline == 1030
    assert (state.error is not None) == expired
    assert bool(state.sent_tokens) != expired
    assert worker._send_blocks.call_count == (0 if expired else 1)
    assert worker._ready_event.synchronize.call_count == (0 if expired else 1)
    worker._control_channel.schedule_completion_ack.assert_called_once_with(
        "transfer", meta.worker_key, state.error
    )


@pytest.mark.parametrize("second_timeout", [20, 30])
def test_device_wait_cancels_only_expired_requests_in_batch(
    worker, monkeypatch, second_timeout
):
    now = 1000
    monkeypatch.setattr(lw.time, "monotonic", lambda: now)
    for tid, remote_block, timeout in (("a", 9, 20), ("b", 10, second_timeout)):
        meta = msgspec.structs.replace(
            _metadata({f"d{tid}": (tid, [[remote_block]])}),
            recv_timeout_s=timeout,
        )
        _ready(worker, meta)
    worker.process_decode_ready_requests()
    tasks = tuple(
        lw.LayerwiseSendTask(
            tid,
            f"p{tid}",
            worker.requests_to_send[tid].destinations[meta.worker_key],
            f"layer{layer}",
            ((local_block,),),
            ((remote_block,),),
            16,
        )
        for layer in (0, 1)
        for tid, local_block, remote_block in (("a", 3, 9), ("b", 4, 10))
    )

    def finish_device_wait():
        nonlocal now
        now = 1010

    event = MagicMock()
    event.synchronize.side_effect = finish_device_wait
    result = worker._send_batch("decode:9000", tasks, event)
    expired_requests = ("a", "b") if second_timeout == 20 else ("a",)
    assert result.errors == dict.fromkeys(
        expired_requests, "D receive deadline too close for WRITE"
    )
    if second_timeout == 20:
        assert result.progress == []
        worker._send_blocks.assert_not_called()
    else:
        assert result.progress == [
            ("b", meta.worker_key, f"layer{layer}", 16) for layer in (0, 1)
        ]
        worker._send_blocks.assert_called_once_with(
            "decode:9000", [1128, 2128], [10320, 20320], [32, 32]
        )
    completed = Future()
    completed.set_result(result)
    worker._pending_sends = [(tasks, completed)]
    worker._finish_pending_send(wait=True)
    assert worker.requests_to_send["a"].sent_tokens == {}
    assert worker.requests_to_send["a"].error is not None
    assert (worker.requests_to_send["b"].error is not None) == (second_timeout == 20)
    assert worker.requests_to_send["b"].sent_tokens == (
        {}
        if second_timeout == 20
        else {(meta.worker_key, f"layer{i}"): 16 for i in (0, 1)}
    )


@pytest.mark.parametrize("ready_after_hook", [False, True])
def test_attention_hook_sends_same_layer_indexer_before_forward_end(
    worker, ready_after_hook
):
    """One attention hook fences both caches; different page widths stay intact."""
    state = _state(worker, tokens=16, final=False)
    names = ["model.layers.7.self_attn.attn", "model.layers.7.self_attn.indexer"]
    worker._layer_index_by_name = dict.fromkeys(names, 7)
    spec = next(iter(worker._layer_specs.values()))
    worker._layer_specs = dict.fromkeys(names, spec)
    worker.kv_cache_config = KVCacheConfig(
        num_blocks=100,
        kv_cache_tensors=[],
        kv_cache_groups=[KVCacheGroupSpec(names, spec)],
    )
    meta = msgspec.structs.replace(
        _metadata(), registered_layer_names=names, registered_layer_indices=[7, 7]
    )
    local = [
        lw.TransferRegion(names[0], 7, 1000, 32, 32),
        lw.TransferRegion(names[1], 7, 2000, 8, 8),
    ]
    remote = [
        lw.TransferRegion(names[0], 7, 10000, 32, 32),
        lw.TransferRegion(names[1], 7, 20000, 8, 8),
    ]

    def ready():
        with patch.object(
            worker._control_channel,
            "_prepare_transfer_regions",
            return_value=(local, remote),
        ):
            asyncio.run(
                worker._control_channel.receive_decode_ready(b"id", AsyncMock(), meta)
            )

    if not ready_after_hook:
        ready()
    event = MagicMock()
    with patch.object(worker, "_record_event", return_value=event):
        worker.save_kv_layer(names[0])
    if ready_after_hook:
        ready()
    worker._process_pending_transfers(wait=True)
    assert not state.sealed
    assert state.sent_tokens == {(meta.worker_key, name): 16 for name in names}
    worker._send_blocks.assert_called_once_with(
        "decode:9000", [1096, 2024], [10288, 20072], [32, 8]
    )
    event.synchronize.assert_called_once()
    worker._control_channel.schedule_completion_ack.assert_not_called()


@pytest.mark.parametrize("uniform", [False, True])
def test_connector_accepts_full_attention_subclasses_and_uniform_specs(uniform):
    class IndexerSpec(lw.FullAttentionSpec):
        pass

    specs = {
        name: cls(block_size=16, num_kv_heads=1, head_size=width, dtype=torch.float16)
        for name, cls, width in [
            ("model.layers.0.attn", lw.FullAttentionSpec, 32),
            ("model.layers.0.indexer", IndexerSpec, 8),
        ]
    }
    spec = (
        UniformTypeKVCacheSpecs(block_size=16, kv_cache_specs=specs)
        if uniform
        else specs["model.layers.0.indexer"]
    )
    config = SimpleNamespace(
        kv_transfer_config=SimpleNamespace(
            kv_role="kv_producer", is_kv_producer=True, engine_id="p"
        ),
        compilation_config=SimpleNamespace(
            cudagraph_mode=SimpleNamespace(has_full_cudagraphs=lambda: False)
        ),
        parallel_config=SimpleNamespace(
            pipeline_parallel_size=1,
            decode_context_parallel_size=1,
            prefill_context_parallel_size=1,
        ),
    )
    with (
        patch.object(lw.KVConnectorBase_V1, "__init__", return_value=None),
        patch.object(lw, "MooncakeLayerwiseScheduler"),
    ):
        lw.MooncakeLayerwiseConnector(
            config,
            lw.KVConnectorRole.SCHEDULER,
            SimpleNamespace(transfer_groups=[SimpleNamespace(kv_cache_spec=spec)]),
        )


def test_worker_identity_distinguishes_tp_pp_and_pcp():
    meta = _metadata()
    peers = [
        msgspec.structs.replace(
            meta,
            remote_tp_rank=tp,
            remote_tp_size=2,
            remote_pp_rank=pp,
            remote_pp_size=2,
            remote_pcp_rank=pcp,
            remote_pcp_size=4,
        )
        for tp in range(2)
        for pp in range(2)
        for pcp in range(4)
    ]
    assert len({peer.worker_key for peer in peers}) == len(peers)


def test_destination_writes_run_concurrently(worker):
    state = _state(worker)
    first = _metadata()
    second = msgspec.structs.replace(first, remote_port=9001, remote_pp_rank=1)
    _ready(worker, first)
    _ready(worker, second)
    worker.process_decode_ready_requests()
    state.available_tokens[0] = state.scheduled_end_tokens
    worker._ready_event = MagicMock()
    both_writing = Barrier(2)

    def send(session, src, dst, sizes):
        # Neither endpoint can return until both WRITE calls have started.
        both_writing.wait(timeout=5)
        return 0

    worker._send_blocks.side_effect = send
    with ThreadPoolExecutor(max_workers=2) as executor:
        worker._sender_executor = executor
        worker._submit_available_layers()
        assert len(worker._pending_sends) == 2
        worker._finish_pending_send(wait=True)
    assert state.error is None
    assert {call.args[0] for call in worker._send_blocks.call_args_list} == {
        "decode:9000",
        "decode:9001",
    }
    assert not worker._pending_sends


@pytest.mark.parametrize("same_request", [False, True])
def test_failed_destination_waits_for_other_pending_sends(worker, same_request):
    state = _state(worker)
    meta = _metadata()
    _ready(worker, meta)
    worker.process_decode_ready_requests()
    layer_transfer_task = lw.LayerwiseSendTask(
        "transfer",
        "p",
        state.destinations[meta.worker_key],
        "layer0",
        ((3,),),
        ((9,),),
        16,
    )
    failed = Future()
    failed.set_exception(RuntimeError("WRITE failed"))
    pending = Future()
    other_state = state
    other_task = layer_transfer_task
    if not same_request:
        other_state = lw.RequestState("healthy", req_id="healthy-p")
        worker.requests_to_send[other_state.transfer_id] = other_state
        other_task = lw.replace(
            layer_transfer_task, transfer_id="healthy", req_id="healthy-p"
        )
    worker._pending_sends = [
        ((layer_transfer_task,), failed),
        ((other_task,), pending),
    ]
    worker._finish_pending_send()
    assert len(worker._pending_sends) == 2
    assert state.error is None
    pending.set_result(
        lw.LayerwiseSendResult(
            progress=[]
            if same_request
            else [("healthy", meta.worker_key, "layer0", 16)]
        )
    )
    worker._finish_pending_send()
    assert state.error == "WRITE failed"
    assert state.sent_tokens == {}
    if not same_request:
        assert other_state.error is None
        assert other_state.sent_tokens == {(meta.worker_key, "layer0"): 16}
    assert not worker._pending_sends
