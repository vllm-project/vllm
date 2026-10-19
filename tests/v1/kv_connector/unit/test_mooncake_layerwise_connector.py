# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exercise layerwise scheduling, executor handoff, and completion without RDMA."""

import asyncio
from collections import OrderedDict
from concurrent.futures import Future, ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import msgspec
import pytest
import torch

from vllm.distributed.kv_transfer.kv_connector.v1.mooncake import (
    mooncake_layerwise_connector as lw,
)
from vllm.v1.core.sched.output import KVConnectorBlockState, SchedulerOutput
from vllm.v1.kv_cache_interface import UniformTypeKVCacheSpecs


def _metadata(reqs=None, *, rank=0, tp=1, width=32):
    return lw.MooncakeLayerwiseXferMetadata(
        remote_hostname="decode",
        remote_port=9000,
        remote_tp_size=tp,
        remote_tp_rank=rank,
        req_blocks=reqs or {"d": ("transfer", [[9, 10]])},
        kv_caches_base_addr=[10000, 20000],
        block_lens=[width, width],
        kv_block_lens=[width, width],
        registered_layer_names=["layer0", "layer1"],
        registered_layer_indices=[0, 1],
        registered_group_indices=[0, 0],
        logical_block_size=16,
        kernel_block_size=16,
    )


@pytest.fixture
def worker():
    w = lw.MooncakeLayerwiseWorker.__new__(lw.MooncakeLayerwiseWorker)
    w.shutdown = MagicMock()
    w.is_kv_producer = True
    w.requests_to_send = {}
    w._step_transfer_ids = ()
    w._send_future = None
    w._ready_event = None
    w._ready_layer_index = None
    w._retired = {}
    w._logical_block_size = w.block_size = 16
    w._physical_blocks_per_logical_kv_block = 1
    w.tp_size = 1
    w.tp_rank = 0
    w.use_mla = False
    w.kv_cache_config = SimpleNamespace(
        transfer_groups=[SimpleNamespace(kv_cache_spec=object())],
        has_mamba_layers=False,
    )
    w.registered_layer_names = ["layer0", "layer1"]
    w.registered_layer_indices = [0, 1]
    w._layer_index_by_name = dict(
        zip(w.registered_layer_names, w.registered_layer_indices)
    )
    w._layer_indices = (0, 1)
    w.transfer_topo = SimpleNamespace(
        handshake_target_ranks=lambda size: range(size),
        total_num_kv_heads=8,
        local_replicates_kv_cache=False,
    )
    w._layout_by_peer = OrderedDict()
    w._encoder = msgspec.msgpack.Encoder()
    w._control_channel = lw.LayerwiseControler(w)
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
            return_value=(*_regions(meta), None),
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
        num_scheduled_tokens=tokens,
        is_final_chunk=final,
    )
    worker.requests_to_send[state.transfer_id] = state
    worker._step_transfer_ids = (state.transfer_id,)
    return state


def test_scheduler_snapshots_updated_blocks():
    scheduler = lw.MooncakeLayerwiseScheduler.__new__(lw.MooncakeLayerwiseScheduler)
    scheduler.is_kv_producer = True
    scheduler._is_hma_required = False
    scheduler.kv_cache_config = SimpleNamespace(transfer_group_ids=(0,))
    request = SimpleNamespace(
        num_computed_tokens=0,
        num_prompt_tokens=17,
        kv_transfer_params={"transfer_id": "transfer"},
    )
    scheduler._prefills = {"p": request}
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
    assert not first.requests[0].is_final_chunk
    assert second.requests[0].block_ids == ((3, 4),)
    assert second.requests[0].is_final_chunk
    assert output.has_sync_kv_loads


def test_start_load_updates_state_in_place_without_clearing_send_progress(worker):
    state = _state(worker, tokens=16, final=False)
    state.sent_tokens[0, 0] = 16
    meta = lw.MooncakeLayerwiseMetadata()
    meta.requests = (
        lw.LayerwisePushReqMeta("p", "transfer", ((3, 4),), 16, 1, True, 17),
    )
    worker.start_load_kv(meta)
    assert worker.requests_to_send["transfer"] is state
    assert state.num_computed_tokens == 16
    assert state.block_ids == ((3, 4),)
    assert state.sent_tokens[0, 0] == 16


def test_submission_copies_ranges_and_defers_regions_to_executor(worker):
    state = _state(worker)
    state.destinations[0] = "d", _metadata()
    state.available_tokens[0] = 17
    worker._ready_event = MagicMock()
    pending = Future()
    worker._sender_executor = MagicMock()
    worker._sender_executor.submit.return_value = pending
    worker._control_channel.get_transfer_regions = MagicMock()
    worker._submit_available_layers()
    worker._control_channel.get_transfer_regions.assert_not_called()
    task = worker._sender_executor.submit.call_args.args[1][0]
    state.block_ids = ((99,),)
    state.available_tokens.clear()
    assert task.block_ids == (3, 4)
    assert task.layers == ((0, 0, 2),)
    worker._submit_available_layers()
    worker._sender_executor.submit.assert_called_once()


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
        worker.save_layer("layer0")
    worker._collect_send_results(wait=True)
    assert state.sent_tokens == {(0, 0): expected}
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
    assert worker.get_finished() == ({"p"}, None)
    worker._send_blocks.assert_called_once()
    worker._control_channel.schedule_completion_ack.assert_called_once_with(
        "transfer", 0, None
    )
    assert "transfer" not in worker.requests_to_send


def test_executor_aggregates_requests_and_layers_by_remote_worker(worker):
    meta = _metadata({"d1": ("a", [[9]]), "d2": ("b", [[10]])})
    _ready(worker, meta)
    event = MagicMock()
    tasks = tuple(
        lw.LayerwiseSendTask(
            tid,
            f"p{tid}",
            did,
            meta,
            (block,),
            0,
            ((0, 0, 1), (1, 0, 1)),
        )
        for tid, did, block in [("a", "d1", 3), ("b", "d2", 4)]
    )
    result = worker._sender_executor.submit(worker._send_batch, tasks, event).result()
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
    worker._control_channel.get_transfer_regions = MagicMock(
        return_value={
            i: ([p], [d]) for i, (p, d) in enumerate(zip(*_regions(meta, 32 // p_tp)))
        }
    )
    task = lw.LayerwiseSendTask("transfer", "p", "d", meta, (3,), 0, ((0, 0, 1),))
    result = worker._sender_executor.submit(
        worker._send_batch, (task,), MagicMock()
    ).result()
    assert not result.errors
    worker._send_blocks.assert_called_once_with("decode:9000", [src], [dst], [16])


def test_write_failure_does_not_advance_sent_tokens(worker):
    state = _state(worker)
    _ready(worker, _metadata())
    worker._send_blocks.return_value = -1
    with patch.object(worker, "_record_event", return_value=MagicMock()):
        worker.save_layer("layer0")
    worker._collect_send_results(wait=True)
    assert state.sent_tokens == {}
    assert "WRITE returned -1" in state.error
    worker._send_blocks.reset_mock()
    worker._progress_sends(wait=True)
    worker._send_blocks.assert_not_called()


def test_ack_batch_shares_pending_and_finishes_only_after_last_reply(worker):
    async def run():
        meta = _metadata({"d1": ("a", [[9]]), "d2": ("b", [[10]])})
        sock = AsyncMock()
        with patch.object(
            worker._control_channel,
            "_prepare_transfer_regions",
            return_value=(*_regions(meta), None),
        ):
            await worker._control_channel.receive_decode_ready(b"id", sock, meta)
        channel = worker._control_channel
        batch = channel._ack_contexts["a"][0]
        assert channel._ack_contexts["b"][0] is batch
        assert list(channel.drain_decode_ready()) == [meta]
        await channel._send_completion_ack("a", 0, None)
        assert batch.pending == {"b": "d2"}
        await channel._send_completion_ack("b", 0, None)
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
        assert list(worker._control_channel.drain_decode_ready()) == []

    asyncio.run(run())


def test_timeout_releases_finished_request_but_keeps_active_request(
    worker, monkeypatch
):
    state = _state(worker)
    state.deadline = 1
    monkeypatch.setattr(lw.time, "monotonic", lambda: 2)
    assert worker.get_finished() == (None, None)
    assert "transfer" in worker.requests_to_send
    state.inference_finished = True
    assert worker.get_finished() == ({"p"}, None)
    assert "transfer" not in worker.requests_to_send


@pytest.mark.parametrize("ready_after_hook", [False, True])
def test_attention_hook_sends_same_layer_indexer_before_forward_end(
    worker, ready_after_hook
):
    """One attention hook fences both caches; different page widths stay intact."""
    state = _state(worker, tokens=16, final=False)
    names = ["model.layers.7.self_attn.attn", "model.layers.7.self_attn.indexer"]
    worker._layer_index_by_name = dict.fromkeys(names, 7)
    worker._layer_indices = (7,)
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
            return_value=(local, remote, None),
        ):
            asyncio.run(
                worker._control_channel.receive_decode_ready(
                    b"id", AsyncMock(), _metadata()
                )
            )

    if not ready_after_hook:
        ready()
    event = MagicMock()
    with patch.object(worker, "_record_event", return_value=event):
        worker.save_layer(names[0])
    if ready_after_hook:
        ready()
    worker._progress_sends(wait=True)
    assert not state.sealed
    assert state.sent_tokens == {(0, 7): 16}
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
