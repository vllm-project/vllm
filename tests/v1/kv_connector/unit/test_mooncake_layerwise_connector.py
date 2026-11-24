# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import contextlib
import ctypes
import queue
import time
from concurrent.futures import ThreadPoolExecutor
from threading import Event
from unittest.mock import AsyncMock, MagicMock, patch

import msgspec
import pytest
import torch
import zmq.asyncio
from huggingface_hub.utils import httpx

from tests.v1.kv_connector.unit.offloading_connector.utils import make_kv_cache_group
from vllm.config import ParallelConfig, set_current_vllm_config
from vllm.config.compilation import CUDAGraphMode
from vllm.distributed.kv_transfer.kv_connector.utils import KVOutputAggregator
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake import mooncake_connector
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.mooncake_connector import (
    KVConnectorRole,
    MooncakeConnector,
    MooncakeConnectorMetadata,
    MooncakeXferResponse,
    MooncakeXferResponseStatus,
    PullReqMeta,
)
from vllm.v1.core.single_type_kv_cache_manager import (
    CircularBufferManager,
    MambaManager,
    SlidingWindowManager,
)
from vllm.v1.kv_cache_interface import (
    AttentionSpec,
    KVCacheConfig,
    MambaSpec,
)
from vllm.v1.request import RequestStatus

from .test_mooncake_connector import (
    _BootstrapClientStub,
    _make_test_kv_cache_config,
    _make_worker_vllm_config,
    patch_worker_dependencies,
)
from .utils import create_model_runner_output, create_request, create_scheduler

pytestmark = pytest.mark.cpu_test


@contextlib.contextmanager
def _layerwise_copy_worker(
    *,
    extra=None,
    names=("model.layers.0.self_attn",),
    register=True,
    kv_role="kv_producer",
    writes=None,
    spec=None,
    manager_config=None,
    array_api=False,
    before_copy=None,
):
    config = _make_worker_vllm_config(
        kv_role, {"transfer_mode": "layerwise", **(extra or {})}
    )
    config.compilation_config.cudagraph_mode = CUDAGraphMode.NONE
    spec = spec or AttentionSpec(
        block_size=config.cache_config.block_size,
        num_kv_heads=1,
        head_size=8,
        dtype=torch.float16,
    )
    cache = KVCacheConfig(
        64, [], [make_kv_cache_group(list(names), spec, manager_config=manager_config)]
    )

    def write(_session, sources, destinations, lengths):
        if before_copy is not None:
            before_copy(sources)
        if writes is not None:
            writes.append(tuple(map(int, lengths)))
        for src, dst, length in zip(sources, destinations, lengths):
            ctypes.memmove(int(dst), int(src), int(length))
        return 0

    with (
        set_current_vllm_config(config),
        patch_worker_dependencies(),
        patch.object(mooncake_connector.httpx, "AsyncClient", _BootstrapClientStub),
        patch.object(torch.accelerator, "current_stream", return_value=None),
        patch.object(torch, "Event", return_value=MagicMock()),
        (
            patch.object(
                mooncake_connector.TransferEngine,
                "batch_transfer_sync_write_arrays",
                side_effect=write,
                create=True,
            )
            if array_api
            else contextlib.nullcontext()
        ),
    ):
        worker = MooncakeConnector(
            config, KVConnectorRole.WORKER, cache
        ).connector_worker
        try:
            if register:
                worker.register_kv_caches(
                    {
                        name: torch.zeros((64, spec.page_size_bytes), dtype=torch.uint8)
                        for name in names
                    }
                )
            with patch.object(
                worker.transfer_backend.engine, "batch_transfer_sync_write", write
            ):
                yield worker
        finally:
            worker.shutdown()


def _copy_destination_metadata(worker, destinations, requests):
    pulls = {
        req_id: PullReqMeta(
            d_req_id=req_id,
            transfer_id=req_id,
            local_block_ids=[list(blocks)],
            num_external_tokens=len(blocks) * worker.logical_block_size,
            remote_engine_id=worker.engine_id,
            remote_bootstrap_addr="http://bootstrap:8998",
        )
        for req_id, blocks in requests.items()
    }
    metadata = worker._build_pull_metadata(pulls, include_layout=True)
    return msgspec.structs.replace(
        metadata,
        remote_engine_id="decode-engine",
        kv_caches_base_addr=[tensor.data_ptr() for tensor in destinations],
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("num_peers", [1, 2])
async def test_layerwise_missing_layers_only_fail_the_affected_request(num_peers):
    names = ("model.layers.0.self_attn", "model.layers.1.self_attn")
    with _layerwise_copy_worker(names=names, kv_role="kv_consumer") as worker:
        pulls = {
            req_id: PullReqMeta(
                d_req_id=req_id,
                transfer_id=req_id,
                local_block_ids=[[block]],
                num_external_tokens=worker.logical_block_size,
                remote_engine_id="prefill-engine",
                remote_bootstrap_addr="http://bootstrap:8998",
                pull_tasks_count=num_peers,
            )
            for req_id, block in (("missing", 3), ("healthy", 4), ("pending", 5))
        }
        if num_peers == 2:
            worker.process_pulling_result(
                MooncakeXferResponse(
                    status=MooncakeXferResponseStatus.FINISH,
                    ok_reqs=list(pulls),
                    covered_layers=[names[0]],
                ),
                pulls.copy(),
            )
            assert all(not pull.failed for pull in pulls.values())
            assert not worker.finished_recving_reqs

        responses = [
            MooncakeXferResponse(
                status=MooncakeXferResponseStatus.CONTINUE,
                ok_reqs=["missing"],
                covered_layers=[names[0]],
            ),
            MooncakeXferResponse(
                status=MooncakeXferResponseStatus.CONTINUE,
                ok_reqs=["healthy"],
                covered_layers=list(names),
            ),
            MooncakeXferResponse(
                status=MooncakeXferResponseStatus.FINISH,
                ok_reqs=["pending"],
                covered_layers=list(names),
            ),
        ]
        sock = mooncake_connector.make_zmq_socket.return_value.__enter__.return_value
        sock.recv.side_effect = [worker._encoder.encode(reply) for reply in responses]
        await worker.receive_kv_from_single_worker("tcp://producer:1234", pulls)

        assert pulls["missing"].failed
        assert not pulls["healthy"].failed
        assert not pulls["pending"].failed
        assert all(pull.pull_tasks_count == 0 for pull in pulls.values())
        assert worker.finished_recving_reqs == set(pulls)
        assert worker.get_block_ids_with_load_errors() == {3}


@pytest.mark.asyncio
@pytest.mark.parametrize("source_blocks", [(1, 2), (2, 1), (1, 3)])
async def test_layerwise_batch_copies_each_request_and_layer(source_blocks):
    names = ("model.layers.0.self_attn", "model.layers.1.self_attn")
    with _layerwise_copy_worker(names=names) as worker:
        sources = [worker.device_kv_caches[name] for name in names]
        destinations = [torch.zeros_like(source) for source in sources]
        for layer, source in enumerate(sources):
            for block in range(1, 5):
                source[block].fill_(10 * layer + block)
        remote = _copy_destination_metadata(
            worker, destinations, {"a": (5, 6), "b": (7, 8)}
        )
        await worker.send_kv_to_decode(b"decode", AsyncMock(), remote)
        worker.process_decode_ready_requests()
        tasks = tuple(
            mooncake_connector.SendTask(
                req_id,
                worker.requests_to_send[req_id].destinations[remote.worker_key],
                name,
                (blocks,),
                (remote_blocks,),
                2 * worker.logical_block_size,
            )
            for req_id, blocks, remote_blocks in (
                ("a", source_blocks, (5, 6)),
                ("b", (3, 4), (7, 8)),
            )
            for name in names
        )
        result = worker._send_batch("decode:9000", tasks, torch.Event())
        assert len(result.progress) == len(tasks)
        for source, destination in zip(sources, destinations):
            assert torch.equal(destination[5:7], source[list(source_blocks)])
            assert torch.equal(destination[7:9], source[3:5])
            assert not torch.any(destination[:5])
            assert not torch.any(destination[9:])


@pytest.mark.asyncio
@pytest.mark.parametrize("interval", [1, 2, 4, 8])
@pytest.mark.parametrize("array_api", [False, True], ids=["lists", "arrays"])
async def test_layerwise_interval_batches_layers_and_flushes_each_chunk(
    interval, array_api
):
    """Batch real copies across layers/requests, including a partial final batch."""
    names = tuple(f"model.layers.{i}.self_attn" for i in range(5))
    writes: list[tuple[int, ...]] = []
    with _layerwise_copy_worker(
        names=names,
        extra={"layerwise_transfer_interval": interval},
        writes=writes,
        array_api=array_api,
    ) as worker:
        sources = [worker.device_kv_caches[name] for name in names]
        destinations = [torch.zeros_like(source) for source in sources]
        remote = _copy_destination_metadata(
            worker, destinations, {"a": (5, 6), "b": (7, 8)}
        )
        await worker.send_kv_to_decode(b"decode", AsyncMock(), remote)
        size = worker.logical_block_size
        for chunk in (1, 2):
            writes.clear()
            metadata = MooncakeConnectorMetadata()
            metadata.requests = tuple(
                mooncake_connector.PushReqMeta(
                    req, req, (blocks,), chunk * size, 2 * size
                )
                for req, blocks in (("a", (1, 2)), ("b", (3, 4)))
            )
            worker.prepare_kv_writes(metadata)
            worker.start_load_kv(metadata)
            for i, (name, source) in enumerate(zip(names, sources)):
                source[chunk].fill_(10 + i)
                source[chunk + 2].fill_(20 + i)
                worker.save_kv_layer(name)
                # A repeated hook must not count as another layer.
                if i == 0:
                    worker.save_kv_layer(name)
                worker._finish_pending_send(wait=True)
                assert len(writes) == (i + 1) // interval
                copied = ((i + 1) // interval) * interval
                for j, destination in enumerate(destinations):
                    expected = 10 + j if j < copied else 0
                    assert torch.all(destination[chunk + 4] == expected)
            worker.wait_for_save()
            worker._process_pending_transfers(wait=True)
            assert len(writes) == (len(names) + interval - 1) // interval
            for source, destination in zip(sources, destinations):
                assert torch.equal(destination[5 : 5 + chunk], source[1 : 1 + chunk])
                assert torch.equal(destination[7 : 7 + chunk], source[3 : 3 + chunk])
            assert sum(map(sum, writes)) == 2 * len(names) * sources[0][0].numel()


@pytest.mark.parametrize("interval", [0, -1, True, 1.5, "4"])
def test_layerwise_rejects_invalid_transfer_interval(interval):
    config = _make_worker_vllm_config(
        "kv_producer",
        {"transfer_mode": "layerwise", "layerwise_transfer_interval": interval},
    )
    config.compilation_config.cudagraph_mode = CUDAGraphMode.NONE
    with pytest.raises(ValueError, match="layerwise_transfer_interval"):
        MooncakeConnector(
            config, KVConnectorRole.SCHEDULER, _make_test_kv_cache_config()
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("prefix_delta", [-1, 0, 1])
@pytest.mark.parametrize("ready_before_start", [False, True])
@pytest.mark.parametrize("delayed_send", [False, True])
@pytest.mark.parametrize("windowed", [False, True])
async def test_layerwise_prefix_copy_waits_for_complete_blocks_and_destination(
    prefix_delta, ready_before_start, delayed_send, windowed
):
    manager_config = (
        SlidingWindowManager.config(sliding_window=17) if windowed else None
    )
    with _layerwise_copy_worker(manager_config=manager_config) as worker:
        name = worker.registered_layer_names[0]
        source = worker.device_kv_caches[name]
        destination = torch.zeros_like(source)
        source[1].fill_(7)
        remote = _copy_destination_metadata(worker, [destination], {"a": (3, 4)})
        socket = AsyncMock(spec=zmq.asyncio.Socket, send_multipart=AsyncMock())
        if ready_before_start:
            await worker.send_kv_to_decode(b"decode", socket, remote)
        block_size = worker.logical_block_size
        metadata = MooncakeConnectorMetadata()
        metadata.requests = (
            mooncake_connector.PushReqMeta(
                "a",
                "a",
                ((1, 2),),
                2 * block_size,
                2 * block_size,
                num_computed_tokens=block_size + prefix_delta,
            ),
        )
        worker.prepare_kv_writes(metadata)
        worker.start_load_kv(metadata)
        if not ready_before_start:
            assert not torch.any(destination)
            await worker.send_kv_to_decode(b"decode", socket, remote)
            worker._process_pending_transfers()
        worker._finish_pending_send(wait=True)
        expected_prefix = 7 if prefix_delta >= 0 and not windowed else 0
        assert torch.all(destination[3] == expected_prefix)
        assert not torch.any(destination[4])
        socket.send_multipart.assert_not_awaited()
        source[2].fill_(11)
        # An active prefill has no separate D-ready WRITE deadline.
        delay = mooncake_connector.envs.VLLM_MOONCAKE_ABORT_REQUEST_TIMEOUT + 61
        with patch.object(
            mooncake_connector.time,
            "monotonic",
            return_value=time.monotonic() + (delay if delayed_send else 0),
        ):
            worker.save_kv_layer(name)
            worker.wait_for_save()
            worker._process_pending_transfers(wait=True)
        assert torch.equal(destination[3:5], source[1:3])


@pytest.mark.asyncio
@pytest.mark.parametrize("recurrent", [False, True])
async def test_layerwise_state_copy_waits_for_final_kv_and_excludes_scratch(recurrent):
    spec = (
        MambaSpec(block_size=16, shapes=((8,),), dtypes=(torch.float16,))
        if recurrent
        else None
    )
    manager_config = (
        MambaManager.config(num_speculative_blocks=2)
        if recurrent
        else CircularBufferManager.config()
    )
    source_blocks = (1, 2, 3) if recurrent else (1,)
    destination_blocks = (4, 5, 6) if recurrent else (4,)
    with _layerwise_copy_worker(spec=spec, manager_config=manager_config) as worker:
        name = worker.registered_layer_names[0]
        source = worker.device_kv_caches[name]
        source[1].fill_(7)
        source[2:4].fill_(99)
        destination = torch.zeros_like(source)
        remote = _copy_destination_metadata(
            worker, [destination], {"a": destination_blocks}
        )
        socket = AsyncMock(spec=zmq.asyncio.Socket, send_multipart=AsyncMock())
        await worker.send_kv_to_decode(b"decode", socket, remote)
        size = worker.logical_block_size
        metadata = MooncakeConnectorMetadata()
        metadata.requests = (
            mooncake_connector.PushReqMeta(
                "a",
                "a",
                (source_blocks,),
                4 * size,
                4 * size,
                num_computed_tokens=3 * size,
            ),
        )
        worker.prepare_kv_writes(metadata)
        worker.start_load_kv(metadata)
        worker._finish_pending_send(wait=True)
        assert not torch.any(destination)
        source[1].fill_(11)
        worker.save_kv_layer(name)
        worker.wait_for_save()
        worker._process_pending_transfers(wait=True)
        assert torch.equal(destination[4], source[1])
        assert not torch.any(destination[5:7])


@pytest.mark.asyncio
async def test_layerwise_reuses_slices_across_hooks_and_refreshes_next_forward():
    names = ("model.layers.0.self_attn", "model.layers.1.self_attn")
    with _layerwise_copy_worker(names=names) as worker:
        sources = [worker.device_kv_caches[name] for name in names]
        destinations = [torch.zeros_like(source) for source in sources]
        for source in sources:
            source[1].fill_(7)
            source[2].fill_(9)
            source[5].fill_(11)
        remote = _copy_destination_metadata(worker, destinations, {"a": (3, 4)})
        await worker.send_kv_to_decode(b"decode", AsyncMock(), remote)
        size = worker.logical_block_size
        metadata = MooncakeConnectorMetadata()
        metadata.requests = (
            mooncake_connector.PushReqMeta("a", "a", ((1, 2),), size, 2 * size),
        )
        worker.prepare_kv_writes(metadata)
        worker.start_load_kv(metadata)
        worker.save_kv_layer(names[0])
        worker._finish_pending_send(wait=True)
        (first_slice,) = (
            value for value in worker.cached_block_slices.values() if value is not None
        )
        worker.save_kv_layer(names[1])
        worker._finish_pending_send(wait=True)
        (reused_slice,) = (
            value for value in worker.cached_block_slices.values() if value is not None
        )
        assert reused_slice is first_slice
        worker.wait_for_save()
        worker._process_pending_transfers(wait=True)
        assert not worker.cached_block_slices
        for destination in destinations:
            assert torch.all(destination[3] == 7)
            assert not torch.any(destination[4])

        # The next chunk uses a replacement block and must not reuse old IDs.
        metadata.requests = (
            mooncake_connector.PushReqMeta("a", "a", ((1, 5),), 2 * size, 2 * size),
        )
        worker.prepare_kv_writes(metadata)
        worker.start_load_kv(metadata)
        for name in names:
            worker.save_kv_layer(name)
        worker.wait_for_save()
        worker._process_pending_transfers(wait=True)
        assert not worker.cached_block_slices
        for destination in destinations:
            assert torch.all(destination[3] == 7)
            assert torch.all(destination[4] == 11)


@pytest.mark.asyncio
async def test_layerwise_ready_at_bootstrap_publication_copies_registered_cache():
    with _layerwise_copy_worker(register=False) as worker:
        group = worker.kv_cache_config.transfer_groups[0]
        name = group.layer_names[0]
        source = torch.full(
            (64, group.kv_cache_specs[name].page_size_bytes), 7, dtype=torch.uint8
        )
        destination = torch.zeros_like(source)
        replies: queue.Queue[tuple[bytes, bytes]] = queue.Queue()
        socket = AsyncMock(send_multipart=AsyncMock(side_effect=replies.put))

        async def publish(url, json):
            remote = _copy_destination_metadata(worker, [destination], {"a": (3,)})
            await worker.send_kv_to_decode(b"decode", socket, remote)
            return httpx.Response(200, request=httpx.Request("POST", url))

        with patch.object(_BootstrapClientStub, "post", side_effect=publish):
            worker.register_kv_caches({name: source})
        count = worker.logical_block_size
        metadata = MooncakeConnectorMetadata()
        metadata.requests = (
            mooncake_connector.PushReqMeta("a", "a", ((1,),), count, count),
        )
        metadata.reqs_to_send["a"] = ("a", [[1]])
        worker.prepare_kv_writes(metadata)
        worker.start_load_kv(metadata)
        worker.save_kv_layer(name)
        worker.wait_for_save()
        worker._process_pending_transfers(wait=True)
        assert worker.get_finished()[0] == {"a"}
        response = msgspec.msgpack.decode(
            replies.get(timeout=5)[1], type=MooncakeXferResponse
        )
        assert response.ok_reqs == ["a"]
        assert response.covered_layers == [name]
        assert torch.equal(destination[3], source[1])


@pytest.mark.parametrize("cancel", [False, True])
def test_layerwise_early_completion_releases_blocks_before_worker_retirement(cancel):
    """Early rank reports permit N+4 reuse without double-freeing cancellations."""
    config = _make_worker_vllm_config("kv_producer", {"transfer_mode": "layerwise"})
    config.parallel_config = ParallelConfig(tensor_parallel_size=2)
    config.scheduler_config.async_scheduling = True
    config.cache_config.enable_prefix_caching = False
    config.compilation_config.cudagraph_mode = CUDAGraphMode.NONE
    scheduler = create_scheduler(config, num_blocks=2)
    aggregator = KVOutputAggregator(config.parallel_config.world_size)
    size = config.cache_config.block_size
    request = create_request(num_tokens=size, max_tokens=1, do_remote_decode=True)
    request.kv_transfer_params["transfer_id"] = request.request_id
    scheduler.add_request(request)
    first = scheduler.schedule()  # N+1: final chunk.
    original_blocks = scheduler.kv_cache_manager.get_block_ids(request.request_id)
    waiting = create_request(num_tokens=size, max_tokens=1)
    scheduler.add_request(waiting)
    second = scheduler.schedule()  # N+2 precedes consumption of N+1.
    assert waiting.request_id not in second.num_scheduled_tokens
    if cancel:
        scheduler.finish_requests(request.request_id, RequestStatus.FINISHED_ABORTED)
    scheduler.update_from_output(
        first,
        aggregator.aggregate(
            [
                create_model_runner_output(
                    [request], finished_sending={request.request_id}
                ),
                create_model_runner_output([request]),
            ]
        ),
    )
    if not cancel:
        assert request.request_id in scheduler.requests  # One rank is insufficient.
    third = scheduler.schedule()  # N+3 carries the normal retirement metadata.
    if not cancel:
        assert request.request_id in third.kv_connector_metadata.reqs_to_send
        assert waiting.request_id not in third.num_scheduled_tokens
    # The existing executor aggregator joins reports arriving in different steps.
    scheduler.update_from_output(
        second,
        aggregator.aggregate(
            [
                create_model_runner_output([]),
                create_model_runner_output([], finished_sending={request.request_id}),
            ]
        ),
    )
    assert request.request_id not in scheduler.requests
    fourth = scheduler.schedule()
    if not cancel:
        assert waiting.request_id in fourth.num_scheduled_tokens
        assert scheduler.kv_cache_manager.get_block_ids(waiting.request_id) == (
            original_blocks
        )
    scheduler.update_from_output(
        third,
        create_model_runner_output(
            [waiting] if waiting.request_id in third.num_scheduled_tokens else [],
            finished_sending={request.request_id},
        ),
    )  # Late retirement must not free the same request a second time.


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("warm_prefix", "write_status"), [(False, 0), (True, 0), (False, -1)]
)
@pytest.mark.parametrize("interval", [1, 4])
async def test_layerwise_finished_prefill_drains_late_ready(
    warm_prefix, write_status, interval
):
    names = ("model.layers.0.self_attn", "model.layers.1.self_attn")
    with _layerwise_copy_worker(
        names=names, extra={"layerwise_transfer_interval": interval}
    ) as worker:
        config = worker.vllm_config
        config.scheduler_config.async_scheduling = False
        scheduler = create_scheduler(
            config, num_blocks=64, kv_cache_config=worker.kv_cache_config
        )
        size = worker.logical_block_size
        if warm_prefix:
            warm = create_request(
                num_tokens=2 * size,
                common_prefix_len=2 * size,
                max_tokens=1,
                block_size=size,
            )
            scheduler.add_request(warm)
            scheduled = scheduler.schedule()
            scheduler.update_from_output(scheduled, create_model_runner_output([warm]))
        request = create_request(
            num_tokens=2 * size,
            common_prefix_len=2 * size,
            max_tokens=1,
            block_size=size,
            do_remote_decode=True,
        )
        request.kv_transfer_params["transfer_id"] = request.request_id
        scheduler.add_request(request)
        scheduled = scheduler.schedule()
        metadata = scheduled.kv_connector_metadata
        (snapshot,) = metadata.requests
        assert snapshot.num_computed_tokens == (size if warm_prefix else 0)
        blocks = list(snapshot.block_ids[0])
        sources = [worker.device_kv_caches[name] for name in names]
        destinations = [torch.zeros_like(source) for source in sources]
        for layer, source in enumerate(sources):
            source[blocks] = layer + 7
        worker.prepare_kv_writes(metadata)
        worker.start_load_kv(metadata)
        for name in names:
            worker.save_kv_layer(name)
        worker.wait_for_save()
        worker._process_pending_transfers(wait=True)
        scheduler.update_from_output(scheduled, create_model_runner_output([request]))
        assert scheduler.has_requests()
        empty = scheduler.schedule()
        assert empty.total_num_scheduled_tokens == 0
        worker.prepare_kv_writes(empty.kv_connector_metadata)
        worker.start_load_kv(empty.kv_connector_metadata)
        assert worker.get_finished()[0] is None
        scheduler.update_from_output(empty, create_model_runner_output([]))
        assert scheduler.has_requests()

        replies: queue.Queue[tuple[bytes, bytes]] = queue.Queue()
        socket = AsyncMock(send_multipart=AsyncMock(side_effect=replies.put))
        remote = _copy_destination_metadata(
            worker, destinations, {request.request_id: (3, 4)}
        )
        await worker.send_kv_to_decode(b"decode", socket, remote)
        empty = scheduler.schedule()
        assert empty.total_num_scheduled_tokens == 0
        with (
            (
                patch.object(
                    worker.transfer_backend.engine,
                    "batch_transfer_sync_write",
                    return_value=write_status,
                )
                if write_status
                else contextlib.nullcontext()
            ),
            (
                pytest.raises(RuntimeError, match="DMA drain is unconfirmed")
                if write_status
                else contextlib.nullcontext()
            ),
        ):
            worker.prepare_kv_writes(empty.kv_connector_metadata)
            worker.start_load_kv(empty.kv_connector_metadata)
            worker._process_pending_transfers(wait=True)
        if write_status:
            assert replies.empty()
            assert request.request_id in worker.requests_to_send
            assert scheduler.has_requests()
            assert all(not torch.any(dest) for dest in destinations)
            return
        finished, _ = worker.get_finished()
        assert finished == {request.request_id}
        response = msgspec.msgpack.decode(
            replies.get(timeout=5)[1], type=MooncakeXferResponse
        )
        assert response.status == MooncakeXferResponseStatus.FINISH
        assert response.ok_reqs == [request.request_id]
        assert response.covered_layers == list(names)
        for source, destination in zip(sources, destinations):
            assert torch.equal(destination[3:5], source[blocks])
        scheduler.update_from_output(
            empty, create_model_runner_output([], finished_sending=finished)
        )
        assert not scheduler.has_requests()


@pytest.mark.parametrize(
    ("role", "graph", "error"),
    [
        ("kv_both", CUDAGraphMode.NONE, "fixed kv_role"),
        ("kv_producer", CUDAGraphMode.FULL, "cannot use FULL"),
    ],
)
def test_layerwise_rejects_unsupported_execution_mode(role, graph, error):
    config = _make_worker_vllm_config(role, {"transfer_mode": "layerwise"})
    config.compilation_config.cudagraph_mode = graph
    with pytest.raises(ValueError, match=error):
        MooncakeConnector(
            config, KVConnectorRole.SCHEDULER, _make_test_kv_cache_config()
        )


@pytest.mark.asyncio
async def test_layerwise_rejects_request_protocol_before_copy():
    with _layerwise_copy_worker() as worker:
        source = next(iter(worker.device_kv_caches.values()))
        destination = torch.zeros_like(source)
        remote = _copy_destination_metadata(worker, [destination], {"a": (3,)})
        remote = msgspec.structs.replace(remote, protocol_version=0)
        with pytest.raises(ValueError, match="same transfer_mode"):
            await worker.send_kv_to_decode(b"decode", AsyncMock(), remote)
        assert not torch.any(destination)


@pytest.mark.asyncio
@pytest.mark.parametrize("complete_before_seal", [False, True])
async def test_layerwise_ack_without_another_model_thread_poll(complete_before_seal):
    entered, release, acknowledged = Event(), Event(), Event()
    responses = []

    def before_copy(sources):
        entered.set()
        assert release.wait(10)

    async def receive_ack(parts):
        responses.append(msgspec.msgpack.decode(parts[1], type=MooncakeXferResponse))
        acknowledged.set()

    with _layerwise_copy_worker(before_copy=before_copy) as worker:
        name = worker.registered_layer_names[0]
        source = worker.device_kv_caches[name]
        source[1].fill_(11)
        destination = torch.zeros_like(source)
        remote = _copy_destination_metadata(worker, [destination], {"a": (3,)})
        sock = AsyncMock()
        sock.send_multipart.side_effect = receive_ack
        await worker.send_kv_to_decode(b"decode", sock, remote)
        size = worker.logical_block_size
        metadata = MooncakeConnectorMetadata()
        metadata.requests = (
            mooncake_connector.PushReqMeta("a", "a", ((1,),), size, size),
        )
        worker.prepare_kv_writes(metadata)
        worker.start_load_kv(metadata)
        try:
            worker.save_kv_layer(name)
            assert entered.wait(10)
            if complete_before_seal:
                release.set()
                for _, future in worker._pending_sends:
                    future.result(timeout=10)
                assert not acknowledged.is_set()
            worker.wait_for_save()
            if not complete_before_seal:
                assert not acknowledged.is_set()
                release.set()
            # No get_finished(), next step, or subsequent layer hook.
            assert acknowledged.wait(10)
            assert len(responses) == 1
            assert responses[0].ok_reqs == ["a"]
            assert responses[0].err_reqs is None
            assert torch.all(destination[3] == 11)
        finally:
            release.set()


@pytest.mark.asyncio
async def test_async_layerwise_completes_unrelated_request_without_duplicate_sends():
    entered, release = Event(), Event()
    blocked_sources = []
    names = ("model.layers.0.self_attn", "model.layers.1.self_attn")

    def before_copy(sources):
        if int(sources[0]) == blocked_address:
            blocked_sources.append(int(sources[0]))
            entered.set()
            assert release.wait(10), "test did not release the transport"

    with _layerwise_copy_worker(
        names=names,
        before_copy=before_copy,
    ) as worker:
        sources = [worker.device_kv_caches[name] for name in names]
        blocked_address = sources[0][1].data_ptr()
        destinations = [torch.zeros_like(source) for source in sources]
        for source in sources:
            source[1].fill_(11)
            source[2].fill_(22)
        remote = _copy_destination_metadata(
            worker, destinations, {"a": (3,), "b": (4,)}
        )
        await worker.send_kv_to_decode(b"decode", AsyncMock(), remote)
        size = worker.logical_block_size
        try:
            for req_id, block in (("a", 1), ("b", 2)):
                metadata = MooncakeConnectorMetadata()
                metadata.requests = (
                    mooncake_connector.PushReqMeta(
                        req_id, req_id, ((block,),), size, size
                    ),
                )
                worker.prepare_kv_writes(metadata)
                worker.start_load_kv(metadata)
                worker.save_kv_layer(names[0])
                if req_id == "a":
                    assert entered.wait(10)
                    # Repeating a hook must not enqueue the same source range.
                    worker.save_kv_layer(names[0])
                worker.save_kv_layer(names[1])
                worker.wait_for_save()
            # Await only b's actual transport results, leaving a blocked.
            for tasks, future in tuple(worker._pending_sends):
                if any(task.transfer_id == "b" for task in tasks):
                    future.result(timeout=10)
            assert worker.get_finished()[0] == {"b"}
            assert worker.get_finished()[0] is None
            metadata = MooncakeConnectorMetadata()
            metadata.reqs_to_send = {"a": ("a", [[1]]), "b": ("b", [[2]])}
            metadata.finished_req_ids = {"a", "b"}
            worker.prepare_kv_writes(metadata)
            worker.start_load_kv(metadata)
            assert worker.get_finished()[0] is None
            assert "b" not in worker.requests_to_send
            assert "a" in worker.requests_to_send
            assert blocked_sources == [blocked_address]
            assert not torch.any(destinations[0][3])
            for destination in destinations:
                assert torch.all(destination[4] == 22)
        finally:
            release.set()
        worker._process_pending_transfers(wait=True)
        assert worker.get_finished()[0] == {"a"}
        for destination in destinations:
            assert torch.all(destination[3] == 11)


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel_notice", ["reqs_not_processed", "finished_req_ids"])
async def test_async_layerwise_cancellation_drains_recurrent_send_before_reuse(
    cancel_notice,
):
    entered, release, fence_entered, overwritten = Event(), Event(), Event(), Event()

    def before_copy(sources):
        entered.set()
        assert release.wait(10), "test did not release the transport"

    spec = MambaSpec(block_size=16, shapes=((8,),), dtypes=(torch.float16,))
    with _layerwise_copy_worker(
        spec=spec,
        manager_config=MambaManager.config(),
        before_copy=before_copy,
    ) as worker:
        name = worker.registered_layer_names[0]
        source = worker.device_kv_caches[name]
        source[1].fill_(11)
        destination = torch.zeros_like(source)
        remote = _copy_destination_metadata(worker, [destination], {"a": (3,)})
        await worker.send_kv_to_decode(b"decode", AsyncMock(), remote)
        size = worker.logical_block_size
        metadata = MooncakeConnectorMetadata()
        metadata.requests = (
            mooncake_connector.PushReqMeta("a", "a", ((1,),), size, 2 * size),
        )
        worker.prepare_kv_writes(metadata)
        worker.start_load_kv(metadata)
        worker.save_kv_layer(name)
        worker.wait_for_save()
        assert not worker._pending_sends  # No intermediate Mamba state transfer.
        metadata = MooncakeConnectorMetadata()
        metadata.requests = (
            mooncake_connector.PushReqMeta(
                "a", "a", ((1,),), 2 * size, 2 * size, num_computed_tokens=size
            ),
        )
        worker.prepare_kv_writes(metadata)
        worker.start_load_kv(metadata)
        worker.save_kv_layer(name)
        worker.wait_for_save()
        next_step = MooncakeConnectorMetadata()
        getattr(next_step, cancel_notice).add("a")

        def overwrite():
            fence_entered.set()
            worker.prepare_kv_writes(next_step)
            source[1].fill_(99)
            overwritten.set()

        with ThreadPoolExecutor(max_workers=1) as executor:
            try:
                assert entered.wait(10)
                future = executor.submit(overwrite)
                assert fence_entered.wait(10)
                assert not overwritten.wait(0.1)
            finally:
                release.set()
            future.result(timeout=10)
        assert torch.all(destination[3] == 11)
        assert torch.all(source[1] == 99)
        worker.start_load_kv(next_step)
        assert "a" not in worker.requests_to_send


def test_async_layerwise_requires_early_hook_and_rejects_preemption():
    with _layerwise_copy_worker() as worker:
        metadata = MooncakeConnectorMetadata()
        with pytest.raises(RuntimeError, match="runner pre-write hook"):
            worker.start_load_kv(metadata)
        metadata.preempted_req_ids = {"a"}
        with pytest.raises(RuntimeError, match="preemption recovery"):
            worker.prepare_kv_writes(metadata)
