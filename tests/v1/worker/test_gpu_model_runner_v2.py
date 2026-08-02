# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import gc
import math
import multiprocessing
import tempfile
from contextlib import suppress
from dataclasses import replace
from pathlib import Path

import pytest
import torch
import torch.distributed as dist

from vllm.distributed.parallel_state import GroupCoordinator
from vllm.utils.network_utils import get_open_port
from vllm.v1.core.kv_cache_utils import (
    _sparse_mla_fixed_hbm_bytes,
    _sparse_mla_manifest,
)
from vllm.v1.kv_cache_interface import SparseMLAOffloadMemoryPlan
from vllm.v1.worker.gpu.sparse_mla_offload import SparseMLAOffloadManager

_MAIN_LAYERS = ("main.0", "main.1")
_INDEXER_LAYERS = ("indexer.0", "indexer.1")
_PER_LAYER_BUFFERS = frozenset(
    [
        "resident_main_kv",
        "resident_logical_ids",
        "resident_last_access",
        "resident_generation",
        "newest_main_kv",
        "newest_logical_ids",
        "newest_generation",
        "provisional_slots",
    ]
)


def _make_sparse_mla_plan(tp_size: int, dp_size: int) -> SparseMLAOffloadMemoryPlan:
    manifest = _sparse_mla_manifest(
        num_blocks=2,
        block_size=2,
        main_layer_count=2,
        indexer_layer_count=2,
        main_width=4,
        indexer_width=2,
        main_dtype=torch.float32,
        indexer_dtype=torch.float32,
        max_num_seqs=2,
        resident_rows=2,
        topk=1,
        newest_width=2,
        request_block_width=2,
        local_query_heads=1,
        value_head_dim=2,
    )
    fixed_bytes = _sparse_mla_fixed_hbm_bytes(manifest)
    indexer_bytes = 2 * 2 * 2 * 2 * torch.float32.itemsize
    return SparseMLAOffloadMemoryPlan(
        num_blocks=2,
        feasible_num_blocks=2,
        main_layer_names=_MAIN_LAYERS,
        indexer_layer_names=_INDEXER_LAYERS,
        host_pool_bytes_per_dp_replica=8192,
        worker_available_hbm_bytes_per_tp_rank=fixed_bytes + indexer_bytes,
        fixed_offload_hbm_bytes_per_tp_rank=fixed_bytes,
        effective_available_hbm_bytes_per_tp_rank=indexer_bytes,
        device_bytes_per_tp_rank=fixed_bytes + indexer_bytes,
        num_dp_replicas=dp_size,
        tensor_parallel_size=tp_size,
        manifest=manifest,
    )


def _backing_path(name: str) -> Path:
    root = Path("/dev/shm")
    if not root.exists():
        root = Path(tempfile.gettempdir())
    return root / name


def _sparse_mla_rank(
    rank: int,
    world_size: int,
    tp_size: int,
    dp_size: int,
    port: int,
    case: str,
    results: multiprocessing.Queue,
) -> None:
    tp_group = None
    manager = None
    cuda_patch = pytest.MonkeyPatch()
    try:
        dist.init_process_group(
            backend="gloo",
            init_method=f"tcp://127.0.0.1:{port}",
            world_size=world_size,
            rank=rank,
        )
        rows = [list(range(i, i + tp_size)) for i in range(0, world_size, tp_size)]
        tp_group = GroupCoordinator(
            group_ranks=rows,
            local_rank=0,
            torch_distributed_backend="gloo",
            use_device_communicator=False,
            use_message_queue_broadcaster=False,
            group_name=f"sparse-mla-{case}",
        )
        dist.barrier()
        tp_group.device = torch.device("cpu")
        plan = _make_sparse_mla_plan(tp_size, dp_size)
        inventory = {
            name: torch.arange(8, dtype=torch.float32).view(2, 2, 2)
            for name in _INDEXER_LAYERS
        }
        invalid_host_errors = []
        if case == "local":
            host_entry = plan.manifest[0]
            for shape in (host_entry.shape[:2], (host_entry.shape[0], 0, 4)):
                manifest = (replace(host_entry, shape=shape), *plan.manifest[1:])
                invalid_plan = replace(plan, manifest=manifest)
                try:
                    invalid = SparseMLAOffloadManager.create_for_tp_group(
                        invalid_plan, tp_group, inventory
                    )
                except RuntimeError as error:
                    invalid_host_errors.append(str(error))
                else:
                    invalid.close()
                    invalid.unlink()
                    invalid_host_errors.append("")

            def fail_cuda_entrypoint(*args, **kwargs):
                raise AssertionError("CPU Manager entered the CUDA runtime")

            cuda_patch.setattr(torch.cuda, "is_available", lambda: True)
            cuda_patch.setattr(torch.cuda, "cudart", fail_cuda_entrypoint)
            cuda_patch.setattr(torch.cuda, "Stream", fail_cuda_entrypoint)
            cuda_patch.setattr(torch.cuda, "Event", fail_cuda_entrypoint)
        manager = SparseMLAOffloadManager.create_for_tp_group(plan, tp_group, inventory)
        handle = manager.pool_handle
        if case == "visibility":
            if tp_group.rank_in_group == 0:
                manager.main_host_write_view("main.0").fill_(rank + 17)
            dist.barrier(group=tp_group.cpu_group)
            sentinel = manager.layer_view("main.0").main_host_kv.flatten()[0].item()
            malformed_result = None
            if tp_group.rank_in_group != 0:
                other_nonce = "0" * 32
                if handle.creation_nonce == other_nonce:
                    other_nonce = "1" * 32
                malformed_handles = (
                    replace(handle, schema_version=2),
                    replace(handle, backing_name="invalid/name.mmap"),
                    replace(handle, creation_nonce=other_nonce),
                    replace(
                        handle,
                        layer_offsets=(handle.layer_offsets[0] + 1,)
                        + handle.layer_offsets[1:],
                    ),
                    replace(handle, byte_length=handle.byte_length + 1),
                )
                mappings_hidden = True
                for malformed_handle in malformed_handles:
                    partial = SparseMLAOffloadManager.__new__(SparseMLAOffloadManager)
                    partial._validate_and_layout(plan, tp_group, inventory)
                    with pytest.raises(ValueError):
                        partial._initialize_host_mapping(malformed_handle)
                    mappings_hidden &= partial._mmap is None and partial._fd is None
                    partial.close()
                malformed_result = len(malformed_handles), mappings_hidden
            results.put((rank, handle, sentinel, malformed_result))
        elif case == "lifetime":
            denied = False
            if tp_group.rank_in_group != 0:
                try:
                    manager.main_host_write_view("main.0")
                except PermissionError:
                    denied = True
                manager.close()
                manager.close()
                try:
                    manager.unlink()
                except PermissionError:
                    denied = denied and True
            dist.barrier(group=tp_group.cpu_group)
            owner_live = None
            if tp_group.rank_in_group == 0:
                owner_live = manager.layer_view("main.0").main_host_kv.numel()
                manager.close()
                manager.close()
            dist.barrier(group=tp_group.cpu_group)
            if tp_group.rank_in_group == 0:
                manager.unlink()
                manager.unlink()
            results.put((rank, handle, denied, owner_live))
        elif case == "local":
            view = manager.layer_view("main.0")
            other_view = manager.layer_view("main.1")
            same_view = (
                view is manager.layer_view("main.0")
                and other_view is manager.layer_view("main.1")
                and not any(
                    hasattr(view, name)
                    for name in (
                        "close",
                        "unlink",
                        "layer_view",
                        "main_host_write_view",
                        "request_block_ids",
                    )
                )
            )
            indexer_borrowed = all(
                manager.indexer_kv(name).data_ptr() == inventory[name].data_ptr()
                for name in _INDEXER_LAYERS
            )
            per_layer = _PER_LAYER_BUFFERS
            request_wide = set(manager._local_buffers) - per_layer
            entries = {entry.name: entry for entry in plan.manifest}
            view_contract = (
                len(per_layer) == 8
                and all(
                    view.local_buffers[name].data_ptr()
                    == manager._local_buffers[name][0].data_ptr()
                    for name in per_layer
                )
                and all(
                    view.local_buffers[name] is manager._local_buffers[name]
                    for name in request_wide
                )
                and all(
                    tensor.shape
                    == (
                        entries[name].shape[1:]
                        if name in per_layer
                        else entries[name].shape
                    )
                    for name, tensor in view.local_buffers.items()
                )
            )
            cursor = 0
            layout_matches = True
            slab = manager._device_slab
            assert slab is not None
            for entry in plan.manifest[2:]:
                alignment = entry.alignment_bytes
                offset = (cursor + alignment - 1) // alignment * alignment
                tensor = manager._local_buffers[entry.name]
                layout_matches &= tensor.data_ptr() - slab.data_ptr() == offset
                payload = math.prod(entry.shape) * entry.dtype.itemsize
                cursor = offset + (payload + alignment - 1) // alignment * alignment
            slab_layout = (
                layout_matches,
                slab.numel() * slab.element_size(),
                cursor,
            )
            event_rows = (
                view.fork_ready_events is manager._fork_ready_events[0],
                view.miss_ready_events is manager._miss_ready_events[0],
                other_view.fork_ready_events is manager._fork_ready_events[1],
                other_view.miss_ready_events is manager._miss_ready_events[1],
            )
            fork_event_count = len(view.fork_ready_events)
            miss_event_count = len(view.miss_ready_events)
            side_stream = view.side_stream
            exported_host_buffer = memoryview(manager._mmap)
            try:
                manager.close()
            except BufferError:
                retained_view_failed_close = True
            else:
                retained_view_failed_close = False
            try:
                manager.layer_view("main.0")
            except RuntimeError:
                acquisition_rejected = True
            else:
                acquisition_rejected = False
            del exported_host_buffer, view, other_view
            gc.collect()
            manager.close()
            results.put(
                (
                    rank,
                    handle,
                    same_view,
                    indexer_borrowed,
                    fork_event_count,
                    miss_event_count,
                    side_stream,
                    retained_view_failed_close,
                    acquisition_rejected,
                    len(invalid_host_errors) == 2
                    and all(
                        "invalid sparse MLA manifest dimensions" in error
                        for error in invalid_host_errors
                    ),
                    view_contract,
                    slab_layout,
                    event_rows,
                )
            )
        else:
            raise AssertionError(case)
        manager.close()
        if tp_group.rank_in_group == 0:
            manager.unlink()
    except BaseException as exc:
        results.put((rank, "ERROR", type(exc).__name__, str(exc)))
        raise
    finally:
        if manager is not None:
            with suppress(BaseException):
                manager.close()
        if tp_group is not None:
            tp_group.destroy()
        if dist.is_initialized():
            dist.destroy_process_group()
        cuda_patch.undo()


def _run_sparse_mla_rank_case(
    case: str, world_size: int, tp_size: int, dp_size: int
) -> list[tuple]:
    assert world_size == tp_size * dp_size
    context = multiprocessing.get_context("spawn")
    results = context.Queue(maxsize=world_size)
    port = get_open_port()
    children = [
        context.Process(
            target=_sparse_mla_rank,
            args=(rank, world_size, tp_size, dp_size, port, case, results),
        )
        for rank in range(world_size)
    ]
    try:
        for child in children:
            child.start()
        records = [results.get(timeout=45) for _ in children]
        for child in children:
            child.join(timeout=45)
        assert [child.exitcode for child in children] == [0] * world_size
        assert not any(child.is_alive() for child in children)
        return sorted(records)
    finally:
        for child in children:
            if child.is_alive():
                child.terminate()
            child.join(timeout=5)
        results.close()
        results.join_thread()


@pytest.mark.cpu_test
def test_sparse_mla_pool_handle_and_cross_process_visibility():
    records = _run_sparse_mla_rank_case("visibility", 4, 2, 2)
    handles = [record[1] for record in records]
    assert handles[0] == handles[1]
    assert handles[2] == handles[3]
    assert handles[0] != handles[2]
    assert [handle.dp_replica_id for handle in handles] == [0, 0, 1, 1]
    assert [handle.tp_global_ranks for handle in handles] == [
        (0, 1),
        (0, 1),
        (2, 3),
        (2, 3),
    ]
    assert [record[2] for record in records] == [17, 17, 19, 19]
    assert all(not _backing_path(handle.backing_name).exists() for handle in handles)
    assert records[1][3] == (5, True)
    assert records[3][3] == (5, True)


@pytest.mark.cpu_test
def test_sparse_mla_pool_writer_and_lifetime_ownership():
    records = _run_sparse_mla_rank_case("lifetime", 2, 2, 1)
    assert records[0][3] == 16
    assert records[1][2] is True
    assert records[1][3] is None
    assert records[0][1] == records[1][1]
    assert not _backing_path(records[0][1].backing_name).exists()


@pytest.mark.cpu_test
def test_sparse_mla_manager_owns_local_buffers_and_exposes_borrowed_layer_views():
    (record,) = _run_sparse_mla_rank_case("local", 1, 1, 1)
    assert record[2:7] == (True, True, 2, 2, None)
    (
        retained_view_failed_close,
        acquisition_rejected,
        invalid_host_rejected,
        view_contract,
        slab_layout,
        event_rows,
    ) = record[7:]
    assert retained_view_failed_close
    assert acquisition_rejected
    assert invalid_host_rejected
    assert view_contract
    fixed_bytes = _make_sparse_mla_plan(1, 1).fixed_offload_hbm_bytes_per_tp_rank
    assert slab_layout == (True, fixed_bytes, fixed_bytes)
    assert event_rows == (True, True, True, True)
    assert not _backing_path(record[1].backing_name).exists()
