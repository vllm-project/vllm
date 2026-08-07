# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import fcntl
import gc
import math
import multiprocessing
import os
import tempfile
from contextlib import nullcontext, suppress
from dataclasses import replace
from pathlib import Path
from types import MappingProxyType, SimpleNamespace

import pytest
import torch
import torch.distributed as dist

from vllm.distributed.parallel_state import GroupCoordinator
from vllm.utils.network_utils import get_open_port
from vllm.v1.core.kv_cache_utils import (
    _sparse_mla_fixed_hbm_bytes,
    _sparse_mla_manifest,
)
from vllm.v1.kv_cache_interface import (
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheTensor,
    SparseMLAOffloadMemoryPlan,
)
from vllm.v1.worker.gpu.attn_utils import _allocate_kv_cache
from vllm.v1.worker.gpu.sparse_mla_offload import SparseMLAOffloadManager
from vllm.v1.worker.utils import AttentionGroup

_MAIN_LAYERS = ("main.0", "main.1")
_INDEXER_LAYERS = ("indexer.0", "indexer.1")
_PER_LAYER_BUFFERS = frozenset(
    [
        "resident_main_kv",
        "resident_logical_ids",
        "resident_last_access",
        "newest_main_kv",
        "newest_logical_ids",
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
                    invalid = SparseMLAOffloadManager.create_with_tp_shared_pool(
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
        manager = SparseMLAOffloadManager.create_with_tp_shared_pool(
            plan, tp_group, inventory
        )
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
            assert manager._fd is not None
            access_mode = fcntl.fcntl(manager._fd, fcntl.F_GETFL) & os.O_ACCMODE
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
            results.put((rank, handle, denied, owner_live, access_mode))
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
                len(per_layer) == 6
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
    assert records[0][4] == os.O_RDWR
    assert records[1][4] == os.O_RDONLY
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


class _FailingClearDict(dict):
    def __init__(self, values, failures: int = 0):
        super().__init__(values)
        self.failures = failures

    def clear(self) -> None:
        if self.failures:
            self.failures -= 1
            raise RuntimeError("clear fault")
        super().clear()


def _make_lifecycle_runner(manager, inventory):
    from vllm.v1.worker.gpu.model_runner import GPUModelRunner

    runner = GPUModelRunner.__new__(GPUModelRunner)
    runner.sparse_mla_offload_manager = manager
    runner._sparse_mla_kv_caches_dict = inventory
    runner._sparse_mla_indexer_layer_names = _INDEXER_LAYERS
    runner._sparse_mla_shutdown_started = False
    runner._sparse_mla_terminal_connector_failure = None
    runner._sparse_mla_connector_shutdown_complete = False
    runner._sparse_mla_sync_complete = False
    runner._sparse_mla_graphs_released = False
    runner._sparse_mla_borrower_step = 0
    runner._sparse_mla_borrowers_unbound = False
    runner._sparse_mla_local_closed = False
    runner._sparse_mla_shared_shutdown_complete = False
    runner._sparse_mla_local_cleanup_step = 0
    runner._sparse_mla_local_shutdown_complete = False
    runner.cudagraph_manager = None
    runner.speculator = None
    runner.kv_connector = object()
    runner.kv_caches = list(inventory.values())
    runner.attn_groups = []
    runner.compilation_config = SimpleNamespace(
        static_forward_context={
            name: SimpleNamespace(kv_cache=tensor) for name, tensor in inventory.items()
        }
    )
    runner.vllm_config = SimpleNamespace()
    return runner


def _patch_lifecycle_runtime(monkeypatch, tp_group, connector_shutdown):
    import vllm.v1.worker.gpu.model_runner as model_runner_module

    real_connector_shutdown = model_runner_module.ensure_kv_transfer_shutdown

    def wrapped_connector_shutdown() -> None:
        real_connector_shutdown()
        connector_shutdown()

    monkeypatch.setattr(model_runner_module, "get_tp_group", lambda: tp_group)
    monkeypatch.setattr(
        model_runner_module,
        "ensure_kv_transfer_shutdown",
        wrapped_connector_shutdown,
    )
    monkeypatch.setattr(torch.accelerator, "synchronize", lambda: None)
    monkeypatch.setattr(torch.accelerator, "empty_cache", lambda: None)
    monkeypatch.setattr(
        model_runner_module, "free_before_shutdown", lambda config: None
    )


def _initialize_lifecycle_tp_group(rank: int, port: int, case: str):
    dist.init_process_group(
        backend="gloo",
        init_method=f"tcp://127.0.0.1:{port}",
        world_size=2,
        rank=rank,
    )
    tp_group = GroupCoordinator(
        group_ranks=[[0, 1]],
        local_rank=0,
        torch_distributed_backend="gloo",
        use_device_communicator=False,
        use_message_queue_broadcaster=False,
        group_name=f"c4-lifecycle-{case}",
    )
    dist.barrier()
    tp_group.device = torch.device("cpu")
    return tp_group


def _prepare_initialize_runner(patch, tp_group, plan, inventory, connector_factory):
    import vllm.v1.worker.gpu.model_runner as model_runner_module

    runner = _make_lifecycle_runner(None, {})
    support = SimpleNamespace(min_cg_support=None, min_cg_attn_backend=None)
    support.narrow = lambda *args: support
    runner.max_model_len = 16
    runner.is_encoder_decoder = False
    runner.scheduler_config = SimpleNamespace(max_num_encoder_input_tokens=0)
    runner.model_config = SimpleNamespace(hf_config=SimpleNamespace())
    runner.dcp_size = 1
    runner.dcp_rank = 0
    runner.cp_interleave = 1
    runner.cache_config = SimpleNamespace(
        enable_prefix_caching=False, cache_dtype="auto"
    )
    runner.device = torch.device("cpu")
    runner.model_state = SimpleNamespace(get_additional_cg_support=lambda: ())
    runner.max_num_reqs = 2
    runner.max_num_tokens = 4
    runner.parallel_config = SimpleNamespace(tensor_parallel_size=2)
    runner.supports_mm_inputs = False
    runner.req_states = SimpleNamespace()
    runner.decode_query_len = 1
    runner.lora_capture_cases = [0]
    runner.input_buffers = SimpleNamespace()
    runner.compilation_config = SimpleNamespace(
        static_forward_context={
            name: SimpleNamespace(kv_cache=tensor) for name, tensor in inventory.items()
        },
        resolve_cudagraph_mode_and_sizes=lambda *args, **kwargs: None,
    )
    runner.vllm_config = SimpleNamespace(mamba_config=None)

    kv_cache_spec = SimpleNamespace(block_size=2)
    kv_cache_config = KVCacheConfig(
        num_blocks=2,
        kv_cache_tensors=[
            KVCacheTensor(size=tensor.nbytes, shared_by=[name])
            for name, tensor in inventory.items()
        ],
        kv_cache_groups=[
            KVCacheGroupSpec(
                layer_names=[*_MAIN_LAYERS, *_INDEXER_LAYERS],
                kv_cache_spec=kv_cache_spec,
            )
        ],
        sparse_mla_offload_plan=plan,
    )
    attn_groups = [
        [
            AttentionGroup(object, list(_MAIN_LAYERS), kv_cache_spec, 0),
            AttentionGroup(object, list(_INDEXER_LAYERS), kv_cache_spec, 0),
        ]
    ]
    observations = {}

    def initialize_inventory(
        kv_caches,
        forward_context,
        physical_config,
        physical_attn_groups,
        device,
        *args,
        **kwargs,
    ):
        raw_inventory = _allocate_kv_cache(physical_config, {}, device)
        allocated_inventory = {
            name: tensor.view(torch.float32).view(2, 2, 2)
            for name, tensor in raw_inventory.items()
        }
        kv_caches.extend(allocated_inventory.values())
        observations.update(
            {
                "config": physical_config,
                "attn_groups": physical_attn_groups,
                "inventory": allocated_inventory,
                "raw_numel": sum(tensor.numel() for tensor in raw_inventory.values()),
            }
        )
        return allocated_inventory

    patch.setattr(model_runner_module, "get_tp_group", lambda: tp_group)
    patch.setattr(
        model_runner_module,
        "init_attn_backend",
        lambda *args, **kwargs: (attn_groups, support, [2]),
    )
    patch.setattr(model_runner_module, "BlockTables", lambda *args, **kwargs: None)

    def disable_pcp_manager(
        _vllm_config,
        _device,
        _supports_mm_inputs,
        _req_states,
        _block_tables,
        cls,
    ):
        _ = cls
        return None

    patch.setattr(
        model_runner_module.pcp,
        "maybe_build_pcp_manager",
        disable_pcp_manager,
    )
    patch.setattr(
        model_runner_module, "initialize_mamba_ssu_backend", lambda *args: None
    )
    patch.setattr(
        model_runner_module, "ModelCudaGraphManager", lambda *args, **kwargs: None
    )
    patch.setattr(
        model_runner_module, "check_attention_cp_compatibility", lambda config: None
    )
    patch.setattr(model_runner_module, "init_kv_cache", initialize_inventory)
    patch.setattr(model_runner_module, "get_kv_connector", connector_factory)
    return runner, kv_cache_config, observations


def _lifecycle_init_rank(rank: int, port: int, case: str, results) -> None:
    import vllm.v1.worker.gpu.sparse_mla_offload as sparse_module

    patch = pytest.MonkeyPatch()
    tp_group = None
    manager = None
    handle = None
    try:
        tp_group = _initialize_lifecycle_tp_group(rank, port, case)
        plan = _make_sparse_mla_plan(2, 1)
        inventory = {
            name: torch.arange(8, dtype=torch.float32).view(2, 2, 2)
            for name in _INDEXER_LAYERS
        }
        missing_inventory_error = None
        missing_inventory_factory_calls = None
        if case == "success":
            missing_inventory = dict(inventory)
            del missing_inventory[_INDEXER_LAYERS[-1]]
            factory_calls = 0
            original_factory = SparseMLAOffloadManager.create_with_tp_shared_pool

            def count_factory(cls, *args, **kwargs):
                nonlocal factory_calls
                factory_calls += 1
                return original_factory(*args, **kwargs)

            patch.setattr(
                SparseMLAOffloadManager,
                "create_with_tp_shared_pool",
                classmethod(count_factory),
            )
            missing_runner, missing_config, _ = _prepare_initialize_runner(
                patch,
                tp_group,
                plan,
                missing_inventory,
                lambda config, kv_caches_dict: object(),
            )
            try:
                missing_runner.initialize_kv_cache(missing_config)
            except (RuntimeError, ValueError) as error:
                missing_inventory_error = str(error)
            missing_inventory_factory_calls = factory_calls
        if case == "same_host":
            patch.setattr(
                sparse_module,
                "in_the_same_node_as",
                lambda group, source_rank=0: [rank == 0, rank == 0],
            )
        elif case == "creator_failure":
            original_initialize = SparseMLAOffloadManager._initialize_host_mapping

            def fail_creator(self, pool_handle):
                if rank == 0 and pool_handle is None:
                    raise RuntimeError("creator fault")
                return original_initialize(self, pool_handle)

            patch.setattr(
                SparseMLAOffloadManager, "_initialize_host_mapping", fail_creator
            )
        elif case == "malformed_handle":
            original_broadcast = sparse_module.dist.broadcast_object_list

            def corrupt_handle(values, *args, **kwargs):
                original_broadcast(values, *args, **kwargs)
                if (
                    rank == 1
                    and isinstance(values[0], tuple)
                    and isinstance(values[0][0], sparse_module.SparseMLAPoolHandle)
                ):
                    pool_handle, error = values[0]
                    values[0] = replace(pool_handle, schema_version=2), error

            patch.setattr(sparse_module.dist, "broadcast_object_list", corrupt_handle)
        elif case == "follower_attach_register_failure":
            original_register = SparseMLAOffloadManager._register_host_mapping

            def fail_follower_register(self):
                if rank == 1:
                    raise RuntimeError("follower register fault")
                return original_register(self)

            patch.setattr(
                SparseMLAOffloadManager,
                "_register_host_mapping",
                fail_follower_register,
            )

        connector_inventory_ids = []

        def construct_connector(config, kv_caches_dict):
            connector_inventory_ids.append(id(kv_caches_dict))
            return object()

        runner, kv_cache_config, observations = _prepare_initialize_runner(
            patch, tp_group, plan, inventory, construct_connector
        )
        with pytest.raises(
            AssertionError, match="Some layers are not correctly initialized"
        ):
            _allocate_kv_cache(kv_cache_config, {}, torch.device("cpu"))
        try:
            runner.initialize_kv_cache(kv_cache_config)
        except RuntimeError as error:
            results.put(
                {
                    "rank": rank,
                    "case": case,
                    "error": str(error),
                    "backings": [
                        path.name
                        for path in _backing_path("").parent.glob(
                            "vllm_sparse_mla_*_dp0.mmap"
                        )
                    ],
                }
            )
            return

        manager = runner.sparse_mla_offload_manager
        assert manager is not None
        handle = manager.pool_handle
        if rank == 0:
            manager.main_host_write_view("main.0").fill_(29)
        dist.barrier(group=tp_group.cpu_group)
        connector_calls = []
        _patch_lifecycle_runtime(
            patch, tp_group, lambda: connector_calls.append("connector")
        )
        allocated_inventory = observations["inventory"]
        exact_inventory = runner._sparse_mla_kv_caches_dict is allocated_inventory
        exact_connector_inventory = connector_inventory_ids == [id(allocated_inventory)]
        physical_config = observations["config"]
        physical_attn_groups = observations["attn_groups"]
        physical_group_names = [
            group.layer_names for group in physical_config.kv_cache_groups
        ]
        physical_attn_names = [
            [name for attn_group in groups for name in attn_group.layer_names]
            for groups in physical_attn_groups
        ]
        logical_group_names = [
            group.layer_names for group in runner.kv_cache_config.kv_cache_groups
        ]
        allocated_keys = tuple(sorted(allocated_inventory))
        allocated_devices = tuple(
            str(tensor.device) for tensor in allocated_inventory.values()
        )
        sentinel = manager.layer_view("main.0").main_host_kv.flatten()[0].item()
        runner.shutdown_sparse_mla_shared()
        results.put(
            {
                "rank": rank,
                "case": case,
                "error": None,
                "sentinel": sentinel,
                "exact_inventory": exact_inventory,
                "exact_connector_inventory": exact_connector_inventory,
                "allocated_keys": allocated_keys,
                "allocated_devices": allocated_devices,
                "allocated_raw_numel": observations["raw_numel"],
                "expected_raw_numel": sum(
                    tensor.size for tensor in physical_config.kv_cache_tensors
                ),
                "physical_group_names": physical_group_names,
                "physical_attn_names": physical_attn_names,
                "logical_group_names": logical_group_names,
                "connector_calls": len(connector_calls),
                "manager_cleared": runner.sparse_mla_offload_manager is None,
                "backing": handle.backing_name,
                "missing_inventory_error": missing_inventory_error,
                "missing_inventory_factory_calls": missing_inventory_factory_calls,
            }
        )
        manager = None
    except BaseException as error:
        results.put(
            {
                "rank": rank,
                "case": case,
                "fatal": type(error).__name__,
                "error": str(error),
            }
        )
        raise
    finally:
        if manager is not None:
            with suppress(BaseException):
                manager.close()
            if rank == 0:
                with suppress(BaseException):
                    manager.unlink()
        if tp_group is not None:
            tp_group.destroy()
        if dist.is_initialized():
            dist.destroy_process_group()
        patch.undo()


def _lifecycle_shutdown_rank(rank: int, port: int, case: str, results) -> None:
    import vllm.v1.worker.gpu.model_runner as model_runner_module
    import vllm.v1.worker.gpu_worker as gpu_worker_module
    from vllm.v1.worker.gpu_worker import Worker as GPUWorker

    patch = pytest.MonkeyPatch()
    tp_group = None
    manager = None
    handle = None
    original_close = None
    original_unlink = None
    try:
        tp_group = _initialize_lifecycle_tp_group(rank, port, case)
        real_worker_connector_shutdown = gpu_worker_module.ensure_kv_transfer_shutdown
        collective_counts = {"gather": 0, "barrier": 0, "broadcast": 0}
        original_gather = model_runner_module.dist.all_gather_object
        original_barrier = model_runner_module.dist.barrier
        original_broadcast = model_runner_module.dist.broadcast_object_list

        def count_gather(*args, **kwargs):
            collective_counts["gather"] += 1
            return original_gather(*args, **kwargs)

        def count_barrier(*args, **kwargs):
            collective_counts["barrier"] += 1
            return original_barrier(*args, **kwargs)

        def count_broadcast(*args, **kwargs):
            collective_counts["broadcast"] += 1
            return original_broadcast(*args, **kwargs)

        patch.setattr(model_runner_module.dist, "all_gather_object", count_gather)
        patch.setattr(model_runner_module.dist, "barrier", count_barrier)
        patch.setattr(
            model_runner_module.dist, "broadcast_object_list", count_broadcast
        )
        plan = _make_sparse_mla_plan(2, 1)
        inventory = {
            name: torch.arange(8, dtype=torch.float32).view(2, 2, 2)
            for name in _INDEXER_LAYERS
        }
        connector_calls = []
        generic_connector_calls: list[str] = []
        connector_fault = [True]
        lifecycle_unlink_calls = [0]
        manager_original_unlink = [None]
        original_manager_factory = SparseMLAOffloadManager.create_with_tp_shared_pool

        def create_counted_manager(cls, *args, **kwargs):
            created_manager = original_manager_factory(*args, **kwargs)
            real_unlink = created_manager.unlink
            manager_original_unlink[0] = real_unlink

            def counted_unlink() -> None:
                lifecycle_unlink_calls[0] += 1
                real_unlink()

            created_manager.unlink = counted_unlink
            return created_manager

        patch.setattr(
            SparseMLAOffloadManager,
            "create_with_tp_shared_pool",
            classmethod(create_counted_manager),
        )
        terminal_transition_before = None
        terminal_transition_delta = None

        def connector_shutdown() -> None:
            connector_calls.append("connector")
            if (
                case
                in (
                    "post_factory_registered_connector_failure",
                    "worker_connector_shutdown_terminal_failure",
                )
                and rank == 1
                and connector_fault[0]
            ):
                raise RuntimeError("connector fault")

        _patch_lifecycle_runtime(patch, tp_group, connector_shutdown)
        errors = []
        if case == "post_factory_registered_connector_failure":

            def construct_connector(config, kv_caches_dict):
                nonlocal terminal_transition_before
                terminal_transition_before = collective_counts.copy()
                if rank == 1:
                    raise RuntimeError("init fault")
                return object()

            runner, kv_cache_config, _ = _prepare_initialize_runner(
                patch, tp_group, plan, inventory, construct_connector
            )
            try:
                runner.initialize_kv_cache(kv_cache_config)
            except RuntimeError as error:
                errors.append(str(error))
            assert terminal_transition_before is not None
            terminal_transition_delta = {
                name: collective_counts[name] - terminal_transition_before[name]
                for name in collective_counts
            }
            manager = runner.sparse_mla_offload_manager
            assert manager is not None
        else:
            manager = SparseMLAOffloadManager.create_with_tp_shared_pool(
                plan, tp_group, inventory
            )
            runner = _make_lifecycle_runner(manager, inventory)
        handle = manager.pool_handle
        original_unlink = manager_original_unlink[0]
        assert original_unlink is not None

        def generic_connector_shutdown() -> None:
            real_worker_connector_shutdown()
            generic_connector_calls.append("connector")

        patch.setattr(
            gpu_worker_module,
            "ensure_kv_transfer_shutdown",
            generic_connector_shutdown,
        )
        patch.setattr(gpu_worker_module, "ensure_ec_transfer_shutdown", None)
        patch.setattr(
            gpu_worker_module.current_platform, "is_cuda_alike", lambda: False
        )
        worker = GPUWorker.__new__(GPUWorker)
        worker.use_v2_model_runner = True
        worker.model_runner = runner
        worker.profiler = None
        worker.weight_transfer_engine = None

        if case in (
            "post_factory_registered_connector_failure",
            "worker_connector_shutdown_terminal_failure",
        ):
            if case == "worker_connector_shutdown_terminal_failure":
                terminal_transition_before = collective_counts.copy()
            calls = (
                (worker.shutdown, runner.shutdown)
                if case == "post_factory_registered_connector_failure"
                else (worker.shutdown, worker.shutdown, runner.shutdown)
            )
            terminal_snapshot = None
            for index, call in enumerate(calls):
                try:
                    call()
                except RuntimeError as error:
                    errors.append(str(error))
                if index == 0:
                    if case == "worker_connector_shutdown_terminal_failure":
                        assert terminal_transition_before is not None
                        terminal_transition_delta = {
                            name: collective_counts[name]
                            - terminal_transition_before[name]
                            for name in collective_counts
                        }
                    terminal_snapshot = collective_counts.copy()
            terminal_local_only = terminal_snapshot == collective_counts
            terminal_unlink_calls = lifecycle_unlink_calls[0]
            nonconnector_error = None
            nonconnector_terminal = None
            nonconnector_complete = None
            nonconnector_connector_calls = None
            retained = runner.sparse_mla_offload_manager is manager
            if case == "post_factory_registered_connector_failure":
                manager.close()
                dist.barrier(group=tp_group.cpu_group)
                if rank == 0:
                    original_unlink()
                dist.barrier(group=tp_group.cpu_group)
                manager = None
                connector_fault[0] = False
                connector_calls.clear()
                retry_inventory = {
                    name: torch.arange(8, dtype=torch.float32).view(2, 2, 2)
                    for name in _INDEXER_LAYERS
                }

                def fail_retry_connector(config, kv_caches_dict):
                    if rank == 1:
                        raise RuntimeError("retry init fault")
                    return object()

                retry_runner, retry_config, _ = _prepare_initialize_runner(
                    patch,
                    tp_group,
                    plan,
                    retry_inventory,
                    fail_retry_connector,
                )
                patch.setattr(
                    model_runner_module,
                    "ModelCudaGraphManager",
                    lambda *args, **kwargs: SimpleNamespace(
                        graphs=_FailingClearDict({}, failures=rank),
                        breakable_cg_runner=object(),
                    ),
                )
                try:
                    retry_runner.initialize_kv_cache(retry_config)
                except RuntimeError as error:
                    nonconnector_error = str(error)
                nonconnector_terminal = (
                    retry_runner._sparse_mla_terminal_connector_failure
                )
                retry_runner.shutdown_sparse_mla_shared()
                nonconnector_complete = (
                    retry_runner._sparse_mla_shared_shutdown_complete
                )
                nonconnector_connector_calls = len(connector_calls)
            results.put(
                {
                    "rank": rank,
                    "case": case,
                    "errors": errors,
                    "connector_calls": len(connector_calls),
                    "generic_connector_calls": len(generic_connector_calls),
                    "terminal_local_only": terminal_local_only,
                    "terminal_transition_delta": terminal_transition_delta,
                    "terminal_unlink_calls": terminal_unlink_calls,
                    "collective_counts": collective_counts,
                    "terminal": runner._sparse_mla_terminal_connector_failure,
                    "retained": retained,
                    "backing": handle.backing_name,
                    "nonconnector_error": nonconnector_error,
                    "nonconnector_terminal": nonconnector_terminal,
                    "nonconnector_complete": nonconnector_complete,
                    "nonconnector_connector_calls": nonconnector_connector_calls,
                }
            )
            return

        original_close = manager.close
        original_unlink = manager.unlink
        close_calls = 0
        unlink_calls = 0
        if case == "retained_borrow_close_retry":
            runner.cudagraph_manager = SimpleNamespace(
                graphs=_FailingClearDict({}, failures=rank),
                breakable_cg_runner=object(),
            )
            runner._sparse_mla_kv_caches_dict = _FailingClearDict(
                inventory, failures=rank
            )

            def close_with_fault() -> None:
                nonlocal close_calls
                close_calls += 1
                if rank == 1 and close_calls == 1:
                    raise RuntimeError("close fault")
                original_close()

            manager.close = close_with_fault
            for _ in range(4):
                try:
                    runner.shutdown_sparse_mla_shared()
                except RuntimeError as error:
                    errors.append(str(error))
            assert runner._sparse_mla_shared_shutdown_complete
        elif case == "tp0_unlink_failure_retry":

            def unlink_with_fault() -> None:
                nonlocal unlink_calls
                unlink_calls += 1
                if rank == 0 and unlink_calls == 1:
                    raise RuntimeError("unlink fault")
                original_unlink()

            manager.unlink = unlink_with_fault
            for _ in range(2):
                try:
                    runner.shutdown_sparse_mla_shared()
                except RuntimeError as error:
                    errors.append(str(error))
            assert runner._sparse_mla_shared_shutdown_complete
        else:
            free_calls = 0

            def fail_free_once(config) -> None:
                nonlocal free_calls
                free_calls += 1
                if free_calls == 1:
                    raise RuntimeError("free fault")

            patch.setattr(model_runner_module, "free_before_shutdown", fail_free_once)
            try:
                worker.shutdown()
            except RuntimeError as error:
                errors.append(str(error))
            worker.shutdown()
            assert runner._sparse_mla_local_shutdown_complete
            nonoffload_order = []
            nonoffload_runner = _make_lifecycle_runner(None, {})

            def nonoffload_connector_shutdown() -> None:
                real_worker_connector_shutdown()
                nonoffload_order.append("connector")

            patch.setattr(
                gpu_worker_module,
                "ensure_kv_transfer_shutdown",
                nonoffload_connector_shutdown,
            )
            patch.setattr(
                torch.accelerator,
                "synchronize",
                lambda: nonoffload_order.append("synchronize"),
            )
            patch.setattr(
                model_runner_module,
                "free_before_shutdown",
                lambda config: nonoffload_order.append("free"),
            )
            nonoffload_worker = GPUWorker.__new__(GPUWorker)
            nonoffload_worker.use_v2_model_runner = True
            nonoffload_worker.model_runner = nonoffload_runner
            nonoffload_worker.profiler = None
            nonoffload_worker.weight_transfer_engine = None
            nonoffload_worker.shutdown()
        results.put(
            {
                "rank": rank,
                "case": case,
                "errors": errors,
                "connector_calls": len(connector_calls),
                "generic_connector_calls": len(generic_connector_calls),
                "close_calls": close_calls,
                "unlink_calls": unlink_calls,
                "shared_complete": runner._sparse_mla_shared_shutdown_complete,
                "manager_cleared": runner.sparse_mla_offload_manager is None,
                "backing": handle.backing_name,
                "free_calls": free_calls
                if case == "completed_worker_shutdown_no_collective"
                else None,
                "nonoffload_order": nonoffload_order
                if case == "completed_worker_shutdown_no_collective"
                else None,
            }
        )
        manager = None
    except BaseException as error:
        results.put(
            {
                "rank": rank,
                "case": case,
                "fatal": type(error).__name__,
                "error": str(error),
            }
        )
        raise
    finally:
        if manager is not None:
            if original_close is not None:
                with suppress(BaseException):
                    original_close()
            else:
                with suppress(BaseException):
                    manager.close()
            if rank == 0:
                if original_unlink is not None:
                    with suppress(BaseException):
                        original_unlink()
                else:
                    with suppress(BaseException):
                        manager.unlink()
        if tp_group is not None:
            tp_group.destroy()
        if dist.is_initialized():
            dist.destroy_process_group()
        patch.undo()


def _run_lifecycle_case(target, case: str) -> list[dict]:
    context = multiprocessing.get_context("spawn")
    results = context.Queue(maxsize=2)
    port = get_open_port()
    children = [
        context.Process(target=target, args=(rank, port, case, results))
        for rank in range(2)
    ]
    try:
        for child in children:
            child.start()
        records = [results.get(timeout=60) for _ in children]
        for child in children:
            child.join(timeout=60)
        assert [child.exitcode for child in children] == [0, 0]
        assert not any(child.is_alive() for child in children)
        assert not any("fatal" in record for record in records)
        return sorted(records, key=lambda record: record["rank"])
    finally:
        for child in children:
            if child.is_alive():
                child.terminate()
            child.join(timeout=5)
        results.close()
        results.join_thread()


@pytest.mark.parametrize(
    "case",
    [
        "success",
        "same_host",
        "creator_failure",
        "malformed_handle",
        "follower_attach_register_failure",
    ],
)
def test_gpu_model_runner_v2_sparse_mla_manager_init_and_shared_visibility(
    case,
):
    records = _run_lifecycle_case(_lifecycle_init_rank, case)
    assert [record["case"] for record in records] == [case, case]
    assert records[0]["error"] == records[1]["error"]
    if case == "success":
        assert [record["sentinel"] for record in records] == [29, 29]
        assert all(record["exact_inventory"] for record in records)
        assert all(record["exact_connector_inventory"] for record in records)
        assert all(record["allocated_keys"] == _INDEXER_LAYERS for record in records)
        assert all(record["allocated_devices"] == ("cpu", "cpu") for record in records)
        assert all(
            record["allocated_raw_numel"] == record["expected_raw_numel"] == 64
            for record in records
        )
        assert all(
            record["physical_group_names"] == [list(_INDEXER_LAYERS)]
            for record in records
        )
        assert all(
            record["physical_attn_names"] == [list(_INDEXER_LAYERS)]
            for record in records
        )
        assert all(
            record["logical_group_names"] == [[*_MAIN_LAYERS, *_INDEXER_LAYERS]]
            for record in records
        )
        assert all(record["connector_calls"] == 1 for record in records)
        assert all(record["manager_cleared"] for record in records)
        assert all(
            record["missing_inventory_error"]
            == "Sparse MLA physical KV cache is missing planned Indexer layers: "
            "indexer.1"
            for record in records
        )
        assert all(record["missing_inventory_factory_calls"] == 0 for record in records)
        assert not _backing_path(records[0]["backing"]).exists()
    else:
        expected_errors = {
            "same_host": "Sparse MLA offload same-host proof failed on TP rank 1: "
            "RuntimeError: tensor-parallel ranks are not on one host",
            "creator_failure": "Sparse MLA offload creator mapping failed on TP rank "
            "0: RuntimeError: creator fault",
            "malformed_handle": "Sparse MLA offload local initialization failed on "
            "TP rank 1: ValueError: shared Host pool handle does not match the local "
            "plan",
            "follower_attach_register_failure": (
                "Sparse MLA offload local initialization failed on TP rank 1: "
                "RuntimeError: follower register fault"
            ),
        }
        assert records[0]["error"] == expected_errors[case]
        assert not records[0]["backings"]


@pytest.mark.parametrize(
    "case",
    [
        "post_factory_registered_connector_failure",
        "worker_connector_shutdown_terminal_failure",
        "retained_borrow_close_retry",
        "tp0_unlink_failure_retry",
        "completed_worker_shutdown_no_collective",
    ],
)
def test_gpu_model_runner_v2_sparse_mla_manager_failure_unwind_and_shutdown(
    case,
):
    records = _run_lifecycle_case(_lifecycle_shutdown_rank, case)
    assert [record["case"] for record in records] == [case, case]
    if case in (
        "post_factory_registered_connector_failure",
        "worker_connector_shutdown_terminal_failure",
    ):
        assert records[0]["errors"] == records[1]["errors"]
        assert len(records[0]["errors"]) == 3
        if case == "post_factory_registered_connector_failure":
            assert (
                "; rollback: Sparse MLA offload connector shutdown"
                in records[0]["errors"][0]
            )
            assert records[0]["errors"][1] == records[0]["errors"][2]
            assert records[0]["nonconnector_error"] == records[1]["nonconnector_error"]
            assert (
                "; rollback: Sparse MLA offload graph release"
                in records[0]["nonconnector_error"]
            )
            assert records[0]["nonconnector_terminal"] is None
            assert records[1]["nonconnector_terminal"] is None
            assert all(record["nonconnector_complete"] for record in records)
            assert all(
                record["nonconnector_connector_calls"] == 1 for record in records
            )
        else:
            assert len(set(records[0]["errors"])) == 1
        assert records[0]["terminal"] == records[1]["terminal"]
        assert all(record["connector_calls"] == 1 for record in records)
        assert all(record["generic_connector_calls"] == 0 for record in records)
        assert all(record["terminal_local_only"] for record in records)
        assert all(record["terminal_unlink_calls"] == 0 for record in records)
        assert all(
            record["terminal_transition_delta"]["barrier"] == 1 for record in records
        )
        assert all(
            record["terminal_transition_delta"]["broadcast"] == 1 for record in records
        )
        assert records[0]["collective_counts"] == records[1]["collective_counts"]
        assert all(record["retained"] for record in records)
        assert not _backing_path(records[0]["backing"]).exists()
    elif case == "retained_borrow_close_retry":
        assert records[0]["errors"] == records[1]["errors"]
        assert len(records[0]["errors"]) == 3
        assert "graph release" in records[0]["errors"][0]
        assert "borrower unbind" in records[0]["errors"][1]
        assert "manager close" in records[0]["errors"][2]
        assert all(record["connector_calls"] == 1 for record in records)
        assert all(record["shared_complete"] for record in records)
        assert not _backing_path(records[0]["backing"]).exists()
    elif case == "tp0_unlink_failure_retry":
        assert records[0]["errors"] == records[1]["errors"]
        assert len(records[0]["errors"]) == 1
        assert "owner unlink" in records[0]["errors"][0]
        assert records[0]["unlink_calls"] == 2
        assert records[1]["unlink_calls"] == 0
        assert all(record["connector_calls"] == 1 for record in records)
        assert not _backing_path(records[0]["backing"]).exists()
    else:
        assert all(record["shared_complete"] for record in records)
        assert all(record["manager_cleared"] for record in records)
        assert all(record["connector_calls"] == 1 for record in records)
        assert all(record["generic_connector_calls"] == 2 for record in records)
        assert all(record["free_calls"] == 2 for record in records)
        assert all(record["errors"] == ["free fault"] for record in records)
        assert all(
            record["nonoffload_order"] == ["connector", "synchronize", "free"]
            for record in records
        )
        assert not _backing_path(records[0]["backing"]).exists()


def test_gpu_model_runner_v2_sparse_mla_manager_request_lifecycle(monkeypatch):
    import vllm.v1.worker.gpu.model_runner as model_runner_module

    generation_names = {
        "request_generation",
        "resident_generation",
        "newest_generation",
    }

    class GenerationFreeBuffers(dict):
        def __init__(self, buffers):
            super().__init__(buffers)
            self.generation_accesses = []
            self.legacy_generation = {
                "request_generation": torch.zeros(3, dtype=torch.int64),
                "resident_generation": torch.full((2, 3, 2), -1, dtype=torch.int64),
                "newest_generation": torch.full((2, 3, 2), -1, dtype=torch.int64),
            }

        def __missing__(self, name):
            if name in self.legacy_generation:
                self.generation_accesses.append(name)
                return self.legacy_generation[name]
            raise KeyError(name)

    manager = SparseMLAOffloadManager.__new__(SparseMLAOffloadManager)
    manager._closing = manager._closed = False
    manager._row_request_ids = [None, None, None]
    manager._local_buffers = GenerationFreeBuffers(
        {
            "resident_main_kv": torch.full((2, 3, 2, 2), 101, dtype=torch.float32),
            "resident_logical_ids": torch.full((2, 3, 2), 41, dtype=torch.int64),
            "resident_last_access": torch.full((2, 3, 2), 42, dtype=torch.int64),
            "newest_main_kv": torch.full((2, 3, 2, 2), 102, dtype=torch.float32),
            "newest_logical_ids": torch.full((2, 3, 2), 44, dtype=torch.int64),
            "request_block_ids": torch.full((3, 2), -1, dtype=torch.int32),
            "request_num_blocks": torch.zeros(3, dtype=torch.int32),
            "request_num_tokens": torch.zeros(3, dtype=torch.int32),
            "request_active": torch.zeros(3, dtype=torch.bool),
        }
    )

    runner = model_runner_module.GPUModelRunner.__new__(
        model_runner_module.GPUModelRunner
    )
    runner.sparse_mla_offload_manager = manager
    runner.update_pp_decode_requests = lambda: None
    runner.finish_requests = lambda output: None
    runner.free_states = lambda output: None
    runner.add_requests = lambda output: None
    runner.update_requests = lambda output: None
    runner.block_tables = SimpleNamespace(
        apply_staged_writes=lambda: None,
        blocks_per_kv_block=[1],
        num_blocks=SimpleNamespace(gpu=torch.tensor([[2, 1, 2]], dtype=torch.int32)),
    )
    runner.req_states = SimpleNamespace(
        num_computed_tokens=SimpleNamespace(
            gpu=torch.tensor([8, 9, 10], dtype=torch.int32)
        )
    )
    runner.cudagraph_manager = None
    runner.dp_size = runner.dp_rank = 1
    runner.lora_config = None
    runner.is_encoder_decoder = False
    runner.kv_cache_config = SimpleNamespace()
    input_batch = SimpleNamespace(
        req_ids=["request-a", "request-b"],
        idx_mapping=torch.tensor([1, 0], dtype=torch.int32),
        num_reqs_after_padding=3,
        seq_lens=torch.tensor([10, 9], dtype=torch.int32),
    )
    runner.prepare_inputs = lambda output, desc: input_batch
    gathered = torch.tensor(
        [[11, 12, 999], [21, 22, 998], [0, 0, 0]], dtype=torch.int32
    )
    runner.prepare_attn = lambda batch: ((gathered,), torch.zeros(1, dtype=torch.int64))
    runner.model_state = SimpleNamespace(
        preprocess_state=lambda *args: (_ for _ in ()).throw(
            RuntimeError("stop after publication")
        )
    )
    runner.kv_connector = SimpleNamespace(no_forward=lambda output: "no-forward")
    monkeypatch.setattr(
        model_runner_module,
        "dispatch_cg_and_sync_dp",
        lambda *args, **kwargs: (SimpleNamespace(num_tokens=2), None),
    )
    scheduler_output = SimpleNamespace(
        scheduled_new_reqs=[
            SimpleNamespace(req_id="request-a"),
            SimpleNamespace(req_id="request-b"),
        ],
        num_scheduled_tokens={"request-a": 1, "request-b": 1},
        total_num_scheduled_tokens=2,
        scheduled_encoder_inputs={},
    )

    with pytest.raises(RuntimeError, match="stop after publication"):
        runner.execute_model(scheduler_output)
    buffers = manager._local_buffers
    assert torch.equal(
        buffers["request_block_ids"],
        torch.tensor([[11, 12], [21, 22], [-1, -1]], dtype=torch.int32),
    )
    assert buffers["request_num_blocks"].tolist() == [1, 2, 0]
    assert buffers["request_num_tokens"].tolist() == [10, 9, 0]
    assert buffers["request_active"].tolist() == [True, True, False]
    assert not buffers.generation_accesses
    assert generation_names.isdisjoint(buffers)
    assert manager._row_request_ids == ["request-a", "request-b", None]
    assert torch.all(buffers["resident_logical_ids"][:, :2] == -1)
    assert torch.all(buffers["newest_logical_ids"][:, :2] == -1)
    assert torch.all(buffers["resident_last_access"][:, :2] == 0)
    assert torch.all(buffers["resident_main_kv"] == 101)
    assert torch.all(buffers["newest_main_kv"] == 102)

    buffers["resident_logical_ids"].copy_(
        torch.arange(12, dtype=torch.int64).view(2, 3, 2)
    )
    buffers["newest_logical_ids"].copy_(
        torch.arange(24, 36, dtype=torch.int64).view(2, 3, 2)
    )
    buffers["resident_last_access"].fill_(72)
    preserved_resident_ids = buffers["resident_logical_ids"].clone()
    preserved_newest_ids = buffers["newest_logical_ids"].clone()
    preserved_last_access = buffers["resident_last_access"].clone()
    manager._prepare_decode_batch(
        ["request-a", "request-b"],
        (),
        torch.tensor([0, 2], dtype=torch.int32),
        (torch.tensor([[31, 32], [41, 42]], dtype=torch.int32),),
        torch.tensor([[1, 0, 2]], dtype=torch.int32),
        torch.tensor([18, 20], dtype=torch.int32),
        2,
    )
    assert torch.equal(buffers["resident_logical_ids"], preserved_resident_ids)
    assert torch.equal(buffers["newest_logical_ids"], preserved_newest_ids)
    assert torch.equal(buffers["resident_last_access"], preserved_last_access)
    assert buffers["request_num_tokens"].tolist() == [18, 20, 0]

    manager._prepare_decode_batch(
        ["request-a", "request-b"],
        ("request-a",),
        torch.tensor([2, 0], dtype=torch.int32),
        (torch.tensor([[51, 52], [61, 62]], dtype=torch.int32),),
        torch.tensor([[1, 0, 2]], dtype=torch.int32),
        torch.tensor([30, 28], dtype=torch.int32),
        2,
    )
    assert torch.all(buffers["resident_logical_ids"][:, 0] == -1)
    assert torch.all(buffers["newest_logical_ids"][:, 0] == -1)
    assert torch.all(buffers["resident_last_access"][:, 0] == 0)
    assert torch.equal(
        buffers["resident_logical_ids"][:, 1], preserved_resident_ids[:, 1]
    )
    assert torch.equal(buffers["newest_logical_ids"][:, 1], preserved_newest_ids[:, 1])
    assert torch.equal(
        buffers["resident_last_access"][:, 1], preserved_last_access[:, 1]
    )
    assert manager._row_request_ids == ["request-a", "request-b", None]
    assert buffers["request_active"].tolist() == [True, True, False]
    assert torch.all(buffers["resident_main_kv"] == 101)
    assert torch.all(buffers["newest_main_kv"] == 102)

    clear_targets = {
        buffers[name][:, 1].data_ptr(): name
        for name in (
            "resident_logical_ids",
            "newest_logical_ids",
            "resident_last_access",
        )
    }
    clear_calls = []
    original_fill = torch.Tensor.fill_
    original_zero = torch.Tensor.zero_

    def track_fill(tensor, value):
        name = clear_targets.get(tensor.data_ptr())
        if name is not None and value == -1:
            clear_calls.append(name)
        return original_fill(tensor, value)

    def track_zero(tensor):
        name = clear_targets.get(tensor.data_ptr())
        if name is not None:
            clear_calls.append(name)
        return original_zero(tensor)

    monkeypatch.setattr(torch.Tensor, "fill_", track_fill)
    monkeypatch.setattr(torch.Tensor, "zero_", track_zero)
    manager._prepare_decode_batch(
        ["request-a"],
        (),
        torch.tensor([0], dtype=torch.int32),
        (torch.tensor([[61, 62], [0, 0], [0, 0]], dtype=torch.int32),),
        torch.tensor([[1, 0, 0]], dtype=torch.int32),
        torch.tensor([38], dtype=torch.int32),
        3,
    )
    assert sorted(clear_calls) == [
        "newest_logical_ids",
        "resident_last_access",
        "resident_logical_ids",
    ]
    assert torch.all(buffers["resident_logical_ids"][:, 1] == -1)
    assert torch.all(buffers["newest_logical_ids"][:, 1] == -1)
    assert torch.all(buffers["resident_last_access"][:, 1] == 0)
    assert torch.equal(
        buffers["request_block_ids"],
        torch.tensor([[61, 62], [-1, -1], [-1, -1]], dtype=torch.int32),
    )
    assert buffers["request_num_blocks"].tolist() == [1, 0, 0]
    assert buffers["request_num_tokens"].tolist() == [38, 0, 0]
    assert buffers["request_active"].tolist() == [True, False, False]
    assert manager._row_request_ids == ["request-a", None, None]
    clear_calls.clear()
    manager._prepare_decode_batch(
        ["request-a"],
        (),
        torch.tensor([0], dtype=torch.int32),
        (torch.tensor([[61, 62], [0, 0], [0, 0]], dtype=torch.int32),),
        torch.tensor([[1, 0, 0]], dtype=torch.int32),
        torch.tensor([38], dtype=torch.int32),
        3,
    )
    assert not clear_calls

    manager._prepare_decode_batch(
        ["request-a", "request-b"],
        (),
        torch.tensor([0, 1], dtype=torch.int32),
        (torch.tensor([[71, 72], [81, 82]], dtype=torch.int32),),
        torch.tensor([[1, 1, 0]], dtype=torch.int32),
        torch.tensor([48, 49], dtype=torch.int32),
        2,
    )
    buffers["resident_logical_ids"].fill_(77)
    buffers["newest_logical_ids"].fill_(78)
    buffers["resident_last_access"].fill_(79)
    manager._prepare_decode_batch(
        ["request-b", "request-a"],
        (),
        torch.tensor([1, 0], dtype=torch.int32),
        (torch.tensor([[91, 92], [101, 102]], dtype=torch.int32),),
        torch.tensor([[1, 1, 0]], dtype=torch.int32),
        torch.tensor([59, 58], dtype=torch.int32),
        2,
    )
    assert torch.all(buffers["resident_logical_ids"][:, :2] == -1)
    assert torch.all(buffers["newest_logical_ids"][:, :2] == -1)
    assert torch.all(buffers["resident_last_access"][:, :2] == 0)
    assert manager._row_request_ids == ["request-b", "request-a", None]
    assert buffers["request_active"].tolist() == [True, True, False]
    assert torch.all(buffers["resident_main_kv"] == 101)
    assert torch.all(buffers["newest_main_kv"] == 102)

    unchanged = {
        name: tensor.clone()
        for name, tensor in buffers.items()
        if name not in generation_names
    }
    scheduler_output.total_num_scheduled_tokens = 0
    scheduler_output.num_scheduled_tokens = {}
    assert runner.execute_model(scheduler_output) == "no-forward"
    assert all(torch.equal(buffers[name], value) for name, value in unchanged.items())


def test_gpu_model_runner_v2_sparse_mla_fixed_capture_inventory(monkeypatch):
    import vllm.v1.worker.gpu.model_runner as model_runner_module

    generation_names = {
        "request_generation",
        "resident_generation",
        "newest_generation",
    }

    class GenerationFreeBuffers(dict):
        def __init__(self, buffers):
            super().__init__(buffers)
            self.generation_accesses = []
            self.legacy_generation = {
                "request_generation": torch.zeros(2, dtype=torch.int64),
                "resident_generation": torch.full((1, 2, 2), -1, dtype=torch.int64),
                "newest_generation": torch.full((1, 2, 2), -1, dtype=torch.int64),
            }

        def __missing__(self, name):
            if name in self.legacy_generation:
                self.generation_accesses.append(name)
                return self.legacy_generation[name]
            raise KeyError(name)

    manager = SparseMLAOffloadManager.__new__(SparseMLAOffloadManager)
    manager._closing = manager._closed = False
    manager._row_request_ids = [None, None]
    manager._local_buffers = GenerationFreeBuffers(
        {
            "resident_main_kv": torch.full((1, 2, 2, 2), 11, dtype=torch.float32),
            "resident_logical_ids": torch.full((1, 2, 2), 1, dtype=torch.int64),
            "resident_last_access": torch.full((1, 2, 2), 2, dtype=torch.int64),
            "newest_main_kv": torch.full((1, 2, 2, 2), 12, dtype=torch.float32),
            "newest_logical_ids": torch.full((1, 2, 2), 4, dtype=torch.int64),
            "request_block_ids": torch.full((2, 2), -1, dtype=torch.int32),
            "request_num_blocks": torch.zeros(2, dtype=torch.int32),
            "request_num_tokens": torch.zeros(2, dtype=torch.int32),
            "request_active": torch.zeros(2, dtype=torch.bool),
        }
    )
    inventory = {
        name: (tensor.data_ptr(), tensor.shape, tensor.stride())
        for name, tensor in manager._local_buffers.items()
    }
    inventory_names = set(manager._local_buffers)
    with monkeypatch.context() as no_readback:
        no_readback.setattr(
            torch.Tensor,
            "cpu",
            lambda self: (_ for _ in ()).throw(AssertionError("CPU readback")),
        )
        no_readback.setattr(
            torch.Tensor,
            "item",
            lambda self: (_ for _ in ()).throw(AssertionError("CPU readback")),
        )
        no_readback.setattr(
            torch.Tensor,
            "tolist",
            lambda self: (_ for _ in ()).throw(AssertionError("CPU readback")),
        )
        no_readback.setattr(
            torch.cuda,
            "synchronize",
            lambda *args: (_ for _ in ()).throw(AssertionError("synchronization")),
        )
        manager._prepare_decode_batch(
            ["request-a", "request-b"],
            (),
            torch.tensor([1, 0], dtype=torch.int32),
            (torch.tensor([[10, 11, 12, 13], [20, 21, 22, 23]], dtype=torch.int32),),
            torch.tensor([[2, 1]], dtype=torch.int32),
            torch.tensor([8, 7], dtype=torch.int32),
            2,
        )
    assert torch.equal(
        manager._local_buffers["request_block_ids"],
        torch.tensor([[10, 11], [20, 21]], dtype=torch.int32),
    )
    assert manager._local_buffers["request_num_blocks"].tolist() == [1, 2]
    assert manager._local_buffers["request_num_tokens"].tolist() == [8, 7]
    assert manager._local_buffers["request_active"].tolist() == [True, True]
    assert not manager._local_buffers.generation_accesses
    assert generation_names.isdisjoint(manager._local_buffers)
    assert set(manager._local_buffers) == inventory_names
    assert {
        name: (tensor.data_ptr(), tensor.shape, tensor.stride())
        for name, tensor in manager._local_buffers.items()
    } == inventory

    valid_idx = torch.tensor([0], dtype=torch.int32)
    valid_table = torch.tensor([[1, 2]], dtype=torch.int32)
    valid_counts = torch.tensor([[1, 0]], dtype=torch.int32)
    valid_tokens = torch.tensor([1], dtype=torch.int32)
    buffers_before_rejection = {
        name: tensor.clone() for name, tensor in manager._local_buffers.items()
    }
    malformed = (
        ((), valid_idx, (valid_table,), valid_counts, valid_tokens, 1),
        (("a",), valid_idx, (valid_table, valid_table), valid_counts, valid_tokens, 1),
        (
            ("a",),
            valid_idx.to(torch.int64),
            (valid_table,),
            valid_counts,
            valid_tokens,
            1,
        ),
        (
            ("a",),
            valid_idx,
            (valid_table.to(torch.int64),),
            valid_counts,
            valid_tokens,
            1,
        ),
        (
            ("a",),
            valid_idx,
            (valid_table.reshape(1, 1, 2),),
            valid_counts,
            valid_tokens,
            1,
        ),
        (
            ("a",),
            valid_idx,
            (valid_table[:, :1],),
            valid_counts,
            valid_tokens,
            1,
        ),
        (
            ("a",),
            valid_idx,
            (valid_table,),
            valid_counts[:, :1],
            valid_tokens,
            1,
        ),
        (
            ("a",),
            valid_idx,
            (valid_table,),
            valid_counts,
            torch.tensor([1, 0], dtype=torch.int32),
            1,
        ),
        (("a",), valid_idx, (valid_table,), valid_counts, valid_tokens, 0),
        (("a",), valid_idx, (valid_table,), valid_counts, valid_tokens, 3),
    )
    for req_ids, idx, tables, counts, tokens, padded in malformed:
        with pytest.raises(ValueError):
            manager._prepare_decode_batch(
                req_ids, (), idx, tables, counts, tokens, padded
            )
    with pytest.raises(ValueError):
        manager._prepare_decode_batch(
            ("a",),
            (),
            valid_idx.to(device="meta"),
            (valid_table,),
            valid_counts,
            valid_tokens,
            1,
        )
    assert all(
        torch.equal(manager._local_buffers[name], value)
        for name, value in buffers_before_rejection.items()
    )

    runner = model_runner_module.GPUModelRunner.__new__(
        model_runner_module.GPUModelRunner
    )
    runner.sparse_mla_offload_manager = manager
    runner.update_pp_decode_requests = lambda: None
    runner.finish_requests = lambda output: None
    runner.free_states = lambda output: None
    runner.add_requests = lambda output: None
    runner.update_requests = lambda output: None
    runner.block_tables = SimpleNamespace(
        apply_staged_writes=lambda: None,
        blocks_per_kv_block=[2],
        num_blocks=SimpleNamespace(gpu=valid_counts),
    )
    runner.req_states = SimpleNamespace(
        num_computed_tokens=SimpleNamespace(gpu=valid_tokens)
    )
    runner.cudagraph_manager = None
    runner.dp_size = runner.dp_rank = 1
    runner.lora_config = None
    runner.is_encoder_decoder = False
    runner.kv_cache_config = SimpleNamespace()
    runner.prepare_inputs = lambda output, desc: SimpleNamespace(
        req_ids=["a"], idx_mapping=valid_idx, num_reqs_after_padding=1
    )
    runner.prepare_attn = lambda batch: (
        (valid_table,),
        torch.zeros(1, dtype=torch.int64),
    )
    runner.model_state = SimpleNamespace(preprocess_state=lambda *args: None)
    monkeypatch.setattr(
        model_runner_module,
        "dispatch_cg_and_sync_dp",
        lambda *args, **kwargs: (SimpleNamespace(num_tokens=1), None),
    )
    scheduler_output = SimpleNamespace(
        scheduled_new_reqs=[SimpleNamespace(req_id="a")],
        num_scheduled_tokens={"a": 1},
        total_num_scheduled_tokens=1,
        scheduled_encoder_inputs={},
    )
    with pytest.raises(ValueError, match="blocks_per_kv_block"):
        runner.execute_model(scheduler_output)
    assert all(
        torch.equal(manager._local_buffers[name], value)
        for name, value in buffers_before_rejection.items()
    )


def _make_c6_attention(mqa_inputs):
    from vllm.model_executor.layers.attention.mla_attention import MLAAttention

    def forward_mqa(query, kv_cache, metadata, layer):
        mqa_inputs.append((query, kv_cache, metadata, layer))
        return torch.full((2, 1, 1), 7, dtype=torch.float32), None

    attention = MLAAttention.__new__(MLAAttention)
    attention.impl = SimpleNamespace(
        dcp_world_size=1, is_sparse=False, forward_mqa=forward_mqa
    )
    attention.kv_cache_dtype = "auto"
    attention.use_pcp = False
    attention.qk_nope_head_dim = attention.qk_rope_head_dim = 1
    attention.q_pad_num_heads = None
    attention.is_aiter_triton_fp4_bmm_enabled = False
    attention.is_aiter_triton_fp8_bmm_enabled = False
    attention.W_UK_T = torch.ones((1, 1, 1), dtype=torch.float32)
    attention.W_UK_T_dcp_qrep = None
    attention.num_heads = attention.v_head_dim = 1
    attention._v_up_proj = lambda attn_out, out: out.copy_(attn_out.squeeze(-1))
    attention.layer_name = "main.0"
    attention._sparse_mla_offload_view = None
    metadata = SimpleNamespace(
        num_actual_tokens=2,
        num_decodes=2,
        num_prefills=0,
        num_decode_tokens=2,
        decode=SimpleNamespace(),
    )
    return attention, metadata, torch.empty((2, 1), dtype=torch.float32)


def _make_c6_layer_view(layer_name="main.0"):
    from vllm.v1.worker.gpu.sparse_mla_offload import SparseMLALayerView

    return SparseMLALayerView(
        layer_name=layer_name,
        layer_index=0,
        is_host_writer=True,
        main_host_kv=torch.arange(2 * 576, dtype=torch.bfloat16).view(2, 576),
        main_host_kv_uva=None,
        local_buffers=MappingProxyType({}),
        side_stream=None,
        fork_ready_events=(),
        miss_ready_events=(),
    )


def test_sparse_mla_backend_routes_main_host_and_indexer_device(monkeypatch):
    import vllm.v1.attention.ops.flashmla as flashmla

    mqa_inputs: list[tuple] = []
    attention, metadata, output = _make_c6_attention(mqa_inputs)
    assert (
        attention.forward_impl(
            torch.ones((2, 1, 2)),
            torch.zeros((2, 1)),
            torch.zeros((2, 1, 1)),
            torch.zeros((1,)),
            metadata,
            output,
        )
        is output
    )
    assert torch.equal(output, torch.full((2, 1), 7, dtype=torch.float32))

    topk = torch.tensor([[[3, 4]], [[5, 6]], [[7, 8]]], dtype=torch.int32)
    dependency = torch.empty(0)
    c3_calls: list[tuple] = []

    def cache_plan(main, indices, layer_name):
        c3_calls.append(("plan", main, indices, layer_name))
        return dependency

    def offload_attention(query, main, out, layer_name, dep):
        c3_calls.append(("attention", query, main, out, layer_name, dep))
        out.fill_(11)

    monkeypatch.setattr(flashmla, "sparse_mla_cache_plan", cache_plan)
    monkeypatch.setattr(flashmla, "sparse_mla_offload_attention", offload_attention)
    attention.impl.is_sparse = True
    attention.impl.topk_indices_buffer = topk
    attention._sparse_mla_offload_view = _make_c6_layer_view()
    output.zero_()
    result = attention.forward_impl(
        torch.ones((2, 1, 2)),
        torch.ones((2, 512), dtype=torch.bfloat16),
        torch.full((2, 1, 64), 2, dtype=torch.bfloat16),
        torch.zeros((1,)),
        metadata,
        output,
    )

    assert result is output
    assert [call[0] for call in c3_calls] == ["plan", "attention"]
    _, main, indices, layer_name = c3_calls[0]
    assert main.shape == (2, 576) and main.dtype is torch.bfloat16
    assert torch.equal(main[:, :512], torch.ones((2, 512), dtype=torch.bfloat16))
    assert torch.equal(main[:, 512:], torch.full((2, 64), 2, dtype=torch.bfloat16))
    assert indices.shape == (3, 1, 2) and indices.dtype is torch.int32
    assert indices.is_contiguous() and indices.data_ptr() == topk.data_ptr()
    assert layer_name == "main.0" and c3_calls[1][2] is main
    assert c3_calls[1][-1] is dependency
    assert torch.equal(output, torch.full((2, 1), 11, dtype=torch.float32))

    c3_call_count = len(c3_calls)
    with pytest.raises(RuntimeError, match="sparse MLA offload Main"):
        attention.forward_impl(
            torch.ones((2, 1, 2)),
            torch.ones((2, 512), dtype=torch.bfloat16),
            torch.ones((2, 1, 64), dtype=torch.float32),
            torch.zeros((1,)),
            metadata,
            output,
        )
    assert len(c3_calls) == c3_call_count

    attention._sparse_mla_offload_view = _make_c6_layer_view("other")
    with pytest.raises(RuntimeError, match="sparse MLA offload"):
        attention.forward_impl(
            torch.ones((2, 1, 2)),
            torch.ones((2, 512), dtype=torch.bfloat16),
            torch.ones((2, 1, 64), dtype=torch.bfloat16),
            torch.zeros((1,)),
            metadata,
            output,
        )
    assert len(mqa_inputs) == 1


def _make_c6_full_decode_seed(monkeypatch):
    import vllm.v1.worker.gpu.model_runner as model_runner_module
    from vllm.config.compilation import CUDAGraphMode
    from vllm.v1.core.sched.output import CachedRequestData, SchedulerOutput
    from vllm.v1.worker.gpu.cudagraph_utils import BatchExecutionDescriptor

    events: list[tuple] = []
    req_id_per_token = torch.tensor([7, 8, 9], dtype=torch.int32)
    fence_token = torch.zeros(1, dtype=torch.int32)

    def prepare_decode_batch(*args):
        events.append(("publish", tuple(args[5].tolist())))

    manager = SimpleNamespace(
        _local_buffers={"tp_fence_token": fence_token},
        _prepare_decode_batch=prepare_decode_batch,
    )
    manager.layer_view = lambda name: SimpleNamespace(
        local_buffers=manager._local_buffers
    )
    runner = model_runner_module.GPUModelRunner.__new__(
        model_runner_module.GPUModelRunner
    )
    runner.sparse_mla_offload_manager = manager
    runner._sparse_mla_tp_fence_token = fence_token
    runner.execute_model_state = None
    runner.update_pp_decode_requests = lambda: None
    runner.finish_requests = runner.free_states = runner.add_requests = (
        lambda output: None
    )
    runner.update_requests = lambda output: None
    runner.block_tables = SimpleNamespace(
        apply_staged_writes=lambda: None,
        blocks_per_kv_block=[1],
        num_blocks=SimpleNamespace(gpu=torch.ones((1, 1), dtype=torch.int32)),
    )
    runner.req_states = SimpleNamespace(
        num_computed_tokens=SimpleNamespace(gpu=torch.ones(1, dtype=torch.int32))
    )
    runner.dp_size = runner.dp_rank = 1
    runner.lora_config = None
    runner.is_encoder_decoder = runner.is_encoder_only = False
    runner.supports_mm_inputs = False
    runner.is_first_pp_rank = runner.is_last_pp_rank = True
    runner.use_aux_hidden_state_outputs = False
    runner.routed_experts_capturer = None
    runner.kv_cache_config = SimpleNamespace(kv_cache_groups=[])
    runner.attn_groups = []
    runner.model_config = SimpleNamespace()
    runner.eplb = SimpleNamespace(prepare_forward=lambda *args: None)
    runner.vllm_config = SimpleNamespace(num_speculative_tokens=0)
    input_batch = SimpleNamespace(
        req_ids=["decode"],
        idx_mapping=torch.zeros(1, dtype=torch.int32),
        num_reqs_after_padding=1,
        num_tokens=1,
        num_tokens_after_padding=3,
        input_ids=torch.zeros(3, dtype=torch.int64),
        positions=torch.arange(3),
        seq_lens=torch.tensor([2], dtype=torch.int32),
        is_padding=torch.tensor([False, True, True]),
        num_draft_tokens=0,
    )
    runner.prepare_inputs = lambda output, desc: input_batch
    runner.prepare_attn = lambda batch: (
        (torch.zeros(1, dtype=torch.int64),),
        torch.zeros(1),
    )

    def prepare_model_state_attn(
        _input_batch,
        _cudagraph_mode,
        _block_tables,
        _slot_mappings,
        _attn_groups,
        _kv_cache_config,
        for_capture=False,
    ):
        _ = for_capture
        return {"main.0": SimpleNamespace(req_id_per_token=req_id_per_token)}

    runner.model_state = SimpleNamespace(
        preprocess_state=lambda *args: None,
        prepare_attn=prepare_model_state_attn,
        prepare_inputs=lambda *args: {},
    )
    runner.kv_connector = SimpleNamespace(
        pre_forward=lambda output: events.append(("pre_forward",))
    )

    def run_fullgraph(desc):
        events.append(("target", tuple(req_id_per_token.tolist())))
        return torch.zeros((3, 1))

    runner.cudagraph_manager = SimpleNamespace(run_fullgraph=run_fullgraph)
    scheduler_output = SchedulerOutput(
        scheduled_new_reqs=[],
        scheduled_cached_reqs=CachedRequestData(
            req_ids=["decode"],
            resumed_req_ids=set(),
            new_token_ids=[],
            all_token_ids={},
            new_block_ids=[],
            num_computed_tokens=[],
            num_output_tokens=[1],
        ),
        num_scheduled_tokens={"decode": 1},
        total_num_scheduled_tokens=1,
        scheduled_spec_decode_tokens={},
        scheduled_encoder_inputs={},
        num_common_prefix_blocks=[],
        finished_req_ids=set(),
        free_encoder_mm_hashes=[],
    )
    tp_group = SimpleNamespace(
        broadcast=lambda token, src: events.append(("broadcast", token, src))
    )
    monkeypatch.setattr(model_runner_module, "get_tp_group", lambda: tp_group)
    monkeypatch.setattr(
        model_runner_module,
        "dispatch_cg_and_sync_dp",
        lambda *args, **kwargs: (
            BatchExecutionDescriptor(CUDAGraphMode.FULL, 3, 1),
            None,
        ),
    )
    monkeypatch.setattr(
        model_runner_module, "build_slot_mappings_by_layer", lambda *args: {}
    )
    return runner, scheduler_output, tp_group, events, req_id_per_token, fence_token


def test_gpu_model_runner_v2_sparse_mla_full_decode_fence_and_tail(monkeypatch):
    runner, scheduler_output, _, events, req_ids, fence_token = (
        _make_c6_full_decode_seed(monkeypatch)
    )

    runner.execute_model(scheduler_output)

    assert events[0] == ("publish", (2,))
    assert [event[0] for event in events[1:]] == [
        "pre_forward",
        "broadcast",
        "target",
    ]
    assert events[2][1] is fence_token and events[2][2] == 0
    assert events[3][1] == (7, -1, -1)
    assert tuple(req_ids.tolist()) == (7, -1, -1)

    rejected_runner, rejected_output, _, rejected_events, _, _ = (
        _make_c6_full_decode_seed(monkeypatch)
    )
    mutations = []
    rejected_runner.update_pp_decode_requests = lambda: mutations.append("update_pp")
    rejected_runner.finish_requests = lambda output: mutations.append("finish")
    rejected_runner.free_states = lambda output: mutations.append("free")
    rejected_runner.add_requests = lambda output: mutations.append("add")
    rejected_runner.update_requests = lambda output: mutations.append("update")
    rejected_runner.block_tables.apply_staged_writes = lambda: mutations.append(
        "blocks"
    )
    rejected_output = replace(
        rejected_output,
        scheduled_cached_reqs=replace(
            rejected_output.scheduled_cached_reqs, num_output_tokens=[0]
        ),
    )
    with pytest.raises(ValueError, match="only real non-MTP pure Decode"):
        rejected_runner.execute_model(rejected_output)
    assert mutations == []
    assert rejected_events == []


def test_gpu_model_runner_v2_sparse_mla_broadcast_failure_stops_target(monkeypatch):
    runner, scheduler_output, tp_group, events, _, _ = _make_c6_full_decode_seed(
        monkeypatch
    )
    tp_group.broadcast = lambda token, src: (_ for _ in ()).throw(
        RuntimeError("fence failed")
    )

    with pytest.raises(RuntimeError, match="fence failed"):
        runner.execute_model(scheduler_output)

    assert events == [("publish", (2,)), ("pre_forward",)]
    assert runner.execute_model_state is None


def test_gpu_model_runner_v2_sparse_mla_mtp_fence_order(monkeypatch):
    import vllm.v1.worker.gpu.model_runner as model_runner_module
    from vllm.config.compilation import CUDAGraphMode
    from vllm.v1.core.sched.output import CachedRequestData
    from vllm.v1.worker.gpu.cudagraph_utils import BatchExecutionDescriptor
    from vllm.v1.worker.gpu.spec_decode.autoregressive import speculator as spec_module
    from vllm.v1.worker.gpu.spec_decode.mtp.speculator import MTPSpeculator

    rejected_runner, scheduler_output, _, rejected_events, _, _ = (
        _make_c6_full_decode_seed(monkeypatch)
    )
    mtp_output = replace(
        scheduler_output,
        num_scheduled_tokens={"decode": 3},
        total_num_scheduled_tokens=3,
        scheduled_spec_decode_tokens={"decode": [11, 12]},
    )
    rejected_runner.speculator = SimpleNamespace()
    rejected_runner.vllm_config = SimpleNamespace(num_speculative_tokens=2)
    malformed = (
        mtp_output,
        replace(mtp_output, scheduled_new_reqs=[SimpleNamespace(req_id="new")]),
        replace(
            mtp_output,
            scheduled_cached_reqs=CachedRequestData(
                req_ids=["decode", "context"],
                resumed_req_ids=set(),
                new_token_ids=[],
                all_token_ids={},
                new_block_ids=[],
                num_computed_tokens=[],
                num_output_tokens=[1, 0],
            ),
            num_scheduled_tokens={"decode": 3, "context": 1},
            total_num_scheduled_tokens=4,
        ),
    )
    for output in malformed:
        with pytest.raises(ValueError, match="sparse MLA offload"):
            rejected_runner.execute_model(output)
        assert rejected_events == []

    runner, _, tp_group, events, _, fence_token = _make_c6_full_decode_seed(monkeypatch)
    manager = runner.sparse_mla_offload_manager
    assert manager is not None
    manager._finalize_mtp_batch = lambda idx, computed: events.append(
        ("finalize", tuple(computed.tolist()))
    )

    def fence_mtp_step(step):
        fence_token.copy_(step.to(dtype=torch.int32))
        tp_group.broadcast(fence_token, src=0)

    manager._fence_mtp_step = fence_mtp_step
    input_batch = runner.prepare_inputs(None, None)
    input_batch.num_reqs = 1
    input_batch.num_tokens = input_batch.num_tokens_after_padding = 3
    input_batch.input_ids = torch.zeros(3, dtype=torch.int64)
    input_batch.positions = torch.arange(3, dtype=torch.int64)
    input_batch.is_padding = torch.zeros(3, dtype=torch.bool)
    input_batch.num_draft_tokens = 2
    input_batch.num_scheduled_tokens = torch.tensor([3], dtype=torch.int32)
    input_batch.seq_lens = torch.tensor([3], dtype=torch.int32)
    input_batch.seq_lens_cpu_upper_bound = torch.tensor([3], dtype=torch.int32)
    input_batch.query_start_loc = torch.tensor([0, 3], dtype=torch.int32)
    runner.prepare_inputs = lambda output, desc: input_batch
    runner.vllm_config = SimpleNamespace(num_speculative_tokens=2)
    runner.num_speculative_steps = 3
    runner.eplb.step = lambda **kwargs: None
    runner.req_states.num_computed_tokens.gpu = torch.zeros(1, dtype=torch.int32)
    runner.req_states.last_sampled_tokens = torch.zeros(1, dtype=torch.int64)
    runner.req_states.next_prefill_tokens = torch.zeros(1, dtype=torch.int64)
    runner.req_states.draft_tokens = torch.zeros((1, 3), dtype=torch.int64)
    runner.req_states.all_token_ids = SimpleNamespace(
        gpu=torch.zeros((1, 3), dtype=torch.int64)
    )
    runner.req_states.prompt_len = SimpleNamespace(np=torch.zeros(1, dtype=torch.int32))
    runner.is_last_pp_rank = True
    runner.pp_handler = None
    runner.pcp_manager = None
    runner.model = SimpleNamespace(compute_logits=lambda hidden: hidden)
    runner.prompt_logprobs_worker = SimpleNamespace(
        compute_prompt_logprobs=lambda *args: {}
    )
    runner.sampler = SimpleNamespace(
        sampling_states=SimpleNamespace(
            temperature=SimpleNamespace(gpu=torch.ones(1)),
            seeds=SimpleNamespace(gpu=torch.zeros(1, dtype=torch.int64)),
        )
    )
    runner.main_stream = runner.output_copy_stream = runner.check_ep_fault = None
    runner.draft_tokens_handler = SimpleNamespace(set_draft_tokens=lambda *args: None)
    runner.kv_connector.post_forward = lambda finished: None

    def sample(hidden, batch, grammar):
        events.append(("sample",))
        return (
            SimpleNamespace(sampled_token_ids=torch.zeros((1, 1), dtype=torch.int64)),
            torch.ones(1, dtype=torch.int32),
            torch.zeros(1, dtype=torch.int32),
        )

    runner.sample = sample

    def postprocess(idx, sampled, num_sampled, num_rejected, query_start_loc):
        events.append(("postprocess",))
        runner.req_states.num_computed_tokens.gpu.fill_(2)

    runner.postprocess_sampled = postprocess
    monkeypatch.setattr(
        model_runner_module,
        "AsyncOutput",
        lambda **kwargs: SimpleNamespace(**kwargs),
    )
    monkeypatch.setattr(
        model_runner_module.pcp,
        "maybe_restore_pcp_for_sampling",
        lambda manager, hidden, batch: (hidden, batch),
    )
    monkeypatch.setattr(
        model_runner_module,
        "dispatch_cg_and_sync_dp",
        lambda *args, **kwargs: (
            BatchExecutionDescriptor(CUDAGraphMode.FULL, 3, 1),
            None,
        ),
    )

    speculator = object.__new__(MTPSpeculator)
    speculator._sparse_mla_offload_manager = manager
    speculator.supports_mm_inputs = False
    speculator.num_speculative_steps = 3
    speculator.max_model_len = 8
    speculator.max_num_reqs = 1
    speculator.dp_size = speculator.dp_rank = 1
    speculator.prefill_cudagraph_manager = None
    speculator.decode_cudagraph_manager = None
    speculator.current_draft_step = torch.zeros((), dtype=torch.int64)
    speculator.last_token_indices = torch.zeros(1, dtype=torch.int64)
    speculator.input_buffers = SimpleNamespace(
        input_ids=torch.zeros(3, dtype=torch.int64),
        positions=torch.zeros(3, dtype=torch.int64),
        query_start_loc=torch.tensor([0, 1], dtype=torch.int32),
        seq_lens=torch.ones(1, dtype=torch.int32),
    )
    speculator.idx_mapping = torch.zeros(1, dtype=torch.int64)
    speculator.hidden_states = torch.zeros((3, 1))
    speculator.draft_tokens = torch.zeros((1, 3), dtype=torch.int64)
    speculator.temperature = torch.ones(1)
    speculator.seeds = torch.zeros(1, dtype=torch.int64)
    speculator.draft_logits = torch.zeros((1, 3, 1))
    speculator.vllm_config = SimpleNamespace()
    speculator.block_tables = SimpleNamespace(
        compute_slot_mappings=lambda *args: torch.zeros(1, dtype=torch.int64)
    )
    speculator.kv_cache_config = SimpleNamespace()
    speculator._copy_request_inputs = lambda *args: None
    speculator._prepare_eplb_forward = lambda *args: None
    speculator._build_draft_attn_metadata = lambda **kwargs: {}
    speculator.sample_draft = lambda *args: torch.zeros(1, dtype=torch.int64)

    class DraftModel:
        def __call__(self, **kwargs):
            events.append(("mtp_call", int(fence_token)))
            return torch.zeros((3, 1))

    speculator.model = DraftModel()
    runner.speculator = speculator
    monkeypatch.setattr(
        spec_module,
        "prepare_prefill_inputs",
        lambda *args: speculator.current_draft_step.zero_(),
    )
    monkeypatch.setattr(
        spec_module, "prepare_decode_inputs", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(
        spec_module, "update_draft_inputs", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(
        spec_module,
        "dispatch_cg_and_sync_dp",
        lambda *args, **kwargs: (
            BatchExecutionDescriptor(CUDAGraphMode.NONE, 1, 1),
            None,
        ),
    )
    monkeypatch.setattr(spec_module, "build_slot_mappings_by_layer", lambda *args: {})
    monkeypatch.setattr(
        spec_module, "set_forward_context", lambda *args, **kwargs: nullcontext()
    )

    broadcast_tokens = []

    def broadcast(token, src):
        broadcast_tokens.append((token.data_ptr(), int(token), src))
        events.append(("broadcast", int(token)))

    tp_group.broadcast = broadcast
    runner.cudagraph_manager.run_fullgraph = lambda desc: (
        events.append(("target", int(fence_token))) or torch.zeros((3, 1))
    )
    runner.execute_model(mtp_output)
    target_state = runner.execute_model_state
    assert target_state is not None
    runner.sample_tokens(None)

    assert [event[0] for event in events] == [
        "publish",
        "pre_forward",
        "broadcast",
        "target",
        "sample",
        "postprocess",
        "finalize",
        "broadcast",
        "mtp_call",
        "broadcast",
        "mtp_call",
        "broadcast",
        "mtp_call",
    ]
    assert [event[1] for event in events if event[0] == "broadcast"] == [-1, 0, 1, 2]
    assert events[6] == ("finalize", (2,))
    assert all(pointer == fence_token.data_ptr() for pointer, _, _ in broadcast_tokens)
    assert all(src == 0 for _, _, src in broadcast_tokens)

    def fail_second_mtp_broadcast(token, src):
        events.append(("broadcast", int(token)))
        if int(token) == 1:
            raise RuntimeError("step-one fence failed")

    events.clear()
    tp_group.broadcast = fail_second_mtp_broadcast
    with pytest.raises(RuntimeError, match="step-one fence failed"):
        speculator.propose(
            input_batch,
            target_state.attn_metadata,
            target_state.slot_mappings_by_layer,
            target_state.hidden_states,
            target_state.aux_hidden_states,
            torch.ones(1, dtype=torch.int32),
            torch.zeros(1, dtype=torch.int32),
            runner.req_states.last_sampled_tokens,
            runner.req_states.next_prefill_tokens,
            runner.sampler.sampling_states.temperature.gpu,
            runner.sampler.sampling_states.seeds.gpu,
        )
    assert events == [
        ("broadcast", 0),
        ("mtp_call", 0),
        ("broadcast", 1),
    ]
