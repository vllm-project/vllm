# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Focused tests for PCP symmetric-memory peer allocation and direct-final writes."""

import multiprocessing as mp
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist

from vllm.utils.network_utils import get_open_port
from vllm.utils.system_utils import update_environment_variables

mp.set_start_method("spawn", force=True)


def _distributed_run(fn, world_size: int) -> None:
    port = str(get_open_port())
    processes: list[mp.Process] = []
    for rank in range(world_size):
        env = {
            "RANK": str(rank),
            "LOCAL_RANK": str(rank),
            "WORLD_SIZE": str(world_size),
            "LOCAL_WORLD_SIZE": str(world_size),
            "MASTER_ADDR": "localhost",
            "MASTER_PORT": port,
        }
        process = mp.Process(target=fn, args=(env,))
        processes.append(process)
        process.start()

    for process in processes:
        process.join(timeout=180)

    for process in processes:
        if process.is_alive():
            process.kill()
            process.join()
        assert process.exitcode == 0


def _worker_peer_allocation(env: dict[str, str]) -> None:
    update_environment_variables(env)
    rank = int(env["RANK"])
    torch.accelerator.set_device_index(rank)
    dist.init_process_group(
        backend="nccl", rank=rank, world_size=int(env["WORLD_SIZE"])
    )
    from vllm.distributed.device_communicators.symm_mem import (
        allocate_symmetric_memory,
        rendezvous_symmetric_memory,
    )

    device = torch.device(f"cuda:{rank}")
    second_group = dist.new_group(ranks=list(range(dist.get_world_size())))
    allocation = allocate_symmetric_memory(
        (4096,), torch.int8, device, dist.group.WORLD
    )
    handle = allocation.handle
    second_handle = rendezvous_symmetric_memory(allocation.storage, second_group)

    assert int(handle.rank) == rank
    assert int(handle.world_size) == dist.get_world_size()
    assert len(handle.buffer_ptrs) == dist.get_world_size()
    assert all(int(ptr or 0) != 0 for ptr in handle.buffer_ptrs)
    assert int(handle.buffer_ptrs_dev or 0) != 0
    assert int(second_handle.buffer_ptrs_dev or 0) != 0
    assert (
        int(handle.buffer_ptrs[rank]) + int(handle.offset)
        == allocation.storage.data_ptr()
    )

    view = allocation.storage.view(torch.int32)[16:32]
    assert view.data_ptr() == allocation.storage.data_ptr() + 64

    offset = 64 + rank * 16
    sentinel = torch.full((16,), rank + 7, dtype=torch.int8, device=device)
    for peer in range(dist.get_world_size()):
        peer_view = handle.get_remote_tensor(peer, (4096,), torch.int8)
        assert peer_view.data_ptr() == int(handle.buffer_ptrs[peer]) + int(
            handle.offset
        )
        peer_view[offset : offset + 16].copy_(sentinel)

    dist.barrier()

    for source in range(dist.get_world_size()):
        source_offset = 64 + source * 16
        expected = torch.full((16,), source + 7, dtype=torch.int8, device=device)
        assert torch.equal(
            allocation.storage[source_offset : source_offset + 16], expected
        )

    torch.accelerator.synchronize()
    del peer_view, view, second_handle, handle, allocation
    dist.destroy_process_group(second_group)
    dist.destroy_process_group()


def _worker_barrier_ubatch_graph_replay(env: dict[str, str]) -> None:
    update_environment_variables(env)
    rank = int(env["RANK"])
    torch.accelerator.set_device_index(rank)
    dist.init_process_group(
        backend="nccl", rank=rank, world_size=int(env["WORLD_SIZE"])
    )
    import vllm.model_executor.layers.attention.direct_kv as direct_kv

    device = torch.device(f"cuda:{rank}")
    active_ubatch = [0]
    domain = direct_kv.KVCacheSymmMemDomain(
        SimpleNamespace(
            device_group=dist.group.WORLD,
            world_size=dist.get_world_size(),
            rank_in_group=rank,
        ),
        num_barriers=2,
        barrier_index_provider=lambda: active_ubatch[0],
    )
    from vllm.distributed.device_communicators.symm_mem import (
        allocate_symmetric_memory_storage,
    )

    storage = allocate_symmetric_memory_storage(1, torch.int8, device)
    domain.rendezvous(storage)
    domain.finalize({"cache": storage}, storage)
    epoch = domain._barrier_epoch
    assert epoch is not None
    # Compile both ubatch specializations before graph capture.
    domain.barrier(storage)
    active_ubatch[0] = 1
    domain.barrier(storage)
    torch.accelerator.synchronize()
    dist.barrier()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        domain.barrier(storage)

    torch.accelerator.synchronize()
    dist.barrier()
    epoch_before = epoch.cpu().tolist()

    active_ubatch[0] = 0
    domain.barrier(storage)
    torch.accelerator.synchronize()
    assert epoch.cpu().tolist() == [epoch_before[0] + 1, epoch_before[1]]

    for replay in range(1, 3):
        graph.replay()
        torch.accelerator.synchronize()
        assert epoch.cpu().tolist() == [
            epoch_before[0] + 1,
            epoch_before[1] + replay,
        ]

    domain.close()
    dist.destroy_process_group()


@pytest.mark.skipif(torch.accelerator.device_count() < 2, reason="needs 2+ GPUs")
def test_symm_mem_peer_allocation_two_gpu():
    _distributed_run(_worker_peer_allocation, world_size=2)


@pytest.mark.skipif(torch.accelerator.device_count() < 2, reason="needs 2+ GPUs")
def test_barrier_ubatch_indices_and_cuda_graph_replay():
    _distributed_run(_worker_barrier_ubatch_graph_replay, world_size=2)


def test_kv_cache_symm_mem_view_uses_native_handle_metadata():
    from vllm.model_executor.layers.attention.direct_kv import (
        KVCacheSymmMemView,
        _peer_view_offset_bytes,
    )

    storage = torch.empty(128, dtype=torch.uint8)
    handle = SimpleNamespace(offset=1024, buffer_ptrs_dev=1234, world_size=4)
    view = storage[16:64:2]
    symm_mem_view = KVCacheSymmMemView(
        peer_ptrs=int(handle.buffer_ptrs_dev),
        offset_bytes=_peer_view_offset_bytes(storage, handle, view),
    )

    assert symm_mem_view.offset_bytes == 1040
    assert symm_mem_view.peer_ptrs == 1234


def test_kv_cache_owns_symm_mem_domains(monkeypatch):
    import vllm.model_executor.layers.attention.direct_kv as direct_kv
    from vllm.distributed.device_communicators.symm_mem import (
        SymmetricMemoryAllocation,
    )
    from vllm.v1.worker import utils as worker_utils
    from vllm.v1.worker.utils import KVCache

    storage = torch.empty(128, dtype=torch.uint8)
    pcp_handle = SimpleNamespace(
        offset=1024,
        buffer_ptrs_dev=1234,
    )
    tp_handle = SimpleNamespace(
        offset=1024,
        buffer_ptrs_dev=4321,
    )

    pcp_signal_storage = torch.empty((1, 2, 2), dtype=torch.int32)
    pcp_signal_allocation = SymmetricMemoryAllocation(
        storage=pcp_signal_storage,
        handle=SimpleNamespace(offset=2048, buffer_ptrs_dev=5678),
    )
    tp_signal_storage = torch.empty((1, 2, 4), dtype=torch.int32)
    tp_signal_allocation = SymmetricMemoryAllocation(
        storage=tp_signal_storage,
        handle=SimpleNamespace(offset=4096, buffer_ptrs_dev=8765),
    )
    signal_allocations = iter((pcp_signal_allocation, tp_signal_allocation))

    pcp_group = object()
    tp_group = object()
    monkeypatch.setattr(
        worker_utils,
        "allocate_symmetric_memory_storage",
        lambda *_args, **_kwargs: storage,
    )

    def allocate_kv_cache(*_args, buffer_allocator, **_kwargs):
        backing = buffer_allocator(128, torch.device("cpu"))
        return {"layer": backing[16:64]}

    monkeypatch.setattr(worker_utils, "allocate_kv_cache", allocate_kv_cache)
    monkeypatch.setattr(
        direct_kv,
        "rendezvous_symmetric_memory",
        lambda _storage, group: pcp_handle if group is pcp_group else tp_handle,
    )
    monkeypatch.setattr(
        direct_kv,
        "allocate_symmetric_memory",
        lambda *_args, **_kwargs: next(signal_allocations),
    )

    state = SimpleNamespace(barrier_calls=[], synchronizations=0)

    def fake_barrier(
        mla_kv_cache,
        indexer_k_cache,
        signals,
        epoch,
        peer_ptrs,
        offset_bytes,
        source_rank,
        world_size,
        barrier_index,
    ):
        state.barrier_calls.append((peer_ptrs, world_size))
        assert mla_kv_cache is storage
        assert indexer_k_cache is None
        assert barrier_index == 0
        assert epoch.numel() == 1

        if world_size == 2:
            assert signals is pcp_signal_storage
            assert peer_ptrs == 5678
            assert offset_bytes == 2048
            assert source_rank == 1
        else:
            assert signals is tp_signal_storage
            assert peer_ptrs == 8765
            assert offset_bytes == 4096
            assert source_rank == 2

    monkeypatch.setattr(torch.ops.vllm, "direct_kv_barrier", fake_barrier)
    monkeypatch.setattr(
        torch.accelerator,
        "synchronize",
        lambda device: setattr(state, "synchronizations", state.synchronizations + 1),
    )

    pcp_domain = direct_kv.KVCacheSymmMemDomain(
        SimpleNamespace(device_group=pcp_group, world_size=2, rank_in_group=1),
    )
    tp_domain = direct_kv.KVCacheSymmMemDomain(
        SimpleNamespace(device_group=tp_group, world_size=4, rank_in_group=2),
    )
    kv_cache = KVCache(pcp_domain=pcp_domain, tp_domain=tp_domain)

    kv_caches = kv_cache.allocate(object(), torch.device("cpu"), object())
    assert kv_caches["layer"].data_ptr() == storage[16:64].data_ptr()
    assert kv_cache.storage is storage

    assert pcp_domain.view("layer").offset_bytes == 1040
    assert pcp_domain.view("layer").peer_ptrs == 1234
    assert tp_domain.view("layer").offset_bytes == 1040
    assert tp_domain.view("layer").peer_ptrs == 4321

    pcp_domain.barrier(storage)
    tp_domain.barrier(storage)
    assert state.barrier_calls == [(5678, 2), (8765, 4)]

    kv_cache.tensors.append(storage)
    kv_cache.close()
    assert state.synchronizations == 2
    assert not kv_cache.tensors
    assert kv_cache.storage is None
    assert kv_cache.pcp_domain is None
    assert kv_cache.tp_domain is None
    with pytest.raises(RuntimeError, match="Missing direct-KV view"):
        pcp_domain.view("layer")
    with pytest.raises(RuntimeError, match="Missing direct-KV view"):
        tp_domain.view("layer")


def _pcp_direct_kv_config(pcp_size: int = 2):
    from vllm.config import CUDAGraphMode

    return SimpleNamespace(
        parallel_config=SimpleNamespace(
            prefill_context_parallel_size=pcp_size,
            decode_context_parallel_size=1,
            data_parallel_size=1,
            use_ubatching=False,
        ),
        model_config=SimpleNamespace(enable_sleep_mode=False),
        compilation_config=SimpleNamespace(
            static_forward_context={
                "attn": SimpleNamespace(
                    use_pcp=True,
                    supports_direct_kv=True,
                ),
                "mamba": SimpleNamespace(
                    use_pcp=False,
                    supports_direct_kv=False,
                ),
            },
            cudagraph_mode=CUDAGraphMode.NONE,
        ),
        scheduler_config=SimpleNamespace(async_scheduling=False),
        cache_config=SimpleNamespace(
            cache_dtype="fp8_ds_mla", enable_prefix_caching=False
        ),
        kv_transfer_config=None,
    )


def test_pcp_direct_kv_auto_selects_supported_config(monkeypatch):
    import vllm.v1.worker.gpu.pcp_manager as pcp

    monkeypatch.delenv("VLLM_USE_PCP_DIRECT_KV", raising=False)
    monkeypatch.setattr(pcp.current_platform, "is_cuda", lambda: True)
    monkeypatch.setattr(pcp, "symm_mem_available", True)

    assert pcp.use_pcp_direct_kv(_pcp_direct_kv_config())


def test_pcp_direct_kv_does_not_gate_unrelated_features(monkeypatch):
    import vllm.v1.worker.gpu.pcp_manager as pcp
    from vllm.config import CUDAGraphMode

    monkeypatch.delenv("VLLM_USE_PCP_DIRECT_KV", raising=False)
    monkeypatch.setattr(pcp.current_platform, "is_cuda", lambda: True)
    monkeypatch.setattr(pcp, "symm_mem_available", True)

    config = _pcp_direct_kv_config()
    config.parallel_config.data_parallel_size = 2
    config.parallel_config.use_ubatching = True
    config.scheduler_config.async_scheduling = True
    config.compilation_config.cudagraph_mode = CUDAGraphMode.PIECEWISE
    config.cache_config.enable_prefix_caching = True
    config.kv_transfer_config = SimpleNamespace(kv_connector="NixlConnector")
    config.model_config.enable_sleep_mode = True

    assert pcp.use_pcp_direct_kv(config)


def test_pcp_direct_kv_requires_symmetric_memory(monkeypatch):
    import vllm.v1.worker.gpu.pcp_manager as pcp

    monkeypatch.delenv("VLLM_USE_PCP_DIRECT_KV", raising=False)
    monkeypatch.setattr(pcp.current_platform, "is_cuda", lambda: True)
    monkeypatch.setattr(pcp, "symm_mem_available", False)

    assert not pcp.use_pcp_direct_kv(_pcp_direct_kv_config())


def test_pcp_direct_kv_rejects_participating_unsupported_layer(monkeypatch):
    import vllm.v1.worker.gpu.pcp_manager as pcp

    monkeypatch.delenv("VLLM_USE_PCP_DIRECT_KV", raising=False)
    monkeypatch.setattr(pcp.current_platform, "is_cuda", lambda: True)

    config = _pcp_direct_kv_config()
    config.compilation_config.static_forward_context["unsupported"] = SimpleNamespace(
        use_pcp=True,
        supports_direct_kv=False,
    )

    assert not pcp.use_pcp_direct_kv(config)


def test_pcp_direct_kv_auto_skips_unsupported_config(monkeypatch):
    import vllm.v1.worker.gpu.pcp_manager as pcp

    monkeypatch.delenv("VLLM_USE_PCP_DIRECT_KV", raising=False)
    assert not pcp.use_pcp_direct_kv(_pcp_direct_kv_config(pcp_size=1))


def test_pcp_direct_kv_disabled_skips_support_checks(monkeypatch):
    import vllm.v1.worker.gpu.pcp_manager as pcp

    monkeypatch.setenv("VLLM_USE_PCP_DIRECT_KV", "0")
    assert not pcp.use_pcp_direct_kv(object())


def test_pcp_direct_kv_forced_without_pcp_warns_and_disables(monkeypatch, caplog):
    import vllm.v1.worker.gpu.pcp_manager as pcp

    monkeypatch.setenv("VLLM_USE_PCP_DIRECT_KV", "1")
    result = pcp.maybe_build_pcp_manager(
        _pcp_direct_kv_config(pcp_size=1),
        torch.device("cpu"),
        supports_mm_inputs=False,
        req_states=None,
        block_tables=None,
    )

    assert result == (None, None)
    assert "VLLM_USE_PCP_DIRECT_KV=1 was ignored because PCP is disabled" in caplog.text


def test_pcp_direct_kv_forced_with_dcp_warns_and_disables(monkeypatch, caplog):
    import vllm.v1.worker.gpu.pcp_manager as pcp

    monkeypatch.setenv("VLLM_USE_PCP_DIRECT_KV", "1")
    config = _pcp_direct_kv_config()
    config.parallel_config.decode_context_parallel_size = 2

    assert not pcp.use_pcp_direct_kv(config)
    assert "VLLM_USE_PCP_DIRECT_KV=1 was ignored" in caplog.text
    assert "decode-context-parallel-size must be 1" in caplog.text


def run_fp8_ds_mla_indexer_oracle(
    group: dist.ProcessGroup | None = None,
    seed_offset: int = 0,
    use_cuda_graph: bool = False,
) -> None:
    """Byte-exact fused peer stores vs gather+insert for production GLM layout.

    Packed uint8 storage with a nonzero layer offset. MLA is fp8_ds_mla (656 B)
    plus Indexer-K (128 fp8 + 4-byte scale).
    """
    from vllm.distributed.device_communicators.symm_mem import (
        allocate_symmetric_memory_storage,
    )
    from vllm.model_executor.layers.attention.direct_kv import (
        KVCacheSymmMemDomain,
    )
    from vllm.models.deepseek_v32.common.kernels import fused_norm_rope

    group = group or dist.group.WORLD
    rank = dist.get_rank(group)
    world = dist.get_world_size(group)
    device = torch.device(f"cuda:{dist.get_rank()}")

    torch.manual_seed(seed_offset)

    # Allocate one packed symmetric storage for MLA and indexer caches. The
    # nonzero prefix verifies that peer writes honor each view's byte offset.
    local = 13
    q_dim, kv_dim, rope, idx_dim = 1536, 512, 64, 128
    mla_entry, idx_entry = 656, 132
    block_size, num_blocks = 64, 32
    layer_offset = 4096
    mla_nbytes = num_blocks * block_size * mla_entry
    idx_nbytes = num_blocks * block_size * idx_entry
    domain = KVCacheSymmMemDomain(
        SimpleNamespace(
            device_group=group,
            world_size=world,
            rank_in_group=rank,
        )
    )
    storage = allocate_symmetric_memory_storage(
        layer_offset + mla_nbytes + idx_nbytes, torch.int8, device
    ).view(torch.uint8)
    domain.rendezvous(storage)

    guard = 0xA5
    storage.fill_(guard)
    mla_view = storage[layer_offset : layer_offset + mla_nbytes].view(
        num_blocks, block_size, mla_entry
    )
    idx_view = storage[
        layer_offset + mla_nbytes : layer_offset + mla_nbytes + idx_nbytes
    ].view(num_blocks, block_size, idx_entry)
    mla_view.zero_()
    idx_view.zero_()
    domain.finalize(
        {"mla": mla_view, "indexer": idx_view},
        storage,
    )
    mla_symm_mem_view = domain.view("mla")
    indexer_symm_mem_view = domain.view("indexer")

    # Build deterministic local inputs for this producer rank.
    q_w = torch.ones(q_dim, device=device, dtype=torch.bfloat16)
    kv_w = torch.ones(kv_dim, device=device, dtype=torch.bfloat16)
    idx_w = torch.ones(idx_dim, device=device, dtype=torch.float32)
    idx_b = torch.zeros(idx_dim, device=device, dtype=torch.float32)
    cos_sin = torch.randn(8192, rope, device=device, dtype=torch.float32)
    torch.manual_seed(seed_offset + rank + 1)
    slots = torch.arange(local, device=device, dtype=torch.int64) + rank * local
    positions = torch.arange(local, device=device)
    q_c = torch.randn(local, q_dim, device=device, dtype=torch.bfloat16)
    kv_c = torch.randn(local, kv_dim, device=device, dtype=torch.bfloat16)
    k_pe = torch.randn(local, rope, device=device, dtype=torch.bfloat16)
    index_k = torch.randn(local, idx_dim, device=device, dtype=torch.bfloat16)

    # Gather the local inputs to build the existing gather-and-insert result.
    total = world * local
    g_q = torch.empty(total, q_dim, device=device, dtype=q_c.dtype)
    g_kv = torch.empty(total, kv_dim, device=device, dtype=kv_c.dtype)
    g_pe = torch.empty(total, rope, device=device, dtype=k_pe.dtype)
    g_ik = torch.empty(total, idx_dim, device=device, dtype=index_k.dtype)
    g_pos = torch.empty(total, device=device, dtype=positions.dtype)
    g_slots = torch.empty(total, device=device, dtype=slots.dtype)
    dist.all_gather_into_tensor(g_q, q_c.contiguous(), group=group)
    dist.all_gather_into_tensor(g_kv, kv_c.contiguous(), group=group)
    dist.all_gather_into_tensor(g_pe, k_pe.contiguous(), group=group)
    dist.all_gather_into_tensor(g_ik, index_k.contiguous(), group=group)
    dist.all_gather_into_tensor(g_pos, positions.contiguous(), group=group)
    dist.all_gather_into_tensor(g_slots, slots.contiguous(), group=group)

    ref_mla = torch.zeros_like(mla_view)
    ref_idx = torch.zeros_like(idx_view)
    ref_kv_c = torch.empty_like(g_kv)
    ref_k_pe = torch.empty_like(g_pe)

    fused_norm_rope(
        g_pos,
        g_q,
        q_w,
        1e-6,
        g_kv,
        kv_w,
        1e-6,
        g_pe,
        cos_sin,
        g_ik,
        idx_w,
        idx_b,
        1e-6,
        cos_sin,
        torch.empty(total, 64, device=device, dtype=torch.int32),
        slot_mapping=g_slots,
        indexer_k_cache=ref_idx,
        mla_kv_cache=ref_mla,
        mla_kv_cache_dtype="fp8_ds_mla",
        has_indexer=True,
        index_rope_interleave=True,
        kv_c_out=ref_kv_c,
        k_pe_out=ref_k_pe,
    )

    # Reset the symmetric views, then produce the same cache directly.
    mla_view.zero_()
    idx_view.zero_()
    dist.barrier(group=group)
    direct_q_out = torch.empty_like(q_c)
    direct_kv_c = torch.empty_like(kv_c)
    direct_k_pe = torch.empty_like(k_pe)
    direct_topk = torch.empty(local, 64, device=device, dtype=torch.int32)

    def write_direct() -> None:
        fused_norm_rope(
            positions,
            q_c,
            q_w,
            1e-6,
            kv_c,
            kv_w,
            1e-6,
            k_pe,
            cos_sin,
            index_k,
            idx_w,
            idx_b,
            1e-6,
            cos_sin,
            direct_topk,
            slot_mapping=slots,
            indexer_k_cache=idx_view,
            mla_kv_cache=mla_view,
            mla_kv_cache_dtype="fp8_ds_mla",
            has_indexer=True,
            index_rope_interleave=True,
            q_c_out=direct_q_out,
            kv_c_out=direct_kv_c,
            k_pe_out=direct_k_pe,
            mla_peer_ptrs=mla_symm_mem_view.peer_ptrs,
            mla_cache_offset_bytes=mla_symm_mem_view.offset_bytes,
            indexer_peer_ptrs=indexer_symm_mem_view.peer_ptrs,
            indexer_cache_offset_bytes=indexer_symm_mem_view.offset_bytes,
            kv_replica_world_size=world,
        )
        domain.barrier(mla_view, idx_view)

    if use_cuda_graph:
        # Compile the Triton kernels and custom barrier before capture.
        write_direct()
        torch.accelerator.synchronize()

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            write_direct()

        # Clear the capture-time writes so correctness depends on graph replay.
        torch.accelerator.synchronize()
        mla_view.zero_()
        idx_view.zero_()
        torch.accelerator.synchronize()
        dist.barrier(group=group)
        graph.replay()
    else:
        write_direct()

    torch.accelerator.synchronize()

    # Both cache regions must match byte-for-byte, and the prefix must remain
    # untouched.
    assert torch.equal(mla_view, ref_mla)
    assert torch.equal(idx_view, ref_idx)
    local_slice = slice(rank * local, (rank + 1) * local)
    assert torch.equal(direct_kv_c, ref_kv_c[local_slice])
    assert torch.equal(direct_k_pe, ref_k_pe[local_slice])
    expected_guard = torch.full(
        (layer_offset,), guard, dtype=torch.uint8, device=device
    )
    assert torch.equal(storage[:layer_offset], expected_guard)

    domain.close()


def _worker_fused_direct_matches_gather_insert(env: dict[str, str]) -> None:
    update_environment_variables(env)
    rank = int(env["RANK"])
    torch.accelerator.set_device_index(rank)
    dist.init_process_group(
        backend="nccl", rank=rank, world_size=int(env["WORLD_SIZE"])
    )

    run_fp8_ds_mla_indexer_oracle()
    dist.destroy_process_group()


def _worker_fused_direct_cudagraph_matches_gather_insert(
    env: dict[str, str],
) -> None:
    update_environment_variables(env)
    rank = int(env["RANK"])
    torch.accelerator.set_device_index(rank)
    dist.init_process_group(
        backend="nccl", rank=rank, world_size=int(env["WORLD_SIZE"])
    )

    run_fp8_ds_mla_indexer_oracle(use_cuda_graph=True)
    dist.destroy_process_group()


def _worker_fused_direct_tp2_pcp2(env: dict[str, str]) -> None:
    update_environment_variables(env)
    global_rank = int(env["RANK"])
    world_size = int(env["WORLD_SIZE"])
    torch.accelerator.set_device_index(global_rank)
    from vllm.config import ParallelConfig, VllmConfig, set_current_vllm_config
    from vllm.distributed.parallel_state import (
        destroy_distributed_environment,
        destroy_model_parallel,
        get_pcp_group,
        get_tp_group,
        init_distributed_environment,
        initialize_model_parallel,
    )

    config = VllmConfig()
    config.parallel_config = ParallelConfig(
        tensor_parallel_size=2,
        prefill_context_parallel_size=2,
    )

    with set_current_vllm_config(config):
        init_distributed_environment(
            world_size=world_size,
            rank=global_rank,
            local_rank=global_rank,
            backend="nccl",
        )
        initialize_model_parallel(
            tensor_model_parallel_size=2,
            prefill_context_model_parallel_size=2,
        )
        pcp_group = get_pcp_group()
        tp_rank = get_tp_group().rank_in_group

        gathered_tp_ranks = [None] * pcp_group.world_size
        dist.all_gather_object(gathered_tp_ranks, tp_rank, group=pcp_group.cpu_group)
        assert gathered_tp_ranks == [tp_rank] * pcp_group.world_size

        run_fp8_ds_mla_indexer_oracle(
            group=pcp_group.device_group,
            seed_offset=1000 * (tp_rank + 1),
        )
        destroy_model_parallel()
        destroy_distributed_environment()


@pytest.mark.skipif(torch.accelerator.device_count() < 2, reason="needs 2+ GPUs")
def test_fused_direct_fp8_ds_mla_indexer_pcp2():
    _distributed_run(_worker_fused_direct_matches_gather_insert, world_size=2)


@pytest.mark.skipif(torch.accelerator.device_count() < 2, reason="needs 2+ GPUs")
def test_fused_direct_fp8_ds_mla_indexer_pcp2_cudagraph_replay():
    _distributed_run(
        _worker_fused_direct_cudagraph_matches_gather_insert,
        world_size=2,
    )


@pytest.mark.skipif(torch.accelerator.device_count() < 4, reason="needs 4 GPUs")
def test_fused_direct_fp8_ds_mla_indexer_pcp4():
    _distributed_run(_worker_fused_direct_matches_gather_insert, world_size=4)


@pytest.mark.skipif(torch.accelerator.device_count() < 4, reason="needs 4 GPUs")
def test_fused_direct_fp8_ds_mla_indexer_tp2_pcp2_subgroups():
    _distributed_run(_worker_fused_direct_tp2_pcp2, world_size=4)
