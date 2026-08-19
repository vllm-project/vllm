# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Focused tests for PCP symmetric-memory peer allocation and direct-final writes."""

import multiprocess as mp
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
    torch.cuda.set_device(rank)
    dist.init_process_group(backend="nccl", rank=rank, world_size=int(env["WORLD_SIZE"]))
    from vllm.distributed.device_communicators.symm_mem import allocate_symm_mem_peer

    device = torch.device(f"cuda:{rank}")
    allocation = allocate_symm_mem_peer((4096,), torch.int8, device, dist.group.WORLD)
    assert allocation.peer_ptrs.numel() == dist.get_world_size()
    assert int(allocation.peer_ptrs[rank].item()) == int(allocation.storage.data_ptr())
    assert not bool((allocation.peer_ptrs == 0).any().item())

    view = allocation.storage.view(torch.int32)[16:32]
    view_ptrs = allocation.peer_ptrs_for_view(view)
    assert int(view_ptrs[rank].item()) == int(view.data_ptr())

    offset = 64 + rank * 16
    sentinel = torch.full((16,), rank + 7, dtype=torch.int8, device=device)
    for peer_view in allocation._peer_views:
        peer_view[offset : offset + 16].copy_(sentinel)
    dist.barrier()
    for source in range(dist.get_world_size()):
        source_offset = 64 + source * 16
        expected = torch.full((16,), source + 7, dtype=torch.int8, device=device)
        assert torch.equal(
            allocation.storage[source_offset : source_offset + 16], expected
        )
    allocation.close()
    dist.destroy_process_group()


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="needs 2+ GPUs")
def test_symm_mem_peer_allocation_two_gpu():
    _distributed_run(_worker_peer_allocation, world_size=2)


@pytest.mark.skipif(torch.cuda.device_count() < 4, reason="needs 4 GPUs")
def test_symm_mem_peer_allocation_four_gpu():
    _distributed_run(_worker_peer_allocation, world_size=4)


def test_select_pcp_direct_slot_row_uses_this_rank_not_rank0():
    from vllm.v1.worker.gpu.pcp_manager import select_pcp_direct_slot_row

    # Rank-major gathered slots: two groups, PCP=4, padded=3.
    # Rank 2's local tokens are 10,11 then PAD.
    gathered = torch.tensor(
        [
            [0, 1, 2, 3, 4, 5, 10, 11, -1, 20, 21, 22],
            [100, 101, 102, 103, 104, 105, 110, 111, -1, 120, 121, 122],
        ],
        dtype=torch.int64,
    )
    rank2 = select_pcp_direct_slot_row(gathered, pcp_world_size=4, pcp_rank=2, local_tokens=2)
    assert rank2.tolist() == [[10, 11], [110, 111]]
    rank0 = select_pcp_direct_slot_row(gathered, pcp_world_size=4, pcp_rank=0, local_tokens=3)
    assert rank0.tolist() == [[0, 1, 2], [100, 101, 102]]


def test_select_pcp_direct_slot_row_world_size_one_is_identity():
    from vllm.v1.worker.gpu.pcp_manager import select_pcp_direct_slot_row

    slots = torch.arange(6, dtype=torch.int64).view(2, 3)
    assert torch.equal(select_pcp_direct_slot_row(slots, 1, 0, 3), slots)


def test_select_pcp_direct_slot_row_rejects_global_token_count():
    from vllm.v1.worker.gpu.pcp_manager import select_pcp_direct_slot_row

    gathered = torch.zeros(2, 4 * 1024, dtype=torch.int64)
    with pytest.raises(ValueError, match="exceeds padded PCP row"):
        select_pcp_direct_slot_row(gathered, 4, 0, 2048)
    row = select_pcp_direct_slot_row(gathered, 4, 0, 1024)
    assert row.shape == (2, 1024)
    # Padded local row is the producer length; unpadded is a prefix of it.
    unpadded = select_pcp_direct_slot_row(gathered, 4, 1, 512)
    assert unpadded.shape == (2, 512)


def _worker_fused_direct_matches_gather_insert(env: dict[str, str]) -> None:
    """Shipped fused_norm_rope: peer-store replicas == gather+local-insert."""
    update_environment_variables(env)
    rank = int(env["RANK"])
    torch.cuda.set_device(rank)
    dist.init_process_group(backend="nccl", rank=rank, world_size=int(env["WORLD_SIZE"]))
    from vllm.distributed.device_communicators.symm_mem import allocate_symm_mem_peer
    from vllm.model_executor.layers.attention.pcp_direct_kv import PCPPeerCacheFence
    from vllm.models.deepseek_v32.common.kernels import fused_norm_rope

    world = dist.get_world_size()
    device = torch.device(f"cuda:{rank}")
    group = dist.group.WORLD
    torch.manual_seed(0)
    local = 13
    q_dim, kv_dim, rope = 1536, 512, 64
    mla = allocate_symm_mem_peer(
        (32, 64, kv_dim + rope), torch.float8_e4m3fn, device, group
    )
    fence = PCPPeerCacheFence(group, device)
    slots = torch.arange(local, device=device, dtype=torch.int64) + rank * local
    positions = torch.arange(local, device=device)
    q_c = torch.randn(local, q_dim, device=device, dtype=torch.bfloat16)
    kv_c = torch.randn(local, kv_dim, device=device, dtype=torch.bfloat16)
    k_pe = torch.randn(local, rope, device=device, dtype=torch.bfloat16)
    q_w = torch.ones(q_dim, device=device, dtype=torch.bfloat16)
    kv_w = torch.ones(kv_dim, device=device, dtype=torch.bfloat16)
    cos_sin = torch.randn(8192, rope, device=device, dtype=torch.float32)
    scale = torch.ones(1, device=device, dtype=torch.float32)
    total = world * local
    g_q = torch.empty(total, q_dim, device=device, dtype=q_c.dtype)
    g_kv = torch.empty(total, kv_dim, device=device, dtype=kv_c.dtype)
    g_pe = torch.empty(total, rope, device=device, dtype=k_pe.dtype)
    g_pos = torch.empty(total, device=device, dtype=positions.dtype)
    g_slots = torch.empty(total, device=device, dtype=slots.dtype)
    dist.all_gather_into_tensor(g_q, q_c.contiguous())
    dist.all_gather_into_tensor(g_kv, kv_c.contiguous())
    dist.all_gather_into_tensor(g_pe, k_pe.contiguous())
    dist.all_gather_into_tensor(g_pos, positions.contiguous())
    dist.all_gather_into_tensor(g_slots, slots.contiguous())
    ref = torch.zeros_like(mla.storage)
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
        None,
        None,
        None,
        1e-6,
        None,
        torch.empty(total, 64, device=device, dtype=torch.int32),
        slot_mapping=g_slots,
        mla_kv_cache=ref,
        mla_kv_cache_dtype="fp8",
        mla_k_scale=scale,
        has_indexer=False,
    )
    mla.storage.zero_()
    dist.barrier()
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
        None,
        None,
        None,
        1e-6,
        None,
        torch.empty(local, 64, device=device, dtype=torch.int32),
        slot_mapping=slots,
        mla_kv_cache=mla.storage,
        mla_kv_cache_dtype="fp8",
        mla_k_scale=scale,
        has_indexer=False,
        mla_peer_ptrs=mla.peer_ptrs,
        pcp_world_size=world,
    )
    fence()
    torch.cuda.synchronize()
    assert torch.equal(mla.storage.view(torch.uint8), ref.view(torch.uint8))
    fence.close()
    mla.close()
    dist.destroy_process_group()


@pytest.mark.skipif(torch.cuda.device_count() < 4, reason="needs 4 GPUs")
def test_fused_direct_peer_stores_match_gather_insert():
    _distributed_run(_worker_fused_direct_matches_gather_insert, world_size=4)
