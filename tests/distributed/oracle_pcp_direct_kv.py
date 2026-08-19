#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""PCP4 oracle: shipped fused_norm_rope gather+insert vs SymmMem peer stores."""

from __future__ import annotations

import torch
import torch.distributed as dist

from vllm.distributed.device_communicators.symm_mem import allocate_symm_mem_peer
from vllm.model_executor.layers.attention.pcp_direct_kv import PCPPeerCacheFence
from vllm.models.deepseek_v32.common.kernels import fused_norm_rope


def run_oracle() -> None:
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world = dist.get_world_size()
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")
    group = dist.group.WORLD

    torch.manual_seed(0)
    local = 13
    q_dim, kv_dim, rope, idx_dim = 1536, 512, 64, 128
    block_size, num_blocks = 64, 32
    entry = kv_dim + rope

    mla = allocate_symm_mem_peer(
        (num_blocks, block_size, entry), torch.float8_e4m3fn, device, group
    )
    fence = PCPPeerCacheFence(group, device)

    slots = torch.arange(local, device=device, dtype=torch.int64) + rank * local
    positions = torch.arange(local, device=device)
    q_c = torch.randn(local, q_dim, device=device, dtype=torch.bfloat16)
    kv_c = torch.randn(local, kv_dim, device=device, dtype=torch.bfloat16)
    k_pe = torch.randn(local, rope, device=device, dtype=torch.bfloat16)
    index_k = torch.randn(local, idx_dim, device=device, dtype=torch.bfloat16)
    q_w = torch.ones(q_dim, device=device, dtype=torch.bfloat16)
    kv_w = torch.ones(kv_dim, device=device, dtype=torch.bfloat16)
    idx_w = torch.ones(idx_dim, device=device, dtype=torch.float32)
    idx_b = torch.zeros(idx_dim, device=device, dtype=torch.float32)
    cos_sin = torch.randn(8192, rope, device=device, dtype=torch.float32)
    topk = torch.empty(local, 64, device=device, dtype=torch.int32)
    scale = torch.ones(1, device=device, dtype=torch.float32)

    total = world * local
    g_q = torch.empty(total, q_dim, device=device, dtype=q_c.dtype)
    g_kv = torch.empty(total, kv_dim, device=device, dtype=kv_c.dtype)
    g_pe = torch.empty(total, rope, device=device, dtype=k_pe.dtype)
    g_ik = torch.empty(total, idx_dim, device=device, dtype=index_k.dtype)
    g_pos = torch.empty(total, device=device, dtype=positions.dtype)
    g_slots = torch.empty(total, device=device, dtype=slots.dtype)
    dist.all_gather_into_tensor(g_q, q_c.contiguous())
    dist.all_gather_into_tensor(g_kv, kv_c.contiguous())
    dist.all_gather_into_tensor(g_pe, k_pe.contiguous())
    dist.all_gather_into_tensor(g_ik, index_k.contiguous())
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
        g_ik,
        idx_w,
        idx_b,
        1e-6,
        cos_sin,
        torch.empty(total, 64, device=device, dtype=torch.int32),
        slot_mapping=g_slots,
        mla_kv_cache=ref,
        mla_kv_cache_dtype="fp8",
        mla_k_scale=scale,
        has_indexer=False,
    )
    torch.cuda.synchronize()

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
        index_k,
        idx_w,
        idx_b,
        1e-6,
        cos_sin,
        topk,
        slot_mapping=slots,
        mla_kv_cache=mla.storage,
        mla_kv_cache_dtype="fp8",
        mla_k_scale=scale,
        has_indexer=True,
        index_rope_interleave=True,
        mla_peer_ptrs=mla.peer_ptrs,
        pcp_world_size=world,
    )
    fence()
    torch.cuda.synchronize()
    match = bool(torch.equal(mla.storage.view(torch.uint8), ref.view(torch.uint8)))
    print(
        f"rank{rank} mla_match={match} mcast={mla.multicast_ptr != 0}",
        flush=True,
    )
    ok = torch.tensor(int(match), device=device)
    dist.all_reduce(ok, op=dist.ReduceOp.MIN)
    fence.close()
    mla.close()
    dist.destroy_process_group()
    if int(ok.item()) != 1:
        raise SystemExit("oracle mismatch")


if __name__ == "__main__":
    run_oracle()
