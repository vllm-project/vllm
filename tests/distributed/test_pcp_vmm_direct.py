# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Direct-final stores into owner KV backing, including graph replay after growth."""

import os
from types import SimpleNamespace as NS

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def _worker(rank, port, pcp_size):
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    torch.accelerator.set_device_index(rank)
    dist.init_process_group("gloo", rank=rank, world_size=4)
    groups = []
    for lane in range(4 // pcp_size):
        ranks = list(range(lane, 4, 4 // pcp_size))
        groups.append(
            (
                dist.new_group(ranks, backend="gloo"),
                dist.new_group(ranks, backend="nccl"),
            )
        )
    cpu, nccl = groups[rank % (4 // pcp_size)]
    local_rank = dist.get_rank(cpu)
    for cache_dtype in ("auto", "fp8", "fp8_ds_mla"):
        _check_dtype(rank, pcp_size, cpu, nccl, local_rank, cache_dtype)
    _check_rejected_layouts(rank, cpu, local_rank)
    dist.destroy_process_group()


def _check_rejected_layouts(rank, cpu, local_rank):
    from vllm.utils.extensible_tensor import ExtensibleTensor

    owner = ExtensibleTensor(
        (4 + local_rank) * 1024 * 1024,
        device=torch.device("cuda", rank),
        exportable=True,
    )
    owner.resize_per_segment_(1024, zero_new=True)
    with pytest.raises(RuntimeError, match="identical same-host VMM layouts"):
        owner.share_with(cpu)
    owner.free()

    owner = ExtensibleTensor(
        4 * 1024 * 1024, device=torch.device("cuda", rank), exportable=True
    )
    owner.resize_per_segment_(1024, zero_new=True)
    peers = owner.share_with(cpu)
    with pytest.raises(RuntimeError, match="identical same-host VMM layouts"):
        owner.resize_per_segment_(2048 + local_rank, zero_new=True)
    assert peers._failed
    owner.free()


def _check_dtype(rank, pcp_size, cpu, nccl, local_rank, cache_dtype):
    from vllm.model_executor.layers.attention.direct_kv import KVCacheVmmDomain
    from vllm.models.deepseek_v32.common.kernels import fused_norm_rope
    from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheTensor
    from vllm.v1.worker.extensible_kv_cache import ExtensibleKVCache

    stride, capacity = 128 * 1024, 64
    segment = stride * capacity
    config = KVCacheConfig(
        capacity,
        [
            KVCacheTensor(
                size=2 * segment,
                layers=["main", "index"],
                layer_stride=segment,
                block_stride=stride,
            )
        ],
        [],
    )
    owner = ExtensibleKVCache(config, torch.device("cuda", rank), exportable=True)
    owner.commit(16)
    raw = owner.buffer.full_view()
    dtype = torch.bfloat16 if cache_dtype == "auto" else torch.uint8
    width = 656 if cache_dtype == "fp8_ds_mla" else 576
    main = (
        raw[:segment]
        .view(dtype)
        .as_strided(
            (capacity, 64, width),
            (stride // torch.empty((), dtype=dtype).element_size(), width, 1),
        )
    )
    index = raw[segment:].as_strided((capacity, 64, 132), (stride, 132, 1))
    domain = KVCacheVmmDomain(
        NS(cpu_group=cpu, world_size=pcp_size, rank_in_group=local_rank),
        owner.buffer,
    )
    domain.bind({"main": main, "index": index}, {})
    peer_ptr = domain.peers.pointers.data_ptr()
    base = owner.buffer.base_ptr
    peer_bases = tuple(domain.peers.bases)
    q = torch.randn(1, 64, device="cuda", dtype=torch.bfloat16)
    kv = torch.randn(1, 512, device="cuda", dtype=torch.bfloat16)
    rope = torch.randn(1, 64, device="cuda", dtype=torch.bfloat16)
    ik = torch.randn(1, 128, device="cuda", dtype=torch.bfloat16)
    q_out, q_ref = torch.empty_like(q), torch.empty_like(q)
    pos = torch.zeros(1, device="cuda", dtype=torch.int64)
    slots = torch.tensor([local_rank + 1], device="cuda", dtype=torch.int64)
    cos = torch.cat((torch.ones(16, 32), torch.zeros(16, 32)), dim=1).cuda()
    index_cos = torch.cat((torch.ones(16, 64), torch.zeros(16, 64)), dim=1).cuda()
    weights = [
        torch.ones(n, device="cuda", dtype=torch.bfloat16) for n in (64, 512, 128)
    ]
    bias = torch.zeros_like(weights[2])
    topk = torch.empty(1, 8, device="cuda", dtype=torch.int32)
    scale = torch.ones(1, device="cuda")
    ref_main, ref_index = torch.zeros_like(main), torch.zeros_like(index)

    def run(cache, index_cache, out, direct):
        fused_norm_rope(
            pos,
            q,
            weights[0],
            1e-6,
            kv,
            weights[1],
            1e-6,
            rope,
            cos,
            ik,
            weights[2],
            bias,
            1e-6,
            index_cos,
            topk,
            slot_mapping=slots,
            indexer_slot_mapping=slots,
            indexer_k_cache=index_cache,
            mla_kv_cache=cache,
            mla_kv_cache_dtype=cache_dtype,
            mla_k_scale=scale,
            index_rope_interleave=True,
            q_c_out=out,
            mla_peer_ptrs=peer_ptr if direct else None,
            indexer_peer_ptrs=peer_ptr if direct else None,
            indexer_cache_offset_bytes=segment,
            kv_replica_world_size=pcp_size if direct else 1,
        )
        if direct:
            domain.barrier(cache, index_cache)

    run(ref_main, ref_index, q_ref, False)
    run(main, index, q_out, True)
    torch.accelerator.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run(main, index, q_out, True)
    for iteration in range(4):
        if iteration == 1:
            owner.commit(48)
            slots.fill_(40 * 64 + local_rank)
        elif iteration == 2:
            # Recommit different physical allocations at the captured VAs.
            owner.release_physical()
            owner.recommit()
            ref_main.zero_()
            ref_index.zero_()
        elif iteration == 3 and local_rank == 0:
            slots.fill_(-1)
        q.normal_()
        kv.normal_()
        ik.normal_()
        run(ref_main, ref_index, q_ref, False)
        graph.replay()
        expected_main = ref_main[: owner.num_committed_blocks].clone()
        expected_index = ref_index[: owner.num_committed_blocks].clone()
        dist.all_reduce(expected_main, group=nccl)
        dist.all_reduce(expected_index, group=nccl)
        torch.testing.assert_close(
            main[: owner.num_committed_blocks], expected_main, rtol=0, atol=0
        )
        torch.testing.assert_close(
            index[: owner.num_committed_blocks], expected_index, rtol=0, atol=0
        )
        torch.testing.assert_close(q_out, q_ref, rtol=0, atol=0)
        assert owner.buffer.base_ptr == base and tuple(domain.peers.bases) == peer_bases
        assert domain.peers.pointers.data_ptr() == peer_ptr
    del graph
    domain.close()
    owner.free()


@pytest.mark.parametrize("pcp_size", [4, 2])
@pytest.mark.skipif(
    torch.accelerator.device_count() < 4, reason="requires four CUDA GPUs"
)
def test_vmm_direct_final_graph_growth(pcp_size):
    from vllm.utils.network_utils import get_open_port

    mp.spawn(_worker, args=(get_open_port(), pcp_size), nprocs=4, join=True)
