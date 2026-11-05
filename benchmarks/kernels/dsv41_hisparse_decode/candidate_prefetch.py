# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Experimental full-candidate materialization, using the existing HiSparse LRU."""

from types import MethodType

import torch

from vllm.triton_utils import tl, triton
from vllm.v1.hisparse.runtime import (
    ResolvedHiSparseConfig,
    _check_residency_shared_memory,
    _create_shared_topk_state,
)
from vllm.v1.worker.gpu_worker import Worker


@triton.jit
def _expand_candidates(
    BLOCKS,
    SEQ_LENS,
    REQS,
    VALID,
    OUT,
    BS: tl.constexpr,
    N: tl.constexpr,
    TILE: tl.constexpr,
):
    row = tl.program_id(0)
    x = tl.program_id(1) * TILE + tl.arange(0, TILE)
    block = tl.load(BLOCKS + row * BS + x // 8, x < N, other=-1)
    req = tl.load(REQS + row)
    length = tl.load(SEQ_LENS + tl.maximum(req, 0))
    valid = tl.load(VALID + row) & (req >= 0)
    position = block * 8 + x % 8
    tl.store(
        OUT + row * N + x,
        tl.where(valid & (block >= 0) & (position < length), position, -1),
        x < N,
    )


@triton.jit
def _lookup_candidates(
    BLOCKS,
    PHYSICAL,
    SELECTED,
    VALID,
    OUT,
    LENS,
    STATS,
    BS: tl.constexpr,
    K: tl.constexpr,
    N: tl.constexpr,
    TILE: tl.constexpr,
):
    row = tl.program_id(0)
    x = tl.arange(0, TILE)
    selected = tl.load(SELECTED + row * K + x, x < K, other=-1)
    valid = (x < K) & (selected >= 0) & tl.load(VALID + row)
    target = selected // 8
    low = tl.full((TILE,), 0, tl.int32)
    high = tl.full((TILE,), BS, tl.int32)
    for _ in range(12):
        middle = (low + high) // 2
        value = tl.load(BLOCKS + row * BS + middle, middle < BS, other=2147483647)
        take_right = (middle < BS) & (value < target)
        low = tl.where(take_right, middle + 1, low)
        high = tl.where(take_right, high, middle)
    block = tl.load(BLOCKS + row * BS + low, low < BS, other=-2)
    found = valid & (low < BS) & (block == target)
    physical = tl.load(PHYSICAL + row * N + low * 8 + selected % 8, found, other=-1)
    tl.store(OUT + row * K + x, tl.where(found, physical, -1), x < K)
    tl.store(LENS + row, tl.sum((found & (physical >= 0)).to(tl.int32)))
    tl.atomic_add(STATS, tl.sum(valid.to(tl.int64)))
    tl.atomic_add(STATS + 1, tl.sum((valid & ((~found) | (physical < 0))).to(tl.int64)))


class CandidatePool:
    def __init__(self, model):
        layers = {
            m.layer_id: m
            for m in model.modules()
            if type(m).__name__ == "DeepseekV4FlashMLAAttention"
        }
        self.source = layers[20]
        cache = self.source.hisparse_cache
        assert cache is not None
        runtime = cache.runtime
        assert runtime.region_stride >= 16384
        config = ResolvedHiSparseConfig(16384, runtime.region_stride, 16384)
        _check_residency_shared_memory(runtime.device, config)
        runtime.max_union_rows = 16384
        runtime.index_group.shared_topk = _create_shared_topk_state(
            runtime.device, runtime.max_swap_rows, 16384
        )
        self.original_swap = cache.swap_in
        self.blocks = None
        self.physical = None
        self.stats = torch.zeros(2, dtype=torch.int64, device=runtime.device)
        rows = runtime.max_swap_rows
        self.positions = torch.empty(
            (rows, 16384), dtype=torch.int32, device=runtime.device
        )
        self.selected = torch.empty(
            (rows, 512), dtype=torch.int32, device=runtime.device
        )
        self.lengths = torch.empty(rows, dtype=torch.int32, device=runtime.device)
        original = self.source._forward_decode

        def decode(attn, **kwargs):
            if attn.hisparse_cache is not None:
                self.stage(kwargs["attn_metadata"], kwargs["swa_metadata"])
            return original(**kwargs)

        self.source._forward_decode = MethodType(decode, self.source)

        def lookup(
            handle, req_id_per_token, block_table, logical_topk_indices, **kwargs
        ):
            assert self.blocks is not None and self.physical is not None
            count = logical_topk_indices.shape[0]
            assert logical_topk_indices.shape[1] == 512
            _lookup_candidates[(count,)](
                self.blocks,
                self.physical,
                logical_topk_indices,
                self.valid,
                self.selected,
                self.lengths,
                self.stats,
                BS=2048,
                K=512,
                N=16384,
                TILE=512,
            )
            return self.selected[:count], self.lengths[:count]

        cache.swap_in = MethodType(lookup, cache)

    def stage(self, metadata, swa):
        assert swa.max_decode_query_len == 1
        count = swa.num_decode_tokens
        candidates = self.source.candidate_block_buffer[:count]
        assert candidates.shape[1] == 2048
        self.blocks = torch.sort(candidates, dim=1).values
        self.valid = swa.is_valid_token[:count]
        positions = self.positions[:count]
        _expand_candidates[(count, 16)](
            self.blocks,
            swa.seq_lens,
            swa.token_to_req_indices,
            self.valid,
            positions,
            BS=2048,
            N=16384,
            TILE=1024,
        )
        self.source.hisparse_cache.runtime.begin_forward()
        self.physical, _ = self.original_swap(
            metadata.req_id_per_token,
            metadata.block_table[: swa.num_decodes],
            positions,
            block_size=64,
            num_valid_rows=swa.query_start_loc[swa.num_decodes : swa.num_decodes + 1],
            return_valid_counts=True,
        )


class CandidateWorker(Worker):
    validate_candidates = False

    def compile_or_warm_up_model(self):
        model = self.model_runner.get_model()
        self.candidate_pool = CandidatePool(model)
        if self.validate_candidates:
            from graph_validate import install_validators

            install_validators(self)
        return super().compile_or_warm_up_model()


class CandidateValidatingWorker(CandidateWorker):
    validate_candidates = True


def control(worker, action):
    pool = worker.candidate_pool
    if action == "reset":
        pool.stats.zero_()
        return dict(rank=worker.rank, reset=True)
    assert action == "read"
    selected, missing = pool.stats.cpu().tolist()
    assert selected > 0 and missing == 0, (selected, missing)
    return dict(rank=worker.rank, selected_rows=selected, missing_pool_rows=missing)
