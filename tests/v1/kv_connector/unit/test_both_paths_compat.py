"""
Descriptor-by-descriptor comparison of OLD (upstream) vs NEW (build_region_meta)
for BOTH _build_fa_local and _build_fa_remote.

Run:
    cd ~/code/vllm && source .venv/bin/activate
    python tests/v1/kv_connector/unit/test_both_paths_compat.py
"""

from __future__ import annotations

import sys
from math import prod
from dataclasses import dataclass

import torch

# ---------------------------------------------------------------------------
# Imports from the codebase
# ---------------------------------------------------------------------------
sys.path.insert(0, ".")

from vllm.v1.kv_cache_interface import (
    KVCacheLayout,
    num_states_for,
    _DIM4_B,
    _DIM4_H,
    get_dtype_size,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.worker import (
    NixlConnectorWorker,
    build_region_meta,
)


# ---------------------------------------------------------------------------
# Minimal stubs — just enough to reproduce descriptor tuples
# ---------------------------------------------------------------------------
@dataclass
class FakeSpec:
    """Minimal KVCacheSpec-like object for testing."""
    num_kv_heads: int
    head_size: int
    dtype: torch.dtype
    tokens_per_state: int = 1
    is_mamba: bool = False
    state_content_size_bytes_val: int | None = None

    @property
    def state_content_size_bytes(self) -> int:
        if self.state_content_size_bytes_val is not None:
            return self.state_content_size_bytes_val
        return self.num_kv_heads * self.head_size * get_dtype_size(self.dtype)

    def transfer_shapes(
        self,
        shape: tuple[int, int, int, int],
        virtually_split: bool,
        mamba_view: bool = False,
    ) -> list[tuple[int, int, int, int]]:
        if self.is_mamba:
            if virtually_split:
                B, _, N, flat_C = shape
                half = flat_C // 2
                return [(B, 1, N, half), (B, 1, N, half)]
            return [shape]
        B, _, N, flat_C = shape
        if virtually_split:
            H = flat_C // (2 * self.head_size)
            return [(B, H, N, self.head_size), (B, H, N, self.head_size)]
        H = flat_C // self.head_size
        return [(B, H, N, self.head_size)]

    def slice_for_tp_transfer(
        self, tensor, my_tp, my_rank, other_tp, other_rank, total_num_kv_heads
    ):
        if self.is_mamba:
            return [tensor]
        tp_ratio = other_tp // my_tp
        if tp_ratio >= 1:
            heads_per_remote_rank = total_num_kv_heads // my_tp
            heads_per_local_rank = total_num_kv_heads // other_tp
            local_rank_in_remote = other_rank % tp_ratio
            start = local_rank_in_remote * heads_per_local_rank
            h_in_meta = tensor.shape[_DIM4_H]
            remote_start = my_rank * heads_per_remote_rank
            overlap_start = max(start, remote_start)
            overlap_end = min(start + heads_per_local_rank, remote_start + h_in_meta)
            if overlap_end <= overlap_start:
                return []
            local_h_start = overlap_start - remote_start
            local_h_len = overlap_end - overlap_start
            return [tensor.narrow(_DIM4_H, local_h_start, local_h_len)]
        else:
            return [tensor]


# ---------------------------------------------------------------------------
# OLD upstream _build_fa_local (verbatim from origin/main)
# ---------------------------------------------------------------------------
def old_build_fa_local(
    base_addresses: list[int],
    block_size_ratio: int,
    num_blocks: int,
    block_len_per_layer: list[int],
    virtually_split: bool,
    device_id: int,
) -> list[tuple[int, int, int]]:
    num_blocks_adj = num_blocks * block_size_ratio
    result: list[tuple[int, int, int]] = []
    for i, base_addr in enumerate(base_addresses):
        kv_block_len = block_len_per_layer[i] // block_size_ratio
        if virtually_split:
            kv_block_len = kv_block_len // 2
        page_stride = block_len_per_layer[i] // block_size_ratio
        for block_id in range(num_blocks_adj):
            block_offset = block_id * page_stride
            addr = base_addr + block_offset
            result.append((addr, kv_block_len, device_id))

        if virtually_split:
            second_split = block_len_per_layer[i] // block_size_ratio // 2
            for block_id in range(num_blocks_adj):
                block_offset = block_id * page_stride
                addr = base_addr + block_offset
                v_addr = addr + kv_block_len
                result.append((v_addr, second_split, device_id))
    return result


# ---------------------------------------------------------------------------
# NEW _build_fa_local (using build_region_meta)
# ---------------------------------------------------------------------------
def new_build_fa_local(
    base_addresses: list[int],
    block_size_ratio: int,
    num_blocks: int,
    block_size: int,
    block_len_per_layer: list[int],
    block_stride_per_layer: list[int],
    spec_per_region: list[FakeSpec],
    virtually_split: bool,
    layout: KVCacheLayout,
    device_id: int,
) -> list[tuple[int, int, int]]:
    num_blocks_adj = num_blocks * block_size_ratio
    result: list[tuple[int, int, int]] = []
    for i, base_addr in enumerate(base_addresses):
        spec = spec_per_region[i]
        block_len = block_len_per_layer[i]

        metas = build_region_meta(
            spec=spec,
            num_blocks=num_blocks_adj,
            block_size=block_size,
            layout=layout,
            block_stride_bytes=block_stride_per_layer[i] // block_size_ratio,
            region_content_bytes=block_len // block_size_ratio,
            virtually_split=virtually_split,
        )

        for meta in metas:
            result.extend(NixlConnectorWorker._view_to_descriptors(
                meta, base_addr, device_id))
    return result


# ---------------------------------------------------------------------------
# OLD upstream _build_fa_remote (verbatim from origin/main)
# ---------------------------------------------------------------------------
def old_build_fa_remote(
    base_addresses: list[int],
    block_size_ratio: int,
    local_block_len_per_layer: list[int],
    remote_block_lens: list[int],
    num_blocks: int,
    num_attn_reads: int,
    rank_offset_factor: int,
    virtually_split: bool,
    device_id: int,
) -> list[tuple[int, int, int]]:
    result: list[tuple[int, int, int]] = []
    for i, base_addr in enumerate(base_addresses):
        local_block_len = local_block_len_per_layer[i]
        if virtually_split:
            local_block_len = local_block_len // 2
        remote_kv_block_len = local_block_len // block_size_ratio
        if block_size_ratio > 1:
            local_block_len = remote_kv_block_len

        local_block_len = local_block_len // num_attn_reads
        rank_offset = rank_offset_factor * remote_kv_block_len

        page_size = remote_block_lens[i]
        for block_id in range(num_blocks):
            block_offset = block_id * page_size
            addr = base_addr + block_offset + rank_offset
            result.append((addr, local_block_len, device_id))

        if virtually_split:
            second_split = local_block_len_per_layer[i] // 2
            if virtually_split:
                pass  # already halved above
            second_split = second_split // num_attn_reads
            for block_id in range(num_blocks):
                block_offset = block_id * page_size
                addr = base_addr + block_offset + rank_offset
                v_addr = addr + remote_block_lens[i] // 2
                result.append((v_addr, second_split, device_id))
    return result


# ---------------------------------------------------------------------------
# NEW _build_fa_remote (using build_region_meta + slice_for_tp_transfer)
# ---------------------------------------------------------------------------
def new_build_fa_remote(
    base_addresses: list[int],
    block_size_ratio: int,
    remote_block_lens: list[int],
    remote_block_strides: list[int],
    remote_block_size: int,
    num_blocks: int,
    spec_per_region: list[FakeSpec],
    remote_tp_rank: int,
    remote_tp_size: int,
    local_tp: int,
    local_rank: int,
    total_kv_heads: int,
    virtually_split: bool,
    layout: KVCacheLayout,
    device_id: int,
) -> list[tuple[int, int, int]]:
    result: list[tuple[int, int, int]] = []
    for i, base_addr in enumerate(base_addresses):
        spec = spec_per_region[i]
        block_len = remote_block_lens[i]

        metas = build_region_meta(
            spec=spec,
            num_blocks=num_blocks,
            block_size=remote_block_size,
            layout=layout,
            block_stride_bytes=remote_block_strides[i],
            region_content_bytes=block_len,
            virtually_split=virtually_split,
        )

        for meta in metas:
            slices = spec.slice_for_tp_transfer(
                meta,
                my_tp=remote_tp_size,
                my_rank=remote_tp_rank,
                other_tp=local_tp,
                other_rank=local_rank,
                total_num_kv_heads=total_kv_heads,
            )

            if slices:
                view = slices[0]
                if block_size_ratio > 1:
                    view = torch.as_strided(
                        torch.empty(1, dtype=view.dtype, device="meta"),
                        size=(
                            view.shape[_DIM4_B] * block_size_ratio,
                            view.shape[_DIM4_H],
                            view.shape[2] // block_size_ratio,
                            view.shape[3],
                        ),
                        stride=(
                            view.stride(_DIM4_B) // block_size_ratio,
                            view.stride(_DIM4_H)
                            if view.stride(_DIM4_H) <= view.stride(2)
                            else view.stride(_DIM4_H) // block_size_ratio,
                            view.stride(2),
                            view.stride(3),
                        ),
                        storage_offset=view.storage_offset(),
                    )
                result.extend(NixlConnectorWorker._view_to_descriptors(
                    view, base_addr, device_id))
    return result


# ---------------------------------------------------------------------------
# Test configurations
# ---------------------------------------------------------------------------
def make_attention_spec(num_kv_heads, head_size, dtype=torch.bfloat16):
    return FakeSpec(
        num_kv_heads=num_kv_heads,
        head_size=head_size,
        dtype=dtype,
    )


def make_mamba_spec(state_content_size_bytes):
    return FakeSpec(
        num_kv_heads=1,
        head_size=1,
        dtype=torch.int8,
        tokens_per_state=-1,
        is_mamba=True,
        state_content_size_bytes_val=state_content_size_bytes,
    )


def compare_descriptors(old, new, label):
    if len(old) != len(new):
        print(f"  FAIL [{label}]: count mismatch old={len(old)} vs new={len(new)}")
        return False

    mismatches = []
    for idx, (o, n) in enumerate(zip(old, new)):
        if o != n:
            mismatches.append((idx, o, n))

    if mismatches:
        print(f"  FAIL [{label}]: {len(mismatches)} descriptor mismatches "
              f"(of {len(old)} total)")
        for idx, o, n in mismatches[:5]:
            print(f"    idx={idx}: old={o} new={n}")
            # Show which field differs
            addr_match = o[0] == n[0]
            len_match = o[1] == n[1]
            dev_match = o[2] == n[2]
            print(f"      addr_match={addr_match} len_match={len_match} "
                  f"dev_match={dev_match}")
        if len(mismatches) > 5:
            print(f"    ... and {len(mismatches) - 5} more")
        return False

    print(f"  PASS [{label}]: {len(old)} descriptors match")
    return True


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------
def test_local_attention_basic():
    """Attention-only, no virtual split, ratio=1, HND layout."""
    num_regions = 2  # K and V separate
    num_kv_heads = 2
    head_size = 128
    block_size = 64
    num_blocks = 100
    dtype = torch.bfloat16
    elem = get_dtype_size(dtype)
    block_len = num_kv_heads * block_size * head_size * elem  # 32768

    base_addresses = [i * num_blocks * block_len for i in range(num_regions)]
    block_len_per_layer = [block_len] * num_regions
    block_stride_per_layer = [block_len] * num_regions
    specs = [make_attention_spec(num_kv_heads, head_size, dtype)] * num_regions
    layout = KVCacheLayout.from_layout_string("HND")

    old = old_build_fa_local(
        base_addresses, 1, num_blocks, block_len_per_layer, False, 0)
    new = new_build_fa_local(
        base_addresses, 1, num_blocks, block_size, block_len_per_layer,
        block_stride_per_layer, specs, False, layout, 0)

    return compare_descriptors(old, new, "local_attention_basic")


def test_local_attention_virtually_split():
    """Attention with virtual split (FlashInfer-like), ratio=1."""
    num_kv_heads = 4
    head_size = 128
    block_size = 16
    num_blocks = 50
    dtype = torch.bfloat16
    elem = get_dtype_size(dtype)
    block_len = 2 * num_kv_heads * block_size * head_size * elem  # K+V interleaved

    base_addresses = [0]  # single region with K+V
    block_len_per_layer = [block_len]
    block_stride_per_layer = [block_len]
    specs = [make_attention_spec(num_kv_heads, head_size, dtype)]
    layout = KVCacheLayout.from_layout_string("HND")

    old = old_build_fa_local(
        base_addresses, 1, num_blocks, block_len_per_layer, True, 0)
    new = new_build_fa_local(
        base_addresses, 1, num_blocks, block_size, block_len_per_layer,
        block_stride_per_layer, specs, True, layout, 0)

    return compare_descriptors(old, new, "local_attention_virtually_split")


def test_local_mamba_no_split():
    """Mamba region, no virtual split, ratio=1."""
    state_bytes = 2_134_016  # conv + SSM
    block_len = 32768  # physical sub-block

    base_addresses = [0]
    block_len_per_layer = [block_len]
    block_stride_per_layer = [block_len]
    specs = [make_mamba_spec(state_bytes)]
    layout = KVCacheLayout.from_layout_string("HND")
    num_blocks = 100

    old = old_build_fa_local(
        base_addresses, 1, num_blocks, block_len_per_layer, False, 0)
    new = new_build_fa_local(
        base_addresses, 1, num_blocks, 64, block_len_per_layer,
        block_stride_per_layer, specs, False, layout, 0)

    return compare_descriptors(old, new, "local_mamba_no_split")


def test_local_mamba_virtually_split():
    """Mamba region, virtual split, ratio=1."""
    state_bytes = 2_134_016
    block_len = 32768

    base_addresses = [0]
    block_len_per_layer = [block_len]
    block_stride_per_layer = [block_len]
    specs = [make_mamba_spec(state_bytes)]
    layout = KVCacheLayout.from_layout_string("HND")
    num_blocks = 100

    old = old_build_fa_local(
        base_addresses, 1, num_blocks, block_len_per_layer, True, 0)
    new = new_build_fa_local(
        base_addresses, 1, num_blocks, 64, block_len_per_layer,
        block_stride_per_layer, specs, True, layout, 0)

    return compare_descriptors(old, new, "local_mamba_virtually_split")


def test_local_hybrid_no_split():
    """Hybrid: 10 attention regions + 2 mamba regions, no virtual split."""
    num_kv_heads = 2
    head_size = 128
    block_size = 64
    dtype = torch.bfloat16
    elem = get_dtype_size(dtype)
    attn_block_len = num_kv_heads * block_size * head_size * elem  # 32768
    mamba_state_bytes = 2_134_016
    mamba_block_len = 32768
    num_blocks = 100

    num_attn_regions = 10
    num_mamba_regions = 2

    base_addresses = [i * 10_000_000 for i in range(num_attn_regions + num_mamba_regions)]
    block_len_per_layer = [attn_block_len] * num_attn_regions + [mamba_block_len] * num_mamba_regions
    block_stride_per_layer = list(block_len_per_layer)
    specs = (
        [make_attention_spec(num_kv_heads, head_size, dtype)] * num_attn_regions
        + [make_mamba_spec(mamba_state_bytes)] * num_mamba_regions
    )
    layout = KVCacheLayout.from_layout_string("HND")

    old = old_build_fa_local(
        base_addresses, 1, num_blocks, block_len_per_layer, False, 0)
    new = new_build_fa_local(
        base_addresses, 1, num_blocks, block_size, block_len_per_layer,
        block_stride_per_layer, specs, False, layout, 0)

    return compare_descriptors(old, new, "local_hybrid_no_split")


def test_remote_attention_homo_tp():
    """Remote attention, homogeneous TP (P=2, D=2), ratio=1."""
    num_kv_heads = 2
    head_size = 128
    block_size = 64
    dtype = torch.bfloat16
    elem = get_dtype_size(dtype)
    block_len = num_kv_heads * block_size * head_size * elem  # 32768
    num_blocks = 50
    total_kv_heads = 4
    remote_tp_size = 2
    local_tp = 2

    base_addresses = [0, 5_000_000]  # 2 regions (K, V)
    local_block_len_per_layer = [block_len, block_len]
    remote_block_lens = [block_len, block_len]
    remote_block_strides = [block_len, block_len]
    specs = [make_attention_spec(num_kv_heads, head_size, dtype)] * 2
    layout = KVCacheLayout.from_layout_string("HND")

    # Test for local_rank=0 reading from remote_rank=0
    num_attn_reads = 1  # homo TP: each local reads from one remote
    rank_offset_factor = 0  # rank 0

    old = old_build_fa_remote(
        base_addresses, 1, local_block_len_per_layer, remote_block_lens,
        num_blocks, num_attn_reads, rank_offset_factor, False, 0)
    new = new_build_fa_remote(
        base_addresses, 1, remote_block_lens, remote_block_strides,
        block_size, num_blocks, specs,
        remote_tp_rank=0, remote_tp_size=remote_tp_size,
        local_tp=local_tp, local_rank=0,
        total_kv_heads=total_kv_heads,
        virtually_split=False, layout=layout, device_id=0)

    return compare_descriptors(old, new, "remote_attention_homo_tp")


def test_remote_attention_hetero_tp():
    """Remote attention, hetero TP (P=1, D=2), ratio=1."""
    total_kv_heads = 4
    head_size = 128
    block_size = 64
    dtype = torch.bfloat16
    elem = get_dtype_size(dtype)
    # Remote (prefill) has all 4 heads
    remote_kv_heads = 4
    remote_block_len = remote_kv_heads * block_size * head_size * elem  # 65536
    # Local (decoder) has 2 heads
    local_kv_heads = 2
    local_block_len = local_kv_heads * block_size * head_size * elem  # 32768
    num_blocks = 50

    remote_tp_size = 1
    local_tp = 2

    base_addresses = [0, 10_000_000]  # 2 regions (K, V)
    local_block_len_per_layer = [local_block_len, local_block_len]
    remote_block_lens = [remote_block_len, remote_block_len]
    remote_block_strides = [remote_block_len, remote_block_len]
    specs = [make_attention_spec(remote_kv_heads, head_size, dtype)] * 2
    layout = KVCacheLayout.from_layout_string("HND")

    # For P=1, D=2: each decoder rank reads from the single prefill rank
    # num_attn_reads = 1 (only 1 source rank in source_ranks_per_group)
    # rank_offset_factor selects which head slice to read
    num_attn_reads = 1
    rank_offset_factor = 0  # local rank 0 → first half

    old = old_build_fa_remote(
        base_addresses, 1, local_block_len_per_layer, remote_block_lens,
        num_blocks, num_attn_reads, rank_offset_factor, False, 0)
    new = new_build_fa_remote(
        base_addresses, 1, remote_block_lens, remote_block_strides,
        block_size, num_blocks, specs,
        remote_tp_rank=0, remote_tp_size=remote_tp_size,
        local_tp=local_tp, local_rank=0,
        total_kv_heads=total_kv_heads,
        virtually_split=False, layout=layout, device_id=0)

    return compare_descriptors(old, new, "remote_attention_hetero_tp_rank0")


def test_remote_attention_hetero_tp_rank1():
    """Remote attention, hetero TP (P=1, D=2), rank1 reads second half."""
    total_kv_heads = 4
    head_size = 128
    block_size = 64
    dtype = torch.bfloat16
    elem = get_dtype_size(dtype)
    remote_kv_heads = 4
    remote_block_len = remote_kv_heads * block_size * head_size * elem  # 65536
    local_kv_heads = 2
    local_block_len = local_kv_heads * block_size * head_size * elem  # 32768
    num_blocks = 50

    remote_tp_size = 1
    local_tp = 2

    base_addresses = [0, 10_000_000]
    local_block_len_per_layer = [local_block_len, local_block_len]
    remote_block_lens = [remote_block_len, remote_block_len]
    remote_block_strides = [remote_block_len, remote_block_len]
    specs = [make_attention_spec(remote_kv_heads, head_size, dtype)] * 2
    layout = KVCacheLayout.from_layout_string("HND")

    num_attn_reads = 1
    rank_offset_factor = 1  # local rank 1 → second half of heads

    old = old_build_fa_remote(
        base_addresses, 1, local_block_len_per_layer, remote_block_lens,
        num_blocks, num_attn_reads, rank_offset_factor, False, 0)
    new = new_build_fa_remote(
        base_addresses, 1, remote_block_lens, remote_block_strides,
        block_size, num_blocks, specs,
        remote_tp_rank=0, remote_tp_size=remote_tp_size,
        local_tp=local_tp, local_rank=1,
        total_kv_heads=total_kv_heads,
        virtually_split=False, layout=layout, device_id=0)

    return compare_descriptors(old, new, "remote_attention_hetero_tp_rank1")


def test_remote_attention_hetero_tp_d1p2():
    """Remote attention, hetero TP (P=2, D=1), local reads 2 heads from remote."""
    total_kv_heads = 4
    head_size = 128
    block_size = 64
    dtype = torch.bfloat16
    elem = get_dtype_size(dtype)
    remote_kv_heads = 2  # each prefill rank has 2 heads
    remote_block_len = remote_kv_heads * block_size * head_size * elem  # 32768
    local_kv_heads = 4  # decoder has all 4 heads
    local_block_len = local_kv_heads * block_size * head_size * elem  # 65536
    num_blocks = 50

    remote_tp_size = 2
    local_tp = 1

    base_addresses = [0, 10_000_000]
    local_block_len_per_layer = [local_block_len, local_block_len]
    remote_block_lens = [remote_block_len, remote_block_len]
    remote_block_strides = [remote_block_len, remote_block_len]
    specs = [make_attention_spec(remote_kv_heads, head_size, dtype)] * 2
    layout = KVCacheLayout.from_layout_string("HND")

    # D=1, P=2: local rank reads from 2 remote ranks
    # For remote_rank=0: num_attn_reads=2, rank_offset_factor=0
    num_attn_reads = 2
    rank_offset_factor = 0

    old = old_build_fa_remote(
        base_addresses, 1, local_block_len_per_layer, remote_block_lens,
        num_blocks, num_attn_reads, rank_offset_factor, False, 0)
    new = new_build_fa_remote(
        base_addresses, 1, remote_block_lens, remote_block_strides,
        block_size, num_blocks, specs,
        remote_tp_rank=0, remote_tp_size=remote_tp_size,
        local_tp=local_tp, local_rank=0,
        total_kv_heads=total_kv_heads,
        virtually_split=False, layout=layout, device_id=0)

    return compare_descriptors(old, new, "remote_attention_hetero_tp_d1p2")


def test_remote_mamba_homo_tp():
    """Remote mamba, homogeneous TP (P=2, D=2), ratio=1."""
    state_bytes = 2_134_016
    block_len = 32768
    num_blocks = 50

    base_addresses = [0]
    local_block_len_per_layer = [block_len]
    remote_block_lens = [block_len]
    remote_block_strides = [block_len]
    specs = [make_mamba_spec(state_bytes)]
    layout = KVCacheLayout.from_layout_string("HND")

    num_attn_reads = 1
    rank_offset_factor = 0

    old = old_build_fa_remote(
        base_addresses, 1, local_block_len_per_layer, remote_block_lens,
        num_blocks, num_attn_reads, rank_offset_factor, False, 0)
    new = new_build_fa_remote(
        base_addresses, 1, remote_block_lens, remote_block_strides,
        64, num_blocks, specs,
        remote_tp_rank=0, remote_tp_size=2,
        local_tp=2, local_rank=0,
        total_kv_heads=4,
        virtually_split=False, layout=layout, device_id=0)

    return compare_descriptors(old, new, "remote_mamba_homo_tp")


def test_remote_hybrid_homo_tp():
    """Remote hybrid (attn + mamba), homo TP (P=2, D=2), ratio=1."""
    num_kv_heads = 2
    head_size = 128
    block_size = 64
    dtype = torch.bfloat16
    elem = get_dtype_size(dtype)
    attn_block_len = num_kv_heads * block_size * head_size * elem  # 32768
    mamba_state_bytes = 2_134_016
    mamba_block_len = 32768
    num_blocks = 50
    total_kv_heads = 4

    num_attn_regions = 10
    num_mamba_regions = 2

    base_addresses = [i * 10_000_000 for i in range(num_attn_regions + num_mamba_regions)]
    local_block_lens = [attn_block_len] * num_attn_regions + [mamba_block_len] * num_mamba_regions
    remote_block_lens = list(local_block_lens)
    remote_block_strides = list(local_block_lens)
    specs = (
        [make_attention_spec(num_kv_heads, head_size, dtype)] * num_attn_regions
        + [make_mamba_spec(mamba_state_bytes)] * num_mamba_regions
    )
    layout = KVCacheLayout.from_layout_string("HND")

    num_attn_reads = 1
    rank_offset_factor = 0

    old = old_build_fa_remote(
        base_addresses, 1, local_block_lens, remote_block_lens,
        num_blocks, num_attn_reads, rank_offset_factor, False, 0)
    new = new_build_fa_remote(
        base_addresses, 1, remote_block_lens, remote_block_strides,
        block_size, num_blocks, specs,
        remote_tp_rank=0, remote_tp_size=2,
        local_tp=2, local_rank=0,
        total_kv_heads=total_kv_heads,
        virtually_split=False, layout=layout, device_id=0)

    return compare_descriptors(old, new, "remote_hybrid_homo_tp")


def test_local_attention_ratio_gt1():
    """Attention, ratio > 1 (block_size_ratio=2)."""
    num_kv_heads = 2
    head_size = 128
    block_size = 64
    dtype = torch.bfloat16
    elem = get_dtype_size(dtype)
    block_len = num_kv_heads * block_size * head_size * elem  # 32768
    num_blocks = 100
    block_size_ratio = 2

    base_addresses = [0, 5_000_000]
    block_len_per_layer = [block_len, block_len]
    block_stride_per_layer = [block_len, block_len]
    specs = [make_attention_spec(num_kv_heads, head_size, dtype)] * 2
    layout = KVCacheLayout.from_layout_string("HND")

    old = old_build_fa_local(
        base_addresses, block_size_ratio, num_blocks, block_len_per_layer, False, 0)
    new = new_build_fa_local(
        base_addresses, block_size_ratio, num_blocks, block_size, block_len_per_layer,
        block_stride_per_layer, specs, False, layout, 0)

    return compare_descriptors(old, new, "local_attention_ratio_gt1")


def test_local_attention_nhd_layout():
    """Attention with NHD layout, no split, ratio=1."""
    num_kv_heads = 4
    head_size = 128
    block_size = 16
    dtype = torch.bfloat16
    elem = get_dtype_size(dtype)
    block_len = num_kv_heads * block_size * head_size * elem

    base_addresses = [0]
    block_len_per_layer = [block_len]
    block_stride_per_layer = [block_len]
    specs = [make_attention_spec(num_kv_heads, head_size, dtype)]
    layout = KVCacheLayout.from_layout_string("NHD")

    old = old_build_fa_local(
        base_addresses, 1, 50, block_len_per_layer, False, 0)
    new = new_build_fa_local(
        base_addresses, 1, 50, block_size, block_len_per_layer,
        block_stride_per_layer, specs, False, layout, 0)

    return compare_descriptors(old, new, "local_attention_nhd_layout")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    tests = [
        test_local_attention_basic,
        test_local_attention_virtually_split,
        test_local_mamba_no_split,
        test_local_mamba_virtually_split,
        test_local_hybrid_no_split,
        test_local_attention_ratio_gt1,
        test_local_attention_nhd_layout,
        test_remote_attention_homo_tp,
        test_remote_attention_hetero_tp,
        test_remote_attention_hetero_tp_rank1,
        test_remote_attention_hetero_tp_d1p2,
        test_remote_mamba_homo_tp,
        test_remote_hybrid_homo_tp,
    ]

    print(f"Running {len(tests)} comparison tests...\n")
    passed = 0
    failed = 0
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"  ERROR [{test.__name__}]: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print(f"\n{'='*60}")
    print(f"Results: {passed} passed, {failed} failed (of {len(tests)} total)")
    if failed > 0:
        sys.exit(1)
    else:
        print("All tests passed!")
