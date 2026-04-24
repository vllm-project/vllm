# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for plan-based transfer executors.

These tests verify that the plan-based design produces correct
outputs (descriptor tuples, descriptor IDs, read specs, split handles).
No GPU or NIXL required.
"""

from __future__ import annotations

from dataclasses import dataclass
from unittest.mock import MagicMock

import pytest

from vllm.distributed.kv_transfer.kv_connector.utils import (
    EngineTransferInfo,
    TransferTopology,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.transfer_plan import (
    EngineTransferPlan,
    RegionPlan,
    generate_dense_plan,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.worker import (
    NixlConnectorWorker,
)
from vllm.v1.kv_cache_interface import FullAttentionSpec, MambaSpec

# ======================================================================
# Test fixtures / helpers
# ======================================================================

ENGINE_ID = "remote_engine"


@dataclass
class FakeNixlAgentMeta:
    """Minimal mock of NixlAgentMetadata for testing."""

    engine_id: str
    agent_metadata: bytes
    kv_caches_base_addr: list[int]
    device_id: int
    num_blocks: int
    block_lens: list[int]
    kv_cache_layout: str
    block_size: int
    ssm_sizes: tuple[int, int]
    attn_backend_name: str


def _make_fake_topo(
    tp_rank: int = 0,
    tp_size: int = 1,
    is_mla: bool = False,
    total_num_kv_heads: int = 8,
    block_size: int = 16,
    is_blocks_first: bool = False,
) -> TransferTopology:
    """Build a lightweight TransferTopology mock (skips __post_init__)."""
    topo = MagicMock(spec=TransferTopology)
    topo.tp_rank = tp_rank
    topo.tp_size = tp_size
    topo.is_mla = is_mla
    topo.total_num_kv_heads = total_num_kv_heads
    topo.block_size = block_size
    topo.is_kv_layout_blocks_first = is_blocks_first
    return topo


def _common_plan_params(
    tp_rank: int = 0,
    tp_size: int = 1,
    is_mla: bool = False,
    num_kv_heads: int = 8,
    block_size: int = 16,
    is_blocks_first: bool = False,
    block_len_per_layer: list[int] | None = None,
    remote_tp_size: int = 1,
    remote_block_size: int = 16,
    remote_num_blocks: int = 256,
    remote_block_lens: list[int] | None = None,
    remote_physical_blocks_per_logical: int = 1,
    local_physical_blocks_per_logical: int = 1,
) -> dict:
    """Build common kwargs for plan generators."""
    if block_len_per_layer is None:
        slot_size = num_kv_heads * 128 * 2  # num_heads * head_size * dtype_bytes
        block_len_per_layer = [slot_size * block_size] * 2
    if remote_block_lens is None:
        remote_block_lens = list(block_len_per_layer)
    return dict(
        transfer_topo=_make_fake_topo(
            tp_rank=tp_rank,
            tp_size=tp_size,
            is_mla=is_mla,
            total_num_kv_heads=num_kv_heads,
            block_size=block_size,
            is_blocks_first=is_blocks_first,
        ),
        block_len_per_layer=block_len_per_layer,
        remote_info=EngineTransferInfo(
            remote_tp_size=remote_tp_size,
            remote_block_size=remote_block_size,
            remote_block_len=remote_block_lens[0],
            remote_physical_blocks_per_logical=remote_physical_blocks_per_logical,
        ),
        remote_meta=_make_nixl_meta(
            base_addrs=[0] * len(block_len_per_layer),
            num_blocks=remote_num_blocks,
            block_lens=remote_block_lens,
            block_size=remote_block_size,
        ),
        group_spec_types=(FullAttentionSpec,),
        local_physical_blocks_per_logical=local_physical_blocks_per_logical,
    )


def _make_nixl_meta(
    base_addrs: list[int],
    num_blocks: int,
    block_lens: list[int],
    device_id: int = 0,
    block_size: int = 16,
) -> FakeNixlAgentMeta:
    return FakeNixlAgentMeta(
        engine_id=ENGINE_ID,
        agent_metadata=b"",
        kv_caches_base_addr=base_addrs,
        device_id=device_id,
        num_blocks=num_blocks,
        block_lens=block_lens,
        kv_cache_layout="HND",
        block_size=block_size,
        ssm_sizes=(0, 0),
        attn_backend_name="FlashAttentionBackend",
    )


# ======================================================================
# Dense equivalence tests
# ======================================================================


class TestDensePlanExecutors:
    """Verify plan-based executors produce correct outputs for dense models."""

    @pytest.mark.parametrize(
        "tp_size,remote_tp_size",
        [
            (1, 1),
            (2, 1),
            (4, 2),
            (1, 2),
            (2, 4),
        ],
    )
    @pytest.mark.parametrize("tp_rank_frac", [0.0, 0.5])
    def test_build_remote_descs(self, tp_size, remote_tp_size, tp_rank_frac):
        tp_rank = int(tp_rank_frac * (tp_size - 1)) if tp_size > 1 else 0
        num_kv_heads = 8
        block_size = 16
        num_blocks = 64
        num_layers = 2
        slot_size = num_kv_heads * 128 * 2
        block_len = slot_size * block_size
        block_len_per_layer = [block_len] * num_layers

        if tp_size >= remote_tp_size:
            tp_ratio = tp_size // remote_tp_size
            remote_block_lens = [bl * tp_ratio for bl in block_len_per_layer]
        else:
            tp_ratio_neg = remote_tp_size // tp_size
            remote_block_lens = [bl // tp_ratio_neg for bl in block_len_per_layer]

        base_addrs = [0x1000 * (i + 1) for i in range(num_layers)]
        plan = generate_dense_plan(
            **_common_plan_params(
                tp_rank=tp_rank,
                tp_size=tp_size,
                num_kv_heads=num_kv_heads,
                block_size=block_size,
                block_len_per_layer=block_len_per_layer,
                remote_tp_size=remote_tp_size,
                remote_block_size=block_size,
                remote_num_blocks=num_blocks,
                remote_block_lens=remote_block_lens,
            ),
        )
        meta = _make_nixl_meta(
            base_addrs, num_blocks, remote_block_lens, block_size=block_size
        )
        descs = NixlConnectorWorker._build_remote_descs_from_plan(plan, meta)

        expected_count = len(plan.fa_regions) * num_blocks
        assert len(descs) == expected_count
        for addr, length, dev in descs:
            assert length > 0
            assert dev == 0

    @pytest.mark.parametrize(
        "tp_size,remote_tp_size",
        [(1, 1), (2, 1), (1, 2)],
    )
    def test_compute_desc_ids(self, tp_size, remote_tp_size):
        num_kv_heads = 8
        block_size = 16
        num_blocks = 64
        num_layers = 2
        slot_size = num_kv_heads * 128 * 2
        block_len = slot_size * block_size
        block_len_per_layer = [block_len] * num_layers

        if tp_size >= remote_tp_size:
            tp_ratio = tp_size // remote_tp_size
            remote_block_lens = [bl * tp_ratio for bl in block_len_per_layer]
        else:
            tp_ratio_neg = remote_tp_size // tp_size
            remote_block_lens = [bl // tp_ratio_neg for bl in block_len_per_layer]

        plan = generate_dense_plan(
            **_common_plan_params(
                tp_size=tp_size,
                num_kv_heads=num_kv_heads,
                block_size=block_size,
                block_len_per_layer=block_len_per_layer,
                remote_tp_size=remote_tp_size,
                remote_block_size=block_size,
                remote_num_blocks=num_blocks,
                remote_block_lens=remote_block_lens,
            ),
        )

        block_ids = ([1, 5, 10, 20],)
        ids = NixlConnectorWorker._compute_desc_ids_from_plan(
            plan, block_ids, dst_num_blocks=num_blocks,
            block_size_ratio=None, physical_blocks_per_logical=1,
        )

        num_regions = len(plan.fa_regions)
        assert len(ids) == num_regions * len(block_ids[0])
        assert ids[0] == 1

    @pytest.mark.parametrize(
        "tp_size,remote_tp_size",
        [(1, 1), (2, 1), (1, 2)],
    )
    def test_compute_read_specs(self, tp_size, remote_tp_size):
        num_kv_heads = 8
        block_size = 16
        num_blocks = 64
        num_layers = 2
        slot_size = num_kv_heads * 128 * 2
        block_len = slot_size * block_size
        block_len_per_layer = [block_len] * num_layers

        if tp_size >= remote_tp_size:
            tp_ratio = tp_size // remote_tp_size
            remote_block_lens = [bl * tp_ratio for bl in block_len_per_layer]
        else:
            tp_ratio_neg = remote_tp_size // tp_size
            remote_block_lens = [bl // tp_ratio_neg for bl in block_len_per_layer]

        plan = generate_dense_plan(
            **_common_plan_params(
                tp_size=tp_size,
                num_kv_heads=num_kv_heads,
                block_size=block_size,
                block_len_per_layer=block_len_per_layer,
                remote_tp_size=remote_tp_size,
                remote_block_size=block_size,
                remote_num_blocks=num_blocks,
                remote_block_lens=remote_block_lens,
            ),
        )

        local_ids = ([1, 2, 3],)
        remote_ids = ([4, 5, 6],)
        specs = NixlConnectorWorker._compute_read_specs_from_plan(plan, local_ids, remote_ids)

        assert len(specs) == len(plan.all_source_ranks)
        for spec in specs:
            assert list(spec.local_block_ids[0]) == [1, 2, 3]
            assert list(spec.remote_block_ids[0]) == [4, 5, 6]

    @pytest.mark.parametrize("remote_tp_size", [2, 4])
    def test_build_src_split_handles(self, remote_tp_size):
        tp_rank = 0
        tp_size = 1
        num_kv_heads = 8
        block_size = 16
        num_blocks = 64
        num_layers = 2
        slot_size = num_kv_heads * 128 * 2
        block_len = slot_size * block_size
        block_len_per_layer = [block_len] * num_layers

        tp_ratio_neg = remote_tp_size // tp_size
        remote_block_lens = [bl // tp_ratio_neg for bl in block_len_per_layer]

        plan = generate_dense_plan(
            **_common_plan_params(
                tp_rank=tp_rank,
                tp_size=tp_size,
                num_kv_heads=num_kv_heads,
                block_size=block_size,
                block_len_per_layer=block_len_per_layer,
                remote_tp_size=remote_tp_size,
                remote_block_size=block_size,
                remote_num_blocks=num_blocks,
                remote_block_lens=remote_block_lens,
            ),
        )

        src_blocks_data = [(0x2000 + i * 1024, 1024, 0) for i in range(8)]
        num_descs = len(src_blocks_data)
        splits = NixlConnectorWorker._build_local_splits_from_plan(
            plan,
            src_blocks_data,
            num_descs,
        )

        assert len(splits) == remote_tp_size
        for handle in splits:
            assert len(handle) == len(src_blocks_data)
            for _, length, _ in handle:
                assert length == 1024 // remote_tp_size


class TestDensePlanStructure:
    def test_source_ranks_homogeneous(self):
        plan = generate_dense_plan(
            **_common_plan_params(tp_size=2, tp_rank=1, remote_tp_size=2),
        )
        assert plan.all_source_ranks == (1,)

    def test_source_ranks_d_gt_p(self):
        plan = generate_dense_plan(
            **_common_plan_params(tp_size=4, tp_rank=2, remote_tp_size=2),
        )
        assert plan.all_source_ranks == (1,)

    def test_source_ranks_p_gt_d(self):
        plan = generate_dense_plan(
            **_common_plan_params(tp_size=1, tp_rank=0, remote_tp_size=2),
        )
        assert plan.all_source_ranks == (0, 1)

    def test_no_ssm_regions(self):
        plan = generate_dense_plan(**_common_plan_params())
        assert plan.ssm_regions == ()
        assert plan.group_spec_types == (FullAttentionSpec,)

    def test_blocks_first_has_k_and_v(self):
        plan = generate_dense_plan(
            **_common_plan_params(is_blocks_first=True),
        )
        num_layers = 2
        assert len(plan.fa_regions) == num_layers * 2  # K + V per layer

    def test_not_blocks_first_has_only_k(self):
        plan = generate_dense_plan(
            **_common_plan_params(is_blocks_first=False),
        )
        num_layers = 2
        assert len(plan.fa_regions) == num_layers  # K only per layer


# ======================================================================
# Mamba equivalence tests
# ======================================================================


def _make_mamba_plan_for_desc_ids(
    num_fa_regions: int,
    num_ssm_regions: int,
    group_spec_types: tuple[type, ...],
    fa_num_blocks: int = 100,
    ssm_num_blocks: int = 100,
) -> EngineTransferPlan:
    """Build a minimal plan with enough structure for compute_desc_ids."""
    fa_regions = tuple(
        RegionPlan(
            layer_idx=i,
            descriptor_bytes=100,
            offset_in_page=0,
            page_stride=100,
            num_blocks=fa_num_blocks,
        )
        for i in range(num_fa_regions)
    )
    ssm_regions = tuple(
        RegionPlan(
            layer_idx=i % (num_ssm_regions // 4) if num_ssm_regions >= 4 else 0,
            descriptor_bytes=50,
            offset_in_page=0,
            page_stride=200,
            num_blocks=ssm_num_blocks,
        )
        for i in range(num_ssm_regions)
    )
    all_ranks = (0,)
    source_ranks_per_group = tuple(all_ranks for _ in group_spec_types)
    return EngineTransferPlan(
        fa_regions=fa_regions,
        ssm_regions=ssm_regions,
        group_spec_types=group_spec_types,
        source_ranks_per_group=source_ranks_per_group,
        all_source_ranks=(0,),
        rank_to_attention_slot={0: 0},
        remote_expansion_stride=1,
    )


class TestMambaPlanDescIds:
    """Verify plan-based desc IDs for hybrid FA+SSM models."""

    def test_hybrid_ssm_ratio_1(self):
        """Equivalent to test_get_block_descs_ids_hybrid_ssm."""
        plan = _make_mamba_plan_for_desc_ids(
            num_fa_regions=2,
            num_ssm_regions=4,  # 4 regions per layer, 1 layer
            group_spec_types=(FullAttentionSpec, MambaSpec),
            fa_num_blocks=100,
            ssm_num_blocks=100,
        )

        fa_blocks = [3, 5]
        ssm_blocks = [1, 2]

        result = NixlConnectorWorker._compute_desc_ids_from_plan(
            plan,
            block_ids=(fa_blocks, ssm_blocks),
            dst_num_blocks=100,
            block_size_ratio=None,
            physical_blocks_per_logical=1,
        )

        expected = [3, 5, 103, 105, 201, 202, 301, 302, 401, 402, 501, 502]
        assert list(result) == expected, f"Expected {expected}, got {list(result)}"

    def test_kernel_block_mismatch(self):
        """Equivalent to test_get_block_descs_ids_kernel_block_mismatch."""
        ratio = 4
        logical_blocks = 100
        num_blocks = logical_blocks * ratio  # 400

        plan = _make_mamba_plan_for_desc_ids(
            num_fa_regions=2,
            num_ssm_regions=4,
            group_spec_types=(FullAttentionSpec, MambaSpec),
            fa_num_blocks=num_blocks,
            ssm_num_blocks=logical_blocks,
        )

        fa_blocks = [3, 7]
        ssm_blocks = [1, 2]

        result = NixlConnectorWorker._compute_desc_ids_from_plan(
            plan,
            block_ids=(fa_blocks, ssm_blocks),
            dst_num_blocks=num_blocks,
            block_size_ratio=None,
            physical_blocks_per_logical=ratio,
        )

        expected = [3, 7, 403, 407, 801, 802, 901, 902, 1001, 1002, 1101, 1102]
        assert list(result) == expected, f"Expected {expected}, got {list(result)}"


class TestMambaPlanReadSpecs:
    """Verify plan-based read specs handle FA group filtering correctly."""

    def test_all_source_ranks_serve_fa(self):
        """When all ranks are FA sources, no filtering happens."""
        both = (0, 1)
        plan = EngineTransferPlan(
            fa_regions=(),
            ssm_regions=(),
            group_spec_types=(FullAttentionSpec, MambaSpec),
            source_ranks_per_group=(both, both),
            all_source_ranks=(0, 1),
            rank_to_attention_slot={0: 0, 1: 1},
            remote_expansion_stride=1,
        )

        local_ids = ([1, 2], [3, 4])
        remote_ids = ([5, 6], [7, 8])

        specs = NixlConnectorWorker._compute_read_specs_from_plan(plan, local_ids, remote_ids)
        assert len(specs) == 2
        for spec in specs:
            assert list(spec.local_block_ids[0]) == [1, 2]
            assert list(spec.local_block_ids[1]) == [3, 4]

    def test_non_fa_rank_skips_fa_groups(self):
        """Ranks not in source_ranks_per_group get groups zeroed out."""
        fa_readers = (0,)
        ssm_readers = (0, 1, 2)
        plan = EngineTransferPlan(
            fa_regions=(),
            ssm_regions=(),
            group_spec_types=(FullAttentionSpec, MambaSpec),
            source_ranks_per_group=(fa_readers, ssm_readers),
            all_source_ranks=(0, 1, 2),
            rank_to_attention_slot={0: 0},
            remote_expansion_stride=1,
        )

        local_ids = ([1, 2], [3, 4])
        remote_ids = ([5, 6], [7, 8])

        specs = NixlConnectorWorker._compute_read_specs_from_plan(plan, local_ids, remote_ids)
        assert len(specs) == 3

        # Rank 0 (FA source): gets all groups
        assert list(specs[0].local_block_ids[0]) == [1, 2]
        assert list(specs[0].local_block_ids[1]) == [3, 4]

        # Rank 1 (not FA): FA group zeroed, Mamba group preserved
        assert specs[1].local_block_ids[0] == []
        assert list(specs[1].local_block_ids[1]) == [3, 4]

        # Rank 2 (not FA): same
        assert specs[2].local_block_ids[0] == []
        assert list(specs[2].local_block_ids[1]) == [3, 4]


class TestMambaPlanSplitHandles:
    """Verify plan-based split handles for Mamba with FA/SSM distinction."""

    def test_fa_and_ssm_different_split_factors(self):
        """Section 0 split by num_attn_reads, section 1 by abs_tp."""
        fa_readers = (0,)
        ssm_readers = (0, 1)
        plan = EngineTransferPlan(
            fa_regions=(),
            ssm_regions=(
                RegionPlan(
                    layer_idx=0,
                    descriptor_bytes=100,
                    offset_in_page=0,
                    page_stride=100,
                    num_blocks=10,
                ),
            ),
            group_spec_types=(FullAttentionSpec, MambaSpec),
            source_ranks_per_group=(fa_readers, ssm_readers),
            all_source_ranks=(0, 1),
            rank_to_attention_slot={0: 0, 1: 0},
            remote_expansion_stride=1,
        )

        # 2 FA descs + 1 SSM desc
        src_blocks_data = [
            (1000, 200, 0),  # FA desc 0
            (2000, 200, 0),  # FA desc 1
            (3000, 400, 0),  # SSM desc 0
        ]

        splits = NixlConnectorWorker._build_local_splits_from_plan(plan, src_blocks_data, 2)

        assert len(splits) == 2  # 2 source ranks

        # Rank 0 (FA source, p_idx=0):
        # FA: chunk=200//1=200, slot=0 → (1000, 200, 0), (2000, 200, 0)
        # SSM: chunk=400//2=200, idx=0 → (3000, 200, 0)
        assert splits[0] == [(1000, 200, 0), (2000, 200, 0), (3000, 200, 0)]

        # Rank 1 (not FA source, p_idx=1):
        # FA: chunk=200//1=200, slot=0 (skip_fa) → (1000, 200, 0), (2000, 200, 0)
        # SSM: chunk=400//2=200, idx=1 → (3200, 200, 0)
        assert splits[1] == [(1000, 200, 0), (2000, 200, 0), (3200, 200, 0)]
