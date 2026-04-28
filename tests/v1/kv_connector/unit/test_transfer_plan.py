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
    generate_gemma4_plan,
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
            plan,
            block_ids,
            dst_num_blocks=num_blocks,
            block_size_ratio=None,
            physical_blocks_per_logical=1,
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
        specs = NixlConnectorWorker._compute_read_specs_from_plan(
            plan, local_ids, remote_ids
        )

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
        rank_to_attention_slot=({0: 0},) * len(group_kinds),
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
            rank_to_attention_slot=({0: 0, 1: 1}, {0: 0, 1: 1}),
            remote_expansion_stride=1,
        )

        local_ids = ([1, 2], [3, 4])
        remote_ids = ([5, 6], [7, 8])

        specs = NixlConnectorWorker._compute_read_specs_from_plan(
            plan, local_ids, remote_ids
        )
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
            rank_to_attention_slot=({0: 0}, {0: 0}),
            remote_expansion_stride=1,
        )

        local_ids = ([1, 2], [3, 4])
        remote_ids = ([5, 6], [7, 8])

        specs = NixlConnectorWorker._compute_read_specs_from_plan(
            plan, local_ids, remote_ids
        )
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
            rank_to_attention_slot=({0: 0, 1: 0}, {0: 0, 1: 0}),
            remote_expansion_stride=1,
        )

        # 2 FA descs + 1 SSM desc
        src_blocks_data = [
            (1000, 200, 0),  # FA desc 0
            (2000, 200, 0),  # FA desc 1
            (3000, 400, 0),  # SSM desc 0
        ]

        splits = NixlConnectorWorker._build_local_splits_from_plan(
            plan, src_blocks_data, 2
        )

        assert len(splits) == 2  # 2 source ranks

        # Rank 0 (FA source, p_idx=0):
        # FA: chunk=200//1=200, slot=0 → (1000, 200, 0), (2000, 200, 0)
        # SSM: chunk=400//2=200, idx=0 → (3000, 200, 0)
        assert splits[0] == [(1000, 200, 0), (2000, 200, 0), (3000, 200, 0)]

        # Rank 1 (not FA source, p_idx=1):
        # FA: chunk=200//1=200, slot=0 (skip_fa) → (1000, 200, 0), (2000, 200, 0)
        # SSM: chunk=400//2=200, idx=1 → (3200, 200, 0)
        assert splits[1] == [(1000, 200, 0), (2000, 200, 0), (3200, 200, 0)]


# ======================================================================
# Gemma4 HeteroTP tests
# ======================================================================


def _make_gemma4_plan_params(
    tp_rank: int = 0,
    tp_size: int = 4,
    remote_tp_size: int = 2,
) -> dict:
    """Build kwargs for generate_gemma4_plan at 2p4d.

    Gemma4-26B at P_TP=2, D_TP=4:
      SWA: 25 layers, K=8, head_dim=256, block_size=16 on both sides
      FA:   5 layers, K=2, head_dim=512, P block_size=32, D block_size=16

    With page unification + HMA, all groups share one physical pool.
    page_size: P=65536, D=32768 → remote_to_local_page_ratio=2.
    For simplicity, use 2 physical layers in tests.
    """
    # D side (local): kv_heads_per_rank for all groups = page_size / block_size
    # page_size = 32768 for both groups at D_TP=4.
    d_page = 32768
    p_page = 65536
    num_layers = 2

    return dict(
        transfer_topo=_make_fake_topo(
            tp_rank=tp_rank,
            tp_size=tp_size,
            is_mla=False,
            total_num_kv_heads=8,
            block_size=16,
            is_blocks_first=False,
        ),
        block_len_per_layer=[d_page] * num_layers,
        remote_info=EngineTransferInfo(
            remote_tp_size=remote_tp_size,
            remote_block_size=16,
            remote_block_len=p_page,
            remote_physical_blocks_per_logical=1,
        ),
        remote_meta=_make_nixl_meta(
            base_addrs=[0x10000 * (i + 1) for i in range(num_layers)],
            num_blocks=500,
            block_lens=[p_page] * num_layers,
            block_size=16,
        ),
        group_kinds=(GroupKind.SWA, GroupKind.FA),
        total_num_kv_heads_per_group=(8, 2),
        local_tokens_per_block=(16, 16),
        remote_tokens_per_block=(16, 32),
    )


class TestGemma4PlanStructure:
    """Verify plan structure for Gemma4-style heterogeneous attention."""

    def test_plan_fields_2p4d_rank0(self):
        """D rank 0 at 2p4d: ratio=2, SWA head-split, FA multi-block."""
        plan = generate_gemma4_plan(**_make_gemma4_plan_params(tp_rank=0))

        assert plan.remote_to_local_page_ratio == 2
        assert plan.group_kinds == (GroupKind.SWA, GroupKind.FA)
        assert plan.local_blocks_per_remote_block == (1, 2)
        assert plan.remote_desc_offset_per_group == (0, 0)  # rank 0: index=0
        assert plan.all_source_ranks == (0,)
        assert plan.source_ranks_per_group == ((0,), (0,))

    def test_plan_fields_2p4d_rank1(self):
        """D rank 1 at 2p4d: SWA reads second descriptor (index=1)."""
        plan = generate_gemma4_plan(**_make_gemma4_plan_params(tp_rank=1))

        assert plan.remote_desc_offset_per_group == (1, 0)  # rank 1: SWA=1
        assert plan.local_blocks_per_remote_block == (1, 2)
        assert plan.all_source_ranks == (0,)

    def test_plan_fields_2p4d_rank2(self):
        """D rank 2 reads from P rank 1."""
        plan = generate_gemma4_plan(**_make_gemma4_plan_params(tp_rank=2))

        assert plan.all_source_ranks == (1,)
        assert plan.remote_desc_offset_per_group == (0, 0)

    def test_fa_regions_have_multiple_descs_per_block(self):
        """FA regions should have descs_per_block = page ratio."""
        plan = generate_gemma4_plan(**_make_gemma4_plan_params())

        for region in plan.fa_regions:
            assert region.descs_per_block == 2
            assert region.desc_stride_bytes == 32768  # D page size


class TestGemma4RemoteDescs:
    """Verify remote descriptor building with sub-descriptors."""

    def test_descs_per_block(self):
        """Each region produces num_blocks * descs_per_block descriptors."""
        plan = generate_gemma4_plan(**_make_gemma4_plan_params())
        meta = _make_nixl_meta(
            base_addrs=[0x10000, 0x20000],
            num_blocks=500,
            block_lens=[65536, 65536],
        )
        descs = build_remote_descs_from_plan(plan, meta)

        # 2 layers × 1 region/layer × 500 blocks × 2 descs/block = 2000
        expected_count = 2 * 500 * 2
        assert len(descs) == expected_count

    def test_desc_stride_within_block(self):
        """Descriptors within a block should be desc_stride_bytes apart."""
        plan = generate_gemma4_plan(**_make_gemma4_plan_params())
        meta = _make_nixl_meta(
            base_addrs=[0x10000, 0x20000],
            num_blocks=500,
            block_lens=[65536, 65536],
        )
        descs = build_remote_descs_from_plan(plan, meta)

        # First block, layer 0: descriptor 0 and descriptor 1
        addr_d0, len_d0, _ = descs[0]
        addr_d1, len_d1, _ = descs[1]
        assert addr_d1 - addr_d0 == 32768  # desc_stride_bytes
        assert len_d0 == len_d1 == 32768  # descriptor_bytes


class TestGemma4DescIds:
    """Verify desc ID computation with sub-desc block IDs."""

    def test_remapped_block_ids(self):
        """After remapping, descriptor indices are correct."""
        plan = generate_gemma4_plan(**_make_gemma4_plan_params())

        # SWA blocks [3, 7], FA blocks [10, 11]
        # Remapped to descriptor indices:
        #   SWA (sub_desc_index=0): [3*2+0, 7*2+0] = [6, 14]
        #   FA  (2 local per remote): [10*2, 10*2+1, 11*2, 11*2+1] = [20,21,22,23]
        #
        # dst_num_blocks = 500 * 2 = 1000 (num_blocks * descs_per_block)
        # 2 fa_regions (2 layers), each with 1000 desc slots
        # SWA: [0*1000+6, 0*1000+14, 1*1000+6, 1*1000+14]
        #    = [6, 14, 1006, 1014]
        # FA:  [0*1000+20, 0*1000+21, 0*1000+22, 0*1000+23,
        #       1*1000+20, 1*1000+21, 1*1000+22, 1*1000+23]
        #    = [20, 21, 22, 23, 1020, 1021, 1022, 1023]

        # First remap via read specs to get descriptor-level block IDs
        local_swa = [10, 11]
        local_fa = [20, 21, 22, 23]
        remote_swa = [3, 7]
        remote_fa = [10, 11]

        specs = compute_read_specs_from_plan(
            plan,
            local_block_ids=(local_swa, local_fa),
            remote_block_ids=(remote_swa, remote_fa),
        )
        assert len(specs) == 1  # Single source rank
        spec = specs[0]

        # Verify remapped remote block IDs
        assert list(spec.remote_block_ids[0]) == [6, 14]  # SWA: b*2+0
        assert list(spec.remote_block_ids[1]) == [20, 21, 22, 23]  # FA: 2 per

        # Verify local block IDs unchanged
        assert list(spec.local_block_ids[0]) == [10, 11]
        assert list(spec.local_block_ids[1]) == [20, 21, 22, 23]

        # Now compute desc IDs with the remapped remote blocks
        remote_ids = compute_desc_ids_from_plan(
            plan,
            block_ids=spec.remote_block_ids,
            dst_num_blocks=500 * 2,  # num_blocks * descs_per_block
        )
        expected_remote = [6, 14, 1006, 1014, 20, 21, 22, 23, 1020, 1021, 1022, 1023]
        assert list(remote_ids) == expected_remote

        # Local desc IDs (standard, descs_per_block=1 locally)
        local_ids = compute_desc_ids_from_plan(
            plan,
            block_ids=spec.local_block_ids,
            dst_num_blocks=1000,  # local num_blocks
        )
        expected_local = [10, 11, 1010, 1011, 20, 21, 22, 23, 1020, 1021, 1022, 1023]
        assert list(local_ids) == expected_local

        # Both have same length → can be paired for transfer
        assert len(remote_ids) == len(local_ids)


# ======================================================================
# Gemma4 Gather-Read tests (local page > remote page)
# ======================================================================


def _make_gemma4_gather_plan_params(
    tp_rank: int = 0,
    tp_size: int = 2,
    remote_tp_size: int = 4,
) -> dict:
    """Build kwargs for generate_gemma4_plan at 4p2d (gather-read).

    Gemma4-26B at P_TP=4, D_TP=2:
      SWA: K=8, head_dim=256, P_tpb=16, D_tpb=16  → concat (2 P ranks)
      FA:  K=2, head_dim=512, P_tpb=16, D_tpb=32  → gather (2P→1D block)

    page_size: P=32768, D=65536 → local_to_remote_page_ratio=2.
    """
    d_page = 65536
    p_page = 32768
    num_layers = 2

    return dict(
        transfer_topo=_make_fake_topo(
            tp_rank=tp_rank,
            tp_size=tp_size,
            is_mla=False,
            total_num_kv_heads=8,
            block_size=16,
            is_blocks_first=False,
        ),
        block_len_per_layer=[d_page] * num_layers,
        remote_info=EngineTransferInfo(
            remote_tp_size=remote_tp_size,
            remote_block_size=16,
            remote_block_len=p_page,
            remote_physical_blocks_per_logical=1,
        ),
        remote_meta=_make_nixl_meta(
            base_addrs=[0x10000 * (i + 1) for i in range(num_layers)],
            num_blocks=500,
            block_lens=[p_page] * num_layers,
            block_size=16,
        ),
        group_kinds=(GroupKind.SWA, GroupKind.FA),
        total_num_kv_heads_per_group=(8, 2),
        local_tokens_per_block=(16, 32),
        remote_tokens_per_block=(16, 16),
    )


class TestGemma4GatherReadPlanStructure:
    """Verify plan structure for gather-read (4p2d)."""

    def test_plan_fields_4p2d_rank0(self):
        """D rank 0 at 4p2d: gather_ratio=2, SWA concat, FA gather."""
        plan = generate_gemma4_plan(**_make_gemma4_gather_plan_params(tp_rank=0))

        assert plan.local_to_remote_page_ratio == 2
        assert plan.remote_to_local_page_ratio == 1
        assert plan.group_kinds == (GroupKind.SWA, GroupKind.FA)
        assert plan.remote_blocks_per_local_block == (1, 2)
        assert plan.local_blocks_per_remote_block == (1, 1)
        # SWA: D rank 0 reads from P rank 0 and P rank 1
        assert (0,) in plan.source_ranks_per_group[0] or len(
            plan.source_ranks_per_group[0]
        ) == 2
        # FA: after GQA dedup, D rank 0 reads from P rank 0 only
        assert len(plan.source_ranks_per_group[1]) == 1

    def test_no_assertion_error(self):
        """4p2d should NOT crash (old code had assert page_ratio >= 1)."""
        plan = generate_gemma4_plan(**_make_gemma4_gather_plan_params())
        assert plan is not None

    def test_fa_regions_standard_descs(self):
        """Gather-read: FA regions have descs_per_block=1 (standard)."""
        plan = generate_gemma4_plan(**_make_gemma4_gather_plan_params())

        for region in plan.fa_regions:
            assert region.descs_per_block == 1
            assert region.descriptor_bytes == 32768  # remote page size


class TestGemma4GatherReadRemoteDescs:
    """Verify remote descriptor building for gather-read."""

    def test_standard_descs_per_block(self):
        """Gather-read: 1 desc per block (no remote sub-descs)."""
        plan = generate_gemma4_plan(**_make_gemma4_gather_plan_params())
        meta = _make_nixl_meta(
            base_addrs=[0x10000, 0x20000],
            num_blocks=500,
            block_lens=[32768, 32768],
        )
        descs = build_remote_descs_from_plan(plan, meta)

        # 2 layers × 1 region/layer × 500 blocks × 1 desc/block = 1000
        assert len(descs) == 2 * 500 * 1

    def test_desc_bytes_match_remote_page(self):
        """Each remote desc should be remote_page_size bytes."""
        plan = generate_gemma4_plan(**_make_gemma4_gather_plan_params())
        meta = _make_nixl_meta(
            base_addrs=[0x10000, 0x20000],
            num_blocks=500,
            block_lens=[32768, 32768],
        )
        descs = build_remote_descs_from_plan(plan, meta)

        for _, length, _ in descs:
            assert length == 32768


class TestGemma4GatherReadSpecs:
    """Verify read spec computation for gather-read."""

    def test_gather_read_specs_4p2d_rank0(self):
        """4p2d rank 0: SWA from 2 ranks, FA from 1 rank (gather)."""
        plan = generate_gemma4_plan(**_make_gemma4_gather_plan_params(tp_rank=0))

        # D has 2 SWA blocks and 1 FA block (32 tokens)
        local_swa = [10, 11]
        local_fa = [20]
        # P has 2 SWA blocks per rank and 2 FA blocks (16 tokens each)
        remote_swa = [5, 6]
        remote_fa = [30, 31]

        specs = compute_read_specs_from_plan(
            plan,
            local_block_ids=(local_swa, local_fa),
            remote_block_ids=(remote_swa, remote_fa),
        )

        # SWA reads from 2 P ranks → 2 specs
        assert len(specs) == 2

        # Spec 0 (P rank 0):
        # SWA: local sub-desc slot 0 → [10*2+0, 11*2+0] = [20, 22]
        # FA: expanded → [20*2+0, 20*2+1] = [40, 41]
        spec0 = specs[0]
        assert list(spec0.local_block_ids[0]) == [20, 22]  # SWA slot 0
        assert list(spec0.local_block_ids[1]) == [40, 41]  # FA gather
        assert list(spec0.remote_block_ids[0]) == [5, 6]  # SWA blocks
        assert list(spec0.remote_block_ids[1]) == [30, 31]  # FA blocks

        # Spec 1 (P rank 1):
        # SWA: local sub-desc slot 1 → [10*2+1, 11*2+1] = [21, 23]
        # FA: empty (rank 1 not in FA source_ranks after GQA dedup)
        spec1 = specs[1]
        assert list(spec1.local_block_ids[0]) == [21, 23]  # SWA slot 1
        assert list(spec1.remote_block_ids[0]) == [5, 6]  # SWA blocks
        assert spec1.local_block_ids[1] == []  # FA empty for rank 1
        assert spec1.remote_block_ids[1] == []

    def test_gather_read_desc_ids_match(self):
        """Local and remote desc IDs should have same length for NIXL."""
        plan = generate_gemma4_plan(**_make_gemma4_gather_plan_params(tp_rank=0))

        local_swa = [10, 11]
        local_fa = [20]
        remote_swa = [5, 6]
        remote_fa = [30, 31]

        specs = compute_read_specs_from_plan(
            plan,
            local_block_ids=(local_swa, local_fa),
            remote_block_ids=(remote_swa, remote_fa),
        )

        for spec in specs:
            # Remote desc IDs: standard (no sub-descs), num_blocks=500
            remote_ids = compute_desc_ids_from_plan(
                plan,
                block_ids=spec.remote_block_ids,
                dst_num_blocks=500,
            )
            # Local desc IDs: gather sub-descs, num_blocks=1000*gather_ratio
            local_ids = compute_desc_ids_from_plan(
                plan,
                block_ids=spec.local_block_ids,
                dst_num_blocks=1000 * 2,  # local_num_blocks * gather_ratio
            )
            assert len(remote_ids) == len(local_ids), (
                f"Desc ID length mismatch for rank {spec.remote_rank}: "
                f"remote={len(remote_ids)}, local={len(local_ids)}"
            )


class TestGemma4GatherReadPlan4p1d:
    """Verify gather-read for 4p1d (D_TP=1, P_TP=4)."""

    def test_4p1d_no_crash(self):
        """4p1d should not crash."""
        params = _make_gemma4_gather_plan_params(tp_rank=0, tp_size=1, remote_tp_size=4)
        # D_TP=1: D_page = 131072 (8 heads * 256 * 2 * 16 * 2 for SWA)
        # P_TP=4: P_page = 32768
        params["block_len_per_layer"] = [131072, 131072]
        params["local_tokens_per_block"] = (16, 32)
        params["remote_tokens_per_block"] = (16, 16)
        plan = generate_gemma4_plan(**params)

        assert plan.local_to_remote_page_ratio == 4
        assert plan.remote_to_local_page_ratio == 1
        assert plan.remote_blocks_per_local_block == (1, 2)
