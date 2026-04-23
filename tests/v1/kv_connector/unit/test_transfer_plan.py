# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Equivalence tests: plan-based executors vs current ABC policy.

These tests verify that the new plan-based design produces identical
outputs (descriptor tuples, descriptor IDs, read specs) to the current
ModelBlockTransferPolicy ABC hierarchy.  No GPU or NIXL required.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from vllm.distributed.kv_transfer.kv_connector.utils import (
    TransferTopology,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.block_transfer_policy import (
    DenseModelBlockTransferPolicy,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.transfer_plan import (
    EngineTransferPlan,
    RegionKind,
    RegionPlan,
    build_local_splits_from_plan,
    build_remote_descs_from_plan,
    compute_desc_ids_from_plan,
    compute_read_specs_from_plan,
    generate_dense_plan,
    visualize_plan,
)

# ======================================================================
# Test fixtures / helpers
# ======================================================================

ENGINE_ID = "remote_engine"
LOCAL_ENGINE_ID = "local_engine"


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


def _make_kv_cache_config(
    block_size: int = 16,
    num_blocks: int = 256,
    num_layers: int = 2,
    head_size: int = 128,
    num_kv_heads: int = 8,
):
    """Create a minimal KVCacheConfig for Dense models."""
    import torch

    from vllm.v1.kv_cache_interface import (
        FullAttentionSpec,
        KVCacheConfig,
        KVCacheGroupSpec,
    )

    spec = FullAttentionSpec(
        block_size=block_size,
        num_kv_heads=num_kv_heads,
        head_size=head_size,
        dtype=torch.float16,
    )
    layers = [f"layer_{i}" for i in range(num_layers)]
    return KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=[],
        kv_cache_groups=[KVCacheGroupSpec(layers, spec)],
    )


def _make_transfer_topo(
    tp_rank: int = 0,
    tp_size: int = 1,
    block_size: int = 16,
    is_mla: bool = False,
    num_kv_heads: int = 8,
):
    """Create a TransferTopology for testing without real attention backend."""
    from vllm.v1.attention.backends.flash_attn import FlashAttentionBackend

    return TransferTopology(
        tp_rank=tp_rank,
        tp_size=tp_size,
        block_size=block_size,
        engine_id=LOCAL_ENGINE_ID,
        is_mla=is_mla,
        is_mamba=False,
        total_num_kv_heads=num_kv_heads,
        attn_backends=[FlashAttentionBackend],
        physical_blocks_per_logical=1,
    )


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
) -> dict:
    """Build common kwargs for plan generators."""
    if block_len_per_layer is None:
        slot_size = num_kv_heads * 128 * 2  # num_heads * head_size * dtype_bytes
        block_len_per_layer = [slot_size * block_size] * 2
    if remote_block_lens is None:
        remote_block_lens = list(block_len_per_layer)
    return dict(
        tp_rank=tp_rank,
        tp_size=tp_size,
        is_mla=is_mla,
        total_num_kv_heads=num_kv_heads,
        is_blocks_first=is_blocks_first,
        block_len_per_layer=block_len_per_layer,
        block_size=block_size,
        remote_tp_size=remote_tp_size,
        remote_block_size=remote_block_size,
        remote_num_blocks=remote_num_blocks,
        remote_block_lens=remote_block_lens,
        remote_physical_blocks_per_logical=remote_physical_blocks_per_logical,
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


class TestDensePlanEquivalence:
    """Verify plan-based outputs match current DenseModelBlockTransferPolicy."""

    @pytest.mark.parametrize(
        "tp_size,remote_tp_size",
        [
            (1, 1),  # homogeneous
            (2, 1),  # D_TP > P_TP
            (4, 2),  # D_TP > P_TP (larger)
            (1, 2),  # P_TP > D_TP
            (2, 4),  # P_TP > D_TP (larger)
        ],
    )
    @pytest.mark.parametrize("tp_rank_frac", [0.0, 0.5])
    def test_build_remote_descs(
        self,
        tp_size,
        remote_tp_size,
        tp_rank_frac,
    ):
        tp_rank = int(tp_rank_frac * (tp_size - 1)) if tp_size > 1 else 0
        num_kv_heads = 8
        head_size = 128
        block_size = 16
        num_blocks = 64
        num_layers = 2
        slot_size = num_kv_heads * head_size * 2
        block_len = slot_size * block_size
        block_len_per_layer = [block_len] * num_layers

        # Adjust remote block_lens for hetero TP
        if tp_size >= remote_tp_size:
            tp_ratio = tp_size // remote_tp_size
            remote_block_lens = [bl * tp_ratio for bl in block_len_per_layer]
        else:
            tp_ratio_neg = remote_tp_size // tp_size
            remote_block_lens = [bl // tp_ratio_neg for bl in block_len_per_layer]

        base_addrs = [0x1000 * (i + 1) for i in range(num_layers)]

        # ---- Old path ----
        kv_config = _make_kv_cache_config(
            block_size=block_size,
            num_blocks=num_blocks,
            num_layers=num_layers,
            num_kv_heads=num_kv_heads,
            head_size=head_size,
        )
        policy = DenseModelBlockTransferPolicy(kv_config, 1)
        topo = _make_transfer_topo(
            tp_rank=tp_rank,
            tp_size=tp_size,
            block_size=block_size,
            num_kv_heads=num_kv_heads,
        )
        is_blocks_first = topo.is_kv_layout_blocks_first
        transfer_info = policy.build_engine_transfer_info(
            transfer_topo=topo,
            local_block_len=block_len_per_layer[0],
            remote_tp_size=remote_tp_size,
            remote_block_size=block_size,
            remote_block_len=remote_block_lens[0],
            remote_physical_blocks_per_logical=1,
        )
        topo.register_remote_engine(ENGINE_ID, transfer_info)
        meta = _make_nixl_meta(
            base_addrs,
            num_blocks,
            remote_block_lens,
            block_size=block_size,
        )
        old_descs = policy.build_remote_descs(
            topo,
            ENGINE_ID,
            meta,
            block_len_per_layer,
        )

        # ---- New path ----
        plan = generate_dense_plan(
            **_common_plan_params(
                tp_rank=tp_rank,
                tp_size=tp_size,
                num_kv_heads=num_kv_heads,
                block_size=block_size,
                is_blocks_first=is_blocks_first,
                block_len_per_layer=block_len_per_layer,
                remote_tp_size=remote_tp_size,
                remote_block_size=block_size,
                remote_num_blocks=num_blocks,
                remote_block_lens=remote_block_lens,
            ),
        )
        new_descs = build_remote_descs_from_plan(plan, meta)

        assert old_descs == new_descs, (
            f"Descriptor mismatch for tp={tp_size}/{remote_tp_size}, "
            f"rank={tp_rank}.\nOld: {old_descs[:5]}...\nNew: {new_descs[:5]}..."
        )

    @pytest.mark.parametrize(
        "tp_size,remote_tp_size",
        [
            (1, 1),
            (2, 1),
            (1, 2),
        ],
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

        topo = _make_transfer_topo(
            tp_size=tp_size,
            block_size=block_size,
            num_kv_heads=num_kv_heads,
        )
        is_blocks_first = topo.is_kv_layout_blocks_first

        kv_config = _make_kv_cache_config(
            block_size=block_size,
            num_blocks=num_blocks,
            num_layers=num_layers,
            num_kv_heads=num_kv_heads,
        )
        policy = DenseModelBlockTransferPolicy(kv_config, 1)
        plan = generate_dense_plan(
            **_common_plan_params(
                tp_size=tp_size,
                num_kv_heads=num_kv_heads,
                block_size=block_size,
                is_blocks_first=is_blocks_first,
                block_len_per_layer=block_len_per_layer,
                remote_tp_size=remote_tp_size,
                remote_block_size=block_size,
                remote_num_blocks=num_blocks,
                remote_block_lens=remote_block_lens,
            ),
        )

        num_regions = len(plan.fa_regions)
        block_ids = ([1, 5, 10, 20],)

        old_ids = policy.get_block_descs_ids(
            block_ids=block_ids,
            num_regions=num_regions,
            dst_num_blocks=num_blocks,
            block_len_per_layer=block_len_per_layer,
        )
        new_ids = compute_desc_ids_from_plan(
            plan,
            block_ids,
            dst_num_blocks=num_blocks,
        )

        np.testing.assert_array_equal(old_ids, new_ids)

    @pytest.mark.parametrize(
        "tp_size,remote_tp_size",
        [
            (1, 1),
            (2, 1),
            (1, 2),
        ],
    )
    def test_compute_read_specs(self, tp_size, remote_tp_size):
        tp_rank = 0
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

        kv_config = _make_kv_cache_config(
            block_size=block_size,
            num_blocks=num_blocks,
            num_layers=num_layers,
            num_kv_heads=num_kv_heads,
        )
        policy = DenseModelBlockTransferPolicy(kv_config, 1)
        topo = _make_transfer_topo(
            tp_rank=tp_rank,
            tp_size=tp_size,
            block_size=block_size,
            num_kv_heads=num_kv_heads,
        )
        is_blocks_first = topo.is_kv_layout_blocks_first
        transfer_info = policy.build_engine_transfer_info(
            transfer_topo=topo,
            local_block_len=block_len_per_layer[0],
            remote_tp_size=remote_tp_size,
            remote_block_size=block_size,
            remote_block_len=remote_block_lens[0],
            remote_physical_blocks_per_logical=1,
        )
        topo.register_remote_engine(ENGINE_ID, transfer_info)
        remote_ranks = topo.target_remote_ranks(ENGINE_ID)

        plan = generate_dense_plan(
            **_common_plan_params(
                tp_rank=tp_rank,
                tp_size=tp_size,
                num_kv_heads=num_kv_heads,
                block_size=block_size,
                is_blocks_first=is_blocks_first,
                block_len_per_layer=block_len_per_layer,
                remote_tp_size=remote_tp_size,
                remote_block_size=block_size,
                remote_num_blocks=num_blocks,
                remote_block_lens=remote_block_lens,
            ),
        )

        local_ids = ([1, 2, 3],)
        remote_ids = ([4, 5, 6],)

        old_specs = policy.compute_read_specs(
            local_ids,
            remote_ids,
            remote_ranks,
            transfer_info,
        )
        new_specs = compute_read_specs_from_plan(plan, local_ids, remote_ids)

        assert len(old_specs) == len(new_specs)
        for old, new in zip(old_specs, new_specs):
            assert old.remote_rank == new.remote_rank
            assert list(old.local_block_ids[0]) == list(new.local_block_ids[0])
            assert list(old.remote_block_ids[0]) == list(new.remote_block_ids[0])

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

        kv_config = _make_kv_cache_config(
            block_size=block_size,
            num_blocks=num_blocks,
            num_layers=num_layers,
            num_kv_heads=num_kv_heads,
        )
        policy = DenseModelBlockTransferPolicy(kv_config, 1)
        topo = _make_transfer_topo(
            tp_rank=tp_rank,
            tp_size=tp_size,
            block_size=block_size,
            num_kv_heads=num_kv_heads,
        )
        is_blocks_first = topo.is_kv_layout_blocks_first
        transfer_info = policy.build_engine_transfer_info(
            transfer_topo=topo,
            local_block_len=block_len_per_layer[0],
            remote_tp_size=remote_tp_size,
            remote_block_size=block_size,
            remote_block_len=remote_block_lens[0],
            remote_physical_blocks_per_logical=1,
        )
        topo.register_remote_engine(ENGINE_ID, transfer_info)

        plan = generate_dense_plan(
            **_common_plan_params(
                tp_rank=tp_rank,
                tp_size=tp_size,
                num_kv_heads=num_kv_heads,
                block_size=block_size,
                is_blocks_first=is_blocks_first,
                block_len_per_layer=block_len_per_layer,
                remote_tp_size=remote_tp_size,
                remote_block_size=block_size,
                remote_num_blocks=num_blocks,
                remote_block_lens=remote_block_lens,
            ),
        )

        src_blocks_data = [(0x2000 + i * 1024, 1024, 0) for i in range(8)]
        num_descs = len(src_blocks_data)

        old_splits = policy.build_src_split_handles(
            topo,
            ENGINE_ID,
            src_blocks_data,
            num_descs,
        )
        new_splits = build_local_splits_from_plan(
            plan,
            src_blocks_data,
            num_descs,
        )

        assert len(old_splits) == len(new_splits), (
            f"Split count mismatch: {len(old_splits)} vs {len(new_splits)}"
        )
        for i, (old, new) in enumerate(zip(old_splits, new_splits)):
            assert old == new, f"Split {i} mismatch"


class TestDensePlanVisualization:
    def test_visualize_produces_output(self):
        plan = generate_dense_plan(
            **_common_plan_params(),
        )
        output = visualize_plan(plan)
        assert "FA regions" in output
        assert "fa_k" in output


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
        assert plan.is_mamba_group == (False,)

    def test_blocks_first_has_k_and_v(self):
        plan = generate_dense_plan(
            **_common_plan_params(is_blocks_first=True),
        )
        kinds = [r.kind.value for r in plan.fa_regions]
        assert "fa_k" in kinds
        assert "fa_v" in kinds

    def test_not_blocks_first_has_only_k(self):
        plan = generate_dense_plan(
            **_common_plan_params(is_blocks_first=False),
        )
        kinds = [r.kind.value for r in plan.fa_regions]
        assert "fa_k" in kinds
        assert "fa_v" not in kinds


# ======================================================================
# Mamba equivalence tests
# ======================================================================


def _make_mamba_plan_for_desc_ids(
    num_fa_regions: int,
    num_ssm_regions: int,
    is_mamba_group: list[bool],
    fa_num_blocks: int = 100,
    ssm_num_blocks: int = 100,
) -> EngineTransferPlan:
    """Build a minimal plan with enough structure for compute_desc_ids."""
    fa_regions = tuple(
        RegionPlan(
            kind=RegionKind.FA_K,
            layer_idx=i,
            descriptor_bytes=100,
            offset_in_page=0,
            page_stride=100,
            num_blocks=fa_num_blocks,
            physical_per_logical=1,
        )
        for i in range(num_fa_regions)
    )
    ssm_regions = tuple(
        RegionPlan(
            kind=RegionKind.SSM_CONV_X,
            layer_idx=i % (num_ssm_regions // 4) if num_ssm_regions >= 4 else 0,
            descriptor_bytes=50,
            offset_in_page=0,
            page_stride=200,
            num_blocks=ssm_num_blocks,
            physical_per_logical=1,
        )
        for i in range(num_ssm_regions)
    )
    physical_per_logical = tuple(1 if m else 1 for m in is_mamba_group)
    return EngineTransferPlan(
        fa_regions=fa_regions,
        ssm_regions=ssm_regions,
        physical_per_logical=physical_per_logical,
        is_mamba_group=tuple(is_mamba_group),
        all_source_ranks=(0,),
        fa_source_ranks=(0,),
        fa_source_set=frozenset({0}),
        num_fa_reads=1,
        num_mamba_reads=1,
        fa_head_slots={0: 0},
        remote_tp_size=1,
        remote_block_size=16,
        remote_block_len=0,
        remote_physical_blocks_per_logical=1,
    )


class TestMambaPlanDescIds:
    """Verify plan-based desc IDs match MambaModelBlockTransferPolicy."""

    def test_hybrid_ssm_ratio_1(self):
        """Equivalent to test_get_block_descs_ids_hybrid_ssm."""
        plan = _make_mamba_plan_for_desc_ids(
            num_fa_regions=2,
            num_ssm_regions=4,  # 4 regions per layer, 1 layer
            is_mamba_group=[False, True],
            fa_num_blocks=100,
            ssm_num_blocks=100,
        )

        fa_blocks = [3, 5]
        ssm_blocks = [1, 2]

        result = compute_desc_ids_from_plan(
            plan,
            block_ids=(fa_blocks, ssm_blocks),
            dst_num_blocks=100,
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
            is_mamba_group=[False, True],
            fa_num_blocks=num_blocks,
            ssm_num_blocks=logical_blocks,
        )

        fa_blocks = [3, 7]
        ssm_blocks = [1, 2]

        result = compute_desc_ids_from_plan(
            plan,
            block_ids=(fa_blocks, ssm_blocks),
            dst_num_blocks=num_blocks,
            physical_blocks_per_logical=ratio,
        )

        expected = [3, 7, 403, 407, 801, 802, 901, 902, 1001, 1002, 1101, 1102]
        assert list(result) == expected, f"Expected {expected}, got {list(result)}"


class TestMambaPlanReadSpecs:
    """Verify plan-based read specs handle FA group filtering correctly."""

    def test_all_source_ranks_serve_fa(self):
        """When all ranks are FA sources, no filtering happens."""
        plan = EngineTransferPlan(
            fa_regions=(),
            ssm_regions=(),
            physical_per_logical=(1, 1),
            is_mamba_group=(False, True),
            all_source_ranks=(0, 1),
            fa_source_ranks=(0, 1),
            fa_source_set=frozenset({0, 1}),
            num_fa_reads=2,
            num_mamba_reads=2,
            fa_head_slots={0: 0, 1: 1},
            remote_tp_size=2,
            remote_block_size=16,
            remote_block_len=0,
            remote_physical_blocks_per_logical=1,
        )

        local_ids = ([1, 2], [3, 4])
        remote_ids = ([5, 6], [7, 8])

        specs = compute_read_specs_from_plan(plan, local_ids, remote_ids)
        assert len(specs) == 2
        for spec in specs:
            assert list(spec.local_block_ids[0]) == [1, 2]
            assert list(spec.local_block_ids[1]) == [3, 4]

    def test_non_fa_rank_skips_fa_groups(self):
        """Ranks not in fa_source_set get FA groups zeroed out."""
        plan = EngineTransferPlan(
            fa_regions=(),
            ssm_regions=(),
            physical_per_logical=(1, 1),
            is_mamba_group=(False, True),
            all_source_ranks=(0, 1, 2),
            fa_source_ranks=(0,),
            fa_source_set=frozenset({0}),
            num_fa_reads=1,
            num_mamba_reads=3,
            fa_head_slots={0: 0},
            remote_tp_size=3,
            remote_block_size=16,
            remote_block_len=0,
            remote_physical_blocks_per_logical=1,
        )

        local_ids = ([1, 2], [3, 4])
        remote_ids = ([5, 6], [7, 8])

        specs = compute_read_specs_from_plan(plan, local_ids, remote_ids)
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
        """FA descs split by num_fa_reads, SSM descs split by abs_tp."""
        plan = EngineTransferPlan(
            fa_regions=(),
            ssm_regions=(
                RegionPlan(
                    kind=RegionKind.SSM_STATE,
                    layer_idx=0,
                    descriptor_bytes=100,
                    offset_in_page=0,
                    page_stride=100,
                    num_blocks=10,
                    physical_per_logical=1,
                ),
            ),
            physical_per_logical=(1, 1),
            is_mamba_group=(False, True),
            all_source_ranks=(0, 1),
            fa_source_ranks=(0,),
            fa_source_set=frozenset({0}),
            num_fa_reads=1,
            num_mamba_reads=2,
            fa_head_slots={0: 0},
            remote_tp_size=2,
            remote_block_size=16,
            remote_block_len=0,
            remote_physical_blocks_per_logical=1,
        )

        # 2 FA descs + 1 SSM desc
        src_blocks_data = [
            (1000, 200, 0),  # FA desc 0
            (2000, 200, 0),  # FA desc 1
            (3000, 400, 0),  # SSM desc 0
        ]
        num_fa_descs = 2

        splits = build_local_splits_from_plan(plan, src_blocks_data, num_fa_descs)

        assert len(splits) == 2  # 2 source ranks

        # Rank 0 (FA source, p_idx=0):
        # FA: chunk=200//1=200, slot=0 → (1000, 200, 0), (2000, 200, 0)
        # SSM: chunk=400//2=200, idx=0 → (3000, 200, 0)
        assert splits[0] == [(1000, 200, 0), (2000, 200, 0), (3000, 200, 0)]

        # Rank 1 (not FA source, p_idx=1):
        # FA: chunk=200//1=200, slot=0 (skip_fa) → (1000, 200, 0), (2000, 200, 0)
        # SSM: chunk=400//2=200, idx=1 → (3200, 200, 0)
        assert splits[1] == [(1000, 200, 0), (2000, 200, 0), (3200, 200, 0)]
