# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for EP weight filtering during model loading."""

import glob
import tempfile
from types import SimpleNamespace

import huggingface_hub.constants
import pytest
import torch

from vllm.config import LoadConfig
from vllm.model_executor.model_loader.default_loader import DefaultModelLoader
from vllm.model_executor.model_loader.ep_weight_filter import (
    compute_eplb_local_expert_ids,
    compute_local_expert_ids,
    parse_expert_id,
    should_skip_weight,
)
from vllm.model_executor.model_loader.weight_utils import (
    safetensors_weights_iterator,
)
from vllm.model_executor.models.interfaces import MixtureOfExperts

# ---------------------------------------------------------------------------
# Unit tests for parse_expert_id
# ---------------------------------------------------------------------------


class TestParseExpertId:
    def test_routed_expert(self):
        name = "model.layers.0.mlp.experts.42.gate_proj.weight"
        assert parse_expert_id(name) == 42

    def test_large_expert_id(self):
        name = "model.layers.60.mlp.experts.383.down_proj.weight"
        assert parse_expert_id(name) == 383

    def test_shared_expert(self):
        # Shared experts use a different naming convention in most models
        name = "model.layers.0.mlp.shared_experts.gate_proj.weight"
        assert parse_expert_id(name) is None

    def test_attention_weight(self):
        name = "model.layers.0.self_attn.q_proj.weight"
        assert parse_expert_id(name) is None

    def test_embedding(self):
        name = "model.embed_tokens.weight"
        assert parse_expert_id(name) is None

    def test_layernorm(self):
        name = "model.layers.0.input_layernorm.weight"
        assert parse_expert_id(name) is None

    def test_fused_3d_expert(self):
        # 3D fused-expert tensors (e.g. gpt-oss) have no numeric expert id.
        # They must NOT be filtered — slicing happens later in weight_loader.
        name = "model.layers.0.mlp.experts.gate_proj.weight"
        assert parse_expert_id(name) is None

    def test_fused_3d_expert_down_proj(self):
        name = "model.layers.10.mlp.experts.down_proj.weight"
        assert parse_expert_id(name) is None

    def test_expert_scale(self):
        # NVFP4 quantized models have scale tensors for experts
        name = "model.layers.5.mlp.experts.100.gate_proj.weight_scale"
        assert parse_expert_id(name) == 100

    def test_expert_zero_id(self):
        name = "model.layers.0.mlp.experts.0.up_proj.weight"
        assert parse_expert_id(name) == 0


# ---------------------------------------------------------------------------
# Unit tests for compute_local_expert_ids
# ---------------------------------------------------------------------------


class TestComputeLocalExpertIds:
    def test_ep_disabled(self):
        assert compute_local_expert_ids(64, ep_size=1, ep_rank=0) is None

    def test_even_split(self):
        # 64 experts, EP=8 → 8 per rank
        ids = compute_local_expert_ids(64, ep_size=8, ep_rank=0)
        assert ids == set(range(0, 8))

        ids = compute_local_expert_ids(64, ep_size=8, ep_rank=7)
        assert ids == set(range(56, 64))

    def test_uneven_split(self):
        # 10 experts, EP=3 → ranks get 4, 3, 3
        ids_0 = compute_local_expert_ids(10, ep_size=3, ep_rank=0)
        ids_1 = compute_local_expert_ids(10, ep_size=3, ep_rank=1)
        ids_2 = compute_local_expert_ids(10, ep_size=3, ep_rank=2)
        assert ids_0 is not None and ids_1 is not None and ids_2 is not None

        assert len(ids_0) == 4
        assert len(ids_1) == 3
        assert len(ids_2) == 3
        # All experts covered, no overlap
        assert ids_0 | ids_1 | ids_2 == set(range(10))
        assert ids_0.isdisjoint(ids_1)
        assert ids_1.isdisjoint(ids_2)

    def test_384_experts_ep8(self):
        # Kimi-K2.5 config: 384 experts, EP=8
        for rank in range(8):
            ids = compute_local_expert_ids(384, ep_size=8, ep_rank=rank)
            assert len(ids) == 48

        # All experts covered
        all_ids: set[int] = set()
        for rank in range(8):
            ids = compute_local_expert_ids(384, ep_size=8, ep_rank=rank)
            assert ids is not None
            all_ids |= ids
        assert all_ids == set(range(384))

    def test_384_experts_ep16(self):
        for rank in range(16):
            ids = compute_local_expert_ids(384, ep_size=16, ep_rank=rank)
            assert len(ids) == 24

    def test_384_experts_ep24(self):
        # 384 / 24 = 16 exactly
        for rank in range(24):
            ids = compute_local_expert_ids(384, ep_size=24, ep_rank=rank)
            assert len(ids) == 16

    # round_robin placement tests

    def test_round_robin_basic(self):
        # 8 experts, EP=2: rank 0 → {0,2,4,6}, rank 1 → {1,3,5,7}
        rr = "round_robin"
        ids_0 = compute_local_expert_ids(8, 2, 0, placement=rr)
        ids_1 = compute_local_expert_ids(8, 2, 1, placement=rr)
        assert ids_0 == {0, 2, 4, 6}
        assert ids_1 == {1, 3, 5, 7}

    def test_round_robin_full_coverage(self):
        # 384 experts, EP=8: all experts covered, no overlap
        rr = "round_robin"
        all_ids: set[int] = set()
        for rank in range(8):
            ids = compute_local_expert_ids(384, 8, rank, placement=rr)
            assert ids is not None and len(ids) == 48
            assert all_ids.isdisjoint(ids)
            all_ids |= ids
        assert all_ids == set(range(384))

    def test_round_robin_uneven(self):
        # 10 experts, EP=3: rank 0→{0,3,6,9}, rank 1→{1,4,7}, rank 2→{2,5,8}
        rr = "round_robin"
        ids_0 = compute_local_expert_ids(10, 3, 0, placement=rr)
        ids_1 = compute_local_expert_ids(10, 3, 1, placement=rr)
        ids_2 = compute_local_expert_ids(10, 3, 2, placement=rr)
        assert ids_0 == {0, 3, 6, 9}
        assert ids_1 == {1, 4, 7}
        assert ids_2 == {2, 5, 8}
        assert ids_0 | ids_1 | ids_2 == set(range(10))


# ---------------------------------------------------------------------------
# Unit tests for compute_eplb_local_expert_ids
# ---------------------------------------------------------------------------


class TestComputeEplbLocalExpertIds:
    def test_ep_disabled(self):
        assert compute_eplb_local_expert_ids(8, 2, ep_size=1, ep_rank=0) is None

    def test_no_redundant_matches_plain_partition(self):
        # With zero redundant experts EPLB placement degenerates to the
        # plain linear EP partition.
        for ep_size in (2, 4, 8):
            for ep_rank in range(ep_size):
                assert compute_eplb_local_expert_ids(
                    8, 0, ep_size, ep_rank
                ) == compute_local_expert_ids(8, ep_size, ep_rank, placement="linear")

    def test_redundant_slots_pull_remote_logical_experts(self):
        # 8 logical + 2 redundant = 10 physical slots, EP=2.
        # Rank 1 owns physical slots 5-9; slots 8 and 9 map to logical 0, 1.
        ids = compute_eplb_local_expert_ids(8, 2, ep_size=2, ep_rank=1)
        assert ids == {0, 1, 5, 6, 7}

    def test_matches_model_side_placement(self):
        # The filter set must equal what the model requests for its local
        # physical slots: determine_expert_map (linear, as forced under EPLB)
        # mapped through EplbState's initial physical-to-logical map.
        from vllm.distributed.eplb.eplb_state import EplbState
        from vllm.model_executor.layers.fused_moe.expert_map_manager import (
            determine_expert_map,
        )

        for num_experts, num_redundant, ep_size in [
            (8, 2, 2),
            (256, 0, 16),
            (256, 32, 16),
            (384, 16, 8),
        ]:
            initial_map = EplbState.build_initial_global_physical_to_logical_map(
                num_experts, num_redundant
            )
            for ep_rank in range(ep_size):
                _, expert_map, _ = determine_expert_map(
                    ep_size, ep_rank, num_experts + num_redundant, "linear"
                )
                assert expert_map is not None
                expected = {
                    initial_map[g] for g in torch.where(expert_map != -1)[0].tolist()
                }
                assert (
                    compute_eplb_local_expert_ids(
                        num_experts, num_redundant, ep_size, ep_rank
                    )
                    == expected
                )

    def test_full_coverage_no_starved_experts(self):
        # The union over ranks must cover every logical expert, and no rank
        # loads more logical experts than its physical slot count.
        for num_experts, num_redundant, ep_size in [(8, 2, 2), (256, 32, 16)]:
            max_slots = -(-(num_experts + num_redundant) // ep_size)
            union: set[int] = set()
            for ep_rank in range(ep_size):
                ids = compute_eplb_local_expert_ids(
                    num_experts, num_redundant, ep_size, ep_rank
                )
                assert ids is not None
                assert len(ids) <= max_slots
                union |= ids
            assert union == set(range(num_experts))

    def test_model_side_forces_linear_placement(self):
        # The filter always splits physical slots linearly under EPLB, which
        # relies on the model refusing round-robin placement there as well.
        from vllm.model_executor.layers.fused_moe.expert_map_manager import (
            determine_expert_placement_strategy,
        )

        for num_redundant in (0, 32):
            strategy = determine_expert_placement_strategy(
                "round_robin",
                moe_parallel_config=None,
                num_expert_group=8,
                num_redundant_experts=num_redundant,
                enable_eplb=True,
            )
            assert strategy == "linear"


# ---------------------------------------------------------------------------
# Unit tests for DefaultModelLoader._init_ep_weight_filter
# ---------------------------------------------------------------------------


class _EplbMoEModel(torch.nn.Module, MixtureOfExperts):
    """MixtureOfExperts exposing only the expert layout the loader checks."""

    def __init__(self, num_logical_experts: int, num_physical_experts: int):
        super().__init__()
        self.num_moe_layers = 1
        self.num_logical_experts = num_logical_experts
        self.num_physical_experts = num_physical_experts


class TestInitEpWeightFilter:
    """Expert set picked by the loader for EP=2 over 8 logical experts."""

    @staticmethod
    def _local_expert_ids(
        monkeypatch: pytest.MonkeyPatch,
        model: torch.nn.Module,
        *,
        ep_rank: int = 1,
        enable_eplb: bool = True,
        num_redundant: int = 2,
        enable_elastic_ep: bool = False,
        placement: str = "linear",
    ) -> set[int] | None:
        parallel_config = SimpleNamespace(
            enable_expert_parallel=True,
            enable_ep_weight_filter=True,
            data_parallel_size=2,
            tensor_parallel_size=1,
            prefill_context_parallel_size=1,
            enable_eplb=enable_eplb,
            enable_elastic_ep=enable_elastic_ep,
            eplb_config=SimpleNamespace(num_redundant_experts=num_redundant),
            expert_placement_strategy=placement,
        )
        monkeypatch.setattr(
            "vllm.config.get_current_vllm_config",
            lambda: SimpleNamespace(parallel_config=parallel_config),
        )
        monkeypatch.setattr(
            "vllm.distributed.get_dp_group",
            lambda: SimpleNamespace(rank_in_group=ep_rank),
        )
        model_config = SimpleNamespace(is_moe=True, get_num_experts=lambda: 8)
        loader = DefaultModelLoader(LoadConfig())
        loader._init_ep_weight_filter(model, model_config)
        return loader.local_expert_ids

    def test_eplb_loads_initial_placement(self, monkeypatch):
        # Rank 1 owns physical slots 5-9; slots 8 and 9 replicate logical 0, 1.
        model = _EplbMoEModel(8, 10)
        assert self._local_expert_ids(monkeypatch, model) == {0, 1, 5, 6, 7}

    def test_eplb_ignores_round_robin_request(self, monkeypatch):
        model = _EplbMoEModel(8, 10)
        ids = self._local_expert_ids(monkeypatch, model, placement="round_robin")
        assert ids == {0, 1, 5, 6, 7}

    def test_eplb_elastic_ep_loads_everything(self, monkeypatch):
        model = _EplbMoEModel(8, 10)
        ids = self._local_expert_ids(monkeypatch, model, enable_elastic_ep=True)
        assert ids is None

    @pytest.mark.parametrize(
        "model",
        [
            pytest.param(torch.nn.Module(), id="not-moe"),
            pytest.param(_EplbMoEModel(8, 8), id="no-redundant-slots"),
            pytest.param(_EplbMoEModel(10, 12), id="other-logical-count"),
        ],
    )
    def test_eplb_unexpected_layout_loads_everything(self, monkeypatch, model):
        # The set is derived from the config; a model whose slots follow a
        # different layout must not have any of its expert weights skipped.
        assert self._local_expert_ids(monkeypatch, model) is None

    def test_without_eplb_uses_plain_partition(self, monkeypatch):
        ids = self._local_expert_ids(
            monkeypatch, torch.nn.Module(), enable_eplb=False, num_redundant=0
        )
        assert ids == {4, 5, 6, 7}


# ---------------------------------------------------------------------------
# Unit tests for should_skip_weight
# ---------------------------------------------------------------------------


class TestShouldSkipWeight:
    def setup_method(self):
        # Simulate EP=8, rank=0 → experts 0-47
        self.local_ids = compute_local_expert_ids(384, ep_size=8, ep_rank=0)

    def test_no_filter(self):
        assert not should_skip_weight("anything", None)

    def test_dense_not_skipped(self):
        assert not should_skip_weight(
            "model.layers.0.self_attn.q_proj.weight", self.local_ids
        )

    def test_local_expert_not_skipped(self):
        assert not should_skip_weight(
            "model.layers.0.mlp.experts.10.gate_proj.weight", self.local_ids
        )

    def test_remote_expert_skipped(self):
        assert should_skip_weight(
            "model.layers.0.mlp.experts.200.gate_proj.weight", self.local_ids
        )

    def test_local_packed_expert_not_skipped(self):
        assert not should_skip_weight(
            "model.layers.0.mlp.experts.10.gate_proj.weight_packed", self.local_ids
        )

    def test_remote_packed_expert_skipped(self):
        assert should_skip_weight(
            "model.layers.0.mlp.experts.200.gate_proj.weight_packed", self.local_ids
        )

    def test_remote_expert_scale_not_skipped(self):
        assert not should_skip_weight(
            "model.layers.0.mlp.experts.200.gate_proj.weight_scale", self.local_ids
        )

    def test_boundary_expert(self):
        # Expert 47 is local (last one), 48 is not
        assert not should_skip_weight(
            "model.layers.0.mlp.experts.47.gate_proj.weight", self.local_ids
        )
        assert should_skip_weight(
            "model.layers.0.mlp.experts.48.gate_proj.weight", self.local_ids
        )

    def test_shared_expert_not_skipped(self):
        assert not should_skip_weight(
            "model.layers.0.mlp.shared_experts.gate_proj.weight", self.local_ids
        )

    def test_embedding_not_skipped(self):
        assert not should_skip_weight("model.embed_tokens.weight", self.local_ids)

    def test_fused_3d_expert_not_skipped(self):
        # 3D fused-expert tensors (gpt-oss style) have no numeric id.
        # Must not be skipped — weight_loader handles slicing later.
        assert not should_skip_weight(
            "model.layers.0.mlp.experts.gate_proj.weight", self.local_ids
        )


# ---------------------------------------------------------------------------
# Integration test: safetensors_weights_iterator with EP filtering
# ---------------------------------------------------------------------------


class TestSafetensorsWeightsIteratorWithEpFilter:
    """Verify that EP filtering produces a strict subset of unfiltered loading
    and that all expected dense + local expert weights are present."""

    @pytest.fixture(scope="class")
    def gpt2_files(self):
        """Download GPT-2 safetensors to a temp dir (shared across class)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            huggingface_hub.constants.HF_HUB_OFFLINE = False
            from vllm.model_executor.model_loader.weight_utils import (
                download_weights_from_hf,
            )

            download_weights_from_hf(
                "openai-community/gpt2",
                allow_patterns=["*.safetensors"],
                cache_dir=tmpdir,
            )
            files = glob.glob(f"{tmpdir}/**/*.safetensors", recursive=True)
            assert len(files) > 0
            yield files

    def test_no_filter_returns_all(self, gpt2_files):
        """With local_expert_ids=None, all weights are returned (no MoE)."""
        all_weights = dict(safetensors_weights_iterator(gpt2_files, False))
        filtered_weights = dict(
            safetensors_weights_iterator(gpt2_files, False, local_expert_ids=None)
        )
        assert set(all_weights.keys()) == set(filtered_weights.keys())

    def test_empty_filter_skips_experts_only(self, gpt2_files):
        """GPT-2 has no expert weights, so even an empty local_expert_ids
        set should return all weights (all are dense)."""
        all_weights = dict(safetensors_weights_iterator(gpt2_files, False))
        filtered_weights = dict(
            safetensors_weights_iterator(gpt2_files, False, local_expert_ids=set())
        )
        # GPT-2 has no experts, so nothing should be filtered
        assert set(all_weights.keys()) == set(filtered_weights.keys())


class TestEpFilterOnSyntheticMoeWeights:
    """Create synthetic safetensors files with expert-like naming and verify
    that the filter correctly skips non-local experts."""

    @pytest.fixture
    def synthetic_moe_files(self, tmp_path):
        """Create synthetic safetensors with expert-patterned tensor names."""
        from safetensors.torch import save_file

        tensors = {}
        # Dense weights
        tensors["model.embed_tokens.weight"] = torch.randn(100, 64)
        tensors["model.layers.0.self_attn.q_proj.weight"] = torch.randn(64, 64)
        tensors["model.layers.0.input_layernorm.weight"] = torch.randn(64)
        # Expert weights: 8 experts
        for expert_id in range(8):
            tensors[f"model.layers.0.mlp.experts.{expert_id}.gate_proj.weight"] = (
                torch.randn(128, 64)
            )
            tensors[f"model.layers.0.mlp.experts.{expert_id}.up_proj.weight"] = (
                torch.randn(128, 64)
            )
            tensors[f"model.layers.0.mlp.experts.{expert_id}.down_proj.weight"] = (
                torch.randn(64, 128)
            )
        # Shared expert (should never be filtered)
        tensors["model.layers.0.mlp.shared_experts.gate_proj.weight"] = torch.randn(
            128, 64
        )

        filepath = str(tmp_path / "model-00001-of-00001.safetensors")
        save_file(tensors, filepath)
        return [filepath], tensors

    def test_no_filter_returns_all(self, synthetic_moe_files):
        files, expected = synthetic_moe_files
        loaded = dict(safetensors_weights_iterator(files, False))
        assert set(loaded.keys()) == set(expected.keys())

    def test_ep2_rank0_gets_half_experts(self, synthetic_moe_files):
        files, expected = synthetic_moe_files
        # EP=2, rank=0 → experts 0-3
        local_ids = compute_local_expert_ids(8, ep_size=2, ep_rank=0)
        assert local_ids is not None
        loaded = dict(
            safetensors_weights_iterator(files, False, local_expert_ids=local_ids)
        )

        # Should have all dense + shared + experts 0-3 only
        for name in loaded:
            eid = parse_expert_id(name)
            if eid is not None:
                assert eid in local_ids, f"Non-local expert {eid} was loaded"

        # Check expert count: 4 experts × 3 weights = 12
        expert_names = [n for n in loaded if parse_expert_id(n) is not None]
        assert len(expert_names) == 4 * 3

        # Check all dense weights present
        assert "model.embed_tokens.weight" in loaded
        assert "model.layers.0.self_attn.q_proj.weight" in loaded
        assert "model.layers.0.input_layernorm.weight" in loaded
        assert "model.layers.0.mlp.shared_experts.gate_proj.weight" in loaded

    def test_ep2_rank1_gets_other_half(self, synthetic_moe_files):
        files, expected = synthetic_moe_files
        local_ids = compute_local_expert_ids(8, ep_size=2, ep_rank=1)
        assert local_ids is not None
        loaded = dict(
            safetensors_weights_iterator(files, False, local_expert_ids=local_ids)
        )

        expert_names = [n for n in loaded if parse_expert_id(n) is not None]
        assert len(expert_names) == 4 * 3
        for name in expert_names:
            assert parse_expert_id(name) in local_ids

    def test_ep8_each_rank_gets_one_expert(self, synthetic_moe_files):
        files, _ = synthetic_moe_files
        all_expert_names = set()
        for rank in range(8):
            local_ids = compute_local_expert_ids(8, ep_size=8, ep_rank=rank)
            loaded = dict(
                safetensors_weights_iterator(files, False, local_expert_ids=local_ids)
            )
            expert_names = {n for n in loaded if parse_expert_id(n) is not None}
            # 1 expert × 3 weights
            assert len(expert_names) == 3
            all_expert_names |= expert_names

        # All 8 experts × 3 weights covered across ranks
        assert len(all_expert_names) == 24

    def test_tensor_values_match(self, synthetic_moe_files):
        """Filtered tensors have identical values to unfiltered ones."""
        files, _ = synthetic_moe_files
        all_weights = dict(safetensors_weights_iterator(files, False))

        local_ids = compute_local_expert_ids(8, ep_size=2, ep_rank=0)
        filtered = dict(
            safetensors_weights_iterator(files, False, local_expert_ids=local_ids)
        )

        for name, tensor in filtered.items():
            assert torch.equal(tensor, all_weights[name]), f"Tensor mismatch for {name}"

    def test_eplb_redundant_slot_weights_not_skipped(self, synthetic_moe_files):
        """Regression test for EPLB: a rank whose redundant physical slots map
        to *remote* logical experts must still read those experts from disk."""
        files, _ = synthetic_moe_files
        # 8 logical + 2 redundant = 10 physical slots, EP=2.
        # Rank 1 owns physical 5-9, i.e. logical {5,6,7} plus redundant {0,1}.
        local_ids = compute_eplb_local_expert_ids(8, 2, ep_size=2, ep_rank=1)
        loaded = dict(
            safetensors_weights_iterator(files, False, local_expert_ids=local_ids)
        )

        for eid in (0, 1, 5, 6, 7):
            for proj in ("gate_proj", "up_proj", "down_proj"):
                name = f"model.layers.0.mlp.experts.{eid}.{proj}.weight"
                assert name in loaded, f"Needed expert weight {name} was skipped"

        # Experts 2, 3, 4 are on rank 0's plain partition and stay filtered.
        expert_names = {n for n in loaded if parse_expert_id(n) is not None}
        assert len(expert_names) == 5 * 3
