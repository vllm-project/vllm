# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for EP weight filtering during model loading."""

import glob
import tempfile

import huggingface_hub.constants
import pytest
import torch

from vllm.model_executor.model_loader.ep_weight_filter import (
    compute_local_expert_ids,
    parse_expert_id,
    should_skip_weight,
)
from vllm.model_executor.model_loader.weight_utils import (
    safetensors_weights_iterator,
)


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
        all_ids = set()
        for rank in range(8):
            ids = compute_local_expert_ids(384, ep_size=8, ep_rank=rank)
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
        assert not should_skip_weight(
            "model.embed_tokens.weight", self.local_ids
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
        all_weights = dict(
            safetensors_weights_iterator(gpt2_files, False)
        )
        filtered_weights = dict(
            safetensors_weights_iterator(
                gpt2_files, False, local_expert_ids=None
            )
        )
        assert set(all_weights.keys()) == set(filtered_weights.keys())

    def test_empty_filter_skips_experts_only(self, gpt2_files):
        """GPT-2 has no expert weights, so even an empty local_expert_ids
        set should return all weights (all are dense)."""
        all_weights = dict(
            safetensors_weights_iterator(gpt2_files, False)
        )
        filtered_weights = dict(
            safetensors_weights_iterator(
                gpt2_files, False, local_expert_ids=set()
            )
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
        tensors["model.layers.0.mlp.shared_experts.gate_proj.weight"] = (
            torch.randn(128, 64)
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
        loaded = dict(
            safetensors_weights_iterator(
                files, False, local_expert_ids=local_ids
            )
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
        loaded = dict(
            safetensors_weights_iterator(
                files, False, local_expert_ids=local_ids
            )
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
                safetensors_weights_iterator(
                    files, False, local_expert_ids=local_ids
                )
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
            safetensors_weights_iterator(
                files, False, local_expert_ids=local_ids
            )
        )

        for name, tensor in filtered.items():
            assert torch.equal(tensor, all_weights[name]), (
                f"Tensor mismatch for {name}"
            )
