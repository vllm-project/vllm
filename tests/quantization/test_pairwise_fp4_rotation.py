# SPDX-License-Identifier: Apache-2.0
"""Unit tests for pairwise_fp4 algorithm core modules.

Covers:
  - rotation_applier: forward/inverse, identity, orthogonality, edge cases
  - channel_monitor: risk score computation (max_abs, dynamic_range), save/load
  - pair_constructor: all 5 policies, edge cases
  - angle_solver: heuristic, small_search
  - fp4_quant_policy: estimate_global_scale, quantize_weight_to_fp4, packing
  - utils: RotationPlan dataclass, serialization, save/load plan
  - rotation_plan: RotationPlanBuilder pipeline, caching

Run:
    pytest tests/quantization/test_pairwise_fp4_rotation.py -v
"""

from __future__ import annotations

import json
import math
import os
import tempfile

import pytest
import torch

# =====================================================================
# rotation_applier
# =====================================================================
from vllm.model_executor.layers.quantization.pairwise_fp4.rotation_applier import (
    apply_givens_rotation,
)


class TestGivensRotation:
    """Tests for apply_givens_rotation."""

    def _random_pairs_angles(self, C: int, n_pairs: int):
        """Generate random non-overlapping pairs and random angles."""
        indices = torch.randperm(C)[: n_pairs * 2]
        pairs = indices.reshape(n_pairs, 2).to(torch.int64)
        angles = torch.randn(n_pairs, dtype=torch.float32)
        return pairs, angles

    def test_forward_inverse_roundtrip(self):
        """R^{-1}(R(x)) ≈ x to float32 precision."""
        torch.manual_seed(0)
        x = torch.randn(4, 64, dtype=torch.float32)
        pairs, angles = self._random_pairs_angles(64, 8)

        y = apply_givens_rotation(x, pairs, angles)
        x_rec = apply_givens_rotation(y, pairs, angles, inverse=True)
        torch.testing.assert_close(x_rec, x, atol=1e-5, rtol=1e-5)

    def test_identity_angle_zero(self):
        """Angle = 0 implies no change."""
        torch.manual_seed(1)
        x = torch.randn(3, 32, dtype=torch.float32)
        pairs = torch.tensor([[0, 1], [4, 5]], dtype=torch.int64)
        angles = torch.zeros(2, dtype=torch.float32)

        y = apply_givens_rotation(x, pairs, angles)
        torch.testing.assert_close(y, x)

    def test_orthogonality_norm_preservation(self):
        """||R(x)|| == ||x|| for every row."""
        torch.manual_seed(2)
        x = torch.randn(8, 128, dtype=torch.float32)
        pairs, angles = self._random_pairs_angles(128, 16)

        y = apply_givens_rotation(x, pairs, angles)
        norms_x = x.norm(dim=-1)
        norms_y = y.norm(dim=-1)
        torch.testing.assert_close(norms_y, norms_x, atol=1e-5, rtol=1e-5)

    def test_empty_pairs_noop(self):
        """Empty pairs = identity."""
        x = torch.randn(2, 16, dtype=torch.float32)
        pairs = torch.empty(0, 2, dtype=torch.int64)
        angles = torch.empty(0, dtype=torch.float32)

        y = apply_givens_rotation(x, pairs, angles)
        torch.testing.assert_close(y, x)

    def test_single_pair_known_value(self):
        """Manual check for a single 45-degree rotation on a 2-D vector."""
        x = torch.tensor([[1.0, 0.0]])  # (1, 2)
        pairs = torch.tensor([[0, 1]], dtype=torch.int64)
        angles = torch.tensor([math.pi / 4], dtype=torch.float32)

        y = apply_givens_rotation(x, pairs, angles)
        expected = torch.tensor([[math.cos(math.pi / 4), -math.sin(math.pi / 4)]])
        torch.testing.assert_close(y, expected, atol=1e-6, rtol=1e-6)

    def test_batched_input(self):
        """Works with 3-D tensors (batch, seq, hidden)."""
        torch.manual_seed(3)
        x = torch.randn(2, 3, 32, dtype=torch.float32)
        pairs, angles = self._random_pairs_angles(32, 4)

        y = apply_givens_rotation(x, pairs, angles)
        assert y.shape == x.shape
        # Norm preservation per-vector
        norms_x = x.norm(dim=-1)
        norms_y = y.norm(dim=-1)
        torch.testing.assert_close(norms_y, norms_x, atol=1e-5, rtol=1e-5)

    def test_validation_errors(self):
        """Bad input shapes raise ValueError."""
        x = torch.randn(2, 8)
        with pytest.raises(ValueError, match="pairs must have shape"):
            apply_givens_rotation(x, torch.zeros(3), torch.zeros(3))
        with pytest.raises(ValueError, match="angles must have shape"):
            apply_givens_rotation(
                x,
                torch.zeros(2, 2, dtype=torch.int64),
                torch.zeros(2, 2),
            )
        with pytest.raises(ValueError, match="same length"):
            apply_givens_rotation(
                x,
                torch.zeros(2, 2, dtype=torch.int64),
                torch.zeros(3),
            )


# =====================================================================
# channel_monitor
# =====================================================================
from vllm.model_executor.layers.quantization.pairwise_fp4.channel_monitor import (
    compute_risk_scores,
    load_risk_scores,
    save_risk_scores,
)


class TestChannelMonitor:
    def test_max_abs_known_values(self):
        """Hand-crafted tensor, max_abs per channel."""
        # 2 rows, 4 channels
        t = torch.tensor([[1.0, -3.0, 0.5, 2.0], [2.0, 1.0, -0.5, -4.0]])
        scores = compute_risk_scores(t, method="max_abs", channel_dim=-1)
        # max_abs per column: [2, 3, 0.5, 4]
        expected = torch.tensor([2.0, 3.0, 0.5, 4.0])
        torch.testing.assert_close(scores, expected)

    def test_dynamic_range_known_values(self):
        """Dynamic range = max_abs / min_nonzero_abs."""
        t = torch.tensor([[1.0, 0.0, 2.0], [3.0, 0.0, 0.5]])
        scores = compute_risk_scores(t, method="dynamic_range", channel_dim=-1)
        # col 0: max=3, min_nz=1 → 3/(1+eps) ≈ 3
        # col 1: all zero → max=0, min stays eps → ~0
        # col 2: max=2, min_nz=0.5 → 2/(0.5+eps) ≈ 4
        assert scores.shape == (3,)
        assert scores[0] > 2.9
        assert scores[1] < 0.01
        assert scores[2] > 3.9

    def test_risk_scores_shape(self):
        """Shape equals the channel dimension."""
        t = torch.randn(8, 64)
        scores = compute_risk_scores(t, method="max_abs")
        assert scores.shape == (64,)

    def test_save_load_roundtrip(self, tmp_path):
        """Saved risk scores round-trip to identical values."""
        scores = torch.randn(32, dtype=torch.float32)
        path = str(tmp_path / "risk.json")
        meta = {
            "layer_index": "layer.0",
            "target": "weight",
            "method": "max_abs",
            "shard_id": 0,
            "num_channels": 32,
        }
        save_risk_scores(scores, path, meta)
        loaded_scores, loaded_meta = load_risk_scores(path)
        torch.testing.assert_close(loaded_scores, scores, atol=1e-6, rtol=1e-6)
        assert loaded_meta["layer_index"] == "layer.0"

    def test_save_missing_metadata_raises(self, tmp_path):
        with pytest.raises(ValueError, match="Missing required metadata"):
            save_risk_scores(
                torch.zeros(4),
                str(tmp_path / "bad.json"),
                {"layer_index": "x"},
            )


# =====================================================================
# pair_constructor
# =====================================================================
from vllm.model_executor.layers.quantization.pairwise_fp4.pair_constructor import (
    construct_pairs,
)


class TestPairConstructor:
    def _scores(self, C: int):
        torch.manual_seed(42)
        return torch.rand(C, dtype=torch.float32)

    @pytest.mark.parametrize(
        "policy",
        ["adjacent_sorted", "high_high", "high_low", "random_baseline"],
    )
    def test_policy_basic_properties(self, policy):
        """Each policy returns valid (N, 2) int64 pairs, all unique."""
        scores = self._scores(128)
        pairs = construct_pairs(scores, policy=policy, top_ratio=0.1)
        assert pairs.ndim == 2 and pairs.shape[1] == 2
        assert pairs.dtype == torch.int64
        flat = pairs.flatten()
        assert flat.unique().shape[0] == flat.shape[0], "duplicate indices"

    def test_joint_compatible_requires_secondary(self):
        with pytest.raises(ValueError, match="secondary_risk_scores is required"):
            construct_pairs(
                torch.ones(32),
                policy="joint_compatible",
                secondary_risk_scores=None,
            )

    def test_joint_compatible(self):
        scores = self._scores(128)
        sec = self._scores(128)
        pairs = construct_pairs(
            scores, policy="joint_compatible", top_ratio=0.1,
            secondary_risk_scores=sec,
        )
        assert pairs.ndim == 2 and pairs.shape[1] == 2

    def test_top_ratio_zero_returns_empty(self):
        pairs = construct_pairs(torch.ones(64), top_ratio=0.0)
        assert pairs.shape == (0, 2)

    def test_top_ratio_small_channel_count(self):
        """Very few channels with tiny top_ratio can still return empty."""
        pairs = construct_pairs(torch.ones(4), top_ratio=0.01)
        assert pairs.shape[0] == 0

    def test_pair_count_scales_with_top_ratio(self):
        scores = self._scores(256)
        p1 = construct_pairs(scores, top_ratio=0.05)
        p2 = construct_pairs(scores, top_ratio=0.2)
        assert p2.shape[0] >= p1.shape[0]


# =====================================================================
# angle_solver
# =====================================================================
from vllm.model_executor.layers.quantization.pairwise_fp4.angle_solver import (
    solve_angles,
)


class TestAngleSolver:
    def test_heuristic_output_shape(self):
        pairs = torch.tensor([[0, 1], [2, 3]], dtype=torch.int64)
        risk = torch.tensor([1.0, 2.0, 0.5, 3.0])
        angles = solve_angles(pairs, solver="heuristic", risk_scores=risk)
        assert angles.shape == (2,)
        assert angles.dtype == torch.float32

    def test_heuristic_symmetric_risk_gives_zero(self):
        """Equal risks → angle ≈ 0."""
        pairs = torch.tensor([[0, 1]], dtype=torch.int64)
        risk = torch.tensor([5.0, 5.0])
        angles = solve_angles(pairs, solver="heuristic", risk_scores=risk)
        assert angles.abs().item() < 1e-6

    def test_heuristic_bounded(self):
        """Heuristic angles should be in [0, π/4]."""
        torch.manual_seed(7)
        pairs = torch.randint(0, 64, (10, 2), dtype=torch.int64)
        risk = torch.rand(64)
        angles = solve_angles(pairs, solver="heuristic", risk_scores=risk)
        assert (angles >= -0.01).all()
        assert (angles <= math.pi / 4 + 0.01).all()

    def test_small_search_output_shape(self):
        w = torch.randn(4, 8)
        pairs = torch.tensor([[0, 1], [2, 3]], dtype=torch.int64)
        angles = solve_angles(pairs, solver="small_search", weight=w)
        assert angles.shape == (2,)

    def test_small_search_bounded(self):
        w = torch.randn(8, 16)
        pairs = torch.tensor([[0, 1], [4, 5], [8, 9]], dtype=torch.int64)
        angles = solve_angles(pairs, solver="small_search", weight=w)
        assert (angles >= -math.pi / 4 - 0.01).all()
        assert (angles <= math.pi / 4 + 0.01).all()

    def test_empty_pairs(self):
        angles = solve_angles(
            torch.empty(0, 2, dtype=torch.int64),
            solver="heuristic",
            risk_scores=torch.zeros(4),
        )
        assert angles.shape == (0,)


# =====================================================================
# fp4_quant_policy
# =====================================================================
from vllm.model_executor.layers.quantization.pairwise_fp4.fp4_quant_policy import (
    estimate_global_scale,
    pack_fp4_to_uint8,
    quantize_weight_to_fp4,
)


class TestFP4QuantPolicy:
    def test_estimate_global_scale_positive(self):
        w = torch.randn(16, 32)
        gs = estimate_global_scale(w)
        assert gs.item() > 0

    def test_estimate_global_scale_zero_weight(self):
        w = torch.zeros(8, 32)
        gs = estimate_global_scale(w)
        assert gs.item() == 1.0

    def test_quantize_shape(self):
        w = torch.randn(64, 128)
        gs = estimate_global_scale(w)
        packed, scales, gs_out = quantize_weight_to_fp4(w, gs)
        assert packed.shape == (64, 64)  # 128/2
        assert packed.dtype == torch.uint8
        assert scales.shape == (64, 8)  # 128/16
        assert scales.dtype == torch.float8_e4m3fn

    def test_pack_fp4_roundtrip_values(self):
        """Known FP4 values pack to expected bytes."""
        # Two FP4 values: 1.0 and -1.5
        # 1.0 → nibble 0b0010 = 2, -1.5 → 0b1011 = 11
        vals = torch.tensor([[1.0, -1.5]])
        packed = pack_fp4_to_uint8(vals)
        byte_val = packed[0, 0].item()
        expected = 2 | (11 << 4)  # 0xB2 = 178
        assert byte_val == expected

    def test_quantize_invalid_dims(self):
        with pytest.raises(ValueError, match="must be 2-D"):
            quantize_weight_to_fp4(
                torch.randn(4, 8, 16),
                torch.tensor(1.0),
            )

    def test_quantize_invalid_divisibility(self):
        with pytest.raises(ValueError, match="divisible by"):
            quantize_weight_to_fp4(
                torch.randn(8, 17),
                torch.tensor(1.0),
                block_size=16,
            )


# =====================================================================
# utils: RotationPlan dataclass
# =====================================================================
from vllm.model_executor.layers.quantization.pairwise_fp4.utils import (
    RotationPlan,
    empty_angles,
    empty_pairs,
    load_plan,
    save_plan,
)


class TestRotationPlan:
    def _make_plan(self):
        return RotationPlan(
            mode="weight_only",
            layer_index="layer.0.qkv",
            pairs=torch.tensor([[0, 1], [2, 3]], dtype=torch.int64),
            angles=torch.tensor([0.1, 0.2], dtype=torch.float32),
            pair_meta={"policy": "high_high"},
            angle_meta={"solver": "heuristic"},
        )

    def test_validate_ok(self):
        self._make_plan().validate()

    def test_validate_bad_mode(self):
        plan = self._make_plan()
        plan.mode = "bad"
        with pytest.raises(ValueError, match="Invalid mode"):
            plan.validate()

    def test_validate_shape_mismatch(self):
        plan = self._make_plan()
        plan.angles = torch.tensor([0.1], dtype=torch.float32)
        with pytest.raises(ValueError, match="length mismatch"):
            plan.validate()

    def test_serialization_roundtrip(self):
        plan = self._make_plan()
        d = plan.to_dict()
        plan2 = RotationPlan.from_dict(d)
        assert plan2.mode == plan.mode
        assert plan2.layer_index == plan.layer_index
        torch.testing.assert_close(plan2.pairs, plan.pairs)
        torch.testing.assert_close(plan2.angles, plan.angles)

    def test_save_load_roundtrip(self, tmp_path):
        plan = self._make_plan()
        path = str(tmp_path / "plan.json")
        save_plan(plan, path)
        plan2 = load_plan(path)
        torch.testing.assert_close(plan2.pairs, plan.pairs)
        torch.testing.assert_close(plan2.angles, plan.angles)

    def test_empty_plan(self):
        plan = RotationPlan(
            mode="weight_only",
            layer_index="x",
            pairs=empty_pairs(),
            angles=empty_angles(),
        )
        assert plan.is_empty
        assert plan.num_pairs == 0
        plan.validate()

    def test_empty_plan_serialization_roundtrip(self, tmp_path):
        plan = RotationPlan(
            mode="weight_only",
            layer_index="x",
            pairs=empty_pairs(),
            angles=empty_angles(),
        )
        path = str(tmp_path / "empty.json")
        save_plan(plan, path)
        plan2 = load_plan(path)
        assert plan2.is_empty


# =====================================================================
# rotation_plan: RotationPlanBuilder
# =====================================================================
from vllm.model_executor.layers.quantization.pairwise_fp4.rotation_plan import (
    RotationPlanBuilder,
)


class TestRotationPlanBuilder:
    def test_weight_only_build(self):
        torch.manual_seed(10)
        w = torch.randn(64, 128)
        builder = RotationPlanBuilder({"top_ratio": 0.1})
        plan = builder.build("layer.0", mode="weight_only", weight=w)
        assert plan.mode == "weight_only"
        plan.validate()
        assert plan.num_pairs > 0

    def test_weight_only_missing_weight_raises(self):
        builder = RotationPlanBuilder()
        with pytest.raises(ValueError, match="requires a weight tensor"):
            builder.build("layer.0", mode="weight_only")

    def test_activation_only_missing_activation_raises(self):
        builder = RotationPlanBuilder()
        with pytest.raises(ValueError, match="requires an activation tensor"):
            builder.build("layer.0", mode="activation_only")

    def test_joint_build(self):
        torch.manual_seed(11)
        w = torch.randn(64, 128)
        a = torch.randn(16, 128)
        builder = RotationPlanBuilder({"top_ratio": 0.1})
        plan = builder.build("layer.0", mode="joint", weight=w, activation=a)
        assert plan.mode == "joint"
        plan.validate()

    def test_cache_creation_and_reuse(self, tmp_path):
        """Builder creates risk-score cache on first call, reuses on second."""
        torch.manual_seed(12)
        w = torch.randn(32, 64)
        cache_dir = str(tmp_path / "cache")
        builder = RotationPlanBuilder({
            "top_ratio": 0.1,
            "risk_cache_dir": cache_dir,
        })
        plan1 = builder.build("lay.0", mode="weight_only", weight=w)

        # Cache file should exist
        cache_files = os.listdir(cache_dir)
        assert len(cache_files) > 0

        # Build again — should use cache (no error even if weight differs)
        w2 = torch.randn(32, 64)
        plan2 = builder.build("lay.0", mode="weight_only", weight=w2)
        # Plans from same cache → same pairs (since risk scores are cached)
        torch.testing.assert_close(plan1.pairs, plan2.pairs)

    def test_small_search_solver(self):
        torch.manual_seed(13)
        w = torch.randn(32, 64)
        builder = RotationPlanBuilder({
            "top_ratio": 0.1,
            "angle_solver": "small_search",
        })
        plan = builder.build("layer.0", mode="weight_only", weight=w)
        plan.validate()
        assert plan.angles.dtype == torch.float32


# =====================================================================
# config
# =====================================================================
from vllm.model_executor.layers.quantization.pairwise_fp4.config import (
    PairwiseFP4Config,
)


class TestPairwiseFP4Config:
    def test_defaults(self):
        cfg = PairwiseFP4Config()
        assert cfg.get_name() == "pairwise_fp4"
        assert cfg.mode == "weight_only"
        assert cfg.group_size == 16

    def test_invalid_mode(self):
        with pytest.raises(ValueError, match="Invalid mode"):
            PairwiseFP4Config(mode="bad")

    def test_from_config(self):
        d = {"mode": "joint", "top_ratio": 0.2}
        cfg = PairwiseFP4Config.from_config(d)
        assert cfg.mode == "joint"
        assert cfg.top_ratio == 0.2

    def test_plan_builder_config(self):
        cfg = PairwiseFP4Config(risk_method="dynamic_range", top_ratio=0.3)
        pbc = cfg.get_plan_builder_config()
        assert pbc["risk_method"] == "dynamic_range"
        assert pbc["top_ratio"] == 0.3

    def test_get_config_filenames(self):
        assert "pairwise_fp4_config.json" in PairwiseFP4Config.get_config_filenames()

    def test_supported_dtypes(self):
        cfg = PairwiseFP4Config()
        dtypes = cfg.get_supported_act_dtypes()
        assert torch.bfloat16 in dtypes
        assert torch.half in dtypes
