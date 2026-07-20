# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for MiniCPM-Robot pooling model (VLM + DiT ActionHead)."""

import pytest
import torch

from vllm.model_executor.layers.pooler.dit_action import DiTActionPooler
from vllm.model_executor.models.minicpm_robot import MiniCPMRobotForHiddenStates
from vllm.model_executor.models.minicpm_robot_action_head import GR00tActionHead

pytestmark = pytest.mark.cpu_test


# ---------------------------------------------------------------------------
# Architecture / registration checks
# ---------------------------------------------------------------------------


def test_model_is_pooling() -> None:
    """MiniCPMRobotForHiddenStates must be registered as a pooling model."""
    assert MiniCPMRobotForHiddenStates.is_pooling_model is True


def test_model_inherits_vllm_pooling() -> None:
    """Verify MRO includes the vLLM pooling interface."""
    from vllm.model_executor.models.interfaces_base import VllmModelForPooling

    assert VllmModelForPooling in MiniCPMRobotForHiddenStates.__mro__


# ---------------------------------------------------------------------------
# Pooler / ActionHead unit checks
# ---------------------------------------------------------------------------


class TestDiTActionPooler:
    """Check pooler config propagation and basic structure."""

    def test_default_ctor_does_not_crash(self) -> None:
        """DiTActionPooler should initialize without errors using defaults."""
        pooler = DiTActionPooler()
        assert isinstance(pooler.action_head, GR00tActionHead)
        assert pooler.action_head.action_dim == 80
        assert pooler.action_head.state_dim == 80
        assert pooler.action_head.action_horizon == 30

    def test_custom_dims(self) -> None:
        pooler = DiTActionPooler(
            action_dim=10,
            state_dim=20,
            action_horizon=5,
        )
        assert pooler.action_head.action_dim == 10
        assert pooler.action_head.state_dim == 20
        assert pooler.action_head.action_horizon == 5

    def test_supported_tasks(self) -> None:
        pooler = DiTActionPooler()
        tasks = pooler.get_supported_tasks()
        assert "embed" in tasks
        assert "token_embed" in tasks

    def test_forward_with_robot_state_produces_nonzero_actions(self) -> None:
        """Full pooler path: VLM hidden_states + robot_state → actions.

        Exercises the real DiT inference branch (list->tensor→reshape→
        predict_action), not just the warmup/zero branch.
        """
        import numpy as np

        from vllm.pooling_params import PoolingParams
        from vllm.v1.pool.metadata import PoolingMetadata, PoolingStates

        pooler = DiTActionPooler(
            action_dim=80,
            state_dim=80,
            action_horizon=30,
            num_inference_timesteps=4,
        )

        # Simulate VLM hidden_states: 32 vision tokens, 1024-dim
        hidden = torch.randn(32, 1024)

        pooler_np = PoolingParams(
            task="token_embed",
            extra_kwargs={"robot_state": np.random.randn(80).astype(np.float32)},
        )
        metadata = PoolingMetadata(
            prompt_lens=torch.tensor([32]),
            prompt_token_ids=None,
            prompt_token_ids_cpu=None,
            pooling_params=[pooler_np],
            pooling_states=[PoolingStates()],
        )
        out_np = pooler.forward(hidden, metadata)
        assert len(out_np) == 1
        assert out_np[0].shape == (30, 80)
        assert not torch.allclose(out_np[0], torch.zeros(30, 80), atol=1e-6), (
            "pooler should produce non-zero actions when robot_state is given"
        )

        # Also test the list→tensor path
        pooler_list = PoolingParams(
            task="token_embed",
            extra_kwargs={"robot_state": [float(i) for i in range(80)]},
        )
        metadata2 = PoolingMetadata(
            prompt_lens=torch.tensor([32]),
            prompt_token_ids=None,
            prompt_token_ids_cpu=None,
            pooling_params=[pooler_list],
            pooling_states=[PoolingStates()],
        )
        out_list = pooler.forward(hidden, metadata2)
        assert len(out_list) == 1
        assert out_list[0].shape == (30, 80)

    def test_forward_without_robot_state_returns_zeros(self) -> None:
        """When robot_state is None, pooler returns a zero actions tensor.

        This is the warmup / dummy-request path.
        """
        from vllm.pooling_params import PoolingParams
        from vllm.v1.pool.metadata import PoolingMetadata, PoolingStates

        pooler = DiTActionPooler()
        hidden = torch.randn(10, 1024)  # 10 tokens, 1024-dim
        pooling_params = PoolingParams(task="token_embed")
        prompt_lens = torch.tensor([10])
        metadata = PoolingMetadata(
            prompt_lens=prompt_lens,
            prompt_token_ids=None,
            prompt_token_ids_cpu=None,
            pooling_params=[pooling_params],
            pooling_states=[PoolingStates()],
        )

        result = pooler.forward(hidden, metadata)
        assert len(result) == 1
        assert result[0].shape == (30, 80)
        assert torch.all(result[0] == 0)

    def test_pooler_load_weights_filters_by_shape(self) -> None:
        """load_weights only accepts keys whose shapes match the model."""
        pooler = DiTActionPooler(action_dim=10, state_dim=20, action_horizon=5)
        good = pooler.action_head.state_dict()
        bad = {"noise.extra": torch.randn(1)}
        mixed = {**good, **bad}
        loaded = pooler.load_weights(mixed)
        # Only matching keys should be consumed
        assert len(loaded) == len(good)
        for k in good:
            assert k in loaded
        for k in bad:
            assert k not in loaded

    def test_pooler_load_weights_filters_by_dtype(self) -> None:
        """load_weights converts tensor dtype to match model parameter."""
        pooler = DiTActionPooler()
        sd = pooler.action_head.state_dict()
        if not sd:
            pytest.skip("ActionHead has no parameters")
        first_key = next(iter(sd))
        orig_dtype = sd[first_key].dtype
        weights = {first_key: torch.randn_like(sd[first_key], dtype=torch.float64)}
        loaded = pooler.load_weights(weights)
        assert first_key in loaded
        assert sd[first_key].dtype == orig_dtype


# ---------------------------------------------------------------------------
# GR00tActionHead unit checks
# ---------------------------------------------------------------------------


class TestGR00tActionHead:
    """Verify action head inference-only path shapes."""

    def test_ctor_diffusion_cfg_merge(self) -> None:
        head = GR00tActionHead(
            hidden_size=1024,
            action_model_type="DiT-B",
            diffusion_model_cfg={"cross_attention_dim": 512},
        )
        # Check that user-provided cfg overwrites the default
        assert head.model.config.cross_attention_dim == 512

    def test_ctor_default_diffusion_cfg(self) -> None:
        head = GR00tActionHead(hidden_size=1024, action_model_type="DiT-B")
        # Default cross_attention_dim should be 1024
        assert head.model.config.cross_attention_dim == 1024

    def test_predict_action_output_shape(self) -> None:
        """predict_action returns (B, action_horizon, action_dim)."""
        head = GR00tActionHead(
            hidden_size=1024,
            action_model_type="DiT-B",
            action_dim=80,
            state_dim=80,
            action_horizon=30,
            num_inference_timesteps=4,
        )
        head.eval()
        with torch.no_grad():
            # Simulate pooled VLM hidden states (batch=1, seq=32, dim=1024)
            vl_embs = torch.randn(1, 32, 1024)
            state = torch.randn(1, 1, 80)
            actions = head.predict_action(vl_embs=vl_embs, state=state)
        assert actions.shape == (1, 30, 80)
        assert not torch.all(actions == 0), (
            "predict_action should produce non-zero actions with random inputs"
        )


# ---------------------------------------------------------------------------
# Processor / multimodal setup checks
# ---------------------------------------------------------------------------


class TestMiniCPMRobotProcessor:
    """Verify the model reuses MiniCPM-V 4.6 processor registration."""

    def test_model_has_multimodal_processor(self) -> None:
        """Our model should have a registered multimodal processor via
        the @MULTIMODAL_REGISTRY decorator."""
        assert hasattr(MiniCPMRobotForHiddenStates, "_processor_factory"), (
            "MiniCPMRobotForHiddenStates should have a registered multimodal processor"
        )

    def test_processor_is_minicpmv4_6(self) -> None:
        """Processor factory should be MiniCPMV4_6MultiModalProcessor."""
        from vllm.model_executor.models.minicpmv4_6 import (
            MiniCPMV4_6MultiModalProcessor,
        )

        factory = MiniCPMRobotForHiddenStates._processor_factory
        assert factory.processor is MiniCPMV4_6MultiModalProcessor

    def test_dummy_inputs_is_minicpmv(self) -> None:
        """Dummy inputs builder should be MiniCPMVDummyInputsBuilder."""
        from vllm.model_executor.models.minicpmv import MiniCPMVDummyInputsBuilder

        factory = MiniCPMRobotForHiddenStates._processor_factory
        assert factory.dummy_inputs is MiniCPMVDummyInputsBuilder


# ---------------------------------------------------------------------------
# ActionHead init-in-isolation test
# ---------------------------------------------------------------------------


def test_action_head_configuration() -> None:
    """GR00tActionHead should accept runtime configuration knobs."""
    head = GR00tActionHead(
        hidden_size=1024,
        action_model_type="DiT-B",
        action_dim=128,
        state_dim=256,
        action_horizon=16,
        num_inference_timesteps=8,
        proprio_inject="concat",
        prediction_type="clean_action",
    )
    assert head.num_inference_timesteps == 8
    assert head.proprio_inject == "concat"
    assert head.prediction_type == "clean_action"
