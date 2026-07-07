# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the FlashInfer autotune warmup context manager and the SM100
autotune guard in TrtLlmBf16Experts."""

import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEParallelConfig,
    FusedMoEQuantConfig,
    RoutingMethodType,
)
from vllm.model_executor.layers.fused_moe.experts.trtllm_bf16_moe import (
    TrtLlmBf16Experts,
)


###############################################################################
# Fixtures
###############################################################################


@pytest.fixture(autouse=True)
def _fake_flashinfer_for_trtllm_tests(monkeypatch):
    """Install a fake flashinfer module so patches to flashinfer.fused_moe work
    without requiring the real FlashInfer package."""
    fake_trtllm = MagicMock()
    fake_fused_moe = SimpleNamespace(trtllm_bf16_moe=fake_trtllm)
    fake_flashinfer = SimpleNamespace(fused_moe=fake_fused_moe)
    monkeypatch.setitem(sys.modules, "flashinfer", fake_flashinfer)
    return fake_trtllm


def _make_dummy_moe_config() -> FusedMoEConfig:
    """Create a minimal FusedMoEConfig for testing TrtLlmBf16Experts."""
    return FusedMoEConfig(
        num_experts=64,
        experts_per_token=3,
        hidden_dim=512,
        intermediate_size=1024,
        num_local_experts=32,
        num_logical_experts=64,
        moe_parallel_config=FusedMoEParallelConfig.make_no_parallel(),
        activation=MoEActivation.SILU,
        in_dtype=torch.bfloat16,
        device="cuda",
        routing_method=RoutingMethodType.TopK,
        max_num_tokens=512,
    )


def _make_dummy_quant_config() -> FusedMoEQuantConfig:
    """Create a minimal unquantized FusedMoEQuantConfig."""
    return FusedMoEQuantConfig()


def _create_trtllm_experts() -> TrtLlmBf16Experts:
    """Create a TrtLlmBf16Experts instance with dummy configs."""
    moe_config = _make_dummy_moe_config()
    quant_config = _make_dummy_quant_config()
    return TrtLlmBf16Experts(moe_config=moe_config, quant_config=quant_config)


def _make_dummy_tensors() -> dict:
    """Create dummy tensors matching a minimal MoE forward call."""
    return {
        "hidden_states": torch.randn(4, 512, dtype=torch.bfloat16),
        "w1": torch.randn(32, 2048, 512, dtype=torch.bfloat16),
        "w2": torch.randn(32, 512, 1024, dtype=torch.bfloat16),
        "router_logits": torch.randn(4, 64, dtype=torch.bfloat16),
        "activation": MoEActivation.SILU,
        "global_num_experts": 64,
        "expert_map": None,
        "a1q_scale": None,
        "apply_router_weight_on_input": False,
    }


###############################################################################
# autotune_warmup context manager tests
###############################################################################


class TestAutotuneWarmupContext:
    """Tests for the autotune_warmup() context manager state transitions."""

    def test_default_false(self):
        """Returns False by default."""
        from vllm.utils.flashinfer import in_flashinfer_autotune_warmup

        assert in_flashinfer_autotune_warmup() is False

    def test_sets_flag_inside(self):
        """Returns True inside the context."""
        from vllm.utils.flashinfer import (
            autotune_warmup,
            in_flashinfer_autotune_warmup,
        )

        with autotune_warmup():
            assert in_flashinfer_autotune_warmup() is True

    def test_clears_flag_after_exit(self):
        """Returns False after exiting the context."""
        from vllm.utils.flashinfer import (
            autotune_warmup,
            in_flashinfer_autotune_warmup,
        )

        with autotune_warmup():
            pass
        assert in_flashinfer_autotune_warmup() is False

    def test_clears_flag_on_exception(self):
        """Returns False even if an exception occurs inside."""
        from vllm.utils.flashinfer import (
            autotune_warmup,
            in_flashinfer_autotune_warmup,
        )

        class _TestError(Exception):
            pass

        with pytest.raises(_TestError):
            with autotune_warmup():
                assert in_flashinfer_autotune_warmup() is True
                raise _TestError()
        assert in_flashinfer_autotune_warmup() is False

    def test_nested_contexts(self):
        """Nested contexts both report True."""
        from vllm.utils.flashinfer import (
            autotune_warmup,
            in_flashinfer_autotune_warmup,
        )

        with autotune_warmup():
            assert in_flashinfer_autotune_warmup() is True
            with autotune_warmup():
                assert in_flashinfer_autotune_warmup() is True
            assert in_flashinfer_autotune_warmup() is True
        assert in_flashinfer_autotune_warmup() is False

    def test_passes_args_through(self):
        """Args are forwarded to autotune()."""
        from vllm.utils.flashinfer import autotune_warmup

        mock_cm = MagicMock()
        mock_cm.__enter__ = MagicMock(return_value=None)
        mock_cm.__exit__ = MagicMock(return_value=None)

        with patch(
            "vllm.utils.flashinfer.autotune", return_value=mock_cm
        ) as mock_autotune:
            with autotune_warmup(tune_mode=True, cache="/tmp/cache.json"):
                pass

            mock_autotune.assert_called_once_with(
                tune_mode=True, cache="/tmp/cache.json"
            )

    def test_no_args(self):
        """No-args call passes through correctly."""
        from vllm.utils.flashinfer import autotune_warmup

        mock_cm = MagicMock()
        mock_cm.__enter__ = MagicMock(return_value=None)
        mock_cm.__exit__ = MagicMock(return_value=None)

        with patch(
            "vllm.utils.flashinfer.autotune", return_value=mock_cm
        ) as mock_autotune:
            with autotune_warmup():
                pass

            mock_autotune.assert_called_once_with()


###############################################################################
# TrtLlmBf16Experts SM100 autotune guard tests
###############################################################################


class TestTrtLlmBf16ExpertsGuard:
    """Tests for the SM100 + warmup autotune guard in TrtLlmBf16Experts."""

    @patch(
        "vllm.model_executor.layers.fused_moe.experts.trtllm_bf16_moe.flashinfer_autotune_fn"  # noqa: E501
    )
    @patch(
        "vllm.model_executor.layers.fused_moe.experts.trtllm_bf16_moe.in_flashinfer_autotune_warmup"  # noqa: E501
    )
    @patch(
        "vllm.model_executor.layers.fused_moe.experts.trtllm_bf16_moe.current_platform.is_device_capability_family"  # noqa: E501
    )
    def test_sm100_warmup_guard_applied(
        self,
        mock_is_device_family,
        mock_in_warmup,
        mock_autotune_fn,
    ):
        """On SM100 during warmup, autotune(False) is applied."""
        mock_is_device_family.return_value = True
        mock_in_warmup.return_value = True
        mock_autotune_cm = MagicMock()
        mock_autotune_cm.__enter__ = MagicMock(return_value=None)
        mock_autotune_cm.__exit__ = MagicMock(return_value=None)
        mock_autotune_fn.return_value = mock_autotune_cm

        experts = _create_trtllm_experts()
        experts.apply(**_make_dummy_tensors())

        mock_autotune_fn.assert_called_once_with(False)
        mock_autotune_cm.__enter__.assert_called_once()

    @patch(
        "vllm.model_executor.layers.fused_moe.experts.trtllm_bf16_moe.flashinfer_autotune_fn"  # noqa: E501
    )
    @patch(
        "vllm.model_executor.layers.fused_moe.experts.trtllm_bf16_moe.in_flashinfer_autotune_warmup"  # noqa: E501
    )
    @patch(
        "vllm.model_executor.layers.fused_moe.experts.trtllm_bf16_moe.current_platform.is_device_capability_family"  # noqa: E501
    )
    def test_sm90_warmup_no_guard(
        self,
        mock_is_device_family,
        mock_in_warmup,
        mock_autotune_fn,
    ):
        """On SM90, no guard — is_device_capability_family(100) is False."""
        mock_is_device_family.return_value = False  # SM90 is not SM100 family
        mock_in_warmup.return_value = True

        experts = _create_trtllm_experts()
        experts.apply(**_make_dummy_tensors())

        mock_autotune_fn.assert_not_called()
        assert sys.modules["flashinfer"].fused_moe.trtllm_bf16_moe.called

    @patch(
        "vllm.model_executor.layers.fused_moe.experts.trtllm_bf16_moe.flashinfer_autotune_fn"  # noqa: E501
    )
    @patch(
        "vllm.model_executor.layers.fused_moe.experts.trtllm_bf16_moe.in_flashinfer_autotune_warmup"  # noqa: E501
    )
    @patch(
        "vllm.model_executor.layers.fused_moe.experts.trtllm_bf16_moe.current_platform.is_device_capability_family"  # noqa: E501
    )
    def test_sm100_outside_warmup_no_guard(
        self,
        mock_is_device_family,
        mock_in_warmup,
        mock_autotune_fn,
    ):
        """On SM100 but outside warmup, no guard."""
        mock_is_device_family.return_value = True
        mock_in_warmup.return_value = False

        experts = _create_trtllm_experts()
        experts.apply(**_make_dummy_tensors())

        mock_autotune_fn.assert_not_called()
        assert sys.modules["flashinfer"].fused_moe.trtllm_bf16_moe.called

    @patch(
        "vllm.model_executor.layers.fused_moe.experts.trtllm_bf16_moe.flashinfer_autotune_fn"  # noqa: E501
    )
    @patch(
        "vllm.model_executor.layers.fused_moe.experts.trtllm_bf16_moe.in_flashinfer_autotune_warmup"  # noqa: E501
    )
    @patch(
        "vllm.model_executor.layers.fused_moe.experts.trtllm_bf16_moe.current_platform.is_device_capability_family"  # noqa: E501
    )
    def test_non_blackwell_device_no_guard(
        self,
        mock_is_device_family,
        mock_in_warmup,
        mock_autotune_fn,
    ):
        """On non-Blackwell devices, no guard."""
        mock_is_device_family.return_value = False
        mock_in_warmup.return_value = True

        experts = _create_trtllm_experts()
        experts.apply(**_make_dummy_tensors())

        mock_autotune_fn.assert_not_called()
        assert sys.modules["flashinfer"].fused_moe.trtllm_bf16_moe.called

    @patch(
        "vllm.model_executor.layers.fused_moe.experts.trtllm_bf16_moe.flashinfer_autotune_fn"  # noqa: E501
    )
    @patch(
        "vllm.model_executor.layers.fused_moe.experts.trtllm_bf16_moe.in_flashinfer_autotune_warmup"  # noqa: E501
    )
    @patch(
        "vllm.model_executor.layers.fused_moe.experts.trtllm_bf16_moe.current_platform.is_device_capability_family"  # noqa: E501
    )
    def test_kernel_args_preserved(
        self,
        mock_is_device_family,
        mock_in_warmup,
        mock_autotune_fn,
    ):
        """When guard is active, kernel arguments pass through unchanged."""
        mock_is_device_family.return_value = True
        mock_in_warmup.return_value = True
        mock_autotune_cm = MagicMock()
        mock_autotune_cm.__enter__ = MagicMock(return_value=None)
        mock_autotune_cm.__exit__ = MagicMock(return_value=None)
        mock_autotune_fn.return_value = mock_autotune_cm

        experts = _create_trtllm_experts()
        experts.topk = 3
        experts.intermediate_size_per_partition = 1024
        experts.ep_rank = 0
        experts.local_num_experts = 32

        tensors = _make_dummy_tensors()
        tensors["global_num_experts"] = 64
        experts.apply(**tensors)

        # Access the fake flashinfer module installed by the fixture
        call_kwargs = sys.modules["flashinfer"].fused_moe.trtllm_bf16_moe.call_args.kwargs
        assert call_kwargs["num_experts"] == 64
        assert call_kwargs["top_k"] == 3
        assert call_kwargs["intermediate_size"] == 1024
        assert call_kwargs["local_expert_offset"] == 0
        assert call_kwargs["local_num_experts"] == 32
