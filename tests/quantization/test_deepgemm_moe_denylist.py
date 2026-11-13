# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Regression test for GitHub issue #50332.

When ``quant_config.use_deep_gemm`` is set to ``False`` (by the per-model-type
Blackwell denylist in ``VllmConfig._verify_quantization``), the MoE backend
selector must NOT choose DEEPGEMM or BATCHED_DEEPGEMM.

This is a CPU-only unit test: it exercises ``select_fp8_moe_backend`` directly
via ``force_disable_deep_gemm=True``, and also exercises the end-to-end path
through ``Fp8MoEMethod.__init__`` using monkeypatching so that no GPU is
required.
"""

import unittest.mock as mock

from vllm.model_executor.layers.fused_moe.oracle.fp8 import (
    Fp8MoeBackend,
    select_fp8_moe_backend,
)

# ---------------------------------------------------------------------------
# Helpers – minimal stubs for FusedMoEConfig / FusedMoEParallelConfig
# ---------------------------------------------------------------------------


def _make_parallel_config() -> "object":
    """Return a stub FusedMoEParallelConfig with single-process defaults."""
    from vllm.model_executor.layers.fused_moe.config import FusedMoEParallelConfig

    return FusedMoEParallelConfig(
        tp_size=1,
        pcp_size=1,
        dp_size=1,
        ep_size=1,
        tp_rank=0,
        pcp_rank=0,
        dp_rank=0,
        ep_rank=0,
        sp_size=1,
        use_ep=False,
        all2all_backend="none",
        enable_eplb=False,
    )


def _make_moe_config(moe_backend: str = "auto") -> "object":
    """Return a minimal FusedMoEConfig for backend-selector unit tests."""
    import torch

    from vllm.model_executor.layers.fused_moe.config import (
        FusedMoEConfig,
        MoEActivation,
    )
    from vllm.model_executor.layers.fused_moe.routing import (
        RoutingType,
        get_default_routing_method,
    )

    routing = get_default_routing_method(
        RoutingType.TOPK,
        num_experts=8,
        experts_per_token=2,
        dp_size=1,
    )
    return FusedMoEConfig(
        num_experts=8,
        experts_per_token=2,
        hidden_dim=256,
        intermediate_size_per_partition=512,
        num_local_experts=8,
        num_logical_experts=8,
        activation=MoEActivation.SiluAndMul,
        device=torch.device("cpu"),
        routing_method=routing,
        moe_parallel_config=_make_parallel_config(),
        in_dtype=torch.bfloat16,
        moe_backend=moe_backend,
    )


# ---------------------------------------------------------------------------
# Test 1: force_disable_deep_gemm removes DEEPGEMM variants from candidates
# ---------------------------------------------------------------------------


class TestForceDisableDeepGemm:
    """Unit tests for the ``force_disable_deep_gemm`` parameter."""

    def test_without_flag_deepgemm_may_be_available(self, monkeypatch):
        """Baseline: without the flag, DEEPGEMM backends are in the list.

        We monkeypatch ``_get_priority_backends`` to return a predictable list
        so this test does not depend on the current platform.
        """
        import vllm.model_executor.layers.fused_moe.oracle.fp8 as fp8_oracle

        # Make every backend appear "supported" so we can see what's selected.
        mock_cls = mock.MagicMock()
        mock_cls.is_supported_config.return_value = (True, None)

        monkeypatch.setattr(
            fp8_oracle,
            "_get_priority_backends",
            lambda *a, **kw: [
                Fp8MoeBackend.DEEPGEMM,
                Fp8MoeBackend.BATCHED_DEEPGEMM,
                Fp8MoeBackend.TRITON,
            ],
        )
        monkeypatch.setattr(
            fp8_oracle,
            "backend_to_kernel_cls",
            lambda backend: [mock_cls],
        )
        # Suppress env-var guards
        monkeypatch.setattr(fp8_oracle.envs, "is_set", lambda _: False)
        monkeypatch.setattr(fp8_oracle.envs, "VLLM_TEST_FORCE_FP8_MARLIN", False)

        config = _make_moe_config()
        backend, _ = select_fp8_moe_backend(
            config=config,
            weight_key=None,
            activation_key=None,
            force_disable_deep_gemm=False,
        )
        # Without the flag the first supported backend (DEEPGEMM) is picked.
        assert backend == Fp8MoeBackend.DEEPGEMM

    def test_with_flag_deepgemm_skipped(self, monkeypatch):
        """With force_disable_deep_gemm=True, DEEPGEMM must NOT be selected."""
        import vllm.model_executor.layers.fused_moe.oracle.fp8 as fp8_oracle

        mock_cls = mock.MagicMock()
        mock_cls.is_supported_config.return_value = (True, None)

        monkeypatch.setattr(
            fp8_oracle,
            "_get_priority_backends",
            lambda *a, **kw: [
                Fp8MoeBackend.DEEPGEMM,
                Fp8MoeBackend.BATCHED_DEEPGEMM,
                Fp8MoeBackend.TRITON,
            ],
        )
        monkeypatch.setattr(
            fp8_oracle,
            "backend_to_kernel_cls",
            lambda backend: [mock_cls],
        )
        monkeypatch.setattr(fp8_oracle.envs, "is_set", lambda _: False)
        monkeypatch.setattr(fp8_oracle.envs, "VLLM_TEST_FORCE_FP8_MARLIN", False)

        config = _make_moe_config()
        backend, _ = select_fp8_moe_backend(
            config=config,
            weight_key=None,
            activation_key=None,
            force_disable_deep_gemm=True,
        )
        # DEEPGEMM and BATCHED_DEEPGEMM must have been removed; TRITON is next.
        assert backend not in (Fp8MoeBackend.DEEPGEMM, Fp8MoeBackend.BATCHED_DEEPGEMM)
        assert backend == Fp8MoeBackend.TRITON

    def test_batched_deepgemm_also_skipped(self, monkeypatch):
        """BATCHED_DEEPGEMM must also be skipped when the flag is set."""
        import vllm.model_executor.layers.fused_moe.oracle.fp8 as fp8_oracle

        mock_cls = mock.MagicMock()
        mock_cls.is_supported_config.return_value = (True, None)

        monkeypatch.setattr(
            fp8_oracle,
            "_get_priority_backends",
            lambda *a, **kw: [
                Fp8MoeBackend.BATCHED_DEEPGEMM,
                Fp8MoeBackend.BATCHED_TRITON,
            ],
        )
        monkeypatch.setattr(
            fp8_oracle,
            "backend_to_kernel_cls",
            lambda backend: [mock_cls],
        )
        monkeypatch.setattr(fp8_oracle.envs, "is_set", lambda _: False)
        monkeypatch.setattr(fp8_oracle.envs, "VLLM_TEST_FORCE_FP8_MARLIN", False)

        config = _make_moe_config()
        backend, _ = select_fp8_moe_backend(
            config=config,
            weight_key=None,
            activation_key=None,
            force_disable_deep_gemm=True,
        )
        assert backend == Fp8MoeBackend.BATCHED_TRITON


# ---------------------------------------------------------------------------
# Test 2: Fp8MoEMethod.__init__ passes force_disable_deep_gemm correctly
# ---------------------------------------------------------------------------


class TestFp8MoEMethodDenylist:
    """End-to-end path: quant_config.use_deep_gemm=False => no DEEPGEMM."""

    def _make_quant_config(self, use_deep_gemm: bool | None):
        from vllm.model_executor.layers.quantization.fp8 import Fp8Config

        cfg = Fp8Config(
            is_checkpoint_fp8_serialized=True,
            activation_scheme="dynamic",
        )
        cfg.use_deep_gemm = use_deep_gemm
        return cfg

    def test_use_deep_gemm_false_passes_flag(self, monkeypatch):
        """When use_deep_gemm is False, Fp8MoEMethod must pass
        force_disable_deep_gemm=True to select_fp8_moe_backend."""
        import vllm.model_executor.layers.quantization.fp8 as fp8_mod

        captured: dict = {}

        def fake_select(
            config,
            weight_key,
            activation_key,
            allow_vllm_cutlass=False,
            force_disable_deep_gemm=False,
        ):
            captured["force_disable_deep_gemm"] = force_disable_deep_gemm
            return (Fp8MoeBackend.TRITON, mock.MagicMock())

        monkeypatch.setattr(fp8_mod, "select_fp8_moe_backend", fake_select)

        quant_config = self._make_quant_config(use_deep_gemm=False)
        layer = mock.MagicMock()
        layer.moe_config = _make_moe_config()

        fp8_mod.Fp8MoEMethod(quant_config=quant_config, layer=layer)

        assert captured.get("force_disable_deep_gemm") is True, (
            "Expected force_disable_deep_gemm=True when use_deep_gemm is False"
        )

    def test_use_deep_gemm_none_does_not_pass_flag(self, monkeypatch):
        """When use_deep_gemm is None (not set), force_disable_deep_gemm
        must be False (no forced exclusion)."""
        import vllm.model_executor.layers.quantization.fp8 as fp8_mod

        captured: dict = {}

        def fake_select(
            config,
            weight_key,
            activation_key,
            allow_vllm_cutlass=False,
            force_disable_deep_gemm=False,
        ):
            captured["force_disable_deep_gemm"] = force_disable_deep_gemm
            return (Fp8MoeBackend.TRITON, mock.MagicMock())

        monkeypatch.setattr(fp8_mod, "select_fp8_moe_backend", fake_select)

        quant_config = self._make_quant_config(use_deep_gemm=None)
        layer = mock.MagicMock()
        layer.moe_config = _make_moe_config()

        fp8_mod.Fp8MoEMethod(quant_config=quant_config, layer=layer)

        assert captured.get("force_disable_deep_gemm") is False, (
            "Expected force_disable_deep_gemm=False when use_deep_gemm is None"
        )

    def test_use_deep_gemm_true_does_not_pass_flag(self, monkeypatch):
        """When use_deep_gemm is True (explicitly allowed), the flag must be
        False so DEEPGEMM remains a candidate."""
        import vllm.model_executor.layers.quantization.fp8 as fp8_mod

        captured: dict = {}

        def fake_select(
            config,
            weight_key,
            activation_key,
            allow_vllm_cutlass=False,
            force_disable_deep_gemm=False,
        ):
            captured["force_disable_deep_gemm"] = force_disable_deep_gemm
            return (Fp8MoeBackend.TRITON, mock.MagicMock())

        monkeypatch.setattr(fp8_mod, "select_fp8_moe_backend", fake_select)

        quant_config = self._make_quant_config(use_deep_gemm=True)
        layer = mock.MagicMock()
        layer.moe_config = _make_moe_config()

        fp8_mod.Fp8MoEMethod(quant_config=quant_config, layer=layer)

        assert captured.get("force_disable_deep_gemm") is False, (
            "Expected force_disable_deep_gemm=False when use_deep_gemm is True"
        )
