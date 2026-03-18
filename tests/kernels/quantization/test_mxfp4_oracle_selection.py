# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Test MXFP4 oracle for-loop selection logic.
Tests that the priority-based backend selection and MXFP8 activation
key mapping work correctly.
"""

from dataclasses import dataclass
from unittest.mock import MagicMock, patch

import pytest

from vllm.model_executor.layers.fused_moe.oracle.mxfp4 import (
    Mxfp4MoeBackend,
    _backend_activation_key,
    _get_priority_backends,
    select_mxfp4_moe_backend,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    kMxfp8Dynamic,
)


@dataclass
class _MockParallelConfig:
    tp_size: int = 1
    pcp_size: int = 1
    dp_size: int = 1
    ep_size: int = 1
    tp_rank: int = 0
    pcp_rank: int = 0
    dp_rank: int = 0
    ep_rank: int = 0
    sp_size: int = 1
    use_ep: bool = False
    all2all_backend: str = "naive"
    enable_eplb: bool = False
    use_batched_activation_format: bool = False

    @property
    def use_all2all_kernels(self):
        return self.dp_size > 1 and self.use_ep

    @property
    def use_deepep_ll_kernels(self):
        return False


class TestBackendActivationKey:
    """Test _backend_activation_key returns correct keys."""

    def test_bf16_backends_return_none(self):
        for backend in [
            Mxfp4MoeBackend.FLASHINFER_TRTLLM_MXFP4_BF16,
            Mxfp4MoeBackend.FLASHINFER_CUTLASS_MXFP4_BF16,
            Mxfp4MoeBackend.CK,
            Mxfp4MoeBackend.TRITON,
            Mxfp4MoeBackend.TRITON_UNFUSED,
            Mxfp4MoeBackend.MARLIN,
            Mxfp4MoeBackend.BATCHED_MARLIN,
        ]:
            assert _backend_activation_key(backend) is None, (
                f"{backend} should have activation_key=None"
            )

    def test_mxfp8_backends_return_kMxfp8Dynamic(self):
        for backend in [
            Mxfp4MoeBackend.FLASHINFER_TRTLLM_MXFP4_MXFP8,
            Mxfp4MoeBackend.FLASHINFER_CUTLASS_MXFP4_MXFP8,
        ]:
            assert _backend_activation_key(backend) is kMxfp8Dynamic, (
                f"{backend} should have activation_key=kMxfp8Dynamic"
            )


class TestGetPriorityBackends:
    """Test _get_priority_backends returns correct priority list."""

    def test_returns_bf16_backends_only(self):
        backends = _get_priority_backends()
        mxfp8_backends = {
            Mxfp4MoeBackend.FLASHINFER_TRTLLM_MXFP4_MXFP8,
            Mxfp4MoeBackend.FLASHINFER_CUTLASS_MXFP4_MXFP8,
        }
        for b in backends:
            assert b not in mxfp8_backends, (
                f"MXFP8 backend {b} should not be in default priority list"
            )

    def test_trtllm_before_ck(self):
        backends = _get_priority_backends()
        fi_idx = backends.index(Mxfp4MoeBackend.FLASHINFER_TRTLLM_MXFP4_BF16)
        ck_idx = backends.index(Mxfp4MoeBackend.CK)
        assert fi_idx < ck_idx, "FlashInfer TRTLLM BF16 should come before CK"

    def test_ck_before_triton(self):
        backends = _get_priority_backends()
        ck_idx = backends.index(Mxfp4MoeBackend.CK)
        triton_idx = backends.index(Mxfp4MoeBackend.TRITON)
        assert ck_idx < triton_idx, "CK should come before Triton"

    def test_triton_before_cutlass(self):
        backends = _get_priority_backends()
        triton_idx = backends.index(Mxfp4MoeBackend.TRITON)
        cutlass_idx = backends.index(Mxfp4MoeBackend.FLASHINFER_CUTLASS_MXFP4_BF16)
        assert triton_idx < cutlass_idx, (
            "Triton should come before FlashInfer CUTLASS BF16"
        )

    def test_cutlass_before_marlin(self):
        backends = _get_priority_backends()
        cutlass_idx = backends.index(Mxfp4MoeBackend.FLASHINFER_CUTLASS_MXFP4_BF16)
        marlin_idx = backends.index(Mxfp4MoeBackend.MARLIN)
        assert cutlass_idx < marlin_idx, (
            "FlashInfer CUTLASS BF16 should come before Marlin"
        )

    def test_no_xpu_in_list(self):
        backends = _get_priority_backends()
        assert Mxfp4MoeBackend.XPU not in backends

    def test_returns_fresh_list(self):
        """Mutating returned list should not affect next call."""
        a = _get_priority_backends()
        a.pop()
        b = _get_priority_backends()
        assert len(b) > len(a)


class TestSelectMxfp4MoeBackendForLoop:
    """Test the for-loop default selection path in select_mxfp4_moe_backend."""

    def _make_config(self, moe_backend="auto", is_lora=False, **parallel_kw):
        parallel = _MockParallelConfig(**parallel_kw)
        config = MagicMock()
        config.moe_backend = moe_backend
        config.is_lora_enabled = is_lora
        config.moe_parallel_config = parallel
        return config

    @patch("vllm.model_executor.layers.fused_moe.oracle.mxfp4.current_platform")
    @patch(
        "vllm.model_executor.layers.fused_moe.oracle.mxfp4.has_triton_kernels",
        return_value=True,
    )
    @patch("vllm.model_executor.layers.fused_moe.oracle.mxfp4.envs")
    def test_first_supported_backend_wins(self, mock_envs, mock_triton, mock_platform):
        """The for-loop should return the first backend whose kernel
        reports is_supported_config=True."""
        mock_platform.get_device_capability.return_value = (10, 0)
        mock_platform.is_cuda.return_value = True
        mock_platform.is_rocm.return_value = False
        mock_platform.is_xpu.return_value = False
        mock_envs.is_set.return_value = False
        mock_envs.VLLM_MXFP4_USE_MARLIN = None

        config = self._make_config()

        # Mock backend_to_kernel_cls to control which backends are "supported"
        fake_supported_cls = MagicMock()
        fake_supported_cls.is_supported_config.return_value = (True, None)

        fake_unsupported_cls = MagicMock()
        fake_unsupported_cls.is_supported_config.return_value = (False, "not supported")

        # FLASHINFER_TRTLLM_BF16 -> unsupported, FLASHINFER_CUTLASS_BF16 -> supported
        def mock_backend_to_kernel_cls(backend):
            if backend == Mxfp4MoeBackend.FLASHINFER_TRTLLM_MXFP4_BF16:
                return [fake_unsupported_cls]
            elif backend == Mxfp4MoeBackend.FLASHINFER_CUTLASS_MXFP4_BF16:
                return [fake_supported_cls]
            return [fake_unsupported_cls]

        with patch(
            "vllm.model_executor.layers.fused_moe.oracle.mxfp4.backend_to_kernel_cls",
            side_effect=mock_backend_to_kernel_cls,
        ):
            backend, k_cls = select_mxfp4_moe_backend(config)

        assert backend == Mxfp4MoeBackend.FLASHINFER_CUTLASS_MXFP4_BF16
        assert k_cls is fake_supported_cls

    @patch("vllm.model_executor.layers.fused_moe.oracle.mxfp4.current_platform")
    @patch(
        "vllm.model_executor.layers.fused_moe.oracle.mxfp4.has_triton_kernels",
        return_value=True,
    )
    @patch("vllm.model_executor.layers.fused_moe.oracle.mxfp4.envs")
    def test_triton_selected_when_flashinfer_unsupported(
        self, mock_envs, mock_triton, mock_platform
    ):
        """If FlashInfer backends are unsupported, Triton should be next."""
        mock_platform.get_device_capability.return_value = (9, 0)
        mock_platform.is_cuda.return_value = True
        mock_platform.is_rocm.return_value = False
        mock_platform.is_xpu.return_value = False
        mock_envs.is_set.return_value = False
        mock_envs.VLLM_MXFP4_USE_MARLIN = None

        config = self._make_config()

        fake_supported_cls = MagicMock()
        fake_supported_cls.is_supported_config.return_value = (True, None)
        fake_unsupported_cls = MagicMock()
        fake_unsupported_cls.is_supported_config.return_value = (False, "not supported")

        def mock_backend_to_kernel_cls(backend):
            if backend == Mxfp4MoeBackend.TRITON:
                return [fake_supported_cls]
            return [fake_unsupported_cls]

        with patch(
            "vllm.model_executor.layers.fused_moe.oracle.mxfp4.backend_to_kernel_cls",
            side_effect=mock_backend_to_kernel_cls,
        ):
            backend, k_cls = select_mxfp4_moe_backend(config)

        assert backend == Mxfp4MoeBackend.TRITON
        assert k_cls is fake_supported_cls

    @patch("vllm.model_executor.layers.fused_moe.oracle.mxfp4.current_platform")
    @patch(
        "vllm.model_executor.layers.fused_moe.oracle.mxfp4.has_triton_kernels",
        return_value=True,
    )
    @patch("vllm.model_executor.layers.fused_moe.oracle.mxfp4.envs")
    def test_fi_bf16_disabled_removes_flashinfer(
        self, mock_envs, mock_triton, mock_platform
    ):
        """When VLLM_USE_FLASHINFER_MOE_MXFP4_BF16=0, FlashInfer BF16
        backends should be skipped."""
        mock_platform.get_device_capability.return_value = (10, 0)
        mock_platform.is_cuda.return_value = True
        mock_platform.is_rocm.return_value = False
        mock_platform.is_xpu.return_value = False
        # fi_bf16_disabled = is_set("...BF16") and not ...BF16
        mock_envs.is_set.side_effect = (
            lambda k: k == "VLLM_USE_FLASHINFER_MOE_MXFP4_BF16"
        )
        mock_envs.VLLM_USE_FLASHINFER_MOE_MXFP4_BF16 = False
        mock_envs.VLLM_MXFP4_USE_MARLIN = None

        config = self._make_config()

        fake_supported_cls = MagicMock()
        fake_supported_cls.is_supported_config.return_value = (True, None)

        visited_backends = []

        def mock_backend_to_kernel_cls(backend):
            visited_backends.append(backend)
            return [fake_supported_cls]

        with patch(
            "vllm.model_executor.layers.fused_moe.oracle.mxfp4.backend_to_kernel_cls",
            side_effect=mock_backend_to_kernel_cls,
        ):
            backend, k_cls = select_mxfp4_moe_backend(config)

        # FlashInfer BF16 backends should NOT appear
        assert Mxfp4MoeBackend.FLASHINFER_TRTLLM_MXFP4_BF16 not in visited_backends
        assert Mxfp4MoeBackend.FLASHINFER_CUTLASS_MXFP4_BF16 not in visited_backends
        # First visited should be CK (second in priority after FlashInfer)
        assert backend == Mxfp4MoeBackend.CK

    @patch("vllm.model_executor.layers.fused_moe.oracle.mxfp4.current_platform")
    @patch(
        "vllm.model_executor.layers.fused_moe.oracle.mxfp4.has_triton_kernels",
        return_value=True,
    )
    @patch("vllm.model_executor.layers.fused_moe.oracle.mxfp4.envs")
    def test_no_supported_backend_raises_on_cuda(
        self, mock_envs, mock_triton, mock_platform
    ):
        """If no backend is supported on CUDA, should raise NotImplementedError."""
        mock_platform.get_device_capability.return_value = (9, 0)
        mock_platform.is_cuda.return_value = True
        mock_platform.is_rocm.return_value = False
        mock_platform.is_xpu.return_value = False
        mock_envs.is_set.return_value = False
        mock_envs.VLLM_MXFP4_USE_MARLIN = None

        config = self._make_config()

        fake_unsupported_cls = MagicMock()
        fake_unsupported_cls.is_supported_config.return_value = (False, "not supported")

        with (
            patch(
                "vllm.model_executor.layers.fused_moe.oracle.mxfp4.backend_to_kernel_cls",
                return_value=[fake_unsupported_cls],
            ),
            pytest.raises(NotImplementedError, match="No MXFP4 MoE backend"),
        ):
            select_mxfp4_moe_backend(config)

    @patch("vllm.model_executor.layers.fused_moe.oracle.mxfp4.current_platform")
    @patch(
        "vllm.model_executor.layers.fused_moe.oracle.mxfp4.has_triton_kernels",
        return_value=True,
    )
    @patch("vllm.model_executor.layers.fused_moe.oracle.mxfp4.envs")
    def test_xpu_fallback_after_loop(self, mock_envs, mock_triton, mock_platform):
        """If no for-loop backend supports config but platform is XPU,
        should return XPU backend with None cls."""
        mock_platform.get_device_capability.return_value = (0, 0)
        mock_platform.is_cuda.return_value = False
        mock_platform.is_rocm.return_value = False
        mock_platform.is_xpu.return_value = True
        mock_envs.is_set.return_value = False
        mock_envs.VLLM_MXFP4_USE_MARLIN = None

        config = self._make_config()

        fake_unsupported_cls = MagicMock()
        fake_unsupported_cls.is_supported_config.return_value = (False, "not supported")

        with patch(
            "vllm.model_executor.layers.fused_moe.oracle.mxfp4.backend_to_kernel_cls",
            return_value=[fake_unsupported_cls],
        ):
            backend, k_cls = select_mxfp4_moe_backend(config)

        assert backend == Mxfp4MoeBackend.XPU
        assert k_cls is None

    @patch("vllm.model_executor.layers.fused_moe.oracle.mxfp4.current_platform")
    @patch(
        "vllm.model_executor.layers.fused_moe.oracle.mxfp4.has_triton_kernels",
        return_value=True,
    )
    @patch("vllm.model_executor.layers.fused_moe.oracle.mxfp4.envs")
    def test_monolithic_fallback_to_modular_for_dp_ep(
        self, mock_envs, mock_triton, mock_platform
    ):
        """With DP+EP, monolithic should reject and modular should accept
        within the same backend's kernel class list."""
        mock_platform.get_device_capability.return_value = (10, 0)
        mock_platform.is_cuda.return_value = True
        mock_platform.is_rocm.return_value = False
        mock_platform.is_xpu.return_value = False
        mock_envs.is_set.return_value = False
        mock_envs.VLLM_MXFP4_USE_MARLIN = None

        config = self._make_config(dp_size=2, use_ep=True, ep_size=2)

        monolithic_cls = MagicMock()
        monolithic_cls.is_supported_config.return_value = (
            False,
            "monolithic rejects DP+EP",
        )
        modular_cls = MagicMock()
        modular_cls.is_supported_config.return_value = (True, None)

        # FLASHINFER_TRTLLM returns [monolithic, modular]
        def mock_backend_to_kernel_cls(backend):
            if backend == Mxfp4MoeBackend.FLASHINFER_TRTLLM_MXFP4_BF16:
                return [monolithic_cls, modular_cls]
            unsupported = MagicMock()
            unsupported.is_supported_config.return_value = (False, "no")
            return [unsupported]

        with patch(
            "vllm.model_executor.layers.fused_moe.oracle.mxfp4.backend_to_kernel_cls",
            side_effect=mock_backend_to_kernel_cls,
        ):
            backend, k_cls = select_mxfp4_moe_backend(config)

        assert backend == Mxfp4MoeBackend.FLASHINFER_TRTLLM_MXFP4_BF16
        assert k_cls is modular_cls

    @patch("vllm.model_executor.layers.fused_moe.oracle.mxfp4.current_platform")
    @patch(
        "vllm.model_executor.layers.fused_moe.oracle.mxfp4.has_triton_kernels",
        return_value=True,
    )
    @patch("vllm.model_executor.layers.fused_moe.oracle.mxfp4.envs")
    def test_env_var_mxfp8_trtllm_bypasses_loop(
        self, mock_envs, mock_triton, mock_platform
    ):
        """VLLM_USE_FLASHINFER_MOE_MXFP4_MXFP8=1 should force TRTLLM MXFP8
        via _return_or_raise, not the for-loop."""
        mock_platform.get_device_capability.return_value = (10, 0)
        mock_platform.is_cuda.return_value = True
        mock_platform.is_rocm.return_value = False
        mock_platform.is_xpu.return_value = False

        def is_set_side_effect(k):
            return k == "VLLM_USE_FLASHINFER_MOE_MXFP4_MXFP8"

        mock_envs.is_set.side_effect = is_set_side_effect
        mock_envs.VLLM_USE_FLASHINFER_MOE_MXFP4_MXFP8 = True
        mock_envs.VLLM_USE_FLASHINFER_MOE_MXFP4_BF16 = None
        mock_envs.VLLM_MXFP4_USE_MARLIN = None

        config = self._make_config()

        fake_supported_cls = MagicMock()
        fake_supported_cls.is_supported_config.return_value = (True, None)

        def mock_backend_to_kernel_cls(backend):
            return [fake_supported_cls]

        with patch(
            "vllm.model_executor.layers.fused_moe.oracle.mxfp4.backend_to_kernel_cls",
            side_effect=mock_backend_to_kernel_cls,
        ):
            backend, k_cls = select_mxfp4_moe_backend(config)

        assert backend == Mxfp4MoeBackend.FLASHINFER_TRTLLM_MXFP4_MXFP8

    @patch("vllm.model_executor.layers.fused_moe.oracle.mxfp4.current_platform")
    @patch(
        "vllm.model_executor.layers.fused_moe.oracle.mxfp4.has_triton_kernels",
        return_value=True,
    )
    @patch("vllm.model_executor.layers.fused_moe.oracle.mxfp4.envs")
    def test_env_var_mxfp8_cutlass_bypasses_loop(
        self, mock_envs, mock_triton, mock_platform
    ):
        """VLLM_USE_FLASHINFER_MOE_MXFP4_MXFP8_CUTLASS=1 should force
        CUTLASS MXFP8 via _return_or_raise."""
        mock_platform.get_device_capability.return_value = (9, 0)
        mock_platform.is_cuda.return_value = True
        mock_platform.is_rocm.return_value = False
        mock_platform.is_xpu.return_value = False

        def is_set_side_effect(k):
            return k == "VLLM_USE_FLASHINFER_MOE_MXFP4_MXFP8_CUTLASS"

        mock_envs.is_set.side_effect = is_set_side_effect
        mock_envs.VLLM_USE_FLASHINFER_MOE_MXFP4_MXFP8_CUTLASS = True
        mock_envs.VLLM_USE_FLASHINFER_MOE_MXFP4_BF16 = None
        mock_envs.VLLM_USE_FLASHINFER_MOE_MXFP4_MXFP8 = None
        mock_envs.VLLM_MXFP4_USE_MARLIN = None

        config = self._make_config()

        fake_supported_cls = MagicMock()
        fake_supported_cls.is_supported_config.return_value = (True, None)

        def mock_backend_to_kernel_cls(backend):
            return [fake_supported_cls]

        with patch(
            "vllm.model_executor.layers.fused_moe.oracle.mxfp4.backend_to_kernel_cls",
            side_effect=mock_backend_to_kernel_cls,
        ):
            backend, k_cls = select_mxfp4_moe_backend(config)

        assert backend == Mxfp4MoeBackend.FLASHINFER_CUTLASS_MXFP4_MXFP8
