# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for Helion linear backends.

Run `pytest tests/model_executor/kernels/linear/test_helion_linear_backend.py`.
"""

import math
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

from vllm.utils.import_utils import has_helion

if not has_helion():
    pytest.skip(
        "Helion is not installed. Install with: pip install vllm[helion]",
        allow_module_level=True,
    )

from tests.kernels.helion.utils import skip_if_platform_unsupported
from tests.kernels.utils import to_fp8, to_int8
from vllm import _custom_ops as ops
from vllm.config import CUDAGraphMode
from vllm.kernels.helion.case_key import CaseKey
from vllm.kernels.helion.ops.block_scaled_mm import baseline as block_scaled_mm_baseline
from vllm.kernels.helion.ops.scaled_mm import baseline as scaled_mm_baseline
from vllm.model_executor.kernels.linear.scaled_mm.cutlass import (
    CutlassFP8ScaledMMLinearKernel,
)
from vllm.model_executor.kernels.linear.scaled_mm.helion import (
    HelionFP8BlockScaledMMLinearKernel,
    HelionFP8ScaledMMLinearKernel,
    HelionINT8ScaledMMLinearKernel,
)
from vllm.model_executor.kernels.linear.scaled_mm.ScaledMMLinearKernel import (
    FP8ScaledMMLinearLayerConfig,
    Int8ScaledMMLinearLayerConfig,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    GroupShape,
    QuantKey,
    ScaleDesc,
    kFp8Dynamic128Sym,
    kFp8Static128BlockSym,
    kFp8StaticTensorSym,
)
from vllm.platforms import current_platform
from vllm.utils.torch_utils import set_random_seed

INT8_PARAM_NAMES = [
    "weight",
    "weight_scale",
    "input_scale",
    "input_zero_point",
    "azp_adj",
]


def _make_fp8_config(K: int, N: int) -> FP8ScaledMMLinearLayerConfig:
    # weight_shape is (N, K) == (output_size, input_size); can_implement
    # unpacks it as ``N, K = c.weight_shape``.
    return FP8ScaledMMLinearLayerConfig(
        weight_quant_key=kFp8StaticTensorSym,
        activation_quant_key=kFp8StaticTensorSym,
        weight_shape=(N, K),
        input_dtype=current_platform.fp8_dtype(),
        out_dtype=torch.bfloat16,
    )


@contextmanager
def _patch_can_implement_env(
    config_keys: list[CaseKey],
    capture_sizes: list[int],
    max_capture_size: int,
    disabled: bool = False,
    disabled_reason: str | None = None,
):
    """Patch the globals ``can_implement`` reads: the scaled_mm wrapper and
    the current vLLM compilation config."""
    mock_scaled_mm = MagicMock()
    mock_scaled_mm._disabled = disabled
    mock_scaled_mm._disabled_reason = disabled_reason
    mock_scaled_mm.get_configured_op.return_value.configs = {
        key: None for key in config_keys
    }

    mock_compilation = MagicMock()
    mock_compilation.max_cudagraph_capture_size = max_capture_size
    mock_compilation.cudagraph_capture_sizes = capture_sizes
    mock_vllm_config = MagicMock()
    mock_vllm_config.compilation_config = mock_compilation

    with (
        patch(
            "vllm.kernels.helion.ops.scaled_mm.scaled_mm",
            mock_scaled_mm,
        ),
        patch(
            "vllm.model_executor.kernels.linear.scaled_mm.helion"
            ".get_current_vllm_config",
            return_value=mock_vllm_config,
        ),
    ):
        yield


@contextmanager
def _patch_is_supported_env(
    cudagraph_mode: CUDAGraphMode = CUDAGraphMode.FULL,
    max_capture_size: int = 128,
):
    """Patch the platform gates ``is_supported`` checks so it reaches the
    later branches on any host."""
    mock_compilation = MagicMock()
    mock_compilation.cudagraph_mode = cudagraph_mode
    mock_compilation.max_cudagraph_capture_size = max_capture_size
    mock_vllm_config = MagicMock()
    mock_vllm_config.compilation_config = mock_compilation

    helion = "vllm.model_executor.kernels.linear.scaled_mm.helion"
    with (
        patch(f"{helion}.has_helion", return_value=True),
        patch(f"{helion}.current_platform.is_cuda", return_value=True),
        patch(f"{helion}.current_platform.is_device_capability", return_value=True),
        patch(f"{helion}.get_current_vllm_config", return_value=mock_vllm_config),
    ):
        yield


class TestHelionFP8ScaledMMLinearKernel:
    # M values Helion is dispatched for are cudagraph_capture_sizes capped at
    # HELION_SCALED_MM_MAX_NUM_TOKENS (=32).
    CAPTURE_SIZES = [1, 2, 4, 8, 16, 24, 32, 64, 128]
    COVERED_M = [1, 2, 4, 8, 16, 24, 32]

    def _keys_for(self, K: int, N: int, m_values: list[int]) -> list[CaseKey]:
        in_dtype = str(current_platform.fp8_dtype())
        return [
            CaseKey({"K": K, "N": N, "M": m, "in_dtype": in_dtype}) for m in m_values
        ]

    @pytest.mark.cpu_test
    def test_is_supported(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("VLLM_BATCH_INVARIANT", "0")
        with _patch_is_supported_env():
            is_supported, reason = HelionFP8ScaledMMLinearKernel.is_supported()
        assert is_supported, reason
        assert reason is None

    @pytest.mark.cpu_test
    def test_not_supported_without_helion(self):
        helion = "vllm.model_executor.kernels.linear.scaled_mm.helion"
        with patch(f"{helion}.has_helion", return_value=False):
            is_supported, reason = HelionFP8ScaledMMLinearKernel.is_supported()
        assert not is_supported
        assert reason is not None
        assert "helion" in reason

    @pytest.mark.cpu_test
    def test_not_supported_on_non_cuda(self):
        helion = "vllm.model_executor.kernels.linear.scaled_mm.helion"
        with (
            patch(f"{helion}.has_helion", return_value=True),
            patch(f"{helion}.current_platform.is_cuda", return_value=False),
        ):
            is_supported, reason = HelionFP8ScaledMMLinearKernel.is_supported()
        assert not is_supported
        assert reason is not None
        assert "CUDA" in reason

    @pytest.mark.cpu_test
    def test_not_supported_on_non_sm90(self):
        helion = "vllm.model_executor.kernels.linear.scaled_mm.helion"
        with (
            patch(f"{helion}.has_helion", return_value=True),
            patch(f"{helion}.current_platform.is_cuda", return_value=True),
            patch(
                f"{helion}.current_platform.is_device_capability", return_value=False
            ),
        ):
            is_supported, reason = HelionFP8ScaledMMLinearKernel.is_supported()
        assert not is_supported
        assert reason is not None
        assert "SM90" in reason

    @pytest.mark.cpu_test
    def test_not_supported_for_batch_invariant(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("VLLM_BATCH_INVARIANT", "1")
        with _patch_is_supported_env():
            is_supported, reason = HelionFP8ScaledMMLinearKernel.is_supported()
        assert not is_supported
        assert reason is not None
        assert "batch invariant" in reason

    @pytest.mark.cpu_test
    def test_not_supported_without_cuda_graph(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("VLLM_BATCH_INVARIANT", "0")
        with _patch_is_supported_env(cudagraph_mode=CUDAGraphMode.NONE):
            is_supported, reason = HelionFP8ScaledMMLinearKernel.is_supported()
        assert not is_supported
        assert reason is not None
        assert "CUDA Graph" in reason

    @pytest.mark.cpu_test
    def test_can_implement(self):
        K, N = 4096, 6144
        keys = self._keys_for(K, N, self.COVERED_M)
        with _patch_can_implement_env(keys, self.CAPTURE_SIZES, 128):
            can_impl, reason = HelionFP8ScaledMMLinearKernel.can_implement(
                _make_fp8_config(K, N)
            )
        assert can_impl, reason
        assert reason is None

    @pytest.mark.cpu_test
    def test_cannot_implement_missing_one_m_config(self):
        K, N = 4096, 6144
        # Drop M=16 so coverage is incomplete.
        keys = self._keys_for(K, N, [m for m in self.COVERED_M if m != 16])
        with _patch_can_implement_env(keys, self.CAPTURE_SIZES, 128):
            can_impl, reason = HelionFP8ScaledMMLinearKernel.can_implement(
                _make_fp8_config(K, N)
            )
        assert not can_impl
        assert reason is not None
        assert "no pre-tuned config" in reason
        assert "16" in reason

    @pytest.mark.cpu_test
    def test_cannot_implement_missing_config_for_other_shape(self):
        # Configs exist, but for a different (K, N) than the layer needs.
        keys = self._keys_for(2048, 2048, self.COVERED_M)
        with _patch_can_implement_env(keys, self.CAPTURE_SIZES, 128):
            can_impl, reason = HelionFP8ScaledMMLinearKernel.can_implement(
                _make_fp8_config(4096, 6144)
            )
        assert not can_impl
        assert reason is not None
        assert "K=4096" in reason and "N=6144" in reason

    @pytest.mark.cpu_test
    def test_can_implement_m_sizes_capped_by_helion_max(self):
        # Configs only for M <= 8, but capture sizes go up to 128. With
        # max_cudagraph_capture_size=8 the covered M range is [1, 2, 4, 8].
        K, N = 4096, 6144
        keys = self._keys_for(K, N, [1, 2, 4, 8])
        with _patch_can_implement_env(keys, [1, 2, 4, 8], 8):
            can_impl, reason = HelionFP8ScaledMMLinearKernel.can_implement(
                _make_fp8_config(K, N)
            )
        assert can_impl, reason

    @pytest.mark.cpu_test
    def test_cannot_implement_disabled_op(self):
        with _patch_can_implement_env(
            [], [1, 2, 4], 4, disabled=True, disabled_reason="no configs for platform"
        ):
            can_impl, reason = HelionFP8ScaledMMLinearKernel.can_implement(
                _make_fp8_config(4096, 6144)
            )
        assert not can_impl
        assert reason is not None
        assert "disabled" in reason
        assert "no configs for platform" in reason

    @pytest.mark.cpu_test
    def test_can_implement_skip_config_check_env(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("VLLM_HELION_LINEAR_SKIP_CONFIG_CHECK", "1")
        K, N = 4096, 6144
        # No matching configs, but the env var bypasses the coverage check.
        with _patch_can_implement_env([], self.CAPTURE_SIZES, 128):
            can_impl, reason = HelionFP8ScaledMMLinearKernel.can_implement(
                _make_fp8_config(K, N)
            )
        assert can_impl, reason

    @staticmethod
    def _make_apply_kernel(
        logical_output_size: int, helion_max_num_tokens: int
    ) -> HelionFP8ScaledMMLinearKernel:
        kernel = object.__new__(HelionFP8ScaledMMLinearKernel)
        fallback = object.__new__(CutlassFP8ScaledMMLinearKernel)
        fallback.logical_output_size = logical_output_size
        kernel.fallback = fallback
        kernel.helion_max_num_tokens = helion_max_num_tokens
        return kernel

    def _run_apply_scaled_mm(
        self, M: int, N: int, K: int, use_bias: bool, helion_max_num_tokens: int
    ) -> None:
        skip_if_platform_unsupported("scaled_mm")
        if not current_platform.supports_fp8():
            pytest.skip("Platform does not support FP8")
        cap = current_platform.get_device_capability()
        assert cap is not None
        if not torch.ops._C.cutlass_scaled_mm_supports_fp8(cap.to_int()):
            pytest.skip("CUTLASS scaled_mm does not support FP8 on this platform")

        set_random_seed(0)
        in_dtype = current_platform.fp8_dtype()
        out_dtype = torch.bfloat16

        a = to_fp8(0.25 * torch.randn((M, K), device="cuda")).to(in_dtype)
        b = to_fp8(0.25 * torch.randn((N, K), device="cuda")).to(in_dtype).t()
        scale_a = 0.25 * torch.rand((M, 1), device="cuda", dtype=torch.float32)
        scale_b = 0.25 * torch.rand((N, 1), device="cuda", dtype=torch.float32)
        bias = torch.rand((N,), device="cuda", dtype=out_dtype) if use_bias else None

        # baseline uses the original, unpadded operands.
        baseline_out = torch.empty((M, N), dtype=out_dtype, device="cuda")
        scaled_mm_baseline(baseline_out, a, b, scale_a, scale_b, bias)

        # process_weights_after_loading pads the weight (and per-channel weight
        # scale) up to 16-element alignment; mirror that here.
        pad_k = (16 - K % 16) % 16
        pad_n = (16 - N % 16) % 16
        if pad_k > 0 or pad_n > 0:
            b = torch.nn.functional.pad(b.t().contiguous(), (0, pad_k, 0, pad_n)).t()
            if pad_n > 0:
                scale_b = torch.nn.functional.pad(scale_b, (0, 0, 0, pad_n), value=1.0)

        kernel = self._make_apply_kernel(
            logical_output_size=N, helion_max_num_tokens=helion_max_num_tokens
        )
        out = kernel.apply_scaled_mm(
            A=a,
            B=b,
            out_dtype=out_dtype,
            As=scale_a,
            Bs=scale_b,
            bias=bias,
            output_shape=[M, N],
        )

        assert out.shape == (M, N)
        torch.testing.assert_close(out, baseline_out, rtol=1e-1, atol=1e-1)

    @pytest.mark.skipif(
        not current_platform.is_cuda(), reason="apply_scaled_mm requires CUDA"
    )
    @pytest.mark.parametrize("M", [4, 32])
    @pytest.mark.parametrize("N,K", [(256, 128), (496, 256)])
    @pytest.mark.parametrize("use_bias", [True, False])
    def test_apply_scaled_mm_aligned(self, M, N, K, use_bias):
        self._run_apply_scaled_mm(M, N, K, use_bias, helion_max_num_tokens=16)

    @pytest.mark.skipif(
        not current_platform.is_cuda(), reason="apply_scaled_mm requires CUDA"
    )
    @pytest.mark.parametrize("M", [4, 32])
    @pytest.mark.parametrize("N,K", [(255, 513), (100, 200), (1280, 342)])
    @pytest.mark.parametrize("use_bias", [True, False])
    def test_apply_scaled_mm_padded(self, M, N, K, use_bias):
        self._run_apply_scaled_mm(M, N, K, use_bias, helion_max_num_tokens=16)

    @pytest.mark.skipif(
        not current_platform.is_cuda(), reason="apply_scaled_mm requires CUDA"
    )
    @pytest.mark.parametrize("M", [4, 32])
    @pytest.mark.parametrize("N,K", [(255, 513), (100, 200)])
    @pytest.mark.parametrize("use_bias", [True, False])
    def test_apply_scaled_mm_triton_fallback(self, M, N, K, use_bias):
        skip_if_platform_unsupported("scaled_mm")
        if not current_platform.supports_fp8():
            pytest.skip("Platform does not support FP8")

        set_random_seed(0)
        in_dtype = current_platform.fp8_dtype()
        out_dtype = torch.bfloat16

        a = to_fp8(0.25 * torch.randn((M, K), device="cuda")).to(in_dtype)
        b = to_fp8(0.25 * torch.randn((N, K), device="cuda")).to(in_dtype).t()
        scale_a = 0.25 * torch.rand((M, 1), device="cuda", dtype=torch.float32)
        scale_b = 0.25 * torch.rand((N, 1), device="cuda", dtype=torch.float32)
        bias = torch.rand((N,), device="cuda", dtype=out_dtype) if use_bias else None

        baseline_out = torch.empty((M, N), dtype=out_dtype, device="cuda")
        scaled_mm_baseline(baseline_out, a, b, scale_a, scale_b, bias)

        # B is left unpadded (K/N not 16-aligned) -> triton_scaled_mm branch.
        kernel = self._make_apply_kernel(
            logical_output_size=N, helion_max_num_tokens=16
        )
        out = kernel.apply_scaled_mm(
            A=a,
            B=b,
            out_dtype=out_dtype,
            As=scale_a,
            Bs=scale_b,
            bias=bias,
            output_shape=[M, N],
        )

        assert out.shape == (M, N)
        torch.testing.assert_close(out, baseline_out, rtol=1e-1, atol=1e-1)


def _make_int8_config(K: int, N: int) -> Int8ScaledMMLinearLayerConfig:
    # weight_shape is (N, K) == (output_size, input_size); can_implement
    # unpacks it as ``N, K = c.weight_shape``.
    return Int8ScaledMMLinearLayerConfig(
        is_static_input_scheme=False,
        is_channelwise=True,
        input_symmetric=True,
        weight_shape=(N, K),
    )


@contextmanager
def _patch_int8_can_implement_env(
    disabled: bool = False, disabled_reason: str | None = None
):
    mock_scaled_mm = MagicMock()
    mock_scaled_mm._disabled = disabled
    mock_scaled_mm._disabled_reason = disabled_reason
    with patch(
        "vllm.kernels.helion.ops.scaled_mm.scaled_mm",
        mock_scaled_mm,
    ):
        yield


class TestHelionINT8ScaledMMLinearKernel:
    # M values Helion is dispatched for are cudagraph_capture_sizes capped at
    # HELION_SCALED_MM_MAX_NUM_TOKENS (=32).
    CAPTURE_SIZES = [1, 2, 4, 8, 16, 24, 32, 64, 128]
    COVERED_M = [1, 2, 4, 8, 16, 24, 32]

    def _keys_for(self, K: int, N: int, m_values: list[int]) -> list[CaseKey]:
        in_dtype = str(torch.int8)
        return [
            CaseKey({"K": K, "N": N, "M": m, "in_dtype": in_dtype}) for m in m_values
        ]

    @pytest.mark.cpu_test
    def test_is_supported(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("VLLM_BATCH_INVARIANT", "0")
        with _patch_is_supported_env():
            is_supported, reason = HelionINT8ScaledMMLinearKernel.is_supported()
        assert is_supported, reason
        assert reason is None

    @pytest.mark.cpu_test
    def test_not_supported_without_helion(self):
        helion = "vllm.model_executor.kernels.linear.scaled_mm.helion"
        with patch(f"{helion}.has_helion", return_value=False):
            is_supported, reason = HelionINT8ScaledMMLinearKernel.is_supported()
        assert not is_supported
        assert reason is not None
        assert "helion" in reason

    @pytest.mark.cpu_test
    def test_not_supported_on_non_cuda(self):
        helion = "vllm.model_executor.kernels.linear.scaled_mm.helion"
        with (
            patch(f"{helion}.has_helion", return_value=True),
            patch(f"{helion}.current_platform.is_cuda", return_value=False),
        ):
            is_supported, reason = HelionINT8ScaledMMLinearKernel.is_supported()
        assert not is_supported
        assert reason is not None
        assert "CUDA" in reason

    @pytest.mark.cpu_test
    def test_not_supported_on_non_sm90(self):
        helion = "vllm.model_executor.kernels.linear.scaled_mm.helion"
        with (
            patch(f"{helion}.has_helion", return_value=True),
            patch(f"{helion}.current_platform.is_cuda", return_value=True),
            patch(
                f"{helion}.current_platform.is_device_capability", return_value=False
            ),
        ):
            is_supported, reason = HelionINT8ScaledMMLinearKernel.is_supported()
        assert not is_supported
        assert reason is not None
        assert "SM90" in reason

    @pytest.mark.cpu_test
    def test_not_supported_for_batch_invariant(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("VLLM_BATCH_INVARIANT", "1")
        with _patch_is_supported_env():
            is_supported, reason = HelionINT8ScaledMMLinearKernel.is_supported()
        assert not is_supported
        assert reason is not None
        assert "batch invariant" in reason

    @pytest.mark.cpu_test
    def test_not_supported_without_cuda_graph(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("VLLM_BATCH_INVARIANT", "0")
        with _patch_is_supported_env(cudagraph_mode=CUDAGraphMode.NONE):
            is_supported, reason = HelionINT8ScaledMMLinearKernel.is_supported()
        assert not is_supported
        assert reason is not None
        assert "CUDA Graph" in reason

    @pytest.mark.cpu_test
    def test_can_implement(self):
        K, N = 4096, 6144
        keys = self._keys_for(K, N, self.COVERED_M)
        with _patch_can_implement_env(keys, self.CAPTURE_SIZES, 128):
            can_impl, reason = HelionINT8ScaledMMLinearKernel.can_implement(
                _make_int8_config(K, N)
            )
        assert can_impl, reason
        assert reason is None

    @pytest.mark.cpu_test
    def test_cannot_implement_missing_one_m_config(self):
        K, N = 4096, 6144
        # Drop M=16 so coverage is incomplete.
        keys = self._keys_for(K, N, [m for m in self.COVERED_M if m != 16])
        with _patch_can_implement_env(keys, self.CAPTURE_SIZES, 128):
            can_impl, reason = HelionINT8ScaledMMLinearKernel.can_implement(
                _make_int8_config(K, N)
            )
        assert not can_impl
        assert reason is not None
        assert "no pre-tuned config" in reason
        assert "16" in reason

    @pytest.mark.cpu_test
    def test_cannot_implement_missing_fp8_only_config(self):
        # Configs exist for the shape, but only for the fp8 in_dtype.
        K, N = 4096, 6144
        fp8_dtype = str(current_platform.fp8_dtype())
        keys = [
            CaseKey({"K": K, "N": N, "M": m, "in_dtype": fp8_dtype})
            for m in self.COVERED_M
        ]
        with _patch_can_implement_env(keys, self.CAPTURE_SIZES, 128):
            can_impl, reason = HelionINT8ScaledMMLinearKernel.can_implement(
                _make_int8_config(K, N)
            )
        assert not can_impl
        assert reason is not None
        assert "torch.int8" in reason

    @pytest.mark.cpu_test
    def test_cannot_implement_disabled_op(self):
        with _patch_int8_can_implement_env(
            disabled=True, disabled_reason="no configs for platform"
        ):
            can_impl, reason = HelionINT8ScaledMMLinearKernel.can_implement(
                _make_int8_config(4096, 6144)
            )
        assert not can_impl
        assert reason is not None
        assert "disabled" in reason
        assert "no configs for platform" in reason

    @pytest.mark.cpu_test
    def test_can_implement_skip_config_check_env(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("VLLM_HELION_LINEAR_SKIP_CONFIG_CHECK", "1")
        K, N = 4096, 6144
        # No matching configs, but the env var bypasses the coverage check.
        with _patch_can_implement_env([], self.CAPTURE_SIZES, 128):
            can_impl, reason = HelionINT8ScaledMMLinearKernel.can_implement(
                _make_int8_config(K, N)
            )
        assert can_impl, reason

    @staticmethod
    def _make_apply_kernel(
        helion_max_num_tokens: int,
    ) -> HelionINT8ScaledMMLinearKernel:
        kernel = object.__new__(HelionINT8ScaledMMLinearKernel)
        kernel.layer_param_names = INT8_PARAM_NAMES
        kernel.helion_max_num_tokens = helion_max_num_tokens
        return kernel

    @staticmethod
    def _layer(
        w_q: torch.Tensor,
        w_s: torch.Tensor,
        i_s: torch.Tensor | None = None,
        i_zp: torch.Tensor | None = None,
        azp_adj: torch.Tensor | None = None,
    ) -> SimpleNamespace:
        return SimpleNamespace(
            weight=w_q,
            weight_scale=w_s,
            input_scale=i_s,
            input_zero_point=i_zp,
            azp_adj=azp_adj,
        )

    def _run_symmetric(self, M: int, N: int, K: int, use_bias: bool) -> None:
        skip_if_platform_unsupported("scaled_mm")

        # M above the Helion threshold and 16-aligned weight -> CUTLASS fallback.
        aligned = K % 16 == 0 and N % 16 == 0
        if M > 16 and aligned:
            pytest.skip("CUTLASS int8 scaled_mm unsupported")

        set_random_seed(0)
        out_dtype = torch.bfloat16

        x = torch.randn((M, K), device="cuda", dtype=out_dtype)
        # channelwise weight, dynamic per-token activation (symmetric).
        # Weight scale is [N, 1] to match the checkpoint's channelwise layout.
        w_q = to_int8(torch.randn((N, K), device="cuda") * 5).t()
        w_s = torch.rand((N, 1), device="cuda", dtype=torch.float32) / 10
        bias = torch.rand((N,), device="cuda", dtype=out_dtype) if use_bias else None

        kernel = self._make_apply_kernel(helion_max_num_tokens=16)
        layer = self._layer(w_q, w_s)
        out = kernel.apply_weights(layer, x, bias)

        x_q, x_s, _ = ops.scaled_int8_quant(x.contiguous(), None, None, symmetric=True)
        expected = torch.empty((M, N), dtype=out_dtype, device="cuda")
        scaled_mm_baseline(expected, x_q, w_q, x_s, w_s, bias)

        assert out.shape == (M, N)
        torch.testing.assert_close(out, expected, rtol=1e-1, atol=1e0)

    @pytest.mark.skipif(
        not current_platform.is_cuda(), reason="apply_weights requires CUDA"
    )
    @pytest.mark.parametrize("M", [4, 32])
    @pytest.mark.parametrize("N,K", [(256, 128), (496, 256)])
    @pytest.mark.parametrize("use_bias", [True, False])
    def test_apply_weights_symmetric_aligned(self, M, N, K, use_bias):
        self._run_symmetric(M, N, K, use_bias)

    @pytest.mark.skipif(
        not current_platform.is_cuda(), reason="apply_weights requires CUDA"
    )
    @pytest.mark.parametrize("M", [4, 32])
    @pytest.mark.parametrize("N,K", [(255, 513), (100, 200)])
    @pytest.mark.parametrize("use_bias", [True, False])
    def test_apply_weights_symmetric_triton_fallback(self, M, N, K, use_bias):
        # N or K not 16-aligned -> triton_scaled_mm branch.
        self._run_symmetric(M, N, K, use_bias)

    @pytest.mark.skipif(
        not current_platform.is_cuda(), reason="apply_weights requires CUDA"
    )
    @pytest.mark.parametrize("M", [4, 32])
    @pytest.mark.parametrize("N,K", [(256, 128)])
    @pytest.mark.parametrize("use_bias", [True, False])
    def test_apply_weights_asymmetric_uses_cutlass(self, M, N, K, use_bias):
        # Helion is never used for the asymmetric (azp) case; it must route
        # through ops.cutlass_scaled_mm_azp and skip the Helion hybrid op.
        skip_if_platform_unsupported("scaled_mm")

        set_random_seed(0)
        out_dtype = torch.bfloat16

        x = torch.randn((M, K), device="cuda", dtype=out_dtype)
        w_q = to_int8(torch.randn((N, K), device="cuda") * 5).t()
        w_s = torch.rand((1, N), device="cuda", dtype=torch.float32) / 10
        # dynamic per-token asymmetric: azp_adj set, input_zero_point None.
        azp_adj = w_q.sum(dim=0, keepdim=True, dtype=torch.int32)
        bias = torch.rand((N,), device="cuda", dtype=out_dtype) if use_bias else None

        kernel = self._make_apply_kernel(helion_max_num_tokens=16)
        layer = self._layer(w_q, w_s, azp_adj=azp_adj)

        sentinel = torch.zeros((M, N), device="cuda", dtype=out_dtype)
        with (
            patch.object(
                ops, "cutlass_scaled_mm_azp", return_value=sentinel
            ) as mock_azp,
            patch.object(
                torch.ops._C, "helion_cutlass_hybrid_scaled_mm"
            ) as mock_helion,
        ):
            out = kernel.apply_weights(layer, x, bias)

        mock_azp.assert_called_once()
        mock_helion.assert_not_called()
        assert out is sentinel


def _make_fp8_block_config(
    K: int,
    N: int,
    out_dtype: torch.dtype = torch.bfloat16,
    act_quant_key: QuantKey = kFp8Dynamic128Sym,
) -> FP8ScaledMMLinearLayerConfig:
    # weight_shape is (N, K) == (output_size, input_size); can_implement
    # unpacks it as ``N, K = config.weight_shape``.
    return FP8ScaledMMLinearLayerConfig(
        weight_quant_key=kFp8Static128BlockSym,
        activation_quant_key=act_quant_key,
        weight_shape=(N, K),
        input_dtype=current_platform.fp8_dtype(),
        out_dtype=out_dtype,
    )


@contextmanager
def _patch_block_can_implement_env(
    config_keys: list[CaseKey],
    capture_sizes: list[int],
    max_capture_size: int,
    disabled: bool = False,
    disabled_reason: str | None = None,
    auto_disable_deep_gemm: bool = False,
    use_deepgemm: bool = True,
):
    """Patch the globals ``HelionFP8BlockScaledMMLinearKernel.can_implement``
    reads: the block_scaled_mm wrapper, the current vLLM config, and the
    DeepGEMM suitability helpers."""
    mock_block = MagicMock()
    mock_block._disabled = disabled
    mock_block._disabled_reason = disabled_reason
    mock_block.get_configured_op.return_value.configs = {
        key: None for key in config_keys
    }

    mock_compilation = MagicMock()
    mock_compilation.max_cudagraph_capture_size = max_capture_size
    mock_compilation.cudagraph_capture_sizes = capture_sizes
    mock_vllm_config = MagicMock()
    mock_vllm_config.compilation_config = mock_compilation
    mock_vllm_config.model_config.hf_text_config.model_type = "test_model"

    helion = "vllm.model_executor.kernels.linear.scaled_mm.helion"
    with (
        patch(
            "vllm.kernels.helion.ops.block_scaled_mm.block_scaled_mm",
            mock_block,
        ),
        patch(f"{helion}.get_current_vllm_config", return_value=mock_vllm_config),
        patch(
            f"{helion}.should_auto_disable_deep_gemm",
            return_value=auto_disable_deep_gemm,
        ),
        patch(
            f"{helion}.should_use_deepgemm_for_fp8_linear",
            return_value=use_deepgemm,
        ),
    ):
        yield


@contextmanager
def _patch_block_is_supported_env(
    cudagraph_mode: CUDAGraphMode = CUDAGraphMode.FULL,
    max_capture_size: int = 128,
):
    """Patch the platform gates ``is_supported`` checks so it reaches the
    later branches on any host. Unlike the FP8/INT8 kernels, the block kernel
    also gates on DeepGEMM availability."""
    mock_compilation = MagicMock()
    mock_compilation.cudagraph_mode = cudagraph_mode
    mock_compilation.max_cudagraph_capture_size = max_capture_size
    mock_vllm_config = MagicMock()
    mock_vllm_config.compilation_config = mock_compilation

    helion = "vllm.model_executor.kernels.linear.scaled_mm.helion"
    with (
        patch(f"{helion}.has_helion", return_value=True),
        patch(f"{helion}.has_deep_gemm", return_value=True),
        patch(f"{helion}.current_platform.is_cuda", return_value=True),
        patch(f"{helion}.current_platform.is_device_capability", return_value=True),
        patch(f"{helion}.get_current_vllm_config", return_value=mock_vllm_config),
    ):
        yield


class TestHelionFP8BlockScaledMMLinearKernel:
    # M values Helion is dispatched for are cudagraph_capture_sizes capped at
    # HELION_BLOCK_SCALED_MM_MAX_NUM_TOKENS (=32).
    CAPTURE_SIZES = [1, 2, 4, 8, 16, 24, 32, 64, 128]
    COVERED_M = [1, 2, 4, 8, 16, 24, 32]

    def _keys_for(self, K: int, N: int, m_values: list[int]) -> list[CaseKey]:
        in_dtype = str(current_platform.fp8_dtype())
        return [
            CaseKey({"K": K, "N": N, "M": m, "in_dtype": in_dtype}) for m in m_values
        ]

    @pytest.mark.cpu_test
    def test_is_supported(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("VLLM_USE_DEEP_GEMM", "1")
        monkeypatch.setenv("VLLM_BATCH_INVARIANT", "0")
        with _patch_block_is_supported_env():
            is_supported, reason = HelionFP8BlockScaledMMLinearKernel.is_supported()
        assert is_supported, reason
        assert reason is None

    @pytest.mark.cpu_test
    def test_not_supported_without_helion(self):
        helion = "vllm.model_executor.kernels.linear.scaled_mm.helion"
        with patch(f"{helion}.has_helion", return_value=False):
            is_supported, reason = HelionFP8BlockScaledMMLinearKernel.is_supported()
        assert not is_supported
        assert reason is not None
        assert "helion" in reason

    @pytest.mark.cpu_test
    def test_not_supported_on_non_cuda(self):
        helion = "vllm.model_executor.kernels.linear.scaled_mm.helion"
        with (
            patch(f"{helion}.has_helion", return_value=True),
            patch(f"{helion}.current_platform.is_cuda", return_value=False),
        ):
            is_supported, reason = HelionFP8BlockScaledMMLinearKernel.is_supported()
        assert not is_supported
        assert reason is not None
        assert "CUDA" in reason

    @pytest.mark.cpu_test
    def test_not_supported_without_deep_gemm(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("VLLM_USE_DEEP_GEMM", "0")
        with _patch_block_is_supported_env():
            is_supported, reason = HelionFP8BlockScaledMMLinearKernel.is_supported()
        assert not is_supported
        assert reason is not None
        assert "DeepGEMM" in reason

    @pytest.mark.cpu_test
    def test_not_supported_on_non_sm90(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("VLLM_USE_DEEP_GEMM", "1")
        helion = "vllm.model_executor.kernels.linear.scaled_mm.helion"
        with (
            _patch_block_is_supported_env(),
            patch(
                f"{helion}.current_platform.is_device_capability", return_value=False
            ),
        ):
            is_supported, reason = HelionFP8BlockScaledMMLinearKernel.is_supported()
        assert not is_supported
        assert reason is not None
        assert "SM90" in reason

    @pytest.mark.cpu_test
    def test_not_supported_for_batch_invariant(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("VLLM_USE_DEEP_GEMM", "1")
        monkeypatch.setenv("VLLM_BATCH_INVARIANT", "1")
        with _patch_block_is_supported_env():
            is_supported, reason = HelionFP8BlockScaledMMLinearKernel.is_supported()
        assert not is_supported
        assert reason is not None
        assert "batch invariant" in reason

    @pytest.mark.cpu_test
    def test_not_supported_without_cuda_graph(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("VLLM_USE_DEEP_GEMM", "1")
        monkeypatch.setenv("VLLM_BATCH_INVARIANT", "0")
        with _patch_block_is_supported_env(cudagraph_mode=CUDAGraphMode.NONE):
            is_supported, reason = HelionFP8BlockScaledMMLinearKernel.is_supported()
        assert not is_supported
        assert reason is not None
        assert "CUDA Graph" in reason

    @pytest.mark.cpu_test
    def test_can_implement(self):
        K, N = 4096, 6144
        keys = self._keys_for(K, N, self.COVERED_M)
        with _patch_block_can_implement_env(keys, self.CAPTURE_SIZES, 128):
            can_impl, reason = HelionFP8BlockScaledMMLinearKernel.can_implement(
                _make_fp8_block_config(K, N)
            )
        assert can_impl, reason
        assert reason is None

    @pytest.mark.cpu_test
    def test_cannot_implement_missing_one_m_config(self):
        K, N = 4096, 6144
        # Drop M=16 so coverage is incomplete.
        keys = self._keys_for(K, N, [m for m in self.COVERED_M if m != 16])
        with _patch_block_can_implement_env(keys, self.CAPTURE_SIZES, 128):
            can_impl, reason = HelionFP8BlockScaledMMLinearKernel.can_implement(
                _make_fp8_block_config(K, N)
            )
        assert not can_impl
        assert reason is not None
        assert "no pre-tuned config" in reason
        assert "16" in reason

    @pytest.mark.cpu_test
    def test_cannot_implement_missing_config_for_other_shape(self):
        # Configs exist, but for a different (K, N) than the layer needs.
        keys = self._keys_for(2048, 2048, self.COVERED_M)
        with _patch_block_can_implement_env(keys, self.CAPTURE_SIZES, 128):
            can_impl, reason = HelionFP8BlockScaledMMLinearKernel.can_implement(
                _make_fp8_block_config(4096, 6144)
            )
        assert not can_impl
        assert reason is not None
        assert "K=4096" in reason and "N=6144" in reason

    @pytest.mark.cpu_test
    def test_can_implement_m_sizes_capped_by_helion_max(self):
        # Configs only for M <= 8, but capture sizes go up to 128. With
        # max_cudagraph_capture_size=8 the covered M range is [1, 2, 4, 8].
        K, N = 4096, 6144
        keys = self._keys_for(K, N, [1, 2, 4, 8])
        with _patch_block_can_implement_env(keys, [1, 2, 4, 8], 8):
            can_impl, reason = HelionFP8BlockScaledMMLinearKernel.can_implement(
                _make_fp8_block_config(K, N)
            )
        assert can_impl, reason

    @pytest.mark.cpu_test
    def test_can_implement_skip_config_check_env(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("VLLM_HELION_LINEAR_SKIP_CONFIG_CHECK", "1")
        K, N = 4096, 6144
        # No matching configs, but the env var bypasses the coverage check.
        with _patch_block_can_implement_env([], self.CAPTURE_SIZES, 128):
            can_impl, reason = HelionFP8BlockScaledMMLinearKernel.can_implement(
                _make_fp8_block_config(K, N)
            )
        assert can_impl, reason

    @pytest.mark.cpu_test
    def test_cannot_implement_disabled_op(self):
        with _patch_block_can_implement_env(
            [], [1, 2, 4], 4, disabled=True, disabled_reason="no configs for platform"
        ):
            can_impl, reason = HelionFP8BlockScaledMMLinearKernel.can_implement(
                _make_fp8_block_config(4096, 6144)
            )
        assert not can_impl
        assert reason is not None
        assert "disabled" in reason
        assert "no configs for platform" in reason

    @pytest.mark.cpu_test
    def test_cannot_implement_non_bf16_output(self):
        K, N = 4096, 6144
        keys = self._keys_for(K, N, self.COVERED_M)
        with _patch_block_can_implement_env(keys, self.CAPTURE_SIZES, 128):
            can_impl, reason = HelionFP8BlockScaledMMLinearKernel.can_implement(
                _make_fp8_block_config(K, N, out_dtype=torch.float16)
            )
        assert not can_impl
        assert reason is not None
        assert "bfloat16" in reason

    @pytest.mark.cpu_test
    def test_cannot_implement_wrong_activation_group_shape(self):
        K, N = 4096, 6144
        keys = self._keys_for(K, N, self.COVERED_M)
        # Per-token activation with a 64-wide group is unsupported; only
        # group_shape=(1,128) is allowed.
        act_key = QuantKey(
            current_platform.fp8_dtype(),
            ScaleDesc(torch.float32, False, GroupShape(1, 64)),
        )
        with _patch_block_can_implement_env(keys, self.CAPTURE_SIZES, 128):
            can_impl, reason = HelionFP8BlockScaledMMLinearKernel.can_implement(
                _make_fp8_block_config(K, N, act_quant_key=act_key)
            )
        assert not can_impl
        assert reason is not None
        assert "group_shape=(1,128)" in reason

    @pytest.mark.cpu_test
    def test_cannot_implement_auto_disabled_deep_gemm_model(self):
        K, N = 4096, 6144
        keys = self._keys_for(K, N, self.COVERED_M)
        with _patch_block_can_implement_env(
            keys, self.CAPTURE_SIZES, 128, auto_disable_deep_gemm=True
        ):
            can_impl, reason = HelionFP8BlockScaledMMLinearKernel.can_implement(
                _make_fp8_block_config(K, N)
            )
        assert not can_impl
        assert reason is not None
        assert "DeepGEMM is not supported" in reason

    @pytest.mark.cpu_test
    def test_cannot_implement_unsupported_deepgemm_metadata(self):
        K, N = 4096, 6144
        keys = self._keys_for(K, N, self.COVERED_M)
        with _patch_block_can_implement_env(
            keys, self.CAPTURE_SIZES, 128, use_deepgemm=False
        ):
            can_impl, reason = HelionFP8BlockScaledMMLinearKernel.can_implement(
                _make_fp8_block_config(K, N)
            )
        assert not can_impl
        assert reason is not None
        assert "metadata is not supported" in reason

    @staticmethod
    def _make_apply_kernel(
        helion_max_num_tokens: int, use_deep_gemm_e8m0: bool
    ) -> HelionFP8BlockScaledMMLinearKernel:
        kernel = object.__new__(HelionFP8BlockScaledMMLinearKernel)
        kernel.weight_group_shape = GroupShape(128, 128)
        kernel.fallback = SimpleNamespace(use_deep_gemm_e8m0=use_deep_gemm_e8m0)
        kernel.helion_max_num_tokens = helion_max_num_tokens
        return kernel

    def _run_apply_block_scaled_mm(self, M: int, N: int, K: int, use_helion: bool):
        skip_if_platform_unsupported("block_scaled_mm")
        from vllm.model_executor.layers.quantization.utils.fp8_utils import (
            per_token_group_quant_fp8,
        )
        from vllm.utils.deep_gemm import _ceil_to_ue8m0, is_deep_gemm_e8m0_used

        set_random_seed(0)
        use_deep_gemm_e8m0 = is_deep_gemm_e8m0_used()
        in_dtype = current_platform.fp8_dtype()

        scale = 1.0 / math.sqrt(K)
        x = scale * (0.5 + torch.rand((M, K), device="cuda", dtype=torch.bfloat16))
        weight = (
            scale * (0.5 + torch.rand((N, K), device="cuda", dtype=torch.float32))
        ).to(in_dtype)
        weight_scale = 0.5 + torch.rand(
            (N // 128, K // 128), device="cuda", dtype=torch.float32
        )
        if use_deep_gemm_e8m0:
            weight_scale = _ceil_to_ue8m0(weight_scale)
        weight_scale = weight_scale.t().contiguous().t()

        # helion_max_num_tokens gates the dispatch: M <= threshold uses Helion,
        # otherwise DeepGEMM. Force whichever branch we want to exercise.
        helion_max_num_tokens = M if use_helion else 0
        kernel = self._make_apply_kernel(helion_max_num_tokens, use_deep_gemm_e8m0)
        # As is unused
        placeholder = x.new_empty(1)
        out = kernel.apply_block_scaled_mm(
            A=x, B=weight, As=placeholder, Bs=weight_scale
        )

        x_q, x_s = per_token_group_quant_fp8(
            x,
            group_size=kernel.weight_group_shape.col,
            column_major_scales=True,
            use_ue8m0=use_deep_gemm_e8m0,
        )
        expected = torch.empty((M, N), dtype=torch.bfloat16, device="cuda")
        block_scaled_mm_baseline(expected, x_q, weight.t(), x_s, weight_scale.t())

        assert out.shape == (M, N)
        assert out.dtype == torch.bfloat16
        torch.testing.assert_close(out, expected, rtol=1e-1, atol=1e-1)

    @pytest.mark.skipif(
        not current_platform.is_cuda(), reason="apply_block_scaled_mm requires CUDA"
    )
    @pytest.mark.parametrize("use_helion", [True, False])
    @pytest.mark.parametrize("M", [1, 16, 32])
    @pytest.mark.parametrize("N,K", [(256, 128), (512, 1024), (4096, 4096)])
    def test_apply_block_scaled_mm(self, M, N, K, use_helion):
        self._run_apply_block_scaled_mm(M, N, K, use_helion)
