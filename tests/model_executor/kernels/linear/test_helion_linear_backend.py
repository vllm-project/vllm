# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for Helion linear backends.

Run `pytest tests/model_executor/kernels/linear/test_helion_linear_backend.py`.
"""

from contextlib import contextmanager
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
from tests.kernels.utils import to_fp8
from vllm.kernels.helion.case_key import CaseKey
from vllm.kernels.helion.ops.scaled_mm import baseline
from vllm.model_executor.kernels.linear.scaled_mm.cutlass import (
    CutlassFP8ScaledMMLinearKernel,
)
from vllm.model_executor.kernels.linear.scaled_mm.helion import (
    HelionFP8ScaledMMLinearKernel,
)
from vllm.model_executor.kernels.linear.scaled_mm.ScaledMMLinearKernel import (
    FP8ScaledMMLinearLayerConfig,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    kFp8StaticTensorSym,
)
from vllm.platforms import current_platform
from vllm.utils.torch_utils import set_random_seed


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


class TestHelionFP8ScaledMMLinearKernel:
    # M values Helion is dispatched for are cudagraph_capture_sizes capped at
    # HELION_SCALED_MM_MAX_NUM_TOKENS (=32).
    CAPTURE_SIZES = [1, 2, 4, 8, 16, 24, 32, 64, 128]
    COVERED_M = [1, 2, 4, 8, 16, 24, 32]

    def _keys_for(self, K: int, N: int, m_values: list[int]) -> list[CaseKey]:
        return [CaseKey({"K": K, "N": N, "M": m}) for m in m_values]

    @pytest.mark.cpu_test
    def test_full_coverage(self):
        K, N = 4096, 6144
        keys = self._keys_for(K, N, self.COVERED_M)
        with _patch_can_implement_env(keys, self.CAPTURE_SIZES, 128):
            can_impl, reason = HelionFP8ScaledMMLinearKernel.can_implement(
                _make_fp8_config(K, N)
            )
        assert can_impl, reason
        assert reason is None

    @pytest.mark.cpu_test
    def test_missing_one_m_config(self):
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
    def test_missing_config_for_other_shape(self):
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
    def test_m_sizes_capped_by_helion_max(self):
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
    def test_disabled_op(self):
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
    def test_skip_config_check_env(self, monkeypatch: pytest.MonkeyPatch):
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
        baseline(baseline_out, a, b, scale_a, scale_b, bias)

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
        baseline(baseline_out, a, b, scale_a, scale_b, bias)

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
