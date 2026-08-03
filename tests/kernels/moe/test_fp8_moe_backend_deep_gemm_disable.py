# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FP8 MoE backend selection honors the per-model DeepGEMM auto-disable.

Regression test for https://github.com/vllm-project/vllm/issues/47169: when
``quant_config.use_deep_gemm`` is explicitly ``False`` (set by
:func:`vllm.utils.deep_gemm.should_auto_disable_deep_gemm` for Qwen3.5/3.6
hybrid models on Blackwell, where DeepGEMM's E8M0 scale format degrades
accuracy), the oracle must never select the DeepGEMM FP8 MoE backends, even
when they would otherwise be the top priority.

GPU-free: the backend priority list and kernel ``is_supported_config`` checks
are mocked, so selection is deterministic on every platform.
"""

from tests.kernels.moe.utils import make_dummy_moe_config
from vllm.model_executor.layers.fused_moe.oracle.fp8 import (
    Fp8MoeBackend,
    select_fp8_moe_backend,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    kFp8Dynamic128Sym,
    kFp8Static128BlockSym,
)


class _FakeKernel:
    """A kernel class whose is_supported_config always succeeds."""

    @classmethod
    def is_supported_config(
        cls, k_cls, config, weight_key, activation_key, activation_format
    ):
        return True, None


def _dummy_config():
    return make_dummy_moe_config(
        num_experts=8,
        experts_per_token=2,
        hidden_dim=512,
        intermediate_size=1024,
    )


def _install_fake_selection(monkeypatch):
    """Force DeepGEMM to be the top-priority backend with fake kernels."""
    monkeypatch.delenv("VLLM_USE_DEEP_GEMM", raising=False)
    monkeypatch.delenv("VLLM_MOE_USE_DEEP_GEMM", raising=False)

    def fake_priority(config, weight_key, activation_key):
        return [
            Fp8MoeBackend.DEEPGEMM,
            Fp8MoeBackend.TRITON,
            Fp8MoeBackend.VLLM_CUTLASS,
            Fp8MoeBackend.BATCHED_VLLM_CUTLASS,
        ]

    monkeypatch.setattr(
        "vllm.model_executor.layers.fused_moe.oracle.fp8._get_priority_backends",
        fake_priority,
    )
    monkeypatch.setattr(
        "vllm.model_executor.layers.fused_moe.oracle.fp8.backend_to_kernel_cls",
        lambda backend: [_FakeKernel],
    )


def test_select_fp8_moe_backend_excludes_deepgemm_when_disabled(monkeypatch):
    _install_fake_selection(monkeypatch)
    backend, _ = select_fp8_moe_backend(
        config=_dummy_config(),
        weight_key=kFp8Static128BlockSym,
        activation_key=kFp8Dynamic128Sym,
        use_deep_gemm=False,
    )
    assert backend == Fp8MoeBackend.TRITON
    assert backend not in (
        Fp8MoeBackend.DEEPGEMM,
        Fp8MoeBackend.BATCHED_DEEPGEMM,
    )


def test_select_fp8_moe_backend_keeps_deepgemm_by_default(monkeypatch):
    _install_fake_selection(monkeypatch)
    backend, _ = select_fp8_moe_backend(
        config=_dummy_config(),
        weight_key=kFp8Static128BlockSym,
        activation_key=kFp8Dynamic128Sym,
    )
    assert backend == Fp8MoeBackend.DEEPGEMM


def test_select_fp8_moe_backend_keeps_deepgemm_when_explicitly_enabled(
    monkeypatch,
):
    _install_fake_selection(monkeypatch)
    backend, _ = select_fp8_moe_backend(
        config=_dummy_config(),
        weight_key=kFp8Static128BlockSym,
        activation_key=kFp8Dynamic128Sym,
        use_deep_gemm=True,
    )
    assert backend == Fp8MoeBackend.DEEPGEMM


def test_select_fp8_moe_backend_env_override_wins_over_disable(monkeypatch):
    _install_fake_selection(monkeypatch)
    # Explicitly setting the DeepGEMM env vars is a user override that must
    # win over the per-model auto-disable.
    monkeypatch.setenv("VLLM_USE_DEEP_GEMM", "1")
    monkeypatch.setenv("VLLM_MOE_USE_DEEP_GEMM", "1")
    backend, _ = select_fp8_moe_backend(
        config=_dummy_config(),
        weight_key=kFp8Static128BlockSym,
        activation_key=kFp8Dynamic128Sym,
        use_deep_gemm=False,
    )
    assert backend == Fp8MoeBackend.DEEPGEMM
