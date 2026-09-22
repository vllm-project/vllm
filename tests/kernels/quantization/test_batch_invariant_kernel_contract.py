# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.model_executor.layers.quantization.utils import fp8_utils


def test_required_batch_invariant_kernel_uses_packaged_default(monkeypatch, tmp_path):
    monkeypatch.delenv("VLLM_BATCH_INVARIANT_KERNEL_LIB", raising=False)
    packaged = tmp_path / "_vllm_batch_invariant_C.so"
    monkeypatch.setattr(fp8_utils, "_PACKAGED_BATCH_INVARIANT_KERNEL", packaged)

    with pytest.raises(RuntimeError, match="library does not exist"):
        fp8_utils.require_batch_invariant_quant_kernel()


def test_required_batch_invariant_kernel_loads_configured_library(monkeypatch):
    calls = []
    monkeypatch.setenv("VLLM_BATCH_INVARIANT_KERNEL_LIB", "/test/bi-kernel.so")
    monkeypatch.setattr(
        fp8_utils,
        "_load_batch_invariant_kernel_library",
        lambda path: calls.append(path),
    )

    fp8_utils.require_batch_invariant_quant_kernel()

    assert calls == ["/test/bi-kernel.so"]


def test_required_batch_invariant_kernel_rejects_missing_library(
    monkeypatch, tmp_path
):
    missing = tmp_path / "_vllm_batch_invariant_C.so"
    monkeypatch.setenv("VLLM_BATCH_INVARIANT_KERNEL_LIB", str(missing))

    with pytest.raises(RuntimeError, match="library does not exist"):
        fp8_utils.require_batch_invariant_quant_kernel()


def test_contiguous_deepgemm_forwards_clamp_to_bi_kernel(monkeypatch):
    from types import SimpleNamespace

    from vllm.model_executor.layers.fused_moe import MoEActivation
    from vllm.model_executor.layers.fused_moe.experts import deep_gemm_moe
    from vllm.utils.deep_gemm import DeepGemmQuantScaleFMT

    calls = []

    def fused(value, **kwargs):
        calls.append(kwargs)
        return kwargs["output_q"], torch.ones(1)

    monkeypatch.setattr(
        deep_gemm_moe, "is_batch_invariant_quant_kernel_enabled", lambda: True
    )
    monkeypatch.setattr(
        DeepGemmQuantScaleFMT,
        "from_oracle",
        staticmethod(lambda: DeepGemmQuantScaleFMT.FLOAT32_CEIL_UE8M0),
    )
    monkeypatch.setattr(
        deep_gemm_moe, "fused_silu_mul_per_token_group_quant_fp8", fused
    )
    expert = SimpleNamespace(
        block_shape=[128, 128],
        gemm1_clamp_limit=10.0,
        gemm1_alpha=1.0,
        gemm1_beta=0.0,
        adjust_N_for_activation=lambda n, _activation: n // 2,
    )
    value = torch.randn(2, 256, dtype=torch.bfloat16)
    output = torch.empty(2, 128, dtype=torch.float8_e4m3fn)

    quantized, _scales = deep_gemm_moe.DeepGemmExperts._act_mul_quant(
        expert, value, output, MoEActivation.SILU
    )

    assert quantized is output
    assert len(calls) == 1
    assert calls[0]["output_q"] is output
    assert calls[0]["use_ue8m0"] is False
    assert calls[0]["round_scale"] is True
    assert calls[0]["clamp_limit"] == 10.0
    assert calls[0]["masked_m"] is None
    assert calls[0]["group_size"] == 128


def test_masked_deepgemm_forwards_clamp_to_bi_kernel(monkeypatch):
    from vllm.model_executor.layers.fused_moe.experts import batched_deep_gemm_moe
    from vllm.utils.deep_gemm import DeepGemmQuantScaleFMT

    calls = []

    def fused(value, **kwargs):
        calls.append(kwargs)
        return torch.empty(1), torch.empty(1)

    monkeypatch.setattr(
        batched_deep_gemm_moe,
        "is_batch_invariant_quant_kernel_enabled",
        lambda: True,
    )
    monkeypatch.setattr(
        batched_deep_gemm_moe,
        "fused_silu_mul_per_token_group_quant_fp8",
        fused,
    )
    value = torch.randn(2, 3, 256, dtype=torch.bfloat16)
    counts = torch.tensor([2, 3], dtype=torch.int32)

    batched_deep_gemm_moe.persistent_masked_m_silu_mul_quant(
        value,
        counts,
        quant_scale_fmt=DeepGemmQuantScaleFMT.FLOAT32_CEIL_UE8M0,
        clamp_limit=10.0,
    )

    assert calls[0]["round_scale"] is True
    assert calls[0]["clamp_limit"] == 10.0
    assert calls[0]["masked_m"] is counts
