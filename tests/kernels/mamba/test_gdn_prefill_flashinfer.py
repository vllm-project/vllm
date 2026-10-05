# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.platforms import current_platform

if current_platform.is_rocm():
    pytest.skip(
        reason="FlashInfer GDN prefill is not supported on ROCm.",
        allow_module_level=True,
    )

import flashinfer.gdn_prefill  # noqa: E402

from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import (
    fi_chunk_gated_delta_rule,
)  # noqa: E402


@pytest.mark.parametrize("cu_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("g_is_exp", [False, True])
def test_flashinfer_gdn_prefill_gate_and_lengths(monkeypatch, cu_dtype, g_is_exp):
    """FI consumes linear gates and reuses already prepared int64 lengths."""
    captured_cu_seqlens = None
    captured_g = None

    def fake_chunk_gated_delta_rule(**kwargs):
        nonlocal captured_cu_seqlens, captured_g
        captured_cu_seqlens = kwargs["cu_seqlens"]
        captured_g = kwargs["g"]
        return kwargs["q"]

    monkeypatch.setattr(
        flashinfer.gdn_prefill,
        "chunk_gated_delta_rule",
        fake_chunk_gated_delta_rule,
    )
    q = torch.zeros(1, 2, 1, 2)
    cu_seqlens = torch.tensor([0, 2], dtype=cu_dtype)
    log_g = torch.tensor([[[-0.5], [-2.0]]])
    g = log_g.exp() if g_is_exp else log_g

    output, final_state = fi_chunk_gated_delta_rule(
        q=q,
        k=q,
        v=q,
        g=g,
        beta=torch.zeros(1, 2, 1),
        initial_state=torch.zeros(1, 1, 2, 2),
        output_final_state=False,
        cu_seqlens=cu_seqlens,
        use_qk_l2norm_in_kernel=False,
        g_is_exp=g_is_exp,
    )

    assert captured_cu_seqlens is not None
    assert captured_cu_seqlens.dtype == torch.int64
    torch.testing.assert_close(captured_cu_seqlens, cu_seqlens, check_dtype=False)
    if cu_dtype == torch.int64:
        assert captured_cu_seqlens is cu_seqlens
    assert captured_g is not None
    torch.testing.assert_close(captured_g, log_g.exp().squeeze(0))
    if g_is_exp:
        assert captured_g.data_ptr() == g.data_ptr()
    assert output.shape == q.shape
    assert final_state is None
