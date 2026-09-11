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


@pytest.mark.parametrize("output_final_state", [False, True])
def test_flashinfer_gdn_prefill_reuses_output_and_uses_int64_cu_seqlens(
    monkeypatch, output_final_state
):
    captured_cu_seqlens = None
    captured_output = None
    expected_final_state = torch.ones(1, 1, 2, 2)

    def fake_chunk_gated_delta_rule(**kwargs):
        nonlocal captured_cu_seqlens, captured_output
        captured_cu_seqlens = kwargs["cu_seqlens"]
        captured_output = kwargs["output"]
        if kwargs["output_final_state"]:
            return kwargs["output"], expected_final_state
        return kwargs["output"]

    monkeypatch.setattr(
        flashinfer.gdn_prefill,
        "chunk_gated_delta_rule",
        fake_chunk_gated_delta_rule,
    )
    q = torch.zeros(1, 2, 1, 2)
    core_attn_out = torch.empty(3, 1, 2)
    cu_seqlens = torch.tensor([0, 2], dtype=torch.int32)

    output, final_state = fi_chunk_gated_delta_rule(
        q=q,
        k=q,
        v=q,
        g=torch.zeros(1, 2, 1),
        beta=torch.zeros(1, 2, 1),
        initial_state=torch.zeros(1, 1, 2, 2),
        output_final_state=output_final_state,
        cu_seqlens=cu_seqlens,
        use_qk_l2norm_in_kernel=False,
        core_attn_out=core_attn_out,
    )

    assert captured_cu_seqlens is not None
    assert captured_cu_seqlens.dtype == torch.int64
    assert captured_output.shape == (2, 1, 2)
    assert captured_output.data_ptr() == core_attn_out.data_ptr()
    assert output.data_ptr() == core_attn_out.data_ptr()
    assert output.shape == q.shape
    if output_final_state:
        assert final_state is expected_final_state
    else:
        assert final_state is None
