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


def test_flashinfer_gdn_prefill_uses_int64_cu_seqlens(monkeypatch):
    captured_cu_seqlens = None

    def fake_chunk_gated_delta_rule(**kwargs):
        nonlocal captured_cu_seqlens
        captured_cu_seqlens = kwargs["cu_seqlens"]
        return kwargs["q"]

    monkeypatch.setattr(
        flashinfer.gdn_prefill,
        "chunk_gated_delta_rule",
        fake_chunk_gated_delta_rule,
    )
    q = torch.zeros(1, 2, 1, 2)
    cu_seqlens = torch.tensor([0, 2], dtype=torch.int32)

    output, final_state = fi_chunk_gated_delta_rule(
        q=q,
        k=q,
        v=q,
        g=torch.zeros(1, 2, 1),
        beta=torch.zeros(1, 2, 1),
        initial_state=torch.zeros(1, 1, 2, 2),
        output_final_state=False,
        cu_seqlens=cu_seqlens,
        use_qk_l2norm_in_kernel=False,
    )

    assert captured_cu_seqlens is not None
    assert captured_cu_seqlens.dtype == torch.int64
    assert output.shape == q.shape
    assert final_state is None
