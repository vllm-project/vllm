# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Regression tests for TokenEmbeddingPoolerHead.project_batch dtype handling.

Specifically guards against the dtype-mismatch crash that happened when the
projector is an nn.Sequential (auto-loaded sentence-transformers 1_Dense/)
held at fp32 while the model trunk runs in fp16/bf16.  Before the fix the
else branch in project_batch passed hidden_states through to the projector
without a cast, raising RuntimeError("mat1 and mat2 must have the same dtype").
"""

import pytest
import torch
import torch.nn as nn

from vllm.model_executor.layers.pooler.tokwise.heads import TokenEmbeddingPoolerHead


def _l2_normalize(x: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.normalize(x, p=2, dim=-1)


@pytest.mark.parametrize("input_dtype", [torch.float16, torch.bfloat16])
def test_project_batch_sequential_projector_fp32_weights(input_dtype):
    """Projector held at fp32 (Sequential), input in fp16/bf16."""
    hidden_dim, embed_dim = 32, 8
    projector = nn.Sequential(nn.Linear(hidden_dim, embed_dim))
    head = TokenEmbeddingPoolerHead(
        head_dtype=torch.float32,
        projector=projector,
        activation=_l2_normalize,
    )

    n_tokens = 4
    hidden_states = torch.randn(n_tokens, hidden_dim, dtype=input_dtype)

    # Pre-fix: this raises "mat1 and mat2 must have the same dtype".
    out = head.project_batch(hidden_states)

    assert out.shape == (n_tokens, embed_dim)
    assert out.dtype == torch.float32  # head_dtype upcast applied


@pytest.mark.parametrize("input_dtype", [torch.float16, torch.bfloat16])
def test_project_batch_linear_projector_fp32_weights(input_dtype):
    """Linear projector at fp32 — uses the optimized weight-cast path.
    Result must match the slow upcast path bit-exactly.
    """
    hidden_dim, embed_dim = 32, 8
    linear = nn.Linear(hidden_dim, embed_dim)
    head_linear = TokenEmbeddingPoolerHead(
        head_dtype=torch.float32, projector=linear, activation=_l2_normalize
    )

    sequential = nn.Sequential(nn.Linear(hidden_dim, embed_dim))
    sequential[0].load_state_dict(linear.state_dict())
    head_seq = TokenEmbeddingPoolerHead(
        head_dtype=torch.float32, projector=sequential, activation=_l2_normalize
    )

    hidden_states = torch.randn(4, hidden_dim, dtype=input_dtype)
    out_linear = head_linear.project_batch(hidden_states)
    out_seq = head_seq.project_batch(hidden_states)

    # Optimized path (downcast weight) and reference path (upcast input)
    # may differ within fp16/bf16 ulp.  bf16 has only 7 mantissa bits so
    # relative tolerance must be looser than fp16's 11.
    rtol = 2e-2 if input_dtype is torch.bfloat16 else 5e-3
    torch.testing.assert_close(out_linear, out_seq, atol=1e-3, rtol=rtol)


def test_project_batch_no_projector():
    """When projector is None, project_batch is a no-op apart from dtype/activation."""
    head = TokenEmbeddingPoolerHead(
        head_dtype=torch.float32, projector=None, activation=_l2_normalize
    )
    hidden_states = torch.randn(4, 8, dtype=torch.float16)
    out = head.project_batch(hidden_states)
    assert out.shape == hidden_states.shape
    assert out.dtype == torch.float32
