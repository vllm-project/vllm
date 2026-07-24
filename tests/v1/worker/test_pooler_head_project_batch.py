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
@pytest.mark.parametrize("wrap_sequential", [False, True])
def test_project_batch_matches_forward_chunk_numerics(input_dtype, wrap_sequential):
    """project_batch must project at head_dtype exactly like forward_chunk
    (PR #40337 review, point 1): queries (chunk path) and documents (batch
    path) must use identical projection precision. Bit-exact — the batch
    path performs the same upcast-then-project pipeline, so no tolerance
    is needed or allowed.
    """
    hidden_dim, embed_dim = 32, 8
    linear = nn.Linear(hidden_dim, embed_dim)
    projector = nn.Sequential(linear) if wrap_sequential else linear
    head = TokenEmbeddingPoolerHead(
        head_dtype=torch.float32, projector=projector, activation=_l2_normalize
    )

    hidden_states = torch.randn(4, hidden_dim, dtype=input_dtype)
    # forward_chunk numerics: upcast to head_dtype, then project, then act.
    ref = _l2_normalize(linear(hidden_states.to(torch.float32)))

    out = head.project_batch(hidden_states)
    assert out.dtype == torch.float32
    torch.testing.assert_close(out, ref, atol=0.0, rtol=0.0)


def test_project_batch_no_projector():
    """When projector is None, project_batch is a no-op apart from dtype/activation."""
    head = TokenEmbeddingPoolerHead(
        head_dtype=torch.float32, projector=None, activation=_l2_normalize
    )
    hidden_states = torch.randn(4, 8, dtype=torch.float16)
    out = head.project_batch(hidden_states)
    assert out.shape == hidden_states.shape
    assert out.dtype == torch.float32
