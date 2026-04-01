# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensorMode

from vllm.utils.torch_utils import direct_register_custom_op


def _indexer_weights_and_k_proj_fake(
    hidden_states: torch.Tensor,
    layer_name: str,
    n_head: int,
    head_dim: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fake implementation for torch.compile shape inference."""
    num_tokens = hidden_states.shape[0]
    return (
        hidden_states.new_empty(num_tokens, n_head),
        hidden_states.new_empty(num_tokens, head_dim),
    )


# Register the op with a no-op real impl (tests only exercise the fake impl).
if not hasattr(torch.ops.vllm, "indexer_weights_and_k_proj"):
    direct_register_custom_op(
        op_name="indexer_weights_and_k_proj",
        op_func=_indexer_weights_and_k_proj_fake,
        mutates_args=[],
        fake_impl=_indexer_weights_and_k_proj_fake,
    )


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required for custom op tests"
)
class TestIndexerWeightsAndKProjOp:
    """Tests for the indexer_weights_and_k_proj custom op fake implementation."""

    def test_fake_output_shapes_and_strides(self):
        """Verify fake impl produces correct shapes and contiguous strides."""
        num_tokens = 32
        n_head = 64
        head_dim = 128

        with FakeTensorMode():
            hidden_states = torch.empty(
                num_tokens, 7168, device="cuda", dtype=torch.bfloat16
            )
            weights, k = torch.ops.vllm.indexer_weights_and_k_proj(
                hidden_states, "test_layer", n_head, head_dim
            )

        # Shape checks
        assert weights.shape == (num_tokens, n_head)
        assert k.shape == (num_tokens, head_dim)

        # Both outputs must be contiguous (canonical strides).
        # This is the regression test for the stride mismatch bug where
        # torch.split views leaked non-contiguous strides across the
        # custom op boundary.
        assert weights.stride() == (n_head, 1)
        assert k.stride() == (head_dim, 1)

    @pytest.mark.parametrize(
        "num_tokens,n_head,head_dim",
        [(1, 64, 128), (16, 64, 128), (128, 64, 128), (256, 32, 64)],
    )
    def test_fake_output_shapes_parametrized(self, num_tokens, n_head, head_dim):
        """Verify fake impl across different dimension combinations."""
        with FakeTensorMode():
            hidden_states = torch.empty(
                num_tokens, 7168, device="cuda", dtype=torch.bfloat16
            )
            weights, k = torch.ops.vllm.indexer_weights_and_k_proj(
                hidden_states, "test_layer", n_head, head_dim
            )

        assert weights.shape == (num_tokens, n_head)
        assert k.shape == (num_tokens, head_dim)
        assert weights.stride() == (n_head, 1)
        assert k.stride() == (head_dim, 1)
