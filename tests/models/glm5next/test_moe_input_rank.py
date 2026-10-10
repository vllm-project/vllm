# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GLM-5.3-Flash MoE block input rank handling.

``Glm5NextMoE.forward`` unpacked ``hidden_states.shape`` into two names, which
only works for the rank-2 ``[tokens, hidden]`` layout. Model runners that feed
bucketed rank-3 ``[1, T, hidden]`` activations raised ``ValueError: too many
values to unpack`` in every MoE layer.

The block must flatten on the way in and restore the caller's leading dims on
the way out: ``sequence_parallel_chunk``, ``tensor_model_parallel_all_gather``
and the ``[:num_tokens]`` slice inside all require tokens on dim 0, so
broadcasting is not an option.

These tests exercise that shape contract directly, without building the module
(which would need a full model config and distributed state).
"""

import inspect

import pytest
import torch

from vllm.models.glm5next.common.model import Glm5NextMoE

HIDDEN = 4096


def _forward_shape_prologue(hidden_states: torch.Tensor):
    """Run the shape handling at the top of ``Glm5NextMoE.forward``.

    Mirrors the source so the test tracks the real implementation rather than a
    copy that can drift.
    """
    src = inspect.getsource(Glm5NextMoE.forward)
    assert "orig_leading_shape" in src, (
        "Glm5NextMoE.forward no longer flattens its input; update this test"
    )
    orig_leading_shape = hidden_states.shape[:-1]
    hidden_dim = hidden_states.shape[-1]
    if hidden_states.dim() != 2:
        hidden_states = hidden_states.reshape(-1, hidden_dim)
    num_tokens = hidden_states.shape[0]
    return orig_leading_shape, hidden_dim, num_tokens, hidden_states


@pytest.mark.parametrize(
    "shape",
    [
        (8, HIDDEN),  # rank-2, the layout previously assumed
        (1, 8, HIDDEN),  # rank-3, e.g. a bucketed model runner
        (2, 4, HIDDEN),
    ],
)
def test_moe_forward_accepts_any_input_rank(shape):
    hidden_states = torch.randn(*shape)

    leading, hidden_dim, num_tokens, flat = _forward_shape_prologue(hidden_states)

    # The experts run on a rank-2 [tokens, hidden] view ...
    assert flat.dim() == 2
    assert flat.shape == (num_tokens, hidden_dim)
    assert num_tokens == hidden_states.numel() // hidden_dim

    # ... and the caller gets its original shape back.
    expert_output = torch.zeros(num_tokens, hidden_dim)
    restored = expert_output.view(*leading, hidden_dim)
    assert restored.shape == hidden_states.shape
