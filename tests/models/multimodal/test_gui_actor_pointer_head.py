# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Dtype contract of the GUI-Actor pointer head.

The head's projections are created under the model's dtype, so it must accept
hidden states in that dtype. `GuiActorPooler` casts both sides to the head's own
weight dtype for exactly this reason; feeding it float32 without that cast used
to raise "mat1 and mat2 must have the same dtype" on the first request, which
took the engine down with it.

These run on CPU and need no checkpoint.
"""

import pytest
import torch

from vllm.model_executor.models.gui_actor import PointerHead, pointer_temperature

pytestmark = pytest.mark.cpu_test

D_MODEL = 32


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("input_dtype", [torch.bfloat16, torch.float32])
def test_pointer_head_accepts_the_model_dtype(dtype, input_dtype):
    """Either side may arrive in another dtype once it is cast to the head's."""
    torch.manual_seed(0)
    head = PointerHead(D_MODEL, D_MODEL).to(dtype)
    patches = torch.randn(7, D_MODEL).to(input_dtype)
    pointer = torch.randn(1, D_MODEL).to(input_dtype)

    weight_dtype = head.projection_enc[0].weight.dtype
    attn, patch_logits = head(patches.to(weight_dtype), pointer.to(weight_dtype))

    # One pointer query scored against the seven patches.
    assert patch_logits.shape == (1, 7)
    assert attn.shape == (1, 7)
    assert patch_logits.dtype == dtype
    # The pooler hands the decode a distribution that sums to one per query.
    assert attn.sum(dim=-1).tolist() == pytest.approx([1.0], abs=1e-5)


def test_pointer_head_rejects_mismatched_dtypes():
    """The failure the pooler's cast exists to prevent, pinned as a test."""
    torch.manual_seed(0)
    head = PointerHead(D_MODEL, D_MODEL).to(torch.bfloat16)
    with pytest.raises(RuntimeError, match="same dtype"):
        head(torch.randn(7, D_MODEL), torch.randn(1, D_MODEL))


@pytest.mark.parametrize("value,expected", [(None, 1.0), (2.5, 2.5), ("3", 3.0)])
def test_pointer_temperature_falls_back_and_coerces(value, expected):
    """Validation lives in the model config; this only reads the value back."""
    config = torch.nn.Module()  # any object carrying the attribute
    config.pointer_temperature = value
    assert pointer_temperature(config) == expected
