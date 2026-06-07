# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest
import torch

from vllm.model_executor.models.utils import _merge_multimodal_embeddings


def test_numerical_parity():
    # Setup
    seq_len = 10
    hidden_size = 4
    inputs_embeds = torch.randn(seq_len, hidden_size)

    is_multimodal = torch.tensor(
        [False, True, True, False, False, True, False, True, False, False]
    )
    num_mm = is_multimodal.sum().item()
    mm_embeds_flat = torch.randn(num_mm, hidden_size)
    multimodal_embeddings = [mm_embeds_flat]

    # Old logic (boolean indexing)
    expected_embeds = inputs_embeds.clone()
    expected_embeds[is_multimodal] = mm_embeds_flat

    # New logic (Dummy Row + functional return)
    actual_embeds = _merge_multimodal_embeddings(
        inputs_embeds.clone(), multimodal_embeddings, is_multimodal
    )

    assert torch.allclose(expected_embeds, actual_embeds)


def test_empty_input():
    seq_len = 5
    hidden_size = 4
    inputs_embeds = torch.randn(seq_len, hidden_size)
    is_multimodal = torch.zeros(seq_len, dtype=torch.bool)

    actual_embeds = _merge_multimodal_embeddings(inputs_embeds, [], is_multimodal)

    assert actual_embeds is inputs_embeds  # Check identity
    assert torch.allclose(actual_embeds, inputs_embeds)


@pytest.mark.parametrize(
    "device", ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]
)
def test_mismatch_robustness(device):
    if device == "cuda" and not torch.cuda.is_available():
        return

    seq_len = 5
    hidden_size = 4
    inputs_embeds = torch.randn(seq_len, hidden_size, device=device)

    # 3 True values, but we provide 2 embeddings
    is_multimodal = torch.tensor([False, True, True, True, False], device=device)
    mm_embeds_flat = torch.randn(2, hidden_size, device=device)

    # The "Operand Padding" naturally guards against out-of-bounds gathers.
    out = _merge_multimodal_embeddings(inputs_embeds, [mm_embeds_flat], is_multimodal)
    assert out is not None
    assert out.shape == (seq_len, hidden_size)


@pytest.mark.parametrize(
    "device", ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_device_dtype_parity(device, dtype):
    if device == "cuda" and not torch.cuda.is_available():
        return

    seq_len = 8
    hidden_size = 16
    inputs_embeds = torch.randn(seq_len, hidden_size, dtype=dtype, device=device)

    is_multimodal = torch.tensor(
        [True, False, True, False, True, False, False, False], device=device
    )
    num_mm = 3
    mm_embeds_flat = torch.randn(num_mm, hidden_size, dtype=dtype, device=device)

    expected_embeds = inputs_embeds.clone()
    expected_embeds[is_multimodal] = mm_embeds_flat

    actual_embeds = _merge_multimodal_embeddings(
        inputs_embeds.clone(), [mm_embeds_flat], is_multimodal
    )

    assert actual_embeds.device.type == device
    assert actual_embeds.dtype == dtype
    assert torch.equal(expected_embeds, actual_embeds)
