# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the fused lora_shrink_expand op.

Verifies that the fused op produces identical results to calling
lora_shrink + lora_expand separately.
"""

import pytest
import torch

from vllm.lora.ops.triton_ops import LoRAKernelMeta, lora_expand, lora_shrink
from vllm.lora.ops.triton_ops.lora_fused_op import lora_shrink_expand

# Clear cached pointers between tests to avoid stale pointer issues
from vllm.lora.ops.triton_ops.utils import _LORA_A_PTR_DICT, _LORA_B_PTR_DICT
from vllm.utils.torch_utils import set_random_seed

from .utils import (
    PunicaTensors,
    assert_close,
    generate_data_for_nslices,
)

_dict_lock = __import__("threading").Lock()


@pytest.fixture(autouse=True)
def reset_device(reset_default_device):
    yield


RANKS = [16, 32, 64]
BATCH_SIZES = [1, 4, 16]
NUM_LORAS = [1, 2, 4]
HIDDEN_SIZES = [512, 2048]
NSLICES_LIST = [1, 3]
DTYPES = [torch.float16, torch.bfloat16]
SCALING = 0.5


@pytest.mark.parametrize("batches", BATCH_SIZES)
@pytest.mark.parametrize("num_loras", NUM_LORAS)
@pytest.mark.parametrize("rank", RANKS)
@pytest.mark.parametrize("hidden_size", HIDDEN_SIZES)
@pytest.mark.parametrize("nslices", NSLICES_LIST)
@pytest.mark.parametrize("dtype", DTYPES)
def test_fused_matches_separate(
    batches: int,
    num_loras: int,
    rank: int,
    hidden_size: int,
    nslices: int,
    dtype: torch.dtype,
):
    """Fused shrink+expand produces same output as separate ops."""
    set_random_seed(42)
    device = "cuda"
    seq_length = 1

    # Generate shrink data
    shrink_data: PunicaTensors = generate_data_for_nslices(
        batches,
        hidden_size,
        num_loras,
        rank,
        seq_length,
        nslices,
        dtype,
        "shrink",
        device,
    )
    _, token_nums = shrink_data.meta()

    # Generate expand data (B weights)
    # For expand, output_size can differ per slice; use hidden_size for simplicity
    lora_b_weights = []
    for _ in range(nslices):
        w = torch.randn(num_loras, 1, hidden_size, rank, dtype=dtype, device=device)
        lora_b_weights.append(w)

    # Build LoRA kernel metadata
    lora_meta = LoRAKernelMeta.make(
        max_loras=num_loras,
        max_num_tokens=token_nums,
        device=device,
    )
    lora_meta.prepare_tensors(shrink_data.token_lora_mapping)

    meta_args = lora_meta.meta_args(
        token_nums=token_nums,
        specialize_active_lora=False,
    )

    inputs = shrink_data.inputs_tensor
    lora_a_weights = shrink_data.lora_weights
    if not isinstance(lora_a_weights, list):
        lora_a_weights = [lora_a_weights]

    # ── Separate shrink + expand ──
    shrink_buffer_sep = torch.zeros(
        nslices,
        token_nums,
        rank,
        dtype=torch.float32,
        device=device,
    )
    output_sep = torch.randn(
        token_nums,
        hidden_size * nslices,
        dtype=dtype,
        device=device,
    )
    output_sep_copy = output_sep.clone()

    with _dict_lock:
        _LORA_A_PTR_DICT.clear()
        _LORA_B_PTR_DICT.clear()
        lora_shrink(
            inputs,
            lora_a_weights,
            shrink_buffer_sep,
            *meta_args,
            SCALING,
        )
        lora_expand(
            shrink_buffer_sep,
            lora_b_weights,
            output_sep,
            *meta_args,
            offset_start=0,
            add_inputs=True,
        )

    # ── Fused shrink + expand ──
    shrink_buffer_fused = torch.zeros(
        nslices,
        token_nums,
        rank,
        dtype=torch.float32,
        device=device,
    )
    output_fused = output_sep_copy.clone()

    with _dict_lock:
        _LORA_A_PTR_DICT.clear()
        _LORA_B_PTR_DICT.clear()
        lora_shrink_expand(
            inputs,
            lora_a_weights,
            shrink_buffer_fused,
            lora_b_weights,
            output_fused,
            *meta_args,
            SCALING,
            offset_start=0,
        )

    # ── Compare ──
    assert_close(shrink_buffer_fused, shrink_buffer_sep)
    assert_close(output_fused, output_sep)


@pytest.mark.parametrize("batches", [1, 8])
@pytest.mark.parametrize("rank", [16, 64])
def test_fused_no_lora_noop(batches: int, rank: int):
    """Fused op is a no-op when no tokens need LoRA."""
    set_random_seed(42)
    device = "cuda"
    hidden_size = 512
    num_loras = 2
    nslices = 1
    dtype = torch.float16

    data: PunicaTensors = generate_data_for_nslices(
        batches,
        hidden_size,
        num_loras,
        rank,
        1,
        nslices,
        dtype,
        "shrink",
        device,
    )
    _, token_nums = data.meta()

    lora_b_weights = [
        torch.randn(num_loras, 1, hidden_size, rank, dtype=dtype, device=device)
    ]

    lora_meta = LoRAKernelMeta.make(
        max_loras=num_loras,
        max_num_tokens=token_nums,
        device=device,
    )
    # Map all tokens to no-lora (-1)
    no_lora_mapping = torch.full(
        (token_nums,),
        -1,
        dtype=torch.long,
        device=device,
    )
    lora_meta.prepare_tensors(no_lora_mapping)

    meta_args = lora_meta.meta_args(
        token_nums=token_nums,
        specialize_active_lora=False,
    )

    inputs = data.inputs_tensor
    lora_a_weights = data.lora_weights
    if not isinstance(lora_a_weights, list):
        lora_a_weights = [lora_a_weights]

    output = torch.randn(
        token_nums,
        hidden_size * nslices,
        dtype=dtype,
        device=device,
    )
    output_before = output.clone()
    shrink_buffer = torch.zeros(
        nslices,
        token_nums,
        rank,
        dtype=torch.float32,
        device=device,
    )

    with _dict_lock:
        _LORA_A_PTR_DICT.clear()
        _LORA_B_PTR_DICT.clear()
        lora_shrink_expand(
            inputs,
            lora_a_weights,
            shrink_buffer,
            lora_b_weights,
            output,
            *meta_args,
            SCALING,
            offset_start=0,
        )

    # Output should be unchanged when no tokens need LoRA
    assert torch.equal(output, output_before)
