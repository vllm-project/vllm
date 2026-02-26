# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from tests.lora.utils import (
    PunicaTensors,
    assert_close,
    generate_data,
    generate_data_for_expand_nslices,
)
from vllm.lora.ops.xpu_ops import bgmv_expand, bgmv_expand_slice, bgmv_shrink
from vllm.platforms import current_platform


def torch_bgmv_expand(
    inputs: torch.Tensor,
    lora_b_weights: torch.Tensor,
    output_tensor: torch.Tensor,
    lora_indices_tensor: torch.Tensor,
    add_inputs: bool = True,
):
    selected_loras = lora_b_weights[lora_indices_tensor].to(dtype=output_tensor.dtype)
    if len(selected_loras.shape) == 4:
        selected_loras = selected_loras.squeeze(dim=1)
    inputs = inputs.to(dtype=output_tensor.dtype)
    outputs = torch.einsum("bi, boi -> bo", inputs, selected_loras)

    limit = output_tensor.shape[0]
    if outputs.shape[0] == 1 and output_tensor.shape[0] != 1:
        limit = 1

    # LoRA adapter and model may add different amounts of padding to output
    common_len = min(outputs.shape[1], output_tensor.shape[1])

    if add_inputs:
        output_tensor[:, :common_len] += outputs[:limit, :common_len]
    else:
        output_tensor[:, :common_len] = outputs[:limit, :common_len]


def torch_bgmv_shrink(
    inputs: torch.Tensor,
    lora_b_weights: torch.Tensor,
    output_tensor: torch.Tensor,
    lora_indices_tensor: torch.Tensor,
    scaling: float = 1.0,
):
    selected_loras = lora_b_weights[lora_indices_tensor].to(dtype=output_tensor.dtype)
    if len(selected_loras.shape) == 4:
        selected_loras = selected_loras.squeeze(dim=1)
    inputs = inputs.to(dtype=output_tensor.dtype)
    outputs = torch.einsum("bi, boi -> bo", inputs, selected_loras)

    output_tensor[:, : outputs.shape[1]] = scaling * outputs[:]


def torch_bgmv_expand_slice(
    inputs: torch.Tensor,
    lora_b_weights: torch.Tensor,
    output_tensor: torch.Tensor,
    lora_indices_tensor: torch.Tensor,
    slice_offset: int,
    slice_size: int,
    add_inputs: bool = True,
):
    selected_loras = lora_b_weights[lora_indices_tensor].to(dtype=output_tensor.dtype)
    inputs = inputs.to(dtype=output_tensor.dtype)
    if len(selected_loras.shape) == 4:
        selected_loras = selected_loras.squeeze(dim=1)
    outputs = torch.einsum("bi, boi -> bo", inputs, selected_loras)

    if add_inputs:
        output_tensor[:, slice_offset : slice_offset + slice_size] += outputs[:]
    else:
        output_tensor[:, slice_offset : slice_offset + slice_size] = outputs[:]


def check_bgmv_shrink(
    batches: int,
    num_loras: int,
    rank: int,
    hidden_size: int,
    dtype: torch.dtype,
    device: str,
    scaling: float,
):
    """
    Compare vllm.bgmv_shrink against a reference implementation.
    """
    seq_length = 1
    data: PunicaTensors = generate_data(
        batches,
        hidden_size,
        num_loras,
        rank,
        seq_length,
        dtype,
        "shrink",
        device,
    )

    bgmv_shrink(
        data.inputs_tensor,
        data.lora_weights,
        data.our_out_tensor,
        data.token_lora_mapping,
        scaling,
    )

    torch_bgmv_shrink(
        data.inputs_tensor,
        data.lora_weights,
        data.ref_out_tensor,
        data.token_lora_mapping,
        scaling,
    )

    data.ref_out_tensor = data.ref_out_tensor.to(torch.float32)
    assert_close(data.our_out_tensor, data.ref_out_tensor)


def check_bgmv_expand(
    batches: int,
    num_loras: int,
    rank: int,
    hidden_size: int,
    dtype: torch.dtype,
    device: str,
    add_inputs: bool,
):
    """
    Compare vllm.bgmv_expand against a reference implementation.
    """
    seq_length = 1
    data: PunicaTensors = generate_data(
        batches,
        hidden_size,
        num_loras,
        rank,
        seq_length,
        dtype,
        "expand",
        device,
    )

    bgmv_expand(
        data.inputs_tensor,
        data.lora_weights,
        data.our_out_tensor,
        data.token_lora_mapping,
        add_inputs=add_inputs,
    )
    torch_bgmv_expand(
        data.inputs_tensor,
        data.lora_weights,
        data.ref_out_tensor,
        data.token_lora_mapping,
        add_inputs=add_inputs,
    )
    assert_close(data.ref_out_tensor, data.our_out_tensor)


def check_bgmv_expand_slice(
    batches: int,
    num_loras: int,
    rank: int,
    hidden_size: int,
    nslices: int,
    dtype: torch.dtype,
    device: str,
    add_inputs: bool,
):
    """
    Compare vllm.bgmv_expand_slice against a reference implementation.
    """
    seq_length = 1
    data: PunicaTensors = generate_data_for_expand_nslices(
        batches,
        hidden_size,
        num_loras,
        rank,
        seq_length,
        dtype,
        nslices,
        device,
    )

    slice_offset = 0
    for index in range(nslices):
        bgmv_expand_slice(
            data.inputs_tensor,
            data.lora_weights[index],
            data.our_out_tensor,
            data.token_lora_mapping,
            slice_offset,
            slice_size=hidden_size,
            add_inputs=add_inputs,
        )
        torch_bgmv_expand_slice(
            data.inputs_tensor,
            data.lora_weights[index],
            data.ref_out_tensor,
            data.token_lora_mapping,
            slice_offset,
            slice_size=hidden_size,
            add_inputs=add_inputs,
        )

        slice_offset += hidden_size
    assert_close(data.ref_out_tensor, data.our_out_tensor)


# General tests params that tests for variations in all dimensions
# except hidden_size.
test_params = {
    "hidden_sizes": [2049],
    "batches": [4],
    "num_loras": [4],
    "max_ranks": [32],
}

DTYPES = [torch.float16, torch.bfloat16]
DEVICES = [f"xpu:{0}"]
SEED = [0]


@pytest.mark.parametrize("batches", test_params["batches"])
@pytest.mark.parametrize("num_loras", test_params["num_loras"])
@pytest.mark.parametrize("rank", test_params["max_ranks"])
@pytest.mark.parametrize("hidden_size", test_params["hidden_sizes"])
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("seed", SEED)
@pytest.mark.parametrize("op_type", ["shrink", "expand"])
@pytest.mark.skipif(not current_platform.is_xpu(), reason="skip for non xpu platform")
def test_bgmv(
    batches: int,
    num_loras: int,
    rank: int,
    hidden_size: int,
    dtype: torch.dtype,
    device: str,
    seed: int,
    op_type: str,
):
    if op_type == "shrink":
        check_bgmv_shrink(
            batches=batches,
            num_loras=num_loras,
            rank=rank,
            hidden_size=hidden_size,
            dtype=dtype,
            device=device,
            scaling=0.5,
        )
    else:
        check_bgmv_expand(
            batches=batches,
            num_loras=num_loras,
            rank=rank,
            hidden_size=hidden_size,
            dtype=dtype,
            device=device,
            add_inputs=True,
        )


@pytest.mark.parametrize("batches", test_params["batches"])
@pytest.mark.parametrize("num_loras", test_params["num_loras"])
@pytest.mark.parametrize("rank", test_params["max_ranks"])
@pytest.mark.parametrize("hidden_size", test_params["hidden_sizes"])
@pytest.mark.parametrize("nslices", [2, 3])
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("seed", SEED)
@pytest.mark.skipif(not current_platform.is_xpu(), reason="skip for non xpu platform")
def test_bgmv_expand_nslices(
    batches: int,
    num_loras: int,
    rank: int,
    hidden_size: int,
    nslices: int,
    dtype: torch.dtype,
    device: str,
    seed: int,
):
    check_bgmv_expand_slice(
        batches=batches,
        num_loras=num_loras,
        rank=rank,
        hidden_size=hidden_size,
        nslices=nslices,
        dtype=dtype,
        device=device,
        add_inputs=True,
    )
