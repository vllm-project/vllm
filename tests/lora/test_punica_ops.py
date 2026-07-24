# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from threading import Lock

import pytest
import torch

import vllm.lora.ops.triton_ops as triton_ops
from vllm.lora.ops.triton_ops import LoRAKernelMeta
from vllm.lora.ops.triton_ops.utils import _LORA_A_PTR_DICT, _LORA_B_PTR_DICT
from vllm.platforms import current_platform
from vllm.utils.torch_utils import set_random_seed

from .utils import PunicaTensors, assert_close, generate_data_for_nslices

DEVICE_TYPE = current_platform.device_type


@pytest.fixture(autouse=True)
def reset_device(reset_default_device):
    pass


@pytest.fixture(autouse=True)
def cleanup_fixture():
    """Override conftest's cleanup_fixture— not needed for punica tests."""
    yield


@pytest.fixture(autouse=True)
def dynamo_reset():
    """Override conftest's dynamo_reset — not needed for punica tests."""
    yield


def _cpu_bgmv_shrink(
    inputs, lora_weight, output, seq_len_tensor, lora_indices, scaling=1.0
):
    """Memory-efficient shrink reference: per-LoRA matmul loop on CPU.
    output[mask] = scaling * inputs[mask] @ weight.T"""
    exploded = torch.repeat_interleave(lora_indices, seq_len_tensor)
    for lid in exploded.unique():
        if lid < 0:
            continue
        mask = exploded == lid
        inp = inputs[mask].to(output.dtype)
        w = lora_weight[lid].to(output.dtype)
        output[mask] = scaling * (inp @ w.T)


def _cpu_bgmv_expand(
    inputs,
    lora_weight,
    output,
    seq_len_tensor,
    lora_indices,
    offset=0,
    add_inputs=False,
):
    """Memory-efficient expand reference: per-LoRA matmul loop on CPU.
    output[mask, offset:offset+n] (+)= inputs[mask] @ weight.T"""
    exploded = torch.repeat_interleave(lora_indices, seq_len_tensor)
    for lid in exploded.unique():
        if lid < 0:
            continue
        mask = exploded == lid
        inp = inputs[mask].to(output.dtype)
        w = lora_weight[lid].to(output.dtype)
        n = w.shape[0]
        result = inp @ w.T
        if add_inputs:
            output[mask, offset : offset + n] += result
        else:
            output[mask, offset : offset + n] = result


# Utility shrink and expand operations used as reference implementations.
def sgmv_shrink_for_nslices(
    nslices: int,
    inputs_tensor: torch.Tensor,
    lora_weights_lst: list[torch.Tensor],
    out_tensor: torch.Tensor,
    b_seq_start_loc: torch.Tensor,
    seq_len_tensor: torch.Tensor,
    prompt_lora_mapping: torch.Tensor,
    batches: int,
    max_seq_length: int,
    num_tokens: int,
    scaling: float,
):
    """CPU reference for sgmv_shrink using per-LoRA matmul loop."""
    inp_cpu = inputs_tensor.cpu()
    seq_cpu = seq_len_tensor.cpu()
    idx_cpu = prompt_lora_mapping.cpu()
    out_cpu = out_tensor.cpu()
    for index in range(nslices):
        _cpu_bgmv_shrink(
            inp_cpu,
            lora_weights_lst[index].cpu(),
            out_cpu[index],
            seq_cpu,
            idx_cpu,
            scaling=scaling,
        )
    out_tensor.copy_(out_cpu)


def sgmv_expand_for_nslices(
    nslices: int,
    hidden_size: int,
    inputs_tensor: torch.Tensor,
    lora_weights_lst: list[torch.Tensor],
    out_tensor: torch.Tensor,
    b_seq_start_loc: torch.Tensor,
    seq_len_tensor: torch.Tensor,
    prompt_lora_mapping: torch.Tensor,
    batches: int,
    max_seq_length: int,
    num_tokens: int,
    add_inputs: bool,
) -> None:
    """CPU reference for sgmv_expand using per-LoRA matmul loop."""
    seq_cpu = seq_len_tensor.cpu()
    idx_cpu = prompt_lora_mapping.cpu()
    out_cpu = out_tensor.cpu()
    for index in range(nslices):
        _cpu_bgmv_expand(
            inputs_tensor[index].cpu(),
            lora_weights_lst[index].cpu(),
            out_cpu,
            seq_cpu,
            idx_cpu,
            offset=hidden_size * index,
            add_inputs=add_inputs,
        )
    out_tensor.copy_(out_cpu)


_dict_lock = Lock()


def check_lora_shrink_kernel(
    batches: int,
    num_loras: int,
    rank: int,
    hidden_size: int,
    nslices: int,
    dtype: torch.dtype,
    device: str,
    seq_length: int,
    scaling: float,
):
    """
    Compare outputs of torch_ops.sgmv_shrink and triton_ops.lora_shrink
    kernels.
    """
    data: PunicaTensors = generate_data_for_nslices(
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
    max_seq_length, token_nums = data.meta()

    # Setup metadata information for SGMV and reference kernels
    sgmv_meta_args = (
        data.b_seq_start_loc,
        data.seq_len_tensor,
        data.prompt_lora_mapping,
        batches,
        max_seq_length,
        token_nums,
    )

    # Setup metadata information for the LoRA kernel.
    lora_meta = LoRAKernelMeta.make(
        max_loras=num_loras,
        max_num_tokens=token_nums,
        device=DEVICE_TYPE,
    )
    lora_meta.prepare_tensors(data.token_lora_mapping)

    ref_out_tensor = data.ref_out_tensor
    out_tensor = data.our_out_tensor.clone()

    # Preventing cache error pointer.
    with _dict_lock:
        # lora_shrink kernel
        _LORA_A_PTR_DICT.clear()
        triton_ops.lora_shrink(
            data.inputs_tensor,
            data.lora_weights,
            out_tensor,
            *lora_meta.meta_args(token_nums=token_nums, specialize_active_lora=False),
            scaling,
        )

    # Reference
    sgmv_shrink_for_nslices(
        nslices,
        data.inputs_tensor,
        data.lora_weights,
        ref_out_tensor,
        *sgmv_meta_args,
        scaling,
    )

    assert_close(out_tensor, ref_out_tensor)


def check_lora_expand_kernel(
    batches: int,
    num_loras: int,
    rank: int,
    hidden_size: int,
    nslices: int,
    dtype: torch.dtype,
    device: str,
    seq_length: int,
    add_inputs: bool,
):
    """
    Compare outputs of torch_ops.sgmv_expand and triton_ops.lora_expand
    kernels.
    """
    data: PunicaTensors = generate_data_for_nslices(
        batches,
        hidden_size,
        num_loras,
        rank,
        seq_length,
        nslices,
        dtype,
        "expand",
        device,
    )

    max_seq_length, token_nums = data.meta()

    # Setup metadata information for SGMV and reference kernels
    sgmv_meta_args = (
        data.b_seq_start_loc,
        data.seq_len_tensor,
        data.prompt_lora_mapping,
        batches,
        max_seq_length,
        token_nums,
    )

    # Setup metadata information for the LoRA kernel.
    lora_meta = LoRAKernelMeta.make(
        max_loras=num_loras,
        max_num_tokens=token_nums,
        device=DEVICE_TYPE,
    )
    lora_meta.prepare_tensors(data.token_lora_mapping)

    # Setup output tensors
    ref_out_tensor = data.ref_out_tensor
    out_tensor = data.our_out_tensor.clone()

    with _dict_lock:
        # lora_expand kernel
        _LORA_B_PTR_DICT.clear()
        triton_ops.lora_expand(
            data.inputs_tensor,
            data.lora_weights,
            out_tensor,
            *lora_meta.meta_args(token_nums=token_nums, specialize_active_lora=False),
            offset_start=0,
            add_inputs=add_inputs,
        )

    # Reference
    sgmv_expand_for_nslices(
        nslices,
        hidden_size,
        data.inputs_tensor,
        data.lora_weights,
        ref_out_tensor,
        *sgmv_meta_args,
        add_inputs=add_inputs,
    )

    assert_close(out_tensor, ref_out_tensor)


# Tests
# We test the punica kernels along 2 verticals mainly.
# 1. Variations in hidden_dim size
# 2. Variations in all other parameters like (batch_size, max_rank, num_loras
#  etc.)

# We have collected the hidden_sizes included in the LoRA models
# currently supported by vLLM. It tests whether the corresponding Triton
# kernel can run normally when tensor parallelism is set to
# [1, 2, 4, 8, 16, 32, 64].
HIDDEN_SIZES = [
    128,
    256,
    512,
    896,
    1024,
    1152,
    1216,
    1280,
    1536,
    1664,
    2048,
    2240,
    2304,
    2368,
    2432,
    2560,
    2752,
    3072,
    3328,
    3456,
    3584,
    3712,
    4096,
    4480,
    4608,
    4736,
    4864,
    5120,
    5504,
    5632,
    5888,
    6144,
    6400,
    6848,
    6912,
    7168,
    7424,
    8192,
    8960,
    9216,
    9472,
    10240,
    11008,
    11264,
    13824,
    14336,
    14784,
    14848,
    15360,
    18944,
    22016,
    22528,
    24576,
    27392,
    27648,
    29568,
    29696,
    32000,
    32256,
    32512,
    32768,
    33024,
    36864,
    43264,
    49152,
    49408,
    60544,
    60672,
    64000,
    64256,
    102400,
    102656,
    128000,
    128256,
]
# The size of TP
divisibility = [1, 2, 8, 16, 64]

all_hidden_size = []
for div in divisibility:
    for hidden_size in HIDDEN_SIZES:
        all_hidden_size.append(hidden_size // div)

HIDDEN_SIZES = list(set(all_hidden_size))

# Test params that focuses on hidden_size variation.
hs_test_params = {
    "hidden_sizes": HIDDEN_SIZES,
    "batches": [4],
    "num_loras": [4],
    "max_ranks": [32],
}

# General tests params that tests for variations in all dimensions
# except hidden_size.
test_params = {
    "hidden_sizes": [2049],
    "batches": [1, 4, 16, 32],
    "num_loras": [1, 8, 32, 128],
    "max_ranks": [1, 4, 8, 16, 32, 64, 128, 256],
}

DTYPES = [torch.float16, torch.bfloat16]
DEVICES = [f"{DEVICE_TYPE}:{0}"]
SEED = [0]


@pytest.mark.parametrize("batches", test_params["batches"])
@pytest.mark.parametrize("num_loras", test_params["num_loras"])
@pytest.mark.parametrize("rank", test_params["max_ranks"])
@pytest.mark.parametrize("hidden_size", test_params["hidden_sizes"])
@pytest.mark.parametrize("nslices", [1, 2, 3])
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("seed", SEED)
@pytest.mark.parametrize("op_type", ["shrink", "expand"])
def test_kernels(
    batches: int,
    num_loras: int,
    rank: int,
    hidden_size: int,
    nslices: int,
    dtype: torch.dtype,
    device: str,
    seed: int,
    op_type: str,
):
    """
    Tests LoRA kernels.
    """
    torch.set_default_device(device)
    torch.accelerator.set_device_index(device)
    set_random_seed(seed)

    if op_type == "shrink":
        check_lora_shrink_kernel(
            batches=batches,
            num_loras=num_loras,
            rank=rank,
            hidden_size=hidden_size,
            nslices=nslices,
            dtype=dtype,
            device=device,
            seq_length=128,
            scaling=0.5,
        )
    else:
        check_lora_expand_kernel(
            batches=batches,
            num_loras=num_loras,
            rank=rank,
            hidden_size=hidden_size,
            nslices=nslices,
            dtype=dtype,
            device=device,
            seq_length=128,
            add_inputs=True,
        )


@pytest.mark.parametrize("batches", hs_test_params["batches"])
@pytest.mark.parametrize("num_loras", hs_test_params["num_loras"])
@pytest.mark.parametrize("rank", hs_test_params["max_ranks"])
@pytest.mark.parametrize("hidden_size", hs_test_params["hidden_sizes"])
@pytest.mark.parametrize("nslices", [1, 2, 3])
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("seed", SEED)
@pytest.mark.parametrize("op_type", ["shrink", "expand"])
def test_kernels_hidden_size(
    batches: int,
    num_loras: int,
    rank: int,
    hidden_size: int,
    nslices: int,
    dtype: torch.dtype,
    device: str,
    seed: int,
    op_type: str,
):
    """
    Tests SGMV and LoRA kernels.
    """
    torch.set_default_device(device)
    torch.accelerator.set_device_index(device)
    set_random_seed(seed)

    if op_type == "shrink":
        check_lora_shrink_kernel(
            batches=batches,
            num_loras=num_loras,
            rank=rank,
            hidden_size=hidden_size,
            nslices=nslices,
            dtype=dtype,
            device=device,
            seq_length=128,
            scaling=0.5,
        )
    else:
        check_lora_expand_kernel(
            batches=batches,
            num_loras=num_loras,
            rank=rank,
            hidden_size=hidden_size,
            nslices=nslices,
            dtype=dtype,
            device=device,
            seq_length=128,
            add_inputs=True,
        )
