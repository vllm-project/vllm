# SPDX-License-Identifier: Apache-2.0
from threading import Lock

import pytest
import torch

import vllm.lora.ops.triton_ops  # noqa: F401
from vllm.lora.ops.torch_ops import (bgmv_expand, bgmv_expand_slice,
                                     bgmv_shrink, sgmv_expand,
                                     sgmv_expand_slice, sgmv_shrink)
from vllm.lora.ops.triton_ops.utils import _LORA_A_PTR_DICT, _LORA_B_PTR_DICT
from vllm.platforms import current_platform

from .utils import (PunicaTensors, assert_close, generate_data,
                    generate_data_for_expand_nslices,
                    generate_data_for_nslices)


# Utility reference implementations for DoRA operations
def apply_dora_norm_magnitudes(lora_a: torch.Tensor, lora_b: torch.Tensor,
                               magnitude_param: torch.Tensor) -> torch.Tensor:
    """
    Apply DoRA normalization and magnitude scaling to LoRA weights.
    
    Args:
        lora_a: The LoRA A weights [input_dim, rank]
        lora_b: The LoRA B weights [rank, output_dim]
        magnitude_param: The DoRA magnitude parameters [output_dim]
        
    Returns:
        The result after applying DoRA transformation
    """
    # In DoRA, the product of lora_a and lora_b is normalized column-wise
    # and then scaled by the magnitude parameter

    # Compute lora_a @ lora_b
    lora_product = torch.matmul(lora_a, lora_b)

    # Normalize the columns of the product
    # For each output dimension, we normalize the weights coming from each rank
    norm = torch.norm(lora_product, dim=0, keepdim=True)
    normalized_product = lora_product / (
        norm + 1e-5)  # Add epsilon for numerical stability

    # Scale each column by the corresponding magnitude parameter
    magnitude_scaled = normalized_product * magnitude_param.view(1, -1)

    return magnitude_scaled


# Utility shrink and expand operations used as reference implementations.
def sgmv_shrink_for_nslices(
        nslices: int, inputs_tensor: torch.Tensor,
        lora_weights_lst: list[torch.Tensor], out_tensor: torch.Tensor,
        b_seq_start_loc: torch.Tensor, seq_len_tensor: torch.Tensor,
        prompt_lora_mapping: torch.Tensor, batches: int, max_seq_length: int,
        num_tokens: int, scaling: float):
    """
    Wrapper around sgmv_shrink that handles any nslices.
    """
    for index in range(nslices):
        sgmv_shrink(
            inputs_tensor,
            lora_weights_lst[index],
            out_tensor[index],
            b_seq_start_loc,
            seq_len_tensor,
            prompt_lora_mapping,
            batches,
            max_seq_length,
            num_tokens,
            scaling,
        )


def sgmv_expand_for_nslices(nslices: int, hidden_size: int,
                            inputs_tensor: torch.Tensor,
                            lora_weights_lst: list[torch.Tensor],
                            out_tensor: torch.Tensor,
                            b_seq_start_loc: torch.Tensor,
                            seq_len_tensor: torch.Tensor,
                            prompt_lora_mapping: torch.Tensor, batches: int,
                            max_seq_length: int, num_tokens: int,
                            add_inputs: bool) -> None:
    """
    Wrapper around sgmv_expand that handles any nslices.
    """
    if nslices == 1:
        # Verify the torch's sgmv_expand op
        sgmv_expand(
            inputs_tensor[0],
            lora_weights_lst[0],
            out_tensor,
            b_seq_start_loc,
            seq_len_tensor,
            prompt_lora_mapping,
            batches,
            max_seq_length,
            num_tokens,
            add_inputs=add_inputs,
        )
    else:
        slice_offset = 0
        for index in range(nslices):
            lora_weights = lora_weights_lst[index]
            sgmv_expand_slice(
                inputs_tensor[index],
                lora_weights,
                out_tensor,
                b_seq_start_loc,
                seq_len_tensor,
                prompt_lora_mapping,
                batches,
                max_seq_length,
                num_tokens,
                slice_offset,
                hidden_size,
                add_inputs=add_inputs,
            )
            slice_offset += hidden_size


_dict_lock = Lock()


def check_sgmv_shrink(batches: int, num_loras: int, rank: int,
                      hidden_size: int, nslices: int, dtype: torch.dtype,
                      device: str, seq_length: int, scaling: float):
    """
    Compare outputs of vllm.sgmv_shrink kernel against a reference
    implementation.
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

    # Preventing cache error pointer.
    with _dict_lock:
        _LORA_A_PTR_DICT.clear()
        torch.ops.vllm.sgmv_shrink(
            data.inputs_tensor,
            data.lora_weights,
            data.our_out_tensor,
            data.b_seq_start_loc,
            data.seq_len_tensor,
            data.prompt_lora_mapping,
            batches,
            max_seq_length,
            token_nums,
            scaling,
        )

        sgmv_shrink_for_nslices(
            nslices,
            data.inputs_tensor,
            data.lora_weights,
            data.ref_out_tensor,
            data.b_seq_start_loc,
            data.seq_len_tensor,
            data.prompt_lora_mapping,
            batches,
            max_seq_length,
            token_nums,
            scaling,
        )
    assert_close(data.our_out_tensor, data.ref_out_tensor)


def check_sgmv_expand(batches: int, num_loras: int, rank: int,
                      hidden_size: int, nslices: int, dtype: torch.dtype,
                      device: str, seq_length: int, add_inputs: bool):
    """
    Compare outputs of vllm.sgmv_expand kernel against a reference
    implementation.
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

    with _dict_lock:
        _LORA_B_PTR_DICT.clear()
        torch.ops.vllm.sgmv_expand(
            data.inputs_tensor,
            data.lora_weights,
            data.our_out_tensor,
            data.b_seq_start_loc,
            data.seq_len_tensor,
            data.prompt_lora_mapping,
            batches,
            max_seq_length,
            token_nums,
            offset_start=0,
            add_inputs=add_inputs,
        )

    sgmv_expand_for_nslices(nslices,
                            hidden_size,
                            data.inputs_tensor,
                            data.lora_weights,
                            data.ref_out_tensor,
                            data.b_seq_start_loc,
                            data.seq_len_tensor,
                            data.prompt_lora_mapping,
                            batches,
                            max_seq_length,
                            token_nums,
                            add_inputs=add_inputs)

    assert_close(data.our_out_tensor, data.ref_out_tensor)


def check_bgmv_shrink(batches: int, num_loras: int, rank: int,
                      hidden_size: int, dtype: torch.dtype, device: str,
                      scaling: float):
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

    torch.ops.vllm.bgmv_shrink(
        data.inputs_tensor,
        data.lora_weights,
        data.our_out_tensor,
        data.token_lora_mapping,
        scaling,
    )

    bgmv_shrink(
        data.inputs_tensor,
        data.lora_weights,
        data.ref_out_tensor,
        data.token_lora_mapping,
        scaling,
    )

    data.ref_out_tensor = data.ref_out_tensor.to(torch.float32)
    assert_close(data.our_out_tensor, data.ref_out_tensor)


def check_bgmv_expand(batches: int, num_loras: int, rank: int,
                      hidden_size: int, dtype: torch.dtype, device: str,
                      add_inputs: bool):
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

    torch.ops.vllm.bgmv_expand(
        data.inputs_tensor,
        data.lora_weights,
        data.our_out_tensor,
        data.token_lora_mapping,
        add_inputs=add_inputs,
    )
    bgmv_expand(
        data.inputs_tensor,
        data.lora_weights,
        data.ref_out_tensor,
        data.token_lora_mapping,
        add_inputs=add_inputs,
    )
    assert_close(data.our_out_tensor, data.ref_out_tensor)


def check_bgmv_expand_slice(batches: int, num_loras: int, rank: int,
                            hidden_size: int, nslices: int, dtype: torch.dtype,
                            device: str, add_inputs: bool):
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
        torch.ops.vllm.bgmv_expand_slice(
            data.inputs_tensor,
            data.lora_weights[index],
            data.our_out_tensor,
            data.token_lora_mapping,
            slice_offset,
            slice_size=hidden_size,
            add_inputs=add_inputs,
        )
        bgmv_expand_slice(
            data.inputs_tensor,
            data.lora_weights[index],
            data.ref_out_tensor,
            data.token_lora_mapping,
            slice_offset,
            slice_size=hidden_size,
            add_inputs=add_inputs,
        )

        slice_offset += hidden_size
    assert_close(data.our_out_tensor, data.ref_out_tensor)


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
#The size of TP
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
DEVICES = [f"cuda:{0}"]
SEED = [0]


@pytest.mark.parametrize("batches", test_params['batches'])
@pytest.mark.parametrize("num_loras", test_params['num_loras'])
@pytest.mark.parametrize("rank", test_params['max_ranks'])
@pytest.mark.parametrize("hidden_size", test_params['hidden_sizes'])
@pytest.mark.parametrize("nslices", [1, 2, 3])
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("seed", SEED)
@pytest.mark.parametrize("op_type", ["shrink", "expand"])
def test_punica_sgmv(
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
    torch.set_default_device(device)
    current_platform.seed_everything(seed)

    if op_type == "shrink":
        check_sgmv_shrink(batches=batches,
                          num_loras=num_loras,
                          rank=rank,
                          hidden_size=hidden_size,
                          nslices=nslices,
                          dtype=dtype,
                          device=device,
                          seq_length=128,
                          scaling=0.5)
    else:
        check_sgmv_expand(batches=batches,
                          num_loras=num_loras,
                          rank=rank,
                          hidden_size=hidden_size,
                          nslices=nslices,
                          dtype=dtype,
                          device=device,
                          seq_length=128,
                          add_inputs=True)


@pytest.mark.parametrize("batches", hs_test_params['batches'])
@pytest.mark.parametrize("num_loras", hs_test_params['num_loras'])
@pytest.mark.parametrize("rank", hs_test_params['max_ranks'])
@pytest.mark.parametrize("hidden_size", hs_test_params['hidden_sizes'])
@pytest.mark.parametrize("nslices", [1, 2, 3])
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("seed", SEED)
@pytest.mark.parametrize("op_type", ["shrink", "expand"])
def test_punica_sgmv_hidden_size(
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
    torch.set_default_device(device)
    current_platform.seed_everything(seed)

    if op_type == "shrink":
        check_sgmv_shrink(batches=batches,
                          num_loras=num_loras,
                          rank=rank,
                          hidden_size=hidden_size,
                          nslices=nslices,
                          dtype=dtype,
                          device=device,
                          seq_length=128,
                          scaling=0.5)
    else:
        check_sgmv_expand(batches=batches,
                          num_loras=num_loras,
                          rank=rank,
                          hidden_size=hidden_size,
                          nslices=nslices,
                          dtype=dtype,
                          device=device,
                          seq_length=128,
                          add_inputs=True)


@pytest.mark.parametrize("batches", test_params['batches'])
@pytest.mark.parametrize("num_loras", test_params['num_loras'])
@pytest.mark.parametrize("rank", test_params['max_ranks'])
@pytest.mark.parametrize("hidden_size", test_params['hidden_sizes'])
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("seed", SEED)
@pytest.mark.parametrize("op_type", ["shrink", "expand"])
def test_punica_bgmv(
    batches: int,
    num_loras: int,
    rank: int,
    hidden_size: int,
    dtype: torch.dtype,
    device: str,
    seed: int,
    op_type: str,
):
    torch.set_default_device(device)
    current_platform.seed_everything(seed)

    if op_type == "shrink":
        check_bgmv_shrink(batches=batches,
                          num_loras=num_loras,
                          rank=rank,
                          hidden_size=hidden_size,
                          dtype=dtype,
                          device=device,
                          scaling=0.5)
    else:
        check_bgmv_expand(batches=batches,
                          num_loras=num_loras,
                          rank=rank,
                          hidden_size=hidden_size,
                          dtype=dtype,
                          device=device,
                          add_inputs=True)


@pytest.mark.parametrize("batches", hs_test_params['batches'])
@pytest.mark.parametrize("num_loras", hs_test_params['num_loras'])
@pytest.mark.parametrize("rank", hs_test_params['max_ranks'])
@pytest.mark.parametrize("hidden_size", hs_test_params['hidden_sizes'])
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("seed", SEED)
@pytest.mark.parametrize("op_type", ["shrink", "expand"])
def test_punica_bgmv_hidden_size(
    batches: int,
    num_loras: int,
    rank: int,
    hidden_size: int,
    dtype: torch.dtype,
    device: str,
    seed: int,
    op_type: str,
):
    torch.set_default_device(device)
    current_platform.seed_everything(seed)

    if op_type == "shrink":
        check_bgmv_shrink(batches=batches,
                          num_loras=num_loras,
                          rank=rank,
                          hidden_size=hidden_size,
                          dtype=dtype,
                          device=device,
                          scaling=0.5)
    else:
        check_bgmv_expand(batches=batches,
                          num_loras=num_loras,
                          rank=rank,
                          hidden_size=hidden_size,
                          dtype=dtype,
                          device=device,
                          add_inputs=True)


@pytest.mark.parametrize("batches", test_params['batches'])
@pytest.mark.parametrize("num_loras", test_params['num_loras'])
@pytest.mark.parametrize("rank", test_params['max_ranks'])
@pytest.mark.parametrize("hidden_size", test_params['hidden_sizes'])
@pytest.mark.parametrize("nslices", [2, 3])
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("seed", SEED)
def test_punica_bgmv_expand_nslices(batches: int, num_loras: int, rank: int,
                                    hidden_size: int, nslices: int,
                                    dtype: torch.dtype, device: str,
                                    seed: int):

    torch.set_default_device(device)
    current_platform.seed_everything(seed)

    check_bgmv_expand_slice(batches=batches,
                            num_loras=num_loras,
                            rank=rank,
                            hidden_size=hidden_size,
                            nslices=nslices,
                            dtype=dtype,
                            device=device,
                            add_inputs=True)


@pytest.mark.parametrize("batches", hs_test_params['batches'])
@pytest.mark.parametrize("num_loras", hs_test_params['num_loras'])
@pytest.mark.parametrize("rank", hs_test_params['max_ranks'])
@pytest.mark.parametrize("hidden_size", hs_test_params['hidden_sizes'])
@pytest.mark.parametrize("nslices", [2, 3])
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("seed", SEED)
def test_punica_bgmv_expand_nslices_hidden_size(batches: int, num_loras: int,
                                                rank: int, hidden_size: int,
                                                nslices: int,
                                                dtype: torch.dtype,
                                                device: str, seed: int):

    torch.set_default_device(device)
    current_platform.seed_everything(seed)

    check_bgmv_expand_slice(batches=batches,
                            num_loras=num_loras,
                            rank=rank,
                            hidden_size=hidden_size,
                            nslices=nslices,
                            dtype=dtype,
                            device=device,
                            add_inputs=True)


########################### DoRA specific tests ##########################


def test_dora_normalization():
    """Test that DoRA normalization and magnitude scaling works as expected."""
    # Create random LoRA weights and magnitude parameters
    input_dim = 64
    rank = 16
    output_dim = 32
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Create random LoRA weights
    lora_a = torch.rand((input_dim, rank), device=device)
    lora_b = torch.rand((rank, output_dim), device=device)
    # Magnitude param should match output_dim, not rank
    magnitude_param = torch.rand((output_dim, ), device=device)

    # Compute the DoRA transformation
    dora_output = apply_dora_norm_magnitudes(lora_a, lora_b, magnitude_param)

    # Manual verification:
    # 1. Compute lora_a @ lora_b
    lora_product = torch.matmul(lora_a, lora_b)

    # 2. Normalize column-wise
    norm = torch.norm(lora_product, dim=0, keepdim=True)
    normalized_product = lora_product / (norm + 1e-5)

    # 3. Apply magnitude scaling
    expected_output = normalized_product * magnitude_param.view(1, -1)

    # Check that our implementation matches the expected output
    assert torch.allclose(dora_output, expected_output, rtol=1e-5, atol=1e-5)

    # Check that the column norms are approximately equal to the magnitude parameters
    actual_col_norms = torch.norm(dora_output, dim=0)
    assert torch.allclose(actual_col_norms,
                          magnitude_param,
                          rtol=1e-4,
                          atol=1e-4)


@pytest.mark.parametrize("device",
                         ["cuda"] if torch.cuda.is_available() else ["cpu"])
def test_dora_vs_lora_performance(device):
    """Compare the performance of DoRA vs regular LoRA."""
    # Set up dimensions
    input_dim = 4096
    rank = 32
    output_dim = 32  # Changed to match magnitude param dimension
    batch_size = 8

    # Create input tensor
    x = torch.rand((batch_size, input_dim), device=device)

    # Create LoRA weights
    lora_a = torch.rand((input_dim, rank), device=device)
    lora_b = torch.rand((rank, output_dim), device=device)

    # Create DoRA magnitude parameters - should match output_dim
    magnitude_param = torch.rand((output_dim, ), device=device)

    # Regular LoRA forward pass (no magnitude normalization)
    def lora_forward():
        return torch.matmul(x, torch.matmul(lora_a, lora_b))

    # DoRA forward pass (with magnitude normalization)
    def dora_forward():
        lora_product = torch.matmul(lora_a, lora_b)
        norm = torch.norm(lora_product, dim=0, keepdim=True)
        normalized_product = lora_product / (norm + 1e-5)
        magnitude_scaled = normalized_product * magnitude_param.view(1, -1)
        return torch.matmul(x, magnitude_scaled)

    # Warm up
    for _ in range(5):
        lora_forward()
        dora_forward()

    # Test performance
    import time

    # Measure LoRA performance
    torch.cuda.synchronize() if device == "cuda" else None
    start_time = time.time()
    for _ in range(100):
        lora_result = lora_forward()
    torch.cuda.synchronize() if device == "cuda" else None
    lora_time = time.time() - start_time

    # Measure DoRA performance
    torch.cuda.synchronize() if device == "cuda" else None
    start_time = time.time()
    for _ in range(100):
        dora_result = dora_forward()
    torch.cuda.synchronize() if device == "cuda" else None
    dora_time = time.time() - start_time

    # Print performance comparison
    print(f"Regular LoRA time: {lora_time:.6f} seconds for 100 iterations")
    print(f"DoRA time: {dora_time:.6f} seconds for 100 iterations")
    print(f"Overhead: {(dora_time/lora_time - 1)*100:.2f}%")

    # DoRA is expected to have some overhead (acceptable up to 200%)
    assert dora_time < lora_time * 3.0, "DoRA overhead too high"


# Test implementing DoRA with packed lora weights
@pytest.mark.parametrize("device",
                         ["cuda"] if torch.cuda.is_available() else ["cpu"])
def test_dora_with_packed_weights(device):
    """Test DoRA with packed lora weights."""
    from vllm.lora.lora import LoRALayerWeights, PackedLoRALayerWeights

    # Set up dimensions
    input_dim = 128
    rank = 8
    output_dim = 64

    # Create two LoRA weights with magnitude parameters (DoRA)
    lora1 = LoRALayerWeights(
        module_name="module1",
        rank=rank,
        lora_alpha=1,
        lora_a=torch.rand((input_dim, rank), device=device),
        lora_b=torch.rand((rank, output_dim), device=device),
        magnitude_param=torch.rand((output_dim, ),
                                   device=device),  # Should match output_dim
    )

    lora2 = LoRALayerWeights(
        module_name="module2",
        rank=rank,
        lora_alpha=1,
        lora_a=torch.rand((input_dim, rank), device=device),
        lora_b=torch.rand((rank, output_dim), device=device),
        magnitude_param=torch.rand((output_dim, ),
                                   device=device),  # Should match output_dim
    )

    # Pack the LoRA weights
    packed_lora = PackedLoRALayerWeights.pack([lora1, lora2])

    # Check that the packed LoRA weights have the magnitude parameters
    assert packed_lora.magnitude_param is not None

    # Verify the shape of the magnitude parameters
    assert isinstance(packed_lora.magnitude_param, list)
    assert len(packed_lora.magnitude_param) == 2
    assert packed_lora.magnitude_param[0].shape == (
        output_dim, )  # Should match output_dim
    assert packed_lora.magnitude_param[1].shape == (
        output_dim, )  # Should match output_dim


@pytest.mark.parametrize("batches", [1, 2])
@pytest.mark.parametrize("num_loras", [1, 2])
@pytest.mark.parametrize("rank", [8])
@pytest.mark.parametrize("hidden_size", [128])
@pytest.mark.parametrize("device",
                         ["cuda"] if torch.cuda.is_available() else ["cpu"])
def test_dora_in_lora_ops(batches: int, num_loras: int, rank: int,
                          hidden_size: int, device: str):
    """Test how DoRA would be used in LoRA operations."""
    # Create test data like a regular LoRA test
    torch.set_default_device(device)

    # Create test data
    seq_length = 4
    seq_len_tensor = torch.ones(batches, dtype=torch.long) * seq_length
    b_seq_start_loc = torch.cumsum(
        torch.tensor([0] + seq_len_tensor[:-1].tolist(), dtype=torch.long),
        dim=0,
    )
    total_tokens = seq_len_tensor.sum()

    # Create input tensor
    inputs_tensor = torch.rand((total_tokens, hidden_size))

    # We need to transpose the weight matrices for matrix multiplication
    # Matrix A: [input_dim, rank]
    # Matrix B: [rank, output_dim]
    lora_a_weights = torch.rand(
        (num_loras, hidden_size, rank))  # [hidden_size, rank] for each lora
    lora_b_weights = torch.rand(
        (num_loras, rank, hidden_size))  # [rank, hidden_size] for each lora

    # DoRA magnitude parameters - should match output dimension (hidden_size)
    magnitude_params = torch.rand((num_loras, hidden_size))

    # Create output tensors for regular LoRA and DoRA
    ref_lora_out_tensor = torch.zeros((total_tokens, hidden_size))
    ref_dora_out_tensor = torch.zeros((total_tokens, hidden_size))

    # Create lora indices tensor
    lora_indices_tensor = torch.randint(0, num_loras, (batches, ))

    # Expand indices to token level
    exploded_indices = torch.repeat_interleave(lora_indices_tensor,
                                               seq_len_tensor)

    # Implement regular LoRA for reference
    for i in range(total_tokens):
        lora_idx = exploded_indices[i].item()
        lora_a = lora_a_weights[lora_idx]  # [hidden_size, rank]
        lora_b = lora_b_weights[lora_idx]  # [rank, hidden_size]

        # For regular LoRA: x @ (A @ B)
        # First compute A @ B to get [hidden_size, hidden_size]
        lora_product = torch.matmul(lora_a, lora_b)

        # Then apply to input: [1, hidden_size] @ [hidden_size, hidden_size] = [1, hidden_size]
        token_input = inputs_tensor[i].unsqueeze(0)  # [1, hidden_size]
        lora_output = torch.matmul(token_input,
                                   lora_product)  # [1, hidden_size]
        ref_lora_out_tensor[i] = lora_output

    # Implement DoRA for reference
    for i in range(total_tokens):
        lora_idx = exploded_indices[i].item()
        lora_a = lora_a_weights[lora_idx]  # [hidden_size, rank]
        lora_b = lora_b_weights[lora_idx]  # [rank, hidden_size]
        magnitude = magnitude_params[lora_idx]  # [hidden_size]

        # For DoRA: x @ norm(A @ B) * magnitude
        lora_product = torch.matmul(lora_a,
                                    lora_b)  # [hidden_size, hidden_size]

        # Normalize column-wise
        norm = torch.norm(lora_product, dim=0, keepdim=True)
        normalized_product = lora_product / (norm + 1e-5)

        # Scale by magnitudes
        magnitude_scaled = normalized_product * magnitude.unsqueeze(0)

        # Apply to input
        token_input = inputs_tensor[i].unsqueeze(0)  # [1, hidden_size]
        dora_output = torch.matmul(token_input,
                                   magnitude_scaled)  # [1, hidden_size]
        ref_dora_out_tensor[i] = dora_output

    # Compare the outputs - they should be different due to normalization
    norm_diff = torch.norm(ref_lora_out_tensor - ref_dora_out_tensor).item()
    print(f"Norm difference between LoRA and DoRA outputs: {norm_diff:.6f}")
    assert norm_diff > 0.01, "LoRA and DoRA outputs should differ significantly"

    # Verify that the DoRA outputs have reasonable values
    dora_output_norm = torch.norm(ref_dora_out_tensor).item()
    assert dora_output_norm > 0, "DoRA output should have non-zero norm"
