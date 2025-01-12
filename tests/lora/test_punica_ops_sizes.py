"""
This script is mainly used to tests various hidden_sizes. We have collected the
hidden_sizes included in the LoRA models currently supported by vLLM. It tests
whether the corresponding Triton kernel can run normally when tensor parallelism
is set to [1, 2, 4, 8, 16, 32, 64].
"""
from threading import Lock

import pytest
import torch

import vllm.lora.ops.triton_ops  # noqa: F401
from vllm.lora.ops.torch_ops import (bgmv_expand, bgmv_expand_slice,
                                     bgmv_shrink, sgmv_expand,
                                     sgmv_expand_slice, sgmv_shrink)
from vllm.lora.ops.triton_ops.utils import _LORA_A_PTR_DICT, _LORA_B_PTR_DICT
from vllm.platforms import current_platform

from .utils import (assert_close, generate_data,
                    generate_data_for_expand_nslices,
                    generate_data_for_nslices)

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

BATCHES = [4]
NUM_LORA = [4]
DTYPES = [torch.float16, torch.bfloat16]
MAX_RANKS = [32]
SCALES = [0.5]
SEED = [0]
DEVICES = [f"cuda:{0}"]

_dict_lock = Lock()


@pytest.mark.parametrize("batches", BATCHES)
@pytest.mark.parametrize("num_loras", NUM_LORA)
@pytest.mark.parametrize("rank", MAX_RANKS)
@pytest.mark.parametrize("hidden_size", HIDDEN_SIZES)
@pytest.mark.parametrize("scaling", SCALES)
@pytest.mark.parametrize("nslices", [1, 2, 3])
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("op_type", ["shrink", "expand"])
@pytest.mark.parametrize("seed", SEED)
@pytest.mark.parametrize("device", DEVICES)
def test_punica_sgmv(
    batches: int,
    num_loras: int,
    rank: int,
    hidden_size: int,
    scaling: float,
    nslices: int,
    dtype: torch.dtype,
    op_type: str,
    seed: int,
    device: str,
):
    torch.set_default_device(device)
    current_platform.seed_everything(seed)

    seq_length = 128
    (
        inputs_tensor,
        lora_weights_lst,
        our_out_tensor,
        ref_out_tensor,
        b_seq_start_loc,
        lora_indices_tensor,
        seq_len_tensor,
        indices,
    ) = generate_data_for_nslices(
        batches,
        hidden_size,
        num_loras,
        rank,
        seq_length,
        nslices,
        dtype,
        op_type,
        device,
    )
    max_seq_length = seq_len_tensor.max()
    token_nums = seq_len_tensor.sum().item()
    if isinstance(max_seq_length, tuple):
        max_seq_length = max_seq_length[0].item()
    else:
        max_seq_length = max_seq_length.item()
    if op_type == "shrink":
        # Preventing cache error pointer.
        with _dict_lock:
            _LORA_A_PTR_DICT.clear()
            torch.ops.vllm.sgmv_shrink(
                inputs_tensor,
                lora_weights_lst,
                our_out_tensor,
                b_seq_start_loc,
                seq_len_tensor,
                lora_indices_tensor,
                batches,
                max_seq_length,
                token_nums,
                scaling,
            )
        for index in range(nslices):
            sgmv_shrink(
                inputs_tensor,
                lora_weights_lst[index],
                ref_out_tensor[index],
                b_seq_start_loc,
                seq_len_tensor,
                lora_indices_tensor,
                batches,
                max_seq_length,
                token_nums,
                scaling,
            )

    else:
        with _dict_lock:
            _LORA_B_PTR_DICT.clear()
            torch.ops.vllm.sgmv_expand(
                inputs_tensor,
                lora_weights_lst,
                our_out_tensor,
                b_seq_start_loc,
                seq_len_tensor,
                lora_indices_tensor,
                batches,
                max_seq_length,
                token_nums,
                offset_start=0,
                add_inputs=True,
            )
        if nslices == 1:
            # Verify the torch's sgmv_expand op
            sgmv_expand(
                inputs_tensor[0],
                lora_weights_lst[0],
                ref_out_tensor,
                b_seq_start_loc,
                seq_len_tensor,
                lora_indices_tensor,
                batches,
                max_seq_length,
                token_nums,
                add_inputs=True,
            )
        else:
            slice_offset = 0
            for index in range(nslices):
                lora_weights = lora_weights_lst[index]
                sgmv_expand_slice(
                    inputs_tensor[index],
                    lora_weights,
                    ref_out_tensor,
                    b_seq_start_loc,
                    seq_len_tensor,
                    lora_indices_tensor,
                    batches,
                    max_seq_length,
                    token_nums,
                    slice_offset,
                    hidden_size,
                    add_inputs=True,
                )
                slice_offset += hidden_size

    assert_close(our_out_tensor, ref_out_tensor)


@pytest.mark.parametrize("batches", BATCHES)
@pytest.mark.parametrize("num_loras", NUM_LORA)
@pytest.mark.parametrize("rank", MAX_RANKS)
@pytest.mark.parametrize("hidden_size", HIDDEN_SIZES)
@pytest.mark.parametrize("scaling", SCALES)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("op_type", ["shrink", "expand"])
@pytest.mark.parametrize("seed", SEED)
@pytest.mark.parametrize("device", DEVICES)
def test_punica_bgmv(
    batches: int,
    num_loras: int,
    rank: int,
    hidden_size: int,
    scaling: float,
    dtype: torch.dtype,
    op_type: str,
    seed: int,
    device: str,
):
    torch.set_default_device(device)
    current_platform.seed_everything(seed)

    seq_length = 1
    (
        inputs_tensor,
        lora_weights,
        our_out_tensor,
        ref_out_tensor,
        b_seq_start_loc,
        lora_indices_tensor,
        seq_len_tensor,
        indices,
    ) = generate_data(
        batches,
        hidden_size,
        num_loras,
        rank,
        seq_length,
        dtype,
        op_type,
        device,
    )
    if op_type == "shrink":
        torch.ops.vllm.bgmv_shrink(
            inputs_tensor,
            lora_weights,
            our_out_tensor,
            indices,
            scaling,
        )

        bgmv_shrink(
            inputs_tensor,
            lora_weights,
            ref_out_tensor,
            indices,
            scaling,
        )

    else:
        torch.ops.vllm.bgmv_expand(
            inputs_tensor,
            lora_weights,
            our_out_tensor,
            indices,
            add_inputs=True,
        )
        bgmv_expand(
            inputs_tensor,
            lora_weights,
            ref_out_tensor,
            indices,
            add_inputs=True,
        )

    if op_type == "shrink":
        ref_out_tensor = ref_out_tensor.to(torch.float32)
    assert_close(our_out_tensor, ref_out_tensor)


@pytest.mark.parametrize("batches", BATCHES)
@pytest.mark.parametrize("num_loras", NUM_LORA)
@pytest.mark.parametrize("rank", MAX_RANKS)
@pytest.mark.parametrize("hidden_size", HIDDEN_SIZES)
@pytest.mark.parametrize("nslices", [2, 3])
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("seed", SEED)
@pytest.mark.parametrize("device", DEVICES)
def test_punica_bgmv_expand_nslices(
    batches: int,
    num_loras: int,
    rank: int,
    hidden_size: int,
    nslices: int,
    dtype: torch.dtype,
    seed: int,
    device: str,
):
    torch.set_default_device(device)
    current_platform.seed_everything(seed)

    seq_length = 1
    (
        inputs_tensor,
        lora_weights_lst,
        our_outputs,
        ref_outputs,
        b_seq_start_loc,
        lora_indices_tensor,
        seq_len_tensor,
        indices,
    ) = generate_data_for_expand_nslices(
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
        lora_weights = lora_weights_lst[index]
        torch.ops.vllm.bgmv_expand_slice(
            inputs_tensor,
            lora_weights,
            our_outputs,
            indices,
            slice_offset,
            slice_size=hidden_size,
            add_inputs=True,
        )
        bgmv_expand_slice(
            inputs_tensor,
            lora_weights,
            ref_outputs,
            indices,
            slice_offset,
            slice_size=hidden_size,
            add_inputs=True,
        )

        slice_offset += hidden_size
    assert_close(our_outputs, ref_outputs)
