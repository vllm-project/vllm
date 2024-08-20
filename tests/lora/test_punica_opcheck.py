"""
This script is used to run torch.library.opcheck on the punica kernels.
"""
import random

import pytest
import torch

from vllm.lora.ops.bgmv_expand import bgmv_expand
from vllm.lora.ops.bgmv_expand_slice import bgmv_expand_slice
from vllm.lora.ops.bgmv_shrink import bgmv_shrink
from vllm.lora.ops.sgmv_expand import sgmv_expand
from vllm.lora.ops.sgmv_expand_slice import sgmv_expand_slice
from vllm.lora.ops.sgmv_shrink import sgmv_shrink

from ..utils import opcheck
from .utils import generate_data, generate_data_for_expand_nslices

HIDDEN_SIZES = [
    128,
    1024,
]
#The size of TP
divisibility = [1, 8, 16]

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
CUDA_DEVICES = [f"cuda:{0}"]


@pytest.mark.parametrize("batches", BATCHES)
@pytest.mark.parametrize("num_loras", NUM_LORA)
@pytest.mark.parametrize("rank", MAX_RANKS)
@pytest.mark.parametrize("hidden_size", HIDDEN_SIZES)
@pytest.mark.parametrize("scaling", SCALES)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("op_type", ["shrink", "expand"])
@pytest.mark.parametrize("seed", SEED)
@pytest.mark.parametrize("device", CUDA_DEVICES)
def test_punica_sgmv(
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
    random.seed(seed)
    torch.set_default_device(device)
    torch.random.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    seq_length = 128
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
    max_seq_length = seq_len_tensor.max()
    if isinstance(max_seq_length, tuple):
        max_seq_length = max_seq_length[0].item()
    else:
        max_seq_length = max_seq_length.item()
    if op_type == "shrink":
        opcheck(sgmv_shrink, (
            inputs_tensor,
            lora_weights,
            our_out_tensor,
            b_seq_start_loc,
            seq_len_tensor,
            lora_indices_tensor,
            batches,
            max_seq_length,
            scaling,
        ))
    else:
        opcheck(sgmv_expand,
                (inputs_tensor, lora_weights, our_out_tensor, b_seq_start_loc,
                 seq_len_tensor, lora_indices_tensor, batches, max_seq_length),
                {
                    'add_inputs': True,
                })


@pytest.mark.parametrize("batches", BATCHES)
@pytest.mark.parametrize("num_loras", NUM_LORA)
@pytest.mark.parametrize("rank", MAX_RANKS)
@pytest.mark.parametrize("hidden_size", HIDDEN_SIZES)
@pytest.mark.parametrize("scaling", SCALES)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("op_type", ["shrink", "expand"])
@pytest.mark.parametrize("seed", SEED)
@pytest.mark.parametrize("device", CUDA_DEVICES)
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
    random.seed(seed)
    torch.set_default_device(device)
    torch.random.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

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
        opcheck(bgmv_shrink, (
            inputs_tensor,
            lora_weights,
            our_out_tensor,
            indices,
            scaling,
        ))
    else:
        opcheck(
            bgmv_expand,
            (inputs_tensor, lora_weights, our_out_tensor, indices),
            {'add_inputs': True},
        )


@pytest.mark.parametrize("batches", BATCHES)
@pytest.mark.parametrize("num_loras", NUM_LORA)
@pytest.mark.parametrize("rank", MAX_RANKS)
@pytest.mark.parametrize("hidden_size", HIDDEN_SIZES)
@pytest.mark.parametrize("nslices", [2, 3])
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("op_type", ["sgmv", "bgmv"])
@pytest.mark.parametrize("seed", SEED)
@pytest.mark.parametrize("device", CUDA_DEVICES)
def test_punica_expand_nslices(
    batches: int,
    num_loras: int,
    rank: int,
    hidden_size: int,
    nslices: int,
    dtype: torch.dtype,
    op_type: str,
    seed: int,
    device: str,
):
    random.seed(seed)
    torch.set_default_device(device)
    torch.random.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    seq_length = 128 if op_type == "sgmv" else 1
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
    max_seq_length = seq_len_tensor.max()
    if isinstance(max_seq_length, tuple):
        max_seq_length = max_seq_length[0].item()
    else:
        max_seq_length = max_seq_length.item()
    slice_offset = 0
    for index in range(nslices):
        lora_weights = lora_weights_lst[index]
        if op_type == "sgmv":
            opcheck(
                sgmv_expand_slice,
                (inputs_tensor, lora_weights, our_outputs, b_seq_start_loc,
                 seq_len_tensor, lora_indices_tensor, batches, max_seq_length,
                 slice_offset, hidden_size),
                {'add_inputs': True},
            )
        else:
            opcheck(
                bgmv_expand_slice,
                (
                    inputs_tensor,
                    lora_weights,
                    our_outputs,
                    indices,
                    slice_offset,
                ),
                {
                    'slice_size': hidden_size,
                    'add_inputs': True
                },
            )

        slice_offset += hidden_size
