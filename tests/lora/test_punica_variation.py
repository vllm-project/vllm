"""
This script is mainly used to test whether trtion kernels can run normally
under different conditions, including various batches, numbers of LoRA , and
maximum ranks.
"""
from threading import Lock

import pytest
import torch

# Enable custom op register
import vllm.lora.ops.bgmv_expand
import vllm.lora.ops.bgmv_expand_slice
import vllm.lora.ops.bgmv_shrink
import vllm.lora.ops.sgmv_expand
import vllm.lora.ops.sgmv_shrink  # noqa: F401
from vllm.lora.ops.utils import _LORA_A_PTR_DICT, _LORA_B_PTR_DICT
from vllm.platforms import current_platform

from .utils import (assert_close, generate_data,
                    generate_data_for_expand_nslices,
                    generate_data_for_nslices, ref_torch_groupgemm)

HIDDEN_SIZES = [4097]

BATCHES = [1, 4, 16, 32]
NUM_LORA = [1, 8, 32, 128]
DTYPES = [torch.float16, torch.bfloat16]
MAX_RANKS = [1, 4, 8, 16, 32, 64, 128, 256]
SCALES = [0.5]
SEED = [0]
CUDA_DEVICES = [f"cuda:{0}"]

# Unlike test_punica_sizes.py, we directly utilize custom op for
# testing, which verifies the correct registration of these ops.
bgmv_expand = torch.ops.vllm.bgmv_expand
bgmv_expand_slice = torch.ops.vllm.bgmv_expand_slice
bgmv_shrink = torch.ops.vllm.bgmv_shrink
sgmv_expand = torch.ops.vllm.sgmv_expand
sgmv_shrink = torch.ops.vllm.sgmv_shrink

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
@pytest.mark.parametrize("device", CUDA_DEVICES)
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
            sgmv_shrink(
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
            ref_torch_groupgemm(
                ref_out_tensor[index],
                inputs_tensor,
                lora_weights_lst[index],
                lora_indices_tensor,
                seq_len_tensor,
                batches,
                scaling,
                op_type,
            )
    else:
        with _dict_lock:
            _LORA_B_PTR_DICT.clear()
            sgmv_expand(
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

        slice_offset = 0
        for index in range(nslices):
            lora_weights = lora_weights_lst[index]
            ref_torch_groupgemm(
                ref_out_tensor[:, slice_offset:slice_offset + hidden_size],
                inputs_tensor[index],
                lora_weights,
                lora_indices_tensor,
                seq_len_tensor,
                batches,
                1.0,
                op_type,
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
        bgmv_shrink(
            inputs_tensor,
            lora_weights,
            our_out_tensor,
            indices,
            scaling,
        )
    else:

        bgmv_expand(
            inputs_tensor,
            lora_weights,
            our_out_tensor,
            indices,
            add_inputs=True,
        )
    ref_torch_groupgemm(
        ref_out_tensor,
        inputs_tensor,
        lora_weights,
        lora_indices_tensor,
        seq_len_tensor,
        batches,
        scaling if op_type == "shrink" else 1.0,
        op_type,
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
@pytest.mark.parametrize("device", CUDA_DEVICES)
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
        bgmv_expand_slice(
            inputs_tensor,
            lora_weights,
            our_outputs,
            indices,
            slice_offset,
            slice_size=hidden_size,
            add_inputs=True,
        )
        ref_torch_groupgemm(
            ref_outputs[:, slice_offset:slice_offset + hidden_size],
            inputs_tensor,
            lora_weights,
            lora_indices_tensor,
            seq_len_tensor,
            batches,
            1.0,
            op_type="expand",
        )

        slice_offset += hidden_size
    assert_close(our_outputs, ref_outputs)
