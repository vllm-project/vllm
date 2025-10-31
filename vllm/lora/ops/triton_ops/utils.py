# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import functools
import json
from pathlib import Path
from typing import Any

import torch

from vllm import envs
from vllm.logger import init_logger

logger = init_logger(__name__)

_LORA_A_PTR_DICT: dict[tuple[int, ...], tuple[torch.tensor, ...]] = {}
_LORA_B_PTR_DICT: dict[tuple[int, ...], tuple[torch.tensor, ...]] = {}


def _get_lora_a_ptr(lora_a_weights: list[torch.Tensor], device: torch.device):
    """
    `_LORA_A_PTR_DICT` collects the required information during `profile_run`,
    After this, it remains constant and subsequent usage is through LUT.
    Refer to:
    https://github.com/triton-lang/triton/blob/release/3.1.x/python/tutorials/08-grouped-gemm.py
    """
    key = tuple(lora_weight.data_ptr() for lora_weight in lora_a_weights)

    if values := _LORA_A_PTR_DICT.get(key):
        return values

    lora_strides_d0 = []
    lora_strides_d1 = []
    lora_strides_d2 = []
    tensor_ptrs = []
    for lora_a_weight in lora_a_weights:
        if lora_a_weight.ndim == 4:  # shape:(lora_num,1,size,rank)
            assert lora_a_weight.size(1) == 1
            lora_a_weight = lora_a_weight.squeeze(dim=1)
        else:
            assert lora_a_weight.ndim == 3  # shape:(lora_num,size,rank)
        assert lora_a_weight.is_contiguous()
        tensor_ptrs.append(lora_a_weight.data_ptr())
        lora_strides_d0.append(lora_a_weight.stride(0))
        lora_strides_d1.append(lora_a_weight.stride(1))
        lora_strides_d2.append(lora_a_weight.stride(2))
    if len(lora_a_weights) > 1:
        lora_ptr_tensor = torch.tensor(tensor_ptrs, device=device, dtype=torch.uint64)
    else:
        lora_ptr_tensor = lora_a_weights[0]

    if (
        len(set(lora_strides_d0)) > 1
        or len(set(lora_strides_d1)) > 1
        or len(set(lora_strides_d2)) > 1
    ):
        raise ValueError("All LoRA weights must have the same stride.")

    _LORA_A_PTR_DICT[key] = (
        lora_ptr_tensor,
        lora_strides_d0[0],
        lora_strides_d1[0],
        lora_strides_d2[0],
    )
    return _LORA_A_PTR_DICT.get(key)


def _get_lora_b_ptr(
    lora_weights: list[torch.Tensor], offset_start: int, device: torch.device
):
    """
     `_LORA_B_PTR_DICT` collects the required information during `profile_run`,
    After this, it remains constant and subsequent usage is through LUT.
    Refer to:
    https://github.com/triton-lang/triton/blob/release/3.1.x/python/tutorials/08-grouped-gemm.py

    """

    key = tuple(lora_weight.data_ptr() for lora_weight in lora_weights)
    if values := _LORA_B_PTR_DICT.get(key):
        return values
    slice_offset_lst = []
    tensor_ptrs = []
    lora_strides_d0 = []
    lora_strides_d1 = []
    lora_strides_d2 = []
    hidden_sizes = []
    slice_offset = offset_start
    for lora_b_weight in lora_weights:
        if lora_b_weight.ndim == 4:  # shape:(lora_num,1,size,rank)
            assert lora_b_weight.size(1) == 1
            lora_b_weight = lora_b_weight.squeeze(dim=1)
        else:
            assert lora_b_weight.ndim == 3  # shape:(lora_num,size,rank)
        assert lora_b_weight.is_contiguous()
        tensor_ptrs.append(lora_b_weight.data_ptr())
        lora_strides_d0.append(lora_b_weight.stride(0))
        lora_strides_d1.append(lora_b_weight.stride(1))
        lora_strides_d2.append(lora_b_weight.stride(2))
        slice_offset_lst.append(slice_offset)
        slice_offset += lora_b_weight.size(1)
        hidden_sizes.append(lora_b_weight.size(1))

    if len(lora_weights) > 1:
        # note these are device tensors
        lora_ptr_tensor = torch.tensor(tensor_ptrs, device=device, dtype=torch.uint64)
        slice_start_tensor = torch.tensor(
            slice_offset_lst, device=device, dtype=torch.uint64
        )
    else:
        slice_start_tensor = slice_offset_lst[0]
        lora_ptr_tensor = lora_b_weight[0]

    # If each lora has the same stride, there's no need to use a
    # tensor for storage.
    if (
        len(set(lora_strides_d0)) == 1
        and len(set(lora_strides_d1)) == 1
        and len(set(lora_strides_d2)) == 1
    ) and len(set(hidden_sizes)) == 1:
        lora_strides_d0_tensor = lora_strides_d0[0]
        lora_strides_d1_tensor = lora_strides_d1[0]
        lora_strides_d2_tensor = lora_strides_d2[0]
        hidden_sizes_tensor = hidden_sizes[0]
        same_stride = True

    else:
        lora_strides_d0_tensor = torch.tensor(lora_strides_d0, device=device)
        lora_strides_d1_tensor = torch.tensor(lora_strides_d1, device=device)
        lora_strides_d2_tensor = torch.tensor(lora_strides_d2, device=device)
        hidden_sizes_tensor = torch.tensor(hidden_sizes, device=device)
        same_stride = False
    # MAX_N is the maximum hidden size among all the lora_b weights
    MAX_N = max(hidden_sizes)
    _LORA_B_PTR_DICT[key] = (
        slice_start_tensor,
        lora_ptr_tensor,
        lora_strides_d0_tensor,
        lora_strides_d1_tensor,
        lora_strides_d2_tensor,
        hidden_sizes_tensor,
        same_stride,
        MAX_N,
    )
    return _LORA_B_PTR_DICT.get(key)


@functools.lru_cache
def load_lora_op_config(op_type: str, add_inputs: bool | None) -> dict | None:
    user_defined_config_folder = envs.VLLM_TUNED_CONFIG_FOLDER
    if user_defined_config_folder is not None:
        gpu_name = torch.cuda.get_device_name()
        gpu_name = gpu_name.replace(" ", "_")
        gpu_name = gpu_name.replace("-", "_")

        config_fname = None
        if op_type == "shrink":
            config_fname = f"{gpu_name}_{op_type.upper()}.json"
        else:
            assert op_type == "expand"
            config_fname = (
                f"{gpu_name}_{op_type.upper()}_{str(add_inputs).upper()}.json"
            )

        config_path = Path(f"{user_defined_config_folder}/{config_fname}")
        if not config_path.exists():
            logger.warning_once(f"No LoRA kernel configs founded in {config_path}")
            return None

        # Load json
        logger.info_once(f"Using tuned LoRA kernel configs from {config_path}.")
        with open(str(config_path)) as f:
            config_data = json.load(f)
    else:
        config_data = None

    return config_data


@functools.lru_cache
def get_lora_op_configs(
    op_type: str,
    max_loras: int,
    batch: int,
    hidden_size: int,
    rank: int,
    num_slices: int,
    add_inputs: bool | None = None,
) -> dict[str, int | None]:
    assert op_type in ["shrink", "expand"]

    # default config
    default = {}
    if op_type == "shrink":
        default = {
            "block_m": 32,
            "block_n": 16,
            "block_k": 256 if batch < 128 else 32,
            "split_k": 64 if batch < 128 else 8,
            "num_warps": 4,
            "num_ctas": 1,
            "num_stages": 2,
            "max_nreg": None,
        }
    else:
        default = {
            "block_m": 64,
            "block_n": 128,
            "block_k": 16,
            "num_warps": 4,
            "num_ctas": 1,
            "num_stages": 2,
            "max_nreg": None,
        }
    m = batch

    k, n = (hidden_size, rank) if op_type == "shrink" else (rank, hidden_size)

    config_data: Any
    config_data = load_lora_op_config(op_type, add_inputs)
    if not config_data:
        logger.warning_once("Using default LoRA kernel configs")
        return default

    # config is structured as config_data[max_loras][num_slices][m][k][n] = {}
    # slice by max_loras
    config_data = (
        config_data.get(str(max_loras))
        or config_data[min(config_data.keys(), key=lambda x: abs(int(x) - max_loras))]
    )
    # slice by num_slices
    config_data = config_data[str(num_slices)]
    # slice by m
    config_data = (
        config_data.get(str(m))
        or config_data[min(config_data.keys(), key=lambda x: abs(int(x) - m))]
    )
    # slice by k
    config_data = (
        config_data.get(str(k))
        or config_data[min(config_data.keys(), key=lambda x: abs(int(x) - k))]
    )
    # slice by n
    config_data = (
        config_data.get(str(n))
        or config_data[min(config_data.keys(), key=lambda x: abs(int(x) - n))]
    )

    assert config_data is not None
    return config_data
