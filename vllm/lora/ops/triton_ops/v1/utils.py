# SPDX-License-Identifier: Apache-2.0

import functools
import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

import torch


@functools.lru_cache(maxsize=100)
def load_v1_op_config(op_type: str,
                      add_inputs: Optional[bool]) -> Optional[Dict]:
    gpu_name = torch.cuda.get_device_name()
    gpu_name = gpu_name.replace(' ', '_')
    gpu_name = gpu_name.replace('-', '_')

    config_fname = None
    if op_type == "shrink":
        config_fname = f"{gpu_name}_{op_type.upper()}.json"
    else:
        assert op_type == "expand"
        config_fname = (f"{gpu_name}_"
                        f"{op_type.upper()}_"
                        f"{str(add_inputs).upper()}.json")

    config_path = Path(
        f'{os.path.dirname(os.path.realpath(__file__))}/configs/{config_fname}'
    )
    if not config_path.exists():
        return None

    # Load json
    config_data = None
    with open(str(config_path)) as f:
        config_data = json.load(f)
    return config_data


@functools.lru_cache(maxsize=100)
def get_v1_op_configs(
        op_type: str,
        max_loras: int,
        batch: int,
        hidden_size: int,
        rank: int,
        num_slices: int,
        add_inputs: Optional[bool] = None) -> dict[str, Optional[int]]:

    assert op_type in ["shrink", "expand"]

    # default config
    default = {}
    if op_type == "shrink":
        default = {
            'block_m': 32,
            'block_n': 16,
            'block_k': 256 if batch < 128 else 32,
            'split_k': 64 if batch < 128 else 8,
            'num_warps': 4,
            'num_ctas': 1,
            'num_stages': 2,
            'max_nreg': None
        }
    else:
        default = {
            'block_m': 64,
            'block_n': 128,
            'block_k': 16,
            'num_warps': 4,
            'num_ctas': 1,
            'num_stages': 2,
            'max_nreg': None
        }
    m = batch

    k, n = (hidden_size, rank) if op_type == "shrink" else (rank, hidden_size)

    config_data: Any
    config_data = load_v1_op_config(op_type, add_inputs)
    if not config_data:
        return default

    # config is structured as config_data[max_loras][num_slices][m][k][n] = {}
    # slice by max_loras
    config_data = config_data.get(str(max_loras)) or config_data[min(
        config_data.keys(), key=lambda x: abs(int(x) - max_loras))]
    # slice by num_slices
    config_data = config_data[str(num_slices)]
    # slice by m
    config_data = config_data.get(str(m)) or config_data[min(
        config_data.keys(), key=lambda x: abs(int(x) - m))]
    # slice by k
    config_data = config_data.get(str(k)) or config_data[min(
        config_data.keys(), key=lambda x: abs(int(x) - k))]
    # slice by n
    config_data = config_data.get(str(n)) or config_data[min(
        config_data.keys(), key=lambda x: abs(int(x) - n))]

    assert config_data is not None
    return config_data
