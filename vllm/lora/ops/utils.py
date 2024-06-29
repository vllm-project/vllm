import functools
import json
import os
from typing import Dict

import torch


def _get_config_file_name(
    op_type: str,
    batches: int,
    hidden_size: int,
) -> str:
    device_name = torch.cuda.get_device_name().replace(" ", "_")
    return (f"op_type={op_type},batches={batches},hidden_size={hidden_size} " +
            f"device_name={device_name}.json")


@functools.lru_cache
def _get_op_configs(op_type: str, batch: int, hidden_size: int):
    FOLDER_NAME = "bgmv_configs"
    json_file_name = _get_config_file_name(op_type, batch, hidden_size)

    config_file_path = os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        FOLDER_NAME,
        json_file_name,
    )
    if os.path.exists(config_file_path):
        with open(config_file_path) as f:
            tuned_config = json.load(f).get(
                f"batches={batch},hidden_size={hidden_size}", None)
            return tuned_config

    # If no optimized configuration is available, return None
    return None


def _get_default_config(op_type: str, batch: int, hidden_size: int):
    if op_type == "expand":
        return {"BLOCK_N": 256, "SPLIT_N": 64, "num_warps": 8}
    else:
        return {"BLOCK_K": 256, "SPLIT_K": 64, "num_warps": 8}


def get_lora_op_configs(op_type: str, batch: int,
                        hidden_size: int) -> Dict[str, int]:
    config = _get_op_configs(op_type, batch, hidden_size)
    if not config:
        config = _get_default_config(op_type, batch, hidden_size)
    return config
