import functools
from typing import Dict


@functools.lru_cache
def _get_op_configs(op_type: str, batch: int, hidden_size: int):
    # TODO: add optimal configurations
    return None


def _check_divisibility(hidden_size: int):
    # The bgmv_expand kernel requires that the hidden_size be divisible by
    # the number below.
    divisibility = [2, 4, 8, 16, 32, 64]
    divisibility.sort(reverse=True)
    for div in divisibility:
        if hidden_size % div == 0:
            return div
    # hidden_size is an odd number
    return 1


def _get_default_config(op_type: str, batch: int, hidden_size: int):
    if op_type == "expand":
        return {
            "BLOCK_N": 256,
            "SPLIT_N": _check_divisibility(hidden_size),
            "num_warps": 8
        }
    else:
        return {"BLOCK_K": 256, "SPLIT_K": 64, "num_warps": 8}


def get_lora_op_configs(op_type: str, batch: int,
                        hidden_size: int) -> Dict[str, int]:
    """Inspired by `fused_moe_kernel`
    The return value will be a dictionary mapping an irregular grid of batch 
    sizes and hidden_size to configurations of the bgmv-related kernel. 
    NOTE: It currently only supports the default configuration. We plan to 
    generate optimal configurations for different hardware in the future using 
    scripts similar to `benchmark_moe.py`.
    """
    config = _get_op_configs(op_type, batch, hidden_size)
    if not config:
        config = _get_default_config(op_type, batch, hidden_size)
    return config
