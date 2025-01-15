from typing import List, Optional, Tuple

import torch

from vllm.scalar_type import ScalarType, scalar_types

MACHETE_SUPPORTED_GROUP_SIZES = [-1, 128]
MACHETE_PREPACKED_BLOCK_SHAPE = [64, 128]


def query_machete_supported_quant_types(zero_points: bool) -> List[ScalarType]:
    if zero_points:
        return [scalar_types.uint4, scalar_types.uint8]
    else:
        return [scalar_types.uint4b8, scalar_types.uint8b128]


def query_machete_supported_act_types(zero_points: bool) -> List[ScalarType]:
    return [torch.float16, torch.bfloat16]


def check_machete_supports_shape(in_features: int, out_featrues: int) \
    -> Tuple[bool, Optional[str]]:
    if in_features % MACHETE_PREPACKED_BLOCK_SHAPE[0] != 0:
        return False, "Input features size must be divisible by "\
            f"{MACHETE_PREPACKED_BLOCK_SHAPE[0]}"
    if out_featrues % MACHETE_PREPACKED_BLOCK_SHAPE[1] != 0:
        return False, "Output features size must be divisible by "\
            f"{MACHETE_PREPACKED_BLOCK_SHAPE[1]}"
    return True, None
