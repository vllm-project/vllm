import pytest
import torch

from vllm.scalar_type import scalar_types


@pytest.mark.parametrize("type_tuple", (
    (-8, 7, scalar_types.int4),
    (0, 15, scalar_types.uint4),
    (-8, 7, scalar_types.uint4b8),
    (-128, 127, scalar_types.uint8b128),
    (-28., 28., scalar_types.float6_e3m2f),
    (torch.int8, scalar_types.int8),
    (torch.uint8, scalar_types.uint8),
    (torch.float8_e5m2, scalar_types.float8_e5m2),
    (torch.float8_e4m3fn, scalar_types.float8_e4m3fn),
    (torch.bfloat16, scalar_types.float16_e8m7),
    (torch.float16, scalar_types.float16_e5m10),
),
                         ids=lambda x: str(x))
def test_scalar_type_min_max(type_tuple):
    print(type_tuple)
    if len(type_tuple) == 3:
        min, max, t = type_tuple
    else:
        torch_type, t = type_tuple
        if torch_type.is_floating_point:
            min = torch.finfo(torch_type).min
            max = torch.finfo(torch_type).max
        else:
            min = torch.iinfo(torch_type).min
            max = torch.iinfo(torch_type).max

    print(t, min, max, t.min(), t.max())
    assert min == t.min()
    assert max == t.max()
