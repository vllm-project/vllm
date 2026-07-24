import pytest
import torch

from vllm.model_executor.layers.quantization.utils.humming_utils import (
    pack_signed_int4_to_uint4_int32,
)


def test_pack_signed_int4_to_uint4_int32() -> None:
    codes = torch.tensor(
        [[-8, -7, -1, 0, 1, 6, 7, -3], [7, 6, 5, 4, 3, 2, 1, 0]],
        dtype=torch.int8,
    )

    packed = pack_signed_int4_to_uint4_int32(codes)

    assert packed.dtype == torch.int32
    assert packed.shape == (2, 1)
    unpacked = torch.stack(
        [((packed >> (4 * lane)) & 0xF).to(torch.int8) - 8 for lane in range(8)],
        dim=-1,
    ).reshape_as(codes)
    assert torch.equal(unpacked, codes)


@pytest.mark.parametrize(
    ("codes", "message"),
    [
        (torch.zeros(2, 7, dtype=torch.int8), "K divisible by 8"),
        (torch.tensor([[-9] + [0] * 7], dtype=torch.int8), "outside"),
        (torch.zeros(2, 8, dtype=torch.int16), "INT8 tensor"),
    ],
)
def test_pack_signed_int4_to_uint4_int32_rejects_invalid_input(
    codes: torch.Tensor, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        pack_signed_int4_to_uint4_int32(codes)
