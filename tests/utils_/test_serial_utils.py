# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest
import torch

from tests.models.utils import check_embeddings_close
from vllm.utils.serial_utils import (
    EMBED_DTYPES,
    ENDIANNESS,
    EmbedDType,
    Endianness,
    binary2tensor,
    tensor2binary,
)

FLOAT_EMBED_DTYPES = tuple(
    embed_dtype
    for embed_dtype, dtype_info in EMBED_DTYPES.items()
    if dtype_info.torch_dtype.is_floating_point
)
INTEGER_EMBED_DTYPES = tuple(
    embed_dtype
    for embed_dtype, dtype_info in EMBED_DTYPES.items()
    if not dtype_info.torch_dtype.is_floating_point
)


def _build_integer_tensor(
    embed_dtype: EmbedDType, shape: tuple[int, ...]
) -> torch.Tensor:
    torch_dtype = EMBED_DTYPES[embed_dtype].torch_dtype

    if torch_dtype is torch.bool:
        return torch.randint(0, 2, shape, dtype=torch.int32).to(torch.bool)
    if torch_dtype is torch.uint8:
        return torch.randint(0, 256, shape, dtype=torch.uint8)
    if torch_dtype is torch.int32:
        return torch.randint(-(2**15), 2**15, shape, dtype=torch.int32)
    if torch_dtype is torch.int64:
        return torch.randint(-(2**31), 2**31, shape, dtype=torch.int64)

    raise AssertionError(f"Unsupported non-floating embed dtype: {embed_dtype}")


@pytest.mark.parametrize("endianness", ENDIANNESS)
@pytest.mark.parametrize("embed_dtype", FLOAT_EMBED_DTYPES)
@torch.inference_mode()
def test_encode_and_decode_floats(embed_dtype: EmbedDType, endianness: Endianness):
    for i in range(10):
        tensor = torch.rand(2, 3, 5, 7, 11, 13, device="cpu", dtype=torch.float32)
        shape = tensor.shape
        binary = tensor2binary(tensor, embed_dtype, endianness)
        new_tensor = binary2tensor(binary, shape, embed_dtype, endianness).to(
            torch.float32
        )

        if embed_dtype in ["float32", "float16"]:
            torch.testing.assert_close(tensor, new_tensor, atol=0.001, rtol=0.001)
        elif embed_dtype == "bfloat16":
            torch.testing.assert_close(tensor, new_tensor, atol=0.01, rtol=0.01)
        else:  # for fp8
            torch.testing.assert_close(tensor, new_tensor, atol=0.1, rtol=0.1)

        check_embeddings_close(
            embeddings_0_lst=tensor.view(1, -1),
            embeddings_1_lst=new_tensor.view(1, -1),
            name_0="gt",
            name_1="new",
            tol=1e-2,
        )


@pytest.mark.parametrize("endianness", ENDIANNESS)
@pytest.mark.parametrize("embed_dtype", INTEGER_EMBED_DTYPES)
@torch.inference_mode()
def test_encode_and_decode_integers(embed_dtype: EmbedDType, endianness: Endianness):
    shape = (2, 3, 5, 7, 11, 13)

    for i in range(10):
        tensor = _build_integer_tensor(embed_dtype, shape)
        binary = tensor2binary(tensor, embed_dtype, endianness)
        new_tensor = binary2tensor(binary, shape, embed_dtype, endianness)

        assert new_tensor.dtype == EMBED_DTYPES[embed_dtype].torch_dtype
        torch.testing.assert_close(tensor, new_tensor, atol=0, rtol=0)
