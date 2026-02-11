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


@pytest.mark.parametrize("endianness", ENDIANNESS)
@pytest.mark.parametrize("embed_dtype", EMBED_DTYPES.keys())
@torch.inference_mode()
def test_encode_and_decode(embed_dtype: EmbedDType, endianness: Endianness):
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
