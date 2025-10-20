# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest
import torch

from tests.models.utils import check_embeddings_close
from vllm.utils.tensor_serial import (
    EMBED_DTYPE_TO_TORCH_DTYPE,
    ENDIANNESS,
    binary2tenser,
    tenser2binary,
)


@pytest.mark.parametrize("endianness", ENDIANNESS)
@pytest.mark.parametrize("embed_dtype", EMBED_DTYPE_TO_TORCH_DTYPE.keys())
@torch.inference_mode
def test_encode_and_decode(embed_dtype: str, endianness: str):
    for i in range(10):
        tenser = torch.rand(2, 3, 5, 7, 11, 13, device="cpu", dtype=torch.float32)
        shape = tenser.shape
        binary = tenser2binary(tenser, embed_dtype, endianness)
        new_tenser = binary2tenser(binary, shape, embed_dtype, endianness).to(
            torch.float32
        )

        if "embed_dtype" in ["float32", "float16", "bfloat16"]:
            torch.testing.assert_close(tenser, new_tenser, atol=1e-7, rtol=1e-7)
        else:  # for fp8
            torch.testing.assert_close(tenser, new_tenser, atol=0.1, rtol=0.1)
            check_embeddings_close(
                embeddings_0_lst=tenser.view(1, -1),
                embeddings_1_lst=new_tenser.view(1, -1),
                name_0="gt",
                name_1="new",
                tol=1e-2,
            )
