# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest
import torch

from tests.models.utils import check_embeddings_close
from vllm.utils.serial_utils import (
    EMBED_DTYPE_TO_TORCH_DTYPE,
    ENDIANNESS,
    binary2tensor,
    tensor2binary,
)


@pytest.mark.parametrize("endianness", ENDIANNESS)
@pytest.mark.parametrize("embed_dtype", EMBED_DTYPE_TO_TORCH_DTYPE.keys())
@torch.inference_mode()
def test_encode_and_decode(embed_dtype: str, endianness: str):
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


@pytest.mark.parametrize("embed_dtype", EMBED_DTYPE_TO_TORCH_DTYPE.keys())
@torch.inference_mode()
def test_binary2tensor_no_warning(embed_dtype: str):
    """Test that binary2tensor does not emit UserWarning about non-writable buffers.

    This addresses issue #26781 where torch.frombuffer on non-writable bytes
    would emit: "UserWarning: The given buffer is not writable..."
    """
    import warnings

    tensor = torch.rand(10, 20, device="cpu", dtype=torch.float32)
    binary = tensor2binary(tensor, embed_dtype, "native")

    # Capture warnings during binary2tensor call
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = binary2tensor(binary, tensor.shape, embed_dtype, "native")

        # Filter for the specific UserWarning about non-writable buffers
        buffer_warnings = [
            warning
            for warning in w
            if issubclass(warning.category, UserWarning)
            and "not writable" in str(warning.message)
        ]
        assert len(buffer_warnings) == 0, (
            f"Expected no warnings about non-writable buffers, got: {buffer_warnings}"
        )

    # Verify the result is correct
    result_float = result.to(torch.float32)
    if embed_dtype in ["float32", "float16"]:
        torch.testing.assert_close(tensor, result_float, atol=0.001, rtol=0.001)
    elif embed_dtype == "bfloat16":
        torch.testing.assert_close(tensor, result_float, atol=0.01, rtol=0.01)
    else:  # fp8
        torch.testing.assert_close(tensor, result_float, atol=0.1, rtol=0.1)
