# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import base64
import sys
from typing import Literal

import numpy as np
import torch
from typing_extensions import assert_never

from vllm import PoolingRequestOutput

sys_byteorder = sys.byteorder


EMBED_DTYPE_TO_TORCH_DTYPE = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    # I'm not sure if other platforms' CPUs support the fp8 data format.
    # EMBED_DTYPE only uses the fp8 data representation,
    # does not use fp8 computation, and only occurs on the CPU.
    # Apologize for any possible break.
    "fp8_e4m3": torch.float8_e4m3fn,
    "fp8_e5m2": torch.float8_e5m2,
}


EMBED_DTYPE_TO_TORCH_DTYPE_VIEW = {
    "float32": torch.float32,
    "float16": torch.float16,
    # numpy does not support bfloat16 and fp8
    "bfloat16": torch.float16,
    "fp8_e4m3": torch.uint8,
    "fp8_e5m2": torch.uint8,
}

EMBED_DTYPE_TO_NUMPY_DTYPE_VIEW = {
    "float32": np.float32,
    "float16": np.float16,
    # numpy does not support bfloat16 and fp8
    "bfloat16": np.float16,
    "fp8_e4m3": np.uint8,
    "fp8_e5m2": np.uint8,
}


ENDIANNESS = ["native", "big", "little"]


def tenser2binary(tenser: torch.Tensor, embed_dtype: str, endianness: str) -> bytes:
    assert isinstance(tenser, torch.Tensor)
    assert embed_dtype in EMBED_DTYPE_TO_TORCH_DTYPE
    assert endianness in ENDIANNESS

    torch_dtype = EMBED_DTYPE_TO_TORCH_DTYPE[embed_dtype]
    torch_view_dtype = EMBED_DTYPE_TO_TORCH_DTYPE_VIEW[embed_dtype]

    np_array = (
        tenser.to(torch_dtype).flatten().contiguous().view(torch_view_dtype).numpy()
    )

    if endianness != "native" and endianness != sys_byteorder:
        np_array = np_array.byteswap()

    return np_array.tobytes()


def binary2tenser(
    binary: bytes, shape: tuple[int, ...], embed_dtype: str, endianness: str
) -> torch.Tensor:
    assert embed_dtype in EMBED_DTYPE_TO_TORCH_DTYPE
    assert embed_dtype in EMBED_DTYPE_TO_NUMPY_DTYPE_VIEW
    assert endianness in ENDIANNESS

    torch_dtype = EMBED_DTYPE_TO_TORCH_DTYPE[embed_dtype]
    np_dtype = EMBED_DTYPE_TO_NUMPY_DTYPE_VIEW[embed_dtype]

    np_array = np.frombuffer(binary, dtype=np_dtype).reshape(shape)

    if endianness != "native" and endianness != sys_byteorder:
        np_array = np_array.byteswap()

    return torch.from_numpy(np_array).view(torch_dtype)


def encoding_pooling_output(
    output: PoolingRequestOutput,
    encoding_format: Literal["float", "base64"],
    embed_dtype: str,
    endianness: str,
) -> list[float] | str:
    if encoding_format == "float":
        return output.outputs.data.tolist()
    elif encoding_format == "base64":
        embedding_bytes = tenser2binary(output.outputs.data, embed_dtype, endianness)
        return base64.b64encode(embedding_bytes).decode("utf-8")
    assert_never(encoding_format)
