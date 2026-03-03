# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import io
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal, get_args

import numpy as np
import numpy.typing as npt
import pybase64
import torch

sys_byteorder = sys.byteorder


@dataclass(frozen=True)
class DTypeInfo:
    torch_dtype: torch.dtype

    torch_view_dtype: torch.dtype
    numpy_view_dtype: npt.DTypeLike

    @property
    def nbytes(self) -> int:
        return self.torch_dtype.itemsize


EmbedDType = Literal["float32", "float16", "bfloat16", "fp8_e4m3", "fp8_e5m2"]
Endianness = Literal["native", "big", "little"]
EncodingFormat = Literal["float", "base64", "bytes", "bytes_only"]

# I'm not sure if other platforms' CPUs support the fp8 data format.
# EMBED_DTYPE only uses the fp8 data representation,
# does not use fp8 computation, and only occurs on the CPU.
# Apologize for any possible break.
# NOTE: numpy does not support bfloat16 and fp8
EMBED_DTYPES: Mapping[EmbedDType, DTypeInfo] = {
    "float32": DTypeInfo(torch.float32, torch.float32, np.float32),
    "float16": DTypeInfo(torch.float16, torch.float16, np.float16),
    "bfloat16": DTypeInfo(torch.bfloat16, torch.float16, np.float16),
    "fp8_e4m3": DTypeInfo(torch.float8_e4m3fn, torch.uint8, np.uint8),
    "fp8_e5m2": DTypeInfo(torch.float8_e5m2, torch.uint8, np.uint8),
}
ENDIANNESS: tuple[Endianness, ...] = get_args(Endianness)


def tensor2base64(x: torch.Tensor) -> str:
    with io.BytesIO() as buf:
        torch.save(x, buf)
        buf.seek(0)
        binary_data = buf.read()

    return pybase64.b64encode(binary_data).decode("utf-8")


def tensor2binary(
    tensor: torch.Tensor,
    embed_dtype: EmbedDType,
    endianness: Endianness,
) -> bytes:
    assert isinstance(tensor, torch.Tensor)
    assert embed_dtype in EMBED_DTYPES
    assert endianness in ENDIANNESS

    dtype_info = EMBED_DTYPES[embed_dtype]

    np_array = (
        tensor.to(dtype_info.torch_dtype)
        .flatten()
        .contiguous()
        .view(dtype_info.torch_view_dtype)
        .numpy()
    )

    if endianness != "native" and endianness != sys_byteorder:
        np_array = np_array.byteswap()

    return np_array.tobytes()


def binary2tensor(
    binary: bytes,
    shape: tuple[int, ...],
    embed_dtype: EmbedDType,
    endianness: Endianness,
) -> torch.Tensor:
    assert embed_dtype in EMBED_DTYPES
    assert endianness in ENDIANNESS

    dtype_info = EMBED_DTYPES[embed_dtype]

    np_array = np.frombuffer(binary, dtype=dtype_info.numpy_view_dtype).reshape(shape)

    if endianness != "native" and endianness != sys_byteorder:
        np_array = np_array.byteswap()

    return torch.from_numpy(np_array).view(dtype_info.torch_dtype)
