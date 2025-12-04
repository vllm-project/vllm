# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import base64
import io
import sys
from dataclasses import dataclass
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

EmbedDType = Literal["float32", "float16", "bfloat16", "fp8_e4m3", "fp8_e5m2"]
Endianness = Literal["native", "big", "little"]
EncodingFormat = Literal["float", "base64", "bytes"]


def tensor2base64(x: torch.Tensor) -> str:
    with io.BytesIO() as buf:
        torch.save(x, buf)
        buf.seek(0)
        binary_data = buf.read()

    return base64.b64encode(binary_data).decode("utf-8")


def tensor2binary(
    tensor: torch.Tensor, embed_dtype: EmbedDType, endianness: Endianness
) -> bytes:
    assert isinstance(tensor, torch.Tensor)
    assert embed_dtype in EMBED_DTYPE_TO_TORCH_DTYPE
    assert endianness in ENDIANNESS

    torch_dtype = EMBED_DTYPE_TO_TORCH_DTYPE[embed_dtype]
    torch_view_dtype = EMBED_DTYPE_TO_TORCH_DTYPE_VIEW[embed_dtype]

    np_array = (
        tensor.to(torch_dtype).flatten().contiguous().view(torch_view_dtype).numpy()
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
    assert embed_dtype in EMBED_DTYPE_TO_TORCH_DTYPE
    assert embed_dtype in EMBED_DTYPE_TO_NUMPY_DTYPE_VIEW
    assert endianness in ENDIANNESS

    torch_dtype = EMBED_DTYPE_TO_TORCH_DTYPE[embed_dtype]
    np_dtype = EMBED_DTYPE_TO_NUMPY_DTYPE_VIEW[embed_dtype]

    np_array = np.frombuffer(binary, dtype=np_dtype).reshape(shape)

    if endianness != "native" and endianness != sys_byteorder:
        np_array = np_array.byteswap()

    return torch.from_numpy(np_array).view(torch_dtype)


def encode_pooling_output(
    output: PoolingRequestOutput,
    encoding_format: EncodingFormat,
    embed_dtype: EmbedDType,
    endianness: Endianness,
) -> list[float] | str | bytes:
    if encoding_format == "float":
        return output.outputs.data.tolist()
    elif encoding_format == "base64":
        embedding_bytes = tensor2binary(output.outputs.data, embed_dtype, endianness)
        return base64.b64encode(embedding_bytes).decode("utf-8")
    elif encoding_format == "bytes":
        return tensor2binary(output.outputs.data, embed_dtype, endianness)
    assert_never(encoding_format)


@dataclass
class MetadataItem:
    index: int
    embed_dtype: EmbedDType
    endianness: Endianness
    start: int
    end: int
    shape: tuple[int, ...]


def encode_pooling_bytes(
    pooling_outputs: list[PoolingRequestOutput],
    embed_dtype: EmbedDType,
    endianness: Endianness,
):
    num_prompt_tokens = 0
    items: list[dict[str, MetadataItem]] = []
    body = []
    offset = 0
    for idx, output in enumerate(pooling_outputs):
        binary = tensor2binary(
            tensor=output.outputs.data,
            embed_dtype=embed_dtype,
            endianness=endianness,
        )
        size = len(binary)

        item = {
            "index": idx,
            "embed_dtype": embed_dtype,
            "endianness": endianness,
            "start": offset,
            "end": offset + size,
            "shape": output.outputs.data.shape,
        }

        body.append(binary)
        items.append(item)
        prompt_token_ids = output.prompt_token_ids
        num_prompt_tokens += len(prompt_token_ids)
        offset += size

    usage = {
        "prompt_tokens": num_prompt_tokens,
        "total_tokens": num_prompt_tokens,
    }
    return body, items, usage


def decode_pooling_output(items: list[MetadataItem], body: bytes) -> list[torch.Tensor]:
    items.sort(key=lambda x: x.index)

    tensor_list: list[torch.Tensor] = []
    for item in items:
        binary = body[item.start : item.end]
        tensor = binary2tensor(binary, item.shape, item.embed_dtype, item.endianness)
        tensor_list.append(tensor)
    return tensor_list
