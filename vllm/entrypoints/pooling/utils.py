# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import math
from dataclasses import dataclass

import pybase64
import torch

from vllm.outputs import PoolingRequestOutput
from vllm.utils.serial_utils import (
    EMBED_DTYPES,
    EmbedDType,
    Endianness,
    binary2tensor,
    tensor2binary,
)


@dataclass
class MetadataItem:
    index: int
    embed_dtype: EmbedDType
    endianness: Endianness
    start: int
    end: int
    shape: tuple[int, ...]


def build_metadata_items(
    embed_dtype: EmbedDType,
    endianness: Endianness,
    shape: tuple[int, ...],
    n_request: int,
) -> list[MetadataItem]:
    n_bytes = EMBED_DTYPES[embed_dtype].nbytes
    size = math.prod(shape)

    return [
        MetadataItem(
            index=i,
            embed_dtype=embed_dtype,
            endianness=endianness,
            start=i * size * n_bytes,
            end=(i + 1) * size * n_bytes,
            shape=shape,
        )
        for i in range(n_request)
    ]


def encode_pooling_output_float(output: PoolingRequestOutput) -> list[float]:
    return output.outputs.data.tolist()


def encode_pooling_output_binary(
    output: PoolingRequestOutput,
    embed_dtype: EmbedDType,
    endianness: Endianness,
) -> bytes:
    return tensor2binary(output.outputs.data, embed_dtype, endianness)


def encode_pooling_output_base64(
    output: PoolingRequestOutput,
    embed_dtype: EmbedDType,
    endianness: Endianness,
) -> str:
    embedding_bytes = tensor2binary(output.outputs.data, embed_dtype, endianness)
    return pybase64.b64encode(embedding_bytes).decode("utf-8")


def encode_pooling_bytes(
    pooling_outputs: list[PoolingRequestOutput],
    embed_dtype: EmbedDType,
    endianness: Endianness,
) -> tuple[list[bytes], list[MetadataItem], dict[str, int]]:
    num_prompt_tokens = 0
    items: list[MetadataItem] = []
    body: list[bytes] = []
    offset = 0
    for idx, output in enumerate(pooling_outputs):
        binary = tensor2binary(
            tensor=output.outputs.data,
            embed_dtype=embed_dtype,
            endianness=endianness,
        )
        size = len(binary)

        item = MetadataItem(
            index=idx,
            embed_dtype=embed_dtype,
            endianness=endianness,
            start=offset,
            end=offset + size,
            shape=output.outputs.data.shape,
        )

        body.append(binary)
        items.append(item)
        prompt_token_ids = output.prompt_token_ids
        num_prompt_tokens += len(prompt_token_ids)
        offset += size

    # Dictionary form of UsageInfo
    usage = dict(
        prompt_tokens=num_prompt_tokens,
        total_tokens=num_prompt_tokens,
    )

    return body, items, usage


def decode_pooling_output(items: list[MetadataItem], body: bytes) -> list[torch.Tensor]:
    return [
        binary2tensor(
            body[item.start : item.end],
            item.shape,
            item.embed_dtype,
            item.endianness,
        )
        for item in sorted(items, key=lambda x: x.index)
    ]
