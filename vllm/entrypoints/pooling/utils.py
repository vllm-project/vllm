# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import importlib.util
import math
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import pybase64
import torch
from fastapi.responses import JSONResponse

from vllm.config import ModelConfig
from vllm.logger import init_logger
from vllm.outputs import PoolingRequestOutput
from vllm.tasks import SupportedTask
from vllm.utils.serial_utils import (
    EMBED_DTYPES,
    EmbedDType,
    Endianness,
    binary2tensor,
    tensor2binary,
)

logger = init_logger(__name__)


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


def encode_pooling_output_float_or_ndarray(output: PoolingRequestOutput) -> Any:
    """Return an ndarray when the response renderer can serialize NumPy."""
    try:
        data = output.outputs.data
        if not data.is_contiguous():
            data = data.contiguous()
        return data.numpy()
    except (RuntimeError, TypeError):
        return output.outputs.data.tolist()


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
) -> tuple[list[bytes], list[dict[str, Any]], dict[str, Any]]:
    num_prompt_tokens = 0
    items: list[dict[str, Any]] = []
    body: list[bytes] = []
    offset = 0
    for idx, output in enumerate(pooling_outputs):
        binary = tensor2binary(
            tensor=output.outputs.data,
            embed_dtype=embed_dtype,
            endianness=endianness,
        )
        size = len(binary)

        # Dictionary form of MetadataItem
        item = dict(
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


@lru_cache(maxsize=1)
def get_json_response_cls() -> type[JSONResponse]:
    if importlib.util.find_spec("orjson") is not None:
        from fastapi.responses import ORJSONResponse

        return ORJSONResponse
    logger.warning_once(
        "To make v1/embeddings API fast, please install orjson by `pip install orjson`"
    )
    return JSONResponse


def enable_scoring_api(
    supported_tasks: tuple["SupportedTask", ...],
    model_config: ModelConfig | None = None,
) -> bool:
    if model_config is None:
        return False

    pooling_task = model_config.get_pooling_task(supported_tasks)
    if pooling_task in ("embed", "token_embed"):
        return True

    if pooling_task == "classify":
        num_labels = getattr(model_config.hf_config, "num_labels", 0)
        if num_labels != 1:
            logger.debug_once("Scoring API is only enabled for num_labels == 1.")
            return False
        return True

    return False
