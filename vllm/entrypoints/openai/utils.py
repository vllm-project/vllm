# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import base64
import io
import json
import zipfile
from typing import Literal

import torch
from typing_extensions import assert_never

from vllm import PoolingRequestOutput
from vllm.entrypoints.openai.protocol import EMBED_DTYPE_TO_TORCH_DTYPE

EMBED_DTYPE_TO_TORCH_DTYPE_View = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.float16,
    "fp8_e4m3": torch.uint8,
    "fp8_e5m2": torch.uint8,
}


def encoding_pooling_output(
    output: PoolingRequestOutput,
    encoding_format: Literal["float", "base64"],
    embed_dtype: str,
) -> list[float] | str:
    if encoding_format == "float":
        return output.outputs.data.tolist()
    elif encoding_format == "base64":
        assert embed_dtype in EMBED_DTYPE_TO_TORCH_DTYPE
        torch_dtype = EMBED_DTYPE_TO_TORCH_DTYPE[embed_dtype]
        embedding_bytes = (
            output.outputs.data.to(torch_dtype)
            .flatten()
            .contiguous()
            .view(torch.uint8)
            .numpy()
            .tobytes()
        )
        return base64.b64encode(embedding_bytes).decode("utf-8")

    assert_never(encoding_format)


def response_compression_pooling_output(
    metadata: dict, tensers: list[torch.Tensor], embed_dtype: str, endianness: str
):
    zip_buffer = io.BytesIO()
    metadata_bytes = json.dumps(metadata).encode("utf-8")
    torch_dtype = EMBED_DTYPE_TO_TORCH_DTYPE[embed_dtype]
    torch_view_dtype = EMBED_DTYPE_TO_TORCH_DTYPE_View[embed_dtype]
    order = ">" if endianness == "big-endian" else "<"

    with zipfile.ZipFile(zip_buffer, "a", zipfile.ZIP_DEFLATED, False) as zip_file:
        zip_file.writestr("metadata.json", metadata_bytes)
        for meta, tenser in zip(metadata["data"], tensers):
            filename = meta["filename"]
            tenser_bytes = (
                tenser.to(torch_dtype)
                .flatten()
                .contiguous()
                .view(torch_view_dtype)
                .numpy()
                .newbyteorder(order)
            )
            zip_file.writestr(filename, tenser_bytes)

    return zip_buffer.getvalue()
