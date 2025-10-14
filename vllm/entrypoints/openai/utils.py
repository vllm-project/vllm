# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import base64
from typing import Literal

import torch
from typing_extensions import assert_never

from vllm import PoolingRequestOutput
from vllm.entrypoints.openai.protocol import EMBED_DTYPE_TO_TORCH_DTYPE


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
