# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from io import BytesIO
from typing import TYPE_CHECKING

import pybase64
import torch

from vllm.exceptions import VLLMValidationError

if TYPE_CHECKING:
    from vllm.config import ModelConfig


def safe_load_prompt_embeds(
    model_config: "ModelConfig",
    embed: bytes,
) -> torch.Tensor:
    if not model_config.enable_prompt_embeds:
        raise VLLMValidationError(
            "You must set `--enable-prompt-embeds` to input `prompt_embeds`.",
            parameter="prompt_embeds",
        )

    # Enable sparse tensor integrity checks to prevent out-of-bounds
    # writes from maliciously crafted tensors
    with torch.sparse.check_sparse_tensor_invariants():
        tensor = torch.load(
            BytesIO(pybase64.b64decode(embed, validate=True)),
            weights_only=True,
            map_location=torch.device("cpu"),
        )
        assert isinstance(tensor, torch.Tensor) and tensor.dtype in (
            torch.float32,
            torch.bfloat16,
            torch.float16,
        )
        tensor = tensor.to_dense()

    if tensor.dim() > 2:
        tensor = tensor.squeeze(0)
        assert tensor.dim() == 2

    return tensor
