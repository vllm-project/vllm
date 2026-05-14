# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from io import BytesIO
from typing import TYPE_CHECKING

import pybase64
import torch

from vllm.exceptions import VLLMValidationError
from vllm.utils.async_utils import make_async

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
        if not isinstance(tensor, torch.Tensor):
            raise VLLMValidationError(
                "`prompt_embeds` payload did not deserialize to a torch.Tensor.",
                parameter="prompt_embeds",
            )
        tensor = tensor.to_dense()

    if tensor.dim() > 2:
        tensor = tensor.squeeze(0)
    if tensor.dim() != 2:
        raise VLLMValidationError(
            "`prompt_embeds` must be a 2D tensor of shape "
            f"(num_tokens, hidden_size); got shape {tuple(tensor.shape)}.",
            parameter="prompt_embeds",
        )

    # Pin each tensor to the model's hidden_size. Validating here
    # also transitively guarantees cross-tensor consistency for requests that
    # include multiple `prompt_embeds` parts, which is required by downstream
    # concatenation in `_build_mixed_prompt_embeds`.
    expected_hidden_size = model_config.get_hidden_size()
    if tensor.shape[1] != expected_hidden_size:
        raise VLLMValidationError(
            f"`prompt_embeds` hidden_size {tensor.shape[1]} does not match "
            f"the model's hidden_size {expected_hidden_size}.",
            parameter="prompt_embeds",
        )

    # Cast to the model's dtype so API clients don't need to know the server's
    # `--dtype` setting ahead of time. Only floating-point source dtypes are
    # allowed. integer / bool / complex inputs almost certainly indicate caller
    # error (e.g. quantized payloads, wrong tensor), and a silent `.to()`
    # could hide a real mistake.
    expected_dtype = model_config.dtype
    if tensor.dtype != expected_dtype:
        if not tensor.is_floating_point():
            raise VLLMValidationError(
                f"`prompt_embeds` dtype {tensor.dtype} is not a floating-point "
                f"type, cannot safely cast to the model's dtype {expected_dtype}.",
                parameter="prompt_embeds",
            )
        tensor = tensor.to(expected_dtype)

    return tensor


safe_load_prompt_embeds_async = make_async(safe_load_prompt_embeds)
"""Async variant of `safe_load_prompt_embeds` that defers the decode to a
thread-pool executor, so the asyncio event loop is not blocked by the base64
decode + `torch.load` work."""
