# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import TYPE_CHECKING, Final

import pybase64
import torch

from vllm.exceptions import VLLMValidationError
from vllm.utils.async_utils import make_async
from vllm.utils.sparse_utils import load_torch_payload

if TYPE_CHECKING:
    from vllm.config import ModelConfig


def safe_load_prompt_embeds(
    model_config: "ModelConfig",
    embed: bytes | str,
) -> torch.Tensor:
    if not model_config.enable_prompt_embeds:
        raise VLLMValidationError(
            "You must set `--enable-prompt-embeds` to input `prompt_embeds`.",
            parameter="prompt_embeds",
        )

    # Decode before loading so a body which is not base64 keeps raising
    # binascii.Error, which is a ValueError and already answered 400.
    payload: Final = pybase64.b64decode(embed, validate=True)
    tensor = load_torch_payload(payload, parameter="prompt_embeds")

    if tensor.dim() > 2:
        tensor = tensor.squeeze(0)
    if tensor.dim() != 2:
        raise VLLMValidationError(
            "`prompt_embeds` must be a 2D tensor of shape "
            f"(num_tokens, hidden_size); got shape {tuple(tensor.shape)}.",
            parameter="prompt_embeds",
        )

    # Pin each tensor to the model's text embedding width. Validating here
    # also transitively guarantees cross-tensor consistency for requests that
    # include multiple `prompt_embeds` parts, which is required by downstream
    # concatenation in `_build_mixed_prompt_embeds`.
    # OPT projects token embeddings to hidden_size inside its decoder. CLIP
    # and SigLIP still accept text embeddings at hidden_size, not projection_dim.
    expected_hidden_size = getattr(
        model_config.hf_text_config,
        "word_embed_proj_dim",
        model_config.get_hidden_size(),
    )
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
