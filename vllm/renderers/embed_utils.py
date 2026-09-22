# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from io import BytesIO
from typing import TYPE_CHECKING, Final

import pybase64
import torch

from vllm.exceptions import VLLMValidationError
from vllm.utils.async_utils import make_async
from vllm.utils.sparse_utils import (
    check_sparse_tensor_invariants_threadsafe,
    safe_to_dense,
)

if TYPE_CHECKING:
    from vllm.config import ModelConfig


# How much of torch's own reason is repeated back to the caller. The reason is
# built from bytes the caller sent, so it is bounded rather than echoed whole.
_MAX_EMBED_ERROR_REASON_CHARS: Final = 200


def _truncated_reason(exc: Exception) -> str:
    """`exc`'s message, capped, saying how much was left out when it is capped.

    Without the count a capped reason reads like the whole one, so a caller
    debugging a large payload cannot tell that torch said more than this.
    """
    reason: Final = str(exc).strip() or type(exc).__name__
    omitted: Final = len(reason) - _MAX_EMBED_ERROR_REASON_CHARS
    if omitted <= 0:
        return reason
    return (
        f"{reason[:_MAX_EMBED_ERROR_REASON_CHARS]}"
        f"... ({omitted} more characters truncated)"
    )


def safe_load_prompt_embeds(
    model_config: "ModelConfig",
    embed: bytes,
) -> torch.Tensor:
    if not model_config.enable_prompt_embeds:
        raise VLLMValidationError(
            "You must set `--enable-prompt-embeds` to input `prompt_embeds`.",
            parameter="prompt_embeds",
        )

    # Decoded outside the try below so that a body which is not base64 keeps
    # raising binascii.Error, which is a ValueError and already answered 400.
    payload: Final = pybase64.b64decode(embed, validate=True)

    with check_sparse_tensor_invariants_threadsafe():
        try:
            tensor = torch.load(
                BytesIO(payload),
                weights_only=True,
                map_location=torch.device("cpu"),
            )
        except (torch.OutOfMemoryError, MemoryError):
            # Running out of memory is the server's condition, not the caller's.
            raise
        except Exception as exc:
            # Everything else here is about the bytes the caller sent, and torch
            # has no single error type for "this is not a tensor file": the zip
            # reader raises `RuntimeError`, the legacy pickle path raises
            # `UnpicklingError`, `KeyError` or `IndexError`, an empty payload
            # raises `EOFError`. None of them is a `ValueError`, so the
            # entrypoints' fallback mapped them all to 500. Catching by
            # behaviour rather than by type also survives torch changing which
            # error it raises for a given malformed payload.
            #
            # torch's reason is worth keeping -- for a malformed sparse tensor
            # it names the offending index -- but it is built from the caller's
            # own bytes, so it is truncated rather than echoed whole.
            raise VLLMValidationError(
                "`prompt_embeds` could not be deserialized as a torch tensor: "
                f"{_truncated_reason(exc)}",
                parameter="prompt_embeds",
            ) from exc
        tensor = safe_to_dense(tensor, parameter="prompt_embeds")

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
