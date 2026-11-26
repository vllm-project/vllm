# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Checks of a request's auxiliary-output sampling parameters."""

from __future__ import annotations

from typing import TYPE_CHECKING

from vllm.distributed.aux_output_connector.object_store import (
    NAMESPACE_SALT_MARKER,
    aux_namespace_cache_salt,
)
from vllm.exceptions import VLLMValidationError
from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.config import AuxOutputConfig
    from vllm.sampling_params import SamplingParams

logger = init_logger(__name__)


def prepare_aux_request(
    config: AuxOutputConfig,
    params: SamplingParams,
    prompt_len: int,
    resumable: bool,
    cache_salt: str | None,
) -> str | None:
    """Check a request's auxiliary-output parameters on its own copy of them
    and fold the deprecated prompt-start alias. Returns its cache salt."""
    if not 0 <= params.aux_prompt_start <= prompt_len:
        raise VLLMValidationError(
            f"aux_prompt_start must be between 0 and "
            f"the prompt length ({prompt_len}), inclusive.",
            parameter="aux_prompt_start",
            value=params.aux_prompt_start,
        )
    if config.enabled:
        _fold_prompt_start_alias(params)
    if (namespace := params.aux_namespace) is None:
        if config.object_key_prefix and NAMESPACE_SALT_MARKER in (cache_salt or ""):
            raise VLLMValidationError(
                f"cache_salt must not contain {NAMESPACE_SALT_MARKER!r} while "
                "AuxOutputConfig.object_key_prefix is set.",
                parameter="cache_salt",
            )
        return cache_salt
    verify_aux_namespace(params)
    if (key_prefix := config.object_key_prefix) is None or not config.enabled:
        raise VLLMValidationError(
            "aux_namespace requires AuxOutputConfig.object_key_prefix and "
            "enable_return_routed_experts.",
            parameter="aux_namespace",
        )
    if resumable:
        raise VLLMValidationError(
            "aux_namespace does not support streaming input.",
            parameter="aux_namespace",
        )
    return aux_namespace_cache_salt(key_prefix, namespace, cache_salt)


def verify_aux_namespace(params: SamplingParams) -> None:
    """Check `aux_namespace` and the rules of a request with one. Called
    once the request's parameters are final. Its objects are under
    `AuxOutputConfig.object_key_prefix`; `/` separates the segments of a
    namespace nested in the namespace of its leading segments, whose release
    also removes it."""
    if (namespace := params.aux_namespace) is None:
        return
    if not namespace:
        reason = "must not be empty"
    elif params.stop:
        reason = "does not support stop strings; use stop_token_ids"
    elif params.n > 1:
        reason = "does not support n > 1"
    else:
        return
    raise VLLMValidationError(
        f"aux_namespace {reason}.", parameter="aux_namespace", value=namespace
    )


def _fold_prompt_start_alias(params: SamplingParams) -> None:
    """Fold the deprecated `routed_experts_prompt_start` into
    `aux_prompt_start`; 0 counts as unset."""
    alias = params.routed_experts_prompt_start
    if not alias:
        return
    logger.warning_once(
        "routed_experts_prompt_start is deprecated; use aux_prompt_start."
    )
    if params.aux_prompt_start not in (0, alias):
        raise VLLMValidationError(
            f"aux_prompt_start ({params.aux_prompt_start}) and "
            f"routed_experts_prompt_start ({alias}) disagree.",
            parameter="aux_prompt_start",
        )
    params.aux_prompt_start = alias
    params.routed_experts_prompt_start = 0
