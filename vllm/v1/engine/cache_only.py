# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Request-scoped validation for cache-only P/D prefill."""

from typing import Any

from vllm.exceptions import VLLMValidationError
from vllm.pooling_params import PoolingParams
from vllm.sampling_params import SamplingParams


def validate_dsv41_cache_only_request(
    sampling_params: SamplingParams | None,
    pooling_params: PoolingParams | None,
    has_mm_features: bool,
) -> None:
    """Reject unsupported P requests before they reach the EngineCore loop."""
    if pooling_params is not None or sampling_params is None:
        raise VLLMValidationError(
            "dsv41_encoder_only_prefill supports generative requests only."
        )
    extra_args = sampling_params.extra_args
    params: Any = (
        extra_args.get("kv_transfer_params") if isinstance(extra_args, dict) else None
    )
    if (
        not isinstance(params, dict)
        or not params.get("do_remote_decode")
        or params.get("do_remote_prefill")
        or not params.get("cache_only")
    ):
        raise VLLMValidationError(
            "A dsv41_encoder_only_prefill producer requires a cache-only "
            "remote decode request on NixlConnector."
        )
    if has_mm_features:
        raise VLLMValidationError(
            "dsv41_encoder_only_prefill initially supports text-only requests."
        )
    if sampling_params.prompt_logprobs is not None:
        raise VLLMValidationError(
            "dsv41_encoder_only_prefill does not support prompt logprobs; use "
            "the conventional P/D path."
        )
