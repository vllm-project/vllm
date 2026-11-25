# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Request validation for discrete diffusion models."""

from typing import TYPE_CHECKING

from vllm.exceptions import VLLMValidationError
from vllm.sampling_params import _SAMPLING_EPS, MAX_LOGPROB_TOKEN_IDS

if TYPE_CHECKING:
    from vllm.sampling_params import SamplingParams


def validate_constrained_diffusion(
    params: "SamplingParams",
    *,
    vocab_size: int,
    tensor_parallel_size: int = 1,
) -> None:
    extra = params.extra_args
    constrained = extra.get("diffusion_constrained") if extra else None
    if constrained is None:
        return
    if not isinstance(constrained, (bool, int)) or constrained not in (0, 1):
        raise VLLMValidationError(
            "diffusion_constrained must be a boolean (or 0/1).",
            parameter="diffusion_constrained",
        )
    if not constrained:
        return
    if tensor_parallel_size > 1:
        raise VLLMValidationError(
            "Constrained DiffusionGemma reads require tensor parallel size 1.",
            parameter="diffusion_constrained",
        )
    token_ids = params.logprob_token_ids
    if not token_ids:
        raise VLLMValidationError(
            "diffusion_constrained needs logprob_token_ids: they are the allowed set.",
            parameter="logprob_token_ids",
        )
    if len(token_ids) > MAX_LOGPROB_TOKEN_IDS or any(
        not isinstance(token_id, int) or not 0 <= token_id < vocab_size
        for token_id in token_ids
    ):
        raise VLLMValidationError(
            "Constrained reads require valid token IDs within the list limit.",
            parameter="logprob_token_ids",
        )
    if params.logprobs is not None and params.logprobs != len(token_ids):
        raise VLLMValidationError(
            "logprobs must equal the number of constrained token IDs.",
            parameter="logprobs",
        )


def validate_diffusion_sampling_params(
    params: "SamplingParams",
    *,
    canvas_length: int | None,
    vocab_size: int,
    tensor_parallel_size: int = 1,
) -> None:
    """Validate request options and normalize one-canvas reads at admission.

    Diffusion models denoise a whole canvas per step, so most per-request
    sampling parameters are not supported. Models whose sampler ignores the
    request temperature or penalties warn about it.
    """
    if (
        params.min_p > _SAMPLING_EPS
        or params.seed is not None
        or params.min_tokens > 0
        or params.logit_bias
        or params.bad_words
        or params.allowed_token_ids
    ):
        raise VLLMValidationError(
            "The min_p, seed, min_tokens, logit_bias, "
            "bad_words, and allowed_token_ids sampling parameters "
            "are not yet supported with diffusion models."
        )

    extra = params.extra_args
    if not extra:
        return

    def check(valid: bool, message: str) -> None:
        if not valid:
            raise VLLMValidationError(message, parameter="extra_args")

    width = extra.get("diffusion_canvas_length")
    if width is not None:
        check(
            isinstance(width, int)
            and not isinstance(width, bool)
            and width > 0
            and (canvas_length is None or width <= canvas_length),
            "diffusion_canvas_length must be a positive integer no larger "
            "than the served canvas.",
        )
    # The request's canvas: its own width, else the served one.
    expected_len = width or canvas_length

    seed = extra.get("diffusion_seed_canvas")
    if seed is not None:
        check(
            isinstance(seed, (list, tuple))
            and all(isinstance(t, int) and not isinstance(t, bool) for t in seed),
            "diffusion_seed_canvas must be a list of token ids.",
        )
        check(
            all(0 <= t < vocab_size for t in seed),
            f"diffusion_seed_canvas ids must be in [0, {vocab_size}).",
        )
        check(
            expected_len is None or len(seed) == expected_len,
            f"diffusion_seed_canvas must hold exactly {expected_len} ids, "
            f"got {len(seed)}.",
        )

    pins = extra.get("diffusion_pinned")
    if pins is not None:
        check(
            isinstance(pins, (list, tuple))
            and all(isinstance(p, int) and not isinstance(p, bool) for p in pins),
            "diffusion_pinned must be a list of canvas positions.",
        )
        check(
            seed is not None,
            "diffusion_pinned needs a diffusion_seed_canvas to hold.",
        )
        check(
            all(p >= 0 and (expected_len is None or p < expected_len) for p in pins),
            "diffusion_pinned positions must lie inside the canvas.",
        )

    cap = extra.get("diffusion_max_steps")
    if cap is not None:
        check(
            isinstance(cap, int) and not isinstance(cap, bool) and cap > 0,
            "diffusion_max_steps must be a positive integer.",
        )

    # The OpenAI server's vllm_xargs narrows JSON booleans to 0/1.
    read_only = extra.get("diffusion_read_only")
    if read_only is not None:
        check(
            isinstance(read_only, (bool, int)) and read_only in (0, 1),
            "diffusion_read_only must be a boolean (or 0/1).",
        )
    if read_only:
        # One canvas is the whole output. Cap max_tokens so the scheduler
        # ends the request there, and ignore EOS so an end-of-turn token
        # drawn into a noise slot does not end it early.
        if expected_len is not None:
            params.max_tokens = min(params.max_tokens or expected_len, expected_len)
        params.ignore_eos = True

    validate_constrained_diffusion(
        params, vocab_size=vocab_size, tensor_parallel_size=tensor_parallel_size
    )
