# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Request validation for discrete diffusion models."""

from typing import TYPE_CHECKING

from vllm.exceptions import VLLMValidationError

if TYPE_CHECKING:
    from vllm.sampling_params import SamplingParams


def validate_diffusion_sampling_params(
    params: "SamplingParams",
    *,
    canvas_length: int | None,
    vocab_size: int,
    async_scheduling: bool | None,
) -> None:
    """Validate request options and normalize one-canvas reads at admission."""
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

    check(
        bool(async_scheduling)
        or width is None
        or canvas_length is None
        or width == canvas_length,
        "A diffusion_canvas_length smaller than the served canvas "
        "requires --async-scheduling.",
    )
