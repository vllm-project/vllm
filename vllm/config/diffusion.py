# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Configuration for discrete diffusion (dLLM) models."""

from typing import Literal

from pydantic import Field

from vllm.config.utils import config


@config
class DiffusionConfig:
    """Configuration for discrete diffusion language models (dLLMs).

    dLLMs generate tokens via iterative denoising over a fixed-length canvas
    rather than left-to-right autoregressive decoding. They reuse the
    speculative-decoding data path (draft token ids, scheduled spec decode
    tokens) with overloaded semantics for block-based generation.
    """

    canvas_length: int = Field(default=None, gt=0)  # type: ignore[assignment]
    """Length of the denoising canvas (block).  Also determines the number of
    speculative tokens scheduled per step."""

    max_denoising_steps: int | None = None
    """Maximum number of denoising iterations per canvas block.
    If not set, read from the model's generation_config.json."""

    temperature: float | None = Field(default=None, ge=0)
    """Sampling temperature for the denoising sampler (engine-wide, since
    per-request sampling parameters are not supported for diffusion models).
    0 means greedy. If not set, read from the model's generation_config.json
    (model-specific key, e.g. ``diffusion_temperature``), defaulting to
    greedy for masked-diffusion models."""

    selection_policy: (
        Literal["low_confidence", "leftmost", "confidence_threshold"] | None
    ) = None
    """Which masked positions to unmask each denoising step (masked-diffusion
    models). ``low_confidence``: the most confident predictions first (LLaDA),
    top-k per step with k from the even transfer schedule.
    ``leftmost``: strictly left-to-right — with one token per step this yields
    clean per-token policy logprobs matching a leftmost-reveal RL recompute.
    ``confidence_threshold``: unmask every position whose chosen-token
    probability exceeds ``confidence_threshold`` (always at least the single
    most confident one, so each step makes progress) — the SGLang
    FastDiffuser thresholding decode.
    If not set, read from the model's generation_config.json
    (``diffusion_selection_policy``), defaulting to the model's selection policy."""

    confidence_threshold: float | None = Field(default=None, gt=0, lt=1)
    """Probability threshold for ``selection_policy="confidence_threshold"``.
    If not set, read from the model's generation_config.json
    (``diffusion_confidence_threshold``), defaulting to 0.9. Ignored by the
    other selection policies."""
