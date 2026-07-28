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

    selection_policy: Literal["low_confidence", "leftmost"] | None = None
    """Which masked positions to unmask each denoising step (masked-diffusion
    models). ``low_confidence``: the most confident predictions first (LLaDA).
    ``leftmost``: strictly left-to-right — with one token per step this yields
    clean per-token policy logprobs matching a leftmost-reveal RL recompute.
    If not set, read from the model's generation_config.json
    (``diffusion_selection_policy``), defaulting to ``low_confidence``."""
