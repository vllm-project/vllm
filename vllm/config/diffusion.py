# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Configuration for discrete diffusion (dLLM) models."""

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
