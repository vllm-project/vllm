# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Configuration for discrete diffusion (dLLM) models."""

from typing import Any

from pydantic import Field

from vllm.config.utils import config


@config
class DiffusionModelOptions:
    """Base class for model-specific `DiffusionConfig.model_options`."""


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

    model_options: dict[str, Any] | DiffusionModelOptions = Field(default_factory=dict)
    """Model-specific options. The accepted keys depend on the model and are
    validated once the model is known, e.g. `NemotronDiffusionOptions` or
    `DiffusionGemmaDiffusionOptions` in `vllm.model_executor.models.config`."""
