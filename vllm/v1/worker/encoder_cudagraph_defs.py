# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Data transfer objects for encoder CUDA graph management."""

from collections.abc import Callable
from dataclasses import dataclass, field

import torch

EncoderCudaGraphPaddingLogic = Callable[[torch.Tensor, torch.Tensor], None]


@dataclass
class EncoderItemSpec:
    """Description of a single encoder input item.

    Returned by ``get_encoder_cudagraph_item_specs()`` to describe each
    image or video in a batch without the manager needing to understand
    model-specific input formats.
    """

    input_size: int
    """Number of input patches/rows for this item."""

    output_tokens: int
    """Number of output tokens after encoder processing (e.g. after
    spatial merge)."""

    path_output_tokens: dict[str, int] = field(default_factory=dict)
    """Per-path output token counts for multi-path encoders.

    Single-path encoders leave this empty and use ``output_tokens`` for the
    default path.
    """

    def get_path_output_tokens(self, path: str) -> int:
        if path == "default" and not self.path_output_tokens:
            return self.output_tokens
        return self.path_output_tokens.get(path, 0)


@dataclass(frozen=True)
class EncoderCudaGraphPathConfig:
    """Capture policy for one independently replayable encoder path."""

    min_token_budget: int | None = None
    """Smallest capture budget, or the model default minimum when unset."""

    allow_zero_tokens: bool = False
    """Whether a batch may omit this path entirely."""


@dataclass
class EncoderCudaGraphConfig:
    """Configuration for encoder CUDA graph management.

    Provided by the model at init time via
    ``get_encoder_cudagraph_config()``. Values are fixed for the
    lifetime of the manager.
    """

    modalities: list[str]
    """Supported modalities (e.g. ["image"])."""

    buffer_keys: list[str]
    """Keys for the tensor buffers recorded into the CUDA graph.
    Before replay the manager zeros then slice-copies new data
    into these buffers."""

    out_hidden_size: int
    """Output hidden dim of the vision encoder.
    Used for DP gather buffer allocation."""

    padding_logics: dict[str, EncoderCudaGraphPaddingLogic] = field(
        default_factory=dict
    )
    """Optional per-buffer replay padding/copy logic.
    If absent for a key, the manager zeros the capture buffer and slice-copies
    the replay buffer into it."""

    max_frames_per_video: int = 1
    """Maximum number of frames per video.
    Only relevant when "video" is in ``modalities``.
    Image-only models can use the default of 1."""

    paths: dict[str, EncoderCudaGraphPathConfig] = field(
        default_factory=lambda: {"default": EncoderCudaGraphPathConfig()}
    )
    """Independently captured encoder paths keyed by their forward name."""

    enable_secondary_capture_axis: bool = False
    """Whether to enable secondary capture axis."""


@dataclass
class EncoderCudaGraphCaptureInputs:
    """Everything needed for one CUDA graph capture.

    Returned by ``prepare_encoder_cudagraph_capture_inputs()``.
    """

    values: dict[str, torch.Tensor]
    """Precomputed tensor buffers that will be recorded into the
    CUDA graph.  The manager stores references to these exact
    tensor objects and copies new data into them before each
    ``graph.replay()`` call (buffer identity invariant)."""


@dataclass
class EncoderCudaGraphReplayBuffers:
    """New buffer values for graph replay, computed by the model from
    actual batch inputs.

    Returned by ``prepare_encoder_cudagraph_replay_buffers()``.
    Keys match ``EncoderCudaGraphConfig.buffer_keys``.
    """

    values: dict[str, torch.Tensor | None]
    """Data to copy into the captured buffers before replay.
    ``None`` values leave the corresponding captured buffer
    unchanged."""
