# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Data transfer objects for encoder CUDA graph management."""

from dataclasses import dataclass
from typing import Any

import torch


@dataclass
class EncoderCudaGraphConfig:
    """Configuration for encoder CUDA graph management.

    Provided by the model at init time via
    ``get_encoder_cudagraph_config()``. Values are fixed for the
    lifetime of the manager.
    """

    modalities: list[str]
    """Supported modalities (e.g. ["image"])."""

    input_key_by_modality: dict[str, str]
    """Per-modality input tensor key mapping, e.g.
    {"image": "pixel_values", "video": "pixel_values_videos"}.
    """

    buffer_keys: list[str]
    """Keys for the tensor buffers recorded into the CUDA graph.
    Before replay the manager zeros then slice-copies new data
    into these buffers."""

    out_hidden_size: int
    """Output hidden dim of the vision encoder.
    Used for DP gather buffer allocation."""


@dataclass
class EncoderCudaGraphCaptureInputs:
    """Everything needed for one CUDA graph capture.

    Returned by ``prepare_encoder_cudagraph_capture_inputs()``.
    """

    mm_kwargs: dict[str, Any]
    """Dummy forward inputs (model-specific keys).
    For Qwen3-VL this contains pixel_values and grid_thw."""

    buffers: dict[str, torch.Tensor]
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

    buffers: dict[str, torch.Tensor | None]
    """Data to copy into the captured buffers before replay.
    ``None`` values leave the corresponding captured buffer
    unchanged."""
