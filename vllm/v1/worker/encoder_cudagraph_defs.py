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

    global_output_tokens: int = 0
    """Number of output tokens from the global image path.
    Only used when ``EncoderCudaGraphConfig.enable_dual_path_graph`` is True."""

    local_output_tokens: int = 0
    """Number of output tokens from the local patch path.
    Only used when ``EncoderCudaGraphConfig.enable_dual_path_graph`` is True."""


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

    enable_dual_path_graph: bool = False
    """If True, the manager captures two independent graph sets
    (global + local) and runs dual-path graph selection during inference."""

    global_token_per_image: int = 0
    """Tokens per global image (e.g. 272 for DeepSeek-OCR).
    Only used when ``enable_dual_path_graph`` is True."""

    local_token_per_patch: int = 0
    """Tokens per local patch (e.g. 100 for DeepSeek-OCR).
    Only used when ``enable_dual_path_graph`` is True."""

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
