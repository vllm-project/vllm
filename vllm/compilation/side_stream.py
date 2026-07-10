# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared high-priority CUDA side streams."""

import torch

from vllm.platforms import current_platform
from vllm.utils.torch_utils import current_stream

_streams: dict[int, torch.cuda.Stream] = {}


def get_side_stream() -> torch.cuda.Stream | None:
    """Return the current device's shared high-priority CUDA stream."""
    if not current_platform.is_cuda():
        return None

    main_stream = current_stream()
    device_index = main_stream.device.index
    assert device_index is not None
    stream = _streams.get(device_index)
    if stream is None:
        _, high_priority = torch.cuda.Stream.priority_range()
        stream = torch.cuda.Stream(device=device_index, priority=high_priority)
        _streams[device_index] = stream
    return stream


def wait_side_stream() -> None:
    """Make the current stream wait for work issued on the side stream.

    No-op if the current device's side stream was never created.
    """
    if not _streams:
        return
    main_stream = current_stream()
    stream = _streams.get(main_stream.device.index)
    if stream is not None:
        main_stream.wait_stream(stream)
