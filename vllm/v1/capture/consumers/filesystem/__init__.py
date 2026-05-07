# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Filesystem capture consumer — streams activations to disk."""

from __future__ import annotations

from vllm.v1.capture.consumers.filesystem.consumer import FilesystemConsumer
from vllm.v1.capture.consumers.filesystem.types import (
    FilesystemCaptureRequest,
)

__all__ = [
    "FilesystemCaptureRequest",
    "FilesystemConsumer",
]
