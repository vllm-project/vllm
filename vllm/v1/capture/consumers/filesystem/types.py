# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Types for the filesystem capture consumer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class FilesystemCaptureRequest:
    """Per-request client spec for the filesystem consumer.

    This is the raw shape that arrives via
    ``SamplingParams.capture["filesystem"]``. The consumer's
    ``validate_client_spec`` converts it into a ``CaptureSpec`` after
    delegating to the existing activation-storing admission validator.
    """

    request_id: str
    tag: str
    hooks: dict[str, Any]
    positions: str | list[int]


@dataclass
class FilesystemConsumerParams:
    """Consumer-level configuration parsed from ``params`` dict.

    Mirrors the ``ActivationWriter`` constructor arguments so the
    consumer can forward them without loss.
    """

    root: str
    writer_threads: int = 4
    queue_size: int = 1024
    timeout_seconds: float = 180.0
    on_collision: str = "overwrite"
    fd_cache_size: int = 256
