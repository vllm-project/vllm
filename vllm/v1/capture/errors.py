# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exception types for the capture-consumer framework."""

from __future__ import annotations


class CaptureValidationError(ValueError):
    """Raised by ``CaptureConsumer.validate_client_spec`` when a
    per-request client spec is invalid.

    The serving layer converts this into an HTTP 400 at admission time.
    """


class UnknownCaptureConsumerError(ValueError):
    """Raised at engine init when a configured consumer name has no
    matching entry point in the ``vllm.capture_consumers`` group."""
