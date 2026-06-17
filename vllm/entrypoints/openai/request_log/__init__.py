# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Request/response jsonl logging via a separate writer subprocess.

The package ``__init__`` deliberately does not eagerly import
``RequestLoggerHub`` so that the ``spawn``-launched writer subprocess
can import ``writer_proc`` without dragging in the parent-side ZMQ
client and its dependencies.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vllm.entrypoints.openai.request_log.client import RequestLoggerHub


def __getattr__(name):
    if name == "RequestLoggerHub":
        from vllm.entrypoints.openai.request_log.client import (
            RequestLoggerHub,
        )

        return RequestLoggerHub
    raise AttributeError(name)


__all__ = ["RequestLoggerHub"]
