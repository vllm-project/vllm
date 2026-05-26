# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Backward-compatible re-export of NixlPullConnector as NixlConnector."""

from vllm.distributed.kv_transfer.kv_connector.v1.nixl.pull_connector import (
    NixlPullConnector,
)

# Backward compatibility: NixlConnector is the pull-based connector.
NixlConnector = NixlPullConnector

__all__ = ["NixlConnector", "NixlPullConnector"]
