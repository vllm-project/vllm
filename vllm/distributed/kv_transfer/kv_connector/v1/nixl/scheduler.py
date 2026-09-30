# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Backward-compatible re-export of NixlPullConnectorScheduler."""

from vllm.distributed.kv_transfer.kv_connector.v1.nixl.pull_scheduler import (
    NixlPullConnectorScheduler,
)

# Backward compatibility: NixlConnectorScheduler is the pull-based scheduler.
NixlConnectorScheduler = NixlPullConnectorScheduler

__all__ = ["NixlConnectorScheduler", "NixlPullConnectorScheduler"]
