# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Backward-compatible re-export of NixlPullConnectorWorker."""

from vllm.distributed.kv_transfer.kv_connector.v1.nixl.pull_worker import (
    NixlPullConnectorWorker,
)

# Backward compatibility: NixlConnectorWorker is the pull-based worker.
NixlConnectorWorker = NixlPullConnectorWorker


__all__ = ["NixlConnectorWorker", "NixlPullConnectorWorker"]
