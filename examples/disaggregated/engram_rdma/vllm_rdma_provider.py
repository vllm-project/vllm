# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compatibility alias for the bundled experimental Engram provider."""

from vllm.distributed.engram_rdma.provider import close_reader, get_reader

__all__ = ["close_reader", "get_reader"]
