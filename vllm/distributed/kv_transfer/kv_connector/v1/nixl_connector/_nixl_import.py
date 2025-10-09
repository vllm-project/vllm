# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Centralized lazy import for NIXL wrapper to avoid circular dependencies."""

from vllm.logger import init_logger

logger = init_logger(__name__)

# Lazy import nixl_wrapper to avoid loading nixl_bindings if nixl is not used
try:
    from nixl._api import nixl_agent as NixlWrapper
    from nixl._bindings import nixlXferTelemetry

    logger.info("NIXL is available")
except ImportError:
    logger.warning("NIXL is not available")
    NixlWrapper = None
    nixlXferTelemetry = None

try:
    from nixl._api import nixl_agent_config
except ImportError:
    nixl_agent_config = None
    logger.warning("NIXL agent config is not available")

__all__ = ["NixlWrapper", "nixlXferTelemetry", "nixl_agent_config"]
