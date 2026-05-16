# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
import sys

from vllm.logger import init_logger
from vllm.platforms import current_platform

logger = init_logger(__name__)

if "UCX_RCACHE_MAX_UNRELEASED" not in os.environ:
    if "nixl" in sys.modules or "rixl" in sys.modules:
        logger.warning_once(
            "NIXL was already imported, we can't reset "
            "UCX_RCACHE_MAX_UNRELEASED. "
            "Please set it to '1024' manually."
        )
    else:
        logger.info_once(
            "Setting UCX_RCACHE_MAX_UNRELEASED to '1024' to avoid a rare "
            "memory leak in UCX when using NIXL."
        )
        os.environ["UCX_RCACHE_MAX_UNRELEASED"] = "1024"

try:
    if not current_platform.is_rocm():
        from nixl._api import nixl_agent as NixlWrapper
    else:
        from rixl._api import nixl_agent as NixlWrapper

    logger.info_once("NIXL is available")
except ImportError:
    logger.warning_once("NIXL is not available")
    NixlWrapper = None  # type: ignore[assignment, misc]

try:
    if not current_platform.is_rocm():
        from nixl._api import nixl_agent_config
    else:
        from rixl._api import nixl_agent_config
except ImportError:
    nixl_agent_config = None  # type: ignore[assignment]
    logger.warning_once("NIXL agent config is not available")

try:
    if not current_platform.is_rocm():
        from nixl._bindings import nixlXferTelemetry
    else:
        from rixl._bindings import nixlXferTelemetry
except ImportError:
    nixlXferTelemetry = None  # type: ignore[assignment, misc]

__all__ = ["NixlWrapper", "nixl_agent_config", "nixlXferTelemetry"]
