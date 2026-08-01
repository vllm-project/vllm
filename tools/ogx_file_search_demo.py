# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compatibility import for the OGX file_search demo handler.

Register as a vLLM plugin by adding the following to your ``pyproject.toml``::

    [project.entry-points."vllm.file_search_plugins"]
    ogx = "vllm.plugins.file_search.ogx_handler:create_handler"

Environment variables:
  OGX_URL  - Base URL of the OGX server (default: http://localhost:8321)
  OGX_TIMEOUT - HTTP timeout in seconds (default: 10)
"""

from vllm.plugins.file_search.ogx_handler import (
    OGXFileSearchHandler,
    create_handler,
)

__all__ = ["OGXFileSearchHandler", "create_handler"]
