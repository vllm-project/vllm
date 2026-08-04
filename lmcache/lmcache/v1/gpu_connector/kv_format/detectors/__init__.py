# SPDX-License-Identifier: Apache-2.0
"""Per-engine detectors.

``base.py`` is the interface, ``<engine>.py`` are the per-engine
implementations, and ``registry.py`` is the engine -> detector table.
"""

# First Party
from lmcache.v1.gpu_connector.kv_format.detectors.registry import get_detector

__all__ = ["get_detector"]
