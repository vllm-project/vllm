# SPDX-License-Identifier: Apache-2.0
"""KVFormatSpec geometry layer.

``base.py`` is the interface + shape rendering, ``<engine_kv_format>.py`` are
the per-format implementations, and ``registry.py`` is the format -> spec table.
"""

# First Party
from lmcache.v1.gpu_connector.kv_format.specs.base import (
    KVFormatSpec,
    concrete_shape,
    describe_shape,
)
from lmcache.v1.gpu_connector.kv_format.specs.registry import get_spec, get_spec_class

__all__ = [
    "KVFormatSpec",
    "concrete_shape",
    "describe_shape",
    "get_spec",
    "get_spec_class",
]
