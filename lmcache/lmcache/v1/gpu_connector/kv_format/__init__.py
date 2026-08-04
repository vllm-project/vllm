# SPDX-License-Identifier: Apache-2.0
"""Format-dispatched geometry for GPU KV caches.

Public surface:

- :class:`KVFormatSpec` -- per-format geometry interface.
- :func:`get_spec` / :func:`get_spec_class` -- look up the spec for a format.
- :func:`detect_format` -- normalize a raw ``kv_caches`` and discover its format.
- :func:`describe_shape` / :func:`concrete_shape` -- render a format's symbolic /
  numeric shape string.
"""

# First Party
from lmcache.v1.gpu_connector.kv_format.detection import (
    detect_format,
    extract_kv_cache_shapes,
)
from lmcache.v1.gpu_connector.kv_format.specs import (
    KVFormatSpec,
    concrete_shape,
    describe_shape,
    get_spec,
    get_spec_class,
)

__all__ = [
    "KVFormatSpec",
    "concrete_shape",
    "describe_shape",
    "detect_format",
    "extract_kv_cache_shapes",
    "get_spec",
    "get_spec_class",
]
