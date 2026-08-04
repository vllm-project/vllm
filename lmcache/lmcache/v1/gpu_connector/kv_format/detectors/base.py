# SPDX-License-Identifier: Apache-2.0
"""Per-engine KV cache discovery interface.

One :class:`EngineDetector` per serving engine reshapes a raw ``kv_caches`` into
the canonical structure the specs expect and identifies its ``EngineKVFormat``,
in a single step. The engine -> detector table is in ``registry.py``.
"""

# mypy: disable-error-code="union-attr"
# Standard
from abc import ABC, abstractmethod
from typing import ClassVar, Optional

# First Party
from lmcache.utils import EngineType
from lmcache.v1.gpu_connector.kv_format.types import DiscoverableKVCache, LayoutHints
import lmcache.c_ops as lmc_ops


def measure_list_depth_until_tensor(
    kv_caches: DiscoverableKVCache,
) -> tuple[int, int, DiscoverableKVCache]:
    """Return ``(list_depth, tensor_ndim, first_tensor)`` for *kv_caches*.

    Descends the first element of each list down to the inner tensor, counting
    the list-nesting depth on the way.
    """
    list_depth = 0
    node = kv_caches
    while isinstance(node, list):
        if not node:
            raise ValueError("encountered an empty kv_caches list")
        list_depth += 1
        node = node[0]
    return list_depth, node.ndim, node


class EngineDetector(ABC):
    """Reshape + identify the ``EngineKVFormat`` for one serving engine."""

    engine_type: ClassVar[EngineType]

    @abstractmethod
    def discover(
        self, kv_caches: DiscoverableKVCache, layout_hints: LayoutHints
    ) -> "tuple[Optional[lmc_ops.EngineKVFormat], DiscoverableKVCache]":
        """Return ``(format, canonical_kv_caches)``, or ``(None, kv_caches)``.

        Reshapes this engine's raw layout into the canonical structure the specs
        expect, and identifies which ``EngineKVFormat`` it is (``None`` if the
        structure is unrecognized).
        """
