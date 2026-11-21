# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Parallelism-free layouts of objects that hold whole KV cache layers.

A connector that shares KV between instances with different parallel configs
stores a block as objects whose bytes depend only on the model. Placing whole
layers back to back in model order gives every rank of every pipeline-parallel
config the same layout: a rank fills the slots of the layers it holds, and the
layers of all pipeline stages together fill the object.
"""

from collections.abc import Iterable, Mapping
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class LayerSlot:
    """The bytes of one layer within an object."""

    offset: int
    size: int


@dataclass(frozen=True, slots=True)
class LayerObjectLayout:
    """Byte layout of an object holding whole KV cache layers back to back."""

    slots: Mapping[str, LayerSlot]
    """Each layer's slot, in object order."""

    size: int
    """Total bytes of the object."""

    @classmethod
    def pack(cls, layer_sizes: Iterable[tuple[str, int]]) -> "LayerObjectLayout":
        """Place the layers back to back in the given order.

        Args:
            layer_sizes: Each layer's name and size in bytes, in the order every
                instance sharing the object agrees on (e.g. model order).

        Returns:
            The layout, with the first layer at offset 0.

        Raises:
            ValueError: If a layer repeats or a size is not positive.

        """
        slots: dict[str, LayerSlot] = {}
        offset = 0
        for layer_name, size in layer_sizes:
            if size <= 0:
                raise ValueError(f"Layer {layer_name} has non-positive size {size}")
            if layer_name in slots:
                raise ValueError(f"Layer {layer_name} appears twice")
            slots[layer_name] = LayerSlot(offset, size)
            offset += size
        return cls(slots, offset)
