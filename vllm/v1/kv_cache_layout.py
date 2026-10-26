# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""KV cache axis semantics and physical layout descriptors.

A leaf module so ``vllm.config`` can import the enum without pulling in the
full KV cache interface.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum, IntEnum

# Logical dim indices in the 5D stride permutation [L, B, H, N, C] (see: RFC #42082).
_DIM_L, _DIM_B, _DIM_H, _DIM_N, _DIM_C = 0, 1, 2, 3, 4


class KVCacheAxis(IntEnum):
    """Semantic tensor axes; KV orders K before V and STATE follows the spec."""

    BLOCK = 0
    KV = 1
    HEAD = 2
    STATE = 3
    HEAD_DIM = 4
    CONTENT = 5
    SINGLETON = 6


@dataclass(frozen=True)
class KVCacheTensorLayout:
    """Axis semantics in tensor indexing order, independent of strides.

    STATE indexes token positions or recurrent states; the cache spec defines
    their token granularity. CONTENT is opaque, with packing and encoding owned
    by the spec. Shape, stride, dtype and storage offset come from the tensor.
    """

    axes: tuple[KVCacheAxis, ...]

    def __post_init__(self) -> None:
        named_axes = [axis for axis in self.axes if axis != KVCacheAxis.SINGLETON]
        if len(set(named_axes)) != len(named_axes):
            raise ValueError("KV cache axes must be unique except SINGLETON")

    def axis_index(self, axis: KVCacheAxis) -> int:
        """Return the tensor dimension for an axis; raise ValueError if absent."""
        return self.axes.index(axis)

    def validate_shape(self, shape: Sequence[int]) -> None:
        """Check rank and fixed-size axes without reading tensor data."""
        if len(shape) != len(self.axes):
            raise ValueError(
                f"KV cache rank {len(shape)} != layout rank {len(self.axes)}"
            )
        for axis, size in zip(self.axes, shape):
            if axis == KVCacheAxis.KV and size != 2:
                raise ValueError("KV axis must have size 2")
            if axis == KVCacheAxis.SINGLETON and size != 1:
                raise ValueError("SINGLETON axis must have size 1")


class KVCacheLayout(Enum):
    """Physical layout descriptor for a KV cache group.

    The logical shape is always [L, B, H, N, <content>] (RFC #42082).
    Each member's value is a stride permutation that maps logical axes
    to physical (memory) order.
    """

    LBHNC = (0, 1, 2, 3, 4)  # [L, B, H, N, C] (identity)
    LBNHC = (0, 1, 3, 2, 4)  # [L, B, N, H, C]
    LHBNC = (0, 2, 1, 3, 4)  # [L, H, B, N, C]
    BLHNC = (1, 0, 2, 3, 4)  # [B, L, H, N, C]
    BLNHC = (1, 0, 3, 2, 4)  # [B, L, N, H, C]
    BHLNC = (1, 2, 0, 3, 4)  # [B, H, L, N, C]

    @property
    def stride_order(self) -> tuple[int, ...]:
        return self.value

    @property
    def layer_view_order(self) -> tuple[int, ...]:
        """Physical axis order of a logical 4D per-layer cache view."""
        return tuple(i - 1 for i in self.value if i != _DIM_L)

    @property
    def is_layer_compact(self) -> bool:
        """True when the layer is compact; i.e. the L dimension is outermost."""
        return self.value[_DIM_L] == 0

    @property
    def is_block_contiguous(self) -> bool:
        """True when [H, N, C] is contiguous within a block."""
        return self.value[-3:] == (_DIM_H, _DIM_N, _DIM_C)

    @property
    def is_block_compact(self) -> bool:
        """True when each page's [H, N, C] bytes form one contiguous run; i.e.
        the L and B dimensions are outermost."""
        return set(self.value[:2]) == {_DIM_L, _DIM_B}

    @property
    def is_block_outermost(self) -> bool:
        """True when B is the outermost physical dimension."""
        return self.value[0] == _DIM_B
