# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import bisect
import math
import operator
from array import array
from collections.abc import Callable, Iterable, Iterator, MutableSequence
from typing import TYPE_CHECKING, overload

from vllm.utils.gc_utils import untracked_list
from vllm.utils.import_utils import LazyLoader

from .inputs import (
    BaseMultiModalField,
    MultiModalFeatureSpec,
    MultiModalFieldElem,
    MultiModalKwargsItem,
    PlaceholderRange,
)

if TYPE_CHECKING:
    import torch
else:
    torch = LazyLoader("torch", globals(), "torch")

_Layout = tuple[tuple[str, BaseMultiModalField], ...]

# `_layout_ids` values that are not layout indices.
_NO_DATA = -1
_OPAQUE = -2

# Slot starts are aligned so that every dtype can view its bytes.
_ALIGNMENT = 64
_MIN_SEGMENT_BYTES = 1 << 20
_MAX_SEGMENT_BYTES = 8 << 20
# Layouts are found by a linear scan. Fields with per-item state, e.g. the
# slices of a flat field, can make every feature's layout distinct; past this
# many layouts, features with a new one are kept whole.
_MAX_LAYOUTS = 16


class PackedMMFeatures(MutableSequence[MultiModalFeatureSpec]):
    """The multimodal features of a request, with a leading run packed.

    A streaming session keeps the feature of every input chunk to replay them
    after a preemption. Each feature is about a dozen objects that the cyclic
    GC tracks, so full collections slow down as sessions grow. Packed features
    keep their metadata in arrays and their CPU tensors in a few shared byte
    segments, and are rebuilt on access, as new objects whose tensors are
    views of the segments. Indices stay stable: they are the input ids that the
    scheduler and the workers exchange.

    Writing to a packed index first unpacks the features from that index on.
    """

    def __init__(self, live: list[MultiModalFeatureSpec] | None = None) -> None:
        self._live = [] if live is None else live
        """The features from index `num_packed` on, kept as objects."""

        # One entry per packed feature.
        self._offsets = array("q")
        self._ends = array("q")
        self._modalities: list[str] = untracked_list()
        self._identifiers: list[str] = untracked_list()
        self._mm_hashes: list[str | None] = untracked_list()
        self._layout_ids = array("i")
        """Indices into `_layouts`, or `_NO_DATA` or `_OPAQUE`."""
        self._slot_starts = array("q", [0])
        """Feature `i` stores its tensors in slots `_slot_starts[i:i + 2]`."""
        self._opaque: dict[int, MultiModalFeatureSpec] = {}
        """Packed features that are kept whole, as their data is not CPU
        tensors."""

        self._layouts: list[_Layout] = []
        """The distinct `(key, field)` sequences of the packed data."""
        self._dtypes: list[torch.dtype] = []

        # One entry per slot, i.e. per packed tensor.
        self._slot_segments = array("i")
        self._slot_byte_starts = array("q")
        self._slot_dtypes = array("b")
        self._slot_dim_starts = array("q", [0])
        self._dims = array("q")

        self._segments: list[torch.Tensor] = []
        """Byte buffers of the packed tensors. Unpacking never reclaims their
        space: the unpacked features' tensors still view it."""
        self._segment_end = 0
        """Bytes used in the last segment."""

    @property
    def num_packed(self) -> int:
        return len(self._offsets)

    def pack_while(
        self, predicate: Callable[[int, MultiModalFeatureSpec], bool]
    ) -> int:
        """Pack the leading unpacked features while `predicate(input_id,
        feature)` holds, and return how many were packed."""
        count = 0
        for input_id, feature in enumerate(self._live, self.num_packed):
            if not predicate(input_id, feature):
                break
            count += 1
        for feature in self._live[:count]:
            self._pack(feature)
        del self._live[:count]
        return count

    def window(self, start: int, end: int) -> tuple[int, int]:
        """`get_mm_features_in_window`, without rebuilding packed features."""
        num_packed = self.num_packed
        lo = bisect.bisect_left(self._ends, start + 1)
        if lo == num_packed:
            lo += bisect.bisect_left(
                self._live,
                start + 1,
                key=lambda f: f.mm_position.offset + f.mm_position.length,
            )
        hi = bisect.bisect_left(self._offsets, end)
        if hi == num_packed:
            hi += bisect.bisect_left(
                self._live, end, key=lambda f: f.mm_position.offset
            )
        return lo, hi

    def __len__(self) -> int:
        return self.num_packed + len(self._live)

    def __iter__(self) -> Iterator[MultiModalFeatureSpec]:
        for index in range(self.num_packed):
            yield self._rebuild(index)
        yield from self._live

    @overload
    def __getitem__(self, index: int) -> MultiModalFeatureSpec: ...

    @overload
    def __getitem__(self, index: slice) -> list[MultiModalFeatureSpec]: ...

    def __getitem__(
        self, index: int | slice
    ) -> MultiModalFeatureSpec | list[MultiModalFeatureSpec]:
        if isinstance(index, slice):
            return [self[i] for i in range(*index.indices(len(self)))]
        index = self._normalize(index)
        num_packed = self.num_packed
        if index < num_packed:
            return self._rebuild(index)
        return self._live[index - num_packed]

    @overload
    def __setitem__(self, index: int, value: MultiModalFeatureSpec) -> None: ...

    @overload
    def __setitem__(
        self, index: slice, value: Iterable[MultiModalFeatureSpec]
    ) -> None: ...

    def __setitem__(self, index, value) -> None:
        if isinstance(index, slice):
            self._unpack_from(0)
            self._live[index] = value
            return
        index = self._normalize(index)
        self._unpack_from(index)
        self._live[index - self.num_packed] = value

    def __delitem__(self, index: int | slice) -> None:
        if isinstance(index, slice):
            self._unpack_from(0)
            del self._live[index]
            return
        index = self._normalize(index)
        self._unpack_from(index)
        del self._live[index - self.num_packed]

    def insert(self, index: int, value: MultiModalFeatureSpec) -> None:
        index = min(max(operator.index(index), -len(self)), len(self))
        if index < 0:
            index += len(self)
        self._unpack_from(index)
        self._live.insert(index - self.num_packed, value)

    def __repr__(self) -> str:
        return f"PackedMMFeatures(num_packed={self.num_packed}, live={self._live!r})"

    def _normalize(self, index: int) -> int:
        length = len(self)
        normalized = operator.index(index)
        if normalized < 0:
            normalized += length
        if not 0 <= normalized < length:
            raise IndexError(f"index {index} out of range for {length} features")
        return normalized

    def _pack(self, feature: MultiModalFeatureSpec) -> None:
        index = self.num_packed
        position = feature.mm_position
        self._offsets.append(position.offset)
        self._ends.append(position.offset + position.length)
        self._modalities.append(feature.modality)
        self._identifiers.append(feature.identifier)
        self._mm_hashes.append(feature.mm_hash)

        layout_id = self._layout_id(feature)
        self._layout_ids.append(layout_id)
        if layout_id == _OPAQUE:
            self._opaque[index] = feature
        elif feature.data is not None:
            for elem in feature.data.values():
                assert isinstance(elem.data, torch.Tensor)
                self._pack_tensor(elem.data)
        self._slot_starts.append(len(self._slot_segments))

    def _layout_id(self, feature: MultiModalFeatureSpec) -> int:
        data = feature.data
        if feature.mm_position.is_embed is not None:
            return _OPAQUE
        if data is None:
            return _NO_DATA
        if not all(_is_packable(elem.data) for elem in data.values()):
            return _OPAQUE
        layout = tuple((key, elem.field) for key, elem in data.items())
        if layout not in self._layouts:
            if len(self._layouts) == _MAX_LAYOUTS:
                return _OPAQUE
            self._layouts.append(layout)
        return self._layouts.index(layout)

    def _pack_tensor(self, tensor: "torch.Tensor") -> None:
        nbytes = tensor.numel() * tensor.element_size()
        start = -(-self._segment_end // _ALIGNMENT) * _ALIGNMENT
        if not self._segments or start + nbytes > self._segments[-1].numel():
            last = self._segments[-1].numel() if self._segments else 0
            size = min(max(2 * last, _MIN_SEGMENT_BYTES), _MAX_SEGMENT_BYTES)
            self._segments.append(torch.empty(max(size, nbytes), dtype=torch.uint8))
            start = 0
        self._segment_end = start + nbytes
        segment = self._segments[-1][start : start + nbytes]
        segment.view(tensor.dtype).view(tensor.shape).copy_(tensor)

        if tensor.dtype not in self._dtypes:
            self._dtypes.append(tensor.dtype)
        self._slot_segments.append(len(self._segments) - 1)
        self._slot_byte_starts.append(start)
        self._slot_dtypes.append(self._dtypes.index(tensor.dtype))
        self._dims.extend(tensor.shape)
        self._slot_dim_starts.append(len(self._dims))

    def _tensor(self, slot: int) -> "torch.Tensor":
        dtype = self._dtypes[self._slot_dtypes[slot]]
        shape = self._dims[
            self._slot_dim_starts[slot] : self._slot_dim_starts[slot + 1]
        ]
        start = self._slot_byte_starts[slot]
        nbytes = math.prod(shape) * dtype.itemsize
        segment = self._segments[self._slot_segments[slot]]
        return segment[start : start + nbytes].view(dtype).view(tuple(shape))

    def _rebuild(self, index: int) -> MultiModalFeatureSpec:
        layout_id = self._layout_ids[index]
        if layout_id == _OPAQUE:
            return self._opaque[index]
        data = None
        if layout_id != _NO_DATA:
            slots = range(self._slot_starts[index], self._slot_starts[index + 1])
            data = MultiModalKwargsItem(
                {
                    key: MultiModalFieldElem(data=self._tensor(slot), field=field)
                    for slot, (key, field) in zip(slots, self._layouts[layout_id])
                }
            )
        offset = self._offsets[index]
        return MultiModalFeatureSpec(
            data=data,
            modality=self._modalities[index],
            identifier=self._identifiers[index],
            mm_position=PlaceholderRange(
                offset=offset, length=self._ends[index] - offset
            ),
            mm_hash=self._mm_hashes[index],
        )

    def _unpack_from(self, index: int) -> None:
        """Move the packed features from `index` on back to `_live`."""
        num_packed = self.num_packed
        if index >= num_packed:
            return
        self._live[:0] = [self._rebuild(i) for i in range(index, num_packed)]

        for column in (
            self._offsets,
            self._ends,
            self._modalities,
            self._identifiers,
            self._mm_hashes,
            self._layout_ids,
        ):
            del column[index:]
        slot = self._slot_starts[index]
        del self._slot_starts[index + 1 :]
        for slot_column in (
            self._slot_segments,
            self._slot_byte_starts,
            self._slot_dtypes,
        ):
            del slot_column[slot:]
        del self._dims[self._slot_dim_starts[slot] :]
        del self._slot_dim_starts[slot + 1 :]
        for opaque_index in [i for i in self._opaque if i >= index]:
            del self._opaque[opaque_index]


def _is_packable(data: object) -> bool:
    return (
        isinstance(data, torch.Tensor)
        and data.device.type == "cpu"
        and data.layout == torch.strided
        and not data.requires_grad
    )
