# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from abc import ABC, abstractmethod
from collections import UserDict, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import cached_property, partial
from itertools import accumulate
from typing import (
    TYPE_CHECKING,
    Any,
    Literal,
    Optional,
    TypeAlias,
    TypedDict,
    Union,
    cast,
    final,
)

import numpy as np
from PIL.Image import Image
from typing_extensions import NotRequired, TypeVar

from vllm.utils.collection_utils import is_list_of
from vllm.utils.import_utils import LazyLoader
from vllm.utils.jsontree import json_map_leaves

if TYPE_CHECKING:
    import torch
    import torch.types
    from transformers.feature_extraction_utils import BatchFeature

    from .media import MediaWithBytes
else:
    torch = LazyLoader("torch", globals(), "torch")

_T = TypeVar("_T")

HfImageItem: TypeAlias = Union["Image", np.ndarray, "torch.Tensor"]
"""
A `transformers.image_utils.ImageInput` representing a single image
item, which can be passed to a HuggingFace `ImageProcessor`.
"""

HfVideoItem: TypeAlias = Union[
    list["Image"], np.ndarray, "torch.Tensor", list[np.ndarray], list["torch.Tensor"]
]
"""
A `transformers.image_utils.VideoInput` representing a single video
item, which can be passed to a HuggingFace `VideoProcessor`.
"""

HfAudioItem: TypeAlias = Union[list[float], np.ndarray, "torch.Tensor"]
"""
Represents a single audio
item, which can be passed to a HuggingFace `AudioProcessor`.
"""

ImageItem: TypeAlias = Union[HfImageItem, "torch.Tensor", "MediaWithBytes[HfImageItem]"]
"""
A `transformers.image_utils.ImageInput` representing a single image
item, which can be passed to a HuggingFace `ImageProcessor`.

Alternatively, a 3-D tensor or batch of 2-D tensors,
which are treated as image embeddings;
these are directly passed to the model without HF processing.
"""

VideoItem: TypeAlias = Union[
    HfVideoItem, "torch.Tensor", tuple[HfVideoItem, dict[str, Any]]
]
"""
A `transformers.video_utils.VideoInput` representing a single video item. 
This can be passed to a HuggingFace `VideoProcessor` 
with `transformers.video_utils.VideoMetadata`.

Alternatively, a 3-D tensor or batch of 2-D tensors,
which are treated as video embeddings;
these are directly passed to the model without HF processing.
"""

AudioItem: TypeAlias = Union[HfAudioItem, tuple[np.ndarray, float], "torch.Tensor"]
"""
Represents a single audio
item, which can be passed to a HuggingFace `AudioProcessor`.

Alternatively, a tuple `(audio, sampling_rate)`, where the sampling rate
is different from that expected by the model;
these are resampled to the model's sampling rate before being processed by HF.

Alternatively, a 3-D tensor or batch of 2-D tensors,
which are treated as audio embeddings;
these are directly passed to the model without HF processing.
"""

ModalityData: TypeAlias = _T | list[_T | None] | None
"""
Either a single data item, or a list of data items. Can only be None if UUID
is provided.

The number of data items allowed per modality is restricted by
`--limit-mm-per-prompt`.
"""


class VisionChunkImage(TypedDict):
    """Represents an image wrapped as a vision chunk."""

    type: Literal["image"]
    image: Image
    uuid: str | None


class VisionChunkVideo(TypedDict):
    """Represents a video chunk with metadata."""

    type: Literal["video_chunk"]
    video_chunk: list[Image]
    uuid: str | None
    prompt: str
    video_idx: int


VisionChunk = VisionChunkImage | VisionChunkVideo
"""A vision chunk is either an image or a video chunk."""


@final
class MultiModalDataBuiltins(TypedDict, total=False):
    """Type annotations for modality types predefined by vLLM."""

    image: ModalityData[ImageItem]
    """The input image(s)."""

    video: ModalityData[VideoItem]
    """The input video(s)."""

    audio: ModalityData[AudioItem]
    """The input audio(s)."""

    vision_chunk: ModalityData[VisionChunk]
    """The input visual atom(s) - unified modality for images and video chunks."""


MultiModalDataDict: TypeAlias = Mapping[str, ModalityData[Any]]
"""
A dictionary containing an entry for each modality type to input.

The built-in modalities are defined by
[`MultiModalDataBuiltins`][vllm.multimodal.inputs.MultiModalDataBuiltins].
"""

MultiModalUUIDDict: TypeAlias = Mapping[str, list[str | None] | str]
"""
A dictionary containing user-provided UUIDs for items in each modality.
If a UUID for an item is not provided, its entry will be `None` and
MultiModalHasher will compute a hash for the item.

The UUID will be used to identify the item for all caching purposes
(input processing caching, embedding caching, prefix caching, etc).
"""


@dataclass(frozen=True)
class PlaceholderRange:
    """
    Placeholder location information for multi-modal data.

    Example:

    Prompt: `AAAA BBBB What is in these images?`

    Images A and B will have:

    ```
    A: PlaceholderRange(offset=0, length=4)
    B: PlaceholderRange(offset=5, length=4)
    ```
    """

    offset: int
    """The start index of the placeholder in the prompt."""

    length: int
    """The length of the placeholder."""

    is_embed: Optional["torch.Tensor"] = None
    """
    A boolean mask of shape `(length,)` indicating which positions
    between `offset` and `offset + length` to assign embeddings to.
    """

    @cached_property
    def embeds_cumsum(self) -> torch.Tensor | None:
        return None if self.is_embed is None else self.is_embed.cumsum(dim=0)

    @cached_property
    def get_num_embeds(self) -> int:
        if self.embeds_cumsum is None:
            return self.length

        return int(self.embeds_cumsum[-1])

    def get_embeds_indices_in_range(
        self, start_idx: int, end_idx: int
    ) -> tuple[int, int]:
        """
        Returns the starting and ending indices of the embeddings of encoder outputs
        in the range of [start_idx, end_idx) in the placeholders.

        For example, given:
        PlaceholderRange(offset=2, length=5, is_embed=[False, True, False, True, True])

        If start_idx=3 and end_idx=5, the output is (1, 3) because we want to get
        the second and the third embeddings from the encoder output.
        """
        if self.embeds_cumsum is None:
            return start_idx, end_idx

        embeds_start_idx = (
            int(self.embeds_cumsum[start_idx - 1]) if start_idx > 0 else 0
        )
        embeds_end_idx = int(self.embeds_cumsum[end_idx - 1])

        return embeds_start_idx, embeds_end_idx

    def extract_embeds_range(self) -> list[tuple[int, int]]:
        """Extract the start and end indices of the embedded region in prompt.

        For example, given `PlaceholderRange(offset=2, length=5)` and
        `is_embed = [False, True, False, True, True]`, the output is
        `[(1 + offset, 1 + offset), (3 + offset, 4 + offset)]`.

        Returns:
            A tuple `(start, end)` representing the start and end
            indices (inclusive) of the embedded region.
            Returns full placeholder range if `is_embed` is `None`.
        """
        if self.is_embed is None:
            return [(self.offset, self.offset + self.length - 1)]

        mask_i = self.is_embed.int()
        starts = torch.nonzero(
            torch.diff(mask_i, prepend=mask_i.new_zeros(1)) == 1
        ).flatten()
        ends = torch.nonzero(
            torch.diff(mask_i, append=mask_i.new_zeros(1)) == -1
        ).flatten()
        ranges = torch.stack((starts, ends), dim=1) + self.offset
        return [tuple(x) for x in ranges.tolist()]

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, self.__class__):
            return False
        if not (self.offset, self.length) == (other.offset, other.length):
            return False

        if self.is_embed is None:
            return other.is_embed is None
        if other.is_embed is None:
            return self.is_embed is None

        return nested_tensors_equal(self.is_embed, other.is_embed)


NestedTensors: TypeAlias = Union[
    list["NestedTensors"],
    list["torch.Tensor"],
    "torch.Tensor",
    tuple["torch.Tensor", ...],
]
"""
Uses a list instead of a tensor if the dimensions of each element do not match.
"""


def nested_tensors_equal(a: NestedTensors, b: NestedTensors) -> bool:
    """
    Equality check between
    [`NestedTensors`][vllm.multimodal.inputs.NestedTensors] objects.
    """
    if isinstance(a, torch.Tensor):
        return isinstance(b, torch.Tensor) and torch.equal(a, b)
    elif isinstance(b, torch.Tensor):
        return isinstance(a, torch.Tensor) and torch.equal(b, a)

    if isinstance(a, list):
        return isinstance(b, list) and all(
            nested_tensors_equal(a_, b_) for a_, b_ in zip(a, b)
        )
    if isinstance(b, list):
        return isinstance(a, list) and all(
            nested_tensors_equal(b_, a_) for b_, a_ in zip(b, a)
        )

    # Both a and b are scalars
    return a == b


def _nested_tensors_h2d(
    tensors: NestedTensors,
    device: torch.types.Device,
) -> NestedTensors:
    if device is None:
        return tensors

    return json_map_leaves(
        (
            lambda x: x.to(device=device, non_blocking=True)
            if isinstance(x, torch.Tensor)
            else x
        ),
        tensors,
    )


BatchedTensorInputs: TypeAlias = dict[str, NestedTensors]
"""
A dictionary containing nested tensors which have been batched via
[`MultiModalKwargsItems.get_data`][vllm.multimodal.inputs.MultiModalKwargsItems.get_data].
"""


def batched_tensors_equal(a: BatchedTensorInputs, b: BatchedTensorInputs) -> bool:
    """
    Equality check between
    [`BatchedTensorInputs`][vllm.multimodal.inputs.BatchedTensorInputs] objects.
    """
    return all(k in b and nested_tensors_equal(a[k], b[k]) for k in a)


@dataclass
class MultiModalFeatureSpec:
    """
    Represents a single multimodal input with its processed data and metadata.

    Used to track multimodal data through processing and caching.
    A request containing multiple multimodal items will have one
    `MultiModalFeatureSpec` per item.
    """

    data: Optional["MultiModalKwargsItem"]
    """
    Represents multimodal data for this feature.

    Can be `None` if the item is cached, to skip IPC between API server
    and engine core processes.
    """

    modality: str
    """The input modality, e.g., `"image"`, `"audio"`, `"video"`."""

    identifier: str
    """The hash for caching encoder outputs (with LoRA prefix if applicable)."""

    mm_position: PlaceholderRange
    """
    The location of the `modality` tokens corresponding to this item
    in the prompt, e.g., `PlaceholderRange(offset=2, length=336)`.
    """

    mm_hash: str | None = None
    """The hash for caching processor outputs (without LoRA prefix)."""

    @staticmethod
    def gather_kwargs(features: list["MultiModalFeatureSpec"], keys: set[str]):
        kwargs = defaultdict[str, list[NestedTensors]](list)

        for f in features:
            item = f.data
            if item is not None:
                for k in keys:
                    if k in item:
                        kwargs[k].append(item[k].data)

        return dict(kwargs)


@dataclass
class MultiModalFieldElem:
    """
    Represents a processed keyword argument to pass to a model for a
    [`MultiModalKwargsItem`][vllm.multimodal.inputs.MultiModalKwargsItem].
    """

    data: NestedTensors
    """
    The tensor data of this field in
    [`MultiModalKwargsItem`][vllm.multimodal.inputs.MultiModalKwargsItem],
    i.e. the value of the keyword argument to be passed to the model.

    It may be set to `None` if it is determined that the item is cached
    in `EngineCore`.
    """

    field: "BaseMultiModalField"
    """
    Defines how to combine the tensor data of this field with others
    in order to batch multi-modal items together for model inference.
    """

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, self.__class__):
            return False

        if self.data is None:
            data_equal = other.data is None
        elif other.data is None:
            data_equal = self.data is None
        else:
            data_equal = nested_tensors_equal(self.data, other.data)

        return data_equal and type(self.field) is type(other.field)  # noqa: E721


@dataclass(frozen=True, kw_only=True)
class BaseMultiModalField(ABC):
    """
    Defines how to interpret tensor data belonging to a keyword argument for
    [`MultiModalKwargsItems`][vllm.multimodal.inputs.MultiModalKwargsItems],
    and vice versa.
    """

    keep_on_cpu: bool = False
    """
    If `True`, then this field is excluded from being moved to the accelerator
    when `MultiModalKwargsItems.get_data()` is called to batch the data.
    """

    def _field_factory(self):
        f = partial(MultiModalFieldElem, field=self)

        # Allow passing data as positional argument
        def factory(data: NestedTensors) -> MultiModalFieldElem:
            return f(data=data)

        return factory

    @abstractmethod
    def build_elems(
        self,
        modality: str,
        key: str,
        data: NestedTensors,
    ) -> Sequence[MultiModalFieldElem]:
        """
        Construct
        [`MultiModalFieldElem`][vllm.multimodal.inputs.MultiModalFieldElem]
        instances to represent the provided data.

        This is the inverse of
        [`reduce_data`][vllm.multimodal.inputs.BaseMultiModalField.reduce_data].
        """
        raise NotImplementedError

    @abstractmethod
    def _reduce_data(
        self,
        batch: list[NestedTensors],
        *,
        pin_memory: bool,
    ) -> NestedTensors:
        raise NotImplementedError

    def reduce_data(
        self,
        elems: list[MultiModalFieldElem],
        *,
        device: torch.types.Device = None,
        pin_memory: bool = False,
    ) -> NestedTensors:
        """
        Merge the data from multiple instances of
        [`MultiModalFieldElem`][vllm.multimodal.inputs.MultiModalFieldElem].

        This is the inverse of
        [`build_elems`][vllm.multimodal.inputs.BaseMultiModalField.build_elems].
        """
        field_types = [type(item.field) for item in elems]
        if len(set(field_types)) > 1:
            raise ValueError(f"Cannot merge different {field_types=}")

        if device is not None and self.keep_on_cpu:
            device = "cpu"
        if pin_memory and self.keep_on_cpu:
            pin_memory = False

        batch = [elem.data for elem in elems]
        out = self._reduce_data(batch, pin_memory=pin_memory)
        return _nested_tensors_h2d(out, device=device)


@dataclass(frozen=True, kw_only=True)
class MultiModalBatchedField(BaseMultiModalField):
    """
    Info:
        [`MultiModalFieldConfig.batched`][vllm.multimodal.inputs.MultiModalFieldConfig.batched]
    """

    def build_elems(
        self,
        modality: str,
        key: str,
        data: NestedTensors,
    ) -> Sequence[MultiModalFieldElem]:
        field_factory = self._field_factory()
        return [field_factory(item) for item in data]

    def _reduce_data(
        self,
        batch: list[NestedTensors],
        *,
        pin_memory: bool,
    ) -> NestedTensors:
        if len(batch) > 0 and is_list_of(batch, torch.Tensor, check="all"):
            batch = cast(list[torch.Tensor], batch)
            if len(batch) == 1:
                # An optimization when `batch` contains only one tensor:
                # - produce exactly same result as `torch.stack(batch)`
                # - will achieve zero-copy if the tensor is contiguous
                return batch[0].unsqueeze(0).contiguous()
            first_shape = batch[0].shape
            if all(elem.shape == first_shape for elem in batch):
                out = torch.empty(
                    (len(batch), *batch[0].shape),
                    dtype=batch[0].dtype,
                    device=batch[0].device,
                    pin_memory=pin_memory,
                )
                return torch.stack(batch, out=out)

        return batch


@dataclass(frozen=True, kw_only=True)
class MultiModalFlatField(BaseMultiModalField):
    """
    Info:
        [`MultiModalFieldConfig.flat`][vllm.multimodal.inputs.MultiModalFieldConfig.flat]
        [`MultiModalFieldConfig.flat_from_sizes`][vllm.multimodal.inputs.MultiModalFieldConfig.flat_from_sizes]
    """

    slices: Sequence[slice] | Sequence[Sequence[slice]]
    dim: int = 0

    def build_elems(
        self,
        modality: str,
        key: str,
        data: NestedTensors,
    ) -> Sequence[MultiModalFieldElem]:
        field_factory = self._field_factory()
        if not is_list_of(self.slices, slice, check="all"):
            assert isinstance(data, torch.Tensor), (
                "torch.Tensor is required for multiple slices"
            )
        return [field_factory(data[cast(slice, s)]) for s in self.slices]

    def _reduce_data(
        self,
        batch: list[NestedTensors],
        *,
        pin_memory: bool,
    ) -> NestedTensors:
        if len(batch) > 0 and is_list_of(batch, torch.Tensor, check="all"):
            batch = cast(list[torch.Tensor], batch)
            if len(batch) == 1:
                # An optimization when `batch` contains only one tensor:
                # - produce exactly same result as `torch.concat(batch)`
                # - will achieve zero-copy if the tensor is contiguous
                return batch[0].contiguous()

            dim = self.dim + (self.dim < 0) * len(batch[0].shape)

            def _shape_before_after(tensor: torch.Tensor):
                return tensor.shape[:dim], tensor.shape[dim + 1 :]

            first_shape = _shape_before_after(batch[0])

            if all(_shape_before_after(elem) == first_shape for elem in batch):
                shape_before, shape_after = first_shape
                shape_concat = sum(item.shape[dim] for item in batch)
                out = torch.empty(
                    (*shape_before, shape_concat, *shape_after),
                    dtype=batch[0].dtype,
                    device=batch[0].device,
                    pin_memory=pin_memory,
                )
                return torch.concat(batch, dim=self.dim, out=out)

            # Variable-length case: non-concat dimensions differ
            # (e.g., Ultravox with different audio durations).
            # Use slice-assign approach (more efficient than padding).
            # See: https://github.com/vllm-project/vllm/issues/31658

            ndim = batch[0].ndim

            # Step 1: Compute output shape
            # - Non-concat dims: take max across batch
            # - Concat dim: sum across batch
            max_sizes: list[int] = []
            for d in range(ndim):
                if d == dim:
                    max_sizes.append(sum(t.shape[d] for t in batch))
                else:
                    max_sizes.append(max(t.shape[d] for t in batch))

            # Step 2: Create zero-initialized output tensor
            out = torch.zeros(
                max_sizes,
                dtype=batch[0].dtype,
                device=batch[0].device,
                pin_memory=pin_memory,
            )

            # Step 3: Slice-assign each tensor to its proper position
            concat_offset = 0
            for tensor in batch:
                slices: list[slice] = []
                for d in range(ndim):
                    if d == dim:
                        slices.append(
                            slice(concat_offset, concat_offset + tensor.shape[d])
                        )
                    else:
                        slices.append(slice(0, tensor.shape[d]))
                out[tuple(slices)] = tensor
                concat_offset += tensor.shape[dim]

            return out

        assert self.dim == 0, "dim == 0 is required for nested list"
        return [e for elem in batch for e in elem]


@dataclass(frozen=True, kw_only=True)
class MultiModalSharedField(BaseMultiModalField):
    """
    Info:
        [`MultiModalFieldConfig.shared`][vllm.multimodal.inputs.MultiModalFieldConfig.shared]
    """

    batch_size: int

    def build_elems(
        self,
        modality: str,
        key: str,
        data: NestedTensors,
    ) -> Sequence[MultiModalFieldElem]:
        field_factory = self._field_factory()
        return [field_factory(data)] * self.batch_size

    def _reduce_data(
        self,
        batch: list[NestedTensors],
        *,
        pin_memory: bool,
    ) -> NestedTensors:
        return batch[0]


@dataclass(frozen=True)
class MultiModalFieldConfig:
    @staticmethod
    def batched(modality: str, *, keep_on_cpu: bool = False):
        """
        Defines a field where an element in the batch is obtained by
        indexing into the first dimension of the underlying data.

        Args:
            modality: The modality of the multi-modal item that uses this
                keyword argument.
            keep_on_cpu: Whether to keep this field on the CPU for the model inputs.

        Example:

        ```
        Input:
            Data: [[AAAA]
                [BBBB]
                [CCCC]]

        Output:
            Element 1: [AAAA]
            Element 2: [BBBB]
            Element 3: [CCCC]
        ```
        """
        return MultiModalFieldConfig(
            field=MultiModalBatchedField(keep_on_cpu=keep_on_cpu),
            modality=modality,
        )

    @staticmethod
    def flat(
        modality: str,
        slices: Sequence[slice] | Sequence[Sequence[slice]],
        dim: int = 0,
        *,
        keep_on_cpu: bool = False,
    ):
        """
        Defines a field where an element in the batch is obtained by
        slicing along the first dimension of the underlying data.

        Args:
            modality: The modality of the multi-modal item that uses this
                keyword argument.
            slices: For each multi-modal item, a slice (dim=0) or a tuple of
                slices (dim>0) that is used to extract the data corresponding
                to it.
            dim: The dimension to extract data, default to 0.
            keep_on_cpu: Whether to keep this field on the CPU for the model inputs.

        Example:

        ```
        Given:
            slices: [slice(0, 3), slice(3, 7), slice(7, 9)]

        Input:
            Data: [AAABBBBCC]

        Output:
            Element 1: [AAA]
            Element 2: [BBBB]
            Element 3: [CC]
        ```

        ```
        Given:
            slices: [
                (slice(None), slice(0, 3)),
                (slice(None), slice(3, 7)),
                (slice(None), slice(7, 9))]
            dim: 1

        Input:
            Data: [[A],[A],[A],[B],[B],[B],[B],[C],[C]]

        Output:
            Element 1: [[A],[A],[A]]
            Element 2: [[B],[B],[B],[B]]
            Element 3: [[C],[C]]
        ```
        """
        return MultiModalFieldConfig(
            field=MultiModalFlatField(
                slices=slices,
                dim=dim,
                keep_on_cpu=keep_on_cpu,
            ),
            modality=modality,
        )

    @staticmethod
    def flat_from_sizes(
        modality: str,
        size_per_item: "torch.Tensor",
        dim: int = 0,
        *,
        keep_on_cpu: bool = False,
    ):
        """
        Defines a field where an element in the batch is obtained by
        slicing along the first dimension of the underlying data.

        Args:
            modality: The modality of the multi-modal item that uses this
                keyword argument.
            size_per_item: For each multi-modal item, the size of the slice
                that is used to extract the data corresponding to it.
            dim: The dimension to slice, default to 0.
            keep_on_cpu: Whether to keep this field on the CPU for the model inputs.

        Example:

        ```
        Given:
            size_per_item: [3, 4, 2]

        Input:
            Data: [AAABBBBCC]

        Output:
            Element 1: [AAA]
            Element 2: [BBBB]
            Element 3: [CC]
        ```

        ```
        Given:
            size_per_item: [3, 4, 2]
            dim: 1

        Input:
            Data: [[A],[A],[A],[B],[B],[B],[B],[C],[C]]

        Output:
            Element 1: [[A],[A],[A]]
            Element 2: [[B],[B],[B],[B]]
            Element 3: [[C],[C]]
        ```

        Info:
            [`MultiModalFieldConfig.flat`][vllm.multimodal.inputs.MultiModalFieldConfig.flat]
        """

        if size_per_item.ndim != 1:
            raise ValueError(
                "size_per_item should be a 1-D tensor, "
                f"but found shape: {size_per_item.shape}"
            )

        slice_idxs = [0, *accumulate(size_per_item)]
        slices = [
            (slice(None, None, None),) * dim
            + (slice(slice_idxs[i], slice_idxs[i + 1]),)
            for i in range(len(size_per_item))
        ]

        return MultiModalFieldConfig.flat(
            modality,
            slices,
            dim=dim,
            keep_on_cpu=keep_on_cpu,
        )

    @staticmethod
    def shared(
        modality: str,
        batch_size: int,
        *,
        keep_on_cpu: bool = False,
    ):
        """
        Defines a field where an element in the batch is obtained by
        taking the entirety of the underlying data.

        This means that the data is the same for each element in the batch.

        Args:
            modality: The modality of the multi-modal item that uses this
                keyword argument.
            batch_size: The number of multi-modal items which share this data.
            keep_on_cpu: Whether to keep this field on the CPU for the model inputs.

        Example:

        ```
        Given:
            batch_size: 4

        Input:
            Data: [XYZ]

        Output:
            Element 1: [XYZ]
            Element 2: [XYZ]
            Element 3: [XYZ]
            Element 4: [XYZ]
        ```
        """
        return MultiModalFieldConfig(
            field=MultiModalSharedField(
                batch_size=batch_size,
                keep_on_cpu=keep_on_cpu,
            ),
            modality=modality,
        )

    field: BaseMultiModalField
    modality: str

    def build_elems(
        self,
        key: str,
        batch: NestedTensors,
    ) -> Sequence[MultiModalFieldElem]:
        return self.field.build_elems(self.modality, key, batch)


class MultiModalKwargsItem(UserDict[str, MultiModalFieldElem]):
    """
    A dictionary of processed keyword arguments to pass to the model,
    corresponding to a single item in
    [`MultiModalDataItems`][vllm.multimodal.parse.MultiModalDataItems].
    """

    @staticmethod
    def dummy(nbytes: int = 1):
        """Convenience class for testing."""
        mm_elem = MultiModalFieldElem(
            data=torch.empty(nbytes, dtype=torch.uint8),
            field=MultiModalSharedField(batch_size=1),
        )
        return MultiModalKwargsItem({"dummy": mm_elem})

    def get_data(self) -> dict[str, NestedTensors]:
        return {key: elem.data for key, elem in self.items()}


_I = TypeVar(
    "_I",
    MultiModalKwargsItem,
    MultiModalKwargsItem | None,
    default=MultiModalKwargsItem,
)


class MultiModalKwargsItems(UserDict[str, Sequence[_I]]):
    """
    A dictionary of processed multi-modal inputs by modality.

    For example, given a processor that processes
    images into `pixel_values` and `image_grid_thw`,
    and audios into `input_audio_features`,
    a prompt with 2 images and 1 audio will be processed
    into a `MultiModalKwargsItems` with the following structure:

    ```python
    MultiModalKwargsItems(
        {
            "image": [
                # For the first image
                MultiModalKwargsItem({"pixel_values": ..., "image_grid_thw": ...}),
                # For the second imgae
                MultiModalKwargsItem({"pixel_values": ..., "image_grid_thw": ...}),
            ],
            "audio": [
                # For the first audio
                MultiModalKwargsItem({"input_audio_features": ...}),
            ],
        }
    )
    ```

    Unlike HF processing which returns all items
    in a single dictionary with batched keyword arguments,
    we split up the items because some of them may already be cached.
    Also, items from multiple requests may be batched together to improve throughput,
    using the logic defined by the
    [`BaseMultiModalField`][vllm.multimodal.inputs.BaseMultiModalField]
    for each keyword argument.
    """

    @staticmethod
    def from_hf_inputs(
        hf_inputs: "BatchFeature",
        config_by_key: Mapping[str, MultiModalFieldConfig],
    ):
        # NOTE: This skips fields in `hf_inputs` that are not in `config_by_key`
        # We assume that those fields are not used in vLLM
        elems_by_key = dict[str, Sequence[MultiModalFieldElem]]()
        keys_by_modality = defaultdict[str, set[str]](set)
        for key, config in config_by_key.items():
            batch = hf_inputs.get(key)
            if batch is not None:
                elems = config.build_elems(key, batch)
                if len(elems) > 0:
                    elems_by_key[key] = elems
                    keys_by_modality[config.modality].add(key)

        items_by_modality = dict[str, list[MultiModalKwargsItem]]()
        for modality, keys in keys_by_modality.items():
            elems_in_modality = {k: elems_by_key[k] for k in keys}
            batch_sizes = {k: len(v) for k, v in elems_in_modality.items()}

            if len(set(batch_sizes.values())) > 1:
                raise ValueError(
                    f"Cannot merge different batch sizes for {modality=}! "
                    f"Found: {batch_sizes=}"
                )

            batch_size = next(iter(batch_sizes.values()))
            items_by_modality[modality] = [
                MultiModalKwargsItem({k: v[i] for k, v in elems_in_modality.items()})
                for i in range(batch_size)
            ]

        return MultiModalKwargsItems(items_by_modality)

    def __getitem__(self, modality: str) -> Sequence[_I]:
        if modality not in self:
            raise KeyError(
                f"Modality {modality!r} not found. "
                f"Available modalities: {set(self.keys())}"
            )

        return super().__getitem__(modality)  # type: ignore[return-value]

    def require_data(self) -> "MultiModalKwargsItems[MultiModalKwargsItem]":
        for modality, items in self.items():
            for i, item in enumerate(items):
                if item is None:
                    raise RuntimeError(f"Found empty mm_items[{modality}][{i}]")

        return self  # type: ignore[return-value]

    def get_data(
        self,
        *,
        device: torch.types.Device = None,
        pin_memory: bool = False,
    ) -> BatchedTensorInputs:
        """Construct a dictionary of keyword arguments to pass to the model."""
        elems_by_key = defaultdict[str, list[MultiModalFieldElem]](list)
        for modality, items in self.items():
            for i, item in enumerate(items):
                if item is None:
                    raise RuntimeError(
                        f"Cannot build data from empty mm_items[{modality}][{i}]"
                    )

                for key, elem in item.items():
                    elems_by_key[key].append(elem)

        data = {
            key: elems[0].field.reduce_data(
                elems,
                device=device,
                pin_memory=pin_memory,
            )
            for key, elems in elems_by_key.items()
        }

        return data


MultiModalKwargsOptionalItems: TypeAlias = (
    MultiModalKwargsItems[MultiModalKwargsItem]
    | MultiModalKwargsItems[MultiModalKwargsItem | None]
)


MultiModalHashes = dict[str, list[str]]
"""
A dictionary containing per-item hashes for each modality.
"""


MultiModalPlaceholderDict: TypeAlias = Mapping[str, Sequence[PlaceholderRange]]
"""
A dictionary containing per-item placeholder ranges for each modality.
"""


class MultiModalInputs(TypedDict):
    """
    Represents the outputs of
    [`BaseMultiModalProcessor`][vllm.multimodal.processing.BaseMultiModalProcessor],
    ready to be passed to vLLM internals.
    """

    type: Literal["multimodal"]
    """The type of inputs."""

    prompt_token_ids: list[int]
    """The processed token IDs which includes placeholder tokens."""

    mm_kwargs: MultiModalKwargsOptionalItems
    """Keyword arguments to be directly passed to the model after batching."""

    mm_hashes: MultiModalHashes
    """The hashes of the multi-modal data."""

    mm_placeholders: MultiModalPlaceholderDict
    """
    For each modality, information about the placeholder tokens in
    `prompt_token_ids`.
    """

    cache_salt: NotRequired[str]
    """
    Optional cache salt to be used for prefix caching.
    """


class MultiModalEncDecInputs(MultiModalInputs):
    """
    Represents the outputs of
    [`EncDecMultiModalProcessor`][vllm.multimodal.processing.EncDecMultiModalProcessor]
    ready to be passed to vLLM internals.
    """

    encoder_prompt_token_ids: list[int]
    """The processed token IDs of the encoder prompt."""
