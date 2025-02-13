# SPDX-License-Identifier: Apache-2.0

from abc import ABC, abstractmethod
from collections import UserDict, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import partial
from itertools import accumulate
from typing import (TYPE_CHECKING, Any, Literal, Optional, TypedDict, TypeVar,
                    Union, cast, final)

import numpy as np
import torch
import torch.types
from PIL.Image import Image
from transformers import BatchFeature
from typing_extensions import NotRequired, TypeAlias

from vllm.utils import JSONTree, full_groupby, is_list_of, json_map_leaves

if TYPE_CHECKING:
    from .hasher import MultiModalHashDict

_T = TypeVar("_T")

HfImageItem: TypeAlias = Union[Image, np.ndarray, torch.Tensor]
"""
A :class:`transformers.image_utils.ImageInput` representing a single image
item, which can be passed to a HuggingFace :code:`ImageProcessor`.
"""

HfVideoItem: TypeAlias = Union[list[Image], np.ndarray, torch.Tensor,
                               list[np.ndarray], list[torch.Tensor]]
"""
A :class:`transformers.image_utils.VideoInput` representing a single video
item, which can be passed to a HuggingFace :code:`VideoProcessor`.
"""

HfAudioItem: TypeAlias = Union[list[float], np.ndarray, torch.Tensor]
"""
Represents a single audio
item, which can be passed to a HuggingFace :code:`AudioProcessor`.
"""

ImageItem: TypeAlias = Union[HfImageItem, torch.Tensor]
"""
A :class:`transformers.image_utils.ImageInput` representing a single image
item, which can be passed to a HuggingFace :code:`ImageProcessor`.

Alternatively, a 3-D tensor or batch of 2-D tensors,
which are treated as image embeddings;
these are directly passed to the model without HF processing.
"""

VideoItem: TypeAlias = Union[HfVideoItem, torch.Tensor]
"""
A :class:`transformers.image_utils.VideoInput` representing a single video
item, which can be passed to a HuggingFace :code:`VideoProcessor`.

Alternatively, a 3-D tensor or batch of 2-D tensors,
which are treated as video embeddings;
these are directly passed to the model without HF processing.
"""

AudioItem: TypeAlias = Union[HfAudioItem, tuple[np.ndarray, float],
                             torch.Tensor]
"""
Represents a single audio
item, which can be passed to a HuggingFace :code:`AudioProcessor`.

Alternatively, a tuple `(audio, sampling_rate)`, where the sampling rate
is different from that expected by the model;
these are resampled to the model's sampling rate before being processed by HF.

Alternatively, a 3-D tensor or batch of 2-D tensors,
which are treated as audio embeddings;
these are directly passed to the model without HF processing.
"""

ModalityData: TypeAlias = Union[_T, list[_T]]
"""
Either a single data item, or a list of data items.

The number of data items allowed per modality is restricted by
:code:`--limit-mm-per-prompt`.
"""


@final
class MultiModalDataBuiltins(TypedDict, total=False):
    """Type annotations for modality types predefined by vLLM."""

    image: ModalityData[ImageItem]
    """The input image(s)."""

    video: ModalityData[VideoItem]
    """The input video(s)."""

    audio: ModalityData[AudioItem]
    """The input audio(s)."""


MultiModalDataDict: TypeAlias = Mapping[str, ModalityData[Any]]
"""
A dictionary containing an entry for each modality type to input.

The built-in modalities are defined by :class:`MultiModalDataBuiltins`.
"""


class PlaceholderRange(TypedDict):
    """
    Placeholder location information for multi-modal data.

    Example:

        Prompt: :code:`AAAA BBBB What is in these images?`

        Images A and B will have:

        .. code-block::

            A: { "offset": 0, "length": 4 }
            B: { "offset": 5, "length": 4 }
    """

    offset: int
    """The start index of the placeholder in the prompt."""

    length: int
    """The length of the placeholder."""


NestedTensors = Union[list["NestedTensors"], list[torch.Tensor], torch.Tensor,
                      tuple[torch.Tensor, ...]]
"""
Uses a list instead of a tensor if the dimensions of each element do not match.
"""


def nested_tensors_equal(a: NestedTensors, b: NestedTensors) -> bool:
    """Equality check between :data:`NestedTensors` objects."""
    if isinstance(a, torch.Tensor):
        return isinstance(b, torch.Tensor) and torch.equal(a, b)
    elif isinstance(b, torch.Tensor):
        return isinstance(a, torch.Tensor) and torch.equal(b, a)

    if isinstance(a, list):
        return (isinstance(b, list)
                and all(nested_tensors_equal(a_, b_) for a_, b_ in zip(a, b)))
    if isinstance(b, list):
        return (isinstance(a, list)
                and all(nested_tensors_equal(b_, a_) for b_, a_ in zip(b, a)))

    # Both a and b are scalars
    return a == b


BatchedTensorInputs: TypeAlias = Mapping[str, NestedTensors]
"""
A dictionary containing nested tensors which have been batched via
:meth:`MultiModalKwargs.batch`.
"""


@dataclass(frozen=True)
class MultiModalFieldElem:
    """
    Represents a keyword argument corresponding to a multi-modal item
    in :class:`MultiModalKwargs`.
    """

    modality: str
    """
    The modality of the corresponding multi-modal item.
    Each multi-modal item can consist of multiple keyword arguments.
    """

    key: str
    """
    The key of this field in :class:`MultiModalKwargs`,
    i.e. the name of the keyword argument to be passed to the model.
    """

    data: NestedTensors
    """
    The tensor data of this field in :class:`MultiModalKwargs`,
    i.e. the value of the keyword argument to be passed to the model.
    """

    field: "BaseMultiModalField"
    """
    Defines how to combine the tensor data of this field with others
    in order to batch multi-modal items together for model inference.
    """

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, self.__class__):
            return False

        return ((self.modality, self.key) == (other.modality, other.key)
                and nested_tensors_equal(self.data, other.data)
                and type(self.field) == type(other.field))  # noqa: E721


@dataclass(frozen=True)
class BaseMultiModalField(ABC):
    """
    Defines how to interpret tensor data belonging to a keyword argument in
    :class:`MultiModalKwargs` for multiple multi-modal items, and vice versa.
    """

    def _field_factory(self, *, modality: str, key: str):
        f = partial(
            MultiModalFieldElem,
            modality=modality,
            key=key,
            field=self,
        )

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
        Construct :class:`MultiModalFieldElem` instances to represent
        the provided data.
        
        This is the inverse of :meth:`reduce_data`.
        """
        raise NotImplementedError

    @abstractmethod
    def _reduce_data(self, batch: list[NestedTensors]) -> NestedTensors:
        raise NotImplementedError

    def reduce_data(self, elems: list[MultiModalFieldElem]) -> NestedTensors:
        """
        Merge the data from multiple instances of :class:`MultiModalFieldElem`.

        This is the inverse of :meth:`build_elems`.
        """
        field_types = [type(item.field) for item in elems]
        if len(set(field_types)) > 1:
            raise ValueError(f"Cannot merge different {field_types=}")

        return self._reduce_data([item.data for item in elems])


@dataclass(frozen=True)
class MultiModalBatchedField(BaseMultiModalField):
    """
    See also:
        :func:`MultiModalFieldConfig.batched`
    """

    def build_elems(
        self,
        modality: str,
        key: str,
        data: NestedTensors,
    ) -> Sequence[MultiModalFieldElem]:
        field_factory = self._field_factory(modality=modality, key=key)
        return [field_factory(item) for item in data]

    def _reduce_data(self, batch: list[NestedTensors]) -> NestedTensors:
        if len(batch) > 0 and is_list_of(batch, torch.Tensor, check="all"):
            if len(batch) == 1:
                # An optimization when `batch` contains only one tensor:
                # - produce exactly same result as `torch.stack(batch)`
                # - will achieve zero-copy if the tensor is contiguous
                return batch[0].unsqueeze(0).contiguous()
            first_shape = batch[0].shape
            if all(elem.shape == first_shape for elem in batch):
                return torch.stack(batch)

        return batch


@dataclass(frozen=True)
class MultiModalFlatField(BaseMultiModalField):
    """
    See also:
        :func:`MultiModalFieldConfig.flat`
        :func:`MultiModalFieldConfig.flat_from_sizes`
    """
    slices: Sequence[slice]

    def build_elems(
        self,
        modality: str,
        key: str,
        data: NestedTensors,
    ) -> Sequence[MultiModalFieldElem]:
        field_factory = self._field_factory(modality=modality, key=key)
        return [field_factory(data[s]) for s in self.slices]

    def _reduce_data(self, batch: list[NestedTensors]) -> NestedTensors:
        if len(batch) > 0 and is_list_of(batch, torch.Tensor, check="all"):
            if len(batch) == 1:
                # An optimization when `batch` contains only one tensor:
                # - produce exactly same result as `torch.concat(batch)`
                # - will achieve zero-copy if the tensor is contiguous
                return batch[0].contiguous()
            first_shape = batch[0].shape
            if all(elem.shape[1:] == first_shape[1:] for elem in batch):
                return torch.concat(batch)

        return [e for elem in batch for e in elem]


@dataclass(frozen=True)
class MultiModalSharedField(BaseMultiModalField):
    """
    See also:
        :func:`MultiModalFieldConfig.shared`
    """
    batch_size: int

    def build_elems(
        self,
        modality: str,
        key: str,
        data: NestedTensors,
    ) -> Sequence[MultiModalFieldElem]:
        field_factory = self._field_factory(modality=modality, key=key)
        return [field_factory(data)] * self.batch_size

    def _reduce_data(self, batch: list[NestedTensors]) -> NestedTensors:
        return batch[0]


class MultiModalFieldConfig:

    @staticmethod
    def batched(modality: str):
        """
        Defines a field where an element in the batch is obtained by
        indexing into the first dimension of the underlying data.

        Args:
            modality: The modality of the multi-modal item that uses this
                keyword argument.

        Example:

            .. code-block::

                Input:
                    Data: [[AAAA]
                        [BBBB]
                        [CCCC]]

                Output:
                    Element 1: [AAAA]
                    Element 2: [BBBB]
                    Element 3: [CCCC]
        """
        return MultiModalFieldConfig(
            field=MultiModalBatchedField(),
            modality=modality,
        )

    @staticmethod
    def flat(modality: str, slices: Sequence[slice]):
        """
        Defines a field where an element in the batch is obtained by
        slicing along the first dimension of the underlying data.

        Args:
            modality: The modality of the multi-modal item that uses this
                keyword argument.
            slices: For each multi-modal item, a slice that is used to extract
                the data corresponding to it.

        Example:

            .. code-block::
        
                Given:
                    slices: [slice(0, 3), slice(3, 7), slice(7, 9)]

                Input:
                    Data: [AAABBBBCC]

                Output:
                    Element 1: [AAA]
                    Element 2: [BBBB]
                    Element 3: [CC]
        """
        return MultiModalFieldConfig(
            field=MultiModalFlatField(slices=slices),
            modality=modality,
        )

    @staticmethod
    def flat_from_sizes(modality: str, size_per_item: torch.Tensor):
        """
        Defines a field where an element in the batch is obtained by
        slicing along the first dimension of the underlying data.

        Args:
            modality: The modality of the multi-modal item that uses this
                keyword argument.
            slices: For each multi-modal item, the size of the slice that
                is used to extract the data corresponding to it.

        Example:

            .. code-block::
        
                Given:
                    size_per_item: [3, 4, 2]

                Input:
                    Data: [AAABBBBCC]

                Output:
                    Element 1: [AAA]
                    Element 2: [BBBB]
                    Element 3: [CC]
    
        See also:
            :func:`MultiModalFieldConfig.flat`
        """

        slice_idxs = [0, *accumulate(size_per_item)]
        slices = [
            slice(slice_idxs[i], slice_idxs[i + 1])
            for i in range(len(size_per_item))
        ]

        return MultiModalFieldConfig.flat(modality, slices)

    @staticmethod
    def shared(modality: str, batch_size: int):
        """
        Defines a field where an element in the batch is obtained by
        taking the entirety of the underlying data.

        This means that the data is the same for each element in the batch.

        Args:
            modality: The modality of the multi-modal item that uses this
                keyword argument.
            batch_size: The number of multi-modal items which share this data.

        Example:

            .. code-block::
        
                Given:
                    batch_size: 4

                Input:
                    Data: [XYZ]

                Output:
                    Element 1: [XYZ]
                    Element 2: [XYZ]
                    Element 3: [XYZ]
                    Element 4: [XYZ]
        """
        return MultiModalFieldConfig(
            field=MultiModalSharedField(batch_size),
            modality=modality,
        )

    def __init__(self, field: BaseMultiModalField, modality: str) -> None:
        super().__init__()

        self.field = field
        self.modality = modality

    def build_elems(
        self,
        key: str,
        batch: NestedTensors,
    ) -> Sequence[MultiModalFieldElem]:
        return self.field.build_elems(self.modality, key, batch)


class MultiModalKwargsItem(UserDict[str, MultiModalFieldElem]):
    """
    A collection of :class:`MultiModalFieldElem`
    corresponding to a data item in :class:`MultiModalDataItems`.
    """

    @staticmethod
    def from_elems(elems: Sequence[MultiModalFieldElem]):
        return MultiModalKwargsItem({elem.key: elem for elem in elems})

    @property
    def modality(self) -> str:
        modalities = {elem.modality for elem in self.data.values()}
        assert len(modalities) == 1, f"Found different modalities={modalities}"
        return next(iter(modalities))


# NOTE: UserDict is for V0 compatibility.
# V1 should access individual items via `get_item`.
class MultiModalKwargs(UserDict[str, NestedTensors]):
    """
    A dictionary that represents the keyword arguments to
    :meth:`~torch.nn.Module.forward`.

    The metadata :code:`items` enables us to obtain the keyword arguments
    corresponding to each data item in :class:`MultiModalDataItems`, via
    :meth:`get_item` and :meth:`get_items`.
    """

    @staticmethod
    def from_hf_inputs(
        hf_inputs: BatchFeature,
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

        items = list[MultiModalKwargsItem]()
        for modality, keys in keys_by_modality.items():
            elems_in_modality = {k: elems_by_key[k] for k in keys}
            batch_sizes = {k: len(v) for k, v in elems_in_modality.items()}

            if len(set(batch_sizes.values())) > 1:
                raise ValueError(
                    f"Cannot merge different batch sizes for {modality=}! "
                    f"Found: {batch_sizes=}")

            batch_size = next(iter(batch_sizes.values()))
            for item_idx in range(batch_size):
                elems = [v[item_idx] for v in elems_in_modality.values()]
                items.append(MultiModalKwargsItem.from_elems(elems))

        return MultiModalKwargs.from_items(items)

    @staticmethod
    def from_items(items: Sequence[MultiModalKwargsItem]):
        """Construct a new :class:`MultiModalKwargs` from multiple items."""
        elems_by_key = defaultdict[str, list[MultiModalFieldElem]](list)
        for item in items:
            for key, elem in item.items():
                elems_by_key[key].append(elem)

        data = {
            key: elems[0].field.reduce_data(elems)
            for key, elems in elems_by_key.items() if len(elems) > 0
        }

        return MultiModalKwargs(data, items=items)

    def __init__(
        self,
        data: Mapping[str, NestedTensors],
        *,
        items: Optional[Sequence[MultiModalKwargsItem]] = None,
    ) -> None:
        super().__init__(data)

        items_by_modality = full_groupby(items or [], key=lambda x: x.modality)
        self._items_by_modality = dict(items_by_modality)

    @property
    def modalities(self):
        return self._items_by_modality.keys()

    @staticmethod
    def _try_stack(nested_tensors: NestedTensors) -> NestedTensors:
        """
        Stack the inner dimensions that have the same shape in
        a nested list of tensors.

        Thus, a dimension represented by a list means that the inner
        dimensions are different for each element along that dimension.
        """
        if isinstance(nested_tensors, torch.Tensor):
            return nested_tensors

        # TODO: Remove these once all models have been migrated
        if isinstance(nested_tensors, np.ndarray):
            return torch.from_numpy(nested_tensors)
        if isinstance(nested_tensors, (int, float)):
            return torch.tensor(nested_tensors)

        stacked = [MultiModalKwargs._try_stack(t) for t in nested_tensors]
        if not is_list_of(stacked, torch.Tensor, check="all"):
            # Only tensors (not lists) can be stacked.
            return stacked

        tensors_ = cast(list[torch.Tensor], stacked)
        if len(tensors_) == 1:
            # An optimization when `tensors_` contains only one tensor:
            # - produce exactly same result as `torch.stack(tensors_)`
            # - will achieve zero-copy if the tensor is contiguous
            return tensors_[0].unsqueeze(0).contiguous()

        if any(t.shape != tensors_[0].shape for t in tensors_):
            # The tensors have incompatible shapes and can't be stacked.
            return tensors_

        return torch.stack(tensors_)

    @staticmethod
    def batch(inputs_list: list["MultiModalKwargs"]) -> BatchedTensorInputs:
        """
        Batch multiple inputs together into a dictionary.

        The resulting dictionary has the same keys as the inputs.
        If the corresponding value from each input is a tensor and they all
        share the same shape, the output value is a single batched tensor;
        otherwise, the output value is a list containing the original value
        from each input.
        """
        if len(inputs_list) == 0:
            return {}

        # We need to consider the case where each item in the batch
        # contains different modalities (i.e. different keys).
        item_lists = defaultdict[str, list[NestedTensors]](list)

        for inputs in inputs_list:
            for k, v in inputs.items():
                item_lists[k].append(v)

        return {
            k: MultiModalKwargs._try_stack(item_list)
            for k, item_list in item_lists.items()
        }

    @staticmethod
    def as_kwargs(
        batched_inputs: BatchedTensorInputs,
        *,
        device: torch.types.Device,
    ) -> BatchedTensorInputs:
        json_inputs = cast(JSONTree[torch.Tensor], batched_inputs)

        json_mapped = json_map_leaves(
            lambda x: x.to(device, non_blocking=True),
            json_inputs,
        )

        return cast(BatchedTensorInputs, json_mapped)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, self.__class__):
            return False
        if self._items_by_modality != other._items_by_modality:
            return False

        ks = self.keys()
        return (ks == other.keys()
                and all(nested_tensors_equal(self[k], other[k]) for k in ks))

    def _validate_modality(self, method_name: str, modality: str) -> None:
        if not self._items_by_modality:
            raise RuntimeError(
                f"`{method_name}` is not supported when "
                "MultiModalKwargs is not initialized with `items`")

        if modality not in self._items_by_modality:
            available_modalities = set(self._items_by_modality.keys())
            raise KeyError(f"Modality {modality!r} not found. "
                           f"Available modalities: {available_modalities}")

    def get_item_count(self, modality: str) -> int:
        """Get the number of items belonging to a modality."""
        self._validate_modality("get_item_count", modality)
        return len(self._items_by_modality[modality])

    def get_item(self, modality: str, item_index: int) -> MultiModalKwargsItem:
        """
        Get the keyword arguments corresponding to an item identified by
        its modality and index.
        """
        self._validate_modality("get_item", modality)
        return self._items_by_modality[modality][item_index]

    def get_items(self, modality: str) -> Sequence[MultiModalKwargsItem]:
        """
        Get the keyword arguments corresponding to each item belonging to
        a modality.
        """
        self._validate_modality("get_items", modality)
        return self._items_by_modality[modality]


MultiModalPlaceholderDict = Mapping[str, Sequence[PlaceholderRange]]
"""
A dictionary containing placeholder ranges for each modality.
"""


class MultiModalInputs(TypedDict):
    """
    Represents the outputs of
    :class:`vllm.multimodal.processing.BaseMultiModalProcessor`,
    ready to be passed to vLLM internals.
    """

    type: Literal["multimodal"]
    """The type of inputs."""

    prompt: str
    """The processed prompt text."""

    prompt_token_ids: list[int]
    """The processed token IDs which includes placeholder tokens."""

    token_type_ids: NotRequired[list[int]]
    """The token type IDs of the prompt."""

    mm_kwargs: MultiModalKwargs
    """Keyword arguments to be directly passed to the model after batching."""

    mm_hashes: NotRequired[Optional["MultiModalHashDict"]]
    """The hashes of the multi-modal data."""

    mm_placeholders: MultiModalPlaceholderDict
    """
    For each modality, information about the placeholder tokens in
    :code:`prompt_token_ids`.
    """


class MultiModalEncDecInputs(MultiModalInputs):
    """
    Represents the outputs of :class:`vllm.multimodal.EncDecMultiModalProcessor`
    ready to be passed to vLLM internals.
    """

    encoder_prompt: str
    """The processed encoder prompt text."""

    encoder_prompt_token_ids: list[int]
    """The processed token IDs of the encoder prompt."""

    encoder_token_type_ids: NotRequired[list[int]]
    """The token type IDs of the encoder prompt."""
