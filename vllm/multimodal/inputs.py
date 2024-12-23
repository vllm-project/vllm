from abc import ABC, abstractmethod
from collections import UserDict, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import partial
from typing import (Any, Literal, Optional, TypedDict, TypeVar, Union, cast,
                    final)

import numpy as np
import torch
import torch.types
from PIL.Image import Image
from transformers import BatchFeature
from typing_extensions import NotRequired, TypeAlias, assert_never

from vllm.utils import JSONTree, is_list_of, json_map_leaves

_T = TypeVar("_T")

# yapf: disable
ImageItem: TypeAlias = Union[Image, np.ndarray, torch.Tensor]
"""
A :class:`transformers.image_utils.ImageInput` representing a single image
item, which can be passed to a HuggingFace :code:`ImageProcessor`.
"""

VideoItem: TypeAlias = Union[
    list[Image],
    np.ndarray,
    torch.Tensor,
    list[np.ndarray],
    list[torch.Tensor],
]
"""
A :class:`transformers.image_utils.VideoInput` representing a single video
item, which can be passed to a HuggingFace :code:`VideoProcessor`.
"""

AudioItem: TypeAlias = Union[
    np.ndarray,
    list[float],
    # `(audio, sampling_rate)`: If the audio's sampling rate is different
    # from that expected by the model, we need to resample it.
    tuple[np.ndarray, float],
]
"""
Represents a single audio
item, which can be passed to a HuggingFace :code:`AudioProcessor`.
"""
# yapf: enable

MultiModalData: TypeAlias = Union[_T, list[_T]]
"""
Either a single data item, or a list of data items.

The number of data items allowed per modality is restricted by
:code:`--limit-mm-per-prompt`.
"""


@final
class MultiModalDataBuiltins(TypedDict, total=False):
    """Type annotations for modality types predefined by vLLM."""

    image: MultiModalData[ImageItem]
    """The input image(s)."""

    video: MultiModalData[VideoItem]
    """The input video(s)."""

    audio: MultiModalData[AudioItem]
    """The input audio(s)."""


MultiModalDataDict: TypeAlias = Mapping[str, MultiModalData[Any]]
"""
A dictionary containing an entry for each modality type to input.

Note:
    This dictionary also accepts modality keys defined outside
    :class:`MultiModalDataBuiltins` as long as a customized plugin
    is registered through the :class:`~vllm.multimodal.MULTIMODAL_REGISTRY`.
    Read more on that :ref:`here <adding_multimodal_plugin>`.
"""


class PlaceholderRange(TypedDict):
    """
    Placeholder location information for multi-modal data.

    For example:
        Prompt: AAAA BBBB What is in these images?
        Images A and B will have:
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
        return isinstance(b, torch.Tensor) and bool((a == b).all().item())
    elif isinstance(b, torch.Tensor):
        return isinstance(a, torch.Tensor) and bool((b == a).all().item())

    if isinstance(a, list):
        return (isinstance(b, list)
                and all(nested_tensors_equal(a_, b_) for a_, b_ in zip(a, b)))
    if isinstance(b, list):
        return (isinstance(a, list)
                and all(nested_tensors_equal(b_, a_) for b_, a_ in zip(b, a)))

    # Both a and b are scalars
    return a == b


BatchedTensorInputs: TypeAlias = dict[str, NestedTensors]
"""
A dictionary containing nested tensors which have been batched via
:meth:`MultiModalKwargs.batch`.
"""


@dataclass(frozen=True)
class MultiModalFieldItem:
    """Represents an item in :class:`MultiModalKwargs`."""
    field: "MultiModalField"
    modality: str
    data: NestedTensors

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, self.__class__):
            return False

        a_tupl = (type(self.field), self.modality)
        b_tupl = (type(other.field), other.modality)
        return a_tupl == b_tupl and nested_tensors_equal(self.data, other.data)


class MultiModalField(ABC):
    """Represents a field in :class:`MultiModalKwargs`."""

    @abstractmethod
    def _reduce_data(self, batch: list[NestedTensors]) -> NestedTensors:
        raise NotImplementedError

    def _build_item(
        self,
        modality: str,
        data: NestedTensors,
    ) -> MultiModalFieldItem:
        return MultiModalFieldItem(self, modality, data)

    def reduce(self, batch: list[MultiModalFieldItem]) -> MultiModalFieldItem:
        """Merge multiple instances of :class:`MultiModalFieldItem` together."""
        field_types = [type(item.field) for item in batch]
        if len(set(field_types)) > 1:
            raise ValueError(
                f"Cannot merge different fields together: {field_types}")

        modalities = [item.modality for item in batch]
        if len(set(modalities)) > 1:
            raise ValueError(
                f"Cannot merge different modalities together: {modalities}")

        data = self._reduce_data([item.data for item in batch])

        return self._build_item(modalities[0], data)


class MultiModalBatchedField(MultiModalField):
    """
    A :class:`MultiModalField` implementation where an item is obtained by
    directly indexing into the first dimension of the underlying data.
    """

    def build_items(
        self,
        batch: NestedTensors,
        modality: str,
    ) -> list[MultiModalFieldItem]:
        return [self._build_item(modality, item) for item in batch]

    def _reduce_data(self, batch: list[NestedTensors]) -> NestedTensors:
        if len(batch) > 0 and is_list_of(batch, torch.Tensor, check="all"):
            first_shape = batch[0].shape
            if all(item.shape == first_shape for item in batch):
                return torch.stack(batch)

        return batch


class MultiModalFlatField(MultiModalField):
    """
    A :class:`MultiModalField` implementation where an item is obtained by
    slicing along the first dimension of the underlying data.
    """

    def build_items(
        self,
        batch: NestedTensors,
        modality: str,
        slices: Sequence[slice],
    ) -> list[MultiModalFieldItem]:
        return [self._build_item(modality, batch[slice_]) for slice_ in slices]

    def _reduce_data(self, batch: list[NestedTensors]) -> NestedTensors:
        if len(batch) > 0 and is_list_of(batch, torch.Tensor, check="all"):
            first_shape = batch[0].shape
            if all(item.shape[1:] == first_shape[1:] for item in batch):
                return torch.concat(batch)

        return [elem for item in batch for elem in item]


class MultiModalFieldConfig:

    @staticmethod
    def batched(modality: str):
        return MultiModalFieldConfig("batched", modality=modality)

    @staticmethod
    def flat(modality: str, slices: Sequence[slice]):
        return MultiModalFieldConfig("flat", modality=modality, slices=slices)

    def __init__(
        self,
        field_type: Literal["batched", "flat"],
        **field_config: Any,
    ) -> None:
        super().__init__()

        field: Any
        if field_type == "batched":
            field = MultiModalBatchedField()
        elif field_type == "flat":
            field = MultiModalFlatField()
        else:
            assert_never(field_type)

        self._factory = partial(field.build_items, **field_config)

    def build_items(self, items: NestedTensors) -> list[MultiModalFieldItem]:
        return self._factory(items)


class MultiModalKwargs(UserDict[str, NestedTensors]):
    """
    A dictionary that represents the keyword arguments to
    :meth:`~torch.nn.Module.forward`.

    Passing :code:`fields` enables the use of :meth:`slice` to
    obtain individual items by modality.
    """

    @staticmethod
    def from_hf_inputs(
        hf_inputs: BatchFeature,
        config_by_key: Mapping[str, MultiModalFieldConfig],
        *,
        enable_sanity_checks: bool = False,
    ):
        # NOTE: This skips fields in `hf_inputs` that are not in `config_by_key`
        # We assume that those fields are not used in vLLM
        items_by_key = {
            key: config.build_items(hf_inputs[key])
            for key, config in config_by_key.items() if key in hf_inputs
        }

        data = {
            k: items[0].field.reduce(items).data
            for k, items in items_by_key.items()
        }

        return MultiModalKwargs(data,
                                items_by_key=items_by_key,
                                enable_sanity_checks=enable_sanity_checks)

    def __init__(
        self,
        data: Mapping[str, NestedTensors],
        *,
        items_by_key: Optional[Mapping[str, list[MultiModalFieldItem]]] = None,
        enable_sanity_checks: bool = False,
    ) -> None:
        if items_by_key is None:
            items_by_key = {}

        super().__init__(data)

        # Shallow copy to avoid footgun in case a defaultdict is passed in
        self._items_by_key = dict(items_by_key)

        keys_by_modality = defaultdict[str, set[str]](set)
        for key, items in items_by_key.items():
            for item in items:
                keys_by_modality[item.modality].add(key)

        self._keys_by_modality = dict(keys_by_modality)

        if enable_sanity_checks:
            for modality, keys in keys_by_modality.items():
                items_in_modality = {k: items_by_key[k] for k in keys}
                batch_sizes = {k: len(v) for k, v in items_in_modality.items()}
                batch_size = next(iter(batch_sizes.values()), 0)
                assert all(bs == batch_size
                           for bs in batch_sizes.values()), dict(
                               modality=modality,
                               batch_sizes=batch_sizes,
                               items_by_key=items_by_key)

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
        if self._items_by_key != other._items_by_key:
            return False

        ks = self.keys()
        return (ks == other.keys()
                and all(nested_tensors_equal(self[k], other[k]) for k in ks))

    def get_item(self, key: str, item_index: int) -> MultiModalFieldItem:
        return self._items_by_key[key][item_index]

    def get_items_by_modality(
        self,
        modality: str,
        item_index: int,
    ) -> Mapping[str, MultiModalFieldItem]:
        """
        Get the keyword arguments corresponding to an item identified by
        its modality and index.
        """
        keys_to_gather = self._keys_by_modality[modality]

        return {
            key: self.get_item(key, item_index)
            for key in keys_to_gather if key in self
        }

    @staticmethod
    def from_items_by_modality(
        items_by_modality: Mapping[str, list[Mapping[str,
                                                     MultiModalFieldItem]]],
        *,
        enable_sanity_checks: bool = False,
    ) -> "MultiModalKwargs":
        """
        Construct a new :class:`MultiModalKwargs` from multiple items returned
        by :meth:`get_fields_by_modality`.
        """
        items_by_key = defaultdict[str, list[MultiModalFieldItem]](list)
        for fields in items_by_modality.values():
            for field in fields:
                for k, v in field.items():
                    items_by_key[k].append(v)

        data = {
            k: items[0].field.reduce(items).data
            for k, items in items_by_key.items()
        }

        return MultiModalKwargs(data,
                                items_by_key=items_by_key,
                                enable_sanity_checks=enable_sanity_checks)


MultiModalPlaceholderDict = Mapping[str, Sequence[PlaceholderRange]]
"""
A dictionary containing placeholder ranges.
"""


class MultiModalInputsV2(TypedDict):
    """
    Represents the outputs of :class:`vllm.multimodal.MultiModalProcessor`,
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

    mm_hashes: NotRequired[list[str]]
    """The hashes of the multi-modal data."""

    mm_placeholders: MultiModalPlaceholderDict
    """
    For each modality, information about the placeholder tokens in
    :code:`prompt_token_ids`.
    """
