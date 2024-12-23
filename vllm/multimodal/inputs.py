from abc import ABC, abstractmethod
from collections import UserDict, defaultdict
from collections.abc import Mapping, Sequence
from typing import (Any, Literal, NamedTuple, Optional, TypedDict, TypeVar,
                    Union, cast, final)

import numpy as np
import torch
import torch.types
from PIL.Image import Image
from typing_extensions import NotRequired, TypeAlias

from vllm.utils import JSONTree, full_groupby, is_list_of, json_map_leaves

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

BatchedTensorInputs: TypeAlias = dict[str, NestedTensors]
"""
A dictionary containing nested tensors which have been batched via
:meth:`MultiModalKwargs.batch`.
"""


class MultiModalFieldTag(ABC):
    """Metadata for a field in :class:`MultiModalKwargs`."""

    def __init__(self, modality: str) -> None:
        super().__init__()

        self.modality = modality

    @abstractmethod
    def get(
        self,
        ref: "MultiModalKwargs",
        key: str,
        item_index: int,
    ) -> NestedTensors:
        raise NotImplementedError

    @classmethod
    def reduce(cls, batch: list[NestedTensors]) -> NestedTensors:
        """Merge elements returned by multiple calls of :meth:`get`."""
        raise NotImplementedError

    def __repr__(self) -> str:
        return (f"{type(self).__name__}(modality={self.modality!r})")


class MultiModalIndexedFieldTag(MultiModalFieldTag):
    """
    A :class:`MultiModalFieldTag` implementation where indexing into
    the batch dimension directly indexes into the first dimension.
    """

    def get(
        self,
        ref: "MultiModalKwargs",
        key: str,
        item_index: int,
    ) -> NestedTensors:
        return ref[key][item_index]

    @classmethod
    def reduce(cls, batch: list[NestedTensors]) -> NestedTensors:
        if len(batch) > 0 and is_list_of(batch, torch.Tensor, check="all"):
            first_shape = batch[0].shape
            if all(item.shape == first_shape for item in batch):
                return torch.stack(batch)

        return batch


class MultiModalFlatFieldTag(MultiModalFieldTag):
    """
    A :class:`MultiModalFieldTag` implementation where indexing into
    the batch dimension corresponds to a slice along the first dimension.
    """

    def __init__(self, modality: str, slices: Sequence[slice]) -> None:
        super().__init__(modality)

        self.slices = slices

    def __repr__(self) -> str:
        return (f"{type(self).__name__}(modality={self.modality!r}, "
                f"slices={self.slices})")

    def get(
        self,
        ref: "MultiModalKwargs",
        key: str,
        item_index: int,
    ) -> NestedTensors:
        return ref[key][self.slices[item_index]]

    @classmethod
    def reduce(cls, batch: list[NestedTensors]) -> NestedTensors:
        if len(batch) > 0 and is_list_of(batch, torch.Tensor, check="all"):
            first_shape = batch[0].shape
            if all(item.shape == first_shape for item in batch):
                return torch.concat(batch)

        return [elem for item in batch for elem in item]


class MultiModalFieldTags:
    """
    Convenience class containing factory methods for
    :class:`MultiModalFieldTag`.
    """
    indexed = MultiModalIndexedFieldTag
    flat = MultiModalFlatFieldTag


class MultiModalField(NamedTuple):
    tag: MultiModalFieldTag
    data: NestedTensors


class MultiModalKwargs(UserDict[str, NestedTensors]):
    """
    A dictionary that represents the keyword arguments to
    :meth:`~torch.nn.Module.forward`.

    Passing :code:`fields` enables the use of :meth:`slice` to
    obtain individual items by modality.
    """

    def __init__(
        self,
        data: Mapping[str, NestedTensors],
        *,
        tags: Optional[Mapping[str, MultiModalFieldTag]] = None,
    ) -> None:
        if tags is None:
            tags = {}

        super().__init__(data)

        self._tags = tags
        self._tags_by_modality = dict(
            full_groupby(tags.items(), key=lambda x: x[1].modality))

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

    def get_fields_by_modality(
        self,
        modality: str,
        item_index: int,
    ) -> Mapping[str, MultiModalField]:
        """
        Get the keyword arguments corresponding to an item identified by
        its modality and index.
        """
        tags_to_gather = self._tags_by_modality[modality]

        return {
            key: MultiModalField(tag, tag.get(self, key, item_index))
            for key, tag in tags_to_gather if key in self
        }

    @staticmethod
    def from_fields_by_modality(
        fields_by_modality: Mapping[str, list[Mapping[str, MultiModalField]]],
        *,
        enable_sanity_checks: bool = True,
    ) -> "MultiModalKwargs":
        """
        Construct a new :class:`MultiModalKwargs` from multiple items returned
        by :meth:`get_fields_by_modality`.
        """
        tag_per_key = defaultdict[str, MultiModalFieldTag]()
        data_per_key = defaultdict[str, list[NestedTensors]](list)
        for fields in fields_by_modality.values():
            for field in fields:
                for k, v in field.items():
                    tag_per_key[k] = v.tag
                    data_per_key[k].append(v.data)

        if enable_sanity_checks:
            batch_sizes = {k: len(v) for k, v in data_per_key.items()}
            batch_size = next(iter(batch_sizes.values()), None)
            if batch_size:
                assert all(bs == batch_size
                           for bs in batch_sizes.values()), batch_sizes

        data = {k: tag_per_key[k].reduce(vs) for k, vs in data_per_key.items()}

        return MultiModalKwargs(data, tags=tag_per_key)


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
