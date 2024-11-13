from collections import UserDict, defaultdict
from typing import (Any, Dict, List, Literal, Mapping, Sequence, Tuple,
                    TypedDict, TypeVar, Union, cast, final)

import numpy as np
import torch
import torch.types
from PIL.Image import Image
from typing_extensions import TypeAlias

from vllm.utils import JSONTree, is_list_of, json_map_leaves

_T = TypeVar("_T")

# yapf: disable
ImageItem: TypeAlias = Union[Image, np.ndarray, torch.Tensor]
"""
A :class:`transformers.image_utils.ImageInput` representing a single image,
which can be passed to a HuggingFace :code:`ImageProcessor`.
"""

VideoItem: TypeAlias = Union[
    List[Image],
    np.ndarray,
    torch.Tensor,
    List[np.ndarray],
    List[torch.Tensor],
]
"""

A :class:`transformers.image_utils.VideoInput` representing a single video,
which can be passed to a HuggingFace :code:`VideoProcessor`.
"""

AudioItem: TypeAlias = Union[
    np.ndarray,
    List[float],
    Tuple[np.ndarray, float],  # DEPRECATED: Use mm_processor_kwargs instead
]
"""
Represents a single audio that can be inputted to a HuggingFace
:code:`AudioProcessor`.
"""
# yapf: enable

MultiModalData: TypeAlias = Union[_T, List[_T]]
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


NestedTensors = Union[List["NestedTensors"], List[torch.Tensor], torch.Tensor]
"""
Uses a list instead of a tensor if the dimensions of each element do not match.
"""

BatchedTensorInputs: TypeAlias = Dict[str, NestedTensors]
"""
A dictionary containing nested tensors which have been batched via
:meth:`MultiModalKwargs.batch`.
"""


class MultiModalKwargs(UserDict[str, NestedTensors]):
    """
    A dictionary that represents the keyword arguments to
    :meth:`~torch.nn.Module.forward`.
    """

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

        tensors_ = cast(List[torch.Tensor], stacked)
        if any(t.shape != tensors_[0].shape for t in tensors_):
            # The tensors have incompatible shapes and can't be stacked.
            return tensors_

        return torch.stack(tensors_)

    @staticmethod
    def batch(inputs_list: List["MultiModalKwargs"]) -> BatchedTensorInputs:
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
        item_lists: Dict[str, List[NestedTensors]] = defaultdict(list)

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
    """
    The original, unprocessed prompt text.

    Note:
        Since prompt text is not required by vLLM internals, we leave this
        unprocessed to save CPU computation. You can still call
        :code:`tokenizer.decode(prompt_token_ids)` to get the processed text.
    """

    prompt_token_ids: List[int]
    """The processed token IDs which includes placeholder tokens."""

    mm_kwargs: MultiModalKwargs
    """Keyword arguments to be directly passed to the model after batching."""

    mm_placeholders: MultiModalPlaceholderDict
    """
    For each modality, information about the placeholder tokens in
    :code:`prompt_token_ids`.
    """
