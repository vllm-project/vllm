from dataclasses import dataclass
from functools import lru_cache
from itertools import groupby
from typing import (Any, Callable, Collection, Generic, List, Mapping,
                    Optional, TypeVar, Union, final)

from transformers import BatchFeature
from typing_extensions import TypeAlias, TypedDict

from vllm.inputs import InputProcessingContext
from vllm.transformers_utils.tokenizer import AnyTokenizer, MistralTokenizer
from vllm.utils import is_list_of

from .inputs import (AudioItem, ImageItem, MultiModalDataDict,
                     MultiModalInputsV2, MultiModalKwargs, PlaceholderRange,
                     VideoItem)

_T = TypeVar("_T")


class PlaceholderReplacement(TypedDict, Generic[_T]):
    token_id: int
    """The ID of the placeholder token."""

    count: Union[Callable[[_T, BatchFeature, int], int], int]
    """
    Given the original data item, HF-processed data, and index of the processed
    item, output the number of replacement tokens to be allocated in vLLM.

    For convenience, you can pass in an integer if this number is a constant.
    """


@dataclass
class ModalityProcessingMetadata(Generic[_T]):
    placeholder_replacements: Mapping[str, PlaceholderReplacement[_T]]
    """
    A dictionary that maps each substring to search in the original prompt text
    to the corresponding replacement.
    """


class MultiModalProcessingMetadataBuiltins(TypedDict, total=False):
    """Type annotations for modality types predefined by vLLM."""

    image: ModalityProcessingMetadata[ImageItem]
    video: ModalityProcessingMetadata[VideoItem]
    audio: ModalityProcessingMetadata[AudioItem]


MultiModalProcessingMetadata: TypeAlias = \
    Mapping[str, ModalityProcessingMetadata[Any]]
"""
A dictionary containing an entry for each modality type to process.

Note:
    This dictionary also accepts modality keys defined outside
    :class:`MultiModalProcessingMetadataBuiltins` as long as a customized plugin
    is registered through the :class:`~vllm.multimodal.MULTIMODAL_REGISTRY`.
    Read more on that :ref:`here <adding_multimodal_plugin>`.
"""

MultiModalMultiData: TypeAlias = List[_T]
"""
A list of data items, where the number of data items allowed
per modality is restricted by :code:`--limit-mm-per-prompt`.
"""


@final
class MultiModalMultiDataBuiltins(TypedDict, total=False):
    """Type annotations for modality types predefined by vLLM."""

    image: MultiModalMultiData[ImageItem]
    """The input images."""

    video: MultiModalMultiData[VideoItem]
    """The input videos."""

    audio: MultiModalMultiData[AudioItem]
    """The input audios."""


MultiModalMultiDataDict: TypeAlias = Mapping[str, MultiModalMultiData[Any]]
"""
A dictionary containing an entry for each modality type to input.

Note:
    This dictionary also accepts modality keys defined outside
    :class:`MultiModalMultiDataBuiltins` as long as a customized plugin
    is registered through the :class:`~vllm.multimodal.MULTIMODAL_REGISTRY`.
    Read more on that :ref:`here <adding_multimodal_plugin>`.
"""


def to_multi_format(data: MultiModalDataDict) -> MultiModalMultiDataDict:
    """
    Convert a :class:`MultiModalDataDict` containing single data items
    to a :class:`MultiModalMultiDataDict` containing multiple data items
    per entry.
    """
    multi_data: Mapping[str, MultiModalMultiData[Any]] = {}

    for k, v in data.items():
        # yapf: disable
        if k == "video":
            # Special case since even a single item can be a list
            multi_data[k] = v if is_list_of(v, list) else [v]  # type: ignore[index]
        elif k in ("image", "audio"):
            multi_data[k] = v if isinstance(v, list) else [v]  # type: ignore[index]
        else:
            multi_data[k] = v if isinstance(v, list) else [v]  # type: ignore[index]
        # yapf: enable

    return multi_data


def iter_token_runs(token_ids: List[int]):
    """
    Yield the starting index and length of each run of tokens that are the same.
    """
    start_idx = 0

    for token_id, it in groupby(token_ids):
        length = sum(1 for _ in it)
        yield token_id, PlaceholderRange(offset=start_idx, length=length)

        start_idx += length


def encode_no_special_tokens(
    tokenizer: AnyTokenizer,
    text: str,
) -> List[int]:
    """
    Backend-agnostic equivalent of HF's
    :code:`tokenizer.encode(text, add_special_tokens=False)`.
    """
    if isinstance(tokenizer, MistralTokenizer):
        return tokenizer.tokenizer.encode(text, bos=False, eos=False)

    return tokenizer.encode(text, add_special_tokens=False)


@lru_cache
def candidate_placeholders(
    tokenizer: AnyTokenizer,
    placeholder_text: str,
) -> Collection[List[int]]:
    """Generate token ID sequences that may represent a placeholder text."""
    # When the placeholder text is not mapped to a special token ID,
    # it may be tokenized differently based on whether it is at the start/end
    # of the string. So, we go through each combination of whether the text
    # is at the start and end boundaries of the string

    # Matches the placeholder when it is in the middle of the string
    start_id, = encode_no_special_tokens(tokenizer, "a")
    end_id, = encode_no_special_tokens(tokenizer, "b")

    candidate_basic = encode_no_special_tokens(tokenizer, placeholder_text)

    start_id_, *candidate_a = encode_no_special_tokens(
        tokenizer,
        f"a{placeholder_text}",
    )
    assert start_id == start_id_

    start_id_, *candidate_ab, end_id_ = encode_no_special_tokens(
        tokenizer,
        f"a{placeholder_text}b",
    )
    assert start_id == start_id_ and end_id == end_id_

    *candidate_b, end_id_ = encode_no_special_tokens(
        tokenizer,
        f"{placeholder_text}b",
    )
    assert end_id == end_id_

    # Remove duplicates (need to convert to tuple to be hashable)
    unique_candidates = {
        tuple(c)
        for c in [candidate_basic, candidate_a, candidate_ab, candidate_b]
    }

    # Convert back to list
    return [list(c) for c in unique_candidates]


def apply_placeholders(
    token_ids: List[int],
    match_ids: List[int],
    replacement_id: int,
    replacement_count: int,
) -> Optional[PlaceholderRange]:
    """
    Find the first occurrence of :code:`placeholder_ids`,
    and replace it with the output of :code:`get_replacement_ids`.

    This function updates :code:`token_ids` in place.
    """
    if len(match_ids) == 0:
        raise ValueError("Match tokens should not be empty")
    if replacement_id in match_ids:
        raise ValueError(f"Match tokens ({match_ids}) should not include "
                         f"replacement token ({replacement_id})")

    placeholder_length = len(match_ids)

    for start_idx in range(len(token_ids) - placeholder_length + 1):
        end_idx = start_idx + placeholder_length

        if token_ids[start_idx:end_idx] == match_ids:
            replacement_ids = [replacement_id] * replacement_count
            token_ids[start_idx:end_idx] = replacement_ids

            return PlaceholderRange(offset=start_idx, length=replacement_count)

    return None


class MultiModalProcessor:
    """
    Helper class to process multi-modal inputs to be used in vLLM.
    """

    def __init__(
        self,
        ctx: InputProcessingContext,
        metadata: MultiModalProcessingMetadata,
    ) -> None:
        super().__init__()

        self.ctx = ctx
        self.metadata = metadata

    def __call__(
        self,
        prompt: str,
        mm_data: MultiModalDataDict,
        mm_processor_kwargs: Mapping[str, object],
    ) -> MultiModalInputsV2:
        return self.apply(prompt, mm_data, mm_processor_kwargs)

    def apply(
        self,
        prompt: str,
        mm_data: MultiModalDataDict,
        mm_processor_kwargs: Mapping[str, object],
    ) -> MultiModalInputsV2:
        tokenizer = self.ctx.tokenizer
        hf_processor = self.ctx.get_hf_processor()

        processed_inputs = hf_processor(
            text=prompt,  # type: ignore
            **mm_data,
            **mm_processor_kwargs,
        )
        new_token_ids, = processed_inputs.pop("input_ids").tolist()
        mm_kwargs = MultiModalKwargs(processed_inputs)

        mm_placeholders: Mapping[str, List[PlaceholderRange]] = {}

        for modality, orig_inputs in to_multi_format(mm_data).items():
            assert isinstance(orig_inputs, list)

            metadata = self.metadata[modality]
            placeholder_repls = metadata.placeholder_replacements
            repl_token_ids = {
                replacement["token_id"]
                for replacement in placeholder_repls.values()
            }

            modality_placeholders: List[PlaceholderRange] = []

            # In case HF processor already inserts placeholder tokens
            for new_token_id, run_info in iter_token_runs(new_token_ids):
                if new_token_id in repl_token_ids:
                    modality_placeholders.append(run_info)

            # Otherwise, we insert them ourselves
            if not modality_placeholders:
                for item_idx, orig_item in enumerate(orig_inputs):
                    for match_str, replacement in placeholder_repls.items():
                        replacement_count = replacement["count"]
                        if callable(replacement_count):
                            replacement_count = replacement_count(
                                orig_item,
                                processed_inputs,
                                item_idx,
                            )

                        for match_ids in candidate_placeholders(
                                tokenizer, match_str):
                            # TODO(youkaichao): Don't update new_token_ids
                            placeholders = apply_placeholders(
                                new_token_ids,
                                match_ids,
                                replacement["token_id"],
                                replacement_count,
                            )

                            if placeholders is not None:
                                modality_placeholders.append(placeholders)

            mm_placeholders[modality] = modality_placeholders  # type: ignore[index]  # yapf: disable

        return MultiModalInputsV2(
            type="multimodal",
            prompt=prompt,
            prompt_token_ids=new_token_ids,
            mm_kwargs=mm_kwargs,
            mm_placeholders=mm_placeholders,
        )
