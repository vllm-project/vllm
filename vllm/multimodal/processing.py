from dataclasses import dataclass
from functools import lru_cache
from heapq import nsmallest
from itertools import groupby
from typing import (Any, Callable, Generic, List, Mapping, NamedTuple,
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


def _encode(
    tokenizer: AnyTokenizer,
    text: str,
    *,
    add_special_tokens: bool = False,
) -> List[int]:
    """
    Backend-agnostic equivalent of HF's
    :code:`tokenizer.encode(text, add_special_tokens=...)`.
    """
    if isinstance(tokenizer, MistralTokenizer):
        return tokenizer.tokenizer.encode(text,
                                          bos=add_special_tokens,
                                          eos=add_special_tokens)

    return tokenizer.encode(text, add_special_tokens=add_special_tokens)


@lru_cache
def _max_vocab_token_len(tokenizer: AnyTokenizer) -> int:
    return max(len(token_text) for token_text in tokenizer.get_vocab())


class _TokenMatch(NamedTuple):
    start_idx: int
    end_idx: int


def find_token_match(token_ids: List[int], match_ids: List[int]):
    """
    Find the first occurrence of :code:`match_ids` in :code:`token_ids`.
    """
    match_len = len(match_ids)

    for start_idx in range(len(token_ids) - match_len + 1):
        end_idx = start_idx + match_len
        if token_ids[start_idx:end_idx] == match_ids:
            return _TokenMatch(start_idx, end_idx)

    return None


class _Candidate(NamedTuple):
    start_idx: int
    end_idx: int
    distance: int


def find_token_match_by_text(
    tokenizer: AnyTokenizer,
    token_ids: List[int],
    token_text: str,
    match_text: str,
):
    """
    Find the first occurrence of the tokenized :code:`match_text` in
    :code:`token_ids`.
    """
    match_ids = _encode(tokenizer, match_text, add_special_tokens=False)
    if (match := find_token_match(token_ids, match_ids)):
        return match

    # When `match_text` is not mapped to a special token ID,
    # it may be tokenized differently based on the surrounding tokens
    # as well as whether it is at the start/end of the string.
    # Therefore, we need to use `token_text` as a reference.
    text_start_idx = token_text.find(match_text)
    if text_start_idx == -1:
        return None

    text_end_idx = text_start_idx + len(match_text)

    # In case the left/right side of `match_text` is fused with the
    # string immediately before/after it during tokenization
    text_buffer = _max_vocab_token_len(tokenizer) - 1
    left_text = token_text[:max(0, text_start_idx - text_buffer)]
    right_text = token_text[:text_end_idx + text_buffer]

    left_idx = len(_encode(tokenizer, left_text, add_special_tokens=False))
    right_idx = len(_encode(tokenizer, right_text, add_special_tokens=True))
    window_size = len(match_ids)

    valid_candidates = list[_Candidate]()
    for start_idx in range(left_idx, right_idx - window_size + 1):
        end_idx = start_idx + window_size
        candidate_text = tokenizer.decode(
            token_ids[start_idx:end_idx],
            skip_special_tokens=False,
        )

        if match_text in candidate_text:
            candidate = _Candidate(
                start_idx=start_idx,
                end_idx=end_idx,
                distance=len(candidate_text) - len(match_text),
            )
            valid_candidates.append(candidate)

            if candidate.distance == 0:
                break

    assert len(valid_candidates) > 0, dict(
        # To facilitate debugging
        token_ids=token_ids,
        match_ids=match_ids,
        left_text=left_text,
        right_text=right_text,
        left_idx=left_idx,
        right_idx=right_idx,
    )

    best_candidate, = nsmallest(1, valid_candidates, key=lambda x: x.distance)
    return best_candidate.start_idx, best_candidate.end_idx


def apply_placeholders(
    tokenizer: AnyTokenizer,
    token_ids: List[int],
    token_text: str,
    match_text: str,
    replacement_id: int,
    replacement_count: int,
) -> Optional[PlaceholderRange]:
    """
    Find the first occurrence of the tokenized :code:`match_text` in
    :code:`token_ids`, and replace it with
    :code:`[replacement_id] * replacement_count`.

    This function updates :code:`token_ids` in place.
    """
    match = find_token_match_by_text(
        tokenizer,
        token_ids,
        token_text,
        match_text,
    )

    if match is None:
        return None

    # TODO(youkaichao): Don't update new_token_ids
    start_idx, end_idx = match
    token_ids[start_idx:end_idx] = [replacement_id] * replacement_count

    return PlaceholderRange(offset=start_idx, length=replacement_count)


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

                        placeholders = apply_placeholders(
                            tokenizer,
                            new_token_ids,
                            prompt,
                            match_str,
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
