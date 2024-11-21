from dataclasses import dataclass
from functools import lru_cache
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


_cached_encode = lru_cache(_encode)


@lru_cache
def _max_vocab_token_len(tokenizer: AnyTokenizer) -> int:
    return max(len(token_text) for token_text in tokenizer.get_vocab())


class _TokenMatch(NamedTuple):
    start_idx: int
    end_idx: int


def find_token_match(
    token_ids: List[int],
    match_ids: List[int],
) -> Optional[_TokenMatch]:
    """
    Find the first occurrence of :code:`match_ids` in :code:`token_ids`.
    """
    match_len = len(match_ids)

    for start_idx in range(len(token_ids) - match_len + 1):
        end_idx = start_idx + match_len
        if token_ids[start_idx:end_idx] == match_ids:
            return _TokenMatch(start_idx, end_idx)

    return None


class _TokenMatchFromTextCandidate(NamedTuple):
    start_idx: int
    end_idx: int

    match_text_prefix: str
    match_text_suffix: str

    @property
    def distance(self) -> int:
        return len(self.match_text_prefix) + len(self.match_text_suffix)


class _TokenMatchFromText(NamedTuple):
    start_idx: int
    end_idx: int

    match_prefix: List[int]
    match_suffix: List[int]

    match_text_prefix: str
    match_text_suffix: str


def find_token_match_by_text(
    tokenizer: AnyTokenizer,
    token_ids: List[int],
    token_text: str,
    match_text: str,
) -> Optional[_TokenMatchFromText]:
    """
    Find the first occurrence of the tokenized :code:`match_text` in
    :code:`token_ids`.
    """
    match_ids = _cached_encode(tokenizer, match_text, add_special_tokens=False)
    if (match := find_token_match(token_ids, match_ids)):
        return _TokenMatchFromText(
            match.start_idx,
            match.end_idx,
            match_prefix=[],
            match_suffix=[],
            match_text_prefix="",
            match_text_suffix="",
        )

    # When `match_text` is not mapped to a special token ID,
    # it may be tokenized differently based on the surrounding tokens
    # as well as whether it is at the start/end of the string.
    # Therefore, we need to use `token_text` as a reference.
    text_start_idx = token_text.find(match_text)
    if text_start_idx == -1:
        return None

    text_end_idx = text_start_idx + len(match_text)

    # In case the left/right side of `match_text` is fused with the
    # string immediately before/after it as a single token
    text_buffer = _max_vocab_token_len(tokenizer) - 1
    left_text = token_text[:max(0, text_start_idx - text_buffer)]
    right_text = token_text[:text_end_idx + text_buffer]

    left_idx = len(_encode(tokenizer, left_text, add_special_tokens=False))
    right_idx = len(_encode(tokenizer, right_text, add_special_tokens=True))
    window_size = len(match_ids)

    best_distance = len(token_text)
    best_candidate = None

    for start_idx in range(left_idx, right_idx - window_size + 1):
        end_idx = start_idx + window_size
        candidate_text = tokenizer.decode(
            token_ids[start_idx:end_idx],
            # In case match_text is a special token
            skip_special_tokens=False,
        )

        if match_text in candidate_text:
            candidate = _TokenMatchFromTextCandidate(
                start_idx,
                end_idx,
                *candidate_text.split(match_text, 1),
            )

            if candidate.distance < best_distance:
                best_candidate = candidate
                best_distance = candidate.distance

            if best_distance == 0:
                break

    assert best_candidate is not None, dict(
        # To facilitate debugging
        token_ids=token_ids,
        match_ids=match_ids,
        left_text=left_text,
        right_text=right_text,
        left_idx=left_idx,
        right_idx=right_idx,
    )

    match_token_prefix = _cached_encode(
        tokenizer,
        best_candidate.match_text_prefix,
        add_special_tokens=False,
    )
    match_token_suffix = _cached_encode(
        tokenizer,
        best_candidate.match_text_suffix,
        add_special_tokens=False,
    )

    return _TokenMatchFromText(
        start_idx=best_candidate.start_idx,
        end_idx=best_candidate.end_idx,
        match_prefix=match_token_prefix,
        match_suffix=match_token_suffix,
        match_text_prefix=best_candidate.match_text_prefix,
        match_text_suffix=best_candidate.match_text_suffix,
    )


def replace_by_text(
    tokenizer: AnyTokenizer,
    token_ids: List[int],
    token_text: str,
    match_text: str,
    replacement_id: int,
    replacement_count: int,
) -> tuple[List[int], str, Optional[PlaceholderRange]]:
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
        return token_ids, token_text, None

    start_idx, end_idx, prefix_ids, suffix_ids, prefix_str, suffix_str = match

    replacement_ids = (prefix_ids + [replacement_id] * replacement_count +
                       suffix_ids)
    replacement_text = tokenizer.decode(
        replacement_ids,
        # In case match_text is a special token
        skip_special_tokens=False,
    )

    token_ids[start_idx:end_idx] = replacement_ids
    token_text = token_text.replace(prefix_str + match_text + suffix_str,
                                    replacement_text, 1)

    return (token_ids, token_text,
            PlaceholderRange(offset=start_idx + len(prefix_ids),
                             length=replacement_count))


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

        new_prompt = prompt
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

            if modality_placeholders:
                new_prompt = tokenizer.decode(new_token_ids)
            else:  # Otherwise, we insert them ourselves
                for item_idx, orig_item in enumerate(orig_inputs):
                    for match_str, replacement in placeholder_repls.items():
                        replacement_count = replacement["count"]
                        if callable(replacement_count):
                            replacement_count = replacement_count(
                                orig_item,
                                processed_inputs,
                                item_idx,
                            )

                        (
                            new_token_ids,
                            new_prompt,
                            placeholders,
                        ) = replace_by_text(
                            tokenizer,
                            new_token_ids,
                            new_prompt,
                            match_str,
                            replacement["token_id"],
                            replacement_count,
                        )

                        if placeholders is not None:
                            modality_placeholders.append(placeholders)

            mm_placeholders[modality] = modality_placeholders  # type: ignore[index]  # yapf: disable

        return MultiModalInputsV2(
            type="multimodal",
            prompt=new_prompt,
            prompt_token_ids=new_token_ids,
            mm_kwargs=mm_kwargs,
            mm_placeholders=mm_placeholders,
        )
