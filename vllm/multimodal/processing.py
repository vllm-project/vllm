import re
from abc import ABC, abstractmethod
from collections.abc import Callable, ItemsView, Iterable, Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache
from itertools import groupby
from typing import Any, Generic, NamedTuple, Optional, Protocol, TypeVar, Union

import numpy as np
from transformers import BatchFeature
from typing_extensions import TypeAlias, TypedDict

from vllm.inputs import InputProcessingContext
from vllm.transformers_utils.tokenizer import AnyTokenizer, MistralTokenizer
from vllm.utils import flatten_2d_lists, full_groupby, is_list_of

from .inputs import (AudioItem, ImageItem, MultiModalDataDict,
                     MultiModalInputsV2, MultiModalKwargs, PlaceholderRange,
                     VideoItem)


def bind_prompt_sequence(
    seq: Union[str, list[int]],
    tokenizer: AnyTokenizer,
) -> "_BoundPromptSequence":
    """
    Bind a text or token sequence to a tokenizer so that it can be
    lazily converted into the other format on demand.
    """
    return _BoundPromptSequence(
        tokenizer=tokenizer,
        _text=seq if isinstance(seq, str) else None,
        _token_ids=seq if isinstance(seq, list) else None,
    )


_T = TypeVar("_T")
_S = TypeVar("_S", str, list[int])


@dataclass
class PromptReplacement(Generic[_S, _T]):
    target: _S
    """The text or token sequence to find and replace."""

    repl_unit: _S
    """
    The unit making up the replacement text or token sequence.
    
    See :code:`repl_count` for more details.
    """

    repl_count: Union[Callable[[list[_T], BatchFeature, int], int], int]
    """
    Given the original multi-modal items for this modality, HF-processed data,
    and index of the processed item, output the number of repetitions of
    :code:`repl_unit` to build up the replacement text or token sequence.

    For convenience, you can pass in an integer if the number of repetitions is
    a constant.
    """

    def __repr__(self) -> str:
        return (f"{type(self).__name__}(target={self.target!r}, "
                f"repl_unit={self.repl_unit!r})")

    def bind(
        self,
        modality: str,
        tokenizer: AnyTokenizer,
    ) -> "_BoundPromptReplacement[_T]":
        return _BoundPromptReplacement(
            modality=modality,
            target=bind_prompt_sequence(self.target, tokenizer),
            repl_unit=bind_prompt_sequence(self.repl_unit, tokenizer),
            repl_count=self.repl_count,
        )


@dataclass
class ModalityProcessingMetadata(Generic[_T]):
    prompt_repls: Sequence[Union[PromptReplacement[str, _T],
                                 PromptReplacement[list[int], _T]]]
    """
    Defines each text or token sequence to replace in the HF-processed prompt.

    This is skipped if the HF-processed prompt is found to already contain
    the replacement prompts.
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


def _encode(
    tokenizer: AnyTokenizer,
    text: str,
    *,
    add_special_tokens: bool = False,
) -> list[int]:
    """
    Backend-agnostic equivalent of HF's
    :code:`tokenizer.encode(text, add_special_tokens=...)`.
    """
    if isinstance(tokenizer, MistralTokenizer):
        return tokenizer.tokenizer.encode(text,
                                          bos=add_special_tokens,
                                          eos=add_special_tokens)

    return tokenizer.encode(text, add_special_tokens=add_special_tokens)


@lru_cache(maxsize=2048)
def _cached_encode(
    tokenizer: AnyTokenizer,
    text: str,
    *,
    add_special_tokens: bool = False,
) -> list[int]:
    return _encode(tokenizer, text, add_special_tokens=add_special_tokens)


def _decode(
    tokenizer: AnyTokenizer,
    token_ids: list[int],
    *,
    skip_special_tokens: bool = False,
) -> str:
    """
    Backend-agnostic equivalent of HF's
    :code:`tokenizer.decode(token_ids, skip_special_tokens=...)`.
    """
    return tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)


@lru_cache(maxsize=2048)
def _cached_decode(
    tokenizer: AnyTokenizer,
    token_ids: tuple[int, ...],
    *,
    skip_special_tokens: bool = False,
) -> str:
    return _decode(tokenizer,
                   list(token_ids),
                   skip_special_tokens=skip_special_tokens)


class _HasModalityAttr(Protocol):
    modality: str


class _HasModalityProp(Protocol):

    @property
    def modality(self) -> str:
        ...


_M = TypeVar("_M", bound=Union[_HasModalityAttr, _HasModalityProp])


def full_groupby_modality(values: Iterable[_M]) -> ItemsView[str, list[_M]]:
    """Convenience function to apply :func:`full_groupby` based on modality."""
    return full_groupby(values, key=lambda x: x.modality)


@dataclass
class _BoundPromptSequence:
    tokenizer: AnyTokenizer
    _text: Optional[str]
    _token_ids: Optional[list[int]]

    def __post_init__(self) -> None:
        if self._text is None and self._token_ids is None:
            raise ValueError("At least one of 'text' and 'token_ids' must be "
                             "specified")

    @property
    def text(self) -> str:
        if self._text is None:
            assert self._token_ids is not None
            self._text = _cached_decode(self.tokenizer, tuple(self._token_ids))

        return self._text

    @property
    def token_ids(self) -> list[int]:
        if self._token_ids is None:
            assert self._text is not None
            self._token_ids = _cached_encode(self.tokenizer, self._text)

        return self._token_ids

    def __repr__(self) -> str:
        return (f"{type(self).__name__}(_text={self._text!r}, "
                f"_token_ids={self._token_ids!r})")


@dataclass
class _BoundPromptReplacement(Generic[_T]):
    modality: str
    target: _BoundPromptSequence
    repl_unit: _BoundPromptSequence
    repl_count: Union[Callable[[list[_T], BatchFeature, int], int], int]

    def get_count(
        self,
        mm_items: list[_T],
        hf_inputs: BatchFeature,
        item_idx: int,
    ) -> int:
        repl_count = self.repl_count
        if isinstance(repl_count, int):
            return repl_count

        return repl_count(mm_items, hf_inputs, item_idx)


def to_multi_format(data: MultiModalDataDict) -> dict[str, list[Any]]:
    """
    Convert a :class:`MultiModalDataDict` containing single data items
    to a :class:`MultiModalMultiDataDict` containing multiple data items
    per entry.
    """
    multi_data = dict[str, list[Any]]()

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


class _TokenRun(NamedTuple):
    token_id: int

    start_idx: int
    length: int


def iter_token_runs(token_ids: list[int]) -> Iterable[_TokenRun]:
    """
    Yield the starting index and length of each run of tokens that are the same.
    """
    start_idx = 0

    for token_id, it in groupby(token_ids):
        length = sum(1 for _ in it)
        yield _TokenRun(token_id=token_id, start_idx=start_idx, length=length)

        start_idx += length


class _PlaceholderInfo(NamedTuple):
    modality: str
    offset: int
    length: int

    def to_range(self) -> PlaceholderRange:
        return PlaceholderRange(offset=self.offset, length=self.length)


def iter_placeholders(
    prompt_repls: Sequence[_BoundPromptReplacement[Any]],
    token_ids: list[int],
    *,
    min_placeholder_count: int,
) -> Iterable[_PlaceholderInfo]:
    """Yield each set of placeholder tokens found in :code:`token_ids`."""
    placeholder_ids_by_modality = {
        modality: {
            token_id
            for prompt_repl in repls
            for token_id in prompt_repl.repl_unit.token_ids
        }
        for modality, repls in full_groupby_modality(prompt_repls)
    }

    for run_info in iter_token_runs(token_ids):
        if run_info.length > min_placeholder_count:
            for (modality,
                 placeholder_ids) in placeholder_ids_by_modality.items():
                if run_info.token_id in placeholder_ids:
                    yield _PlaceholderInfo(
                        modality=modality,
                        offset=run_info.start_idx,
                        length=run_info.length,
                    )


class _TokenMatch(NamedTuple):
    start_idx: int
    end_idx: int


def iter_token_matches(
    token_ids: list[int],
    match_ids: list[int],
) -> Iterable[_TokenMatch]:
    """Yield each occurrence of :code:`match_ids` in :code:`token_ids`."""
    match_len = len(match_ids)

    last_end_idx = 0
    for start_idx in range(len(token_ids) - match_len + 1):
        if start_idx < last_end_idx:
            continue  # Exclude overlapping matches

        end_idx = start_idx + match_len
        if token_ids[start_idx:end_idx] == match_ids:
            yield _TokenMatch(start_idx=start_idx, end_idx=end_idx)
            last_end_idx = end_idx


class _PromptReplacementMatch(ABC, Generic[_T, _S]):
    prompt_repl: _BoundPromptReplacement[_T]

    @property
    def modality(self) -> str:
        return self.prompt_repl.modality

    @property
    @abstractmethod
    def start_idx(self) -> int:
        raise NotImplementedError

    @property
    @abstractmethod
    def end_idx(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def get_repl(
        self,
        mm_items: list[_T],
        hf_inputs: BatchFeature,
        item_idx: int,
    ) -> _S:
        raise NotImplementedError

    def __repr__(self) -> str:
        return (f"{type(self).__name__}(modality={self.modality!r}, "
                f"start_idx={self.start_idx!r}, end_idx={self.end_idx!r})")


@dataclass(repr=False)
class _PromptReplacementTokenMatch(_PromptReplacementMatch[_T, list[int]]):
    prompt_repl: _BoundPromptReplacement[_T]
    match: _TokenMatch

    @property
    def start_idx(self) -> int:
        return self.match.start_idx

    @property
    def end_idx(self) -> int:
        return self.match.end_idx

    def get_repl(
        self,
        mm_items: list[_T],
        hf_inputs: BatchFeature,
        item_idx: int,
    ) -> list[int]:
        prompt_repl = self.prompt_repl
        count = prompt_repl.get_count(mm_items, hf_inputs, item_idx)
        return prompt_repl.repl_unit.token_ids * count


@dataclass(repr=False)
class _PromptReplacementTextMatch(_PromptReplacementMatch[_T, str]):
    prompt_repl: _BoundPromptReplacement[_T]
    match: re.Match[str]

    @property
    def start_idx(self) -> int:
        return self.match.start()

    @property
    def end_idx(self) -> int:
        return self.match.end()

    def get_repl(
        self,
        mm_items: list[_T],
        hf_inputs: BatchFeature,
        item_idx: int,
    ) -> str:
        prompt_repl = self.prompt_repl
        count = prompt_repl.get_count(mm_items, hf_inputs, item_idx)
        return prompt_repl.repl_unit.text * count


def find_token_matches(
    prompt: list[int],
    prompt_repls: Sequence[_BoundPromptReplacement[_T]],
) -> list[_PromptReplacementTokenMatch[_T]]:
    """Return each target of :code:`prompt_repls` found in :code:`prompt`."""
    return [
        _PromptReplacementTokenMatch(prompt_repl, match)
        for prompt_repl in prompt_repls
        for match in iter_token_matches(prompt, prompt_repl.target.token_ids)
    ]


def find_text_matches(
    prompt: str,
    prompt_repls: Sequence[_BoundPromptReplacement[_T]],
) -> list[_PromptReplacementTextMatch[_T]]:
    """Return each target of :code:`prompt_repls` found in :code:`prompt`."""
    return [
        _PromptReplacementTextMatch(prompt_repl, match)
        for prompt_repl in prompt_repls
        for match in re.finditer(re.escape(prompt_repl.target.text), prompt)
    ]


def _resolve_matches(
    prompt: _S,
    matches: Sequence[_PromptReplacementMatch[_T, _S]],
) -> list[_PromptReplacementMatch[_T, _S]]:
    """
    Resolve :code:`matches` to ensure that there are no overlapping matches,
    and sort them such that earlier matches take priority over later ones.
    """
    num_matches_by_idx = np.zeros(len(prompt), dtype=int)
    for match in matches:
        num_matches_by_idx[match.start_idx:match.end_idx] += 1

    duplicate_matches_idxs, = np.nonzero(num_matches_by_idx > 1)
    if len(duplicate_matches_idxs) > 0:
        raise ValueError("Unable to find a unique replacement "
                         f"at indices={duplicate_matches_idxs} "
                         f"of prompt={prompt}")

    return sorted(matches, key=lambda x: x.start_idx)


def _replace_matches(
    prompt: _S,
    matches: Sequence[_PromptReplacementMatch[_T, _S]],
    mm_items_by_modality: Mapping[str, list[_T]],
    hf_inputs: BatchFeature,
) -> list[_S]:
    out_seqs = list[_S]()
    prev_end_idx = 0
    next_idx_by_modality = {modality: 0 for modality in mm_items_by_modality}

    for match in _resolve_matches(prompt, matches):
        modality = match.modality
        mm_items = mm_items_by_modality[modality]

        item_idx = next_idx_by_modality[modality]
        if item_idx >= len(mm_items):
            continue

        start_idx = match.start_idx
        end_idx = match.end_idx
        repl_ids = match.get_repl(mm_items, hf_inputs, item_idx)

        out_seqs.append(prompt[prev_end_idx:start_idx] + repl_ids)
        prev_end_idx = end_idx
        next_idx_by_modality[modality] += 1

    out_seqs.append(prompt[prev_end_idx:])

    return out_seqs


def replace_token_matches(
    prompt: list[int],
    matches: Sequence[_PromptReplacementMatch[_T, list[int]]],
    mm_items_by_modality: Mapping[str, list[_T]],
    hf_inputs: BatchFeature,
) -> list[int]:
    """Apply :code:`prompt_repls` to :code:`prompt`."""
    if not matches:
        return prompt

    token_id_seqs = _replace_matches(
        prompt,
        matches,
        mm_items_by_modality,
        hf_inputs,
    )

    return flatten_2d_lists(token_id_seqs)


def replace_text_matches(
    prompt: str,
    matches: Sequence[_PromptReplacementMatch[_T, str]],
    mm_items_by_modality: Mapping[str, list[_T]],
    hf_inputs: BatchFeature,
) -> str:
    """Apply :code:`prompt_repls` to :code:`prompt`."""
    if not matches:
        return prompt

    texts = _replace_matches(
        prompt,
        matches,
        mm_items_by_modality,
        hf_inputs,
    )

    return "".join(texts)


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

    def _find_placeholders(
        self,
        all_prompt_repls: Sequence[_BoundPromptReplacement[Any]],
        new_token_ids: list[int],
        *,
        # To avoid false positives from multi-input when detecting
        # whether placeholder tokens have been inserted, in case
        # the target sequence is a subset of the replacement tokens
        min_placeholder_count: int = 16,
    ) -> list[_PlaceholderInfo]:
        return list(
            iter_placeholders(
                all_prompt_repls,
                new_token_ids,
                min_placeholder_count=min_placeholder_count,
            ))

    def _apply_hf_processor(
        self,
        prompt: str,
        mm_data: MultiModalDataDict,
        mm_processor_kwargs: Mapping[str, object],
    ) -> BatchFeature:
        hf_processor = self.ctx.get_hf_processor()

        return hf_processor(
            text=prompt,  # type: ignore
            **mm_data,
            **mm_processor_kwargs,
        )

    def _bind_prompt_replacements(
        self,
        mm_data: MultiModalDataDict,
    ) -> list[_BoundPromptReplacement[Any]]:
        tokenizer = self.ctx.tokenizer

        return [
            prompt_repl.bind(modality, tokenizer)
            for modality, metadata in self.metadata.items()
            if modality in mm_data for prompt_repl in metadata.prompt_repls
        ]

    def _apply_prompt_replacements(
        self,
        mm_data: MultiModalDataDict,
        hf_inputs: BatchFeature,
        token_ids: list[int],
        prompt_repls: Sequence[_BoundPromptReplacement[Any]],
    ) -> tuple[list[int], str, list[_PlaceholderInfo]]:
        tokenizer = self.ctx.tokenizer

        mm_items = to_multi_format(mm_data)
        token_matches = find_token_matches(token_ids, prompt_repls)

        # If the search text does not represent a special token,
        # it may have different token IDs in the prompt, because
        # the tokens may go across the boundaries of the search text.
        # ----
        # e.g. when searching for "foo" in "food", if "food" itself makes
        # up a token, then the token ID of "foo" will not appear at all
        # ----
        # Since it is inefficient to search for all possible tokenizations
        # of the search text in the prompt, we instead perform string
        # replacement on the decoded token IDs, then encode them back.
        if all(
            len(matches) >= len(mm_data[modality])
            for modality, matches in full_groupby_modality(token_matches)
        ):  # yapf: disable
            token_ids = replace_token_matches(
                token_ids,
                token_matches,
                mm_items,
                hf_inputs,
            )

            text = _decode(tokenizer, token_ids)
            matched_repls = [match.prompt_repl for match in token_matches]
        else:
            text = _decode(tokenizer, token_ids)

            text_matches = find_text_matches(text, prompt_repls)
            text = replace_text_matches(
                text,
                text_matches,
                mm_items,
                hf_inputs,
            )

            token_ids = _encode(tokenizer, text)
            matched_repls = [match.prompt_repl for match in text_matches]

        placeholders = self._find_placeholders(matched_repls, token_ids)

        # Sanity check
        assert len(placeholders) == len(matched_repls), dict(
            # Log this information for easier debugging
            text=text,
            token_ids=token_ids,
            placeholders=placeholders,
            matched_repls=matched_repls,
        )

        return token_ids, text, placeholders

    def apply(
        self,
        prompt_text: str,
        mm_data: MultiModalDataDict,
        mm_processor_kwargs: Mapping[str, object],
    ) -> MultiModalInputsV2:
        """
        Process multi-modal inputs to be used in vLLM.

        The main steps are:

        1. Apply HF Processor on prompt text and multi-modal data together,
           outputting token IDs and processed tensors.
        2. Find and replace sequences in the token IDs with placeholder tokens.
           The number of placeholder tokens equals the feature size of the
           multi-modal data outputted by the multi-modal encoder.
        3. Extract information about the placeholder tokens from the
           processed token IDs.
        """
        tokenizer = self.ctx.tokenizer

        hf_inputs = self._apply_hf_processor(prompt_text, mm_data,
                                             mm_processor_kwargs)
        prompt_ids, = hf_inputs.pop("input_ids").tolist()
        mm_kwargs = MultiModalKwargs(hf_inputs)

        all_prompt_repls = self._bind_prompt_replacements(mm_data)

        # If HF processor already inserts placeholder tokens,
        # there is no need for us to insert them
        all_placeholders = self._find_placeholders(all_prompt_repls,
                                                   prompt_ids)
        if all_placeholders:
            prompt_text = _decode(tokenizer, prompt_ids)
        else:
            (
                prompt_ids,
                prompt_text,
                all_placeholders,
            ) = self._apply_prompt_replacements(
                mm_data,
                hf_inputs,
                prompt_ids,
                all_prompt_repls,
            )

        mm_placeholders = {
            modality: [item.to_range() for item in items]
            for modality, items in full_groupby_modality(all_placeholders)
        }

        return MultiModalInputsV2(
            type="multimodal",
            prompt=prompt_text,
            prompt_token_ids=prompt_ids,
            mm_kwargs=mm_kwargs,
            mm_placeholders=mm_placeholders,
        )
