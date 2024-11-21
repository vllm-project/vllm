import re
from dataclasses import dataclass
from functools import lru_cache
from itertools import groupby
from typing import (Any, Callable, Generic, Iterable, Mapping, NamedTuple,
                    Optional, Sequence, TypeVar, Union)

from transformers import BatchFeature
from typing_extensions import TypeAlias, TypedDict

from vllm.inputs import InputProcessingContext
from vllm.transformers_utils.tokenizer import AnyTokenizer, MistralTokenizer
from vllm.utils import full_groupby, is_list_of

from .inputs import (AudioItem, ImageItem, MultiModalDataDict,
                     MultiModalInputsV2, MultiModalKwargs, PlaceholderRange,
                     VideoItem)


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


PromptSegment: TypeAlias = Union[str, list[int]]


def bind_segment(
    prompt_segment: PromptSegment,
    tokenizer: AnyTokenizer,
) -> "_BoundPromptSegment":
    return _BoundPromptSegment(
        tokenizer=tokenizer,
        _text=prompt_segment if isinstance(prompt_segment, str) else None,
        _token_ids=prompt_segment
        if isinstance(prompt_segment, list) else None,
    )


_S_co = TypeVar("_S_co", bound=PromptSegment, covariant=True)
_T = TypeVar("_T")


@dataclass
class PromptReplacement(Generic[_S_co, _T]):
    target: _S_co
    """The prompt segment to find and replace."""

    repl_unit: _S_co
    """
    The unit making up the replacement prompt segment.
    
    See :code:`repl_count` for more details.
    """

    repl_count: Callable[[_T, BatchFeature, int], int]
    """
    Given the original data item, HF-processed data, and index of the processed
    item, output the number of repetitions of :code:`repl_unit` to build up the
    replacement prompt segment.
    """

    def bind(
        self,
        modality: str,
        tokenizer: AnyTokenizer,
    ) -> "_BoundPromptReplacement[_T]":
        return _BoundPromptReplacement(
            modality=modality,
            target=bind_segment(self.target, tokenizer),
            repl_unit=bind_segment(self.repl_unit, tokenizer),
            repl_count=self.repl_count,
        )


@dataclass
class _BoundPromptSegment:
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


@dataclass
class _BoundPromptReplacement(Generic[_T]):
    modality: str
    target: _BoundPromptSegment
    repl_unit: _BoundPromptSegment
    repl_count: Callable[[_T, BatchFeature, int], int]


@dataclass
class ModalityProcessingMetadata(Generic[_T]):
    prompt_repls: Sequence[PromptReplacement[PromptSegment, _T]]
    """
    Defines each segment to replace in the HF-processed prompt.

    This is skipped if the HF-processed prompt is found to already contain
    the replacement prompts.
    """

    def bind_prompt_repls(
        self,
        modality: str,
        tokenizer: AnyTokenizer,
    ) -> list[_BoundPromptReplacement[_T]]:
        return [
            prompt_repl.bind(modality, tokenizer)
            for prompt_repl in self.prompt_repls
        ]


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


class _TokenRun(TypedDict):
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


class _TokenMatch(NamedTuple):
    start_idx: int
    end_idx: int

    @property
    def length(self) -> int:
        return self.end_idx - self.start_idx


def iter_token_matches(
    token_ids: list[int],
    match_ids: list[int],
) -> Iterable[_TokenMatch]:
    """
    Yield each occurrence of :code:`match_ids` in :code:`token_ids`.
    """
    match_len = len(match_ids)

    for start_idx in range(len(token_ids) - match_len + 1):
        end_idx = start_idx + match_len
        if token_ids[start_idx:end_idx] == match_ids:
            yield _TokenMatch(start_idx, end_idx)


class _BoundPlaceholderRange(TypedDict):
    modality: str
    offset: int
    length: int


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

    def _extract_placeholder_ranges(
        self,
        all_prompt_repls: Sequence[_BoundPromptReplacement[Any]],
        new_token_ids: list[int],
        *,
        # To avoid false positives from multi-input when detecting
        # whether HF processor already inserts placeholder tokens
        min_placeholder_count: int = 16,
    ) -> Iterable[_BoundPlaceholderRange]:
        placeholder_ids_by_modality = {
            modality: {
                token_id
                for prompt_repl in prompt_repls
                for token_id in prompt_repl.repl_unit.token_ids
            }
            for modality, prompt_repls in full_groupby(
                all_prompt_repls, key=lambda x: x.modality)
        }

        # In case HF processor already inserts placeholder tokens
        for run_info in iter_token_runs(new_token_ids):
            if run_info["length"] > min_placeholder_count:
                for (modality,
                     placeholder_ids) in placeholder_ids_by_modality.items():
                    if run_info["token_id"] in placeholder_ids:
                        yield _BoundPlaceholderRange(
                            modality=modality,
                            offset=run_info["start_idx"],
                            length=run_info["length"],
                        )

    def _find_token_id_matches(
        self,
        token_ids: list[int],
        prompt_repls: Sequence[_BoundPromptReplacement[_T]],
    ) -> list[tuple[str, _TokenMatch]]:
        return [(prompt_repl.target.text, match)
                for prompt_repl in prompt_repls for match in
                iter_token_matches(token_ids, prompt_repl.target.token_ids)]

    def _replace_token_id_matches(
        self,
        token_ids: list[int],
        prompt_repls: Sequence[_BoundPromptReplacement[_T]],
        matches: Sequence[tuple[str, _TokenMatch]],
        mm_items_by_modality: Mapping[str, list[_T]],
        hf_inputs: BatchFeature,
    ) -> list[int]:
        prompt_repls_by_target_text = {
            prompt_repl.target.text: prompt_repl
            for prompt_repl in prompt_repls
        }

        # To ensure that later replacements don't affect
        # the placeholder ranges of earlier ones
        sorted_matches = sorted(matches, key=lambda x: x[1].start_idx)

        out_token_ids = list[int]()
        prev_end_idx = 0

        for i, (target_text, (start_idx,
                              end_idx)) in enumerate(sorted_matches):
            prompt_repl = prompt_repls_by_target_text[target_text]
            mm_items = mm_items_by_modality[prompt_repl.modality]

            repl_count = prompt_repl.repl_count(mm_items[i], hf_inputs, i)
            repl_ids = prompt_repl.repl_unit.token_ids * repl_count

            out_token_ids.extend(token_ids[prev_end_idx:start_idx] + repl_ids)
            prev_end_idx = end_idx

        return out_token_ids

    def _iter_text_matches(
        self,
        token_text: str,
        prompt_repls: Sequence[_BoundPromptReplacement[_T]],
    ) -> Iterable[tuple[str, re.Match]]:
        for prompt_repl in prompt_repls:
            target_text = prompt_repl.target.text
            for match in re.finditer(re.escape(target_text), token_text):
                yield target_text, match

    def _find_and_replace_token_text_matches(
        self,
        token_ids: list[int],
        prompt_repls: Sequence[_BoundPromptReplacement[_T]],
        mm_items_by_modality: Mapping[str, list[_T]],
        hf_inputs: BatchFeature,
    ) -> list[int]:
        tokenizer = self.ctx.tokenizer
        token_text = _decode(tokenizer, token_ids)

        prompt_repls_by_target_text = {
            prompt_repl.target.text: prompt_repl
            for prompt_repl in prompt_repls
        }

        # To ensure that later replacements don't affect
        # the placeholder ranges of earlier ones
        sorted_matches = sorted(
            self._iter_text_matches(token_text, prompt_repls),
            key=lambda x: x[1].start(),
        )

        out_texts = list[str]()
        prev_end_idx = 0

        for i, (target_text, match) in enumerate(sorted_matches):
            prompt_repl = prompt_repls_by_target_text[target_text]
            mm_items = mm_items_by_modality[prompt_repl.modality]

            repl_count = prompt_repl.repl_count(mm_items[i], hf_inputs, i)
            repl_text = prompt_repl.repl_unit.text * repl_count

            out_texts.extend(token_text[prev_end_idx:match.start()] +
                             repl_text)

            prev_end_idx = match.end()

        return _encode(tokenizer, "".join(out_texts))

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

    def _apply_prompt_replacements(
        self,
        mm_data: MultiModalDataDict,
        hf_inputs: BatchFeature,
        new_token_ids: list[int],
    ) -> tuple[list[int], list[_BoundPlaceholderRange]]:
        tokenizer = self.ctx.tokenizer
        all_prompt_repls = [
            prompt_repl for modality, metadata in self.metadata.items()
            if modality in mm_data
            for prompt_repl in metadata.bind_prompt_repls(modality, tokenizer)
        ]

        # In case HF processor already inserts placeholder tokens
        all_placeholder_ranges = list(
            self._extract_placeholder_ranges(all_prompt_repls, new_token_ids))

        # Otherwise, we insert them ourselves
        if not all_placeholder_ranges:
            mm_items = to_multi_format(mm_data)
            token_id_matches = self._find_token_id_matches(
                new_token_ids,
                all_prompt_repls,
            )

            if len(token_id_matches) == len(mm_items):
                new_token_ids = self._replace_token_id_matches(
                    new_token_ids,
                    all_prompt_repls,
                    token_id_matches,
                    mm_items,
                    hf_inputs,
                )
            else:
                new_token_ids = self._find_and_replace_token_text_matches(
                    new_token_ids,
                    all_prompt_repls,
                    mm_items,
                    hf_inputs,
                )

            all_placeholder_ranges = list(
                self._extract_placeholder_ranges(all_prompt_repls,
                                                 new_token_ids))

        return new_token_ids, all_placeholder_ranges

    def apply(
        self,
        prompt: str,
        mm_data: MultiModalDataDict,
        mm_processor_kwargs: Mapping[str, object],
    ) -> MultiModalInputsV2:
        tokenizer = self.ctx.tokenizer

        hf_inputs = self._apply_hf_processor(prompt, mm_data,
                                             mm_processor_kwargs)
        new_token_ids, = hf_inputs.pop("input_ids").tolist()
        mm_kwargs = MultiModalKwargs(hf_inputs)

        (
            new_token_ids,
            all_placeholder_ranges,
        ) = self._apply_prompt_replacements(
            mm_data,
            hf_inputs,
            new_token_ids,
        )

        mm_placeholders = {
            modality: [
                PlaceholderRange(offset=item["offset"], length=item["length"])
                for item in items
            ]
            for modality, items in full_groupby(all_placeholder_ranges,
                                                key=lambda x: x["modality"])
        }

        return MultiModalInputsV2(
            type="multimodal",
            prompt=_decode(tokenizer, new_token_ids),
            prompt_token_ids=new_token_ids,
            mm_kwargs=mm_kwargs,
            mm_placeholders=mm_placeholders,
        )
