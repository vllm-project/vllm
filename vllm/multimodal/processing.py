import re
from abc import ABC, abstractmethod
from collections import UserDict
from collections.abc import Callable, ItemsView, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, NamedTuple, Optional, Protocol, TypeVar, Union

import numpy as np
import torch
from PIL.Image import Image
from transformers import BatchFeature, ProcessorMixin
from typing_extensions import assert_never

from vllm.inputs import DummyData, InputProcessingContext
from vllm.logger import init_logger
from vllm.transformers_utils.tokenizer import AnyTokenizer, MistralTokenizer
from vllm.utils import flatten_2d_lists, full_groupby, is_list_of

from .inputs import (AudioItem, ImageItem, MultiModalDataDict,
                     MultiModalInputsV2, MultiModalKwargs, PlaceholderRange,
                     VideoItem)

logger = init_logger(__name__)

_S = TypeVar("_S", str, list[int])
_PromptSeq = Union[str, list[int]]


@dataclass
class PromptReplacement:
    modality: str
    """The modality for which the replacement is made"""

    target: _PromptSeq
    """The text or token sequence to find and replace."""

    replacement: Union[Callable[[int], _PromptSeq],
                       _PromptSeq] = field(repr=False)
    """
    Given the index of the processed item within :attr:`modality`, output the
    replacement text or token sequence.

    For convenience, you can pass in the replacement instead of a function
    if it does not depend on the input.
    """

    def bind(self, tokenizer: AnyTokenizer) -> "_BoundPromptReplacement":
        return _BoundPromptReplacement(
            tokenizer=tokenizer,
            modality=self.modality,
            _target=self.target,
            _replacement=self.replacement,
        )


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
    tokenizer: AnyTokenizer = field(repr=False)

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
class _BoundPromptReplacement:
    tokenizer: AnyTokenizer = field(repr=False)
    modality: str

    _target: _PromptSeq
    _replacement: Union[Callable[[int], _PromptSeq],
                        _PromptSeq] = field(repr=False)

    def __post_init__(self) -> None:
        self._replacement_cache = dict[int, _BoundPromptSequence]()

    @property
    def target(self) -> _BoundPromptSequence:
        target = self._target

        return _BoundPromptSequence(
            tokenizer=self.tokenizer,
            _text=target if isinstance(target, str) else None,
            _token_ids=target if isinstance(target, list) else None,
        )

    def get_replacement(self, item_idx: int) -> _BoundPromptSequence:
        replacement = self._replacement
        if callable(replacement):
            cache_key = item_idx
            if cache_key in self._replacement_cache:
                return self._replacement_cache[cache_key]

            replacement = replacement(item_idx)
        else:
            cache_key = None

        bound_replacement = _BoundPromptSequence(
            tokenizer=self.tokenizer,
            _text=replacement if isinstance(replacement, str) else None,
            _token_ids=replacement if isinstance(replacement, list) else None,
        )

        if cache_key is not None:
            self._replacement_cache[cache_key] = bound_replacement

        return bound_replacement


class ImageSize(NamedTuple):
    width: int
    height: int


class MultiModalDataItems(UserDict[str, list[Any]]):
    """
    As :class:`MultiModalDataDict`, but normalized such that each entry
    corresponds to a list.
    """

    @property
    def image(self) -> list[ImageItem]:
        return self["image"]

    @property
    def video(self) -> list[VideoItem]:
        return self["video"]

    @property
    def audio(self) -> list[AudioItem]:
        return self["audio"]

    def get_image_size(self, item_idx: int) -> ImageSize:
        image = self.image[item_idx]

        if isinstance(image, Image):
            return ImageSize(*image.size)
        if isinstance(image, (np.ndarray, torch.Tensor)):
            _, h, w = image.shape
            return ImageSize(w, h)

        assert_never(image)


def to_multi_format(data: MultiModalDataDict) -> MultiModalDataItems:
    """
    Normalize :class:`MultiModalDataDict` to :class:`MultiModalDataItems`.
    """
    multi_data = MultiModalDataItems()

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


class _TokenMatch(NamedTuple):
    start_idx: int
    end_idx: int


def iter_token_matches(
    token_ids: list[int],
    match_ids: list[int],
) -> Iterable[_TokenMatch]:
    """
    Yield each occurrence of :code:`match_ids` in :code:`token_ids`.

    Note that empty matches are ignored.
    """
    prompt_len = len(token_ids)
    match_len = len(match_ids)

    if match_len == 0:
        return

    start_idx = 0
    while start_idx < prompt_len - match_len + 1:
        end_idx = start_idx + match_len

        if token_ids[start_idx:end_idx] == match_ids:
            yield _TokenMatch(start_idx=start_idx, end_idx=end_idx)

            # Exclude overlapping matches
            start_idx = end_idx
        else:
            start_idx += 1


@dataclass(repr=False)
class _PromptReplacementMatch(ABC):
    prompt_repl: _BoundPromptReplacement

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

    def __repr__(self) -> str:
        return (f"{type(self).__name__}(modality={self.modality!r}, "
                f"start_idx={self.start_idx!r}, end_idx={self.end_idx!r})")


@dataclass(repr=False)
class _PromptReplacementTokenMatch(_PromptReplacementMatch):
    match: _TokenMatch

    @property
    def start_idx(self) -> int:
        return self.match.start_idx

    @property
    def end_idx(self) -> int:
        return self.match.end_idx


@dataclass(repr=False)
class _PromptReplacementTextMatch(_PromptReplacementMatch):
    match: re.Match[str]

    @property
    def start_idx(self) -> int:
        return self.match.start()

    @property
    def end_idx(self) -> int:
        return self.match.end()


class _PlaceholderInfo(NamedTuple):
    modality: str
    start_idx: int
    replacement: list[int]

    @property
    def length(self) -> int:
        return len(self.replacement)

    def to_range(self) -> PlaceholderRange:
        return PlaceholderRange(
            offset=self.start_idx,
            length=self.length,
        )


def find_token_matches(
    prompt: list[int],
    prompt_repls: Sequence[_BoundPromptReplacement],
) -> list[_PromptReplacementTokenMatch]:
    """Return each target of :code:`prompt_repls` found in :code:`prompt`."""
    return [
        _PromptReplacementTokenMatch(prompt_repl, match)
        for prompt_repl in prompt_repls
        for match in iter_token_matches(prompt, prompt_repl.target.token_ids)
    ]


def find_text_matches(
    prompt: str,
    prompt_repls: Sequence[_BoundPromptReplacement],
) -> list[_PromptReplacementTextMatch]:
    """Return each target of :code:`prompt_repls` found in :code:`prompt`."""
    return [
        _PromptReplacementTextMatch(prompt_repl, match)
        for prompt_repl in prompt_repls
        for match in re.finditer(re.escape(prompt_repl.target.text), prompt)
    ]


def _resolve_matches(
    prompt: _PromptSeq,
    matches: Sequence[_PromptReplacementMatch],
) -> list[_PromptReplacementMatch]:
    """
    Resolve :code:`matches` to ensure that there are no overlapping matches,
    and sort them such that earlier matches take priority over later ones.
    """
    seen_matches: list[Optional[_PromptReplacementMatch]] = [None
                                                             ] * len(prompt)

    for match in matches:
        for idx in range(match.start_idx, match.end_idx):
            if seen_matches[idx] is not None:
                raise ValueError("Found overlapping matches "
                                 f"({seen_matches[idx]} and {match}) "
                                 f"at index={idx} of prompt={prompt}")

            seen_matches[idx] = match

    return sorted(matches, key=lambda x: x.start_idx)


def _replace_matches(
    prompt: _S,
    matches: Sequence[_PromptReplacementMatch],
    mm_items: MultiModalDataItems,
) -> list[_S]:
    out_seqs = list[_S]()
    prev_end_idx = 0
    next_idx_by_modality = {modality: 0 for modality in mm_items}

    for match in _resolve_matches(prompt, matches):
        modality = match.modality
        modal_items = mm_items[modality]

        item_idx = next_idx_by_modality[modality]
        if item_idx >= len(modal_items):
            continue

        start_idx = match.start_idx
        end_idx = match.end_idx

        repl_info = match.prompt_repl
        replacement = repl_info.get_replacement(item_idx)

        if isinstance(prompt, str):
            repl_seq = replacement.text
            out_seqs.append(prompt[prev_end_idx:start_idx] + repl_seq)
        else:
            repl_seq = replacement.token_ids
            out_seqs.append(prompt[prev_end_idx:start_idx] + repl_seq)

        prev_end_idx = end_idx
        next_idx_by_modality[modality] += 1

    out_seqs.append(prompt[prev_end_idx:])

    return out_seqs


def replace_token_matches(
    prompt: list[int],
    matches: Sequence[_PromptReplacementTokenMatch],
    mm_items: MultiModalDataItems,
) -> list[int]:
    """Apply :code:`prompt_repls` to :code:`prompt`."""
    if not matches:
        return prompt

    token_id_seqs = _replace_matches(prompt, matches, mm_items)

    return flatten_2d_lists(token_id_seqs)


def replace_text_matches(
    prompt: str,
    matches: Sequence[_PromptReplacementTextMatch],
    mm_items: MultiModalDataItems,
) -> str:
    """Apply :code:`prompt_repls` to :code:`prompt`."""
    if not matches:
        return prompt

    texts = _replace_matches(prompt, matches, mm_items)

    return "".join(texts)


def _iter_modality_placeholders(
    prompt: list[int],
    modality: str,
    modality_repls: Sequence[_BoundPromptReplacement],
    modal_items: list[Any],
) -> Iterable[_PlaceholderInfo]:
    if len(modal_items) == 0:
        return

    prompt_len = len(prompt)
    item_index = 0

    start_idx = 0
    while start_idx < prompt_len:
        found = False

        for repl_info in modality_repls:
            replacement = repl_info.get_replacement(item_index)
            repl_tokens = replacement.token_ids
            repl_len = len(repl_tokens)
            end_idx = start_idx + repl_len

            if repl_len == 0 or end_idx > prompt_len:
                continue

            if prompt[start_idx:end_idx] == repl_tokens:
                yield _PlaceholderInfo(
                    modality=modality,
                    start_idx=start_idx,
                    replacement=repl_tokens,
                )

                item_index += 1
                if item_index >= len(modal_items):
                    return

                # Exclude overlapping matches
                start_idx = end_idx
                found = True
                break

        if not found:
            start_idx += 1


def iter_placeholders(
    prompt_repls: Sequence[_BoundPromptReplacement],
    prompt: list[int],
    mm_items: MultiModalDataItems,
) -> Iterable[_PlaceholderInfo]:
    """
    Yield each set of placeholder tokens found in :code:`prompt`.

    Note that empty matches are ignored.
    """
    repls_by_modality = dict(full_groupby_modality(prompt_repls))

    for modality, modal_items in mm_items.items():
        if modality in repls_by_modality:
            yield from _iter_modality_placeholders(
                prompt,
                modality,
                repls_by_modality[modality],
                modal_items,
            )


class ProcessorInputs(NamedTuple):
    """Keyword arguments to :meth:`BaseMultiModalProcessor`"""
    prompt_text: str
    mm_data: MultiModalDataDict
    mm_processor_kwargs: Mapping[str, object]


class BaseMultiModalProcessor(ABC):
    """
    Abstract base class to process multi-modal inputs to be used in vLLM.
    """

    def __init__(self, ctx: InputProcessingContext) -> None:
        super().__init__()

        self.ctx = ctx

    def __call__(
        self,
        prompt: str,
        mm_data: MultiModalDataDict,
        mm_processor_kwargs: Mapping[str, object],
    ) -> MultiModalInputsV2:
        return self.apply(prompt, mm_data, mm_processor_kwargs)

    def _get_hf_processor(self) -> ProcessorMixin:
        """
        Subclasses can add keyword arguments to this method to accept
        additional kwargs from model config or user inputs.
        """
        return self.ctx.get_hf_processor()

    def _get_tokenizer(self) -> AnyTokenizer:
        return self.ctx.tokenizer

    @abstractmethod
    def _get_prompt_replacements(
        self,
        mm_items: MultiModalDataItems,
        hf_inputs: BatchFeature,
        mm_processor_kwargs: Mapping[str, object],
    ) -> list[PromptReplacement]:
        """
        Given the original multi-modal items for this modality
        and HF-processed data, output the replacements to perform.

        Note:
            Even when the HF processor already performs replacement for us,
            we still use this replacement information to determine
            the placeholder token positions for each multi-modal item.
        """
        raise NotImplementedError

    def _find_placeholders(
        self,
        all_prompt_repls: Sequence[_BoundPromptReplacement],
        new_token_ids: list[int],
        mm_items: MultiModalDataItems,
    ) -> list[_PlaceholderInfo]:
        return list(
            iter_placeholders(all_prompt_repls, new_token_ids, mm_items))

    def _apply_hf_processor(
        self,
        prompt: str,
        mm_data: MultiModalDataDict,
        mm_processor_kwargs: Mapping[str, object],
    ) -> BatchFeature:
        hf_processor = self._get_hf_processor(**mm_processor_kwargs)

        processor_data = dict[str, Any]()
        passthrough_data = dict[str, Any]()
        for k, v in mm_data.items():
            # TODO: Make a separate modality for embedding inputs
            # to avoid confusion
            if k in ("image", "video", "audio"):
                if isinstance(v, torch.Tensor) and v.ndim == 3:
                    # Pass through embedding inputs (single)
                    passthrough_data[f"{k}_embeds"] = [v]
                elif is_list_of(v, torch.Tensor) and v[0].ndim == 2:
                    # Pass through embedding inputs (multi)
                    passthrough_data[f"{k}_embeds"] = v
                else:
                    # Map keys to plural form, e.g.: image -> images
                    processor_data[f"{k}s"] = v
            else:
                processor_data[k] = v

        assert callable(hf_processor)
        mm_processor_kwargs = self.ctx.resolve_hf_processor_call_kwargs(
            hf_processor,
            mm_processor_kwargs,
        )

        try:
            hf_inputs = hf_processor(
                text=prompt,  # type: ignore
                **processor_data,
                **mm_processor_kwargs,
                return_tensors="pt",
            )
        except Exception as exc:
            data = dict(text=prompt, **processor_data)

            raise RuntimeError(
                f"Failed to apply {type(hf_processor).__name__} "
                f"on data={data} with kwargs={mm_processor_kwargs}") from exc

        hf_inputs.update(passthrough_data)

        return hf_inputs

    def _bind_prompt_replacements(
        self,
        prompt_repls: list[PromptReplacement],
    ) -> list[_BoundPromptReplacement]:
        tokenizer = self._get_tokenizer()

        return [prompt_repl.bind(tokenizer) for prompt_repl in prompt_repls]

    def _apply_prompt_replacements(
        self,
        mm_items: MultiModalDataItems,
        hf_inputs: BatchFeature,
        token_ids: list[int],
        prompt_repls: Sequence[_BoundPromptReplacement],
    ) -> tuple[list[int], str, list[_PlaceholderInfo]]:
        tokenizer = self._get_tokenizer()

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
            len(matches) >= len(mm_items[modality])
            for modality, matches in full_groupby_modality(token_matches)
        ):  # yapf: disable
            token_ids = replace_token_matches(
                token_ids,
                token_matches,
                mm_items,
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
            )

            token_ids = _encode(tokenizer, text)
            matched_repls = [match.prompt_repl for match in text_matches]

        placeholders = self._find_placeholders(matched_repls, token_ids,
                                               mm_items)

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
        tokenizer = self._get_tokenizer()

        hf_inputs = self._apply_hf_processor(prompt_text, mm_data,
                                             mm_processor_kwargs)
        prompt_ids, = hf_inputs.pop("input_ids").tolist()
        mm_kwargs = MultiModalKwargs(hf_inputs)

        mm_items = to_multi_format(mm_data)
        prompt_repls = self._get_prompt_replacements(mm_items, hf_inputs,
                                                     mm_processor_kwargs)
        all_prompt_repls = self._bind_prompt_replacements(prompt_repls)

        # If HF processor already inserts placeholder tokens,
        # there is no need for us to insert them
        all_placeholders = self._find_placeholders(all_prompt_repls,
                                                   prompt_ids, mm_items)

        if all_placeholders:
            prompt_text = _decode(tokenizer, prompt_ids)
        else:
            (
                prompt_ids,
                prompt_text,
                all_placeholders,
            ) = self._apply_prompt_replacements(
                mm_items,
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

    @abstractmethod
    def _get_dummy_mm_inputs(
        self,
        mm_counts: Mapping[str, int],
    ) -> ProcessorInputs:
        """
        Build the multi-modal portion of the input which, after processing,
        results in `mm_max_tokens` in :meth:`get_dummy_data`.
        """
        raise NotImplementedError

    def get_dummy_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        mm_max_tokens: Mapping[str, int],
    ) -> DummyData:
        # Avoid circular import
        from vllm.sequence import SequenceData

        processor_inputs = self._get_dummy_mm_inputs(mm_counts)
        mm_inputs = self.apply(*processor_inputs)

        prompt_token_ids = mm_inputs["prompt_token_ids"]
        placeholders_by_modality = mm_inputs["mm_placeholders"]

        total_placeholders_by_modality = dict[str, int]()
        for modality, placeholders in placeholders_by_modality.items():
            num_placeholders = sum(item["length"] for item in placeholders)
            max_tokens = mm_max_tokens[modality]

            if num_placeholders != max_tokens:
                logger.warning(
                    "The processed dummy data has a total of %d placeholder "
                    "tokens for the '%s' modality, which is not the expected "
                    "%d tokens.", num_placeholders, modality, max_tokens)

            total_placeholders_by_modality[modality] = num_placeholders

        total_len = len(prompt_token_ids)
        if total_len > seq_len:
            logger.warning(
                "The context length (%d) of the model is too short "
                "to hold the multi-modal embeddings in the worst case "
                "(%d tokens in total, out of which %s are reserved for "
                "multi-modal embeddings). This may cause certain multi-modal "
                "inputs to fail during inference, even when the input text is "
                "short. To avoid this, you should increase `max_model_len`, "
                "reduce `max_num_seqs`, and/or reduce `mm_counts`.", seq_len,
                total_len, total_placeholders_by_modality)

        prompt_token_ids.extend([0] * (seq_len - len(prompt_token_ids)))

        return DummyData(
            seq_data=SequenceData.from_seqs(prompt_token_ids),
            multi_modal_data=mm_inputs["mm_kwargs"],
            multi_modal_placeholders=placeholders_by_modality,
        )
