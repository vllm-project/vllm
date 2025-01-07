import re
from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import Callable, ItemsView, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from functools import lru_cache
from typing import NamedTuple, Optional, Protocol, TypeVar, Union

from transformers import PretrainedConfig, ProcessorMixin

from vllm.inputs import InputProcessingContext
from vllm.logger import init_logger
from vllm.transformers_utils.tokenizer import (AnyTokenizer, decode_tokens,
                                               encode_tokens)
from vllm.utils import LRUCache, flatten_2d_lists, full_groupby

from .hasher import MultiModalHasher
from .inputs import MultiModalKwargsItem, PlaceholderRange

logger = init_logger(__name__)

_S = TypeVar("_S", str, list[int])
_PromptSeq = Union[str, list[int]]


@dataclass
class PromptReplacement:
    modality: str
    """The modality for which the replacement is made."""

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

    def bind(self, tokenizer: AnyTokenizer) -> "BoundPromptReplacement":
        return BoundPromptReplacement(
            tokenizer=tokenizer,
            modality=self.modality,
            _target=self.target,
            _replacement=self.replacement,
        )


@lru_cache(maxsize=2048)
def _cached_encode(
    tokenizer: AnyTokenizer,
    text: str,
    *,
    add_special_tokens: bool = False,
) -> list[int]:
    return encode_tokens(tokenizer,
                         text,
                         add_special_tokens=add_special_tokens)


@lru_cache(maxsize=2048)
def _cached_decode(
    tokenizer: AnyTokenizer,
    token_ids: tuple[int, ...],
    *,
    skip_special_tokens: bool = False,
) -> str:
    return decode_tokens(tokenizer,
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
class BoundPromptReplacement:
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
    prompt_repl: BoundPromptReplacement

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


@dataclass
class PlaceholderInfo:
    modality: str
    item_idx: int
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
    prompt_repls: Sequence[BoundPromptReplacement],
) -> list[_PromptReplacementTokenMatch]:
    """Return each target of :code:`prompt_repls` found in :code:`prompt`."""
    return [
        _PromptReplacementTokenMatch(prompt_repl, match)
        for prompt_repl in prompt_repls
        for match in iter_token_matches(prompt, prompt_repl.target.token_ids)
    ]


def find_text_matches(
    prompt: str,
    prompt_repls: Sequence[BoundPromptReplacement],
) -> list[_PromptReplacementTextMatch]:
    """Return each target of :code:`prompt_repls` found in :code:`prompt`."""
    return [
        _PromptReplacementTextMatch(prompt_repl, match)
        for prompt_repl in prompt_repls
        for match in re.finditer(re.escape(prompt_repl.target.text), prompt)
    ]


def _resolve_matches(
    prompt: _PromptSeq,
    mm_matches: Mapping[str, Sequence[_PromptReplacementMatch]],
) -> list[_PromptReplacementMatch]:
    """
    Resolve :code:`mm_matches` to ensure that there are no overlapping matches,
    and sort them such that earlier matches take priority over later ones.
    """
    matches = [m for matches in mm_matches.values() for m in matches]

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
    mm_matches: Mapping[str, Sequence[_PromptReplacementMatch]],
    mm_item_counts: Mapping[str, int],
) -> list[_S]:
    """Apply the replacements in :code:`mm_matches` to :code:`prompt`."""
    out_seqs = list[_S]()
    prev_end_idx = 0
    next_idx_by_modality = defaultdict[str, int](lambda: 0)

    for match in _resolve_matches(prompt, mm_matches):
        modality = match.modality

        item_idx = next_idx_by_modality[modality]
        if item_idx >= mm_item_counts.get(modality, 0):
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
    mm_matches: Mapping[str, Sequence[_PromptReplacementTokenMatch]],
    mm_item_counts: Mapping[str, int],
) -> list[int]:
    """Apply the replacements in :code:`mm_matches` to :code:`prompt`."""
    if not mm_matches:
        return prompt

    token_id_seqs = _replace_matches(prompt, mm_matches, mm_item_counts)

    return flatten_2d_lists(token_id_seqs)


def replace_text_matches(
    prompt: str,
    mm_matches: Mapping[str, Sequence[_PromptReplacementTextMatch]],
    mm_item_counts: Mapping[str, int],
) -> str:
    """Apply the replacements in :code:`mm_matches` to :code:`prompt`."""
    if not mm_matches:
        return prompt

    texts = _replace_matches(prompt, mm_matches, mm_item_counts)

    return "".join(texts)


def _iter_modality_placeholders(
    prompt: list[int],
    modality: str,
    modality_repls: Sequence[BoundPromptReplacement],
    modal_item_count: int,
) -> Iterable[PlaceholderInfo]:
    if modal_item_count == 0:
        return

    prompt_len = len(prompt)
    item_idx = 0

    start_idx = 0
    while start_idx < prompt_len:
        found = False

        for repl_info in modality_repls:
            replacement = repl_info.get_replacement(item_idx)
            repl_tokens = replacement.token_ids
            repl_len = len(repl_tokens)
            end_idx = start_idx + repl_len

            if repl_len == 0 or end_idx > prompt_len:
                continue

            if prompt[start_idx:end_idx] == repl_tokens:
                yield PlaceholderInfo(
                    modality=modality,
                    item_idx=item_idx,
                    start_idx=start_idx,
                    replacement=repl_tokens,
                )

                item_idx += 1
                if item_idx >= modal_item_count:
                    return

                # Exclude overlapping matches
                start_idx = end_idx
                found = True
                break

        if not found:
            start_idx += 1


def _iter_placeholders(
    mm_prompt_repls: Mapping[str, Sequence[BoundPromptReplacement]],
    prompt: list[int],
    mm_item_counts: Mapping[str, int],
) -> Iterable[PlaceholderInfo]:
    """
    For each modality, yield each set of placeholder tokens found in
    :code:`prompt`.

    Note that empty matches are ignored.
    """
    for modality, modal_item_count in mm_item_counts.items():
        if modality in mm_prompt_repls:
            yield from _iter_modality_placeholders(
                prompt,
                modality,
                mm_prompt_repls[modality],
                modal_item_count,
            )


def find_mm_placeholders(
    mm_prompt_repls: Mapping[str, Sequence[BoundPromptReplacement]],
    prompt: list[int],
    mm_item_counts: Mapping[str, int],
) -> Mapping[str, list[PlaceholderInfo]]:
    it = _iter_placeholders(mm_prompt_repls, prompt, mm_item_counts)
    return dict(full_groupby_modality(it))


class ProcessingCache:

    def __init__(self, capacity: int) -> None:
        super().__init__()

        # DEBUG: Set to None to disable
        self.debug_cache_hit_ratio_steps: Optional[int] = None

        self._cache = LRUCache[str, MultiModalKwargsItem](capacity)

    def _maybe_log_cache_stats(self) -> None:
        steps = self.debug_cache_hit_ratio_steps
        if not steps:
            return

        cache_stats = self._cache.stat()
        if cache_stats.total % steps == 0:
            logger.debug("ProcessingCache: hit_ratio = %.2f",
                         cache_stats.hit_ratio)

    def get(
        self,
        model_id: str,
        modality: str,
        input_item: object,
        input_kwargs: Mapping[str, object],
    ) -> Optional[MultiModalKwargsItem]:
        """
        Get a processed multi-modal item from the cache
        according to its dependencies, including:

        - The model ID
        - The modality of the item
        - The original data item passed to the HF processor
        - The configuration options of the HF processor
        """
        self._maybe_log_cache_stats()

        cache_key = MultiModalHasher.hash_kwargs(model_id=model_id,
                                                 **{modality: input_item},
                                                 **input_kwargs)
        return self._cache.get(cache_key)

    def put(
        self,
        model_id: str,
        modality: str,
        input_item: object,
        input_kwargs: Mapping[str, object],
        output_kwargs: MultiModalKwargsItem,
    ) -> None:
        """
        Put a processed multi-modal item into the cache
        according to its dependencies (see :meth:`get`).
        """
        cache_key = MultiModalHasher.hash_kwargs(model_id=model_id,
                                                 **{modality: input_item},
                                                 **input_kwargs)
        self._cache.put(cache_key, output_kwargs)


class BaseProcessingInfo:
    """Base class containing information to perform processing."""

    def __init__(self, ctx: InputProcessingContext) -> None:
        super().__init__()

        self.ctx = ctx

    @property
    def model_id(self) -> str:
        return self.ctx.model_config.model

    def get_tokenizer(self) -> AnyTokenizer:
        return self.ctx.tokenizer

    def get_hf_config(self) -> PretrainedConfig:
        return self.ctx.get_hf_config()

    def get_hf_processor(self, **kwargs: object) -> ProcessorMixin:
        """
        Subclasses can override this method to handle
        specific kwargs from model config or user inputs.
        """
        return self.ctx.get_hf_processor(**kwargs)

    @abstractmethod
    def get_supported_mm_limits(self) -> Mapping[str, Optional[int]]:
        """
        Return the maximum supported number of items for each modality.

        A value of `None` means unlimited number of items.

        Omitting a modality from the returned dictionary means that
        it is not supported at all.
        """
        raise NotImplementedError

    @abstractmethod
    def get_mm_max_tokens_per_item(self, seq_len: int) -> Mapping[str, int]:
        """
        Get the maximum possible number of tokens per data item
        for each modality.

        The dictionary returned by this method should have the same
        keys as that returned by :meth:`get_supported_mm_limits`.
        """
        raise NotImplementedError
