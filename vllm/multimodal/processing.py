# SPDX-License-Identifier: Apache-2.0

import re
from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import (Callable, Generator, ItemsView, Iterable, Mapping,
                             Sequence)
from dataclasses import dataclass, field
from enum import Enum
from functools import lru_cache
from typing import (TYPE_CHECKING, Generic, NamedTuple, Optional, Protocol,
                    TypeVar, Union, cast)

from transformers import BatchFeature, PretrainedConfig, ProcessorMixin
from typing_extensions import assert_never

from vllm.inputs import InputProcessingContext
from vllm.logger import init_logger
from vllm.transformers_utils.tokenizer import (AnyTokenizer, decode_tokens,
                                               encode_tokens)
from vllm.utils import LRUCache, flatten_2d_lists, full_groupby

from .hasher import MultiModalHasher
from .inputs import (MultiModalDataDict, MultiModalEncDecInputs,
                     MultiModalFieldConfig, MultiModalInputs, MultiModalKwargs,
                     MultiModalKwargsItem, PlaceholderRange)
from .parse import (DictEmbeddingItems, EmbeddingItems, MultiModalDataItems,
                    MultiModalDataParser)

if TYPE_CHECKING:
    from .profiling import BaseDummyInputsBuilder

logger = init_logger(__name__)

_S = TypeVar("_S", str, list[int])

PromptSeq = Union[str, list[int]]
"""A token sequence (list of token IDs) or text."""


@dataclass
class PromptIndex:
    """Resolves to an index in the prompt."""
    get_match_index: Callable[[AnyTokenizer, PromptSeq], Optional[int]]


class PromptIndexTargets:

    @staticmethod
    def start() -> PromptIndex:
        """
        Resolves to the start of the prompt (before the first token).

        This results in a match even if the prompt is empty.
        """
        return PromptIndex(lambda tok, prompt: 0)

    @staticmethod
    def prefix(seq: PromptSeq) -> PromptIndex:
        """
        Resolves to a location in the prompt after the given prefix.
        """

        def get_match_index(
            tokenizer: AnyTokenizer,
            prompt: PromptSeq,
        ) -> Optional[int]:
            prefix = seq

            if isinstance(prompt, str):
                if not isinstance(prefix, str):
                    # Make both `str`
                    prefix = decode_tokens(tokenizer, prefix)
            else:
                if isinstance(prefix, str):
                    # Make both `list[int]`
                    prefix = encode_tokens(tokenizer, prefix)

            match_idx = len(prefix)
            return match_idx if prompt[:match_idx] == prefix else None

        return PromptIndex(get_match_index)

    @staticmethod
    def end() -> PromptIndex:
        """
        Resolves to the end of the prompt (after the last token).

        This results in a match even if the prompt is empty.
        """
        return PromptIndex(lambda tok, prompt: len(prompt))


PromptTarget = Union[PromptSeq, PromptIndex]
"""
The token sequence or text to update.
"""


@dataclass
class PromptUpdateDetails:
    """Details about the token sequence or text that are part of the update."""

    full: PromptSeq
    """The full content."""

    features: PromptSeq
    """
    The part of the content that corresponds to feature placeholders;
    this will be replaced by the output of the vision encoder during model
    inference.
    """

    @staticmethod
    def from_seq(seq: PromptSeq) -> "PromptUpdateDetails":
        return PromptUpdateDetails(full=seq, features=seq)


PromptUpdateInfo = Union[PromptSeq, PromptUpdateDetails]
"""
The token sequence or text that are part of the update.

If only part of the content corresponds to feature placeholders, you can
use :class:`PromptUpdateDetails` to specify which part.
"""

PromptUpdateContent = Union[Callable[[int], PromptUpdateInfo],
                            PromptUpdateInfo]
"""
Given the index of the processed item within :attr:`modality`,
output the corresponding token sequence (or text).

For convenience, you can directly pass in the token sequence (or text)
instead of a function if it does not depend on the input.
"""


class UpdateMode(str, Enum):
    INSERT = "insert"
    REPLACE = "replace"


@dataclass
class PromptUpdate(ABC):
    """
    Defines how to update a prompt with placeholder tokens.
    """

    modality: str
    """The modality for which the update is made."""

    target: PromptTarget
    """The token sequence (or text) to update."""

    @property
    @abstractmethod
    def content(self) -> PromptUpdateContent:
        """The placeholder tokens that are part of the update."""
        raise NotImplementedError

    @property
    @abstractmethod
    def mode(self) -> UpdateMode:
        """Defines how to update the prompt."""
        raise NotImplementedError

    def bind(self, tokenizer: AnyTokenizer) -> "BoundPromptUpdate":
        return BoundPromptUpdate(
            _origin=self,
            tokenizer=tokenizer,
        )


@dataclass
class PromptInsertion(PromptUpdate):
    """
    Defines how to insert placeholder tokens into a prompt.

    Example:

        For each image, insert a number of ``<image>`` feature placeholders
        equal to the feature size of the vision encoder after the ``<s>`` token:

        .. code-block:: python

            PromptInsertion(
                modality="image",
                target="<s>",
                insertion="<image>" * image_feature_size,
            )

        Insert these tokens at the start of the prompt:

        .. code-block:: python

            PromptInsertion(
                modality="image",
                target=PromptIndexTargets.start(),
                insertion="<image>" * image_feature_size,
            )

        Insert these tokens after a prefix ``Images:``:

        .. code-block:: python

            PromptInsertion(
                modality="image",
                target=PromptIndexTargets.prefix("Images:"),
                insertion="<image>" * image_feature_size,
            )

        Insert these tokens at the end of the prompt:

        .. code-block:: python

            PromptInsertion(
                modality="image",
                target=PromptIndexTargets.end(),
                insertion="<image>" * image_feature_size,
            )
    """

    insertion: PromptUpdateContent = field(repr=False)
    """
    Given the index of the processed item within :attr:`modality`,
    output the token sequence (or text) to insert right after :attr:`target`.

    For convenience, you can directly pass in the token sequence (or text)
    instead of a function if it does not depend on the input.
    """

    @property
    def content(self) -> PromptUpdateContent:
        return self.insertion

    @property
    def mode(self) -> UpdateMode:
        return UpdateMode.INSERT


@dataclass
class PromptReplacement(PromptUpdate):
    """
    Defines how to replace portions of an input prompt with placeholder tokens.

    Example:

        For each image, replace one ``<image>`` input placeholder in the prompt
        with a number of ``<image>`` feature placeholders
        equal to the feature size of the vision encoder:

        .. code-block:: python

            PromptReplacement(
                modality="image",
                target="<image>",
                replacement="<image>" * image_feature_size,
            )

        As above, but further pad the feature placeholders with ``<image_bos>``
        and `<image_eos>``, which are not supposed to be passed to the vision
        encoder:

        .. code-block:: python

            PromptReplacement(
                modality="image",
                target="<image>",
                replacement=PromptUpdateDetails(
                    full="".join([
                        "<image_bos>",
                        "<image>" * image_feature_size,
                        "<image_eos>",
                    ]),
                    features="<image>" * image_feature_size,
                ),
            )

        To avoid unnecessary tokenization during prompt replacement,
        we recommended passing token sequences instead of text:

        .. code-block:: python

            PromptReplacement(
                modality="image",
                target=[image_token_id],
                replacement=PromptUpdateDetails(
                    full=([image_bos_id] + [image_token_id] * image_feature_size
                          + [image_eos_id]),
                    features=[image_token_id] * image_feature_size,
                ),
            )
    """

    replacement: PromptUpdateContent = field(repr=False)
    """
    Given the index of the processed item within :attr:`modality`,
    output the token sequence (or text) to replace :attr:`target`.

    For convenience, you can directly pass in the token sequence (or text)
    instead of a function if it does not depend on the input.
    """

    @property
    def content(self) -> PromptUpdateContent:
        return self.replacement

    @property
    def mode(self) -> UpdateMode:
        return UpdateMode.REPLACE


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
    """
    A :data:`_PromptSeq` bound to a tokenizer to automatically
    convert between token sequence and text representations.
    """
    tokenizer: AnyTokenizer = field(repr=False)

    _text: Optional[str]
    _token_ids: Optional[list[int]]

    @staticmethod
    def from_seq(
        tokenizer: AnyTokenizer,
        seq: PromptSeq,
    ) -> "_BoundPromptSequence":
        return _BoundPromptSequence(
            tokenizer=tokenizer,
            _text=seq if isinstance(seq, str) else None,
            _token_ids=seq if isinstance(seq, list) else None,
        )

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
class _BoundPromptContent:
    full: _BoundPromptSequence
    features: _BoundPromptSequence


@dataclass
class BoundPromptUpdate:
    """
    A :class:`PromptUpdate` bound to a tokenizer to automatically convert
    :attr:`target` and the result of :meth:`get_content` between
    token sequence and text representations.
    """
    _origin: PromptUpdate
    tokenizer: AnyTokenizer = field(repr=False)

    def __post_init__(self) -> None:
        self._content_cache = dict[int, _BoundPromptContent]()

    @property
    def modality(self) -> str:
        return self._origin.modality

    @property
    def target(self) -> Union[_BoundPromptSequence, PromptIndex]:
        """The token sequence (or text) to update."""
        target = self._origin.target

        if isinstance(target, PromptIndex):
            return target

        return _BoundPromptSequence.from_seq(self.tokenizer, target)

    @property
    def content(self) -> PromptUpdateContent:
        """The placeholder tokens that are part of the update."""
        return self._origin.content

    @property
    def mode(self) -> UpdateMode:
        """Defines how to update the prompt."""
        return self._origin.mode

    def get_content(self, item_idx: int) -> _BoundPromptContent:
        """
        Given the index of the processed item within :attr:`modality`,
        output the token sequence (or text) to update.
        """
        content = self.content
        if callable(content):
            cache_key = item_idx
            if cache_key in self._content_cache:
                return self._content_cache[cache_key]

            content = content(item_idx)
        else:
            cache_key = None

        if not isinstance(content, PromptUpdateDetails):
            content = PromptUpdateDetails.from_seq(content)

        bound_full = _BoundPromptSequence.from_seq(self.tokenizer,
                                                   content.full)
        bound_features = _BoundPromptSequence.from_seq(self.tokenizer,
                                                       content.features)
        bound_content = _BoundPromptContent(full=bound_full,
                                            features=bound_features)

        if cache_key is not None:
            self._content_cache[cache_key] = bound_content

        return bound_content


class _TokenMatch(NamedTuple):
    start_idx: int
    end_idx: int


def iter_token_matches(
    token_ids: list[int],
    match_ids: list[int],
) -> Generator[_TokenMatch]:
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
class _PromptTargetMatch(ABC):
    _origin: BoundPromptUpdate

    @property
    def modality(self) -> str:
        return self._origin.modality

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
class _PromptTargetIndexMatch(_PromptTargetMatch):
    match_idx: int

    @property
    def start_idx(self) -> int:
        return self.match_idx

    @property
    def end_idx(self) -> int:
        return self.match_idx


@dataclass(repr=False)
class _PromptTargetTokenMatch(_PromptTargetMatch):
    match: _TokenMatch

    @property
    def start_idx(self) -> int:
        return self.match.start_idx

    @property
    def end_idx(self) -> int:
        return self.match.end_idx


@dataclass(repr=False)
class _PromptTargetTextMatch(_PromptTargetMatch):
    match: re.Match[str]

    @property
    def start_idx(self) -> int:
        return self.match.start()

    @property
    def end_idx(self) -> int:
        return self.match.end()


@dataclass
class PlaceholderFeaturesInfo:
    modality: str
    item_idx: int
    start_idx: int
    tokens: list[int]

    @property
    def length(self) -> int:
        return len(self.tokens)

    def to_range(self) -> PlaceholderRange:
        return PlaceholderRange(
            offset=self.start_idx,
            length=self.length,
        )


def find_token_matches(
    prompt: list[int],
    prompt_updates: Sequence[BoundPromptUpdate],
) -> Sequence[_PromptTargetMatch]:
    """Return each target of :code:`prompt_updates` found in :code:`prompt`."""

    def get_matches(update: BoundPromptUpdate):
        target = update.target

        if isinstance(target, PromptIndex):
            match_idx = target.get_match_index(update.tokenizer, prompt)
            if match_idx is None:
                return []

            return [_PromptTargetIndexMatch(update, match_idx)]

        return [
            _PromptTargetTokenMatch(update, match)
            for match in iter_token_matches(prompt, target.token_ids)
        ]

    return [
        match for update in prompt_updates for match in get_matches(update)
    ]


def find_text_matches(
    prompt: str,
    prompt_updates: Sequence[BoundPromptUpdate],
) -> Sequence[_PromptTargetMatch]:
    """Return each target of :code:`prompt_updates` found in :code:`prompt`."""

    def get_matches(update: BoundPromptUpdate):
        target = update.target

        if isinstance(target, PromptIndex):
            match_idx = target.get_match_index(update.tokenizer, prompt)
            if match_idx is None:
                return []

            return [_PromptTargetIndexMatch(update, match_idx)]

        return [
            _PromptTargetTextMatch(update, match)
            for match in re.finditer(re.escape(target.text), prompt)
        ]

    return [
        match for update in prompt_updates for match in get_matches(update)
    ]


def _resolve_matches(
    prompt: PromptSeq,
    mm_matches: Mapping[str, Sequence[_PromptTargetMatch]],
) -> list[_PromptTargetMatch]:
    """
    Resolve :code:`mm_matches` to ensure that there are no overlapping matches,
    and sort them such that earlier matches take priority over later ones.
    """
    matches = [m for matches in mm_matches.values() for m in matches]

    seen_matches: list[Optional[_PromptTargetMatch]] = [None] * len(prompt)

    for match in matches:
        for idx in range(match.start_idx, match.end_idx):
            if seen_matches[idx] is not None:
                raise ValueError("Found overlapping matches "
                                 f"({seen_matches[idx]} and {match}) "
                                 f"at index={idx} of prompt={prompt}")

            seen_matches[idx] = match

    return sorted(matches, key=lambda x: x.start_idx)


def _apply_matches(
    prompt: _S,
    mm_matches: Mapping[str, Sequence[_PromptTargetMatch]],
    mm_item_counts: Mapping[str, int],
) -> list[_S]:
    """Apply the updates in :code:`mm_matches` to :code:`prompt`."""
    out_seqs = list[Union[str, list[int]]]()
    prev_end_idx = 0
    next_idx_by_modality = defaultdict[str, int](lambda: 0)

    for match in _resolve_matches(prompt, mm_matches):
        modality = match.modality

        item_start_idx = next_idx_by_modality[modality]
        max_item_count = mm_item_counts.get(modality, 0)
        if item_start_idx >= max_item_count:
            continue

        start_idx = match.start_idx
        end_idx = match.end_idx
        origin = match._origin
        mode = origin.mode

        if mode == UpdateMode.INSERT:
            out_seqs.append(prompt[prev_end_idx:end_idx])
            num_inserts = max_item_count
        elif mode == UpdateMode.REPLACE:
            out_seqs.append(prompt[prev_end_idx:start_idx])
            num_inserts = max_item_count if start_idx == end_idx else 1
        else:
            assert_never(mode)

        item_end_idx = min(item_start_idx + num_inserts, max_item_count)

        for item_idx in range(item_start_idx, item_end_idx):
            content = origin.get_content(item_idx)
            insert_seq = (content.full.text if isinstance(prompt, str) else
                          content.full.token_ids)

            out_seqs.append(insert_seq)

        prev_end_idx = end_idx
        next_idx_by_modality[modality] += item_end_idx - item_start_idx

    out_seqs.append(prompt[prev_end_idx:])

    return cast(list[_S], out_seqs)


def apply_token_matches(
    prompt: list[int],
    mm_matches: Mapping[str, Sequence[_PromptTargetMatch]],
    mm_item_counts: Mapping[str, int],
) -> list[int]:
    """Apply the updates in :code:`mm_matches` to :code:`prompt`."""
    if not mm_matches:
        return prompt

    token_id_seqs = _apply_matches(prompt, mm_matches, mm_item_counts)

    return flatten_2d_lists(token_id_seqs)


def apply_text_matches(
    prompt: str,
    mm_matches: Mapping[str, Sequence[_PromptTargetMatch]],
    mm_item_counts: Mapping[str, int],
) -> str:
    """Apply the updates in :code:`mm_matches` to :code:`prompt`."""
    if not mm_matches:
        return prompt

    texts = _apply_matches(prompt, mm_matches, mm_item_counts)

    return "".join(texts)


def _iter_placeholders(
    mm_prompt_updates: Mapping[str, Sequence[BoundPromptUpdate]],
    prompt: list[int],
    mm_item_counts: Mapping[str, int],
) -> Iterable[PlaceholderFeaturesInfo]:
    """
    Yield each set of placeholder tokens found in :code:`prompt`.

    Matches are exclusive even when multiple modalities share
    the same placeholder tokens. In that case, the modality that
    appears earlier in `mm_prompt_updates` takes priority.

    Note that empty matches are ignored.
    """
    prompt_len = len(prompt)
    item_idx_by_modality = defaultdict[str, int](lambda: 0)

    start_idx = 0
    while start_idx < prompt_len:
        found = False

        for modality, modality_updates in mm_prompt_updates.items():
            item_idx = item_idx_by_modality[modality]
            if item_idx >= mm_item_counts.get(modality, 0):
                continue

            for update_info in modality_updates:
                content = update_info.get_content(item_idx)
                content_tokens_full = content.full.token_ids
                content_len_full = len(content_tokens_full)
                end_idx_full = start_idx + content_len_full

                if content_len_full == 0 or end_idx_full > prompt_len:
                    continue

                if prompt[start_idx:end_idx_full] == content_tokens_full:
                    content_tokens_feat = content.features.token_ids

                    try:
                        match = next(
                            iter_token_matches(content_tokens_full,
                                               content_tokens_feat))
                        yield PlaceholderFeaturesInfo(
                            modality=modality,
                            item_idx=item_idx,
                            start_idx=start_idx + match.start_idx,
                            tokens=content_tokens_feat,
                        )
                    except StopIteration:
                        raise AssertionError(
                            f"{content_tokens_feat=} should be a "
                            f"subsequence of {content_tokens_full=}") from None

                    # Exclude overlapping matches
                    start_idx = end_idx_full
                    item_idx_by_modality[modality] += 1
                    found = True
                    break

            if found:
                break  # Go back to the outer while loop

        if not found:
            start_idx += 1


def find_mm_placeholders(
    mm_prompt_updates: Mapping[str, Sequence[BoundPromptUpdate]],
    prompt: list[int],
    mm_item_counts: Mapping[str, int],
) -> Mapping[str, list[PlaceholderFeaturesInfo]]:
    it = _iter_placeholders(mm_prompt_updates, prompt, mm_item_counts)
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
    """Base class to provide the information necessary for data processing."""

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
    def get_mm_max_tokens_per_item(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
    ) -> Mapping[str, int]:
        """
        Get the maximum possible number of tokens per data item
        for each modality.

        The dictionary returned by this method should have the same
        keys as that returned by :meth:`get_supported_mm_limits`.
        """
        raise NotImplementedError


_I = TypeVar("_I", bound=BaseProcessingInfo)


class BaseMultiModalProcessor(ABC, Generic[_I]):
    """
    Abstract base class to process multi-modal inputs to be used in vLLM.

    Not to be confused with :class:`transformers.ProcessorMixin`.
    """

    def __init__(self,
                 info: _I,
                 dummy_inputs: "BaseDummyInputsBuilder[_I]",
                 *,
                 cache: Optional[ProcessingCache] = None,
                 enable_sanity_checks: bool = True) -> None:
        if get_repls := getattr(self, "_get_prompt_replacements", None):
            logger.warning_once("`_get_prompt_replacements` has been renamed "
                                "to `_get_prompt_updates`. The old name will "
                                "be removed in an upcoming release.")
            self._get_prompt_updates = get_repls  # type: ignore[method-assign]

        super().__init__()

        self.info = info
        self.dummy_inputs = dummy_inputs
        self.cache = cache
        self.enable_sanity_checks = enable_sanity_checks

        self.data_parser = self._get_data_parser()

    def __call__(
        self,
        prompt: str,
        mm_data: MultiModalDataDict,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> MultiModalInputs:
        return self.apply(prompt, mm_data, hf_processor_mm_kwargs)

    def _get_data_parser(self) -> MultiModalDataParser:
        """
        Construct a parser to preprocess multi-modal data items
        before passing them to :meth:`_get_hf_mm_data`.

        You can support additional modalities by creating a subclass
        of :class:`MultiModalDataParser` that has additional subparsers.
        """
        return MultiModalDataParser()

    def _to_mm_items(
        self,
        mm_data: MultiModalDataDict,
    ) -> MultiModalDataItems:
        """
        Normalize :class:`MultiModalDataDict` to :class:`MultiModalDataItems`
        before passing them to :meth:`_get_hf_mm_data`.
        """
        mm_items = self.data_parser.parse_mm_data(mm_data)

        mm_limits = self.info.ctx.get_mm_config().limit_per_prompt
        for modality, items in mm_items.items():
            limit = mm_limits.get(modality, 1)
            if len(items) > limit:
                raise ValueError(
                    f"You set {modality}={limit} (or defaulted to 1) in "
                    f"`--limit-mm-per-prompt`, but passed {len(items)} "
                    f"{modality} items in the same prompt.")

        return mm_items

    @abstractmethod
    def _get_mm_fields_config(
        self,
        hf_inputs: BatchFeature,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        """Given the HF-processed data, output the metadata of each field."""
        raise NotImplementedError

    @abstractmethod
    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        out_mm_kwargs: MultiModalKwargs,
    ) -> list[PromptUpdate]:
        """
        Given the original multi-modal items for this modality
        and HF-processed data, output the updates to perform.

        Notes:
            - You should not assume that HF processor always performs prompt
              updates: in :meth:`_apply_hf_processor_missing`, this method
              is called on text-only and multimodal-only inputs separately,
              instead of passing them in the same call.
            - The update information returned by this method is also used to
              determine the placeholder token positions for each multi-modal
              item.
        """
        raise NotImplementedError

    def _find_mm_placeholders(
        self,
        mm_prompt_updates: Mapping[str, Sequence[BoundPromptUpdate]],
        new_token_ids: list[int],
        mm_item_counts: Mapping[str, int],
    ) -> Mapping[str, list[PlaceholderFeaturesInfo]]:
        return find_mm_placeholders(mm_prompt_updates, new_token_ids,
                                    mm_item_counts)

    def _get_hf_mm_data(
        self,
        mm_items: MultiModalDataItems,
    ) -> tuple[Mapping[str, object], Mapping[str, object]]:
        processor_data = dict[str, object]()
        passthrough_data = dict[str, object]()

        for items in mm_items.values():
            processor_data.update(items.get_processor_data())
            passthrough_data.update(items.get_passthrough_data())

        return processor_data, passthrough_data

    def _call_hf_processor(
        self,
        prompt: str,
        # Not to be confused with `mm_data` in `self.apply`.
        # This refers to the data to be passed to HF processor.
        mm_data: Mapping[str, object],
        mm_kwargs: Mapping[str, object],
    ) -> BatchFeature:
        """
        Call the HF processor on the prompt text and
        associated multi-modal data.
        """
        return self.info.ctx.call_hf_processor(
            self.info.get_hf_processor(**mm_kwargs),
            dict(text=prompt, **mm_data),
            mm_kwargs,
        )

    def _hf_processor_applies_updates(
        self,
        prompt_text: str,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> bool:
        """
        Return whether the HF processor applies prompt updates.

        For most HF processors, this should be :code:`True` when multi-modal
        data items are passed, but :code:`False` when multi-modal embeddings
        are passed.
        """
        return not any(
            isinstance(items, (EmbeddingItems, DictEmbeddingItems))
            for items in mm_items.values())

    def _apply_hf_processor_text_mm(
        self,
        prompt_text: str,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> tuple[list[int], MultiModalKwargs, bool]:
        """
        Apply the HF processor on the prompt text and multi-modal data
        together.

        In addition, return whether prompt updates have been applied.
        """
        processor_data, passthrough_data = self._get_hf_mm_data(mm_items)

        processed_data = self._call_hf_processor(
            prompt=prompt_text,
            mm_data=processor_data,
            mm_kwargs=hf_processor_mm_kwargs,
        )
        processed_data.update(passthrough_data)

        prompt_ids, = processed_data.pop("input_ids").tolist()

        mm_kwargs = MultiModalKwargs.from_hf_inputs(
            processed_data,
            self._get_mm_fields_config(processed_data, hf_processor_mm_kwargs),
        )

        is_update_applied = self._hf_processor_applies_updates(
            prompt_text=prompt_text,
            mm_items=mm_items,
            hf_processor_mm_kwargs=hf_processor_mm_kwargs,
        )

        return prompt_ids, mm_kwargs, is_update_applied

    def _apply_hf_processor_text_only(self, prompt_text: str) -> list[int]:
        """
        Apply the HF processor on the prompt text only.

        Since HF processor requires that text and multi-modal items
        correspond to each other, we create dummy multi-modal items
        to go along with the text.
        """
        prompt_ids, _, _ = self._apply_hf_processor_text_mm(
            prompt_text=prompt_text,
            mm_items=MultiModalDataItems({}),
            hf_processor_mm_kwargs={},
        )

        return prompt_ids

    def _apply_hf_processor_tokens_only(
        self,
        prompt_tokens: list[int],
    ) -> list[int]:
        """
        Apply the HF processor on the prompt tokens only.

        Most HF processors accept prompt text but not prompt tokens.
        If the HF processor adds or removes tokens that are not related to
        multi-modal data, you should override this method so it is consistent
        with the output of :meth:`_apply_hf_processor_text_only` on the
        corresponding text.
        """
        return prompt_tokens

    def _apply_hf_processor_mm_only(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> MultiModalKwargs:
        """
        Apply the HF processor on the multi-modal data only.

        Since HF processor requires that text and multi-modal items
        correspond to each other, we generate dummy text using
        :class:`DummyInputsBuilder` to go along with the multi-modal data.
        """
        mm_counts = mm_items.get_all_counts()

        dummy_inputs = self.dummy_inputs.get_dummy_processor_inputs(
            self.info.ctx.model_config.max_model_len,
            mm_counts,
        )

        _, mm_kwargs, _ = self._apply_hf_processor_text_mm(
            prompt_text=dummy_inputs.prompt_text,
            mm_items=mm_items,
            hf_processor_mm_kwargs=hf_processor_mm_kwargs,
        )

        return mm_kwargs

    def _apply_hf_processor_main(
        self,
        prompt: Union[str, list[int]],
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        *,
        enable_hf_prompt_update: bool,
    ) -> tuple[list[int], MultiModalKwargs, bool]:
        """
        Apply the HF processor on the prompt text and multi-modal data.

        In addition, return whether prompt updates have been applied
        (for most HF processors, this should be :code:`True`).

        Note:
            If :code:`enable_hf_prompt_update=False`, we use HF processor
            to perform prompt updates if available; HF processor requires
            that the prompt corresponds to multi-modal items.
        """
        if isinstance(prompt, str):
            if enable_hf_prompt_update:
                return self._apply_hf_processor_text_mm(
                    prompt_text=prompt,
                    mm_items=mm_items,
                    hf_processor_mm_kwargs=hf_processor_mm_kwargs,
                )

            prompt_ids = self._apply_hf_processor_text_only(prompt)
        else:
            prompt_ids = self._apply_hf_processor_tokens_only(prompt)

        mm_kwargs = self._apply_hf_processor_mm_only(
            mm_items=mm_items,
            hf_processor_mm_kwargs=hf_processor_mm_kwargs,
        )

        return prompt_ids, mm_kwargs, False

    def _cached_apply_hf_processor(
        self,
        prompt: Union[str, list[int]],
        mm_data_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> tuple[list[int], MultiModalKwargs, bool]:
        """
        Apply the HF processor on the full prompt text,
        caching the results and reusing cached results.
        """
        cache = self.cache
        model_id = self.info.model_id

        _, passthrough_data = self._get_hf_mm_data(mm_data_items)
        if cache is None or passthrough_data:
            return self._apply_hf_processor_main(
                prompt=prompt,
                mm_items=mm_data_items,
                hf_processor_mm_kwargs=hf_processor_mm_kwargs,
                enable_hf_prompt_update=True,
            )

        mm_maybe_cached_kw_items = {
            modality: [
                cache.get(model_id, modality, item, hf_processor_mm_kwargs)
                for item in items
            ]
            for modality, items in mm_data_items.items()
        }

        mm_missing_idxs = {
            modality:
            [idx for idx, item in enumerate(kw_items) if item is None]
            for modality, kw_items in mm_maybe_cached_kw_items.items()
        }
        mm_missing_data = {
            modality: [mm_data_items[modality][idx] for idx in idxs]
            for modality, idxs in mm_missing_idxs.items()
        }
        mm_missing_data_items = self._to_mm_items(mm_missing_data)

        # NOTE: `prompt` does not correspond to `mm_missing_data_items`,
        # so we can't apply prompt updates until the new multimodal
        # items are combined with the cached multimodal items
        (
            prompt_ids,
            mm_missing_kwargs,
            is_update_applied,
        ) = self._apply_hf_processor_main(
            prompt=prompt,
            mm_items=mm_missing_data_items,
            hf_processor_mm_kwargs=hf_processor_mm_kwargs,
            enable_hf_prompt_update=False,
        )

        mm_missing_next_idx = {
            modality: 0
            for modality in mm_missing_data_items
        }

        merged_kw_items = list[MultiModalKwargsItem]()
        for modality, kw_items in mm_maybe_cached_kw_items.items():
            for idx, kw_item in enumerate(kw_items):
                if kw_item is None:
                    kw_item = mm_missing_kwargs.get_item(
                        modality,
                        mm_missing_next_idx[modality],
                    )

                    cache.put(
                        model_id,
                        modality,
                        mm_data_items[modality][idx],
                        hf_processor_mm_kwargs,
                        kw_item,
                    )

                    mm_missing_next_idx[modality] += 1

                merged_kw_items.append(kw_item)

        if self.enable_sanity_checks:
            mm_missing_counts = mm_missing_data_items.get_all_counts()
            assert all(
                item_count == mm_missing_counts[modality]
                for modality, item_count in mm_missing_next_idx.items()), dict(
                    mm_missing_next_idx=mm_missing_next_idx,
                    mm_missing_counts=mm_missing_counts)

        mm_kwargs = MultiModalKwargs.from_items(merged_kw_items)

        return prompt_ids, mm_kwargs, is_update_applied

    def _bind_and_group_updates(
        self,
        prompt_updates: list[PromptUpdate],
    ) -> dict[str, list[BoundPromptUpdate]]:
        tokenizer = self.info.get_tokenizer()

        it = (update.bind(tokenizer) for update in prompt_updates)
        return dict(full_groupby_modality(it))

    def _apply_prompt_updates(
        self,
        token_ids: list[int],
        mm_prompt_updates: Mapping[str, Sequence[BoundPromptUpdate]],
        mm_item_counts: Mapping[str, int],
    ) -> tuple[list[int], str, Mapping[str, list[PlaceholderFeaturesInfo]]]:
        tokenizer = self.info.get_tokenizer()

        mm_token_matches = {
            modality: find_token_matches(token_ids, updates)
            for modality, updates in mm_prompt_updates.items()
        }
        mm_match_counts = {
            modality: len(matches)
            for modality, matches in mm_token_matches.items()
        }

        # If the search text does not represent a special token,
        # it may have different token IDs in the prompt, because
        # the tokens may go across the boundaries of the search text.
        # ----
        # e.g. when searching for "foo" in "food", if "food" itself makes
        # up a token, then the token ID of "foo" will not appear at all
        # ----
        # Since it is inefficient to search for all possible tokenizations
        # of the search text in the prompt, we instead perform string-based
        # updates on the decoded token IDs, then encode them back.
        if all(
            mm_match_counts.get(modality, 0) >= item_count
            for modality, item_count in mm_item_counts.items()
        ):  # yapf: disable
            token_ids = apply_token_matches(
                token_ids,
                mm_token_matches,
                mm_item_counts,
            )

            text = decode_tokens(tokenizer, token_ids)
            matched_updates = {
                modality: [match._origin for match in token_matches]
                for modality, token_matches in mm_token_matches.items()
            }
        else:
            text = decode_tokens(tokenizer, token_ids)

            mm_text_matches = {
                modality: find_text_matches(text, updates)
                for modality, updates in mm_prompt_updates.items()
            }
            text = apply_text_matches(
                text,
                mm_text_matches,
                mm_item_counts,
            )

            token_ids = encode_tokens(tokenizer,
                                      text,
                                      add_special_tokens=False)
            matched_updates = {
                modality: [match._origin for match in token_matches]
                for modality, token_matches in mm_text_matches.items()
            }

        placeholders = self._find_mm_placeholders(
            matched_updates,
            token_ids,
            mm_item_counts,
        )

        return token_ids, text, placeholders

    def _validate_mm_kwargs(
        self,
        mm_kwargs: MultiModalKwargs,
        mm_item_counts: Mapping[str, int],
    ) -> None:
        for modality, item_count in mm_item_counts.items():
            if modality in mm_kwargs.modalities:
                items = mm_kwargs.get_items(modality)
            else:
                items = []

            if len(items) != item_count:
                raise RuntimeError(
                    f"Expected there to be {item_count} {modality} items in "
                    f"keyword arguments corresponding to {item_count} "
                    f"{modality} data items, but only found {len(items)}! "
                    "There is likely a problem with your "
                    "implementation of merged multi-modal processor for this "
                    "model (usually arising from an inconsistency between "
                    "`_call_hf_processor` and `_get_mm_fields_config`).")

    def _validate_mm_placeholders(
        self,
        mm_placeholders: Mapping[str, list[PlaceholderFeaturesInfo]],
        mm_item_counts: Mapping[str, int],
    ) -> None:
        for modality, item_count in mm_item_counts.items():
            placeholders = mm_placeholders.get(modality, [])

            if len(placeholders) != item_count:
                raise RuntimeError(
                    f"Expected there to be {item_count} prompt updates "
                    f"corresponding to {item_count} {modality} items, but "
                    f"instead found {len(placeholders)} prompt updates! "
                    "Either the prompt text has missing/incorrect tokens for "
                    "multi-modal inputs, or there is a problem with your "
                    "implementation of merged multi-modal processor for this "
                    "model (usually arising from an inconsistency between "
                    "`_call_hf_processor` and `_get_prompt_updates`).")

    def apply(
        self,
        prompt: Union[str, list[int]],
        mm_data: MultiModalDataDict,
        hf_processor_mm_kwargs: Mapping[str, object],
        return_mm_hashes: bool = False,
    ) -> MultiModalInputs:
        """
        Process multi-modal inputs to be used in vLLM.

        The main steps are:

        1. Apply HF Processor on prompt text and multi-modal data together,
           outputting token IDs and processed tensors.
        2. Find and update sequences in the token IDs with placeholder tokens.
           The number of placeholder tokens equals the feature size of the
           multi-modal data outputted by the multi-modal encoder.
        3. Extract information about the placeholder tokens from the
           processed token IDs.
        """
        mm_items = self._to_mm_items(mm_data)

        # Create MM hashes to be returned (only used in V1)
        # TODO: Use these hash keys for caching operations in apply_hf_processor
        # instead of rehashing.

        if return_mm_hashes:
            model_id = self.info.model_id
            mm_hashes = {
                modality: [
                    MultiModalHasher.hash_kwargs(model_id=model_id,
                                                 **{modality: item},
                                                 **hf_processor_mm_kwargs)
                    for item in items
                ]
                for modality, items in mm_items.items()
            }
        else:
            mm_hashes = None

        (
            prompt_ids,
            mm_kwargs,
            is_update_applied,
        ) = self._cached_apply_hf_processor(
            prompt,
            mm_items,
            hf_processor_mm_kwargs,
        )

        unbound_prompt_updates = self._get_prompt_updates(
            mm_items,
            hf_processor_mm_kwargs,
            mm_kwargs,
        )
        mm_prompt_updates = self._bind_and_group_updates(
            unbound_prompt_updates)

        mm_item_counts = mm_items.get_all_counts()
        self._validate_mm_kwargs(mm_kwargs, mm_item_counts)

        if is_update_applied:
            mm_placeholders = self._find_mm_placeholders(
                mm_prompt_updates,
                prompt_ids,
                mm_item_counts,
            )
            self._validate_mm_placeholders(mm_placeholders, mm_item_counts)

            tokenizer = self.info.get_tokenizer()
            prompt = decode_tokens(tokenizer, prompt_ids)
        else:
            (
                prompt_ids,
                prompt,
                mm_placeholders,
            ) = self._apply_prompt_updates(
                prompt_ids,
                mm_prompt_updates,
                mm_item_counts,
            )
            self._validate_mm_placeholders(mm_placeholders, mm_item_counts)

        mm_placeholder_ranges = {
            modality: [item.to_range() for item in placeholders]
            for modality, placeholders in mm_placeholders.items()
        }

        return MultiModalInputs(
            type="multimodal",
            prompt=prompt,
            prompt_token_ids=prompt_ids,
            mm_kwargs=mm_kwargs,
            mm_hashes=mm_hashes,
            mm_placeholders=mm_placeholder_ranges,
        )


class EncDecMultiModalProcessor(BaseMultiModalProcessor[_I]):

    @abstractmethod
    def create_encoder_prompt(
        self,
        prompt: Union[str, list[int]],
        mm_data: MultiModalDataDict,
    ) -> Union[str, list[int]]:
        """
        Create input prompt for the encoder. HF processor will be applied on 
        this prompt during profiling and generation.
        """
        raise NotImplementedError

    def create_decoder_prompt(
        self,
        prompt: Union[str, list[int]],
        mm_data: MultiModalDataDict,
    ) -> Union[str, list[int]]:
        """Create input prompt for the decoder."""
        return prompt

    def apply(
        self,
        prompt: Union[str, list[int]],
        mm_data: MultiModalDataDict,
        hf_processor_mm_kwargs: Mapping[str, object],
        return_mm_hashes: bool = False,
    ) -> MultiModalEncDecInputs:
        """
        Process multi-modal inputs to be used in vLLM.
        The main processing steps are modified to fit encoder-decoder model:
        1. Create encoder prompt from input prompt text.
        2. Apply the HF processor on encoder prompt.
        3. Copy the input prompt text as decoder prompt inputs.
        """
        encoder_prompt = self.create_encoder_prompt(prompt, mm_data)
        encoder_inputs = super().apply(
            encoder_prompt,
            mm_data,
            hf_processor_mm_kwargs,
            return_mm_hashes,
        )

        tokenizer = self.info.get_tokenizer()
        decoder_prompt = self.create_decoder_prompt(prompt, mm_data)
        if isinstance(decoder_prompt, str):
            decoder_prompt_ids = encode_tokens(tokenizer,
                                               decoder_prompt,
                                               add_special_tokens=False)
        else:
            decoder_prompt_ids = decoder_prompt
            decoder_prompt = decode_tokens(tokenizer, decoder_prompt)

        mm_inputs = MultiModalEncDecInputs(
            encoder_prompt=encoder_inputs["prompt"],
            encoder_prompt_token_ids=encoder_inputs["prompt_token_ids"],
            **encoder_inputs)
        mm_inputs.update({
            "prompt": decoder_prompt,
            "prompt_token_ids": decoder_prompt_ids
        })
        return mm_inputs
