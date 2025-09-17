# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import (Callable, Generator, ItemsView, Iterable, Mapping,
                             Sequence)
from dataclasses import dataclass, field, replace
from enum import Enum
from functools import lru_cache
from typing import (TYPE_CHECKING, Generic, NamedTuple, Optional, Protocol,
                    TypeVar, Union, cast)

import regex as re
import torch
from typing_extensions import assert_never

from vllm.inputs import InputProcessingContext
from vllm.logger import init_logger
from vllm.transformers_utils.tokenizer import (AnyTokenizer, decode_tokens,
                                               encode_tokens)
from vllm.utils import flatten_2d_lists, full_groupby

from .hasher import MultiModalHasher
from .inputs import (MultiModalDataDict, MultiModalEncDecInputs,
                     MultiModalFieldConfig, MultiModalInputs,
                     MultiModalKwargsItem, MultiModalKwargsItems,
                     MultiModalKwargsOptionalItems, MultiModalUUIDDict,
                     PlaceholderRange)
from .parse import (DictEmbeddingItems, EmbeddingItems, MultiModalDataItems,
                    MultiModalDataParser)

if TYPE_CHECKING:
    from transformers.configuration_utils import PretrainedConfig
    from transformers.feature_extraction_utils import BatchFeature
    from transformers.processing_utils import ProcessorMixin

    from .cache import BaseMultiModalProcessorCache
    from .profiling import BaseDummyInputsBuilder

logger = init_logger(__name__)

_S = TypeVar("_S", str, list[int])

PromptSeq = Union[str, list[int]]
"""A token sequence (list of token IDs) or text."""


@lru_cache(maxsize=2048)
def _cached_encode(
    tokenizer: AnyTokenizer,
    text: str,
    *,
    add_special_tokens: Optional[bool] = None,
) -> list[int]:
    return encode_tokens(tokenizer,
                         text,
                         add_special_tokens=add_special_tokens)


@lru_cache(maxsize=2048)
def _cached_decode(
    tokenizer: AnyTokenizer,
    token_ids: tuple[int, ...],
    *,
    skip_special_tokens: Optional[bool] = None,
) -> str:
    return decode_tokens(tokenizer,
                         list(token_ids),
                         skip_special_tokens=skip_special_tokens)


def _seq2text(tokenizer: AnyTokenizer, seq: PromptSeq) -> str:
    if isinstance(seq, str):
        return seq

    return _cached_decode(tokenizer, tuple(seq))


def _seq2tokens(tokenizer: AnyTokenizer, seq: PromptSeq) -> list[int]:
    if isinstance(seq, str):
        return _cached_encode(tokenizer, seq, add_special_tokens=False)

    return seq


class _GetMatchIndex(Protocol):

    def __call__(
        self,
        tokenizer: AnyTokenizer,
        prompt: PromptSeq,
        start_idx: int = 0,
    ) -> Optional[int]:
        ...


@dataclass
class PromptIndex:
    """Resolves to an index in the prompt."""
    get_match_index: _GetMatchIndex


class PromptIndexTargets:

    @staticmethod
    def start() -> PromptIndex:
        """
        Resolves to the start of the prompt (before the first token).

        This results in a match even if the prompt is empty.
        """
        return PromptIndex(lambda tokenizer, prompt, start_idx=0: 0)

    @staticmethod
    def prefix(seq: PromptSeq) -> PromptIndex:
        """
        Resolves to a location in the prompt after the given prefix.
        """

        def get_match_index(
            tokenizer: AnyTokenizer,
            prompt: PromptSeq,
            start_idx: int = 0,
        ) -> Optional[int]:
            if start_idx != 0:
                return None

            prefix = seq

            if isinstance(prompt, str):
                if not isinstance(prefix, str):
                    # Make both `str`
                    prefix = decode_tokens(tokenizer, prefix)
            else:
                if isinstance(prefix, str):
                    # Make both `list[int]`
                    prefix = encode_tokens(tokenizer,
                                           prefix,
                                           add_special_tokens=False)

            match_idx = len(prefix)
            return match_idx if prompt[:match_idx] == prefix else None

        return PromptIndex(get_match_index)

    @staticmethod
    def end() -> PromptIndex:
        """
        Resolves to the end of the prompt (after the last token).

        This results in a match even if the prompt is empty.
        """
        return PromptIndex(lambda tokenizer, prompt, start_idx=0: len(prompt))


UpdateTarget = Union[PromptSeq, PromptIndex]
"""
The token sequence or text to update.
"""

PromptUpdateTarget = Union[Callable[[int], UpdateTarget], UpdateTarget]
"""
Given the index of the processed item within
[`modality`][vllm.multimodal.processing.PromptUpdate.modality],
output the corresponding token sequence (or text).

For convenience, you can directly pass in the token sequence (or text)
instead of a function if it does not depend on the input.
"""


@dataclass
class PromptUpdateDetails(Generic[_S]):
    """Details about the token sequence or text that are part of the update."""

    full: _S
    """The full content."""

    is_embed: Optional[Callable[[AnyTokenizer, PromptSeq],
                                torch.Tensor]] = None
    """
    Given [`full`][vllm.multimodal.processing.PromptUpdateDetails.full],
    return a boolean mask of shape `(len(full),)` indicating which positions
    of `full` to assign embeddings to.

    `None` (default) means to assign embeddings to all positions of `full`.

    The embeddings are obtained by calling
    [`SupportsMultiModal.get_multimodal_embeddings`][vllm.model_executor.models.interfaces.SupportsMultiModal.get_multimodal_embeddings].
    """

    @staticmethod
    def from_seq(seq: _S) -> "PromptUpdateDetails[_S]":
        return PromptUpdateDetails(full=seq)

    @staticmethod
    def select_text(
        seq: _S,
        embed_text: str,
    ) -> "PromptUpdateDetails[_S]":

        def is_embed(tokenizer: AnyTokenizer, full: PromptSeq) -> torch.Tensor:
            embed_token_ids = encode_tokens(tokenizer, embed_text)
            token_ids = _seq2tokens(tokenizer, full)

            return torch.isin(
                torch.tensor(token_ids),
                torch.tensor(embed_token_ids),
            )

        return PromptUpdateDetails(full=seq, is_embed=is_embed)

    @staticmethod
    def select_token_id(
        seq: _S,
        embed_token_id: int,
    ) -> "PromptUpdateDetails[_S]":

        def is_embed(tokenizer: AnyTokenizer, full: PromptSeq) -> torch.Tensor:
            token_ids = _seq2tokens(tokenizer, full)

            return torch.tensor(token_ids) == embed_token_id

        return PromptUpdateDetails(full=seq, is_embed=is_embed)


PromptUpdateInfo = Union[PromptSeq, PromptUpdateDetails]
"""
The token sequence or text that are part of the update.

If only part of the content corresponds to feature placeholders, you can
use [`PromptUpdateDetails`][vllm.multimodal.processing.PromptUpdateDetails] to
specify which part.
"""

PromptUpdateContent = Union[Callable[[int], PromptUpdateInfo],
                            PromptUpdateInfo]
"""
Given the index of the processed item within
[`modality`][vllm.multimodal.processing.PromptUpdate.modality],
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

    target: PromptUpdateTarget
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

    def _resolve_target(self, item_idx: int) -> UpdateTarget:
        target = self.target
        if callable(target):
            target = target(item_idx)

        return target

    def _resolve_content(self, item_idx: int) -> PromptUpdateDetails:
        content = self.content
        if callable(content):
            content = content(item_idx)

        if not isinstance(content, PromptUpdateDetails):
            content = PromptUpdateDetails.from_seq(content)

        return content

    def resolve(self, item_idx: int) -> "ResolvedPromptUpdate":
        """
        Given the index of the processed item within
        [`modality`][vllm.multimodal.processing.PromptUpdate.modality],
        output a copy of this object with its lazy attributes resolved.
        """
        return ResolvedPromptUpdate(
            modality=self.modality,
            item_idx=item_idx,
            mode=self.mode,
            target=self._resolve_target(item_idx),
            content=self._resolve_content(item_idx),
        )


@dataclass
class PromptInsertion(PromptUpdate):
    """
    Defines how to insert placeholder tokens into a prompt.

    Example:

    For each image, insert a number of ``<image>`` feature placeholders
    equal to the feature size of the vision encoder after the ``<s>`` token:

    ```python
    PromptInsertion(
        modality="image",
        target="<s>",
        insertion="<image>" * image_feature_size,
    )
    ```

    Insert these tokens at the start of the prompt:

    ```python
    PromptInsertion(
        modality="image",
        target=PromptIndexTargets.start(),
        insertion="<image>" * image_feature_size,
    )
    ```

    Insert these tokens after a prefix ``Images:``:

    ```python
    PromptInsertion(
        modality="image",
        target=PromptIndexTargets.prefix("Images:"),
        insertion="<image>" * image_feature_size,
    )
    ```

    Insert these tokens at the end of the prompt:

    ```python
    PromptInsertion(
        modality="image",
        target=PromptIndexTargets.end(),
        insertion="<image>" * image_feature_size,
    )
    ```
    """

    insertion: PromptUpdateContent = field(repr=False)
    """
    Given the index of the processed item within
    [`modality`][vllm.multimodal.processing.PromptUpdate.modality],
    output the token sequence (or text) to insert right after
    [`target`][vllm.multimodal.processing.PromptUpdate.target].

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

    ```python
    PromptReplacement(
        modality="image",
        target="<image>",
        replacement="<image>" * image_feature_size,
    )
    ```

    As above, but further pad the feature placeholders with ``<image_bos>``
    and `<image_eos>``, which are not supposed to be passed to the vision
    encoder:

    ```python
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
    ```

    To avoid unnecessary tokenization during prompt replacement,
    we recommended passing token sequences instead of text:

    ```python
    PromptReplacement(
        modality="image",
        target=[image_token_id],
        replacement=PromptUpdateDetails(
            full=([image_bos_id] + [image_token_id] * image_feature_size
                    + [image_eos_id]),
            features=[image_token_id] * image_feature_size,
        ),
    )
    ```
    """

    replacement: PromptUpdateContent = field(repr=False)
    """
    Given the index of the processed item within
    [`modality`][vllm.multimodal.processing.PromptUpdate.modality],
    output the token sequence (or text) to replace
    [`target`][vllm.multimodal.processing.PromptUpdate.target].

    For convenience, you can directly pass in the token sequence (or text)
    instead of a function if it does not depend on the input.
    """

    @property
    def content(self) -> PromptUpdateContent:
        return self.replacement

    @property
    def mode(self) -> UpdateMode:
        return UpdateMode.REPLACE


class _HasModalityAttr(Protocol):
    modality: str


class _HasModalityProp(Protocol):

    @property
    def modality(self) -> str:
        ...


_M = TypeVar("_M", bound=Union[_HasModalityAttr, _HasModalityProp])


def full_groupby_modality(values: Iterable[_M]) -> ItemsView[str, list[_M]]:
    """Convenience function to apply [`full_groupby`][vllm.utils.full_groupby]
    based on modality."""
    return full_groupby(values, key=lambda x: x.modality)


class PromptTargetMatch(NamedTuple):
    start_idx: int
    end_idx: int


@dataclass(frozen=True)
class ResolvedPromptUpdate:
    """
    A [`PromptUpdate`][vllm.multimodal.processing.PromptUpdate] with its
    lazy attributes resolved, apart from those related to tokenization.
    """

    modality: str
    """The modality for which the update is made."""

    item_idx: int
    """The index within `modality` of the item this update pertains to."""

    mode: UpdateMode
    """Defines how to update the prompt."""

    target: UpdateTarget
    """The token sequence (or text) to update."""

    content: PromptUpdateDetails = field(repr=False)
    """The placeholder tokens that are part of the update."""

    def iter_token_matches(
        self,
        prompt: list[int],
        tokenizer: AnyTokenizer,
        *,
        start_idx: int = 0,
    ) -> Generator[PromptTargetMatch]:
        """Yield each instance of `self.target` found in `prompt`."""
        target = self.target

        if isinstance(target, PromptIndex):
            match_idx = target.get_match_index(tokenizer, prompt, start_idx)
            if match_idx is not None:
                yield PromptTargetMatch(match_idx, match_idx)

            return

        target_token_ids = _seq2tokens(tokenizer, target)

        for match in iter_token_matches(prompt,
                                        target_token_ids,
                                        start_idx=start_idx):
            yield PromptTargetMatch(match.start_idx, match.end_idx)

    def iter_text_matches(
        self,
        prompt: str,
        tokenizer: AnyTokenizer,
        *,
        start_idx: int = 0,
    ) -> Generator[PromptTargetMatch]:
        """Yield each instance of `self.target` found in `prompt`."""
        target = self.target

        if isinstance(target, PromptIndex):
            match_idx = target.get_match_index(tokenizer, prompt, start_idx)
            if match_idx is not None:
                yield PromptTargetMatch(match_idx, match_idx)

            return

        target_text = _seq2text(tokenizer, target)

        for match in re.finditer(re.escape(target_text), prompt,
                                 pos=start_idx):
            yield PromptTargetMatch(match.start(), match.end())

    def iter_matches(
        self,
        prompt: Union[list[int], str],
        tokenizer: AnyTokenizer,
        *,
        start_idx: int = 0,
    ) -> Generator[PromptTargetMatch]:
        """Yield each instance of `self.target` found in `prompt`."""
        if isinstance(prompt, str):
            return self.iter_text_matches(prompt,
                                          tokenizer,
                                          start_idx=start_idx)

        return self.iter_token_matches(prompt, tokenizer, start_idx=start_idx)

    def with_target(self, target: UpdateTarget):
        return replace(self, target=target)

    def with_content(self, content: PromptUpdateInfo):
        if not isinstance(content, PromptUpdateDetails):
            content = PromptUpdateDetails.from_seq(content)

        return replace(self, content=content)


class _TokenMatch(NamedTuple):
    start_idx: int
    end_idx: int


def iter_token_matches(
    token_ids: list[int],
    match_ids: list[int],
    *,
    start_idx: int = 0,
) -> Generator[_TokenMatch]:
    """
    Yield each occurrence of `match_ids` in `token_ids`.

    Note that empty matches are ignored.
    """
    prompt_len = len(token_ids)
    match_len = len(match_ids)

    if match_len == 0:
        return

    while start_idx < prompt_len - match_len + 1:
        end_idx = start_idx + match_len

        if token_ids[start_idx:end_idx] == match_ids:
            yield _TokenMatch(start_idx=start_idx, end_idx=end_idx)

            # Exclude overlapping matches
            start_idx = end_idx
        else:
            start_idx += 1


def replace_token_matches(
    token_ids: list[int],
    match_ids: list[int],
    new_ids: list[int],
) -> list[int]:
    """
    Replace each occurrence of `match_ids` in `token_ids`
    with `new_ids`.

    Note that empty matches are ignored.
    """
    out_seqs = list[list[int]]()
    prev_end_idx = 0

    for match in iter_token_matches(token_ids, match_ids):
        start_idx = match.start_idx
        end_idx = match.end_idx

        out_seqs.append(token_ids[prev_end_idx:start_idx])
        out_seqs.append(new_ids)
        prev_end_idx = end_idx

    out_seqs.append(token_ids[prev_end_idx:])

    return flatten_2d_lists(out_seqs)


@dataclass
class PlaceholderFeaturesInfo:
    modality: str
    item_idx: int
    start_idx: int
    tokens: list[int]
    is_embed: Optional[torch.Tensor]

    @property
    def length(self) -> int:
        return len(self.tokens)

    def to_range(self) -> PlaceholderRange:
        # TODO: Is it worth it to optimize this by stripping the
        # leading and ending positions where `is_embed=False`?
        return PlaceholderRange(
            offset=self.start_idx,
            length=self.length,
            is_embed=self.is_embed,
        )


_MatchToApply = tuple[tuple[str, int], tuple[PromptTargetMatch, int]]


def _find_matches(
    prompt: _S,
    mm_prompt_updates: "MultiModalPromptUpdates",
    tokenizer: AnyTokenizer,
    *,
    prev_end_idx: int = 0,
    current_result: "MultiModalPromptUpdatesApplyResult",
) -> tuple[Optional[UpdateMode], list[_MatchToApply]]:
    mode: Optional[UpdateMode] = None
    mm_matches = dict[tuple[str, int], tuple[PromptTargetMatch, int]]()

    for modality, modality_updates in mm_prompt_updates.items():
        for item_idx, item_updates in enumerate(modality_updates):
            if current_result[modality][item_idx] is not None:
                continue  # Updates have already been applied for this item

            for update_idx, update in enumerate(item_updates):
                if (modality, item_idx) in mm_matches:
                    break  # Already found a match for this item

                for match in update.iter_matches(
                        prompt,
                        tokenizer,
                        start_idx=prev_end_idx,
                ):
                    # All matches should share the same mode
                    if mode is None:
                        mode = update.mode
                    elif mode != update.mode:
                        continue

                    mm_matches[(modality, item_idx)] = match, update_idx
                    break  # Get only the first valid match per item

    # Prioritize earlier matches
    matches_to_apply = sorted(mm_matches.items(), key=lambda item: item[1][0])

    # To avoid conflicts, only replace one non-empty item at a time
    if mode == UpdateMode.REPLACE:
        matches_to_apply_ = list[_MatchToApply]()
        has_non_empty_matches = False

        for item in matches_to_apply:
            _, (match, _) = item
            if match.start_idx == match.end_idx:
                matches_to_apply_.append(item)
            elif not has_non_empty_matches:
                has_non_empty_matches = True
                matches_to_apply_.append(item)

        matches_to_apply = matches_to_apply_

    return mode, matches_to_apply


def _apply_matches(
    prompt: _S,
    mm_prompt_updates: "MultiModalPromptUpdates",
    tokenizer: AnyTokenizer,
) -> tuple[list[_S], "MultiModalPromptUpdatesApplyResult"]:
    prompt_len = len(prompt)

    out_seqs = list[Union[str, list[int]]]()
    out_result: MultiModalPromptUpdatesApplyResult = {
        m: [None] * len(items)
        for m, items in mm_prompt_updates.items()
    }

    start_idx = prev_end_idx = 0
    while start_idx < max(prompt_len, 1):  # Allow inserts into empty prompt
        found = False

        mode, matches_to_apply = _find_matches(
            prompt,
            mm_prompt_updates,
            tokenizer,
            prev_end_idx=prev_end_idx,
            current_result=out_result,
        )

        if mode is not None:
            for (modality, item_idx), (match, update_idx) in matches_to_apply:
                found = True

                matched_update = mm_prompt_updates[modality][item_idx][
                    update_idx]
                matched_content = matched_update.content.full

                if mode == UpdateMode.INSERT:
                    end_idx_to_insert = match.end_idx
                elif mode == UpdateMode.REPLACE:
                    end_idx_to_insert = match.start_idx
                else:
                    assert_never(mode)

                out_seqs.append(prompt[prev_end_idx:end_idx_to_insert])
                out_seqs.append(
                    _seq2text(tokenizer, matched_content
                              ) if isinstance(prompt, str) else _seq2tokens(
                                  tokenizer, matched_content))
                out_result[modality][item_idx] = update_idx

                # Exclude overlapping matches
                start_idx = prev_end_idx = match.end_idx

        if not found:
            start_idx += 1

    out_seqs.append(prompt[prev_end_idx:])

    return cast(list[_S], out_seqs), out_result


def apply_token_matches(
    prompt: list[int],
    mm_prompt_updates: "MultiModalPromptUpdates",
    tokenizer: AnyTokenizer,
) -> tuple[list[int], "MultiModalPromptUpdatesApplyResult"]:
    """
    Apply the updates in `mm_prompt_updates` to `prompt`.

    Matches are exclusive even when multiple modalities share
    the same placeholder tokens. In that case, the modality that
    appears earlier in `mm_prompt_updates` takes priority.
    """
    token_id_seqs, result = _apply_matches(prompt, mm_prompt_updates,
                                           tokenizer)

    return flatten_2d_lists(token_id_seqs), result


def apply_text_matches(
    prompt: str,
    mm_prompt_updates: "MultiModalPromptUpdates",
    tokenizer: AnyTokenizer,
) -> tuple[str, "MultiModalPromptUpdatesApplyResult"]:
    """
    Apply the updates in `mm_prompt_updates` to `prompt`.

    Matches are exclusive even when multiple modalities share
    the same placeholder tokens. In that case, the modality that
    appears earlier in `mm_prompt_updates` takes priority.
    """
    texts, result = _apply_matches(prompt, mm_prompt_updates, tokenizer)

    return "".join(texts), result


def _iter_placeholders(
    prompt: list[int],
    mm_prompt_updates: "MultiModalPromptUpdates",
    tokenizer: AnyTokenizer,
) -> Iterable[PlaceholderFeaturesInfo]:
    """
    Yield each set of placeholder tokens found in `prompt`.

    Matches are exclusive even when multiple modalities share
    the same placeholder tokens. In that case, the modality that
    appears earlier in `mm_prompt_updates` takes priority.

    Note that empty matches are ignored.
    """
    prompt_len = len(prompt)
    mm_item_counts = {m: len(items) for m, items in mm_prompt_updates.items()}

    item_idx_by_modality = defaultdict[str, int](lambda: 0)

    start_idx = 0
    while start_idx < prompt_len:
        found = False

        for modality, modality_updates in mm_prompt_updates.items():
            item_idx = item_idx_by_modality[modality]
            if item_idx >= mm_item_counts.get(modality, 0):
                continue

            for update in modality_updates[item_idx]:
                content = update.content
                content_tokens_full = _seq2tokens(tokenizer, content.full)
                content_len_full = len(content_tokens_full)
                end_idx_full = start_idx + content_len_full

                if content_len_full == 0 or end_idx_full > prompt_len:
                    continue

                if prompt[start_idx:end_idx_full] == content_tokens_full:
                    content_is_embed = content.is_embed
                    if content_is_embed is not None:
                        content_is_embed = content_is_embed(
                            tokenizer, content.full)

                    yield PlaceholderFeaturesInfo(
                        modality=modality,
                        item_idx=item_idx,
                        start_idx=start_idx,
                        tokens=content_tokens_full,
                        is_embed=content_is_embed,
                    )

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
    prompt: list[int],
    mm_prompt_updates: "MultiModalPromptUpdates",
    tokenizer: AnyTokenizer,
) -> Mapping[str, list[PlaceholderFeaturesInfo]]:
    it = _iter_placeholders(prompt, mm_prompt_updates, tokenizer)
    return dict(full_groupby_modality(it))


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

    def get_hf_config(self) -> "PretrainedConfig":
        return self.ctx.get_hf_config()

    def get_hf_processor(self, **kwargs: object) -> "ProcessorMixin":
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

    def get_allowed_mm_limits(self) -> Mapping[str, int]:
        """Return the maximum allowed number of items for each modality."""
        supported_mm_limits = self.get_supported_mm_limits()
        mm_config = self.ctx.get_mm_config()

        allowed_limits = dict[str, int]()
        for modality, supported_limit in supported_mm_limits.items():
            user_limit = mm_config.get_limit_per_prompt(modality)

            allowed_limits[modality] = (user_limit if supported_limit is None
                                        else min(user_limit, supported_limit))

        return allowed_limits

    def get_mm_max_tokens_per_item(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
    ) -> Optional[Mapping[str, int]]:
        """
        Return the maximum number of tokens per item of for each modality.
        
        When `None` (the default) is returned, vLLM will generate dummy inputs
        (images/videos) at maximum possible sizes and process them to determine
        the maximum token count per modality.

        This approach works but can be very slow for certain models (e.g.,
        Qwen2.5-VL), leading to very long startup time. For better performance,
        each model can override this method to return pre-computed maximum token
        counts, avoiding the need for dummy input generation and processing.

        Note:
            The maximum number of tokens per item of each modality returned 
            from this function should respect the model's maximum sequence
            length and the maximum number of items of each modality allowed,
            and agree with dummy inputs (images/videos) at maximum possible
            sizes.
        """
        return None


_I = TypeVar("_I", bound=BaseProcessingInfo)

MultiModalHashes = dict[str, list[str]]
"""
A collection of hashes with a similar structure as
[`MultiModalKwargsItems`][vllm.multimodal.inputs.MultiModalKwargsItems].
"""

MultiModalPromptUpdates = Mapping[str, list[Sequence[ResolvedPromptUpdate]]]
"""
A collection of prompt updates with a similar structure as
[`MultiModalKwargsItems`][vllm.multimodal.inputs.MultiModalKwargsItems].
"""

MultiModalPromptUpdatesApplyResult = Mapping[str, list[Optional[int]]]
"""
For an item `MultiModalPromptUpdates[k][i]`,
`MultiModalPromptUpdatesApplyResult[k][i]` represents the index of the
`ResolvedPromptUpdate` instance that has been applied, or `None` if none of the
`ResolvedPromptUpdate` instances have been applied.
"""


class MultiModalProcessingInfo(NamedTuple):
    kwargs: MultiModalKwargsOptionalItems
    hashes: MultiModalHashes
    prompt_updates: MultiModalPromptUpdates


class BaseMultiModalProcessor(ABC, Generic[_I]):
    """
    Abstract base class to process multi-modal inputs to be used in vLLM.

    Not to be confused with `transformers.ProcessorMixin`.
    """

    def __init__(
        self,
        info: _I,
        dummy_inputs: "BaseDummyInputsBuilder[_I]",
        *,
        cache: Optional["BaseMultiModalProcessorCache"] = None,
    ) -> None:
        super().__init__()

        self.info = info
        self.dummy_inputs = dummy_inputs
        self.cache = cache

        self.data_parser = self._get_data_parser()

        # Avoid unnecessary recomputation
        self._supported_mm_limits = self.info.get_supported_mm_limits()
        self._allowed_mm_limits = self.info.get_allowed_mm_limits()

    @property
    def supported_mm_limits(self):
        return self._supported_mm_limits

    @property
    def allowed_mm_limits(self):
        return self._allowed_mm_limits

    def __call__(
        self,
        prompt: str,
        mm_data: MultiModalDataDict,
        hf_processor_mm_kwargs: Mapping[str, object],
        *,
        mm_uuids: Optional[MultiModalUUIDDict] = None,
    ) -> MultiModalInputs:
        return self.apply(prompt,
                          mm_data,
                          hf_processor_mm_kwargs,
                          mm_uuids=mm_uuids)

    def _get_data_parser(self) -> MultiModalDataParser:
        """
        Construct a parser to preprocess multi-modal data items
        before passing them to
        [`_get_hf_mm_data`][vllm.multimodal.processing.BaseMultiModalProcessor._get_hf_mm_data].

        You can support additional modalities by creating a subclass
        of [`MultiModalDataParser`][vllm.multimodal.parse.MultiModalDataParser]
        that has additional subparsers.
        """
        return MultiModalDataParser()

    def validate_num_items(
        self,
        modality: str,
        num_items: int,
    ) -> None:
        supported_limit = self.supported_mm_limits.get(modality, 0)
        allowed_limit = self.allowed_mm_limits.get(modality, 0)

        if supported_limit is None:
            supported_limit = allowed_limit

        limit = min(supported_limit, allowed_limit)

        if num_items > limit:
            msg = (f"At most {limit} {modality}(s) may be provided in "
                   "one prompt.")

            if num_items <= supported_limit:
                msg += " Set `--limit-mm-per-prompt` to increase this limit."

            raise ValueError(msg)

    def _to_mm_items(
        self,
        mm_data: MultiModalDataDict,
    ) -> MultiModalDataItems:
        """
        Normalize
        [`MultiModalDataDict`][vllm.multimodal.inputs.MultiModalDataDict]
        to [`MultiModalDataItems`][vllm.multimodal.parse.MultiModalDataItems]
        before passing them to
        [`_get_hf_mm_data`][vllm.multimodal.processing.BaseMultiModalProcessor._get_hf_mm_data].
        """
        mm_items = self.data_parser.parse_mm_data(mm_data)
        for modality, items in mm_items.items():
            self.validate_num_items(modality, len(items))

        return mm_items

    @abstractmethod
    def _get_mm_fields_config(
        self,
        hf_inputs: "BatchFeature",
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        """Given the HF-processed data, output the metadata of each field."""
        raise NotImplementedError

    @abstractmethod
    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        out_mm_kwargs: MultiModalKwargsItems,
    ) -> Sequence[PromptUpdate]:
        """
        Given the original multi-modal items for this modality
        and HF-processed data, output the updates to perform.

        The information returned by this method is used to update token inputs
        which bypass the HF processor. It is also used to update the output of
        HF processor if the HF process does not apply prompt updates to text
        inputs.

        Moreover, this information is critical to determine the token positions
        in order to construct
        [`PlaceholderRange`][vllm.multimodal.inputs.PlaceholderRange]
        for each multi-modal item.
        """
        raise NotImplementedError

    def _bind_and_group_updates(
        self,
        prompt_updates: Sequence[PromptUpdate],
        mm_item_counts: Mapping[str, int],
    ) -> MultiModalPromptUpdates:
        return {
            modality: [[update.resolve(item_idx) for update in updates]
                       for item_idx in range(mm_item_counts.get(modality, 0))]
            for modality, updates in full_groupby_modality(prompt_updates)
        }

    def _get_mm_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        out_mm_kwargs: MultiModalKwargsItems,
    ) -> MultiModalPromptUpdates:
        unbound_prompt_updates = self._get_prompt_updates(
            mm_items=mm_items,
            hf_processor_mm_kwargs=hf_processor_mm_kwargs,
            out_mm_kwargs=out_mm_kwargs,
        )

        mm_prompt_updates = self._bind_and_group_updates(
            unbound_prompt_updates,
            mm_items.get_all_counts(),
        )

        for modality, prompt_updates in mm_prompt_updates.items():
            for item_idx, item_prompt_updates in enumerate(prompt_updates):
                if len(item_prompt_updates) > 1:
                    logger.warning_once(
                        "Detected %d prompt updates for `mm_items[%r][%s]`. "
                        "Multiple prompt updates per item is now "
                        "deprecated and may be removed in v0.13. "
                        "Instead, please specify dynamic update targets "
                        "in the same prompt update definition by passing "
                        "a function to `PromptUpdate.target`.",
                        len(prompt_updates),
                        modality,
                        item_idx,
                    )

        return mm_prompt_updates

    def _find_mm_placeholders(
        self,
        new_token_ids: list[int],
        mm_prompt_updates: MultiModalPromptUpdates,
    ) -> Mapping[str, list[PlaceholderFeaturesInfo]]:
        tokenizer = self.info.get_tokenizer()

        return find_mm_placeholders(new_token_ids, mm_prompt_updates,
                                    tokenizer)

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
        tok_kwargs: Mapping[str, object],
    ) -> "BatchFeature":
        """
        Call the HF processor on the prompt text and
        associated multi-modal data.
        """
        return self.info.ctx.call_hf_processor(
            self.info.get_hf_processor(**mm_kwargs),
            dict(text=prompt, **mm_data),
            dict(**mm_kwargs, **tok_kwargs),
        )

    def _hf_processor_applies_updates(
        self,
        prompt_text: str,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        tokenization_kwargs: Mapping[str, object],
    ) -> bool:
        """
        Return whether the HF processor applies prompt updates.

        For most HF processors, this should be `True` when multi-modal
        data items are passed, but `False` when multi-modal embeddings
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
        tokenization_kwargs: Mapping[str, object],
    ) -> tuple[list[int], "BatchFeature", bool]:
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
            tok_kwargs=tokenization_kwargs,
        )
        processed_data.update(passthrough_data)

        prompt_ids, = processed_data.pop("input_ids").tolist()

        is_update_applied = self._hf_processor_applies_updates(
            prompt_text=prompt_text,
            mm_items=mm_items,
            hf_processor_mm_kwargs=hf_processor_mm_kwargs,
            tokenization_kwargs=tokenization_kwargs,
        )

        return prompt_ids, processed_data, is_update_applied

    def _apply_hf_processor_text_only(
        self,
        prompt_text: str,
        tokenization_kwargs: Mapping[str, object],
    ) -> list[int]:
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
            tokenization_kwargs=tokenization_kwargs,
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
        with the output of
        [`_apply_hf_processor_text_only`][vllm.multimodal.processing.BaseMultiModalProcessor._apply_hf_processor_text_only]
        on the
        corresponding text.
        """
        return prompt_tokens

    def _apply_hf_processor_mm_only(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        tokenization_kwargs: Mapping[str, object],
    ) -> "BatchFeature":
        """
        Apply the HF processor on the multi-modal data only.

        Since HF processor requires that text and multi-modal items
        correspond to each other, we generate dummy text using
        [`DummyInputsBuilder`][vllm.multimodal.profiling.BaseDummyInputsBuilder]
        to go along with the multi-modal data.
        """
        mm_counts = mm_items.get_all_counts()

        _, mm_processed_data, _ = self._apply_hf_processor_text_mm(
            prompt_text=self.dummy_inputs.get_dummy_text(mm_counts),
            mm_items=mm_items,
            hf_processor_mm_kwargs=hf_processor_mm_kwargs,
            tokenization_kwargs=tokenization_kwargs,
        )

        return mm_processed_data

    def _apply_hf_processor_main(
        self,
        prompt: Union[str, list[int]],
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        tokenization_kwargs: Mapping[str, object],
        *,
        enable_hf_prompt_update: bool,
    ) -> tuple[list[int], "BatchFeature", bool]:
        """
        Apply the HF processor on the prompt text and multi-modal data.

        In addition, return whether prompt updates have been applied
        (for most HF processors, this should be `True`).

        Note:
            If `enable_hf_prompt_update=False`, we use HF processor
            to perform prompt updates if available; HF processor requires
            that the prompt corresponds to multi-modal items.
        """
        if isinstance(prompt, str):
            if enable_hf_prompt_update:
                return self._apply_hf_processor_text_mm(
                    prompt_text=prompt,
                    mm_items=mm_items,
                    hf_processor_mm_kwargs=hf_processor_mm_kwargs,
                    tokenization_kwargs=tokenization_kwargs,
                )

            prompt_ids = self._apply_hf_processor_text_only(
                prompt, tokenization_kwargs)
        else:
            prompt_ids = self._apply_hf_processor_tokens_only(prompt)

        mm_processed_data = self._apply_hf_processor_mm_only(
            mm_items=mm_items,
            hf_processor_mm_kwargs=hf_processor_mm_kwargs,
            tokenization_kwargs=tokenization_kwargs,
        )

        return prompt_ids, mm_processed_data, False

    def _hash_mm_items(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        tokenization_kwargs: Mapping[str, object],
        *,
        mm_uuids: Optional[MultiModalUUIDDict] = None,
    ) -> MultiModalHashes:
        """Create MM hashes to be returned (only used in V1).


        Note: When overrides are provided via callers of `apply`,
        `_hash_mm_items` will be bypassed and the overrides will be used.
        """
        model_id = self.info.model_id

        hashes: MultiModalHashes = {}
        mm_uuids = mm_uuids or {}

        for modality, items in mm_items.items():
            if modality in mm_uuids:
                mm_uuids_per_modality = mm_uuids[modality]
                if isinstance(mm_uuids_per_modality, str):
                    mm_uuids_per_modality = [mm_uuids_per_modality]

                # For None entries, compute a hash; otherwise, use provided ID.
                computed: list[str] = []
                for i, item in enumerate(items):
                    item_uuid = mm_uuids_per_modality[i]

                    # NOTE: Even if a item_uuid is provided, we still compute a
                    # hash if `hf_processor_mm_kwargs` or `tokenization_kwargs`
                    # are provided. This is because the processed multimodal
                    # inputs can be different depending on the processor kwargs.
                    if item_uuid is None or \
                        hf_processor_mm_kwargs or \
                        tokenization_kwargs:

                        # NOTE: use provided hash string to hash with kwargs
                        # if available for better performance.
                        item = item_uuid if item_uuid is not None else item
                        computed.append(
                            MultiModalHasher.hash_kwargs(
                                model_id=model_id,
                                **{modality: item},
                                **hf_processor_mm_kwargs,
                                **tokenization_kwargs))
                    else:
                        computed.append(item_uuid)
                hashes[modality] = computed
            else:
                hashes[modality] = [
                    MultiModalHasher.hash_kwargs(model_id=model_id,
                                                 **{modality: item},
                                                 **hf_processor_mm_kwargs,
                                                 **tokenization_kwargs)
                    for item in items
                ]

        return hashes

    def _get_cache_missing_items(
        self,
        cache: "BaseMultiModalProcessorCache",
        mm_data_items: MultiModalDataItems,
        mm_hashes: MultiModalHashes,
    ) -> MultiModalDataItems:
        mm_is_cached = {
            modality: cache.is_cached(hashes)
            for modality, hashes in mm_hashes.items()
        }

        mm_missing_idxs = {
            modality: [
                idx for idx, item_is_cached in enumerate(items_is_cached)
                if not item_is_cached
            ]
            for modality, items_is_cached in mm_is_cached.items()
        }
        mm_missing_data = {}
        for modality, idxs in mm_missing_idxs.items():
            missing_modality_data = []
            for idx in idxs:
                data = mm_data_items[modality][idx]
                if data is None:
                    raise ValueError(
                        f"Cache miss for {modality} at index {idx} "
                        f"but data is not provided.")
                else:
                    missing_modality_data.append(data)
            mm_missing_data[modality] = missing_modality_data

        return self._to_mm_items(mm_missing_data)

    def _recompute_cached_prompt_update(
        self,
        cached_update: ResolvedPromptUpdate,
        new_item_idx: int,
    ) -> ResolvedPromptUpdate:
        """
        Override this if other attributes of `ResolvedPromptUpdate`
        also need to be recomputed after retrieving from the cache.
        """
        return replace(cached_update, item_idx=new_item_idx)

    def _merge_mm_kwargs(
        self,
        cache: "BaseMultiModalProcessorCache",
        mm_hashes: MultiModalHashes,
        mm_missing_kwargs: MultiModalKwargsItems,
        mm_missing_prompt_updates: MultiModalPromptUpdates,
    ) -> tuple[MultiModalKwargsOptionalItems, MultiModalPromptUpdates]:
        # Need to calculate this at the beginning to avoid skipping cache logic
        # for subsequently repeated items in the same modality
        mm_is_cached = {
            modality: cache.is_cached(hashes)
            for modality, hashes in mm_hashes.items()
        }

        mm_missing_next_idx = defaultdict[str, int](lambda: 0)

        merged_kwargs = defaultdict[str,
                                    list[Optional[MultiModalKwargsItem]]](list)
        merged_prompt_updates = defaultdict[
            str, list[Sequence[ResolvedPromptUpdate]]](list)
        for modality, hashes in mm_hashes.items():
            missing_kwargs = mm_missing_kwargs.get(modality, [])
            missing_prompt_updates = mm_missing_prompt_updates.get(
                modality, [])

            for item_idx, item_hash in enumerate(hashes):
                kwargs: Optional[MultiModalKwargsItem]
                if not mm_is_cached[modality][item_idx]:
                    missing_next_idx = mm_missing_next_idx[modality]
                    kwargs = missing_kwargs[missing_next_idx]
                    updates = missing_prompt_updates[missing_next_idx]

                    mm_missing_next_idx[modality] += 1

                    item = kwargs, updates
                else:
                    item = None

                kwargs, updates = cache.get_and_update_item(item, item_hash)

                merged_kwargs[modality].append(kwargs)
                merged_prompt_updates[modality].append([
                    self._recompute_cached_prompt_update(update, item_idx)
                    for update in updates
                ])

        mm_kwargs = MultiModalKwargsItems(merged_kwargs)
        mm_prompt_updates = dict(merged_prompt_updates)

        return mm_kwargs, mm_prompt_updates

    def _apply_hf_processor(
        self,
        prompt: Union[str, list[int]],
        mm_data_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        tokenization_kwargs: Mapping[str, object],
        *,
        mm_uuids: Optional[MultiModalUUIDDict] = None,
    ) -> tuple[list[int], MultiModalProcessingInfo, bool]:
        (
            prompt_ids,
            mm_processed_data,
            is_update_applied,
        ) = self._apply_hf_processor_main(
            prompt=prompt,
            mm_items=mm_data_items,
            hf_processor_mm_kwargs=hf_processor_mm_kwargs,
            tokenization_kwargs=tokenization_kwargs,
            enable_hf_prompt_update=True,
        )

        mm_kwargs = MultiModalKwargsItems.from_hf_inputs(
            mm_processed_data,
            self._get_mm_fields_config(mm_processed_data,
                                       hf_processor_mm_kwargs),
        )

        # Use overrides if provided; fallback to data-dependent hashing.
        mm_hashes = self._hash_mm_items(mm_data_items,
                                        hf_processor_mm_kwargs,
                                        tokenization_kwargs,
                                        mm_uuids=mm_uuids)

        mm_prompt_updates = self._get_mm_prompt_updates(
            mm_data_items,
            hf_processor_mm_kwargs,
            mm_kwargs,
        )

        mm_info = MultiModalProcessingInfo(
            kwargs=mm_kwargs,
            hashes=mm_hashes,
            prompt_updates=mm_prompt_updates,
        )

        return prompt_ids, mm_info, is_update_applied

    def _cached_apply_hf_processor(
        self,
        prompt: Union[str, list[int]],
        mm_data_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        tokenization_kwargs: Mapping[str, object],
        *,
        mm_uuids: Optional[MultiModalUUIDDict] = None,
    ) -> tuple[list[int], MultiModalProcessingInfo, bool]:
        """
        Apply the HF processor on the full prompt text,
        caching the results and reusing cached results.
        """
        cache = self.cache

        _, passthrough_data = self._get_hf_mm_data(mm_data_items)
        if cache is None or passthrough_data:
            return self._apply_hf_processor(
                prompt=prompt,
                mm_data_items=mm_data_items,
                hf_processor_mm_kwargs=hf_processor_mm_kwargs,
                tokenization_kwargs=tokenization_kwargs,
                mm_uuids=mm_uuids,
            )

        mm_hashes = self._hash_mm_items(mm_data_items,
                                        hf_processor_mm_kwargs,
                                        tokenization_kwargs,
                                        mm_uuids=mm_uuids)

        mm_missing_data_items = self._get_cache_missing_items(
            cache=cache,
            mm_data_items=mm_data_items,
            mm_hashes=mm_hashes,
        )

        # NOTE: `prompt` does not correspond to `mm_missing_data_items`,
        # so we can't apply prompt updates until the new multimodal
        # items are combined with the cached multimodal items
        (
            prompt_ids,
            mm_missing_processed_data,
            is_update_applied,
        ) = self._apply_hf_processor_main(
            prompt=prompt,
            mm_items=mm_missing_data_items,
            hf_processor_mm_kwargs=hf_processor_mm_kwargs,
            tokenization_kwargs=tokenization_kwargs,
            enable_hf_prompt_update=False,
        )

        mm_missing_kwargs = MultiModalKwargsItems.from_hf_inputs(
            mm_missing_processed_data,
            self._get_mm_fields_config(mm_missing_processed_data,
                                       hf_processor_mm_kwargs),
        )

        mm_missing_prompt_updates = self._get_mm_prompt_updates(
            mm_missing_data_items,
            hf_processor_mm_kwargs,
            mm_missing_kwargs,
        )

        mm_kwargs, mm_prompt_updates = self._merge_mm_kwargs(
            cache,
            mm_hashes=mm_hashes,
            mm_missing_kwargs=mm_missing_kwargs,
            mm_missing_prompt_updates=mm_missing_prompt_updates,
        )

        mm_info = MultiModalProcessingInfo(
            kwargs=mm_kwargs,
            hashes=mm_hashes,
            prompt_updates=mm_prompt_updates,
        )

        return prompt_ids, mm_info, is_update_applied

    def _apply_token_matches(
        self,
        prompt: list[int],
        mm_prompt_updates: MultiModalPromptUpdates,
    ) -> tuple[list[int], MultiModalPromptUpdatesApplyResult]:
        tokenizer = self.info.get_tokenizer()
        return apply_token_matches(prompt, mm_prompt_updates, tokenizer)

    def _apply_text_matches(
        self,
        prompt: str,
        mm_prompt_updates: MultiModalPromptUpdates,
    ) -> tuple[str, MultiModalPromptUpdatesApplyResult]:
        tokenizer = self.info.get_tokenizer()
        return apply_text_matches(prompt, mm_prompt_updates, tokenizer)

    def _apply_prompt_updates(
        self,
        token_ids: list[int],
        mm_prompt_updates: MultiModalPromptUpdates,
    ) -> tuple[list[int], str, Mapping[str, list[PlaceholderFeaturesInfo]]]:
        tokenizer = self.info.get_tokenizer()

        new_token_ids, match_result = self._apply_token_matches(
            token_ids,
            mm_prompt_updates,
        )

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
                all(update_idx is not None for update_idx in update_idxs)
                for update_idxs in match_result.values()):
            new_text = decode_tokens(tokenizer, new_token_ids)
        else:
            new_text, match_result = self._apply_text_matches(
                decode_tokens(tokenizer, token_ids),
                mm_prompt_updates,
            )

            new_token_ids = encode_tokens(
                tokenizer,
                new_text,
                add_special_tokens=False,
            )

        matched_updates = defaultdict[
            str, list[Sequence[ResolvedPromptUpdate]]](list)
        for modality, update_idxs in match_result.items():
            for item_idx, update_idx in enumerate(update_idxs):
                assert update_idx is not None, (
                    "Failed to apply prompt replacement for "
                    f"mm_items[{modality!r}][{item_idx}]")

                matched_updates[modality].append(
                    [mm_prompt_updates[modality][item_idx][update_idx]])

        placeholders = self._find_mm_placeholders(
            new_token_ids,
            dict(matched_updates),
        )

        return new_token_ids, new_text, placeholders

    def _validate_mm_kwargs(
        self,
        mm_kwargs: MultiModalKwargsOptionalItems,
        mm_item_counts: Mapping[str, int],
    ) -> None:
        for modality, item_count in mm_item_counts.items():
            items = mm_kwargs.get(modality, [])

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
                # NOTE: If you are a model developer, this can also arise from
                # an inconsistency between `_call_hf_processor` and
                # `_get_mm_fields_config` implementations
                raise RuntimeError(
                    f"Expected there to be {item_count} prompt updates "
                    f"corresponding to {item_count} {modality} items, but "
                    f"instead found {len(placeholders)} prompt updates! "
                    "This is likely because you forgot to include input "
                    "placeholder tokens (e.g., `<image>`, `<|image_pad|>`) "
                    "in the prompt. If the model has a chat template, make "
                    "sure you have applied it before calling `LLM.generate`.")

    def _maybe_apply_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        prompt_ids: list[int],
        mm_kwargs: MultiModalKwargsOptionalItems,
        mm_prompt_updates: MultiModalPromptUpdates,
        is_update_applied: bool,
    ) -> tuple[list[int], str, Mapping[str, list[PlaceholderFeaturesInfo]]]:
        mm_item_counts = mm_items.get_all_counts()
        self._validate_mm_kwargs(mm_kwargs, mm_item_counts)

        if is_update_applied:
            mm_placeholders = self._find_mm_placeholders(
                prompt_ids,
                mm_prompt_updates,
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
            )
            self._validate_mm_placeholders(mm_placeholders, mm_item_counts)

        return prompt_ids, prompt, mm_placeholders

    def apply(
        self,
        prompt: Union[str, list[int]],
        mm_data: MultiModalDataDict,
        hf_processor_mm_kwargs: Mapping[str, object],
        tokenization_kwargs: Optional[Mapping[str, object]] = None,
        *,
        mm_uuids: Optional[MultiModalUUIDDict] = None,
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

        if tokenization_kwargs is None:
            tokenization_kwargs = {}

        (
            prompt_ids,
            mm_info,
            is_update_applied,
        ) = self._cached_apply_hf_processor(
            prompt,
            mm_items,
            hf_processor_mm_kwargs,
            tokenization_kwargs=tokenization_kwargs,
            mm_uuids=mm_uuids,
        )

        # NOTE: tokenization_kwargs are not required to init processor
        prompt_ids, prompt, mm_placeholders = self._maybe_apply_prompt_updates(
            mm_items=mm_items,
            prompt_ids=prompt_ids,
            mm_kwargs=mm_info.kwargs,
            mm_prompt_updates=mm_info.prompt_updates,
            is_update_applied=is_update_applied,
        )

        mm_placeholder_ranges = {
            modality: [item.to_range() for item in placeholders]
            for modality, placeholders in mm_placeholders.items()
        }

        return MultiModalInputs(
            type="multimodal",
            prompt=prompt,
            prompt_token_ids=prompt_ids,
            mm_kwargs=mm_info.kwargs,
            mm_hashes=mm_info.hashes,
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

    @property
    def pad_dummy_encoder_prompt(self) -> bool:
        return False

    def create_decoder_prompt(
        self,
        prompt: Union[str, list[int]],
        mm_data: MultiModalDataDict,
    ) -> Union[str, list[int]]:
        """Create input prompt for the decoder."""
        return prompt

    def _get_enc_dec_inputs(
        self,
        prompt: Union[str, list[int]],
        mm_data: MultiModalDataDict,
        encoder_inputs: MultiModalInputs,
    ):
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

    def apply(
        self,
        prompt: Union[str, list[int]],
        mm_data: MultiModalDataDict,
        hf_processor_mm_kwargs: Mapping[str, object],
        tokenization_kwargs: Optional[Mapping[str, object]] = None,
        *,
        mm_uuids: Optional[MultiModalUUIDDict] = None,
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
            tokenization_kwargs,
            mm_uuids=mm_uuids,
        )

        return self._get_enc_dec_inputs(
            prompt=prompt,
            mm_data=mm_data,
            encoder_inputs=encoder_inputs,
        )
