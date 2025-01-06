import pickle
import re
from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import Callable, ItemsView, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, NamedTuple, Optional, Protocol, TypeVar, Union

import numpy as np
import torch
from blake3 import blake3
from PIL import Image
from transformers import BatchFeature, PretrainedConfig, ProcessorMixin

from vllm.inputs import DummyData, InputProcessingContext
from vllm.logger import init_logger
from vllm.transformers_utils.tokenizer import (AnyTokenizer, decode_tokens,
                                               encode_tokens)
from vllm.utils import LRUCache, flatten_2d_lists, full_groupby

from .inputs import (MultiModalDataDict, MultiModalFieldConfig,
                     MultiModalInputsV2, MultiModalKwargs,
                     MultiModalKwargsItem, PlaceholderRange)
from .parse import MultiModalDataItems, MultiModalDataParser
from .profiling import BaseProfilingInfo

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

    def bind(self, tokenizer: AnyTokenizer) -> "_BoundPromptReplacement":
        return _BoundPromptReplacement(
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


@dataclass
class _PlaceholderInfo:
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
    modality_repls: Sequence[_BoundPromptReplacement],
    modal_item_count: int,
) -> Iterable[_PlaceholderInfo]:
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
                yield _PlaceholderInfo(
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
    mm_prompt_repls: Mapping[str, Sequence[_BoundPromptReplacement]],
    prompt: list[int],
    mm_item_counts: Mapping[str, int],
) -> Iterable[_PlaceholderInfo]:
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
    mm_prompt_repls: Mapping[str, Sequence[_BoundPromptReplacement]],
    prompt: list[int],
    mm_item_counts: Mapping[str, int],
) -> Mapping[str, list[_PlaceholderInfo]]:
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

    def _serialize_item(self, obj: object) -> bytes:
        # Simple cases
        if isinstance(obj, str):
            return obj.encode("utf-8")
        if isinstance(obj, bytes):
            return obj
        if isinstance(obj, Image.Image):
            return obj.tobytes()

        # Convertible to NumPy arrays
        if isinstance(obj, torch.Tensor):
            obj = obj.numpy()
        if isinstance(obj, (int, float)):
            obj = np.array(obj)
        if isinstance(obj, np.ndarray):
            return obj.tobytes()

        logger.warning(
            "No serialization method found for %s. "
            "Falling back to pickle.", type(obj))

        return pickle.dumps(obj)

    def _item_to_bytes(
        self,
        key: str,
        obj: object,
    ) -> Iterable[tuple[bytes, bytes]]:
        # Recursive cases
        if isinstance(obj, (list, tuple)):
            for i, elem in enumerate(obj):
                yield from self._item_to_bytes(f"{key}.{i}", elem)
        elif isinstance(obj, dict):
            for k, v in obj.items():
                yield from self._item_to_bytes(f"{key}.{k}", v)
        else:
            key_bytes = self._serialize_item(key)
            value_bytes = self._serialize_item(obj)
            yield key_bytes, value_bytes

    def _hash_kwargs(self, **kwargs: object) -> str:
        hasher = blake3()

        for k, v in kwargs.items():
            for k_bytes, v_bytes in self._item_to_bytes(k, v):
                hasher.update(k_bytes)
                hasher.update(v_bytes)

        return hasher.hexdigest()

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

        cache_key = self._hash_kwargs(model_id=model_id,
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
        cache_key = self._hash_kwargs(model_id=model_id,
                                      **{modality: input_item},
                                      **input_kwargs)
        self._cache.put(cache_key, output_kwargs)


class ProcessingMixin:
    """
    Contains helper functions to perform processing.

    Not to be confused with :class:`transformers.ProcessorMixin`.
    """
    ctx: InputProcessingContext

    def _get_tokenizer(self) -> AnyTokenizer:
        return self.ctx.tokenizer

    def _get_hf_config(self) -> PretrainedConfig:
        return self.ctx.get_hf_config()

    def _get_hf_processor(self, **kwargs: object) -> ProcessorMixin:
        """
        Subclasses can override this method to handle
        specific kwargs from model config or user inputs.
        """
        return self.ctx.get_hf_processor(**kwargs)


class BaseMultiModalProcessor(ProcessingMixin, ABC):
    """
    Abstract base class to process multi-modal inputs to be used in vLLM.

    Not to be confused with :class:`transformers.ProcessorMixin`.
    """

    def __init__(self,
                 ctx: InputProcessingContext,
                 *,
                 cache: Optional[ProcessingCache] = None,
                 enable_sanity_checks: bool = True) -> None:
        super().__init__()

        self.ctx = ctx
        self.cache = cache
        self.enable_sanity_checks = enable_sanity_checks

        self.data_parser = self._get_data_parser()
        self.profiling_info = self._get_profiling_info()

    def __call__(
        self,
        prompt: str,
        mm_data: MultiModalDataDict,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> MultiModalInputsV2:
        return self.apply(prompt, mm_data, hf_processor_mm_kwargs)

    def _get_data_parser(self) -> MultiModalDataParser:
        """
        Construct a parser to preprocess multi-modal data items
        before passing them to :meth:`_get_hf_mm_data`.

        You can support additional modalities by creating a subclass
        of :class:`MultiModalDataParser` that has additional subparsers.
        """
        return MultiModalDataParser()

    def _get_profiling_info(self) -> BaseProfilingInfo:
        """
        Get the profiling information to find the worst-case memory usage of
        the model.
        """
        raise NotImplementedError

    def _to_mm_items(
        self,
        mm_data: MultiModalDataDict,
    ) -> MultiModalDataItems:
        """
        Normalize :class:`MultiModalDataDict` to :class:`MultiModalDataItems`
        before passing them to :meth:`_get_hf_mm_data`.
        """
        mm_items = self.data_parser.parse_mm_data(mm_data)

        mm_limits = self.ctx.get_mm_config().limit_per_prompt
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
    def _get_prompt_replacements(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        out_mm_kwargs: MultiModalKwargs,
    ) -> list[PromptReplacement]:
        """
        Given the original multi-modal items for this modality
        and HF-processed data, output the replacements to perform.

        Notes:
            - You should not assume that HF processor always performs prompt
              replacement: in :meth:`_apply_hf_processor_missing`, this method
              is called on text-only and multimodal-only inputs separately,
              instead of passing them in the same call.
            - The replacement information returned by this method is also used
              to determine the placeholder token positions for each multi-modal
              item.
        """
        raise NotImplementedError

    def _find_mm_placeholders(
        self,
        mm_prompt_repls: Mapping[str, Sequence[_BoundPromptReplacement]],
        new_token_ids: list[int],
        mm_item_counts: Mapping[str, int],
    ) -> Mapping[str, list[_PlaceholderInfo]]:
        return find_mm_placeholders(mm_prompt_repls, new_token_ids,
                                    mm_item_counts)

    def _get_hf_mm_data(
        self,
        mm_items: MultiModalDataItems,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        processor_data = dict[str, Any]()
        passthrough_data = dict[str, Any]()

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
        return self.ctx.call_hf_processor(
            self._get_hf_processor(**mm_kwargs),
            dict(text=prompt, **mm_data),
            mm_kwargs,
        )

    def _apply_hf_processor(
        self,
        prompt_text: str,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> tuple[list[int], MultiModalKwargs]:
        """
        Wrapper of :meth:`_call_hf_processor` that applies
        additional pre-processing and post-processing.
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

        return prompt_ids, mm_kwargs

    def _apply_hf_processor_missing(
        self,
        prompt_text: str,
        mm_missing_data_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
    ):
        """
        Apply the HF processor on the full prompt text, but only on the
        multi-modal data that are missing from the cache.

        Note:
            We pass prompt text and multi-modal data into the HF processor
            in separate calls to avoid HF prompt replacement being done for
            cached items; instead, we rely on our own prompt replacement logic
            (:meth:`_get_prompt_replacements`) for the full text.
        """
        mm_missing_counts = mm_missing_data_items.get_all_counts()

        prompt_ids, _ = self._apply_hf_processor(
            prompt_text=prompt_text,
            mm_items=MultiModalDataItems({}),
            hf_processor_mm_kwargs={},
        )

        # Some HF processors (e.g. Qwen2-VL) expect corresponding
        # multi-modal tokens to be in the prompt text
        dummy_inputs = self.profiling_info.get_dummy_processor_inputs(
            self.ctx.model_config.max_model_len,
            mm_missing_counts,
        )

        _, mm_missing_kwargs = self._apply_hf_processor(
            prompt_text=dummy_inputs.prompt_text,
            mm_items=mm_missing_data_items,
            hf_processor_mm_kwargs=hf_processor_mm_kwargs,
        )

        return prompt_ids, mm_missing_kwargs

    def _cached_apply_hf_processor(
        self,
        prompt_text: str,
        mm_data_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> tuple[list[int], MultiModalKwargs]:
        """
        Apply the HF processor on the full prompt text,
        caching the results and reusing cached results.
        """
        cache = self.cache
        model_id = self.ctx.model_config.model

        _, passthrough_data = self._get_hf_mm_data(mm_data_items)
        if cache is None or passthrough_data:
            return self._apply_hf_processor(
                prompt_text=prompt_text,
                mm_items=mm_data_items,
                hf_processor_mm_kwargs=hf_processor_mm_kwargs,
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

        prompt_ids, mm_missing_kwargs = self._apply_hf_processor_missing(
            prompt_text=prompt_text,
            mm_missing_data_items=mm_missing_data_items,
            hf_processor_mm_kwargs=hf_processor_mm_kwargs,
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

        return prompt_ids, mm_kwargs

    def _bind_and_group_repls(
        self,
        prompt_repls: list[PromptReplacement],
    ) -> dict[str, list[_BoundPromptReplacement]]:
        tokenizer = self._get_tokenizer()

        it = (prompt_repl.bind(tokenizer) for prompt_repl in prompt_repls)
        return dict(full_groupby_modality(it))

    def _always_apply_prompt_replacements(self) -> bool:
        """
        A flag which can be overridden so that
        :meth:`_apply_prompt_replacements` is always called even if we
        detect that HF has performed processing via
        :meth:`_find_placeholders_by_modality`.

        This is useful in cases where :meth:`_find_placeholders_by_modality`
        cannot be reliably used to detect whether HF has performed processing.
        """
        return False

    def _apply_prompt_replacements(
        self,
        token_ids: list[int],
        mm_prompt_repls: Mapping[str, Sequence[_BoundPromptReplacement]],
        mm_item_counts: Mapping[str, int],
    ) -> tuple[list[int], str, Mapping[str, list[_PlaceholderInfo]]]:
        tokenizer = self._get_tokenizer()

        mm_token_matches = {
            modality: find_token_matches(token_ids, prompt_repls)
            for modality, prompt_repls in mm_prompt_repls.items()
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
        # of the search text in the prompt, we instead perform string
        # replacement on the decoded token IDs, then encode them back.
        if all(
            mm_match_counts.get(modality, 0) >= item_count
            for modality, item_count in mm_item_counts.items()
        ):  # yapf: disable
            token_ids = replace_token_matches(
                token_ids,
                mm_token_matches,
                mm_item_counts,
            )

            text = decode_tokens(tokenizer, token_ids)
            matched_repls = {
                modality: [match.prompt_repl for match in token_matches]
                for modality, token_matches in mm_token_matches.items()
            }
        else:
            text = decode_tokens(tokenizer, token_ids)

            mm_text_matches = {
                modality: find_text_matches(text, prompt_repls)
                for modality, prompt_repls in mm_prompt_repls.items()
            }
            text = replace_text_matches(
                text,
                mm_text_matches,
                mm_item_counts,
            )

            token_ids = encode_tokens(tokenizer,
                                      text,
                                      add_special_tokens=False)
            matched_repls = {
                modality: [match.prompt_repl for match in token_matches]
                for modality, token_matches in mm_text_matches.items()
            }

        placeholders = self._find_mm_placeholders(
            matched_repls,
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
        mm_placeholders: Mapping[str, list[_PlaceholderInfo]],
        mm_item_counts: Mapping[str, int],
        *,
        allow_missing: bool = False,
    ) -> Mapping[str, int]:
        missing_repl_counts = dict[str, int]()

        for modality, item_count in mm_item_counts.items():
            placeholders = mm_placeholders.get(modality, [])

            if len(placeholders) != item_count and not allow_missing:
                raise RuntimeError(
                    f"Expected there to be {item_count} prompt replacements "
                    f"corresponding to {item_count} {modality} items, but only "
                    f"found {len(placeholders)} prompt replacements! Either "
                    "the prompt text has missing/incorrect tokens for "
                    "multi-modal inputs, or there is a problem with your "
                    "implementation of merged multi-modal processor for this "
                    "model (usually arising from an inconsistency between "
                    "`_call_hf_processor` and `_get_prompt_replacements`).")

            missing_repl_counts[modality] = item_count - len(placeholders)

        return missing_repl_counts

    def apply(
        self,
        prompt_text: str,
        mm_data: MultiModalDataDict,
        hf_processor_mm_kwargs: Mapping[str, object],
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
        mm_items = self._to_mm_items(mm_data)

        prompt_ids, mm_kwargs = self._cached_apply_hf_processor(
            prompt_text,
            mm_items,
            hf_processor_mm_kwargs,
        )

        unbound_prompt_repls = self._get_prompt_replacements(
            mm_items,
            hf_processor_mm_kwargs,
            mm_kwargs,
        )
        mm_prompt_repls = self._bind_and_group_repls(unbound_prompt_repls)

        mm_item_counts = mm_items.get_all_counts()
        self._validate_mm_kwargs(mm_kwargs, mm_item_counts)

        hf_mm_placeholders = self._find_mm_placeholders(
            mm_prompt_repls,
            prompt_ids,
            mm_item_counts,
        )

        if self._always_apply_prompt_replacements():
            mm_missing_repl_counts = mm_item_counts
            mm_missing_repls = dict(mm_prompt_repls)
        else:
            mm_missing_repl_counts = self._validate_mm_placeholders(
                hf_mm_placeholders,
                mm_item_counts,
                allow_missing=True,
            )

            mm_missing_repls = dict[str, list[_BoundPromptReplacement]]()
            for modality, missing_repl_count in mm_missing_repl_counts.items():
                if missing_repl_count == 0:
                    mm_missing_repls[modality] = []
                elif missing_repl_count == mm_item_counts.get(modality, 0):
                    mm_missing_repls[modality] = mm_prompt_repls[modality]
                else:
                    raise ValueError("Partial prompt replacement within "
                                     f"{modality=} is not supported")

        # If HF processor already inserts placeholder tokens,
        # there is no need for us to insert them
        if all(len(repls) == 0 for repls in mm_missing_repls.items()):
            tokenizer = self._get_tokenizer()
            prompt_text = decode_tokens(tokenizer, prompt_ids)
            mm_placeholders = hf_mm_placeholders
        else:
            (
                prompt_ids,
                prompt_text,
                missing_mm_placeholders,
            ) = self._apply_prompt_replacements(
                prompt_ids,
                mm_missing_repls,
                mm_missing_repl_counts,
            )

            mm_placeholders = {**hf_mm_placeholders, **missing_mm_placeholders}

        self._validate_mm_placeholders(mm_placeholders, mm_item_counts)

        mm_placeholder_ranges = {
            modality: [item.to_range() for item in placeholders]
            for modality, placeholders in mm_placeholders.items()
        }

        return MultiModalInputsV2(
            type="multimodal",
            prompt=prompt_text,
            prompt_token_ids=prompt_ids,
            mm_kwargs=mm_kwargs,
            mm_placeholders=mm_placeholder_ranges,
        )

    def _get_dummy_mm_inputs(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
    ) -> MultiModalInputsV2:
        profiling = self.profiling_info
        processor_inputs = profiling.get_dummy_processor_inputs(
            seq_len, mm_counts)

        return self.apply(
            prompt_text=processor_inputs.prompt_text,
            mm_data=processor_inputs.mm_data,
            hf_processor_mm_kwargs=processor_inputs.hf_processor_mm_kwargs,
        )

    def get_dummy_data(self, seq_len: int) -> DummyData:
        # Avoid circular import
        from vllm.sequence import SequenceData

        profiling = self.profiling_info
        mm_counts = profiling.get_mm_limits()
        mm_max_tokens_per_item = profiling.get_mm_max_tokens_per_item(seq_len)
        if mm_counts.keys() != mm_max_tokens_per_item.keys():
            raise AssertionError(
                "The keys returned by `get_supported_mm_limits`"
                f"({set(mm_counts.keys())}) should be the same as those "
                "returned by `get_mm_max_tokens_per_item` "
                f"({set(mm_max_tokens_per_item.keys())})")

        mm_inputs = self._get_dummy_mm_inputs(seq_len, mm_counts)
        prompt_token_ids = mm_inputs["prompt_token_ids"]
        placeholders_by_modality = mm_inputs["mm_placeholders"]

        total_placeholders_by_modality = {
            modality: sum(item["length"] for item in placeholders)
            for modality, placeholders in placeholders_by_modality.items()
        }
        expected_placeholders_by_modality = {
            modality: mm_max_tokens_per_item[modality] * mm_counts[modality]
            for modality in placeholders_by_modality
        }
        if total_placeholders_by_modality != expected_placeholders_by_modality:
            raise AssertionError(
                f"The processed dummy data has a total of "
                f"{total_placeholders_by_modality} placeholder tokens, which "
                f"is not the expected {expected_placeholders_by_modality} "
                "tokens.")

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

            return DummyData(
                seq_data=SequenceData.from_prompt_token_counts((0, seq_len)),
                multi_modal_data=None,
                multi_modal_placeholders=None,
            )

        prompt_token_ids.extend([0] * (seq_len - len(prompt_token_ids)))

        return DummyData(
            seq_data=SequenceData.from_seqs(prompt_token_ids),
            multi_modal_data=mm_inputs["mm_kwargs"],
            multi_modal_placeholders=placeholders_by_modality,
        )
