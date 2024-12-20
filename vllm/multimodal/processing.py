import re
from abc import ABC, abstractmethod
from collections import UserDict, defaultdict
from collections.abc import Callable, ItemsView, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from functools import lru_cache, partial
from typing import (Any, Literal, NamedTuple, Optional, Protocol, TypeVar,
                    Union, cast)

import numpy as np
import torch
from blake3 import blake3
from PIL.Image import Image
from transformers import BatchFeature, ProcessorMixin
from typing_extensions import assert_never

from vllm.inputs import DummyData, InputProcessingContext
from vllm.logger import init_logger
from vllm.transformers_utils.tokenizer import AnyTokenizer, MistralTokenizer
from vllm.utils import LRUCache, flatten_2d_lists, full_groupby, is_list_of

from .audio import resample_audio
from .inputs import (AudioItem, ImageItem, MultiModalDataDict,
                     MultiModalInputsV2, MultiModalKwargs, NestedTensors,
                     PlaceholderRange, VideoItem)

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

    @staticmethod
    def from_dict(data: MultiModalDataDict) -> "MultiModalDataItems":
        """
        Normalize :class:`MultiModalDataDict` to :class:`MultiModalDataItems`.
        """
        multi_data = MultiModalDataItems()

        for k, v in data.items():
            # TODO: Make a separate modality for embedding inputs
            # to avoid confusion
            # yapf: disable
            if k == "video":
                # Special case since even a single item can be a list
                multi_data[k] = (  # type: ignore[index]
                    v if (isinstance(v, torch.Tensor)
                          or is_list_of(v, list)) else [v]
                )
            elif k in ("image", "audio"):
                multi_data[k] = (  # type: ignore[index]
                    v if isinstance(v, (torch.Tensor, list)) else [v]
                )
            else:
                multi_data[k] = v if isinstance(v, list) else [v]  # type: ignore[index]
            # yapf: enable

        return multi_data

    # NOTE: When a field (e.g. `images`) doesn't exist, directly appending to
    # `self.images` doesn't update this dictionary, which may be confusing
    # We annotate the getter methods as `Sequence` to prevent others from
    # trying to update the list in this way
    @property
    def images(self) -> Sequence[ImageItem]:
        return self.get("image", [])

    @property
    def videos(self) -> Sequence[VideoItem]:
        return self.get("video", [])

    @property
    def audios(self) -> Sequence[AudioItem]:
        return self.get("audio", [])

    def get_item_counts(self) -> Mapping[str, int]:
        return {m: len(items) for m, items in self.items()}

    def get_image_size(self, item_idx: int) -> ImageSize:
        image = self.images[item_idx]

        if isinstance(image, Image):
            return ImageSize(*image.size)
        if isinstance(image, (np.ndarray, torch.Tensor)):
            _, h, w = image.shape
            return ImageSize(w, h)

        assert_never(image)

    def get_audio_with_sr(
        self,
        item_idx: int,
        *,
        default_sr: float,
    ) -> tuple[np.ndarray, float]:
        audio = self.audios[item_idx]

        if isinstance(audio, tuple):
            return audio
        if isinstance(audio, list):
            return np.array(audio), default_sr
        if isinstance(audio, np.ndarray):
            return audio, default_sr

        assert_never(audio)

    def resample_audios(self, new_sr: float, *, drop_sr: bool = True) -> None:
        """
        If :code:`drop_sr=True`, the audio items in this dictionary are updated
        to be NumPy arrays which implicitly means that their sampling rate is
        the same as the model's expected sampling rate; otherwise, they remain
        as :code:`(audio, new_sr)` tuples.
        """
        if not self.audios:
            return

        new_audios = []
        for item_idx in range(len(self.audios)):
            audio, sr = self.get_audio_with_sr(item_idx, default_sr=new_sr)
            audio = resample_audio(audio, orig_sr=sr, target_sr=new_sr)

            new_audios.append(audio if drop_sr else (audio, new_sr))

        self["audio"] = new_audios


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
    mm_item_counts: Mapping[str, int],
) -> list[_S]:
    out_seqs = list[_S]()
    prev_end_idx = 0
    next_idx_by_modality = {modality: 0 for modality in mm_item_counts}

    for match in _resolve_matches(prompt, matches):
        modality = match.modality

        item_idx = next_idx_by_modality[modality]
        if item_idx >= mm_item_counts[modality]:
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
    mm_item_counts: Mapping[str, int],
) -> list[int]:
    """Apply :code:`prompt_repls` to :code:`prompt`."""
    if not matches:
        return prompt

    token_id_seqs = _replace_matches(prompt, matches, mm_item_counts)

    return flatten_2d_lists(token_id_seqs)


def replace_text_matches(
    prompt: str,
    matches: Sequence[_PromptReplacementTextMatch],
    mm_item_counts: Mapping[str, int],
) -> str:
    """Apply :code:`prompt_repls` to :code:`prompt`."""
    if not matches:
        return prompt

    texts = _replace_matches(prompt, matches, mm_item_counts)

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
                if item_index >= modal_item_count:
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
    mm_item_counts: Mapping[str, int],
) -> Iterable[_PlaceholderInfo]:
    """
    Yield each set of placeholder tokens found in :code:`prompt`.

    Note that empty matches are ignored.
    """
    repls_by_modality = dict(full_groupby_modality(prompt_repls))

    for modality, modal_item_count in mm_item_counts.items():
        if modality in repls_by_modality:
            yield from _iter_modality_placeholders(
                prompt,
                modality,
                repls_by_modality[modality],
                modal_item_count,
            )


class ProcessorInputs(NamedTuple):
    """Keyword arguments to :meth:`BaseMultiModalProcessor`"""
    prompt_text: str
    mm_data: MultiModalDataDict
    hf_mm_kwargs: Mapping[str, object]


class ProcessingCache:

    def __init__(self, capacity: int) -> None:
        super().__init__()

        # DEBUG: Set to None to disable
        self.debug_cache_hit_ratio_steps: Optional[int] = None

        self._fine_text_cache = LRUCache[str, BatchFeature](capacity)
        self._fine_mm_cache = LRUCache[str, BatchFeature](capacity)
        self._coarse_cache = LRUCache[str, BatchFeature](capacity)

    def maybe_log_cache_stats(self, cache: LRUCache, name: str) -> None:
        steps = self.debug_cache_hit_ratio_steps
        if not steps:
            return

        cache_stats = cache.stat()
        if cache_stats.total % steps == 0:
            logger.debug("ProcessingCache: %s.hit_ratio = %.2f", name,
                         cache_stats.hit_ratio)

    def _iter_bytes_to_hash(self, key: str, obj: object) -> Iterable[bytes]:
        # Recursive cases
        if isinstance(obj, (list, tuple)):
            for i, elem in enumerate(obj):
                yield from self._iter_bytes_to_hash(f"{key}.{i}", elem)
            return
        if isinstance(obj, dict):
            for k, v in obj.items():
                yield from self._iter_bytes_to_hash(f"{key}.{k}", v)
            return

        # Simple cases
        if isinstance(obj, str):
            yield key.encode("utf-8")
            yield obj.encode("utf-8")
            return
        if isinstance(obj, bytes):
            yield key.encode("utf-8")
            yield obj
            return
        if isinstance(obj, Image):
            yield key.encode("utf-8")
            yield obj.tobytes()
            return

        # Convertible to NumPy arrays
        if isinstance(obj, torch.Tensor):
            obj = obj.numpy()
        if isinstance(obj, (int, float)):
            obj = np.array(obj)
        if isinstance(obj, np.ndarray):
            yield key.encode("utf-8")
            yield obj.tobytes()
            return

        msg = f"Unable to hash object of type {type(obj)}"
        raise NotImplementedError(msg)

    def _hash_kwargs(self, **kwargs: object) -> str:
        hasher = blake3()

        for k, v in kwargs.items():
            for item_bytes in self._iter_bytes_to_hash(k, v):
                hasher.update(item_bytes)

        return hasher.hexdigest()

    def _cached_call_fine(
        self,
        ctx: InputProcessingContext,
        hf_processor: ProcessorMixin,
        text: str,
        mm_data: Mapping[Literal["images", "videos", "audios"], list[Any]],
        mm_kwargs: Mapping[str, object],
    ) -> BatchFeature:
        self.maybe_log_cache_stats(self._fine_text_cache, "fine_text_cache")

        processed_text = self._fine_text_cache.get_or_put(
            text,
            default_factory=partial(
                ctx.call_hf_processor,
                ctx.get_modality_processor(hf_processor, "text"),
                dict(text=text),
            ),
        )

        processed_data = dict(**processed_text)
        for data_key, items in mm_data.items():
            processed_modal_items = defaultdict[str, Union[
                list[torch.Tensor], list[NestedTensors]]](list)

            for item in items:
                self.maybe_log_cache_stats(self._fine_mm_cache,
                                           "fine_mm_cache")

                modal_item = cast(Mapping[str, object], {data_key: item})
                processed_modal_item = self._fine_mm_cache.get_or_put(
                    self._hash_kwargs(**modal_item, **mm_kwargs),
                    default_factory=partial(
                        ctx.call_hf_processor,
                        ctx.get_modality_processor(hf_processor, data_key),
                        modal_item,
                        mm_kwargs,
                    ),
                )

                for k, v in processed_modal_item.items():
                    # Remove the extra batch dimension
                    processed_modal_items[k].append(v[0])

            for k, vs in processed_modal_items.items():
                # Try to merge elements into a single tensor
                if is_list_of(vs, torch.Tensor, check="all") and len(vs) > 0:
                    first_shape = vs[0].shape
                    if all(v.shape == first_shape for v in vs):
                        vs = torch.stack(vs)

                processed_data[k] = vs

        return BatchFeature(processed_data)

    def _cached_call_coarse(
        self,
        ctx: InputProcessingContext,
        hf_processor: ProcessorMixin,
        text: str,
        mm_data: Mapping[str, object],
        mm_kwargs: Mapping[str, object],
    ) -> BatchFeature:
        self.maybe_log_cache_stats(self._coarse_cache, "coarse_cache")

        processed_data = self._coarse_cache.get_or_put(
            self._hash_kwargs(text=text, **mm_data, **mm_kwargs),
            default_factory=partial(
                ctx.call_hf_processor,
                hf_processor,
                dict(text=text, **mm_data),
                mm_kwargs,
            ),
        )

        # Shallow copy to avoid footgun when downstream methods
        # mutate the returned dictionary (since the result is cached)
        return BatchFeature(processed_data)  # type: ignore[arg-type]

    def call_hf_processor(
        self,
        ctx: InputProcessingContext,
        # Assumes that hf_processor has been initialized according to kwargs
        hf_processor: ProcessorMixin,
        text: str,
        mm_data: Mapping[str, object],
        mm_kwargs: Mapping[str, object],
    ) -> BatchFeature:
        # Try to cache each item separately to improve hit rate
        extra_keys = mm_data.keys() - {"images", "videos", "audios"}
        if (mm_data and not extra_keys
                and all(isinstance(v, list) for v in mm_data.values())):
            try:
                return self._cached_call_fine(
                    ctx,
                    hf_processor,
                    text=text,
                    mm_data=mm_data,  # type: ignore[arg-type]
                    mm_kwargs=mm_kwargs,
                )
            except Exception:
                logger.exception(
                    "Failed to apply processor on each item separately! "
                    "Falling back to coarse caching.",
                    stack_info=True,
                )

        return self._cached_call_coarse(
            ctx,
            hf_processor,
            text=text,
            mm_data=mm_data,
            mm_kwargs=mm_kwargs,
        )


class BaseMultiModalProcessor(ABC):
    """
    Abstract base class to process multi-modal inputs to be used in vLLM.
    """

    def __init__(
        self,
        ctx: InputProcessingContext,
        *,
        cache: Optional[ProcessingCache] = None,
    ) -> None:
        super().__init__()

        self.ctx = ctx
        self.cache = cache

    def __call__(
        self,
        prompt: str,
        mm_data: MultiModalDataDict,
        hf_mm_kwargs: Mapping[str, object],
    ) -> MultiModalInputsV2:
        return self.apply(prompt, mm_data, hf_mm_kwargs)

    def _get_hf_processor(self) -> ProcessorMixin:
        """
        Subclasses can add keyword arguments to this method to accept
        additional kwargs from model config or user inputs.
        """
        return self.ctx.get_hf_processor()

    def _get_tokenizer(self) -> AnyTokenizer:
        return self.ctx.tokenizer

    def _get_mm_items(
        self,
        mm_data: MultiModalDataDict,
    ) -> MultiModalDataItems:
        return MultiModalDataItems.from_dict(mm_data)

    @abstractmethod
    def _get_prompt_replacements(
        self,
        mm_items: MultiModalDataItems,
        hf_inputs: BatchFeature,
        hf_mm_kwargs: Mapping[str, object],
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
        mm_item_counts: Mapping[str, int],
    ) -> list[_PlaceholderInfo]:
        return list(
            iter_placeholders(all_prompt_repls, new_token_ids, mm_item_counts))

    def _get_hf_mm_data(
        self,
        mm_items: MultiModalDataItems,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        processor_data = dict[str, Any]()
        passthrough_data = dict[str, Any]()

        for k, v in mm_items.items():
            # TODO: Make a separate modality for embedding inputs
            # to avoid confusion
            if k in ("image", "video", "audio"):
                if isinstance(v, torch.Tensor) and v.ndim == 3:
                    # Pass through embedding inputs (single)
                    passthrough_data[f"{k}_embeds"] = [v]
                elif (is_list_of(v, torch.Tensor) and len(v) > 0
                      and v[0].ndim == 2):
                    # Pass through embedding inputs (multi)
                    passthrough_data[f"{k}_embeds"] = v
                else:
                    # Map keys to plural form, e.g.: image -> images
                    processor_data[f"{k}s"] = v
            else:
                processor_data[k] = v

        return processor_data, passthrough_data

    def _call_hf_processor(
        self,
        prompt: str,
        # Not to be confused with `mm_data` in `self.apply`.
        # This refers to the data to be passed to HF processor.
        mm_data: Mapping[str, object],
        mm_kwargs: Mapping[str, object],
    ) -> BatchFeature:
        if self.cache is None:
            return self.ctx.call_hf_processor(
                self._get_hf_processor(**mm_kwargs),
                dict(text=prompt, **mm_data),
                mm_kwargs,
            )

        return self.cache.call_hf_processor(
            self.ctx,
            self._get_hf_processor(**mm_kwargs),
            text=prompt,
            mm_data=mm_data,
            mm_kwargs=mm_kwargs,
        )

    def _apply_hf_processor(
        self,
        prompt: str,
        mm_items: MultiModalDataItems,
        hf_mm_kwargs: Mapping[str, object],
    ) -> BatchFeature:
        processor_data, passthrough_data = self._get_hf_mm_data(mm_items)

        processed_data = self._call_hf_processor(
            prompt=prompt,
            mm_data=processor_data,
            mm_kwargs=hf_mm_kwargs,
        )
        processed_data.update(passthrough_data)

        return processed_data

    def _bind_prompt_replacements(
        self,
        prompt_repls: list[PromptReplacement],
    ) -> list[_BoundPromptReplacement]:
        tokenizer = self._get_tokenizer()

        return [prompt_repl.bind(tokenizer) for prompt_repl in prompt_repls]

    def _apply_prompt_replacements(
        self,
        token_ids: list[int],
        prompt_repls: Sequence[_BoundPromptReplacement],
        mm_item_counts: Mapping[str, int],
    ) -> tuple[list[int], str, list[_PlaceholderInfo]]:
        tokenizer = self._get_tokenizer()

        token_matches = find_token_matches(token_ids, prompt_repls)
        mm_match_counts = {
            modality: len(matches)
            for modality, matches in full_groupby_modality(token_matches)
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
                token_matches,
                mm_item_counts,
            )

            text = _decode(tokenizer, token_ids)
            matched_repls = [match.prompt_repl for match in token_matches]
        else:
            text = _decode(tokenizer, token_ids)

            text_matches = find_text_matches(text, prompt_repls)
            text = replace_text_matches(
                text,
                text_matches,
                mm_item_counts,
            )

            token_ids = _encode(tokenizer, text)
            matched_repls = [match.prompt_repl for match in text_matches]

        placeholders = self._find_placeholders(matched_repls, token_ids,
                                               mm_item_counts)

        return token_ids, text, placeholders

    def apply(
        self,
        prompt_text: str,
        mm_data: MultiModalDataDict,
        hf_mm_kwargs: Mapping[str, object],
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
        mm_items = self._get_mm_items(mm_data)

        hf_inputs = self._apply_hf_processor(prompt_text, mm_items,
                                             hf_mm_kwargs)
        prompt_ids, = hf_inputs.pop("input_ids").tolist()
        mm_kwargs = MultiModalKwargs(hf_inputs)

        prompt_repls = self._get_prompt_replacements(mm_items, hf_inputs,
                                                     hf_mm_kwargs)
        all_prompt_repls = self._bind_prompt_replacements(prompt_repls)

        # If HF processor already inserts placeholder tokens,
        # there is no need for us to insert them
        mm_item_counts = mm_items.get_item_counts()
        all_placeholders = self._find_placeholders(all_prompt_repls,
                                                   prompt_ids, mm_item_counts)

        if all_placeholders:
            tokenizer = self._get_tokenizer()
            prompt_text = _decode(tokenizer, prompt_ids)
        else:
            (
                prompt_ids,
                prompt_text,
                all_placeholders,
            ) = self._apply_prompt_replacements(
                prompt_ids,
                all_prompt_repls,
                mm_item_counts,
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
