from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from typing import Any, Generic, Optional, TypeVar

from transformers import BatchFeature

from vllm import envs
from vllm.transformers_utils.tokenizer import decode_tokens, encode_tokens

from .hasher import MultiModalHasher
from .inputs import (MultiModalDataDict, MultiModalFieldConfig,
                     MultiModalInputsV2, MultiModalKwargs,
                     MultiModalKwargsItem)
from .parse import MultiModalDataItems, MultiModalDataParser
from .processing import (BaseProcessingInfo, BoundPromptReplacement,
                         PlaceholderInfo, ProcessingCache, PromptReplacement,
                         find_mm_placeholders, find_text_matches,
                         find_token_matches, full_groupby_modality,
                         replace_text_matches, replace_token_matches)
from .profiling import BaseDummyDataBuilder

_I = TypeVar("_I", bound=BaseProcessingInfo)


class BaseMultiModalProcessor(ABC, Generic[_I]):
    """
    Abstract base class to process multi-modal inputs to be used in vLLM.

    Not to be confused with :class:`transformers.ProcessorMixin`.
    """

    def __init__(self,
                 info: _I,
                 dummy_data_builder: BaseDummyDataBuilder[_I],
                 *,
                 cache: Optional[ProcessingCache] = None,
                 enable_sanity_checks: bool = True) -> None:
        super().__init__()

        self.info = info
        self.dummy_data_builder = dummy_data_builder
        self.cache = cache
        self.enable_sanity_checks = enable_sanity_checks

        self.data_parser = self._get_data_parser()

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
        mm_prompt_repls: Mapping[str, Sequence[BoundPromptReplacement]],
        new_token_ids: list[int],
        mm_item_counts: Mapping[str, int],
    ) -> Mapping[str, list[PlaceholderInfo]]:
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
        return self.info.ctx.call_hf_processor(
            self.info.get_hf_processor(**mm_kwargs),
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
        dummy_inputs = self.dummy_data_builder.get_dummy_processor_inputs(
            self.info.ctx.model_config.max_model_len,
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
        model_id = self.info.model_id

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
    ) -> dict[str, list[BoundPromptReplacement]]:
        tokenizer = self.info.get_tokenizer()

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
        mm_prompt_repls: Mapping[str, Sequence[BoundPromptReplacement]],
        mm_item_counts: Mapping[str, int],
    ) -> tuple[list[int], str, Mapping[str, list[PlaceholderInfo]]]:
        tokenizer = self.info.get_tokenizer()

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
        mm_placeholders: Mapping[str, list[PlaceholderInfo]],
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

        # Create MM hashes (only used in V1)
        # TODO: Use these hash keys for caching operations in apply_hf_processor
        # instead of rehashing.

        if envs.VLLM_USE_V1:
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

            mm_missing_repls = dict[str, list[BoundPromptReplacement]]()
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
            tokenizer = self.info.get_tokenizer()
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
            mm_hashes=mm_hashes,
            mm_placeholders=mm_placeholder_ranges,
        )
