# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import time
from collections.abc import Mapping
from typing import Any, Literal, cast

from vllm.config import VllmConfig
from vllm.inputs.data import (
    ProcessorInputs,
    PromptType,
    SingletonInputs,
    SingletonPrompt,
)
from vllm.inputs.parse import split_enc_dec_inputs
from vllm.inputs.preprocess import InputPreprocessor
from vllm.logger import init_logger
from vllm.lora.request import LoRARequest
from vllm.multimodal import MULTIMODAL_REGISTRY, MultiModalRegistry
from vllm.multimodal.encoder_budget import MultiModalBudget
from vllm.multimodal.inputs import (
    MultiModalDataDict,
    MultiModalFeatureSpec,
    MultiModalUUIDDict,
)
from vllm.multimodal.parse import ModalityDataItems, MultiModalDataItems
from vllm.multimodal.processing.context import set_request_id
from vllm.multimodal.utils import argsort_mm_positions
from vllm.pooling_params import PoolingParams
from vllm.renderers import BaseRenderer, renderer_from_config
from vllm.renderers.inputs import DictPrompt, TokPrompt
from vllm.sampling_params import SamplingParams
from vllm.tasks import POOLING_TASKS, SupportedTask
from vllm.tokenizers import TokenizerLike
from vllm.utils import length_from_prompt_token_ids_or_embeds, random_uuid
from vllm.utils.torch_utils import set_default_torch_num_threads
from vllm.v1.engine import EngineCoreRequest
from vllm.v1.metrics.stats import MultiModalCacheStats

logger = init_logger(__name__)


class InputProcessor:
    def __init__(
        self,
        vllm_config: VllmConfig,
        renderer: BaseRenderer | None = None,
        *,
        mm_registry: MultiModalRegistry = MULTIMODAL_REGISTRY,
    ) -> None:
        self.vllm_config = vllm_config
        self.model_config = model_config = vllm_config.model_config
        self.cache_config = vllm_config.cache_config
        self.lora_config = vllm_config.lora_config
        self.scheduler_config = vllm_config.scheduler_config
        self.speculative_config = vllm_config.speculative_config
        self.structured_outputs_config = vllm_config.structured_outputs_config
        self.observability_config = vllm_config.observability_config

        self.generation_config_fields = model_config.try_get_generation_config()

        self.renderer = renderer or renderer_from_config(model_config)
        self.mm_registry = mm_registry
        self.mm_processor_cache = mm_registry.processor_cache_from_config(vllm_config)

        self.supports_mm_inputs = mm_registry.supports_multimodal_inputs(model_config)
        self.mm_encoder_cache_size = 0
        self.skip_prompt_length_check = False
        if self.supports_mm_inputs:
            mm_budget = MultiModalBudget(vllm_config, mm_registry)
            self.mm_encoder_cache_size = mm_budget.encoder_cache_size
            self.skip_prompt_length_check = (
                mm_budget.processor.info.skip_prompt_length_check
            )
            mm_budget.reset_cache()  # Not used anymore

        self.input_preprocessor = InputPreprocessor(
            model_config,
            self.observability_config,
            renderer=renderer,
            mm_registry=mm_registry,
            mm_processor_cache=self.mm_processor_cache,
        )

    @property
    def tokenizer(self) -> TokenizerLike | None:
        return self.renderer.tokenizer

    def get_tokenizer(self) -> TokenizerLike:
        return self.renderer.get_tokenizer()

    def _validate_params(
        self,
        params: SamplingParams | PoolingParams,
        # TODO: Validate generation tasks as well once `supported_tasks`
        # is passed to all `process_inputs` calls
        supported_tasks: tuple[SupportedTask, ...] | None,
    ):
        """Raise `ValueError` if SamplingParams or PoolingParams is not valid."""
        if isinstance(params, SamplingParams):
            params.verify(
                self.model_config,
                self.speculative_config,
                self.structured_outputs_config,
                self.tokenizer,
            )
        elif isinstance(params, PoolingParams):
            if supported_tasks is None:
                raise RuntimeError("`supported_tasks` must be passed for pooling")

            supported_pooling_tasks = [
                task for task in supported_tasks if task in POOLING_TASKS
            ]

            if params.task is None:
                if not supported_pooling_tasks:
                    raise ValueError("Pooling tasks are not supported")

                if "token_embed" in supported_pooling_tasks:
                    params.task = "token_embed"
                elif "token_classify" in supported_pooling_tasks:
                    params.task = "token_classify"
                elif "plugin" in supported_pooling_tasks:
                    params.task = "plugin"

            if params.task not in supported_pooling_tasks:
                raise ValueError(
                    f"Unsupported task: {params.task!r} "
                    f"Supported tasks: {supported_pooling_tasks}"
                )

            params.verify(self.model_config)
        else:
            raise TypeError(
                f"params must be either SamplingParams or PoolingParams, "
                f"but got {type(params).__name__}"
            )

    def _parse_mm_items(self, mm_data: MultiModalDataDict) -> MultiModalDataItems:
        mm_processor = self.input_preprocessor._get_mm_processor()
        return mm_processor.info.parse_mm_data(mm_data)

    def _validate_singleton_mm_uuids(self, prompt: SingletonPrompt) -> None:
        if not isinstance(prompt, dict):
            return

        mm_data = cast(MultiModalDataDict, prompt.get("multi_modal_data") or {})
        mm_uuids = cast(MultiModalUUIDDict, prompt.get("multi_modal_uuids") or {})
        if not mm_data and not mm_uuids:
            return

        mm_data_parsed = self._parse_mm_items(
            {k: v for k, v in mm_data.items() if v is not None}
        )
        mm_uuids_parsed = {
            k: [v] if isinstance(v, str) else v
            for k, v in mm_uuids.items()
            if v is not None
        }

        # NOTE: Include the keys corresponding to `None`
        modalities = mm_data.keys() | mm_uuids.keys()

        for modality in modalities:
            data_items = cast(
                ModalityDataItems | list[Any], mm_data_parsed.get(modality, [])
            )
            uuid_items = cast(list[str | None], mm_uuids_parsed.get(modality, []))

            if len(data_items) > 0:
                if len(uuid_items) > 0 and len(data_items) != len(uuid_items):
                    raise ValueError(
                        f"If given, multi_modal_uuids[{modality!r}] must have "
                        f"same length as multi_modal_data[{modality!r}], but "
                        f"got {len(uuid_items)} vs {len(data_items)}."
                    )

                for i, item in enumerate(data_items):
                    if item is None:
                        if not uuid_items:
                            raise ValueError(
                                f"multi_modal_data[{modality!r}][{i}] is empty but "
                                f"multi_modal_uuids[{modality!r}] is missing."
                            )

                        if uuid_items[i] is None:
                            raise ValueError(
                                f"multi_modal_data[{modality!r}][{i}] is empty but "
                                f"multi_modal_uuids[{modality!r}][{i}] is missing."
                            )
            else:
                if len(uuid_items) == 0:
                    raise ValueError(
                        f"multi_modal_data[{modality!r}] is empty but "
                        f"multi_modal_uuids[{modality!r}] is missing."
                    )

    def _validate_mm_uuids(self, prompt: PromptType | DictPrompt | TokPrompt) -> None:
        """
        Validate that user-provided multi_modal_uuids align with
        multi_modal_data in the incoming request prompt(s).
        Only checks lengths; `None` entries are allowed and will be
        auto-hashed downstream.
        """

        if isinstance(prompt, dict) and "encoder_prompt" in prompt:
            self._validate_singleton_mm_uuids(prompt["encoder_prompt"])  # type: ignore[typeddict-item]

            if (dec_prompt := prompt["decoder_prompt"]) is not None:  # type: ignore[typeddict-item]
                self._validate_singleton_mm_uuids(dec_prompt)
        else:
            self._validate_singleton_mm_uuids(prompt)

    def _validate_lora(self, lora_request: LoRARequest | None) -> None:
        if lora_request is None:
            return

        # LoRA request passed in while LoRA is not enabled
        if not self.lora_config:
            raise ValueError(
                f"Got lora_request {lora_request} but LoRA is not enabled!"
            )

        if self.tokenizer is not None:
            logger.warning_once(
                "vLLM has deprecated support for supporting different "
                "tokenizers for different LoRAs. By default, vLLM uses base "
                "model's tokenizer. If you are using a LoRA "
                "with its own tokenizer, consider specifying `--tokenizer "
                "[lora_path]` to use the LoRA tokenizer."
            )

    def _extract_singleton_mm_data(
        self, prompt: SingletonPrompt
    ) -> MultiModalDataDict | None:
        if not isinstance(prompt, dict):
            return None

        return prompt.get("multi_modal_data")

    def _extract_mm_data(
        self, prompt: PromptType | DictPrompt | TokPrompt
    ) -> MultiModalDataDict | None:
        if isinstance(prompt, dict) and "encoder_prompt" in prompt:
            return self._extract_singleton_mm_data(prompt["encoder_prompt"])  # type: ignore[typeddict-item]
        else:
            return self._extract_singleton_mm_data(prompt)

    def _maybe_build_mm_uuids(
        self,
        request_id: str,
        prompt: PromptType | DictPrompt | TokPrompt,
    ) -> MultiModalUUIDDict | None:
        """Build per-item multimodal hash overrides when enabled. In this case,
        multimodal data items are identified by their request id, modality and
        index rather than their content.

        Returns a dictionary of modality -> list[str] of overrides, or None if
        disabled or no multimodal data is present.
        """
        mm_data = self._extract_mm_data(prompt)
        if not mm_data:
            return None

        mm_items = self._parse_mm_items(
            {k: v for k, v in mm_data.items() if v is not None}
        )

        return {
            modality: [f"{request_id}-{modality}-{i}" for i in range(data_count)]
            for modality, data_count in mm_items.get_all_counts().items()
        }

    def _get_mm_identifier(
        self,
        mm_hash: str,
        lora_request: LoRARequest | None,
    ) -> str:
        """
        When enable_tower_connector_lora is True, multi-modal embeddings
        vary depending on the LoRA request. Therefore, the mm_hash must be
        generated based on the LoRA request to prevent incorrect cache hits.
        """
        if (
            lora_request is None
            or self.lora_config is None
            or not self.lora_config.enable_tower_connector_lora
        ):
            return mm_hash
        return f"{lora_request.lora_name}:{mm_hash}"

    @staticmethod
    def assign_request_id(request: EngineCoreRequest):
        """Replace the externally supplied request ID with an internal request ID
        that adds 8 random characters in order to ensure uniquness.
        """
        if request.external_req_id is not None:
            raise ValueError(
                "The external_req_id field should not be set on EngineCoreRequests"
                " passed to vLLM; use the request_id field."
            )
        request.external_req_id = request.request_id
        request.request_id = f"{request.external_req_id}-{random_uuid():.8}"

    def process_inputs(
        self,
        request_id: str,
        prompt: PromptType | DictPrompt | TokPrompt,
        params: SamplingParams | PoolingParams,
        arrival_time: float | None = None,
        lora_request: LoRARequest | None = None,
        tokenization_kwargs: dict[str, Any] | None = None,
        trace_headers: Mapping[str, str] | None = None,
        priority: int = 0,
        data_parallel_rank: int | None = None,
        supported_tasks: tuple[SupportedTask, ...] | None = None,
        resumable: bool = False,
    ) -> EngineCoreRequest:
        self._validate_lora(lora_request)
        self._validate_params(params, supported_tasks)

        parallel_config = self.vllm_config.parallel_config
        dp_size = parallel_config.data_parallel_size
        dp_local_size = parallel_config.data_parallel_size_local
        num_ranks = dp_local_size if parallel_config.local_engines_only else dp_size
        if data_parallel_rank is not None and not (0 <= data_parallel_rank < num_ranks):
            raise ValueError(
                f"data_parallel_rank {data_parallel_rank} "
                f"is out of range [0, {num_ranks})."
            )

        if arrival_time is None:
            arrival_time = time.time()

        # Optionally generate multimodal hash overrides to avoid hashing
        # multimodal data items by their content as their identifiers.

        # NOTE: when users explicitly turn off BOTH prefix caching and input
        # processing caching, no multimodal features or embeddings will be
        # reused across requests, therefore identifying multimodal data items
        # by their content is no longer necessary, and we create uuids with
        # request id-modality-index as multimodal hash overrides.
        if (
            self.model_config.multimodal_config
            and self.model_config.multimodal_config.mm_processor_cache_gb == 0
            and not self.cache_config.enable_prefix_caching
        ):
            mm_uuids = self._maybe_build_mm_uuids(request_id, prompt)
        else:
            # Otherwise, use user-provided uuids as multimodal hash overrides
            # if provided.
            self._validate_mm_uuids(prompt)
            if isinstance(prompt, dict):
                mm_uuids = cast(
                    MultiModalUUIDDict | None, prompt.get("multi_modal_uuids")
                )
            else:
                mm_uuids = None

        # Process inputs, which includes:
        # 1. Tokenize text prompt, with LoRA request if one exists.
        # 2. For multimodal models with a merged preprocessor, preprocess
        #   multimodal data and expand prompt token ids accordingly.
        with set_request_id(request_id), set_default_torch_num_threads():
            processed_inputs: ProcessorInputs = self.input_preprocessor.preprocess(
                prompt,
                tokenization_kwargs=tokenization_kwargs,
                mm_uuids=mm_uuids,
            )

        from vllm.platforms import current_platform

        current_platform.validate_request(
            prompt=prompt,
            params=params,
            processed_inputs=processed_inputs,
        )

        eos_token_id = self.input_preprocessor.get_eos_token_id()

        encoder_inputs, decoder_inputs = split_enc_dec_inputs(processed_inputs)
        self._validate_model_inputs(encoder_inputs, decoder_inputs)

        # Mypy can be conservative for TypedDict unions; normalize access.
        if decoder_inputs["type"] == "embeds":
            prompt_token_ids = None
            prompt_embeds = decoder_inputs["prompt_embeds"]
        else:
            prompt_token_ids = decoder_inputs["prompt_token_ids"]
            prompt_embeds = None

        sampling_params = None
        pooling_params = None
        if isinstance(params, SamplingParams):
            # TODO: can we avoid cloning here in multiproc case?
            sampling_params = params.clone()
            # If unset max tokens, then generate up to the max_model_len.
            if sampling_params.max_tokens is None:
                seq_len = length_from_prompt_token_ids_or_embeds(
                    prompt_token_ids, prompt_embeds
                )
                sampling_params.max_tokens = self.model_config.max_model_len - seq_len

            sampling_params.update_from_generation_config(
                self.generation_config_fields,
                None if self.tokenizer is None else self.tokenizer.eos_token_id,
            )
            if self.tokenizer is not None:
                sampling_params.update_from_tokenizer(self.tokenizer)
        else:
            pooling_params = params.clone()

        # Multimodal related.
        mm_features: list[MultiModalFeatureSpec] | None = None

        if decoder_inputs["type"] == "multimodal":
            decoder_mm_inputs = decoder_inputs["mm_kwargs"]
            decoder_mm_positions = decoder_inputs["mm_placeholders"]
            decoder_mm_hashes = decoder_inputs["mm_hashes"]

            # Merge and flatten multimodal placeholders, hashes and inputs
            # from dictionaries to lists, and sort them by each item's position
            # in the input sequence.
            sorted_mm_idxs = argsort_mm_positions(decoder_mm_positions)

            mm_features = []
            for modality, idx in sorted_mm_idxs:
                base_mm_hash = decoder_mm_hashes[modality][idx]
                mm_features.append(
                    MultiModalFeatureSpec(
                        data=decoder_mm_inputs[modality][idx],
                        modality=modality,
                        identifier=self._get_mm_identifier(
                            base_mm_hash,
                            lora_request,
                        ),
                        mm_position=decoder_mm_positions[modality][idx],
                        mm_hash=base_mm_hash,
                    )
                )

        return EngineCoreRequest(
            request_id=request_id,
            prompt_token_ids=prompt_token_ids,
            prompt_embeds=prompt_embeds,
            mm_features=mm_features,
            sampling_params=sampling_params,
            pooling_params=pooling_params,
            eos_token_id=eos_token_id,
            arrival_time=arrival_time,
            lora_request=lora_request,
            cache_salt=decoder_inputs.get("cache_salt"),
            priority=priority,
            data_parallel_rank=data_parallel_rank,
            trace_headers=trace_headers,
            resumable=resumable,
        )

    def _validate_prompt_len(
        self,
        prompt_len: int,
        prompt_type: Literal["encoder", "decoder"],
    ):
        if self.skip_prompt_length_check:
            return

        if prompt_len == 0 and prompt_type == "decoder":
            raise ValueError(f"The {prompt_type} prompt cannot be empty")

        model_config = self.model_config
        max_prompt_len = (
            model_config.max_model_len
            if prompt_type == "decoder"
            else self.mm_encoder_cache_size
        )
        if prompt_len > max_prompt_len:
            if self.supports_mm_inputs:
                suggestion = (
                    "Make sure that `max_model_len` is no smaller than the "
                    "number of text tokens plus multimodal tokens. For image "
                    "inputs, the number of image tokens depends on the number "
                    "of images, and possibly their aspect ratios as well."
                )
            else:
                suggestion = (
                    "Make sure that `max_model_len` is no smaller than the "
                    "number of text tokens."
                )

            raise ValueError(
                f"The {prompt_type} prompt (length {prompt_len}) is "
                f"longer than the maximum model length of {max_prompt_len}. "
                f"{suggestion}"
            )
        elif prompt_len == max_prompt_len and model_config.runner_type == "generate":
            suggestion = (
                "Make sure that `max_model_len` is no smaller than the "
                "number of text tokens (prompt + requested output tokens)."
            )
            raise ValueError(
                f"The {prompt_type} prompt (length {prompt_len}) plus the number of "
                f"requested output tokens (at least 1) is longer than the maximum "
                f"model length of {max_prompt_len}. {suggestion}"
            )

    def _validate_model_input(
        self,
        prompt_inputs: SingletonInputs,
        prompt_type: Literal["encoder", "decoder"],
    ) -> None:
        model_config = self.model_config
        tokenizer = self.tokenizer

        prompt_ids = (
            None
            if prompt_inputs["type"] == "embeds"
            else prompt_inputs["prompt_token_ids"]
        )
        prompt_embeds = (
            prompt_inputs["prompt_embeds"]
            if prompt_inputs["type"] == "embeds"
            else None
        )

        prompt_len = length_from_prompt_token_ids_or_embeds(prompt_ids, prompt_embeds)
        self._validate_prompt_len(prompt_len, prompt_type)

        if prompt_inputs["type"] == "multimodal":
            decoder_mm_positions = prompt_inputs["mm_placeholders"]
            for modality, mm_positions in decoder_mm_positions.items():
                for mm_position in mm_positions:
                    embed_length = mm_position.get_num_embeds()
                    if embed_length > self.mm_encoder_cache_size:
                        raise ValueError(
                            f"The {prompt_type} prompt contains a(n) {modality} item "
                            f"with length {embed_length}, which exceeds the "
                            f"pre-allocated encoder cache size "
                            f"{self.mm_encoder_cache_size}. Please reduce the input "
                            f"size or increase the encoder cache size "
                            f"by setting --limit-mm-per-prompt at startup."
                        )

        if prompt_ids and tokenizer is not None:
            max_input_id = max(prompt_ids, default=0)

            # NOTE: tokenizer.max_token_id is the tokenizer’s vocab size while
            # self.model_config.get_vocab_size() is the model’s vocab size.
            # For Qwen3 models, the language model has extra tokens that do
            # not exist in the tokenizer, and vice versa for multimodal
            # placeholder tokens in some multimodal models.
            # See https://github.com/QwenLM/Qwen3/issues/29#issuecomment-1933720399 # noqa: E501
            # and https://github.com/vllm-project/vllm/pull/22471#discussion_r2312251421 # noqa: E501

            # Here we take the max of the two to determine if a token id is
            # truly out-of-vocabulary.
            model_vocab_size = model_config.get_vocab_size()
            if max_input_id > max(tokenizer.max_token_id, model_vocab_size - 1):
                raise ValueError(f"Token id {max_input_id} is out of vocabulary")

    def _validate_model_inputs(
        self,
        encoder_inputs: SingletonInputs | None,
        decoder_inputs: SingletonInputs,
    ):
        if encoder_inputs is not None:
            self._validate_model_input(encoder_inputs, prompt_type="encoder")

        self._validate_model_input(decoder_inputs, prompt_type="decoder")

    def stat_mm_cache(self) -> MultiModalCacheStats | None:
        return self.input_preprocessor.stat_mm_cache()

    def clear_mm_cache(self) -> None:
        self.input_preprocessor.clear_mm_cache()

    def close(self) -> None:
        if self.mm_processor_cache is not None:
            self.mm_processor_cache.close()
