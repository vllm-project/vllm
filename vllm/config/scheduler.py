# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import hashlib
import importlib
import inspect
import json
import os
from collections.abc import Callable
from dataclasses import InitVar
from typing import Any, Literal

from pydantic import Field, SkipValidation, field_validator, model_validator
from pydantic.dataclasses import dataclass
from typing_extensions import Self

from vllm.config.utils import config
from vllm.logger import init_logger
from vllm.utils import (
    DEFAULT_MAX_NUM_BATCHED_TOKENS,
    MULTIMODAL_MODEL_MAX_NUM_BATCHED_TOKENS,
    POOLING_MODEL_MAX_NUM_BATCHED_TOKENS,
)

logger = init_logger(__name__)

RunnerType = Literal["generate", "pooling", "draft"]
SchedulerPolicy = Literal["fcfs", "priority"]


@config
@dataclass
class SchedulerConfig:
    """Scheduler configuration."""

    runner_type: RunnerType = "generate"
    """The runner type to launch for the model."""

    max_num_batched_tokens: SkipValidation[int] = None  # type: ignore
    """Maximum number of tokens to be processed in a single iteration.

    This config has no static default. If left unspecified by the user, it will
    be set in `EngineArgs.create_engine_config` based on the usage context."""

    max_num_seqs: SkipValidation[int] = None  # type: ignore
    """Maximum number of sequences to be processed in a single iteration.

    This config has no static default. If left unspecified by the user, it will
    be set in `EngineArgs.create_engine_config` based on the usage context."""

    max_model_len: SkipValidation[int] = None  # type: ignore
    """Maximum length of a sequence (including prompt and generated text). This
    is primarily set in `ModelConfig` and that value should be manually
    duplicated here."""

    max_num_partial_prefills: int = 1
    """For chunked prefill, the maximum number of sequences that can be
    partially prefilled concurrently."""

    max_long_partial_prefills: int = 1
    """For chunked prefill, the maximum number of prompts longer than
    long_prefill_token_threshold that will be prefilled concurrently. Setting
    this less than max_num_partial_prefills will allow shorter prompts to jump
    the queue in front of longer prompts in some cases, improving latency."""

    long_prefill_token_threshold: int = 0
    """For chunked prefill, a request is considered long if the prompt is
    longer than this number of tokens."""

    num_lookahead_slots: int = 0
    """The number of slots to allocate per sequence per
    step, beyond the known token ids. This is used in speculative
    decoding to store KV activations of tokens which may or may not be
    accepted.

    NOTE: This will be replaced by speculative config in the future; it is
    present to enable correctness tests until then."""

    enable_chunked_prefill: SkipValidation[bool] = None  # type: ignore
    """If True, prefill requests can be chunked based
    on the remaining max_num_batched_tokens."""

    is_multimodal_model: bool = False
    """True if the model is multimodal."""

    is_encoder_decoder: InitVar[bool] = False
    """True if the model is an encoder-decoder model.

    Note: This is stored in the ModelConfig, and is used only here to
    disable chunked prefill and prefix caching for encoder-decoder models.
    """

    # TODO (ywang96): Make this configurable.
    max_num_encoder_input_tokens: int = Field(init=False)
    """Multimodal encoder compute budget, only used in V1.

    NOTE: This is not currently configurable. It will be overridden by
    max_num_batched_tokens in case max multimodal embedding size is larger."""

    # TODO (ywang96): Make this configurable.
    encoder_cache_size: int = Field(init=False)
    """Multimodal encoder cache size, only used in V1.

    NOTE: This is not currently configurable. It will be overridden by
    max_num_batched_tokens in case max multimodal embedding size is larger."""

    policy: SchedulerPolicy = "fcfs"
    """The scheduling policy to use:\n
    - "fcfs" means first come first served, i.e. requests are handled in order
    of arrival.\n
    - "priority" means requests are handled based on given priority (lower
    value means earlier handling) and time of arrival deciding any ties)."""

    chunked_prefill_enabled: bool = Field(init=False)
    """True if chunked prefill is enabled."""

    disable_chunked_mm_input: bool = False
    """If set to true and chunked prefill is enabled, we do not want to
    partially schedule a multimodal item. Only used in V1
    This ensures that if a request has a mixed prompt
    (like text tokens TTTT followed by image tokens IIIIIIIIII) where only
    some image tokens can be scheduled (like TTTTIIIII, leaving IIIII),
    it will be scheduled as TTTT in one step and IIIIIIIIII in the next."""

    # scheduler class or path. "vllm.v1.core.sched.scheduler.Scheduler"
    # (default) or "mod.custom_class".
    scheduler_cls: str | type[object] = "vllm.v1.core.sched.scheduler.Scheduler"
    """The scheduler class to use. "vllm.v1.core.sched.scheduler.Scheduler" is
    the default scheduler. Can be a class directly or the path to a class of
    form "mod.custom_class"."""

    external_parameters: dict | None = None
    """ A dictionary of external parameters for custom scheduler implementations.
    If a user-defined scheduler requires additional configuration values, they
    can be provided here directly as a dict. Alternatively, a JSON file can be
    placed in the directory of the class specified by ``scheduler_cls``; in that
    case, the file will be automatically loaded and its contents stored in this
    field.
    """

    disable_hybrid_kv_cache_manager: bool = False
    """If set to True, KV cache manager will allocate the same size of KV cache
    for all attention layers even if there are multiple type of attention layers
    like full attention and sliding window attention.
    """

    async_scheduling: bool = False
    """EXPERIMENTAL: If set to True, perform async scheduling. This may help
    reduce the CPU overheads, leading to better latency and throughput. However,
    async scheduling is currently not supported with some features such as
    structured outputs, speculative decoding, and pipeline parallelism.
    """

    def compute_hash(self) -> str:
        """
        WARNING: Whenever a new field is added to this config,
        ensure that it is included in the factors list if
        it affects the computation graph.

        Provide a hash that uniquely identifies all the configs
        that affect the structure of the computation
        graph from input ids/embeddings to the final hidden states,
        excluding anything before input ids/embeddings and after
        the final hidden states.
        """
        # no factors to consider.
        # this config will not affect the computation graph.
        factors: list[Any] = []
        hash_str = hashlib.md5(str(factors).encode(), usedforsecurity=False).hexdigest()
        return hash_str

    @field_validator(
        "max_num_batched_tokens",
        "max_num_seqs",
        "max_model_len",
        "enable_chunked_prefill",
        mode="wrap",
    )
    @classmethod
    def _skip_none_validation(cls, value: Any, handler: Callable) -> Any:
        """Skip validation if the value is `None` when initialisation is delayed."""
        if value is None:
            return value
        return handler(value)

    def __post_init__(self, is_encoder_decoder: bool) -> None:
        if self.max_model_len is None:
            self.max_model_len = 8192

        if self.max_num_seqs is None:
            self.max_num_seqs = 128

        if is_encoder_decoder:
            # Chunked prefill should be disabled for encoder-decoder models.
            self.disable_chunked_mm_input = True
            self.chunked_prefill_enabled = False
            self.enable_chunked_prefill = False
            self.long_prefill_token_threshold = 0
            logger.info(
                "Encoder-decoder models do not support chunked prefill nor"
                " prefix caching; disabling both."
            )

        if self.max_num_batched_tokens is None:
            if self.enable_chunked_prefill:
                self.max_num_batched_tokens = DEFAULT_MAX_NUM_BATCHED_TOKENS
            else:
                # If max_model_len is too short, use
                # DEFAULT_MAX_NUM_BATCHED_TOKENS as the default value
                # for higher throughput.
                self.max_num_batched_tokens = max(
                    self.max_model_len, DEFAULT_MAX_NUM_BATCHED_TOKENS
                )

            if self.runner_type == "pooling":
                # Choose specific value for higher throughput
                self.max_num_batched_tokens = max(
                    self.max_num_batched_tokens,
                    POOLING_MODEL_MAX_NUM_BATCHED_TOKENS,
                )
            if self.is_multimodal_model:
                # The value needs to be at least the number of multimodal tokens
                self.max_num_batched_tokens = max(
                    self.max_num_batched_tokens,
                    MULTIMODAL_MODEL_MAX_NUM_BATCHED_TOKENS,
                )

            # When using default settings,
            # Ensure max_num_batched_tokens does not exceed model limit.
            # Some models (e.g., Whisper) have embeddings tied to max length.
            self.max_num_batched_tokens = min(
                self.max_num_seqs * self.max_model_len, self.max_num_batched_tokens
            )

        self.max_num_encoder_input_tokens = self.max_num_batched_tokens
        self.encoder_cache_size = self.max_num_batched_tokens

        if self.enable_chunked_prefill:
            logger.info(
                "Chunked prefill is enabled with max_num_batched_tokens=%d.",
                self.max_num_batched_tokens,
            )

        self.chunked_prefill_enabled = self.enable_chunked_prefill
        if self.max_num_partial_prefills > 1:
            if self.long_prefill_token_threshold == 0:
                self.long_prefill_token_threshold = int(self.max_model_len * 0.04)

            logger.info(
                "Concurrent partial prefills enabled with "
                "max_num_partial_prefills=%d, max_long_partial_prefills=%d, "
                "long_prefill_token_threshold=%d",
                self.max_num_partial_prefills,
                self.max_long_partial_prefills,
                self.long_prefill_token_threshold,
            )

        if self.async_scheduling:
            ewsjf_scheduler_cls = (
                "vllm.v1.core.sched.ewsjf_scheduler.scheduler.EWSJFScheduler"
            )

            if self.scheduler_cls == ewsjf_scheduler_cls:
                self.scheduler_cls = (
                    "vllm.v1.core.sched.ewsjf_scheduler."
                    "async_scheduler.AsyncEWSJFScheduler"
                )
            else:
                self.scheduler_cls = "vllm.v1.core.sched.async_scheduler.AsyncScheduler"

        if isinstance(self.scheduler_cls, str):
            self.load_external_parameters()

    def load_external_parameters(self):
        module_name, class_name = self.scheduler_cls.rsplit(".", 1)
        module = importlib.import_module(module_name)
        cls = getattr(module, class_name)

        module_file = inspect.getfile(cls)
        module_dir = os.path.dirname(module_file)

        config_path = os.path.join(module_dir, "config.json")
        # Load config.json only if no external parameters were provided
        if self.external_parameters is None and os.path.exists(config_path):
            with open(config_path, encoding="utf-8") as f:
                self.external_parameters = json.load(f)

    @model_validator(mode="after")
    def _verify_args(self) -> Self:
        if (
            self.max_num_batched_tokens < self.max_model_len
            and not self.chunked_prefill_enabled
        ):
            raise ValueError(
                f"max_num_batched_tokens ({self.max_num_batched_tokens}) is "
                f"smaller than max_model_len ({self.max_model_len}). "
                "This effectively limits the maximum sequence length to "
                "max_num_batched_tokens and makes vLLM reject longer "
                "sequences. Please increase max_num_batched_tokens or "
                "decrease max_model_len."
            )

        if self.max_num_batched_tokens < self.max_num_seqs:
            raise ValueError(
                f"max_num_batched_tokens ({self.max_num_batched_tokens}) must "
                "be greater than or equal to max_num_seqs "
                f"({self.max_num_seqs})."
            )

        if self.max_num_batched_tokens > self.max_num_seqs * self.max_model_len:
            logger.warning(
                "max_num_batched_tokens (%d) exceeds max_num_seqs "
                "* max_model_len (%d). This may lead to unexpected behavior.",
                self.max_num_batched_tokens,
                self.max_num_seqs * self.max_model_len,
            )

        if self.num_lookahead_slots < 0:
            raise ValueError(
                "num_lookahead_slots "
                f"({self.num_lookahead_slots}) must be greater than or "
                "equal to 0."
            )

        if self.max_num_partial_prefills < 1:
            raise ValueError(
                f"max_num_partial_prefills ({self.max_num_partial_prefills}) "
                "must be greater than or equal to 1."
            )
        elif self.max_num_partial_prefills > 1:
            if not self.chunked_prefill_enabled:
                raise ValueError(
                    "Chunked prefill must be enabled to set "
                    "max_num_partial_prefills > 1."
                )

            if self.long_prefill_token_threshold > self.max_model_len:
                raise ValueError(
                    "long_prefill_token_threshold "
                    f"({self.long_prefill_token_threshold}) cannot be greater "
                    f"than the max_model_len ({self.max_model_len})."
                )

        if (self.max_long_partial_prefills < 1) or (
            self.max_long_partial_prefills > self.max_num_partial_prefills
        ):
            raise ValueError(
                f"max_long_partial_prefills ({self.max_long_partial_prefills}) "
                "must be greater than or equal to 1 and less than or equal to "
                f"max_num_partial_prefills ({self.max_num_partial_prefills})."
            )

        return self
