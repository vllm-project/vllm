# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Callable
from dataclasses import InitVar
from typing import TYPE_CHECKING, Any, ClassVar, Literal, cast

from pydantic import Field, field_validator
from typing_extensions import Self

from vllm.config.utils import config
from vllm.logger import init_logger
from vllm.utils.hashing import safe_hash
from vllm.utils.import_utils import resolve_obj_by_qualname

if TYPE_CHECKING:
    from vllm.v1.core.sched.interface import SchedulerInterface

logger = init_logger(__name__)

RunnerType = Literal["generate", "pooling", "draft"]
SchedulerPolicy = Literal["fcfs", "priority"]


@config
class SchedulerConfig:
    """Scheduler configuration."""

    max_model_len: InitVar[int]
    """Maximum length of a sequence (including prompt and generated text).

    Note: This is stored in the ModelConfig, and is used only here to
    provide fallbacks and validate other attributes."""

    is_encoder_decoder: InitVar[bool]
    """True if the model is an encoder-decoder model.

    Note: This is stored in the ModelConfig, and is used only here to
    disable chunked prefill and prefix caching for encoder-decoder models.
    """

    DEFAULT_MAX_NUM_BATCHED_TOKENS: ClassVar[int] = 2048
    DEFAULT_MAX_NUM_SEQS: ClassVar[int] = 128

    runner_type: RunnerType = "generate"
    """The runner type to launch for the model."""

    max_num_batched_tokens: int = Field(default=DEFAULT_MAX_NUM_BATCHED_TOKENS, ge=1)
    """Maximum number of tokens to be processed in a single iteration.

    The default value here is mainly for convenience when testing.
    In real usage, this should be set in `EngineArgs.create_engine_config`.
    """

    max_num_seqs: int = Field(default=DEFAULT_MAX_NUM_SEQS, ge=1)
    """Maximum number of sequences to be processed in a single iteration.

    The default value here is mainly for convenience when testing.
    In real usage, this should be set in `EngineArgs.create_engine_config`.
    """

    max_num_partial_prefills: int = Field(default=1, ge=1)
    """For chunked prefill, the maximum number of sequences that can be
    partially prefilled concurrently."""

    max_long_partial_prefills: int = Field(default=1, ge=1)
    """For chunked prefill, the maximum number of prompts longer than
    long_prefill_token_threshold that will be prefilled concurrently. Setting
    this less than max_num_partial_prefills will allow shorter prompts to jump
    the queue in front of longer prompts in some cases, improving latency."""

    long_prefill_token_threshold: int = 0
    """For chunked prefill, a request is considered long if the prompt is
    longer than this number of tokens."""

    enable_chunked_prefill: bool = True
    """If True, prefill requests can be chunked based
    on the remaining `max_num_batched_tokens`.

    The default value here is mainly for convenience when testing.
    In real usage, this should be set in `EngineArgs.create_engine_config`.
    """

    is_multimodal_model: bool = False
    """True if the model is multimodal."""

    max_num_batched_encoder_embeds: int | None = None
    """Maximum number of encoder embeddings to be processed in a single iteration.

    Defaults to `max_num_batched_tokens`. This will be overridden by the
    maximum possible multimodal embedding count of the model if it is larger,
    in order to ensure that a multimodal item will be eventually scheduled
    regardless of its embedding count.

    Note that the number of encoder embeddings might be smaller than the
    number of input tokens used to represent the multimodal input; see
    [vllm.v1.core.encoder_cache_manager.EncoderCacheManager][] for more details.
    """

    encoder_cache_size: int | None = None
    """Maximum number of encoder embeddings that can be stored in the encoder cache;
    it must be no smaller than `max_num_batched_encoder_embeds`.

    Defaults to `max_num_batched_encoder_embeds`. This will be overridden by the
    maximum possible multimodal embedding count of the model if it is larger,
    in order to ensure that a multimodal item will be eventually scheduled
    regardless of its embedding count.

    Note that the number of encoder embeddings might be smaller than the
    number of input tokens used to represent the multimodal input; see
    [vllm.v1.core.encoder_cache_manager.EncoderCacheManager][] for more details.
    """

    policy: SchedulerPolicy = "fcfs"
    """The scheduling policy to use:\n
    - "fcfs" means first come first served, i.e. requests are handled in order
    of arrival.\n
    - "priority" means requests are handled based on given priority (lower
    value means earlier handling) and time of arrival deciding any ties)."""

    disable_chunked_mm_input: bool = False
    """If set to true and chunked prefill is enabled, we do not want to
    partially schedule a multimodal item.

    This ensures that if a request has a mixed prompt
    (like text tokens `TTTT` followed by image tokens `IIIIIIIIII`) where only
    some image tokens can be scheduled (like `TTTTIIIII`, leaving `IIIII`),
    it will be scheduled as `TTTT` in one step and `IIIIIIIIII` in the next."""

    # scheduler class or path. "vllm.v1.core.sched.scheduler.Scheduler"
    # (default) or "mod.custom_class".
    scheduler_cls: str | type[object] = Field(default=None)
    """The scheduler class to use. "vllm.v1.core.sched.scheduler.Scheduler" is
    the default scheduler. Can be a class directly or the path to a class of
    form "mod.custom_class"."""

    disable_hybrid_kv_cache_manager: bool | None = None
    """If set to True, KV cache manager will allocate the same size of KV cache
    for all attention layers even if there are multiple type of attention layers
    like full attention and sliding window attention.
    If set to None, the default value will be determined based on the environment
    and starting configuration.
    """

    async_scheduling: bool = Field(default=None)
    """If set to False, disable async scheduling. Async scheduling helps to
    avoid gaps in GPU utilization, leading to better latency and throughput.
    """

    stream_interval: int = Field(default=1, ge=1)
    """The interval (or buffer size) for streaming in terms of token length.
    A smaller value (1) makes streaming smoother by sending each token immediately,
    while a larger value (e.g., 10) reduces host overhead and may increase throughput
    by batching multiple tokens before sending."""

    @staticmethod
    def default_factory(**kwargs):
        """
        Factory method to create `SchedulerConfig` with default values for `InitVar`s.
        """
        if "max_model_len" not in kwargs:
            kwargs["max_model_len"] = 8192
        if "is_encoder_decoder" not in kwargs:
            kwargs["is_encoder_decoder"] = False
        return SchedulerConfig(**kwargs)

    def get_scheduler_cls(self) -> type["SchedulerInterface"]:
        if self.scheduler_cls is None:
            if self.async_scheduling:
                from vllm.v1.core.sched.async_scheduler import AsyncScheduler

                return AsyncScheduler
            from vllm.v1.core.sched.scheduler import Scheduler

            return Scheduler

        # This warning can be removed once the Scheduler interface is
        # finalized and we can maintain support for scheduler classes that
        # implement it
        logger.warning_once(
            "Using custom scheduler class %s. This scheduler interface is "
            "not public and compatibility may not be maintained.",
            self.scheduler_cls,
        )
        if not isinstance(self.scheduler_cls, str):
            return cast(type["SchedulerInterface"], self.scheduler_cls)
        return resolve_obj_by_qualname(self.scheduler_cls)

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
        factors: list[Any] = []

        # max_num_batched_tokens need to be included in the hash due
        # to two reasons:
        # 1. LoRA creates static buffers based on max_num_batched_tokens.
        #   The tensor sizes and strides get captured in the torch.compile
        #   graph explicitly.
        # 2. Inductor decides whether using 32-bit or 64-bit indexing integer
        #   based on the data sizes. `max_num_batched_tokens` has an
        #   impact on that. For more details, please check
        #   https://github.com/vllm-project/vllm/issues/29585
        factors.append(self.max_num_batched_tokens)

        hash_str = safe_hash(str(factors).encode(), usedforsecurity=False).hexdigest()
        return hash_str

    @field_validator("scheduler_cls", "async_scheduling", mode="wrap")
    @classmethod
    def _skip_none_validation(cls, value: Any, handler: Callable) -> Any:
        """Skip validation if the value is `None` when initialisation is delayed."""
        return None if value is None else handler(value)

    def __post_init__(self, max_model_len: int, is_encoder_decoder: bool) -> None:
        if is_encoder_decoder:
            # Chunked prefill should be disabled for encoder-decoder models.
            self.disable_chunked_mm_input = True
            self.enable_chunked_prefill = False
            self.long_prefill_token_threshold = 0
            logger.info(
                "Encoder-decoder models do not support chunked prefill nor"
                " prefix caching; disabling both."
            )

        if self.enable_chunked_prefill:
            logger.info(
                "Chunked prefill is enabled with max_num_batched_tokens=%d.",
                self.max_num_batched_tokens,
            )

        if self.max_num_partial_prefills > 1:
            if self.long_prefill_token_threshold == 0:
                self.long_prefill_token_threshold = int(max_model_len * 0.04)

            logger.info(
                "Concurrent partial prefills enabled with "
                "max_num_partial_prefills=%d, max_long_partial_prefills=%d, "
                "long_prefill_token_threshold=%d",
                self.max_num_partial_prefills,
                self.max_long_partial_prefills,
                self.long_prefill_token_threshold,
            )

        self.verify_max_model_len(max_model_len)

    def verify_max_model_len(self, max_model_len: int) -> Self:
        if (
            self.max_num_batched_tokens < max_model_len
            and not self.enable_chunked_prefill
        ):
            raise ValueError(
                f"max_num_batched_tokens ({self.max_num_batched_tokens}) is "
                f"smaller than max_model_len ({max_model_len}). "
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

        if self.max_num_batched_tokens > self.max_num_seqs * max_model_len:
            logger.warning(
                "max_num_batched_tokens (%d) exceeds max_num_seqs "
                "* max_model_len (%d). This may lead to unexpected behavior.",
                self.max_num_batched_tokens,
                self.max_num_seqs * max_model_len,
            )

        if self.max_num_partial_prefills > 1:
            if not self.enable_chunked_prefill:
                raise ValueError(
                    "Chunked prefill must be enabled to set "
                    "max_num_partial_prefills > 1."
                )

            if self.long_prefill_token_threshold > max_model_len:
                raise ValueError(
                    "long_prefill_token_threshold "
                    f"({self.long_prefill_token_threshold}) cannot be greater "
                    f"than the max_model_len ({max_model_len})."
                )

        if self.max_long_partial_prefills > self.max_num_partial_prefills:
            raise ValueError(
                f"{self.max_long_partial_prefills=} must be less than or equal to "
                f"{self.max_num_partial_prefills=}."
            )

        return self
