# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import contextvars
import threading
import time
from abc import abstractmethod
from collections.abc import Generator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import (
    TYPE_CHECKING,
    Any,
    overload,
)

import torch
from typing_extensions import TypeVar

from vllm.logger import init_logger
from vllm.tokenizers import TokenizerLike
from vllm.transformers_utils.processor import cached_processor_from_config
from vllm.utils.func_utils import get_allowed_kwarg_only_overrides
from vllm.utils.jsontree import JSONTree, json_map_leaves

if TYPE_CHECKING:
    from transformers.configuration_utils import PretrainedConfig
    from transformers.feature_extraction_utils import BatchFeature
    from transformers.processing_utils import ProcessorMixin

    from vllm.config import ModelConfig, ObservabilityConfig
else:
    PretrainedConfig = object
    BatchFeature = object
    ProcessorMixin = object

    ModelConfig = object
    ObservabilityConfig = object

logger = init_logger(__name__)


_request_id_context: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "_request_id_context", default=None
)


def get_current_request_id() -> str | None:
    """Get the current request_id from the context, if available."""
    return _request_id_context.get()


@contextmanager
def set_request_id(request_id: str) -> Generator[None, None, None]:
    """Context manager to set the request_id for the current context."""
    token = _request_id_context.set(request_id)
    try:
        yield
    finally:
        _request_id_context.reset(token)


@dataclass
class MultiModalProcessorTimingStats:
    """Per-request timing statistics for multimodal processor stages."""

    hf_processor_time: float = 0.0
    """Time spent in HuggingFace processor calls (seconds)."""

    hashing_time: float = 0.0
    """Time spent computing multimodal item hashes (seconds)."""

    cache_lookup_time: float = 0.0
    """Time spent in cache lookups and merges (seconds)."""

    prompt_update_time: float = 0.0
    """Time spent applying prompt updates and finding placeholders (seconds)."""

    preprocessor_total_time: float = 0.0
    """Total preprocessing time (seconds)."""

    def to_dict(self) -> dict[str, float]:
        """Convert stats to a dictionary for JSON serialization."""
        return {
            "hf_processor_time": self.hf_processor_time,
            "hashing_time": self.hashing_time,
            "cache_lookup_time": self.cache_lookup_time,
            "prompt_update_time": self.prompt_update_time,
            "preprocessor_total_time": self.preprocessor_total_time,
        }


def get_timing_stats_from_engine_client(
    engine_client: Any,
) -> dict[str, dict[str, float]]:
    """
    Get all multimodal timing stats from the engine client.

    Collects both preprocessing stats (HF processor, hashing, cache lookup,
    prompt update) and encoder forward pass timing, merged by request_id.

    Args:
        engine_client: The engine client (has input_processor and workers).

    Returns:
        Dictionary mapping request_id to merged stats dict containing
        both preprocessing and encoder timing metrics.

    Example:
        {
            'request-123': {
                'hf_processor_time': 0.45,
                'hashing_time': 0.02,
                'cache_lookup_time': 0.01,
                'prompt_update_time': 0.03,
                'preprocessor_total_time': 0.51,
                'encoder_forward_time': 0.23,
                'num_encoder_calls': 1
            }
        }
    """
    try:
        if not engine_client.vllm_config.observability_config.enable_mm_processor_stats:
            return {}
    except (AttributeError, RuntimeError):
        return {}

    preprocessing_stats = {}
    try:
        input_processor = engine_client.input_processor
        input_preprocessor = input_processor.input_preprocessor

        if hasattr(input_preprocessor, "_get_mm_processor"):
            mm_processor = input_preprocessor._get_mm_processor()
            if mm_processor is not None and hasattr(mm_processor, "info"):
                ctx = mm_processor.info.ctx
                preprocessing_stats = ctx.get_all_timing_stats()
    except (AttributeError, RuntimeError):
        pass

    encoder_stats = {}
    try:
        if hasattr(engine_client, "collective_rpc"):
            encoder_stats_results = engine_client.collective_rpc(
                "get_encoder_timing_stats"
            )
            if encoder_stats_results and len(encoder_stats_results) > 0:
                for worker_stats in encoder_stats_results:
                    if not worker_stats:
                        continue
                    for request_id, stats_dict in worker_stats.items():
                        if request_id not in encoder_stats:
                            encoder_stats[request_id] = dict(stats_dict)
                        else:
                            # Aggregate timing metrics across workers
                            current_time = encoder_stats[request_id].get(
                                "encoder_forward_time", 0.0
                            )
                            new_time = stats_dict.get("encoder_forward_time", 0.0)
                            encoder_stats[request_id]["encoder_forward_time"] = max(
                                current_time, new_time
                            )

                            current_calls = encoder_stats[request_id].get(
                                "num_encoder_calls", 0
                            )
                            new_calls = stats_dict.get("num_encoder_calls", 0)
                            encoder_stats[request_id]["num_encoder_calls"] = max(
                                current_calls, new_calls
                            )
    except (AttributeError, RuntimeError):
        pass

    merged_stats = {}

    for request_id, prep_dict in preprocessing_stats.items():
        merged_stats[request_id] = dict(prep_dict)

    for request_id, enc_dict in encoder_stats.items():
        if request_id in merged_stats:
            merged_stats[request_id].update(enc_dict)
            continue

        # In V1 engine, the request_id in encoder_stats has a suffix
        # appended to the original request_id (which is used in
        # preprocessing_stats).
        # We try to strip the suffix to find the matching request.
        possible_original_id = request_id.rpartition("-")[0]
        if possible_original_id and possible_original_id in merged_stats:
            merged_stats[possible_original_id].update(enc_dict)
        else:
            merged_stats[request_id] = dict(enc_dict)

    return merged_stats


@contextmanager
def timed_preprocessor_operation(ctx: "InputProcessingContext", stage_name: str):
    """
    Context manager to time an operation using the context's timing stats.

    The request_id is automatically retrieved from the context variable,
    so it doesn't need to be passed as a parameter.

    Args:
        ctx: The InputProcessingContext containing the timing stats registry.
        stage_name: Name of the stage being timed.
    """
    request_id = get_current_request_id()
    if ctx is None or request_id is None:
        yield
        return

    stats = ctx.get_timing_stats(request_id)
    if stats is None:
        yield
        return

    start_time = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start_time
        if stage_name == "hf_processor":
            stats.hf_processor_time += elapsed
        elif stage_name == "hashing":
            stats.hashing_time += elapsed
        elif stage_name == "cache_lookup":
            stats.cache_lookup_time += elapsed
        elif stage_name == "prompt_update":
            stats.prompt_update_time += elapsed
        stats.preprocessor_total_time += elapsed


_T = TypeVar("_T")
_C = TypeVar("_C", bound=PretrainedConfig, default=PretrainedConfig)
_P = TypeVar("_P", bound=ProcessorMixin, default=ProcessorMixin)


@dataclass(frozen=True)
class InputProcessingContext:
    """
    Contains information about the model which may be used to
    modify the inputs.
    """

    model_config: ModelConfig
    """The configuration of the model."""

    tokenizer: TokenizerLike | None
    """The tokenizer used to tokenize the inputs."""

    observability_config: "ObservabilityConfig | None" = field(
        default=None, compare=False, repr=False
    )
    """Configuration for observability features."""

    timing_stats_registry: dict[str, MultiModalProcessorTimingStats] = field(
        default_factory=dict, compare=False, repr=False
    )
    """Registry for storing timing stats keyed by request_id."""

    _timing_stats_registry_lock: threading.Lock = field(
        default_factory=threading.Lock, compare=False, repr=False
    )
    """Lock for thread-safe access to timing_stats_registry."""

    def get_tokenizer(self) -> TokenizerLike:
        if self.tokenizer is None:
            raise ValueError(
                "You cannot pass text prompts when `skip_tokenizer_init=True`"
            )

        return self.tokenizer

    @overload
    def get_hf_config(self, /) -> PretrainedConfig: ...

    @overload
    def get_hf_config(
        self,
        typ: type[_C] | tuple[type[_C], ...],
        /,
    ) -> _C: ...

    def get_hf_config(
        self,
        typ: type[Any] | tuple[type[Any], ...] | None = None,
        /,
    ) -> Any:
        """
        Get the HuggingFace configuration
        (`transformers.PretrainedConfig`) of the model,
        additionally checking its type.

        Raises:
            TypeError: If the configuration is not of the specified type.
        """
        if typ is None:
            from transformers.configuration_utils import PretrainedConfig

            typ = PretrainedConfig

        hf_config = self.model_config.hf_config
        if not isinstance(hf_config, typ):
            raise TypeError(
                "Invalid type of HuggingFace config. "
                f"Expected type: {typ}, but "
                f"found type: {type(hf_config)}"
            )

        return hf_config

    def get_hf_image_processor_config(self) -> dict[str, Any]:
        """
        Get the HuggingFace image processor configuration of the model.
        """
        return self.model_config.hf_image_processor_config

    def get_mm_config(self):
        """
        Get the multimodal config of the model.

        Raises:
            RuntimeError: If the model is not a multimodal model.
        """
        mm_config = self.model_config.multimodal_config
        if mm_config is None:
            raise RuntimeError("Not a multimodal model")

        return mm_config

    @overload
    def get_hf_processor(self, /, **kwargs: object) -> ProcessorMixin: ...

    @overload
    def get_hf_processor(
        self,
        typ: type[_P] | tuple[type[_P], ...],
        /,
        **kwargs: object,
    ) -> _P: ...

    def get_hf_processor(
        self,
        typ: type[Any] | tuple[type[Any], ...] | None = None,
        /,
        **kwargs: object,
    ) -> Any:
        """
        Get the HuggingFace processor
        (`transformers.ProcessorMixin`) of the model,
        additionally checking its type.

        Raises:
            TypeError: If the processor is not of the specified type.
        """
        if typ is None:
            from transformers.processing_utils import ProcessorMixin

            typ = ProcessorMixin

        from vllm.tokenizers.mistral import MistralTokenizer

        tokenizer = self.tokenizer
        if isinstance(tokenizer, MistralTokenizer):
            tokenizer = tokenizer.transformers_tokenizer

        return cached_processor_from_config(
            self.model_config,
            processor_cls=typ,
            tokenizer=tokenizer,
            **kwargs,
        )

    def init_processor(
        self,
        typ: type[_T],
        /,
        **kwargs: object,
    ) -> _T:
        """
        Initialize a HuggingFace-like processor class, merging the
        keyword arguments with those in the model's configuration.
        """
        mm_config = self.model_config.get_multimodal_config()
        base_kwargs = mm_config.mm_processor_kwargs
        if base_kwargs is None:
            base_kwargs = {}

        merged_kwargs = {**base_kwargs, **kwargs}

        return typ(**merged_kwargs)

    def _postprocess_output(
        self,
        output: JSONTree,
    ) -> JSONTree:
        def _postprocess_one(x: object):
            if isinstance(x, torch.Tensor):  # noqa: SIM102
                # This mimics the behavior of transformers.BatchFeature
                if x.is_floating_point():
                    x = x.to(dtype=self.model_config.dtype)

            return x

        return json_map_leaves(_postprocess_one, output)

    def call_hf_processor(
        self,
        hf_processor: ProcessorMixin,
        data: Mapping[str, object],
        kwargs: Mapping[str, object] = {},
        *,
        num_tries: int = 1,
        max_tries: int = 5,
    ) -> BatchFeature | JSONTree:
        """
        Call `hf_processor` on the prompt `data`
        (text, image, audio...) with configurable options `kwargs`.
        """
        assert callable(hf_processor)

        mm_config = self.model_config.get_multimodal_config()
        merged_kwargs = mm_config.merge_mm_processor_kwargs(kwargs)

        allowed_kwargs = get_allowed_kwarg_only_overrides(
            hf_processor,
            merged_kwargs,
            requires_kw_only=False,
            allow_var_kwargs=True,
        )

        try:
            output = hf_processor(**data, **allowed_kwargs, return_tensors="pt")
        except Exception as exc:
            # See https://github.com/huggingface/tokenizers/issues/537
            if (
                isinstance(exc, RuntimeError)
                and exc
                and exc.args[0] == "Already borrowed"
                and num_tries < max_tries
            ):
                logger.warning(
                    "Failed to acquire tokenizer in current thread. "
                    "Retrying (%d/%d)...",
                    num_tries,
                    max_tries,
                )
                time.sleep(0.5)
                return self.call_hf_processor(
                    hf_processor,
                    data,
                    kwargs,
                    num_tries=num_tries + 1,
                    max_tries=max_tries,
                )

            msg = (
                f"Failed to apply {type(hf_processor).__name__} "
                f"on data={data} with kwargs={allowed_kwargs}"
            )

            raise ValueError(msg) from exc

        # this emulates output.to(dtype=self.model_config.dtype)
        from transformers.feature_extraction_utils import BatchFeature

        if isinstance(output, BatchFeature):
            output_ = self._postprocess_output(output.data)
            return BatchFeature(output_)

        logger.warning_once(
            "%s did not return `BatchFeature`. "
            "Make sure to match the behaviour of `ProcessorMixin` when "
            "implementing custom processors.",
            type(hf_processor).__name__,
        )

        return self._postprocess_output(output)

    def get_timing_stats(
        self, request_id: str
    ) -> MultiModalProcessorTimingStats | None:
        """
        Get timing stats for a request.
        """
        if (
            self.observability_config is None
            or not self.observability_config.enable_mm_processor_stats
        ):
            return None
        with self._timing_stats_registry_lock:
            return self.timing_stats_registry.get(request_id)

    def create_timing_stats(self, request_id: str) -> MultiModalProcessorTimingStats:
        """
        Create and store timing stats in the registry for a request.

        This should be called at the start of processing for a request.
        The stats object is created immediately and stored in the registry.
        """
        if (
            self.observability_config is None
            or not self.observability_config.enable_mm_processor_stats
        ):
            return MultiModalProcessorTimingStats()

        with self._timing_stats_registry_lock:
            if request_id in self.timing_stats_registry:
                raise ValueError(
                    f"Timing stats already exist for request_id: {request_id}"
                )
            stats = MultiModalProcessorTimingStats()
            self.timing_stats_registry[request_id] = stats
            return stats

    def clear_timing_stats_registry(self) -> int:
        """
        Clear all stats from the registry. Returns the number of stats cleared.
        """
        if (
            self.observability_config is None
            or not self.observability_config.enable_mm_processor_stats
        ):
            return 0
        with self._timing_stats_registry_lock:
            count = len(self.timing_stats_registry)
            self.timing_stats_registry.clear()
            return count

    def get_all_timing_stats(self) -> dict[str, dict[str, float]]:
        """
        Get all timing stats as a dictionary for API endpoints.
        """
        if (
            self.observability_config is None
            or not self.observability_config.enable_mm_processor_stats
        ):
            return {}
        with self._timing_stats_registry_lock:
            return {
                rid: stats.to_dict()
                for rid, stats in self.timing_stats_registry.items()
            }


class BaseProcessingInfo:
    """Base class to provide the information necessary for data processing."""

    def __init__(self, ctx: InputProcessingContext) -> None:
        super().__init__()

        self.ctx = ctx

    @property
    def model_id(self) -> str:
        return self.ctx.model_config.model

    def get_tokenizer(self) -> TokenizerLike:
        return self.ctx.get_tokenizer()

    def get_hf_config(self) -> PretrainedConfig:
        return self.ctx.get_hf_config()

    def get_hf_processor(self, **kwargs: object) -> ProcessorMixin:
        """
        Subclasses can override this method to handle
        specific kwargs from model config or user inputs.
        """
        return self.ctx.get_hf_processor(**kwargs)

    @property
    def skip_prompt_length_check(self) -> bool:
        return False

    @abstractmethod
    def get_supported_mm_limits(self) -> Mapping[str, int | None]:
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

            allowed_limits[modality] = (
                user_limit
                if supported_limit is None
                else min(user_limit, supported_limit)
            )

        return allowed_limits

    def get_mm_max_tokens_per_item(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
    ) -> Mapping[str, int] | None:
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
