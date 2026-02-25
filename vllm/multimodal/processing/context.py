# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import time
from abc import abstractmethod
from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from functools import cached_property
from typing import TYPE_CHECKING, Any, overload

import torch
from typing_extensions import TypeVar

from vllm.logger import init_logger
from vllm.multimodal.inputs import MultiModalDataDict
from vllm.multimodal.parse import (
    DictEmbeddingItems,
    EmbeddingItems,
    MultiModalDataItems,
    MultiModalDataParser,
)
from vllm.renderers import TokenizeParams
from vllm.tokenizers import TokenizerLike
from vllm.transformers_utils.processor import cached_processor_from_config
from vllm.utils.func_utils import get_allowed_kwarg_only_overrides
from vllm.utils.jsontree import JSONTree, json_map_leaves
from vllm.utils.mistral import is_mistral_tokenizer

if TYPE_CHECKING:
    from transformers.configuration_utils import PretrainedConfig
    from transformers.feature_extraction_utils import BatchFeature
    from transformers.processing_utils import ProcessorMixin

    from vllm.config import ModelConfig
else:
    PretrainedConfig = object
    BatchFeature = object
    ProcessorMixin = object

    ModelConfig = object

logger = init_logger(__name__)


@dataclass
class TimingContext:
    """Helper class to record execution times during multi-modal processing."""

    enabled: bool = True
    """If disabled, `TimingContext.record` becomes a no-op."""

    stage_secs: dict[str, float] = field(default_factory=dict)
    """The execution time (in seconds) for each processing stage."""

    @property
    def total_secs(self) -> float:
        return sum(self.stage_secs.values())

    @contextmanager
    def record(self, stage: str):
        """Record the execution time for a processing stage."""
        if not self.enabled:
            yield
            return

        start_time = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - start_time
            self.stage_secs.setdefault(stage, 0.0)
            self.stage_secs[stage] += elapsed

    def get_stats_dict(self):
        stats_dict = {
            f"{stage}_secs": time_s for stage, time_s in self.stage_secs.items()
        }
        stats_dict["preprocessor_total_secs"] = self.total_secs

        return stats_dict


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

        tokenizer = self.tokenizer
        if is_mistral_tokenizer(tokenizer):
            tokenizer = tokenizer.transformers_tokenizer

        merged_kwargs = self.get_merged_mm_kwargs(kwargs)
        merged_kwargs.pop("tokenizer", None)

        return cached_processor_from_config(
            self.model_config,
            processor_cls=typ,
            tokenizer=tokenizer,
            **merged_kwargs,
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
        merged_kwargs = self.get_merged_mm_kwargs(kwargs)

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

    def get_merged_mm_kwargs(self, kwargs: Mapping[str, object]):
        mm_config = self.model_config.get_multimodal_config()
        return mm_config.merge_mm_processor_kwargs(kwargs)

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

        merged_kwargs = self.get_merged_mm_kwargs(kwargs)

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

    def get_default_tok_params(self) -> TokenizeParams:
        """Construct the default parameters for tokenization."""
        model_config = self.ctx.model_config
        encoder_config = model_config.encoder_config or {}

        return TokenizeParams(
            max_total_tokens=model_config.max_model_len,
            do_lower_case=encoder_config.get("do_lower_case", False),
            add_special_tokens=True,
        )

    @cached_property
    def default_tok_params(self) -> TokenizeParams:
        return self.get_default_tok_params()

    def _get_expected_hidden_size(self) -> int | None:
        """
        Get expected hidden size for embedding validation if `mm_embeds` are enabled.

        This validates hidden dimensions to prevent a vulnerability where embeddings
        with correct `ndim` but wrong `shape` could cause crashes at inference time.
        """
        model_config = self.ctx.model_config
        mm_config = model_config.get_multimodal_config()

        if mm_config.enable_mm_embeds:
            return model_config.get_inputs_embeds_size()

        return None

    def get_data_parser(self) -> MultiModalDataParser:
        """
        Constructs a parser to preprocess multi-modal data items
        before passing them to
        [`_get_hf_mm_data`][vllm.multimodal.processing.BaseMultiModalProcessor._get_hf_mm_data].

        You can support additional modalities by creating a subclass
        of [`MultiModalDataParser`][vllm.multimodal.parse.MultiModalDataParser]
        that has additional subparsers.
        """
        return MultiModalDataParser(
            expected_hidden_size=self._get_expected_hidden_size(),
        )

    @cached_property
    def data_parser(self) -> MultiModalDataParser:
        return self.get_data_parser()

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

    @cached_property
    def supported_mm_limits(self) -> Mapping[str, int | None]:
        """The maximum supported number of items for each modality."""
        return self.get_supported_mm_limits()

    @cached_property
    def allowed_mm_limits(self) -> Mapping[str, int]:
        """The maximum allowed number of items for each modality."""
        mm_config = self.ctx.get_mm_config()

        allowed_limits = dict[str, int]()
        for modality, supported_limit in self.supported_mm_limits.items():
            user_limit = mm_config.get_limit_per_prompt(modality)

            allowed_limits[modality] = (
                user_limit
                if supported_limit is None
                else min(user_limit, supported_limit)
            )

        return allowed_limits

    def validate_num_items(self, modality: str, num_items: int) -> None:
        """
        Raise `ValueError` if the number of input items for the given modality
        is invalid.
        """
        supported_limit = self.supported_mm_limits.get(modality, 0)
        allowed_limit = self.allowed_mm_limits.get(modality, 0)

        if supported_limit is None:
            supported_limit = allowed_limit

        limit = min(supported_limit, allowed_limit)

        if num_items > limit:
            msg = f"At most {limit} {modality}(s) may be provided in one prompt."

            if num_items <= supported_limit:
                msg += " Set `--limit-mm-per-prompt` to increase this limit."

            raise ValueError(msg)

    def parse_mm_data(
        self,
        mm_data: MultiModalDataDict,
        *,
        validate: bool = True,
    ) -> MultiModalDataItems:
        """
        Normalize
        [`MultiModalDataDict`][vllm.multimodal.inputs.MultiModalDataDict]
        to [`MultiModalDataItems`][vllm.multimodal.parse.MultiModalDataItems]
        before passing them to
        [`_get_hf_mm_data`][vllm.multimodal.processing.BaseMultiModalProcessor._get_hf_mm_data].
        """
        mm_items = self.data_parser.parse_mm_data(mm_data)

        if validate:
            mm_config = self.ctx.get_mm_config()

            for modality, items in mm_items.items():
                if isinstance(items, (EmbeddingItems, DictEmbeddingItems)):
                    if not mm_config.enable_mm_embeds:
                        raise ValueError(
                            f"You must set `--enable-mm-embeds` to input "
                            f"`{modality}_embeds`"
                        )
                    if mm_config.get_limit_per_prompt(modality) == 0:
                        logger.debug(
                            "Skipping count validation for modality "
                            "'%s' (embeddings with limit=0)",
                            modality,
                        )
                        continue
                self.validate_num_items(modality, len(items))

        return mm_items

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
