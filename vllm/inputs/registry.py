# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Union

import torch
from transformers import BatchFeature, PretrainedConfig, ProcessorMixin
from typing_extensions import TypeVar

from vllm.logger import init_logger
from vllm.transformers_utils.processor import cached_processor_from_config
from vllm.utils import get_allowed_kwarg_only_overrides
from vllm.utils.jsontree import JSONTree, json_map_leaves

if TYPE_CHECKING:
    from vllm.config import ModelConfig
    from vllm.transformers_utils.tokenizer import AnyTokenizer
else:
    ModelConfig = Any
    AnyTokenizer = Any

_T = TypeVar("_T")
_C = TypeVar("_C", bound=PretrainedConfig, default=PretrainedConfig)
_P = TypeVar("_P", bound=ProcessorMixin, default=ProcessorMixin)

logger = init_logger(__name__)


@dataclass(frozen=True)
class InputContext:
    """
    Contains information about the model which may be used to
    modify the inputs.
    """

    model_config: ModelConfig
    """The configuration of the model."""

    def get_hf_config(
        self,
        typ: Union[type[_C], tuple[type[_C], ...]] = PretrainedConfig,
        /,
    ) -> _C:
        """
        Get the HuggingFace configuration
        (`transformers.PretrainedConfig`) of the model,
        additionally checking its type.

        Raises:
            TypeError: If the configuration is not of the specified type.
        """
        hf_config = self.model_config.hf_config
        if not isinstance(hf_config, typ):
            raise TypeError("Invalid type of HuggingFace config. "
                            f"Expected type: {typ}, but "
                            f"found type: {type(hf_config)}")

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

    def get_hf_processor(
        self,
        typ: Union[type[_P], tuple[type[_P], ...]] = ProcessorMixin,
        /,
        **kwargs: object,
    ) -> _P:
        """
        Get the HuggingFace processor
        (`transformers.ProcessorMixin`) of the model,
        additionally checking its type.

        Raises:
            TypeError: If the processor is not of the specified type.
        """
        return cached_processor_from_config(
            self.model_config,
            processor_cls=typ,
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


@dataclass(frozen=True)
class InputProcessingContext(InputContext):
    tokenizer: AnyTokenizer
    """The tokenizer used to tokenize the inputs."""

    def get_hf_processor(
        self,
        typ: Union[type[_P], tuple[type[_P], ...]] = ProcessorMixin,
        /,
        **kwargs: object,
    ) -> _P:
        return super().get_hf_processor(
            typ,
            tokenizer=self.tokenizer,
            **kwargs,
        )

    def call_hf_processor(
        self,
        hf_processor: ProcessorMixin,
        data: Mapping[str, object],
        kwargs: Mapping[str, object] = {},
    ) -> Union[BatchFeature, JSONTree]:
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

        def maybe_cast_dtype(x):
            # This mimics the behavior of transformers.BatchFeature
            if isinstance(x, torch.Tensor) and x.is_floating_point():
                return x.to(dtype=self.model_config.dtype)
            return x

        try:
            output = hf_processor(**data,
                                  **allowed_kwargs,
                                  return_tensors="pt")
            # this emulates output.to(dtype=self.model_config.dtype)
            if isinstance(output, BatchFeature):
                cast_output = json_map_leaves(maybe_cast_dtype, output.data)
                return BatchFeature(cast_output)

            cast_output = json_map_leaves(maybe_cast_dtype, output)

            logger.warning_once(
                f"{type(hf_processor).__name__} did not return `BatchFeature`. "
                "Make sure to match the behaviour of `ProcessorMixin` when "
                "implementing custom processors.")
            return cast_output

        except Exception as exc:
            msg = (f"Failed to apply {type(hf_processor).__name__} "
                   f"on data={data} with kwargs={allowed_kwargs}")

            raise ValueError(msg) from exc
