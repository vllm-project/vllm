# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from pathlib import Path
from typing import TYPE_CHECKING, Any

from vllm.logger import init_logger

if TYPE_CHECKING:
    from transformers import PretrainedConfig

logger = init_logger(__name__)


class ModelFormatHandler:
    """Extension hook for out-of-tree model formats.

    Handlers can customize how a model reference is interpreted across vLLM,
    such as model/config discovery, tokenizer and processor resolution, and
    engine-arg defaults.
    """

    name: str = ""

    def matches(self, model: str | Path | None) -> bool:
        return False

    def update_engine_args(self, engine_args: Any) -> None:
        return

    def prepare_hf_config_load(
        self,
        model: str | Path,
        revision: str | None = None,
        kwargs: dict[str, Any] | None = None,
    ) -> tuple[str | Path, dict[str, Any]]:
        return model, kwargs or {}

    def should_use_hf_config_parser(
        self,
        original_model: str | Path,
        resolved_model: str | Path,
    ) -> bool:
        return False

    def get_missing_hf_config_error(
        self,
        original_model: str | Path,
        resolved_model: str | Path,
    ) -> str | None:
        return None

    def patch_parsed_hf_config(
        self,
        original_model: str | Path,
        config_dict: dict[str, Any],
        config: "PretrainedConfig",
    ) -> "PretrainedConfig":
        return config

    def patch_model_hf_config(
        self,
        original_model: str | Path,
        hf_config: "PretrainedConfig",
    ) -> "PretrainedConfig":
        return hf_config

    def resolve_tokenizer_init(
        self,
        tokenizer_name: str | Path,
        *args: Any,
        revision: str | None = None,
        runner_type: str = "generate",
        tokenizer_mode: str = "auto",
        **kwargs: Any,
    ) -> tuple[str | Path, tuple[Any, ...], dict[str, Any]]:
        return tokenizer_name, args, kwargs

    def resolve_processor_source(
        self,
        model_config: Any,
        component: str,
    ) -> tuple[str | Path, str | None]:
        return model_config.model, model_config.revision

    def validate_model_config(self, model_config: Any) -> None:
        return

    def resolve_sentence_transformer_source(
        self,
        model: str | Path,
        revision: str | None = None,
    ) -> str | Path:
        return model

    def resolve_image_processor_source(
        self,
        model: str | Path,
        revision: str | None = None,
    ) -> str | Path:
        return model

    def should_skip_generation_config(self, model: str | Path) -> bool:
        return False


_MODEL_FORMAT_HANDLERS: list[ModelFormatHandler] = []


def register_model_format(handler: ModelFormatHandler) -> ModelFormatHandler:
    if not isinstance(handler, ModelFormatHandler):
        raise ValueError("The model format handler must subclass `ModelFormatHandler`.")

    replaced = False
    if handler.name:
        for idx, existing in enumerate(_MODEL_FORMAT_HANDLERS):
            if existing.name == handler.name:
                logger.warning(
                    "The model format handler %r already exists and will be "
                    "overwritten by %s.",
                    handler.name,
                    type(handler),
                )
                _MODEL_FORMAT_HANDLERS[idx] = handler
                replaced = True
                break

    if not replaced:
        _MODEL_FORMAT_HANDLERS.append(handler)

    return handler


def get_model_format_handler(model: str | Path | None) -> ModelFormatHandler | None:
    for handler in reversed(_MODEL_FORMAT_HANDLERS):
        if handler.matches(model):
            return handler
    return None


def prepare_hf_model_reference(
    model: str | Path,
    revision: str | None = None,
    **kwargs: Any,
) -> tuple[ModelFormatHandler | None, str | Path, dict[str, Any]]:
    handler = get_model_format_handler(model)
    if handler is None:
        return None, model, kwargs
    resolved_model, resolved_kwargs = handler.prepare_hf_config_load(
        model,
        revision=revision,
        kwargs=kwargs,
    )
    return handler, resolved_model, resolved_kwargs


__all__ = [
    "ModelFormatHandler",
    "get_model_format_handler",
    "prepare_hf_model_reference",
    "register_model_format",
]
