from __future__ import annotations

from typing import TYPE_CHECKING

from vllm.logger import init_logger
from vllm.platforms import CpuArchEnum, current_platform

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer

    from vllm.config import ModelConfig
    from vllm.logits_process import LogitsProcessor
    from vllm.sampling_params import GuidedDecodingParams

logger = init_logger(__name__)


def has_xgrammar_unsupported_json_features(schema: dict) -> bool:
    """Check if JSON schema contains features unsupported by xgrammar."""

    def check_object(obj: dict) -> bool:
        if not isinstance(obj, dict):
            return False

        # Check for pattern restrictions
        if "pattern" in obj:
            return True

        # Check for numeric ranges
        if obj.get("type") in ("integer", "number") and any(
                key in obj for key in [
                    "minimum", "maximum", "exclusiveMinimum",
                    "exclusiveMaximum", "multipleOf"
                ]):
            return True

        # Recursively check all nested objects and arrays
        for value in obj.values():
            if isinstance(value, dict):
                if check_object(value):
                    return True
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, dict) and check_object(item):
                        return True

        return False

    return check_object(schema)


def maybe_backend_fallback(
        guided_params: GuidedDecodingParams) -> GuidedDecodingParams:
    # lm-format-enforce doesn't support grammar, fallback to xgrammar
    if (guided_params.backend == "lm-format-enforcer"
            and guided_params.grammar is not None):
        logger.warning(
            "lm-format-enforcer does not support grammar guided decoding. "
            "Falling back to use xgrammar instead.")
        guided_params.backend = "xgrammar"

    if guided_params.backend == "xgrammar":
        # xgrammar only has x86 wheels for linux, fallback to outlines
        if current_platform.get_cpu_architecture() is not CpuArchEnum.X86:
            logger.warning("xgrammar is only supported on x86 CPUs. "
                           "Falling back to use outlines instead.")
            guided_params.backend = "outlines"

        # xgrammar doesn't support regex or choice, fallback to outlines
        if guided_params.regex is not None or guided_params.choice is not None:
            logger.warning(
                "xgrammar only supports json or grammar guided decoding. "
                "Falling back to use outlines instead.")
            guided_params.backend = "outlines"

        # xgrammar doesn't support some JSON schema features
        elif (guided_params.json is not None
              and has_xgrammar_unsupported_json_features(guided_params.json)):
            logger.warning(
                "xgrammar does not support advanced JSON schema features like "
                "patterns or numeric ranges. "
                "Falling back to use outlines instead.")
            guided_params.backend = "outlines"

    return guided_params


async def get_guided_decoding_logits_processor(
        guided_params: GuidedDecodingParams, tokenizer: PreTrainedTokenizer,
        model_config: ModelConfig) -> LogitsProcessor | None:
    guided_params = maybe_backend_fallback(guided_params)
    # CFG grammar not supported by LMFE, so we use outlines instead
    if guided_params.backend == 'outlines':
        # NOTE: lazy import outlines to avoid https://github.com/vllm-project/vllm/issues/4193
        from vllm.model_executor.guided_decoding.outlines_decoding import (  # noqa
            get_outlines_guided_decoding_logits_processor)
        return await get_outlines_guided_decoding_logits_processor(
            guided_params, tokenizer)
    if guided_params.backend == 'lm-format-enforcer':
        from vllm.model_executor.guided_decoding.lm_format_enforcer_decoding import (  # noqa
            get_local_lm_format_enforcer_guided_decoding_logits_processor)
        return get_local_lm_format_enforcer_guided_decoding_logits_processor(
            guided_params, tokenizer)
    if guided_params.backend == 'xgrammar':
        from vllm.model_executor.guided_decoding.xgrammar_decoding import (  # noqa
            get_local_xgrammar_guided_decoding_logits_processor)
        return get_local_xgrammar_guided_decoding_logits_processor(
            guided_params, tokenizer, model_config)

    raise ValueError(
        f"Unknown guided decoding backend '{guided_params.backend}'. "
        "Must be one of 'outlines, 'lm-format-enforcer', 'xgrammar'")


def get_local_guided_decoding_logits_processor(
        guided_params: GuidedDecodingParams, tokenizer: PreTrainedTokenizer,
        model_config: ModelConfig) -> LogitsProcessor | None:
    guided_params = maybe_backend_fallback(guided_params)
    # CFG grammar not supported by LMFE, so we use outlines instead
    if guided_params.backend == 'outlines':
        # NOTE: lazy import outlines to avoid https://github.com/vllm-project/vllm/issues/4193
        from vllm.model_executor.guided_decoding.outlines_decoding import (  # noqa
            get_local_outlines_guided_decoding_logits_processor)
        return get_local_outlines_guided_decoding_logits_processor(
            guided_params, tokenizer)
    if guided_params.backend == 'lm-format-enforcer':
        from vllm.model_executor.guided_decoding.lm_format_enforcer_decoding import (  # noqa
            get_local_lm_format_enforcer_guided_decoding_logits_processor)
        return get_local_lm_format_enforcer_guided_decoding_logits_processor(
            guided_params, tokenizer)
    if guided_params.backend == 'xgrammar':
        from vllm.model_executor.guided_decoding.xgrammar_decoding import (  # noqa
            get_local_xgrammar_guided_decoding_logits_processor)
        return get_local_xgrammar_guided_decoding_logits_processor(
            guided_params, tokenizer, model_config)

    raise ValueError(
        f"Unknown guided decoding backend '{guided_params.backend}'. "
        "Must be one of 'outlines, 'lm-format-enforcer', 'xgrammar'")
