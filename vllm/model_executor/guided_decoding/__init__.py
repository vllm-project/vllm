from __future__ import annotations

from typing import TYPE_CHECKING

from vllm.logger import init_logger
from vllm.model_executor.guided_decoding.utils import (
    convert_lark_to_gbnf, grammar_is_likely_lark,
    has_lmf_unsupported_json_features, has_xgrammar_unsupported_json_features)
from vllm.platforms import CpuArchEnum

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer

    from vllm.config import ModelConfig
    from vllm.logits_process import LogitsProcessor
    from vllm.sampling_params import GuidedDecodingParams

logger = init_logger(__name__)


def maybe_backend_fallback(
        guided_params: GuidedDecodingParams) -> GuidedDecodingParams:
    # lm-format-enforce doesn't support grammar, fallback to xgrammar
    if guided_params.backend == "lm-format-enforcer":
        if guided_params.grammar is not None:
            logger.warning(
                "lm-format-enforcer does not support grammar guided decoding. "
                "Falling back to use xgrammar instead.")
            guided_params.backend = "xgrammar"

        # lm-format-enforcer doesn't support some JSON schema features
        elif (guided_params.json is not None
              and has_lmf_unsupported_json_features(guided_params.json)):
            logger.warning(
                "lm-format-enforcer does not support advanced JSON schema "
                "features like patterns or numeric ranges. "
                "Falling back to use outlines instead.")
            guided_params.backend = "outlines"

    if guided_params.backend == "xgrammar":
        # xgrammar only has x86 wheels for linux, fallback to outlines
        from vllm.platforms import current_platform
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

        # xgrammar only supports GBNF grammars, so we must convert Lark.
        # We must check if the grammar is likely Lark and if that
        # grammar is convertible to GBNF
        elif (guided_params.grammar is not None
              and grammar_is_likely_lark(guided_params.grammar)):
            try:
                convert_lark_to_gbnf(guided_params.grammar)
            except Exception:
                logger.warning(
                    "xgrammar does not support Lark grammars and the "
                    "grammar failed to convert to GBNF. "
                    "Falling back to use outlines instead.")
                guided_params.backend = "outlines"

    if (guided_params.backend == "outlines"
            and guided_params.json_object is not None):
        # outlines doesn't support json_object, fallback to xgrammar
        logger.warning("outlines does not support json_object. "
                       "Falling back to use xgrammar instead.")
        guided_params.backend = "xgrammar"

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
