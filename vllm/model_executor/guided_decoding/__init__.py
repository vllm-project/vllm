# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING

from vllm.logger import init_logger
from vllm.model_executor.guided_decoding.utils import (
    convert_lark_to_gbnf, grammar_is_likely_lark,
    has_lmf_unsupported_json_features, has_xgrammar_unsupported_json_features)
from vllm.reasoning import ReasoningParserManager

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer

    from vllm.config import ModelConfig
    from vllm.logits_process import LogitsProcessor
    from vllm.sampling_params import GuidedDecodingParams

logger = init_logger(__name__)


def maybe_backend_fallback(
        guided_params: GuidedDecodingParams) -> GuidedDecodingParams:

    def fallback_or_error(guided_params: GuidedDecodingParams, message: str,
                          fallback: str) -> None:
        """Change the backend to the specified fallback with a warning log,
        or raise a ValueError if the `no-fallback` option is specified."""
        if guided_params.no_fallback():
            raise ValueError(message)

        logger.warning("%s Falling back to use %s instead.", message, fallback)
        guided_params.backend = fallback

    # `auto` was added for V1 to explicitly declare a mode that has fallbacks
    # in place. If that is specified with V0, treat it as `xgrammar`, as we have
    # fallbacks enabled for that and it is the V0 default.
    if guided_params.backend == "auto":
        guided_params.backend = "xgrammar"

    # lm-format-enforce doesn't support grammar, fallback to xgrammar
    if guided_params.backend_name == "lm-format-enforcer":
        if guided_params.grammar is not None:
            fallback_or_error(
                guided_params,
                "lm-format-enforcer does not support grammar guided decoding.",
                "xgrammar")

        # lm-format-enforcer doesn't support some JSON schema features
        elif (guided_params.json is not None
              and has_lmf_unsupported_json_features(guided_params.json)):
            fallback_or_error(
                guided_params,
                "lm-format-enforcer does not support advanced JSON schema "
                "features like patterns or numeric ranges.", "outlines")

    if guided_params.backend_name == "xgrammar":
        from vllm.model_executor.guided_decoding.xgrammar_decoding import (
            xgr_installed)

        # xgrammar doesn't support some JSON schema features
        if (guided_params.json is not None and
                has_xgrammar_unsupported_json_features(guided_params.json)):
            fallback_or_error(
                guided_params,
                "xgrammar does not support advanced JSON schema features like "
                "string length, item limits, or property bounds.", "outlines")

        # xgrammar only supports GBNF grammars, so we must convert Lark.
        # We must check if the grammar is likely Lark and if that
        # grammar is convertible to GBNF
        elif (guided_params.grammar is not None
              and grammar_is_likely_lark(guided_params.grammar)):
            try:
                convert_lark_to_gbnf(guided_params.grammar)
            except Exception:
                fallback_or_error(
                    guided_params,
                    "xgrammar does not support Lark grammars and the "
                    "grammar failed to convert to GBNF.", "outlines")

        # If the xgrammar module cannot be imported successfully,
        # we should still allow users to use guided decoding with a fallback.
        elif not xgr_installed:
            fallback_or_error(
                guided_params,
                "xgrammar module cannot be imported successfully.", "outlines")

    if (guided_params.backend_name == "outlines"
            and guided_params.json_object is not None):
        # outlines doesn't support json_object, fallback to guidance
        fallback_or_error(guided_params,
                          "outlines does not support json_object.", "guidance")

    return guided_params


async def get_guided_decoding_logits_processor(
        guided_params: GuidedDecodingParams,
        tokenizer: PreTrainedTokenizer,
        model_config: ModelConfig,
        reasoning_backend: str | None = None) -> LogitsProcessor | None:

    reasoner = None
    if reasoning_backend is not None:
        reasoner_class = ReasoningParserManager.get_reasoning_parser(
            reasoning_backend)
        reasoner = reasoner_class(tokenizer)

    guided_params = maybe_backend_fallback(guided_params)

    # CFG grammar not supported by LMFE, so we use outlines instead
    if guided_params.backend_name == 'outlines':
        # NOTE: lazy import outlines to avoid https://github.com/vllm-project/vllm/issues/4193
        from vllm.model_executor.guided_decoding.outlines_decoding import (  # noqa
            get_outlines_guided_decoding_logits_processor)
        return await get_outlines_guided_decoding_logits_processor(
            guided_params, tokenizer, reasoner)
    if guided_params.backend == 'lm-format-enforcer':
        from vllm.model_executor.guided_decoding.lm_format_enforcer_decoding import (  # noqa
            get_local_lm_format_enforcer_guided_decoding_logits_processor)
        return get_local_lm_format_enforcer_guided_decoding_logits_processor(
            guided_params, tokenizer)
    if guided_params.backend_name == 'xgrammar':
        from vllm.model_executor.guided_decoding.xgrammar_decoding import (  # noqa
            get_local_xgrammar_guided_decoding_logits_processor)
        return get_local_xgrammar_guided_decoding_logits_processor(
            guided_params, tokenizer, model_config, reasoner)
    if guided_params.backend_name == 'guidance':
        from vllm.model_executor.guided_decoding.guidance_decoding import (
            get_local_guidance_guided_decoding_logits_processor)
        return get_local_guidance_guided_decoding_logits_processor(
            guided_params, tokenizer)
    raise ValueError(
        f"Unknown guided decoding backend '{guided_params.backend}'. "
        "Must be one of 'outlines, 'lm-format-enforcer', 'xgrammar', 'guidance'"
    )


def get_local_guided_decoding_logits_processor(
        guided_params: GuidedDecodingParams,
        tokenizer: PreTrainedTokenizer,
        model_config: ModelConfig,
        reasoning_backend: str | None = None) -> LogitsProcessor | None:
    guided_params = maybe_backend_fallback(guided_params)

    reasoner = None
    if reasoning_backend is not None:
        reasoner_class = ReasoningParserManager.get_reasoning_parser(
            reasoning_backend)
        reasoner = reasoner_class(tokenizer)

    # CFG grammar not supported by LMFE, so we use outlines instead
    if guided_params.backend_name == 'outlines':
        # NOTE: lazy import outlines to avoid https://github.com/vllm-project/vllm/issues/4193
        from vllm.model_executor.guided_decoding.outlines_decoding import (  # noqa
            get_local_outlines_guided_decoding_logits_processor)
        return get_local_outlines_guided_decoding_logits_processor(
            guided_params, tokenizer, reasoner)
    if guided_params.backend_name == 'lm-format-enforcer':
        from vllm.model_executor.guided_decoding.lm_format_enforcer_decoding import (  # noqa
            get_local_lm_format_enforcer_guided_decoding_logits_processor)
        return get_local_lm_format_enforcer_guided_decoding_logits_processor(
            guided_params, tokenizer)
    if guided_params.backend_name == 'xgrammar':
        from vllm.model_executor.guided_decoding.xgrammar_decoding import (  # noqa
            get_local_xgrammar_guided_decoding_logits_processor)
        return get_local_xgrammar_guided_decoding_logits_processor(
            guided_params, tokenizer, model_config, reasoner)
    if guided_params.backend_name == 'guidance':
        from vllm.model_executor.guided_decoding.guidance_decoding import (
            get_local_guidance_guided_decoding_logits_processor)
        return get_local_guidance_guided_decoding_logits_processor(
            guided_params, tokenizer)

    raise ValueError(
        f"Unknown guided decoding backend '{guided_params.backend}'. "
        "Must be one of 'outlines, 'lm-format-enforcer', 'xgrammar', 'guidance'"
    )
