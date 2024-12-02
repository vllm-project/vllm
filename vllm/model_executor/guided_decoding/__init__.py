from __future__ import annotations

from typing import TYPE_CHECKING

from vllm.logger import init_logger

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer

    from vllm.config import ModelConfig
    from vllm.logits_process import LogitsProcessor
    from vllm.sampling_params import GuidedDecodingParams

logger = init_logger(__name__)


def maybe_backend_fallback(
        guided_params: GuidedDecodingParams) -> GuidedDecodingParams:
    # Since lm-format-enforce doesn't support grammar, fallback to xgrammar
    if (guided_params.backend == "lm-format-enforcer"
            and guided_params.grammar is not None):
        logger.warning(
            "lm-format-enforcer does not support grammar guided decoding. "
            "Falling back to use xgrammar instead.")
        guided_params.backend = "xgrammar"

    # Since xgrammar backend only supports json/grammar, fallback to outlines
    xgrammar_supports = any(
        [guided_params.json is not None, guided_params.grammar is not None])
    if guided_params.backend == "xgrammar" and not xgrammar_supports:
        logger.warning(
            "xgrammar only supports json or grammar guided decoding. "
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
