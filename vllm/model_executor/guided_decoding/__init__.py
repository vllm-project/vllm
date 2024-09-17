from typing import Optional

from vllm.sampling_params import GuidedDecodingParams, LogitsProcessor
from vllm.transformers_utils.tokenizer import MistralTokenizer


async def get_guided_decoding_logits_processor(
        guided_params: GuidedDecodingParams,
        tokenizer) -> Optional[LogitsProcessor]:
    # CFG grammar not supported by LMFE, so we use outlines instead
    if guided_params.backend == 'outlines' or guided_params.grammar:
        if isinstance(tokenizer, MistralTokenizer):
            raise NotImplementedError(
                "Guided decoding with 'outlines' is currently not supported "
                "for Mistral tokenizer. Please consider contributing to the "
                "'outlines' project if you are interested in this feature.")
        # NOTE: lazy import outlines to avoid https://github.com/vllm-project/vllm/issues/4193
        from vllm.model_executor.guided_decoding.outlines_decoding import (  # noqa
            get_outlines_guided_decoding_logits_processor)
        return await get_outlines_guided_decoding_logits_processor(
            guided_params, tokenizer)
    if guided_params.backend == 'lm-format-enforcer':
        if isinstance(tokenizer, MistralTokenizer):
            raise NotImplementedError(
                "Guided decoding with 'lm-format-enforcer' is currently not "
                "supported for Mistral tokenizer. Please consider contributing "
                "to the 'lm-format-enforcer' project if you are interested "
                "in this feature.")
        from vllm.model_executor.guided_decoding.lm_format_enforcer_decoding import (  # noqa
            get_local_lm_format_enforcer_guided_decoding_logits_processor)
        return get_local_lm_format_enforcer_guided_decoding_logits_processor(
            guided_params, tokenizer)

    raise ValueError(
        f"Unknown guided decoding backend '{guided_params.backend}'. "
        "Must be one of 'outlines, 'lm-format-enforcer'")


def get_local_guided_decoding_logits_processor(
        guided_params: GuidedDecodingParams,
        tokenizer) -> Optional[LogitsProcessor]:
    # CFG grammar not supported by LMFE, so we use outlines instead
    if guided_params.backend == 'outlines' or guided_params.grammar:
        if isinstance(tokenizer, MistralTokenizer):
            raise NotImplementedError(
                "Guided decoding with 'outlines' is currently not supported "
                "for Mistral tokenizer. Please consider contributing to the "
                "'outlines' project if you are interested in this feature.")
        # NOTE: lazy import outlines to avoid https://github.com/vllm-project/vllm/issues/4193
        from vllm.model_executor.guided_decoding.outlines_decoding import (  # noqa
            get_local_outlines_guided_decoding_logits_processor)
        return get_local_outlines_guided_decoding_logits_processor(
            guided_params, tokenizer)
    if guided_params.backend == 'lm-format-enforcer':

        if isinstance(tokenizer, MistralTokenizer):
            raise NotImplementedError(
                "Guided decoding with 'lm-format-enforcer' is currently not "
                "supported for Mistral tokenizer. Please consider contributing "
                "to the 'lm-format-enforcer' project if you are interested "
                "in this feature.")
        from vllm.model_executor.guided_decoding.lm_format_enforcer_decoding import (  # noqa
            get_local_lm_format_enforcer_guided_decoding_logits_processor)
        return get_local_lm_format_enforcer_guided_decoding_logits_processor(
            guided_params, tokenizer)

    raise ValueError(
        f"Unknown guided decoding backend '{guided_params.backend}'. "
        "Must be one of 'outlines, 'lm-format-enforcer'")
