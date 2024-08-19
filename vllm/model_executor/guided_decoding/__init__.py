from typing import Optional

from vllm.model_executor.guided_decoding.guided_fields import (
    GuidedDecodingRequest)
from vllm.model_executor.guided_decoding.outlines_decoding import (
    get_local_outlines_guided_decoding_logits_processor)
from vllm.sampling_params import LogitsProcessor


def get_local_guided_decoding_logits_processor(
        guided_decoding_backend: str, guided_options: GuidedDecodingRequest,
        tokenizer) -> Optional[LogitsProcessor]:
    # request = _adapt_request_for_tool_use(request)

    if guided_decoding_backend == 'outlines':
        return get_local_outlines_guided_decoding_logits_processor(
            guided_options, tokenizer)
    if guided_decoding_backend == 'lm-format-enforcer':
        from vllm.model_executor.guided_decoding.lm_format_enforcer_decoding import (  # noqa
            get_local_lm_format_enforcer_guided_decoding_logits_processor)
        return get_local_lm_format_enforcer_guided_decoding_logits_processor(
            guided_options, tokenizer)

    raise ValueError(
        f"Unknown guided decoding backend '{guided_decoding_backend}'. "
        "Must be one of 'outlines, 'lm-format-enforcer'")
