# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# mypy: ignore-errors
"""
gRPC Request Manager for vLLM

Manages request lifecycle for gRPC requests, converting between protobuf
and vLLM types. Much simpler than SGLang's implementation since we can
use AsyncLLM directly (no ZMQ needed).

Key optimization: Sets detokenize=False in SamplingParams to skip
detokenization and return token IDs only.
"""

from vllm.grpc import vllm_engine_pb2
from vllm.logger import init_logger
from vllm.sampling_params import (
    RequestOutputKind,
    SamplingParams,
    StructuredOutputsParams,
)

logger = init_logger(__name__)


def create_sampling_params_from_proto(
    proto_params: vllm_engine_pb2.SamplingParams,
    stream: bool = True,
) -> SamplingParams:
    """
    Convert protobuf SamplingParams to vLLM SamplingParams.

    Args:
        proto_params: Protobuf SamplingParams message
        stream: Whether streaming is enabled

    Returns:
        vLLM SamplingParams with detokenize=False and structured_outputs
    """
    # Build stop sequences
    stop = list(proto_params.stop) if proto_params.stop else None
    stop_token_ids = (
        list(proto_params.stop_token_ids) if proto_params.stop_token_ids else None
    )

    # Handle structured outputs constraints
    structured_outputs = None
    constraint_field = proto_params.WhichOneof("constraint")
    if constraint_field:
        if constraint_field == "json_schema":
            structured_outputs = StructuredOutputsParams(json=proto_params.json_schema)
        elif constraint_field == "regex":
            structured_outputs = StructuredOutputsParams(regex=proto_params.regex)
        elif constraint_field == "grammar":
            structured_outputs = StructuredOutputsParams(grammar=proto_params.grammar)
        elif constraint_field == "structural_tag":
            structured_outputs = StructuredOutputsParams(
                structural_tag=proto_params.structural_tag
            )
        elif constraint_field == "json_object":
            structured_outputs = StructuredOutputsParams(
                json_object=proto_params.json_object
            )
        elif constraint_field == "choice":
            structured_outputs = StructuredOutputsParams(
                choice=list(proto_params.choice.choices)
            )

    # Handle logit_bias
    logit_bias = None
    if proto_params.logit_bias:
        logit_bias = dict(proto_params.logit_bias)

    # Create SamplingParams with detokenize=False and output_kind=DELTA
    # detokenize=False: KEY OPTIMIZATION that skips detokenization!
    # output_kind=DELTA: Return only new tokens in each chunk (for streaming)
    return SamplingParams(
        temperature=proto_params.temperature if proto_params.temperature > 0 else 1.0,
        top_p=proto_params.top_p if proto_params.top_p > 0 else 1.0,
        top_k=proto_params.top_k if proto_params.top_k > 0 else -1,
        min_p=proto_params.min_p if proto_params.min_p > 0 else 0.0,
        frequency_penalty=proto_params.frequency_penalty,
        presence_penalty=proto_params.presence_penalty,
        repetition_penalty=proto_params.repetition_penalty
        if proto_params.repetition_penalty > 0
        else 1.0,
        max_tokens=proto_params.max_tokens
        if proto_params.HasField("max_tokens")
        else None,
        min_tokens=proto_params.min_tokens if proto_params.min_tokens > 0 else 0,
        stop=stop,
        stop_token_ids=stop_token_ids,
        skip_special_tokens=proto_params.skip_special_tokens,
        spaces_between_special_tokens=proto_params.spaces_between_special_tokens,
        ignore_eos=proto_params.ignore_eos,
        n=proto_params.n if proto_params.n > 0 else 1,
        logprobs=proto_params.logprobs if proto_params.HasField("logprobs") else None,
        prompt_logprobs=proto_params.prompt_logprobs
        if proto_params.HasField("prompt_logprobs")
        else None,
        seed=proto_params.seed if proto_params.HasField("seed") else None,
        include_stop_str_in_output=proto_params.include_stop_str_in_output,
        logit_bias=logit_bias,
        truncate_prompt_tokens=proto_params.truncate_prompt_tokens
        if proto_params.HasField("truncate_prompt_tokens")
        else None,
        structured_outputs=structured_outputs,
        # detokenize must be True if stop strings are used (SamplingParams validation)
        detokenize=bool(stop),
        output_kind=RequestOutputKind.DELTA if stream else RequestOutputKind.FINAL_ONLY,
    )
