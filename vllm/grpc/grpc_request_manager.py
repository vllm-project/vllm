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

import asyncio
from collections.abc import AsyncGenerator

from vllm.grpc import vllm_engine_pb2
from vllm.inputs import TokensPrompt
from vllm.logger import init_logger
from vllm.outputs import RequestOutput
from vllm.sampling_params import (
    RequestOutputKind,
    SamplingParams,
    StructuredOutputsParams,
)
from vllm.v1.engine import EngineCoreRequest
from vllm.v1.engine.async_llm import AsyncLLM
from vllm.v1.engine.output_processor import RequestOutputCollector

logger = init_logger(__name__)


class GrpcRequestManager:
    """
    Manages gRPC request lifecycle for vLLM.

    Responsibilities:
    - Convert protobuf requests to vLLM EngineCoreRequest
    - Set detokenize=False in SamplingParams (key optimization!)
    - Submit requests to AsyncLLM
    - Stream token IDs (not text) back to gRPC clients
    - Handle abort/cancel operations
    """

    def __init__(self, async_llm: AsyncLLM):
        """
        Initialize the request manager.

        Args:
            async_llm: The AsyncLLM engine instance to submit requests to
        """
        self.async_llm = async_llm
        self.rid_to_collector: dict[str, RequestOutputCollector] = {}

        logger.info("GrpcRequestManager initialized")

    async def generate(
        self,
        request_id: str,
        prompt_token_ids: list[int],
        sampling_params: SamplingParams,
        arrival_time: float,
    ) -> AsyncGenerator[RequestOutput, None]:
        """
        Submit a generation request and stream outputs.

        Args:
            request_id: Unique request identifier
            prompt_token_ids: Pre-tokenized input from Rust router
            sampling_params: Sampling parameters (with detokenize=False!)
            arrival_time: Request arrival timestamp

        Yields:
            RequestOutput objects containing token IDs (text will be empty)
        """
        try:
            # Use processor.process_inputs() with pre-tokenized input
            prompt: TokensPrompt = {"prompt_token_ids": prompt_token_ids}

            engine_request = self.async_llm.processor.process_inputs(
                request_id=request_id,
                prompt=prompt,
                params=sampling_params,
                arrival_time=arrival_time,
            )

            collector = RequestOutputCollector(output_kind=sampling_params.output_kind)
            self.rid_to_collector[request_id] = collector

            # Submit to AsyncLLM - it will call add_request internally
            # and populate our collector
            await self._submit_request(engine_request, collector)

            # Stream outputs from collector
            while True:
                try:
                    output = await collector.get()
                    yield output

                    if output.finished:
                        break

                except asyncio.CancelledError:
                    logger.info("Request %s cancelled by client.", request_id)
                    # Clean up the request in output_processor and engine_core
                    await self.async_llm.abort([request_id])
                    raise  # Re-raise to let gRPC server handle cleanup

        except Exception:
            logger.exception("Error in generate for %s", request_id)
            raise
        finally:
            # Cleanup
            self.rid_to_collector.pop(request_id, None)

    async def _submit_request(
        self,
        request: EngineCoreRequest,
        collector: RequestOutputCollector,
    ) -> None:
        """
        Internal method to submit request to AsyncLLM.

        Args:
            request: The EngineCoreRequest to submit
            collector: The output collector for this request
        """
        try:
            # Add request to output processor
            # Use None for prompt since we have pre-tokenized input
            # TODO: Support sampling_params.n > 1 (parallel sampling)
            # When n > 1, we need to:
            # 1. Create a ParentRequest to track all child requests
            # 2. Fan out multiple child EngineCoreRequests with different
            #    request_index values
            # 3. Aggregate outputs from all children
            # For now, we only support n=1, so parent_req=None and
            # request_index=0
            self.async_llm.output_processor.add_request(
                request=request,
                prompt=None,
                parent_req=None,
                request_index=0,
                queue=collector,
            )

            # Submit to engine core
            await self.async_llm.engine_core.add_request_async(request)

        except Exception as e:
            logger.exception("Error submitting request %s", request.request_id)
            # Put error in collector
            collector.put(e)

    async def abort(self, request_id: str) -> bool:
        """
        Abort a running request.

        Args:
            request_id: The request ID to abort

        Returns:
            True if request was found and aborted, False otherwise
        """
        try:
            # Check if request exists
            collector = self.rid_to_collector.get(request_id)

            if collector is None:
                logger.debug(
                    "Abort: request %s not found (may have already completed).",
                    request_id,
                )
                return False

            # Abort in AsyncLLM (this handles both engine_core and output_processor)
            await self.async_llm.abort([request_id])

            # Remove from our tracking
            self.rid_to_collector.pop(request_id, None)

            logger.info("Request %s aborted.", request_id)
            return True

        except Exception:
            logger.exception("Error aborting request %s", request_id)
            self.rid_to_collector.pop(request_id, None)
            return False

    async def health_check(self) -> tuple[bool, str]:
        """
        Check if the engine is healthy.

        Returns:
            Tuple of (is_healthy, message)
        """
        try:
            # Check if engine is running and not errored
            if self.async_llm.errored:
                return False, "Engine is not alive"

            return True, "Healthy"

        except Exception as e:
            logger.exception("Health check error")
            return False, f"Error: {e}"

    def get_model_config(self) -> dict:
        """
        Get model configuration information.

        Returns:
            Dictionary with model config details
        """
        model_config = self.async_llm.model_config

        return {
            "model_path": model_config.model,
            "is_generation": model_config.runner_type == "generate",
            "max_context_length": model_config.max_model_len,
            "vocab_size": model_config.get_vocab_size(),
            "supports_vision": model_config.is_multimodal_model,
        }

    def get_num_unfinished_requests(self) -> int:
        """
        Get the number of currently running requests.

        Returns:
            Number of unfinished requests
        """
        return len(self.rid_to_collector)


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
