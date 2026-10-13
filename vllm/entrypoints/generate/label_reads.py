# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""One-token reads that return the logprobs of chosen label tokens."""

from collections.abc import AsyncGenerator, Mapping, Sequence
from dataclasses import dataclass

from vllm.engine.protocol import EngineClient
from vllm.inputs import EngineInput
from vllm.lora.request import LoRARequest
from vllm.outputs import RequestOutput
from vllm.sampling_params import SamplingParams
from vllm.utils.async_utils import merge_async_iterators


@dataclass(frozen=True)
class LabelRead:
    result: RequestOutput
    logprobs: list[float]
    """Full-vocabulary logprob of each label token, in ``logprob_token_ids``
    order."""


async def next_token_label_reads(
    engine_client: EngineClient,
    engine_inputs: Sequence[EngineInput],
    sampling_params: Sequence[SamplingParams],
    request_id: str,
    *,
    lora_request: LoRARequest | None = None,
    trace_headers: Mapping[str, str] | None = None,
    priority: int = 0,
) -> list[LabelRead]:
    """Runs one read per input at once, as request ``{request_id}-{i}``. Each
    read's params set ``max_tokens=1`` and the label tokens as
    ``logprob_token_ids``. Raises ValueError naming the item when a read has
    no output or lacks a label's logprob."""
    generators: list[AsyncGenerator[RequestOutput, None]] = [
        engine_client.generate(
            engine_input,
            params,
            f"{request_id}-{i}",
            lora_request=lora_request,
            trace_headers=trace_headers,
            priority=priority,
        )
        for i, (engine_input, params) in enumerate(zip(engine_inputs, sampling_params))
    ]
    results: list[RequestOutput | None] = [None] * len(generators)
    async for i, res in merge_async_iterators(*generators):
        results[i] = res

    reads = []
    for i, (result, params) in enumerate(zip(results, sampling_params)):
        if result is None:
            raise ValueError(f"Failed to generate result for item {i}")
        if not result.outputs:
            raise ValueError(f"No output generated for item {i}")
        output = result.outputs[0]
        if output.finish_reason == "error":
            raise ValueError(f"Generation error for item {i}")
        if not output.logprobs:
            raise ValueError(
                f"No logprobs available for item {i}. "
                "This might indicate an issue with logprobs configuration."
            )
        logprobs = output.logprobs[0]
        label_ids = params.logprob_token_ids or []
        missing = [t for t in label_ids if t not in logprobs]
        if missing:
            raise ValueError(
                f"Token IDs {missing} not found in logprobs for item {i}. "
                "This might indicate the tokens are outside the model's vocabulary."
            )
        reads.append(LabelRead(result, [logprobs[t].logprob for t in label_ids]))
    return reads
