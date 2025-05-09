# SPDX-License-Identifier: Apache-2.0

import asyncio
from typing import Optional

from vllm import SamplingParams
from vllm.inputs import PromptType
from vllm.sampling_params import RequestOutputKind
from vllm.v1.engine.async_llm import AsyncLLM


async def generate_dp(
        engine: AsyncLLM,
        request_id: str,
        prompt: PromptType,
        output_kind: RequestOutputKind,
        max_tokens: int,
        prompt_logprobs: Optional[int] = None) -> tuple[int, str]:
    # Ensure generate doesn't complete too fast for cancellation test.
    await asyncio.sleep(0.2)

    count = 0
    sampling_params = SamplingParams(max_tokens=max_tokens,
                                     ignore_eos=True,
                                     output_kind=output_kind,
                                     temperature=0,
                                     prompt_logprobs=prompt_logprobs)
    async for out in engine.generate(request_id=request_id,
                                     prompt=prompt,
                                     sampling_params=sampling_params):

        num_tokens = len(out.outputs[0].token_ids)
        if output_kind == RequestOutputKind.DELTA:
            count += num_tokens
        else:
            count = num_tokens

        await asyncio.sleep(0.)

    return count, request_id
