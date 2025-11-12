import asyncio
import os
import signal

import psutil
import torch

from vllm import SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.inputs import TokensPrompt
from vllm.logger import init_logger
from vllm.multimodal.cache import MultiModalBatchedField
from vllm.multimodal.hasher import MultiModalHasher
from vllm.multimodal.inputs import MultiModalFeatureSpec, MultiModalFieldElem, MultiModalFlatField, MultiModalKwargsItem, PlaceholderRange
from vllm.pooling_params import PoolingParams
from vllm.sampling_params import RequestOutputKind
from vllm.usage.usage_lib import UsageContext
from vllm.utils import FlexibleArgumentParser
from vllm.v1.engine.async_llm import AsyncLLM
from vllm.v1.request import EngineCoreRequest

logger = init_logger('vllm.guard')


async def handle_request(
    guard_engine: AsyncLLM,
    request_id: str,
    query_prompt: TokensPrompt,
    message_list: list[list[int]],
):
    response = guard_engine.encode(
        query_prompt, pooling_params=PoolingParams(
            task="encode",
            output_kind=RequestOutputKind.DELTA,
        ), request_id=request_id, resumable=True)

    response_index, conversation_results = 0, []

    async for resp in response:   
        # Wait the last token to avoid the "abort" error
        if response_index != 0:
            conversation_results.append(resp)
        response_index += 1

        if not message_list:
            continue

        next_chunk = message_list.pop(0)
        await guard_engine.resume_request(
            request_id=request_id, prompt_token_ids=next_chunk,
            finish_forever=not message_list,
        )



def parse_args():
    parser = FlexibleArgumentParser(
        description="Demo on using the LLMEngine class directly"
    )
    parser = AsyncEngineArgs.add_cli_args(parser)
    return parser.parse_args()



async def main():
    engine_args = AsyncEngineArgs(
        model="...",
        tokenizer="...",
        enforce_eager=True,
        max_num_batched_tokens=8192,
    )
    engine = AsyncLLM.from_engine_args(engine_args, usage_context=UsageContext.API_SERVER)

    prompts = ["my favourite person is", ". The capital of France is", ". The capital of Germany is"]
    tokens = [(await engine.get_tokenizer()).encode(p) for p in prompts]

    sampling_params = SamplingParams(max_tokens=1)

    mm_data = [
        MultiModalFieldElem(
            modality="audio",
            key="audio_arrays",
            data=torch.zeros(1280, dtype=torch.float32),
            field=MultiModalBatchedField(),
        ),
    ]
    q = await engine.add_request(
        request_id="a",
        prompt=EngineCoreRequest(
            request_id="a",
            prompt_token_ids=tokens[0],
            sampling_params=sampling_params,
            mm_features=[
                MultiModalFeatureSpec(
                    data=MultiModalKwargsItem.from_elems(mm_data),
                    modality="audio",
                    identifier=MultiModalHasher.hash_kwargs(
                        offset=0,
                    ),
                    mm_position=PlaceholderRange(offset=0, length=1),
                ),
            ],
            pooling_params=None,
            eos_token_id=2,
            arrival_time=0.0,
            lora_request=None,
            cache_salt=None,
            data_parallel_rank=0,
        ),
        params=sampling_params
    )

    async def read_q():
        for _ in range(1):
            out = q.get_nowait() or await q.get()
            for output in out.outputs:
                print(output)

    await read_q()

    engine.shutdown()
    current_process = psutil.Process()
    children = current_process.children(recursive=True)
    for child in children:
        os.kill(child.pid, signal.SIGTERM)

if __name__ == '__main__':
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user")
