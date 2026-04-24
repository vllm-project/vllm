# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import asyncio
import uuid

from datatools.tokenizer.bpe import TemplatedBPTokenizer
from test_cohere2_vision import (
    IMG_1,
    IMG_2,
    MSG_2,
    encode_with_turns,
    image_to_base64,
    process_image,
)
from vllm.cohere.multimodal_tokeniser.continuous import (
    MM_C3_AGENTS_TEXT_TOKENISER_CONT,
    Tokeniser,
)

from vllm import AsyncEngineArgs, AsyncLLMEngine, SamplingParams, TokensPrompt

NUM_TEST_PROMPTS = 20


async def generate_response(engine, params, prompt: str) -> str:
    req_id = str(uuid.uuid4())  # unique ID for this request
    stream = engine.generate(prompt, params, request_id=req_id)
    final_output = None
    async for output in stream:  # iterate over streamed chunks
        final_output = output  # keep updating until final chunk
    text = final_output.outputs[0].text if final_output else ""
    return text


async def test_async_engine(
    llm_instance, sampling_params, tokenizer_txt, tokenizer_img
):
    msg_2 = [
        {"image_sizes": process_image(IMG_1)[1]},
        {"image_sizes": process_image(IMG_2)[1]},
        {"text": MSG_2},
    ]
    # Tokenize conversations
    token_ids_2 = encode_with_turns(msg_2, tokenizer_txt, tokenizer_img)
    # Prepare engine inputs
    engine_input_2 = TokensPrompt(
        prompt_token_ids=token_ids_2,
        multi_modal_data={"image": [image_to_base64(IMG_1), image_to_base64(IMG_2)]},
    )

    llm_instance.reset_prefix_cache()
    prompts = [engine_input_2] * NUM_TEST_PROMPTS
    tasks = [
        asyncio.create_task(generate_response(llm_instance, sampling_params, p))
        for p in prompts
    ]
    results = await asyncio.gather(*tasks)
    for i, res in enumerate(results, 1):
        print(f"Result {i}: {res}\n")


if __name__ == "__main__":
    tokenizer_txt = TemplatedBPTokenizer(
        MM_C3_AGENTS_TEXT_TOKENISER_CONT,
        chat_template_name="chat-command-turn_tokens-v2",
    )

    tokenizer_img = Tokeniser(
        min_image_size=512, max_image_size=512, downsample_ratio=16, max_crops=12
    )
    sampling_params = SamplingParams(top_p=0.7, temperature=0.3, max_tokens=256)
    llm_instance = AsyncLLMEngine.from_engine_args(
        AsyncEngineArgs(
            model="/host/ckpts/tmp_tif_export_111b/poseidon/",
            max_model_len=32000,
            tensor_parallel_size=4,
            gpu_memory_utilization=0.95,
            enable_chunked_prefill=True,
            max_num_batched_tokens=4096,
            # limit_mm_per_prompt={"image": 4}
            guided_decoding_backend="xgrammar",
            disable_mm_preprocessor_cache=True,
            enable_prefix_caching=False,
        )
    )
    asyncio.run(
        test_async_engine(llm_instance, sampling_params, tokenizer_txt, tokenizer_img)
    )
