# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import asyncio
import random
import time
import uuid

from datatools.tokenizer.bpe import TemplatedBPTokenizer
from test_cohere2_vision import (
    MSG_1,
    encode_with_turns,
    image_to_base64,
    process_image,
)
from test_utils import generate_random_image
from vllm.cohere.multimodal_tokeniser.continuous import (
    MM_C3_AGENTS_TEXT_TOKENISER_CONT,
    Tokeniser,
)

from vllm import AsyncEngineArgs, AsyncLLMEngine, SamplingParams, TokensPrompt

NUM_TEST_PROMPTS = 16
NUM_IMAGES_PER_PROMPT = 1


async def generate_response(engine, params, prompt: str) -> str:
    req_id = str(uuid.uuid4())  # unique ID for this request
    stream = engine.generate(prompt, params, request_id=req_id)
    final_output = None
    async for output in stream:  # iterate over streamed chunks
        final_output = output  # keep updating until final chunk
    text = final_output.outputs[0].text if final_output else ""
    return text


def generate_prompts_image(tokenizer_txt, tokenizer_img):
    msg = []
    img_data = []
    for i in range(NUM_IMAGES_PER_PROMPT):
        # 6144, 512 is the image size with max number of crops
        img = generate_random_image(6144, 512)
        msg.append(
            {"image_sizes": process_image(img)[1]},
        )
        img_data.append(image_to_base64(img))
        # img_data.append(img)
    msg.append({"text": MSG_1})
    token_ids = encode_with_turns(msg, tokenizer_txt, tokenizer_img)
    engine_input = TokensPrompt(
        prompt_token_ids=token_ids,
        multi_modal_data={"image": img_data},
    )
    return engine_input


def generate_prompts_text(tokenizer_txt):
    vocab_size = len(tokenizer_txt)
    # 3333 is roughly the number of tokens of a max crop image
    # using vocab_size - 20 to avoid generating OOV tokens
    token_ids = [random.randint(0, vocab_size - 20) for _ in range(3333)]
    engine_input = TokensPrompt(prompt_token_ids=token_ids)
    return engine_input


async def test_async_engine(
    llm_instance, sampling_params, tokenizer_txt, tokenizer_img
):
    llm_instance.reset_prefix_cache()
    prompts = [
        generate_prompts_image(tokenizer_txt, tokenizer_img)
        for _ in range(NUM_TEST_PROMPTS)
    ]
    # prompts = [generate_prompts_text(tokenizer_txt) for _ in range(NUM_TEST_PROMPTS)]
    tasks = [
        asyncio.create_task(generate_response(llm_instance, sampling_params, p))
        for p in prompts
    ]
    results = await asyncio.gather(*tasks)
    for i, res in enumerate(results, 1):
        print(f"Result {i}: {res}\n")


if __name__ == "__main__":
    """
    For a wholistic nvtx range capture, it's helpful to insert the ranges
    in following places:
    collective_rpc() and worker_busy_loop() from
    vllm/v1/executor/multiproc_executor.py execute_model() from
    vllm/v1/worker/gpu_model_runner.py
    """
    tokenizer_txt = TemplatedBPTokenizer(
        MM_C3_AGENTS_TEXT_TOKENISER_CONT,
        chat_template_name="chat-command-turn_tokens-v2",
    )

    tokenizer_img = Tokeniser(
        min_image_size=512, max_image_size=512, downsample_ratio=16, max_crops=12
    )
    sampling_params = SamplingParams(top_p=0.7, temperature=0.3, max_tokens=1)
    llm_instance = AsyncLLMEngine.from_engine_args(
        AsyncEngineArgs(
            # text model: c3-111b-instruct-offline-pref-i3ba20ti-c3-temp-fp8-vllm
            model="/host/ckpts/tmp_tif_export_111b_fp8/poseidon/",  # vision model
            max_model_len=256000,
            tensor_parallel_size=2,
            gpu_memory_utilization=0.95,
            enable_chunked_prefill=True,
            max_num_batched_tokens=4096,
            # limit_mm_per_prompt={"image": 4}
            # disable_mm_preprocessor_cache=True,
            # enable_prefix_caching=False,
            max_num_seqs=4,
            guided_decoding_backend="xgrammar",
        )
    )
    time.sleep(10)
    import torch.cuda.profiler as profiler

    profiler.start()
    asyncio.run(
        test_async_engine(llm_instance, sampling_params, tokenizer_txt, tokenizer_img)
    )
    profiler.stop()
    del llm_instance
