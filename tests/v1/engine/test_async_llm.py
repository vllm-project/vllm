# SPDX-License-Identifier: Apache-2.0

import asyncio
from contextlib import ExitStack
from typing import List, Optional, Tuple

import pytest

from tests.v1.engine.utils import PLP_APC_UNSUPPORTED_MSG
from vllm import SamplingParams
from vllm.assets.image import ImageAsset
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.inputs import PromptType
from vllm.platforms import current_platform
from vllm.sampling_params import RequestOutputKind
from vllm.v1.engine.async_llm import AsyncLLM

if not current_platform.is_cuda():
    pytest.skip(reason="V1 currently only supported on CUDA.",
                allow_module_level=True)

TEXT_ENGINE_ARGS = AsyncEngineArgs(model="meta-llama/Llama-3.2-1B-Instruct",
                                   enforce_eager=True,
                                   disable_log_requests=True)

VISION_ENGINE_ARGS = AsyncEngineArgs(model="Qwen/Qwen2-VL-2B-Instruct",
                                     enforce_eager=True,
                                     disable_log_requests=True)

TEXT_PROMPT = "Hello my name is Robert and"

VISION_PROMPT_TEMPLATE = (
    "<|im_start|>system\nYou are a helpful assistant.<|im_end|>"
    "\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>"
    "What is in the image?<|im_end|>\n"
    "<|im_start|>assistant\n")
VISION_PROMPT = {
    "prompt": VISION_PROMPT_TEMPLATE,
    "multi_modal_data": {
        "image": ImageAsset("stop_sign").pil_image
    }
}


async def generate(engine: AsyncLLM,
                   request_id: str,
                   prompt: PromptType,
                   output_kind: RequestOutputKind,
                   max_tokens: int,
                   prompt_logprobs: Optional[int] = None) -> Tuple[int, str]:
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


@pytest.mark.parametrize(
    "output_kind", [RequestOutputKind.DELTA, RequestOutputKind.FINAL_ONLY])
@pytest.mark.asyncio
async def test_async_llm_refuses_prompt_logprobs_with_apc(
        monkeypatch, output_kind: RequestOutputKind):
    """Test passes if AsyncLLM raises an exception when it is configured
    for automatic prefix caching and it receives a request with
    prompt_logprobs enabled, which is incompatible."""
    # TODO(rickyx): Remove monkeypatch VLLM_USE_V1 setting once we have a
    # better way to test V1 so that in the future when we switch, we don't
    # have to change all the tests.
    monkeypatch.setenv("VLLM_USE_V1", "1")
    # Create AsyncLLM engine with APC
    apc_engine_args = AsyncEngineArgs(model="facebook/opt-125m",
                                      enable_prefix_caching=True,
                                      gpu_memory_utilization=0.8,
                                      disable_log_requests=True)
    engine = AsyncLLM.from_engine_args(apc_engine_args)
    try:
        with pytest.raises(ValueError) as excinfo:
            # Issue a request with prompt logprobs enabled, which should fail
            await asyncio.create_task(
                generate(engine,
                         "request-0",
                         TEXT_PROMPT,
                         output_kind,
                         10,
                         prompt_logprobs=5))
        # Validate exception string is correct
        assert str(excinfo.value) == PLP_APC_UNSUPPORTED_MSG
    finally:
        # Shut down engine
        engine.shutdown()


@pytest.mark.parametrize(
    "output_kind", [RequestOutputKind.DELTA, RequestOutputKind.FINAL_ONLY])
@pytest.mark.parametrize("engine_args_and_prompt",
                         [(TEXT_ENGINE_ARGS, TEXT_PROMPT),
                          (VISION_ENGINE_ARGS, VISION_PROMPT)])
@pytest.mark.asyncio
async def test_load(monkeypatch, output_kind: RequestOutputKind,
                    engine_args_and_prompt: Tuple[AsyncEngineArgs,
                                                  PromptType]):
    # TODO(rickyx): Remove monkeypatch once we have a better way to test V1
    # so that in the future when we switch, we don't have to change all the
    # tests.
    with monkeypatch.context() as m, ExitStack() as after:
        m.setenv("VLLM_USE_V1", "1")
        engine_args, prompt = engine_args_and_prompt

        engine = AsyncLLM.from_engine_args(engine_args)
        after.callback(engine.shutdown)

        NUM_REQUESTS = 100
        NUM_EXPECTED_TOKENS = 10

        request_ids = [f"request-{i}" for i in range(NUM_REQUESTS)]

        # Create concurrent requests.
        tasks = []
        for request_id in request_ids:
            tasks.append(
                asyncio.create_task(
                    generate(engine, request_id, prompt, output_kind,
                             NUM_EXPECTED_TOKENS)))

        # Confirm that we got all the EXPECTED tokens from the requests.
        done, pending = await asyncio.wait(tasks,
                                           return_when=asyncio.FIRST_EXCEPTION)
        for task in pending:
            task.cancel()
        for task in done:
            num_generated_tokens, request_id = await task
            assert num_generated_tokens == NUM_EXPECTED_TOKENS, (
                f"{request_id} generated {num_generated_tokens} but "
                f"expected {NUM_EXPECTED_TOKENS}")

        assert not engine.output_processor.has_unfinished_requests()


@pytest.mark.parametrize(
    "output_kind", [RequestOutputKind.DELTA, RequestOutputKind.FINAL_ONLY])
@pytest.mark.parametrize("engine_args_and_prompt",
                         [(TEXT_ENGINE_ARGS, TEXT_PROMPT),
                          (VISION_ENGINE_ARGS, VISION_PROMPT)])
@pytest.mark.asyncio
async def test_abort(monkeypatch, output_kind: RequestOutputKind,
                     engine_args_and_prompt: Tuple[AsyncEngineArgs,
                                                   PromptType]):

    with monkeypatch.context() as m, ExitStack() as after:
        m.setenv("VLLM_USE_V1", "1")
        engine_args, prompt = engine_args_and_prompt

        engine = AsyncLLM.from_engine_args(engine_args)
        after.callback(engine.shutdown)

        NUM_REQUESTS = 100
        NUM_EXPECTED_TOKENS = 100
        REQUEST_IDS_TO_ABORT = range(1, 100, 10)

        request_ids = [f"request-{i}" for i in range(NUM_REQUESTS)]

        # Create concurrent requests.
        tasks: List[asyncio.Task] = []
        for request_id in request_ids:
            tasks.append(
                asyncio.create_task(
                    generate(engine, request_id, prompt, output_kind,
                             NUM_EXPECTED_TOKENS)))

        # API server cancels requests when they disconnect.
        for idx in REQUEST_IDS_TO_ABORT:
            tasks[idx].cancel()
            await asyncio.sleep(0.1)

        # Confirm the other requests are okay.
        for idx, task in enumerate(tasks):
            # Confirm that it was actually canceled.
            if idx in REQUEST_IDS_TO_ABORT:
                with pytest.raises(asyncio.CancelledError):
                    await task
            else:
                # Otherwise, make sure the request was not impacted.
                num_generated_tokens, request_id = await task
                assert num_generated_tokens == NUM_EXPECTED_TOKENS, (
                    f"{request_id} generated {num_generated_tokens} but "
                    f"expected {NUM_EXPECTED_TOKENS}")

        assert not engine.output_processor.has_unfinished_requests()

        # Confirm we can do another generation.
        request_id = f"request-{REQUEST_IDS_TO_ABORT[0]}"
        task = asyncio.create_task(
            generate(engine, request_id, prompt, output_kind,
                     NUM_EXPECTED_TOKENS))
        num_generated_tokens, request_id = await task
        assert num_generated_tokens == NUM_EXPECTED_TOKENS
        assert not engine.output_processor.has_unfinished_requests()
