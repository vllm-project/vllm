# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare Qwen2-VL-2B-Instruct outputs between V0 and V1 engines for parity.

Uses the same prompt and image for both engines and asserts that generated
text and logprobs match within the test harness epsilon (check_logprobs_close).
"""

import gc
from contextlib import ExitStack, nullcontext

import pytest
import torch

from vllm import SamplingParams
from vllm.assets.image import ImageAsset
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.inputs import PromptType
from vllm.platforms import current_platform
from vllm.v1.engine.async_llm import AsyncLLM

try:
    from vllm.utils.torch_utils import set_default_torch_num_threads
except ModuleNotFoundError:

    def set_default_torch_num_threads(num_threads: int | None = None):
        return nullcontext()


from tests.conftest import VllmRunner
from tests.models.utils import check_logprobs_close

MODEL = "Qwen/Qwen2-VL-2B-Instruct"
SEED = 0

# Qwen2-VL chat template; placeholder filled by image.
VISION_PROMPT_TEMPLATE = (
    "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
    "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>"
    "What is in the image?<|im_end|>\n<|im_start|>assistant\n"
)

if not current_platform.is_cuda():
    pytest.skip(
        reason="V1 engine currently only supported on CUDA.",
        allow_module_level=True,
    )


def _get_v0_outputs(
    vllm_runner: type[VllmRunner],
    prompts: list[str],
    images: list,
    max_tokens: int,
    num_logprobs: int,
) -> list:
    """Run V0 engine and return list of (token_ids, text, logprobs) per prompt."""
    with vllm_runner(
        MODEL,
        enforce_eager=True,
        max_model_len=2048,
        dtype="half",
        trust_remote_code=True,
        default_torch_num_threads=1,
    ) as vllm_model:
        outputs = vllm_model.generate_greedy_logprobs(
            prompts,
            max_tokens=max_tokens,
            num_logprobs=num_logprobs,
            images=images,
        )
    return outputs


async def _get_v1_outputs(
    prompts_with_mm: list[PromptType],
    max_tokens: int,
    num_logprobs: int,
) -> list:
    """Run V1 engine and return list of (generated_token_ids, generated_text, logprobs)."""
    engine_args = AsyncEngineArgs(
        model=MODEL,
        enforce_eager=True,
        max_model_len=2048,
        dtype="half",
        trust_remote_code=True,
    )
    with ExitStack() as stack:
        stack.enter_context(set_default_torch_num_threads(1))
        engine = AsyncLLM.from_engine_args(engine_args)
        stack.callback(engine.shutdown)

        sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=max_tokens,
            logprobs=num_logprobs,
            seed=SEED,
        )

        results: list[tuple[list[int], str, list]] = []
        for i, prompt_input in enumerate(prompts_with_mm):
            request_id = f"qwen2vl_parity_{i}"
            final_output = None
            async for out in engine.generate(
                request_id=request_id,
                prompt=prompt_input,
                sampling_params=sampling_params,
            ):
                final_output = out

            assert final_output is not None and final_output.finished
            assert len(final_output.outputs) >= 1
            completion = final_output.outputs[0]

            gen_ids = list(completion.token_ids)
            gen_text = completion.text
            logprobs = completion.logprobs
            if logprobs is not None and hasattr(logprobs, "__iter__"):
                logprobs = list(logprobs)
            results.append((gen_ids, gen_text, logprobs))

    return results


@pytest.mark.parametrize("max_tokens", [32, 128])
@pytest.mark.parametrize("num_logprobs", [5])
@pytest.mark.asyncio
async def test_qwen2_vl_v0_v1_parity(vllm_runner, max_tokens, num_logprobs):
    """Compare Qwen2-VL-2B-Instruct generation and logprobs between V0 and V1."""
    prompts = [VISION_PROMPT_TEMPLATE]
    image = ImageAsset("stop_sign").pil_image
    images = [image]

    # 1) V0 run
    v0_outputs = _get_v0_outputs(
        vllm_runner,
        prompts=prompts,
        images=images,
        max_tokens=max_tokens,
        num_logprobs=num_logprobs,
    )

    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    # 2) V1 run
    v1_prompt_inputs: list[PromptType] = [
        {"prompt": p, "multi_modal_data": {"image": img}}
        for p, img in zip(prompts, images)
    ]
    v1_outputs = await _get_v1_outputs(
        v1_prompt_inputs,
        max_tokens=max_tokens,
        num_logprobs=num_logprobs,
    )

    assert len(v0_outputs) == len(v1_outputs), (
        f"V0 returned {len(v0_outputs)} outputs, V1 returned {len(v1_outputs)}"
    )

    v0_generated = []
    for output_ids, output_str, logprobs in v0_outputs:
        n_log = len(logprobs) if logprobs else 0
        prompt_len = len(output_ids) - n_log
        gen_ids = output_ids[prompt_len:]
        v0_generated.append((gen_ids, output_str, logprobs))

    check_logprobs_close(
        outputs_0_lst=v0_generated,
        outputs_1_lst=v1_outputs,
        name_0="V0",
        name_1="V1",
    )
