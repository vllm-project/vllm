# Test the AsyncLLMEngine with multi-step-decoding and chunked prefill

from typing import List

import pytest

from ..utils import RemoteOpenAIServer

MODELS = [
    "facebook/opt-125m",
    "meta-llama/Llama-2-7b-hf",
]
NUM_SCHEDULER_STEPS = [8, 16]  # Multi-step decoding steps
NUM_PROMPTS = [100]

# TODO (varun) : Expand tests for multiple TP & PP
DEFAULT_SERVER_ARGS: List[str] = [
    "--disable-log-requests",
    "--use-v2-block-manager",
    "--worker-use-ray",
    "--gpu-memory-utilization",
    "0.90",
    "--swap-space",
    "16",
    "--tensor-parallel-size",
    "1",
    "--pipeline-parallel-size",
    "1",
]


async def completions_with_server_args(prompts: List[str], model_name: str,
                                       server_cli_args: List[str]):

    outputs = None
    with RemoteOpenAIServer(model_name, server_cli_args) as server:
        client = server.get_async_client()
        outputs = await client.completions.create(model=model_name,
                                                  prompt=prompts,
                                                  temperature=0,
                                                  stream=False,
                                                  max_tokens=150)
    assert outputs is not None

    return outputs


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("num_scheduler_steps", NUM_SCHEDULER_STEPS)
@pytest.mark.parametrize("num_prompts", NUM_PROMPTS)
@pytest.mark.asyncio
async def test_mutli_step_with_chunked_prefill(example_prompts, model: str,
                                               num_scheduler_steps: int,
                                               num_prompts: int):

    prompts = example_prompts
    if len(prompts) < num_prompts:
        prompts = prompts * ((num_prompts // len(prompts)) + 1)
    prompts = prompts[:num_prompts]
    assert len(prompts) == num_prompts

    server_args = DEFAULT_SERVER_ARGS + \
        ["--num-scheduler-steps", f"{num_scheduler_steps}"]

    ref_completions = await completions_with_server_args(
        prompts, model, server_args)
    test_completions = await completions_with_server_args(
        prompts, model, server_args + ["--enable-chunked-prefill"])

    def get_text_generations(completions):
        return [x.text for x in completions.choices]

    ref_generations = get_text_generations(ref_completions)
    test_generations = get_text_generations(test_completions)
    assert ref_generations == test_generations
