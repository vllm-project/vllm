# Test the AsyncLLMEngine with multi-step-decoding

from typing import List

import pytest

from ..utils import RemoteOpenAIServer

MODELS = [
    "JackFram/llama-160m",
]
NUM_SCHEDULER_STEPS = [8]  # Multi-step decoding steps
NUM_PROMPTS = [10]

DEFAULT_SERVER_ARGS: List[str] = [
    "--disable-log-requests",
    "--use-v2-block-manager",
    "--worker-use-ray",
    "--gpu-memory-utilization",
    "0.85",
    "--swap-space",
    "16",
]


async def completions_with_server_args(prompts: List[str], model_name: str,
                                       server_cli_args: List[str]):

    outputs = None
    with RemoteOpenAIServer(model_name, server_cli_args) as server:
        async with server.get_async_client() as client:
            outputs = await client.completions.create(model=model_name,
                                                      prompt=prompts,
                                                      temperature=0,
                                                      stream=False,
                                                      max_tokens=5)
    assert outputs is not None

    return outputs


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize(("tp_size, pp_size"), [
    (1, 1),
    (2, 2),
])
@pytest.mark.parametrize("eager_mode", [False, True])
@pytest.mark.parametrize("num_scheduler_steps", NUM_SCHEDULER_STEPS)
@pytest.mark.parametrize("num_prompts", NUM_PROMPTS)
@pytest.mark.asyncio
async def test_multi_step(example_prompts, model: str, tp_size: int,
                          pp_size: int, eager_mode: int,
                          num_scheduler_steps: int, num_prompts: int):

    prompts = example_prompts
    if len(prompts) < num_prompts:
        prompts = prompts * ((num_prompts // len(prompts)) + 1)
    prompts = prompts[:num_prompts]
    assert len(prompts) == num_prompts

    server_args = DEFAULT_SERVER_ARGS + ["--enforce-eager"]
    ms_server_args = DEFAULT_SERVER_ARGS + \
        ["--num-scheduler-steps", f"{num_scheduler_steps}"]

    # Disable output proc callback as its not supported
    # with multi-step right now
    ms_server_args += ["--disable-async-output-proc"]
    if eager_mode:
        ms_server_args.append("--enforce-eager")

    distributed_args = [
        "--tensor-parallel-size",
        str(tp_size),
        "--pipeline-parallel-size",
        str(pp_size),
    ]

    ref_completions = await completions_with_server_args(
        prompts, model, server_args + distributed_args)
    test_completions = await completions_with_server_args(
        prompts, model, ms_server_args + distributed_args)

    def get_text_generations(completions):
        return [x.text for x in completions.choices]

    ref_generations = get_text_generations(ref_completions)
    test_generations = get_text_generations(test_completions)
    assert ref_generations == test_generations
