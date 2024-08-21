# Test the AsyncLLMEngine with multi-step-decoding

from typing import Dict, List, Optional

import pytest
from openai.types.completion import Completion

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

NUM_LOGPROBS = [None, 5]  # `logprobs` argument to OpenAI completions API


async def completions_with_server_args(
    prompts: List[str],
    model_name: str,
    server_cli_args: List[str],
    num_logprobs: Optional[int],
) -> Completion:
    '''
    Construct a remote OpenAI server, obtain an async client to the
    server & invoke the completions API to obtain completions.

    Arguments:

    * prompts: test prompts
    * model_name: model to spin up on the vLLM server
    * server_cli_args: CLI args for starting the server

    Returns:

    * OpenAI Completion instance
    '''

    outputs = None
    with RemoteOpenAIServer(model_name, server_cli_args) as server:
        client = server.get_async_client()
        outputs = await client.completions.create(model=model_name,
                                                  prompt=prompts,
                                                  temperature=0,
                                                  stream=False,
                                                  max_tokens=5,
                                                  logprobs=num_logprobs)
    assert outputs is not None

    return outputs


def get_text_generations(completions: Completion):
    '''Obtain generated tokens'''
    return [x.text for x in completions.choices]


'''
Logprobs values are extracted as List[List[Dict[str,float]]], i.e.:
* For each :class:`SequenceGroup`,
* for each token offset in a sequence,
* a mapping from str(token) -> logprob
'''
LogprobType = List[Optional[List[Dict[str, float]]]]


def get_logprob_generations(completions: Completion) -> LogprobType:
    '''Obtain top-rank logprobs for each token in each :class:`SequenceGroup`'''
    return [(None if x.logprobs is None else x.logprobs.top_logprobs)
            for x in completions.choices]


def assert_all_close_logprobs(
    ref_logprobs: LogprobType,
    test_logprobs: LogprobType,
    atol: float = 1e-3,
    rtol: float = 1e-3,
) -> None:
    '''
    Asserts that logprobs produced by the vLLM engine instance under test
    are very close to a set of ground-truth reference values.

    Each individual reference logprob must be close to the test logprob,
    according to the formula

    assert abs(tok_top_test_logprob - 
            tok_top_ref_logprob) <= (atol + 
                                    rtol * abs(
                                        tok_top_ref_logprob))

    Arguments:

    * ref_logprobs: ground-truth logprobs
    * test_logprobs: logprobs produced by vLLM engine under test
    * atol: absolute mismatch tolerance when comparing single logprobs
    * rtol: relative mismatch tolerance when comparing single logprobs
    '''

    assert len(ref_logprobs) == len(test_logprobs), (
        "Reference & test logprob SequenceGroup counts must match.")

    if ref_logprobs[0] is None:
        # It is expected that if one :class:`SequenceGroup` has
        # `None` logprobs, then all :class:`SequenceGroup`s
        # in the reference list have `None` logprobs.
        # Validate this.
        assert all([x is None for x in ref_logprobs])

        # Next, assert that this is also true for
        # test logprobs.
        assert all([x is None for x in test_logprobs])
        return

    for (group_ref_logprobs,
         group_test_logprobs) in zip(ref_logprobs, test_logprobs):

        assert group_ref_logprobs is not None
        assert group_test_logprobs is not None
        assert len(group_ref_logprobs) == len(group_test_logprobs), (
            "Reference & test logprob seq lens must match.")

        for (token_ref_logprobs,
             token_test_logprobs) in zip(group_ref_logprobs,
                                         group_test_logprobs):
            assert token_ref_logprobs.keys() == token_test_logprobs.keys(), (
                "Reference & test top-logprob token sets must match.")
            for (tok_str_ref,
                 tok_top_ref_logprob) in token_ref_logprobs.items():
                tok_top_test_logprob = token_test_logprobs[tok_str_ref]

                assert abs(tok_top_test_logprob - tok_top_ref_logprob) <= (
                    atol + rtol * abs(tok_top_ref_logprob))


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize(("tp_size, pp_size"), [
    (1, 1),
    (2, 2),
])
@pytest.mark.parametrize("eager_mode", [False, True])
@pytest.mark.parametrize("num_scheduler_steps", NUM_SCHEDULER_STEPS)
@pytest.mark.parametrize("num_prompts", NUM_PROMPTS)
@pytest.mark.parametrize("num_logprobs", NUM_LOGPROBS)
@pytest.mark.asyncio
async def test_multi_step(example_prompts, model: str, tp_size: int,
                          pp_size: int, eager_mode: int,
                          num_scheduler_steps: int, num_prompts: int,
                          num_logprobs: Optional[int]):
    '''
    Test vLLM engine with multi-step scheduling in an OpenAI-protocol
    client/server environment.

    Set up an engine with single-step scheduling as a ground-truth reference.

    Send a completions API request to both engines with the same prompts.

    Validate:
    * Generated tokens match
    * Generated logprobs are all very close

    Arguments:

    * example_prompts: test fixture providing example prompts
    * model: model under test (same for single- and multi-step engines)
    * tp_size: degree of tensor-parallelism
    * pp_size: degree of pipeline-parallelism
    * eager_mode
    * num_scheduler_steps: for multi-step scheduling, GPU-side steps per
                           GPU -> CPU output transfer
    * num_prompts: number of example prompts under test
    * num_logprobs: corresponds to the `logprobs` argument to the OpenAI
                    completions endpoint; `None` -> no logprobs
    '''

    prompts = example_prompts
    if len(prompts) < num_prompts:
        prompts = prompts * ((num_prompts // len(prompts)) + 1)
    prompts = prompts[:num_prompts]
    assert len(prompts) == num_prompts

    server_args = DEFAULT_SERVER_ARGS + ["--enforce-eager"]
    ms_server_args = DEFAULT_SERVER_ARGS + \
        ["--num-scheduler-steps", f"{num_scheduler_steps}"]

    if eager_mode:
        ms_server_args.append("--enforce-eager")

    distributed_args = [
        "--tensor-parallel-size",
        str(tp_size),
        "--pipeline-parallel-size",
        str(pp_size),
    ]

    ref_completions = await completions_with_server_args(
        prompts, model, server_args + distributed_args, num_logprobs)
    test_completions = await completions_with_server_args(
        prompts, model, ms_server_args + distributed_args, num_logprobs)

    # Assert multi-step scheduling produces identical tokens
    # to single-step scheduling.
    ref_generations = get_text_generations(ref_completions)
    test_generations = get_text_generations(test_completions)
    assert ref_generations == test_generations

    # Assert multi-step scheduling produces identical logprobs
    # to single-step scheduling.
    ref_logprobs = get_logprob_generations(ref_completions)
    test_logprobs = get_logprob_generations(test_completions)
    assert_all_close_logprobs(
        ref_logprobs,
        test_logprobs,
        atol=1e-5,
        rtol=1e-5,
    )
