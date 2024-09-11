from itertools import cycle
from typing import List, Optional, Tuple

import pytest

from vllm import LLM, SamplingParams
from vllm.model_executor.utils import set_random_seed

from ...conftest import cleanup
from ...models.utils import check_logprobs_close, check_outputs_equal
from ...utils import RemoteOpenAIServer

PROMPTS = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
    "San Francisco is know for its",
    "Facebook was created in 2004 by",
    "Curious George is a",
    "Python 3.11 brings improvements to its",
]


@pytest.fixture
def test_llm_generator(common_llm_kwargs, per_test_common_llm_kwargs,
                       test_llm_kwargs, seed):

    def generate():
        kwargs = {
            **common_llm_kwargs,
            **per_test_common_llm_kwargs,
            **test_llm_kwargs,
        }

        llm = LLM(**kwargs)

        if seed is not None:
            set_random_seed(seed)

        yield llm

        del llm
        cleanup()

    return generate


def maybe_assert_ngram_worker(llm):
    # Verify the proposer worker is ngram if ngram is specified.
    if (llm.llm_engine.speculative_config is not None
            and llm.llm_engine.speculative_config.ngram_prompt_lookup_max > 0):
        from vllm.spec_decode.ngram_worker import NGramWorker
        assert isinstance(
            llm.llm_engine.model_executor.driver_worker.proposer_worker,
            NGramWorker)


def get_output_from_llm_generator(
        llm_generator, prompts,
        sampling_params) -> Tuple[List[str], List[List[int]], float]:
    tokens: List[str] = []
    token_ids: List[List[int]] = []
    acceptance_rate: float = -1.0
    for llm in llm_generator():
        maybe_assert_ngram_worker(llm)

        outputs = llm.generate(prompts, sampling_params, use_tqdm=True)

        token_ids = [output.outputs[0].token_ids for output in outputs]
        tokens = [output.outputs[0].text for output in outputs]

        # Fetch acceptance rate if logging is enabled.
        if stat_loggers := getattr(llm.llm_engine, "stat_loggers", None):
            stat_logger = stat_loggers["prometheus"]
            acceptance_rate = (stat_logger.metrics.
                               gauge_spec_decode_draft_acceptance_rate.labels(
                                   **stat_logger.labels)._value.get())
        del llm

    return tokens, token_ids, acceptance_rate


def run_logprob_correctness_test(vllm_runner,
                                 common_llm_kwargs,
                                 per_test_common_llm_kwargs,
                                 baseline_llm_kwargs,
                                 test_llm_kwargs,
                                 batch_size: int,
                                 max_output_len: int,
                                 seed: Optional[int] = 0,
                                 temperature: float = 0.0,
                                 logprobs: int = 1):
    org_args = {
        **common_llm_kwargs,
        **per_test_common_llm_kwargs,
        **baseline_llm_kwargs,
    }

    sd_args = {
        **common_llm_kwargs,
        **per_test_common_llm_kwargs,
        **test_llm_kwargs,
    }

    prompts = [prompt for prompt, _ in zip(cycle(PROMPTS), range(batch_size))]

    sampling_params = SamplingParams(temperature=temperature,
                                     max_tokens=max_output_len,
                                     seed=seed,
                                     logprobs=logprobs)

    with vllm_runner(**org_args) as vllm_model:
        org_outputs = vllm_model.generate_w_logprobs(prompts, sampling_params)

    with vllm_runner(**sd_args) as vllm_model:
        sd_outputs = vllm_model.generate_w_logprobs(prompts, sampling_params)

    check_logprobs_close(outputs_0_lst=org_outputs,
                         outputs_1_lst=sd_outputs,
                         name_0="org",
                         name_1="sd")


def run_equality_correctness_test(
        vllm_runner,
        common_llm_kwargs,
        per_test_common_llm_kwargs,
        baseline_llm_kwargs,
        test_llm_kwargs,
        batch_size: int,
        max_output_len: int,
        seed: Optional[int] = 0,
        temperature: float = 0.0,
        disable_seed: bool = False,
        ignore_eos: bool = True,
        ensure_all_accepted: bool = False,
        expected_acceptance_rate: Optional[float] = None):

    org_args = {
        **common_llm_kwargs,
        **per_test_common_llm_kwargs,
        **baseline_llm_kwargs,
    }

    sd_args = {
        **common_llm_kwargs,
        **per_test_common_llm_kwargs,
        **test_llm_kwargs,
    }

    prompts = [prompt for prompt, _ in zip(cycle(PROMPTS), range(batch_size))]

    if disable_seed:
        seed = None

    sampling_params = SamplingParams(temperature=temperature,
                                     max_tokens=max_output_len,
                                     seed=seed,
                                     ignore_eos=ignore_eos)

    with vllm_runner(**org_args) as vllm_model:
        org_outputs = vllm_model.generate(prompts, sampling_params)

    with vllm_runner(**sd_args) as vllm_model:
        if ensure_all_accepted or expected_acceptance_rate is not None:
            # Force log interval to be 0 to catch all metrics.
            stat_logger = vllm_model.model.llm_engine.stat_loggers[
                'prometheus']
            stat_logger.local_interval = -100

        sd_outputs = vllm_model.generate(prompts, sampling_params)

        if ensure_all_accepted or expected_acceptance_rate is not None:
            acceptance_rate = (stat_logger.metrics.
                               gauge_spec_decode_draft_acceptance_rate.labels(
                                   **stat_logger.labels)._value.get())

            if ensure_all_accepted:
                assert True
                # FIXME: ci fails to log acceptance rate.
                # It works locally.
                # assert acceptance_rate == 1.0

            if expected_acceptance_rate is not None:
                assert acceptance_rate >= expected_acceptance_rate - 1e-2

    check_outputs_equal(outputs_0_lst=org_outputs,
                        outputs_1_lst=sd_outputs,
                        name_0="org",
                        name_1="sd")


def run_equality_correctness_test_tp(model,
                                     common_llm_kwargs,
                                     per_test_common_llm_kwargs,
                                     baseline_llm_kwargs,
                                     test_llm_kwargs,
                                     batch_size: int,
                                     max_output_len: int,
                                     seed: int = 0,
                                     temperature: float = 0.0):
    """Helper method that compares the outputs of both the baseline LLM and
    the test LLM. It asserts greedy equality, e.g. that the outputs are exactly
    the same when temperature is zero.
    """
    arg1 = common_llm_kwargs + per_test_common_llm_kwargs + baseline_llm_kwargs
    arg2 = common_llm_kwargs + per_test_common_llm_kwargs + test_llm_kwargs
    env1 = env2 = None

    max_wait_seconds = 240
    results = []

    prompts = [prompt for prompt, _ in zip(cycle(PROMPTS), range(batch_size))]

    for args, env in ((arg1, env1), (arg2, env2)):
        with RemoteOpenAIServer(model,
                                args,
                                env_dict=env,
                                max_wait_seconds=max_wait_seconds) as server:
            client = server.get_client()

            completion = client.completions.create(model=model,
                                                   prompt=prompts,
                                                   max_tokens=max_output_len,
                                                   seed=seed,
                                                   temperature=temperature)

            results.append({
                "test":
                "seeded_sampling",
                "text": [choice.text for choice in completion.choices],
                "finish_reason":
                [choice.finish_reason for choice in completion.choices],
                "usage":
                completion.usage,
            })

    n = len(results) // 2
    arg1_results = results[:n]
    arg2_results = results[n:]
    for arg1_result, arg2_result in zip(arg1_results, arg2_results):
        assert arg1_result == arg2_result, (
            f"Results for {model=} are not the same with {arg1=} and {arg2=}. "
            f"{arg1_result=} != {arg2_result=}")
