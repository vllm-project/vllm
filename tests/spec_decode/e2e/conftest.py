from itertools import cycle
from typing import Dict, List, Tuple

import pytest

from vllm import LLM
from vllm.model_executor.utils import set_random_seed
from vllm.sequence import Logprob

from ...conftest import cleanup
from ...utils import RemoteOpenAIServer


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


def get_logprobs_from_llm_generator(
        llm_generator, prompts,
        sampling_params) -> List[List[Dict[int, Logprob]]]:
    """Returns a dict of (token_id: Logprob) for each generated position, for
    each sequence in the batch.
    """
    for llm in llm_generator():
        outputs = llm.generate(prompts, sampling_params, use_tqdm=True)
        logprobs = [output.outputs[0].logprobs[:] for output in outputs]
        del llm

    return logprobs


def run_equality_correctness_test(model,
                                  common_llm_kwargs,
                                  per_test_common_llm_kwargs,
                                  baseline_llm_kwargs,
                                  test_llm_kwargs,
                                  batch_size: int,
                                  max_output_len: int,
                                  seed: int = 0,
                                  temperature: float = 0.0,
                                  disable_seed: bool = False,
                                  ensure_all_accepted: bool = False,
                                  force_output_len: bool = True):
    """Helper method that compares the outputs of both the baseline LLM and
    the test LLM. It asserts greedy equality, e.g. that the outputs are exactly
    the same when temperature is zero.
    """
    arg1 = common_llm_kwargs + per_test_common_llm_kwargs + baseline_llm_kwargs
    arg2 = common_llm_kwargs + per_test_common_llm_kwargs + test_llm_kwargs
    env1 = env2 = None

    max_wait_seconds = 240
    results = []

    prompts = [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
        "San Francisco is know for its",
        "Facebook was created in 2004 by",
        "Curious George is a",
        "Python 3.11 brings improvements to its",
    ]

    # TODO: Implement force_output_len.

    prompts = [prompt for prompt, _ in zip(cycle(prompts), range(batch_size))]

    for args, env in ((arg1, env1), (arg2, env2)):
        with RemoteOpenAIServer(model,
                                args,
                                env_dict=env,
                                max_wait_seconds=max_wait_seconds) as server:
            client = server.get_client()

            if disable_seed:
                completion = client.completions.create(
                    model=model,
                    prompt=prompts,
                    max_tokens=max_output_len,
                    temperature=temperature)
            else:
                completion = client.completions.create(
                    model=model,
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

            if ensure_all_accepted:
                # TODO: Implement this.
                print(server.get_metrics())
                # assert acceptance_rate == 1.0

    n = len(results) // 2
    arg1_results = results[:n]
    arg2_results = results[n:]
    for arg1_result, arg2_result in zip(arg1_results, arg2_results):
        assert arg1_result == arg2_result, (
            f"Results for {model=} are not the same with {arg1=} and {arg2=}. "
            f"{arg1_result=} != {arg2_result=}")


# def run_equality_correctness_test(
#         baseline_llm_generator,
#         test_llm_generator,
#         batch_size,
#         max_output_len,
#         force_output_len: bool,
#         temperature: float,
#         seeded: bool,
#         print_tokens: bool = False,
#         ensure_all_accepted: bool = False,
#         expected_acceptance_rate: Optional[float] = None):
#     """Helper method that compares the outputs of both the baseline LLM and
#     the test LLM. It asserts greedy equality, e.g. that the outputs are exactly
#     the same when temperature is zero (or when temperature is > 0 and seeded).
#     """

#     prompts = [
#         "Hello, my name is",
#         "The president of the United States is",
#         "The capital of France is",
#         "The future of AI is",
#         "San Francisco is know for its",
#         "Facebook was created in 2004 by",
#         "Curious George is a",
#         "Python 3.11 brings improvements to its",
#     ]

#     prompts = [prompt for prompt, _ in zip(cycle(prompts), range(batch_size))]

#     # If the test requires that we generated max_output_len tokens, then set the
#     # sampling params to ignore eos token.
#     ignore_eos = force_output_len

#     if seeded:
#         sampling_params = [
#             SamplingParams(
#                 max_tokens=max_output_len,
#                 ignore_eos=ignore_eos,
#                 temperature=temperature,
#                 seed=i,
#             ) for i in range(len(prompts))
#         ]
#     else:
#         sampling_params = SamplingParams(
#             max_tokens=max_output_len,
#             ignore_eos=ignore_eos,
#             temperature=temperature,
#         )

#     (spec_batch_tokens, spec_batch_token_ids,
#      acceptance_rate) = get_output_from_llm_generator(test_llm_generator,
#                                                       prompts, sampling_params)

#     (baseline_batch_tokens, baseline_batch_token_ids,
#      _) = get_output_from_llm_generator(baseline_llm_generator, prompts,
#                                         sampling_params)

#     assert len(baseline_batch_token_ids) == len(prompts)
#     assert len(spec_batch_token_ids) == len(prompts)

#     for i, (baseline_token_ids, baseline_tokens, spec_token_ids,
#             spec_tokens) in enumerate(
#                 zip(baseline_batch_token_ids, baseline_batch_tokens,
#                     spec_batch_token_ids, spec_batch_tokens)):
#         if print_tokens:
#             print(f'{i=} {baseline_tokens=}')
#             print(f'{i=}     {spec_tokens=}')
#         print(f'{i=} {baseline_token_ids=}')
#         print(f'{i=}     {spec_token_ids=}')
#         assert baseline_token_ids == spec_token_ids

#     print(f'{acceptance_rate=}')

#     if ensure_all_accepted:
#         assert acceptance_rate == 1.0

#     if expected_acceptance_rate is not None:
#         assert acceptance_rate >= expected_acceptance_rate - 1e-2
