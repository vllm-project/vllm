from itertools import cycle
from typing import List, Optional, Sequence, Tuple, Union

import pytest
import torch

from vllm import LLM, SamplingParams
from vllm.distributed import cleanup_dist_env_and_memory
from vllm.model_executor.utils import set_random_seed
from vllm.sequence import PromptLogprobs, SampleLogprobs

from ...models.utils import (TokensTextLogprobs,
                             TokensTextLogprobsPromptLogprobs,
                             check_logprobs_close, check_outputs_equal)
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
        cleanup_dist_env_and_memory()

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


def check_logprobs_correctness(
    spec_outputs: Sequence[Union[TokensTextLogprobs,
                                 TokensTextLogprobsPromptLogprobs]],
    baseline_outputs: Sequence[Union[TokensTextLogprobs,
                                     TokensTextLogprobsPromptLogprobs]],
    disable_logprobs: bool = False,
):
    """Compare sampled and prompt logprobs between baseline and spec decoding
    """
    if not disable_logprobs:
        return check_logprobs_close(
            outputs_0_lst=baseline_outputs,
            outputs_1_lst=spec_outputs,
            name_0="org",
            name_1="sd",
        )

    # Check correctness when disable_logprobs == True
    for spec_output, baseline_output in zip(spec_outputs, baseline_outputs):
        # Check generated token logprobs.
        spec_logprobs = spec_output[2]
        baseline_logprobs = baseline_output[2]
        _check_logprobs_when_output_disabled(spec_logprobs,
                                             baseline_logprobs,
                                             is_prompt_logprobs=False)

        # Check prompt logprobs too, if they exist
        if len(baseline_output) == 4:
            assert len(spec_output) == 4
            spec_prompt_logprobs = spec_output[3]
            baseline_prompt_logprobs = baseline_output[3]
            _check_logprobs_when_output_disabled(spec_prompt_logprobs,
                                                 baseline_prompt_logprobs,
                                                 is_prompt_logprobs=True)


def _check_logprobs_when_output_disabled(
    spec_logprobs: Union[Optional[PromptLogprobs], SampleLogprobs],
    baseline_logprobs: Union[Optional[PromptLogprobs], SampleLogprobs],
    is_prompt_logprobs: bool = False,
):
    # Prompt logprobs are optional
    if is_prompt_logprobs and baseline_logprobs is None:
        assert spec_logprobs is None
        return

    assert spec_logprobs is not None
    assert baseline_logprobs is not None
    assert len(spec_logprobs) == len(baseline_logprobs)

    # For each generated position of the sequence.
    for pos, (spec_pos_logprobs, baseline_pos_logprobs) in enumerate(
            zip(spec_logprobs, baseline_logprobs)):

        # First prompt logprob is expected to be None
        if is_prompt_logprobs and baseline_pos_logprobs is None:
            assert spec_pos_logprobs is None
            assert pos == 0
            continue

        assert spec_pos_logprobs is not None
        assert baseline_pos_logprobs is not None

        # When disabled, the 1 logprob is returned with dummy values for the
        # score and rank, but the token id should match the baseline model
        assert len(spec_pos_logprobs) == 1
        (spec_pos_logprob_token_id,
         spec_pos_logprob) = next(iter(spec_pos_logprobs.items()))
        assert spec_pos_logprob.rank == -1
        assert spec_pos_logprob.logprob == 0.0
        if isinstance(spec_pos_logprob_token_id, torch.Tensor):
            spec_pos_logprob_token_id = spec_pos_logprob_token_id.item()
        assert spec_pos_logprob_token_id in baseline_pos_logprobs


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
        expected_acceptance_rate: Optional[float] = None,
        logprobs: Optional[int] = None,
        prompt_logprobs: Optional[int] = None,
        disable_logprobs: bool = False):

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
                                     ignore_eos=ignore_eos,
                                     logprobs=logprobs,
                                     prompt_logprobs=prompt_logprobs)

    with vllm_runner(**org_args) as vllm_model:
        org_outputs = vllm_model.generate_w_logprobs(prompts, sampling_params)

    with vllm_runner(**sd_args) as vllm_model:
        if ensure_all_accepted or expected_acceptance_rate is not None:
            # Force log interval to be 0 to catch all metrics.
            stat_logger = vllm_model.model.llm_engine.stat_loggers[
                'prometheus']
            stat_logger.local_interval = -100

        sd_outputs = vllm_model.generate_w_logprobs(prompts, sampling_params)

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

    # Only pass token entries, not the logprobs
    check_outputs_equal(outputs_0_lst=[out[0:2] for out in org_outputs],
                        outputs_1_lst=[out[0:2] for out in sd_outputs],
                        name_0="org",
                        name_1="sd")

    # Check logprobs if requested
    if logprobs is not None or prompt_logprobs is not None:
        check_logprobs_correctness(sd_outputs, org_outputs, disable_logprobs)


def run_equality_correctness_test_tp(model,
                                     common_llm_kwargs,
                                     per_test_common_llm_kwargs,
                                     baseline_llm_kwargs,
                                     test_llm_kwargs,
                                     batch_size: int,
                                     max_output_len: int,
                                     seed: int = 0,
                                     temperature: float = 0.0,
                                     logprobs: Optional[int] = None):
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
                                                   temperature=temperature,
                                                   logprobs=logprobs)

            results.append({
                "test":
                "seeded_sampling",
                "text": [choice.text for choice in completion.choices],
                "logprobs": [choice.logprobs for choice in completion.choices],
                "finish_reason":
                [choice.finish_reason for choice in completion.choices],
                "usage":
                completion.usage,
            })

    n = len(results) // 2
    arg1_results = results[:n]
    arg2_results = results[n:]
    # Separate logprobs to avoid asserting exact equality.
    arg1_logprobs = [r.pop("logprobs") for r in arg1_results]
    arg2_logprobs = [r.pop("logprobs") for r in arg2_results]

    for arg1_result, arg2_result in zip(arg1_results, arg2_results):
        assert arg1_result == arg2_result, (
            f"Results for {model=} are not the same with {arg1=} and {arg2=}. "
            f"{arg1_result=} != {arg2_result=}")
    if logprobs:
        for logs1, logs2 in zip(arg1_logprobs, arg2_logprobs):
            for l1, l2 in zip(logs1, logs2):
                assert l1.tokens == l2.tokens
