# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import random
from typing import TYPE_CHECKING

import pytest

from tests.utils import create_new_process_for_each_test
from vllm import LLM
from vllm.sampling_params import (
    RequestOutputKind,
    SamplingParams,
    StructuredOutputsParams,
)
from vllm.v1.metrics.reader import Counter, Gauge, Histogram, Metric, Vector

if TYPE_CHECKING:
    from tests.conftest import VllmRunner
else:
    VllmRunner = object

MODEL = "facebook/opt-125m"
DTYPE = "half"


def _vllm_model(
    apc: bool,
    vllm_runner: type[VllmRunner],
    *,
    skip_tokenizer_init: bool = False,
):
    """Set up VllmRunner instance."""
    return vllm_runner(
        MODEL,
        dtype=DTYPE,
        max_model_len=128,
        enforce_eager=True,
        enable_prefix_caching=apc,
        gpu_memory_utilization=0.5,
        skip_tokenizer_init=skip_tokenizer_init,
    )


@pytest.fixture(
    # Function scope decouples tests & allows
    # env var adjustment via monkeypatch
    scope="function",
    # Prefix caching
    params=[False, True],
)
def vllm_model(vllm_runner, request):
    """VllmRunner test fixture parameterized by APC True/False."""
    with _vllm_model(request.param, vllm_runner) as vllm_model:
        yield vllm_model


@pytest.fixture(scope="function")
def vllm_model_apc(vllm_runner):
    """VllmRunner test fixture with APC."""
    with _vllm_model(True, vllm_runner) as vllm_model:
        yield vllm_model


@pytest.fixture(
    # Function scope decouples tests & allows
    # env var adjustment via monkeypatch
    scope="function",
    # Prefix caching
    params=[False, True],
)
def vllm_model_skip_tokenizer_init(vllm_runner, request):
    """VllmRunner test fixture with APC."""
    with _vllm_model(
        request.param,
        vllm_runner,
        skip_tokenizer_init=True,
    ) as vllm_model:
        yield vllm_model


def _get_test_sampling_params(
    prompt_list: list[str],
    seed: int | None = 42,
    structured_outputs: bool = False,
) -> tuple[list[SamplingParams], list[int]]:
    """Generate random sampling params for a batch."""
    rng = random.Random(seed)

    def get_mostly_n_gt1() -> int:
        r"""Mostly n \in [2,20], ~1/3 n=1"""
        x = rng.randint(0, 28)
        if x < 10:
            return 1
        else:
            return x - 8

    n_list = [get_mostly_n_gt1() for _ in range(len(prompt_list))]
    # High temperature to maximize the chance of unique completions
    return [
        SamplingParams(
            temperature=0.95,
            top_p=0.95,
            n=n,
            seed=seed,
            structured_outputs=StructuredOutputsParams(regex="[0-9]+")
            if structured_outputs
            else None,
        )
        for n in n_list
    ], n_list


def test_compatibility_with_skip_tokenizer_init(
    vllm_model_skip_tokenizer_init: VllmRunner,
    example_prompts: list[str],
):
    # Case 1: Structured output request should raise an error.
    sampling_params_list, _ = _get_test_sampling_params(
        example_prompts,
        structured_outputs=True,
    )
    llm: LLM = vllm_model_skip_tokenizer_init.llm
    with pytest.raises(ValueError):
        _ = llm.generate(example_prompts, sampling_params_list)


def test_parallel_sampling(vllm_model, example_prompts) -> None:
    """Test passes if parallel sampling `n>1` yields `n` unique completions.

    Args:
      vllm_model: VllmRunner instance under test.
      example_prompt: test fixture providing prompts for testing.
    """
    sampling_params_list, n_list = _get_test_sampling_params(example_prompts)
    llm: LLM = vllm_model.llm
    outputs = llm.generate(example_prompts, sampling_params_list)

    # Validate each request response
    for out, n in zip(outputs, n_list):
        completion_counts: dict[str, int] = {}
        # Assert correct number of completions
        assert len(out.outputs) == n, f"{len(out.outputs)} completions; {n} expected."
        for idx in range(n):
            comp = out.outputs[idx]
            # Assert correct completion indices
            assert comp.index == idx, f"Index {comp.index}; expected {idx}."
            text = comp.text
            completion_counts[text] = completion_counts.get(text, 0) + 1
        # Assert unique completions
        if len(completion_counts) != n:
            repeats = {txt: num for (txt, num) in completion_counts.items() if num > 1}
            raise AssertionError(
                f"{len(completion_counts)} unique completions; expected"
                f" {n}. Repeats: {repeats}"
            )


def test_engine_metrics(vllm_runner, example_prompts):
    max_tokens = 100
    # Use spec decoding to test num_accepted_tokens_per_pos
    speculative_config = {
        "method": "ngram",
        "prompt_lookup_max": 5,
        "prompt_lookup_min": 3,
        "num_speculative_tokens": 5,
    }

    with vllm_runner(
        MODEL,
        speculative_config=speculative_config,
        disable_log_stats=False,
    ) as vllm_model:
        llm: LLM = vllm_model.llm
        sampling_params = SamplingParams(temperature=0.0, max_tokens=max_tokens)
        outputs = llm.generate(example_prompts, sampling_params)

        n_prompts = len(example_prompts)
        assert len(outputs) == n_prompts

        total_tokens = 0
        for out in outputs:
            assert len(out.outputs) == 1
            total_tokens += len(out.outputs[0].token_ids)
        assert total_tokens == max_tokens * n_prompts

        metrics = llm.get_metrics()

        def find_metric(name) -> list[Metric]:
            found = []
            for metric in metrics:
                if metric.name == name:
                    found.append(metric)
            return found

        num_requests_running = find_metric("vllm:num_requests_running")
        assert len(num_requests_running) == 1
        assert isinstance(num_requests_running[0], Gauge)
        assert num_requests_running[0].value == 0.0

        generation_tokens = find_metric("vllm:generation_tokens")
        assert len(generation_tokens) == 1
        assert isinstance(generation_tokens[0], Counter)
        assert generation_tokens[0].value == total_tokens

        request_generation_tokens = find_metric("vllm:request_generation_tokens")
        assert len(request_generation_tokens) == 1
        assert isinstance(request_generation_tokens[0], Histogram)
        assert "+Inf" in request_generation_tokens[0].buckets
        assert request_generation_tokens[0].buckets["+Inf"] == n_prompts
        assert request_generation_tokens[0].count == n_prompts
        assert request_generation_tokens[0].sum == total_tokens

        num_accepted_tokens_per_pos = find_metric(
            "vllm:spec_decode_num_accepted_tokens_per_pos"
        )
        assert len(num_accepted_tokens_per_pos) == 1
        assert isinstance(num_accepted_tokens_per_pos[0], Vector)
        assert len(num_accepted_tokens_per_pos[0].values) == 5


@pytest.mark.parametrize("model", ["meta-llama/Llama-3.2-1B-Instruct"])
def test_skip_tokenizer_initialization(model: str):
    # This test checks if the flag skip_tokenizer_init skips the initialization
    # of tokenizer and detokenizer. The generated output is expected to contain
    # token ids.
    llm = LLM(
        model=model,
        skip_tokenizer_init=True,
        enforce_eager=True,
    )
    sampling_params = SamplingParams(prompt_logprobs=True, detokenize=True)

    with pytest.raises(ValueError, match="`skip_tokenizer_init=True`"):
        llm.generate("abc", sampling_params)

    outputs = llm.generate(
        {"prompt_token_ids": [1, 2, 3]}, sampling_params=sampling_params
    )
    assert len(outputs) > 0
    completions = outputs[0].outputs
    assert len(completions) > 0
    assert completions[0].text == ""
    assert completions[0].token_ids


@pytest.mark.parametrize("output_kind", list(RequestOutputKind))
@pytest.mark.parametrize("level", [0, 1, 2])
@create_new_process_for_each_test()
def test_inproc_sleep_wait_preserves_requests(
    vllm_runner, monkeypatch, output_kind, level
):
    """Drain running generations without losing outputs or admitting waiters."""
    from vllm.v1.engine.output_processor import RequestOutputCollector

    monkeypatch.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    with vllm_runner(
        MODEL,
        dtype=DTYPE,
        max_model_len=128,
        max_num_seqs=1,
        enforce_eager=level != 1,
        enable_sleep_mode=True,
        enable_prefix_caching=False,
        async_scheduling=level == 1,
        compilation_config={
            "mode": 0,
            "cudagraph_mode": "FULL",
            "cudagraph_capture_sizes": [1],
        },
        gpu_memory_utilization=0.2,
    ) as model:
        llm = model.llm
        engine = llm.llm_engine
        prompt = "The capital of France is"
        params = SamplingParams(
            temperature=0,
            max_tokens=16,
            ignore_eos=True,
            logprobs=3,
            output_kind=RequestOutputKind.FINAL_ONLY,
        )
        expected = llm.generate([prompt], params, use_tqdm=False)[0].outputs[0]
        if output_kind == RequestOutputKind.DELTA:
            params = SamplingParams(
                temperature=0,
                max_tokens=16,
                ignore_eos=True,
                logprobs=3,
                output_kind=RequestOutputKind.FINAL_ONLY,
                stop=[llm.get_tokenizer().decode(expected.token_ids[:4])],
            )
            expected = llm.generate([prompt], params, use_tqdm=False)[0].outputs[0]
            assert expected.finish_reason == "stop"
            assert len(expected.token_ids) < params.max_tokens
        params.output_kind = output_kind
        collector = RequestOutputCollector(output_kind, "running")
        engine.add_request("running", prompt, params)
        core = engine.engine_core.engine_core
        while not core.scheduler.running:
            for output in engine.step():
                collector.put(output)
        engine.add_request("waiting", prompt, params)

        llm.sleep(level=level, mode="wait")

        assert not core.scheduler.running
        assert len(core.scheduler.waiting) == 1
        assert engine.is_sleeping()
        assert engine.has_unfinished_requests()
        assert engine.get_num_unfinished_requests() == 2
        drained = engine.step()
        assert [output.request_id for output in drained] == ["running"]
        for output in drained:
            collector.put(output)
        result = collector.get_nowait()
        assert result is not None and result.finished
        actual = result.outputs[0]
        assert actual.token_ids == expected.token_ids
        assert actual.text == expected.text
        assert actual.finish_reason == expected.finish_reason
        assert len(actual.logprobs) == len(expected.logprobs)
        for actual_step, expected_step in zip(actual.logprobs, expected.logprobs):
            assert actual_step.keys() == expected_step.keys()
            for token_id in actual_step:
                assert actual_step[token_id].logprob == pytest.approx(
                    expected_step[token_id].logprob, abs=1e-4
                )
        assert len(core.scheduler.waiting) == 1

        assert engine.get_num_unfinished_requests() == 1
        if level >= 1:
            llm.wake_up(tags=["weights"])
            assert engine.is_sleeping()
            assert engine.step() == []
            if level == 2:
                llm.collective_rpc("reload_weights")
            llm.wake_up(tags=["kv_cache"])
        else:
            llm.wake_up(tags=["scheduling"])
        waiting = RequestOutputCollector(output_kind, "waiting")
        while engine.has_unfinished_requests():
            for output in engine.step():
                assert output.request_id == "waiting"
                waiting.put(output)
        result = waiting.get_nowait()
        assert result is not None and result.finished
        assert result.outputs[0].token_ids == expected.token_ids


def _inproc_client_for_drain():
    from collections import deque
    from unittest.mock import MagicMock

    from vllm.v1.engine.core_client import InprocClient

    client = object.__new__(InprocClient)
    client.engine_core = MagicMock()
    client.engine_core.vllm_config.parallel_config.data_parallel_size = 1
    client.engine_core.model_executor.is_sleeping = False
    client.engine_core.sleep.return_value = None
    client.engine_core.batch_queue = []
    client._pending_outputs = deque()
    return client


def test_inproc_wait_delivers_raw_outputs_after_batch_cleanup():
    """A drained client retains every output and flushes queued model batches."""
    from vllm.v1.engine import EngineCoreOutput, EngineCoreOutputs

    client = _inproc_client_for_drain()
    core = client.engine_core
    core.scheduler.has_requests.side_effect = [True, False, False]
    core.batch_queue = [object()]
    count = 0

    def step():
        nonlocal count
        count += 1
        if count == 2:
            core.batch_queue.clear()
        return {0: EngineCoreOutputs(outputs=[EngineCoreOutput("r", [count])])}, True

    core.step_fn.side_effect = step
    client.sleep(level=1, mode="wait")
    core.sleep.assert_called_once_with(1, "keep")
    assert core.post_step.call_count == 2
    assert client.get_output().outputs[0].new_token_ids == [1]
    assert client.get_output().outputs[0].new_token_ids == [2]
    assert core.step_fn.call_count == 2


@pytest.mark.parametrize("dp_size,asleep", [(2, False), (1, True)])
def test_inproc_drain_rejects_unsafe_state_before_scheduling(dp_size, asleep):
    """Never run a local-only drain across DP ranks or unmapped memory."""
    from unittest.mock import MagicMock

    client = _inproc_client_for_drain()
    core = client.engine_core
    core.vllm_config.parallel_config.data_parallel_size = dp_size
    core.model_executor.is_sleeping = asleep
    step = MagicMock()
    with pytest.raises(ValueError, match="data parallelism|Wake the engine"):
        client.drain_requests(step)
    core.scheduler.set_pause_state.assert_not_called()
    step.assert_not_called()
    core.sleep.assert_not_called()


def test_inproc_wait_does_not_sleep_after_step_failure():
    """A failed drain must propagate the error before releasing GPU memory."""
    client = _inproc_client_for_drain()
    core = client.engine_core
    core.scheduler.has_requests.return_value = True
    core.step_fn.side_effect = RuntimeError("model execution failed")
    with pytest.raises(RuntimeError, match="model execution failed"):
        client.sleep(level=1, mode="wait")
    core.sleep.assert_not_called()


def test_inproc_wait_with_no_running_requests_does_not_step():
    """Waiting-only and idle engines can sleep without admitting a new request."""
    client = _inproc_client_for_drain()
    core = client.engine_core
    core.scheduler.has_requests.return_value = False
    client.sleep(level=0, mode="wait")
    core.step_fn.assert_not_called()
    core.sleep.assert_called_once_with(0, "keep")
