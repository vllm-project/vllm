# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import random
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, call

import pytest

from vllm import LLM
from vllm.engine.arg_utils import EngineArgs
from vllm.sampling_params import (
    RequestOutputKind,
    SamplingParams,
    StructuredOutputsParams,
)
from vllm.v1.engine.llm_engine import LLMEngine
from vllm.v1.metrics.reader import Counter, Gauge, Histogram, Metric, Vector
from vllm.v1.outputs import RequestOutput

if TYPE_CHECKING:
    from tests.conftest import VllmRunner
else:
    VllmRunner = object

MODEL = "facebook/opt-125m"
DTYPE = "half"


@pytest.mark.skip_global_cleanup
def test_step_invalidates_mm_processor_cache_misses():
    engine = object.__new__(LLMEngine)
    engine.should_execute_dummy_batch = False
    engine.log_stats = False
    engine.logger_manager = None

    mm_processor_cache = MagicMock()
    engine.renderer = MagicMock(mm_processor_cache=mm_processor_cache)

    engine_core_output = MagicMock(
        mm_cache_miss_hashes=["missing-hash-1", "missing-hash-2"]
    )
    outputs = MagicMock(
        outputs=[engine_core_output],
        timestamp=0.0,
        scheduler_stats=None,
    )
    engine.engine_core = MagicMock()
    engine.engine_core.get_output.return_value = outputs

    processed_outputs = MagicMock(reqs_to_abort=[], request_outputs=[])
    engine.output_processor = MagicMock()
    engine.output_processor.process_outputs.return_value = processed_outputs

    assert engine.step() == []
    assert mm_processor_cache.invalidate.call_args_list == [
        call("missing-hash-1"),
        call("missing-hash-2"),
    ]


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
        r"""Mostly n \in [2,20], ~1/3 n=1."""
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


@pytest.mark.parametrize(
    "output_kind",
    [
        RequestOutputKind.CUMULATIVE,
        RequestOutputKind.DELTA,
        RequestOutputKind.FINAL_ONLY,
    ],
)
def test_parallel_sampling_llm_engine_output_kinds(
    output_kind: RequestOutputKind,
) -> None:
    """LLMEngine.add_request()/step() with parallel sampling (n>1) yields n
    unique completions for every RequestOutputKind.

    Regression guard for https://github.com/vllm-project/vllm/issues/21948:
    the LLMEngine + CUMULATIVE combination was reported broken, and the
    LLMEngine add_request()/step() path had no explicit parallel-sampling
    coverage at all. Each entry point x output_kind combination must be
    exercised explicitly.
    """
    engine_args = EngineArgs(
        model=MODEL,
        dtype=DTYPE,
        max_model_len=128,
        enforce_eager=True,
        gpu_memory_utilization=0.5,
    )
    engine = LLMEngine.from_engine_args(engine_args)
    try:
        n = 2
        max_tokens = 8
        engine.add_request(
            "request-0",
            "Hello, my name is",
            SamplingParams(
                max_tokens=max_tokens,
                n=n,
                output_kind=output_kind,
                ignore_eos=True,
                temperature=0.5,
                seed=33,
            ),
        )
        outputs: list[RequestOutput] = []
        while engine.has_unfinished_requests():
            outputs.extend(engine.step())

        # Collect per-completion token-length histories across all steps.
        per_seq: dict[int, list[int]] = {i: [] for i in range(n)}
        finished: set[int] = set()
        for out in outputs:
            for comp in out.outputs:
                per_seq[comp.index].append(len(comp.token_ids))
                if comp.finished:
                    finished.add(comp.index)

        # Every one of the n parallel completions must finish.
        assert finished == set(range(n)), (
            f"{output_kind}: expected all {n} completions to finish, got {finished}"
        )
        # Every completion must reach the requested length.
        for i in range(n):
            lens = per_seq[i]
            # DELTA reports per-step token deltas; CUMULATIVE/FINAL_ONLY
            # report cumulative snapshots / the full final sequence.
            total = sum(lens) if output_kind == RequestOutputKind.DELTA else lens[-1]
            assert total == max_tokens, (
                f"{output_kind}: completion {i} must reach {max_tokens} "
                f"tokens, got {lens}"
            )
        # CUMULATIVE snapshots must be monotone non-decreasing.
        if output_kind == RequestOutputKind.CUMULATIVE:
            for i in range(n):
                lens = per_seq[i]
                assert lens == sorted(lens), (
                    f"CUMULATIVE: snapshot lengths for completion {i} "
                    f"must be monotone non-decreasing, got {lens}"
                )
    finally:
        # Best-effort cleanup; LLMEngine has no public shutdown().
        engine.engine_core.shutdown()
