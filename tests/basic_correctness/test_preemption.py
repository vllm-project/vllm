# SPDX-License-Identifier: Apache-2.0
"""Compare the short outputs of HF and vLLM when using greedy sampling.

VLLM_TEST_ENABLE_ARTIFICIAL_PREEMPT=1 has to be set before running this test.

Run `VLLM_TEST_ENABLE_ARTIFICIAL_PREEMPT=1
pytest tests/basic_correctness/test_preemption.py`.
"""
import pytest
from prometheus_client import REGISTRY

import vllm.envs as envs
from vllm import SamplingParams
from vllm.core.scheduler import (ARTIFICIAL_PREEMPTION_MAX_CNT,
                                 ENABLE_ARTIFICIAL_PREEMPT)

from ..models.utils import check_outputs_equal

MODELS = [
    "distilbert/distilgpt2",
]


@pytest.fixture(scope="module", autouse=True)
def check_settings():
    assert ENABLE_ARTIFICIAL_PREEMPT is True, (
        "Use an env var VLLM_TEST_ENABLE_ARTIFICIAL_PREEMPT=1."
        "`VLLM_TEST_ENABLE_ARTIFICIAL_PREEMPT=1 "
        "pytest tests/basic_correctness/test_preemption.py`")


@pytest.fixture
def distributed_executor_backend() -> str:
    # When SPMD worker is used, use distributed_executor_backend="ray"
    # to test delta input optimization works with preemption.
    return "ray" if envs.VLLM_USE_RAY_SPMD_WORKER else "mp"


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("dtype", ["half"])
@pytest.mark.parametrize("max_tokens", [96])
@pytest.mark.parametrize("chunked_prefill_token_size", [16])
def test_chunked_prefill_recompute(
    hf_runner,
    vllm_runner,
    example_prompts,
    model: str,
    dtype: str,
    max_tokens: int,
    chunked_prefill_token_size: int,
    distributed_executor_backend: str,
) -> None:
    """Ensure that chunked prefill works with preemption."""
    max_num_seqs = min(chunked_prefill_token_size, 256)
    enable_chunked_prefill = False
    max_num_batched_tokens = None
    if chunked_prefill_token_size != -1:
        enable_chunked_prefill = True
        max_num_batched_tokens = chunked_prefill_token_size

    with hf_runner(model, dtype=dtype) as hf_model:
        hf_outputs = hf_model.generate_greedy(example_prompts, max_tokens)

    with vllm_runner(
            model,
            dtype=dtype,
            max_num_batched_tokens=max_num_batched_tokens,
            enable_chunked_prefill=enable_chunked_prefill,
            max_num_seqs=max_num_seqs,
            distributed_executor_backend=distributed_executor_backend,
            disable_log_stats=False,
    ) as vllm_model:
        vllm_outputs = vllm_model.generate_greedy(example_prompts, max_tokens)
        assert (vllm_model.model.llm_engine.scheduler[0].artificial_preempt_cnt
                < ARTIFICIAL_PREEMPTION_MAX_CNT)

    for i in range(len(example_prompts)):
        hf_output_ids, hf_output_str = hf_outputs[i]
        vllm_output_ids, vllm_output_str = vllm_outputs[i]
        assert hf_output_str == vllm_output_str, (
            f"Test{i}:\nHF: {hf_output_str!r}\nvLLM: {vllm_output_str!r}")
        assert hf_output_ids == vllm_output_ids, (
            f"Test{i}:\nHF: {hf_output_ids}\nvLLM: {vllm_output_ids}")


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("dtype", ["float"])
@pytest.mark.parametrize("max_tokens", [96])
def test_preemption(
    caplog_vllm,
    hf_runner,
    vllm_runner,
    example_prompts,
    model: str,
    dtype: str,
    max_tokens: int,
    distributed_executor_backend: str,
) -> None:
    """By default, recompute preemption is enabled"""

    with hf_runner(model, dtype=dtype) as hf_model:
        hf_outputs = hf_model.generate_greedy(example_prompts, max_tokens)

    with vllm_runner(
            model,
            dtype=dtype,
            disable_log_stats=False,
            distributed_executor_backend=distributed_executor_backend,
    ) as vllm_model:
        vllm_outputs = vllm_model.generate_greedy(example_prompts, max_tokens)
        assert (vllm_model.model.llm_engine.scheduler[0].artificial_preempt_cnt
                < ARTIFICIAL_PREEMPTION_MAX_CNT)
        total_preemption = (
            vllm_model.model.llm_engine.scheduler[0].num_cumulative_preemption)

    check_outputs_equal(
        outputs_0_lst=hf_outputs,
        outputs_1_lst=vllm_outputs,
        name_0="hf",
        name_1="vllm",
    )

    assert ("is preempted by PreemptionMode.RECOMPUTE mode because there "
            "is not enough KV cache space." in caplog_vllm.text)
    # Ensure the count bucket of request-level histogram metrics matches
    # the number of requests as a simple sanity check to ensure metrics are
    # generated
    preemption_metrics = None
    for m in REGISTRY.collect():
        if m.name == "vllm:num_preemptions":
            preemption_metrics = m
    assert preemption_metrics is not None
    total_recorded_preemption = 0
    for sample in preemption_metrics.samples:
        total_recorded_preemption += sample.value
    assert total_preemption == total_recorded_preemption


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("dtype", ["float"])
@pytest.mark.parametrize("max_tokens", [96])
def test_preemption_infeasible(
    vllm_runner,
    example_prompts,
    model: str,
    dtype: str,
    max_tokens: int,
    distributed_executor_backend: str,
) -> None:
    """Verify infeasible preemption request will be ignored."""
    BLOCK_SIZE = 16
    prefill_blocks = 2
    decode_blocks = max_tokens // BLOCK_SIZE
    with vllm_runner(
            model,
            dtype=dtype,
            block_size=BLOCK_SIZE,
            # Not enough gpu blocks to complete a single sequence.
            # preemption should happen, and the sequence should be
            # ignored instead of hanging forever.
            num_gpu_blocks_override=prefill_blocks + decode_blocks // 2,
            max_model_len=((prefill_blocks + decode_blocks // 2) * BLOCK_SIZE),
            distributed_executor_backend=distributed_executor_backend,
    ) as vllm_model:
        sampling_params = SamplingParams(max_tokens=max_tokens,
                                         ignore_eos=True)
        req_outputs = vllm_model.model.generate(
            example_prompts,
            sampling_params=sampling_params,
        )

        assert (vllm_model.model.llm_engine.scheduler[0].artificial_preempt_cnt
                < ARTIFICIAL_PREEMPTION_MAX_CNT)

    # Verify the request is ignored and not hang.
    for req_output in req_outputs:
        outputs = req_output.outputs
        assert len(outputs) == 1
        assert outputs[0].finish_reason == "length"
