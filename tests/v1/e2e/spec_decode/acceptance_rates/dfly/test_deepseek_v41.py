# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Real-weight DeepSeek-V4.1 DFly2 verification, cache reuse and model evaluation."""

import pytest

from tests.evals.gsm8k.gsm8k_eval import evaluate_gsm8k_offline
from tests.models.utils import check_logprobs_close
from tests.utils import multi_gpu_only

from ...utils import (
    _skip_if_insufficient_gpus_for_tp,
    compute_acceptance_len,
    compute_acceptance_rate,
)


@multi_gpu_only(num_gpus=8)
@pytest.mark.slow_test
@pytest.mark.parametrize("eager", [True, False])
def test_dsv41_dfly2_preserves_target_generation(vllm_runner, eager):
    _skip_if_insufficient_gpus_for_tp(8)
    prompts = [
        "The capital of France is",
        "Explain speculative decoding in one sentence.",
        "1 + 1 =",
        "A book has 120 pages. Sam reads 30 pages a day. How many days does he need?",
        "Context token. " * 400 + "\nWrite a Python function that adds two integers.",
    ]
    options = dict(
        tensor_parallel_size=8,
        dtype="bfloat16",
        max_model_len=4096,
        max_num_batched_tokens=1024,
        max_num_seqs=4,
        kv_cache_dtype="fp8_ds_mla",
        block_size=64,
        enable_prefix_caching=True,
        enable_chunked_prefill=True,
        disable_log_stats=False,
        safetensors_load_strategy="prefetch",
        compilation_config={
            "decode_cudagraph_capture_request_sizes": [1, 2, 4],
            "prefill_cudagraph_capture_token_sizes": [128, 512, 1024],
        },
    )
    target = "deepseek-ai/DeepSeek-V4.1-Flash-1017"
    with vllm_runner(target, enforce_eager=eager, **options) as runner:
        baseline = runner.generate_greedy_logprobs(
            prompts, max_tokens=64, num_logprobs=5
        )
        baseline_eval = evaluate_gsm8k_offline(
            runner.llm,
            num_questions=32,
            max_tokens=256,
            temperature=0,
        )
        print(f"target eager={eager}: GSM8K={baseline_eval}")
        print(f"target outputs: {[output[1] for output in baseline]}")

    with vllm_runner(
        target,
        enforce_eager=eager,
        speculative_config={
            "method": "dfly",
            "model": "AngelSlim/DS-V4.1-Flash-DFly2-B6",
            "num_speculative_tokens": 5,
        },
        **options,
    ) as runner:
        output = runner.generate_greedy_logprobs(prompts, max_tokens=64, num_logprobs=5)
        check_logprobs_close(
            outputs_0_lst=baseline,
            outputs_1_lst=output,
            name_0="target",
            name_1=f"dfly2 eager={eager}",
        )
        # Change request order and batch size after populating the prefix cache.
        cached = runner.generate_greedy_logprobs(
            [prompts[4], prompts[0]],
            max_tokens=64,
            num_logprobs=5,
        )
        check_logprobs_close(
            outputs_0_lst=[baseline[4], baseline[0]],
            outputs_1_lst=cached,
            name_0="target",
            name_1="dfly2 cached",
        )
        result = evaluate_gsm8k_offline(
            runner.llm,
            num_questions=32,
            max_tokens=256,
            temperature=0,
        )
        metrics = runner.llm.get_metrics()
        acceptance = compute_acceptance_rate(metrics)
        length = compute_acceptance_len(metrics)
        print(
            f"eager={eager}: GSM8K={result}, acceptance={acceptance}, length={length}"
        )
        assert result["accuracy"] >= baseline_eval["accuracy"]
        assert 0 < acceptance < 1
        assert 1 < length <= 6
