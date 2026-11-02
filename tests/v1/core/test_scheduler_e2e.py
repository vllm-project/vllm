# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm import LLM, SamplingParams

MODEL = "hmellor/tiny-random-LlamaForCausalLM"
PROMPT = "Hello my name is Robert and I"


@pytest.fixture(scope="module")
def llm() -> LLM:
    return LLM(
        MODEL,
        enforce_eager=True,
        enable_prefix_caching=True,
        long_prefill_token_threshold=2,
        max_num_batched_tokens=6,
        max_num_seqs=3,
        block_size=16,
    )


def test_concurrent_partial_prefill(llm):
    outputs = llm.generate([PROMPT] * 3)
    assert len(outputs) == 3
    for output in outputs:
        assert len(output.outputs) == 1


def test_prefix_cache_stats_is_recorded(llm):
    # 17 tokens will make sure first 16 tokens are cached in a block
    input_tokens = {"prompt_token_ids": [101] * 17}
    _ = llm.generate([input_tokens])
    outputs = llm.generate([input_tokens])
    assert outputs[0].num_cached_tokens == 16


def test_residual_sjf_orders_cached_work_without_changing_output(
    vllm_runner, monkeypatch
):
    """Cached remaining work changes admission order, preserving model outputs."""
    monkeypatch.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    monkeypatch.setenv("VLLM_BATCH_INVARIANT", "1")
    results = []
    for policy in ("fcfs", "residual_sjf"):
        with vllm_runner(
            "facebook/opt-125m",
            enforce_eager=True,
            async_scheduling=False,
            enable_prefix_caching=True,
            enable_chunked_prefill=True,
            scheduling_policy=policy,
            residual_sjf_max_wait_ms=60_000,
            max_model_len=1024,
            max_num_batched_tokens=32,
            max_num_seqs=1,
            gpu_memory_utilization=0.1,
            kv_cache_memory_bytes=64 * 1024 * 1024,
        ) as model:
            engine = model.llm.llm_engine
            params = SamplingParams(temperature=0, max_tokens=8, ignore_eos=True)
            model.llm.generate([{"prompt_token_ids": [42] * 513}], params)
            engine.add_request("cold", {"prompt_token_ids": [43] * 64}, params)
            engine.add_request(
                "cached", {"prompt_token_ids": [42] * 513 + [44] * 16}, params
            )
            outputs = {}
            for _ in range(64):
                if not engine.has_unfinished_requests():
                    break
                for output in engine.step():
                    if output.finished:
                        outputs[output.request_id] = output.outputs[0].token_ids
            assert not engine.has_unfinished_requests()
            assert list(outputs) == (
                ["cold", "cached"] if policy == "fcfs" else ["cached", "cold"]
            )
            results.append(outputs)
    assert results[0] == results[1]
