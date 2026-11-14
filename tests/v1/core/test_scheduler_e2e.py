# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import time
from types import SimpleNamespace

import pytest

from vllm import LLM, SamplingParams, TokensPrompt
from vllm.transformers_utils.utils import maybe_model_redirect

MODEL = "vllm-project/tiny-random-LlamaForCausalLM"
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


@pytest.mark.cpu_test
def test_cache_aware_admission_ages_cold_request(vllm_runner, monkeypatch):
    """Warm requests reuse their prefix and pass a cold one until it has waited
    cache_aware_admission_max_wait_ms, then the cold one is served.
    """
    monkeypatch.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    with vllm_runner(
        maybe_model_redirect("Qwen/Qwen3-0.6B"),
        load_format="dummy",
        skip_tokenizer_init=True,
        enforce_eager=True,
        enable_prefix_caching=True,
        enable_chunked_prefill=True,
        async_scheduling=False,
        max_model_len=128,
        max_num_batched_tokens=128,
        max_num_seqs=1,
        num_gpu_blocks_override=64,
        block_size=32,
        cache_aware_admission_window=4,
        cache_aware_admission_threshold=0,
        cache_aware_admission_max_wait_ms=1000,
    ) as runner:
        params = SamplingParams(
            temperature=0, max_tokens=1, ignore_eos=True, detokenize=False
        )
        warm = TokensPrompt(prompt_token_ids=[0] * 65)
        cold = TokensPrompt(prompt_token_ids=[1] * 65)
        runner.llm.generate([warm], params)
        engine = runner.llm.llm_engine
        clock = SimpleNamespace(now=100.0)
        fake_time = SimpleNamespace(**vars(time))
        fake_time.monotonic = lambda: clock.now
        monkeypatch.setattr("vllm.v1.core.sched.scheduler.time", fake_time)
        engine.add_request("cold", cold, params)
        served = []
        for now in (100.0, 100.5, 100.999, 101.0):
            clock.now = now
            engine.add_request(f"warm@{now}", warm, params)
            (output,) = engine.step()
            assert output.finished and len(output.outputs[0].token_ids) == 1
            served.append((output.request_id, output.num_cached_tokens))
        assert served == [
            ("warm@100.0", 64),
            ("warm@100.5", 64),
            ("warm@100.999", 64),
            ("cold", 0),
        ]
        (output,) = engine.step()
        assert (output.request_id, output.num_cached_tokens) == ("warm@101.0", 64)
        assert not engine.has_unfinished_requests()
