# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import time

import pytest

from tests.models.utils import check_logprobs_close
from vllm import LLM, SamplingParams, TokensPrompt
from vllm.config import KVTransferConfig
from vllm.v1.metrics.reader import Counter

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


@pytest.mark.timeout(180)
def test_partial_offload_loads_finish_under_kv_pressure(vllm_runner):
    """Concurrent CPU prefix loads must finish their uncached tails."""
    with vllm_runner(
        MODEL,
        skip_tokenizer_init=True,
        enforce_eager=True,
        gpu_memory_utilization=0.3,
        max_model_len=512,
        max_num_batched_tokens=64,
        max_num_seqs=8,
        enable_chunked_prefill=True,
        block_size=16,
        num_gpu_blocks_override=50,
        enable_prefix_caching=True,
        disable_log_stats=False,
        kv_transfer_config=KVTransferConfig(
            kv_connector="OffloadingConnector",
            kv_role="kv_both",
            kv_connector_extra_config={"cpu_bytes_to_use": 64 << 20},
        ),
    ) as runner:
        engine = runner.get_llm()
        prompts = [
            TokensPrompt(prompt_token_ids=[i + 3] * 128 + [i + 20] * 128)
            for i in range(8)
        ]
        params = SamplingParams(
            temperature=0,
            max_tokens=16,
            ignore_eos=True,
            logprobs=5,
            detokenize=False,
        )
        cold_params = params.clone()
        cold_params.extra_args = {
            "kv_transfer_params": {"max_load_tokens": 0, "max_offload_tokens": 0}
        }
        cold = engine.generate(prompts, cold_params, use_tqdm=False)
        for prompt in prompts:
            engine.generate(
                TokensPrompt(prompt_token_ids=prompt["prompt_token_ids"][:128]),
                params,
                use_tqdm=False,
            )

        deadline = time.monotonic() + 30
        while not engine.reset_prefix_cache():
            assert time.monotonic() < deadline, "CPU offloads did not finish"
            engine.llm_engine.step()

        def external_tokens():
            return sum(
                metric.value
                for metric in engine.get_metrics()
                if isinstance(metric, Counter)
                and metric.name == "vllm:prompt_tokens_by_source"
                and metric.labels.get("source") == "external_kv_transfer"
            )

        before = external_tokens()
        warm = engine.generate(prompts, params, use_tqdm=False)
        assert len(warm) == len(prompts)
        assert all(
            output.finished and len(output.outputs[0].token_ids) == 16
            for output in warm
        )
        assert external_tokens() > before
        check_logprobs_close(
            outputs_0_lst=[
                (list(output.outputs[0].token_ids), "", output.outputs[0].logprobs)
                for output in cold
            ],
            outputs_1_lst=[
                (list(output.outputs[0].token_ids), "", output.outputs[0].logprobs)
                for output in warm
            ],
            name_0="cold",
            name_1="CPU prefix reload",
            always_check_logprobs=True,
        )
