# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""End-to-end checks of `--kv-cache-dtype kitty` through `LLM`.

Kitty outputs depend on the schedule: a chunk or a recomputed request reads
pages sealed in earlier steps dequantized (see `kitty_attn.py`). These tests
therefore compare against the unquantized cache with `check_logprobs_close`;
the exact behavior of each path is covered by
`tests/kernels/attention/test_kitty_attn.py`.
"""

import os

import pytest

from tests.models.utils import check_logprobs_close
from vllm.platforms import current_platform
from vllm.utils.math_utils import cdiv
from vllm.v1.core.kv_cache_utils import get_max_concurrency_for_kv_cache_config
from vllm.v1.metrics.reader import Counter

MODEL = "Qwen/Qwen3-0.6B"
NUM_LOGPROBS = 5
# `vllm_runner` passes block_size=16 by default; None lets Kitty pick 128, as
# with `vllm serve --kv-cache-dtype kitty`.
CACHE_ARGS = {
    "auto": {"kv_cache_dtype": "auto"},
    "kitty": {"kv_cache_dtype": "kitty", "block_size": None},
}

pytestmark = pytest.mark.skipif(
    not current_platform.is_cuda() or not current_platform.has_device_capability(80),
    reason="Kitty needs CUDA with compute capability >= 8.0",
)


def _long_prompt(num_words: int) -> str:
    path = os.path.join(os.path.dirname(__file__), "../../../prompts/summary.txt")
    with open(path) as f:
        return " ".join(f.read().split()[:num_words])


def _num_preemptions(llm) -> int:
    return sum(
        metric.value
        for metric in llm.get_metrics()
        if metric.name == "vllm:num_preemptions" and isinstance(metric, Counter)
    )


def test_kitty_close_to_bf16(vllm_runner, example_prompts):
    """One batch takes every Kitty path: a short prompt, and a ~1000-token prompt
    prefilled in chunks of at most 256 tokens, so later chunks read sealed K and
    V pages; then 300 decode steps cross page boundaries and seal pages."""
    prompts = [example_prompts[0], _long_prompt(750)]
    outputs = {}
    for kv_cache_dtype in ("auto", "kitty"):
        with vllm_runner(
            MODEL,
            **CACHE_ARGS[kv_cache_dtype],
            max_model_len=2048,
            max_num_batched_tokens=256,
            enable_chunked_prefill=True,
        ) as llm:
            outputs[kv_cache_dtype] = llm.generate_greedy_logprobs(
                prompts, 300, NUM_LOGPROBS
            )
    check_logprobs_close(
        outputs_0_lst=outputs["auto"],
        outputs_1_lst=outputs["kitty"],
        name_0="bf16",
        name_1="kitty",
    )


def test_kitty_preemption(vllm_runner, example_prompts):
    """Requests preempted by recompute get a fresh residual block and re-seal
    their pages from the recomputed prompt; outputs stay close to bf16."""
    max_tokens = 300
    with vllm_runner(MODEL, max_model_len=1024) as llm:
        reference = llm.generate_greedy_logprobs(
            example_prompts, max_tokens, NUM_LOGPROBS
        )
    # A request holds one block per residual group (28 on this model) plus one
    # per 128 tokens: 3 prompts fit with one block spare, so the second
    # running request to cross a page boundary preempts another.
    with vllm_runner(
        MODEL,
        **CACHE_ARGS["kitty"],
        max_model_len=1024,
        num_gpu_blocks_override=3 * 29 + 1,
        disable_log_stats=False,
    ) as llm:
        kitty = llm.generate_greedy_logprobs(example_prompts, max_tokens, NUM_LOGPROBS)
        assert _num_preemptions(llm.llm) > 0
    check_logprobs_close(
        outputs_0_lst=reference,
        outputs_1_lst=kitty,
        name_0="bf16",
        name_1="kitty",
    )


def test_kitty_kv_capacity(vllm_runner, monkeypatch):
    """A request at `max_model_len` costs the KV memory the Kitty layout
    predicts: one packed block of every layer's pages per 128 tokens, plus one
    block per residual group, each holding as many residual pages as fit. On
    this model that is 4.4x less than bf16."""
    monkeypatch.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    max_model_len = 8192
    bytes_per_request = {}
    for kv_cache_dtype in ("auto", "kitty"):
        with vllm_runner(
            MODEL,
            **CACHE_ARGS[kv_cache_dtype],
            max_model_len=max_model_len,
            gpu_memory_utilization=0.5,
        ) as llm:
            engine_core = llm.llm.llm_engine.engine_core.engine_core
            config = engine_core.scheduler.kv_cache_config
            # All tensors alias one backing allocation.
            (pool_bytes,) = {t.size for t in config.kv_cache_tensors}
            concurrency = get_max_concurrency_for_kv_cache_config(
                engine_core.vllm_config, config
            )
            bytes_per_request[kv_cache_dtype] = pool_bytes / concurrency
            hf_config = engine_core.vllm_config.model_config.hf_text_config

    layers = hf_config.num_hidden_layers
    kv_heads = hf_config.num_key_value_heads
    head_dim = hf_config.head_dim
    packed_block = layers * kv_heads * 128 * 81
    residual_page = kv_heads * 640 * head_dim * 2
    residual_groups = cdiv(layers, packed_block // residual_page)
    expected = (cdiv(max_model_len, 128) + residual_groups) * packed_block
    bf16 = max_model_len * layers * kv_heads * head_dim * 2 * 2
    assert bytes_per_request["kitty"] == pytest.approx(expected, rel=1e-9)
    assert bytes_per_request["auto"] == pytest.approx(bf16, rel=1e-9)
