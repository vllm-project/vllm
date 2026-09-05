# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""End-to-end ROCm test: unified attention works with KV connectors."""

import pytest

from vllm import LLM, SamplingParams
from vllm._aiter_ops import is_aiter_found_and_supported
from vllm.config import AttentionConfig, KVTransferConfig

pytestmark = pytest.mark.skipif(
    not is_aiter_found_and_supported(),
    reason="ROCM_AITER_UNIFIED_ATTN requires aiter on a supported ROCm device",
)

MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"
PROMPTS = [
    "The capital of France is the city of Paris, which is famous for",
    "In a distant galaxy, a lone explorer discovered an ancient signal that",
]
MAX_TOKENS = 16


def test_unified_attn_with_decode_bench_connector(monkeypatch):
    """Ensure generate() runs with unified attention + DecodeBenchConnector."""
    monkeypatch.setenv("VLLM_ROCM_USE_AITER", "1")
    llm = LLM(
        model=MODEL_NAME,
        max_model_len=2048,
        max_num_seqs=8,
        gpu_memory_utilization=0.6,
        enforce_eager=True,
        attention_config=AttentionConfig(backend="ROCM_AITER_UNIFIED_ATTN"),
        kv_transfer_config=KVTransferConfig(
            kv_connector="DecodeBenchConnector",
            kv_role="kv_both",
        ),
    )

    outputs = llm.generate(
        PROMPTS,
        SamplingParams(temperature=0.0, max_tokens=MAX_TOKENS),
    )

    for output in outputs:
        prompt_len = len(output.prompt_token_ids)
        completion = output.outputs[0]
        print(
            f"[decode_bench] prompt_len={prompt_len} "
            f"num_cached_tokens={output.num_cached_tokens} "
            f"num_generated={len(completion.token_ids)}\n"
            f"  prompt: {output.prompt!r}\n"
            f"  output: {completion.text!r}"
        )

    assert len(outputs) == len(PROMPTS)
    for output in outputs:
        completion = output.outputs[0]
        assert len(completion.token_ids) == MAX_TOKENS, (
            f"expected {MAX_TOKENS} generated tokens, got "
            f"{len(completion.token_ids)}: {completion.text!r}"
        )

        prompt_len = len(output.prompt_token_ids)
        # DecodeBenchConnector fills every prompt token except the last
        assert output.num_cached_tokens == prompt_len - 1, (
            f"connector should serve prompt_len-1={prompt_len - 1} tokens, "
            f"got num_cached_tokens={output.num_cached_tokens}"
        )
