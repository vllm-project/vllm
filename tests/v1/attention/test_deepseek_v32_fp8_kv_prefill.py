# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from tests.utils import fork_new_process_for_each_test
from vllm.platforms import current_platform

INDEX_TOPK = 256


@pytest.mark.skipif(
    not current_platform.is_device_capability_family(100),
    reason="FlashInfer sparse MLA with an fp8 query requires SM 10.x",
)
@fork_new_process_for_each_test
def test_fp8_kv_dense_mha_prefill_under_cudagraphs(
    monkeypatch: pytest.MonkeyPatch, vllm_runner
):
    """With an fp8 KV cache the piecewise graph packs the MQA query as fp8; a
    prefill longer than the decode threshold (128 at 8 heads) but within
    `index_topk` replays it on the dense-MHA route, which must RoPE the bf16
    q_pe itself."""
    monkeypatch.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    monkeypatch.setenv("VLLM_DEEP_GEMM_WARMUP", "skip")
    prompt = [1000 + i for i in range(192)]

    with vllm_runner(
        "deepseek-ai/DeepSeek-V3.2",
        load_format="dummy",
        hf_overrides={
            "num_hidden_layers": 1,
            "hidden_size": 256,
            "intermediate_size": 512,
            "num_attention_heads": 8,
            "num_key_value_heads": 1,
            "n_routed_experts": 8,
            "num_experts_per_tok": 2,
            "index_topk": INDEX_TOPK,
        },
        kv_cache_dtype="fp8",
        attention_backend="FLASHINFER_MLA_SPARSE",
        block_size=64,
        max_model_len=512,
        max_num_seqs=4,
        compilation_config={"cudagraph_capture_sizes": [4, 256]},
    ) as runner:
        [(token_ids, _)] = runner.generate_greedy([prompt], max_tokens=4)

    assert len(token_ids) > len(prompt)
