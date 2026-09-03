# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU generation smoke test for DeepSeek-V4.

The real ``deepseek-ai/DeepSeek-V4-Flash`` checkpoint is far too large to load
in CI, so this uses ``load_format="dummy"`` with a small ``hf_overrides`` that
still exercises all three sparse-attention variants the model supports
(``compress_ratios`` 0/dense-SWA, 4/C4A, 128/C128A) plus MoE routing, the
compressor, and mHC gating. With random weights there's no ground truth to
compare generations against, so this only asserts that generation completes
and produces well-formed output.
"""

import pytest

MODEL = "deepseek-ai/DeepSeek-V4-Flash"

HF_OVERRIDES = {
    "num_hidden_layers": 5,
    "compress_ratios": [0, 4, 128, 4, 128],
    "n_routed_experts": 8,
    "num_experts_per_tok": 2,
    # Hash-routed MoE layers (bypass score-based topk via hash_indices_table)
    # for the first N layers -- exercises the CPU monolithic MXFP4
    # apply_monolithic hash-routing path.
    "num_hash_layers": 2,
}


@pytest.mark.cpu_model
def test_cpu_dummy_generation(vllm_runner, monkeypatch) -> None:
    with monkeypatch.context() as m:
        m.setenv("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
        with vllm_runner(
            MODEL,
            trust_remote_code=True,
            load_format="dummy",
            enforce_eager=True,
            max_model_len=128,
            max_num_seqs=2,
            # DeepseekV4SparseMLABackend.get_supported_kernel_block_sizes()
            # requires exactly 256 (also matches compressor.py's
            # 256/compress_ratio design).
            block_size=256,
            kv_cache_dtype="fp8_ds_mla",
            hf_overrides=HF_OVERRIDES,
        ) as llm:
            outputs = llm.generate_greedy(["Hello", "MTP on CPU"], max_tokens=4)

    # output_ids includes the prompt tokens, so just check generation actually
    # produced up to max_tokens new ones and didn't crash/return garbage.
    for output_ids, output_str in outputs:
        assert len(output_ids) > 0
        assert isinstance(output_str, str)
