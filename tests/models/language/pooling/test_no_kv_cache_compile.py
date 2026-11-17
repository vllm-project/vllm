# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from vllm.inputs import TokensPrompt

from ...utils import check_embeddings_close

INTERMEDIATE_SIZE = 65536
INT32_LIMIT_TOKENS = 2**31 // (2 * INTERMEDIATE_SIZE)  # 16384
MAX_TOKENS = INT32_LIMIT_TOKENS + 1024


def test_no_kv_cache_long_step_compiled_matches_eager(vllm_runner):
    # No KV cache (bidirectional). Hidden size and eps make the dummy MLP
    # output matter, so wrong values fail the check, not only a crash.
    hf_overrides = {
        "is_causal": False,
        "num_hidden_layers": 1,
        "hidden_size": 4096,
        "rms_norm_eps": 1e-12,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "head_dim": 64,
        "intermediate_size": INTERMEDIATE_SIZE,
    }
    prompt = TokensPrompt(
        prompt_token_ids=[(i * 7919) % 150000 for i in range(MAX_TOKENS - 8)]
    )
    outputs = {}
    for enforce_eager in (True, False):
        with vllm_runner(
            "Qwen/Qwen3-0.6B",
            runner="pooling",
            convert="embed",
            load_format="dummy",
            hf_overrides=hf_overrides,
            max_model_len=MAX_TOKENS,
            max_num_batched_tokens=MAX_TOKENS,
            enforce_eager=enforce_eager,
        ) as model:
            [out] = model.llm.embed([prompt])
            outputs[enforce_eager] = out.outputs.embedding

    check_embeddings_close(
        embeddings_0_lst=[outputs[True]],
        embeddings_1_lst=[outputs[False]],
        name_0="eager",
        name_1="compiled",
    )
