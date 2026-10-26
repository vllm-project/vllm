# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""ROCm equivalents of the quantized-MoE model startup/generation contract.

Like test_blackwell_moe.py, these tests use reduced models and dummy weights.
They validate model/backend integration, not trained-model accuracy.
"""

import math

import pytest

from tests.utils import RemoteOpenAIServer
from vllm.platforms import current_platform
from vllm.platforms.rocm import on_gfx950

pytestmark = pytest.mark.skipif(
    not current_platform.is_rocm() or not on_gfx950(),
    reason="ROCm FP8/MXFP4 model integration requires gfx950",
)

COMMON_OVERRIDES = {
    "num_hidden_layers": 4,
    "hidden_size": 512,
    "intermediate_size": 1024,
    "num_attention_heads": 8,
    "num_key_value_heads": 2,
}

CASES = [
    pytest.param(
        "RedHatAI/DeepSeek-Coder-V2-Lite-Instruct-FP8",
        "efe1ced428db63e7ccbcf367334596f77e9af140",
        "triton",
        "TRITON_MLA",
        {
            "n_routed_experts": 8,
            "n_shared_experts": 1,
            "num_experts_per_tok": 2,
            "moe_intermediate_size": 512,
        },
        ["--code-revision", "e434a23f91ba5b4923cf6c9d9a238eb4a08e3a11"],
        id="deepseek-fp8-per-tensor-triton",
    ),
    pytest.param(
        "Qwen/Qwen3-30B-A3B-FP8",
        "d206ba732169f29bb77fbf80fc2c4b81d4d30782",
        "triton",
        "TRITON_ATTN",
        {
            "num_experts": 8,
            "num_experts_per_tok": 2,
            "moe_intermediate_size": 512,
            "head_dim": 64,
        },
        [],
        id="qwen3-fp8-block-triton",
    ),
    pytest.param(
        "openai/gpt-oss-20b",
        "6cee5e81ee83917806bbde320786a8fb61efebee",
        "aiter",
        "TRITON_ATTN",
        {
            "num_local_experts": 8,
            "num_experts_per_tok": 2,
            "experts_per_token": 2,
            "layer_types": ["sliding_attention", "full_attention"] * 2,
        },
        [],
        id="gptoss-mxfp4-aiter",
    ),
]


@pytest.mark.parametrize("enforce_eager", [True, False], ids=["eager", "compiled"])
@pytest.mark.parametrize(
    "model,revision,moe_backend,attention_backend,overrides,extra_args", CASES
)
def test_rocm_quantized_moe_generation(
    model,
    revision,
    moe_backend,
    attention_backend,
    overrides,
    extra_args,
    enforce_eager,
):
    """Quantized MoE models warm up and return finite logits through serving."""
    args = [
        "--revision",
        revision,
        "--load-format",
        "dummy",
        "--trust-remote-code",
        "--dtype",
        "bfloat16",
        "--max-model-len",
        "2048",
        "--max-num-batched-tokens",
        "256",
        "--max-num-seqs",
        "8",
        "--gpu-memory-utilization",
        "0.02",
        "--kv-cache-memory-bytes",
        str(256 * 1024 * 1024),
        "--moe-backend",
        moe_backend,
        "--attention-backend",
        attention_backend,
        *extra_args,
    ]
    if enforce_eager:
        args.append("--enforce-eager")

    with RemoteOpenAIServer(
        model,
        args,
        env_dict={"VLLM_ROCM_USE_AITER": "1", "VLLM_USE_V2_MODEL_RUNNER": "1"},
        override_hf_configs={**COMMON_OVERRIDES, **overrides},
        max_wait_seconds=1200,
    ) as server:
        prompts = [[1, 2, 3], [1, 2, 3, 4] * 4 + [5]]
        completion = server.get_client().completions.create(
            model=model,
            prompt=prompts,
            temperature=0,
            max_tokens=2,
            logprobs=1,
            extra_body={"ignore_eos": True, "skip_special_tokens": False},
        )
        assert len(completion.choices) == len(prompts)
        assert completion.usage is not None
        assert completion.usage.prompt_tokens == sum(map(len, prompts))
        assert completion.usage.completion_tokens == 2 * len(prompts)
        for choice in completion.choices:
            assert choice.finish_reason == "length"
            assert choice.text
            assert choice.logprobs is not None
            assert choice.logprobs.token_logprobs is not None
            assert len(choice.logprobs.token_logprobs) == 2
            assert all(
                p is not None and math.isfinite(p)
                for p in choice.logprobs.token_logprobs
            )
