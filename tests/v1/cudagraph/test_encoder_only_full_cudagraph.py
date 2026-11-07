# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Encoder-only (non-causal) attention under FULL CUDA graphs: a single prompt
replayed through a graph captured on a many-request dummy batch must give the
same result as eager execution."""

import pytest
import torch

from vllm import LLM, SamplingParams
from vllm.platforms import current_platform

LLAMA = "hmellor/tiny-random-LlamaForCausalLM"


def _tiny_config(arch: str):
    """Two-layer random configs for architectures that enable encoder-only
    attention through ``is_causal=False`` (head sizes >= 32 so the default
    FP32 backend is used)."""
    from transformers import DeepseekV2Config, Gemma3TextConfig, Qwen3Config

    common = dict(
        vocab_size=32000,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=4,
        max_position_embeddings=2048,
        tie_word_embeddings=False,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
    )
    if arch == "qwen3":
        return Qwen3Config(num_key_value_heads=2, head_dim=32, **common)
    if arch == "gemma3":
        # One sliding-window layer (window shorter than most prompts) and one
        # full-attention layer.
        return Gemma3TextConfig(
            num_key_value_heads=2,
            head_dim=32,
            sliding_window=64,
            layer_types=["sliding_attention", "full_attention"],
            **common,
        )
    if arch == "deepseek_v2":
        # is_causal=False turns MLA off (ModelConfig.use_mla), so this covers
        # the non-MLA DeepseekV2Attention path.
        return DeepseekV2Config(
            num_key_value_heads=4,
            kv_lora_rank=64,
            q_lora_rank=None,
            qk_nope_head_dim=32,
            qk_rope_head_dim=16,
            v_head_dim=32,
            n_routed_experts=4,
            num_experts_per_tok=2,
            n_shared_experts=1,
            moe_intermediate_size=32,
            first_k_dense_replace=2,
            topk_method="greedy",
            n_group=1,
            topk_group=1,
            **common,
        )
    raise ValueError(arch)


@pytest.fixture(scope="module")
def model_path(request, tmp_path_factory) -> str:
    arch = request.param
    if arch == "llama":
        return LLAMA
    from transformers import AutoModelForCausalLM, AutoTokenizer

    path = tmp_path_factory.mktemp(f"tiny-{arch}")
    torch.manual_seed(0)
    model = AutoModelForCausalLM.from_config(_tiny_config(arch), dtype=torch.float32)
    model.save_pretrained(path)
    AutoTokenizer.from_pretrained(LLAMA).save_pretrained(path)
    return str(path)


def _last_logprobs(
    model: str, cudagraph_mode: str, prompts: list[list[int]]
) -> list[list[float]]:
    llm = LLM(
        model,
        hf_overrides={"is_causal": False},
        max_model_len=2048,
        max_num_batched_tokens=4096,
        enable_prefix_caching=False,
        gpu_memory_utilization=0.3,
        dtype="float32",
        compilation_config={
            "cudagraph_mode": cudagraph_mode,
            "max_cudagraph_capture_size": 1024,
        },
    )
    params = SamplingParams(
        max_tokens=1, temperature=0.0, logprob_token_ids=list(range(100, 132))
    )
    out = []
    for prompt in prompts:  # one request per step: each replays a FULL graph
        (result,) = llm.generate([{"prompt_token_ids": prompt}], params, use_tqdm=False)
        logprobs = result.outputs[0].logprobs
        assert logprobs is not None
        out.append([lp.logprob for lp in logprobs[0].values()])
    del llm
    torch.accelerator.empty_cache()
    return out


@pytest.mark.skipif(not current_platform.is_cuda(), reason="CUDA graphs need CUDA")
@pytest.mark.parametrize(
    "model_path", ["llama", "qwen3", "gemma3", "deepseek_v2"], indirect=True
)
def test_encoder_only_full_cudagraph_matches_eager(model_path: str):
    generator = torch.Generator().manual_seed(0)
    prompts = [
        torch.randint(100, 30000, (n,), generator=generator).tolist()
        for n in (23, 150, 600, 1000)
    ]
    expected = _last_logprobs(model_path, "NONE", prompts)
    actual = _last_logprobs(model_path, "FULL", prompts)
    for exp, act in zip(expected, actual):
        torch.testing.assert_close(
            torch.tensor(act), torch.tensor(exp), atol=1e-4, rtol=1e-4
        )
