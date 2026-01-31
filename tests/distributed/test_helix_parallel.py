# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Integration tests for Helix parallelism.

WARNING: These tests require multi-GPU (4+ GPUs) with compute capability 9.0+
(Hopper/Blackwell). Tests will be skipped if requirements are not met.

Helix parallelism decouples attention (KVP) and FFN (TP) parallelism strategies.
For MLA models (effective K=1), TPA=1 and KVP=N where N is the number of GPUs.
"""

import os
from typing import NamedTuple

import pytest
import torch

from tests.evals.gsm8k.gsm8k_eval import evaluate_gsm8k
from tests.utils import RemoteOpenAIServer, create_new_process_for_each_test
from vllm.logger import init_logger

from ..models.registry import HF_EXAMPLE_MODELS

logger = init_logger("test_helix_parallel")

VLLM_MULTI_NODE = os.getenv("VLLM_MULTI_NODE", "0") == "1"

# Helix test models
# - MLA models: TPA=1 is optimal (effective K=1)
# - GQA models: TPA can be > 1 (multiple KV heads)
HELIX_TEST_MODELS = [
    "deepseek-ai/DeepSeek-V2-Lite-Chat",  # MLA
    "Qwen/Qwen2.5-1.5B-Instruct",          # GQA
]

# GSM8K eval configuration
NUM_QUESTIONS = 256  # Fast eval for CI
NUM_SHOTS = 5  # Few-shot examples
# Accuracy threshold (should match DCP baseline with 2% buffer)
MIN_ACCURACY = {
    "deepseek-ai/DeepSeek-V2-Lite-Chat": 0.64,
    "Qwen/Qwen2.5-1.5B-Instruct": 0.52,
}


class HelixParallelSetup(NamedTuple):
    tp_size: int
    dcp_size: int  # Also helix_kvp_size
    cp_kv_cache_interleave_size: int
    chunked_prefill: bool


class HelixTestOptions(NamedTuple):
    multi_node_only: bool
    attn_backend: str | None = None


# Helix test configurations
#
# TPA (Tensor Parallel for Attention) shards attention heads across GPUs.
# Constraint: TPA ≤ K (number of KV heads)
#
# For MLA models (DeepSeek-V2):
#   - Effective K=1 (KV compressed to single latent)
#   - TPA must be 1 (can't shard 1 head)
#   - Benefit comes from KVP (sequence sharding)
#
# For GQA models (Qwen, Llama):
#   - K = num_kv_heads (e.g., 2, 8)
#   - TPA can be up to K
#   - Both TPA and KVP provide benefits
#
HELIX_TEST_CONFIGS = {
    # DeepSeek-V2-Lite-Chat: MLA attention
    # - num_attention_heads: 16, num_key_value_heads: 1 (MLA latent)
    # - TPA must be 1 (effective K=1)
    "deepseek-ai/DeepSeek-V2-Lite-Chat": [
        # 4 GPU: TP=4, DCP=4 -> TPA=1, KVP=4
        (
            HelixParallelSetup(
                tp_size=4,
                dcp_size=4,
                cp_kv_cache_interleave_size=64,
                chunked_prefill=True,
            ),
            HelixTestOptions(multi_node_only=False, attn_backend="FLASHMLA"),
        ),
    ],
    # Qwen2.5-1.5B-Instruct: GQA attention
    # - num_attention_heads: 12, num_key_value_heads: 2
    # - TPA can be up to 2
    "Qwen/Qwen2.5-1.5B-Instruct": [
        # 4 GPU: TP=4, DCP=2 -> TPA=2, KVP=2
        (
            HelixParallelSetup(
                tp_size=4,
                dcp_size=2,
                cp_kv_cache_interleave_size=16,
                chunked_prefill=True,
            ),
            HelixTestOptions(multi_node_only=False, attn_backend="FLASH_ATTN"),
        ),
        # Same config with FlashInfer backend
        (
            HelixParallelSetup(
                tp_size=4,
                dcp_size=2,
                cp_kv_cache_interleave_size=16,
                chunked_prefill=True,
            ),
            HelixTestOptions(multi_node_only=False, attn_backend="FLASHINFER"),
        ),
    ],
}


def _test_helix_gsm8k(
    model_id: str,
    parallel_setup: HelixParallelSetup,
    test_options: HelixTestOptions,
    num_gpus_available: int,
):
    """Run GSM8K evaluation with Helix parallelism enabled."""
    tp_size, dcp_size, interleave_size, chunked_prefill = parallel_setup
    multi_node_only, attn_backend = test_options

    model_info = HF_EXAMPLE_MODELS.find_hf_info(model_id)
    model_info.check_transformers_version(on_fail="skip")
    model_info.check_available_online(on_fail="skip")

    trust_remote_code = model_info.trust_remote_code
    tokenizer_mode = model_info.tokenizer_mode

    if num_gpus_available < tp_size:
        pytest.skip(f"Need at least {tp_size} GPUs, have {num_gpus_available}")

    if multi_node_only and not VLLM_MULTI_NODE:
        pytest.skip("Not in multi-node setting")

    # Helix with DCP requires compute capability 9.0+ (Hopper/Blackwell)
    # Both MLA and GQA models with context parallelism need this
    if torch.cuda.get_device_capability() < (9, 0):
        pytest.skip("Helix with DCP requires compute capability 9.0+")

    server_args = [
        "--dtype", "bfloat16",
        "--max-model-len", "4096",
        "--max-num-seqs", "64",
        # Helix-specific args
        "--helix-mode",
        "--tensor-parallel-size", str(tp_size),
        "--decode-context-parallel-size", str(dcp_size),
        "--cp-kv-cache-interleave-size", str(interleave_size),
    ]

    if chunked_prefill:
        server_args.append("--enable-chunked-prefill")

    if trust_remote_code:
        server_args.append("--trust-remote-code")

    if tokenizer_mode:
        server_args.extend(["--tokenizer-mode", tokenizer_mode])

    if attn_backend:
        server_args.extend(["--attention-backend", attn_backend])

    with RemoteOpenAIServer(
        model_id,
        server_args,
        num_gpus=tp_size,
    ) as server:
        client = server.get_client()

        accuracy = evaluate_gsm8k(
            client,
            model_id,
            num_questions=NUM_QUESTIONS,
            num_shots=NUM_SHOTS,
        )

        logger.info(
            f"Helix GSM8K accuracy for {model_id} "
            f"(TP={tp_size}, KVP={dcp_size}): {accuracy:.2%}"
        )

        min_acc = MIN_ACCURACY.get(model_id, 0.5)
        assert accuracy >= min_acc, (
            f"Helix accuracy {accuracy:.2%} below threshold {min_acc:.2%} "
            f"for {model_id}"
        )


def _generate_test_params():
    """Generate test parameters for parametrize decorator."""
    params = []
    for model_id in HELIX_TEST_MODELS:
        if model_id in HELIX_TEST_CONFIGS:
            for setup, options in HELIX_TEST_CONFIGS[model_id]:
                params.append((model_id, setup, options))
    return params


@pytest.mark.parametrize(
    ("model_id", "parallel_setup", "test_options"),
    _generate_test_params(),
)
@create_new_process_for_each_test()
def test_helix_generation(
    model_id: str,
    parallel_setup: HelixParallelSetup,
    test_options: HelixTestOptions,
    num_gpus_available,
):
    """Test Helix parallelism with GSM8K evaluation."""
    _test_helix_gsm8k(
        model_id,
        parallel_setup,
        test_options,
        num_gpus_available,
    )


# Additional test: Compare Helix vs non-Helix output consistency
@pytest.mark.parametrize("model_id", HELIX_TEST_MODELS)
@create_new_process_for_each_test()
def test_helix_output_consistency(model_id: str, num_gpus_available):
    """
    Verify Helix produces consistent outputs with non-Helix DCP.
    
    This test runs the same prompts with and without helix_mode
    and verifies outputs are identical (both use exact attention).
    """
    if num_gpus_available < 4:
        pytest.skip("Need at least 4 GPUs")

    if torch.cuda.get_device_capability() < (9, 0):
        pytest.skip("Helix with MLA requires compute capability 9.0+")

    model_info = HF_EXAMPLE_MODELS.find_hf_info(model_id)
    model_info.check_transformers_version(on_fail="skip")
    model_info.check_available_online(on_fail="skip")

    # Test prompts
    prompts = [
        "What is 2 + 2?",
        "Explain machine learning in one sentence.",
    ]

    base_args = [
        "--dtype", "bfloat16",
        "--max-model-len", "512",
        "--tensor-parallel-size", "4",
        "--decode-context-parallel-size", "4",
        "--attention-backend", "FLASHMLA",
    ]

    if model_info.trust_remote_code:
        base_args.append("--trust-remote-code")

    # Run with Helix
    helix_args = base_args + ["--helix-mode"]
    with RemoteOpenAIServer(model_id, helix_args, num_gpus=4) as server:
        client = server.get_client()
        helix_outputs = []
        for prompt in prompts:
            response = client.completions.create(
                model=model_id,
                prompt=prompt,
                max_tokens=50,
                temperature=0,  # Deterministic
            )
            helix_outputs.append(response.choices[0].text)

    # Run without Helix (standard DCP)
    with RemoteOpenAIServer(model_id, base_args, num_gpus=4) as server:
        client = server.get_client()
        dcp_outputs = []
        for prompt in prompts:
            response = client.completions.create(
                model=model_id,
                prompt=prompt,
                max_tokens=50,
                temperature=0,
            )
            dcp_outputs.append(response.choices[0].text)

    # Compare outputs
    for i, (helix_out, dcp_out) in enumerate(zip(helix_outputs, dcp_outputs)):
        assert helix_out == dcp_out, (
            f"Output mismatch for prompt {i}:\n"
            f"Helix: {helix_out}\n"
            f"DCP: {dcp_out}"
        )

    logger.info(f"Helix output consistency verified for {model_id}")
