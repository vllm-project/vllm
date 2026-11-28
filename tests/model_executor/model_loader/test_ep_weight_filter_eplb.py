# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Weight loading with EPLB, end to end on real checkpoints."""

import logging

import pytest
import torch
from safetensors.torch import save_file
from transformers import Qwen3MoeConfig, Qwen3MoeForCausalLM

from tests.utils import multi_gpu_marks
from vllm.config import CompilationConfig
from vllm.platforms import current_platform

# Already used by other CI tests. Its 72 routed experts are stored per expert;
# with EP2 and 2 redundant experts each rank holds 37 slots, and rank 1's
# redundant slots replicate experts 0 and 1 from rank 0's plain EP shard.
MODEL = "luccafong/deepseek_mtp_main_random"
PROMPTS = [
    "The capital of France is",
    "The future of AI is",
    "1 + 1 =",
    "def fibonacci(n):",
]
MAX_TOKENS = 16
FILTER_LOG = "EP weight filter: ep_size=2, ep_rank=0, loading 37/72 experts"

pytestmark = [
    *multi_gpu_marks(num_gpus=2),
    pytest.mark.skipif(
        not current_platform.is_cuda_alike(),
        reason="Needs a CUDA or ROCm GPU.",
    ),
]


def test_ep_weight_filter_matches_unfiltered_with_eplb(
    vllm_runner, caplog_mp_spawn, monkeypatch: pytest.MonkeyPatch
):
    # Batch invariance makes the two runs comparable token for token, and
    # spawned workers let caplog_mp_spawn read their logs.
    monkeypatch.setenv("VLLM_BATCH_INVARIANT", "1")
    monkeypatch.setenv("VLLM_ROCM_USE_AITER", "0")
    monkeypatch.setenv("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

    outputs = {}
    for enable_ep_weight_filter in (False, True):
        with (
            caplog_mp_spawn(logging.INFO) as log_holder,
            vllm_runner(
                MODEL,
                tensor_parallel_size=2,
                enable_expert_parallel=True,
                enable_eplb=True,
                # A small window and interval make EPLB rebalance mid-generation;
                # synchronous torch_nccl keeps the test independent of NIXL.
                eplb_config={
                    "num_redundant_experts": 2,
                    "window_size": 2,
                    "step_interval": 4,
                    "use_async": False,
                    "communicator": "torch_nccl",
                },
                enable_ep_weight_filter=enable_ep_weight_filter,
                max_model_len=2048,
                block_size=None,
                enable_chunked_prefill=None,
                enable_prefix_caching=False,
                enforce_eager=True,
                compilation_config=CompilationConfig(),
            ) as runner,
        ):
            outputs[enable_ep_weight_filter] = runner.generate_greedy(
                PROMPTS, MAX_TOKENS
            )
        assert (FILTER_LOG in log_holder.text) == enable_ep_weight_filter, (
            f"expected the filter {'on' if enable_ep_weight_filter else 'off'}"
        )

    assert outputs[True] == outputs[False]


def test_stacked_expert_checkpoint_loads_redundant_experts(
    vllm_runner, tmp_path, monkeypatch: pytest.MonkeyPatch
):
    """Stacked `[num_experts, ...]` expert tensors (the transformers v5 layout of
    e.g. Qwen3.5-MoE checkpoints) must also load the redundant EPLB experts."""
    # EPLB routes some tokens to the redundant experts from the first step, and
    # batch invariance makes their outputs match the originals' token for token,
    # so a redundant expert that was not loaded changes the generated tokens.
    monkeypatch.setenv("VLLM_BATCH_INVARIANT", "1")
    monkeypatch.setenv("VLLM_ROCM_USE_AITER", "0")
    config = Qwen3MoeConfig(
        vocab_size=1024,
        hidden_size=128,
        intermediate_size=256,
        moe_intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=32,
        num_experts=8,
        num_experts_per_tok=2,
        initializer_range=0.1,
        architectures=["Qwen3MoeForCausalLM"],
    )
    torch.manual_seed(0)
    state_dict = Qwen3MoeForCausalLM(config).to(torch.bfloat16).state_dict()
    assert state_dict["model.layers.0.mlp.experts.gate_up_proj"].dim() == 3
    # save_pretrained would convert the experts back to the per-expert layout.
    save_file(
        {name: tensor.contiguous() for name, tensor in state_dict.items()},
        tmp_path / "model.safetensors",
    )
    config.save_pretrained(tmp_path)

    prompts = [list(range(start, start + 32)) for start in range(0, 256, 32)]
    outputs = {}
    for num_redundant_experts in (0, 4):
        with vllm_runner(
            str(tmp_path),
            skip_tokenizer_init=True,
            tensor_parallel_size=2,
            enable_expert_parallel=True,
            enable_eplb=True,
            eplb_config={
                "num_redundant_experts": num_redundant_experts,
                "use_async": False,
                "communicator": "torch_nccl",
            },
            max_model_len=256,
            enforce_eager=True,
            compilation_config=CompilationConfig(),
        ) as runner:
            outputs[num_redundant_experts] = runner.generate_greedy(prompts, MAX_TOKENS)

    assert outputs[4] == outputs[0]
