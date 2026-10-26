# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Multi-GPU equivalence of the Muse Spark token-sharded EP paths."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.multiprocessing as mp
from torch import nn

from vllm.platforms import current_platform

if not current_platform.is_rocm():
    pytest.skip("The token-sharded path is ROCm-only", allow_module_level=True)

from tests.utils import multi_gpu_test
from vllm.config import ParallelConfig, VllmConfig, set_current_vllm_config
from vllm.distributed import (
    cleanup_dist_env_and_memory,
    init_distributed_environment,
    initialize_model_parallel,
    tensor_model_parallel_all_gather,
)
from vllm.forward_context import set_forward_context
from vllm.model_executor.models.utils import sequence_parallel_chunk
from vllm.models.muse_spark.amd import model as muse_model
from vllm.models.muse_spark.common.layers import (
    materialize_residual_path_constants,
    norm_then_gated_residual,
)
from vllm.transformers_utils.configs.muse_spark import MuseSparkTextConfig
from vllm.utils.network_utils import get_open_port
from vllm.v1.worker.workspace import init_workspace_manager, reset_workspace_manager

NUM_LAYERS = 2
HIDDEN_SIZE = 256


class _TokenMixingAttention(nn.Module):
    """Mixes all tokens like attention does, without a KV cache."""

    def __init__(self, *args, prefix: str = "", **kwargs) -> None:
        super().__init__()
        self.layer_name = prefix

    def forward(self, q, k, v):
        return q + q.float().mean(0).to(q.dtype)


def _make_config() -> VllmConfig:
    config = VllmConfig(
        parallel_config=ParallelConfig(
            tensor_parallel_size=2,
            enable_expert_parallel=True,
            all2all_backend="allgather_reducescatter",
        ),
    )
    config.model_config = SimpleNamespace(
        architecture="MuseSpark1xForCausalLM",
        is_moe=True,
        sleep_mode_offload_cudagraph=False,
        max_model_len=4096,
        hf_text_config=MuseSparkTextConfig(
            hidden_act="silu",
            attention_bias=False,
            rope_parameters={"rope_type": "default", "rope_theta": 500000.0},
            num_hidden_layers=NUM_LAYERS,
            layer_types=["sliding_attention", "full_attention"],
            mlp_layer_types=["sparse"] * NUM_LAYERS,
            layer_rope_theta=[500000.0, 0.0],
            sliding_window=128,
            max_position_embeddings=4096,
            hidden_size=HIDDEN_SIZE,
            intermediate_size=512,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=64,
            qk_scale_factor=5.47,
            rms_norm_eps=1e-5,
            post_norm_eps=1e-8,
            residual_gate_temperature=0.3,
            num_local_experts=8,
            num_experts_per_tok=2,
            num_shared_experts=0,
            moe_hidden_size=128,
            moe_intermediate_size=128,
            routed_scaling_factor=1.0,
            output_multiplier=0.18,
        ),
        dtype=torch.bfloat16,
    )
    return config


def _make_moe(config: VllmConfig, rank: int) -> nn.Module:
    """A production MoE block with identical weights plus distinct expert shards."""
    with set_current_vllm_config(config):
        block = muse_model.MuseSpark1xMoE(
            config, layer_idx=0, prefix="model.layers.0.feed_forward"
        )
    torch.manual_seed(7)
    for param in block.parameters():
        param.normal_(std=0.08)
    assert block.shard_tokens
    experts = block.experts.routed_experts
    # Distinct rank contributions expose missing or duplicate reductions.
    experts.w13_weight.add_(rank * 0.01)
    experts.quant_method.process_weights_after_loading(experts)
    materialize_residual_path_constants(block)
    return block


def _check_moe_block(rank: int, config: VllmConfig) -> None:
    block = _make_moe(config, rank)
    # Token counts with and without padding to the TP size.
    for num_tokens in (301, 512):
        torch.manual_seed(num_tokens)
        x = torch.randn(num_tokens, HIDDEN_SIZE)
        with (
            set_current_vllm_config(config),
            set_forward_context(None, config, num_tokens=num_tokens),
        ):
            actual = block(x).clone()
            # The all-reduce path that smaller batches take.
            with patch.object(muse_model, "_TOKEN_SHARDED_MIN_TOKENS", 1 << 30):
                expected = block(x).clone()
        torch.testing.assert_close(actual, expected, atol=0.002, rtol=0.02)


def _make_layers(config: VllmConfig, rank: int) -> nn.ModuleList:
    """Production layers with identical weights plus distinct rank shards."""
    with (
        set_current_vllm_config(config),
        patch.object(muse_model, "Attention", _TokenMixingAttention),
        patch.object(muse_model, "supports_fused_kv_insert", lambda attn: False),
    ):
        layers = nn.ModuleList(
            muse_model.MuseSpark1xDecoderLayer(config, prefix=f"model.layers.{i}")
            for i in range(NUM_LAYERS)
        )
    torch.manual_seed(7)
    for param in layers.parameters():
        param.normal_(std=0.08)
    for layer in layers:
        assert layer.token_sharded
        # Distinct rank contributions expose missing or duplicate reductions.
        layer.self_attn.o_proj.weight.add_(rank * 0.01)
        experts = layer.feed_forward.experts.routed_experts
        experts.w13_weight.add_(rank * 0.01)
        experts.quant_method.process_weights_after_loading(experts)
    materialize_residual_path_constants(layers)
    return layers


def _run_layers(layers, positions, x, token_sharded: bool) -> torch.Tensor:
    num_tokens = positions.shape[0]
    if token_sharded:
        x = sequence_parallel_chunk(x)
    for layer in layers:
        x = layer(positions, x, token_sharded)
    out = norm_then_gated_residual(x)
    if token_sharded:
        out = tensor_model_parallel_all_gather(out.contiguous(), 0)[:num_tokens]
    return out.clone()


def _check_token_sharded(rank: int, config: VllmConfig) -> None:
    layers = _make_layers(config, rank)
    # Token counts with and without padding to the TP size.
    for num_tokens in (301, 512):
        torch.manual_seed(num_tokens)
        x = torch.randn(num_tokens, HIDDEN_SIZE, dtype=torch.float32)
        positions = torch.arange(num_tokens)
        with (
            set_current_vllm_config(config),
            set_forward_context(None, config, num_tokens=num_tokens),
        ):
            expected = _run_layers(layers, positions, x, token_sharded=False)
            actual = _run_layers(layers, positions, x, token_sharded=True)
        torch.testing.assert_close(actual, expected, atol=0.002, rtol=0.02)


def _run(rank: int, port: int, check) -> None:
    torch.accelerator.set_device_index(rank)
    torch.set_num_threads(1)
    config = _make_config()
    init_distributed_environment(
        world_size=2,
        rank=rank,
        local_rank=rank,
        distributed_init_method=f"tcp://127.0.0.1:{port}",
    )
    try:
        init_workspace_manager(torch.device(f"cuda:{rank}"))
        with set_current_vllm_config(config):
            initialize_model_parallel(tensor_model_parallel_size=2)
            with torch.device(f"cuda:{rank}"), torch.inference_mode():
                torch.set_default_dtype(torch.bfloat16)
                check(rank, config)
    finally:
        reset_workspace_manager()
        cleanup_dist_env_and_memory()


@multi_gpu_test(num_gpus=2)
def test_token_sharded_moe_block_matches_all_reduce() -> None:
    """The token-sharded MoE block matches its all-reduce path."""
    mp.spawn(_run, args=(get_open_port(), _check_moe_block), nprocs=2)


@multi_gpu_test(num_gpus=2)
def test_token_sharded_decoder_layers_match_full_tokens() -> None:
    """The sharded residual stream and MoE block match the replicated path."""
    mp.spawn(_run, args=(get_open_port(), _check_token_sharded), nprocs=2)
