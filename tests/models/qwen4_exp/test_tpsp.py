# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Multi-GPU equivalence tests for Qwen4Exp HC and MoE sequence parallelism."""

from dataclasses import MISSING, fields
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.multiprocessing as mp
from torch import nn
from transformers import Qwen4ExpTextConfig

from vllm.platforms import current_platform

if not current_platform.is_cuda():
    pytest.skip("Qwen4Exp SP requires CUDA", allow_module_level=True)

from tests.utils import multi_gpu_test
from vllm.config import (
    EngramConfig,
    ParallelConfig,
    VllmConfig,
    set_current_vllm_config,
)
from vllm.distributed import (
    cleanup_dist_env_and_memory,
    init_distributed_environment,
    initialize_model_parallel,
    tensor_model_parallel_all_reduce,
)
from vllm.forward_context import set_forward_context
from vllm.model_executor.layers.linear import RowParallelLinear
from vllm.model_executor.layers.mamba.mamba_utils import is_conv_state_dim_first
from vllm.models.common.ops.sequence_parallel import (
    sp_all_gather,
    sp_padding_mask,
    sp_shard,
)
from vllm.models.qwen4_exp.nvidia import model as qwen4_model
from vllm.models.qwen4_exp.nvidia.model import Qwen4ExpDecoderLayer
from vllm.utils.network_utils import get_open_port
from vllm.v1.attention.backends.short_conv_attn import PleShortConvAttentionMetadata
from vllm.v1.worker.workspace import init_workspace_manager, reset_workspace_manager

from .test_ple import _ConvBatchCase, _make_conv_metadata


class _TPAttentionProjection(nn.Module):
    """Keep real TP communication without setting up an attention KV cache."""

    def __init__(self, config, *, reduce_results: bool, **kwargs) -> None:
        super().__init__()
        self.proj = RowParallelLinear(
            config.hidden_size,
            config.hidden_size,
            bias=False,
            input_is_parallel=False,
            reduce_results=reduce_results,
        )

    def forward(self, hidden_states: torch.Tensor):
        return self.proj(hidden_states)[0]


def _make_config(rank: int, *, hc_sp: bool = False, moe_sp: bool = False) -> VllmConfig:
    """Configure HC SP explicitly or MoE SP through TP=2, DP=2 and EP."""
    config = VllmConfig(
        parallel_config=ParallelConfig(
            tensor_parallel_size=2,
            data_parallel_size=2 if moe_sp else 1,
            data_parallel_rank=rank // 2,
            enable_expert_parallel=moe_sp,
            enable_hc_sp=hc_sp,
            all2all_backend="allgather_reducescatter",
        ),
    )
    config.model_config = SimpleNamespace(
        architecture="Qwen4ExpForCausalLM",
        is_moe=True,
        sleep_mode_offload_cudagraph=False,
        hf_text_config=Qwen4ExpTextConfig(
            vocab_size=128,
            eos_token_id=0,
            hidden_size=128,
            hc_count=2,
            hc_lowrank=16,
            num_hidden_layers=1,
            layer_types=["linear_attention"],
            num_experts=4,
            num_experts_per_tok=2,
            moe_intermediate_size=256,
            shared_expert_intermediate_size=256,
            ple_layer_ids=[] if moe_sp else [1],
            ple_embed_dim=128,
            heads_per_ngram=2,
            ngram_vocab_size_base=32,
            make_ngram_vocab_size_divisible_by=64,
        ),
        dtype=torch.bfloat16,
    )
    config.engram_config = EngramConfig(cpu_offload=False)
    return config


def _make_decoder(config: VllmConfig, rank: int) -> Qwen4ExpDecoderLayer:
    """Use the production constructor and identical initial weights for both modes."""
    with (
        set_current_vllm_config(config),
        patch.object(qwen4_model, "QwenGatedDeltaNetAttention", _TPAttentionProjection),
    ):
        layer = Qwen4ExpDecoderLayer(
            config, config.model_config.hf_text_config.layer_types[0], "model.layers.0"
        )
        torch.manual_seed(12)
        for param in layer.parameters():
            param.normal_(std=0.08)
        # Distinct rank contributions expose missing or duplicate reductions.
        layer.linear_attn.proj.weight.add_(rank * 0.01)
        experts = layer.mlp.experts.routed_experts
        if config.parallel_config.enable_expert_parallel:
            experts.w13_weight.add_(rank * 0.01)
        experts.quant_method.process_weights_after_loading(experts)
    return layer


def _check_decoder_ple(rank: int) -> None:
    """Compare HC SP with full tokens across a PLE prefill and cached decode."""
    configs = [_make_config(rank, hc_sp=enabled) for enabled in (False, True)]
    layers = [_make_decoder(config, rank) for config in configs]
    states = []
    for layer in layers:
        ple = layer.ple
        assert ple is not None
        state = torch.zeros(2, ple.hc_hidden_size, ple.conv_state_len)
        if not is_conv_state_dim_first():
            state = state.transpose(-1, -2).contiguous()
        ple.kv_cache = (state,)
        states.append(state)

    # Three prefill tokens straddle the TP=2 boundary and require SP padding.
    for case in (
        _ConvBatchCase(prefill_query_lens=(3,)),
        _ConvBatchCase(num_decodes=1),
    ):
        layout, num_tokens = _make_conv_metadata(case, torch.device("cuda", rank))
        metadata_args = {
            f.name: None
            for f in fields(PleShortConvAttentionMetadata)
            if f.default is MISSING and f.default_factory is MISSING
        }
        metadata = PleShortConvAttentionMetadata(**(metadata_args | vars(layout)))
        ids = torch.randint(0, 128, (num_tokens,))
        positions = torch.arange(num_tokens).expand(3, -1)
        qsl = torch.tensor([0, num_tokens], dtype=torch.int32)
        context = torch.zeros(1, layers[0].config.ngram_size - 1, dtype=torch.int64)
        hidden = torch.randn(num_tokens, 256)
        block = torch.randn(num_tokens, 128)
        injection = torch.randn(num_tokens, 2)
        outputs = []
        for config, layer in zip(configs, layers):
            assert layer.ple is not None
            inputs = (hidden, block, injection)
            if config.parallel_config.enable_hc_sp:
                inputs = tuple(sp_shard(x) for x in inputs)
            with (
                set_current_vllm_config(config),
                set_forward_context({layer.ple.prefix: metadata}, config),
            ):
                layer.ple.start_prefetch(ids, qsl, context)
                output = layer(
                    *inputs,
                    positions,
                    input_ids=ids,
                    query_start_loc=qsl,
                    ngram_context=context,
                )
                if config.parallel_config.enable_hc_sp:
                    output = tuple(sp_all_gather(x)[:num_tokens] for x in output)
                # A later MoE call may reuse workspace-backed output storage.
                outputs.append(tuple(x.clone() for x in output))
        torch.testing.assert_close(outputs[1], outputs[0], atol=0.002, rtol=0.02)
        torch.testing.assert_close(states[1], states[0], atol=0.002, rtol=0.02)


def _check_moe_sp(rank: int, config: VllmConfig) -> None:
    """Compare native SP with full-token GR and the MoE wrapper's public interface."""
    layer = _make_decoder(config, rank)
    dp_rank = config.parallel_config.data_parallel_rank
    counts = torch.tensor([3, 5], device="cpu")
    num_tokens = int(counts[dp_rank])
    torch.manual_seed(30 + dp_rank)
    hidden = torch.randn(num_tokens, 256)
    positions = torch.arange(num_tokens)
    # Include runner padding in addition to the padding introduced by SP.
    padding = sp_padding_mask(positions == num_tokens - 1, hidden)
    with (
        set_current_vllm_config(config),
        set_forward_context(
            None,
            config,
            num_tokens=num_tokens,
            num_tokens_across_dp=counts,
            is_padding=padding,
        ),
    ):
        # The reference keeps full GR rows. MoE handles its own chunk and gather.
        full_hidden, attn_input, injection = layer.attn_hyper_connection.mix(hidden)
        attn_output = tensor_model_parallel_all_reduce(layer.linear_attn(attn_input))
        full_hidden, moe_input, injection = layer.mlp_hyper_connection.combine_and_mix(
            full_hidden, attn_output, injection
        )
        expected = tuple(
            x.clone() for x in (full_hidden, layer.mlp(moe_input), injection)
        )
        output = layer(
            sp_shard(hidden),
            None,
            None,
            positions,
            input_ids=None,
            query_start_loc=None,
            ngram_context=None,
        )
        actual = tuple(sp_all_gather(x)[:num_tokens] for x in output)
    torch.testing.assert_close(actual, expected, atol=0.002, rtol=0.02)


def _run_sp(rank: int, port: int, moe_sp: bool) -> None:
    """Initialize rank-local runtime state and run one equivalence check."""
    torch.accelerator.set_device_index(rank)
    torch.set_num_threads(1)
    config = _make_config(rank, moe_sp=moe_sp)
    init_distributed_environment(
        world_size=4 if moe_sp else 2,
        rank=rank,
        local_rank=rank,
        distributed_init_method=f"tcp://127.0.0.1:{port}",
    )
    try:
        # Spawned ranks bypass the GPU worker's workspace initialization.
        init_workspace_manager(torch.device(f"cuda:{rank}"))
        with set_current_vllm_config(config):
            initialize_model_parallel(tensor_model_parallel_size=2)
            with torch.device(f"cuda:{rank}"), torch.inference_mode():
                torch.set_default_dtype(torch.bfloat16)
                if moe_sp:
                    _check_moe_sp(rank, config)
                else:
                    _check_decoder_ple(rank)
    finally:
        reset_workspace_manager()
        cleanup_dist_env_and_memory()


@multi_gpu_test(num_gpus=2)
def test_hc_sp_decoder_and_ple() -> None:
    """HC SP preserves decoder output and PLE cache across a token-shard boundary."""
    mp.spawn(_run_sp, args=(get_open_port(), False), nprocs=2)


@multi_gpu_test(num_gpus=4)
def test_moe_sp_equivalence(monkeypatch) -> None:
    """Native MoE SP preserves outputs with unequal DP batches and padding."""
    monkeypatch.setenv("VLLM_MOE_SKIP_PADDING", "1")
    mp.spawn(_run_sp, args=(get_open_port(), True), nprocs=4)
