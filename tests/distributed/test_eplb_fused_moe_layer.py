# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Test that the interaction between EPLB and FusedMoE Layer is okay

from dataclasses import dataclass

import pytest
import torch

from vllm.config import VllmConfig, set_current_vllm_config
from vllm.distributed.eplb.rebalance_execute import rearrange_expert_weights_inplace
from vllm.distributed.parallel_state import (
    ensure_model_parallel_initialized,
    get_tp_group,
)
from vllm.model_executor.layers.fused_moe.layer import FusedMoE

from .eplb_utils import distributed_run, set_env_vars_and_device


@dataclass
class TestConfig:
    num_layers: int
    num_experts: int
    num_local_experts: int
    num_topk: int
    hidden_size: int
    intermediate_size: int
    weight_dtype: torch.dtype
    weight_scale_dtype: torch.dtype | None
    column_major_scales: bool


def make_expert_weights(
    layer_idx: int,
    global_expert_idx: int,
    global_num_experts: int,
    tensor_shape: tuple[int, ...],
    tensor_dtype: torch.dtype,
    tensor_device: torch.device,
    is_column_major: bool,
) -> torch.Tensor:
    assert len(tensor_shape) == 2

    if is_column_major:
        tensor_shape = (tensor_shape[1], tensor_shape[0])

    x = torch.empty(tensor_shape, dtype=tensor_dtype, device=tensor_device)
    value_offset = (layer_idx * global_num_experts + global_expert_idx) * x.numel()
    x.view(-1).copy_(
        torch.arange(
            value_offset,
            value_offset + x.numel(),
            dtype=tensor_dtype,
            device=tensor_device,
        )
    )

    if is_column_major:
        x = torch.transpose(x, 1, 0)
        assert not x.is_contiguous()
    return x


def make_fused_moe_layer(
    rank: int,
    layer_idx: int,
    test_config: TestConfig,
) -> FusedMoE:
    fml = FusedMoE(
        num_experts=test_config.num_experts,
        top_k=test_config.num_topk,
        hidden_size=test_config.hidden_size,
        intermediate_size=test_config.intermediate_size,
        prefix=f"dummy_layer_{layer_idx}",
        activation="silu",
        is_act_and_mul=True,
        params_dtype=test_config.weight_dtype,
    )

    device = torch.device(f"cuda:{rank}")

    from functools import partial

    _make_expert_weights = partial(
        make_expert_weights,
        layer_idx=layer_idx,
        global_num_experts=test_config.num_experts,
        tensor_device=device,
    )

    assert isinstance(fml.w13_weight.data, torch.Tensor)
    assert isinstance(fml.w2_weight.data, torch.Tensor)
    fml.w13_weight.data = fml.w13_weight.data.to(device=device)
    fml.w2_weight.data = fml.w2_weight.data.to(device=device)
    w13_weight = fml.w13_weight.data
    w2_weight = fml.w2_weight.data
    assert w13_weight.size(0) == test_config.num_local_experts
    for i in range(test_config.num_local_experts):
        g_i = rank * test_config.num_local_experts + i
        w13_weight_e = w13_weight[i]
        w2_weight_e = w2_weight[i]
        w13_weight_e.copy_(
            _make_expert_weights(
                global_expert_idx=g_i,
                tensor_shape=w13_weight_e.shape,
                tensor_dtype=w13_weight_e.dtype,
                is_column_major=False,
            )
        )
        w2_weight_e.copy_(
            _make_expert_weights(
                global_expert_idx=g_i,
                tensor_shape=w2_weight_e.shape,
                tensor_dtype=w2_weight_e.dtype,
                is_column_major=False,
            )
        )

    block_size = 16

    def block_quant_scales_shape(
        shape: tuple[int, ...], is_column_major: bool
    ) -> tuple[int, ...]:
        assert len(shape) == 3
        if not is_column_major:
            return (shape[0], shape[1] // block_size, shape[2] // block_size)
        else:
            return (shape[0], shape[2] // block_size, shape[1] // block_size)

    is_column_major = test_config.column_major_scales
    w13_weight_scale_inv = torch.empty(
        block_quant_scales_shape(w13_weight.shape, is_column_major),
        dtype=test_config.weight_dtype,
        device=device,
    )
    w2_weight_scale_inv = torch.empty(
        block_quant_scales_shape(w2_weight.shape, is_column_major),
        dtype=test_config.weight_dtype,
        device=device,
    )

    for i in range(test_config.num_local_experts):
        g_i = rank * test_config.num_local_experts + i
        w13_s_e = w13_weight_scale_inv[i]
        w2_s_e = w2_weight_scale_inv[i]
        w13_s_e.copy_(
            _make_expert_weights(
                global_expert_idx=g_i,
                tensor_shape=w13_s_e.shape,
                tensor_dtype=w13_s_e.dtype,
                # Fill data in row-major and then
                # transpose if test_config requires col-major.
                is_column_major=False,
            )
        )
        w2_s_e.copy_(
            _make_expert_weights(
                global_expert_idx=g_i,
                tensor_shape=w2_s_e.shape,
                tensor_dtype=w2_s_e.dtype,
                is_column_major=False,
            )
        )
    if is_column_major:
        w13_weight_scale_inv = torch.transpose(w13_weight_scale_inv, 1, 2)
        w2_weight_scale_inv = torch.transpose(w2_weight_scale_inv, 1, 2)
        assert not w13_weight_scale_inv.is_contiguous()
        assert not w2_weight_scale_inv.is_contiguous()

    # Add scales to the parameter list
    fml.w13_weight_scale_inv = torch.nn.Parameter(
        w13_weight_scale_inv, requires_grad=False
    )
    fml.w2_weight_scale_inv = torch.nn.Parameter(
        w2_weight_scale_inv, requires_grad=False
    )

    return fml


def _test_eplb_fml(env, world_size: int, test_config: TestConfig):
    # Initialize model parallel (using tensor parallel as an entrypoint
    # to expert parallel)
    set_env_vars_and_device(env)

    vllm_config = VllmConfig()
    vllm_config.parallel_config.tensor_parallel_size = world_size
    vllm_config.parallel_config.enable_expert_parallel = True

    with set_current_vllm_config(vllm_config):
        ensure_model_parallel_initialized(
            tensor_model_parallel_size=world_size, pipeline_model_parallel_size=1
        )

        ep_group = get_tp_group().cpu_group
        ep_rank = torch.distributed.get_rank()

        fml_layers = [
            make_fused_moe_layer(ep_rank, layer_idx, test_config)
            for layer_idx in range(test_config.num_layers)
        ]
        rank_expert_weights = [fml.get_expert_weights() for fml in fml_layers]

        indices = torch.zeros(
            test_config.num_layers, test_config.num_experts, dtype=torch.long
        )
        for lidx in range(test_config.num_layers):
            indices[lidx] = torch.Tensor(range(test_config.num_experts))

        shuffled_indices = torch.zeros_like(indices)
        for lidx in range(test_config.num_layers):
            shuffled_indices[lidx] = torch.randperm(test_config.num_experts)

        rearrange_expert_weights_inplace(
            indices,
            shuffled_indices,
            rank_expert_weights,
            ep_group,
            is_profile=False,
        )

        num_local_experts = test_config.num_local_experts
        num_global_experts = test_config.num_experts
        for lidx, fml in enumerate(fml_layers):
            for name, w in fml.named_parameters():
                for e in range(num_local_experts):
                    g_e = shuffled_indices[lidx][ep_rank * num_local_experts + e]
                    ref = make_expert_weights(
                        layer_idx=lidx,
                        global_expert_idx=int(g_e.item()),
                        global_num_experts=num_global_experts,
                        tensor_shape=w[e].shape,
                        tensor_dtype=w[e].dtype,
                        tensor_device=w[e].device,
                        is_column_major=not w[e].is_contiguous(),
                    )
                    assert w[e].shape == ref.shape and w[e].stride() == ref.stride(), (
                        f"w[{e}] {w[e].size()} {w[e].stride()} vs "
                        f"ref {ref.size()} {ref.stride()}"
                    )
                    torch.testing.assert_close(w[e], ref)


@pytest.mark.parametrize("world_size", [2])
@pytest.mark.parametrize("num_layers", [4])
@pytest.mark.parametrize("num_experts", [16])
@pytest.mark.parametrize("hidden_size", [256])
@pytest.mark.parametrize("intermediate_size", [256])
@pytest.mark.parametrize("column_major_scales", [True, False])
def test_eplb_fml(
    world_size: int,
    num_layers: int,
    num_experts: int,
    hidden_size: int,
    intermediate_size: int,
    column_major_scales: bool,
):
    if torch.cuda.device_count() < world_size:
        pytest.skip(f"Need at least {world_size} GPUs to run the test")

    num_local_experts = num_experts // world_size
    num_topk = 4
    # The dtypes are fine as we are essentially just checking data-copies
    weight_dtype = torch.bfloat16
    weight_scale_dtype = torch.bfloat16

    test_config = TestConfig(
        num_layers=num_layers,
        num_experts=num_experts,
        num_local_experts=num_local_experts,
        num_topk=num_topk,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        weight_dtype=weight_dtype,
        weight_scale_dtype=weight_scale_dtype,
        column_major_scales=column_major_scales,
    )

    distributed_run(
        _test_eplb_fml,
        world_size,
        test_config,
    )
