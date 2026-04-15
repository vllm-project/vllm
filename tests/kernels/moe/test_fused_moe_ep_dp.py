# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Layer-level test for FusedMoE with DP+EP and all-to-all backends.

Exercises the full FusedMoE nn.Module end-to-end including:
- Router (fused_topk)
- All-to-all dispatch/combine (DeepEP HT, DeepEP LL, allgather-reducescatter)
- Weight handling and expert sharding
- The DefaultMoERunner orchestration

Run: pytest -v -s tests/kernels/moe/test_fused_moe_ep_dp.py
"""

import os

import pytest
import torch
import torch.multiprocessing as mp

from tests.kernels.moe.utils import make_test_quant_config
from tests.kernels.quantization.nvfp4_utils import (
    FLOAT4_E2M1_MAX,
    FLOAT8_E4M3_MAX,
    dequantize_nvfp4_to_dtype,
)
from vllm import _custom_ops as ops
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.distributed.parallel_state import (
    ensure_model_parallel_initialized,
    init_distributed_environment,
)
from vllm.forward_context import set_forward_context
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.fused_moe.layer import FusedMoE
from vllm.model_executor.layers.fused_moe.router.fused_topk_router import (
    fused_topk,
)
from vllm.model_executor.layers.quantization.modelopt import (
    ModelOptNvFp4Config,
    ModelOptNvFp4FusedMoE,
)
from vllm.platforms import current_platform
from vllm.utils.import_utils import has_deep_ep
from vllm.utils.network_utils import get_open_port
from vllm.utils.system_utils import update_environment_variables
from vllm.utils.torch_utils import set_random_seed
from vllm.v1.worker.workspace import init_workspace_manager

from ...utils import multi_gpu_test

mp.set_start_method("spawn", force=True)


def _distributed_run(fn, world_size, *args, extra_env=None):
    """Launch fn across world_size processes with dynamic port allocation."""
    port = get_open_port()
    processes: list[mp.Process] = []
    for i in range(world_size):
        env: dict[str, str] = {
            "RANK": str(i),
            "LOCAL_RANK": str(i),
            "WORLD_SIZE": str(world_size),
            "LOCAL_WORLD_SIZE": str(world_size),
            "MASTER_ADDR": "localhost",
            "MASTER_PORT": str(port),
        }
        if extra_env:
            env.update(extra_env)
        p = mp.Process(target=fn, args=(env, world_size, *args))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()

    for p in processes:
        assert p.exitcode == 0


def _init_worker(env: dict[str, str]):
    """Initialize distributed environment for a worker process."""
    update_environment_variables(env)
    rank = int(os.environ["LOCAL_RANK"])
    torch.accelerator.set_device_index(rank)
    device = torch.device(f"cuda:{rank}")

    bare_config = VllmConfig()
    with set_current_vllm_config(bare_config):
        init_distributed_environment()

    return rank, device


def torch_moe_ref(
    a: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
) -> torch.Tensor:
    """Single-GPU PyTorch MoE reference (all experts, no sharding)."""
    m, _ = a.shape
    topk = topk_ids.size(1)
    out = torch.zeros_like(a)
    for i in range(m):
        for j in range(topk):
            e = topk_ids[i][j]
            e_w = topk_weights[i][j]
            x = a[i] @ w1[e].transpose(0, 1)
            x = SiluAndMul()(x.unsqueeze(0)).squeeze(0)
            out[i] += (x @ w2[e].transpose(0, 1)) * e_w
    return out


# ---------------------------------------------------------------------------
# BF16 DP+EP test
# ---------------------------------------------------------------------------


def _worker_bf16(
    env: dict[str, str],
    world_size: int,
    backend: str,
    num_experts: int,
    topk: int,
    hidden_size: int,
    intermediate_size: int,
    M: int,
):
    dtype = torch.bfloat16
    rank, device = _init_worker(env)
    init_workspace_manager(device)

    vllm_config = VllmConfig()
    vllm_config.parallel_config.data_parallel_size = world_size
    vllm_config.parallel_config.data_parallel_rank = rank
    vllm_config.parallel_config.enable_expert_parallel = True
    vllm_config.parallel_config.all2all_backend = backend
    vllm_config.parallel_config.is_moe_model = True
    vllm_config.compilation_config.fast_moe_cold_start = False

    with set_current_vllm_config(vllm_config):
        ensure_model_parallel_initialized(
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
        )

        # Generate full weights (same seed on all ranks)
        set_random_seed(42)
        w13_full = (
            torch.randn(
                num_experts,
                2 * intermediate_size,
                hidden_size,
                device=device,
                dtype=dtype,
            )
            / 10
        )
        w2_full = (
            torch.randn(
                num_experts,
                hidden_size,
                intermediate_size,
                device=device,
                dtype=dtype,
            )
            / 10
        )

        # Generate per-rank input (different per rank)
        set_random_seed(100 + rank)
        hidden_states = (
            torch.randn(
                M,
                hidden_size,
                device=device,
                dtype=dtype,
            )
            / 10
        )
        router_logits = (
            torch.randn(
                M,
                num_experts,
                device=device,
                dtype=dtype,
            )
            / 10
        )

        # Reference output using all experts locally
        topk_weights, topk_ids, _ = fused_topk(
            hidden_states,
            router_logits.float(),
            topk,
            renormalize=True,
        )
        ref_output = torch_moe_ref(
            hidden_states,
            w13_full,
            w2_full,
            topk_ids,
            topk_weights,
        )

        # Create FusedMoE layer
        fml = FusedMoE(
            num_experts=num_experts,
            top_k=topk,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            prefix=f"test_bf16_{rank}",
            activation="silu",
            is_act_and_mul=True,
            params_dtype=dtype,
            reduce_results=False,
        )
        fml = fml.to(device)

        # Fill local expert weights from the full set
        ep_rank = fml.moe_parallel_config.ep_rank
        n_local = fml.local_num_experts
        start = ep_rank * n_local
        end = start + n_local
        fml.w13_weight.data.copy_(w13_full[start:end])
        fml.w2_weight.data.copy_(w2_full[start:end])

        # Process weights and init kernel
        fml.quant_method.process_weights_after_loading(fml)
        fml.maybe_init_modular_kernel()

        # Forward
        num_tokens_across_dp = torch.tensor(
            [M] * world_size,
            dtype=torch.int,
            device="cpu",
        )
        with set_forward_context(
            None,
            vllm_config,
            num_tokens=M,
            num_tokens_across_dp=num_tokens_across_dp,
        ):
            output = fml(hidden_states, router_logits)

        torch.testing.assert_close(ref_output, output, atol=5e-2, rtol=5e-2)


BACKENDS = [
    "allgather_reducescatter",
    "deepep_high_throughput",
    "deepep_low_latency",
]

BF16_SHAPES = [
    (16, 2048, 512),
    (64, 4096, 1024),
]


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("m,hidden_size,intermediate_size", BF16_SHAPES)
@pytest.mark.parametrize("num_experts,topk", [(8, 2)])
@multi_gpu_test(num_gpus=2)
def test_fused_moe_ep_dp(
    backend: str,
    m: int,
    hidden_size: int,
    intermediate_size: int,
    num_experts: int,
    topk: int,
):
    if (
        backend in ("deepep_high_throughput", "deepep_low_latency")
        and not has_deep_ep()
    ):
        pytest.skip("DeepEP not available")

    _distributed_run(
        _worker_bf16,
        2,
        backend,
        num_experts,
        topk,
        hidden_size,
        intermediate_size,
        m,
    )


# ---------------------------------------------------------------------------
# NVFP4 + DP+EP test (CuTeDSL experts + DeepEP LL)
# ---------------------------------------------------------------------------


def _worker_nvfp4(
    env: dict[str, str],
    world_size: int,
    num_experts: int,
    topk: int,
    hidden_size: int,
    intermediate_size: int,
    M: int,
):
    dtype = torch.bfloat16
    backend = "deepep_low_latency"
    rank, device = _init_worker(env)
    init_workspace_manager(device)

    vllm_config = VllmConfig()
    vllm_config.parallel_config.data_parallel_size = world_size
    vllm_config.parallel_config.data_parallel_rank = rank
    vllm_config.parallel_config.enable_expert_parallel = True
    vllm_config.parallel_config.all2all_backend = backend
    vllm_config.parallel_config.is_moe_model = True
    vllm_config.compilation_config.fast_moe_cold_start = False

    with set_current_vllm_config(vllm_config):
        ensure_model_parallel_initialized(
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
        )

        # Number of local experts for this rank
        num_local_experts = num_experts // world_size

        # Create FusedMoE layer with NVFP4 quant config
        quant_config = ModelOptNvFp4Config(
            is_checkpoint_nvfp4_serialized=True,
            kv_cache_quant_algo=None,
            exclude_modules=[],
        )
        fml = FusedMoE(
            num_experts=num_experts,
            top_k=topk,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            prefix=f"test_nvfp4_{rank}",
            activation="silu",
            is_act_and_mul=True,
            params_dtype=dtype,
            reduce_results=False,
            quant_config=quant_config,
        )

        # Register NVFP4 weight parameters
        nvfp4_method = ModelOptNvFp4FusedMoE(quant_config, fml)
        nvfp4_method.create_weights(
            fml,
            num_local_experts,
            hidden_size,
            intermediate_size,
            params_dtype=torch.uint8,
            global_num_experts=num_experts,
        )
        fml = fml.to(device)

        # Generate quantized weights (same seed on all ranks)
        set_random_seed(42)
        w1_q, w2_q, _ = make_test_quant_config(
            num_local_experts,
            intermediate_size,
            hidden_size,
            in_dtype=dtype,
            quant_dtype="nvfp4",
            block_shape=None,
            per_act_token_quant=False,
        )

        # Fill local expert weights and scales
        fml.w13_weight.data = w1_q
        fml.w2_weight.data = w2_q
        fml.w2_input_scale.data = torch.randn_like(fml.w2_input_scale.data) / 5
        fml.w13_input_scale.data = torch.randn_like(fml.w13_input_scale.data) / 5
        fml.w2_weight_scale_2.data = torch.randn_like(fml.w2_weight_scale_2.data) / 5
        fml.w13_weight_scale_2.data = torch.randn_like(fml.w13_weight_scale_2.data) / 5
        fml.w2_weight_scale.data = (
            torch.randn(fml.w2_weight_scale.data.shape, device=device) / 5
        ).to(fml.w2_weight_scale.data.dtype)
        fml.w13_weight_scale.data = (
            torch.randn(fml.w13_weight_scale.data.shape, device=device) / 5
        ).to(fml.w13_weight_scale.data.dtype)

        # Dequantize weights BEFORE process_weights_after_loading, which
        # transforms scales into kernel-specific format.
        w13_deq = torch.empty(
            num_local_experts,
            2 * intermediate_size,
            hidden_size,
            device=device,
            dtype=dtype,
        )
        w2_deq = torch.empty(
            num_local_experts,
            hidden_size,
            intermediate_size,
            device=device,
            dtype=dtype,
        )
        for idx in range(num_local_experts):
            w13_deq[idx] = dequantize_nvfp4_to_dtype(
                fml.w13_weight.data[idx],
                fml.w13_weight_scale.data[idx],
                fml.w13_weight_scale_2.data[idx, 0],
                dtype,
                device,
            )
            w2_deq[idx] = dequantize_nvfp4_to_dtype(
                fml.w2_weight.data[idx],
                fml.w2_weight_scale.data[idx],
                fml.w2_weight_scale_2.data[idx],
                dtype,
                device,
            )

        # Now transform weights for the kernel
        nvfp4_method.process_weights_after_loading(fml)
        fml.maybe_init_modular_kernel()

        # Generate per-rank input (different per rank)
        set_random_seed(100 + rank)
        hidden_states = (
            torch.randn(
                M,
                hidden_size,
                device=device,
                dtype=dtype,
            )
            / 10
        )
        router_logits = (
            torch.randn(
                M,
                num_experts,
                device=device,
                dtype=dtype,
            )
            / 10
        )

        # Build full-expert dequantized weights for reference
        # (allgather across ranks)
        w13_full = torch.zeros(
            num_experts,
            2 * intermediate_size,
            hidden_size,
            device=device,
            dtype=dtype,
        )
        w2_full = torch.zeros(
            num_experts,
            hidden_size,
            intermediate_size,
            device=device,
            dtype=dtype,
        )
        ep_rank = fml.moe_parallel_config.ep_rank
        start = ep_rank * num_local_experts
        end = start + num_local_experts
        w13_full[start:end] = w13_deq
        w2_full[start:end] = w2_deq
        torch.distributed.all_reduce(w13_full)
        torch.distributed.all_reduce(w2_full)

        # Reference: quantize activations, dequantize, run torch_moe_ref
        a_global_scale = (
            (FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX)
            / torch.amax(hidden_states.abs().flatten(), dim=-1)
        ).to(torch.float32)
        a_fp4, a_scale = ops.scaled_fp4_quant(hidden_states, a_global_scale)
        a_deq = dequantize_nvfp4_to_dtype(
            a_fp4,
            a_scale,
            a_global_scale,
            dtype,
            device,
        )

        topk_weights, topk_ids, _ = fused_topk(
            hidden_states,
            router_logits.float(),
            topk,
            renormalize=True,
        )
        ref_output = torch_moe_ref(
            a_deq,
            w13_full,
            w2_full,
            topk_ids,
            topk_weights,
        )

        # Forward
        num_tokens_across_dp = torch.tensor(
            [M] * world_size,
            dtype=torch.int,
            device="cpu",
        )
        with set_forward_context(
            None,
            vllm_config,
            num_tokens=M,
            num_tokens_across_dp=num_tokens_across_dp,
        ):
            output = fml(hidden_states, router_logits)

        torch.testing.assert_close(ref_output, output, atol=1e-1, rtol=1e-1)


NVFP4_SHAPES = [
    # hidden_size must be in DeepEP LL SUPPORTED_HIDDEN_SIZES
    (16, 2048, 256),
    (64, 4096, 512),
]


@pytest.mark.parametrize("m,hidden_size,intermediate_size", NVFP4_SHAPES)
@pytest.mark.parametrize("num_experts,topk", [(8, 2)])
@multi_gpu_test(num_gpus=2)
def test_fused_moe_ep_dp_nvfp4(
    m: int,
    hidden_size: int,
    intermediate_size: int,
    num_experts: int,
    topk: int,
):
    if not current_platform.has_device_capability(100):
        pytest.skip("NVFP4 CuTeDSL requires SM100+ (Blackwell)")

    if not has_deep_ep():
        pytest.skip("DeepEP not available")

    _distributed_run(
        _worker_nvfp4,
        2,
        num_experts,
        topk,
        hidden_size,
        intermediate_size,
        m,
        extra_env={
            "VLLM_USE_FLASHINFER_MOE_FP4": "1",
            "VLLM_FLASHINFER_MOE_BACKEND": "cutedsl",
        },
    )
