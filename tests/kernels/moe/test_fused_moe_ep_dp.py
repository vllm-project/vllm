# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Layer-level test for FusedMoE with DP+EP and all-to-all backends.

Exercises the full FusedMoE nn.Module end-to-end including:
- Router (fused_topk)
- All-to-all dispatch/combine (DeepEP HT, DeepEP LL, allgather-reducescatter)
- Weight handling and expert sharding
- The DefaultMoERunner orchestration

Launches processes once per backend and loops over shapes inside the worker
to minimize process creation overhead.

Run: pytest -v -s tests/kernels/moe/test_fused_moe_ep_dp.py
"""

import os

import pytest
import torch
import torch.multiprocessing as mp

from tests.kernels.moe.utils import make_test_weights
from tests.kernels.quantization.nvfp4_utils import (
    FLOAT4_E2M1_MAX,
    FLOAT8_E4M3_MAX,
    dequantize_nvfp4_to_dtype,
)
from vllm import _custom_ops as ops
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.distributed.parallel_state import (
    ensure_model_parallel_initialized,
    get_ep_group,
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
)
from vllm.platforms import current_platform
from vllm.utils.import_utils import has_deep_ep
from vllm.utils.network_utils import get_open_port
from vllm.utils.system_utils import update_environment_variables
from vllm.utils.torch_utils import set_random_seed
from vllm.v1.worker.workspace import init_workspace_manager

from ...utils import multi_gpu_test

mp.set_start_method("spawn", force=True)


# ---------------------------------------------------------------------------
# Distributed launch helpers
# ---------------------------------------------------------------------------


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


def _cleanup_between_configs(prefix: str, backend: str):
    """Clean up state between test config iterations.

    Follows the pattern from test_moe_layer.py:
    clear compilation state and destroy DeepEP all2all buffers on SM100.
    """
    vllm_config = VllmConfig()
    cc = vllm_config.compilation_config
    if prefix in cc.static_forward_context:
        del cc.static_forward_context[prefix]
        if prefix in cc.static_all_moe_layers:
            cc.static_all_moe_layers.remove(prefix)

    cap = current_platform.get_device_capability()
    if (
        cap is not None
        and cap.major == 10
        and backend in ("deepep_low_latency", "deepep_high_throughput")
    ):
        torch.accelerator.synchronize()
        all2all_manager = get_ep_group().device_communicator.all2all_manager
        if all2all_manager is not None:
            all2all_manager.destroy()

    torch.accelerator.empty_cache()


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
    shapes: list[tuple[int, int, int]],
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

        for m, hidden_size, intermediate_size in shapes:
            prefix = f"test_bf16_{rank}_{m}_{hidden_size}"

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

            set_random_seed(100 + rank)
            hidden_states = torch.randn(m, hidden_size, device=device, dtype=dtype) / 10
            router_logits = torch.randn(m, num_experts, device=device, dtype=dtype) / 10

            topk_weights, topk_ids, _ = fused_topk(
                hidden_states,
                router_logits.float(),
                topk,
                renormalize=True,
            )
            ref_output = torch_moe_ref(
                hidden_states, w13_full, w2_full, topk_ids, topk_weights
            )

            fml = FusedMoE(
                num_experts=num_experts,
                top_k=topk,
                hidden_size=hidden_size,
                intermediate_size=intermediate_size,
                prefix=prefix,
                activation="silu",
                is_act_and_mul=True,
                params_dtype=dtype,
                reduce_results=False,
            )
            fml = fml.to(device)

            ep_rank = fml.moe_parallel_config.ep_rank
            n_local = fml.local_num_experts
            start = ep_rank * n_local
            end = start + n_local
            fml.w13_weight.data.copy_(w13_full[start:end])
            fml.w2_weight.data.copy_(w2_full[start:end])

            fml.quant_method.process_weights_after_loading(fml)
            fml.maybe_init_modular_kernel()

            num_tokens_across_dp = torch.tensor(
                [m] * world_size, dtype=torch.int, device="cpu"
            )
            with set_forward_context(
                None,
                vllm_config,
                num_tokens=m,
                num_tokens_across_dp=num_tokens_across_dp,
            ):
                output = fml(hidden_states, router_logits)

            torch.testing.assert_close(ref_output, output, atol=5e-2, rtol=5e-2)

            del fml
            _cleanup_between_configs(prefix, backend)


BACKENDS = [
    "allgather_reducescatter",
    "deepep_high_throughput",
    "deepep_low_latency",
]

BF16_SHAPES = [
    (16, 2048, 512),
    (16, 4096, 1024),
]


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("num_experts,topk", [(8, 2)])
@multi_gpu_test(num_gpus=2)
def test_fused_moe_ep_dp(
    backend: str,
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
        BF16_SHAPES,
    )


# ---------------------------------------------------------------------------
# NVFP4 + DP+EP test (CuTeDSL experts)
# ---------------------------------------------------------------------------


def _worker_nvfp4(
    env: dict[str, str],
    world_size: int,
    backend: str,
    num_experts: int,
    topk: int,
    shapes: list[tuple[int, int, int]],
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
    # Force CuteDSL backend. Auto-upgrades to batched variant for
    # DeepEP LL (BatchedExperts format) via oracle lines 212-217.
    vllm_config.kernel_config.moe_backend = "flashinfer_cutedsl"

    with set_current_vllm_config(vllm_config):
        ensure_model_parallel_initialized(
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
        )

        for m, hidden_size, intermediate_size in shapes:
            prefix = f"test_nvfp4_{rank}_{m}_{hidden_size}"
            num_local_experts = num_experts // world_size

            # Generate FULL quantized weights with matching scales
            # (same seed on all ranks so weights are identical)
            set_random_seed(42)
            (_, w13_q, w13_bs, w13_gs), (_, w2_q, w2_bs, w2_gs) = make_test_weights(
                num_experts,
                intermediate_size,
                hidden_size,
                in_dtype=dtype,
                quant_dtype="nvfp4",
            )
            assert w13_bs is not None and w13_gs is not None
            assert w2_bs is not None and w2_gs is not None

            # Dequantize FULL weights for reference BEFORE any transforms
            w13_deq = torch.empty(
                num_experts,
                2 * intermediate_size,
                hidden_size,
                device="cuda",
                dtype=dtype,
            )
            w2_deq = torch.empty(
                num_experts,
                hidden_size,
                intermediate_size,
                device="cuda",
                dtype=dtype,
            )
            for idx in range(num_experts):
                w13_deq[idx] = dequantize_nvfp4_to_dtype(
                    w13_q[idx], w13_bs[idx], w13_gs[idx], dtype, device
                )
                w2_deq[idx] = dequantize_nvfp4_to_dtype(
                    w2_q[idx], w2_bs[idx], w2_gs[idx], dtype, device
                )

            # Create FusedMoE layer with NVFP4 quant config
            # FusedMoE.__init__ creates quant_method and registers weights
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
                prefix=prefix,
                activation="silu",
                is_act_and_mul=True,
                params_dtype=dtype,
                reduce_results=False,
                quant_config=quant_config,
            )
            fml = fml.to(device)

            # Fill LOCAL expert weights and scales from make_test_weights
            ep_rank = fml.moe_parallel_config.ep_rank
            start = ep_rank * num_local_experts
            end = start + num_local_experts
            fml.w13_weight.data.copy_(w13_q[start:end])
            fml.w2_weight.data.copy_(w2_q[start:end])
            fml.w13_weight_scale.data.copy_(w13_bs[start:end])
            fml.w2_weight_scale.data.copy_(w2_bs[start:end])
            # weight_scale_2 stores 1/global_scale (the kernel applies it
            # as alpha = 1/(a_scale * w_scale), see test_cutedsl_moe.py:428)
            fml.w13_weight_scale_2.data[:, 0] = 1.0 / w13_gs[start:end]
            fml.w13_weight_scale_2.data[:, 1] = 1.0 / w13_gs[start:end]
            fml.w2_weight_scale_2.data.copy_(1.0 / w2_gs[start:end])
            fml.w13_input_scale.data.fill_(1.0)
            fml.w2_input_scale.data.fill_(1.0)

            fml.quant_method.process_weights_after_loading(fml)
            fml.maybe_init_modular_kernel()

            # Generate per-rank input
            set_random_seed(100 + rank)
            hidden_states = torch.randn(m, hidden_size, device=device, dtype=dtype) / 10
            router_logits = torch.randn(m, num_experts, device=device, dtype=dtype) / 10

            # Reference: quantize+dequantize activations, then torch_moe_ref
            a_global_scale = (
                (FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX)
                / torch.amax(hidden_states.abs().flatten(), dim=-1)
            ).to(torch.float32)
            a_fp4, a_scale = ops.scaled_fp4_quant(hidden_states, a_global_scale)
            a_deq = dequantize_nvfp4_to_dtype(
                a_fp4, a_scale, a_global_scale, dtype, device
            )

            topk_weights, topk_ids, _ = fused_topk(
                hidden_states,
                router_logits.float(),
                topk,
                renormalize=True,
            )
            ref_output = torch_moe_ref(a_deq, w13_deq, w2_deq, topk_ids, topk_weights)

            # Forward
            num_tokens_across_dp = torch.tensor(
                [m] * world_size, dtype=torch.int, device="cpu"
            )
            with set_forward_context(
                None,
                vllm_config,
                num_tokens=m,
                num_tokens_across_dp=num_tokens_across_dp,
            ):
                output = fml(hidden_states, router_logits)

            # FP4 quantization + distributed dispatch introduces error.
            # test_cutedsl_moe.py uses atol=2e-1 for kernel-level tests.
            torch.testing.assert_close(ref_output, output, atol=4e-1, rtol=4e-1)

            del fml
            _cleanup_between_configs(prefix, backend)


NVFP4_SHAPES = [
    # hidden_size must be in DeepEP LL SUPPORTED_HIDDEN_SIZES.
    # M kept small for DeepEP LL buffer constraints.
    (16, 2048, 256),
    (32, 4096, 512),
]

NVFP4_BACKENDS = [
    # deepep_high_throughput -> Standard format -> masked_gemm CuteDSL
    "deepep_high_throughput",
    # TODO: deepep_low_latency needs LL buffer pre-allocation for
    # batched CuteDSL. Add once buffer sizing is resolved.
]


@pytest.mark.parametrize("backend", NVFP4_BACKENDS)
@pytest.mark.parametrize("num_experts,topk", [(8, 2)])
@multi_gpu_test(num_gpus=2)
def test_fused_moe_ep_dp_nvfp4(
    backend: str,
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
        backend,
        num_experts,
        topk,
        NVFP4_SHAPES,
    )
