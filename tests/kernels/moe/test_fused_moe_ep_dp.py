# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Layer-level FusedMoE tests with data-parallel + expert-parallel routing.

Two public tests, each launching 2 worker processes once and sweeping
configs internally to amortize spawn cost:

- test_fused_moe_ep_dp_bf16:   bf16 weights, sweeps 3 a2a backends x 2 shapes
- test_fused_moe_ep_dp_nvfp4:  NVFP4 weights + FlashInfer CuTeDSL (masked_gemm)
                               + DeepEP low-latency, sweeps 2 shapes

Reference for NVFP4 follows the convention in tests/kernels/moe/test_cutedsl_moe.py:
per-expert weight global scales from amax, unit input global scale, and
quant-dequant both activations and weights before a pure-torch MoE reference.
"""

import gc

import pytest
import torch
from torch.distributed import ProcessGroup

from tests.kernels.moe.modular_kernel_tools.parallel_utils import (
    ProcessGroupInfo,
    parallel_launch_with_config,
)
from tests.kernels.moe.utils import make_test_weights
from tests.kernels.quantization.nvfp4_utils import (
    dequantize_nvfp4_to_dtype,
)
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.forward_context import set_forward_context
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.fused_moe.layer import FusedMoE
from vllm.model_executor.layers.fused_moe.prepare_finalize.deepep_ll import (
    DeepEPLLPrepareAndFinalize,
)
from vllm.model_executor.layers.fused_moe.router.fused_topk_router import fused_topk
from vllm.model_executor.layers.quantization.modelopt import ModelOptNvFp4Config
from vllm.platforms import current_platform
from vllm.utils.import_utils import has_deep_ep
from vllm.utils.torch_utils import set_random_seed
from vllm.v1.worker.workspace import init_workspace_manager

from ...utils import multi_gpu_test

NUM_EXPERTS = 8
TOPK = 2
WORLD_SIZE = 2

BF16_BACKENDS = [
    "allgather_reducescatter",
    "deepep_high_throughput",
    "deepep_low_latency",
]

# (M, hidden_size, intermediate_size).
# hidden_size must be in DeepEPLLPrepareAndFinalize.SUPPORTED_HIDDEN_SIZES
# because deepep_low_latency rounds the layer's hidden_size up to the nearest
# supported value, which would then mismatch the weight tensor shape.
BF16_SHAPES = [
    (16, 2048, 512),
    (64, 4096, 1024),
]

NVFP4_SHAPES = [
    (16, 2048, 256),
    (64, 4096, 512),
]

BF16_TOL = dict(atol=5e-2, rtol=5e-2)


def _nvfp4_tol(hidden_size: int) -> dict:
    # Matches tests/kernels/moe/test_moe_layer.py modelopt_fp4 formula:
    # error scales linearly with the GEMM K-dimension.
    tol = 1e-1 + hidden_size * 5e-4
    return dict(atol=tol, rtol=tol)


def _torch_moe_ref(
    a: torch.Tensor,
    w13: torch.Tensor,
    w2: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
) -> torch.Tensor:
    """Dense per-token MoE reference in the tensor dtype of `a`."""
    m = a.size(0)
    topk = topk_ids.size(1)
    silu_and_mul = SiluAndMul()
    out = torch.zeros_like(a)
    for i in range(m):
        for j in range(topk):
            e = topk_ids[i, j]
            w = topk_weights[i, j]
            out[i] += (silu_and_mul(a[i] @ w13[e].T) @ w2[e].T) * w
    return out


def _make_forward_ctx(vllm_config: VllmConfig, m: int):
    num_tokens_across_dp = torch.tensor([m] * WORLD_SIZE, dtype=torch.int, device="cpu")
    return set_forward_context(
        None,
        vllm_config,
        num_tokens=m,
        num_tokens_across_dp=num_tokens_across_dp,
    )


def _cleanup_between_configs() -> None:
    gc.collect()
    torch.accelerator.synchronize()
    torch.accelerator.empty_cache()
    torch.distributed.barrier()


def _run_bf16_one(
    pgi: ProcessGroupInfo,
    vllm_config: VllmConfig,
    m: int,
    hidden_size: int,
    intermediate_size: int,
) -> None:
    dtype = torch.bfloat16
    backend = vllm_config.parallel_config.all2all_backend

    with set_current_vllm_config(vllm_config):
        set_random_seed(42)
        w13_full = (
            torch.randn(
                NUM_EXPERTS,
                2 * intermediate_size,
                hidden_size,
                device="cuda",
                dtype=dtype,
            )
            / 10
        )
        w2_full = (
            torch.randn(
                NUM_EXPERTS,
                hidden_size,
                intermediate_size,
                device="cuda",
                dtype=dtype,
            )
            / 10
        )

        set_random_seed(100 + pgi.rank)
        hidden_states = torch.randn(m, hidden_size, device="cuda", dtype=dtype) / 10
        router_logits = torch.randn(m, NUM_EXPERTS, device="cuda", dtype=dtype) / 10

        topk_weights, topk_ids, _ = fused_topk(
            hidden_states,
            router_logits.float(),
            TOPK,
            renormalize=True,
        )
        ref_output = _torch_moe_ref(
            hidden_states, w13_full, w2_full, topk_ids, topk_weights
        )

        fml = FusedMoE(
            num_experts=NUM_EXPERTS,
            top_k=TOPK,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            prefix=f"bf16_{backend}_{m}_{hidden_size}_{intermediate_size}",
            activation="silu",
            is_act_and_mul=True,
            params_dtype=dtype,
            reduce_results=False,
        ).to("cuda")

        ep_rank = fml.moe_parallel_config.ep_rank
        n_local = fml.local_num_experts
        start, end = ep_rank * n_local, (ep_rank + 1) * n_local
        fml.w13_weight.data.copy_(w13_full[start:end])
        fml.w2_weight.data.copy_(w2_full[start:end])

        fml.quant_method.process_weights_after_loading(fml)
        fml.maybe_init_modular_kernel()

        with _make_forward_ctx(vllm_config, m):
            output = fml(hidden_states, router_logits)

        torch.testing.assert_close(
            output,
            ref_output,
            msg=(
                f"bf16 mismatch: backend={backend} "
                f"M={m} H={hidden_size} I={intermediate_size} rank={pgi.rank}"
            ),
            **BF16_TOL,
        )

    del fml


def _run_nvfp4_one(
    pgi: ProcessGroupInfo,
    vllm_config: VllmConfig,
    m: int,
    hidden_size: int,
    intermediate_size: int,
) -> None:
    dtype = torch.bfloat16

    with set_current_vllm_config(vllm_config):
        set_random_seed(42)
        (_, w13_q, w13_bs, w13_gs), (_, w2_q, w2_bs, w2_gs) = make_test_weights(
            NUM_EXPERTS,
            intermediate_size,
            hidden_size,
            in_dtype=dtype,
            quant_dtype="nvfp4",
        )
        assert w13_bs is not None and w13_gs is not None
        assert w2_bs is not None and w2_gs is not None

        # Convention match test_cutedsl_moe.py: unit input global scale on
        # both layer and reference so they use the same effective scale.
        input_gs = torch.ones(1, dtype=torch.float32, device="cuda")

        set_random_seed(100 + pgi.rank)
        hidden_states = torch.randn(m, hidden_size, device="cuda", dtype=dtype) / 10
        router_logits = torch.randn(m, NUM_EXPERTS, device="cuda", dtype=dtype) / 10

        from flashinfer import fp4_quantize

        a_fp4, a_sf = fp4_quantize(hidden_states, input_gs)
        a_deq = dequantize_nvfp4_to_dtype(
            a_fp4, a_sf, input_gs, dtype, hidden_states.device
        )

        w13_deq = torch.empty(
            NUM_EXPERTS,
            2 * intermediate_size,
            hidden_size,
            device="cuda",
            dtype=dtype,
        )
        w2_deq = torch.empty(
            NUM_EXPERTS, hidden_size, intermediate_size, device="cuda", dtype=dtype
        )
        for e in range(NUM_EXPERTS):
            w13_deq[e] = dequantize_nvfp4_to_dtype(
                w13_q[e], w13_bs[e], w13_gs[e], dtype, w13_q.device
            )
            w2_deq[e] = dequantize_nvfp4_to_dtype(
                w2_q[e], w2_bs[e], w2_gs[e], dtype, w2_q.device
            )

        topk_weights, topk_ids, _ = fused_topk(
            hidden_states,
            router_logits.float(),
            TOPK,
            renormalize=True,
        )
        ref_output = _torch_moe_ref(a_deq, w13_deq, w2_deq, topk_ids, topk_weights)

        quant_config = ModelOptNvFp4Config(
            is_checkpoint_nvfp4_serialized=True,
            kv_cache_quant_algo=None,
            exclude_modules=[],
        )
        fml = FusedMoE(
            num_experts=NUM_EXPERTS,
            top_k=TOPK,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            prefix=f"nvfp4_{m}_{hidden_size}_{intermediate_size}",
            activation="silu",
            is_act_and_mul=True,
            params_dtype=dtype,
            reduce_results=False,
            quant_config=quant_config,
        ).to("cuda")

        ep_rank = fml.moe_parallel_config.ep_rank
        n_local = fml.local_num_experts
        start, end = ep_rank * n_local, (ep_rank + 1) * n_local

        fml.w13_weight.data.copy_(w13_q[start:end])
        fml.w2_weight.data.copy_(w2_q[start:end])
        fml.w13_weight_scale.data.copy_(w13_bs[start:end])
        fml.w2_weight_scale.data.copy_(w2_bs[start:end])

        # weight_scale_2 stores the RECIPROCAL of the global scale: the kernel
        # computes g_alphas = a_scale * weight_scale_2, which needs 1/w_gs for
        # correct dequantization (see tests/kernels/moe/test_moe_layer.py).
        # w13 packs gate and up; replicate for both halves.
        w13_inv_gs = (1.0 / w13_gs[start:end]).unsqueeze(1).expand(-1, 2).contiguous()
        fml.w13_weight_scale_2.data.copy_(w13_inv_gs)
        fml.w2_weight_scale_2.data.copy_(1.0 / w2_gs[start:end])

        # Match the reference's unit input scale.
        fml.w13_input_scale.data.fill_(1.0)
        fml.w2_input_scale.data.fill_(1.0)

        fml.quant_method.process_weights_after_loading(fml)
        fml.maybe_init_modular_kernel()

        with _make_forward_ctx(vllm_config, m):
            output = fml(hidden_states, router_logits)

        torch.testing.assert_close(output, ref_output, **_nvfp4_tol(hidden_size))

    del fml


def _worker_bf16_all_shapes(
    pgi: ProcessGroupInfo,
    vllm_config: VllmConfig,
    cpu_group: ProcessGroup,
    shapes: list[tuple[int, int, int]],
) -> None:
    init_workspace_manager(pgi.device)
    for m, h, i in shapes:
        _run_bf16_one(pgi, vllm_config, m, h, i)
        _cleanup_between_configs()


def _worker_nvfp4_all_shapes(
    pgi: ProcessGroupInfo,
    vllm_config: VllmConfig,
    cpu_group: ProcessGroup,
    shapes: list[tuple[int, int, int]],
) -> None:
    init_workspace_manager(pgi.device)
    for m, h, i in shapes:
        _run_nvfp4_one(pgi, vllm_config, m, h, i)
        _cleanup_between_configs()


def _base_vllm_config(all2all_backend: str) -> VllmConfig:
    cfg = VllmConfig()
    cfg.parallel_config.data_parallel_size = WORLD_SIZE
    cfg.parallel_config.enable_expert_parallel = True
    cfg.parallel_config.is_moe_model = True
    cfg.parallel_config.all2all_backend = all2all_backend
    cfg.compilation_config.fast_moe_cold_start = False
    # Keep moe.max_num_tokens below the DeepEP LL nvshmem_qp_depth bound
    # (default qp_depth=1024 -> assertion needs max_tokens_per_rank <= 511).
    cfg.scheduler_config.max_num_batched_tokens = 128
    return cfg


@pytest.mark.parametrize("backend", BF16_BACKENDS)
@multi_gpu_test(num_gpus=WORLD_SIZE)
def test_fused_moe_ep_dp_bf16(backend: str):
    if torch.accelerator.device_count() < WORLD_SIZE:
        pytest.skip(f"Need {WORLD_SIZE} GPUs, have {torch.accelerator.device_count()}")
    if (
        backend in ("deepep_high_throughput", "deepep_low_latency")
        and not has_deep_ep()
    ):
        pytest.skip("DeepEP not available")

    parallel_launch_with_config(
        WORLD_SIZE,
        _worker_bf16_all_shapes,
        _base_vllm_config(backend),
        None,
        BF16_SHAPES,
    )


@multi_gpu_test(num_gpus=WORLD_SIZE)
def test_fused_moe_ep_dp_nvfp4():
    if not current_platform.has_device_capability(100):
        pytest.skip("NVFP4 + CuTeDSL masked_gemm requires SM100+")
    if torch.accelerator.device_count() < WORLD_SIZE:
        pytest.skip(f"Need {WORLD_SIZE} GPUs, have {torch.accelerator.device_count()}")
    if not has_deep_ep():
        pytest.skip("DeepEP not available")

    for _, h, _ in NVFP4_SHAPES:
        assert h in DeepEPLLPrepareAndFinalize.SUPPORTED_HIDDEN_SIZES, (
            f"hidden_size={h} not in DeepEP LL supported sizes"
        )

    cfg = _base_vllm_config("deepep_low_latency")
    cfg.kernel_config.moe_backend = "flashinfer_cutedsl"
    parallel_launch_with_config(
        WORLD_SIZE, _worker_nvfp4_all_shapes, cfg, None, NVFP4_SHAPES
    )
