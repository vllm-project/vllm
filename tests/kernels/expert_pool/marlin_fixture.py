# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Small NVFP4 Marlin MoE layers for the expert pool CUDA tests."""

import pytest
import torch

from tests.kernels.moe.modular_kernel_tools.parallel_utils import _set_vllm_config
from tests.kernels.moe.utils import _scaled_fp4_quant_emulated
from vllm.config import (
    CompilationConfig,
    ParallelConfig,
    VllmConfig,
    set_current_vllm_config,
)
from vllm.model_executor.layers.fused_moe import FusedMoEFactory
from vllm.model_executor.layers.quantization.modelopt import ModelOptNvFp4Config
from vllm.utils.torch_utils import set_random_seed
from vllm.v1.worker.workspace import (
    init_workspace_manager,
    is_workspace_manager_initialized,
)

E, K, N, TOP_K, M = 8, 256, 128, 2, 8

EXPERT_TENSORS = ("w13_weight", "w2_weight", "w13_weight_scale", "w2_weight_scale")


def vllm_config(pool_rows: int) -> VllmConfig:
    cfg = VllmConfig(
        parallel_config=ParallelConfig(), compilation_config=CompilationConfig()
    )
    cfg.kernel_config.moe_backend = "marlin"
    cfg.offload_config.moe_expert_pool_rows = pool_rows
    return cfg


@pytest.fixture(scope="module")
def dist_env():
    cfg = vllm_config(0)
    _set_vllm_config(cfg, 1, rank=0, local_rank=0)
    if not is_workspace_manager_initialized():
        init_workspace_manager(torch.accelerator.current_accelerator())
    return cfg


def _quantize_row_major(w: torch.Tensor):
    qs, ss, gs = [], [], []
    for i in range(w.shape[0]):
        amax = w[i].abs().max().to(torch.float32)
        g = torch.tensor(448.0 * 6.0, device=w.device) / amax
        q, s = _scaled_fp4_quant_emulated(w[i], g)
        qs.append(q)
        ss.append(s)
        gs.append(g)
    return torch.stack(qs), torch.stack(ss), torch.stack(gs)


def quantized_weights(device, n: int = N, seed_offset: int = 0):
    set_random_seed(11 + seed_offset)
    w1 = torch.randn(E, 2 * n, K, dtype=torch.bfloat16, device=device)
    w2 = torch.randn(E, K, n, dtype=torch.bfloat16, device=device)
    # Distinct per-expert magnitudes so the global scales differ per expert.
    mag = torch.tensor([0.5 + i for i in range(E)], device=device).view(E, 1, 1)
    w1 = (w1 * mag).to(torch.bfloat16)
    w2 = (w2 * mag.flip(0)).to(torch.bfloat16)
    # Row-major [E, rows, K/16] block scales as a checkpoint stores them. The
    # CUDA quant op would return the swizzled 128x4 layout padded to 128 rows,
    # which is neither the checkpoint layout nor what Marlin's permute reads.
    w1q, w1s, w1gs = _quantize_row_major(w1)
    w2q, w2s, w2gs = _quantize_row_major(w2)
    params = {
        "w13_weight": w1q,
        "w2_weight": w2q,
        "w13_weight_scale": w1s,
        "w2_weight_scale": w2s,
        "w13_weight_scale_2": (1.0 / w1gs).unsqueeze(1).expand(-1, 2).contiguous(),
        "w2_weight_scale_2": 1.0 / w2gs,
        "w13_input_scale": torch.ones((E, 2), dtype=torch.float32, device=device),
        "w2_input_scale": torch.ones(E, dtype=torch.float32, device=device),
    }
    assert params["w13_weight_scale"].shape == (E, 2 * n, K // 16)
    assert params["w2_weight_scale"].shape == (E, K, n // 16)
    assert torch.unique(params["w13_weight_scale_2"][:, 0]).numel() == E
    assert torch.unique(params["w2_weight_scale_2"]).numel() == E
    return params


def make_layer(
    cfg: VllmConfig, params: dict[str, torch.Tensor], host_source: bool = False
):
    """Build the layer; with host_source the per-expert tensors are
    registered as pinned CPU tensors, the layout the loader restores when
    the pool is enabled."""
    with set_current_vllm_config(cfg):
        # Any construction error is a failure: the Marlin capability gate is
        # the module-level skip in the test, and the backend is pinned.
        layer = FusedMoEFactory(
            num_experts=E,
            top_k=TOP_K,
            hidden_size=K,
            intermediate_size=N,
            params_dtype=torch.bfloat16,
            renormalize=False,
            quant_config=ModelOptNvFp4Config(
                is_checkpoint_nvfp4_serialized=True,
                kv_cache_quant_algo=None,
                exclude_modules=[],
            ),
            tp_size=1,
            dp_size=1,
            prefix="from_forward_context",
        )
        if cfg.offload_config.moe_expert_pool_rows > 0:
            # create_weights wiring: expert tensors start in pinned host memory.
            for name in EXPERT_TENSORS:
                p = getattr(layer.routed_experts, name)
                assert p.device.type == "cpu" and p.is_pinned(), name
        for name, value in params.items():
            data = value.clone()
            if host_source and name in EXPERT_TENSORS:
                data = data.cpu().pin_memory()
            layer.routed_experts.register_parameter(
                name, torch.nn.Parameter(data, requires_grad=False)
            )
        layer._quant_method.process_weights_after_loading(layer.routed_experts)
    return layer


def routing(order: list[int], device) -> torch.Tensor:
    # Token t routes to experts (order[t], order[(t + 1) % E]); every forward
    # touches all E experts, more than the pool holds.
    logits = torch.full((M, E), -10.0, device=device)
    for t in range(M):
        logits[t, order[t % E]] = 3.0
        logits[t, order[(t + 1) % E]] = 2.0
    return logits
