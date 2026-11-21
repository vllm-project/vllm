# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Hopper coverage for MiniMax-M3.1's NVFP4 experts, which declare a per-token
FC1 scheme that the SM90 W4A16 backends do not use."""

import pytest
import torch

from tests.kernels.utils import torch_experts
from vllm.config import (
    KernelConfig,
    ParallelConfig,
    VllmConfig,
    set_current_vllm_config,
)
from vllm.forward_context import set_forward_context
from vllm.model_executor.layers.fused_moe import FusedMoEFactory, fused_topk
from vllm.model_executor.layers.fused_moe.experts.marlin_moe import MarlinExperts
from vllm.model_executor.layers.quantization.modelopt import ModelOptMxFp8Config
from vllm.model_executor.layers.quantization.utils.marlin_utils_fp4 import (
    is_fp4_marlin_supported,
)
from vllm.model_executor.layers.quantization.utils.nvfp4_emulation_utils import (
    dequantize_to_dtype,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    kNvfp4DynamicToken,
)
from vllm.platforms import current_platform
from vllm.utils.import_utils import has_humming

if not (current_platform.is_device_capability_family(90) and is_fp4_marlin_supported()):
    pytest.skip("Requires an SM90 GPU with Marlin", allow_module_level=True)

E, N, K, TOPK, M = 8, 256, 512, 2, 8


def _checkpoint() -> dict[str, torch.Tensor]:
    g = torch.Generator(device="cuda").manual_seed(0)

    def packed(*shape):
        return torch.randint(
            0, 256, shape, dtype=torch.uint8, device="cuda", generator=g
        )

    def scales(*shape):
        return torch.ones(*shape, dtype=torch.float8_e4m3fn, device="cuda")

    # One global scale per expert, shared by w1 and w3 as in the checkpoint.
    w13_global = torch.rand(E, 1, device="cuda", generator=g) * 50 + 50
    return {
        "w13_weight_packed": packed(E, 2 * N, K // 2),
        "w13_weight_scale": scales(E, 2 * N, K // 16),
        "w13_weight_global_scale": w13_global.expand(E, 2).contiguous(),
        "w13_input_global_scale": torch.ones(E, 2, device="cuda"),
        "w2_weight_packed": packed(E, K, N // 2),
        "w2_weight_scale": scales(E, K, N // 16),
        "w2_weight_global_scale": torch.rand(E, device="cuda", generator=g) * 50 + 50,
        "w2_input_global_scale": torch.ones(E, device="cuda"),
    }


def _dequant(q, scale, global_scale):
    return dequantize_to_dtype(
        q, scale, 1.0 / global_scale, torch.bfloat16, block_size=16, swizzle=False
    )


@pytest.mark.parametrize("moe_backend", ["humming", "marlin"])
@torch.inference_mode()
def test_per_token_nvfp4_experts_run_as_w4a16(
    dist_init, workspace_init, moe_backend
) -> None:
    if moe_backend == "humming" and not has_humming():
        pytest.skip("Requires humming")
    ckpt = _checkpoint()
    quant_config = ModelOptMxFp8Config.from_config(
        {
            "quant_method": "mxfp8",
            "moe_quant_algo": "NVFP4",
            "moe_quant_format": "nvfp4-pack-quantized",
            "group_size": 16,
            "ignored_layers": [],
        }
    )
    vllm_config = VllmConfig(
        parallel_config=ParallelConfig(),
        kernel_config=KernelConfig(moe_backend=moe_backend),
    )
    with set_current_vllm_config(vllm_config):
        runner = FusedMoEFactory(
            num_experts=E,
            top_k=TOPK,
            hidden_size=K,
            intermediate_size=N,
            params_dtype=torch.bfloat16,
            quant_config=quant_config,
            prefix="model.layers.0.block_sparse_moe.experts",
            qat_activation_key=kNvfp4DynamicToken,
        )
    runner.cuda()
    routed = runner.routed_experts
    for name, tensor in ckpt.items():
        getattr(routed, name).data.copy_(tensor)
    routed.quant_method.process_weights_after_loading(routed)
    moe_kernel = routed.quant_method.moe_kernel
    assert moe_kernel is not None
    experts = moe_kernel.fused_experts
    assert (type(experts) is MarlinExperts) == (moe_backend == "marlin")

    x = torch.randn(M, K, dtype=torch.bfloat16, device="cuda") / 10
    logits = torch.randn(M, E, dtype=torch.float32, device="cuda")
    with set_current_vllm_config(vllm_config), set_forward_context(None, vllm_config):
        out = runner.forward(hidden_states=x, router_logits=logits)

    w13 = torch.empty(E, 2 * N, K, dtype=torch.bfloat16, device="cuda")
    for shard, rows in ((0, slice(0, N)), (1, slice(N, 2 * N))):
        w13[:, rows] = _dequant(
            ckpt["w13_weight_packed"][:, rows],
            ckpt["w13_weight_scale"][:, rows],
            ckpt["w13_weight_global_scale"][:, shard],
        )
    w2 = _dequant(
        ckpt["w2_weight_packed"],
        ckpt["w2_weight_scale"],
        ckpt["w2_weight_global_scale"],
    )
    topk_weights, topk_ids, _ = fused_topk(x, logits, TOPK, renormalize=True)
    expected = torch_experts(x, w13, w2, topk_weights, topk_ids).float()
    assert (out.float() - expected).norm() / expected.norm() < 2e-2
