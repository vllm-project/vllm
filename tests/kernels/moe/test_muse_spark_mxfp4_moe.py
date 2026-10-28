# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Humming and Marlin coverage for MuseSpark's compressed-tensors MXFP4 experts."""

import pytest
import torch

from tests.kernels.moe.muse_spark_utils import (
    assert_cudagraph_replays,
    make_runner,
    run_moe,
    swiglu_moe_ref,
)
from vllm.model_executor.layers.fused_moe.experts.fused_humming_moe import (
    HummingGroupedExperts,
    HummingIndexedExperts,
)
from vllm.model_executor.layers.fused_moe.experts.marlin_moe import MarlinExperts
from vllm.model_executor.layers.fused_moe.oracle import mxfp4 as mxfp4_oracle
from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors import (  # noqa: E501
    CompressedTensorsConfig,
)
from vllm.model_executor.layers.quantization.utils.marlin_utils_fp4 import (
    is_fp4_marlin_supported,
)
from vllm.model_executor.layers.quantization.utils.nvfp4_emulation_utils import (
    break_fp4_bytes,
)
from vllm.models.muse_spark.common.layers import MuseSpark1xRMSNormWithInputScale
from vllm.models.muse_spark.nvidia.muse_moe_runner import (
    MuseSparkCompressedTensorsMxfp4FusedMoE,
)
from vllm.platforms import current_platform
from vllm.utils.import_utils import has_humming

if not current_platform.is_cuda():
    pytest.skip("Requires a CUDA GPU", allow_module_level=True)

# (moe_backend, experts class, checkpoint declares FP8 activations). Humming
# applies the block-FP8 activation scheme, which it only offers on SM90+;
# Marlin is W4A16 and ignores the declared activations.
BACKENDS = [
    pytest.param(
        "humming",
        HummingIndexedExperts,
        False,
        marks=pytest.mark.skipif(not has_humming(), reason="needs Humming"),
        id="humming-w4a16",
    ),
    pytest.param(
        "humming",
        HummingIndexedExperts,
        True,
        marks=pytest.mark.skipif(
            not (has_humming() and current_platform.has_device_capability(90)),
            reason="Humming MXFP4 with block-FP8 activations needs SM90+",
        ),
        id="humming-w4a8",
    ),
    pytest.param(
        "humming-grouped",
        HummingGroupedExperts,
        False,
        marks=pytest.mark.skipif(not has_humming(), reason="needs Humming"),
        id="humming-grouped-w4a16",
    ),
    pytest.param(
        "marlin",
        MarlinExperts,
        True,
        marks=pytest.mark.skipif(
            not is_fp4_marlin_supported(), reason="needs FP4 Marlin"
        ),
        id="marlin-w4a16",
    ),
]

E, N, K, TOPK, M = 8, 256, 512, 2, 8
GROUP = 32
PROJECTIONS = (("gate_proj", N, K), ("up_proj", N, K), ("down_proj", K, N))


def _quant_config(fp8_activations: bool) -> CompressedTensorsConfig:
    """MXFP4 experts, optionally with the dynamic FP8 group activations
    llm-compressor emits."""
    input_activations = None
    if fp8_activations:
        input_activations = {
            "dynamic": True,
            "group_size": 128,
            "num_bits": 8,
            "strategy": "group",
            "symmetric": True,
            "type": "float",
        }
    return CompressedTensorsConfig.from_config(
        {
            "config_groups": {
                "experts": {
                    "format": "mxfp4-pack-quantized",
                    "input_activations": input_activations,
                    "targets": [r"re:.*experts\.\d+\.(gate|up|down)_proj$"],
                    "weights": {
                        "dynamic": False,
                        "group_size": GROUP,
                        "num_bits": 4,
                        "strategy": "group",
                        "symmetric": True,
                        "type": "float",
                    },
                }
            },
            "format": "mixed-precision",
            "ignore": [],
            "quant_method": "compressed-tensors",
        }
    )


def _mxfp4_checkpoint() -> dict[str, torch.Tensor]:
    g = torch.Generator(device="cuda").manual_seed(0)

    def randint(low: int, high: int, *shape: int) -> torch.Tensor:
        return torch.randint(
            low, high, shape, dtype=torch.uint8, device="cuda", generator=g
        )

    checkpoint = {}
    for expert in range(E):
        for name, rows, cols in PROJECTIONS:
            prefix = f"{expert}.{name}"
            checkpoint[f"{prefix}.weight_packed"] = randint(0, 256, rows, cols // 2)
            # e8m0 exponents 2^-7 .. 2^-2 keep the SwiGLU inputs O(1).
            checkpoint[f"{prefix}.weight_scale"] = randint(
                120, 126, rows, cols // GROUP
            )
    return checkpoint


def _dequant(packed: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    values = break_fp4_bytes(packed, torch.float32).view(*scale.shape, GROUP)
    values = values * torch.exp2(scale.float() - 127)[..., None]
    return values.view(scale.shape[0], -1).to(torch.bfloat16)


def _dequantized_experts(checkpoint):
    def weight(expert: int, name: str) -> torch.Tensor:
        prefix = f"{expert}.{name}"
        return _dequant(
            checkpoint[f"{prefix}.weight_packed"], checkpoint[f"{prefix}.weight_scale"]
        )

    w13 = torch.stack(
        [
            torch.cat([weight(expert, "gate_proj"), weight(expert, "up_proj")])
            for expert in range(E)
        ]
    )
    w2 = torch.stack([weight(expert, "down_proj") for expert in range(E)])
    return w13, w2


@pytest.mark.parametrize(("moe_backend", "experts_cls", "fp8_activations"), BACKENDS)
@torch.inference_mode()
def test_mxfp4_moe_runs_with_post_expert_finalize(
    dist_init,
    workspace_init,
    monkeypatch,
    moe_backend: str,
    experts_cls: type,
    fp8_activations: bool,
) -> None:
    if moe_backend == "humming-grouped":
        # Grouped is the EP default; force it to cover the permuted-row hook.
        monkeypatch.setenv("VLLM_HUMMING_MOE_GEMM_TYPE", "grouped")
        moe_backend = "humming"
    # The bare test VllmConfig has no model_config to read an override from.
    monkeypatch.setattr(mxfp4_oracle, "_user_moe_activation_override", lambda: None)
    norm = MuseSpark1xRMSNormWithInputScale(
        K,
        eps=1e-8,
        has_weight=True,
        is_post_norm=True,
        zero_centered_gamma=True,
    ).cuda()
    norm.weight.copy_(torch.linspace(-0.2, 0.2, K, device="cuda"))
    expert_bias = torch.linspace(-0.5, 0.5, E, dtype=torch.float32, device="cuda")
    checkpoint = _mxfp4_checkpoint()
    router, runner, vllm_config = make_runner(
        norm,
        expert_bias,
        checkpoint,
        _quant_config(fp8_activations),
        hidden_size=K,
        intermediate_size=N,
        top_k=TOPK,
        moe_backend=moe_backend,
    )
    quant_method = runner.routed_experts.quant_method
    assert isinstance(quant_method, MuseSparkCompressedTensorsMxfp4FusedMoE)
    assert type(quant_method.moe_kernel.fused_experts) is experts_cls

    hidden_states = torch.randn(M, K, dtype=torch.bfloat16, device="cuda") / 10
    router_logits = torch.randn(M, E, dtype=torch.float32, device="cuda")
    output = run_moe(runner, vllm_config, hidden_states, router_logits)
    assert output.shape == hidden_states.shape
    assert output.dtype == hidden_states.dtype
    assert torch.isfinite(output).all()

    w13, w2 = _dequantized_experts(checkpoint)
    topk_weights, topk_ids = router._compute_routing(hidden_states, router_logits, None)
    expected = swiglu_moe_ref(
        hidden_states,
        w13,
        w2,
        topk_weights,
        topk_ids,
        lambda y: norm.forward_native(y, out_dtype=torch.float32),
    )
    assert (output.float() - expected).norm() / expected.norm() < 5e-2

    assert_cudagraph_replays(
        runner,
        vllm_config,
        hidden_states,
        router_logits,
        torch.randn_like(hidden_states),
        torch.randn_like(router_logits),
    )
