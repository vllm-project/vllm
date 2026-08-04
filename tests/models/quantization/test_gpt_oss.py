# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
End-to-end accuracy test for GPT-OSS model quantization.

Config:
    Task:   gsm8k_platinum
    Filter: flexible-extract
    n-shot: 5
    Metric: exact_match

Run: pytest tests/models/quantization/test_gpt_oss.py
"""

import importlib.metadata
import importlib.util
from dataclasses import dataclass

import huggingface_hub
import lm_eval
import pytest
import torch
from packaging import version

from vllm._aiter_ops import is_aiter_found_and_supported, rocm_aiter_ops
from vllm.config import get_current_vllm_config
from vllm.forward_context import set_forward_context
from vllm.model_executor.layers.fused_moe.config import (
    mxfp4_w4a8_moe_quant_config,
    mxfp4_w4a16_moe_quant_config,
)
from vllm.model_executor.layers.fused_moe.experts.aiter_mxfp4_w4a8_moe import (
    aiter_triton_kernel_w4a8_moe_forward,
    aiter_triton_kernel_w4a16_moe_forward,
)
from vllm.model_executor.layers.fused_moe.oracle.mxfp4 import (
    Mxfp4MoeBackend,
    convert_gpt_oss_weight_to_mxfp4_moe_kernel_format,
)
from vllm.platforms import current_platform
from vllm.transformers_utils.repo_utils import hf_api
from vllm.utils.torch_utils import set_random_seed

if current_platform.is_rocm():
    from vllm.platforms.rocm import on_gfx950, on_gfx1250
else:

    def on_gfx950() -> bool:
        return False

    def on_gfx1250() -> bool:
        return False


MODEL_ACCURACIES = {
    # Full quantization: attention linears and MoE linears
    "amd/gpt-oss-20b-WFP8-AFP8-KVFP8": 0.89,
    # MoE linears only quantization
    "amd/gpt-oss-20b-MoE-Quant-W-MXFP4-A-FP8-KV-FP8": 0.89,
    # MoE linears only quantization
    # "amd/gpt-oss-20b-MoE-Quant-W-MXFP4-A-MXFP4-KV-FP8": 0.90,
}

QUARK_MXFP4_AVAILABLE = importlib.util.find_spec("quark") is not None and version.parse(
    importlib.metadata.version("amd-quark")
) >= version.parse("0.9.0")


def has_huggingface_access(repo):
    try:
        hf_api().list_repo_refs(repo)
        return True
    except huggingface_hub.errors.RepositoryNotFoundError:
        return False


HF_HUB_AMD_ORG_ACCESS = all(
    [has_huggingface_access(model_name) for model_name in MODEL_ACCURACIES]
)


@dataclass
class ModelCase:
    model_id: str
    tp: int


@dataclass
class EvaluationConfig:
    model_name: str

    def get_model_args(self, tp_size: int):
        return {
            "pretrained": self.model_name,
            "chat_template_args": {"reasoning_effort": "low"},
            "enable_thinking": True,
            "think_end_token": "200008",
            "tensor_parallel_size": tp_size,
            "dtype": "auto",
            "trust_remote_code": False,
            "enable_prefix_caching": False,
            "enforce_eager": False,
        }


@pytest.mark.skipif(not QUARK_MXFP4_AVAILABLE, reason="amd-quark>=0.9 is not available")
@pytest.mark.skipif(
    not HF_HUB_AMD_ORG_ACCESS,
    reason="Read access to huggingface.co/amd is required for this test.",
)
@pytest.mark.parametrize("tp_size", [1, 2, 4, 8])
@pytest.mark.parametrize("model_name, expected_accuracy", MODEL_ACCURACIES.items())
def test_gpt_oss_attention_quantization(
    model_name: str,
    tp_size: int,
    expected_accuracy: float,
    monkeypatch: pytest.MonkeyPatch,
):
    if tp_size > current_platform.device_count():
        pytest.skip("Not enough GPUs to run this test case")

    if "amd/gpt-oss-20b-MoE-Quant-W-MXFP4-A-FP8-KV-FP8" in model_name and on_gfx950():
        monkeypatch.setenv("VLLM_ROCM_USE_AITER", "1")

    model_args = EvaluationConfig(model_name).get_model_args(tp_size)

    extra_run_kwargs = {
        "gen_kwargs": {"max_gen_toks": 8000},
        "apply_chat_template": True,
        "fewshot_as_multiturn": True,
        "num_fewshot": 5,
    }

    lm_eval_out = lm_eval.simple_evaluate(
        model="vllm",
        model_args=model_args,
        tasks="gsm8k_platinum",
        batch_size="auto",
        **extra_run_kwargs,
    )
    measured_accuracy = float(
        lm_eval_out["results"]["gsm8k_platinum"]["exact_match,flexible-extract"]
    )

    rtol = 0.02
    assert measured_accuracy >= expected_accuracy - rtol, (
        f"Accuracy {measured_accuracy:.4f} is below threshold "
        f"{expected_accuracy - rtol:.4f} (expected >= {expected_accuracy} - {rtol})"
    )


@pytest.mark.skipif(
    not (on_gfx950() or on_gfx1250()),
    reason="AITER MXFP4 MoE requires gfx950 or gfx1250",
)
@pytest.mark.skipif(not is_aiter_found_and_supported(), reason="aiter is not installed")
@pytest.mark.skipif(torch.accelerator.device_count() == 0, reason="no gpu available")
@pytest.mark.parametrize(
    "mxfp4_backend",
    [Mxfp4MoeBackend.AITER_MXFP4_FP8, Mxfp4MoeBackend.AITER_MXFP4_BF16],
    ids=["w4a8", "w4a16"],
)
@pytest.mark.parametrize("num_experts", [32, 64, 128])
def test_aiter_mxfp4_moe_ignores_padded_rows(
    mxfp4_backend: Mxfp4MoeBackend,
    num_experts: int,
    monkeypatch: pytest.MonkeyPatch,
    dist_init,
) -> None:
    """
    Garbage in cudagraph padding rows must not reach the unpadded outputs.

    A cudagraph replay of a size-`TOKENS_PADDED` graph driven by
    `TOKENS_UNPADDED` real tokens leaves the trailing padding row containing
    -inf/inf/nan.
    """
    TOKENS_PADDED = 8
    TOKENS_UNPADDED = 7

    # `rocm_aiter_ops` snapshots the environment at import time, so setting the
    # env var alone is not enough.
    monkeypatch.setenv("VLLM_ROCM_USE_AITER", "1")
    rocm_aiter_ops.refresh_env_variables()

    set_random_seed(0)
    device = torch.device("cuda")
    hidden_size = 2048
    intermediate_size = 512

    num_experts = 32
    topk = 4
    is_w4a8 = mxfp4_backend is Mxfp4MoeBackend.AITER_MXFP4_FP8

    # TODO: clean this in oracle/mxfp4.
    # `convert_gpt_oss_weight_to_mxfp4_moe_kernel_format` maps
    # AITER_MXFP4_BF16 onto the CK layout (`shuffle_weight_a16w4`), which the
    # triton monolithic cannot consume -- it wants the `_swizzle_mxfp4`
    # PrecisionConfig that the TRITON branch of the same converter produces.
    weight_backend = (
        Mxfp4MoeBackend.AITER_MXFP4_FP8 if is_w4a8 else Mxfp4MoeBackend.TRITON
    )

    layer = torch.nn.Module()
    layer.w13_weight = torch.randint(
        0,
        256,
        (num_experts, 2 * intermediate_size, hidden_size // 2),
        dtype=torch.uint8,
        device=device,
    )
    layer.w2_weight = torch.randint(
        0,
        256,
        (num_experts, hidden_size, intermediate_size // 2),
        dtype=torch.uint8,
        device=device,
    )

    # Keep the exponents near 127 (2**0) so the dequantized weights stay in a sane
    # range and 255 (e8m0 NaN) is never hit.
    layer.w13_weight_scale = torch.randint(
        124,
        131,
        (num_experts, 2 * intermediate_size, hidden_size // 32),
        dtype=torch.uint8,
        device=device,
    )
    layer.w2_weight_scale = torch.randint(
        124,
        131,
        (num_experts, hidden_size, intermediate_size // 32),
        dtype=torch.uint8,
        device=device,
    )
    layer.w13_bias = torch.randn(
        (num_experts, 2 * intermediate_size), dtype=torch.float32, device=device
    )
    layer.w2_bias = torch.randn(
        (num_experts, hidden_size), dtype=torch.float32, device=device
    )
    # Read off `layer`, not off the arguments, by the AITER_MXFP4_FP8 branch.
    layer.w13_input_scale = torch.full(
        (num_experts,), 0.1, dtype=torch.float32, device=device
    )
    layer.w2_input_scale = torch.full(
        (num_experts,), 0.1, dtype=torch.float32, device=device
    )

    (
        w13_weight,
        w2_weight,
        w13_scale,
        w2_scale,
        w13_bias,
        w2_bias,
    ) = convert_gpt_oss_weight_to_mxfp4_moe_kernel_format(
        weight_backend,
        layer,
        layer.w13_weight,
        layer.w2_weight,
        layer.w13_weight_scale,
        layer.w2_weight_scale,
        w13_bias=layer.w13_bias,
        w2_bias=layer.w2_bias,
    )

    quant_config_factory = (
        mxfp4_w4a8_moe_quant_config if is_w4a8 else mxfp4_w4a16_moe_quant_config
    )
    quant_config = quant_config_factory(
        w1_scale=w13_scale,
        w2_scale=w2_scale,
        w1_bias=w13_bias,
        w2_bias=w2_bias,
    )

    hidden_states = torch.randn(
        (TOKENS_PADDED, hidden_size), dtype=torch.bfloat16, device=device
    )
    gating_output = torch.randn(
        (TOKENS_PADDED, num_experts), dtype=torch.bfloat16, device=device
    )

    hidden_states[TOKENS_UNPADDED:, 0] = float("inf")
    hidden_states[TOKENS_UNPADDED:, 1] = float("-inf")
    hidden_states[TOKENS_UNPADDED:, 2] = float("nan")

    gating_output[TOKENS_UNPADDED:, :] = float("-inf")
    gating_output[TOKENS_UNPADDED:, 0] = float("inf")
    gating_output[TOKENS_UNPADDED:, 1] = float("nan")
    gating_output[TOKENS_UNPADDED:, 2] = -float("nan")

    is_padding = torch.zeros(TOKENS_PADDED, dtype=torch.bool, device=device)
    is_padding[TOKENS_UNPADDED:] = True

    # Poison memory: after `del blocks`, further `torch.empty` in routing with wrongful
    # expert ids may read into these memory sections resulting in
    # potential memory access fault.
    blocks = []
    for numel in (16, 32, 64, 128, 256, 512, 1024, 4096, 1 << 14, 1 << 16, 1 << 20):
        blocks.append(torch.full((numel,), 100e7, dtype=torch.int32, device=device))
    # return blocks
    del blocks
    torch.accelerator.synchronize()

    with set_forward_context(None, get_current_vllm_config(), is_padding=is_padding):
        if is_w4a8:
            # AITER_MXFP4_FP8 routes through the triton monolithic expert, which
            # does its own top-k inside `aiter.ops.triton.moe.moe_routing`.
            output = aiter_triton_kernel_w4a8_moe_forward(
                hidden_states=hidden_states,
                w1=w13_weight,
                w2=w2_weight,
                gating_output=gating_output,
                topk=topk,
                renormalize=True,
                quant_config=quant_config,
                global_num_experts=num_experts,
                unpadded_N_w1=2 * intermediate_size,
                unpadded_K_w1=hidden_size,
                unpadded_N_w2=hidden_size,
                unpadded_K_w2=intermediate_size,
            )
        else:
            output = aiter_triton_kernel_w4a16_moe_forward(
                hidden_states=hidden_states,
                w1=w13_weight,
                w2=w2_weight,
                gating_output=gating_output,
                topk=topk,
                renormalize=True,
                quant_config=quant_config,
                global_num_experts=num_experts,
                unpadded_N_w1=2 * intermediate_size,
                unpadded_K_w1=hidden_size,
                unpadded_N_w2=hidden_size,
                unpadded_K_w2=intermediate_size,
            )

    assert torch.isfinite(output[:TOKENS_UNPADDED]).all()
