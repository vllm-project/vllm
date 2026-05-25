# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Correctness tests for torch.compile fusion passes.

Compares model outputs between eager mode (baseline) and VLLM_COMPILE mode
with fusion passes enabled. Uses few-layer models with dummy weights to keep
tests fast while verifying that fusion passes don't alter model outputs
beyond acceptable numerical tolerance.

Complements the existing fusion *count* tests (test_tp1_quant.py etc.) which
verify that fusions fire the expected number of times but don't check outputs.
"""

import json
from typing import NamedTuple

import pytest

from vllm.config.compilation import CompilationMode
from vllm.platforms import current_platform

from ...utils import compare_two_settings, create_new_process_for_each_test
from .common import is_blackwell

N_LAYERS = 4


class FusionCorrectnessCase(NamedTuple):
    model_name: str
    hf_overrides: dict
    attn_backend: str
    pass_config: dict
    custom_ops: list[str]
    extra_args: list[str]


# ---------------------------------------------------------------------------
# FP8 models (Hopper+)
# ---------------------------------------------------------------------------

LLAMA3_8B_FP8 = FusionCorrectnessCase(
    model_name="RedHatAI/Meta-Llama-3.1-8B-Instruct-FP8",
    hf_overrides={"num_hidden_layers": N_LAYERS},
    attn_backend="TRITON_ATTN",
    pass_config={
        "fuse_norm_quant": True,
        "fuse_act_quant": True,
        "fuse_attn_quant": True,
    },
    custom_ops=["+quant_fp8", "+rms_norm"],
    extra_args=[],
)

LLAMA4_SCOUT_FP8 = FusionCorrectnessCase(
    model_name="nvidia/Llama-4-Scout-17B-16E-Instruct-FP8",
    hf_overrides={"text_config": {"num_hidden_layers": N_LAYERS}},
    attn_backend="TRITON_ATTN",
    pass_config={
        "fuse_norm_quant": True,
        "fuse_attn_quant": True,
    },
    custom_ops=["+quant_fp8", "+rms_norm"],
    extra_args=[],
)

QWEN3_A3B_FP8 = FusionCorrectnessCase(
    model_name="Qwen/Qwen3-30B-A3B-FP8",
    hf_overrides={"num_hidden_layers": N_LAYERS},
    attn_backend="TRITON_ATTN",
    pass_config={
        "fuse_norm_quant": True,
        "enable_qk_norm_rope_fusion": True,
    },
    custom_ops=["+quant_fp8", "+rms_norm"],
    extra_args=[],
)

DEEPSEEK_CODER_V2_LITE_FP8 = FusionCorrectnessCase(
    model_name="RedHatAI/DeepSeek-Coder-V2-Lite-Instruct-FP8",
    hf_overrides={"num_hidden_layers": N_LAYERS},
    attn_backend="TRITON_MLA",
    pass_config={
        "fuse_norm_quant": True,
        "fuse_act_quant": True,
        "fuse_attn_quant": True,
    },
    custom_ops=["+quant_fp8", "+rms_norm"],
    extra_args=[],
)

DEEPSEEK_V3_FP8 = FusionCorrectnessCase(
    model_name="deepseek-ai/DeepSeek-V3",
    hf_overrides={"num_hidden_layers": N_LAYERS},
    attn_backend="TRITON_MLA",
    pass_config={
        "fuse_norm_quant": True,
        "fuse_act_quant": True,
        "fuse_attn_quant": True,
    },
    custom_ops=["+quant_fp8", "+rms_norm"],
    extra_args=[],
)

# ---------------------------------------------------------------------------
# FP4 models (Blackwell only)
# ---------------------------------------------------------------------------

LLAMA3_8B_FP4 = FusionCorrectnessCase(
    model_name="nvidia/Llama-3.1-8B-Instruct-FP4",
    hf_overrides={"num_hidden_layers": N_LAYERS},
    attn_backend="FLASHINFER",
    pass_config={
        "fuse_act_quant": True,
        "fuse_attn_quant": True,
        "enable_qk_norm_rope_fusion": True,
    },
    custom_ops=["+rms_norm"],
    extra_args=["--kv-cache-dtype", "fp8"],
)

DEEPSEEK_R1_FP4 = FusionCorrectnessCase(
    model_name="nvidia/DeepSeek-R1-0528-NVFP4-v2",
    hf_overrides={"num_hidden_layers": N_LAYERS},
    attn_backend="FLASHINFER_MLA",
    pass_config={
        "fuse_act_quant": True,
        "fuse_attn_quant": True,
    },
    custom_ops=["+rms_norm"],
    extra_args=[],
)


def _run_fusion_correctness(case: FusionCorrectnessCase):
    """Run a single fusion correctness comparison (fused vs eager baseline)."""
    common_args = [
        "--dtype",
        "bfloat16",
        "--max-model-len",
        "1024",
        "--max-num-seqs",
        "8",
        "--load-format",
        "dummy",
        "--gpu-memory-utilization",
        "0.8",
        "--hf-overrides",
        json.dumps(case.hf_overrides),
        f"--attention-backend={case.attn_backend}",
        "--tensor-parallel-size",
        "1",
        "-cc.cudagraph_mode=none",
        *case.extra_args,
    ]

    baseline_args = [*common_args, "--enforce-eager"]

    compilation_config = {
        "mode": CompilationMode.VLLM_COMPILE,
        "splitting_ops": [],
        "custom_ops": case.custom_ops,
        "pass_config": case.pass_config,
    }
    fused_args = [
        *common_args,
        "--compilation_config",
        json.dumps(compilation_config),
    ]

    compare_two_settings(
        case.model_name,
        fused_args,
        baseline_args,
        method="generate_close",
    )


@pytest.mark.parametrize(
    "case",
    [
        pytest.param(LLAMA3_8B_FP8, id="llama3_8b_fp8"),
        pytest.param(
            LLAMA4_SCOUT_FP8,
            id="llama4_scout_fp8",
            marks=pytest.mark.skipif(
                not current_platform.is_cuda(),
                reason="Llama4 Scout FP8 only supported on CUDA",
            ),
        ),
        pytest.param(QWEN3_A3B_FP8, id="qwen3_a3b_fp8"),
        pytest.param(
            DEEPSEEK_CODER_V2_LITE_FP8,
            id="deepseek_coder_v2_lite_fp8",
        ),
        pytest.param(DEEPSEEK_V3_FP8, id="deepseek_v3_fp8"),
    ],
)
@pytest.mark.skipif(
    not current_platform.is_cuda() and not current_platform.is_rocm(),
    reason="Fusion correctness tests require CUDA or ROCm",
)
@create_new_process_for_each_test()
def test_fp8_fusion_correctness_tp1(case: FusionCorrectnessCase):
    """Verify FP8 fused outputs match eager baseline for TP=1 models."""
    _run_fusion_correctness(case)


@pytest.mark.parametrize(
    "case",
    [
        pytest.param(LLAMA3_8B_FP4, id="llama3_8b_fp4"),
        pytest.param(DEEPSEEK_R1_FP4, id="deepseek_r1_fp4"),
    ],
)
@pytest.mark.skipif(not is_blackwell(), reason="FP4 requires Blackwell")
@create_new_process_for_each_test()
def test_fp4_fusion_correctness_tp1(case: FusionCorrectnessCase):
    """Verify FP4 fused outputs match eager baseline for TP=1 models."""
    _run_fusion_correctness(case)
