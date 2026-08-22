# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Callable

import pytest

from vllm.config import PassConfig
from vllm.platforms import current_platform

from ...utils import multi_gpu_test
from .common import (
    INDUCTOR_GRAPH_PARTITION,
    AttentionBackendCase,
    Matches,
    custom_ops_combos,
    is_blackwell,
)
from .models import (
    FLASHINFER_ATTN,
    FLASHINFER_MLA_ATTN,
    ROCM_AITER_MLA_ATTN,
    ROCM_AITER_UNIFIED_ATTN,
    ROCM_ATTN,
    TRITON_ATTN,
    deepseek_coder_v2_lite_fp8,
    deepseek_r1_fp4,
    deepseek_v3_fp8,
    gpt_oss_20b,
    llama3_8b,
    llama3_8b_fp4,
    llama3_8b_fp8,
    llama4_scout_fp4,
    llama4_scout_fp8,
    qwen3_a3b,
    qwen3_a3b_fp8,
)

pytestmark = pytest.mark.skipif(
    not current_platform.is_cuda_alike(), reason="Only test CUDA/ROCm"
)

TP2_FP8_MODELS = (
    [
        llama3_8b_fp8,
        llama4_scout_fp8,
        qwen3_a3b_fp8,
        deepseek_coder_v2_lite_fp8,
        deepseek_v3_fp8,
    ]
    if current_platform.is_cuda()
    else [
        llama3_8b_fp8,
        qwen3_a3b_fp8,
        deepseek_coder_v2_lite_fp8,
    ]
)

TP2_FP8_ATTN_BACKENDS = (
    [TRITON_ATTN, FLASHINFER_ATTN, FLASHINFER_MLA_ATTN]
    if current_platform.is_cuda()
    else [ROCM_AITER_UNIFIED_ATTN, ROCM_AITER_MLA_ATTN]
)


@multi_gpu_test(num_gpus=2)
@pytest.mark.parametrize(
    "model_name, matches_fn, model_kwargs, hf_overrides",
    # Platform-specific lists above retain CUDA's full matrix while adding
    # only ROCm payloads supported by the selected attention backends.
    TP2_FP8_MODELS,
)
@pytest.mark.parametrize("attn_backend", TP2_FP8_ATTN_BACKENDS)
@pytest.mark.parametrize("n_layers", [4])
@pytest.mark.parametrize("custom_ops", custom_ops_combos("quant_fp8", "rms_norm"))
@pytest.mark.parametrize("inductor_graph_partition", INDUCTOR_GRAPH_PARTITION)
@pytest.mark.skipif(not current_platform.is_cuda_alike(), reason="Only test CUDA/ROCm")
def test_tp2_ar_rms_fp8_fusions(
    model_name: str,
    matches_fn: Callable[[int], Matches],
    model_kwargs: dict,
    hf_overrides: Callable[[int], dict],
    attn_backend: AttentionBackendCase,
    n_layers: int,
    custom_ops: str,
    inductor_graph_partition: bool,
    run_e2e_fusion_test,
    monkeypatch,
):
    matches = matches_fn(n_layers)

    model_name_lower = model_name.lower()
    rocm_static_fp8_mla = (
        current_platform.is_rocm()
        and "deepseek-coder" in model_name_lower
        and attn_backend.backend.name == "ROCM_AITER_MLA"
    )
    # Preserve CUDA's existing treatment of all DeepSeek payloads. The ROCm
    # DeepSeek-Coder checkpoint is the static-FP8 exception validated with the
    # native QuantFP8 MLA pattern.
    block_fp8 = "qwen" in model_name_lower or "deepseek" in model_name_lower
    if rocm_static_fp8_mla:
        block_fp8 = False
        if custom_ops != "-quant_fp8,-rms_norm":
            pytest.skip("ROCm static-FP8 MLA parity uses native quant and RMSNorm")
        # AITER's MoE GEMM does not support this reduced dummy model shape;
        # use the supported Triton MoE path while retaining AITER MLA and AR.
        monkeypatch.setenv("VLLM_ROCM_USE_AITER_MOE", "0")
    if block_fp8 and "-quant_fp8" in custom_ops:
        # This is why config forces +quant_fp8 by default
        pytest.skip("native QuantFP8 matching not supported for group quant")

    # Reduce size of model and skip weight loading time
    model_kwargs["hf_overrides"] = hf_overrides(n_layers)
    model_kwargs["load_format"] = "dummy"
    model_kwargs["max_model_len"] = 1024
    model_kwargs["kernel_config"] = {"enable_flashinfer_autotune": False}
    model_kwargs["disable_custom_all_reduce"] = False

    compilation_config = dict(
        use_inductor_graph_partition=inductor_graph_partition,
        custom_ops=custom_ops.split(","),
        pass_config=PassConfig(
            fuse_norm_quant=True,
            fuse_act_quant=True,
            fuse_attn_quant=True,
            enable_qk_norm_rope_fusion=True,
            fuse_allreduce_rms=True,
        ),
    )

    matches_check = [
        "rms_quant_fusion",
        "act_quant_fusion",
        "norm_rope_fusion",
        "attn_quant_fusion",
        "ar_rms_fusion",
    ]

    use_aiter = current_platform.is_rocm()
    if use_aiter:
        matches = matches._replace(aiter_ar_rms_fusion=matches.ar_rms_fusion)
        # The AITER all-reduce pass registers the larger AR+RMS(+group-quant)
        # patterns before its AR+RMS-only patterns. Consequently, the
        # standalone AITER RMS-quant pass does not own the post-AR sites in
        # this test. ROCm's supported contract here is native QK-norm+RoPE and
        # attention-output quant fusion plus AITER AR+RMS; CUDA keeps the full
        # checks above.
        matches_check = [
            "norm_rope_fusion",
            "attn_quant_fusion",
            "aiter_ar_rms_fusion",
        ]

    run_e2e_fusion_test(
        model_name,
        matches,
        model_kwargs,
        attn_backend,
        compilation_config,
        matches_check,
        tp_size=2,
        use_aiter=use_aiter,
    )


@multi_gpu_test(num_gpus=2)
@pytest.mark.parametrize(
    "model_name, matches_fn, model_kwargs, hf_overrides",
    [llama3_8b_fp4, llama4_scout_fp4, deepseek_r1_fp4],
)
@pytest.mark.parametrize(
    "attn_backend",
    [FLASHINFER_ATTN, FLASHINFER_MLA_ATTN],
)
@pytest.mark.parametrize("n_layers", [4])
@pytest.mark.parametrize("custom_ops", custom_ops_combos("rms_norm"))
@pytest.mark.parametrize("inductor_graph_partition", INDUCTOR_GRAPH_PARTITION)
@pytest.mark.skipif(not is_blackwell(), reason="Blackwell required for fp4")
@pytest.mark.skipif(not current_platform.is_cuda(), reason="Only test CUDA")
def test_tp2_ar_rms_fp4_fusions(
    model_name: str,
    matches_fn: Callable[[int], Matches],
    model_kwargs: dict,
    hf_overrides: Callable[[int], dict],
    attn_backend: AttentionBackendCase,
    n_layers: int,
    custom_ops: str,
    inductor_graph_partition: bool,
    run_e2e_fusion_test,
    monkeypatch,
):
    matches = matches_fn(n_layers)

    # Reduce size of model and skip weight loading time
    model_kwargs["hf_overrides"] = hf_overrides(n_layers)
    model_kwargs["load_format"] = "dummy"
    model_kwargs["max_model_len"] = 1024
    model_kwargs["kernel_config"] = {"enable_flashinfer_autotune": False}
    model_kwargs["disable_custom_all_reduce"] = False

    compilation_config = dict(
        use_inductor_graph_partition=inductor_graph_partition,
        custom_ops=custom_ops.split(","),
        pass_config=PassConfig(
            fuse_act_quant=True,
            fuse_attn_quant=True,
            fuse_allreduce_rms=True,
        ),
    )

    matches_check = [
        "act_quant_fusion",
        "attn_quant_fusion",
        "ar_rms_fusion",
    ]

    run_e2e_fusion_test(
        model_name,
        matches,
        model_kwargs,
        attn_backend,
        compilation_config,
        matches_check,
        tp_size=2,
    )


@multi_gpu_test(num_gpus=2)
@pytest.mark.parametrize(
    "model_name, matches_fn, model_kwargs, hf_overrides, model_impl",
    [
        (*llama3_8b, "auto"),
        (*llama3_8b, "transformers"),
        (*qwen3_a3b, "auto"),
        (*gpt_oss_20b, "auto"),
    ],
)
@pytest.mark.parametrize(
    "attn_backend",
    [
        TRITON_ATTN,
        FLASHINFER_ATTN,
        ROCM_ATTN,
        ROCM_AITER_UNIFIED_ATTN,
    ],
)
@pytest.mark.parametrize("n_layers", [4])
@pytest.mark.parametrize("custom_ops", tuple(custom_ops_combos("rms_norm")))
@pytest.mark.parametrize("inductor_graph_partition", INDUCTOR_GRAPH_PARTITION)
@pytest.mark.skipif(not current_platform.is_cuda_alike(), reason="Only test CUDA/ROCm")
def test_tp2_ar_rms_fusions(
    model_name: str,
    matches_fn: Callable[[int], Matches],
    model_kwargs: dict,
    hf_overrides: Callable[[int], dict],
    model_impl: str,
    attn_backend: AttentionBackendCase,
    n_layers: int,
    custom_ops: str,
    inductor_graph_partition: bool,
    run_e2e_fusion_test,
):
    if model_impl == "transformers" and not current_platform.is_rocm():
        pytest.skip("Transformers 3D AR+RMS regression is ROCm-only")

    matches = matches_fn(n_layers)
    if model_impl == "transformers":
        # Transformers add+RMSNorm canonicalization exposes every generic
        # AR+RMS fusion site, including the final norm.
        matches = matches._replace(aiter_ar_rms_fusion=matches.ar_rms_fusion)

    # Reduce size of model and skip weight loading time
    model_kwargs["hf_overrides"] = hf_overrides(n_layers)
    model_kwargs["load_format"] = "dummy"
    model_kwargs["model_impl"] = model_impl
    model_kwargs["max_model_len"] = 1024
    model_kwargs["kernel_config"] = {"enable_flashinfer_autotune": False}
    model_kwargs["disable_custom_all_reduce"] = False

    compilation_config = dict(
        use_inductor_graph_partition=inductor_graph_partition,
        custom_ops=custom_ops.split(","),
        pass_config=PassConfig(
            enable_qk_norm_rope_fusion=True,
            fuse_allreduce_rms=True,
        ),
    )

    matches_check = [
        "norm_rope_fusion",
    ]

    if current_platform.is_rocm():
        matches_check.append("aiter_ar_rms_fusion")
    else:
        matches_check.append("ar_rms_fusion")

    run_e2e_fusion_test(
        model_name,
        matches,
        model_kwargs,
        attn_backend,
        compilation_config,
        matches_check,
        tp_size=2,
        use_aiter=current_platform.is_rocm(),
    )
