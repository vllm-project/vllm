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
    ROCM_AITER_UNIFIED_ATTN,
    ROCM_ATTN,
    TRITON_ATTN,
    llama3_8b,
    llama3_8b_fp8,
    llama4_scout_fp8,
    qwen3_a3b,
)


@multi_gpu_test(num_gpus=2)
@pytest.mark.parametrize(
    "model_name, matches_fn, model_kwargs, hf_overrides",
    [llama3_8b_fp8, llama4_scout_fp8],
)
@pytest.mark.parametrize(
    "attn_backend",
    [TRITON_ATTN, FLASHINFER_ATTN]
    if current_platform.is_cuda()
    else [TRITON_ATTN, ROCM_ATTN, ROCM_AITER_UNIFIED_ATTN],
)
@pytest.mark.parametrize("n_layers", [4])
@pytest.mark.parametrize("custom_ops", list(custom_ops_combos("quant_fp8", "rms_norm")))
@pytest.mark.parametrize("inductor_graph_partition", INDUCTOR_GRAPH_PARTITION)
def test_tp2_async_tp_fp8_fusions(
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

    if is_blackwell():
        # Disable FlashInfer scaled_mm FP8 as it's not supported in async tp patterns
        monkeypatch.setenv("VLLM_DISABLED_KERNELS", "FlashInferFP8ScaledMMLinearKernel")

    # Reduce size of model and skip weight loading time
    model_kwargs["hf_overrides"] = hf_overrides(n_layers)
    model_kwargs["load_format"] = "dummy"
    model_kwargs["max_model_len"] = 1024

    compilation_config = dict(
        use_inductor_graph_partition=inductor_graph_partition,
        custom_ops=custom_ops.split(","),
        pass_config=PassConfig(
            fuse_norm_quant=True,
            fuse_act_quant=True,
            fuse_attn_quant=True,
            enable_qk_norm_rope_fusion=True,
            enable_sp=True,
            fuse_gemm_comms=True,
        ),
    )

    matches_check = [
        "rms_quant_fusion",
        "act_quant_fusion",
        "norm_rope_fusion",
        "attn_quant_fusion",
        "sequence_parallel",
        "async_tp",
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
    "model_name, matches_fn, model_kwargs, hf_overrides",
    [llama3_8b, qwen3_a3b],
)
@pytest.mark.parametrize("attn_backend", [TRITON_ATTN])
@pytest.mark.parametrize("n_layers", [4])
@pytest.mark.parametrize("custom_ops", list(custom_ops_combos("rms_norm")))
@pytest.mark.parametrize("inductor_graph_partition", INDUCTOR_GRAPH_PARTITION)
def test_tp2_async_tp_fusions(
    model_name: str,
    matches_fn: Callable[[int], Matches],
    model_kwargs: dict,
    hf_overrides: Callable[[int], dict],
    attn_backend: AttentionBackendCase,
    n_layers: int,
    custom_ops: str,
    inductor_graph_partition: bool,
    run_e2e_fusion_test,
):
    matches = matches_fn(n_layers)

    # Reduce size of model and skip weight loading time
    model_kwargs["hf_overrides"] = hf_overrides(n_layers)
    model_kwargs["load_format"] = "dummy"
    model_kwargs["max_model_len"] = 1024

    compilation_config = dict(
        use_inductor_graph_partition=inductor_graph_partition,
        custom_ops=custom_ops.split(","),
        pass_config=PassConfig(
            enable_qk_norm_rope_fusion=True,
            enable_sp=True,
            fuse_gemm_comms=True,
        ),
    )

    matches_check = [
        "norm_rope_fusion",
        "sequence_parallel",
        "async_tp",
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
