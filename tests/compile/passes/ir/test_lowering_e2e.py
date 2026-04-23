from collections.abc import Callable
from typing import Any

import pytest

from tests.compile.fusions_e2e.common import AttentionBackendCase, is_blackwell
from tests.compile.fusions_e2e.models import (
    FLASHINFER_ATTN,
    FLASHINFER_MLA_ATTN,
    ROCM_AITER_UNIFIED_ATTN,
    ROCM_ATTN,
    TRITON_ATTN,
    TRITON_MLA_ATTN,
)
from vllm.platforms import current_platform
from vllm.utils.flashinfer import is_flashinfer_fp8_blockscale_gemm_supported

from .models import qwen3_a3b_fp8


@pytest.mark.parametrize(
    "model_name, model_kwargs, hf_overrides, use_deepgemm",
    [(*qwen3_a3b_fp8, False), (*qwen3_a3b_fp8, True)],
)
@pytest.mark.parametrize(
    "attn_backend",
    [
        TRITON_ATTN,
        FLASHINFER_ATTN,
        ROCM_ATTN,
        ROCM_AITER_UNIFIED_ATTN,
        FLASHINFER_MLA_ATTN,
        TRITON_MLA_ATTN,
    ],
)
@pytest.mark.parametrize("n_layers", [6])
def test_op_lowering_e2e(
    model_name: str,
    model_kwargs: dict[str, Any],
    hf_overrides: Callable[[int], dict],
    attn_backend: AttentionBackendCase,
    n_layers: int,
    use_deepgemm: bool,
    run_e2e_lowering_test,
):
    if use_deepgemm and not current_platform.is_cuda():
        pytest.skip("DeepGemm only supported on CUDA")

    if use_deepgemm and is_flashinfer_fp8_blockscale_gemm_supported():
        # Flashinfer block FP8 GEMM has internal quantization, so it can't
        # be fused with other ops.
        pytest.skip("FlashInfer block FP8 GEMM not supported")
    if use_deepgemm and is_blackwell():
        # TODO(luka) DeepGEMM uses different quants, matching not supported
        #  - on Blackwell, uses a special quant fp8, currently not supported
        pytest.skip("DeepGEMM & quant matching not currently supported")

    # Reduce size of model and skip weight loading time
    model_kwargs["hf_overrides"] = hf_overrides(n_layers)
    model_kwargs["load_format"] = "dummy"
    model_kwargs["max_model_len"] = 1024
    model_kwargs["kernel_config"] = {"enable_flashinfer_autotune": False}

    use_aiter = current_platform.is_rocm() and ("qwen" in model_name.lower())

    run_e2e_lowering_test(
        model_name,
        model_kwargs,
        attn_backend,
        {},
        use_deepgemm=use_deepgemm,
        use_aiter=use_aiter,
    )
