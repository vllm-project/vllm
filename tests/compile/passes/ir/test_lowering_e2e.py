import pytest

from tests.compile.fusions_e2e.common import AttentionBackendCase
from tests.compile.fusions_e2e.models import (
    FLASHINFER_ATTN,
    FLASHINFER_MLA_ATTN,
    ROCM_AITER_UNIFIED_ATTN,
    ROCM_ATTN,
    TRITON_ATTN,
    TRITON_MLA_ATTN,
)
from vllm.platforms import current_platform


@pytest.mark.parametrize(
    "model_name",
    [
        "meta-llama/Llama-3.2-1B-Instruct",
        "RedHatAI/Meta-Llama-3.1-8B-Instruct-FP8",
        "Qwen/Qwen3-30B-A3B",
    ],
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
    attn_backend: AttentionBackendCase,
    n_layers: int,
    run_e2e_lowering_test,
):
    # Reduce size of model and skip weight loading time
    model_kwargs = {}
    model_kwargs["hf_overrides"] = {"num_hidden_layers": n_layers}
    model_kwargs["load_format"] = "dummy"
    model_kwargs["max_model_len"] = 1024
    model_kwargs["kernel_config"] = {"enable_flashinfer_autotune": False}

    use_aiter = current_platform.is_rocm() and ("qwen" in model_name.lower())

    run_e2e_lowering_test(
        model_name,
        model_kwargs,
        attn_backend,
        {},
        use_aiter=use_aiter,
    )
