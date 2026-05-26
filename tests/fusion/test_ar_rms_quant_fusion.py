# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Manual-fusion tests for ``fused_ar_rms_norm_quant``.

These tests verify that the manual fusion helper introduced in
``vllm.model_executor.layers.fusion.ar_rms_quant`` actually fires
inside model forward when every compiler-driven fusion pass is disabled. They
do not validate output correctness — that belongs in the accuracy harness.

Tracks the AR+RMS+Quant bullet under
https://github.com/vllm-project/vllm/issues/43224.
"""

import pytest

from vllm import LLM, SamplingParams
from vllm.config import (
    CompilationConfig,
    CompilationMode,
    CUDAGraphMode,
    PassConfig,
)
from vllm.platforms import current_platform

from .conftest import count_matching

pytestmark = pytest.mark.skipif(
    not current_platform.is_cuda(), reason="CUDA-only fusion path"
)


def _build_llm(model: str, tp_size: int, n_layers: int = 4) -> LLM:
    """Build an LLM with every compiler fusion pass off.

    Manual fusion must work without any help from the compiler. Forcing all
    ``PassConfig`` flags off (and disabling the compile cache) leaves the
    fused op call sites in model code as the only thing that can produce a
    fused kernel invocation.
    """
    return LLM(
        model=model,
        tensor_parallel_size=tp_size,
        load_format="dummy",
        hf_overrides={"num_hidden_layers": n_layers},
        max_model_len=1024,
        # Stripped n_layers model is tiny; don't reserve the whole GPU so
        # consecutive parametrized tests can run in the same process.
        gpu_memory_utilization=0.2,
        compilation_config=CompilationConfig(
            mode=CompilationMode.VLLM_COMPILE,
            cudagraph_mode=CUDAGraphMode.NONE,
            inductor_compile_config={"force_disable_caches": True},
            pass_config=PassConfig(
                fuse_norm_quant=False,
                fuse_act_quant=False,
                fuse_attn_quant=False,
                fuse_allreduce_rms=False,
                enable_qk_norm_rope_fusion=False,
                enable_sp=False,
                fuse_gemm_comms=False,
            ),
        ),
        disable_custom_all_reduce=True,
    )


def _generate(llm: LLM) -> None:
    llm.generate(
        ["The capital of France is"],
        SamplingParams(temperature=0, max_tokens=32),
    )


@pytest.mark.parametrize(
    "model,expected_ops",
    [
        # FP8 static-per-tensor: helper dispatches to the C++ norm+quant kernel
        # for every decoder layer with a residual; the first layer (residual is
        # None) takes the rms_norm_static_fp8_quant path. Lower-bound check
        # only — exact count depends on layer count and decode steps.
        pytest.param(
            "RedHatAI/Llama-3.2-1B-Instruct-FP8",
            ("fused_add_rms_norm_static_fp8_quant",),
            id="llama-3.2-1b-fp8",
        ),
        # Unquantized: helper's _allreduce_rms_norm branch falls through to
        # norm(x, residual), which dispatches to the fused_add_rms_norm
        # custom op. Proves the helper is wired into model code and routes
        # the no-quant case correctly.
        pytest.param(
            "meta-llama/Llama-3.2-1B-Instruct",
            ("fused_add_rms_norm",),
            id="llama-3.2-1b",
        ),
    ],
)
def test_tp1_manual_fusion(op_count_session, model: str, expected_ops: tuple[str, ...]):
    llm = _build_llm(model, tp_size=1)
    with op_count_session(llm) as counts:
        _generate(llm)

    for op_substr in expected_ops:
        assert count_matching(counts, op_substr) > 0, (
            f"Manual fusion did not fire: no calls matching '{op_substr}'. "
            f"counts={dict(counts)}"
        )


# TODO(@mgoin): TP=2 variant.
#
# The TP>=2 + residual path dispatches to
# ``flashinfer_trtllm_fused_allreduce_norm``, which is a Python wrapper, not
# a torch op — TorchDispatchMode cannot see it. To assert it fired, add a
# parallel "python-call spy" alongside the dispatch counter that monkey-
# patches the name in ``vllm.model_executor.layers.fusion.ar_rms_quant``
# inside each worker, then check it was called at least once.
