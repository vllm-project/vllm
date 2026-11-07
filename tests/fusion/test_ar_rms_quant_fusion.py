# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""End-to-end check that manual fusion runs without compiler-driven passes.

Drive an engine with every compiler fusion pass disabled and prove the
manual-fusion path still produces fused-kernel work inside model forward.
The counts come from the worker-installed counters in ``conftest.py``.

The model is Qwen3 (the decoder carrying the manual residual-stream fusion
path here), so at TP=1 the observable is the fused add+RMSNorm op dispatched
by the helper, plus the helper being entered at all. At TP>=2 the test
asserts the FlashInfer fused all-reduce+RMSNorm wrapper fired instead of an
unfused all-reduce at the stream boundary.

``ir_enable_torch_wrap`` is set so IR ops (``fused_add_rms_norm``) go through
torch op dispatch and the dispatch counter can observe them; in eager mode
they would otherwise route Python-direct. This changes how the op is invoked,
not which implementation runs.
"""

import pytest

from vllm import LLM, SamplingParams
from vllm.config import CompilationConfig, PassConfig
from vllm.platforms import current_platform

from ..utils import multi_gpu_only
from .conftest import count_matching

pytestmark = pytest.mark.skipif(
    not current_platform.is_cuda(), reason="CUDA-only fusion path"
)

PROMPT = "The capital of France is"
MODEL = "Qwen/Qwen3-0.6B"


def _build_llm(tp_size: int) -> LLM:
    """Every compiler fusion off: manual fusion must not need compiler help."""
    return LLM(
        model=MODEL,
        tensor_parallel_size=tp_size,
        max_model_len=1024,
        gpu_memory_utilization=0.2,
        enforce_eager=True,
        compilation_config=CompilationConfig(
            pass_config=PassConfig(
                fuse_norm_quant=False,
                fuse_act_quant=False,
                fuse_attn_quant=False,
                fuse_allreduce_rms=False,
            ),
            # Route IR ops through torch op dispatch so the dispatch counter
            # can see them (see module docstring).
            ir_enable_torch_wrap=True,
        ),
        disable_custom_all_reduce=True,
    )


def _generate(llm: LLM) -> None:
    llm.generate([PROMPT], SamplingParams(temperature=0, max_tokens=8))


def test_tp1_manual_fusion(ar_counter_session):
    """At TP=1 the helper runs in model code and dispatches the fused add+norm op."""
    llm = _build_llm(tp_size=1)
    with ar_counter_session(llm) as counts:
        _generate(llm)

    # The manual-fusion helper was entered from model forward...
    assert counts["prepare_calls"] > 0, dict(counts)
    # ...and the fused residual-add + RMSNorm op dispatched inside it.
    assert count_matching(counts, "fused_add_rms_norm") > 0, dict(counts)


@multi_gpu_only(num_gpus=2)
def test_tp2_flashinfer_fused_allreduce(ar_counter_session):
    """At TP>=2 the deferred all-reduce is absorbed by the FlashInfer fused kernel.

    The FlashInfer fused-AR entry point is a plain-Python resolution inside the
    helper, so the spy key is the direct assertion; the underlying torch op
    is also visible to the dispatch counter.
    """
    import vllm.model_executor.layers.fusion.ar_rms_quant as arq

    if arq.flashinfer_trtllm_fused_allreduce_norm is None:
        pytest.skip("FlashInfer fused-AR kernel unavailable")

    llm = _build_llm(tp_size=2)
    with ar_counter_session(llm) as counts:
        _generate(llm)

    # Boundaries deferred their reduce and the fused-AR kernel absorbed it...
    assert counts["owed_allreduce"] > 0, dict(counts)
    assert counts["flashinfer_ar_calls"] > 0, dict(counts)
    # ...with no unfused all-reduce left at the stream boundaries.
    assert counts["explicit_allreduce"] == 0, dict(counts)
