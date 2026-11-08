# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""End-to-end tests that the model wiring defers and reduces as expected.

These tests drive a real engine with every compiler fusion pass disabled,
so manual fusion cannot be confused with a compiler pass doing the work.

Counts are collected inside the workers (see ``conftest.py``): the model runs
in spawned processes, so instrumenting the test process observes nothing.
``forwards`` counts ``finalize_norm`` invocations, which lets every other count
be normalized without assuming a prefill/decode step count.
"""

import pytest

from vllm import LLM, SamplingParams
from vllm.config import CompilationConfig, PassConfig
from vllm.platforms import current_platform

from ..utils import multi_gpu_only

pytestmark = pytest.mark.skipif(
    not current_platform.is_cuda(), reason="CUDA-only fusion path"
)

PROMPT = "The capital of France is"


def _build_llm(model: str, tp_size: int) -> LLM:
    """Every compiler fusion off: manual fusion must not need compiler help."""
    return LLM(
        model=model,
        tensor_parallel_size=tp_size,
        max_model_len=1024,
        # The models are tiny; don't reserve the whole GPU so consecutive tests
        # can run in the same process.
        gpu_memory_utilization=0.2,
        # Eager mode keeps exactly one residual-stream pass per model forward.
        enforce_eager=True,
        compilation_config=CompilationConfig(
            pass_config=PassConfig(
                fuse_norm_quant=False,
                fuse_act_quant=False,
                fuse_attn_quant=False,
                fuse_allreduce_rms=False,
            ),
        ),
        disable_custom_all_reduce=True,
    )


def _generate(llm: LLM) -> None:
    llm.generate([PROMPT], SamplingParams(temperature=0, max_tokens=8))


def _expected_counts(llm: LLM, *, migrated: bool) -> tuple[int, int]:
    """(prepares, owed all-reduces) per model forward."""
    n_layers = llm.model_config.get_total_num_hidden_layers()
    if not migrated:
        # Unmigrated decoders only reach the helper from the final norm.
        return 1, 0
    # Attention + MLP boundary per layer, plus the final norm. Layer 0's
    # attention reads the already-reduced embedding (FULL), so every other
    # boundary -- 2 * n_layers of them -- owes exactly one all-reduce at TP>1.
    return 2 * n_layers + 1, 2 * n_layers


def test_qwen3_enters_stream_at_every_boundary(ar_counter_session):
    """Qwen3 runs every attention/MLP boundary and its final norm through the stream."""
    llm = _build_llm("Qwen/Qwen3-0.6B", tp_size=1)
    prepares, _ = _expected_counts(llm, migrated=True)
    with ar_counter_session(llm) as counts:
        _generate(llm)

    forwards = counts["forwards"]
    assert forwards > 0, "no model forward was observed"
    assert counts["prepare_calls"] == forwards * prepares, dict(counts)
    # One rank holds the whole sum, so nothing is ever owed.
    assert counts["owed_allreduce"] == 0, dict(counts)


@multi_gpu_only(num_gpus=2)
def test_qwen3_reduces_once_per_deferred_boundary(ar_counter_session):
    llm = _build_llm("Qwen/Qwen3-0.6B", tp_size=2)
    prepares, allreduces = _expected_counts(llm, migrated=True)
    with ar_counter_session(llm) as counts:
        _generate(llm)

    forwards = counts["forwards"]
    assert forwards > 0, "no model forward was observed"
    assert counts["prepare_calls"] == forwards * prepares, dict(counts)
    assert counts["owed_allreduce"] == forwards * allreduces, (
        f"deferred all-reduce did not fire exactly once per boundary: {dict(counts)}"
    )


def test_qwen2_control_never_defers(ar_counter_session):
    """Unmigrated decoders keep the pre-fusion path: no stream, no deferred AR."""
    llm = _build_llm("Qwen/Qwen2.5-0.5B-Instruct", tp_size=1)
    prepares, _ = _expected_counts(llm, migrated=False)
    with ar_counter_session(llm) as counts:
        _generate(llm)

    forwards = counts["forwards"]
    assert forwards > 0, "no model forward was observed"
    assert counts["prepare_calls"] == forwards * prepares, dict(counts)
    assert counts["owed_allreduce"] == 0, dict(counts)
