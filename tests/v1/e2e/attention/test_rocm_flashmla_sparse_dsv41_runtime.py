# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""ROCM_FLASHMLA_SPARSE_DSV41 through the engine: scheduler, model runners,
CUDA graphs, async scheduling, spec decode and tensor parallelism.

The model is the five-layer DeepSeek-V4.1-Flash stand-in of the unit suite
(``tests/v1/attention/backends/rocm_flashmla_sparse_dsv41/dsv41_harness.py``):
one layer of each attention kind, random weights in the checkpoint's ranges
(peaked logits, unlike stock dummy weights). DeepSeek V4.1 has no second ROCm
backend to compare with, so the reference is the same model on the eager
Triton sparse path under the same scheduler settings, and the per-call
numerics are checked against a float32 reference in the unit suite. The
tolerances were calibrated on MI350X against the aiter sparse MLA path
(``VLLM_ROCM_USE_AITER_TRITON_SPARSE_MLA=1``), a second correct
implementation of the same attention.

The batch mixes short prompts, prompts past the window (128) and past both
top-k thresholds (512 / 1024 compressed entries), a shared prefix (prefix-
cache hits) and, at 2048 batched tokens, chunked prefill (extends), so every
phase runs under each engine mode.
"""

import pytest

from tests.v1.attention.backends.rocm_flashmla_sparse_dsv41.dsv41_harness import (
    AITER_ENV,
    MODEL,
    callable_overrides,
    engine_kwargs,
    random_prompts,
)
from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(
    not current_platform.is_rocm(), reason="ROCM_FLASHMLA_SPARSE_DSV41 is ROCm only"
)

MAX_TOKENS = 16
# Mean |prompt logprob difference| allowed against the eager Triton run.
# Calibrated on MI350X: the same eager run twice differs by 0.004 (the MoE's
# atomic reductions are not deterministic), the aiter path by 0.0044 and
# FULL_AND_PIECEWISE by 0.002. 0.02 is about five times that noise; a
# 64-token window instead of 128 is far outside it (negative control).
TOL = 0.02
# Per-call error against the float32 reference (as in the unit suite).
REF_TOL = {"mxfp8": 0.035, "bf16": 0.01}


def _prompts() -> list[list[int]]:
    prompts = random_prompts([5, 300, 1100, 2100])
    shared = random_prompts([700], seed=3)[0]
    return prompts + [
        shared + random_prompts([50], seed=4)[0],
        shared + random_prompts([60], seed=5)[0],
    ]


def _run(
    vllm_runner,
    monkeypatch,
    *,
    aiter: bool = False,
    cudagraph_mode: str = "NONE",
    async_scheduling: bool = False,
    model_runner_v2: bool = True,
    speculative: bool = False,
    tp: int = 1,
    **overrides,
):
    from vllm import SamplingParams
    from vllm.inputs import TokensPrompt

    monkeypatch.setenv("VLLM_ROCM_USE_AITER", "1")
    monkeypatch.setenv(AITER_ENV, "1" if aiter else "0")
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "1" if model_runner_v2 else "0")
    eager = cudagraph_mode == "NONE"
    kwargs = engine_kwargs(
        **{
            "max_num_batched_tokens": 2048,
            "enforce_eager": eager,
            "async_scheduling": async_scheduling,
            "tensor_parallel_size": tp,
            **overrides,
        }
    )
    if not eager:
        kwargs["compilation_config"] = {
            "cudagraph_mode": cudagraph_mode,
            "cudagraph_capture_sizes": [1, 2, 4, 8, 16],
        }
    if speculative:
        kwargs["speculative_config"] = {"method": "dspark", "num_speculative_tokens": 5}
        kwargs["hf_overrides"] = callable_overrides(kwargs["hf_overrides"])
    prompts = _prompts()
    params = SamplingParams(
        temperature=0.0, max_tokens=MAX_TOKENS, logprobs=5, prompt_logprobs=1
    )
    with vllm_runner(MODEL, **kwargs) as runner:
        # Each attention call (up to a few a layer) is also checked against
        # the float32 reference: the logprobs alone move little when attention
        # is wrong (see test_check_catches_a_wrong_window).
        runner.llm.collective_rpc(
            "probe_install", kwargs={"check": True, "max_checked_calls": 4}
        )
        outs = runner.llm.generate(
            [TokensPrompt(prompt_token_ids=p) for p in prompts], params
        )
        calls = runner.llm.collective_rpc("probe_report")
    result = []
    for prompt, out in zip(prompts, outs):
        lps = out.prompt_logprobs[1:]
        result.append(
            {
                "prompt": [d[t].logprob for t, d in zip(prompt[1:], lps)],
                "top": [max(v.logprob for v in d.values()) for d in lps],
                "tokens": list(out.outputs[0].token_ids),
                "sample": [
                    {t: v.logprob for t, v in d.items()}
                    for d in out.outputs[0].logprobs
                ],
            }
        )
    return result, [c for rank in calls for c in rank]


def _reference_errors(calls) -> dict[str, float]:
    """Worst per-call error against the float32 reference, by output kind."""
    worst: dict[str, float] = {}
    for c in calls:
        assert c.get("finite", True), c
        for key in ("decode_err", "prefill_err"):
            if key in c:
                kind = c["output"]
                worst[kind] = max(worst.get(kind, 0.0), c[key])
    return worst


def _check_reference(calls) -> None:
    worst = _reference_errors(calls)
    assert worst, "no attention call was checked"
    for kind, err in worst.items():
        assert err < REF_TOL[kind], (kind, err)


def _mean_prompt_diff(ref, out) -> float:
    diffs = [
        abs(a - b) for r, o in zip(ref, out) for a, b in zip(r["prompt"], o["prompt"])
    ]
    return sum(diffs) / len(diffs)


def _check_samples(ref, out) -> None:
    """Greedy tokens may diverge only where the other run ranks the token in
    its top 5 (a near tie), as ``check_logprobs_close`` allows."""
    for i, (r, o) in enumerate(zip(ref, out)):
        for step, (a, b) in enumerate(zip(r["tokens"], o["tokens"])):
            if a == b:
                continue
            assert a in o["sample"][step] and b in r["sample"][step], (i, step)
            break


_REFERENCE: dict = {}


def _reference(vllm_runner, monkeypatch, **kwargs):
    key = tuple(sorted(kwargs.items()))
    if key not in _REFERENCE:
        _REFERENCE[key], calls = _run(vllm_runner, monkeypatch, **kwargs)
        _check_reference(calls)
        top = [t for r in _REFERENCE[key] for t in r["top"]]
        # The comparison only has power if the logits are far from uniform:
        # a uniform distribution over the 129k vocabulary gives -11.8, the
        # stand-in's checkpoint-range weights about -6.4.
        assert sum(top) / len(top) > -8.0, "reference logits are nearly flat"
    return _REFERENCE[key]


def _compare(vllm_runner, monkeypatch, tol=None, ref_kwargs=None, **kwargs):
    ref = _reference(vllm_runner, monkeypatch, **(ref_kwargs or {}))
    out, calls = _run(vllm_runner, monkeypatch, **kwargs)
    _check_reference(calls)
    diff = _mean_prompt_diff(ref, out)
    tol = TOL if tol is None else tol
    assert diff <= tol, f"mean prompt logprob diff {diff:.4f} > {tol}"
    _check_samples(ref, out)
    return diff


ENGINE_MODES = [
    # (cudagraph_mode, async_scheduling, model_runner_v2)
    ("NONE", True, True),
    ("PIECEWISE", False, True),
    ("FULL_DECODE_ONLY", True, True),
    ("FULL_AND_PIECEWISE", True, True),
    ("NONE", False, False),
    ("FULL_AND_PIECEWISE", False, False),
]


@pytest.mark.parametrize("aiter", [False, True], ids=["triton", "aiter"])
@pytest.mark.parametrize(
    "cudagraph_mode, async_scheduling, model_runner_v2", ENGINE_MODES
)
def test_engine_modes(
    vllm_runner, monkeypatch, aiter, cudagraph_mode, async_scheduling, model_runner_v2
):
    from vllm.platforms.rocm import on_gfx950

    if aiter and not on_gfx950():
        pytest.skip("the aiter sparse MLA kernel is gfx950-only")
    _compare(
        vllm_runner,
        monkeypatch,
        aiter=aiter,
        cudagraph_mode=cudagraph_mode,
        async_scheduling=async_scheduling,
        model_runner_v2=model_runner_v2,
    )


def test_prefix_caching_and_chunking_do_not_change_logprobs(vllm_runner, monkeypatch):
    """H1: no prefix caching and one-step prefills give the same logprobs as
    the chunked, prefix-cached reference."""
    _compare(
        vllm_runner,
        monkeypatch,
        enable_prefix_caching=False,
        max_num_batched_tokens=8192,
    )


@pytest.mark.parametrize("aiter", [False, True], ids=["triton", "aiter"])
def test_dspark_spec_decode(vllm_runner, monkeypatch, aiter):
    """H4: DSpark drafts verified by the target's multi-token decode rows give
    the target's own greedy tokens (prompt logprobs are prefill-only)."""
    from vllm.platforms.rocm import on_gfx950

    if aiter and not on_gfx950():
        pytest.skip("the aiter sparse MLA kernel is gfx950-only")
    _compare(vllm_runner, monkeypatch, aiter=aiter, speculative=True)


@pytest.mark.parametrize("aiter", [False, True], ids=["triton", "aiter"])
def test_tensor_parallel_2(vllm_runner, monkeypatch, aiter):
    """TP2 shards the 64 heads to 32 a rank (o_groups 8 -> 4); the latent
    cache stays whole. The random weights are drawn per shard, so a TP2 model
    differs from the TP1 one: each rank's attention calls are checked against
    the reference, and graph mode is compared with eager at TP2."""
    import torch

    from vllm.platforms.rocm import on_gfx950

    if torch.accelerator.device_count() < 2:
        pytest.skip("needs 2 GPUs")
    if aiter and not on_gfx950():
        pytest.skip("the aiter sparse MLA kernel is gfx950-only")
    _compare(
        vllm_runner,
        monkeypatch,
        ref_kwargs={"tp": 2},
        tp=2,
        aiter=aiter,
        cudagraph_mode="FULL_AND_PIECEWISE",
        async_scheduling=True,
    )


def test_check_catches_a_wrong_window(vllm_runner, monkeypatch):
    """Negative control. A model whose layers use a 64-token window where the
    reference expects 128 must fail the per-call reference check. Its
    logprobs barely move (0.006 mean on MI350X, against 0.004 of run-to-run
    noise): with random weights the residual stream dominates the output, so
    the engine runs lean on the per-call check, not on logprobs alone."""
    from tests.v1.attention.backends.rocm_flashmla_sparse_dsv41.dsv41_harness import (
        hf_overrides,
    )

    _, calls = _run(
        vllm_runner, monkeypatch, hf_overrides=hf_overrides(sliding_window=64)
    )
    worst = _reference_errors(calls)
    assert worst and all(err > 0.3 for err in worst.values()), worst


def test_dcp_is_refused(vllm_runner, monkeypatch, capfd):
    """H5: DCP is refused at startup (the SWA and compressed KV groups reject
    it; the backend has no DCP impl either), not at the first forward."""
    import torch

    if torch.accelerator.device_count() < 2:
        pytest.skip("needs 2 GPUs")
    with pytest.raises(Exception, match="initialization failed"):
        _run(vllm_runner, monkeypatch, tp=2, decode_context_parallel_size=2)
    out, err = capfd.readouterr()
    assert "DCP not support sliding window" in out + err
