# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Every ``forward_mqa`` dispatch path of the DeepSeek V4.1 ROCm sparse MLA
layer, inside the running engine, against a float32 reference.

Dispatch table of ``DeepseekV41ROCMAiterMLAAttention.forward_mqa``
(``vllm/models/deepseek_v41/amd/rocm.py``). "source" is a layer that is its
own index source (layers 1, 3, 4 of the stand-in); a consumer (layer 2)
reads the ragged top-k its source built this step; the SWA-only layer
(layer 0) has no top-k.

condition -> kernels:

- ``attn_metadata is None`` (profiling) -> none; output zeroed, workspace
  reserved at warmup.
- decode, Triton, MXFP8 out (gfx950) -> ``rocm_sparse_attn_decode`` (its
  reduce epilogue does the inverse RoPE and the MXFP8 quant).
- decode, Triton, bf16 out -> ``rocm_sparse_attn_decode`` +
  ``rocm_inverse_rope_rows_``.
- decode, aiter, MXFP8 out -> ``triton_sparse_mla_fwd`` +
  ``rocm_inverse_rope_mxfp8_rows``.
- decode, aiter, bf16 out -> ``triton_sparse_mla_fwd`` +
  ``rocm_inverse_rope_rows_``.
- decode, compressed layer that is its own source -> also
  ``compute_global_topk_ragged_indices_and_indptr`` (consumers reuse it).
- prefill, Triton -> ``dequantize_and_gather_k_cache`` +
  ``combine_topk_swa_indices`` + ``rocm_sparse_attn_prefill``, then
  ``rocm_inverse_rope_mxfp8_rows`` (MXFP8) or ``rocm_inverse_rope_rows_``.
- prefill, aiter -> ``triton_sparse_mla_fwd`` + the same inverse RoPE.
- prefill, aiter, compressed layer that is its own source -> also
  ``build_prefill_topk_ragged_indices``.
- mixed -> the decode row + the prefill row.

The aiter path is opt-in (``VLLM_ROCM_USE_AITER_TRITON_SPARSE_MLA=1``,
gfx950 only). Every call is also checked for: finite output; the reference
error (bf16 output 1%, MXFP8 output 3.5%: e4m3 rounding alone is ~2.7%); a
negative control (the reference without the compressed rows must miss by
far more); the indexer's selection staying causal and actually selecting
(full top-k rows once the context passes 512 compressed entries); and the
builders' SWA ragged indices and compressed slot mapping against values
rebuilt from positions and block tables.

The workload covers prefill-only, chunked prefill mixed with decodes,
extends (chunk continuations and a prefix-cache hit) and decode-only steps,
with prompts at, below and above the window (128), the 128-token block, and
the ratio-1/ratio-2 top-k thresholds (512 and 1024 tokens).
"""

import pytest
import torch

from tests.v1.attention.backends.rocm_flashmla_sparse_dsv41.dsv41_harness import (
    DSV41_HEADS,
    MODEL,
    aiter_sparse_mla,
    callable_overrides,
    engine_kwargs,
    hf_overrides,
    random_prompts,
)

SOURCES = {1, 3, 4}  # their own index source
COMPRESSED = {1, 2, 3, 4}
MXFP8_TOL, BF16_TOL, NEG_MIN = 0.035, 0.01, 0.3

# Window and block boundaries, ratio-1 top-k (512) and ratio-2 (1024).
BOUNDARY_LENS = [1, 127, 128, 129, 255, 256, 257, 511, 513, 1023, 1024, 1025, 2049]


def expected_kernels(layer: int, phase: str, aiter: bool, output: str) -> set[str]:
    rope = (
        "rocm_inverse_rope_mxfp8_rows"
        if output == "mxfp8"
        else ("rocm_inverse_rope_rows_")
    )
    want: set[str] = set()
    if phase in ("decode", "mixed"):
        if aiter:
            want |= {"triton_sparse_mla_fwd", rope}
        else:
            want.add("rocm_sparse_attn_decode")
            if output == "bf16":
                want.add(rope)
        if layer in SOURCES:
            want.add("compute_global_topk_ragged_indices_and_indptr")
    if phase in ("prefill", "mixed"):
        want.add(rope)
        if aiter:
            want.add("triton_sparse_mla_fwd")
            if layer in SOURCES:
                want.add("build_prefill_topk_ragged_indices")
        else:
            want |= {
                "dequantize_and_gather_k_cache",
                "combine_topk_swa_indices",
                "rocm_sparse_attn_prefill",
            }
    return want


def _run_workload(llm, lens=BOUNDARY_LENS, max_tokens: int = 4):
    from vllm import SamplingParams
    from vllm.inputs import TokensPrompt

    params = SamplingParams(temperature=0.0, max_tokens=max_tokens)
    prompts = random_prompts(lens)
    llm.generate([TokensPrompt(prompt_token_ids=p) for p in prompts], params)
    # A prefix-cache hit: the longest prompt again with a new tail extends
    # from its cached blocks.
    hit = prompts[-1] + random_prompts([300], seed=1)[0]
    llm.generate([TokensPrompt(prompt_token_ids=hit)], params)
    return llm.collective_rpc("probe_report")[0]


def _check_calls(calls: list[dict], aiter: bool) -> dict:
    assert calls, "no forward_mqa call was recorded"
    seen: dict = {"phases": set(), "extend": False, "storages": set()}
    for c in calls:
        layer, phase = c["layer"], c["phase"]
        assert phase != "profile", c
        ctx = (layer, phase, c.get("decode_tokens"), c.get("prefill_tokens"))
        seen["phases"].add(phase)
        seen["extend"] |= c["extend"]
        assert set(c["kernels"]) == expected_kernels(
            layer, phase, aiter, c["output"]
        ), ctx
        if "stats" not in c:
            continue
        assert c["finite"], ctx
        tol = MXFP8_TOL if c["output"] == "mxfp8" else BF16_TOL
        for key in ("decode_err", "prefill_err"):
            if key in c:
                assert c[key] < tol, (ctx, key, c[key])
        assert c["bf16_err"] < BF16_TOL, (ctx, c["bf16_err"])
        assert set(c["bf16_kernels"]) == expected_kernels(
            layer, phase, aiter, "bf16"
        ), ctx
        stats = c["stats"]
        if layer in COMPRESSED:
            assert stats["noncausal"] == 0, (ctx, stats)
            if stats["topk_selected"]:
                assert c["neg_err"] > NEG_MIN, (ctx, c["neg_err"])
        if "decode_swa_ragged_ok" in c:
            assert c["decode_swa_ragged_ok"], ctx
            if phase == "decode":
                seen["storages"].add((layer, c["decode_swa_ragged_storage"]))
        if "prefill_swa_ragged_built" in c:
            # Built only for the aiter prefill, which reads the cache in place.
            assert c["prefill_swa_ragged_built"] is aiter, ctx
            if aiter and "prefill_swa_ragged_ok" in c:
                assert c["prefill_swa_ragged_ok"], ctx
        if "compressed_slot_mapping_ok" in c:
            assert c["compressed_slot_mapping_ok"], ctx
    return seen


def _assert_selection_active(calls: list[dict]) -> None:
    """Contexts past the top-k reach full top-k rows on every compressed
    layer, so the indexer really selected (not "everything visible")."""
    full = {
        c["layer"]
        for c in calls
        if "stats" in c and c["stats"].get("topk_rows_full", 0) > 0
    }
    assert full >= COMPRESSED, full


@pytest.mark.parametrize("aiter", [False, True], ids=["triton", "aiter"])
@pytest.mark.parametrize("heads", DSV41_HEADS, ids=lambda h: h.id)
def test_dispatch_matches_reference(vllm_runner, monkeypatch, aiter, heads):
    from vllm.platforms.rocm import on_gfx950

    if aiter and not on_gfx950():
        pytest.skip("the aiter sparse MLA kernel is gfx950-only")
    with (
        aiter_sparse_mla(monkeypatch, aiter),
        vllm_runner(
            MODEL,
            # 2048 batched tokens: the boundary prompts chunk, so steps mix
            # decodes with prefills and chunk continuations extend.
            **engine_kwargs(
                hf_overrides=hf_overrides(heads), max_num_batched_tokens=2048
            ),
        ) as runner,
    ):
        llm = runner.llm
        config = llm.collective_rpc("probe_install", kwargs={"bf16_rerun": True})[0]
        assert sorted(config) == [0, 1, 2, 3, 4]
        for layer_id, layer in config.items():
            assert layer["use_aiter_sparse_mla"] is aiter
            assert layer["n_local_heads"] == heads.num_heads
            assert layer["kv_cache_dtype"] == "fp8_ds_mla"
            want = "DeepseekV41RocmMxfp4Indexer" if layer_id in SOURCES else None
            assert layer["indexer_cls"] == want, layer_id
        calls = _run_workload(llm)
    seen = _check_calls(calls, aiter)
    assert seen["phases"] == {"prefill", "mixed", "decode"}, seen["phases"]
    assert seen["extend"]
    _assert_selection_active(calls)
    # One persistent ragged buffer per layer group, reused by every decode
    # step. (Eager prefill steps run layers 3-4 on the SWA bounded replay's
    # own metadata, built by another builder, so only decode steps count.)
    storages: dict[int, set[int]] = {}
    for layer, ptr in seen["storages"]:
        storages.setdefault(layer, set()).add(ptr)
    assert all(len(p) == 1 for p in storages.values()), storages


@pytest.mark.parametrize("aiter", [False, True], ids=["triton", "aiter"])
def test_profiling_call_zeroes_output(vllm_runner, monkeypatch, aiter):
    """C2: with no metadata (the profiling run) every layer zeroes its output,
    launches nothing and asks for no workspace past what warmup reserved."""
    from vllm.platforms.rocm import on_gfx950

    if aiter and not on_gfx950():
        pytest.skip("the aiter sparse MLA kernel is gfx950-only")
    with (
        aiter_sparse_mla(monkeypatch, aiter),
        vllm_runner(MODEL, **engine_kwargs()) as runner,
    ):
        llm = runner.llm
        llm.collective_rpc("probe_install", kwargs={"check": False})
        result = llm.collective_rpc("probe_profile_call")[0]
    assert sorted(result) == [0, 1, 2, 3, 4]
    for layer_id, r in result.items():
        assert r["zeroed"], (layer_id, r)
        assert r["kernels"] == [], (layer_id, r)
        assert not r["workspace_grew"], (layer_id, r)
        if on_gfx950():
            assert r["output"] == "QuantizedActivation", (layer_id, r)


@pytest.mark.parametrize("aiter", [False, True], ids=["triton", "aiter"])
def test_spec_decode_rows(vllm_runner, monkeypatch, aiter):
    """Multi-token decode (q_len > 1 per request, spec-as-decode): DSpark
    (V4.1's draft, five tokens a block) verification steps reach the target
    layers as decodes, and every row of them is checked."""
    from vllm.platforms.rocm import on_gfx950

    if aiter and not on_gfx950():
        pytest.skip("the aiter sparse MLA kernel is gfx950-only")
    with (
        aiter_sparse_mla(monkeypatch, aiter),
        vllm_runner(
            MODEL,
            **engine_kwargs(
                hf_overrides=callable_overrides(hf_overrides()),
                speculative_config={"method": "dspark", "num_speculative_tokens": 5},
            ),
        ) as runner,
    ):
        llm = runner.llm
        llm.collective_rpc("probe_install", kwargs={"bf16_rerun": True})
        calls = _run_workload(llm, lens=[129, 600, 1100], max_tokens=8)
    target = [c for c in calls if c["layer"] in range(5)]
    _check_calls(target, aiter)
    multi = [c for c in target if c.get("max_decode_query_len", 1) > 1]
    assert multi, "no verification step reached the target layers"
    assert any("decode_err" in c for c in multi)


def test_kv_insert_round_trip(vllm_runner):
    """D1/D4: the fused Q RoPE + KV RoPE/UE8M0 quant/SWA insert, against
    RoPE in PyTorch; the written rows are decoded by the suite's own record
    decoder, so a wrong record layout fails here and not only through
    attention. NoPE rounds to e4m3 with a power-of-two scale per 64 dims
    (~2.5% RMS); RoPE dims stay bf16."""
    with vllm_runner(MODEL, **engine_kwargs()) as runner:
        llm = runner.llm
        llm.collective_rpc("probe_install", kwargs={"check": False, "kv_check": True})
        _run_workload(llm, lens=[1, 129, 700], max_tokens=3)
        records = llm.collective_rpc("probe_kv_report")[0]
    assert {r["layer"] for r in records} == {0, 1, 2, 3, 4}
    for r in records:
        if r["rows"] == 0:
            continue
        assert r["nope"] < 0.05, r
        assert r["rope"] < 0.01, r
        assert r["q"] < 0.01, r
        assert r["q_pad_zero"], r


@pytest.fixture(autouse=True)
def _no_grad():
    with torch.inference_mode():
        yield
