# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Essential tests for the fused DSpark draft sampling epilogue.

The fused kernel must be BIT-EXACT against the stock chain called exactly
as ``_sample_logits`` calls it: (a) the sampled ids, (b) the
``draft_tokens`` cells, (c) the full ``draft_logits`` cache rows (inert
``idx_map == -1`` rows keep their sentinel in both arms), and (d) the
returned persistent scratch view (stable address for CUDA-graph replay).
The arrival counters must self-clean to zero after every launch.
``VLLM_DISABLE_DRAFT_SAMPLE_EPILOGUE_FUSION=1`` falls back to the stock
chain before any launch; the fusion engages at num draft rows <=
``VLLM_DRAFT_SAMPLING_FUSION_MAX_TOKENS`` (default 2).
"""

import inspect
import os

import pytest
import torch

import vllm.envs as envs
from vllm.v1.worker.gpu.sample.gumbel import gumbel_sample
from vllm.v1.worker.gpu.spec_decode.dspark import (
    sample_epilogue_fused as fused_module,
)
from vllm.v1.worker.gpu.spec_decode.dspark.sample_epilogue_fused import (
    _BLOCK_SIZE,
    SampleEpilogueScratch,
    dspark_sample_epilogue_fused,
    sample_epilogue_fusion_disabled,
)
from vllm.v1.worker.gpu.spec_decode.dspark.speculator import DSparkSpeculator

KILL_VAR = "VLLM_DISABLE_DRAFT_SAMPLE_EPILOGUE_FUSION"
GATE_VAR = "VLLM_DRAFT_SAMPLING_FUSION_MAX_TOKENS"
FULL_VOCAB = 129280  # DeepSeek-V4.1-Flash: 127 blocks of 1024
SENTINEL = -777.0

needs_gpu = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="the fused epilogue oracle needs a GPU"
)


@pytest.fixture(autouse=True)
def _lift_draft_batch_gate():
    # The oracle exercises the kernel itself: lift the draft-batch gate so
    # every batch size engages. The gate policy has its own test below.
    os.environ[GATE_VAR] = "1024"
    yield
    os.environ.pop(GATE_VAR, None)


def _stock_chain(logits, idx_map, temperature, seeds, sample_pos, cache, col, use_fp64):
    return gumbel_sample(
        logits,
        idx_map,
        temperature,
        seeds,
        sample_pos - 1,
        apply_temperature=True,
        is_drafting=True,
        logits_cache=cache,
        logits_cache_col=col,
        use_fp64=use_fp64,
    )


def _make_case(
    b,
    vocab,
    temps,
    use_fp64,
    seed,
    max_rows=16,
    n_cols=5,
    strided=False,
    invalid_rows=(),
):
    """One oracle case: shared inputs plus both arms' output buffers
    (sentinel-filled caches, zeroed draft_tokens)."""
    col = n_cols // 2
    g = torch.Generator(device="cpu").manual_seed(seed)
    dev = torch.device("cuda")

    temperature = torch.zeros(max_rows, dtype=torch.float32, device=dev)
    for i, t in enumerate(temps):
        temperature[i] = t
    seeds = torch.randint(0, 2**62, (max_rows,), dtype=torch.int64, generator=g).to(dev)

    # idx_map: valid rows map to their own request slot; listed rows are
    # inert (-1) exactly as CUDA-graph padding produces.
    idx_row = torch.tensor(
        [r if r not in invalid_rows else -1 for r in range(b)], dtype=torch.int64
    )
    pos_row = torch.arange(100, 100 + b, dtype=torch.int64)
    if strided:
        # The caller's real geometry: columns of [b, n_cols] parents.
        idx_map = idx_row.unsqueeze(1).repeat(1, n_cols).to(dev)
        sample_pos = (
            pos_row.unsqueeze(1) + torch.arange(n_cols, dtype=torch.int64).unsqueeze(0)
        ).to(dev)
        idx_map, sample_pos = idx_map[:, col], sample_pos[:, col]
    else:
        idx_map, sample_pos = idx_row.to(dev), pos_row.to(dev)

    logits_cpu = torch.randn(b, vocab, generator=g, dtype=torch.float32) * 3.0
    if strided:
        # A non-contiguous [b, 2, vocab][:, 1] view, built ON DEVICE -- a
        # .to(dtype) of a non-dense view would materialize a contiguous
        # copy and defeat the stride test.
        parent = torch.zeros(b, 2, vocab, dtype=torch.bfloat16, device=dev)
        parent[:, 1] = logits_cpu.to(torch.bfloat16).to(dev)
        logits = parent[:, 1]
    else:
        logits = logits_cpu.to(torch.bfloat16).to(dev)

    return dict(
        logits=logits,
        idx_map=idx_map,
        sample_pos=sample_pos,
        temperature=temperature,
        seeds=seeds,
        col_t=torch.tensor(col, dtype=torch.int32, device=dev),
        ref_cache=torch.full(
            (max_rows, n_cols, vocab), SENTINEL, dtype=torch.bfloat16, device=dev
        ),
        fused_cache=torch.full(
            (max_rows, n_cols, vocab), SENTINEL, dtype=torch.bfloat16, device=dev
        ),
        ref_draft=torch.zeros((max_rows, n_cols), dtype=torch.int64, device=dev),
        fused_draft=torch.zeros((max_rows, n_cols), dtype=torch.int64, device=dev),
        scratch=SampleEpilogueScratch(max_rows, vocab, dev, use_fp64),
        use_fp64=use_fp64,
        b=b,
        col=col,
    )


def _run_fused(c):
    return dspark_sample_epilogue_fused(
        c["logits"],
        c["idx_map"],
        c["temperature"],
        c["seeds"],
        c["sample_pos"],
        c["fused_cache"],
        c["col_t"],
        c["fused_draft"],
        c["scratch"],
        use_fp64=c["use_fp64"],
    )


def _compare(case, tag):
    c = case
    ref_sampled = _stock_chain(
        c["logits"],
        c["idx_map"],
        c["temperature"],
        c["seeds"],
        c["sample_pos"],
        c["ref_cache"],
        c["col_t"],
        c["use_fp64"],
    )
    # The stock caller's slice copy (what the fused kernel absorbs).
    c["ref_draft"][: c["b"], c["col"]] = ref_sampled

    out = _run_fused(c)
    assert out is not None, f"{tag}: kill var leaked into the oracle arm"
    assert torch.equal(out, ref_sampled), f"{tag}: sampled ids diverged"
    assert torch.equal(c["fused_draft"], c["ref_draft"]), (
        f"{tag}: draft_tokens diverged"
    )
    # Written cells bit-identical; unwritten cells (inert rows, other
    # columns) keep the sentinel in BOTH arms.
    assert torch.equal(c["fused_cache"], c["ref_cache"]), (
        f"{tag}: draft logits cache diverged"
    )
    # The returned tensor is the persistent scratch view (the `prev`
    # handshake: same values, stable address for CUDA-graph capture).
    assert out.data_ptr() == c["scratch"].sampled.data_ptr(), (
        f"{tag}: returned tensor is not the scratch view"
    )
    # Self-cleaning counters (replay safety).
    assert int(c["scratch"].counters.sum()) == 0, (
        f"{tag}: arrival counters did not self-clean"
    )


# --- Bit-exact oracle --------------------------------------------------------


@pytest.mark.parametrize(
    "b,vocab,use_fp64,seed",
    [
        (1, 1000, False, 1),  # single row, one partial block
        (3, 1025, False, 3),  # block boundary + 1 element
        (3, 4096, True, 4),  # fp64 reduction path
        (8, 2048, True, 6),  # largest batch geometry
    ],
)
@needs_gpu
def test_bitexact_oracle_matrix(b, vocab, use_fp64, seed):
    # Mixed per-row temperatures: greedy (0), identity (1.0), hot (1.7).
    temps = [0.0, 1.0, 1.7] * 3
    _compare(
        _make_case(b, vocab, temps[:b], use_fp64, seed),
        f"matrix b={b} v={vocab} fp64={use_fp64}",
    )


@needs_gpu
def test_bitexact_full_vocab_production_geometry():
    # The V4.1-Flash geometry: vocab 129280 -> 127 blocks, bf16 logits.
    for use_fp64, seed in ((False, 11), (True, 12)):
        _compare(
            _make_case(2, FULL_VOCAB, [1.0, 0.7], use_fp64, seed, max_rows=8),
            f"full-vocab fp64={use_fp64}",
        )


@needs_gpu
def test_bitexact_strided_caller_geometry():
    # idx_map / sample_pos as columns of [b, n_cols] parents and logits as
    # a [b, 2, vocab][:, 1] slice: the in-kernel -1 offset and the stride
    # args replace the stock contiguous() copies.
    _compare(_make_case(3, 4096, [1.0, 0.0, 1.7], False, 31, strided=True), "strided")


@needs_gpu
def test_bitexact_inert_rows():
    # CUDA-graph padding geometry: idx_map == -1 rows must not scatter
    # into request slot 0 (the cache sentinel proves no store happened).
    _compare(
        _make_case(5, 4096, [1.0] * 5, False, 41, invalid_rows=(1, 3)), "inert-rows"
    )


@needs_gpu
def test_bitexact_engineered_ties():
    # temp=0 (deterministic argmax): duplicate maxima WITHIN a block
    # (stock tl.max tie behavior must be inherited) and ACROSS blocks
    # (torch.argmax first-max semantics).
    case = _make_case(3, 4096, [0.0, 0.0, 0.0], False, 51)
    logits = torch.full((3, 4096), -5.0, dtype=torch.float32)
    # row 0: within-block tie at offsets 5 and 900 (block 0)
    logits[0, 5] = logits[0, 900] = 7.0
    # row 1: cross-block tie at offsets 10 (block 0) and 1034 (block 1)
    logits[1, 10] = logits[1, 1034] = 7.0
    # row 2: triple tie across blocks 0, 1, 3
    logits[2, 0] = logits[2, 1024] = logits[2, 3072] = 7.0
    case["logits"] = logits.to(torch.bfloat16).to("cuda")
    _compare(case, "ties")
    # The cross-block winner must be the FIRST block's copy.
    out = case["scratch"].sampled[:3]
    assert int(out[1]) == 10, "cross-block tie must pick the first block"
    assert int(out[2]) == 0, "triple tie must pick the first block"


@needs_gpu
def test_replay_determinism_and_counter_self_cleaning():
    # Back-to-back launches with varying data AND batch geometry, all
    # reusing one scratch: every call must stay bit-exact and leave the
    # counters at zero (catches atomic-visibility bugs and counter leaks
    # that would deadlock or corrupt later launches).
    max_rows, vocab = 16, 4096
    scratch = SampleEpilogueScratch(max_rows, vocab, torch.device("cuda"), False)
    for it in range(4):
        b = (it % 4) * 2 + 1  # 1, 3, 5, 7
        case = _make_case(
            b,
            vocab,
            [1.0, 0.5, 1.7, 0.0, 1.0, 2.0, 0.9][:b],
            False,
            seed=1000 + it,
            max_rows=max_rows,
        )
        case["scratch"] = scratch
        _compare(case, f"replay iter={it} b={b}")


# --- Kill switch and engagement gate ----------------------------------------


@needs_gpu
def test_kill_var_falls_back_to_stock(monkeypatch):
    envs.disable_envs_cache()
    monkeypatch.delenv(KILL_VAR, raising=False)
    assert envs.VLLM_DISABLE_DRAFT_SAMPLE_EPILOGUE_FUSION is False
    case = _make_case(2, 1024, [1.0, 1.0], False, 71)
    monkeypatch.setenv(KILL_VAR, "1")
    assert sample_epilogue_fusion_disabled()
    assert _run_fused(case) is None, "kill var must fall back to stock"
    # No fused writes happened: the cache keeps its sentinel everywhere.
    assert torch.equal(
        case["fused_cache"], torch.full_like(case["fused_cache"], SENTINEL)
    )
    assert int(case["scratch"].counters.sum()) == 0
    # The stock chain itself is the fallback and still runs bit-exactly.
    ref = _stock_chain(
        case["logits"],
        case["idx_map"],
        case["temperature"],
        case["seeds"],
        case["sample_pos"],
        case["ref_cache"],
        case["col_t"],
        False,
    )
    assert ref.shape == (2,)


@needs_gpu
def test_draft_batch_gate_falls_back(monkeypatch):
    # The fusion engages at num draft rows <= VLLM_DRAFT_SAMPLING_FUSION_
    # MAX_TOKENS (default 2 -- the measured crossover between
    # launch-elimination wins and the stock sampler at larger batches).
    envs.disable_envs_cache()
    monkeypatch.delenv(GATE_VAR, raising=False)  # default gate: 2
    case3 = _make_case(3, 1024, [1.0] * 3, False, 81)
    assert _run_fused(case3) is None, "batch 3 > default gate 2 must fall back"
    assert torch.equal(
        case3["fused_cache"], torch.full_like(case3["fused_cache"], SENTINEL)
    )
    case2 = _make_case(2, 1024, [1.0, 1.0], False, 82)
    out2 = _run_fused(case2)
    assert out2 is not None and out2.shape == (2,)
    monkeypatch.setenv(GATE_VAR, "8")
    assert _run_fused(case3) is not None, "raised gate must re-engage"


# --- Wiring contracts --------------------------------------------------------


def test_source_contracts():
    # The fused branch sits AFTER the d2t scatter in _sample_logits
    # (reduced-vocab models keep the stock scatter chain), the stock
    # sampler remains as the fallback, and the handshake flag resets
    # before the acceptance estimator read.
    src = inspect.getsource(DSparkSpeculator._sample_logits)
    assert src.index("buf.index_copy_(") < src.index("dspark_sample_epilogue_fused(")
    assert "gumbel_sample" in src
    assert src.index("self._fused_epilogue_wrote = False") < src.index(
        "self._maybe_predict_acceptance("
    )
    # Eligibility: full-vocab probabilistic no-watermarker + the READY
    # census line.
    load_src = inspect.getsource(DSparkSpeculator.load_draft_model)
    assert "self.draft_logits is not None" in load_src
    assert "self._d2t_scatter_index is None" in load_src
    assert "self.draft_watermarker is None" in load_src
    assert "DSpark draft sample-epilogue fusion READY" in load_src
    # The bit-exactness argument rests on the SAME block width and noise
    # helpers as the stock kernel; pin both so a stock change forces a
    # revisit here. The kill var is read per call (eager A/B toggling).
    import vllm.v1.worker.gpu.sample.gumbel as gumbel_mod

    assert "BLOCK_SIZE = 1024" in inspect.getsource(gumbel_mod)
    assert _BLOCK_SIZE == 1024
    fused_src = inspect.getsource(fused_module)
    assert "from vllm.v1.worker.gpu.sample.gumbel import" in fused_src
    for helper in (
        "spec_step_seed",
        "murmur3_uniform32",
        "murmur3_uniform64",
        "_log1p_neg_stable",
        "_DRAFT_NOISE_SALT",
    ):
        assert helper in fused_src
    assert 'os.environ.get("VLLM_DISABLE_DRAFT_SAMPLE_EPILOGUE_FUSION"' in fused_src
