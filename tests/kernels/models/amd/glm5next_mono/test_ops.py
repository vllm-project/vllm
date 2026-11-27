# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Op-level tests of the GLM-5.3-Flash MonoKernel on CDNA4, one process a TP rank.

Every stage of a launch is compared, on the kernel's own input to that stage,
with the fp64 math of `reference.py` (no vLLM op, no activation quantization),
and vLLM's op for the stage runs on the same input beside it. A stage passes
when the MonoKernel's relative L2 error to fp64 is below the bound's `ref` and,
where vLLM has the stage, at most `vllm` times vLLM's own error. Each test
prints its table of errors (run with `-s`). Bounds and widths are below."""

import pytest
import torch

from tests.utils import multi_process_parallel

from . import _dsa, _ffn, _kda, _moe
from ._ffn import _on_cdna4
from .reference import Bound

pytestmark = pytest.mark.skipif(
    not _on_cdna4(), reason="the Flash MonoKernel targets CDNA4"
)


# Step widths M (tokens a launch). The FFN's down stage runs out of LDS at 16 TP2
# rows and the sparse-attention TP2 layer serves at most 4 (`TP_MAX_WIDTH`).
M_FFN = (2, 4, 8, 16)
M_FFN_TP2 = (2, 4, 8)
M_LAYER = (2, 4, 8)
M_DSA_TP2 = (2, 4)

# Bounds, relative L2 to fp64 on the stage's own input. One bf16 rounding of a
# tensor is ~1.6e-3 of it, so a bf16 stage that rounds a few times sits at a few
# 1e-3: `ref` is 1e-2 unless noted. `vllm` is the most the
# MonoKernel's error may be over vLLM's on the same input.
FFN_HALF = {
    # fp32 on both sides: both sit at fp32 rounding (~1e-7), too low for a ratio
    "post mix": Bound(1e-5),
    "comb mix": Bound(1e-5),
    "FFN input": Bound(1e-2, 1.1),
    # experts take MXFP8 inputs (E4M3 a value, E8M0 a 32-block): ~3.5% a GEMM
    "MoE": Bound(1e-1),
    # the same MoE with the inputs rounded as the kernel does: what is left
    "MoE (MXFP8 inputs)": Bound(1e-2),
}
FFN_TOL = {"residual": Bound(1e-2, 1.1), **FFN_HALF}
KDA_TOL = {
    "attention input": Bound(1e-2, 1.1),
    "projection": Bound(1e-2, 1.1),
    "gates": Bound(1e-2, 1.1),
    "KDA output": Bound(1e-2, 1.1),
    "conv state": Bound(1e-2, 1.1),
    "recurrent state": Bound(1e-2, 1.1),
    **FFN_TOL,
}
DSA_TOL = {
    "attention input": Bound(1e-2, 1.1),
    "q_a | kv_a": Bound(1e-2),
    "index k": Bound(1e-2),
    # fp32 out of the GEMM, no bf16 rounding
    "index head weights": Bound(1e-4),
    "index gate": Bound(1e-2),
    "q_a norm": Bound(1e-2),
    # an e4m3 cache row is ~3% off its value (3 mantissa bits)
    "KV cache row (worst)": Bound(6e-2),
    "kpool tail vs vLLM": Bound(1e-2),
    "index cache vs vLLM": Bound(2e-2),
    "new pool vs vLLM (worst)": Bound(3e-2),
    "q_b": Bound(1e-2),
    "index q vs vLLM": Bound(3e-2),
    "index weights vs vLLM": Bound(1e-2),
    "q_lat (W_UK)": Bound(1e-2),
    # fp32 dot products of exact FP8 values
    "pool scores (worst row)": Bound(1e-4),
    "attention (worst row)": Bound(2e-2),
    **FFN_TOL,
}
MOE_TOL = {**FFN_TOL, "MoE": Bound(1e-1, 1.25)}


def _ready(tp: int) -> None:
    if torch.accelerator.device_count() < tp:
        pytest.skip(f"needs {tp} GPUs")
    pytest.importorskip("flydsl")


@pytest.mark.parametrize(
    "tp, separate, widths",
    [
        (2, True, M_FFN_TP2),
        (4, False, M_FFN),
        (4, True, M_FFN),
        (8, False, M_FFN),
    ],
)
def test_ffn_layer_vs_fp64(
    monkeypatch: pytest.MonkeyPatch, tp: int, separate: bool, widths: tuple
):
    """The FFN half (mHC post and pre, RMSNorm, router, top-8, MXFP4 experts with
    the shared one fused or `separate`, MoE TP reduce) against fp64 and vLLM's
    mHC ops (`FFN_TOL`); routing is exactly fp64's top 8. Then a NaN padding row
    stays in its row and 300 graph replays are bit-exact."""
    _ready(tp)
    check = _ffn._worker(_ffn._check, separate, widths, FFN_TOL)
    multi_process_parallel(monkeypatch, tp, 1, check)


@pytest.mark.parametrize("early_out", [False, True])
def test_stalled_rank_reports_expiry(monkeypatch: pytest.MonkeyPatch, early_out: bool):
    """A rank stalled in an FFN launch: the others' waits expire and say so."""
    _ready(4)
    multi_process_parallel(
        monkeypatch, 4, 1, _ffn._worker(_ffn._check_fault, early_out)
    )


def test_late_rank_is_waited_for(monkeypatch: pytest.MonkeyPatch):
    """A rank 10 s late: the others wait it out and nothing expires."""
    _ready(4)
    multi_process_parallel(monkeypatch, 4, 1, _ffn._worker(_ffn._check_slow_rank))


def test_stalled_rank_in_sparse_attention_reports_expiry(
    monkeypatch: pytest.MonkeyPatch,
):
    """A rank stalled in a sparse-attention launch: waits expire, no fault."""
    _ready(4)
    multi_process_parallel(monkeypatch, 4, 1, _ffn._worker(_dsa._check_fault))


@pytest.mark.parametrize("live", [0.0, 0.02, 0.5])
def test_sparse_attention_selects_through_tied_scores(
    monkeypatch: pytest.MonkeyPatch, live: float
):
    """Top-512 selection when thousands of pools tie at the threshold."""
    _ready(4)
    multi_process_parallel(monkeypatch, 4, 1, _ffn._worker(_dsa._check_ties, live))


@pytest.mark.parametrize(
    "tp, dim_first, saturate",
    [
        (2, False, False),
        (4, False, False),
        (4, True, False),
        (8, False, False),
        (4, False, True),
    ],
)
def test_kda_layer_vs_fp64(
    monkeypatch: pytest.MonkeyPatch, tp: int, dim_first: bool, saturate: bool
):
    """The whole KDA layer (attention-side mHC, in-projection, gates, short conv,
    gated delta rule, gated RMSNorm, both state updates, o_proj TP reduce) and
    the FFN half against fp64 and vLLM's ops (`KDA_TOL`), the conv state
    `dim_first` or width-first, `saturate`: gates far past the sigmoid's range."""
    _ready(tp)
    widths = M_LAYER
    check = _ffn._worker(_kda._check, dim_first, saturate, widths, KDA_TOL)
    multi_process_parallel(monkeypatch, tp, 1, check)


@pytest.mark.parametrize(
    "tp, fmt, kv_fp8",
    [
        (2, "mixed", False),
        (4, "fp8", False),
        (4, "bf16", False),
        (8, "mixed", False),
        (4, "mixed", True),
        (8, "mixed", True),
    ],
)
def test_dsa_layer_vs_fp64(
    monkeypatch: pytest.MonkeyPatch, tp: int, fmt: str, kv_fp8: bool
):
    """The whole sparse-attention layer (projections, norms, MLA and kpool index
    caches, index query, pool scores, top-512 selection, sparse attention, o_proj
    TP reduce) and the FFN half against fp64 (`DSA_TOL`; kpool and the FWHT index
    query against vLLM's kernels), weights `fmt`, `kv_fp8`: an e4m3 MLA cache as
    `--kv-cache-dtype fp8_e4m3` gives. The selection is checked exactly."""
    _ready(tp)
    widths = M_DSA_TP2 if tp == 2 else M_LAYER
    check = _ffn._worker(_dsa._check, _dsa.FORMATS[fmt], kv_fp8, widths, DSA_TOL)
    multi_process_parallel(monkeypatch, tp, 1, check)


@pytest.mark.parametrize("tp", [2, 4, 8])
def test_ffn_vs_fp64_and_glm5next_moe(monkeypatch: pytest.MonkeyPatch, tp: int):
    """The FFN half against fp64 and vLLM's own `Glm5NextMoE` built from the model
    config through `process_weights_after_loading`, AITER MoE with the shared
    expert fused, on random weights (`MOE_TOL`)."""
    _ready(tp)
    multi_process_parallel(monkeypatch, tp, 1, _moe._worker(M_LAYER, MOE_TOL))
