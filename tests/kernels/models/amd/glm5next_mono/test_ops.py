# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Op-level tests of the GLM-5.3-Flash MonoKernel on CDNA4, one process a TP rank.

Each launch kind runs on random weights and is checked stage by stage against the
ops vLLM itself runs for the same decode step, on the kernel's own inputs to each
stage, at every width; then every width is captured in one graph and replayed,
bit for bit against the first run. The checks live beside this file:

- ``_ffn``: the FFN half (mHC seam, RMSNorm, router, top-8, shared expert, MXFP4
  experts, MoE TP reduce), a stalled rank's expired waits and a late rank waited
  out;
- ``_kda``: the whole KDA layer (in-projections, short conv, gated delta rule,
  gated norm, o_proj and its TP reduce), then the FFN half;
- ``_dsa``: the whole sparse-attention layer (MLA and kpool index caches, the
  indexer, top-2048 selection, split attention), bf16 or e4m3 KV cache;
- ``_moe``: the FFN stages against vLLM's own ``Glm5NextMoE`` built from the
  model config, through ``process_weights_after_loading``."""

import pytest
import torch

from tests.utils import multi_process_parallel

from . import _dsa, _ffn, _kda, _moe
from ._ffn import _on_cdna4

pytestmark = pytest.mark.skipif(
    not _on_cdna4(), reason="the Flash MonoKernel targets CDNA4"
)


def _ready(tp: int) -> None:
    if torch.accelerator.device_count() < tp:
        pytest.skip(f"needs {tp} GPUs")
    pytest.importorskip("flydsl")


@pytest.mark.parametrize("tp, separate", [(2, True), (4, False), (4, True), (8, False)])
def test_ffn_layer_matches_vllm_ops(
    monkeypatch: pytest.MonkeyPatch, tp: int, separate: bool
):
    _ready(tp)
    multi_process_parallel(monkeypatch, tp, 1, _ffn._worker(_ffn._check, separate))


@pytest.mark.parametrize("early_out", [False, True])
def test_stalled_rank_reports_expiry(monkeypatch: pytest.MonkeyPatch, early_out: bool):
    _ready(4)
    multi_process_parallel(
        monkeypatch, 4, 1, _ffn._worker(_ffn._check_fault, early_out)
    )


def test_late_rank_is_waited_for(monkeypatch: pytest.MonkeyPatch):
    _ready(4)
    multi_process_parallel(monkeypatch, 4, 1, _ffn._worker(_ffn._check_slow_rank))


def test_stalled_rank_in_sparse_attention_reports_expiry(
    monkeypatch: pytest.MonkeyPatch,
):
    _ready(4)
    multi_process_parallel(monkeypatch, 4, 1, _ffn._worker(_dsa._check_fault))


@pytest.mark.parametrize("live", [0.0, 0.02, 0.5])
def test_sparse_attention_selects_through_tied_scores(
    monkeypatch: pytest.MonkeyPatch, live: float
):
    _ready(4)
    multi_process_parallel(monkeypatch, 4, 1, _ffn._worker(_dsa._check_ties, live))


@pytest.mark.parametrize(
    "tp, dim_first", [(2, False), (4, False), (4, True), (8, False)]
)
def test_kda_layer_matches_vllm_ops(
    monkeypatch: pytest.MonkeyPatch, tp: int, dim_first: bool
):
    _ready(tp)
    multi_process_parallel(monkeypatch, tp, 1, _ffn._worker(_kda._check, dim_first))


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
def test_dsa_layer_matches_vllm_ops(
    monkeypatch: pytest.MonkeyPatch, tp: int, fmt: str, kv_fp8: bool
):
    """``kv_fp8``: an e4m3 MLA cache, as ``--kv-cache-dtype fp8_e4m3`` gives."""
    _ready(tp)
    check = _ffn._worker(_dsa._check, _dsa.FORMATS[fmt], kv_fp8)
    multi_process_parallel(monkeypatch, tp, 1, check)


@pytest.mark.parametrize("tp, fmt", [(2, "mixed"), (4, "mixed"), (8, "mixed")])
def test_dsa_layer_mxfp4_kv_cache(monkeypatch: pytest.MonkeyPatch, tp: int, fmt: str):
    """An ``mxfp4_mla`` MLA cache (``--kv-cache-dtype mxfp4_mla``): rows written as
    ``store_mxfp4_mla`` writes them, the attention over their unpacked values."""
    _ready(tp)
    check = _ffn._worker(_dsa._check, _dsa.FORMATS[fmt], False, True)
    multi_process_parallel(monkeypatch, tp, 1, check)


@pytest.mark.parametrize("tp", [2, 4, 8])
def test_ffn_matches_glm5next_moe(monkeypatch: pytest.MonkeyPatch, tp: int):
    _ready(tp)
    multi_process_parallel(monkeypatch, tp, 1, _moe._worker)
