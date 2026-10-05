# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the GLM kpool indexer's DCP top-k merge kernel JIT warmup."""

import sys
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest
import torch

from vllm.config import VllmConfig, set_current_vllm_config

_CUTEDSL_MODULE = "vllm.model_executor.kernels.attention.dsa.dcp_indexer_cutedsl"


class _RecordingKernel:
    """Stand-in for a JIT kernel singleton that records warmup registration."""

    def __init__(self, name: str, registrations: list[str]) -> None:
        self._name = name
        self._registrations = registrations

    def register_warmup(self) -> None:
        self._registrations.append(self._name)


def _build_kpool_indexer(
    monkeypatch: pytest.MonkeyPatch, *, dcp: int, jit_warmup: bool
) -> list[str]:
    """Build a kpool indexer with the DCP merge kernels stubbed out.

    Returns the names of the kernels whose register_warmup() ran. The
    platform guards (CUDA, DeepGEMM, CuteDSL, DCP group) are faked so the
    wiring is exercised on any machine.
    """
    from vllm.models.glm5next.nvidia import sparse_indexer

    registrations: list[str] = []
    fake_cutedsl: Any = ModuleType(_CUTEDSL_MODULE)
    fake_cutedsl._PACK_DCP_TOPK_CANDIDATES_KERNEL = _RecordingKernel(
        "pack_dcp_topk_candidates", registrations
    )
    fake_cutedsl._STABLE_TOPK_FROM_GATHERED_CANDIDATES_KERNEL = _RecordingKernel(
        "stable_topk_from_gathered_candidates", registrations
    )
    monkeypatch.setitem(sys.modules, _CUTEDSL_MODULE, fake_cutedsl)
    monkeypatch.setattr(
        sparse_indexer, "current_platform", SimpleNamespace(is_cuda=lambda: True)
    )
    monkeypatch.setattr(sparse_indexer, "has_deep_gemm", lambda: True)
    # Added by the warmup change under test; absent before it.
    monkeypatch.setattr(sparse_indexer, "has_cutedsl", lambda: True, raising=False)
    monkeypatch.setattr(
        sparse_indexer, "get_dcp_group", lambda: SimpleNamespace(rank_in_group=0)
    )

    config = VllmConfig(
        kernel_config={"enable_jit_warmup": jit_warmup},
        parallel_config={
            "tensor_parallel_size": dcp,
            "decode_context_parallel_size": dcp,
        },
    )
    with set_current_vllm_config(config):
        sparse_indexer.SparseAttnIndexerKpool(
            k_cache=None,
            quant_block_size=128,
            scale_fmt="ue8m0",
            topk_tokens=2048,
            head_dim=128,
            max_pool_len=4096,
            max_total_seq_len=8192,
            topk_indices_buffer=torch.empty(8, 2176, dtype=torch.int32),
        )
    return registrations


def test_kpool_indexer_registers_dcp_merge_kernels_for_jit_warmup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Under DCP with JIT warmup enabled, the kpool indexer registers the two
    DCP top-k merge kernels so they compile at startup instead of on the
    first long-context request, like the base SparseAttnIndexer does."""
    registrations = _build_kpool_indexer(monkeypatch, dcp=2, jit_warmup=True)
    assert registrations == [
        "pack_dcp_topk_candidates",
        "stable_topk_from_gathered_candidates",
    ]


@pytest.mark.parametrize(
    "dcp,jit_warmup",
    [(1, True), (2, False)],
    ids=["no-dcp", "jit-warmup-disabled"],
)
def test_kpool_indexer_skips_dcp_merge_warmup_without_dcp_or_opt_in(
    monkeypatch: pytest.MonkeyPatch, dcp: int, jit_warmup: bool
) -> None:
    """Without DCP the merge kernels never run, and with warmup disabled they
    must not be compiled eagerly: neither case may register them."""
    assert _build_kpool_indexer(monkeypatch, dcp=dcp, jit_warmup=jit_warmup) == []
