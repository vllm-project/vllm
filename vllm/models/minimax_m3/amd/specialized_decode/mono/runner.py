# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Host side of the FFN launch: one compiled kernel, scratch region and the
shared epoch word per row count of one TP rank."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch

from .ffn import (
    EPOCH_WORDS,
    HIDDEN,
    MAX_TOKENS,
    Experts,
    FfnBuild,
    build_ffn,
    scratch_bytes,
    scratch_layout,
)


def _writable_flydsl_cache() -> None:
    """AITER points FlyDSL at its bundled read-only cache."""
    cur = os.environ.get("FLYDSL_RUNTIME_CACHE_DIR")
    if cur and os.access(cur, os.W_OK):
        return
    path = Path.home() / ".flydsl" / "cache"
    path.mkdir(parents=True, exist_ok=True)
    os.environ["FLYDSL_RUNTIME_CACHE_DIR"] = str(path)


class FfnRunner:
    """``sym``: this rank's peer buffer; ``peers``: every rank's, int64 on
    device. The kernels move the epoch on themselves, so every rank that runs
    the same launches agrees on it. Runners on one peer buffer must share
    ``epoch``: a tag repeated over the buffer's lifetime could match a stale
    pair."""

    def __init__(
        self,
        tp: int,
        rank: int,
        sym: int,
        peers: torch.Tensor,
        device: torch.device,
        widths: tuple[int, ...] = tuple(range(1, MAX_TOKENS + 1)),
        experts: Experts = Experts.ALL,
        timeline: bool = False,
        epoch: torch.Tensor | None = None,
    ) -> None:
        _writable_flydsl_cache()
        self.tp, self.rank, self.sym, self.peers = tp, rank, sym, peers
        self.widths = widths
        self.keys = {
            s: FfnBuild(tokens=s, tp=tp, experts=experts, timeline=timeline)
            for s in widths
        }
        if epoch is None:
            epoch = torch.zeros(EPOCH_WORDS, dtype=torch.int32, device=device)
        self.epoch = epoch
        self.scratch = {
            s: torch.zeros(scratch_bytes(s), dtype=torch.uint8, device=device)
            for s in widths
        }
        self._launch = {s: build_ffn(key) for s, key in self.keys.items()}
        self._compiled: dict[int, Any] = {}

    def _args(self, s, part, res_in, res_out, out, w, mode):
        return (
            part.data_ptr(),
            res_in.data_ptr(),
            res_out.data_ptr(),
            out.data_ptr(),
            w.post_norm.data_ptr(),
            w.router.data_ptr(),
            w.router_bias.data_ptr(),
            w.w13.data_ptr(),
            w.s13.data_ptr(),
            w.w2.data_ptr(),
            w.s2.data_ptr(),
            self.scratch[s].data_ptr(),
            self.sym,
            self.peers.data_ptr(),
            self.rank,
            self.epoch.data_ptr(),
            mode,
            torch.cuda.current_stream(),
        )

    def warm(self, weights) -> None:
        """Compile and load every width (a launch that returns at once)."""
        import flydsl.compiler as flyc

        dummy = torch.zeros(1, HIDDEN, dtype=torch.bfloat16, device=self.epoch.device)
        for s in self.widths:
            if s not in self._compiled:
                args = self._args(s, dummy, dummy, dummy, dummy, weights, 0)
                self._compiled[s] = flyc.compile(self._launch[s], *args)

    def __call__(self, part: torch.Tensor, residual: torch.Tensor, weights):
        """``part``: this rank's unreduced o_proj output; returns the reduced
        MoE output and the residual after attention."""
        s = part.shape[0]
        assert part.is_contiguous() and residual.is_contiguous()
        out = torch.empty_like(part)
        res_out = torch.empty_like(residual)
        self._compiled[s](*self._args(s, part, residual, res_out, out, weights, 1))
        return out, res_out

    def region(self, s: int, name: str) -> torch.Tensor:
        off, nbytes = scratch_layout(s)[name]
        return self.scratch[s][off : off + nbytes]
