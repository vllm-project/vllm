# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The KV-balanced MLA core (``kv_balanced_mla``) for the specialized decode under
DCP, selected with ``additional_config={"mla_core": "kv_balanced"}``.

It replaces tokenspeed's fixed split count (sized from the deployment's longest
sequence, so a typical request fills a few clusters) with a device plan from this
rank's KV shard lengths. As in KIMI_K3_KV_BALANCED_MLA, the first MLA layer of a
forward plans and every later one reuses the plan (all MLA layers of a step share
the lengths); it is captured with the decode graph. It writes the DcpBox rows and
log2 LSE that tokenspeed would, PDL chained between MlaQuery and DcpValue as
tokenspeed is, so the rest of the layer is unchanged.

``VLLM_K3_SPEC_KVB="capacity:min_split_tokens:max_waves"`` sets the planner bounds
(default 16:512:8: the merged backend's 8:1024:8 under-fills the GPU at the
specialized decode's small batches)."""

import os

import torch

from vllm.forward_context import get_forward_context, is_forward_context_available

from ....kv_balanced_mla import core as kvb

DEFAULT_BOUNDS = (16, 512, 8)
# Per (device, heads, q): the core and its plan buffers, shared by every MLA layer.
_CORES: dict[tuple, tuple] = {}
# Per device: the split-KV scratch, sized before capture for every core.
_WORKSPACE: dict[torch.device, torch.Tensor] = {}
# Per plan buffer (``work``): the (forward context, KV lengths) of the plan it holds.
# The context is held so that its identity can't be reused by a later forward.
_PLANNED: dict[int, tuple[object, int]] = {}


def planner_bounds() -> tuple[int, int, int]:
    value = os.environ.get("VLLM_K3_SPEC_KVB")
    if not value:
        return DEFAULT_BOUNDS
    capacity, min_split, waves = (int(x) for x in value.split(":"))
    return capacity, min_split, waves


def balanced_cores(core, dcp, heads, pitches, queries, device):
    """One MLA layer's KV-balanced cores per query length q for ``core ==
    "kv_balanced"``, else None. ``heads``: this rank's heads; ``pitches``: the
    layer's batches (tokens)."""
    if core != "kv_balanced":
        return None
    if dcp is None:
        raise ValueError("Kimi K3 MLA core kv_balanced needs DCP")
    most = max(pitches, default=1)
    return {
        q: BalancedCore(dcp.size * heads, q, most // q, device)
        for q in queries
        if most >= q
    }


class BalancedCore:
    """The KV-balanced decode of ``heads`` (DCP x local) heads at ``q`` tokens per
    request for up to ``max_requests`` requests, writing the DcpBox rows and LSE."""

    def __init__(self, heads, q, max_requests, device):
        key = (device, heads, q)
        if key not in _CORES:
            capacity, min_split, waves = planner_bounds()
            core = kvb.KvBalancedMla(
                heads,
                q,
                max_requests,
                True,
                device,
                capacity=capacity,
                min_split_tokens=min_split,
                max_waves=waves,
                pdl=True,
            )
            i32 = dict(dtype=torch.int32, device=device)
            counts = torch.zeros(max_requests, **i32)
            work = torch.full((core.list_len,), -1, **i32)
            _CORES[key] = core, counts, work
            held = _WORKSPACE.get(device)
            if held is None or held.numel() < core.workspace_bytes:
                if torch.cuda.is_current_stream_capturing():
                    raise RuntimeError(
                        "KV-balanced scratch must be sized before capture"
                    )
                _WORKSPACE[device] = torch.empty(
                    core.workspace_bytes, dtype=torch.int8, device=device
                )
        self.core, self.counts, self.work = _CORES[key]
        self.device = device

    def run(self, box, query, cache, block_table, lengths, totals, scales, out, lse):
        """The first ``reqs = lse.shape[0]`` requests' ``query`` rows [reqs, q, H, 576]
        (FP8) over this rank's shard of the KV (``lengths``), masked causally in
        global positions (``totals``, the full lengths), into the DcpBox ``out`` rows
        [reqs, q, H, 512] (BF16) and ``lse`` [reqs, q, H] (log2)."""
        reqs = lse.shape[0]
        lengths, block_table, totals = lengths[:reqs], block_table[:reqs], totals[:reqs]
        # One plan per forward, made by its first MLA layer; calls outside a forward
        # (compile warmup) plan for themselves.
        context = get_forward_context() if is_forward_context_available() else None
        held = _PLANNED.get(self.work.data_ptr())
        if (
            context is None
            or held is None
            or held[0] is not context
            or held[1] != lengths.data_ptr()
        ):
            self.core.plan(lengths, self.counts, self.work)
            _PLANNED[self.work.data_ptr()] = (context, lengths.data_ptr())
        softmax, output_scale = scales
        self.core.decode(
            query.view(-1, self.core.heads, 576),
            cache,
            block_table,
            lengths,
            softmax,
            output_scale,
            self.counts,
            self.work,
            _WORKSPACE[self.device],
            lse=lse,
            causal_lens=totals,
            cp=(box.size, box.rank, box.interleave),
            out=out,
        )
