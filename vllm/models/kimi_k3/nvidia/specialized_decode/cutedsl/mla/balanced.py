# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The KV-balanced MLA core (``kv_balanced_mla``) for the specialized decode under
DCP, selected with ``additional_config={"mla_core": "kv_balanced"}``.

It replaces tokenspeed's fixed split count (sized from the deployment's longest
sequence, so a typical request fills a few clusters) with a device plan from this
rank's KV shard lengths. The plan is made once per step by the first MLA layer and
reused by every later one (all MLA layers of a step share the lengths); it is
captured with the decode graph. ``VLLM_K3_SPEC_KVB="capacity:min_split_tokens:
max_waves"`` sets the planner bounds (default 16:512:8, tuned for the specialized
decode's small batches)."""

import os

import torch

from ....kv_balanced_mla import core as kvb

DEFAULT = (16, 512, 8)
# Per (device, heads, q): the core and its plan buffers, shared by every MLA layer.
_CORES: dict[tuple, tuple] = {}
# Per device: the split-KV scratch, sized before capture for every core.
_WORKSPACE: dict[torch.device, torch.Tensor] = {}


def planner_bounds() -> tuple[int, int, int]:
    value = os.environ.get("VLLM_K3_SPEC_KVB")
    if not value:
        return DEFAULT
    capacity, min_split, waves = (int(x) for x in value.split(":"))
    return capacity, min_split, waves


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
                # PDL chained between MlaQuery and DcpValue, as tokenspeed is.
                pdl=os.environ.get("VLLM_K3_SPEC_KVB_PDL", "1") == "1",
            )
            i32 = dict(dtype=torch.int32, device=device)
            counts = torch.zeros(max_requests, **i32)
            work = torch.full((core.list_len,), -1, **i32)
            _CORES[key] = core, counts, work
            held = _WORKSPACE.get(device)
            if held is None or held.numel() < core.workspace_bytes:
                if torch.cuda.is_current_stream_capturing():
                    raise RuntimeError("KV-balanced scratch must be sized before capture")
                _WORKSPACE[device] = torch.empty(
                    core.workspace_bytes, dtype=torch.int8, device=device
                )
        self.core, self.counts, self.work = _CORES[key]
        self.device = device

    def run(self, query, cache, block_table, lengths, totals, scales, out, lse, cp,
            plan):
        """``query`` [B, H, 576] FP8 rows (q per request) over this rank's shard
        (``lengths``) masked causally in global positions (``totals``) into ``out``
        [B, H, 512] BF16 and ``lse`` [reqs, q, H] (log2). ``plan``: make this step's
        plan first (the step's first MLA layer)."""
        reqs = lengths.shape[0]
        if plan:
            self.core.plan(lengths, self.counts, self.work)
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
            lse=lse.view(reqs, self.core.q, self.core.heads),
            causal_lens=totals,
            cp=cp,
            out=out,
        )
