# SPDX-License-Identifier: Apache-2.0
"""Cascade (shared-prefix) decode attention glue for ROCm backends.

Sibling requests in agentic traffic share long block-aligned prefixes.
Prefix caching stores that KV once; a plain decode kernel still streams
it once per request, and decode attention is bound by those streams.
Cascade attention streams each shared block once per group and merges
in log-sum-exp space -- the same identity split-KV decode relies on.
The kernels and the planner live in AITER
(`aiter.ops.flydsl.pa_decode_cascade{,_plan}`); this module owns the
per-batch-bucket workspaces, the plain-4D layout mapping from vLLM
paged caches, and the decode-row routing for mixed steps.

Enabled with ``VLLM_ROCM_CASCADE_DECODE=1`` (pure-decode steps) and
``VLLM_ROCM_CASCADE_MIXED=1`` (decode rows of mixed prefill/decode
steps). Both default to off while the path is validated. Nothing runs
-- at all, including buffer allocation -- with the flags off.
"""

from __future__ import annotations

import os

import torch

import vllm.envs as envs
from vllm.logger import init_logger

logger = init_logger(__name__)

_PLANNER_TILE_LIMIT = 4096


def _cascade_aiter():
    try:
        from aiter.ops.flydsl.pa_decode_cascade import (  # noqa: F401
            pa_decode_cascade,
            plan_pa_decode_cascade,
        )
        from aiter.ops.flydsl.pa_decode_cascade_plan import (  # noqa: F401
            fill_pa_decode_cascade_plan,
            plan_pa_decode_cascade_groups,
        )
    except ImportError:
        return None
    return (
        pa_decode_cascade,
        plan_pa_decode_cascade,
        fill_pa_decode_cascade_plan,
        plan_pa_decode_cascade_groups,
    )


class CascadeState:
    """Per (geometry, batch bucket) cascade workspace and planner."""

    def __init__(
        self,
        device,
        num_q_heads: int,
        num_kv_heads: int,
        head_dim: int,
        table_width: int,
        max_reqs: int,
    ):
        fns = _cascade_aiter()
        if fns is None:
            raise RuntimeError(
                "aiter.pa_decode_cascade ops not importable"
            )
        (
            self.pa_decode_cascade,
            plan,
            self.fill,
            self.plan_planner,
        ) = fns
        self.device = device
        self.num_q_heads = num_q_heads
        self.num_kv_heads = num_kv_heads
        # Items cover groups and singletons (whole valid runs); singles
        # remain for overflow-only fallback.
        max_items = max(1, max_reqs)
        self.ws = plan(
            device,
            num_q_heads=num_q_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            max_items=max_items,
            max_singles=max_reqs,
            table_width=table_width,
            kv_layout="plain4d",
        )
        self.pl = self.plan_planner(
            device,
            max_rows=max_reqs,
            table_width=table_width,
            max_b_rows=self.ws.max_b_rows,
        )
        self.max_reqs = max_reqs
        self.table_width = table_width
        self.filled_epoch = 0
        self.consumed_epoch = 0

    def chunk_out(self, num_rows: int, shape, dtype, device) -> torch.Tensor:
        return chunk_out(num_rows, shape, dtype, device)

    def ensure_filled(self, block_table, seq_lens, q_map_of_req=None, *, ring=False):
        """Fill cascade metadata if the builder did not already fill this step."""
        if self.filled_epoch > self.consumed_epoch:
            return False
        if ring:
            _ring_record(block_table, seq_lens, q_map_of_req)
        self.fill(self.ws, self.pl, block_table, seq_lens, q_map_of_req=q_map_of_req)
        self.filled_epoch += 1
        self.consumed_epoch = self.filled_epoch
        return True

    def mark_consumed_with(self, generation: int) -> None:
        if generation > self.filled_epoch:
            self.filled_epoch = generation
        self.consumed_epoch = max(self.consumed_epoch, generation)

    def run(self, output, query, k_cache, v_cache, seq_lens, block_table, scale, sinks):
        ring = os.environ.get("VLLM_ROCM_CASCADE_RING") is not None
        prof = os.environ.get("AITER_CASCADE_PROF")
        if prof:
            import time

            t0 = time.perf_counter()
        self.ensure_filled(block_table, seq_lens, ring=ring)
        if prof:
            torch.cuda.synchronize()
            print(f"[cascade-prof] fill={(time.perf_counter()-t0)*1000:6.2f}")
        self.pa_decode_cascade(
            output,
            query,
            k_cache,
            v_cache,
            seq_lens,
            block_table,
            scale,
            sinks=sinks,
            workspace=self.ws,
        )


# One chunk staging buffer per (tail shape, dtype), shared by every
# attention instance -- layers run sequentially, so one buffer each.
_CHUNK_BUFS: dict[tuple, torch.Tensor] = {}


def chunk_out(num_rows: int, shape, dtype, device) -> torch.Tensor:
    key = (device, dtype, shape[1:])
    rows_needed = (num_rows + 1023) // 1024 * 1024
    buf = _CHUNK_BUFS.get(key)
    if buf is None or buf.shape[0] < num_rows:
        _CHUNK_BUFS[key] = torch.empty(
            (rows_needed, *shape[1:]), dtype=dtype, device=device
        )
        buf = _CHUNK_BUFS[key]
    return buf[:num_rows]


_STATES: dict[tuple, CascadeState] = {}
_GATE_REASON: str | None = None


def _log_once(level: int, msg: str, *args) -> None:
    if not getattr(_log_once, f"_{msg}", False):
        setattr(_log_once, f"_{msg}", True)
        logger.info_once(msg, *args, scope="local")


def eligible(
    impl,
    attn_metadata,
    *,
    mixed: bool,
) -> bool:
    """Whether a cascade call may handle this layer and step."""
    global _GATE_REASON
    if not (envs.VLLM_ROCM_CASCADE_DECODE or envs.VLLM_ROCM_CASCADE_MIXED):
        return False
    if _cascade_aiter() is None:
        _GATE_REASON = "aiter cascade ops are not importable"
        return False
    if impl.sliding_window not in (None, -1, (-1, -1)):
        _GATE_REASON = "cascade is off for sliding-window layers"
        return False
    if impl.alibi_slopes is not None:
        _GATE_REASON = "cascade is off for ALiBi"
        return False
    if impl.logits_soft_cap:
        _GATE_REASON = "cascade is off for logits soft cap"
        return False
    if impl.kv_cache_dtype not in (
        "auto",
        "bfloat16",
        "fp16",
        torch.bfloat16,
        torch.float16,
    ):
        _GATE_REASON = f"cascade is off for kv_cache_dtype={impl.kv_cache_dtype}"
        return False
    h = impl.head_size
    if h not in (64, 128, 256, 512):
        _GATE_REASON = f"cascade is off for head_size={h}"
        return False
    if impl.num_kv_heads <= 0 or impl.num_heads % impl.num_kv_heads:
        _GATE_REASON = "cascade needs a GQA group of 1..8 query heads"
        return False
    if impl.num_heads // impl.num_kv_heads > 8:
        _GATE_REASON = "cascade needs a GQA group of 1..8 query heads"
        return False
    if mixed and not envs.VLLM_ROCM_CASCADE_MIXED:
        return False
    if not mixed and not envs.VLLM_ROCM_CASCADE_DECODE:
        return False
    return True


def describe_blocked_call() -> str:
    return _GATE_REASON or "not attempted"


def _state_for(state_key, device, heads, kv_heads, dim, width, max_reqs) -> CascadeState:
    # Geometry-only key: every attention instance of one KV-cache group
    # shares one cascade state. Planner metadata is layer-independent;
    # each layer folds its own query through it in stream order.
    key = (heads, kv_heads, dim, width, max_reqs)
    st = _STATES.get(key)
    if st is None:
        st = CascadeState(device, heads, kv_heads, dim, width, max_reqs)
        _STATES[key] = st
        logger.info_once(
            "ROCm cascade decode workspace allocated: %s", key, scope="local"
        )
    return st


def decode_rows_maps(
    query_start_loc: torch.Tensor,
    seq_lens: torch.Tensor,
    num_actual_tokens: int | None = None,
):
    """Decode-row routing for a mixed step (torch ops, eager-only).

    Returns (dec_req, dec_q_map, chk_req, chk_tok, chk_cu_q, chk_len_k):
    request rows with query length 1, their token rows, and the same
    infrastructure for the chunk rows that the unified kernel keeps.
    """
    qsl = query_start_loc
    qlens = qsl[1:] - qsl[:-1]
    is_dec = qlens.eq(1)
    is_chk = ~is_dec
    dec_req = torch.nonzero(is_dec, as_tuple=False).squeeze(1).to(torch.int32)
    chk_req = torch.nonzero(is_chk, as_tuple=False).squeeze(1)
    dec_q_map = qsl[:-1][dec_req.long()].to(torch.int32)
    if chk_req.numel() == 0:
        empty = torch.empty(0, dtype=torch.int32, device=qsl.device)
        return dec_req, dec_q_map, empty, empty, empty.to(qsl.dtype), empty
    chk_qlens = qlens[chk_req]
    chk_cu = torch.cat(
        [
            torch.zeros(1, dtype=qlens.dtype, device=qsl.device),
            torch.cumsum(chk_qlens, 0),
        ]
    ).to(qsl.dtype)
    total_tokens = int(qsl[-1] - qsl[0])
    if num_actual_tokens is not None:
        total_tokens = min(total_tokens, int(num_actual_tokens))
    tok_of_req = (
        torch.searchsorted(qsl, torch.arange(total_tokens, device=qsl.device), right=True)
        - 1
    )
    chk_tok = torch.nonzero(~is_dec[tok_of_req], as_tuple=False).squeeze(1).to(torch.int32)
    chk_len_k = seq_lens[chk_req.long()]
    return dec_req, dec_q_map, chk_req, chk_tok, chk_cu, chk_len_k


# Rolling record of planner inputs for fault forensics (enabled with
# VLLM_ROCM_CASCADE_RING). Keeps the last N lens tables on host.
_RING_KEEP = 24
_ring_buf = []
_ring_i = 0


def _ring_record(bt, lens, q_map=None):
    keep = bt.to("cpu", non_blocking=True), lens.to("cpu", non_blocking=True), (
        None if q_map is None else q_map.to("cpu", non_blocking=True)
    )
    global _ring_i
    _ring_buf.append((keep[1], keep[0], keep[2], torch.cuda.Event()))
    if len(_ring_buf) > _RING_KEEP:
        _ring_buf.pop(0)
    import os
    path = os.environ.get("VLLM_ROCM_CASCADE_RING_PATH", "/tmp/cascade_ring")
    os.makedirs(path, exist_ok=True)
    # fire-and-forget append
    torch.save((keep[1].to("cpu"), keep[0].to("cpu"), None if keep[2] is None else keep[2].to("cpu")),
               f"{path}/ring_{_ring_i:06d}.pt")
    _ring_i += 1


_FILL_GENERATION = [0]


def maybe_fill_step(
    ws_builder_state,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    max_query_len: int,
):
    """Builder-side per-step planner fill for the shared cascade state.

    Called from an attention metadata builder's build(): advances the
    shared state's fill generation so the group's attention instances
    skip their own fill. Off entirely with the cascade flags off.
    """
    if not (envs.VLLM_ROCM_CASCADE_DECODE or envs.VLLM_ROCM_CASCADE_MIXED):
        return
    if max_query_len != 1:
        # Not a pure-decode step: the mixed path fills its own decode
        # subset per call; the pure fill over the full batch would not be
        # consumed by it.
        return
    if _cascade_aiter() is None:
        # Cascade ops unavailable: run a stock path everywhere (no state
        # allocation, no fill), no failure.
        return
    st = ws_builder_state
    prof = os.environ.get("AITER_CASCADE_PROF")
    if prof:
        import time

        t0 = time.perf_counter()
    gen = _FILL_GENERATION[0] + 1
    _FILL_GENERATION[0] = gen
    st.fill(st.ws, st.pl, block_table, seq_lens)
    st.mark_consumed_with(gen)
    if prof:
        torch.cuda.synchronize()
        c = st.ws.counts.cpu().tolist()
        print(f"[cascade-prof] builder fill gen={gen} time={(time.perf_counter()-t0)*1000:5.2f} items={c[0]} members={c[1]} singles={c[2]}")
    prof = os.environ.get("AITER_CASCADE_PROF")
    if prof:
        import time

        print(f"[cascade-prof] builder fill gen={gen}")
