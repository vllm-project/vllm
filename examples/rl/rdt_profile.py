# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Two-level critical-path profiler for the sharded-RDT weight sync.

Why two levels? The naive "measure every sub-op and sum" approach CANNOT add up
to wall time here, for two reasons:

  * Overlap. gather(N+1) runs *inside* the update_weights(N) interval (software
    pipeline + double-buffered gather buffer). Summing gather + update_weights
    double-counts the hidden part.
  * Separate GPUs/processes. The producer's serve runs on the trainer node
    *inside* the window the consumer is blocked in ``ray.get``. Summing
    "producer time" + "pull time" double-counts, and comparing a producer clock
    to a consumer clock is meaningless.

The only thing that provably sums to wall is a SINGLE timeline. So:

LEVEL 1 -- CRITICAL PATH (additive, ground truth).
  Timed entirely on the driver's single async thread via ``timed(name)``. Every
  interval the driver blocks on lands in exactly one bucket; a hidden gather
  contributes 0 to its own bucket because it elapses *inside* the
  ``update_weights`` interval (we time what the driver WAITS FOR, not what runs
  somewhere). ``driver_misc = wall - sum(buckets)`` makes the buckets sum to the
  measured wall by construction. Immune to overlap and to cross-node clocks.

LEVEL 2 -- ATTRIBUTION (diagnostic, never re-added).
  Explains the big Level-1 buckets from per-process DURATIONS (never timestamps,
  so no cross-node clock subtraction): the consumer's pull/process split (from
  consumer.jsonl) and the producer's gather/serve split (from get_produce_timing).
  The producer's work is shown as the explanation of the consumer's
  ``produce_wait`` (produce is a blocking RPC the consumer ray.gets) -- it is
  NEVER added to the total. That is the double-count this whole design avoids.

  It also prints a PER-WORKER straggler table (bytes / transfer_s / TRUE GB/s =
  bytes/transfer, per worker) and auto-classifies any straggler: equal bytes but
  uneven transfer => TRANSPORT straggler (RoCE congestion; fix is NIC/fabric, not
  code); uneven bytes => EP IMBALANCE (fix = balance the shard). This is what
  settles "is a slow worker carrying more data, or just transferring slower?" --
  measured, not assumed. (Observed: transient transport stragglers on different
  flows each run, equal bytes; see multi_node_rdt.md Part VI.) The per-worker
  ``bytes`` come from the engine's ``_log_timing`` (pull_bytes) into the jsonl.

The transfer bucket itself is at the GPUDirect-RDMA speed-of-light on this
fabric (~24 GB/s/rail = ~48% of line rate, read AND write); see Part VI. The
remaining wall lever is hiding ``process`` (fp8 quant) behind the next pull, not
faster transfer.

Everything here is driver-side + small Ray tasks; nothing imports vLLM internals,
so it ships with the example via ``working_dir`` and needs no venv deploy. The
engine-side counters it reads (produce_wait/recv_wall split, process-phase split)
live in ``vllm/distributed/weight_transfer/{_nixl_profile,sharded_rdt_engine}.py``
and DO need redeploy -- see ``deploy_rdt_profiling.py``.
"""
import json
import time
from collections import defaultdict
from contextlib import contextmanager

# Level-1 driver buckets, in display order. ``update_weights`` is the only one
# guaranteed present every run; the rest may be 0 (e.g. gather_tail in a perfect
# overlap). driver_misc is computed in finish().
_L1_ORDER = [
    "start_weight_update",
    "gather_fill",
    "update_weights",
    "free_group",
    "gather_tail",
    "finish_weight_update",
    "driver_misc",
]
_L1_LABEL = {
    "gather_fill": "gather_fill (grp 0)",
    "update_weights": "update_weights (Σ grp)",
    "free_group": "free_group (Σ grp)",
    "gather_tail": "gather_tail (Σ grp)",
}
_L1_NOTE = {
    "update_weights": "◄ dominant: serialized across groups; gathers hidden under it",
    "gather_tail": "(residual gather NOT hidden; ~0 in steady state)",
    "gather_fill": "(pipeline fill: the one gather with nothing to overlap it)",
}


class CriticalPathProfiler:
    """Level-1 additive timer for ONE sync iteration, plus the Level-2 report.

    Usage (driver):
        cp = CriticalPathProfiler(sync_iter)
        cp.begin()
        with cp.timed("start_weight_update"): await engine.start_weight_update(...)
        ... loop with cp.timed("update_weights"/"free_group"/"gather_*") ...
        with cp.timed("finish_weight_update"): await engine.finish_weight_update()
        cp.finish()
        print(cp.report(consumer_jsonl=..., producer_timings=..., producer_nixl=...))
    """

    def __init__(self, sync_iter: int):
        self.sync_iter = sync_iter
        self.buckets: dict[str, float] = defaultdict(float)
        self._t0: float | None = None
        self.wall: float = 0.0

    def begin(self) -> None:
        self._t0 = time.perf_counter()

    @contextmanager
    def timed(self, name: str):
        t = time.perf_counter()
        try:
            yield
        finally:
            self.buckets[name] += time.perf_counter() - t

    def finish(self) -> None:
        assert self._t0 is not None, "begin() not called"
        self.wall = time.perf_counter() - self._t0
        # By construction: everything not explicitly timed (asyncio slop, the
        # instant gfuts dispatch, create_task) is driver_misc, so buckets sum to
        # wall EXACTLY.
        self.buckets["driver_misc"] = self.wall - sum(self.buckets.values())

    # ---------------- reporting ----------------
    def report(self, *, consumer_jsonl: str, producer_timings: list[dict],
               producer_nixl: list[dict], n_groups: int) -> str:
        out: list[str] = []
        w = self.wall or 1e-9

        def pct(v):
            return f"{100 * v / w:5.1f}%"

        out.append(f"\n══════ RDT weight-sync profile — iter {self.sync_iter} "
                   f"({'cold' if self.sync_iter == 0 else 'warm'}) ══════")
        out.append("CRITICAL PATH  (driver timeline; provably sums to WALL)")
        for k in _L1_ORDER:
            v = self.buckets.get(k, 0.0)
            if k == "update_weights" or v != 0.0 or k == "driver_misc":
                out.append(f"  {_L1_LABEL.get(k, k):<24}{v:8.3f}s {pct(v)}  "
                           f"{_L1_NOTE.get(k, '')}")
        s = sum(self.buckets.values())
        out.append(f"  {'─' * 52}")
        ok = "✓" if abs(s - self.wall) < 1e-6 else f"✗ Δ={s - self.wall:+.4f}"
        out.append(f"  {'WALL':<24}{self.wall:8.3f}s {pct(self.wall)}  "
                   f"(Σbuckets={s:.3f} {ok})")

        # ---- Level 2: attribution ----
        con = _aggregate_consumer(consumer_jsonl)
        out.append("")
        out.append("ATTRIBUTION  (explains the buckets above — NOT re-added; "
                   "all per-process durations)")
        if not con:
            out.append("  (no consumer records — inference-node jsonl collection "
                       "returned nothing; check the Ray reader task / path)")
            return "\n".join(out)

        # Slowest worker = the one that actually bounds each update_weights wall.
        slow = max(con, key=lambda p: con[p]["pull"] + con[p]["process"])
        c = con[slow]
        uw = self.buckets.get("update_weights", 0.0)
        pull, proc = c["pull"], c["process"]
        fanout = uw - (pull + proc)
        recv = c.get("recv_wall_seconds", 0.0)
        pw = c.get("produce_wait_seconds", 0.0)
        xfer = c.get("transfer_seconds", 0.0)
        split_live = (pw > 0 or recv > 0)

        out.append(f"  update_weights {uw:.3f}s  =  slowest worker(pull+process) "
                   f"{pull + proc:.3f}s  +  engine fan-out / worker skew {fanout:.3f}s")

        # ---- per-worker straggler table (TRUE GB/s from measured bytes) ----
        rows = []
        for pid, d in con.items():
            gib = d.get("bytes", 0.0) / (1024 ** 3)
            xf = d.get("transfer_seconds", 0.0)
            gbps = (d.get("bytes", 0.0) / 1e9 / xf) if xf else 0.0
            rows.append((d["pull"] + d["process"], pid, gib, xf, gbps,
                         d["process"], d["pull"]))
        rows.sort(reverse=True)
        out.append("  per-worker (sorted slowest→fastest; TRUE GB/s = bytes/transfer):")
        out.append(f"      {'pid':>7} {'GiB':>7} {'transfer_s':>10} {'GB/s':>6} "
                   f"{'pull_s':>7} {'proc_s':>7}")
        for _, pid, gib, xf, gbps, prc, pl in rows:
            out.append(f"      {pid:>7} {gib:>7.1f} {xf:>10.3f} {gbps:>6.1f} "
                       f"{pl:>7.3f} {prc:>7.3f}")
        # classify the straggler cause from the spread in bytes vs GB/s
        if len(rows) > 1:
            gibs = [r[2] for r in rows]
            xfs = [r[3] for r in rows]
            bw = [r[4] for r in rows]
            byte_spread = (max(gibs) - min(gibs)) / (sum(gibs) / len(gibs)) if sum(gibs) else 0
            xf_spread = (max(xfs) - min(xfs)) / (sum(xfs) / len(xfs)) if sum(xfs) else 0
            bw_spread = (max(bw) - min(bw)) / (sum(bw) / len(bw)) if sum(bw) else 0
            if xf_spread < 0.05:
                verdict = "balanced — no straggler"
            elif byte_spread > 0.5 * xf_spread:
                verdict = ("IMBALANCE — slow workers pull MORE bytes "
                           f"(byte spread {byte_spread:.0%}); fix = balance the EP shard")
            else:
                verdict = ("TRANSPORT straggler — similar bytes, lower GB/s "
                           f"(BW spread {bw_spread:.0%}); fix = NIC/placement, not bytes")
            out.append(f"      transfer spread {xf_spread:.0%} | byte spread "
                       f"{byte_spread:.0%} | BW spread {bw_spread:.0%}  ⇒ {verdict}")
        out.append(f"  └ consumer  (worker pid={slow}, slowest of {len(con)}; "
                   f"per-iter sums over {int(c['_n'])} pulls):")
        out.append(f"      pull                    {pull:8.3f}s")
        if split_live:
            descs_add = max(recv - xfer, 0.0)
            out.append(f"        produce_wait          {pw:8.3f}s   ⊂ BLOCKED on "
                       f"producer (explained below)")
            out.append(f"        recv_wall (NIXL read) {recv:8.3f}s")
            out.append(f"          transfer (RDMA)     {xfer:8.3f}s   ⊂ real RDMA, "
                       f"RoCE-variable — usually the #1 lever")
            out.append(f"          descs + add_agent   {descs_add:8.3f}s   "
                       f"⊂ recv_wall − transfer")
        else:
            out.append(f"        (produce_wait/recv_wall split UNAVAILABLE — engine "
                       f"not redeployed with mark_pull_start; pull−transfer below)")
            out.append(f"        transfer (RDMA)       {xfer:8.3f}s")
            out.append(f"        pull − transfer       {pull - xfer:8.3f}s   "
                       f"⊂ produce_wait + descs + poll")
        out.append(f"      process                 {proc:8.3f}s")
        ph = [("quant (pwa_loading)", "quant_seconds"),
              ("scatter (arena→param)", "scatter_seconds"),
              ("materialize (shells)", "materialize_seconds"),
              ("kernel_copy", "kernel_copy_seconds")]
        if any(c.get(k, 0.0) for _, k in ph):
            for lbl, k in ph:
                out.append(f"        {lbl:<21} {c.get(k, 0.0):8.3f}s")
        else:
            out.append("        (process-phase split UNAVAILABLE — engine not "
                       "redeployed with the PhaseTimer in _replay)")
        out.append(f"      register                {c.get('register_seconds', 0.0):8.3f}s "
                   f"({int(c.get('register_calls', 0))} regs)  "
                   f"◄ want ~0 (pre-registered arenas)")

        # producer side — explains produce_wait, already counted inside it.
        if producer_timings:
            pm = max(t.get("method_seconds", 0.0) for t in producer_timings)
            psl = max(t.get("slice_seconds", 0.0) for t in producer_timings)
            pwt = max(t.get("wait_seconds", 0.0) for t in producer_timings)
            plumb = (pw - pm) if split_live else None
            out.append(f"  └ producer  (trainer, slowest of {len(producer_timings)}) "
                       f"— EXPLAINS produce_wait, already counted, NOT re-added:")
            out.append(f"      gather-cache wait       {pwt:8.3f}s   "
                       f"⊂ want ~0 (gather done before produce)")
            out.append(f"      serve (slice/plan)      {psl:8.3f}s   "
                       f"⊂ want ~0 (cached serve plan, no op-chain replay)")
            out.append(f"      produce method total    {pm:8.3f}s")
            if plumb is not None:
                out.append(f"      ⇒ produce_wait − method {plumb:8.3f}s   "
                           f"= Ray dispatch RTT + producer meta cuda.sync")
        if producer_nixl:
            preg = max(n.get("register_seconds", 0.0) for n in producer_nixl)
            pregc = sum(int(n.get("register_calls", 0)) for n in producer_nixl)
            pext = max(n.get("extract_seconds", 0.0) for n in producer_nixl)
            out.append(f"      producer NIXL register  {preg:8.3f}s ({pregc} regs)  "
                       f"extract(meta)={pext:.3f}s  ◄ want ~0 regs")
        return "\n".join(out)


def _aggregate_consumer(jsonl_text: str) -> dict:
    """Sum every numeric field per worker pid over this iter's replay records.

    The jsonl is truncated per-iter on the inference node, so every non-baseline
    record here belongs to this iteration. Returns {pid: {field: sum, '_n': count}}.
    """
    by_pid: dict = {}
    for line in jsonl_text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except Exception:
            continue
        if r.get("mode") == "baseline":
            continue
        pid = r.get("pid")
        if pid is None:
            continue
        d = by_pid.setdefault(pid, defaultdict(float))
        d["_n"] += 1
        for k, v in r.items():
            if k in ("pid", "mode") or not isinstance(v, (int, float)):
                continue
            d[k] += v
    return by_pid


def make_consumer_file_tasks(node_ip: str, path: str):
    """Return (read, truncate) callables backed by Ray tasks pinned to ``node_ip``
    (the inference node), so the driver can read/clear the workers' local
    consumer.jsonl across nodes with no shared filesystem.

    The 8 inference workers append to ``path`` on their node; appends of one JSON
    line (<4 KiB) are atomic under O_APPEND so they never interleave. Truncate is
    called at each iter's start (before any worker writes that iter), so each read
    returns exactly one iteration's records.
    """
    import os

    import ray

    @ray.remote(num_cpus=0, resources={f"node:{node_ip}": 0.001})
    def _read() -> str:
        try:
            with open(path) as f:
                return f.read()
        except FileNotFoundError:
            return ""

    @ray.remote(num_cpus=0, resources={f"node:{node_ip}": 0.001})
    def _truncate() -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        open(path, "w").close()

    return (lambda: ray.get(_read.remote()),
            lambda: ray.get(_truncate.remote()))
