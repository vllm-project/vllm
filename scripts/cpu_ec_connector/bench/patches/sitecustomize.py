# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Descriptor-count instrumentation for the EC offload benchmark.

Wraps `_coalesce_runs` -- the function that decides how many descriptors an
entry's blocks collapse into -- and records `blocks in` against `descriptors
out`. That ratio is the fragmentation metric: a contiguous entry yields one
descriptor for hundreds of blocks, a scattered one yields nearly as many
descriptors as blocks. Effective bandwidth alone cannot distinguish
fragmentation from PCIe contention or NUMA placement; this can.

The connector is not modified. Both call sites resolve `_coalesce_runs`
through module globals at call time, so replacing the module attribute is
enough.

This file lives in its own directory precisely so it is NOT auto-imported by
the benchmark scripts one level up: Python puts a script's own directory on
`sys.path`, and any `sitecustomize` found there loads automatically. Put
*this* directory on PYTHONPATH only for the server process.

Unset `EC_BENCH_FRAG_FILE` and this module does nothing, so a leaked
PYTHONPATH is harmless.

Enabling this adds a Python frame and a dict update per call -- about 1 us
against a ~40 us descriptor build. Small, but do not report latencies from a
patched process: run the timed arms unpatched and diagnose fragmentation in a
separate run.
"""

import atexit
import json
import os
import sys
import time

_OUT_PATH = os.environ.get("EC_BENCH_FRAG_FILE")
_FLUSH_INTERVAL_S = float(os.environ.get("EC_BENCH_FRAG_FLUSH_S", "10"))


def _install(out_path):
    from vllm.distributed.ec_transfer.ec_connector.cpu import worker as worker_mod

    original = worker_mod._coalesce_runs
    # (elapsed_second, caller) -> [calls, blocks_in, descriptors_out]
    buckets: dict[tuple[int, str], list[int]] = {}
    started = time.time()
    state = {"last_flush": 0.0}
    pid = os.getpid()

    def flush() -> None:
        if not buckets:
            return
        rows = [
            {
                "pid": pid,
                "t_s": second,
                "caller": caller,
                "calls": calls,
                "blocks": blocks,
                "descriptors": descriptors,
                # 1.0 means every block needed its own descriptor (worst case);
                # blocks/descriptors is the mean run length.
                "blocks_per_descriptor": round(blocks / descriptors, 2)
                if descriptors
                else 0,
            }
            for (second, caller), (calls, blocks, descriptors) in sorted(
                buckets.items()
            )
        ]
        buckets.clear()
        try:
            with open(out_path, "a", encoding="utf-8") as f:
                f.write(
                    "".join(json.dumps(r, separators=(",", ":")) + "\n" for r in rows)
                )
        except OSError:
            pass  # never let instrumentation break the server

    def wrapped(block_ids):
        slots, first_blocks, num_blocks = original(block_ids)
        # The caller's name separates the save path from the load path without
        # threading a flag through the connector.
        caller = sys._getframe(1).f_code.co_name
        elapsed = time.time() - started
        entry = buckets.get((int(elapsed), caller))
        if entry is None:
            entry = buckets[(int(elapsed), caller)] = [0, 0, 0]
        entry[0] += 1
        entry[1] += len(block_ids)
        entry[2] += int(slots.size)
        # Periodic rather than signal-driven: vLLM owns SIGTERM, and a flushed
        # prefix survives even a SIGKILL.
        if elapsed - state["last_flush"] >= _FLUSH_INTERVAL_S:
            state["last_flush"] = elapsed
            flush()
        return slots, first_blocks, num_blocks

    worker_mod._coalesce_runs = wrapped
    atexit.register(flush)
    print(f"[ec-bench] descriptor counting active -> {out_path}", file=sys.stderr)


if _OUT_PATH:
    try:
        _install(_OUT_PATH)
    except ImportError as exc:
        # Probe interpreters that never load vllm land here; nothing to patch.
        print(f"[ec-bench] skipping instrumentation ({exc})", file=sys.stderr)
