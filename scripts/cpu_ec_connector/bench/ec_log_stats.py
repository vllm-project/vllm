# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Read EC connector accounting out of a vLLM server log.

The connector reports every batched DMA on a DEBUG line, so the server must
run with `VLLM_LOGGING_LEVEL=DEBUG` or nothing here finds anything. Encoder
inputs come from the per-step iteration line that
`--enable-logging-iteration-details` enables.

Callers take `os.path.getsize()` before and after a workload and pass that
byte range to `read_slice`, which keeps one pass's accounting separate from
the next's.
"""

from __future__ import annotations

import re
import statistics
from collections.abc import Iterator
from datetime import datetime
from typing import NamedTuple

# vLLM's formatter: "LEVEL %m-%d %H:%M:%S [file:lineno] message" (vllm/logger.py).
# Year is absent, so timestamps are only ever used as deltas within one log.
_TS = r"\w+ (\d\d-\d\d \d\d:\d\d:\d\d) \["
_EC_XFER_RE = re.compile(
    _TS + r"[^\]]+\] EC (save|load): (\d+) entr\w+ \((\d+) bytes\) took ([\d.]+) ms"
)
# ECExampleConnector reports one line per item instead of the CPU connector's
# batched "EC load:" line, so a run over shared storage would otherwise read as
# zero loads.
_EC_EXAMPLE_LOAD_RE = re.compile(r"Success load encoder cache for hash")
_ENCODER_INPUTS_RE = re.compile(r"encoder inputs: (\d+)")
_ENCODER_EMBEDS_RE = re.compile(r"encoder output embeddings: (\d+)")

_TS_FORMAT = "%m-%d %H:%M:%S"


class Transfer(NamedTuple):
    """One batched DMA the worker reported as complete."""

    direction: str  # "save" (D2H) or "load" (H2D)
    entries: int  # encoder-cache entries in this batch
    num_bytes: int
    ms: float

    @property
    def gbps(self) -> float:
        """Effective bandwidth. Decay in this number is the fragmentation signal:
        a coalesced entry runs near 53 GB/s, a fully scattered one near 4.4."""
        return self.num_bytes / (self.ms / 1000) / 1e9 if self.ms else 0.0


def read_slice(path: str, start: int, end: int) -> str:
    with open(path, "rb") as f:
        f.seek(start)
        return f.read(end - start).decode("utf-8", errors="replace")


def iter_transfers(text: str) -> Iterator[tuple[float, Transfer]]:
    """Yield `(seconds_since_first_transfer, Transfer)` in log order."""
    first: datetime | None = None
    for ts, direction, entries, nbytes, ms in _EC_XFER_RE.findall(text):
        when = datetime.strptime(ts, _TS_FORMAT)
        if first is None:
            first = when
        yield (
            (when - first).total_seconds(),
            Transfer(direction, int(entries), int(nbytes), float(ms)),
        )


def summarize(text: str) -> dict:
    """Totals for one log slice."""
    out = {
        f"ec_{d}_{k}": 0
        for d in ("save", "load")
        for k in ("entries", "bytes", "transfers")
    }
    out["ec_save_ms"] = out["ec_load_ms"] = 0.0
    for _, t in iter_transfers(text):
        out[f"ec_{t.direction}_entries"] += t.entries
        out[f"ec_{t.direction}_bytes"] += t.num_bytes
        out[f"ec_{t.direction}_transfers"] += 1
        out[f"ec_{t.direction}_ms"] += t.ms
    for key in ("save", "load"):
        ms, nbytes = out[f"ec_{key}_ms"], out[f"ec_{key}_bytes"]
        out[f"ec_{key}_ms"] = round(ms, 3)
        out[f"ec_{key}_gbps"] = round(nbytes / (ms / 1000) / 1e9, 1) if ms else 0.0
    out["ec_example_loads"] = len(_EC_EXAMPLE_LOAD_RE.findall(text))
    out["encoder_inputs_computed"] = sum(
        int(m) for m in _ENCODER_INPUTS_RE.findall(text)
    )
    out["encoder_embeds_computed"] = sum(
        int(m) for m in _ENCODER_EMBEDS_RE.findall(text)
    )
    return out


def window_stats(text: str, window_s: float) -> list[dict]:
    """Per-window bandwidth, for watching fragmentation develop over a long run.

    A region that fragments as it churns describes the same bytes with more
    descriptors, which shows up here as `load_gbps` decaying window over
    window. Bandwidth alone cannot say *why* it decayed -- PCIe contention and
    NUMA placement land the same way -- so read it alongside the descriptor
    counts from the bench sitecustomize patch.
    """
    windows: dict[int, dict] = {}
    for offset, t in iter_transfers(text):
        w = windows.setdefault(
            int(offset // window_s),
            {
                "window": 0,
                "load_bytes": 0,
                "load_ms": 0.0,
                "load_entries": 0,
                "save_bytes": 0,
                "save_ms": 0.0,
                "save_entries": 0,
            },
        )
        w[f"{t.direction}_bytes"] += t.num_bytes
        w[f"{t.direction}_ms"] += t.ms
        w[f"{t.direction}_entries"] += t.entries
    out = []
    for idx in sorted(windows):
        w = windows[idx]
        w["window"] = idx
        w["t_start_s"] = round(idx * window_s, 1)
        for key in ("load", "save"):
            ms, nbytes, n = w[f"{key}_ms"], w[f"{key}_bytes"], w[f"{key}_entries"]
            w[f"{key}_gbps"] = round(nbytes / (ms / 1000) / 1e9, 1) if ms else 0.0
            w[f"{key}_ms_per_entry"] = round(ms / n, 3) if n else 0.0
            w[f"{key}_ms"] = round(ms, 3)
        out.append(w)
    return out


# The modality word changed from "image" to "media" when the proxy gained
# video support, so match whatever noun sits between the count and "item(s)".
_REWROTE_RE = re.compile(r"Rewrote (\d+) \w+ item\(s\)")


def rewritten_items(text: str) -> int:
    """How many image items the EPD proxy replaced with a grid reference.

    The proxy logs this per request and falls back per *item*: one whose encoder
    reported no metadata is forwarded with its pixels intact. So the presence of
    the line proves only that at least one item was rewritten -- the count is
    what says whether the configuration actually covered the workload.
    """
    return sum(int(n) for n in _REWROTE_RE.findall(text))


def stage_summary(text: str) -> dict:
    """Median of each stage the EPD proxy times, in ms.

    The proxy logs one line per request, e.g.

        STAGE rewrite encode=91.2 rewrite=0.7 decode_ttfb=310.4 decode_total=980.1

    Field names differ between its streaming and non-streaming paths, so they are
    read as whatever key=value pairs the line carries rather than assumed. Medians
    because a single slow request should not move the number.
    """
    per_field: dict[str, list[float]] = {}
    modes: set[str] = set()
    for line in text.splitlines():
        marker = line.find("STAGE ")
        if marker < 0:
            continue
        parts = line[marker + len("STAGE ") :].split()
        if parts:
            modes.add(parts[0])
        for part in parts[1:]:
            key, _, value = part.partition("=")
            try:
                per_field.setdefault(key, []).append(float(value))
            except ValueError:
                continue
    out: dict = {
        "requests": max((len(v) for v in per_field.values()), default=0),
        "modes": sorted(modes),
    }
    for key, values in per_field.items():
        out[f"{key}_ms_median"] = round(statistics.median(values), 2)
    return out


def decay_report(windows: list[dict], key: str = "load") -> dict:
    """First-vs-last window bandwidth, so a negative result reads as one.

    `ratio` below 1.0 means the direction got slower per byte over the run.
    """
    active = [w for w in windows if w[f"{key}_entries"]]
    if len(active) < 2:
        return {"windows_with_traffic": len(active), "verdict": "insufficient data"}
    first, last = active[0][f"{key}_gbps"], active[-1][f"{key}_gbps"]
    ratio = last / first if first else 0.0
    return {
        "windows_with_traffic": len(active),
        f"{key}_gbps_first": first,
        f"{key}_gbps_last": last,
        "ratio": round(ratio, 3),
        "verdict": (
            "no decay observed" if ratio >= 0.9 else f"decayed to {ratio * 100:.0f}%"
        ),
    }
