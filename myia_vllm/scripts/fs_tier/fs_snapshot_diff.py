#!/usr/bin/env python3
"""M1 instrument for jsboige/vllm#198 — per-group fs KV tier read-back.

Takes a fresh mtime snapshot of the fs tier, diffs it against a baseline
snapshot taken earlier, and reports per group and per size class:

  created  present now, absent at baseline
  hit      in both, mtime changed (TouchOnHitFsTier bumped it => read back
           in the window)
  stable   in both, mtime unchanged (written, never read back)
  deleted  in baseline, absent now (pruned)

Layout: <root>/<model>_<hash>_r<rank>/<3 hex>/<2 hex>_g<gen>/<block>.bin
Group: g3 = attention, g0-g2 = Gated DeltaNet state snapshots.

Runs INSIDE the engine container (read-only: one `find` + stdin).
The baseline file lives on the host and is piped on stdin:

  MSYS_NO_PATHCONV=1 docker exec -i myia_vllm-medium-swift15-27b \
    python3 -c "$(cat /d/vllm/myia_vllm/scripts/fs_tier/fs_snapshot_diff.py)" \
    /kv-tier-fs - < /d/vllm/.tmp/fs_snapshot_20261025T1759Z.txt

(The repo is not mounted in the container — only myia_vllm/kv_tier is,
read-only — hence `python3 -c "$(cat …)"` with the baseline on stdin.)

Usage: fs_snapshot_diff.py TIER_ROOT BASELINE
  BASELINE = path to a snapshot file, or "-" for stdin.
Snapshot format: one line per file, `%T@ %s %p` (mtime epoch, size, path).

Output: a compact report (~12 lines). Exit 0 on success.
"""

import datetime
import os
import re
import subprocess
import sys
import time
from collections import defaultdict

MIB = 1024 * 1024


def parse_snapshot(text):
    """Parse `%T@ %s %p` lines into {path: (mtime, size)}."""
    out = {}
    for line in text.splitlines():
        parts = line.split(" ", 2)
        if len(parts) != 3:
            continue
        try:
            out[parts[2]] = (float(parts[0]), int(parts[1]))
        except ValueError:
            continue
    return out


def take_snapshot(root):
    out = subprocess.run(
        ["find", root, "-name", "*.bin", "-printf", "%T@ %s %p\n"],
        capture_output=True,
        text=True,
        check=True,
    )
    return parse_snapshot(out.stdout)


def group_of(path):
    # .../<3 hex>/<2 hex>_g<N>/<sha256>.bin
    try:
        seg = path.split("/")[-2]
        return "g" + seg.rsplit("_g", 1)[1]
    except (IndexError, ValueError):
        return "g?"


def bucket_of(group):
    if group == "g3":
        return "attn"
    if group in ("g0", "g1", "g2"):
        return "GDN"
    return "other"


def class_of(size):
    mib = size / MIB
    if mib < 8:
        return "s<8M"
    if mib < 32:
        return "a8-32M"
    if mib < 128:
        return "g32-128M"
    return "L>128M"


CLASSES = ("s<8M", "a8-32M", "g32-128M", "L>128M")
BUCKETS = ("attn", "GDN", "other")


def parse_t0(baseline_arg, explicit):
    """T0 of the window: explicit arg, or the timestamp in the file name."""
    if explicit:
        return datetime.datetime.strptime(explicit, "%Y-%m-%dT%H:%MZ").replace(
            tzinfo=datetime.timezone.utc
        ).timestamp()
    m = re.search(r"(\d{8}T\d{4}Z)", os.path.basename(baseline_arg or ""))
    if m:
        return datetime.datetime.strptime(m.group(1), "%Y%m%dT%H%MZ").replace(
            tzinfo=datetime.timezone.utc
        ).timestamp()
    return None


def main():
    if len(sys.argv) not in (3, 4):
        sys.exit("usage: fs_snapshot_diff.py TIER_ROOT BASELINE(- for stdin) "
                 "[T0_ISO e.g. 2026-10-25T17:59Z]")
    root, baseline_arg = sys.argv[1], sys.argv[2]
    if baseline_arg == "-":
        baseline_text = sys.stdin.read()
        label = sys.argv[3] if len(sys.argv) == 4 else "stdin"
    else:
        with open(baseline_arg) as f:
            baseline_text = f.read()
        label = os.path.basename(baseline_arg)

    t1 = time.time()
    new = take_snapshot(root)
    base = parse_snapshot(baseline_text)
    t0 = parse_t0(None if baseline_arg == "-" else baseline_arg,
                  sys.argv[3] if len(sys.argv) == 4 else None)

    counts = defaultdict(lambda: defaultdict(int))  # (bucket,class)->{field: value}
    for path, (mt, size) in new.items():
        key = (bucket_of(group_of(path)), class_of(size))
        if path not in base:
            counts[key]["created"] += 1
            counts[key]["created_b"] += size
        elif mt > base[path][0]:
            counts[key]["hit"] += 1
            counts[key]["hit_b"] += size
        else:
            counts[key]["stable"] += 1
            counts[key]["stable_b"] += size
    for path in base:
        if path not in new:
            key = (bucket_of(group_of(path)), class_of(base[path][1]))
            counts[key]["deleted"] += 1
            counts[key]["deleted_b"] += base[path][1]

    def line(bucket, klass, c):
        n = {f: c[f"{f}"] for f in ("created", "hit", "stable", "deleted")}
        gb = {f: c[f"{f}_b"] for f in n}
        denom = n["hit"] + n["stable"]
        hitpct = f"{100.0 * n['hit'] / denom:.1f}%" if denom else "-"
        return (f"{bucket:<6} {klass:<9} {n['created']:>7} {n['hit']:>5} "
                f"{n['stable']:>7} {n['deleted']:>5}  {hitpct:>6}  "
                f"c={gb['created'] / MIB / 1024:8.2f}G h={gb['hit'] / MIB / 1024:7.2f}G")

    window = f"{(t1 - t0) / 60:.0f}min" if t0 else "unknown"
    print(f"== M1 fs-tier mtime diff ==  baseline={label}  "
          f"window={window}  T1={time.strftime('%Y-%m-%d %H:%MZ', time.gmtime(t1))}")
    print(f"baseline files={len(base)} now={len(new)}")
    print(f"{'grp':<6} {'class':<9} {'created':>7} {'hit':>5} "
          f"{'stable':>7} {'deleted':>5}  {'hit%':>6}  GB")
    for b in BUCKETS:
        for k in CLASSES:
            c = counts.get((b, k))
            if c and any(c.values()):
                print(line(b, k, c))
    tot = defaultdict(int)
    for c in counts.values():
        for k, v in c.items():
            tot[k] += v
    denom = tot["hit"] + tot["stable"]
    hitpct = f"{100.0 * tot['hit'] / denom:.1f}%" if denom else "-"
    print(f"{'TOTAL':<6} {'-':<9} {tot['created']:>7} {tot['hit']:>5} "
          f"{tot['stable']:>7} {tot['deleted']:>5}  {hitpct:>6}  "
          f"c={tot['created_b'] / MIB / 1024:8.2f}G h={tot['hit_b'] / MIB / 1024:7.2f}G")
    print("hit% = hit/(hit+stable); created excluded (no baseline mtime).")


if __name__ == "__main__":
    main()
