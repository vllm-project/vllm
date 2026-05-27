"""Summarize per-kernel time from rocprofv3 kernel_stats.csv files.

Usage:  python3 analyze_kernel_trace.py <triton_kernel_stats.csv> <cdna_kernel_stats.csv>

Prints, for each side:
- attention kernel(s): total ns, calls, avg, max
- KV-cache write kernel: total ns
- top-5 of all other kernels
Then a diff: attention CDNA-Triton (positive = CDNA slower).
"""

import csv
import sys
from collections import defaultdict


def load(path):
    with open(path) as f:
        rdr = csv.DictReader(f)
        rows = list(rdr)
    # Normalize: numbers as floats.
    for r in rows:
        for k in ("Calls", "TotalDurationNs", "AverageNs", "Percentage",
                  "MinNs", "MaxNs"):
            try:
                r[k] = float(r[k])
            except (KeyError, ValueError):
                r[k] = 0.0
        r["Name"] = r["Name"].strip('"')
    return rows


# Heuristic match for the attention kernel(s) on each side.
ATTENTION_KEYWORDS = ("attn", "attention", "unified", "paged_prefill",
                      "pth_decode", "_attn_fwd")
KV_WRITE_KEYWORDS = ("reshape_cache", "reshape_and_cache")


def match_any(name, keys):
    n = name.lower()
    return any(k in n for k in keys)


def summarize(label, rows):
    rows.sort(key=lambda r: r["TotalDurationNs"], reverse=True)
    total = sum(r["TotalDurationNs"] for r in rows)
    print(f"\n=== {label} ===")
    print(f"Total GPU kernel time:    {total/1e9:.3f} s "
          f"({len(rows)} distinct kernels)")
    attn = [r for r in rows if match_any(r["Name"], ATTENTION_KEYWORDS)]
    kvw = [r for r in rows if match_any(r["Name"], KV_WRITE_KEYWORDS)]
    print("\n  Attention kernels:")
    for r in attn:
        print(f"    {r['Name'][:60]:60s}  calls={int(r['Calls']):6d}  "
              f"total={r['TotalDurationNs']/1e6:8.2f} ms  "
              f"avg={r['AverageNs']:8.0f} ns  max={r['MaxNs']:.0f} ns")
    attn_total = sum(r["TotalDurationNs"] for r in attn)
    print(f"    -> attention total: {attn_total/1e6:.2f} ms "
          f"({100*attn_total/total:.1f}% of GPU time)")

    print("\n  KV-cache writes:")
    for r in kvw:
        print(f"    {r['Name'][:60]:60s}  calls={int(r['Calls']):6d}  "
              f"total={r['TotalDurationNs']/1e6:8.2f} ms  "
              f"avg={r['AverageNs']:8.0f} ns")
    kvw_total = sum(r["TotalDurationNs"] for r in kvw)
    print(f"    -> kv-write total:  {kvw_total/1e6:.2f} ms")

    print("\n  Top-10 non-attention non-kv kernels by total time:")
    other = [r for r in rows
             if not match_any(r["Name"], ATTENTION_KEYWORDS)
             and not match_any(r["Name"], KV_WRITE_KEYWORDS)]
    for r in other[:10]:
        name = r["Name"]
        if len(name) > 80:
            name = name[:77] + "..."
        print(f"    {name:80s}  calls={int(r['Calls']):6d}  "
              f"total={r['TotalDurationNs']/1e6:8.2f} ms")
    return attn_total, kvw_total, total


if len(sys.argv) != 3:
    print("Usage: analyze_kernel_trace.py <triton_stats.csv> <cdna_stats.csv>")
    sys.exit(1)

t_rows = load(sys.argv[1])
c_rows = load(sys.argv[2])
t_attn, t_kvw, t_total = summarize("TRITON baseline", t_rows)
c_attn, c_kvw, c_total = summarize("CDNA-HIP", c_rows)

print("\n=== DIFF (CDNA vs Triton) ===")
print(f"  total GPU kernel time:  {(c_total-t_total)/1e6:+.2f} ms  "
      f"({100*(c_total-t_total)/t_total:+.1f}%)")
print(f"  attention:              {(c_attn-t_attn)/1e6:+.2f} ms  "
      f"({100*(c_attn-t_attn)/t_attn:+.1f}%)")
print(f"  kv-cache writes:        {(c_kvw-t_kvw)/1e6:+.2f} ms  "
      f"({100*(c_kvw-t_kvw)/t_kvw:+.1f}%)")
