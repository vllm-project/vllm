# Mooncake selective recurrent lookup results

Measured on NVIDIA Thor (aarch64, Ubuntu 24.04, Python 3.12.3,
PyTorch 2.13.0+cu130), 2026-10-01. The existing runtime was used without
package installation; the CPU/TCP Mooncake build and Python binding were
loaded from a separate task directory.

- vLLM baseline: `2e52a558f04307b3f6f842e937e35aac079121c1`.
- Mooncake client and Master: `7943c052be6aab81c71b60fea9483952dec21141`
  from [PR #4314](https://github.com/kvcache-ai/Mooncake/pull/4314).
- The dependency implements selected-only `LastHitOnly` responses; its public
  interface is still under review. This connector option defaults to off.

## Correctness

| Check | Baseline | Selective implementation |
| --- | --- | --- |
| Worker, coordinator, connector suites | 261 passed | 305 passed |
| Layout, scheduler, HMA save/load suites | 62 passed | 62 passed |
| Final combined six-suite run | 323 passed across two baseline runs | 367 passed, zero skipped |
| Real TCP lookup, wire metadata, receiver buffer bytes | Not present | 4 passed, included above |
| Upstream Python probe contract against the same Master | — | 3 passed |
| Upstream Master policy, lease deadline and eviction tests | — | 9 passed |

The TCP scenarios cover a complete checkpoint, a missing rank namespace,
KV-capped candidates and Eagle pruning. Both recurrent rank buffers are
checked byte for byte after the connector's actual receiver loads them.
Unit coverage additionally includes 4096 physical existence patterns,
unequal namespace counts, duplicate keys, partial-hash KV tails, recurrent-only
lookup and invalid/error responses without a recurrent existence fallback.
Both baseline and modified runs report the same 14 Torch JIT deprecation
warnings. No model inference or GPU throughput improvement is claimed.

## Lookup latency

`benchmark_mooncake_last_hit_only.py` compares the existing lookup path
(flag off) with selective lookup (flag on), using the same worker and data.
Each scenario ran in three fresh processes: 20 warmups and 1000 measured
lookups per path per process. The second process ran selective lookup first.
There are 128 checkpoints, one KV namespace and four recurrent namespaces;
all recurrent checkpoints exist. KV either covers all 128 checkpoints or only
the first 16. Every result is asserted to equal the KV-supported boundary.

Values below are medians of the three per-process percentiles in microseconds.
Speedup is the ratio of the reported p50 values; the range uses individual
process p50 ratios. TCP uses a loopback Master with independent 60-second
leases, a 32 MiB segment and a 16 MiB local buffer.

| Transport / KV coverage | Before p50 | After p50 | p50 speedup | Run range | Before p95 | After p95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| In-memory / 128 | 397.15 | 375.36 | 1.06× | 1.05–1.06× | 414.92 | 380.31 |
| In-memory / 16 | 308.67 | 162.87 | 1.90× | 1.89–1.90× | 321.89 | 166.67 |
| TCP / 128 | 2037.92 | 1235.93 | 1.65× | 1.43–2.44× | 2072.87 | 1576.90 |
| TCP / 16 | 1694.88 | 678.66 | 2.50× | 2.46–2.96× | 1712.96 | 723.93 |

TCP selective timing varies substantially between processes. These results
measure connector lookup and Store control work on this machine, including
key construction and lease handling; they do not establish serving latency
or token throughput gains.

| KV coverage | Existing queried keys / RPCs | Selective queried keys / RPCs | Existing recurrent keys leased | Selective recurrent keys leased |
| --- | ---: | ---: | ---: | ---: |
| 128 | 640 / 1 | 128 + 512 / 2 | 512 | 4 |
| 16 | 640 / 1 | 128 + 64 / 2 | 512 | 4 |

## Reproduce

Use matching client and Master builds at the pinned dependency, with
`USE_TCP=ON`, `USE_CUDA=OFF` and independent leases. Set `MASTER` to an
isolated loopback Master address and `PYTHON` to the existing environment's
Python; expose the matching binding through `PYTHONPATH` and its libraries
through `LD_LIBRARY_PATH`.

```bash
MOONCAKE_LAST_HIT_MASTER="$MASTER" "$PYTHON" -m pytest -q \
  tests/v1/kv_connector/unit/test_mooncake_store_{worker,coordinator,connector,layout,scheduler,hma_e2e}.py

for divisor in 1 8; do
  "$PYTHON" benchmarks/benchmark_mooncake_last_hit_only.py \
    --checkpoints 128 --iterations 1000 --kv-divisor "$divisor"
  "$PYTHON" benchmarks/benchmark_mooncake_last_hit_only.py \
    --checkpoints 128 --iterations 1000 --kv-divisor "$divisor" --master "$MASTER"
done
```

Repeat the benchmark three times, adding `--selective-first` on the second
run. The harness deletes only its own UUID-prefixed Store keys. The existing
environment, other workloads and their model files are preserved.
