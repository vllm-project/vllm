# Indexer Q-row chunk benchmark

This is an SM100 MXFP4 component benchmark for the combined cooperative and
persistent BF16 implementation, with configurable row chunking. It executes real
DeepGEMM scoring; inputs are deterministic synthetic quantized Q/K tensors.
It is not a model or serving benchmark.

Run in an environment with the matching built native extension, on one idle GPU:

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python benchmarks/kernels/indexer_chunk/benchmark.py \
  --case smoke --out /tmp/indexer-chunk-smoke
```

Use `--runtime /absolute/installed/source/root` for an isolated built runtime and
`--expected-native-sha HASH` to require a particular native library. Then run
`c1_short`, `c1_source`, `c2`, `c4`, and `tail_source` into distinct output directories.
These sweep disabled/512/1024/2048/4096/8192 MiB in forward and reverse order.
The large cases can require tens of GiB of GPU memory and substantial host memory.

Correctness snapshots are outside measured captures. They check exact valid
producer/candidate score bits, legal exact top-k values, sentinel padding and
graph recomputation. Tied index identities are not required. Timing uses cold-L2
CUPTI graph replay; one replay contains every chunk. The original common schedule
is prepared outside caller timing; extra chunk schedules are inside it.

The payload budget is not a total-device or graph-pool cap. Results separately
record additional live peak, graph private reservation and the conservative
profiling scratch estimate. Output, input and existing workspace are excluded
from additional-live figures. Retain all raw results, including regressions.

## Configuring the decode budget

`VLLM_SPARSE_INDEXER_MAX_DECODE_LOGITS_MB` sets the BF16 decode logits payload
budget in MiB. The default is `2048` (2 GiB); `0` disables it. Set it to `4096`
to trade more temporary memory for fewer chunks. This is adjustable per deployment;
no budget is optimal for every workload.

The planner derives the query-row limit from `budget_bytes / (2 * capacity)`
and aligns chunks to 16 rows. A batch that fits is unchanged. With a 1M model
length, a 2 GiB budget fits 1024 rows at compression 1, 2048 at compression 2,
or 4096 at compression 4. Eight query rows per request correspond to 128,
256, or 512 requests, respectively. These are per-step query tokens, not the
sum of query and historical KV tokens. Actual chunks may be smaller because
the planner balances the row ranges, processing larger chunks first so later
CUDA graph nodes can reuse their temporary allocations.

Shorter effective history does not by itself reduce the configured logits
capacity. Candidate scores, physical row padding, and other scratch remain
additional to this payload budget.
