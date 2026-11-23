# Operator kernel buckets

vLLM uses a common integer interval interface for operator dispatch.
Operators retain their existing thresholds, implementation choices and
multidimensional dispatch. M is one possible axis; the interface does not
require every operator to dispatch on M. Prefill, decode, mixed execution and speculative verification have
separate configuration entries. This provides the dispatch vocabulary for a
future Agent Kernel library; it does not install such a library or select new
kernels by default.

The implementation lives in `vllm/kernel_buckets.py`. CUDA and CUTLASS host
dispatch use `csrc/core/kernel_buckets.h`. This initial migration covers NVIDIA host dispatch in the listed operators.
Other model and platform support is preserved; vendor-library internals are not
intercepted by this interface.

## Intervals preserve the existing boundaries

All intervals are half open: `[start, stop)`. An omitted stop means unbounded.
For nonnegative integer rows, `M <= 16` is `[0, 17)` and `M >= 1024` is
`[1024, infinity)`. Matching does not pad or round the actual tensor shape.
An empty input remains an empty input, with the original caller's handling.

`m_in_range` expresses a support or dispatch gate. `MBucketTable` describes a
complete partition and selects an implementation identifier or configuration:

```python
from vllm.kernel_buckets import MBucketTable, m_in_range

small = m_in_range(m, start=1, stop=17)
config = MBucketTable(
    stops=(17, 1025, 4097),
    values=(small_config, medium_config, large_config, fallback_config),
)
selected = config.select(m)
```

Ordinary integer lookup uses binary search; compiled symbolic lookup retains
traceable range comparisons. `select_current` skips phase-context reads when
all phase entries share the fallback table. Distinct phase tables still read
the real execution phase.

Router BF16 and HC dispatch cache up to 256 exact configurations. Cache keys
retain the original shape and dtype dimensions; misses use the same bucket
selection. Whole-batch phase classification reuses the scheduler's prefill and
draft counts. DBO slices still classify their own live request metadata.

Construct tables once during module or operator initialization. Reuse the same
table for launch selection, warmup configuration selection and exported keys.
An operator with a single implementation can use one unbounded interval.
Do not introduce artificial performance boundaries for it.

`MBucketTable.from_points` compresses adjacent equal configurations while
preserving holes and their fallback. An isolated measured shape stays a
singleton bucket. The BF16/FP32 router tables retain such points because their
measurements do not establish a configuration for the intervening shapes.
`from_nearest` converts an existing nearest-point lookup to intervals and
preserves its insertion-order tie rule. `select_nearest_m` keeps that rule for
mutable legacy config dictionaries without rebuilding a table per launch.

Compile-time shape specialization remains separate from configuration buckets.
For example, the router dot-product backend still compiles for exact M and K;
its compile key must retain both even when several M values share the same
block size. A bucketed config alone does not make that kernel shape dynamic.

## Execution phase comes from host metadata

`KernelPhase` has `PREFILL`, `DECODE`, `MIXED`, `VERIFY` and `UNKNOWN` entries.
A large decode batch and a short chunked prefill can have the same M. Therefore
M, graph padding and `max_query_len` alone must not determine the phase.

The V2 runner supplies the phase through `ForwardContext.kernel_phase` using
the live requests' host prefill flags and scheduled draft-token information.
DBO determines the phase independently for each request slice. A slice with
insufficient per-request draft information remains unknown. No device tensor is
read back to classify the phase.

An attention operator processing a prefill or decode subrange of a mixed batch
must pass the subrange's phase explicitly. `current_kernel_phase()` describes
the enclosing forward, so it returns `MIXED` for such a batch. A draft forward
creating a new context must also supply its own phase or keep the fallback.

`PhaseBucketTable` allows each phase to have different M boundaries and different
implementations. A missing phase uses its explicit fallback and never inherits
another phase's specialized implementation:

```python
from vllm.kernel_buckets import KernelPhase, MBucketTable, PhaseBucketTable

buckets = PhaseBucketTable(
    operator="my_library.my_operator",
    fallback=MBucketTable((), (legacy_kernel,)),
    prefill=MBucketTable((1025,), (prefill_kernel, large_prefill_kernel)),
    decode=MBucketTable((17,), (small_decode_kernel, decode_kernel)),
)
implementation = buckets.select(m, phase=KernelPhase.DECODE)
```

The migrated operators initialize phase entries with their existing configs.
They can later acquire independently validated phase implementations without
changing interval semantics. This refactor itself makes no kernel speedup claim.

## Preserve the original multidimensional dispatch

M usually denotes an operator's local token rows. It can differ from the
scheduler's logical batch after padding, sharding, routing or packing. Select
using the rows that the operator actually consumes.

Some choices use a derived work count. QSA sparse attention uses
`M * num_kv_heads`; its tables declare that axis. Its architecture, cache dtype
and existing `use_prefill_config` query profile remain separate keys. That query
profile is an algorithm choice, and must not substitute for `KernelPhase`.
CUTLASS paths that already dispatch on rounded M retain their existing rounding.

Keep existing N/K buckets, dtype and architecture partitions, head-count
choices, sequence-length ranges and their combinations in the operator's
dispatch. Do not replace a multidimensional decision with an M-only table.
The common row interval is only the row component of that original decision.

Keep scale representation, strides, alignment, parallel layout, numerical mode
and execution dependencies in the operator's support checks and specialization. Likewise, sequence length and per-request
query length are separate dimensions. A matching M bucket does not establish
that a kernel supports them.

Collective operators must also use dispatch metadata agreed across their
participating ranks. A rank-local phase or work count must not change the
collective schedule independently on one rank.

## Agent Kernel keys describe the actual dispatch

`PhaseBucketTable.key` returns a hashable `KernelBucketKey` containing the
operator, phase, declared axis, selected interval and specialization tuple.
Rows inside an interval share that key only when all other dispatch dimensions
also agree. `specialization` may contain existing bucket labels for N/K or other
axes, as well as exact attributes required by the implementation:

```python
key = buckets.key(
    m,
    phase=KernelPhase.DECODE,
    specialization=(gpu_arch, input_dtype, output_dtype, n, k, layout),
)
```

The caller must preserve every original dispatch axis in its key and include
exact compile-time dimensions when required. Operators whose original choices
use another work axis keep that axis; they do not acquire artificial M cutoffs. An unknown phase cannot produce an Agent
Kernel key. Registration, availability checks, fallback and the call signature
belong to the future library adapter; this module does not intercept all Torch
or external-library calls.

Shared CUDA graphs, dummy runs and untagged standalone calls retain the unknown
phase and legacy fallback. A graph records the selected implementation during
capture; replay cannot switch it by changing host metadata. Before enabling a
phase-specific implementation, its adapter must either use distinct phase-aware
capture/compile identities or keep the shared graph on the fallback. This also
applies to model wrappers that reuse compiled graphs without checking guards.

## Migration coverage and checks

The source migration covers row gates in GEMM/router, Triton MoE, FP8 activation
and quantization, HC/mHC, QSA, indexer/top-k, GDN overlap and retained communication
paths, including the public custom-op and shared sequence-parallel/fusion
entry points. QSA's architecture/query-profile tables and router point configs use
explicit interval tables. C++ row choices, including constexpr router thresholds
and rounded-M CUTLASS dispatch, use the common host predicate.

The source inventory lists the covered Python scopes and exclusions in
`tools/kernel_bucket_inventory.py`. It finds direct host row thresholds,
normalized predicates, nearest-config lookups and table declarations without
importing CUDA. It deliberately excludes device indexing, assertions, empty
input checks and unrelated backends. It is a syntactic regression check, not
an exhaustive dynamic dependency-closure proof; derived expressions, arbitrary
new variable names and opaque library-internal dispatch need code review.

```bash
.venv/bin/python tools/kernel_bucket_inventory.py --check --output buckets.json
VLLM_TARGET_DEVICE=empty .venv/bin/python -m pytest \
  --confcutdir=tests/model_executor \
  tests/model_executor/test_kernel_buckets.py -q
VLLM_TARGET_DEVICE=empty .venv/bin/python -m pytest \
  --confcutdir=tests/v1/worker \
  tests/v1/worker/test_gpu_input_batch_v2.py -k kernel_phase -q
c++ -std=c++17 -Wall -Wextra -Werror -I csrc \
  tests/kernels/core/kernel_buckets_test.cpp -o /tmp/kernel-buckets-test
/tmp/kernel-buckets-test
```

The host checks are selected by the Operator Buckets Buildkite area. CUDA
compilation, kernel correctness, model quality and end-to-end serving performance
still require the target GPU environment. Host dispatch equality does not
establish GPU numerical correctness or performance. Before publishing or enabling
new implementations, validate both retained models, prefill/decode/verification,
mixed batches and graph replay on the affected NVIDIA architectures.
