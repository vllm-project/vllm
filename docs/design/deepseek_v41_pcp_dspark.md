# DeepSeek-V4.1 PCP with DSpark

This experimental follow-up to
[#59857](https://github.com/vllm-project/vllm/pull/59857) enables fixed-verification
DSpark and optional same-host Engram table sharing. It is based on
`7908ee8906475842ba0ecd1c09eed6d878718f05`; the parent's MoE prerequisite
[#60946](https://github.com/vllm-project/vllm/pull/60946) still applies.

The validated configuration is NVIDIA MRV2, FlashMLA sparse attention,
PP=DP=DCP=1, expert parallelism, eager execution and synchronous scheduling.
Prefix caching, bounded replay, model CUDA graphs, adaptive verification,
microbatching, KV transfer and MegaMoE are outside this validation.

## Compressor state and replicated drafting

Without speculation, the PCP compressor publishes each local chunk's last
row and retains only the newest open ratio-2 row. Rejection can resume before
that row, so enabling DSpark requires preserving additional ring history.

For speculative batches, each rank publishes up to one ring capacity of its
chunk tail. Metadata selects unique writes within the newest ring window,
retaining the rows needed after rejection without conflicting writes to the
same ring slot. The non-speculative path keeps its one-row publication.

The generic replicated DSpark path from
[#56107](https://github.com/vllm-project/vllm/pull/56107) restores target
auxiliary states to global token order before drafting. Replicated draft
forwards can still differ numerically across PCP ranks: a model trace found
identical target inputs and samples but different proposed tokens. Broadcast
the proposal from PCP rank zero so all target shards verify the same tokens.
When present, proposal logits and adaptive confidence buffers follow the same
ownership rule; V4.1 adaptive verification remains disabled by its model guard.

This does not implement
[#54036](https://github.com/vllm-project/vllm/pull/54036)'s sharded draft
context-KV precomputation. Draft work is still replicated.

## Engram sharding and sharing

Enable the optional storage mode with:

```bash
--engram-config '{"cpu_offload":true,"pcp_shared_memory":true}'
```

Engram hash heads retain their existing TP partition. PCP ranks with the same
TP rank map the same physical weight and scale pages; they do not split the
table into new PCP-owned row ranges. For example:

| Topology | TP head partitions | Physical copies of each partition |
| --- | ---: | ---: |
| TP1 / PCP4, private | 1 | 4 |
| TP1 / PCP4, shared | 1 | 1 |
| TP2 / PCP2, private | 2 | 2 |
| TP2 / PCP2, shared | 2 | 1 |

Each PCP replica still looks up its own token IDs. This reuses the existing
`DPSharedEngramStorage` allocation, one-writer loading, CUDA host registration,
UVA views and lifetime handling from
[#57651](https://github.com/vllm-project/vllm/pull/57651). It does not change
weight precision, table layout, hash-head ordering or lookup arithmetic.
Existing DP sharing is unchanged.

The option defaults to false and requires V4.1 CPU offload, PCP>1, DP=PP=1,
same-host/shared-namespace PCP replicas, and sufficient `/dev/shm` capacity.
It cannot be combined with `dp_shared_memory` or `use_thp`. Unsupported
placement or insufficient shared memory raises instead of falling back to
private copies. The existing storage handles mapping/registration failures.

For the tested checkpoint, two full tables require about 188.83 GiB. Four
private PCP copies need about 755.33 GiB, exceeding the test container's
650 GiB limit. Sharing reduced measured table PSS to 188.83 GiB. A 256 GiB
`/dev/shm` capacity was used within that same 650 GiB container limit; it is
not an additional host-memory allocation.

Draft models run with PCP=1. Their copied Engram configuration disables PCP
sharing without modifying the target configuration. This avoids inheriting
an invalid PCP>1 storage requirement during draft initialization.

This is distinct from
[#57195](https://github.com/vllm-project/vllm/pull/57195)'s SP token-owner
lookup path: TP head partitioning and PCP token ownership are preserved here.

## Validation

Four H20 GPUs, real DeepSeek-V4.1-Flash checkpoint, DSpark with five speculative
tokens, chunk size 8192, Humming MoE, `fp8_ds_mla`, APC/replay disabled,
`--enforce-eager`, `--no-async-scheduling` and `VLLM_GPU_SYNC_CHECK=error`:

| Check | Result |
| --- | --- |
| Distributed Engram suite, including original DP and new TP1/PCP4 and TP2/PCP2 sharing | 31 passed |
| DSpark configuration suite | 13 passed |
| Focused PCP compressor GPU regression, PCP2/4 with 0/5/9 speculative tokens | 12 passed |
| Fixed GSM8K canary, TP1/PCP4 + shared storage | 32/32, no HTTP errors or truncated outputs |
| Actual speculation counters | 3900 accepted / 5695 drafted, 1139 draft rounds |
| Shared storage audit | Two backing files, eight rank mappings, total PSS 188.83 GiB |

Storage tests compare private/shared weights, scales and lookup outputs exactly;
exercise uneven TP head slices, distinct and empty PCP inputs, changed/restored
CUDA graph inputs, one-writer visibility, failure cleanup and storage lifetime.
These graph tests validate lookup storage only, not model graph support.
The model canary is not a comprehensive accuracy evaluation.

Run the focused tests in a CUDA vLLM environment with four free GPUs:

```bash
python -m pytest tests/distributed/test_engram_dp_shard.py
python -m pytest tests/v1/worker/test_dspark_utils.py
python -m pytest tests/kernels/test_compressor_kv_cache.py \
  -k 'v41_pcp_ring_links or v41_pcp_compressor_preserves'
```

The local validation used a Python overlay on a prebuilt runtime: source base
`7908ee8` plus this patch, with native extensions and missing runtime files
reused from an `a982` image. A clean build from the complete PR head remains
to be validated. Full-model CI requires the large checkpoint and sufficient
host RAM; the focused tests extend existing Buildkite test files.

## Initial performance evidence

64K input / one output token, concurrency four, eight requests per round with
continuous replenishment, zero cached tokens; one calibration then three
measured rounds per instance:

| Configuration | Round 1 | Round 2 | Round 3 | Median input tok/s |
| --- | ---: | ---: | ---: | ---: |
| TP2 / PCP2, private | 5266.766 | 5270.280 | 5273.134 | 5270.280 |
| TP1 / PCP4, shared | 5658.446 | 5665.052 | 5664.641 | 5664.641 |

The median increased 7.48% in this sequential topology comparison. It is not
three independent A/B campaigns or an isolated shared-storage speedup:
TP/PCP and shared-memory capacity changed, and private PCP4 failed at startup.
The shared mode removes that memory barrier. Calibration samples, excluded
from the medians, were 5256.846 and 4095.625 tok/s respectively.

SGLang's historical approximately 17.2K tok/s is not a matched backend result.
Its TP4/attention-CP4/EP2, FP8-activation MoE and decoder bounded replay differ
from this TP1/PCP4/EP4, Humming A16, no-replay diagnostic configuration. This
change does not claim to close that gap or improve decode throughput.
