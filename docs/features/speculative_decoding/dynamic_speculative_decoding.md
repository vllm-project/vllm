# Dynamic Speculative Decoding

## Why is Dynamic SD needed?

SD methods need to verify K tokens for each sequence during decoding. As BS increases, the effective BS becomes BS\*K which increases the compute requirement during verification. When this BS\*K goes beyond a critical BS then SD negatively impacts the decode speed (TPOT). DSD helps by tuning the K to an optimal value such that we continue to reap the benefits from SD.

## Use cases

* Variable concurrency workload using same deployment. K would decrease as concurrency increases.
* During RL rollout where we start off with high BS but then end up with small BS due to very few long tail request which end up generating a lot of tokens stalling the progress of the current rollout. Here K would go up during the end of rollout.

## `--speculative-config` schema

To use Dynamic SD, add `num_speculative_tokens_per_batch_size` to the config of an SD method which is a list of list. Here, an entry is `[start_bs, end_bs, optimal_K]` which means when the concurrency is within range `[start_bs, end_bs]` then `optimal_K` number of draft tokens are used. For e.g.,

```bash
--speculative-config '{
    "method": "eagle",
    "model": "yuhuili/EAGLE-LLaMA3.1-Instruct-8B",
    "num_speculative_tokens": 3,
    "num_speculative_tokens_per_batch_size": [
      [1, 64, 3],
      [65, 128, 1],
      [129, 512, 0]
    ]
  }'
```

implies that:

* K=3 will be used when the concurrency is in range [1, 64]
* K=1 will be used when the concurrency is in range [65, 128]
* K=0 will be used when the concurrency is in range [129, 512], i.e., no draft tokens will be produced.

## Offline K schedule tuning

`vllm bench sweep tune_speculative_k` converts repeated serving sweep results
into `num_speculative_tokens_per_batch_size`. Keep the global maximum K fixed
across runs and force each candidate through a one-range batch-size schedule so
KV-cache and worker-buffer sizing remain comparable. For example, one entry in
`serve_params.json` for candidate K=3 is:

```json
{
  "speculative_config": {
    "method": "eagle",
    "model": "yuhuili/EAGLE-LLaMA3.1-Instruct-8B",
    "num_speculative_tokens": 7,
    "num_speculative_tokens_per_batch_size": [[1, 64, 3]]
  }
}
```

Add equivalent entries for every candidate, including K=0 when desired. Put
the tuning anchors in `bench_params.json`:

```json
[
  {"max_concurrency": 1},
  {"max_concurrency": 8},
  {"max_concurrency": 32},
  {"max_concurrency": 64}
]
```

Run at least three repetitions per combination, then generate the schedule:

```bash
vllm bench sweep serve \
  --serve-cmd 'vllm serve meta-llama/Llama-3.1-8B-Instruct' \
  --bench-cmd 'vllm bench serve --backend vllm --model meta-llama/Llama-3.1-8B-Instruct --request-rate inf --num-prompts 256 --ignore-eos' \
  --serve-params serve_params.json \
  --bench-params bench_params.json \
  --num-runs 3 \
  --output-dir results \
  --experiment-name spec-k

vllm bench sweep tune_speculative_k results/spec-k --max-batch-size 64
```

The tuner maximizes median output TPS minus a median-absolute-deviation
uncertainty penalty by default. It can instead minimize a latency metric, for
example `--objective-var mean_tpot_ms --objective-direction minimize`; the
uncertainty penalty is then added to the median. Candidates within 1% of the
best conservative objective are treated as tied and the smaller K is
preferred. Candidate K values are selected independently because MoE and other
architectures can have a non-monotonic optimum. Adjacent equal-K ranges are
merged, and the command writes
`speculative_k_schedule.json` in the experiment directory.
Pass `--acceptance-var spec_decode_acceptance_length` to include acceptance
stability diagnostics, and `--strict-acceptance-stability` to reject unstable
positive-K candidates. Acceptance is undefined for K=0 because no draft model
runs, so missing or null K=0 acceptance values are allowed.
Every batch-size anchor must contain the same candidate set, and the deployment
maximum must be a measured anchor; the tuner rejects missing candidates and
unmeasured tail extrapolation. With the default vLLM result fields, it also
rejects candidate K values above the configured global maximum and sweeps that
change the global maximum between candidates. The output records both the
configured global maximum and the largest measured candidate; keep the former
unchanged when deploying the generated schedule so allocation and graph
coverage match the sweep.
Use a representative prompt/output distribution and the same GPU topology,
model, drafter, compilation mode, and latency constraints as production.
Because `max_concurrency` is a ceiling rather than a scheduler batch-size trace,
keep the benchmark saturated with fixed-length requests and verify that it is a
reasonable proxy for the runtime batch sizes being tuned. To optimize under a
latency SLO, pass `--goodput tpot:<milliseconds>` (and any other SLOs) to
`vllm bench serve`, then tune with
`--objective-var request_goodput --objective-direction maximize`. The tuner
optimizes one scalar metric at a time; alternatively exclude candidates that
violate deployment constraints before tuning.

## Online Examples

### Dynamic SD Eagle Drafter

```bash
VLLM_USE_V2_MODEL_RUNNER=0 vllm serve meta-llama/Llama-3.1-8B-Instruct \
  --speculative-config '{
    "method": "eagle",
    "model": "yuhuili/EAGLE-LLaMA3.1-Instruct-8B",
    "num_speculative_tokens": 3,
    "num_speculative_tokens_per_batch_size": [
      [1, 64, 3],
      [65, 128, 1],
      [129, 512, 0]
    ]
  }'
```

### Dynamic SD Eagle3 Drafter

```bash
VLLM_USE_V2_MODEL_RUNNER=0 vllm serve meta-llama/Llama-3.1-8B-Instruct \
  --speculative-config '{
    "method": "eagle3",
    "model": "yuhuili/EAGLE3-LLaMA3.1-Instruct-8B",
    "num_speculative_tokens": 3,
    "num_speculative_tokens_per_batch_size": [
      [1, 16, 5],
      [17, 32, 4],
      [33, 64, 3],
      [65, 128, 1],
      [129, 512, 0]
    ]
  }'

```

## Limitations

* Tested with Eagle, Eagle-3, and DFlash. Other SD methods may or may not work out of the box
* Full Cudagraph only works with Model Runner V2. MRv1 only supports piece-wise cuda graph with this feature
* Not compatible with data parallelism (`--data-parallel-size > 1`). Each DP rank schedules independently, so ranks can pick different K values, causing DP collective divergence and deadlocks. When DP is enabled, vLLM automatically disables `num_speculative_tokens_per_batch_size` and falls back to the static `num_speculative_tokens` value.
