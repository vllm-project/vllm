# Comparing Multimodal Preprocessing Concurrency

Prefer `--api-server-count` for scaling multimodal input processing. Enable
`--mm-processor-num-workers` only when a serving benchmark shows that its
parallelism outweighs the additional process-boundary transfer and memory costs.
Functional tests and standalone rendering benchmarks do not establish that.

## Setup

Use an isolated deployment representative of production, not a live
customer-facing endpoint. No particular GPU model is required. Keep hardware,
total CPU allocation, model/tokenizer revisions, images, pixel limits, cache
settings and engine configuration fixed across cases. Record the vLLM commit,
runtime versions, CPU topology, RAM and shared-memory limits, and exact commands.
Ensure the load generator is not the bottleneck.

| Configuration | API servers | MM workers per renderer | `OMP_NUM_THREADS` |
| --- | ---: | ---: | ---: |
| Baseline | 1 | 1 | 1 |
| Process pool | 1 | 2 | 1 |
| API-server scale-out | 2 | 1 | 1 |

A worker count of `1` uses the existing thread, not a child process. Start one
deployment at a time, assigning `$api_servers` and `$mm_workers` from each row:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
TOKENIZERS_PARALLELISM=false VLLM_MEDIA_LOADING_THREAD_COUNT=1 \
vllm serve "$model" \
    --revision "$revision" --tokenizer-revision "$revision" \
    --api-server-count "$api_servers" \
    --mm-processor-num-workers "$mm_workers" \
    --mm-processor-cache-gb 0 --mm-processor-device cpu \
    --mm-processor-kwargs '{"max_pixels":3145728}' \
    --limit-mm-per-prompt '{"image":1,"video":0,"audio":0}' \
    --no-enable-prefix-caching \
    --max-model-len 4096 --max-num-seqs 16 --max-num-queued-reqs 216
```

Set `$model` and `$revision` to the same checkpoint for all runs. If tuning
intra-op threads, repeat all three cases under the same total CPU allocation;
do not tune only the process-pool case. All server children share that allocation.

## Workloads

Use the existing `vllm bench serve` client with a `custom_image` JSONL dataset.
For example, a row can contain
`{"prompt": "Describe this image.", "image_files": ["/data/page.png"]}`.
Set `$dataset` to its path and `$label` to a unique workload/configuration/trial
name. The following example measures bounded-concurrency throughput:

```bash
vllm bench serve \
    --model "$model" \
    --backend openai-chat --endpoint /v1/chat/completions \
    --dataset-name custom_image --dataset-path "$dataset" \
    --custom-ensure-client-side-data \
    --num-prompts 1200 --custom-output-len 16 \
    --request-rate inf --max-concurrency 16 \
    --num-warmups 32 --ignore-eos --temperature 0 --seed 0 \
    --percentile-metrics ttft,e2el --metric-percentiles 50,99 \
    --save-result --save-detailed --result-filename "$label.json"
```

Also test the deployment's offered load: replace `--request-rate inf
--max-concurrency 16` with, for example, `--request-rate 20 --burstiness inf`.
With 1200 requests this schedules about 60 seconds of arrivals, then drains
outstanding requests. Report both intervals; offered rate is not completed
throughput. A fixed offered rate can hide capacity differences when every case
keeps up.

For the burst from [issue #58266](https://github.com/vllm-project/vllm/issues/58266),
use `--request-rate inf --num-prompts 300 --custom-output-len 1024
--extra-body '{"min_tokens":1024}'` without a client concurrency limit. Exact
reproduction also requires the reporter's custom 4B checkpoint, document
images and B300/eight-vCPU setup; different inputs or hardware are a separate
experiment.

Request-count admission statistics are shared approximately across API servers
at this revision: do not divide `--max-num-queued-reqs` by the API-server count.
The optional `--max-num-queued-tokens` limit remains local to each API process.
Always retain rejections and errors in the results.

## Evaluation

- Warm every process, exclude startup from measurements, run at least three
  trials per case, publish every trial and report medians and ranges. Recheck the
  baseline for drift and run long enough to measure steady state.
- Compare successful requests/second, TTFT, end-to-end latency, GPU/CPU
  utilization, resident memory and shared-memory usage. Verify response
  correctness; deterministic decoding alone does not prove output parity.
- To investigate pre-engine wait, collect the same `llm_request` trace spans in
  every case. Subtract `gen_ai.latency.time_in_model_prefill` and
  `gen_ai.latency.time_in_queue` from `gen_ai.latency.time_to_first_token` within
  each request. Verify timing semantics at the tested revision; do not subtract
  aggregate percentiles or mix client and server timings. This remainder includes
  frontend work and handoff, not just preprocessing.
- Require a repeatable practical benefit over both baseline and API-server
  scale-out, without hiding memory costs, errors or latency regressions.
