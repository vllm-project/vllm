# Comparing Multimodal Preprocessing Concurrency

Prefer `--api-server-count` for scaling multimodal input processing. An additional
preprocessing process pool is only useful if its parallelism outweighs the cost
of transferring inputs and processed tensors across another process boundary.
Compare both approaches before recommending `--mm-processor-num-workers`.

## Comparison matrix

Keep the model, image inputs, pixel limits, cache settings, offered load, engine
configuration, CPU affinity, and total available CPU resources fixed.
Pin model and tokenizer revisions with `--revision` and `--tokenizer-revision`
when running the server.

| Configuration | API/render processes | MM workers per renderer | `OMP_NUM_THREADS` |
| --- | ---: | ---: | ---: |
| Default threading | 1 | 1 | 1 |
| Process pool | 1 | 2 | 1 |
| Frontend scale-out | 2 | 1 | 1 |
| Thread-budget baseline | 1 | 1 | 8 |
| Thread-budget process pool | 1 | 2 | 4 |
| Thread-budget frontend scale-out | 2 | 1 | 4 |

A worker count of `1` uses the existing thread, not a child process. All server
processes and their children must share the same CPU allocation in every case.
The last three rows bound the total configured preprocessing intra-op threads
at eight; this does not include other frontend or engine threads.

Disable processor and prefix caches for an uncached workload. Keep media-loading
thread counts fixed, warm every process before measurement, and exclude startup
from steady-state timing. Run at least three trials, publish every trial, and
report the range as well as the median. Recheck the baseline to detect drift.
Do not select only the model, thread count, or trial that favors the new code.

## CPU-only HTTP rendering comparison

The standalone client in `benchmarks/benchmark_mm_render.py` exercises
`/v1/chat/completions/render` with inline PNGs. It includes HTTP input handling,
image decoding, preprocessing, worker IPC when enabled, and receipt of the full
serialized tensor response. It does not run inference or measure TTFT.

`vllm launch render` starts a single rendering process. It does **not** implement
`--api-server-count` scale-out. For this CPU-only comparison, start two independent
render servers on different ports and pass both URLs to the client. This is a
comparison of rendering processes, not a measurement of the normal
`vllm serve --api-server-count` deployment, shared socket routing, or admission
control.

Example on Linux with at least 16 available logical CPUs:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
TOKENIZERS_PARALLELISM=false VLLM_MEDIA_LOADING_THREAD_COUNT=1 \
taskset -c 0-7 .venv/bin/python -m vllm.entrypoints.launchers.render.entry \
    --model Qwen/Qwen3.5-4B \
    --host 127.0.0.1 --port 8100 \
    --max-model-len 4096 \
    --mm-processor-cache-gb 0 \
    --mm-processor-device cpu \
    --mm-processor-num-workers 1 \
    --mm-processor-kwargs '{"max_pixels":3145728}' \
    --limit-mm-per-prompt '{"image":1,"video":0,"audio":0}' \
    --no-enable-prefix-caching \
    --disable-uvicorn-access-log
```

Run the client from the checkout in another terminal:

```bash
taskset -c 8-15 .venv/bin/python benchmarks/benchmark_mm_render.py \
    --model Qwen/Qwen3.5-4B \
    --base-urls http://127.0.0.1:8100 \
    --label render1_mm1_omp1 \
    --width 1275 --height 1650 \
    --unique-images 16 --seed 0 \
    --num-warmups 32 --num-prompts 96 \
    --concurrency 16 --repetitions 3 \
    --output-json render1_mm1_omp1.json
```

For resource measurements, additionally pass `--server-pids` followed by the
PID of each local render server. The client samples their process trees and the
local `/dev/shm` mount. Run it in the same PID and shared-memory namespaces.
Without server PIDs these resource fields are `null`.

For two render servers, start the same server command on port `8101` too, keeping
both server processes pinned to `0-7`. Pass both base URLs and both server PIDs
to the client. Requests are distributed round-robin between URLs. For process
pool rows, start only one server and set `--mm-processor-num-workers 2`.

The client uses 16 deterministic synthetic images with block noise, resized to
the requested page dimensions before PNG encoding. They are not the original
issue's document images. The processor cache must stay disabled when these
images repeat. The JSON records compressed PNG size, full response size,
versions, affinities, and individual trials.

Interpret the metrics carefully:

- Request latency runs from HTTP submission through receipt of the entire body;
  it excludes local waiting for a concurrency slot.
- Throughput is completed responses divided by measured wall time. Each trial
  uses a closed loop with the specified concurrency.
- Any HTTP error or timeout fails the run rather than counting a partial trial
  as success.
- Average CPU cores are the server process tree's CPU seconds divided by wall
  time.
- Peak summed RSS can double-count shared pages. `/dev/shm` usage is sampled at
  100 ms intervals, so shorter peaks may be missed.
- A separate client affinity avoids competing for the same logical CPUs, but
  does not guarantee isolation from SMT siblings or other host workloads.
- Tensor responses can be much larger than compressed images. HTTP output
  serialization and transport can dominate this benchmark; these costs differ
  from engine-core tensor transport.

Do not treat a render-only result, positive or negative, as a GPU inference
throughput or TTFT result.

## Required inference comparison

On the target GPU host, use **one** `vllm serve` deployment at a time and vary
`--api-server-count`, `--mm-processor-num-workers`, and thread budget according
to the matrix. Keep engine settings unchanged, for example:

```bash
OMP_NUM_THREADS="$threads" MKL_NUM_THREADS="$threads" \
TOKENIZERS_PARALLELISM=false VLLM_MEDIA_LOADING_THREAD_COUNT=1 \
vllm serve Qwen/Qwen3.5-4B \
    --api-server-count "$api_servers" \
    --mm-processor-num-workers "$mm_workers" \
    --mm-processor-cache-gb 0 \
    --mm-processor-device cpu \
    --mm-processor-kwargs '{"max_pixels":3145728}' \
    --limit-mm-per-prompt '{"image":1,"video":0,"audio":0}' \
    --no-enable-prefix-caching \
    --max-model-len 4096 \
    --max-num-seqs 16 \
    --max-num-queued-reqs 216
```

Set the three shell variables to the values in each matrix row. Apply the same
CPU allocation to the entire deployment, including engine processes.

Check admission semantics at the exact tested revision. In this implementation,
multiple API servers share approximate request-count admission statistics, so
`--max-num-queued-reqs` must not be divided again by the API-server count.
`--max-num-queued-tokens` is still checked against each API process's local
prefill backlog. If testing that optional limit, report its per-process values
and aggregate budget separately. Always report rejected requests.

Use a `custom_image` JSONL dataset with the same images for all cases. For example,
each row can contain `{"prompt": "Describe this image.", "image_files": ["/data/page.png"]}`.
Use the issue reporter's images and model revision when reproducing that report;
synthetic or different images are a separate experiment.

```bash
vllm bench serve \
    --model Qwen/Qwen3.5-4B \
    --backend openai-chat --endpoint /v1/chat/completions \
    --dataset-name custom_image --dataset-path "$dataset" \
    --custom-ensure-client-side-data \
    --num-prompts 1200 --custom-output-len 16 \
    --request-rate 20 --burstiness inf \
    --num-warmups 32 --ignore-eos --seed 0 \
    --percentile-metrics ttft,e2el \
    --metric-percentiles 50,99 \
    --save-result --save-detailed \
    --result-filename "$label-steady.json"
```

Also test a burst of 300 simultaneous requests with
`--request-rate inf --num-prompts 300 --custom-output-len 1024`. Do not add a
client concurrency limit to that burst. Retain error details and successful
request counts rather than comparing latency after silently dropping rejections.

Record GPU utilization, CPU usage, resident memory, shared-memory usage, TTFT,
end-to-end latency, and successful requests/second. Compare deterministic
responses across configurations and retain model/tokenizer revisions and the
exact commands. Improvements must persist across repeated runs relative to both
the single-API baseline and API-server scale-out, without hiding memory costs,
errors, or latency regressions.
