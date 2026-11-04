# Reproduce a Quantized Backend Comparison

See the [A100 results and limits](quantized_backends.md).
These commands use the same server and benchmark options as that test.
The [original scripts and data](https://gist.github.com/jiangLLM/b77407a91899553309bfe941ed621cf6)
also include eight short prompt warmups and a 128-question GSM8K check before
the first timed round. Use those scripts to repeat the exact experiment.

`--linear-backend` selects the compute backend. Keep the quantization method
fixed; a different `--quantization` value may change how vLLM loads weights.
See [backend selection](../features/quantization/README.md#selecting-linear-backends-per-quantization).

## Prepare the model and environment

Use an active [vLLM environment](../getting_started/installation/gpu.md) with
Humming and its CUDA dependencies installed. Keep the same vLLM, PyTorch, CUDA,
and Humming versions for both backends. Run these commands outside the source
checkout if you installed a wheel, so the source package cannot shadow it.

Download one model revision before any timed run:

```bash
mkdir quantized-backend-bench
cd quantized-backend-bench
mkdir results caches
hf download Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4 \
  --revision e9c932ac1893a49ae0fc497ad6e1e86e2e39af20 \
  --local-dir model
uv pip freeze > results/packages.txt
nvidia-smi -q > results/gpu.txt
```

This checkpoint uses symmetric 4-bit GPTQ weights, group size 128, and no
act-order. The example uses FP16 activations and tensor parallel size 1. For
another checkpoint, first check that both backends support its format and
shapes. Record the model revision, GPU model, power limit, driver, and whether
clocks are locked. Keep other jobs off the GPU.

## Start one backend

In the first terminal, activate the environment and enter the benchmark
directory. Choose the backend and round. Create a new result directory for each
run; reuse that backend's cache directory on later rounds.

```bash
set -euo pipefail
cd /path/to/quantized-backend-bench
backend=marlin
round=1
out="$PWD/results/$backend-r$round"
mkdir "$out"

export VLLM_CACHE_ROOT="$PWD/caches/$backend/vllm"
export TRITON_CACHE_DIR="$PWD/caches/$backend/triton"
export TORCHINDUCTOR_CACHE_DIR="$PWD/caches/$backend/inductor"
export HUMMING_CACHE_DIR="$PWD/caches/$backend/humming"
export HUMMING_TMP_DIR="$PWD/caches/$backend/humming-tmp"
export CUDA_CACHE_PATH="$PWD/caches/$backend/cuda"

vllm serve "$PWD/model" \
  --served-model-name qwen-gptq --host 127.0.0.1 --port 8000 \
  --quantization auto_gptq --linear-backend "$backend" --dtype float16 \
  --tensor-parallel-size 1 --max-model-len 8192 \
  --max-num-seqs 32 --max-num-batched-tokens 8192 \
  --kv-cache-memory-bytes 8589934592 --gpu-memory-utilization 0.80 \
  --attention-backend FLASH_ATTN --seed 42 --no-enable-prefix-caching \
  --disable-uvicorn-access-log 2>&1 | tee "$out/server.log"
```

Check `server.log` for `Using MarlinLinearKernel for AutoGPTQLinearMethod` or
`Using HummingLinearKernel for AutoGPTQLinearMethod`, as appropriate. Do not
compare results if the requested backend was not selected. Confirm the same
attention backend in both logs too. This example uses the default CUDA Graph
path. If you need eager mode, add `--enforce-eager` to both runs and report it
separately.

The fixed 8 GiB KV cache prevents a change in weight or workspace memory from
also changing KV cache capacity. Keep these server limits, environment
overrides, and prefix-cache settings the same for both backends.

## Measure request performance

In a second terminal, activate the same environment and enter the same
directory. Set `backend` and `round` to match the active server, then wait for
its health endpoint to return HTTP 200.

```bash
set -euo pipefail
cd /path/to/quantized-backend-bench
backend=marlin
round=1
out="$PWD/results/$backend-r$round"
curl --fail http://127.0.0.1:8000/health

for input_len in 128 2048; do
  for concurrency in 1 8 32; do
    name="serve-i${input_len}-o128-c${concurrency}"
    vllm bench serve \
      --backend vllm --base-url http://127.0.0.1:8000 \
      --model qwen-gptq --tokenizer "$PWD/model" \
      --dataset-name random --random-input-len "$input_len" \
      --random-output-len 128 --random-range-ratio 0 \
      --num-prompts 96 --max-concurrency "$concurrency" --request-rate inf \
      --seed 42 --ignore-eos --num-warmups 4 --temperature 0 \
      --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,95,99 \
      --save-result --save-detailed --result-dir "$out" \
      --result-filename "$name.json" 2>&1 | tee "$out/$name.log"
  done
done
```

`--random-range-ratio 0` fixes the prompt lengths. `--ignore-eos` fixes the
output length at 128 tokens. The four warmup requests are outside the 96
measured requests. Check each JSON file: `completed` must be 96, `failed` must
be 0, `errors` must be empty strings, and all input and output lengths must
match the requested lengths. Check the server log for JIT work, preemption,
or errors during the timed interval. Preserve failed runs as well.

After all six cases finish, stop the server with Ctrl-C and wait for its GPU
processes to exit. Repeat the server and benchmark steps in this order:

| Round | First backend | Second backend |
| --- | --- | --- |
| 1 | Marlin | Humming |
| 2 | Humming | Marlin |
| 3 | Marlin | Humming |

Change both terminals' `backend` and `round` values for each run. Do not run two
servers at once or reuse a result directory. Report each round and the median
across rounds for output tokens/s, time to first token (TTFT), and time per
output token (TPOT). Include variability; small gains can depend on clock,
temperature, and load. [Parameter sweeps](sweeps.md) can automate larger grids.

## Check model accuracy

Run an evaluation separately from the throughput benchmark, with the same
server settings for each backend. For this model, the repository's
[GSM8K evaluator](../../tests/evals/gsm8k/gsm8k_eval.py) supports a full 1,319-question
test. Use the script from the same vLLM revision as the server.

Restart each backend with the same settings and a new result directory. Set
`backend`, `round`, and `out` in both terminals to match that run. Run the
evaluation once per backend before you stop its server:

```bash
"$VIRTUAL_ENV/bin/python" /path/to/vllm/tests/evals/gsm8k/gsm8k_eval.py \
  --host http://127.0.0.1 --port 8000 \
  --num-questions 1319 --num-shots 5 --max-tokens 512 \
  --temperature 0 --top-p 1 --seed 42 --max-concurrency 32 \
  --request-timeout-seconds 1800 --save-results "$out/gsm8k.json"
```

Use the Python executable from the active vLLM environment. This command saves
aggregate scores, not per-question responses. If scores differ, retain prompts,
labels, responses, and output token counts with an evaluation harness before
you attribute the difference to a backend. Check for failed requests and
responses at the token limit. Equal total scores do not imply identical
answers; a fixed seed does not guarantee [reproducible outputs](../usage/reproducibility.md).

## Report startup cost and scope

Measure startup from process launch until `/health` succeeds. Report the first
start with fresh per-backend caches separately from warm restarts with those
caches intact. Exclude model download. Note any shared caches, such as
FlashInfer's, and whether model files were already in the OS page cache.
Fresh compiler directories alone do not make the whole run cold.

Include peak device memory and the sample interval if you collect it. A sampled
peak can miss short spikes. Keep the hardware, model and software revisions,
exact commands, logs, raw JSON, and score limits with the report. Results from
one checkpoint support that configuration; other formats, MoE, tensor-parallel
sizes, and architectures need their own checks before a broad default change.
