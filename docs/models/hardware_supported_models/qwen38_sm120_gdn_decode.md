# Experimental Qwen3.8 GDN decode fusion on SM120

`additional_config.gdn_norm_quant_decode` is a **default-off** optimization for
Qwen4Exp's Gated DeltaNet (GDN) output path. It combines RMS normalization,
sigmoid gating and MXFP8 activation quantization for physical decode batches
of one or two tokens, preserving the intermediate BF16 rounding. The existing
output-projection GEMM consumes the quantized activation. This is per-layer
fusion, not a whole-model megakernel.

## Scope and fallback

The measured target is
[`local-inference-lab/Qwen3.8-Flash-Next-NVFP4`](https://huggingface.co/local-inference-lab/Qwen3.8-Flash-Next-NVFP4/tree/ada4da32a583a78aa47299f45a70603c950490b8)
at revision `ada4da32a583a78aa47299f45a70603c950490b8`, on one RTX PRO 6000
Blackwell Server Edition (SM120). This checkpoint uses mixed quantization;
its GDN output projections use MXFP8 despite NVFP4 appearing in the model name.

The opt-in requires BF16 model activations, the native FlashInfer CUTLASS MXFP8
output consumer, 36 GDN layers with 48 value heads of dimension 128,
`CompilationMode.NONE`, and `FULL_DECODE_ONLY` CUDA graphs. Parallel sizes must
be one. Speculative decoding, LoRA, expert/sequence parallelism, microbatching,
batch-invariant mode, generic weight offload and sleep mode are rejected.
PLE CPU offload is supported by the measured recipe. Weight reload/refit and
concurrent invocation of an individual layer are unsupported.

Prefill, mixed batches, physical batches above two, missing metadata and
unsupported runtime layouts retain the original normalization/output path.
Each layer owns its decode buffers. The convolution, recurrence, attention
state, routed/shared experts and HyperConnection paths remain unchanged.
HTTP request concurrency is not necessarily the physical decode batch size.

## Prepare the pinned checkpoint

Use an installed vLLM checkout and a local snapshot of the exact revision above.
The original config contains legacy architecture names; the measured setup used
a separate normalized config/tokenizer directory. Do not overwrite the weights
or infer that the unmodified Hub config loads directly.

Set `SNAPSHOT` to the downloaded revision, then create a new directory:

```bash
export SNAPSHOT=/path/to/ada4da32a583a78aa47299f45a70603c950490b8
export QWEN_CONFIG="$PWD/qwen38-config"
uv run --no-project --python .venv/bin/python python - <<'PY'
import hashlib
import json
import os
import shutil
from pathlib import Path

source = Path(os.environ["SNAPSHOT"])
target = Path(os.environ["QWEN_CONFIG"])
raw = (source / "config.json").read_bytes()
assert hashlib.sha256(raw).hexdigest() == (
    "2d655dfa625c3c018f9685031367bc83f8532d3faf3333d7c838b2bdc13b8a7e"
)
config = json.loads(raw)
config["architectures"] = ["Qwen4ExpForConditionalGeneration"]
config["model_type"] = "qwen4_exp"
config["text_config"]["model_type"] = "qwen4_exp_text"
config["vision_config"]["model_type"] = "qwen4_exp_vision"
normalized = (json.dumps(config, indent=2) + "\n").encode()
assert hashlib.sha256(normalized).hexdigest() == (
    "96ef214ca206fcc8da270304ea4b1bacdb6184de9e5c33baa2fcef8b58da127a"
)
target.mkdir(exist_ok=False)
(target / "config.json").write_bytes(normalized)
for pattern in ("tokenizer*", "special_tokens_map.json", "added_tokens.json",
                "vocab*", "merges.txt", "chat_template*", "hf_quant_config.json"):
    for path in source.glob(pattern):
        if path.is_file():
            shutil.copy2(path, target / path.name)
assert (target / "tokenizer.json").is_file()
assert (target / "tokenizer_config.json").is_file()
PY
```

Only four configuration names change. Model dimensions, layer types and
quantization metadata remain intact; the normalized directory contains no
model weights.

## Enable the experiment

The following uses the measured serving settings. The draft implementation on
the current branch still requires GPU validation; the measurements below are
from its pinned precursor.

```bash
export CUDA_VISIBLE_DEVICES=0
export VLLM_USE_V2_MODEL_RUNNER=1
export MODEL_ID=local-inference-lab/Qwen3.8-Flash-Next-NVFP4
export MODEL_REVISION=ada4da32a583a78aa47299f45a70603c950490b8
uv run --no-project --python .venv/bin/python python -m vllm.entrypoints.cli.main \
  serve "$SNAPSHOT" --revision "$MODEL_REVISION" \
  --hf-config-path "$QWEN_CONFIG" --tokenizer "$QWEN_CONFIG" \
  --served-model-name "$MODEL_ID" --host 127.0.0.1 --port 8000 \
  --tensor-parallel-size 1 --language-model-only --dtype bfloat16 \
  --generation-config vllm --max-model-len 4096 --max-num-seqs 8 \
  --max-num-batched-tokens 2048 --gpu-memory-utilization 0.90 \
  --no-enable-prefix-caching --mamba-ssm-cache-dtype float32 \
  --engram-config '{"cpu_offload":true}' \
  --compilation-config '{"mode":0,"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[1,2,4,8]}' \
  --additional-config '{"gdn_norm_quant_decode":true}' --seed 0
```

For the stock control, omit `--additional-config` or set the boolean to `false`.
Keep all other settings and runtime versions identical. The flag does not
enable an experimental MoE or HyperConnection backend.

## Validation and measurement

The precursor ran on vLLM commit
`0beab6c251e8046c0c6cf2fef114cd23b58cda74`, with Torch 2.14.0+cu130,
Transformers 5.20.2, Triton 3.8.0 and FlashInfer 0.7.2.post1. Its public-layer
probe passed 107 checks; independent comparison of 39 saved tensor artifacts
found zero unequal bytes in 114 comparisons. Eight profiles verified six graph
replays each: 36 fused producers per step at B1/B2, zero at B4/B8, and unchanged
native output-GEMM identities. These are precursor results, not validation of a
rebased or refactored implementation.

One stock/GDN/GDN/stock experiment measured three runs per mode at each
concurrency, with 2048 input tokens and 256 output tokens. Giving equal weight
to each mode's two phases produced:

| Maximum concurrency | Stock TPOT (ms) | GDN TPOT (ms) | TPOT reduction | Output throughput change |
| --- | ---: | ---: | ---: | ---: |
| 1 | 9.4464 | 9.4248 | 0.229% | +0.358% |
| 2 | 10.7074 | 10.5942 | 1.058% | +0.303% |
| 4 | 14.0114 | 14.0183 | -0.049% | -0.302% |
| 8 | 20.7130 | 20.7162 | -0.015% | +0.004% |

All three paired C1/C2 decode runs improved. C2 mean first-token latency was
3.622% higher, and its first throughput comparison regressed. Larger-batch
controls had appreciable phase drift. This is one allocation, not independent
replication or a general throughput guarantee.

To repeat the workload, run stock A1 and GDN B1 with repeat 1, then GDN B2 and
stock A2 with repeats 2/3. For each repeat, measure `C=1,2,4,8` in the same order
for both modes. Use unique result names and keep raw request details:

```bash
C=2 PHASE=stock-a1 REPEAT=1
uv run --no-project --python .venv/bin/python python -m vllm.entrypoints.cli.main \
  bench serve --backend vllm --base-url http://127.0.0.1:8000 \
  --endpoint /v1/completions --model "$SNAPSHOT" \
  --served-model-name "$MODEL_ID" --tokenizer "$QWEN_CONFIG" \
  --dataset-name random --random-input-len 2048 --random-output-len 256 \
  --random-range-ratio 0 --random-prefix-len 0 \
  --num-prompts "$((16 * C))" --num-warmups 8 --request-rate inf \
  --max-concurrency "$C" --ignore-eos --temperature 0 --seed 0 \
  --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,90,99 \
  --save-result --save-detailed --result-dir results \
  --result-filename "${PHASE}-c${C}-r${REPEAT}.json"
```

Run the same full quality workload once per mode, using concurrency two to
exercise the intended decode regime:

```bash
uv run --no-project --python .venv/bin/python python \
  tests/evals/gsm8k/gsm8k_eval.py --model "$MODEL_ID" \
  --num-questions 1319 --num-shots 5 --max-tokens 512 \
  --temperature 0 --seed 42 --max-concurrency 2 \
  --request-timeout-seconds 3600 --use-chat-completions \
  --chat-template-kwargs '{"enable_thinking":false}' \
  --save-results "results/gsm8k-${PHASE}.json"
```

The precursor additionally retained per-question answers: stock scored
1280/1319 and GDN 1277/1319, with eight stock-only and five GDN-only correct
answers and no invalid responses. Four 45-request chat captures passed
functional checks. Strict repeat comparisons failed for stock as well as GDN
(39/45 stock-stock, 35/45 GDN-GDN, 38/45 stock-GDN). **Quality equivalence is not
established.** Exact layer comparisons and baseline instability do not remove
this limitation; the feature remains experimental and disabled by default.
