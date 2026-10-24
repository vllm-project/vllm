# GLM-5.2 LongBench-v2 evaluation

On a fixed 100-request LongBench-v2 subset, GLM-5.2 scores **54% with stock and
58% with DoSA**, an observed difference of **4 percentage points**. A separate
stock A/A run scores 55%; it does not replace the original 54% baseline.

| Configuration | Correct / requests | Accuracy |
| --- | ---: | ---: |
| Stock | 54 / 100 | 54% |
| DoSA | 58 / 100 | 58% |
| Stock A/A, same requests | 55 / 100 | 55% |

Both arms use the same prompts, order and generation settings. Five answers
change from incorrect to correct and one from correct to incorrect. Neither arm
has unparsed or length-capped answers. All percentages refer to this frozen
100-request subset.

## Dataset and protocol

- Dataset: `zai-org/LongBench-v2`, revision
  `2b48e494f2c7a2f0af81aae178e05c7e1dde0fe9`.
- Fixed IDs and original prompt-token counts are committed in
  `benchmarks/dosa/longbench_v2_glm.csv`.
- The 100 requests were selected by domain/256K-token bands before model output.
  Prompts are untruncated; use the supplied list rather than selecting again.
- Zero-shot prompt/parser: `THUDM/LongBench` revision
  `2e00731f8d0bff23dc4325161044d0ed8af94c1e`, `prompts/0shot.txt`.
- Greedy, temperature 0, nonthinking, 128 output tokens, default EOS.
- FP8 KV, TP8/EP8, 16K prefill chunks, max context 1048576 and concurrency 1.
- Each request resets the prefix cache and uses a fresh request ID. The original
  A–D parser counts an unparsed answer as incorrect and retains capped answers.

This is a task-accuracy evaluation using chat completions. The SQLite-prefix,
two-output-token workload in [the TTFT guide](dosa.md#ttft-reproduction) measures
latency separately. DeepSeek is outside this accuracy evaluation's scope.

## Reproduce

Use this checkout installed with CUDA 13, PyTorch 2.14.0+cu130 and FlashInfer 0.7.2
on two nodes with four SM100 GPUs each. From the repository root, activate the
installed environment and download the pinned dataset:

```bash
source /path/to/vllm/.venv/bin/activate
uv run --active --no-sync python - <<'PY'
from huggingface_hub import hf_hub_download
hf_hub_download(
    "zai-org/LongBench-v2", "data.json", repo_type="dataset",
    revision="2b48e494f2c7a2f0af81aae178e05c7e1dde0fe9", local_dir="longbench-v2",
)
PY
```

Run the server on both nodes. Set `MASTER_ADDR` to node 0's address and use
`NODE_RANK=0` or `NODE_RANK=1`. Start with `DOSA=false` for stock; both arms use
all the settings below:

```bash
export MODEL_DIR=/path/to/GLM-5.2-FP8
export MASTER_ADDR=NODE0_IP NODE_RANK=0 DOSA=false
export CUDA_VISIBLE_DEVICES=0,1,2,3 VLLM_SERVER_DEV_MODE=1
export NCCL_NVLS_ENABLE=0 VLLM_ALLREDUCE_USE_FLASHINFER=0
export VLLM_ALLREDUCE_USE_SYMM_MEM=0
unset VLLM_USE_NCCL_SYMM_MEM
NODE_ARGS=()
if [ "$NODE_RANK" = 1 ]; then NODE_ARGS=(--headless); fi
uv run --active --no-sync vllm serve "$MODEL_DIR" \
    --served-model-name dosa-benchmark --tensor-parallel-size 8 --enable-expert-parallel \
    --nnodes 2 --node-rank "$NODE_RANK" --master-addr "$MASTER_ADDR" --master-port 29810 \
    --distributed-executor-backend mp --max-model-len 1048576 \
    --max-num-batched-tokens 16384 --max-num-seqs 1 --kv-cache-dtype fp8 \
    --gpu-memory-utilization 0.90 --seed 233767787 --host 0.0.0.0 --port 30210 \
    --compilation-config '{"pass_config":{"fi_allreduce_fusion_max_size_mb":0}}' \
    --attention-config "{\"indexer_dosa\":$DOSA}" "${NODE_ARGS[@]}"
```

On node 0, evaluate stock using the committed sample list:

```bash
uv run --active --no-sync python benchmarks/dosa/longbench_v2.py \
    --data longbench-v2/data.json --base-url "http://$MASTER_ADDR:30210" \
    --model dosa-benchmark --arm stock --output results/longbench-v2-stock
```

Stop both servers normally and restart with `DOSA=true`, keeping the same model,
hardware and settings. Then repeat with `--arm dosa` and a new output directory:

```bash
uv run --active --no-sync python benchmarks/dosa/longbench_v2.py \
    --data longbench-v2/data.json --base-url "http://$MASTER_ADDR:30210" \
    --model dosa-benchmark --arm dosa --output results/longbench-v2-dosa
```

Each output directory contains one raw JSON response per request and `STATE.json`
with predictions, answers, token counts and the correct/100 summary. A separate
A/A control repeats stock in another output directory. Report that control
separately; do not substitute it for the original baseline.
