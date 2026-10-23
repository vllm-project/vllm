# DoSA sparse indexer

DoSA accelerates DSA prefill by screening keys with score bounds and refining
surviving candidates. It is disabled by default:

```bash
vllm serve MODEL_DIR --tensor-parallel-size 8 --enable-expert-parallel \
    --kv-cache-dtype fp8 --max-num-batched-tokens 32768 \
    --attention-config '{"indexer_dosa": true}'
```

The optimized path requires SM100, CUDA 13, FP8 index keys, 32 or 64 index heads,
head dimension 128, top-k 2048 and gathered contexts of 131072–1048576 keys.
Decode, short spans, MXFP4, context parallelism, speculative decoding and
candidate-block selection use the existing backend. Model context limits remain
separate from these kernel limits.

Selection uses native SM100 MX/BF16 scoring, including BF16 per-head rounding
and weighted accumulation. Capacity overflow and numerical admission failures
use exhaustive recovery. Equal-score ties can select different valid key IDs.
The workspace manager owns scratch and private graphs and accounts for them
before KV-cache sizing; bootstrap and carry state belong to one invocation.

## Measured results

TTFT uses SQLite 3.53.4 source prefixes on eight SM100 GPUs, FP8 KV and TP8/EP8.
Here 1K means 1024 tokens.
Each speedup is stock / DoSA, using three formal requests after one warmup.

| Context | GLM Q16K | GLM Q32K | DeepSeek Q16K | DeepSeek Q32K |
| --- | ---: | ---: | ---: | ---: |
| 128K | 0.99× | 1.01× | 0.93× | 0.88× |
| 256K | 1.23× | 1.08× | — | — |
| 384K | 1.17× | 1.16× | 0.97× | 0.97× |
| 512K | 1.29× | 1.36× | 1.02× | 1.02× |
| 768K | 1.66× | 1.69× | 1.33× | 1.35× |
| ~1M | 2.18× | 2.36× | 1.80× | 1.83× |

GLM-5.2 task accuracy uses 100 fixed LongBench-v2 requests, zero-shot, greedy
nonthinking answers, 128 output tokens and Q16K: stock 54/100, DoSA 58/100.
A separate stock A/A run scores 55/100 and does not replace the baseline.
See the [LongBench-v2 reproduction guide](dosa_longbench_v2.md) for the frozen
sample list, evaluation protocol and commands.
DeepSeek results cover TTFT only. Its native context is 163840 tokens; longer
requests use a 1048576-token server limit for latency extrapolation, without an
answer-quality claim. The SQLite prefixes are not independent agent tasks.

## TTFT reproduction

Use this checkout installed with CUDA 13, PyTorch 2.14.0+cu130 and FlashInfer 0.7.2
on two four-GPU nodes. Activate its virtual environment and generate the inputs
from the repository root:

```bash
source /path/to/vllm/.venv/bin/activate
uv run --active --no-sync python benchmarks/dosa/prepare_inputs.py --output-dir inputs/dosa
```

The default plans contain six GLM lengths and five DeepSeek lengths; DeepSeek
256K was not measured. Both arms use the same raw UTF-8 files. Set the checkpoint,
node 0 address and each node's rank, then run this server command on both nodes:

```bash
export MODEL_KIND=glm MODEL_DIR=/path/to/GLM-5.2-FP8
export MASTER_ADDR=NODE0_IP NODE_RANK=0 CHUNK=32768 DOSA=false
export CUDA_VISIBLE_DEVICES=0,1,2,3
export VLLM_SERVER_DEV_MODE=1 VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
unset NCCL_NVLS_ENABLE VLLM_ALLREDUCE_USE_FLASHINFER
unset VLLM_ALLREDUCE_USE_SYMM_MEM VLLM_USE_NCCL_SYMM_MEM
EXTRA_ARGS=()
if [ "$MODEL_KIND" = glm ]; then
    export NCCL_NVLS_ENABLE=0 VLLM_ALLREDUCE_USE_FLASHINFER=0
    export VLLM_ALLREDUCE_USE_SYMM_MEM=0
    EXTRA_ARGS=(--compilation-config '{"pass_config":{"fi_allreduce_fusion_max_size_mb":0}}')
fi
NODE_ARGS=()
if [ "$NODE_RANK" = 1 ]; then NODE_ARGS=(--headless); fi
uv run --active --no-sync vllm serve "$MODEL_DIR" \
    --served-model-name dosa-benchmark --tensor-parallel-size 8 --enable-expert-parallel \
    --nnodes 2 --node-rank "$NODE_RANK" --master-addr "$MASTER_ADDR" --master-port 29810 \
    --distributed-executor-backend mp --max-model-len 1048576 \
    --max-num-batched-tokens "$CHUNK" --max-num-seqs 1 --kv-cache-dtype fp8 \
    --gpu-memory-utilization 0.90 --seed 233767787 --host 0.0.0.0 --port 30210 \
    --attention-config "{\"indexer_dosa\":$DOSA}" "${EXTRA_ARGS[@]}" "${NODE_ARGS[@]}"
```

On node 0, run the client against the healthy stock server:

```bash
uv run --active --no-sync python benchmarks/dosa/ttft_client.py \
    --requests "inputs/dosa/$MODEL_KIND/requests.json" \
    --base-url "http://$MASTER_ADDR:30210" --model dosa-benchmark \
    --arm stock --source dosa-main --output "results/$MODEL_KIND-stock-$CHUNK"
```

Restart both servers with `DOSA=true`, then repeat with `--arm dosa` and a new
output directory. Run both `CHUNK=16384` and `32768` on the same node pair. For
DeepSeek, set `MODEL_KIND=ds` and its checkpoint path; retain native communication
defaults. The four GLM communication settings above apply equally to its two arms.

The client uses vLLM's `async_request_openai_completions`: one warmup and three
formal requests per prefix, fresh request IDs, a prefix-cache reset before every
request, two output tokens, temperature 0 and default EOS. TTFT ends at the first
streamed completion choice and includes server tokenization. Each receipt and the
median of the three formal requests are saved. No profiling hooks are required.
