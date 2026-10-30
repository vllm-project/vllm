# DoSA sparse indexer

DoSA accelerates DSA prefill by scoring a seed subset, screening keys with score
bounds and refining survivors. Enable with
`--attention-config '{"indexer_dosa":true}'`.

The optimized path supports SM100, CUDA 13, FP8 index keys, 32/64 index heads,
head dimension 128, top-k 2048 and gathered contexts of 131072–1048576 keys.
Decode, short query/key spans, MXFP4, context parallelism, speculative decoding
and candidate-block selection use the existing backend. The feature is disabled
by default.

Selection uses native SM100 MX/BF16 scoring. Capacity and numerical admission
failures recover exhaustively; equal-score ties may choose different valid IDs.
The workspace manager owns scratch and private graphs before KV-cache sizing.
Bootstrap and carry state belong to one invocation.

GLM-5.2 TTFT uses SQLite 3.53.4 prefixes, eight SM100 GPUs, FP8 KV and TP8/EP8.
Each cell is the median of 3 formal runs after 1 warmup. Both arms run on the
same eight GPUs with the same client and communication settings. Here 1K = 1024 tokens.

| Context | Q16K stock / DoSA (s) | Speedup | Q32K stock / DoSA (s) | Speedup |
| --- | ---: | ---: | ---: | ---: |
| 128K | 6.168 / 6.121 | 1.0076× | 6.059 / 6.038 | 1.0035× |
| 256K | 15.386 / 13.691 | 1.1238× | 15.044 / 13.474 | 1.1166× |
| 384K | 26.996 / 22.209 | 1.2155× | 26.514 / 21.790 | 1.2168× |
| 512K | 43.317 / 31.692 | 1.3668× | 42.752 / 31.066 | 1.3762× |
| 768K | 96.243 / 53.371 | 1.8033× | 95.704 / 52.484 | 1.8235× |
| ~1M (1048448) | 188.954 / 78.758 | 2.3992× | 189.598 / 77.505 | 2.4463× |

The [published LongBench-v2 reproduction](https://github.com/llm-sys/dosa-vllm/blob/33f5da8eabdd8f77a1733ab322312f7a4653e23e/docs/features/dosa_longbench_v2.md)
reports stock 54/100, DoSA 58/100 and stock A/A 55/100 on 100 fixed requests.
[Earlier DeepSeek TTFT results](https://github.com/llm-sys/dosa-vllm/blob/33f5da8eabdd8f77a1733ab322312f7a4653e23e/docs/features/dosa.md#measured-results)

## TTFT reproduction

Use CUDA 13, PyTorch 2.14.0+cu130 and FlashInfer 0.7.2.post1 on two four-GPU nodes.
Use the shared-fixes prerequisite for stock and this feature revision for DoSA.
On the feature revision, install the matching core and build the extension:

```bash
uv venv --python 3.12
source .venv/bin/activate
VLLM_USE_PRECOMPILED=1 VLLM_PRECOMPILED_WHEEL_VARIANT=cu130 \
    uv pip install -e . --torch-backend=cu130
uv pip install -r requirements/build/cuda.txt --torch-backend=cu130
uv run --active --no-sync python tools/generate_cmake_presets.py --force-overwrite
cmake --preset release
cmake --build --preset release --target _dosa_C --parallel 8
cmake --install cmake-build-release --component _dosa_C
```

Download the existing reproduction tools and generate the inputs:

```bash
export DOSA_TTFT_TOOLS=results/dosa-ttft-tools
mkdir -p "$DOSA_TTFT_TOOLS"
for file in prepare_inputs.py ttft_client.py; do
    curl -fL "https://raw.githubusercontent.com/llm-sys/dosa-vllm/33f5da8eabdd8f77a1733ab322312f7a4653e23e/benchmarks/dosa/$file" -o "$DOSA_TTFT_TOOLS/$file"
done
uv run --active --no-sync python "$DOSA_TTFT_TOOLS/prepare_inputs.py" --output-dir inputs/dosa
```

On nodes 0/1, set `NODE_RANK=0`/`NODE_RANK=1` and run the following:

```bash
export MODEL_DIR=/path/to/GLM-5.2-FP8
export MASTER_ADDR=NODE0_IP CHUNK=32768 DOSA=false
export NCCL_SOCKET_IFNAME=DATA_NIC_0,DATA_NIC_1
export CUDA_VISIBLE_DEVICES=0,1,2,3 VLLM_SERVER_DEV_MODE=1
export NCCL_NVLS_ENABLE=0 VLLM_ALLREDUCE_USE_FLASHINFER=0
export VLLM_ALLREDUCE_USE_SYMM_MEM=0
unset VLLM_USE_NCCL_SYMM_MEM
NODE_ARGS=()
if [ "$NODE_RANK" = 1 ]; then NODE_ARGS=(--headless); fi
DOSA_ARGS=()
if [ "$DOSA" = true ]; then
    DOSA_ARGS=(--attention-config '{"indexer_dosa":true}')
fi
uv run --active --no-sync vllm serve "$MODEL_DIR" \
    --served-model-name dosa-benchmark --tensor-parallel-size 8 --enable-expert-parallel \
    --nnodes 2 --node-rank "$NODE_RANK" --master-addr "$MASTER_ADDR" --master-port 29810 \
    --distributed-executor-backend mp --max-model-len 1048576 \
    --max-num-batched-tokens "$CHUNK" --max-num-seqs 1 --kv-cache-dtype fp8 \
    --gpu-memory-utilization 0.90 --seed 233767787 --host 0.0.0.0 --port 30210 \
    --compilation-config '{"pass_config":{"fi_allreduce_fusion_max_size_mb":0}}' \
    "${DOSA_ARGS[@]}" "${NODE_ARGS[@]}"
```

Use the same client host for both arms. Run against the stock server:

```bash
CUDA_VISIBLE_DEVICES="" uv run --active --no-sync python \
    "$DOSA_TTFT_TOOLS/ttft_client.py" \
    --requests inputs/dosa/glm/requests.json \
    --base-url "http://$MASTER_ADDR:30210" --model dosa-benchmark \
    --arm stock --source "$(git describe --always --dirty)" --output "results/glm-stock-$CHUNK"
```

Restart both servers with `DOSA=true`, then repeat with `--arm dosa` and a new
output directory. Run both `CHUNK=16384` and `32768` on the same node pair.

The client uses fresh RIDs, prefix resets, temperature 0, a two-token budget and
default EOS. TTFT spans server tokenization through the first completion choice.
