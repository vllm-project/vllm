# nvidia/DeepSeek-V3.2-NVFP4

An optimized implementation for `nvidia/DeepSeek-V3.2-NVFP4` with FP8 FlashInfer MLA on Blackwell GPUs.

The main win comes from aggressively fusing ops in the attention path, across the MLA and sparse-indexer boundary, which is critical for low latency.
On top of manual fusions, the implementation uses `torch.compile` with vLLM's custom fusion passes to fuse remaining miscellaneous ops.
It is compatible with piecewise CUDA graphs for prefill and full CUDA graphs for decode.

TP and EP are supported; PP is not.
MTP is supported.

## Usage

```bash
export VLLM_USE_SPECIALIZED_MODELS=1
export VLLM_USE_V2_MODEL_RUNNER=1
export TRTLLM_ENABLE_PDL=1

NUM_GPUS=4

# With TP
vllm serve nvidia/DeepSeek-V3.2-NVFP4 \
    -tp 4 \
    --compilation-config '{"max_cudagraph_capture_size": 1024}' \
    --speculative-config '{"method": "mtp", "num_speculative_tokens": 1}' \
    --kernel-config.enable_flashinfer_autotune=False

# With attention DP + MoE EP
vllm serve nvidia/DeepSeek-V3.2-NVFP4 \
    -dp $NUM_GPUS -ep \
    --compilation-config '{"max_cudagraph_capture_size": 1024}' \
    --speculative-config '{"method": "mtp", "num_speculative_tokens": 1}' \
    --kernel-config.enable_flashinfer_autotune=False
```
