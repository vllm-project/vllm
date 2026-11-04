# Quantized ReplaySSM checkpoints

ReplaySSM stores recent Mamba2 inputs in a buffer. It updates the stored state checkpoint only when the buffer is flushed.
Quantized checkpoints reduce storage precision and can change model output.
The default, `--replayssm-state-dtype auto`, preserves the existing floating-point checkpoint path.

## Select a checkpoint mode

| Mode | Kernel backend | Checkpoint formats | Decode support |
| --- | --- | --- | --- |
| `shared` | FlashInfer | FP8 E4M3FN or INT8 | Standard and speculative decoding; aligned prefix caching |
| `separate` | CuTe DSL | FP16 recurrence checkpoint plus FP8 or NVFP4 read checkpoint | SM103 standard decode; SM121 standard and speculative decode; no prefix caching |

Both modes require `--use-replayssm`, `--mamba-backend flashinfer`, BF16 model activations, head dimension 64, and state size 128.
The Mamba backend option retains FlashInfer for the existing surrounding execution path. Separate checkpoints select CuTe DSL for ReplaySSM decode.
Unsupported combinations fail during configuration or model initialization.

### Shared checkpoints

One quantized checkpoint supplies both output computation and subsequent checkpoint updates.
FP8 and INT8 use one FP32 scale per state row.
The checkpoint and its scales move together during prefix reuse and state copying.

```bash
vllm serve nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4 \
    --use-replayssm \
    --mamba-backend flashinfer \
    --mamba-ssm-cache-dtype auto \
    --replayssm-checkpoint-mode shared \
    --replayssm-state-dtype fp8 \
    --enable-mamba-cache-stochastic-rounding
```

Use `int8` to select INT8 storage. Shared NVFP4 checkpoints are unsupported.
The existing stochastic-rounding option controls shared decode checkpoint writes.
Stochastic rounding can reduce accumulated quantization error during long generations.
Prefill checkpoint conversion uses nearest rounding.

### Separate checkpoints

The FP16 recurrence checkpoint supplies flush updates. The FP8 or NVFP4 read checkpoint supplies non-flush outputs.
Each flush creates both checkpoints independently from FP32 working values.
Neither checkpoint is rewritten between flushes.
Separate copies add storage relative to a single checkpoint of the same read precision.

```bash
vllm serve nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4 \
    --use-replayssm \
    --mamba-backend flashinfer \
    --mamba-ssm-cache-dtype auto \
    --replayssm-checkpoint-mode separate \
    --replayssm-state-dtype nvfp4 \
    --replayssm-buffer-len 16 \
    --no-enable-prefix-caching
```

Use `fp8` to select an FP8 read checkpoint. Separate INT8 checkpoints are unsupported.
Separate mode requires an SM103 or SM121 GPU and eight or sixteen heads per B/C group.
It requires a replay window of 16 and does not support prefix caching.
SM103 supports standard decoding. SM121 also supports up to seven speculative tokens per verification step.
The kernels always use stochastic FP16 recurrence rounding with seven Philox rounds.
The read copy uses nearest rounding.

NVFP4 stores packed E2M1 values, block-16 E4M3 scales, and an FP32 outer scale.
The decode flush selects the outer scale from maxima recorded for the preceding checkpoint.
This avoids a reduction across thread blocks for the current checkpoint.

### GB10 and speculative verification

GB10 uses the SM121 path. Its CuTe flush kernel uses warp matrix instructions with BF16 operands and FP32 accumulation.
Software stochastic rounding writes the FP16 recurrence checkpoint.
The CuTe read kernel loads each quantized state tile once and reuses the tile across the verification tokens.

The input rings hold 16 committed history entries plus the largest verification window.
Two counters record the committed token position and the pending verification length.
At the next step, the kernel commits only the accepted prefix of the previous verification window.
Rejected tokens never update either checkpoint. A flush occurs only when the committed position crosses a replay boundary.

For Bolt with seven draft tokens, the verification window contains at most eight tokens.
The additional rings and counter require 151,560 bytes per layer and cache slot, before alignment.
The SM103 standard-decode path retains its original ring layout and kernels.

The CuTe kernels derive from [ReplaySSM](https://github.com/Johnny-Liou/ReplaySSM/tree/73a9592e9d8a2268620e8883a82f5cec7442ed36/low_precision).
Their kernel latency results do not establish a full-model throughput gain for this integration.
