# Quantized weight-update validation matrix

This matrix records validation of quantized weight updates on an NVIDIA H200
without downloading models. It distinguishes aliases and online shorthands from
checkpoint formats, and does not treat an initialization failure as a reload
failure.

## Weight-update paths

| Path | Quantization | Transport | Result | Evidence |
|---|---|---|---|---|
| `reload_weights` | online FP8 per-tensor | filesystem | Pass | BF16 checkpoint to dummy FP8 runtime; 451 parameter/buffer bindings retained |
| `reload_weights` | online FP8 per-block | filesystem | Pass | 451 bindings retained and inference completed |
| `reload_weights` | online FP8 per-channel | filesystem | Pass | 451 bindings retained and inference completed |
| `reload_weights` | online INT8 weight-only | filesystem | Pass | 339 bindings retained and inference completed |
| `reload_weights` | online MXFP8 | filesystem | Pass | 451 bindings retained and inference completed |
| `reload_weights` | compressed-tensors FP8 block | filesystem | Pass | Reloaded add checkpoint to multiply checkpoint; 451 bindings retained |
| `reload_weights` | compressed-tensors W4A16 | filesystem | Pass | Reloaded add checkpoint to multiply checkpoint; 563 bindings retained |
| `reload_weights` | compressed-tensors W4A8 MoE | filesystem | Pass | 54 bindings retained and inference completed |
| `reload_weights` | GPTQ / AutoGPTQ / GPTQ-Marlin | filesystem | Pass | 555 bindings retained for each explicit entry point |
| `reload_weights` | experts INT8 MoE | filesystem | Pass | BF16 checkpoint to dummy INT8-expert runtime; 34 bindings retained |
| weight transfer | online FP8 per-tensor | packed CUDA IPC | Pass | BF16 Transformers trainer to dummy FP8 runtime; 451 bindings retained; cold-load tokens matched |
| weight transfer | online FP8 per-tensor | packed NCCL | Pass | GPU trainer to vLLM receiver through real PyNccl communicator; 451 bindings retained |
| weight transfer | unquantized recorder | unpacked NCCL | Pass | Cross-process Ray integration test |
| weight transfer | unquantized recorder | unpacked CUDA IPC, Ray and HTTP metadata paths | Pass | Cross-process Ray integration tests |
| sparse weight update | unquantized parameter patch | sparse NCCL | Pass | Cross-process Ray integration test updated only selected entries |

Every binding count above includes named parameters and named buffers. The
checks compare both Python object identity and underlying storage address.

## Registered quantization names

| Registered name | Effective family | H200 result |
|---|---|---|
| `fp8` | FP8 checkpoint / legacy online FP8 | Pass for checkpoint FP8 and BF16-to-FP8 reload |
| `online` | Configurable online quantization | Covered through all registered online shorthands |
| `fp8_per_tensor` | Online FP8 | Pass |
| `fp8_per_block` | Online FP8 | Pass |
| `fp8_per_channel` | Online FP8 | Pass |
| `int8_per_channel_weight_only` | Online INT8 | Pass |
| `mxfp8` | Online shorthand or ModelOpt checkpoint format | Pass for online shorthand; no local ModelOpt MXFP8 checkpoint |
| `compressed-tensors` | Multiple checkpoint schemes | Pass for FP8 block, W4A16, and W4A8 MoE formats; one older TinyLlama W4A8 checkpoint fails initial load because it contains an unregistered `weight_chan_scale` |
| `gptq` | AutoGPTQ implementation | Pass |
| `auto_gptq` | AutoGPTQ implementation | Pass |
| `gptq_marlin` | AutoGPTQ implementation with Marlin selection | Pass |
| `experts_int8` | Online INT8 MoE experts | Pass |
| `awq`, `auto_awq`, `awq_marlin` | AutoAWQ implementation | Not runtime-tested: no local AWQ checkpoint |
| `moe_wna16` | GPTQ/AWQ MoE WNA16 | Not runtime-tested with a compatible MoE checkpoint; the local dense GPTQ checkpoint fails initial load with missing `g_idx` destination |
| `humming` | Humming checkpoint conversion | Environment-limited: Humming NVRTC compilation fails during initial post-load processing |
| `bitsandbytes` | bitsandbytes | Dependency unavailable in the existing environment; not installed per offline/no-download constraint |
| `torchao` | TorchAO | Dependency unavailable in the existing environment; not installed per offline/no-download constraint |
| `modelopt`, `modelopt_fp4`, `modelopt_mxfp8`, `modelopt_mixed` | TensorRT ModelOpt formats | No compatible local checkpoint; not runtime-tested |
| `quark` | AMD Quark formats | No compatible local checkpoint; not runtime-tested |
| `inc` | Intel Neural Compressor format | No compatible local checkpoint; not runtime-tested |
| `mxfp4`, `gpt_oss_mxfp4` | MXFP4 | Environment-limited: local GPT-OSS checkpoint reaches FlashInfer JIT, which fails because the H200 image lacks `cublasLt.h` |
| `deepseek_v4_fp8` | DeepSeek V4 FP8 | No compatible local FP8 checkpoint; not runtime-tested |
| `fbgemm_fp8` | Deprecated FBGEMM FP8 | No compatible local checkpoint; not runtime-tested |
| `fp_quant` | Blackwell FP4/FP8 | Not supported by H200 (minimum compute capability 10.0) |

## Interpretation

`Pass` means the complete load/update lifecycle and inference executed on H200,
and runtime tensor storage remained stable. `Not runtime-tested` is deliberately
not a support claim. Environment and checkpoint limitations are recorded
separately from failures reached after reload begins.
