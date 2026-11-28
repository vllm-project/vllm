# Mamba2: SSD prefill and selective state update

Return to the [primitive index](../SKILL.md).

SSD computes sequence/chunk outputs; selective state update (SSU) advances
recurrent state. Neither operation is [GDN](gdn.md) or [KDA](kda.md).

## SSD prefill

| Implementation | Language | Origin and pinned source | Phase / contract |
| --- | --- | --- | --- |
| Mamba Triton SSD - `mamba_chunk_scan_combined`, `mamba_chunk_scan_combined_varlen` | Triton | Mamba [upstream v2.2.4][ssd-up]; [vLLM adaptation][ssd-v] | One upstream-derived family. Chunk BMM, discretization/cumsum, chunk state, state passing and output scan are constituent stages. Packed/varlen and initial/final-state contracts vary between upstream and adapted copies. |
| Combined SSD - `SSDCombined`, `ssd_combined_fwd`, `SSDKernel` | CuTe DSL with support kernels | NVIDIA/FlashInfer [composition][ssd-fi] and [device kernel][ssd-kernel] | SM100/103/110; BF16 I/O, final state and token-index checkpoint capture. Explicitly rejects SM120/121 and SM107. Support Triton chunk-state/CUDA scan preparation does not create extra SSD candidates. |
| Cake combined SSD - `CakeSSDCombined` | CUDA, source-built/generated | Cake in FlashInfer [source boundary][cake-ssd] | Combined chunk scan/state propagation alternative to CuTe SSD; route-specific configuration and generated-source requirements. |
| Native selective/chunk scan - `invokeSelectiveScan`, `invokeChunkScan`, `invokeSelectiveScanUpdate` | CUDA | TensorRT-LLM [source][native-scan] | Native state-space scan/update family, separate from Mamba's Triton SSD composition. Match the selective versus chunk scan contract; not every entry is Mamba2 SSD. |
| CPU chunk scan - `mamba_chunk_scan_fwd_cpu_impl` | C++ | vLLM [source][cpu] | CPU Mamba2 prefill, distinct from GPU chunk schedules and CPU decode. |

## Recurrent decode

| Implementation | Language | Origin and pinned source | Phase / contract |
| --- | --- | --- | --- |
| Mamba selective state update - `selective_state_update` | Triton | Mamba [upstream v2.2.4][ssu-up]; [vLLM adaptation][ssu-v] | One upstream-derived SSU family; indexed single/multi-token updates in adapted copies. Not separate framework backends, and not the SSD chunk algorithm. |
| Native selective state update - `selective_state_update` | CUDA | FlashInfer [device kernels][ssu-kernel] and [generic/SM90/SM100 generators][ssu-jit] | State `[cache,heads,dim,dstate]`, optional state indices. Generic and architecture-specialized kernels grouped; dtype/dstate/head-dim determine eligibility. |
| Cake selective state update - `selective_state_update(backend="cake")` | CUDA, source-built/generated | Cake in FlashInfer [source boundary][cake-ssu] | SM100/103 optional decode/MTP route; falls back outside Cake eligibility. |
| CPU selective state update - `selective_state_update_cpu_impl` | C++ | vLLM [source][cpu] | CPU recurrent update; not CPU SSD prefill. |

## Multi-token prediction, verification and replay

| Implementation | Language | Origin and pinned source | Phase / contract |
| --- | --- | --- | --- |
| Multi-token SSU - `selective_state_update` | CUDA | FlashInfer: [simple][mtp-simple], [horizontal][mtp-horizontal], [async horizontal][mtp-async], [vertical][mtp-vertical] | One MTP family with materially different scheduling algorithms. Fixed/varlen token metadata; simple serial, horizontal, asynchronously pipelined horizontal and vertical schedules are not additional provider imports. |
| Tree-state MTP - `selective_state_update_mtp_ssm_cache_trtllm` | CUDA | TensorRT-LLM [call contract][ssu-t] and [kernels][mtp-trt] | Tree-parent state and draft-cache updates differ from ordinary SSU. Scalar/vec4/vec8/vec16 instantiations grouped. |
| Matmul replay checkpointing - `checkpointing_ssu` | CUDA | FlashInfer [API][checkpoint] and [kernel][checkpoint-kernel] | Cached X/B/dt ring and accepted-token replay into in-place state. Precompute stages belong to this family. |
| Quantized-state checkpointing - `checkpointing_ssu(state_scale=...)` | CUDA | FlashInfer [8-bit kernel][checkpoint8] | Separate 8-bit-state representation with per-state scales and stochastic-rounding parameters; not ordinary FP32-state replay. |
| Selected-prefix materialization - `replayssm_materialize` | CUDA | FlashInfer [source][replay] | Replays an accepted-token prefix from a source state slot into a separate destination slot, rather than merely selecting an already-stored state. |

## Companion causal convolution

Convolution maintains a short input window, not the SSM recurrence.

| Implementation | Language | Origin and pinned source | Phase / contract |
| --- | --- | --- | --- |
| Causal-conv1d - `causal_conv1d_fn` / `causal_conv1d_fwd`, `causal_conv1d_update` | CUDA | Dao-AILab [upstream project][conv-up] (copy revision not recorded); [TRTLLM CUDA adaptation][conv-t] | Prefill and recurrent convolution, packed-sequence/window-state variants. CUDA device source is retained here because vLLM's corresponding implementation is Triton, not a CUDA copy. AOT/JIT delivery is not by itself another algorithm. |
| Native context/generation convolution - `invokeMambaConv1dContext`, `invokeMambaConv1dGeneration` | CUDA | TensorRT-LLM [source][native-conv] | Separate native convolution family, not the causal-conv1d package ABI or SSD. |

## Naming and exclusions

TokenSpeed's [`MambaAttnBackend` source][ts-mamba] imports GDN and convolution
operations: its name is **not evidence of Mamba2 SSD or SSU**.
Similarly, vLLM's Mamba1 selective-scan path is not Mamba2 SSD prefill.
FlashInfer imports in serving frameworks are counted only at the provider
above. Local Triton replay-index/scatter helpers, output-only one-offs,
convolution helpers and gated norm/layout glue are excluded; the substantial
upstream Mamba Triton pipelines are retained.

The vLLM SSD and SSU headers name Mamba v2.2.4; its resolved commit is
linked below without updating the retained checkouts. TRTLLM remains for
native CUDA scan/convolution and tree-state MTP, not generic Mamba copies.
The causal-conv1d attribution names upstream `main`, not an exact revision;
the [vLLM convolution source][conv-v] is a Triton adaptation and is not
substituted as evidence for the CUDA family.

[ssd-up]: https://github.com/state-spaces/mamba/blob/95d8aba8a8c75aedcaa6143713b11e745e7cd0d9/mamba_ssm/ops/triton/ssd_combined.py
[ssu-up]: https://github.com/state-spaces/mamba/blob/95d8aba8a8c75aedcaa6143713b11e745e7cd0d9/mamba_ssm/ops/triton/selective_state_update.py
[ssd-v]: https://github.com/vllm-project/vllm/blob/435c96f9dbdd29258cb8e0f433c5b54a00cf6b16/vllm/model_executor/layers/mamba/ops/ssd_combined.py
[ssd-fi]: https://github.com/flashinfer-ai/flashinfer/blob/15e83b7bb9f32d81d84017e2e715c265fb7253f5/flashinfer/mamba/ssd_combined.py
[ssd-kernel]: https://github.com/flashinfer-ai/flashinfer/blob/15e83b7bb9f32d81d84017e2e715c265fb7253f5/flashinfer/mamba/ssd_kernel.py
[cake-ssd]: https://github.com/flashinfer-ai/flashinfer/blob/15e83b7bb9f32d81d84017e2e715c265fb7253f5/flashinfer/mamba/cake_ssd_combined.py
[native-scan]: https://github.com/NVIDIA/TensorRT-LLM/blob/a8ac7e5bccb972b35808dc973f7b4aac96cbbc13/cpp/tensorrt_llm/kernels/selectiveScan/selectiveScan.cu
[cpu]: https://github.com/vllm-project/vllm/blob/435c96f9dbdd29258cb8e0f433c5b54a00cf6b16/csrc/cpu/mamba_cpu.cpp
[ssu-v]: https://github.com/vllm-project/vllm/blob/435c96f9dbdd29258cb8e0f433c5b54a00cf6b16/vllm/model_executor/layers/mamba/ops/mamba_ssm.py
[ssu-t]: https://github.com/NVIDIA/TensorRT-LLM/blob/a8ac7e5bccb972b35808dc973f7b4aac96cbbc13/tensorrt_llm/_torch/modules/mamba/selective_state_update.py
[ssu-kernel]: https://github.com/flashinfer-ai/flashinfer/blob/15e83b7bb9f32d81d84017e2e715c265fb7253f5/include/flashinfer/mamba/kernel_selective_state_update_stp.cuh
[ssu-jit]: https://github.com/flashinfer-ai/flashinfer/blob/15e83b7bb9f32d81d84017e2e715c265fb7253f5/flashinfer/jit/mamba/selective_state_update.py
[cake-ssu]: https://github.com/flashinfer-ai/flashinfer/blob/15e83b7bb9f32d81d84017e2e715c265fb7253f5/flashinfer/mamba/cake_selective_state_update.py
[mtp-simple]: https://github.com/flashinfer-ai/flashinfer/blob/15e83b7bb9f32d81d84017e2e715c265fb7253f5/include/flashinfer/mamba/kernel_selective_state_update_mtp_simple.cuh
[mtp-horizontal]: https://github.com/flashinfer-ai/flashinfer/blob/15e83b7bb9f32d81d84017e2e715c265fb7253f5/include/flashinfer/mamba/kernel_selective_state_update_mtp_horizontal.cuh
[mtp-async]: https://github.com/flashinfer-ai/flashinfer/blob/15e83b7bb9f32d81d84017e2e715c265fb7253f5/include/flashinfer/mamba/kernel_selective_state_update_mtp_async_horizontal.cuh
[mtp-vertical]: https://github.com/flashinfer-ai/flashinfer/blob/15e83b7bb9f32d81d84017e2e715c265fb7253f5/include/flashinfer/mamba/kernel_selective_state_update_mtp_vertical.cuh
[mtp-trt]: https://github.com/NVIDIA/TensorRT-LLM/tree/a8ac7e5bccb972b35808dc973f7b4aac96cbbc13/cpp/tensorrt_llm/kernels/mamba2MTPSSMCache
[checkpoint]: https://github.com/flashinfer-ai/flashinfer/blob/15e83b7bb9f32d81d84017e2e715c265fb7253f5/flashinfer/mamba/checkpointing_ssu.py
[checkpoint-kernel]: https://github.com/flashinfer-ai/flashinfer/blob/15e83b7bb9f32d81d84017e2e715c265fb7253f5/include/flashinfer/mamba/kernel_checkpointing_ssu.cuh
[checkpoint8]: https://github.com/flashinfer-ai/flashinfer/blob/15e83b7bb9f32d81d84017e2e715c265fb7253f5/include/flashinfer/mamba/kernel_checkpointing_ssu_8bit.cuh
[replay]: https://github.com/flashinfer-ai/flashinfer/blob/15e83b7bb9f32d81d84017e2e715c265fb7253f5/csrc/replayssm_materialize.cu
[conv-up]: https://github.com/Dao-AILab/causal-conv1d
[conv-v]: https://github.com/vllm-project/vllm/blob/435c96f9dbdd29258cb8e0f433c5b54a00cf6b16/vllm/model_executor/layers/mamba/ops/causal_conv1d.py
[conv-t]: https://github.com/NVIDIA/TensorRT-LLM/blob/a8ac7e5bccb972b35808dc973f7b4aac96cbbc13/cpp/tensorrt_llm/kernels/causalConv1d/causalConv1d.cu
[native-conv]: https://github.com/NVIDIA/TensorRT-LLM/blob/a8ac7e5bccb972b35808dc973f7b4aac96cbbc13/cpp/tensorrt_llm/kernels/mambaConv1dKernels.cu
[ts-mamba]: https://github.com/lightseekorg/tokenspeed/blob/d0a2d1e02a1a643f58737894de109d0f40d83334/python/tokenspeed/runtime/layers/attention/backends/state/mamba.py
