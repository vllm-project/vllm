# Kimi Delta Attention: prefill, decode and MTP

Return to the [primitive index](../SKILL.md).

KDA uses per-key-channel gates; its state and gate ABI are not interchangeable
with scalar-gated [GDN](gdn.md) or [Mamba2](mamba2.md).

## Chunked prefill

| Implementation | Language | Origin and pinned source | Phase / contract |
| --- | --- | --- | --- |
| FLA KDA - `chunk_kda` | Triton | Flash Linear Attention [upstream v0.5.1][fla-up]; [vLLM adaptation][fla-v] | One FLA-derived chunk family. Gate prefix, triangular interactions/solve, transformed operands, state propagation and output grouped; not a list of stage exports. Copy-specific packed/initial/final-state options; the upstream release is a family reference, not an exact vLLM copy revision. |
| Blackwell chunk KDA - `chunk_kda_cutedsl` | CuTe DSL with Triton prologue | SGLang [source][s-chunk] | KDA adaptation of the local Blackwell chunk pipeline: per-channel gate preparation, KKT/UW, state and output. The prologue is Triton, not CuTe DSL and not another backend. |
| NVIDIA persistent KDA - `chunk_kda_fwd`; fused K123 + persistent K4 or fused K1234 | CuTe DSL | NVIDIA/TRTLLM [K123][nvidia-t123], [K1234][nvidia-t1234] and [K4][nvidia-t4]; KDA_prefill [vendored SGLang pipeline][nvidia-s] | Blackwell persistent family with material fusion variants. SGLang credits the NVIDIA package but records no upstream URL/revision; its reviewed copy is retained for that pipeline. SM100/103, not SM120. Gate preparation, inverse and state/output stages belong to the pipeline; BF16 triangular inverse is not an independent KDA candidate. |
| Hand-written tcgen05 prefill - `chunk_kda_fwd` | CUDA + inline PTX | `kda_prefill` artifact, vendored in SGLang: [device source][ptx], [origin/contract][ptx-api] | Attribution names short commit `33583615` but no resolvable upstream repository. SM103a/GB300, K=V=128, chunk 64. Fused long-sequence and two-launch many-head/sequence routes; TMEM/tcgen05-specific encodings. No claim of speculative rollback or FP32 track-buffer support. |
| BT16 chunked KDA - `kda_chunked_bt16.run` | CuTe DSL | FlashInfer [source][bt16] | Blackwell two-stage prepare/chain algorithm with BT16 chunk factors. |
| SM120 chunked KDA - `decomp.run`, `fused.run` | CuTe DSL | FlashInfer [runtime][sm120], [decomposed][decomp], [fused][fused120] | Two material schedules: decomposed prepare operands + recurrence versus one 512-thread CTA per sequence/head retaining factors in shared memory. |
| Helion chunk KDA - `chunk_kda` | Helion | SGLang [source][helion-prefill] | Complete gate/intra/solve/state/output pipeline with optional Helion dependency, not a Triton import alias. |
| Packaged CuTe KDA - `cutedsl_kda_forward` | CuTe DSL provider | `tokenspeed_cutedsl_kda`; [package integration evidence][ts-cute] | Token-major prefill, V-major state, raw beta logits, dt_bias and gate bound validated against the AOT build; explicit workspace. Package device source was not reviewed, so no equality with the in-tree CuTe families is asserted. |

## Decode and fused recurrence

| Implementation | Language | Origin and pinned source | Phase / contract |
| --- | --- | --- | --- |
| FLA recurrent KDA - `fused_recurrent_kda` | Triton | Flash Linear Attention [upstream v0.5.1][fla-rec-up]; [vLLM adaptation][fla-v]; [TokenSpeed state-pool adaptation][fla-rec-ts] | Recurrent phase of the FLA family; channel-gated state update. TokenSpeed explicitly names v0.5.1 and adds separate read/write pool indices plus an optional gated RMSNorm epilogue. These are adaptation contracts, not another provider. |
| Recurrent KDA - `recurrent_kda` | CuTe DSL | FlashInfer [kernel][fi-rec] | SM100, BF16 `[N,HV,V,K]` state, GQA/varlen and fused multi-token speculative updates. One-warp/grouped-CTA variants grouped. |
| Packed / fully fused KDA - `packed_kda_decode`, `fused_kda_decode` | CuTe DSL | FlashInfer [packed][fi-packed] and [fused][fi-fused] kernels | Packed projections/state versus full width-4 convolution + SiLU + recurrence + gated RMSNorm. Fused route: SM100/103, D=128, heads 12/24/32/48/96; not just a renamed unpacked recurrent API. |
| Fused gate + KDA - `cutedsl_fused_sigmoid_gating_kda_update` | CuTe DSL | SGLang [source][s-rec] | Pooled-state KDA decode, separate from the scalar GDN kernel. |
| Packed native KDA - `kda_packed_decode` | CUDA | SGLang [device source][s-packed], ported from its Triton packed recurrence | Row-streaming state update rather than a whole state tile held in registers. FP32 recurrence with warp reductions; distinct from the fully fused convolution/decode family. |
| Native fused KDA - `invokeKdaDecode`, `fused_kda_decode`, `kda_fused_decode` | CUDA | NVIDIA-authored [vLLM source][v-decode]; [TRTLLM variant][trt-decode]; SGLang [NVIDIA and Moonshot vendored fusion][s-fused] | Grouped native CUDA family with implementation-specific layout/head/feature dispatch, not a claim of identical copies. SGLang's convolution + recurrence + RMSNorm specialization: K=V=128, width 4, T=1, heads 12/6/3. |
| AMD fused KDA - `fused_kda_decode`, `fused_kda_chunk`, `fused_kda_prologue` | HIP | vLLM [decode][v-hip] and [chunk][v-hip-chunk] source | Separate ROCm native recurrence/chunk implementation; CUDA eligibility does not establish HIP support. |
| Helion packed decode - `helion_fused_recurrent_kda_packed_decode` | Helion | SGLang [source][helion-decode] | Packed recurrent state/input validation; distinct algorithm from Helion chunk prefill. |

## Multi-token verification and replay

| Implementation | Language | Origin and pinned source | Phase / contract |
| --- | --- | --- | --- |
| Kimi K3 multi-token recurrence - `kda_decode_mtp_kernel`, `fused_kda_decode_mtp_dspark` | CuTe DSL | NVIDIA/TRTLLM [MTP source][trt-mtp]; SGLang [Dspark variant][s-mtp] | Short multi-token state/update family; Dspark fusion and ReplaySSM ring options are material integration variants, not FlashInfer import wrappers. |
| WY output-only KDA - `kda_wy_output_only`, `kda_recoverssm_verify` | CuTe DSL | FlashInfer [source][wy] | BF16-state speculative/output-only recurrence with a distinct state-write policy. |
| Helion replay decode - `helion_fused_recurrent_kda_replayssm_decode` | Helion | SGLang [source][helion-replay] | Replay-aware recurrent update, separate from ordinary packed decode. |

## Additional provider families

These span phases or are external package boundaries, not serving adapters.

| Implementation | Language | Origin and pinned source | Phase / contract |
| --- | --- | --- | --- |
| FlashKDA - `flash_kda.fwd`, `torch.ops._flashkda_C.fwd`; generated prefill/decode variants | CUDA/CUTLASS; generated CUDA variants | MoonshotAI-derived [FlashKDA device source, vLLM-maintained fork][flash-up]; [vLLM integration][flash-v]; FlashInfer [generated CUDA sources][flash-source], [prefill manifest][flash-gen] and [decode manifest][flash-decode] | The pinned dependency implements two-stage prefill. Reviewed vLLM selection: SM9x/10x/12x, D=128, BF16 inputs, BF16/FP32 state, bounded gate; explicit workspace and optional checkpoint state. FlashInfer's generated/NVRTC/indexed SM100/103 variants and separate decode are not assumed identical to this dependency. The vLLM integration is not device source. |
| Cake KDA - `recurrent_kda(backend="cake")`, packed/fused decode routes | CUDA, source-built/generated | Cake in FlashInfer [prefill][cake], [decode][cake-decode], [packed T1][cake-packed], [fused decode][cake-fused] manifests | SM100/103 affine/BT16 prefill and frozen-plan path; scalar/packed/fused T1 decode are material phase/fusion variants of the provider. |
| cuDNN KDA - `cudnn_recurrent_kda` | External cuDNN | NVIDIA cuDNN; FlashInfer [dependency graph integration][cudnn] | Prefill/decode graph backend with its own state/options restrictions. Device implementation not reviewed. |
| AMD KDA - `gluon_kda_paged_prefill_gfx950`, `gluon_kda_recurrent_decode_gfx950`, `gluon_kda_fused_verify_gfx950`, `gluon_kda_fused_replay_gfx950` and gfx1250 counterparts | Gluon | TokenSpeed AMD in-tree device source: [gfx950 prefill][ts-prefill950] / [decode and replay][ts-decode950], [gfx1250 prefill][ts-prefill1250] / [decode and replay][ts-decode1250] | Architecture-specific chunk prefill, standalone/fused V-major decode and no-store verify/replay. Constituent kernels and architecture variants grouped as one implementation family, not framework import wrappers. |
| AMD/AITER-derived fused KDA - `flydsl_kimi_k3_kda_decode_with_f_b` | FlyDSL | SGLang [vendored device kernel][aiter-device] and [source-selection loader][aiter-loader] | Optional ROCm fused decode with head-local f_b projection, distinct from native HIP above. Loader defaults to the in-tree AMD-authored implementation, then tries AITER; no exact upstream AITER device revision was established for this specialization. The loader itself is not device source. |

Local Triton gate/conv/state-copy one-offs and import-only FlashInfer/FLA
serving surfaces are omitted. The provider families above preserve actual
phase, language and state-layout alternatives without counting every helper
or framework registration as a new implementation.

Other-framework sources remain for substantive local CuTe/CUDA/Helion/Gluon
implementations, speculative-state adaptations, and packages without a
verified equivalent vLLM source. In particular, the NVIDIA KDA_prefill and
hand-written PTX artifacts have no verifiable upstream repository/full
revision in their retained attribution; their vendored device code is the
evidence, not an invented upstream path. TokenSpeed's packaged CuTe loader is a package
boundary, not the package's device implementation.

FlashKDA revision `b59532f1f464fbd536272780e30df5bf6a2ccc02` comes from
the retained [vLLM dependency manifest][flash-pin]. It resolves in
`vllm-project/FlashKDA`, not `MoonshotAI/FlashKDA`, so the exact fork's
device sources take precedence over a project homepage or package loader.

[fla-up]: https://github.com/fla-org/flash-linear-attention/blob/2e38c1fab332174d056928feaf29f8c5fd5ac550/fla/ops/kda/chunk.py
[fla-rec-up]: https://github.com/fla-org/flash-linear-attention/blob/2e38c1fab332174d056928feaf29f8c5fd5ac550/fla/ops/kda/fused_recurrent.py
[fla-v]: https://github.com/vllm-project/vllm/blob/435c96f9dbdd29258cb8e0f433c5b54a00cf6b16/vllm/third_party/flash_linear_attention/ops/kda.py
[s-chunk]: https://github.com/sgl-project/sglang/tree/5c2de3f35567ffceec6cea86ba18e075692e9101/python/sglang/kernels/ops/attention/linear/kda_blackwell
[nvidia-s]: https://github.com/sgl-project/sglang/blob/5c2de3f35567ffceec6cea86ba18e075692e9101/python/sglang/kernels/ops/attention/linear/kda_nvidia_prefill/chunk_fwd.py
[nvidia-t123]: https://github.com/NVIDIA/TensorRT-LLM/blob/a8ac7e5bccb972b35808dc973f7b4aac96cbbc13/tensorrt_llm/_torch/cute_dsl_kernels/blackwell/kimi_k3_kda/fused_k123.py
[nvidia-t1234]: https://github.com/NVIDIA/TensorRT-LLM/blob/a8ac7e5bccb972b35808dc973f7b4aac96cbbc13/tensorrt_llm/_torch/cute_dsl_kernels/blackwell/kimi_k3_kda/fused_k1234.py
[nvidia-t4]: https://github.com/NVIDIA/TensorRT-LLM/blob/a8ac7e5bccb972b35808dc973f7b4aac96cbbc13/tensorrt_llm/_torch/cute_dsl_kernels/blackwell/kimi_k3_kda/k4_persistent.py
[ptx-api]: https://github.com/sgl-project/sglang/blob/5c2de3f35567ffceec6cea86ba18e075692e9101/python/sglang/kernels/ops/attention/linear/kda_ptx_prefill/__init__.py
[ptx]: https://github.com/sgl-project/sglang/blob/5c2de3f35567ffceec6cea86ba18e075692e9101/python/sglang/kernels/jit/csrc/attention/kda_prefill.cu
[bt16]: https://github.com/flashinfer-ai/flashinfer/blob/15e83b7bb9f32d81d84017e2e715c265fb7253f5/flashinfer/kda_kernels/kda_chunked_bt16.py
[sm120]: https://github.com/flashinfer-ai/flashinfer/blob/15e83b7bb9f32d81d84017e2e715c265fb7253f5/flashinfer/kda_kernels/sm120_prefill/runtime.py
[decomp]: https://github.com/flashinfer-ai/flashinfer/blob/15e83b7bb9f32d81d84017e2e715c265fb7253f5/flashinfer/kda_kernels/sm120_prefill/decomp.py
[fused120]: https://github.com/flashinfer-ai/flashinfer/blob/15e83b7bb9f32d81d84017e2e715c265fb7253f5/flashinfer/kda_kernels/sm120_prefill/fused.py
[helion-prefill]: https://github.com/sgl-project/sglang/blob/5c2de3f35567ffceec6cea86ba18e075692e9101/python/sglang/kernels/ops/attention/helion/kda_prefill.py
[ts-cute]: https://github.com/lightseekorg/tokenspeed/blob/d0a2d1e02a1a643f58737894de109d0f40d83334/tokenspeed-kernel/python/tokenspeed_kernel/thirdparty/cutedsl_kda/__init__.py
[fla-rec-ts]: https://github.com/lightseekorg/tokenspeed/blob/d0a2d1e02a1a643f58737894de109d0f40d83334/tokenspeed-kernel/python/tokenspeed_kernel/thirdparty/triton/fla_kda_recurrent.py
[fi-rec]: https://github.com/flashinfer-ai/flashinfer/blob/15e83b7bb9f32d81d84017e2e715c265fb7253f5/flashinfer/kda_kernels/recurrent_kda.py
[fi-packed]: https://github.com/flashinfer-ai/flashinfer/blob/15e83b7bb9f32d81d84017e2e715c265fb7253f5/flashinfer/kda_kernels/packed_kda_decode_cute.py
[fi-fused]: https://github.com/flashinfer-ai/flashinfer/blob/15e83b7bb9f32d81d84017e2e715c265fb7253f5/flashinfer/kda_kernels/fused_kda_decode.py
[s-rec]: https://github.com/sgl-project/sglang/blob/5c2de3f35567ffceec6cea86ba18e075692e9101/python/sglang/kernels/ops/attention/cutedsl_kda.py
[s-packed]: https://github.com/sgl-project/sglang/blob/5c2de3f35567ffceec6cea86ba18e075692e9101/python/sglang/kernels/jit/csrc/attention/kda_packed_decode.cuh
[s-fused]: https://github.com/sgl-project/sglang/blob/5c2de3f35567ffceec6cea86ba18e075692e9101/python/sglang/kernels/jit/csrc/attention/kda_fused_decode.cuh
[trt-decode]: https://github.com/NVIDIA/TensorRT-LLM/blob/a8ac7e5bccb972b35808dc973f7b4aac96cbbc13/cpp/tensorrt_llm/kernels/kdaDecode/kdaDecode.cu
[v-decode]: https://github.com/vllm-project/vllm/blob/435c96f9dbdd29258cb8e0f433c5b54a00cf6b16/csrc/libtorch_stable/kimi_k3/fused_kda_decode_kernel.cu
[v-hip]: https://github.com/vllm-project/vllm/blob/435c96f9dbdd29258cb8e0f433c5b54a00cf6b16/csrc/libtorch_stable/kimi_k3/fused_kda_decode_kernel_rocm.cu
[v-hip-chunk]: https://github.com/vllm-project/vllm/blob/435c96f9dbdd29258cb8e0f433c5b54a00cf6b16/csrc/libtorch_stable/kimi_k3/fused_kda_chunk_kernel_rocm.cu
[helion-decode]: https://github.com/sgl-project/sglang/blob/5c2de3f35567ffceec6cea86ba18e075692e9101/python/sglang/kernels/ops/attention/helion/kda_decode.py
[trt-mtp]: https://github.com/NVIDIA/TensorRT-LLM/blob/a8ac7e5bccb972b35808dc973f7b4aac96cbbc13/tensorrt_llm/_torch/cute_dsl_kernels/blackwell/kimi_k3_kda/kda_mtp_decode.py
[s-mtp]: https://github.com/sgl-project/sglang/blob/5c2de3f35567ffceec6cea86ba18e075692e9101/python/sglang/kernels/ops/kimi_k3/kda_decode_mtp.py
[wy]: https://github.com/flashinfer-ai/flashinfer/blob/15e83b7bb9f32d81d84017e2e715c265fb7253f5/flashinfer/kda_kernels/kda_decode_wy_output_only.py
[helion-replay]: https://github.com/sgl-project/sglang/blob/5c2de3f35567ffceec6cea86ba18e075692e9101/python/sglang/kernels/ops/attention/helion/kda_replayssm.py
[flash-up]: https://github.com/vllm-project/FlashKDA/tree/b59532f1f464fbd536272780e30df5bf6a2ccc02/csrc/smxx
[flash-v]: https://github.com/vllm-project/vllm/blob/435c96f9dbdd29258cb8e0f433c5b54a00cf6b16/vllm/models/kimi_k3/nvidia/kda.py
[flash-pin]: https://github.com/vllm-project/vllm/blob/435c96f9dbdd29258cb8e0f433c5b54a00cf6b16/cmake/external_projects/flashkda.cmake
[flash-source]: https://github.com/flashinfer-ai/flashinfer/tree/15e83b7bb9f32d81d84017e2e715c265fb7253f5/csrc/kda
[flash-gen]: https://github.com/flashinfer-ai/flashinfer/blob/15e83b7bb9f32d81d84017e2e715c265fb7253f5/flashinfer/jit/flash_kda.py
[flash-decode]: https://github.com/flashinfer-ai/flashinfer/blob/15e83b7bb9f32d81d84017e2e715c265fb7253f5/flashinfer/jit/flash_kda_decode.py
[cake]: https://github.com/flashinfer-ai/flashinfer/blob/15e83b7bb9f32d81d84017e2e715c265fb7253f5/flashinfer/jit/cake_kda.py
[cake-decode]: https://github.com/flashinfer-ai/flashinfer/blob/15e83b7bb9f32d81d84017e2e715c265fb7253f5/flashinfer/jit/cake_kda_decode.py
[cake-packed]: https://github.com/flashinfer-ai/flashinfer/blob/15e83b7bb9f32d81d84017e2e715c265fb7253f5/flashinfer/jit/cake_kda_packed_t1.py
[cake-fused]: https://github.com/flashinfer-ai/flashinfer/blob/15e83b7bb9f32d81d84017e2e715c265fb7253f5/flashinfer/jit/cake_fused_kda_decode.py
[cudnn]: https://github.com/flashinfer-ai/flashinfer/blob/15e83b7bb9f32d81d84017e2e715c265fb7253f5/flashinfer/cudnn/linear_attention.py
[ts-prefill950]: https://github.com/lightseekorg/tokenspeed/blob/d0a2d1e02a1a643f58737894de109d0f40d83334/tokenspeed-kernel-amd/python/tokenspeed_kernel_amd/ops/gfx950/attention/kda/prefill.py
[ts-decode950]: https://github.com/lightseekorg/tokenspeed/blob/d0a2d1e02a1a643f58737894de109d0f40d83334/tokenspeed-kernel-amd/python/tokenspeed_kernel_amd/ops/gfx950/attention/kda/decode.py
[ts-prefill1250]: https://github.com/lightseekorg/tokenspeed/blob/d0a2d1e02a1a643f58737894de109d0f40d83334/tokenspeed-kernel-amd/python/tokenspeed_kernel_amd/ops/gfx1250/attention/kda/prefill.py
[ts-decode1250]: https://github.com/lightseekorg/tokenspeed/blob/d0a2d1e02a1a643f58737894de109d0f40d83334/tokenspeed-kernel-amd/python/tokenspeed_kernel_amd/ops/gfx1250/attention/kda/decode.py
[aiter-device]: https://github.com/sgl-project/sglang/blob/5c2de3f35567ffceec6cea86ba18e075692e9101/python/sglang/kernels/ops/kimi_k3/flydsl/kernels/kimi_k3_kda_decode_fb.py
[aiter-loader]: https://github.com/sgl-project/sglang/blob/5c2de3f35567ffceec6cea86ba18e075692e9101/python/sglang/kernels/ops/kimi_k3/flydsl/source.py
