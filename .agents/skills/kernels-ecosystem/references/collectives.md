# AllReduce, AllToAll and fused collectives

[Primitive index](../SKILL.md)

Entries identify transport/compute implementations, not framework communicator
classes. FlashInfer or framework exposure of a TRTLLM/vLLM collective is
folded into that origin's row. Symmetric workspace, peer access, rank layout
and synchronization contracts are part of the implementation.

## AllReduce

| Implementation / entry | Language | Origin and source | Algorithm / constraints |
| --- | --- | --- | --- |
| Custom peer-memory AR: `cross_device_reduce_1stage`, `cross_device_reduce_2stage` | CUDA/HIP | vLLM; [native kernels][vllm-native] | Registered peer buffers; topology/message-size gates. FlashInfer and SGLang copies are not separate families. |
| Custom AR: `customAllReduce` | CUDA | TRTLLM; [source][trt-ar] | Strategy/configuration, rank/size/alignment and IPC buffers; also exposed through FI. |
| Low-precision AR: `customLowPrecisionAllReduce` | CUDA | TRTLLM; [source][low-ar] | Compressed reduction and separate workspace; not the full-precision algorithm. |
| AR residual/norm/quant: `allreduce_fusion_op` | CUDA | TRTLLM; [source][trt-fusion] | Fused epilogues selected by pattern; FI's TRTLLM fusion is an exposure of this family. |
| MoE finalize AR: `moe_allreduce`, `moe_finalize_allreduce` | CUDA | TRTLLM; [source][moe-ar] | Routed expert weighting/finalization plus TP reduction. |
| MNNVL one-shot: `oneshotAllreduceFusionOp` | CUDA | TRTLLM; [source][mnnvl-cuda] | One-shot NVLink-fabric reduction/fusion; shared lineage with FI's TRTLLM MNNVL exposure. |
| MNNVL two-shot: `twoshotAllreduceFusionOp` | CUDA | TRTLLM; [source][mnnvl-cuda] | Separate two-stage algorithm and symmetric fusion workspace. |
| Userbuffers: `userbuffers_allreduce_finalize` | CUDA/C++ | TRTLLM; [source][userbuffers] | Registered symmetric userbuffers; fused residual/norm/finalize. |
| MNNVL LL | CuTe DSL | FlashInfer; [kernel][ll] | `MNNVLCuteDSLAllReduceFusionWorkspace`; low-latency BF16 protocol with split staging. |
| MNNVL HT | CuTe DSL | FlashInfer; [kernel][ht] | Same workspace family, distinct persistent BF16 SM100 protocol. |
| MNNVL BT | CuTe DSL | FlashInfer; [kernel][bt] | Distinct three-stage BF16 SM100 protocol. |
| PCIe IPC AR: `PcieIpcAllReduceWorkspace` | CUDA | FlashInfer; [device kernels][pcie] | PCIe peer-buffer topology/policy/tuning; not NVLink MNNVL. |
| Quantized two-shot: `quantized_all_reduce` | Triton + PTX | FlashInfer; [kernel][qar] | BF16 -> grouped FP8 + FP32 scales -> BF16; PyTorch symmetric memory, default scale group 256. |
| V2 push: `custom_all_reduce`, `ONE_SHOT_PUSH` | CUDA | SGLang; [device source][sgl-v2] | Lamport push-plane workspace/counters; distinct from its vendored vLLM AR. |
| V2 pull: `ONE_SHOT_PULL`, `TWO_SHOT_PULL` | CUDA | SGLang; [device source][sgl-v2] | One-/two-stage pull algorithms with peer buffers and runtime-selected metadata. |
| QuickReduce: `QuickAllReduce` | HIP | QuickReduce implementation retained in vLLM; [source][quick] | AMD-specific collective; SGLang's QuickReduce exposure is not another algorithm. |
| Deterministic AR | HIP | SGLang; [native source directory][sgl-ar] | `deterministic_all_reduce.hip`; reproducible reduction separate from custom AR. |
| Iris producer-direct and two-stage AR | Gluon | TokenSpeed/Iris integration; [kernels][iris] | `iris_reduce_symmetric_gluon_kernel`, `iris_reduce_symmetric_two_stage_gluon_kernel`; AMD symmetric buffers, distinct scratch protocols. |

General library collectives (NCCL/RCCL, PyTorch symmetric-memory multimem
and two-shot, NVSHMEM, MSCCL++) are not repeated as each framework's import
wrapper. The table focuses on concrete alternative implementations in the
reviewed sources.

## Expert AllToAll

These exchange routed expert tokens and combine results. They are not
interchangeable with attention-state AllToAll below.

| Implementation / entry | Language | Origin and source | Algorithm / constraints |
| --- | --- | --- | --- |
| NVLink one-sided: `moe_a2a_dispatch`, `moe_a2a_combine` | CUDA | TRTLLM; [native source][one-sided], [FI generated manifest][comm-jit] | GPU-initiated exchange with rank buffers/expert metadata; FI legacy and SM100a/103a fused MNNVL variants belong to this lineage. |
| NVLink two-sided: `MoeAlltoAll` / `MnnvlMoe` | CUDA | TRTLLM; [native fused comm][two-sided], [FI implementation entry][fi-two] | Prepare/communicate/local-gather stages; distinct protocol/workspaces from one-sided. |
| DeepEP normal: `Buffer.dispatch`, `Buffer.combine` | CUDA | DeepEP; [upstream kernels][deep-ep] | Throughput-oriented dispatch/combine, events and routing layouts. |
| DeepEP low latency: `low_latency_dispatch`, `low_latency_combine` | CUDA | DeepEP; [upstream LL kernels][deep-ll] | Expert-major buffers, bounded tokens/rank and separate event/handle protocol. |
| DeepEP V2: `ElasticBuffer` | CUDA | DeepEP V2; [upstream elastic kernels][deep-v2] | Elastic transport/API and FP8 scale formats, not an alias of V1 `Buffer`. |
| NCCL-EP split dispatch/combine | CUDA library | NCCL-EP; [upstream device source][nccl-ep] | Fleet/handle lifecycle separates communication and compute; not ordinary equal-sized NCCL AllToAll. |
| NIXL-EP split dispatch/combine | CUDA library | NIXL-EP; [upstream kernels][nixl-ep] | Workspace/bootstrap/handle protocol specific to EP. |
| Mooncake EP: `Buffer` | CUDA transport | Mooncake EP; [upstream kernel][moon] | `mooncake_ep_buffer` dispatch/combine; one provider, not an SGLang kernel. |
| MoRI normal / low latency | HIP | MoRI; [upstream kernels][mori] | Distinct normal and LL dispatch/combine configurations and buffers. |
| PPLX: `AllToAll.intranode`, `AllToAll.internode` | CUDA/NVSHMEM | PPLX; [upstream kernels (moving `master`)][pplx] | Intra-node and NVSHMEM inter-node setups; checkout integration does not pin a provider revision. |

Provider rows link directly to their kernel source, using revisions found in
the current checkouts where available. This citation update is not a new
transport compatibility or performance review.
Framework dispatcher classes, AllGather/ReduceScatter emulation and
platform-only import wrappers are omitted.

Compute-integrated push/pull transport (FlashInfer's EP mega kernels,
TRTLLM MegaMoE, DeepGEMM MegaMoE) is listed with the
[fused expert implementations](moe.md), not counted again here.

## Attention-state exchange

| Implementation / entry | Language | Origin and source | Algorithm / constraints |
| --- | --- | --- | --- |
| Helix: `alltoall_helix_native` / `decode_cp_a2a_alltoall` | CUDA | TRTLLM Helix; [native source][helix], [FI entry][dcp] | DCP partial attention output/statistics exchange. Native transport differs from NCCL send/recv composition. |
| Ulysses IPC: `ulysses_a2a` | CUDA | FlashInfer; [device kernels][ulysses] | Attention head/sequence exchange with IPC topology; packing and synchronization helpers are not separate transports. |
| Cake speculative DCP: `run_dcp_spec_decode` | CUDA, generated | FlashInfer Cake; [entry][cake-dcp] | Fused speculative decode and DCP workspace/counter protocol. |

## AllGather, ReduceScatter and GEMM fusions

| Implementation / entry | Language | Origin and source | Algorithm / constraints |
| --- | --- | --- | --- |
| Custom AG/RS: `allgather`, `reduce_scatter`, `mnnvl_lamport_allgather`, `mnnvl_lamport_reduce_scatter` | CUDA | vLLM; [native implementations][vllm-agrs] | Peer-buffer collectives and MNNVL Lamport variants; multicast AllGather is not an additional AllReduce algorithm. |
| Persistent two-shot GEMM-AR: `PersistentDenseGemmKernel` | CuTe DSL | FlashInfer; [kernel][fi-gemm-ar] | Multimem barriers and two-shot reduction integrated with GEMM. |
| CUTLASS GEMM-AR / `Fp4GemmAllreduceRunner` | C++ CUTLASS/CUDA | TRTLLM; [source][trt-gemm-ar] | SM90/SM100 families including NVFP4; userbuffer/group contract, not GEMM followed by NCCL. |
| Wait-signal AG-MM: `all_gather_matmul_triton` | Triton | FlashInfer; [kernel][ag-triton] | Overlaps matrix multiply with gathered blocks using signals. |
| Wait-signal AG-MM: `all_gather_matmul_cutile` | cuTile | FlashInfer; [kernel][ag-tile] | Separate cuTile matmul/barrier implementation. |
| Cake AG-MM: `all_gather_matmul_cake` | CUDA, generated | FlashInfer Cake; [entry][ag-cake] | SM100a/103a prepared descriptors, including packed QKV. |
| Kimi GEMM-RS/AR: `Sm100GemmRsArBF16` | CuTe DSL | Kimi implementation in vLLM; [source][kimi] | SM100 BF16 compute plus reduce-scatter/all-reduce. |
| Latent-tail AR/norm/RS: `AllReduceRMSNormWithReduceScatterEarlyExit` | CuTe DSL | Kimi latent-MoE family in vLLM; [source][kimi] | Fused latent-tail collective; `CollectiveKernel`, early-exit and sentinel protocol, not generic AR. |
| Residual multicast projection: `FusedAddMulticastGemm`, `AdaptiveUpProjectionKernel` | CuTe DSL | Kimi latent-MoE family in vLLM; [source][kimi] | Residual-add plus multicast/up projection; `FusedAddMulticastSkinnyGemm` is its distinct skinny implementation. |

[vllm-native]: https://github.com/vllm-project/vllm/blob/435c96f9dbdd29258cb8e0f433c5b54a00cf6b16/csrc/custom_all_reduce.cuh
[vllm-agrs]: https://github.com/vllm-project/vllm/blob/435c96f9dbdd29258cb8e0f433c5b54a00cf6b16/csrc/libtorch_stable/custom_all_gather_reduce_scatter.cu
[trt-ar]: https://github.com/NVIDIA/TensorRT-LLM/blob/a8ac7e5bccb972b35808dc973f7b4aac96cbbc13/cpp/tensorrt_llm/kernels/customAllReduceKernels.cu
[low-ar]: https://github.com/NVIDIA/TensorRT-LLM/blob/a8ac7e5bccb972b35808dc973f7b4aac96cbbc13/cpp/tensorrt_llm/kernels/communicationKernels/customLowPrecisionAllReduceKernels.cu
[trt-fusion]: https://github.com/NVIDIA/TensorRT-LLM/blob/a8ac7e5bccb972b35808dc973f7b4aac96cbbc13/cpp/tensorrt_llm/kernels/communicationKernels/allReduceFusionKernels.cu
[moe-ar]: https://github.com/NVIDIA/TensorRT-LLM/blob/a8ac7e5bccb972b35808dc973f7b4aac96cbbc13/cpp/tensorrt_llm/kernels/communicationKernels/moeAllReduceFusionKernels.cu
[mnnvl-cuda]: https://github.com/NVIDIA/TensorRT-LLM/blob/a8ac7e5bccb972b35808dc973f7b4aac96cbbc13/cpp/tensorrt_llm/kernels/communicationKernels/mnnvlAllreduceKernels.cu
[userbuffers]: https://github.com/NVIDIA/TensorRT-LLM/tree/a8ac7e5bccb972b35808dc973f7b4aac96cbbc13/cpp/tensorrt_llm/kernels/userbuffers
[ll]: https://github.com/flashinfer-ai/flashinfer/blob/15e83b7bb9f32d81d84017e2e715c265fb7253f5/flashinfer/comm/mnnvl_cutedsl/kernel_ll/device_kernels.py
[ht]: https://github.com/flashinfer-ai/flashinfer/blob/15e83b7bb9f32d81d84017e2e715c265fb7253f5/flashinfer/comm/mnnvl_cutedsl/kernel_ht/device_kernel.py
[bt]: https://github.com/flashinfer-ai/flashinfer/blob/15e83b7bb9f32d81d84017e2e715c265fb7253f5/flashinfer/comm/mnnvl_cutedsl/kernel_bt/device_kernels.py
[pcie]: https://github.com/flashinfer-ai/flashinfer/blob/15e83b7bb9f32d81d84017e2e715c265fb7253f5/include/flashinfer/comm/pcie_ipc_all_reduce.cuh
[qar]: https://github.com/flashinfer-ai/flashinfer/blob/15e83b7bb9f32d81d84017e2e715c265fb7253f5/flashinfer/comm/quantized_allreduce.py
[sgl-v2]: https://github.com/sgl-project/sglang/blob/5c2de3f35567ffceec6cea86ba18e075692e9101/python/sglang/kernels/jit/csrc/distributed/custom_all_reduce.cuh
[quick]: https://github.com/vllm-project/vllm/blob/435c96f9dbdd29258cb8e0f433c5b54a00cf6b16/csrc/quickreduce/quick_reduce_impl.cuh
[sgl-ar]: https://github.com/sgl-project/sglang/tree/5c2de3f35567ffceec6cea86ba18e075692e9101/python/sglang/kernels/aot/csrc/allreduce
[iris]: https://github.com/lightseekorg/tokenspeed/blob/d0a2d1e02a1a643f58737894de109d0f40d83334/tokenspeed-kernel/python/tokenspeed_kernel/ops/communication/iris.py
[one-sided]: https://github.com/NVIDIA/TensorRT-LLM/blob/a8ac7e5bccb972b35808dc973f7b4aac96cbbc13/cpp/tensorrt_llm/kernels/communicationKernels/moeAlltoAllKernels.cu
[comm-jit]: https://github.com/flashinfer-ai/flashinfer/blob/15e83b7bb9f32d81d84017e2e715c265fb7253f5/flashinfer/jit/comm.py
[two-sided]: https://github.com/NVIDIA/TensorRT-LLM/blob/a8ac7e5bccb972b35808dc973f7b4aac96cbbc13/cpp/tensorrt_llm/kernels/fusedMoeCommKernels.cu
[fi-two]: https://github.com/flashinfer-ai/flashinfer/blob/15e83b7bb9f32d81d84017e2e715c265fb7253f5/flashinfer/comm/trtllm_alltoall.py
[deep-ep]: https://github.com/deepseek-ai/DeepEP/tree/d4f41e4e93602a15e95f55f6ee8df8f1aaa0e4bb/csrc/kernels/legacy
[deep-ll]: https://github.com/deepseek-ai/DeepEP/blob/d4f41e4e93602a15e95f55f6ee8df8f1aaa0e4bb/csrc/kernels/legacy/internode_ll.cu
[deep-v2]: https://github.com/deepseek-ai/DeepEP/tree/d4f41e4e93602a15e95f55f6ee8df8f1aaa0e4bb/csrc/kernels/elastic
[nccl-ep]: https://github.com/NVIDIA/nccl-extensions/tree/e57f0dad43dc1ca5bf96f09bf4075afc2eae6599/nccl_ep/device
[nixl-ep]: https://github.com/ai-dynamo/nixl/tree/c28061f9782e099f975bcc79198b7b5a1a36cc40/examples/device/ep/csrc/kernels
[moon]: https://github.com/kvcache-ai/Mooncake/blob/01d1eb2a7ec37fd5e20a88573e9b4956e7846e9a/mooncake-ep/src/mooncake_ep_kernel.cu
[mori]: https://github.com/ROCm/mori/tree/96ffa169710f214e76e07abe5008d686fe54522b/src/ops/dispatch_combine
[pplx]: https://github.com/perplexityai/pplx-kernels/tree/master/csrc/all_to_all
[helix]: https://github.com/NVIDIA/TensorRT-LLM/blob/a8ac7e5bccb972b35808dc973f7b4aac96cbbc13/cpp/tensorrt_llm/kernels/helixAllToAll.cu
[dcp]: https://github.com/flashinfer-ai/flashinfer/blob/15e83b7bb9f32d81d84017e2e715c265fb7253f5/flashinfer/comm/dcp_alltoall.py
[ulysses]: https://github.com/flashinfer-ai/flashinfer/blob/15e83b7bb9f32d81d84017e2e715c265fb7253f5/include/flashinfer/comm/ulysses_all_to_all.cuh
[cake-dcp]: https://github.com/flashinfer-ai/flashinfer/blob/15e83b7bb9f32d81d84017e2e715c265fb7253f5/flashinfer/cake_dcp.py
[fi-gemm-ar]: https://github.com/flashinfer-ai/flashinfer/blob/15e83b7bb9f32d81d84017e2e715c265fb7253f5/flashinfer/cute_dsl/gemm_allreduce_two_shot.py
[trt-gemm-ar]: https://github.com/NVIDIA/TensorRT-LLM/tree/a8ac7e5bccb972b35808dc973f7b4aac96cbbc13/cpp/tensorrt_llm/kernels/cutlass_kernels/allreduce_gemm
[ag-triton]: https://github.com/flashinfer-ai/flashinfer/blob/15e83b7bb9f32d81d84017e2e715c265fb7253f5/flashinfer/comm/all_gather_matmul/all_gather_matmul_triton.py
[ag-tile]: https://github.com/flashinfer-ai/flashinfer/blob/15e83b7bb9f32d81d84017e2e715c265fb7253f5/flashinfer/comm/all_gather_matmul/all_gather_matmul_cutile.py
[ag-cake]: https://github.com/flashinfer-ai/flashinfer/blob/15e83b7bb9f32d81d84017e2e715c265fb7253f5/flashinfer/comm/all_gather_matmul/cake_all_gather_matmul.py
[kimi]: https://github.com/vllm-project/vllm/tree/435c96f9dbdd29258cb8e0f433c5b54a00cf6b16/vllm/models/kimi_k3/nvidia/ops/cute_dsl
