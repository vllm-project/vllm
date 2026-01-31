# Dual Batch Overlap

## Motivation

The core motivation of the DBO system in vLLM is to overlap the sparse all-to-all communication in the MoE layer with the surrounding computation. This system currently only targets DP+EP deployments.

## Introduction

The Dual Batch Overlap system works by splitting the batch in the model runner, creating two worker threads, and then running the model on each of these worker threads. When DBO is enabled, yield points within the `FusedMoEKernelModular` allow the two CPU worker threads (also called UBatch threads) to ping-pong between each other so that when one is running compute, the other is waiting on communication. Throughout the code, ubatch may be used as a short form of microbatch; this is an ASCII-friendly version of the short form µ-batch.

The DBO system includes modifications to `GpuModelRunner` and `ModularKernel`, and defines two utility classes: `UBatchWrapper` and `UBatchContext`. `UBatchWrapper` manages thread lifecycle and CUDA graph execution of the model. `UBatchContext` wraps `ForwardContext` to coordinate synchronization between the two UBatch threads.

Below is the overlap schedule that is currently implemented in vLLM.

```python
# Schedule notation legend:
#    S = Shared expert
#    A0 = MLA qkv proj,
#    A1 = Core attn + out proj + MoE gate
#    D = Dispatch
#    C = Combine

# Comp: |-A0₀-A1₀-||-MLP₁-||-S₁-MLP₀-||-S₀-A0₁-A1₁-|
# Comm: |----D₁---||--D₀--||----C₁---||-----C₀-----|
# Order: D₁ send, A0₀, A1₀, D₁ recv, D₀ send, MLP₁, D₀ recv,
#        C₁ send, S₁, MLP₀, C₁ recv, C₀ send, S₀, A0₁, A1₁, C₀ recv.
# MLP_SHARED_OVERLAP = "mlp_shared_overlap"
```

## Running with DBO

To enable the DBO system pass in the `--enable-dbo` argument to your vllm serve command. This must be run in conjunction with `--data-parallel-size N` where N is greater than 1 and `--enable-expert-parallel`. Additionally, there are two configuration knobs.

* `--dbo-decode-token-threshold` the minimum number of tokens in a decode-only batch required to enable DBO for that batch
* `--dbo-prefill-token-threshold` the minimum number of tokens in a batch containing at least one prefill required to enable DBO for that batch

Currently, DBO is only supported with DeepEP, so DeepEP must be installed and the `--all2all-backend` argument must be set to `deepep_low_latency` if your workload is primarily decode requests, or `deepep_high_throughput` if your workload is primarily prefill requests.

Below is a command that will spin up a two DP rank server with expert parallelism and DBO enabled.
EX: `vllm serve deepseek-ai/DeepSeek-V2-Lite --trust-remote-code --data-parallel-size 2 --enable-expert-parallel --enable-dbo --all2all-backend deepep_low_latency`

Note that there must be at least two GPUs visible in `CUDA_VISIBLE_DEVICES`

## DBO Components

* GPUModelRunner
* UBatchWrapper
* UBatchContext

### GPU Model Runner

The batch is split into microbatches by the `GPUModelRunner` class. This is accomplished in two steps. First, coordination across all DP ranks is performed to determine whether microbatching will be applied. Microbatching must be uniform across all DP ranks. If microbatching is not feasible for any DP rank, it is disabled for all ranks. If all DP ranks are going to microbatch, the total number of tokens is padded up to the max number of tokens amongst all ranks. If any rank would end up with an empty second microbatch after the padding is applied, microbatching will be aborted and no ranks will microbatch. Once microbatching has been initiated by all ranks, the second step is performed. The `CommonAttentionMetadata` is sliced in half by the `GPUModelRunner` so that there is one attention metadata per-microbatch.

### UBatchWrapper

gpu_ubatch_wrapper

The `UBatchWrapper` class is a model wrapper that's responsible for all of the thread, UBatchContext, and CUDA graph management for DBO. It's designed to be relatively transparent to the GPU Model Runner.

The implementation runs the model twice, once for each microbatch. Each model invocation occurs within a UBatch thread. These threads are launched in parallel and are synchronized using the `UBatchContext`. Each thread is provided with a sliced version of the attention metadata that is used to run its half of the batch.

CUDA graphs for DBO are entirely managed by the `UBatchWrapper`. Because of this, DBO only supports running with Full CUDA graphs. However, once a DBO CUDA graph has been captured, it can be replayed without any multithreading or CPU synchronization.

#### Interfaces

The `__init__` method takes in the model, VllmConfig, CUDAGraphMode, and device.

The `forward` method exclusively takes in model arguments. It determines whether or not to run with DBO based on whether a `ubatch_slices` object is present in the `forward_context`. Otherwise, the model is run without DBO.

### UBatchContext

ubatch_context

The `UBatchContext` class is a `ForwardContext` wrapper class that is used by the `UBatchWrapper` class to synchronize the two UBatch threads. It should only be instantiated by using `make_ubatch_contexts`.

When one of the UBatch threads reaches a `dbo_yield` call, it pauses, and starts the other thread which will run until it reaches the same `dbo_yield` call. This "ping-pong" dynamic continues, with threads swapping at each `dbo_yield call`, until the model's execution is complete.

The current implementation has all `dbo_yield` and `dbo_maybe_run_recv_hook` calls in the `FusedMoEKernelModular.forward` method.

#### Interfaces

The `make_ubatch_context` function initializes two `UBatchContexts`, one for each UBatch thread. It takes two CUDA streams, the preexisting `ForwardContexts` and a CPU thread barrier. This function should be used exclusively to instantiate `UBatchContexts`. It will handle all of the event initialization.

The `dbo_register_recv_hook` method registers a callback that can be returned by the `FusedMoEPrepareAndFinalizeModular` class in the other UBatch thread’s `UBatchContext`. The callback will be run when the other thread calls `dbo_maybe_run_recv_hook`. This is typically used to wait on an all-to-all kernel.

The `dbo_maybe_run_recv_hook` method runs a callback that’s set by the `dbo_register_recv_hook` function if that callback exists.

The `dbo_yield` method puts the current thread to sleep and wakes up the other UBatch thread.
