# Dual Batch Overlap

## Motivation

The core motivation of the DBO system in vLLM is to overlap the sparse all-to-all communication in the MoE layer with the surrounding computation. This system currently only targets DP+EP deployments.

## Introduction

The Dual Batch Overlap system works by splitting the batch up in the model runner, creating two worker threads, and then having each of these worker threads run the model. When DBO is enabled, there are yield points within the FusedMoEModularKernel that will allow the two worker threads to ping-pong between each other so that when one is running compute, the other is waiting on communication. Throughout the code you may see ubatch being used as a short form of microbatch; this is just an ASCII friendly version of the short form µ-batch.

The DBO system includes modifications to `GpuModelRunner` and `ModularKernel`, and defines two utility classes: `UBatchWrapper` and `UBatchContext`. `UBatchWrapper` manages thread lifecycle and CUDA graph execution of the model. `UBatchContext` wraps `ForwardContext` to coordinate synchronization between the two UBatch threads.

Below is the overlap schedule that is currently implemented in vLLM.

```python
# Schedule notation legend:
#    S = Shared expert
#    A0 = MLA qkv pro,
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

* `--dbo-decode-token-threshold` the minimum number of tokens in a decode-only batch required to enable DBO for that batch.
* `--dbo-prefill-token-threshold` the minimum number of tokens in a batch containing at least one prefill required to enable DBO for that batch

Currently DBO is only supported with DeepEP so you’ll have to install that and set the `VLLM_ALL2ALL_BACKEND` environment variable to `deepep_low_latency` if your workload is primarily decode requests and `deepep_high_throughput` if your workload is primarily prefill requests.

Below is a command that will spin up a two DP rank server with expert parallelism and DBO enabled.
EX: `VLLM_ALL2ALL_BACKEND=deepep_low_latency vllm serve --model="deepseek-ai/DeepSeek-V2-Lite" --trust-remote-code --data-parallel-size 2 --enable-expert-parallel --enable-dbo`

Note that you'll need to have DeepEp installed and at least two GPUs visible in `CUDA_VISIBLE_DEVICES`

## DBO Components

* GPUModelRunner
* UBatchWrapper
* UBatchContext

### GPU Model Runner

The `GpuModelRunner` is responsible for splitting up the batch into microbatches. Mechanically this requires two steps. The first is to coordinate between all of the DP ranks to decide if we are microbatching. Microbatching must be uniform between all DP ranks. If any DP rank doesn’t want to microbatch, none of them will. If all DP ranks want to microbatch, the total number of tokens is padded up to the max number of tokens amongst all ranks. If any rank would end up with an empty second microbatch after the padding is applied, microbatching will be aborted and no ranks will microbatch. Once all ranks have decided to microbatch, the second step is to slice up the `CommonAttentionMetadata` so that we have one attention metadata per-microbatch.

### UBatchWrapper

gpu_ubatch_wrapper

The `UBatchWrapper` class is a model wrapper that's responsible for all of the thread, UBatchContext, and cuda graph management for DBO. It's designed to be relatively transparent to the GPU Model Runner.

The implementation revolves around running the model twice, once for each microbatch. Each invocation of the model will happen inside of a cpu thread. These threads are launched in parallel and are synchronized using the `UBatchContext`. Each thread is given a “sliced” version of the attention metadata that they will use to run their half of the batch.

Cudagraphs for DBO are entirely managed by the `UBatchWrapper` as well. Because of this, DBO only supports running with Full Cudagraphs. However, once we’ve captured a DBO cudagraph, we can replay it without any multithreading or CPU synchronization.

#### Interfaces

`__init__` method takes in the model, VllmConfig, CUDAGraphMode, and device.

`forward` method exclusively takes in model arguments. It determines whether or not to run with DBO if there's a `ubatch_slices` object in the `forward_context`. Otherwise it just naively runs the model.

### UBatchContext

ubatch_context

The `UBatchContext` class is a `ForwardContext` wrapper class that is used by the `UBatchWrapper` class to synchronize the two UBatch threads. It should only be instantiated by using `make_ubatch_contexts`.

When one of the `UBatch` threads reaches a `dbo_yield` call, it pauses, and starts the other thread which will run until it reaches the same `dbo_yield` call. This "ping-pong" dynamic continues, with threads swapping at each `dbo_yield call`, until the model's execution is complete.

The current implementation has all `dbo_yield` and `dbo_maybe_run_recv_hook` calls in the `FusedMoEModularKernel.forward` method.

#### Interfaces

`make_ubatch_context` function initializes two `UBatchContexts`, one for each UBatch thread. It takes two cuda streams, the preexisting `ForwardContexts` and a cpu thread barrier. You should exclusively use this function to instantiate `UBatchContexts`. It will handle all of the event initialization.

`dbo_register_recv_hook` method registers a callback that can be returned by the `FusedMoEPrepareAndFinalize` class in the other UBatch thread’s `UBatchContext`. The callback will be run when the other thread calls `dbo_maybe_run_recv_hook`. This is typically used to “wait” on an all-to-all kernel

`dbo_maybe_run_recv_hook` method runs a callback that’s set by the `dbo_register_recv_hook` function if that callback exists.

`dbo_yield` method puts the current thread to sleep and wakes up the other UBatch thread
