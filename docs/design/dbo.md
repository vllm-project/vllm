# Dual Batch Overlap

## Motivation

The core motivation of the DBO system in vLLM is to overlap the sparse all-to-all communication in the MoE layer with the surrounding computation. This system currently only targets DP+EP deployments.

## Introduction

The Dual Batch Overlap system works by splitting the batch in the model runner, creating two worker threads, and then running the model on each of these worker threads. When DBO is enabled, yield points within the `FusedMoEModularKernel` allow the two CPU worker threads (also called UBatch threads) to ping-pong between each other so that when one is running compute, the other is waiting on communication. Throughout the code, ubatch may be used as a short form of microbatch; this is an ASCII-friendly version of the short form µ-batch.

The DBO system includes modifications to `GPUModelRunner` and `ModularKernel`, and defines two utility classes: `UBatchRunner` and `UBatchContext`. `UBatchRunner` splits the batch and manages the lifecycle of the UBatch threads. `UBatchContext` wraps `ForwardContext` to coordinate synchronization between the two UBatch threads.

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

DBO needs an all2all backend that supports it: `deepep_low_latency` if your workload is primarily decode requests or `deepep_high_throughput` if it is primarily prefill requests (both need DeepEP installed), or `nixl_ep`. On ROCm, `allgather_reducescatter` and `mori` (when it runs MoRI EPv2, see [MoRI EP](../serving/expert_parallel_deployment.md#mori-ep-rocm)) also support it.

Below is a command that will spin up a two DP rank server with expert parallelism and DBO enabled.
EX: `vllm serve deepseek-ai/DeepSeek-V2-Lite --trust-remote-code --data-parallel-size 2 --enable-expert-parallel --enable-dbo --all2all-backend deepep_low_latency`

Note that there must be at least two GPUs visible in `CUDA_VISIBLE_DEVICES`

## Limitations

DBO cannot be combined with LoRA, pipeline parallelism, prefill context parallelism, multimodal, encoder-decoder or hybrid models, or encoder-only models. CUDA graphs must be Full or disabled: the runner rejects `PIECEWISE`.

## DBO Components

* GPUModelRunner
* UBatchRunner
* UBatchContext

### GPU Model Runner

Before each step, the DP ranks agree on whether to microbatch, in `sync_cudagraph_and_dp_padding`. Microbatching must be uniform across all DP ranks. It only happens if every rank allows it and the rank with the fewest tokens meets the applicable threshold. Every rank then pads to the token count of the largest rank, since the expert all-to-all requires all ranks to split and pad equally. The result is a `BatchExecutionDescriptor` with `num_ubatches` set to 2, which selects a captured Full CUDA graph if one matches, and otherwise runs the step eagerly.

For a step with `num_ubatches > 1`, the runner calls `UBatchRunner.prepare` to split the batch and then `UBatchRunner.run` to execute it, instead of calling the model directly.

### UBatchRunner

`UBatchRunner`, in `vllm/v1/worker/gpu/ubatch_utils.py`, is built by `maybe_build_ubatch_runner` when DBO is enabled and there is more than one DP rank.

The batch is sliced before its attention metadata is built. `prepare` splits the batch at the midpoint of the DP-padded token count. A uniform batch of multi-token queries (speculative decoding verification) is split on a request boundary instead, so that each microbatch stays a uniform decode. `prepare` then builds the attention metadata for each microbatch and stores it in that microbatch's own `ForwardContext`. A microbatch that falls entirely inside the padding has no query tokens, and runs like a dummy batch.

`run` starts one UBatch thread per microbatch, each of which runs the model on its slice of the model inputs, then joins them and merges their outputs. The threads are synchronized using `UBatchContext`.

Capture uses the same path through `begin_capturable_run`, which parks the threads before the caller enters the capture context. `CudaGraphManager` adds a microbatched capture candidate for each eligible uniform Full graph. Once a DBO CUDA graph has been captured, it can be replayed without any multithreading or CPU synchronization.

### UBatchContext

`UBatchContext`, in `vllm/v1/worker/ubatching.py`, is a `ForwardContext` wrapper class that is used by `UBatchRunner` to synchronize the two UBatch threads. It should only be instantiated by using `make_ubatch_contexts`.

When one of the UBatch threads reaches a `dbo_yield` call, it pauses, and starts the other thread which will run until it reaches the same `dbo_yield` call. This "ping-pong" dynamic continues, with threads swapping at each `dbo_yield call`, until the model's execution is complete.

The `dbo_yield` and `dbo_maybe_run_recv_hook` calls are in the `FusedMoEModularKernel` forward path and the prepare/finalize classes it uses for all-to-all communication.

#### Interfaces

The `make_ubatch_contexts` function initializes one `UBatchContext` for each UBatch thread. It takes the compute and communication CUDA streams, the `ForwardContexts` built for each microbatch and a CPU thread barrier. This function should be used exclusively to instantiate `UBatchContexts`. It will handle all of the event initialization.

The `dbo_register_recv_hook` method registers a callback that can be returned by the `FusedMoEPrepareAndFinalizeModular` class in the other UBatch thread’s `UBatchContext`. The callback will be run when the other thread calls `dbo_maybe_run_recv_hook`. This is typically used to wait on an all-to-all kernel.

The `dbo_maybe_run_recv_hook` method runs a callback that’s set by the `dbo_register_recv_hook` function if that callback exists.

The `dbo_yield` method puts the current thread to sleep and wakes up the other UBatch thread.
