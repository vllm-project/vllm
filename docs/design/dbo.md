# Dual Batch Overlap

## Motivation

The core motivation of the DBO system in vLLM is to overlap the sparse all-to-all communication in the MoE layer with the surrounding computation. This targets DP+EP deployments. For TP deployments without expert parallelism, the `all-reduce-overlap` [schedule](#schedules) overlaps the TP all-reduces instead.

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

To enable the DBO system pass in the `--enable-dbo` argument to your vllm serve command. To overlap the all-to-all, this must be run in conjunction with `--data-parallel-size N` where N is greater than 1 and `--enable-expert-parallel`. With tensor parallelism on one DP rank and no expert parallelism, `--enable-dbo` runs the `all-reduce-overlap` [schedule](#schedules) instead. Additionally, there are two configuration knobs.

* `--dbo-decode-token-threshold` the minimum number of tokens in a decode-only batch required to enable DBO for that batch
* `--dbo-prefill-token-threshold` the minimum number of tokens in a batch containing at least one prefill required to enable DBO for that batch

Currently, DBO is only supported with DeepEP, so DeepEP must be installed and the `--all2all-backend` argument must be set to `deepep_low_latency` if your workload is primarily decode requests, or `deepep_high_throughput` if your workload is primarily prefill requests.

Below is a command that will spin up a two DP rank server with expert parallelism and DBO enabled.
EX: `vllm serve deepseek-ai/DeepSeek-V2-Lite --trust-remote-code --data-parallel-size 2 --enable-expert-parallel --enable-dbo --all2all-backend deepep_low_latency`

Note that there must be at least two GPUs visible in `CUDA_VISIBLE_DEVICES`

## Limitations

DBO cannot be combined with LoRA, pipeline parallelism, prefill context parallelism, multimodal, encoder-decoder or hybrid models, or encoder-only models. Multimodal models are supported under `all-reduce-overlap`. CUDA graphs must be Full or disabled: the runner rejects `PIECEWISE`.

## DBO Components

* GPUModelRunner
* UBatchRunner
* UBatchContext

### GPU Model Runner

Before each step, the DP ranks agree on whether to microbatch, in `sync_cudagraph_and_dp_padding`. Microbatching must be uniform across all DP ranks. It only happens if every rank allows it and the rank with the fewest tokens meets the applicable threshold. Every rank then pads to the token count of the largest rank, since the expert all-to-all requires all ranks to split and pad equally. The result is a `BatchExecutionDescriptor` with `num_ubatches` set to 2, which selects a captured Full CUDA graph if one matches, and otherwise runs the step eagerly.

For a step with `num_ubatches > 1`, the runner calls `UBatchRunner.prepare` to split the batch and then `UBatchRunner.run` to execute it, instead of calling the model directly.

### UBatchRunner

`UBatchRunner`, in `vllm/v1/worker/gpu/ubatch_utils.py`, is built by `maybe_build_ubatch_runner` when DBO is enabled and there is more than one DP rank, or the [schedule](#schedules) is `all-reduce-overlap`.

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

## Schedules

`--dbo-schedule` selects what the two microbatches overlap. Left unset, it is `all-reduce-overlap` with tensor parallelism on one DP rank and no expert parallelism, and `dispatch-combine-overlap` otherwise. Both run on the same runtime (`UBatchContext`, the token thresholds); they differ in where the microbatches hand off.

* `dispatch-combine-overlap` is the schedule above: the hand-offs are the all-to-all yield points inside the `FusedMoEModularKernel`, and one microbatch's dispatch / combine overlaps the other's compute.
* `all-reduce-overlap` makes each TP all-reduce a hand-off, in `GroupCoordinator._all_reduce_out_place`: the all-reduce runs on the comm stream while the other microbatch computes. It needs tensor parallelism on one DP rank without expert parallelism.
* `attention-feed-forward-overlap` (experimental) splits the SMs into two CUDA green-context pools and overlaps one microbatch's attention, on the attention pool, with the rest of the other's layer, on the feed-forward pool. Long-context decode attention is bound by HBM bandwidth and the rest of the layer by tensor-core throughput or launch latency, so the two overlap when the attention kernel can saturate HBM from a partial pool. It is never the default.

Under `all-reduce-overlap` there is no DP handshake: the runner decides per step by the thresholds alone, and a split step runs eagerly. No all-to-all backend is needed, and multimodal models are supported, since the vision encoder runs outside the split.

Under `attention-feed-forward-overlap` the microbatches' compute stream is the attention pool's stream (`vllm/v1/worker/gpu/green_context_pools.py`) and each microbatch's comm stream is its own stream in the feed-forward pool. A microbatch starts on the attention pool's stream (the stream FULL CUDA graphs are captured and replayed on), moves to its feed-forward stream on entry and back on exit. Models opt in by bracketing the attention kernel with the existing `dbo_yield_and_switch_from_comm_to_compute("attention-feed-forward-overlap")` and `dbo_yield_and_switch_from_compute_to_comm("attention-feed-forward-overlap")`; given a schedule, these do nothing unless that schedule is running. gpt-oss brackets only its full-attention layers.

* `--dbo-attention-sms` (default 64) sizes the attention pool; the rest of the layer gets the remaining SMs. One of the two pools must be a multiple of 8 SMs.

The schedule pays off only with FULL CUDA graphs, and capturing green-context streams requires CUDA 13.4 or later. It has no all-to-all of its own, so it also runs with `--data-parallel-size 1`, where the runner decides per step by the thresholds alone and takes only a captured microbatched graph, and without expert parallelism no all-to-all backend is needed. With expert parallelism the all-to-all backend must be `deepep_low_latency` or `nixl_ep`: they hand the receive to the peer microbatch through `dbo_register_recv_hook`, which runs on the registering microbatch's stream, while the high-throughput and naive prepare/finalize switch between the compute and comm streams around their collectives, which does not fit a MoE that already runs on the feed-forward pool's stream. The gain depends on the decode attention kernel saturating HBM from the attention pool's SMs; with a kernel that needs the whole GPU for that, the schedule is slower than running the batch whole.
