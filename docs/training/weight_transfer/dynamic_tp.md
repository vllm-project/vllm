# Dynamic TP with IPC refit

The experimental `LLMEngine` layout-transition API switches a fixed two-GPU pool between DP2×TP1 and DP1×TP2 at a drained rollout/update boundary. It retains EngineCore, executor, worker processes, and the bootstrap world. TP/DP groups, model runner, IPC receiver, KV cache, and scheduler are rebuilt for the target layout. Generation resumes after every rank has loaded the complete checkpoint and initialized execution state.

## Initial scope

Use one node with two physical ranks, `external_launcher`, an in-process engine, and the eager V2 model runner. The initial implementation accepts exact dense BF16 `LlamaForCausalLM`, PP/CP=1, and unpacked native IPC. Quantization, MoE, LoRA, speculation, sleep/offload, CuMem allocation, online serving, concurrent API calls, and rank-loss recovery are unsupported.

All ranks must call the same operations in the same order with identical transition IDs, target TP sizes, and weight versions. DP2 ranks may generate different inputs. With TP2, both ranks must submit identical prompts, sampling parameters, and request order.

## Calling sequence

Launch both ranks with `torchrun --standalone --nproc-per-node=2`, `VLLM_ENABLE_V1_MULTIPROCESSING=0`, and `VLLM_USE_V2_MODEL_RUNNER=1`. Construct `LLM` with `tensor_parallel_size=1`, `data_parallel_size=2`, `distributed_executor_backend="external_launcher"`, `enforce_eager=True`, `async_scheduling=False`, and `WeightTransferConfig(backend="ipc")`. Initialize the receiver once on each rank:

```python
llm.init_weight_transfer_engine({"init_info": {"packed": False}})
```

Finish synchronous generation on every rank and call one additional `llm.llm_engine.step()` on each rank to consume the final scheduler completion cleanup. Having zero unfinished frontend requests alone does not establish this boundary. Ensure that no ordinary weight update is active and serialize the following sequence against generation and other control operations:

```python
from vllm.v1.engine.layout_transition import LayoutTransitionRequest

engine = llm.llm_engine
engine.step()  # After synchronous generation, on every rank.
request = LayoutTransitionRequest("rollout-12", 2, "checkpoint-12")
engine.prepare_layout_transition(request)
# Cancellation is possible here via engine.cancel_layout_transition(request).
engine.install_layout_transition(request)
for payload in rank_local_checkpoint_chunks:
    engine.update_layout_weights(request, payload)
engine.finish_layout_transition(request)
```

For the reverse transition, use a new transition ID and checkpoint version with target TP=1. Installation already starts the native refit, and finishing the transition completes it. Do not interleave ordinary `start_weight_update` or `finish_weight_update` calls with this sequence.

## Checkpoint ownership

Each payload follows the unpacked [IPC engine](ipc.md) schema: `names`, `dtype_names`, `shapes`, and `ipc_handles`. Handles must refer to live tensors on the receiving rank's physical GPU. Both ranks provide the same ordered names, full HF shapes, and BF16 dtypes, while their handles may differ. Send complete HF tensors, including separate Q/K/V and gate/up inputs; the new model loader selects the target TP shards.

The full checkpoint is required before completion. Missing, duplicate, unexpected, or incompatible parameters are rejected. Tied embedding/output weights use one canonical entry. Keep all publisher tensors alive until `finish_layout_transition` succeeds, including inputs whose fused layer spans several chunks. If a transition fails after teardown, shut down the receiver before releasing outstanding exports.

A successful transition creates an empty KV cache. Ordinary weight updates between transitions still require the caller to invalidate prefix caches before generating with the new weights.

## Rejection and failure

`LayoutTransitionRejected` during preparation leaves the original layout usable. Successful preparation reserves admission; cancellation releases it while the engine is still prepared. Installation preflight rejection also permits cancellation because teardown has not begun.

During refit, metadata rejection happens collectively before receive and permits a corrected chunk to be retried. An incomplete checkpoint at finish permits sending the remaining weights. Admission remains reserved throughout these retries.

After installation starts, cancellation and rollback are unavailable. A runtime failure keeps the engine reserved; shut it down and recreate it. The protocol assumes all physical ranks remain alive and participate, so it does not recover from a missing rank or a failed collective.
