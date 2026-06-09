# MoE Layer Architecture Design Document

## Overview

The vLLM Mixture of Experts (MoE) subsystem lives under `vllm/model_executor/layers/fused_moe/`. The entry point is the `FusedMoEFactory()` factory function in `layer.py`, which assembles a pipeline of cooperating objects and returns a `MoERunner` — the `nn.Module` that models call directly in their forward pass.

## Object Relationship Diagram

```python
Model (e.g. Mixtral, DeepSeek)
  │
  │  calls FusedMoEFactory(...) factory   ──────────────────────────────────┐
  │                                                                  │
  ▼                                                                  │
MoERunner (nn.Module, is the return value)                           │
  ├── router: FusedMoERouter          ◄── created by factory         │
  ├── routed_experts: RoutedExperts   ◄── created by factory         │
  ├── _shared_experts: SharedExperts? ◄── wraps model-provided layer │
  ├── gate: nn.Module?                ◄── model-provided             │
  ├── shared_expert_gate: nn.Module?  ◄── model-provided             │
  ├── routed_input_transform: nn.Module?                             │
  ├── routed_output_transform: nn.Module?                            │
  └── moe_config: FusedMoEConfig     ◄── created by factory         │
                                                                     │
RoutedExperts (nn.Module)                                            │
  ├── quant_method: FusedMoEMethodBase  (owns expert weight params)  │
  ├── expert_map_manager: ExpertMapManager                           │
  ├── moe_config: FusedMoEConfig                                    │
  └── [w13_weight, w2_weight, scales, ...] (registered parameters)   │
                                                                     │
FusedMoERouter (ABC, not nn.Module)                                  │
  ├── eplb_state: EplbLayerState?                                    │
  └── Concrete: TopKRouter, ZeroExpertRouter, etc.                   │
                                                                     │
ExpertMapManager                                                     │
  ├── expert_map: Tensor?  (global→local mapping)                    │
  ├── routing_tables: tuple[Tensor, Tensor, Tensor]?                 │
  └── placement_strategy: "linear" | "round_robin"                   │
                                                                     │
FusedMoEConfig (dataclass)                                           │
  ├── num_experts, experts_per_token, hidden_dim, ...                │
  └── moe_parallel_config: FusedMoEParallelConfig                    │
         ├── tp_size/rank, dp_size/rank, ep_size/rank, ...           │
         └── all2all_backend, use_ep, enable_eplb                    │
```

## Component Descriptions

### 1. `FusedMoEFactory()` — Factory Function (`layer.py`)

**Role**: Top-level constructor. Models never instantiate the components directly; they call `FusedMoEFactory(...)` which:

1. Builds `FusedMoEParallelConfig` from TP/DP/EP/SP sizes
2. Computes expert counts (logical, global, redundant, fused-shared)
3. Creates `ExpertMapManager` for expert placement/mapping
4. Creates or accepts a `FusedMoERouter` via `create_fused_moe_router()`
5. Creates `FusedMoEConfig` (the single dataclass carrying all MoE dimensions/settings)
6. Creates `RoutedExperts` (which triggers `quant_method.create_weights()`)
7. Creates `MoERunner` and returns it

**Returns**: `MoERunner` — what the model stores as its MoE layer.

### 2. `MoERunner` (`runner/moe_runner.py`)

**Role**: The orchestrator. This is the `nn.Module` that models call `.forward()` on. It coordinates the entire MoE forward pass.

**Inherits**: `MoERunnerInterface` → `PluggableLayer` → `nn.Module`

**Key responsibilities**:

- **Gate application**: If the runner holds the gate (internal router), it applies `F.linear(hidden_states, gate_weight)` to produce `router_logits`. Supports fusing router + shared-expert gate weights.
- **Input/output transforms**: Applies `routed_input_transform` (e.g., latent projection for NemotronH) before expert computation and `routed_output_transform` after.
- **Padding**: Pads `hidden_states` to `moe_config.hidden_dim` when quantization backends require alignment.
- **Dispatch/Combine**: For DP/EP without internal MK support, dispatches tokens across ranks before computation and combines after.
- **Expert execution**: Delegates to `RoutedExperts.forward_modular()` or `.forward_monolithic()` depending on whether the quant method handles routing internally.
- **Shared experts**: Manages `SharedExperts` lifecycle — triggering computation before, after, or overlapped with routed experts via CUDA streams.
- **All-reduce**: Handles TP/EP all-reduce at the correct point (either after combine kernel or after shared+routed sum).
- **Scaling**: Applies `routed_scaling_factor` to output (with FP16 overflow protection).
- **CUDA graph support**: Registers itself as a custom op (`vllm.moe_forward` / `vllm.moe_forward_shared`) for torch.compile compatibility.

**Forward call chain**:

```python
forward()
  → apply_routed_input_transform()
  → _maybe_pad_hidden_states()
  → _forward_entry (custom op wrapper)
    → _forward_impl()
      → _maybe_sync_shared_experts_stream()
      → gate application (if internal)
      → _maybe_dispatch() (DP/EP token redistribution)
      → _apply_quant_method()
        → shared_experts(NO_OVERLAP)  [if MK can't overlap]
        → router.select_experts()    [modular path]
        → routed_experts.forward_modular() / forward_monolithic()
        → shared_experts(MULTI_STREAM_OVERLAPPED)
      → _maybe_combine()
  → _maybe_reduce_shared_expert_output()
  → _maybe_apply_routed_scale_to_output()
  → apply_routed_output_transform()
  → shared_output + fused_output
  → _maybe_reduce_final_output()
  → _maybe_add_zero_expert_output()
```

### 3. `FusedMoERouter` (`router/fused_moe_router.py`)

**Role**: Abstract base class for token-to-expert routing. Given hidden states and router logits, produces `(topk_weights, topk_ids)`.

**Key interface**:

- `select_experts(hidden_states, router_logits) → (topk_weights, topk_ids)` — public entry; calls `_select_experts()` then optionally records routing for replay.
- `routing_method_type` — returns a `RoutingMethodType` enum so MK backends can select specialized kernels.
- `eplb_state` — optional EPLB layer state for expert load balancing.

**Concrete implementations** (via `create_fused_moe_router` factory in `router/router_factory.py`):

- `TopKRouter` — standard softmax/sigmoid + top-k routing
- `ZeroExpertRouter` — adds a "zero expert" bias term to the output
- Custom routing function wrapper

**Not an `nn.Module`**: The router has no trainable parameters in the fused MoE path (the gate weights live on the model or on `MoERunner`).

### 4. `RoutedExperts` (`routed_experts.py`)

**Role**: Container for expert weight parameters and execution logic. This is where `w13_weight`, `w2_weight`, scales, zero points, etc. are registered as `nn.Parameter`s.

**Inherits**: `PluggableLayer` → `nn.Module`

**Key responsibilities**:

- **Weight creation**: Delegates to `quant_method.create_weights(layer=self, ...)` which registers parameters on this module.
- **Weight loading**: Implements `weight_loader()` — a complex method handling TP sharding, quantization-specific loading (per-tensor/channel/group/block scales, zero points, g_idx), and EP expert filtering. Also implements `load_weights()` for the newer fused loading path.
- **Execution**: Two forward paths:
    - `forward_modular(x, topk_weights, topk_ids, ...)` — for decomposed kernels where the router has already selected experts. Calls `quant_method.apply()`.
    - `forward_monolithic(x, router_logits, ...)` — for monolithic kernels that handle routing internally. Calls `quant_method.apply_monolithic()`.
- **Expert mapping**: Maintains `expert_map` (global→local ID tensor) and routing tables via `ExpertMapManager`. Supports EPLB weight rearrangement via `get_expert_weights()`.
- **Quant method**: Holds the `FusedMoEMethodBase` instance that determines which kernel to use.

### 5. `SharedExperts` (`runner/shared_experts.py`)

**Role**: Wrapper around a model-provided shared expert `nn.Module` that adds CUDA stream overlap and DBO (Dynamic Batch Ordering) support.

**Key features**:

- Runs shared experts on a separate CUDA stream when possible, overlapping with routed expert computation.
- Called at specific ordering points (`SharedExpertsOrder`): `NO_OVERLAP` (before MK), `MK_INTERNAL_OVERLAPPED` (by MK), `MULTI_STREAM_OVERLAPPED` (after MK, in aux stream).
- Supports DBO by maintaining per-ubatch output buffers.
- The underlying shared expert layer is a standard `nn.Module` (e.g., another `MLP` layer) provided by the model.

### 6. `FusedMoEConfig` (`config.py`)

**Role**: Central dataclass carrying all MoE configuration. Created once by the factory and shared by `MoERunner`, `RoutedExperts`, and other components.

**Key fields**:

- `num_experts`, `experts_per_token` (top_k), `hidden_dim`, `intermediate_size`
- `num_local_experts`, `num_logical_experts` (for EPLB)
- `activation: MoEActivation` (silu, gelu, etc. with gated/ungated distinction)
- `in_dtype`, `router_logits_dtype`
- `moe_parallel_config: FusedMoEParallelConfig` (all parallelism settings)
- `routing_method: RoutingMethodType`
- Computed: `intermediate_size_per_partition`, ROCm AITER flags

### 7. `FusedMoEParallelConfig` (`config.py`)

**Role**: Dataclass encoding all parallelism dimensions and backend selection.

**Key fields**: `tp_size/rank`, `dp_size/rank`, `ep_size/rank`, `pcp_size/rank`, `sp_size`, `use_ep`, `all2all_backend`, `enable_eplb`.

**Notable behavior**: When EP is enabled, TP is "collapsed" into EP — each device owns a full subset of experts rather than a shard of every expert. The `make()` factory method computes this flattening.

### 8. `ExpertMapManager` (`expert_map_manager.py`)

**Role**: Manages the mapping between global expert IDs and local (per-rank) expert IDs for Expert Parallelism.

**Key outputs**:

- `expert_map`: Tensor of shape `(global_num_experts,)` mapping global→local ID (-1 for non-local experts)
- `expert_mask`: Binary mask for AITER
- `routing_tables`: `(global_to_physical, physical_to_global, local_to_global)` for round-robin placement
- `local_num_experts`: How many experts this rank owns
- `placement_strategy`: "linear" (contiguous blocks) or "round_robin" (interleaved)

### 9. `FusedMoEMethodBase` (not shown in detail)

**Role**: Strategy pattern for quantization-specific expert execution. Each quantization scheme (FP8, INT8, INT4, NVFP4, MXFP4, unquantized, etc.) provides a subclass that knows how to:

- `create_weights()` — register the right parameters on `RoutedExperts`
- `apply()` — execute the fused MoE kernel with pre-computed routing (modular)
- `apply_monolithic()` — execute a kernel that handles routing internally (monolithic)
- `get_fused_moe_quant_config()` — produce a `FusedMoEQuantConfig` describing scales/shapes

**Two execution modes**:

- **Modular**: Router selects experts first → `apply(layer, x, topk_weights, topk_ids)`. Used by Triton, CUTLASS, and most backends.
- **Monolithic**: Kernel handles routing internally → `apply_monolithic(layer, x, router_logits)`. Used by FlashInfer TRTLLM and some specialized backends.

## Data Flow Summary

```python
hidden_states (from transformer block)
    │
    ▼
MoERunner.forward()
    │
    ├── [optional] routed_input_transform (latent projection)
    │
    ├── [optional] gate(hidden_states) → router_logits
    │
    ├── [optional] dispatch (DP/EP token redistribution)
    │
    ├── [MODULAR PATH]
    │   ├── FusedMoERouter.select_experts(hidden_states, router_logits)
    │   │   → (topk_weights, topk_ids)
    │   │
    │   └── RoutedExperts.forward_modular(x, topk_weights, topk_ids)
    │       → quant_method.apply(layer=routed_experts, ...)
    │           → fused MoE kernel (Triton/CUTLASS/etc.)
    │
    ├── [MONOLITHIC PATH]
    │   └── RoutedExperts.forward_monolithic(x, router_logits)
    │       → quant_method.apply_monolithic(layer=routed_experts, ...)
    │           → monolithic kernel (FlashInfer TRTLLM, etc.)
    │
    ├── [PARALLEL] SharedExperts(shared_input) → shared_output
    │   (on aux CUDA stream when possible)
    │
    ├── [optional] combine (DP/EP result aggregation)
    │
    ├── shared_output + fused_output  (element-wise add)
    │
    ├── [optional] routed_output_transform (latent → full dim)
    │
    ├── [optional] all-reduce (TP/EP)
    │
    └── final output → back to transformer block
```

## Key Design Decisions

1. **Factory pattern over constructor**: `FusedMoEFactory()` is a function, not a class. This avoids deep inheritance hierarchies and allows the factory to select different `MoERunner` / `RoutedExperts` subclasses via `runner_cls` / `routed_experts_cls`.

2. **Separation of routing from execution**: `FusedMoERouter` is decoupled from `RoutedExperts`. This allows monolithic kernels to bypass the router entirely while modular kernels use it for expert selection.

3. **Weights live on RoutedExperts, orchestration on MoERunner**: This separates concerns — `RoutedExperts` handles weight lifecycle (creation, loading, quantization) while `MoERunner` handles the forward pass orchestration (dispatch, shared experts, all-reduce).

4. **Quant method as strategy**: `FusedMoEMethodBase` encapsulates all quantization-specific logic, allowing the same `RoutedExperts` / `MoERunner` code to work with FP8, INT4, unquantized, etc.

5. **SharedExperts as wrapper**: Rather than embedding shared expert logic into the runner, `SharedExperts` wraps a model-provided `nn.Module` and adds stream overlap / DBO concerns. The model controls what the shared expert *is*; the MoE subsystem controls *when/how* it runs.

6. **Custom ops for CUDA graph compatibility**: `MoERunner.forward()` goes through `torch.ops.vllm.moe_forward` custom ops to work with `torch.compile` and CUDA graphs. The custom op looks up the runner by name from a static registry.
