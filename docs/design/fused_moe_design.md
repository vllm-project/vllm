# Fused MoE Layer Architecture Design Document

## Overview

The vLLM Mixture of Experts (MoE) subsystem lives under `vllm/model_executor/layers/fused_moe/`. The entry point is the `FusedMoEFactory()` factory function in `layer.py`, which assembles a pipeline of cooperating objects and returns a `MoERunner` — the `nn.Module` that models call directly in their forward pass.

## Directory Structure

```text
fused_moe/
  ├── layer.py                  — FusedMoEFactory() entry point
  ├── config.py                 — FusedMoEConfig, FusedMoEParallelConfig, FusedMoEQuantConfig, RoutingMethodType
  ├── routed_experts.py         — RoutedExperts (weight parameters, loading, execution)
  ├── modular_kernel.py         — FusedMoEKernel, FusedMoEExperts, FusedMoEPrepareAndFinalize base classes
  ├── fused_moe_method_base.py  — FusedMoEMethodBase (quant method strategy)
  ├── expert_map_manager.py     — ExpertMapManager (EP expert placement/mapping)
  ├── topk_weight_and_reduce.py — TopKWeightAndReduce implementations
  ├── activation.py             — MoEActivation definitions
  ├── fused_moe.py              — Legacy fused_experts function (Triton kernel entry)
  ├── runner/                   — MoERunner orchestrator and SharedExperts wrapper
  ├── router/                   — FusedMoERouter ABC and concrete router implementations
  ├── experts/                  — FusedMoEExperts subclasses (Triton, CUTLASS, DeepGemm, FlashInfer, TRTLLM, etc.)
  ├── prepare_finalize/         — FusedMoEPrepareAndFinalize subclasses (DeepEP, FlashInfer, Mori, NIXL, etc.)
  ├── oracle/                   — MoE kernel selection oracles (one per quantization type)
  └── configs/                  — Auto-tuned Triton kernel configs (JSON, keyed by E/N/device/dtype)
```

## Object Relationship Diagram

```text
Model (e.g. Mixtral, DeepSeek)
  │
  │  calls FusedMoEFactory(...) factory   ───────────────────────────┐
  │                                                                  │
  ▼                                                                  │
MoERunner (nn.Module, is the return value)                           │
  ├── router: FusedMoERouter          ◄── created by factory         │
  ├── routed_experts: RoutedExperts   ◄── created by factory         │
  ├── _shared_experts: SharedExperts? ◄── wraps model-provided layer │
  ├── gate: nn.Module?                                               │
  ├── shared_expert_gate: nn.Module?                                 │
  ├── routed_input_transform: nn.Module?                             │
  ├── routed_output_transform: nn.Module?                            │
  ├── routed_scaling_factor: float?                                  │
  ├── enable_dbo: bool                                               │
  ├── layer_name: str                                                │
  └── moe_config: FusedMoEConfig     ◄── created by factory          │
                                                                     │
RoutedExperts (nn.Module)                                            │
  ├── quant_method: FusedMoEMethodBase  (owns expert weight params)  │
  ├── expert_map_manager: ExpertMapManager                           │
  ├── moe_config: FusedMoEConfig                                     │
  └── [w13_weight, w2_weight, scales, ...] (registered parameters)   │
                                                                     │
FusedMoERouter (ABC, not nn.Module)                                  │
  ├── eplb_state: EplbLayerState?                                    │
  └── Concrete: FusedTopKRouter, ZeroExpertRouter,                   │
      GroupedTopKRouter, FusedTopKBiasRouter,                        │
      AiterSharedRoutedFusedMoERouter, RoutingSimulatorRouter, etc.  │
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

The concrete classes used for `MoERunner` and `RoutedExperts` can be overridden by passing `runner_cls` / `routed_experts_cls` (and optional `runner_args` / `routed_experts_args`) to the factory. This allows models to supply specialized subclasses when the default implementations are insufficient.

**Returns**: `MoERunner` — what the model stores as its MoE layer.

### 2. `MoERunner` (`runner/moe_runner.py`)

**Role**: The orchestrator. This is the `nn.Module` that models call `.forward()` on. It coordinates the entire MoE forward pass.

**Inherits**: `MoERunnerInterface(PluggableLayer, ABC)` → `PluggableLayer` → `nn.Module`

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
  → truncate fused_output to og_hidden_dim_pre_xform (if padded)
  → _maybe_reduce_routed_output_before_transform()  (latent MoE pre-reduction)
  → _maybe_reduce_shared_expert_output()
  → _maybe_apply_routed_scale_to_output()
  → apply_routed_output_transform()
  → shared_output + fused_output
  → _maybe_reduce_final_output()
  → _maybe_add_zero_expert_output()
```

**Data flow summary**:

```text
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
    ├── [optional] truncate fused_output (undo hidden_dim padding)
    │
    ├── [optional] pre-transform all-reduce (latent MoE: reduce before non-linear output transform)
    │
    ├── [optional] reduce shared_expert_output (when fused_output already reduced)
    │
    ├── [optional] apply routed_scaling_factor
    │
    ├── [optional] routed_output_transform (latent → full dim)
    │
    ├── shared_output + fused_output  (element-wise add)
    │
    ├── [optional] all-reduce (TP/EP, when not already reduced above)
    │
    └── final output → back to transformer block
```

### 3. `FusedMoERouter` (`router/fused_moe_router.py`)

**Role**: Abstract base class for token-to-expert routing. Given hidden states and router logits, produces `(topk_weights, topk_ids)`.

**Key interface**:

- `select_experts(hidden_states, router_logits) → (topk_weights, topk_ids)` — public entry; calls `_select_experts()` then optionally records routing for replay.
- `routing_method_type` — returns a `RoutingMethodType` enum so MK backends can select specialized kernels.
- `eplb_state` — optional EPLB layer state for expert load balancing.

**Concrete implementations** (via `create_fused_moe_router` factory in `router/router_factory.py`):

- `FusedTopKRouter` — standard softmax/sigmoid + top-k routing
- `FusedTopKBiasRouter` — top-k routing with e_score_bias (DeepSeek V3, MiniMax)
- `GroupedTopKRouter` — grouped top-k routing (DeepSeek V3 style)
- `ZeroExpertRouter` — adds a "zero expert" bias term to the output
- `AiterSharedRoutedFusedMoERouter` — ROCm AITER shared+routed fused router
- `RoutingSimulatorRouter` — routing simulation for testing/analysis
- `CustomRoutingRouter` — wrapper for custom routing functions

**Not an `nn.Module`**: The router has no trainable parameters in the fused MoE path (the gate weights live on the model or on `MoERunner`).

### 4. `RoutedExperts` (`routed_experts.py`)

**Role**: Container for expert weight parameters, weight loading, and execution logic. `RoutedExperts` is the component responsible for all expert weight lifecycle — creation, loading from checkpoints, and passing weights to kernels at inference time. This is where `w13_weight`, `w2_weight`, scales, zero points, etc. are registered as `nn.Parameter`s.

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

### 6. Configuration Classes (`config.py`)

All MoE configuration dataclasses and enums live in `config.py`.

#### 6a. `RoutingMethodType`

**Role**: IntEnum describing the routing algorithm used by the router. Passed to monolithic kernels (e.g. FlashInfer TRTLLM) so they can select specialized routing implementations internally.

**Values**: `Default` (Softmax→TopK), `Renormalize` (TopK→Softmax), `DeepSeekV3` (Sigmoid+Bias→GroupedTopK), `Llama4` (Top1→Sigmoid), `RenormalizeNaive` (Softmax→TopK→Renormalize), `TopK` (TopK only), `SigmoidRenorm` (Sigmoid→TopK→Renormalize), `MiniMax2` (Sigmoid+Bias→TopK→ScaledSumNormalize), `Sigmoid` (Sigmoid→TopK), `DeepseekV4` (SqrtSoftplus+Bias→Normalize), `Custom`, `Simulated`, `Unspecified`.

The helper function `get_routing_method_type()` maps model parameters (`scoring_func`, `top_k`, `renormalize`, `num_expert_group`, `has_e_score_bias`) to the appropriate enum value.

#### 6b. `FusedMoEQuantDesc`

**Role**: Dataclass describing the quantization of a single tensor (one activation or one weight matrix).

**Key fields**:

- `dtype: torch.dtype | str | None` — the quantized type (`None` means unquantized)
- `shape: GroupShape | None` — quantization granularity: `PER_TENSOR` (-1,-1), `PER_TOKEN` (1,-1), or block shape like `(128, 128)`
- `scale: torch.Tensor | PrecisionConfig | None` — quantization scales
- `alpha_or_gscale: torch.Tensor | None` — per-channel scales or global scales (NVFP4, W4A8)
- `zp: torch.Tensor | None` — zero points (INT4/INT8)
- `bias: torch.Tensor | None` — biases (GPT Triton MoE)

#### 6c. `FusedMoEQuantConfig`

**Role**: Dataclass bundling the quantization parameters for a complete fused MoE operation. Contains four `FusedMoEQuantDesc` instances — one for each tensor in the two-GEMM MoE pipeline:

- `_a1` — first activation (input to GEMM1)
- `_w1` — first weight (gate+up projection)
- `_a2` — second activation (intermediate, input to GEMM2)
- `_w2` — second weight (down projection)

Each `FusedMoEMethodBase` subclass implements `get_fused_moe_quant_config()` to construct this from loaded weights. The oracle's `make_quant_config` function also builds these during kernel construction.

**Key convenience properties**: `quant_dtype`, `weight_quant_dtype`, `is_quantized`, `per_act_token_quant`, `per_out_ch_quant`, `block_shape`, `a1_scale`, `a2_scale`, `w1_scale`, `w2_scale`, `w1_zp`, `w1_bias`, `w1_precision`, `w2_precision`.

**Additional fields**: `is_scale_swizzled`, `gemm1_alpha`/`gemm1_beta`/`gemm1_clamp_limit` (MXFP4 TRTLLM SwiGLU clamping), `mx_alignment`.

**Usage**: `FusedMoEQuantConfig` is only used with modular kernels — it is passed to `FusedMoEExperts` constructors. Non-modular MoE methods can set it to `None`.

#### 6d. `FusedMoEParallelConfig`

**Role**: Dataclass encoding all parallelism dimensions and backend selection.

**Key fields**: `tp_size/rank`, `dp_size/rank`, `ep_size/rank`, `pcp_size/rank`, `sp_size`, `use_ep`, `all2all_backend`, `enable_eplb`.

**Computed properties**: `use_all2all_kernels`, `use_deepep_ht_kernels`, `use_deepep_ll_kernels`, `use_deepep_v2_kernels`, `use_fi_nvl_two_sided_kernels`, `use_fi_nvl_one_sided_kernels`, `use_ag_rs_all2all_kernels`, `use_mori_kernels`, `use_nixl_ep_kernels`, `use_batched_activation_format`, `needs_round_robin_routing_tables`, `is_sequence_parallel`.

**Notable behavior**: When EP is enabled, TP is "collapsed" into EP — each device owns a full subset of experts rather than a shard of every expert. The `make()` factory method computes this flattening.

#### 6e. `FusedMoEConfig`

**Role**: Central dataclass carrying all MoE layer configuration. Created once by the factory and shared by `MoERunner`, `RoutedExperts`, and other components.

**Key fields**:

- `num_experts`, `experts_per_token` (top_k), `hidden_dim`, `intermediate_size`
- `num_local_experts`, `num_logical_experts` (for EPLB)
- `activation: MoEActivation` (silu, gelu, situglu, etc. with gated/ungated distinction)
- `in_dtype`, `router_logits_dtype`
- `moe_parallel_config: FusedMoEParallelConfig` (all parallelism settings)
- `routing_method: RoutingMethodType`
- `device: torch.device | str`
- `hidden_dim_unpadded`, `intermediate_size_per_partition_unpadded`, `intermediate_pad` (padding/alignment)
- `moe_backend: MoEBackend` (kernel selection)
- `max_num_tokens`, `has_bias`, `is_lora_enabled`
- `skip_final_all_reduce`, `defer_moe_finalize` (reduction/finalization control)
- `swiglu_limit`, `swiglu_alpha`, `swiglu_beta` (SwiGLU clamp parameters)
- `activation_situ_beta`, `activation_situ_linear_beta` (SituGLU parameters)
- `max_capture_size` (CUDA graph capture)
- Computed: `intermediate_size_per_partition`, `rocm_aiter_fmoe_enabled`, `aiter_fmoe_shared_expert_enabled`

### 7. `ExpertMapManager` (`expert_map_manager.py`)

**Role**: Manages the mapping between global expert IDs and local (per-rank) expert IDs for Expert Parallelism (EP).

**Key outputs**:

- `expert_map`: Tensor of shape `(global_num_experts,)` mapping global→local ID (-1 for non-local experts)
- `expert_mask`: Binary mask for AITER
- `routing_tables`: `(global_to_physical, physical_to_global, local_to_global)` for round-robin placement
- `local_num_experts`: How many experts this rank owns
- `placement_strategy`: "linear" (contiguous blocks) or "round_robin" (interleaved)

### 8. `FusedMoEMethodBase` (`fused_moe_method_base.py`)

**Role**: Strategy pattern for quantization-specific expert execution. Each quantization scheme (FP8, INT8, INT4, NVFP4, MXFP4, unquantized, etc.) provides a subclass that knows how to create weights, build quantization configs, and execute the fused MoE kernel. See [Fused MoE Modular Kernel](./fused_moe_modular_kernel.md) for details on the kernel internals.

**Key attributes**:

- `moe: FusedMoEConfig` — the MoE layer configuration
- `moe_quant_config: FusedMoEQuantConfig | None` — cached quantization config built by `get_fused_moe_quant_config()`
- `moe_kernel: FusedMoEKernel | None` — the modular kernel object constructed by the oracle (see section 9). When set, `apply()` and `apply_monolithic()` delegate to it.

**Key methods**:

- `create_weights()` — register the right parameters on `RoutedExperts`
- `apply()` — execute the fused MoE kernel with pre-computed routing (modular path)
- `apply_monolithic()` — execute a kernel that handles routing internally (monolithic path)
- `get_fused_moe_quant_config()` — produce a `FusedMoEQuantConfig` describing scales/shapes

**Key properties**:

- `supports_internal_mk` — `True` when `moe_kernel` is set; signals `MoERunner` to use the internal modular kernel path (dispatch/combine handled by the kernel's `FusedMoEPrepareAndFinalize`) rather than the legacy external dispatch/combine
- `mk_can_overlap_shared_experts` — `True` when the kernel's prepare/finalize supports async operation, enabling shared expert overlap via DBO
- `is_monolithic` — whether the kernel handles routing internally

**Two execution modes**:

- **Modular**: Router selects experts first → `apply(layer, x, topk_weights, topk_ids)`. Used by Triton, CUTLASS, and most backends.
- **Monolithic**: Kernel handles routing internally → `apply_monolithic(layer, x, router_logits)`. Used by FlashInfer TRTLLM and some specialized backends.

### 9. MoE Kernel Oracle (`oracle/`)

**Role**: Each quantization type has an oracle that selects the best MoE kernel backend for a given (model, hardware, deployment-config) tuple. Oracles live under `fused_moe/oracle/` — one module per quantization type: `unquantized.py`, `fp8.py`, `nvfp4.py`, `mxfp4.py`, `mxfp8.py`, `int8.py`, `int_wna16.py`, `w4a8.py`, `w4a8_int8.py`.

**Target interface**: `MoEKernelOracle` (in `oracle/base.py`) is the abstract base class that all oracles will eventually inherit from. Currently only `UnquantizedMoEKernelOracle` has been migrated to this class-based pattern. The remaining oracles use an equivalent set of module-level free functions following the same informal convention.

**Four responsibilities**:

1. **Backend selection** (`select_backend` / `select_*_moe_backend`): Given a `FusedMoEConfig`, returns the best `(backend_enum, FusedMoEExperts class)` pair. Internally, this enumerates backends in platform-specific priority order (via `get_priority_backends`), maps each to its candidate `FusedMoEExperts` subclasses (via `backend_to_kernel_cls`), and picks the first one where `FusedMoEExperts.is_supported_config()` passes. The user can override the backend via `--moe-backend` (mapped through `map_backend`).

2. **Quantization config construction** (`make_quant_config` / `make_*_moe_quant_config`): Builds a `FusedMoEQuantConfig` from the loaded weight parameters (scales, zero points, block shapes). This config describes the quantization scheme for each of the four tensors (a1, a2, w1, w2) and is passed to the kernel at construction time. Not used by the unquantized oracle.

3. **Weight post-processing** (`convert_to_kernel_format` / `convert_to_*_moe_kernel_format`): Transforms weights into the layout expected by the selected backend after loading. For example, AITER and FlashInfer require specific weight permutations. The default is a no-op pass-through.

4. **Modular kernel construction** (`make_kernel` / `make_*_moe_kernel`): Constructs the `FusedMoEKernel` object by pairing the selected `FusedMoEExperts` subclass with the appropriate `FusedMoEPrepareAndFinalize` subclass (determined by the all2all backend from `FusedMoEParallelConfig`). The resulting `FusedMoEKernel` is stored on `FusedMoEMethodBase.moe_kernel`.

**Free-function oracle convention** (used by all oracles except unquantized):

```text
oracle/{quant_type}.py
  ├── {Quant}MoeBackend (Enum)           — backend choices for this quant type
  ├── _get_priority_backends(...)        — platform-specific backend priority
  ├── backend_to_kernel_cls(...)         — backend → [FusedMoEExperts subclasses]
  ├── map_{quant}_backend(...)           — MoEBackend → {Quant}MoeBackend
  ├── select_{quant}_moe_backend(...)    — primary entry: choose best backend
  ├── make_{quant}_moe_quant_config(...) — build FusedMoEQuantConfig from weights
  ├── convert_to_{quant}_moe_kernel_format(...)  — weight layout transform
  └── make_{quant}_moe_kernel(...)       — construct FusedMoEKernel
```

**`MoEKernelOracle` class** (target interface in `oracle/base.py`):

The abstract methods mirror the free-function convention: `backend_enum_cls`, `get_priority_backends`, `backend_to_kernel_cls`, `map_backend`, `select_backend`, `make_kernel`, `convert_to_kernel_format`, `make_quant_config`. Going forward, all oracles will be migrated to inherit from `MoEKernelOracle`.

## Key Design Decisions

1. **Factory pattern over constructor**: `FusedMoEFactory()` is a function, not a class. This avoids deep inheritance hierarchies and allows the factory to select different `MoERunner` / `RoutedExperts` subclasses via `runner_cls` / `routed_experts_cls`.

2. **Separation of routing from execution**: `FusedMoERouter` is decoupled from `RoutedExperts`. This allows monolithic kernels to bypass the router entirely while modular kernels use it for expert selection.

3. **Weights live on RoutedExperts, orchestration on MoERunner**: This separates concerns — `RoutedExperts` handles weight lifecycle (creation, loading, quantization) while `MoERunner` handles the forward pass orchestration (dispatch, shared experts, all-reduce).

4. **Quant method as strategy**: `FusedMoEMethodBase` encapsulates all quantization-specific logic, allowing the same `RoutedExperts` / `MoERunner` code to work with FP8, INT4, unquantized, etc.

5. **SharedExperts as wrapper**: Rather than embedding shared expert logic into the runner, `SharedExperts` wraps a model-provided `nn.Module` and adds stream overlap / DBO concerns. The model controls what the shared expert *is*; the MoE subsystem controls *when/how* it runs.

6. **Custom ops for CUDA graph compatibility**: `MoERunner.forward()` goes through `torch.ops.vllm.moe_forward` custom ops to work with `torch.compile` and CUDA graphs. The custom op looks up the runner by name from a static registry.
