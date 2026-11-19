# Contributing Fused MoE Implementations

This guide covers adding experts kernels, dispatch/combine backends, MoE quantization methods, and kernel-selection oracles. It describes both the implementation contract and the integration needed to make an implementation selectable in a real model.

Start with the [Fused MoE design](../design/fused_moe_design.md) for component ownership and the [modular kernel design](../design/fused_moe_modular_kernel.md) for execution contracts. Follow the [development setup](README.md#developing) before running the checks below.

## Choose the extension point

| Change | Main implementation | Production integration |
| ------ | ------------------- | ---------------------- |
| New experts compute kernel | `FusedMoEExpertsModular` or `FusedMoEExpertsMonolithic` under `fused_moe/experts/` | Add candidates and any weight conversion to the relevant quantization-family oracle. |
| New dispatch/combine backend | All2All manager and prepare/finalize adapter | Connect backend configuration, communicator construction, activation-format selection, and the prepare/finalize factory. |
| New MoE quantization method | `FusedMoEMethodBase` subclass under `quantization/` | Return it from the quantization config for `RoutedExperts`; implement weight loading, post-processing, and execution. |
| New selection policy or quantization-family oracle | Module under `fused_moe/oracle/` | Connect the quantization method to selection, conversion, quantization-config construction, and kernel assembly. |

The paths in this table are relative to `vllm/model_executor/layers/`, except the distributed All2All manager. These extension points are related, but a new compute backend for an existing quantization scheme usually does not need a new quantization method or oracle module.

!!! important "Production selection and test registration are separate"
    Adding a class to the modular-kernel test registry does not make the production oracle or dispatcher factory select it. Conversely, a production implementation is not automatically exercised by the kernel-combination tests.

## Add an experts kernel

### 1. Implement the execution interface

Add the adapter under [`fused_moe/experts/`](../../vllm/model_executor/layers/fused_moe/experts/). Choose the base class in [`modular_kernel.py`](../../vllm/model_executor/layers/fused_moe/modular_kernel.py) according to who owns routing:

- `FusedMoEExpertsModular` consumes precomputed top-k IDs and weights. Implement `workspace_shapes()`, `finalize_weight_and_reduce_impl()`, and `apply()`. Normally `apply()` writes into the supplied output buffer and returns `None`; only the supported deferred-finalization path returns `UnfinalizedMoEOutput` instead.
- `FusedMoEExpertsMonolithic` consumes router logits and performs routing internally. Implement its `apply()` contract and declare the supported routing methods and router-logits dtypes. It must be paired with a monolithic prepare/finalize adapter.

For modular experts, specify the two scratch-buffer shapes and output shape. Override `workspace_dtype()` if the default activation dtype is unsuitable. Choose the top-k weight/reduction implementation according to whether the experts or finalization perform that work; do not apply router weights a second time when they were already applied to the input.

Use an existing adapter with the same layout and quantization scheme as a reference. For example, [`TritonExperts`](../../vllm/model_executor/layers/fused_moe/experts/triton_moe.py) demonstrates a modular adapter and extending the shared support checker.

### 2. Declare capabilities and input contracts

Implement the required static methods inherited from `FusedMoEExperts`:

- `activation_format()`
- `_supports_current_device()`
- `_supports_no_act_and_mul()`
- `_supports_quant_scheme()`
- `_supports_activation()`
- `_supports_parallel_config()`

The inherited `is_supported_config()` combines these predicates with checks for layout, routing, shape, LoRA, batch invariance, and activation-quantization fallback. It returns `(supported, reason)`, not just a boolean. Prefer implementing the predicates; if additional constraints require overriding the checker, preserve the shared checks and return a useful rejection reason.

In particular:

- Declare `PaddedStandard` only if inactive rows marked by `topk_ids == -1` cannot affect valid results. Such rows may contain uninitialized data. A padding-aware experts implementation can also accept `Standard`, but the reverse is not true.
- For `BatchedExperts`, consume the valid-token metadata rather than treating the entire per-expert capacity as valid. The common kernel builder supplies `max_num_tokens` and `num_dispatchers` to the constructor.
- Set `expects_unquantized_inputs` when input quantization belongs inside the experts. Pair it only with prepare/finalize implementations that support `defer_input_quant=True`.
- If the executed activation quantization differs from the requested scheme, implement `executed_activation_key()` consistently with the kernel and weight-conversion path. This lets the shared checker enforce strict quantization-scheme selection.
- Override optional capability hooks only for features actually implemented, including `supports_lora()`. Passing the ordinary forward test is not evidence of LoRA or all parallel-layout support.

### 3. Make the experts selectable

Extend the oracle for each supported quantization family, following [Extend or add an oracle](#extend-or-add-an-oracle). For an FP8 implementation, the integration points are in [`oracle/fp8.py`](../../vllm/model_executor/layers/fused_moe/oracle/fp8.py):

1. Add the class to `backend_to_kernel_cls()` for the appropriate backend enum. Introduce an enum member only if it represents a distinct backend choice.
2. Update automatic-selection priority and explicit backend mapping where applicable. A class that is never reached by either path will not be used.
3. Add the required weight/scale conversion to `convert_to_fp8_moe_kernel_format()` and any backend-specific settings to quantization-config construction. A new kernel may require different packing even when the checkpoint scheme is unchanged.
4. Pass any nonstandard constructor arguments through kernel assembly. The common `make_moe_kernel()` helper accepts `experts_kwargs` and `prepare_finalize_kwargs`.

Keep model weights owned by `RoutedExperts`. If the adapter needs additional post-load preparation or retained constants, implement its `process_weights_after_loading()`/`kept_constants` contract and ensure the calling quantization method invokes the corresponding initialization hooks.

### 4. Validate selection and computation separately

Register modular experts with `register_experts()` in [`mk_objects.py`](../../tests/kernels/moe/modular_kernel_tools/mk_objects.py), guarded by the appropriate optional-dependency/platform checks. Check that the harness can construct the class and prepare its inputs: custom constructor arguments, quantization schemes, or packed weight layouts may also require changes to `make_fused_experts()`, `MK_QUANT_CONFIGS`, or the helpers in [`common.py`](../../tests/kernels/moe/modular_kernel_tools/common.py).

Then exercise:

- Oracle selection with supported and unsupported configurations, including both automatic and explicit backend selection.
- Numerical correctness in [`test_modular_kernel_combinations.py`](../../tests/kernels/moe/test_modular_kernel_combinations.py) for the layouts and quantization schemes claimed, including empty/uneven expert assignments and padding where applicable.
- The actual layer path in [`test_moe_layer.py`](../../tests/kernels/moe/test_moe_layer.py), so selection, converted weights, and execution are tested together. Monolithic implementations need layer or backend-specific tests; the modular combination registry does not cover their routing interface.

## Add a dispatch/combine backend

### 1. Add or reuse an All2All manager

The manager owns communication setup and reusable handles/resources. Start with [`All2AllManagerBase`](../../vllm/distributed/device_communicators/base_device_communicator.py) and the implementations in [`all2all.py`](../../vllm/distributed/device_communicators/all2all.py). Reuse an existing manager if only the prepare/finalize adapter is changing. Ensure resources are scoped to the appropriate process group and cleaned up by the manager's lifecycle.

### 2. Implement the prepare/finalize adapter

Add the adapter under [`fused_moe/prepare_finalize/`](../../vllm/model_executor/layers/fused_moe/prepare_finalize/), choosing the modular or monolithic interface to match the experts. For a modular adapter, implement `prepare()` and `finalize()` plus the required common contracts:

- `activation_format` is a property describing the output of preparation.
- `topk_indices_dtype()` declares a routing-ID dtype requirement, or returns `None`.
- `max_num_tokens_per_rank()` and `num_dispatchers()` describe dispatch capacity; return `None` for the maximum if there is no fixed token limit.
- `output_is_reduced()` describes cross-rank reduction ownership, not just top-k reduction within a token.

Honor the input-quantization ownership contract. Preserve or return the routing IDs, weights, and token metadata needed by the experts and combine operation. Account for local/global expert IDs and routing tables when expert placement is not contiguous.

If supporting async operation, implement both async entry points and follow the [hook/receiver contract](../design/fused_moe_modular_kernel.md#asynchronous-prepare-and-finalize). `supports_dbo()` defaults to `supports_async()`; a synchronous implementation may opt into DBO only if it performs the required scheduling handoff itself. Do not advertise overlap support merely because the communication library has asynchronous primitives.

### 3. Connect production configuration and construction

For a new user-selectable backend, check all of these integration points:

1. Add its accepted name to `All2AllBackend` in [`vllm/config/parallel.py`](../../vllm/config/parallel.py), with any necessary configuration validation. Changing automatic backend selection is a separate policy decision; do not change defaults just to expose the new choice.
2. Instantiate the manager in the relevant device communicator. [`CudaCommunicator`](../../vllm/distributed/device_communicators/cuda_communicator.py) handles CUDA/ROCm paths; check the corresponding communicator for other platforms. An accepted configuration value is not enough if the communicator still rejects it as unknown.
3. Add or update backend predicates in `FusedMoEParallelConfig` in [`fused_moe/config.py`](../../vllm/model_executor/layers/fused_moe/config.py), with the appropriate EP/DP/TP/SP gating.
4. Update `FusedMoEConfig.activation_format` if necessary. Oracles consult this property before constructing prepare/finalize, so it must predict the format the new adapter actually produces.
5. Add the construction branch to `maybe_make_prepare_finalize()` in [`all2all_utils.py`](../../vllm/model_executor/layers/fused_moe/all2all_utils.py). Pass the manager handle, capacity, quantization settings, and routing tables the adapter needs. Reject unsupported modular/monolithic combinations explicitly.

[`DeepEPLLAll2AllManager`](../../vllm/distributed/device_communicators/all2all.py) and [`DeepEPLLPrepareAndFinalize`](../../vllm/model_executor/layers/fused_moe/prepare_finalize/deepep_ll.py) provide a traceable example of these connections for a batched backend. Their quantization and hardware constraints are specific to DeepEP LL, not requirements for every dispatcher.

### 4. Register and test the backend

Use `register_prepare_and_finalize()` in [`mk_objects.py`](../../tests/kernels/moe/modular_kernel_tools/mk_objects.py), including the production backend name and accurate compatibility metadata. The harness constructs prepare/finalize through the production factory, so this registration does not replace step 3.

Run compatible multi-rank combinations and the layer-level parallel cases. Include empty ranks, uneven token counts, capacity boundaries, expert-ID mapping, and top-k weight/reduction ownership. If claiming DBO, shared-expert overlap, sequence parallelism, or EPLB, exercise those paths explicitly. Also test rejection of incompatible experts layouts and unsupported deferred input quantization.

Document hardware/interconnect/dependency requirements and update the [backend feature table](../design/moe_kernel_features.md#fused-moe-modular-all2all-backends).

## Add a MoE quantization method

Use [`Fp8MoEMethod`](../../vllm/model_executor/layers/quantization/fp8.py) as a lifecycle example, not as a template for every weight representation. Check the closest existing scheme before adding a new class.

### 1. Connect quantization-config selection

Implement a [`FusedMoEMethodBase`](../../vllm/model_executor/layers/fused_moe/fused_moe_method_base.py) subclass and make the relevant `QuantizationConfig.get_quant_method()` return it for `RoutedExperts`. `RoutedExperts._get_quant_method()` uses `resolve_quant_method()` and otherwise falls back to the unquantized method. Defining the subclass alone does not select it.

If extending an existing format, update that format's selection logic, such as [`Fp8Config.get_quant_method()`](../../vllm/model_executor/layers/quantization/fp8.py) or the [`compressed-tensors`](../../vllm/model_executor/layers/quantization/compressed_tensors/compressed_tensors.py) scheme selection. A new top-level format also needs configuration discovery in [`quantization/__init__.py`](../../vllm/model_executor/layers/quantization/__init__.py), checkpoint-config handling, and appropriate platform support. That module provides `register_quantization_config()` for custom configs; built-in formats use its supported-name and config-lookup tables.

### 2. Select a compatible backend and create loadable parameters

Determine the weight and activation `QuantKey`s and call the appropriate oracle. These keys describe scheme compatibility, including scaling granularity; they are not the runtime tensors in `FusedMoEQuantConfig`. If the scheme is new, follow [Extend or add an oracle](#extend-or-add-an-oracle).

Implement `create_weights()` to register expert weights, scales, zero points, and biases as required on `RoutedExperts`. Respect the local expert count and partitioned dimensions supplied by the layer. Preserve the supplied loader attributes, including `weight_loader`, and attach the scale/packing metadata expected by [`RoutedExperts.weight_loader()`](../../vllm/model_executor/layers/fused_moe/routed_experts.py). Account for merged gate/up weights, non-gated models, sharding, and padding where supported.

If the backend requires dimension alignment before allocation, implement the appropriate `maybe_roundup_sizes()` behavior. Do not assume the checkpoint layout is already the selected kernel's runtime layout.

### 3. Complete post-load initialization

Implement `process_weights_after_loading()` with a clear order:

1. Process loaded weights/scales and convert them to the selected backend's format.
2. Replace or update the layer's parameters using the existing parameter helpers as appropriate, preserving the weight-reload contract.
3. Implement `get_fused_moe_quant_config()` using the final runtime tensors and cache the result on `moe_quant_config`.
4. Construct the kernel through the oracle, passing expert routing tables where required, and store it on `moe_kernel`.
5. Run any required experts post-load hooks and retain constants via `keep_constants()`.

`Fp8MoEMethod._setup_kernel()` and `_init_moe_kernel()` show this ordering. Its preprocessed-weight path rebuilds the kernel without converting weights again. Support preprocessed weights or weight reloads only with the corresponding lifecycle implemented and tested.

### 4. Implement execution entry points

Implement `apply()` for modular execution and `apply_monolithic()` if the selected backend supports monolithic execution. Forward the layer's weights, activation/routing settings, expert map, and applicable shared-expert inputs to the kernel. Preserve deferred-finalization results when that path is supported.

!!! warning
    Assigning `moe_kernel` does not supply these entry points: the base `apply()` and `apply_monolithic()` raise `NotImplementedError`. `supports_internal_mk` becomes True when the kernel is set, so the concrete method must implement the interface that the runner will call.

Only advertise `supports_eplb=True` when the method's weight/scale representation and execution remain valid after expert redistribution. EP working with a fixed expert assignment is not sufficient evidence for EPLB.

### 5. Test the loading lifecycle, not only the GEMMs

Extend the nearest quantization and layer tests. Cover config-to-method selection, checkpoint weight/scale loading, conversion, and a real forward pass with the selected backend. Include TP/EP loading and reloads when supported. [`test_moe_weight_loading_padded.py`](../../tests/kernels/moe/test_moe_weight_loading_padded.py) and [`test_moe_layer.py`](../../tests/kernels/moe/test_moe_layer.py) are useful existing suites for layer/loader behavior.

The modular combination harness constructs synthetic inputs and experts directly; it does not validate the complete quantization-method lifecycle. Add or extend a model test using the real checkpoint-loading path under [`tests/models/quantization/`](../../tests/models/quantization/), and run the relevant model evaluation for output-affecting changes.

## Extend or add an oracle

An oracle owns selection and backend-specific conversion/configuration for a quantization family. Prefer extending that family's existing module when adding another compute backend. Add a new oracle when a new family needs its own selection and weight-processing policy, not for every experts class.

### 1. Preserve the current entry points

The shared [`MoEKernelOracle`](../../vllm/model_executor/layers/fused_moe/oracle/base.py) interface is being adopted incrementally. Class-based helpers and module-level functions coexist. Inspect the calling quantization method and keep its selection, conversion, config, and construction calls connected; creating a subclass or module does not auto-register it.

For a new oracle, provide its backend enum, candidate mapping, priority policy, explicit-backend mapping, weight conversion, quantization-config construction, and kernel construction. Implement every entry point used by its quantization method rather than relying on unimplemented target methods in the base class. If compatibility functions delegate to class methods, keep that delegation one-way to avoid recursion.

### 2. Connect backend choices

For the FP8 example, the relevant symbols in [`oracle/fp8.py`](../../vllm/model_executor/layers/fused_moe/oracle/fp8.py) are `Fp8MoeBackend`, `_get_priority_backends()`, `backend_to_kernel_cls()`, `map_fp8_backend()`, and `select_fp8_moe_backend()`.

- Map each backend to an ordered list of candidate experts classes, including layout-specific variants where needed.
- Include a new backend in automatic priority only on the configurations where it should be considered. Compatibility checks belong in the experts support predicates; priority expresses preference among supported choices.
- If exposing a new `--moe-backend` value, update `MoEBackend` in [`vllm/config/kernel.py`](../../vllm/config/kernel.py) and the relevant oracle mapping. A new enum member alone does not create a valid public option. A new implementation behind an existing choice may not need a new public value.
- Preserve existing environment-variable precedence, layout remapping, and quantization-fallback policy. Explicit selection must still be checked against the deployment configuration.

### 3. Reuse probing and assembly helpers

Instantiate your `MoEKernelOracle` subclass with the deployment config and quantization keys. Define its logging `name` and `backend_to_kernel_cls()` mapping. When wrapping a module-level mapping, forward to it at call time rather than capturing it in a static alias.

Use the helpers in [`oracle/base.py`](../../vllm/model_executor/layers/fused_moe/oracle/base.py):

- `try_backend()` probes candidates and returns the selected class, if any, plus rejection reasons. Use it while walking automatic priorities and retain useful diagnostics when no candidate works.
- `return_or_raise()` returns a supported explicitly selected backend/class pair or raises with a rejection reason. Preserve any existing oracle-specific mapping/fallback rules around this helper.
- `make_moe_kernel()` selects prepare/finalize, checks activation-format compatibility, supplies batched constructor parameters, constructs the experts, and returns `FusedMoEKernel`. Use its keyword-extension dictionaries for additional adapter/experts construction inputs.

Kernel assembly does not load or convert checkpoint weights, and does not replace the quantization method's post-load hooks. Connect the oracle's conversion and `FusedMoEQuantConfig` construction to the loading sequence described above. If the family supports online requantization sources, preserve the kernel-less source path; `maybe_skip_for_requantization_source()` uses the oracle's `placeholder_backend` for that case.

### 4. Test policy separately from kernels

Extend [`test_moe_kernel_oracle.py`](../../tests/kernels/moe/test_moe_kernel_oracle.py) for shared probing/assembly behavior and the nearest family-specific suite, such as [`test_unquantized_backend_selection.py`](../../tests/kernels/moe/test_unquantized_backend_selection.py), for policy. Mock device/dependency probes for selection-only tests rather than launching kernels to test a priority list.

Cover automatic ordering and fallback, explicit mapping and rejection behavior, informative failures when no candidate is supported, and activation-format selection from the deployment config. Test weight/activation key mismatches and relevant LoRA, routing, or parallel restrictions. Assembly tests should verify the selected adapter/experts pairing and required construction inputs; numerical correctness and loading still need their separate tests.

## Validation and handoff

Use the smallest existing suite that exercises each changed contract, then validate the integrated path. Do not treat a skipped test or a test of a fallback implementation as coverage of the new backend.

- **Selection:** Demonstrate which backend/class is selected for both automatic and explicit choices, and why incompatible configurations are rejected.
- **Kernel correctness:** Compare against the existing reference at representative token counts, top-k values, dtypes, and supported layouts. Include empty/padded cases where applicable.
- **Integration:** Exercise real layer construction, loading, routing, and execution. For user-visible behavior, use a model-level test through the real `LLM`/serving entry point, not only a direct kernel invocation.
- **Optional features:** Exercise LoRA, shared experts, DBO, CUDA graphs, EPLB, or Transformers-backed models when the change claims or affects that support. Reuse the corresponding suites, including [`tests/lora/`](../../tests/lora/), [`test_eplb_fused_moe_layer.py`](../../tests/distributed/test_eplb_fused_moe_layer.py), and [`tests/models/transformers/fusers/test_moe.py`](../../tests/models/transformers/fusers/test_moe.py).
- **Performance:** Benchmark representative workloads before changing automatic priorities. Record hardware, shapes, quantization, topology, baseline, and results. Use the existing [profiling instructions](../design/fused_moe_modular_kernel.md#how-to-profile) for traces, and put new kernel benchmarks under `benchmarks/kernels/`.
- **Documentation:** Update [kernel capabilities](../design/moe_kernel_features.md), deployment requirements, and any public backend/quantization options affected. Include exact test commands/results and relevant model evaluations in the contribution.

Install the optional backend dependencies and allocate the GPUs/interconnect required by the selected tests. The standalone modular compatibility command defaults to multiple ranks; set `--world-size` deliberately and use the [compatibility-testing instructions](../design/fused_moe_modular_kernel.md#how-to-check-fusedmoeprepareandfinalizemodular-fusedmoeexpertsmodular-compatibility) to select a specific registered pair. A model-loading test and a kernel-combination test serve different purposes; both may be needed for a new backend.
