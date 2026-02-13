# Adding a Backend

This guide explains how to add a new computational backend to vLLM. Backends are implementations of core operations (MoE, attention, quantization) optimized for specific hardware or use cases.

## Overview

vLLM uses an **oracle pattern** for backend selection: a centralized module decides which backend to use based on hardware, configuration, and environment variables. This keeps selection logic in one place rather than scattered throughout the codebase.

Backend code lives in:

- `vllm/model_executor/layers/fused_moe/` — MoE backends
- `vllm/v1/attention/backends/` — Attention backends
- `vllm/model_executor/layers/quantization/` — Quantization backends

Attention backend selection lives in `vllm/v1/attention/selector.py`.

## When to Add a New Backend

Add a new backend when you have:

- A kernel optimized for specific hardware (e.g., Hopper GPUs)
- An alternative algorithm with different performance characteristics
- Hardware-specific optimizations (ROCm, CPU, TPU)

Do **not** add a new backend for:

- Minor parameter tweaks to existing backends
- Model-specific logic (belongs in model code)

## MoE Backends

Unquantized MoE backends are selected via the oracle at
`vllm/model_executor/layers/fused_moe/oracle/unquantized.py`. Quantized MoE
backends use `vllm/model_executor/layers/fused_moe/oracle/fp8.py` and
`vllm/model_executor/layers/fused_moe/oracle/nvfp4.py`.

If your backend does not fit MoE/attention/quantization, follow the same pattern: find the central selection/dispatch point, add any layout conversion, wire kernel creation, add tests, and register env vars as needed.

### Step 1: Add to the Backend Enum

```python
# oracle/unquantized.py

class UnquantizedMoeBackend(Enum):
    FLASHINFER_CUTLASS = "FlashInfer CUTLASS"
    SONIC = "Sonic MoE"  # Add your backend
    AITER = "ROCm AITER"
    TRITON = "TRITON"
    # ...
```

### Step 2: Implement Selection Logic

Update `select_unquantized_moe_backend()` to select your backend when
appropriate:

```python
def select_unquantized_moe_backend(
    use_ep: bool,
    use_dp: bool,
    is_act_and_mul: bool,
    has_bias: bool,
) -> UnquantizedMoeBackend:
    # Add selection logic for your backend.
    if envs.VLLM_USE_SONIC_MOE:
        from vllm.model_executor.layers.fused_moe.sonic_moe import (
            is_sonic_moe_supported,
        )

        if (
            is_sonic_moe_supported()
            and is_act_and_mul
            and not has_bias
            and not use_ep
        ):
            return UnquantizedMoeBackend.SONIC
    # ... existing logic
```

Consider:

- Hardware requirements (GPU architecture, platform)
- Configuration constraints (expert parallelism, data parallelism)
- External package availability (use a support check and lazy import)
- Environment variables for opt-in/opt-out

### Step 3: Implement Weight Conversion

If your backend requires a different weight layout, add conversion logic to
`convert_to_unquantized_kernel_format()`:

```python
def convert_to_unquantized_kernel_format(
    unquantized_backend: UnquantizedMoeBackend,
    layer: Module,
    w13_weight: torch.Tensor | None = None,
    w2_weight: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    # ...
    elif unquantized_backend == UnquantizedMoeBackend.SONIC:
        from vllm.model_executor.layers.fused_moe.sonic_moe import (
            permute_weights_for_sonic,
        )
        w13_weight = permute_weights_for_sonic(layer.w13_weight.data)

    return w13_weight, w2_weight
```

### Step 4: Implement Kernel Creation

Add kernel instantiation to `make_unquantized_moe_kernel()`:

```python
def make_unquantized_moe_kernel(
    layer: torch.nn.Module,
    backend: UnquantizedMoeBackend,
    quant_config: FusedMoEQuantConfig,
    moe_config: FusedMoEConfig,
) -> tuple[mk.FusedMoEModularKernel | None, bool]:
    # ...
    elif backend == UnquantizedMoeBackend.SONIC:
        from vllm.model_executor.layers.fused_moe.sonic_moe import (
            SonicMoeExperts,
        )
        kernel = mk.FusedMoEModularKernel(
            MoEPrepareAndFinalizeNoEP(),
            SonicMoeExperts(
                out_dtype=layer.params_dtype,
                quant_config=quant_config,
                weights_prepermuted=True,
            ),
        )
        use_inplace = False

    return kernel, use_inplace
```

### Step 5: Implement the Experts Class

Create your experts class following the `FusedMoEPermuteExpertsUnpermute`
interface:

```python
# vllm/model_executor/layers/fused_moe/your_backend.py

class YourExperts(mk.FusedMoEPermuteExpertsUnpermute):
    def __init__(
        self,
        out_dtype: torch.dtype,
        quant_config: FusedMoEQuantConfig = FUSED_MOE_UNQUANTIZED_CONFIG,
        **kwargs,
    ):
        super().__init__(quant_config)
        self.out_dtype = out_dtype

    def workspace_shapes(
        self,
        M,
        N,
        K,
        topk,
        global_num_experts,
        local_num_experts,
        expert_tokens_meta,
        activation,
    ) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
        # Return (workspace1_shape, workspace2_shape, output_shape)
        ...

    def apply(
        self,
        output: torch.Tensor,
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        activation: str,
        global_num_experts: int,
        expert_map: torch.Tensor | None,
        a1q_scale: torch.Tensor | None,
        a2_scale: torch.Tensor | None,
        workspace13: torch.Tensor,
        workspace2: torch.Tensor,
        expert_tokens_meta: mk.ExpertTokensMetadata | None,
        apply_router_weight_on_input: bool,
    ) -> None:
        # Run your kernel
        ...
```

See `vllm/model_executor/layers/fused_moe/modular_kernel.py` for the full
interface and existing implementations like `TritonExperts` or
`FlashInferExperts` for reference.

### Step 6: Export Public APIs

Add public symbols to `vllm/model_executor/layers/fused_moe/__init__.py` and
register them in `__all__`. If the backend depends on Triton, append the
symbols under the `if HAS_TRITON:` block to avoid import errors.

```python
from vllm.model_executor.layers.fused_moe.your_backend import (
    YourExperts,
    is_your_backend_supported,
)

if HAS_TRITON:
    __all__ += [
        "YourExperts",
        "is_your_backend_supported",
    ]
```

## Testing

### Unit Tests

Create `tests/kernels/moe/test_your_backend.py`:

```python
def test_is_supported():
    """Test platform detection."""
    result = is_your_backend_supported()
    assert isinstance(result, bool)

def test_weight_conversion():
    """Test weight format conversion."""
    # ...

@pytest.mark.skipif(not is_your_backend_supported(), reason="Backend not supported")
def test_vs_reference():
    """Compare against reference implementation."""
    # Run your backend
    out_yours = your_kernel(...)

    # Run reference (e.g., Triton)
    out_reference = triton_kernel(...)

    # Compare
    diff = calc_diff(out_yours, out_reference)
    assert diff < 0.01  # <1% difference
```

### CI Configuration

Add a CI job in `.buildkite/test_areas/kernels.yaml`:

```yaml
- label: Kernels Your Backend Test (H100)
  timeout_in_minutes: 10
  gpu: h100
  num_gpus: 1
  source_file_dependencies:
  - vllm/model_executor/layers/fused_moe/your_backend.py
  - tests/kernels/moe/test_your_backend.py
  commands:
    - pytest -v -s kernels/moe/test_your_backend.py
```

## Environment Variables

Register any new environment variables in `vllm/envs.py` by adding them to the
`environment_variables` mapping (and, if desired, add a type hint under
`TYPE_CHECKING` for docs tooling):

```python
environment_variables: dict[str, Callable[[], Any]] = {
    # ...
    "VLLM_USE_YOUR_BACKEND": lambda: bool(
        int(os.getenv("VLLM_USE_YOUR_BACKEND", "0"))
    ),
}
```

## Checklist

- [ ] Backend enum added to oracle
- [ ] Selection logic implemented
- [ ] Weight conversion (if needed)
- [ ] Kernel creation function updated
- [ ] Experts class implements full interface
- [ ] Public APIs exported
- [ ] Unit tests with reference comparison
- [ ] CI configuration added
- [ ] Environment variable registered
- [ ] Pre-commit checks pass
