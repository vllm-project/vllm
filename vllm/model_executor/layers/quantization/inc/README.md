# INC Quantization Support

## High-Level Architecture and Call Flow

```bash
                        ┌─────────────────────────────┐
                        │      HuggingFace Config      │
                        │   (quantization_config.json)  │
                        └──────────────┬──────────────┘
                                       │ parsed by
                                       ▼
┌──────────────────────────────────────────────────────────────────────┐
│                            INCConfig                                 │
│                                                                      │
│  get_quant_method(layer, prefix):                                    │
│    1. resolver.resolve(layer, prefix) → INCLayerConfig               │
│    2. resolve_scheme(layer_config)    → INCScheme                    │
│    3. isinstance dispatch:                                           │
│         LinearBase → scheme.get_linear_method() → INCLinearMethod    │
│         FusedMoE   → scheme.get_moe_method()   → concrete MoE method│
│         Attention  → scheme.get_kvcache_method() → (future)          │
└───────────────────────────────┬──────────────────────────────────────┘
                                │
                      resolve_scheme(layer_config)
                                │
                  ┌─────────────┼──────────────┐
                  ▼             ▼              ▼
           INCWna16Scheme  INCFp8Scheme  INCMxfp8Scheme
           (current)       (future)      (future)
                  │
       ┌──────────┼──────────┐
       ▼          ▼          ▼
  get_linear   get_moe    get_kvcache
  _method()    _method()  _method()
       │          │          │
       ▼          ▼          ▼
  INCLinear   concrete    (future)
  Method      MoE method
  (adapter)   (no wrapper)
       │          │
       ▼          ▼
  INCLinear   GPTQMarlin
  Scheme      MoEMethod
  (impl)      (direct)
```

### Dispatch flow (same 3 steps for all layer types)

```bash
resolve config → resolve scheme → scheme.get_xxx_method()
```

### Why Linear has an adapter but MoE doesn't

- **Linear**: `INCLinearScheme` defines a different lifecycle API (`create_weights`/`process_weights_after_loading`/`apply_weights`) than `LinearMethodBase`, so `INCLinearMethod` bridges them.
- **MoE**: Concrete methods already implement `FusedMoEMethodBase` directly. Wrapping would break class identity checks in vLLM's MoE layer.

## Package layout

```bash
inc/
├── inc.py              # INCConfig (entry point + dispatch)
├── resolver.py         # INCLayerConfig + INCConfigResolver
├── inc_linear.py       # INCLinearMethod
└── schemes/
    ├── base.py         # INCScheme ABC + INCLinearScheme ABC
    ├── factory.py      # resolve_scheme() — ordered list
    ├── wna16.py        # INCWna16Scheme (orchestrator)
    ├── wna16_linear.py # INCWNA16LinearScheme
    └── xpu_w4a16_linear.py
```

## Adding a new scheme

### Linear only (e.g., FP8)

```python
# schemes/fp8.py
class INCFp8Scheme(INCScheme):
    @staticmethod
    def can_handle(layer_config):
        return layer_config.is_fp8

    def get_linear_method(self, config, layer, prefix, layer_config):
        return INCLinearMethod(INCFp8LinearScheme(layer_config))

    # get_moe_method — inherits default NotImplementedError


class INCFp8LinearScheme(INCLinearScheme):
    def __init__(self, layer_config):
        self.kernel = init_fp8_linear_kernel()  # vLLM picks best kernel per platform

    def get_min_capability(cls) -> int:
        return 89

    def create_weights(self, layer, ...):
        # Register FP8 weight + per-channel scale
        layer.register_parameter("weight", ...)
        layer.register_parameter("weight_scale", ...)

    def process_weights_after_loading(self, layer):
        self.kernel.process_weights_after_loading(layer)

    def apply_weights(self, layer, x, bias=None):
        return self.kernel.apply_weights(layer, x, bias)
```

### Adding MoE support

```python
# schemes/fp8.py — add to existing scheme
class INCFp8Scheme(INCScheme):
    ...
    def get_moe_method(self, config, layer, prefix, layer_config):
        return INCFp8MoEMethod(layer.moe_config)


# schemes/fp8_moe.py — new file, implements FusedMoEMethodBase directly
class INCFp8MoEMethod(FusedMoEMethodBase):
    """Only 2 abstract methods to implement."""

    def __init__(self, moe_config):
        super().__init__(moe_config)

    def create_weights(self, layer, num_experts, hidden_size,
                       intermediate_size_per_partition, params_dtype, **kw):
        # Register FP8 MoE weights + scales
        ...

    def get_fused_moe_quant_config(self, layer):
        return FusedMoEQuantConfig(...)

    # apply() has a working default — override only if needed
```

### Register (one line)

```python
# schemes/factory.py
_SCHEME_LIST.extend([
    INCFp8Scheme,      # ← add
    INCWna16Scheme,
])
```

**That's it.** Two files touched: one new scheme file + one line in factory.
