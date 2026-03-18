# Online Quantization Configuration Redesign

**Status:** Draft | **Created:** 2026-03-18

---

## User-Facing API

### Parameters

| Parameter | Type | Description |
|---|---|---|
| `quantization` | `str \| None` | Scheme string. Existing values unchanged. New: `"fp8_tensorwise"`, `"fp8_blockwise"`, `"int8_channelwise"` |
| `quantization_config` | `dict[str, Any] \| None` | Structured config passed as `**kwargs` to the config class constructor. Applies to online schemes; ignored for offline. Extensible to offline overrides in the future. |

### Usage Examples

```python
# Explicit online FP8, no extra config
LLM(model, quantization="fp8_blockwise")

# With structured config
LLM(model, quantization="fp8_blockwise", quantization_config={
    "ignore": ["lm_head"],
    "moe_scheme": "fp8_tensorwise",
    "backend": "cutlass",
})

# CLI
vllm serve model --quantization fp8_blockwise \
    --quantization-config '{"ignore": ["lm_head"]}'

# Existing behavior — unchanged
LLM(model, quantization="fp8")       # still works, online/offline auto-detected
LLM(model, quantization="gptq")      # offline, unchanged
```

---

## Class Hierarchy

```
QuantizationConfig (ABC)            # existing, unchanged
├── Fp8Config                       # existing offline/online hybrid, unchanged
├── GptqConfig, AwqConfig, ...      # existing offline configs, unchanged
└── OnlineQuantizationConfig (ABC)  # NEW base for online schemes
    ├── OnlineFp8TensorwiseConfig   # quantization="fp8_tensorwise"
    ├── OnlineFp8BlockwiseConfig    # quantization="fp8_blockwise"
    └── OnlineInt8ChannelwiseConfig # quantization="int8_channelwise"
```

---

## `OnlineQuantizationConfig` Base Class

```python
class OnlineQuantizationConfig(QuantizationConfig, ABC):
    is_online: bool = True                    # dispatch marker

    def __init__(
        self,
        ignore: list[str] | None = None,      # layers to skip
        linear_scheme: str | None = None,      # override for linear layers
        moe_scheme: str | None = None,         # override for MoE layers
        backend: str = "auto",                 # compute backend
    ): ...

    @staticmethod
    def get_config_filenames() -> list[str]:
        return []                              # never reads from checkpoint

    @classmethod
    def from_config(cls, config) -> Self:
        return cls()                           # constructed from user args, not checkpoint
```

### `ignore` Matching

- **Exact match** by default: `"lm_head"` matches only `lm_head`
- **Regex** via `"re:"` prefix: `"re:model\\.layers\\.[0-3]\\..*"` matches layers 0-3

### `moe_scheme` / `linear_scheme` Delegation

The config class interprets these strings internally and selects the appropriate method class directly. No cross-config registry lookups.

---

## Concrete Config Classes

One class per scheme. Each is a thin dispatch layer that delegates to existing method implementations.

```python
class OnlineFp8TensorwiseConfig(OnlineQuantizationConfig):
    def get_name(self) -> str: return "fp8_tensorwise"
    def get_min_capability(cls) -> int: return 89
    def get_quant_method(self, layer, prefix):
        if is_layer_skipped(prefix, self.ignore):
            return UnquantizedLinearMethod()
        if isinstance(layer, LinearBase):
            return Fp8OnlineLinearMethod(self)      # existing
        if isinstance(layer, FusedMoE):
            return Fp8OnlineMoEMethod(self, layer)  # existing
        ...

class OnlineFp8BlockwiseConfig(OnlineQuantizationConfig):
    def __init__(self, weight_block_size=None, **kwargs):
        super().__init__(**kwargs)
        self.weight_block_size = weight_block_size or [128, 128]
    def get_name(self) -> str: return "fp8_blockwise"
    ...

class OnlineInt8ChannelwiseConfig(OnlineQuantizationConfig):
    def get_name(self) -> str: return "int8_channelwise"
    ...
```

---

## Registry

Additive changes to `get_quantization_config()`:

```python
method_to_config = {
    # ... all existing entries unchanged ...
    "fp8_tensorwise": OnlineFp8TensorwiseConfig,
    "fp8_blockwise": OnlineFp8BlockwiseConfig,
    "int8_channelwise": OnlineInt8ChannelwiseConfig,
}
```

---

## Config Instantiation (`get_quant_config()`)

```python
def get_quant_config(model_config, load_config) -> QuantizationConfig:
    quant_cls = get_quantization_config(model_config.quantization)

    if getattr(quant_cls, 'is_online', False):
        user_cfg = model_config.quantization_config or {}
        return quant_cls(**user_cfg)        # <-- online path

    # existing offline path unchanged
    ...
```

---

## Validation: Online Scheme + Quantized Checkpoint

`FutureWarning` (not a hard error) when user requests an online scheme but the checkpoint already has quant metadata. Can be tightened to an error in a future release.

---

## Out-of-Tree Plugins

```python
@register_quantization_config("my_online_quant")
class MyConfig(OnlineQuantizationConfig):
    def __init__(self, my_param="default", **kwargs):
        super().__init__(**kwargs)
        self.my_param = my_param
    ...

# Works automatically — is_online dispatch handles it
LLM(model, quantization="my_online_quant",
    quantization_config={"my_param": "custom", "ignore": ["lm_head"]})
```

---

## Backward Compatibility

All existing behavior is unchanged. New method strings and `quantization_config` are purely additive.

The implicit online behavior of `quantization="fp8"` with a bf16 checkpoint is **not deprecated yet**. Deprecation warning may be added in a future release.

---

## File Changes

| File | Change |
|---|---|
| **New:** `quantization/online_config.py` | `OnlineQuantizationConfig` base class |
| **New:** `quantization/online_fp8.py` | `OnlineFp8TensorwiseConfig`, `OnlineFp8BlockwiseConfig` |
| **New:** `quantization/online_int8.py` | `OnlineInt8ChannelwiseConfig` |
| `quantization/__init__.py` | New scheme strings + registry mappings |
| `config/model.py` | `quantization_config` field, validation warning |
| `model_loader/weight_utils.py` | `is_online` early-return in `get_quant_config()` |
| `engine/arg_utils.py` | `quantization_config` parameter |
| `entrypoints/llm.py` | `quantization_config` parameter |

---

## Implementation Phases

1. **Foundation:** Base class, `OnlineFp8TensorwiseConfig`, `quantization_config` parameter wiring, `is_online` dispatch, validation warning
2. **Expand:** `OnlineFp8BlockwiseConfig`, `OnlineInt8ChannelwiseConfig`, `OnlineMxfp8Config`, `OnlineMxfp4Config`
3. **Advanced:** Per-layer scheme overrides, YAML config file support, deprecation of implicit `quantization="fp8"` online behavior
