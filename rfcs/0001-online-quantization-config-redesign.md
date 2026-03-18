# RFC: Online Quantization Configuration Redesign

**Authors:** vLLM Community
**Status:** Draft
**Created:** 2026-03-18

## 1. Summary

This RFC proposes a structured approach to online (post-training, dynamic) quantization configuration in vLLM. Today, online quantization is configured through the same `quantization` string parameter used for offline (checkpoint-embedded) quantization, with method-specific behavior determined implicitly by checkpoint contents. This makes it difficult to compose schemes, configure per-layer behavior, or understand what will actually happen at runtime.

We propose:
1. A new `OnlineQuantizationConfig` base class that explicitly models online quantization concerns (layer ignoring, per-layer-type scheme overrides, backend selection)
2. New explicit quantization method strings (`fp8_tensorwise`, `fp8_blockwise`, `int8_channelwise`, etc.) that unambiguously select online schemes
3. A `quantization_config` parameter on `LLM`/`EngineArgs` for structured configuration beyond what a string can express
4. Full backward compatibility with existing `quantization="fp8"` behavior and out-of-tree plugins

## 2. Motivation

### Current problems

**Implicit online vs offline detection.** When a user passes `quantization="fp8"`, vLLM determines whether to use online or offline quantization by inspecting the checkpoint's `hf_config.quantization_config`. If the checkpoint has FP8 metadata, offline mode is used; otherwise, online mode kicks in. This is confusing — users cannot tell from the API call alone what will happen.

**No structured configuration for online schemes.** Online quantization has knobs that offline does not: which layers to skip, whether to use different schemes for attention vs MoE vs dense layers, which compute backend to target. Today these are either hardcoded or require modifying checkpoint config files.

**Proliferating method strings.** The `QuantizationMethods` literal type already has 17+ entries, with more being added for each variant (`mxfp8`, `mxfp4`, `modelopt_fp4`, `modelopt_mxfp8`, `modelopt_mixed`, `petit_nvfp4`, etc.). Each new online scheme requires a new string, a new config class, and new wiring — even when the underlying quantization methods (`Fp8LinearMethod`, `Fp8MoEMethod`) are shared.

**Hard to compose.** A user who wants "FP8 blockwise for linear layers but FP8 tensorwise for MoE" cannot express this today. Each `quantization` string maps to a single config that applies uniformly.

### What this enables

- `LLM(model, quantization="fp8_blockwise")` — clear, unambiguous online FP8 with block scaling
- `LLM(model, quantization="fp8_blockwise", quantization_config={"ignore": ["lm_head"], "moe_scheme": "fp8_tensorwise"})` — skip the output head, use tensorwise for MoE. Internally mapped to `OnlineFp8BlockwiseConfig(**kwargs)` when `is_online` is true.
- Out-of-tree plugins can subclass `OnlineQuantizationConfig` and register via `@register_quantization_config` with the same structured knobs
- Existing `quantization="fp8"` behavior is unchanged

## 3. Design

### 3.1 New base class: `OnlineQuantizationConfig`

```python
# vllm/model_executor/layers/quantization/online_config.py

class OnlineQuantizationConfig(QuantizationConfig, ABC):
    """Base class for online (dynamic, post-training) quantization configs.

    Online configs quantize unquantized (bf16/fp16/fp32) model weights at
    load time rather than reading pre-quantized weights from a checkpoint.
    """

    # Class-level marker for dispatch in get_quant_config()
    is_online: bool = True

    def __init__(
        self,
        ignore: list[str] | None = None,
        linear_scheme: str | None = None,
        moe_scheme: str | None = None,
        backend: str = "auto",
    ):
        super().__init__()
        self.ignore = ignore or []
        self.linear_scheme = linear_scheme
        self.moe_scheme = moe_scheme
        self.backend = backend

    @staticmethod
    def get_config_filenames() -> list[str]:
        """Online configs do not read from checkpoint config files."""
        return []

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "OnlineQuantizationConfig":
        """Online configs are constructed from user args, not checkpoint metadata.

        This implementation exists to satisfy the abstract interface. If called
        (e.g., by a checkpoint that happens to have matching quant_method), it
        extracts what it can from the checkpoint config but prefers user-supplied
        values.
        """
        return cls()  # Subclasses may override with richer extraction
```

**Key design decisions:**

- `is_online = True` class attribute enables duck-type checking in `get_quant_config()` without maintaining a hardcoded list. Out-of-tree plugins that subclass `OnlineQuantizationConfig` get this for free.
- `get_config_filenames()` returns `[]` — online configs never read from checkpoint files.
- `from_config()` has a default implementation instead of being left abstract. Online configs are primarily constructed from user arguments, not checkpoint metadata. Subclasses can override if they need to extract partial information from checkpoints.
- `ignore` uses exact string match by default, with opt-in regex via `"re:"` prefix (e.g., `"lm_head"` for exact match, `"re:model\\.layers\\.[0-3]\\..*"` for regex).
- `linear_scheme` and `moe_scheme` allow per-layer-type overrides. When `None`, the default scheme of the config is used for all layer types.
- `backend` selects the compute backend (`"auto"`, `"vllm"`, `"quark"`, `"cutlass"`, etc.).

### 3.2 Concrete online config classes

Each online scheme gets a concrete config class. Initially:

```python
class OnlineFp8TensorwiseConfig(OnlineQuantizationConfig):
    """Online FP8 quantization with per-tensor scaling."""

    def get_name(self) -> str:
        return "fp8_tensorwise"

    def get_supported_act_dtypes(self) -> list[torch.dtype]:
        return [torch.bfloat16, torch.float16]

    @classmethod
    def get_min_capability(cls) -> int:
        return 89  # Ada Lovelace+

    def get_quant_method(self, layer, prefix):
        if is_layer_skipped(prefix, self.ignore):
            return UnquantizedLinearMethod()
        if isinstance(layer, LinearBase):
            return Fp8OnlineLinearMethod(self)  # existing class
        if isinstance(layer, FusedMoE):
            return Fp8OnlineMoEMethod(self, layer)
        if isinstance(layer, Attention):
            return Fp8KVCacheMethod(self)
        return None


class OnlineFp8BlockwiseConfig(OnlineQuantizationConfig):
    """Online FP8 quantization with block-wise scaling."""

    def __init__(self, weight_block_size=None, **kwargs):
        super().__init__(**kwargs)
        self.weight_block_size = weight_block_size or [128, 128]

    def get_name(self) -> str:
        return "fp8_blockwise"

    # ... similar structure, returns blockwise-capable methods ...


class OnlineInt8ChannelwiseConfig(OnlineQuantizationConfig):
    """Online INT8 quantization with per-channel scaling."""

    def get_name(self) -> str:
        return "int8_channelwise"
    # ...
```

These classes **delegate to existing quantization method implementations** (`Fp8OnlineLinearMethod`, `Fp8OnlineMoEMethod`, etc.) rather than duplicating logic. The config class is a thin dispatch layer.

### 3.3 New method strings and registry mapping

New entries in `QuantizationMethods`:

```python
QuantizationMethods = Literal[
    # ... existing entries ...
    "fp8_tensorwise",    # NEW: explicit online FP8 tensorwise
    "fp8_blockwise",     # NEW: explicit online FP8 blockwise
    "int8_channelwise",  # NEW: explicit online INT8 channelwise
    # Future: "nvfp4", "mxfp8_online", "mxfp4_online", etc.
]
```

Registry mapping in `get_quantization_config()`:

```python
method_to_config: dict[str, type[QuantizationConfig]] = {
    # ... existing entries unchanged ...
    "fp8_tensorwise": OnlineFp8TensorwiseConfig,
    "fp8_blockwise": OnlineFp8BlockwiseConfig,
    "int8_channelwise": OnlineInt8ChannelwiseConfig,
}
```

### 3.4 `quantization_config` parameter

A new optional parameter on `LLM` and `EngineArgs`:

```python
class LLM:
    def __init__(
        self,
        model: str,
        *,
        quantization: QuantizationMethods | None = None,
        quantization_config: dict[str, Any] | None = None,  # NEW
        ...
    ):
```

This dict is passed as `**kwargs` to the `OnlineQuantizationConfig` subclass constructor:

```python
# In VllmConfig._get_quantization_config():
quant_cls = get_quantization_config(model_config.quantization)

if getattr(quant_cls, 'is_online', False):
    user_cfg = model_config.quantization_config or {}
    return quant_cls(**user_cfg)
else:
    # Existing offline path unchanged
    ...
```

**CLI equivalent:**

```bash
vllm serve model --quantization fp8_blockwise \
    --quantization-config '{"ignore": ["lm_head"], "backend": "cutlass"}'
```

### 3.5 Changes to `_verify_quantization()`

The validation in `ModelConfig._verify_quantization()` needs a soft check for online-vs-checkpoint conflicts:

```python
def _verify_quantization(self) -> None:
    # ... existing validation ...

    # If user specified an online scheme but checkpoint has quant metadata,
    # warn rather than error — the user may know what they're doing, and
    # out-of-tree plugins may have custom quant_method strings we don't
    # recognize.
    if self.quantization is not None:
        quant_cls = get_quantization_config(self.quantization)
        hf_quant_cfg = getattr(self.hf_config, "quantization_config", None)

        if getattr(quant_cls, 'is_online', False) and hf_quant_cfg is not None:
            checkpoint_method = hf_quant_cfg.get("quant_method", "unknown")
            import warnings
            warnings.warn(
                f"Online quantization scheme '{self.quantization}' was "
                f"requested, but the checkpoint already has quantization "
                f"metadata (quant_method='{checkpoint_method}'). The "
                f"checkpoint's pre-quantized weights will be ignored and "
                f"weights will be re-quantized online. If you intended to "
                f"use pre-quantized weights, set "
                f"quantization='{checkpoint_method}' instead.",
                FutureWarning,
                stacklevel=2,
            )
```

This is a **deprecation-style warning**, not a hard error. Rationale:
- Out-of-tree plugins may define custom `quant_method` strings that we don't recognize
- Some checkpoints may have partial quant metadata (e.g., only KV cache quantized)
- Users experimenting with re-quantization should get informed, not blocked
- Can be tightened to an error in a future release if the warning proves insufficient

### 3.6 Changes to `get_quant_config()` in `weight_utils.py`

The core config instantiation function gets an early-return branch for online schemes:

```python
def get_quant_config(
    model_config: ModelConfig, load_config: LoadConfig
) -> QuantizationConfig:
    quant_cls = get_quantization_config(model_config.quantization)

    # Online schemes: construct from user config, not checkpoint
    if getattr(quant_cls, 'is_online', False):
        user_cfg = getattr(model_config, 'quantization_config', None) or {}
        return quant_cls(**user_cfg)

    # Existing offline path — unchanged
    hf_quant_config = getattr(model_config.hf_config, "quantization_config", None)
    if hf_quant_config is not None:
        return quant_cls.from_config(hf_quant_config)
    # ... rest of existing logic ...
```

### 3.7 Interaction with out-of-tree plugins

Out-of-tree plugins work exactly as before, plus they gain the ability to use `OnlineQuantizationConfig`:

```python
from vllm.model_executor.layers.quantization import register_quantization_config
from vllm.model_executor.layers.quantization.online_config import OnlineQuantizationConfig

@register_quantization_config("my_online_quant")
class MyOnlineQuantConfig(OnlineQuantizationConfig):
    """Custom online quantization scheme."""

    def __init__(self, my_param: str = "default", **kwargs):
        super().__init__(**kwargs)
        self.my_param = my_param

    def get_name(self) -> str:
        return "my_online_quant"

    def get_quant_method(self, layer, prefix):
        if is_layer_skipped(prefix, self.ignore):  # inherited
            return UnquantizedLinearMethod()
        # ... custom logic ...

# Usage:
llm = LLM(
    model="my-bf16-model",
    quantization="my_online_quant",
    quantization_config={"my_param": "custom", "ignore": ["lm_head"]},
)
```

The `is_online` class attribute on `OnlineQuantizationConfig` means the `get_quant_config()` dispatch works automatically — no need to modify core vLLM code for each new plugin.

## 4. Backward Compatibility

### What doesn't change

| Scenario | Behavior |
|---|---|
| `quantization="fp8"` with FP8 checkpoint | Offline path via `Fp8Config`, identical to today |
| `quantization="fp8"` with bf16 checkpoint | Online path via `Fp8Config(is_checkpoint_fp8_serialized=False)`, identical to today |
| `quantization="gptq"`, `"awq"`, etc. | Completely unchanged |
| `@register_quantization_config` plugins | Work as before; `QuantizationConfig` base class unchanged |
| No `quantization` arg | Auto-detection from checkpoint unchanged |

### What's new

| Scenario | Behavior |
|---|---|
| `quantization="fp8_tensorwise"` | Explicit online FP8 tensorwise via `OnlineFp8TensorwiseConfig` |
| `quantization="fp8_blockwise"` | Explicit online FP8 blockwise via `OnlineFp8BlockwiseConfig` |
| `quantization_config={...}` | Structured config passed to online config constructor |
| Online scheme + checkpoint with quant metadata | `FutureWarning` (not an error) |

### Deprecation path

The implicit online behavior of `quantization="fp8"` (when the checkpoint is bf16) is not deprecated immediately. Over time, we may add a deprecation warning suggesting users switch to `quantization="fp8_tensorwise"` for clarity. Timeline TBD based on adoption.

## 5. File Changes Summary

| File | Change | Risk |
|---|---|---|
| `quantization/__init__.py` | Add new scheme strings to `QuantizationMethods`, add mappings in `get_quantization_config()` | Low — additive |
| **New:** `quantization/online_config.py` | `OnlineQuantizationConfig` base class | Low — new file |
| **New:** `quantization/online_fp8.py` | `OnlineFp8TensorwiseConfig`, `OnlineFp8BlockwiseConfig` | Low — new file, delegates to existing methods |
| **New:** `quantization/online_int8.py` | `OnlineInt8ChannelwiseConfig` | Low — new file |
| `config/model.py` | Add `quantization_config` field, update `_verify_quantization()` | Medium — validation path |
| `model_loader/weight_utils.py` | Add early-return branch in `get_quant_config()` for `is_online` | Medium — critical path |
| `engine/arg_utils.py` | Add `quantization_config` parameter | Low — additive |
| `entrypoints/llm.py` | Add `quantization_config` parameter | Low — additive |
| Existing quant method classes | No changes initially | None |

## 6. Open Questions

### 6.1 One class per scheme vs parameterized base — **DECIDED: Option A**

One concrete class per scheme (`OnlineFp8TensorwiseConfig`, `OnlineFp8BlockwiseConfig`, etc.). Each maps to a unique `quantization` string.

Rationale:
- It integrates cleanly with the existing string-based registry (`get_quantization_config("fp8_blockwise")` returns a class)
- Each scheme can have scheme-specific parameters (e.g., `weight_block_size` only on blockwise)
- Plugin authors can subclass a specific scheme rather than the generic base
- The `quantization` string remains the primary dispatch key, consistent with the rest of vLLM

### 6.2 Should `moe_scheme` / `linear_scheme` overrides reference other registered schemes? — **DECIDED: Option B**

The base `OnlineQuantizationConfig` class should delegate internally — interpret the string and select the right method class directly. Cross-config delegation adds complexity and circular dependency risks. The config class knows which method classes exist and can select directly.

### 6.3 How should `ignore` patterns work? — **DECIDED: Exact match + regex via prefix**

`ignore` patterns use **exact string match** by default. To use regex, prefix the pattern with `"re:"`.

Examples:
- `"lm_head"` — exact match against the layer name
- `"re:model\\.layers\\.[0-3]\\..*"` — regex match for layers 0-3

Rationale: Exact match covers the common case simply. Regex (opt-in via `"re:"` prefix) provides full expressiveness when needed, without the ambiguity of glob patterns.

### 6.4 Should online configs support `from_config()` for hybrid scenarios?

Some future scenario: a checkpoint has FP8 tensorwise weights but the user wants to re-quantize to FP8 blockwise. Should `OnlineFp8BlockwiseConfig.from_config()` be able to read the existing checkpoint's quant config and use it as a starting point?

For now, no — the default `from_config()` returns a bare `cls()` instance. We can add richer `from_config()` implementations later if hybrid scenarios emerge. The deprecation warning (Section 3.5) covers the immediate need.

## 7. Alternatives Considered

### 7.1 Extending `Fp8Config` with more flags

Instead of new classes, add `online_mode`, `scaling_type`, etc. to the existing `Fp8Config`.

**Rejected because:** `Fp8Config` already has a complex `is_checkpoint_fp8_serialized` / `activation_scheme` / `weight_block_size` matrix. Adding more flags increases the combinatorial space and makes it harder to validate. Separate classes with clear names are easier to reason about.

### 7.2 A completely separate `--online-quantization` CLI flag

Instead of extending the `quantization` string namespace, introduce a parallel `--online-quantization` argument.

**Rejected because:** It fragments the user experience — users would need to know whether a method is "online" or "offline" before choosing which flag to use. Having all methods in a single `quantization` namespace (with clear naming like `fp8_blockwise`) is simpler.

### 7.3 YAML/TOML config file for quantization

Instead of `quantization_config` as a dict, require a config file path.

**Deferred, not rejected.** A config file makes sense for complex multi-layer quantization recipes. But the dict approach works for the common cases and can be extended to accept a file path later (e.g., `quantization_config="path/to/quant.yaml"`).

## 8. Implementation Plan

### Phase 1: Foundation (this RFC)
- Add `OnlineQuantizationConfig` base class
- Add `OnlineFp8TensorwiseConfig` as the first concrete implementation
- Wire up `quantization_config` parameter through `LLM` → `EngineArgs` → `ModelConfig` → `VllmConfig`
- Add `is_online` dispatch branch in `get_quant_config()`
- Add deprecation warning in `_verify_quantization()`
- Tests: unit tests for config creation, integration test with a small bf16 model

### Phase 2: Expand schemes
- `OnlineFp8BlockwiseConfig` — requires lifting the `weight_block_size` restriction in `Fp8OnlineLinearMethod` or creating a new blockwise online method
- `OnlineInt8ChannelwiseConfig`
- `OnlineMxfp8Config`, `OnlineMxfp4Config`

### Phase 3: Advanced features
- Per-layer scheme overrides (`linear_scheme`, `moe_scheme`)
- Config file support (`quantization_config="path/to/recipe.yaml"`)
- Deprecation warning for implicit online behavior of `quantization="fp8"`
