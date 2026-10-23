# Specialized Decode Kernels

## Motivation

A **specialized decode kernel** replaces part or all of a model's decode step with a single hand-written launch. At the small batch sizes of low-latency serving, the regular path is dominated by launch overhead, so one persistent kernel can be much faster. The monokernel (FlyDSL, AMD) is one. Kimi K3's CuTeDSL executor is another. More are coming.

Each one used to bring its own glue into model files. `vllm/models/common/specialized_decode/` replaces that glue with one contract, so model code remains easy to review as the number of kernels grows.

An author writes the specialized kernel, a `DecodeSpec` saying under what conditions it can run, and a `DecodeImpl` dispatcher to connect the kernel to the model code. The following is then handled by the library:

- JIT warm-up registration
- the config and model checks
- per-forward choice between the kernel and vLLM's path
- TP rank agreement on all of it
- setting CUDA graph capture sizes
- and picking between several kernels for one model, for instance based on hardware architecture.

Key design principles:

- **The kernel owns its layer loop**: the library has no opinion about layers, so a kernel may replace one layer, two, or the whole backbone
- **Default deny**: an feature the spec does not list refuses the deployment at startup
- **Refusals carry reasons**: every check returns strings, which are logged and raised
- **Three touch points in the model**: construct, build, forward. Nothing in the decoder layer
- **No kernel code in the contract**: the library is plain Python and imports no vendor runtime, so it loads on any platform

## Quick Overview

### Declaring a Spec

A spec says when the kernel can run and how to build it. It holds no kernel code, so it is evaluated before anything vendor-specific loads:

```python
# vllm/models/deepseek_v41/amd/specialized_decode/spec.py
from vllm.models.common.specialized_decode.spec import DecodeSpec, Feature


def _refuse_config(vllm_config) -> list[str]:
    reasons = []
    if vllm_config.parallel_config.tensor_parallel_size not in (2, 4):
        reasons.append("needs tensor parallel size 2 or 4")
    return reasons


def _build(vllm_config, model, widths):
    from .runtime.decode import DSV41Kernel  # the kernel import stays lazy

    return DSV41Kernel(vllm_config, model, widths)


DECODE_SPEC = DecodeSpec["DSV41Kernel"](
    name="dsv41-cdna4",
    supports=frozenset({Feature.PIPELINE_PARALLEL, Feature.SPECULATIVE_DECODING}),
    widths=lambda cfg: tuple(range(1, 49)),
    refuse_config=_refuse_config,
    refuse_model=_refuse_model,
    refuse_caches=_refuse_caches,
    skip_step=_skip_step,
    build=_build,
)
```

| Field | Meaning |
| --- | --- |
| `name` | Log and metric name. |
| `supports` | Engine features this spec serves. Default deny. |
| `widths` | The step sizes a launch serves, read from the config. |
| `refuse_config` | Why the config cannot run it. Checked before load. |
| `refuse_model` | Why the loaded model cannot run it. Checked after load. |
| `refuse_caches` | Why the KV caches cannot be read. Checked when they are bound. |
| `skip_step` | Why *this* step should run vLLM's path instead. |
| `build` | Builds this rank's kernel. |
| `pad_steps` | Whether a short step launches at the next width up. Off by default. |
| `set_capture_sizes` | Whether the widths become the capture sizes. On by default. |

`widths` is a function of the config, so a spec can read the GPU. One kernel may take 48 rows on gfx950 but only 42 on gfx942, where the workgroup has less LDS.

### Implementing the Kernel

The kernel stands in for the entire model forward:

```python
class DecodeImpl(Protocol):
    cache_layers: tuple[str, ...]

    def run(self, step: Step, input_ids, inputs_embeds, **kwargs) -> Any: ...
    def bind_caches(self, caches: Mapping[str, Any]) -> None: ...
```

`run` takes and returns exactly what the model's own forward does. `step` carries the row count, the width to launch at, and the positions. There is deliberately no further constraint on how `run` is written: FlyDSL and CuTeDSL kernels are structured very differently, and forcing one shape would cost more than it saves.

A kernel that compiles lazily, from one key per launch, can also implement `DecodeWarmable`:

```python
@runtime_checkable
class DecodeWarmable(Protocol):
    def warmup_keys(self) -> list[Any]: ...
    def compile(self, key: Any) -> None: ...
```

The dispatcher then registers it with vLLM's JIT warm-up, so the first decode step does not pay the compile. This is optional. Kimi K3 compiles eagerly while it builds, implements neither method, and hence registration is skipped.

!!! note
    Warm-up is the one part of the contract with a placement requirement. vLLM's warm-up registry is only active while the model loads, so a kernel that wants automatic warm-up must be built in `process_weights_after_loading`. Building later is fine, it just means warming up eagerly.

### Wiring It Into the Model

Three touch points, all on the causal LM:

```python
class MyForCausalLM(nn.Module):
    def __init__(self, *, vllm_config, prefix=""):
        self.decode_spec = select_decode_spec(vllm_config, SPECS)
        self.specialized_decode: SpecializedDecode[MyKernel] | None = None

    def process_weights_after_loading(self):
        if self.decode_spec is not None:
            self.specialized_decode = SpecializedDecode(
                self.model, self.vllm_config, self.decode_spec
            )

    def forward(self, input_ids, positions, intermediate_tensors=None,
                inputs_embeds=None, **kwargs):
        decode = self.specialized_decode
        if decode is not None:
            step = decode.plan(positions, intermediate_tensors)
            if step is not None:
                return decode.run(step, input_ids, inputs_embeds, **kwargs)
        return self.model(input_ids, positions, intermediate_tensors,
                          inputs_embeds, **kwargs)
```

Plus one hook, so the kernel sees the KV caches once the runner has bound them:

```python
class MyModelState(DefaultModelState):
    def initialize_kv_cache(self, kv_cache_config, block_tables):
        super().initialize_kv_cache(kv_cache_config, block_tables)
        if self.model.specialized_decode is not None:
            self.model.specialized_decode.bind_caches()
```

Nothing else changes. In particular, the decoder layer does not.

!!! warning
    The model must not carry `@support_torch_compile`. A specialized decode replaces the layer loop, so a `torch.compile`'d model is refused with a message saying so.

### Enabling It

```bash
vllm serve deepseek-ai/DeepSeek-V4.1 --specialized-decode monokernel
vllm serve deepseek-ai/DeepSeek-V4.1 --specialized-decode   # auto: vLLM picks
vllm serve deepseek-ai/DeepSeek-V4.1 --monokernel           # shorthand, will be deprecated
```

## Lifecycle

```text
Model.__init__
    ↓
[select_decode_spec] → None if the flag is off
                     → the first spec whose refuse_config() is empty
                     → raises DecodeUnsupported if none fits
                     → optionally sets the CUDA graph capture sizes to the widths
    ↓
process_weights_after_loading
    ↓
[SpecializedDecode(...)] → refuse_model(), then agree across the TP group
                         → spec.build() → the kernel
                         → register warmup_keys()/compile() with the JIT warm-up
    ↓
ModelState.initialize_kv_cache          (runs twice: profiling cache, then the real one)
    ↓
[bind_caches] → refuse_caches(), if ok then kernel.bind_caches()
    ↓
forward, every step
    ↓
[plan] → None (run vLLM's path) if: unserved width,
         no forward context, piecewise CUDA graph, caches not bound,
         or skip_step() gave a reason
    ↓
[run] → kernel.run(step, input_ids, inputs_embeds, **kwargs)
```

## Core Concepts

### Refuse Versus Skip

**Refuse** means the deployment is wrong. It raises `DecodeUnsupported` and the engine does not start. Use it for anything a request cannot change. Example: the parallelism, the GPU, the quantization backend, the cache layout. A silent fallback here would hide a misconfiguration behind a quiet performance loss.

**Skip** means this one step is not for the kernel. `skip_step` returns a reason, `plan()` returns None, and the model's own forward runs. Use it for anything that varies step to step: a prefill in the batch, an unserved row count, missing metadata.

Every rank must reach the same verdict, or the ranks that took the kernel will wait on collectives the others never issue. The library handles this.

### Features

`supports` is default deny. The library works out which features the config turned on, subtracts the spec's set, and refuses whatever is left, naming it. `Feature` covers the parallelism modes, LoRA, KV connectors, EPLB, sleep mode, CPU weight offload, weight updates, routed expert capture and ubatching.

Two checks are enabled by default.

1. the engine must be on Model Runner V2
1. the CUDA graph mode is `FULL`

### Widths and Capture Sizes

A kernel is built for a fixed set of row counts, and a step of any other size cannot launch. There are two ways to handle that:

- **Exact (default).** `select_decode_spec` amends the CUDA graph capture sizes to the served widths, keeping any size larger than the widest launch. A step off the list falls back to vLLM's path.
- **Padded.** With `pad_steps=True` a short step launches at the next width up and the kernel owns the padding. Such a spec usually also sets `set_capture_sizes=False`, so the deployment keeps the capture sizes it asked for. A kernel built at `(1, 2, 4, 5, 6, 8)` then serves a batch of 3 on the size-4 launch.

### Cache Binding

`cache_layers` names the attention layers whose KV caches the kernel reads. `bind_caches()` looks each up in the static forward context, runs `spec.refuse_caches`, and only then hands them to the kernel, which therefore never sees a cache its spec has not accepted. The hook fires twice during startup, once for the profiling cache and once for the real one. `plan()` refuses to run before the first call, so a model that forgets the hook falls back to vLLM's path rather than reading unbound memory.

## Recipes

### Two Kernels for One Model

Write the second kernel, then add a second spec and list it in preference order:

```python
DECODE_SPEC_GFX942 = DecodeSpec["DSV41Gfx942Kernel"](
    name="dsv41-cdna3",
    supports=frozenset({Feature.PIPELINE_PARALLEL}),   # no spec decode here
    widths=lambda cfg: tuple(range(1, 17)),            # fewer rows than gfx950
    refuse_config=_refuse_config_gfx942,               # "needs a gfx942 GPU"
    refuse_model=_refuse_model,                        # shared with gfx950
    refuse_caches=_refuse_caches,
    skip_step=_skip_step,
    build=_build_gfx942,
)

SPECS = (DECODE_SPEC_GFX950, DECODE_SPEC_GFX942)
```

The model already passes `SPECS` to `select_decode_spec`, so nothing else changes. The dispatcher takes the first spec that accepts the config. Specializing on anything other than the architecture works the same way: a kernel that only handles batch size 1 sets `widths=lambda cfg: (1,)` and goes first in the list.

### A Kernel That Covers Part of a Layer

Sometimes we want to write a specialized kernel for part of a layer. For instance, DeepSeek-V4.1 has layers where only the FFN seam uses a specialized kernel but attention runs in normal vLLM code. The library needs no concept of this, because the kernel owns the layer loop and decides per layer:

```python
# in DecodeImpl.run:
for idx in range(self.embed_layer, self.layer_end):
    layer, plan = model.layers[idx], self.plans.get(...)
    ffn = out = None
    if plan.ffn_only:
        ffn = partial(self.launch_ffn, plan)      # a seam, not a layer
    else:
        out = self.launch_whole(plan, hidden, positions, ...)
    if out is None:
        # inject the specialized kernel into the vLLM execution path
        out = layer(hidden, positions, ..., decode_ffn=ffn)
```

The decoder layer can then decide where to use `decode_ffn`.

### A Kernel on a Speculator's Draft Head

A drafter needs its own spec, and additionally needs to plug into the target model's KV cache binding hook.

The draft's causal LM carries the same three touch points as the target:

```python
# vllm/models/deepseek_v41/amd/dspark.py
class DSparkDeepseekV4ForCausalLM(nn.Module):
    def __init__(self, *, vllm_config, prefix=""):
        self.specialized_decode_spec = select_decode_spec(
            vllm_config, DRAFT_DECODE_SPEC
        )
        self.specialized_decode: SpecializedDecode[DSV41DraftKernel] | None = None
        self.model = DSparkDeepseekV4Model(vllm_config=vllm_config, ...)

    def process_weights_after_loading(self):
        if self.specialized_decode_spec is not None and self.specialized_decode is None:
            self.specialized_decode = SpecializedDecode(
                self.model, self.vllm_config, self.specialized_decode_spec
            )
            # register the specialized drafter
            _DRAFT_DECODES.append(self.specialized_decode)

    def forward(self, input_ids, positions, inputs_embeds=None):
        if self.specialized_decode is not None:
            step = self.specialized_decode.plan(positions)
            if step is not None:
                return self.specialized_decode.run(step, input_ids, inputs_embeds)
        return self.model(input_ids, positions, inputs_embeds)
```

Since the drafter doesn't have its own `ModelState`, we leverage the target model's model state to bind the drafter's caches. We do this by amending the target's `initialize_kv_cache` to also bind the cache of any registered drafter:

```python
# vllm/models/deepseek_v41/amd/dspark.py
_DRAFT_DECODES: list[SpecializedDecode] = []


def bind_draft_decode_caches() -> None:
    for decode in _DRAFT_DECODES:
        decode.bind_caches()


# vllm/models/deepseek_v41/amd/model_state.py
class DeepseekV41ROCMModelState(DeepseekV41ModelState):
    def initialize_kv_cache(self, kv_cache_config, block_tables):
        super().initialize_kv_cache(kv_cache_config, block_tables)
        if self.model.specialized_decode is not None:
            self.model.specialized_decode.bind_caches()
        # NEW
        bind_draft_decode_caches()
```

## See Also

- [CUDA Graphs](cuda_graphs.md), the capture sizes a spec amends
- [Model Runner V2](model_runner_v2.md), the runner a specialized decode requires
