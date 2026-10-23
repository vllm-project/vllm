# Specialized Decode Kernels

A *specialized decode kernel* replaces part or all of a model's decode step with a
single hand-written launch, which pays off at the small batch sizes of low-latency
serving. The monokernel (FlyDSL, persistent, AMD) is one such kernel; Kimi K3's
CuTeDSL executor is another. This document describes the shared contract they are
written against, which lives in `vllm/models/common/specialized_decode/`.

The contract exists to (1) improve standardization and (2) limit the lines changed in
model files as more specialized decode kernels are added. The kernels themselves will
naturally remain difficult to review thoroughly, but model code should stay easy to
review.

## Goal

**An author of a specialized decode kernel should write:**

1. A FlyDSL/CuTeDSL implementation and its utilities, in `vllm/models/<model>/<vendor>/`
2. The conditions under which it can run
3. Minimal glue connecting it to the model code

**The author should not have to write:**

1. Another new way to glue the kernel into the model code
2. JIT/warm-up registration
3. Basic config checks during start up, re-used by many specialized kernels anyway
   (parallelism, spec decode, graph mode)
4. Checks of whether to run the specialized kernel or regular vLLM code at every step
5. TP rank agreement of the above
6. Updating of CUDA graph capture sizes
7. Kernel selection when one model has more than one specialized kernel

The shared library implements the latter points.

## How

Declare when the specialized kernel can run:

```python
# refuse causes crash, skip causes fallback to regular vllm execution

@dataclass(frozen=True)
class DecodeSpec(Generic[KernelT]):
    name: str
    supports: frozenset[Feature]
    widths: Callable[[VllmConfig], tuple[int, ...]]
    refuse_config: Callable[[VllmConfig], list[str]]
    refuse_model: Callable[[Any], list[str]]
    refuse_caches: Callable[[VllmConfig, KernelT, Mapping[str, Any]], list[str]]
    skip_step: Callable[[VllmConfig, KernelT, int], str | None]
    build: Callable[[VllmConfig, Any, tuple[int, ...]], KernelT]
    pad_steps: bool = False
    set_capture_sizes: bool = True
```

The two flags cover the two ways a kernel can relate its built widths to the engine's
batch sizes. The default is exact: the served widths become the CUDA graph capture
sizes, and a step of any other row count falls back. With `pad_steps=True` a short step
instead launches at the next width up, which is what `Step.kernel_width` carries, and
such a spec usually also sets `set_capture_sizes=False` so that the deployment keeps the
sizes it asked for. GLM-5.2 wants the second shape: built at `(1, 2, 4, 5, 6, 8)`, it
serves a batch of 3 on the size-4 launch rather than refusing it.

`KernelT` is a stand-in for the entire model forward, implementing

```python
class DecodeImpl(Protocol):
    cache_layers: tuple[str, ...]

    def run(self, step: Step, input_ids, inputs_embeds, **kwargs) -> Any: ...
    def bind_caches(self, caches: Mapping[str, Any]) -> None: ...
```

A kernel that compiles lazily, from a key per launch, can additionally implement

```python
@runtime_checkable
class DecodeWarmable(Protocol):
    def warmup_keys(self) -> list[Any]: ...
    def compile(self, key: Any) -> None: ...
```

and the dispatcher registers it with vLLM's JIT warm-up, so the first decode step does
not pay the compile. This is optional on purpose. A kernel that compiles eagerly while
it is built, as Kimi K3 does in its provider constructors and again in `bind_caches`,
implements neither method and the dispatcher skips registration. It is also the only
part of the contract with a placement requirement: vLLM's warm-up registry is active
only while the model loads, so a kernel that wants automatic warm-up has to be built in
`process_weights_after_loading`. Building later (Kimi K3 builds in its `ModelState`,
because EAGLE3 aux layers are configured after the loader returns) is fine, it just
means warming up eagerly instead.

Once the kernel author has written their CuTeDSL/FlyDSL kernel, the two remaining points

> (2) The conditions under which it can run
>
> (3) Minimal glue connecting it to the model code

are implemented through a `DecodeSpec` and by wiring the kernel code through a
`DecodeImpl`. The remainder is taken care of by the shared `SpecializedDecode` utility.

!!! note
    There is deliberately no further constraint on how one implements `run`. There is a
    big benefit in letting each model decide exactly how to structure this. We have
    already seen many different ways of structuring the underlying FlyDSL/CuTeDSL
    kernels.

### Model-side code

The model-side code looks like this:

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

plus one hook to prepare the KV cache

```python
class MyModelState(DefaultModelState):
    def initialize_kv_cache(self, kv_cache_config, block_tables):
        super().initialize_kv_cache(kv_cache_config, block_tables)
        if self.model.specialized_decode is not None:
            self.model.specialized_decode.bind_caches()
```

On the model side nothing else needs to be touched.

### Directory layout

"Specialized decode" is the contract. "Monokernel" is one implementation of it (FlyDSL,
persistent kernel, AMD). Only the contract and the host-side glue around it are shared:

| Path | Holds |
| --- | --- |
| `vllm/models/common/specialized_decode/` | The contract: spec, guards, dispatch. Plain Python, no kernel code and no vendor runtime in its import graph, so it loads anywhere. |
| `vllm/models/<model>/<vendor>/specialized_decode/spec.py` | One model's `DecodeSpec`: what the deployment has to satisfy. No kernel backend in its import graph, so selection runs before anything vendor-specific loads. |
| `vllm/models/<model>/<vendor>/specialized_decode/runtime/` | That model's `DecodeImpl`, imported lazily from `spec.build`, with its own kernels beneath. |

Every kernel, including the wave and math primitives, the tagged mailbox, FP8/MX
arithmetic, the execution model and the symmetric peer buffers, stays inside the model
that owns it, for example at `vllm/models/deepseek_v41/amd/mono/`. Kimi K3's `cutedsl/`
and GLM-5.2's `flydsl/` sit in the same place for their models.

Cross-model-family sharing of kernel code is deliberately out of scope. It needs
validation that has not been done (DeepSeek against Kimi, in particular), and promoting
those files would make every model's kernels move together. Keeping them model-local
costs some duplication and keeps the blast radius of a contract change at the contract.
A shared `<vendor>/` directory can be added once two families have actually been
cross-checked.

One consequence worth calling out: a model's JIT cache key is typically seeded by a
digest of every `.py` under that model's kernel package, so an edit to one model's
kernels rebuilds nothing of another's.

## Adding a second kernel for the same model

Say DeepSeek-V4.1 gets a gfx942 kernel next to the gfx950 one. Write the FlyDSL kernel,
then:

```python
# vllm/models/deepseek_v41/amd/specialized_decode/runtime/decode_gfx942.py

class DSV41Gfx942Kernel(DSV41Kernel):           # reuse what you want
    def run(self, step: Step, input_ids, inputs_embeds, **kwargs) -> Any:
        ...
        return self.runner.run_gfx942_kernel(...)

    def compile(self, key):
        width, idx = key
        self.runner.kernels(width, self.plans[idx])
```

```python
# vllm/models/deepseek_v41/amd/specialized_decode/spec.py

def _refuse_config_gfx942(vllm_config) -> list[str]:
    reasons = []
    if not current_platform.is_gfx942():
        reasons.append("needs a gfx942 GPU")
    ...
    return reasons


def _build_gfx942(vllm_config, model, widths):  # the kernel import stays lazy
    from .runtime.decode_gfx942 import DSV41Gfx942Kernel

    return DSV41Gfx942Kernel(vllm_config, model, widths)


DECODE_SPEC_GFX942 = DecodeSpec["DSV41Gfx942Kernel"](
    name="dsv41-cdna3",
    supports=frozenset({Feature.PIPELINE_PARALLEL}),   # no spec decode here
    widths=lambda cfg: tuple(range(1, 17)),            # fewer rows than gfx950
    refuse_config=_refuse_config_gfx942,
    refuse_model=_refuse_model,                        # shared with gfx950
    refuse_caches=_refuse_caches,
    skip_step=_skip_step,
    build=_build_gfx942,
)

SPECS = (DECODE_SPEC_GFX950, DECODE_SPEC_GFX942)
```

Then one line in the model:

```python
self.decode_spec = select_decode_spec(vllm_config, SPECS)
```

The dispatcher takes the first compatible spec. Specializing on something other than
arch works the same way. A hyper-specialized batch-size 1 kernel can just set
`widths=lambda cfg: (1,)` and be placed first in the `SPECS` list.

## A kernel that only covers part of a layer

Sometimes a layer cannot be taken whole: DeepSeek-V4.1 has layers where only the FFN
seam is worth a launch and vLLM's attention has to run. The library has no concept of
this, and does not need one, because the kernel owns the layer loop. It decides per
layer whether to launch the whole thing or to run vLLM's layer with the seam injected:

```python
class DSV41Kernel:
    def run(self, step: Step, input_ids, inputs_embeds, **kwargs) -> Any:
        ...
        for idx in range(self.embed_layer, self.layer_end):
            layer, plan = model.layers[idx], self.plans.get(...)
            ffn = out = None
            if plan.ffn_only:
                ffn = partial(self.launch_ffn, plan)      # a seam, not a layer
            else:
                out = self.launch_whole(plan, hidden, positions, ...)
            if out is None:
                out = layer(hidden, positions, ..., decode_ffn=ffn)
            hidden, residual, post_mix, res_mix, pre_mix = out
        ...
```

`decode_ffn` is an argument on DeepSeek-V4.1's own decoder layer, not a library concept,
so a model that needs a seam adds one and a model that does not never hears about it.
The same shape covers a whole-layer launch that declines at runtime: `launch_whole`
returns None and vLLM's layer runs instead, which is also the per-layer fallback (a
layer with no plan simply runs as usual).

The build side follows: `warmup_keys` returns a `(width, "ffn")` key for the seam layers
and a whole-layer key for the rest, and `compile` dispatches on it.

## Worked example: a speculator's draft layers

A specialized decode can also be put on a speculator's draft head. A model-level kernel
binds to one backbone, and the draft head is the speculator's, not the target's, so the
draft needs its own `DecodeSpec` rather than falling out for free. DeepSeek-V4.1 gives
it one, `DRAFT_DECODE_SPEC`, and the three model touch points go on
`DSparkDeepseekV4ForCausalLM` exactly as they do on the target.

The two kernels share `LayerLaunches`, which settles everything at load (the plans, the
runner, the warm-up keys); each subclass owns only its layer loop and its routing. The
draft differs in four ways, all of them data: its own routed experts and top-k
(`dspark_n_routed_experts` / `dspark_num_experts_per_tok`, 128 / 3 on V4.1-Flash), one
fp4 ug group instead of two, SWA rows that are a block's non-causal window rather than a
causal decode token's (`get_dspark_swa_index_width`), and a first layer that is handed
the embedding already broadcast to the hc copies. A draft layer the kernels cannot take
whole runs entirely as vLLM's: there is no FFN-only launch for the draft.

One lifecycle detail: `ModelState` reaches only the target causal LM, so the draft heads
built in this process register themselves and
`DeepseekV41ROCMModelState.initialize_kv_cache` drains the registry, checking the
draft's caches alongside the target's for both the profiling cache and the real one.

## Worked example: a second GPU architecture

Supporting a second architecture (DeepSeek-V4.1 on MI300X and MI325X next to MI355X) is
entirely model-specific and lands inside `vllm/models/deepseek_v41/amd/`, leaving the
contract untouched: CDNA3 joins `CDNA_VERSIONS`, the row cap narrows to 42 on gfx942 (a
64 KB LDS workgroup), and the ARCH-vs-GPU refusal and the gfx950-only MoE backend check
go into `_refuse_model`. The per-layer differences (no fused seam requirement, TP 4
only, vLLM's FP8 indexer instead of the MXFP4 one) sit in `_skip_layer` and `_ffn_only`
next to the checks they qualify.

Two pieces need a different shape:

**The weight copies.** The gfx942 kernels read their own re-encoded copies of a layer's
weights, made at load by wrapping each `quant_method.process_weights_after_loading`, so
that they count as model memory when vLLM sizes the KV cache. Those per-layer quant
hooks run *before* the model-level `process_weights_after_loading`, which is where
`SpecializedDecode` is built, so the wrapping cannot live at the build point. It happens
at model construction instead, from `prepare_weights`, called next to
`select_decode_spec` in the causal LM's `__init__`. That function also records the flag
the gfx942 decode helpers read and builds their top-k hooks. The copy machinery is one
file, `runtime/weights942.py`.

**The draft on gfx942.** There the draft layers take the FFN launch after vLLM's
attention, not a whole-layer launch, so `_plan_layer` returns an FFN plan for them and
the draft's layer loop grows the same `decode_ffn` injection the target's has.
`DSV41DraftKernel` owns its own runner, so on gfx942 it is built at the target's fp4 ug
groups rather than the draft's one, which keeps the kernel configuration that was
validated there. The cost is one extra scratch buffer.

One mechanical follow-on: `runner.ffn_kernel` is keyed by `(tokens, experts, topk)`, so
the warm-up keys carry the routing.

## The flag

A deployment asks for a specialized decode with `--specialized-decode`, whose value is
the backend to run on:

- `--specialized-decode monokernel` runs most or all of a decoder layer in a single
  persistent launch.
- `--specialized-decode` with no value, or `--specialized-decode auto`, lets vLLM pick,
  which today means the monokernel.
- Leaving it unset runs vLLM's regular decode. An unknown backend is rejected by argparse
  at startup, and a deployment that cannot run the selected backend is refused at
  startup rather than falling back silently.

`--monokernel` is a CLI shorthand for `--specialized-decode monokernel`, and
`--no-monokernel` turns off a specialized decode asked for in a config file or through
the Python API. The underlying config field is
`KernelConfig.specialized_decode: Literal["auto", "monokernel"] | None`, so a future
implementation gets a name rather than a second boolean.
