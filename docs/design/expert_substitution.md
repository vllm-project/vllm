# Expert substitution (MoNE)

[Mixture of Novices and Experts (MoNE)](https://arxiv.org/abs/2507.00390)
replaces low-importance MoE experts with "novices" that return constant vectors.
vLLM stores only retained expert MLPs and novice vectors, and combines their
outputs using the model's router weights.

## Loading a MoNE checkpoint

Load a compatible llmcompressor `MoNePruningModifier` export normally, such as
[Qwen3.6-35B-A3B-MoNe-20](https://huggingface.co/inference-optimization/Qwen3.6-35B-A3B-MoNe-20):

```python
from vllm import LLM

llm = LLM(model="inference-optimization/Qwen3.6-35B-A3B-MoNe-20")
```

For serving:

```bash
vllm serve inference-optimization/Qwen3.6-35B-A3B-MoNe-20
```

Checkpoint metadata enables MoNE automatically; no MoNE flag is needed.
Unpruned layers use their normal implementation.

To request TRT-LLM monolithic execution on supported hardware:

```python
llm = LLM(
    model="inference-optimization/Qwen3.6-35B-A3B-MoNe-20",
    kernel_config={"moe_backend": "flashinfer_trtllm"},
)
```

Backend defaults are unchanged. MXFP4 uses the TRT-LLM path when explicitly
requested; INT4 additionally requires `VLLM_USE_FLASHINFER_MOE_INT4=1`.
Automatic selection can fall back to a compatible decomposed backend. An
explicitly requested backend must support the checkpoint and configuration.

## Supported configurations

Let `E` be the original logical expert count and `R` the retained count in a
layer. Backend support is subject to the quantization method's hardware,
shape, and dtype requirements:

| Expert weights | Decomposed backend | TRT-LLM monolithic backend |
| --- | --- | --- |
| Unquantized | Supported | BF16; `E <= 2048` |
| Compressed-tensors FP8, per-tensor or block-scaled | Supported | `E <= 512` |
| Compressed-tensors NVFP4 | Supported | `E <= 2048` |
| Compressed-tensors MXFP4 | Supported | BF16 inputs; `E <= 2048` |
| Compressed-tensors symmetric group-32 INT4 | Supported | MXINT4 kernel; BF16 inputs; `E <= 2048` |

Monolithic execution requires supported Blackwell hardware and ungrouped
softmax routing with selected weights renormalized (`Renormalize` or
`RenormalizeNaive`). Other routing methods require a compatible decomposed
backend. Other quantization methods are not enabled for MoNE.

Models must use `FusedMoEFactory` with the default `RoutedExperts`, with one
substituted MoE per decoder layer and exactly one layer index in its prefix.
Tensor parallelism is supported. The loader also handles pipeline-stage and
MTP layer ownership; these ownership checks are not full distributed
generation coverage.

Expert, data, prefill context, and sequence parallelism, EPLB, MoE LoRA, fused
shared experts, router weighting on expert inputs, custom `RoutedExperts`
classes, and routed input/output transforms are rejected at initialization.
Deferred MoE finalize and layerwise weight reloads are rejected when used.
Paths that bypass `FusedMoEFactory`, such as native MegaMoE, do not use this
implementation.

## Checkpoint format

The checkpoint keeps the original architecture. This `config.json` excerpt
declares retained counts in `quantization_config.moe_sparsity_config`:

```json
{
  "architectures": ["Qwen3MoeForCausalLM"],
  "num_experts": 128,
  "quantization_config": {
    "config_groups": {},
    "format": "fakequant",
    "quant_method": "compressed-tensors",
    "quantization_status": "compressed",
    "moe_sparsity_config": {
      "method": "novice",
      "num_experts": [109, 109, null, 109]
    }
  }
}
```

The list is indexed by decoder layer, including dense layers. Use `null` for
unpruned layers. A count must satisfy `1 <= R <= E`; `R == E` also disables
substitution. Keep the architecture's expert count (`num_experts` here) at
`E` so the router still produces all logical expert scores. VLM metadata can
live on the text config or the outer model's quantization config.

Within each pruned layer, the export must order the router and expert tensors
consistently:

* Router rows `[0, R)` name retained experts; rows `[R, E)` name novices.
* Compact expert tensors contain only the `R` retained MLPs, in router order.
* A floating-point `mlp.experts.novices` tensor has shape `[E - R, hidden_size]`,
  with row `j - R` holding the constant for logical expert `j`.

For the first layer in the example, an unquantized stacked export stores:

```text
model.layers.0.mlp.gate.weight            # [128, hidden]
model.layers.0.mlp.experts.gate_up_proj   # [109, 2 * intermediate, hidden]
model.layers.0.mlp.experts.down_proj      # [109, hidden, intermediate]
model.layers.0.mlp.experts.novices        # [19, hidden]
```

Compact per-expert tensors are also accepted, for example
`mlp.experts.0.gate_proj.weight`, `mlp.experts.0.up_proj.weight`, and
`mlp.experts.0.down_proj.weight`, with expert IDs `0` through `R - 1`.
Projection names depend on the model's expert mapping.

Changing metadata alone does not convert an ordinary checkpoint: the exporter
must also reorder router rows, compact retained weights, and provide novices.
Legacy `approximate_experts` / `.approx_value` checkpoints and
`compression_config.transform_config.expert_substitution` metadata are not
interpreted as this format.

### Quantized tensors

Quantized exports use the same router ordering, sparsity metadata, and
floating-point novice tensor. Packed weights and per-expert scales keep the
compact leading dimension `R`. The `config_groups` must describe the quantized
projections; the empty example above is for unquantized experts. Scheme lookup
also accepts stacked module targets such as `mlp.experts.gate_up_proj` and
`mlp.experts.down_proj`.

| Format | Projection tensors |
| --- | --- |
| FP8 | `weight`, `weight_scale`, and `input_scale` for static activations |
| NVFP4 | `weight_packed`, `weight_scale`, `weight_global_scale`, `input_global_scale` |
| MXFP4 | `weight_packed` (two FP4 values per byte), group-32 exponent `weight_scale` |
| Symmetric group-32 INT4 | `weight_packed` (eight biased values per int32), floating-point `weight_scale` |

Suffixes attach to the projection module, for example
`model.layers.0.mlp.experts.gate_up_proj.weight_packed`. The down projection
uses the same suffix convention.

## Semantics and execution

A novice `j` contributes `w_j(x) * novice_j`. Routing retains the model's
existing top-k, normalization, and scaling rules. Novices participate in any
selected-weight normalization; their weights are not redistributed to retained
experts. Selecting retained expert 0 and novice 3 with final weights
`[0.25, 0.75]` therefore produces:

```text
0.25 * expert_0(x) + 0.75 * novice_3
```

Novice-only selections produce weighted constant vectors. Runner-level output
scaling applies to the combined retained and novice output.

`FusedMoEFactory` constructs `SubstitutedRoutedExperts` for pruned layers.
Both execution paths compute the novice contribution on TP rank 0 only,
using a dense matrix `[num_tokens, E - R]` of accumulated routing weights in
the novice vectors' dtype. Routing weights are cast to that dtype before
accumulation, and the matrix multiplies the stored novice vectors directly.
The contribution is added before output scaling and TP reduction. Adding it
on every rank would multiply the novice contribution by the TP size.

The dense calculation runs even when no novices are selected, avoiding separate
sparse dispatch calls. Its temporary memory grows with `num_tokens * (E - R)`.
Retained IDs already index compact rows; novice indices are logical IDs minus
`R`, so neither path needs an expert-ID lookup table.

### Decomposed backends

The runner computes novice output before `transform_routes` mutates the routes.
That method leaves retained IDs and weights unchanged and turns novice routes
into zero-weight routes to physical expert 0. The backend still receives
`top_k` slots per token, so novice selections can schedule unnecessary
retained-expert GEMMs. Only `R` expert MLPs are stored.

### Monolithic backends

The backend routes over all `E` logical experts but executes only `[0, R)`,
using `local_expert_offset=0` and `local_num_experts=R`. Novice selections run
no retained-expert GEMMs and require no dummy MLPs.

Two generic capabilities enable this path:

* `supports_partial_expert_execution`: execute only the configured local expert
  range, with no renormalization after filtering selections outside it.
* `supports_routing_replay_capture`: capture selected logical IDs, including
  selections outside that local range.

The backend has no novice-specific logic. On TP rank 0, MoNE requests
`routing_replay_out`, gathers the selected logits, and applies softmax to
reconstruct renormalized weights. The full-router denominator cancels, and
using captured IDs avoids a second top-k with potentially different tie choices.
External routing capture remains available alongside this internal capture.
Ordinary checkpoints request no internal capture and run no novice accumulation.

## Weight loading and updates

Retained tensors use the standard expert mapping. The model's `load_weights`
adapter consumes each local `.novices` tensor and maps it to the runtime
`routed_experts.expert_substitution.novices` parameter. Weights belonging to
other pipeline stages or MTP layers pass through to the model's own loader.
Loading fails if a local novice tensor is missing, has the wrong shape, or
contains NaNs.

DeepSeek V3.2 NVIDIA and GLM-5-Next runner-based loaders pass retained expert
tensors to the MoE runner as they arrive, accepting per-expert and stacked
weights/scales without buffering across layers or checkpoint shards.

Direct `load_weights` calls can replace individual layers' novice tensors.
Layerwise checkpoint reloads (`reload_weights`) are rejected while preserving
the previous weights.

## Validation and performance

The MoNE kernel and loader suites check independent numerical references for
retained-only, novice-only, and mixed selections; model-dtype novice calculation;
normalization/scaling; two-rank TP; compact quantized loading; generation
tokens and full-vocabulary log probabilities; and monolithic CUDA graph replay.
`tests/kernels/moe/test_routed_experts_capture_monolithic.py` also checks partial
execution with nonzero local offsets and fully nonlocal selections in all five
monolithic formats. These generated-checkpoint tests verify correctness, not
pretrained model quality or distributed expert-parallel support.

From the repository root, compare complete unquantized ordinary and compact
MoNE layers, including routing, novice accumulation, and TP reduction:

```bash
.venv/bin/python benchmarks/kernels/benchmark_mone.py --backend triton --output mone-triton.jsonl
```

Use `--backend flashinfer_trtllm --reference-backend triton` to additionally
compare monolithic and decomposed MoNE with identical weights and routes.
The benchmark defaults to CUDA graph timing; `--mode eager`, `--cache-mode cold`,
and `torchrun --nproc-per-node=2` allow eager timing, cache flushing, and TP=2.
It varies novice-selection fractions; use `--tokens` and `--retained` to select
batch sizes and retained counts. Fewer stored weights do not guarantee lower
latency; measure the backend and token counts relevant to the workload.
