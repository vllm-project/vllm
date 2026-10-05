# Expert substitution (MoNE)

[Mixture of Novices and Experts (MoNE)](https://arxiv.org/abs/2507.00390)
prunes MoE layers by replacing redundant experts with "novices" that return a
constant vector. vLLM loads the retained experts into compact weights and adds
the novices' output separately, so pruned experts use no MLP weight memory and
no MoE backend needs to know about them.

## Checkpoint format

The checkpoint keeps the original architecture. `config.json` lists the
substituted experts of each decoder layer:

```json
{
  "architectures": ["DeepseekV2ForCausalLM"],
  "n_routed_experts": 64,
  "approximate_experts": {
    "1": [1, 6, 8, 12],
    "2": [0, 3, 17, 40]
  }
}
```

For VLMs, `approximate_experts` may live on the text config. Retained experts
keep their logical checkpoint names. Each substituted expert stores one
`[hidden_size]` tensor instead of its MLP:

```text
model.layers.1.mlp.experts.0.gate_proj.weight
model.layers.1.mlp.experts.1.approx_value
```

Alternatively, `compression_config.transform_config.expert_substitution`
declares the same information with explicitly named constant tensors:

```json
{
  "compression_config": {
    "transform_config": {
      "expert_substitution": {
        "version": 1,
        "router_semantics": {
          "preserve_logical_expert_ids": true,
          "preserve_router_weights": true,
          "renormalize_after_substitution": false
        },
        "targets": {
          "model.layers.1.mlp.experts": {
            "num_logical_experts": 64,
            "weight_layout": "compact_retained_experts",
            "replacements": {
              "1": {
                "format": "constant-v1",
                "tensors": {"value": "model.layers.1.mlp.expert_replacements.1.value"}
              }
            }
          }
        }
      }
    }
  }
}
```

Only the values shown for `version`, `router_semantics`, `weight_layout`, and
`format` are supported. Each target path must contain exactly one layer index.
This form is not yet accepted alongside compressed-tensors quantization, whose
`transform_config` schema does not include `expert_substitution`.

## Semantics

For a substituted expert `j` with constant `v_j` and router weight `w_j(x)`,
the expert contributes `w_j(x) v_j`. Routing is unchanged: router weights are
neither renormalized nor redistributed to retained experts.

## Execution

`FusedMoEFactory` matches each MoE layer to `approximate_experts` by the layer
index in its prefix and, for a match, builds `SubstitutedRoutedExperts`, which:

1. allocates weights only for retained experts, in ascending logical order;
2. after routing, gathers `w_j(x) v_j` for substituted routes in FP32;
3. rewrites substituted routes as zero-weight routes to physical expert 0 and
   retained routes as compact physical IDs, preserving the `[num_tokens, top_k]`
   contract of every decomposed backend;
4. adds the constant output on one tensor-parallel rank, or on every rank when
   the backend already returns a reduced output.

Zero-weight routes may schedule some unnecessary GEMM work; the memory savings
are unaffected. Monolithic backends route internally and are not selected.

## Weight loading

Retained experts are loaded through the standard expert mapping by logical ID
and placed in their compact row. Substituted IDs map to no local row, the same
way expert parallelism skips non-local experts.

Constant tensors are consumed at the model's `load_weights` boundary, so
initial loading and weight reloads share one path. `approx_value` tensors of
layers that the model does not build (other pipeline stages, MTP layers) are
passed through to the model's loader, which skips them like any other weight
it does not own. Explicitly named constants are always consumed, and dropped
when their layer is built elsewhere. Loading fails if a local substituted
expert has no constant.

Direct updates may change individual constants. Layerwise reloads must supply
all constant rows of each updated layer; incomplete updates are rejected.

## Supported configurations

- Models whose MoE layers use `FusedMoEFactory` with the default
  `RoutedExperts`, and whose MoE prefix contains exactly one layer index.
- Unquantized experts.
- Tensor and pipeline parallelism, including MTP draft models.
- Any decomposed MoE backend.

Expert and data parallelism, EPLB, MoE LoRA, fused shared experts, quantized
experts, and routed input/output transforms are rejected at initialization.
