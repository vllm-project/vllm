# Expert substitution (MoNE)

[Mixture of Novices and Experts (MoNE)](https://arxiv.org/abs/2507.00390)
prunes MoE layers by replacing low-importance experts with "novices" that
return a constant vector. vLLM loads the retained experts into compact weights
and adds the novices' output separately, so pruned experts use no MLP weight
memory and no MoE backend needs to know about them.

## Checkpoint format

The checkpoint keeps the original architecture and is produced by
llmcompressor's `MoNePruningModifier`. `config.json` declares the per-layer
retained expert count in `quantization_config.moe_sparsity_config`:

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

The `num_experts` list is indexed by decoder layer; `null` entries denote
layers that keep all of their experts (including dense layers). For VLMs the
`quantization_config` may live on the text config.

Each MoE layer stores:

* the router (`mlp.gate.weight`) at its full original width, with rows
  reordered so the retained experts occupy the leading rows and the collapsed
  experts the trailing rows;
* the retained experts left-shifted into compact stacked tensors
  `mlp.experts.gate_up_proj` (`[retained, 2 * intermediate, hidden]`) and
  `mlp.experts.down_proj` (`[retained, hidden, intermediate]`), so physical
  row `r` holds the expert of router row `r`;
* one `mlp.experts.novices` tensor of shape
  `[num_collapsed, hidden_size]` whose rows match the trailing router rows.

```text
model.layers.0.mlp.gate.weight            # [128, hidden]
model.layers.0.mlp.experts.gate_up_proj   # [109, 2 * intermediate, hidden]
model.layers.0.mlp.experts.down_proj      # [109, hidden, intermediate]
model.layers.0.mlp.experts.novices        # [19, hidden]
```

## Semantics

For a collapsed expert `j` with constant `v_j` and router weight `w_j(x)`, the
expert contributes `w_j(x) v_j`. Routing is unchanged: the router scores all
of its rows and router weights are neither renormalized nor redistributed to
retained experts.

## Execution

`FusedMoEFactory` matches each MoE layer to `moe_sparsity_config` by the layer
index in its prefix and, for a match, builds `SubstitutedRoutedExperts`, which:

1. allocates weights only for retained experts, one per leading router row;
2. after routing, sums `w_j(x) v_j` over collapsed routes in FP32;
3. rewrites collapsed routes as zero-weight routes to physical expert 0 and
   retained routes as compact physical IDs, preserving the
   `[num_tokens, top_k]` contract of every decomposed backend;
4. adds the constant output on one tensor-parallel rank, before the partial
   outputs are all-reduced.

Zero-weight routes may schedule some unnecessary GEMM work; the memory savings
are unaffected. Monolithic backends route internally and are not selected.

## Weight loading

Retained experts are loaded from the compact stacked tensors through the
standard fused expert mapping; the stacked rows already match the compact
physical rows. `novices` tensors are consumed at the model's `load_weights`
boundary, so initial loading and direct weight updates share one path.
`novices` tensors of layers that the model does not build (other pipeline
stages, MTP layers) are passed through to the model's loader, which skips them
like any other weight it does not own. Loading fails if a local collapsed
expert layer has no `novices` tensor.

Direct `load_weights` updates may replace a layer's `novices` tensor.
Layerwise reloads (`reload_weights` in checkpoint format) are rejected, and
the model keeps its previous weights.

## Supported configurations

- Models whose MoE layers use `FusedMoEFactory` with the default
  `RoutedExperts`, and whose MoE prefix contains exactly one layer index.
- Unquantized experts.
- Tensor and pipeline parallelism, including MTP draft models.
- Any decomposed MoE backend.

Expert and data parallelism, EPLB, MoE LoRA, fused shared experts, quantized
experts, and routed input/output transforms are rejected at initialization.
Deferred MoE finalize and layerwise weight reloads are rejected when used.
