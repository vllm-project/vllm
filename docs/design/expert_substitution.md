# Expert substitution

Expert substitution is an inference representation for routed MoE layers. It
removes selected expert MLPs from a checkpoint and replaces each removed expert
with an explicitly described executor. The first supported executor,
`constant-v1`, returns a stored vector.

The compression algorithm and the inference representation are separate. For
example, MoNE may produce a `constant-v1` checkpoint, but vLLM does not execute
MoNE calibration or select experts. It only executes the serialized
representation.

## Checkpoint contract

Expert substitution is stored under the checkpoint's compression metadata. The
model architecture is unchanged.

```json
{
  "model_type": "deepseek_v2",
  "architectures": ["DeepseekV2ForCausalLM"],
  "compression_config": {
    "producer": {
      "name": "llm-compressor",
      "version": "..."
    },
    "provenance": {
      "algorithm": "mone"
    },
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
            "num_logical_experts": 256,
            "weight_layout": "compact_retained_experts",
            "replacements": {
              "3": {
                "format": "constant-v1",
                "tensors": {
                  "value": "model.layers.1.mlp.expert_replacements.3.value"
                }
              }
            }
          }
        }
      }
    }
  }
}
```

`producer` and `provenance` are informational. Runtime selection depends on the
replacement `format`, not on the producing algorithm.

Version 1 accepts only the following contract:

- `weight_layout` is `compact_retained_experts`.
- Every replacement is `constant-v1`.
- Each replacement provides exactly one explicit `value` tensor reference.
- Logical expert IDs and router weights are preserved.
- Router weights are not renormalized after replacement routes are removed from
  the MLP execution path.

Unknown versions, formats, layouts, tensor contracts, or router semantics fail
during model initialization.

## Logical and physical experts

Routers continue to produce stable logical expert IDs. Retained MLPs are stored
in compact physical rows ordered by ascending logical ID. Replacements map to
`-1` and never receive a physical MLP row.

```text
Logical expert IDs:  [0, 1, 2, 3, 4, 5]
Replacements:            1        4
Physical MLP rows:   [0,    1, 2,    3]
Logical to physical: [0, -1, 1, 2, -1, 3]
```

Retained checkpoint tensors remain named by logical expert ID. The loader uses
the layout to place them in compact physical order. Replacement tensors use the
exact checkpoint names in `tensors.value`.

## `constant-v1`

For logical expert `j`, `constant-v1` defines

\[
E_j(x) = v_j
\]

where `v_j` is a one-dimensional tensor in the routed expert output space. If
the router selects expert `j` with weight `w_j(x)`, its contribution is

\[
w_j(x) v_j.
\]

The vector length must equal the routed output hidden size. Values use the
checkpoint parameter dtype and are replicated across tensor-parallel ranks.

## Execution

The runtime performs the following sequence:

1. Route against the full logical expert set.
2. Compute constant contributions from the original IDs and router weights.
3. Set replacement-route weights to zero for regular MoE computation.
4. Map retained logical IDs to compact physical rows.
5. Exclude invalid rows from Triton alignment, padding, GEMM scheduling, and
   regular expert reduction.
6. Reduce retained expert output across tensor-parallel ranks.
7. Add the replicated constant contribution once after that reduction.

This preserves the original routing weights without scheduling dummy MLP work
for substituted experts.

## Supported configurations

The initial implementation supports:

- DeepSeek V2 and V3 implementations sharing the DeepSeek V2 MoE path.
- GLM models inheriting that path.
- Homogeneous `constant-v1` substitutions.
- Unquantized FP16 and BF16 retained experts.
- Tensor parallelism.
- The decomposed Triton MoE backend.
