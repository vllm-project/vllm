# Manifest-driven streaming expert reload

## Problem

A checkpoint reload must not allocate checkpoint-format storage for the entire
model before the first source tensor arrives. MoE checkpoints make this
especially expensive because expert weights dominate model size and a trainer
may transmit experts repeatedly or update only selected layers.

## Source of truth

Streaming reload derives its coverage model from the rank-local sharding
manifest captured during initial checkpoint loading. Each `RankShard` records:

- checkpoint source name;
- destination parameter name;
- loader fragment metadata such as `expert_id` and `shard_id`;
- the rank and parallel scope that consumed the fragment.

No quant-method-specific reload hook is required. Completion is set coverage,
not tensor size or transport-call boundaries.

## Lazy checkpoint staging

`ModelwiseReloadSession` does not install a second expert tracker. At start it
constructs `_LazyCheckpointBindings`, which compiles the manifest once into the
following lightweight indexes:

```text
source_name -> target names
module name -> expected RankShard set
```

The original weight loader still performs expert mapping, TP narrowing, padding,
and fused `w1`/`w3` placement. The manifest's `RankShard.fragment` retains the
loader facts such as `expert_id` and `shard_id`; the same `RankShard` objects are
used for module completeness and PWAL selection.

`before_source()` materializes only targets belonging to the source currently
being loaded. Compatible non-quantized targets can bind directly to runtime
storage. Quantized or incompatible targets receive checkpoint-format storage on
first use. Thus the transaction does not allocate all expert tensors at start.

`ReloadUnit` and `ShardCoverageTracker` remain available for their standalone
unit-level API and tests, but modelwise reload does not install them. Reusing
them as a second manifest-derived coverage system would duplicate the sharding
truth and may allocate additional expert staging slabs.

## Finish semantics

At `finish`, `_LazyCheckpointBindings` classifies manifest modules as:

- **complete**: every expected shard for the touched module arrived;
- **incomplete**: at least one shard arrived, but its module is not complete;
- **untouched**: no shard for the module arrived.

Incomplete and untouched modules are rebound to their original serving
storage. Complete modules participate in normal model PWAL; no separate expert
tracker assembly step is needed because the model's original loaders wrote the
manifest targets directly. Processed values are validated and copied into
stable serving storage.

This permits trainer updates containing only selected layers. It does not make
an incomplete module visible: a partially transmitted linear or MoE module is
discarded at the transaction boundary.

## Ordinary parameters

Non-expert parameters use the same manifest completion rules. Storage is
allocated only when a source targeting the parameter begins loading. When the
checkpoint and serving shape/dtype are compatible and no quantized PWAL schema
conversion is required, the loader may write directly into serving storage.
Otherwise it uses checkpoint-format staging until the owning module is
complete.

## Parallel sharding

The captured manifest already represents the inference rank's expected
fragments, so transport chunking and repeated expert delivery are independent
of correctness. Conversion between different trainer and inference parallel
strategies—for example trainer TP=8 and inference TP=4—is a future transform at
the source-to-target shard layer. The streaming completion and allocation
model does not depend on the two TP sizes being equal, but the transform itself
is not implemented yet.
