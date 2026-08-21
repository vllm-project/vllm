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

## Expert staging

`ExpertShardLoader` remains installed on expert parameters. During reload it
uses the active `ShardCoverageTracker` to redirect each expert fragment into a
checkpoint-format slab for that local expert. The original weight loader still
performs expert mapping, TP narrowing, padding, and fused `w1`/`w3` placement.

`ModelwiseReloadSession.start()` builds trackers directly from the manifest and
construction-time tensor metadata. A tracker therefore knows:

- the complete set of expert/shard fragments expected for the module;
- the checkpoint-format shape and dtype of each per-expert destination;
- which repeated source transmission replaces a previously received fragment.

The tracker retains only expert slices that have started loading. It does not
materialize the full expert tensor merely because the transaction started.

## Finish semantics

At `finish`, `_LazyCheckpointBindings` classifies manifest modules as:

- **complete**: every expected shard for the touched module arrived;
- **incomplete**: at least one shard arrived, but its module is not complete;
- **untouched**: no shard for the module arrived.

Incomplete and untouched modules are rebound to their original serving
storage. Complete expert trackers assemble their staged slices into the
checkpoint-format module tensors, after which normal model PWAL runs. Processed
values are validated and copied into stable serving storage.

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
