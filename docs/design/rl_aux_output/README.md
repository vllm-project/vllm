# Aux Outputs for RL

An RL trainer recomputes the logprobs of the tokens an inference engine sampled. Some values the engine computed on the way cannot be recomputed by the trainer, or only at a high cost, and have to be shipped with the rollout:

- the experts each MoE layer routed every position to (R3, routing replay);
- the selections of a sparse-attention indexer;
- the final hidden states;
- the top-k next-token logprobs, for example of a teacher scoring long prompts.

This folder specifies how vLLM delivers these **aux outputs** for long multi-turn RL rollouts through a shared object store, and records the evidence behind the design. The user-facing guide is [Routed Experts Return](../../training/routed_experts.md#sessions-delivering-rows-through-a-shared-object-store).

## Terms

- **Position.** Position `p` processes token `p` and produces the logits for token `p + 1`. A sequence of `L` tokens has aux outputs for positions `[0, L - 1)`; its rows are those positions' outputs.
- **Session.** A set of requests that may share KV and objects: one agent rollout, one GRPO group, one tree of forks. A request joins a session with `aux_session`.
- **Hash block.** The unit of vLLM's prefix-cache block hashes.
- **Object.** An immutable entry in a shared Mooncake store holding the rows of a span of positions of one session, one array per enabled output. A full object covers one KV block of the full-attention KV cache group (a whole number of hash blocks); a tail object covers the positions after a request's last full object. With pipeline parallelism each stage stores its own part in its own object.
- **Weight version.** A string the RL framework sets on the engine at every weight update (`update_weight_version`); `default` before the first call.
- **Segment.** A pair `[start, version]`: positions from `start` up to the next segment's start carry that weight version.

## The design in short

1. **The weight version is part of the block hash.** A session request carries per-segment cache salts: the hash of every hash block includes the versions of its positions, next to its tokens. Block hashes chain, so a hash names the tokens and versions of the whole prefix.
2. **The session and the key prefix are part of the cache salt.** Only requests of one session, under one key prefix, compute or hit that session's KV.
3. **Every KV connector keyed by block hashes follows without connector code.** KV stored in any tier or on any replica is found only by a request that states the same tokens and versions.
4. **An object is named by a block hash.** A full object's key is `<prefix>/<session>/<hash of the block's last hash block>`. The same name means the same tokens at the same versions and, with batch invariance, the same rows. The store keeps the first object written under a name.
5. **The client carries the versions between turns.** Each response returns `aux_version_segments`. The client sends them with the session's next request; the engine salts the covered prompt positions with them and later positions with the current version.
6. **What a request computes is named by the version that computes it.** When a waiting request is scheduled, every position it will compute, from its prefix-cache hit or its loaded KV on, is salted with the current version. Names of positions it reuses stay as stated.
7. **The response lists keys, not rows.** A finished request returns the keys of the objects covering all its positions, in order. The trainer reads them from the store and releases the session by key pattern after the engine has paused.
8. **The remaining limits are documented, not counted.** A running request keeps the names it was scheduled with, so a few cases name a recomputed position with another version than the computing one ([protocol §12](protocol.md#12-limits)). The engine does not detect or report them.

## Requirements

- Batch invariance: the same tokens at the same weight versions give bitwise identical outputs regardless of batch, chunking, or prefix-cache hit (#62365).
- Replicas that share KV stores or the object pool hash identically: the same hash algorithm (`sha256`, or the same `PYTHONHASHSEED`), hash block size and look-ahead. Replicas that share the object pool also use the same pipeline-parallel size and enabled outputs.
- Prefix caching on, a plain full-attention KV cache group, hits on KV block boundaries, at most one hash look-ahead token, no prefix replay. The engine refuses to start otherwise.
- KV connectors store the blocks that decoding completes (for `MooncakeStoreConnector`, `save_decode_cache: true`).
- The object pool does not share a Mooncake master with a KV store that is reset.

## PR stack

Each layer is one pull request on `main`, in this order:

| # | Layer | Content |
| --- | --- | --- |
| 1 | Per-segment cache salts (generic) | `cache_salt_segments: list[(start, salt)]` on `EngineCoreRequest` and `Request`; `Request.add_cache_salt_segment`. A hash block's extra keys include the segments overlapping it, with starts relative to the block. Requests without segments hash as before. |
| 2 | Session objects | `aux_session` and `aux_version_segments`; the session salt; salts at admission and at weight updates; full and tail objects; per-stage objects under pipeline parallelism; `aux_output_keys` and `aux_version_segments` in the response; the Mooncake object store, `read_aux_rows` and `aux_session_pattern`; writes drained at every pause; startup checks. |
| 3 | Hidden states and top-k logprobs | `enable_return_hidden_states`: field `hidden_states` of session objects. `return_topk_logprobs = k`: fields `topk_token_ids` and `topk_logprobs` of session objects, computed only for session rows, in bounded chunks. |
| 4 | Objects without routed experts | `object_key_prefix` with any enabled output, so dense models can deliver hidden states and top-k logprobs. |
| 5 | Recomputed positions | Just before a waiting request is scheduled, the positions it will compute are salted with the current version. The offloading connector derives its offload keys again from the first chunk whose block hash changed. |

A fix to `collective_rpc` that steps with more than one worker need comes before layer 2, and these documents come before layer 5.

Deferred: a vocab-parallel top-k path that avoids gathering full-vocabulary logits.

Later:

- **Prefill/decode disaggregation.** The prefill's segments travel in `kv_transfer_params`.
- **Rust frontend.** It returns `aux_output_keys` and `aux_version_segments` but does not accept `aux_session`; it must apply the salts before it hashes.

## Documents

| Document | Contents |
| --- | --- |
| [Protocol](protocol.md) | Configuration, request and response fields, how names are formed and when, what the engine writes, weight updates, reset, pause and release, requirements and limits |
| [Consumer contract](consumer.md) | How a trainer reads objects, what each field holds, and what it must do |
| [Design space](design_space.md) | The alternatives for each decision and why this one |
| [Scenarios](scenarios.md) | Sizing: bytes per position, KV tier capacity, object pool, object counts |
| [Experiments](experiments.md) | Validation results, open items, and earlier measurements |
| [Request-wise design](request_wise.md) | An alternative where objects belong to the request that wrote them: names without versions, the `held` rule, release by reference count, and a comparison with this design |

## Status

All six layers are on the branch. Requests without `aux_session` keep the inline behavior: routed experts and indexer selections returned in the response. Validation of the full stack is tracked in [experiments](experiments.md); the runs on the current branch are pending.

RFC #64380 described an earlier protocol (manifests, per-trajectory copies, a writer-side version ledger). This folder replaces it.

## Branches

Both design branches start from the same commit of the session stack, `cfcec38b62`, and each carries documents that describe its own code.

| Branch | Head | Status |
| --- | --- | --- |
| `rl/aux-output` | | This branch: the integration branch with these documents. |
| `rl/aux-output-session-wise` | `04495f3806` | The five layers of the PR stack, plus one commit that makes aborted session requests list their keys. Its protocol and consumer contract add the rule for building a trajectory from several responses, and the limits found in validation. Validated on six of the eight scenarios; two are running ([evidence](request_wise.md#evidence)). |
| `rl/aux-output-request-wise` | `7a3f62ba55` | The request-wise design ([request-wise design](request_wise.md)): 19 code commits and one document commit on the same base. Its own `request_wise.md` describes the code. Validated on all eight scenarios ([evidence](request_wise.md#evidence)). |
