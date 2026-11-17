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
- **Object.** An immutable entry in a shared Mooncake store holding the rows of a span of positions of one session, one array per enabled output. A full object covers one KV block of the full-attention KV cache group (a whole number of hash blocks); a tail object covers the positions after a request's last full object.
- **Weight version.** A string the RL framework sets on the engine at every weight update (`update_weight_version`); `default` before the first call.
- **Segment.** A pair `[start, version]`: positions from `start` up to the next segment's start carry that weight version.

## The design in short

1. **The weight version is part of the block hash.** A session request carries per-segment cache salts: the hash of every hash block includes the versions of its positions, next to its tokens. Block hashes chain, so a hash names the tokens and versions of the whole prefix.
2. **The session and the key prefix are part of the cache salt.** Only requests of one session, under one key prefix, compute or hit that session's KV.
3. **Every KV connector keyed by block hashes follows without connector code.** KV stored in any tier or on any replica is found only by a request that states the same tokens and versions.
4. **An object is named by a block hash.** A full object's key is `<prefix>/<session>/<hash of the block's last hash block>`. The same name means the same tokens at the same versions and, with batch invariance, the same rows. The store keeps the first object written under a name.
5. **The client carries the versions between turns.** Each response returns `aux_version_segments`. The client sends them with the session's next request; the engine salts the covered prompt positions with them and later positions with the current version.
6. **What a request computes is named by the version that computes it.** When a waiting request is scheduled, every position it will compute, from its prefix-cache hit or its loaded KV on, is salted with the current version. Names of positions it reuses stay as stated.
7. **The response lists keys, not rows.** A finished request returns the keys of the objects covering all its positions, in order. The trainer reads them from the store and releases the session by key pattern after the engine has completed a pause.
8. **The remaining limits are documented, not counted.** A failed synchronous KV load reported through `invalid_block_ids` is not supported for session requests, and multi-module MTP draft KV carries the target versions ([protocol §9](protocol.md#9-limits)). The engine does not count them; a failed synchronous load raises, asserts or goes unnoticed depending on the case.

## Requirements

- Batch invariance: the same tokens at the same weight versions give bitwise identical outputs regardless of batch, chunking, or prefix-cache hit (#62365).
- Replicas that share KV stores or the object pool hash identically: the same hash algorithm with `PYTHONHASHSEED` set to one value everywhere or unset everywhere (unset, only a cryptographic algorithm such as `sha256` gives every process the same seed), hash block size, full-attention KV block size (the object size) and look-ahead. Replicas that share the object pool also use the same enabled outputs.
- Prefix caching on, a plain full-attention KV cache group, hits on KV block boundaries, at most one hash look-ahead token, no prefix replay. The engine refuses to start otherwise.
- KV connectors store the blocks that decoding completes (for `MooncakeStoreConnector`, `save_decode_cache: true`).
- The object pool does not share a Mooncake master or pool with a KV store (`mooncake_config_path` names its configuration when the Mooncake KV connector reads `MOONCAKE_CONFIG_PATH`).

## PR stack

Each layer is one pull request on `main`, in this order:

| # | Layer | Content |
| --- | --- | --- |
| 1 | Per-segment cache salts (generic) | `cache_salt_segments: list[(start, salt)]` on `EngineCoreRequest` and `Request`; `Request.add_cache_salt_segment`. A hash block's extra keys include the segments overlapping it, with starts relative to the block. Requests without segments hash as before. |
| 2 | Session objects | `aux_session` and `aux_version_segments`; the session salt; salts at admission and at weight updates; full and tail objects; `aux_output_keys` and `aux_version_segments` in the response; the Mooncake object store with its own configuration (`mooncake_config_path`), the pin choice (`object_hard_pin`, off for the default SSD-first store) and the write retry deadline (`object_put_stall_timeout_s`), `read_aux_rows`, `aux_session_pattern` and `aux_prefix_pattern`; writes drained at every pause; startup checks. |
| 3 | Block hashes from the first computed block | The scheduler sends a session request's block hashes from its first computed block on, so a request resumed after a prefix-cache reset does not send its whole history again. |
| 4 | Hidden states and top-k logprobs | `enable_return_hidden_states`: field `hidden_states` of session objects. `return_topk_logprobs = k`: fields `topk_token_ids` and `topk_logprobs` of session objects, computed only for session rows, in bounded chunks. |
| 5 | Objects without routed experts | `object_key_prefix` with any enabled output, so dense models can deliver hidden states and top-k logprobs. |
| 7 | Recomputed positions | Just before a waiting request is scheduled, the positions it will compute are salted with the current version. The offloading connector derives its offload keys again from the first chunk whose block hash changed. |

Routed expert capture with pipeline parallelism, which merges the stages' routing at the last stage, comes before layer 2. These documents come between layers 5 and 7.

Deferred: a vocab-parallel top-k path that avoids gathering full-vocabulary logits.

Later:

- **Prefill/decode disaggregation.** The prefill's segments travel in `kv_transfer_params`.
- **Rust frontend.** It returns `aux_output_keys` and `aux_version_segments` but does not accept `aux_session`; it must apply the salts before it hashes.

## Documents

| Document | Contents |
| --- | --- |
| [Protocol](protocol.md) | A session step by step, request and response fields, how names are formed and when, what the engine writes, hits and weight updates, pause and release, deployment, limits |
| [Consumer contract](consumer.md) | How a trainer reads objects, what each field holds, and what it must do |
| [Design space](design_space.md) | The alternatives for each decision and why this one |
| [Scenarios](scenarios.md) | Sizing: bytes per position, KV tier capacity, object pool, object counts |
| [Experiments](experiments.md) | Validation results, open items, and earlier measurements |

## Status

Requests without `aux_session` keep the inline behavior: routed experts and indexer selections returned in the response. Validation results are in [experiments](experiments.md).

RFC #64380 described an earlier protocol (manifests, per-trajectory copies, a writer-side version ledger). This folder replaces it.
