# KV hint actions

`KvHintsEnvelope` carries versioned payloads. At request admission, `dispatch.py` decodes registered `vllm.*` actions and either executes them immediately or defers them to successful completion, according to the consumer's registration. The dispatcher skips backend-owned actions and leaves the original envelope and opaque payloads intact.

## G1 RETAIN via SET_PRIORITY

Attach this envelope to an inference request using the existing `kv_hints` field:

```json
{
  "protocol_version": "0.1",
  "message_id": "router/message-1",
  "actions": [{
    "action_id": "retain-prefix",
    "action_type": "vllm.set_priority",
    "action_version": "1.0",
    "payload": {
      "target": "request_cached",
      "location": "local_g1",
      "claim_id": "router/session-1",
      "revision": 1,
      "value": 10,
      "ttl_seconds": 60
    }
  }]
}
```

- Decode at admission; apply at successful stop or length-limit completion, before releasing request references. Abort and error discard deferred actions. Preemption preserves them until completion.
- Scope is the request's currently cached copies in the receiving engine's G1 pool. Uncached tails, null blocks, and other pools are excluded. Empty scope is rejected. Remote and explicit-hash targeting are not implemented.
- Lower integer priorities are evicted first, with LRU breaking ties. The highest unexpired declaration wins; blocks with none have priority 0. Numeric priority never pins memory, reduces available capacity, or changes request ordering.
- Producers namespace `claim_id`. Higher revisions replace that claim's value and TTL, which starts at application. Same/older revisions cannot renew TTL; conflicting payloads at the same revision are rejected.
- To clear a claim, send `value: null`, omit `ttl_seconds`, and increase the revision. Other claims remain effective. Expired/cleared revision records stay until cache invalidation, reuse, or reset; each block admits at most 16 claim IDs.
- Invalid actions are rejected independently. Results are executor-local and describe affected physical copies at application time; no new response channel is added. Existing KV events continue to report actual cache residency changes.

`core/kv_cache_priority.py` owns declarations and eviction ordering. The block pool reports lifetime transitions; the scheduler supplies admission and completion hooks. Dispatch results can be `deferred`; this acknowledges parsing, not physical feasibility at execution time.

## G2 -> G1 prefetch via COPY_TO

Attach this action at request admission (ingress):

```json
{
  "action_id": "prefetch-session-1",
  "action_type": "vllm.copy_to",
  "action_version": "1.0",
  "payload": {
    "target": ["<gpu block hash hex>", "..."],
    "src": {"tier": "cpu"},
    "dst": {"tier": "gpu"}
  }
}
```

- `target` lists one GPU block-hash hex string per GPU block of the prefix, in order from the start - the same hashes the engine publishes in its `BlockStored` KV events. Hash selectors capture their scope at submission time.
- Only local `cpu -> gpu` copies are supported; `engine_id` must be omitted on both locations. Other source/destination combinations report `unsupported`.
- Execution is request-free: the offloading connector checks feasibility against the free GPU blocks behind a configured reserve (`prefetch_reserve_blocks`), reserves destination blocks, submits the load, and publishes the filled blocks into the GPU prefix cache on completion. A synchronous `applied` result means submitted, not readable; the blocks become usable through the normal KV residency events once the transfer completes.
- Outcomes map to hint results: `ACCEPTED`/`ALREADY_AVAILABLE` -> `applied`, `DEFERRED` (reserve exhausted, source still in flight) -> `deferred`, `MISSING` (target not fully offloaded) -> `rejected`, and disabled or unbound connectors -> `unsupported`. An infeasible submission reserves nothing; callers may resubmit later.
- Enable with `kv_connector_extra_config: {"enable_request_free_prefetch": true}` (optionally `prefetch_reserve_blocks`). Requires prefix caching and a single KV cache group whose block size equals the hash unit.

The prefetch mechanism is an execution primitive, not a policy: it never preempts requests or takes referenced blocks, and an unused prefetched block is evicted before any block a request still references.
