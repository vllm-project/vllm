# Cache, request, and peer ownership

- **Complete cache identity.** Include inputs that change the cached result or
  its interpretation. Consider model and adapter identity, media shape and
  mode, preprocessing, position or template state, and representation version.
  Frame composite keys unambiguously and preserve hash strength where
  collisions can expose or substitute another request's data.
- **Sharing scope.** Preserve cache namespaces and `cache_salt` through
  continuations, fan-out, serialization, and external stores where promised.
  Trace prefix, processor, and encoder caches separately. Prefix-cache
  salting covers the prefix cache. Verify the media UUID namespace for the
  processor and encoder caches under the documented deployment assumptions.
  [Media UUID security](../../../../../docs/usage/security.md#multimodal-media-uuid-security)
- **Identity reuse.** Associate cache entries, transfer registrations, and
  requests tracked across processes with their owners. Include a generation
  or equivalent lifetime check when identifiers or slots can be reused.
- **Batch and buffer reuse.** Initialize request-owned data before exposing it.
  Clear or remap sampling masks, histories, token and logprob output, scratch
  buffers, and CUDA graph inputs across batch insertion, removal, compaction,
  and retries. Verify independence with distinct requests.
- **Mirrored caches.** Coordinate publication, eviction, and rollback when a
  frontend and worker hold paired cache state. Check missing entries, rejected
  requests, stale handles, and asymmetric eviction.
- **Transfer binding.** Bind peer, direction, request, block set, byte range,
  and descriptor to authorized live state before scheduling, reading,
  writing, or completing a transfer. Verify both producer and consumer paths.
- **Async cleanup.** Release exactly the resources acquired on cancellation,
  retry, rejection, or timeout. Check duplicate and late events against the
  current owner. Reuse blocks or handles once in-flight work can no longer
  access them.
- **Shared adaptation and counters.** Scope mutable parser, estimator,
  scheduler, and accounting state to the expected owner. Verify that one
  request updates another request's behavior only where sharing is intended.

## Published advisory examples

Preserve cache ownership and namespace
through transformations, errors, and eviction.

- [Shared cache identity](https://github.com/vllm-project/vllm/security/advisories/GHSA-rh6f-3x46-j33q)
- [Cache salt through tool continuations](https://github.com/vllm-project/vllm/security/advisories/GHSA-935w-9g4m-p28p)
- [Mirrored cache consistency](https://github.com/vllm-project/vllm/security/advisories/GHSA-ph3r-5jfg-f84f)
