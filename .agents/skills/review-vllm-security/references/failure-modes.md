# Detailed security failure modes

Screen the families in [the skill](../SKILL.md), then read the sections that
apply. One operation can need several controls. Remote video input, for
example, crosses fetch and codec boundaries, spends resources during
expansion, and produces tensors subject to model and kernel invariants.

## Resource budgets

- **Transport limits.** Apply body, header, multipart, WebSocket, and RPC
  receive limits at the corresponding transport or reader. Check streamed and
  lengthless input and alternate server configurations. Enforce the cumulative
  byte limit while reading, before a full body is materialized.
- **Parsed input limits.** Bound collection sizes, nesting, and string lengths
  before tokenization, copying, rendering, or task submission. Count outer
  lists as well as inner elements. Inspect empty, omitted, and explicitly
  unlimited configurations.
- **Total fan-out.** Account for total prompts, documents, candidates,
  output sequences, and media items across child requests. Check the products
  of limits that multiply work. Reserve capacity before spawning children
  and define how partial failure unwinds it.
- **Token and sampling costs.** Bound input and output tokens, requested
  logprobs, stop and allow/block lists, and per-token work. Include template
  and multimodal token expansion, pretokenized inputs, and combinations of
  truncation parameters that change which work is performed.
- **Downloaded and decoded media.** Check downloaded bytes, decoded pixel and
  sample counts, frames, and total decoded bytes where each consumes a distinct
  resource. Charge the whole request, including repeated media. Enforce totals
  as actual data arrives when metadata is incomplete or inaccurate.
- **Processor amplification.** Bound post-resize and padded geometry, sampled
  frames, resampling output, derived tensors and patches, and generated media
  tokens before materializing them. Check the largest intermediate and final
  dimensions when accounting starts from source dimensions.
- **Profiling and reservation.** Compare runtime media and batch maxima with
  profiling, encoder cache and graph buffer capacity, and host and device
  memory reservations. Include optional and fallback backends that use
  different representations or allocate in a different address space.
- **Long-lived streams.** Apply cumulative limits to incremental or resumed
  sessions. Bound buffered output and queued input, use backpressure with slow
  consumers, and stop producers when the request closes or is cancelled.
- **Workers and pending work.** Bound outstanding parser jobs, renderer threads,
  network requests, and connector tasks as well as per-job cost. Verify
  behavior at capacity and release permits on submission failure, timeout, and
  cancellation.
- **Retained state.** Limit cache and registry entry counts, bytes, and other
  scarce resources such as file descriptors. Verify eviction and expiry for
  large entries, many tiny or empty entries, and failed or never-established
  sessions.
- **Overrides and parity.** Check where request, model, backend, and operator
  settings merge. Verify that aliases, overrides, fallback loaders, and
  backend-specific work honor the effective policy and budget.

## Input invariants and fault containment

- **Effective value.** Check omitted, null, empty, and default values plus
  field combinations and feature capabilities. Recheck after conversions that
  change the value or unit. Validate finite floats and signed ranges before
  arithmetic, loops, or allocation.
- **Derived arithmetic.** Guard casts and derived seeds, sizes, offsets,
  counts, and index extrema in the consumer's integer width. Verify signedness
  and inclusive or exclusive limits before dispatch to native code.
- **Token IDs.** Trace prompt IDs, sampler lists, stop IDs, speculative and
  recovered IDs, and detokenization input. Validate against the model's usable
  vocabulary and consumer buffer dimensions, including differences between
  tokenizer metadata and the loaded model.
- **Tensor metadata and contents.** Check rank, dimensions, dtype, layout,
  stride, storage span, and supported sparse representations. Verify relations
  among parallel inputs before tensor conversion, device copy, or indexing.
- **Media and model relationships.** Check minimum usable media sizes,
  placeholder counts, offsets and lengths, alignment between media and feature
  tensors, and positional or grid capacity. Verify supported combinations of
  embeddings, pretokenized input, and optional model features.
- **Wire representations.** Apply invariants where HTTP, gRPC, render,
  derender, or disaggregated routes accept engine-shaped data. Check extra
  fields, aliases, mixed typed and untyped values, frontend conversions, and
  native defaults.
- **Shared and native boundaries.** Check native index and storage assumptions
  where real shapes and allocations are known. Return a request-local error for
  invalid input before any fatal assertion, panic, unsafe index, or
  worker-fatal exception. Keep genuine internal invariant failures
  distinguishable from client errors.
- **Admission and rollback.** Validate before registering or mutating shared
  state when possible. On failure after registration, use the normal
  request-failure and rollback path. Verify that rejected work leaves no stale
  metadata and that subsequent inference succeeds.

## Cache, request, and peer ownership

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
  [Media UUID security](../../../../docs/usage/security.md#multimodal-media-uuid-security)
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

## Parser and rendering complexity

- **Source and expansion bounds.** Limit nesting, references, branches,
  alternatives, tool definitions, and generated schema size before compilation.
  Check translated schemas and grammar forms as well as raw input.
- **Invalid and incomplete forms.** Check empty, malformed, recursive, and
  incomplete input on each backend. Confirm preprocessing and validation run
  before compiler or native calls for every supported schema format.
- **Repeated work.** Look for repeated copying and rescanning of growing
  state during rendering, token replacement, incremental parsing, and
  sanitization. Check streaming deltas, unmatched delimiters, nested quotes,
  and immediate rejection paths.
- **Pattern matching cost.** Review regex and grammar complexity on caller
  data and model output, including patterns used in error handling. Prefer
  bounded or linear algorithms where feasible. Check compile and match
  controls on every backend that can consume the input.
- **Timeout behavior.** Bound worker occupancy and work that continues after
  a deadline. Check repeated timeouts, cancelled callers, executor capacity,
  and event-loop responsiveness.
- **Streaming semantics.** Compare buffered parsing with chunks split across
  quotes, escapes, or delimiters. Keep parser state request-local. Preserve
  the distinction between quoted data and tool syntax through dispatch.
- **Executable output boundary.** Keep streaming fragments in parser state
  until a complete call and its required arguments are validated. Emit a call
  for dispatch once, and handle invalid or incomplete terminal output without
  executing it.
- **Template execution.** Apply trust policy, permitted template context, and
  execution limits to model-provided templates and alternate renderers.
  Review template argument preparation as well as the render itself.

## Artifact trust and unsafe interpretation

- **Executable loading.** Carry `trust_remote_code` to the actual dynamic
  import or instantiation in model, tokenizer, processor, and config paths.
  Treat artifact-provided class names and plugin declarations as data until
  the trusted loader authorizes the action.
- **Revision pins.** Track `revision` and `code_revision` separately to each
  relevant file lookup and import. Include secondary config reads, weights,
  tokenizer and processor assets, dynamic code dependencies, and fallback
  resolution. Check failure behavior for missing pinned assets and
  cross-repository references.
- **Fallback trust.** Check fast and fallback loaders, remote and local
  snapshots, dependency helpers, and optional formats. Preserve the operator's
  decision through retry, error recovery, and model-specific wrappers.
- **Deserialization.** Use data-only formats and safe APIs for untrusted
  checkpoints, serialized embeddings, and IPC messages. Check indirect
  deserialization through libraries, shared-memory helpers, and alternative
  readers. Validate metadata before materializing tensors.
- **Data-driven dispatch.** Use declarative parsing and validated registries
  for model configuration and tool arguments. Keep authority to select
  executable loaders, callbacks, modules, or plugins with the trusted code.
- **Packed and local artifacts.** Resolve artifact-relative paths under the
  intended root. Review quantization metadata that governs native dimensions,
  offsets, or allocation. Use the project's private-cache trust assumptions
  when a new path consumes cached files.
- **Dependency guarantees.** Verify loader safeguards across the supported
  versions and paths of the dependency, including legacy and fallback readers.

## Fetch, filesystem, and decoder policy

- **Every acquisition path.** Reuse the shared media connector or equivalent
  controls. Trace sync and async fetches, model or tokenizer hooks, file and
  inline handlers, and secondary reads during decoding or rendering.
- **Effective destination.** Enforce the configured scheme, host, address,
  and redirect policy on the destination actually used by the client. Keep
  validation and fetch URL interpretation consistent. Reapply applicable
  policy to each redirect.
- **Request context.** Scope credentials and headers to their intended
  destination. Keep ambient authorization, shared cookies, and caller session
  state out of unrelated media fetches. Check retries and redirects for
  unintended forwarding.
- **Path confinement.** Apply configured local-media, artifact, and
  shared-storage roots to the resolved path. Check absolute paths, symlinks,
  archive members, and caller or artifact names used to construct paths.
  Validate the target used by the actual open or write.
- **Detected format.** Use the detected media format and nested stream type to
  choose an allowed decoder. Review backend selection, fallbacks, and formats
  that can launch helpers or open another file or network resource.
- **Decoder safeguards.** Preserve pixel, sample, duration, and frame
  protections across decoder choices. Keep time and byte limits active during
  download and decode. Review model initialization for changes to shared
  decoder safety settings.
- **Decode failure.** Return a request error for unusable results, partial
  reads, or unexpected dimensions. Verify that a failed decode cannot return
  a buffer from an earlier request, including on native and hardware backends.

## Authorization and capability gates

- **Effective route.** Inspect registered routes, mount prefixes, methods,
  WebSocket upgrades, middleware order, and the deployed frontend. Verify that
  authorization and routing see consistent paths under supported prefix and
  proxy configurations.
- **Access boundary.** Check API-key coverage for the specific route. Apply
  the documented trusted-network model to internal transports. Restrict
  administrative, debug, and dynamic model-loading actions to their intended
  principals.
- **Browser and credential handling.** Check Host and origin behavior when
  changing browser-facing listeners or local endpoints. Verify that gated
  routes use the existing credential comparison path and test expected
  credential handling.
- **Operator-owned switches.** Check enablement of request templates,
  multimodal overrides, insecure serialization, plugins, and privileged
  tools at the operation that uses them. Include nested overrides, backend
  selection, and continuation paths.
- **Internal authority.** Use typed, server-owned context for peer selection,
  tensor handles, cache transfer state, and internal routing. Review merging
  of generic extra arguments with privileged fields.
- **Tool dispatch.** Resolve tool identity and namespace from trusted
  definitions. Enforce the allowlist at execution and share applicable call
  budget counters across continuations. Check same-name tools and who may
  turn a parsed tool call into an action. Treat remote tool results as data
  with the trust of their source. Keep authorization decisions in trusted
  server state.
- **Tool retries.** Track dispatch and outcome by call identity when execution
  can have side effects. Account for duplicate events and disconnects.
  A timeout can leave the result unknown. Safe retries need an idempotency
  guarantee or a way to verify the original outcome.
- **Sessions and peers.** Bind tool sessions and connector actions to the
  intended principal, instance, or authorized peer. Verify that caller
  headers and IDs cannot take over a session or redirect an action.

## Sensitive output and error responses

- **Safe client errors.** Use structured error codes and safe fields. Keep
  internal paths, usernames, native object representations, stack traces, raw
  request bodies, and credentials out of client-visible errors. Compare HTTP,
  gRPC, streaming, and model-specific wrappers.
- **Error amplification.** Limit how many validation errors are collected,
  how much each field echoes, and the total response size. Account for
  validation, redaction, and serialization costs on rejected requests.
- **Logs and metrics.** Keep secret and high-volume input fields out of
  routine diagnostics. Bound log records and use a fixed vocabulary for
  metric labels derived from methods, paths, or error classes.
- **Model-visible context.** Render only intended tool and prompt fields.
  Keep service credentials, connection headers, and private execution state
  in the corresponding transport or server context.
- **Output provenance.** Verify that returned tokens, logprobs, media, and
  intermediate representations come from this request and initialized storage.
  Derive usage and cached-token counts from the trusted accounting source.
- **Signals from shared resources.** Check timing, cache-hit indicators, and
  operational metadata when changing what callers can learn about shared
  state. Tie guarantees to the documented cache and deployment assumptions.
