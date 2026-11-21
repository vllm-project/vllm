# Checks by change

Use these routes into the [detailed failure modes](failure-modes.md) during
design and review. Follow the change across sections when it spans boundaries.

## New endpoint, request field, or frontend

- Find route registration, the request model or RPC conversion, middleware,
  and the serving entry point. Follow the field through render, engine request
  construction, and streaming and error responses. Check
  [authorization](failure-modes.md#authorization-and-capability-gates),
  [resource budgets](failure-modes.md#resource-budgets),
  [input invariants](failure-modes.md#input-invariants-and-fault-containment),
  and [error handling](failure-modes.md#sensitive-output-and-error-responses).
- Look for other consumers of the same field and the same underlying handler.
  For batch or pooling routes, budget across the outer request before creating
  child work. For token-in/token-out, render/derender, and scale-out, validate
  caller-supplied intermediates when they re-enter the service.
- Exercise the registered route under its supported configuration. For an
  authenticated route, verify credential handling with the actual mount
  prefix. For a limit, assert that the upstream consumer receives no work
  when admission rejects the input. Check any distinct frontend exposing it.
- For OpenAI- or Anthropic-compatible fields, also use
  [API compatibility guidance](../../check-api-compat/SKILL.md).

## New multimodal model, processor, or media backend

- Trace acquisition through `MediaConnector` and media I/O, then through
  processor overrides, placeholder construction, encoding, and the engine.
  Check Python speech-to-text and Rust chat multimodal loading when the
  change reaches either stack. Check [fetch and decoder policy](failure-modes.md#fetch-filesystem-and-decoder-policy),
  [resource budgets](failure-modes.md#resource-budgets), and
  [input invariants](failure-modes.md#input-invariants-and-fault-containment).
- Compare `get_supported_mm_limits`, `get_mm_max_tokens_per_item`, and model
  profiling with runtime output shapes. Look for resize, padding, frame
  selection, audio resampling, intermediate tensors, and placeholder or
  token expansion that occur after `MediaDecodedBudget` accounts for decoded
  media. Check backend-specific limits and any trusted per-request overrides.
- Exercise the normal serving path with small configured limits. Verify
  minimum-sized input, mixed or repeated media, and the aggregate limit.
  Use a stub decoder or processor that reports its allocations to check
  that materialization stops at the intended boundary. Compare independent
  code paths such as inline and remote media or CPU and GPU decoding as needed.

## Model, tokenizer, quantization, or artifact loading

- Locate the main loader and secondary configs, weights, tokenizer, processor,
  dynamic modules, packed tensors, and fallback reads used by the model.
  Trace trust settings and revision pins to each consumer. Check
  [artifact trust](failure-modes.md#artifact-trust-and-unsafe-interpretation),
  [path confinement](failure-modes.md#fetch-filesystem-and-decoder-policy),
  and [native input invariants](failure-modes.md#input-invariants-and-fault-containment).
- With stub loaders, record the repo, revision, trust choice, and local path
  at the actual calls. Check a missing pinned asset and confirm that
  resolution fails without an unpinned retry. For a new packed format,
  verify that invalid metadata is rejected before invoking the kernel.

## Structured output, templates, reasoning, and tool calls

- Trace raw fields, generated schemas, backend dispatch, and validation.
  Include the format constructed from tool definitions, streaming argument
  parsing, and model-supplied templates. Check
  [parser complexity](failure-modes.md#parser-and-rendering-complexity) and
  [worker budgets](failure-modes.md#resource-budgets).
- Follow parsed results into actual dispatch and the next model invocation.
  For the Responses API, check both `HarmonyContext` and `ParsableContext`
  where applicable. Treat schemas and results from `MCPToolServer` as data
  from the remote service. Check [tool authority](failure-modes.md#authorization-and-capability-gates),
  [data-only interpretation](failure-modes.md#artifact-trust-and-unsafe-interpretation),
  and [model-visible context](failure-modes.md#sensitive-output-and-error-responses).
- Verify the same valid call across buffered and chunked outputs, and a
  malformed or incomplete form in each affected backend. Use stub workers
  and tools to check that only a complete authorized call is dispatched, that
  duplicate completion does not cause another dispatch, and that timeout
  cleanup and call budgets hold across continuations. Verify that tool
  transport credentials stay out of model-visible prompts.

## Engine, scheduler, sampler, or native kernel

- Trace `SamplingParams` or the relevant request representation from the
  frontend through input processing to the model runner, scheduler, or
  native operation. Compare shared and alternate runners where they consume
  the same field. Check
  [input invariants](failure-modes.md#input-invariants-and-fault-containment),
  [resource budgets](failure-modes.md#resource-budgets), and
  [buffer ownership](failure-modes.md#cache-request-and-peer-ownership).
- Identify the actual dimension or ownership map that governs access. Verify
  behavior at the accepted boundary, rejection of unsupported combinations,
  and healthy inference after a rejected request. Test buffer and slot reuse
  with two distinguishable requests when a change touches co-batched state.

## Shared caches, batching, IPC, or disaggregated serving

- Trace cache-key creation, translation into external keys, worker updates,
  and eviction. When reusing media features, inspect the UUID namespace at
  the processor and encoder caches independently of prefix-cache salting.
  For transfers, follow admission through registration, peer selection,
  completion, and cleanup on both ends. Check
  [ownership](failure-modes.md#cache-request-and-peer-ownership),
  [capability gates](failure-modes.md#authorization-and-capability-gates),
  [resource budgets](failure-modes.md#resource-budgets),
  and [safe IPC decoding](failure-modes.md#artifact-trust-and-unsafe-interpretation).
- With stub peers or caches, exercise interleaved requests, cancellation,
  eviction, and duplicate or delayed completion. Check that resulting state
  still belongs to the correct request and pending counts return to baseline.
  For background workers, verify that a failed worker or stuck queue is
  visible to whichever readiness or liveness signal the deployment uses.

## Deployment, configuration, or plugin changes

- Review defaults and opt-ins with the affected frontend and deployment mode.
  Find which process binds the interface, which caller can reach it, and which
  layer enforces access. Include changes in what a proxy routes, which network
  can reach workers, or which directory holds executable or cached data.
  Check [authorization](failure-modes.md#authorization-and-capability-gates)
  and [artifact trust](failure-modes.md#artifact-trust-and-unsafe-interpretation).
- Check option precedence and merged configuration at the consuming layer.
  Exercise the default restricted behavior and an explicitly permitted
  behavior. Compare administrative routes, custom plugin routes, and internal
  RPC endpoints with their intended trust boundaries.

## Errors, logging, metrics, or usage reporting

- Trace validation, exception conversion, response serialization, and
  diagnostics on all affected paths. Check
  [sensitive output](failure-modes.md#sensitive-output-and-error-responses)
  and [resource budgets](failure-modes.md#resource-budgets).
- Use a harmless sentinel value to verify that private fields stay out of
  responses, logs, metrics, and tool prompts as applicable. Check bounded
  error summaries with small configured caps and verify that usage or cache
  counts are populated by the engine.
