# Resource budgets

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

## Published advisory examples

Limit transport reads, media materialization, and
total prompt or media work across the request.

- [HTTP headers](https://github.com/vllm-project/vllm/security/advisories/GHSA-rxc4-3w6r-4v47)
- [Media downloads](https://github.com/vllm-project/vllm/security/advisories/GHSA-p6g9-7v3x-m8mv)
- [Request-wide media budget](https://github.com/vllm-project/vllm/security/advisories/GHSA-5r3c-5x4v-8gjh)
- [Prompt list fan-out](https://github.com/vllm-project/vllm/security/advisories/GHSA-87x5-vmc3-756j)
