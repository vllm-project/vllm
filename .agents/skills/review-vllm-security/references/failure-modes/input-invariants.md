# Input invariants and fault containment

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

## Published advisory examples

Use the effective model bounds in
each frontend and keep invalid request or compilation errors request-local.

- [Rust and Python token ID bounds](https://github.com/vllm-project/vllm/security/advisories/GHSA-qff2-492f-9fm4)
- [Structured-output error containment](https://github.com/vllm-project/vllm/security/advisories/GHSA-85xf-c7hm-whqw)
