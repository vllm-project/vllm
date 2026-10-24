# Session prefix reuse

Session prefix reuse uses the existing `session_id` Agent Hint to select one
recent candidate for input tokenization and prompt block hash reuse. Enable
the two stages independently:

```console
vllm serve Qwen/Qwen3-0.6B \
    --tokenizer-revision c1899de289a04d12100db370d81485cdf75e47ca \
    --session-token-cache-mb 32 \
    --session-hash-cache-mb 32
```

Both options default to zero (disabled). Budgets are in MiB per renderer or
EngineCore instance, respectively; multiple processes do not share a budget.
The histories occupy CPU memory and do not retain physical KV blocks. Original
KV prefix lookup and existing scheduler optimizations remain active.

Send the existing `session_id` request field or `X-Session-ID` header. The
existing precedence is request field, header, then `vllm_xargs.session_id`.
Session identity selects a candidate; it does not prove equal content or a
parent/child relationship. Each process keeps the most recently successfully
published candidate per session and cache salt. Same-session requests may run
concurrently. Forks or worker routing can reduce reuse, without relaxing exact
content validation. No new request protocol fields are required.

## Initial support

Token reuse supports text Chat Completions using the audited Qwen3 tokenizer
configuration identified by the Qwen3-0.6B revision above, Transformers 5.19.0
and tokenizers 0.23.3. Other model revisions using the same audited configuration
can pass the same checks. Activation checks the complete backend configuration
and public encoding implementation; a model name alone is insufficient.
The tokenizer must remain immutable. Unsupported configurations or library
versions disable token reuse with a warning and use native tokenization, even
when those library versions satisfy vLLM's installation requirements.

This initial native helper path requires a GIL-enabled CPython build.
If the helper extension is unavailable, the corresponding reuse stage is
disabled with a warning and requests use the existing computation path.

### Boundary contract

The certified configuration extracts context-independent added tokens before
NFC normalization. Their literals do not overlap across a complete marker,
consume neighboring whitespace or require word context. Subsequent regex,
ByteLevel and deterministic BPE processing is local to the separated text.
Consequently an unchanged prefix ending at a complete certified marker has
the same IDs when a suffix changes. Snapshots store only the prompt text and
token IDs. On a partial match, the frontend finds the last complete marker in
the common prefix and locates its token by walking from the nearer end of the
snapshot, where text and token IDs are aligned. Every added-token ID on that
walk must match the markers in the text, otherwise the full prompt is encoded.
Only an added token that cannot overlap another added token and needs no word
context ends a reused prefix; a right-stripping one does so only when the new
suffix does not start with whitespace. Other added tokens are still matched
during the walk.
This relies on added-token IDs being disjoint from ordinary BPE IDs. The pinned
backend fingerprint covers these properties; this argument does not apply
automatically to arbitrary special tokens or other tokenizers.

The frontend reruns the chat template and compares the resulting prompt. It
reuses only exact token prefixes ending at certified added-token boundaries.
Plain character matches and offsets alone are not treated as stability proofs.
Changed tools, thinking settings or template tails are handled through actual
rendered content comparison. Explicit input truncation, padding, offsets,
lowercasing, multimodal/embedding inputs, LoRA configurations and connector
configurations do not use the initial frontend fast path.

Hash reuse requires an existing native block hasher, plain token inputs and
compatible salt/context. It independently compares full token IDs against its
own candidate, reuses complete blocks at the native hash granularity, and
continues the native hash chain. Frontend and EngineCore histories need not
refer to the same request. Resumable inputs, multimodal/embedding inputs, LoRA
and KV connectors are outside initial hash support.

Requests without a session hint or with an incompatible configuration use the
existing computation path for the corresponding stage. Without a reusable
token boundary, the frontend encodes the full prompt; independent hash reuse
may still succeed. Insufficient history capacity prevents saving a new
snapshot without discarding results already computed or reused. A non-reusable
request can still incur bounded history publication work. Disabling token reuse
does not disable independent hash reuse, or vice versa.

## Memory and lifecycle

Simple LRU eviction bounds charged history state, including reservations and
snapshots still held by readers after replacement. Oversized items or requests
that cannot obtain a reservation are not cached. Cache-only token copies require
a reservation of the snapshot size before allocation. Accounting conservatively
charges shared Python objects; it is not an exact process RSS limit. Native
request inputs, outputs, Python allocators and other serving state are separate.
Temporary text-matching workspace is outside the retained-history budget.

Cancelled/failed frontend preprocessing does not publish a partial result.
Cache clearing and shutdown invalidate in-flight publication. EngineCore
prefix-cache reset also clears its CPU history. Restart creates empty histories;
there is no cross-worker sharing or persistence.

Use session IDs within the deployment's existing cache isolation domain and
respect `cache_salt`. Session IDs are not authentication credentials. Histories
do not bypass authorization or existing cache isolation policy.

This is an opt-in CPU optimization. Helper and component microbenchmarks do not
establish HTTP/IPC or GPU serving improvements; measure representative text
requests with the enabled stages and normal renderer concurrency.
