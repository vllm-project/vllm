# Attention-only KV hole repair

With `kv_load_failure_policy="repair"`, a connector can report a failed
KV span and certify the healthy cache entries that follow it. For a
single full-attention cache group whose missing object covers every model
layer, the scheduler admits the span as a regular prefill chunk. Healthy
prefill and decode requests can share the same model forward. At most one
repair is admitted per batch; batching multiple repairs is not required.

For a hole `[4096, 4224)` and a certified restored extent of 8064 tokens,
the repair row has 128 query tokens at their original positions and an
attention sequence length of 4224. Existing attention kernels read the
healthy prefix and regenerate the missing KV entries. They cannot attend
to the retained suffix, which remains allocated and unpublished until the
repair commits. On success the request resumes at 8064; other batch rows
emit their ordinary outputs. A repair row emits no output token.

Admission uses the ordinary token budget, including the repair compute
surcharge. Insufficient remaining space defers that request while healthy
traffic continues. Transfer-generation checks, exclusive ownership of the
write blocks, per-layer cache-layout checks, and distributed preflight and
commit agreement remain required before the restored suffix is trusted.

Full-depth repairs support the ordinary compiled forward and piecewise
CUDA graphs. The repair preflight and commit checks run outside the graph;
the authorized forward runs between them. Ordinary decode keeps its normal
full-graph dispatch. Partial-layer repairs retain exclusive eager replay.

This path retains the repair policy's single-group, synchronous scheduling,
and parallelism restrictions. Hybrid recurrent state, sliding-window groups,
speculative decoding, and independent repair streams are outside this
implementation's tested scope. A refused mixed-batch worker preflight is an engine integrity error:
the engine stops before advancing ordinary requests rather than silently
claiming the unexecuted batch completed. Ordinary storage misses with valid
repair descriptors do not take that error path.

## Hybrid attention holes with healthy recurrent checkpoints

A one-GPU hybrid with one full-attention group and healthy Mamba/GDN state
groups can repair an attention-only delivery hole. Terminal, generation-bound
coverage must independently certify the later recurrent checkpoint and every
healthy attention block. Missing recurrent-state recovery is explicitly omitted.
Sliding windows, speculation, sessions, multimodal requests and distributed or
asynchronous scheduling remain unsupported for this path.

The parent stays parked with its healthy later checkpoint and attention suffix.
An internal ordinary prefill request replays from an existing healthy local
joint checkpoint before the hole (or initial state) through the hole's end.
It owns fresh attention and recurrent allocations, consumes the normal scheduler
budget, can chunk across forwards, and can share execution with ordinary traffic.
At most one repair request runs per forward. Its prompt has one extra original
token, keeping replay an unfinished prefill without sampling or client output.
Scratch attention and state never enter prefix caching or external STORE.

After successful completion, the cache manager atomically adopts only the
private attention blocks covering the hole, with normal block references.
Healthy attention mappings and the parent's recurrent checkpoint stay unchanged.
Retiring the internal request releases the replay prefix and scratch state.
Abort, preemption or a stale transfer generation cannot publish a replay;
insufficient scratch capacity uses ordinary recompute. A replay that makes no
progress for 32 scheduler steps also releases its retained cache through ordinary
recompute. LMCache refreshes its
physical block mapping before ordinary missing-range STORE and excludes the
healthy recurrent group with existing null-block suppression. No special
storage retry or attention/recurrent kernel is introduced.

This first implementation reuses already-local earlier checkpoints; it does
not fetch an additional older external checkpoint just to shorten replay.
Consequently a cold GPU cache can require replay from token zero through the
hole even though only the hole is retained. This can substantially reduce the
compute savings compared with attention-only models.

Replay can change prefill chunk and batch shapes. Ownership and checkpoint
preservation do not guarantee bitwise regenerated BF16 KV or identical greedy
outputs across different execution shapes. Numerical/model-quality qualification
is separate; ordinary prefix-cache reuse can also change greedy outputs.
