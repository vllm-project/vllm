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
