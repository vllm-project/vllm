# Top-Tier Push Plan

Primary thesis: Runtime allocator-state accounting can expose fragmentation and residency structure that current inference stacks hide behind generic device allocators.
Honest fallback: Today the runtime can fall back to the standard allocator path with no source-level accounting guarantees.
Next gate: Instrument allocator events and export one stable metric family for fragmentation, allocation latency, and residency bookkeeping.
Missing evidence: End-to-end inference wins or even allocator-path improvements on the target hardware.

Immediate actions:
1. Add instrumentation before proposing any allocator policy.
2. Quantify fragmentation and allocation overhead under realistic serving traces.
3. Keep this line pathology-first until a stable signal emerges.