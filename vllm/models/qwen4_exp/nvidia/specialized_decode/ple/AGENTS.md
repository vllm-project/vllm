# PLE gather protocol (Qwen4Exp specialized decode)

The PLE layer's row gather at TP > 1 (`gather.py`, used by `../runtime/ple.py`).
The package guide is [../AGENTS.md](../AGENTS.md).

`gather.py` states the invariants; these must hold for any change:

- Same calls everywhere: every rank makes the same gather calls in the same
  order on one stream, with the same ids (the step's n-gram ids, computed from
  tokens every rank holds), so every rank agrees on each row's owner. The
  owner is the only writer of the row's slot in each peer's mailbox; an id in
  no rank's range, or ranges that overlap, break the exchange.
- Generations: thread 0 of each CTA reads the generation word before any
  thread of the CTA stores or polls and arrives on the counter; the last CTA
  to arrive stores the next generation (two alternate) and a zero counter at
  its end. Call `n + 2` reuses call `n`'s slots, so no rank may start call
  `n + 2` before every rank has read and reset them to the empty sentinel
  (BF16 negative zero) in call `n`. Unlike the logits gather, only owners
  publish, so the call's own exchange orders this only when every rank owns
  a row of call `n + 1`; otherwise every rank must run a collective (an
  all-reduce, the boundary kernel) after call `n + 1`. The executor calls
  each gather once a step, and every later decoder layer of the step ends
  with a collective over the TP group (the boundary kernel, its row
  split's reduce-scatter, or an all-reduce).
- Order and liveness: the grid has at most the measured resident CTAs; each
  thread publishes all its owned packs before it polls any, and polls in
  increasing order; the kernel waits for its predecessor before it touches
  anything, and a thread triggers the dependents after its publish phase
  (a CTA counts as triggered at its first thread's trigger; a programmatic
  launch waits for every CTA, so every CTA of the grid is resident by
  then). Nothing on another stream that waits on
  the ranks runs beside it: the PLE prefetch's side stream is joined first.

`test_ple_protocol.py` checks the exchange over every interleaving of small
models with two ownership patterns, owners moving between calls and a rank
that owns nothing, the latter with a collective after each call (injected
defects: one generation, no reset, a pack with no owner, a pack with two
owners, an owner-less rank without the collective) and the residency bound (a
grid of one CTA per pack beyond the slots hangs); `test_ple.py` checks the
kernel against the stock path on four GPUs, under graph replay, with the race
stress of `replay_stress.py` (graphs of three gathers of 4 to 64 tokens, ranks
skewed, replayed in a seeded random order; in half of them a rank owns no row
of the middle call, which an all-reduce follows) and a stall report.
