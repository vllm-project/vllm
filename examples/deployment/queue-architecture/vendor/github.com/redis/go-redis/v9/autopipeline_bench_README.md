# Autopipelining benchmarks

These benchmarks validate the goal of autopipelining: batching concurrent
commands into pipelines cuts network round-trips and raises throughput, without
callers writing pipeline code.

Two measurement rules keep every number honest:

- **Throughput is measured on executed commands** — a command is counted only
  after its result has been read (`.Result()` / `.Err()`), never when it is
  merely queued. On the deferred face `ap.Set` returns immediately, so counting
  calls would measure enqueue speed, not throughput.
- **Rates divide by the timed region, not the nominal window.** The
  fixed-duration drivers keep draining whatever was in flight when the deadline
  hit; that drain is part of the timed region, so `ops/sec` cannot be inflated
  by work that finished after the window closed.

## Running

The benchmarks talk to a real Redis on `:6379` (they skip when none answers).
Start one first:

```sh
make docker.start            # or: docker run --rm -p 6379:6379 redis

# the headline three-way throughput comparison:
go test -run '^$' -bench BenchmarkAutoPipelineThroughput -benchtime=1x .

# everything:
go test -run '^$' -bench Benchmark -benchmem -benchtime=1x .
```

The throughput benchmarks run for a fixed wall-clock duration (~3s each) and
report `ops/sec`, so `-benchtime=1x` (one iteration) is correct for them.
**Do not pass a time-based `-benchtime`** (e.g. `-benchtime=5s`): the
fixed-duration drivers ignore `b.N`, so Go's framework would re-run the full
window geometrically trying to fill the time budget.

The per-operation benchmarks (`BenchmarkDispatchPath`,
`BenchmarkAutoPipelineZeroCopy`) are the opposite: their ns/op and allocs/op
are only meaningful at the **default** `-benchtime` — at `-benchtime=1x` the
workers still issue a minimum window each, all billed to a single iteration.
(The "everything" command above also sweeps the repo's other root-package
benchmarks; that is harmless, just broader than this file.)

## The headline benchmark: BenchmarkAutoPipelineThroughput

Three ways to issue the same workload (2000 goroutines), each counting only
executed commands:

1. **Normal** — a plain client; each `Set` is a blocking round-trip. Bounded by
   Redis's non-pipelined ceiling (like `redis-benchmark` without `-P`).
2. **AutoPipelineBlocking** — the blocking face with a parallel-batch config:
   `ap.Set(...)` blocks until executed, the same call shape as a normal client.
   Only one command per caller is in flight, but the flusher batches across the
   2000 callers into deep pipelines.
3. **AutoPipelineWindowed** — the deferred face: each caller submits a window of
   200 commands, then reads the results. Keeps pipelines deepest.

The `WindowedGET` variant repeats (3) with GET instead of SET: SET throughput is
server-bound (Redis's write processing), GET is cheaper on the server, so the
GET number shows the client machinery itself is not the limit.

**Absolute numbers are machine- and load-dependent and vary a lot** — CPU
count, Redis's own ceiling, network path (loopback vs docker veth vs real
network), and noisy neighbors all move them by integer factors. The signal is
the WITHIN-RUN multiplier against the `Normal` baseline measured in the same
environment, plus `allocs/op` (which is exact and stable everywhere):

| variant                 | vs Normal (same run)   |
| ----------------------- | ---------------------- |
| Normal                  | 1x (the baseline)      |
| AutoPipelineBlocking    | ~10x                   |
| AutoPipelineWindowed    | ~25-30x                |
| AutoPipelineWindowedGET | above Windowed (reads) |

As one concrete example: an Apple Silicon laptop with a loopback Redis puts
`Normal` around 80k ops/sec (so ~800k blocking, ~2.5M windowed); a 4-vCPU CI
runner with dockerized Redis lands near half that on the CPU-bound variants —
different absolutes, same multipliers and ordering.

The autopipeline variants use an explicit parallel-batch config
(`MaxBatchSize: 300, MaxConcurrentBatches: 80, Unordered: true`) — **not the
ordered default**. The default (`MaxConcurrentBatches: 1`,
`DefaultAutoPipelineOptions` / `DefaultBlockingAutoPipelineOptions`) serializes
batch execution: blocking usage lands at roughly half the parallel-batch
multiplier, while windowed submission stays well into the tens-of-x even
ordered.

## The other benchmarks

- **BenchmarkIndividualCommands** — plain-client baseline: one blocking
  round-trip per command across GOMAXPROCS workers. Its ns/op is your
  environment's RTT floor; every other number is best read against it.
- **BenchmarkManualPipeline** — hand-built 100-deep `Pipeline().Exec()`,
  sequential: the per-command cost of explicit pipelining (roughly a tenth
  of a round-trip per command). The ceiling autopipelining approaches
  without anyone writing pipeline code.
- **BenchmarkDispatchPath** — the engine's per-command dispatch cost with
  honest `b.N` accounting: ns/op and allocs/op per executed command
  (4 allocs/cmd on the submit path; unordered dispatch roughly halves the
  ordered ns/op), plus the lone-command blocking fast path (~1 RTT).
- **BenchmarkFutureFace** — the typed future face on the ordered default
  config: per-command reads (`InOrder`) vs windowed reads (`Window200`,
  roughly 2x InOrder).
- **BenchmarkAutoPipelineSubmit** — the non-blocking `Submit` entry point,
  windowed, on the ordered default; lands in the same band as
  `FutureFace/Window200`.
- **BenchmarkAutoPipelineZeroCopy** — `GetToBuffer`/`SetFromBuffer` vs regular
  `Get`/`Set` (Set+Get pairs): B/op drops ~10x at 4KiB and ~90x at 64KiB
  (payloads decode into the caller's buffer instead of fresh strings), with
  throughput at parity or better; allocs/op 10 vs 11. The B/op and allocs/op
  ratios are environment-independent.
- **BenchmarkClusterAutoPipelineThroughput** — the same blocking/windowed
  drivers against a local 3-master cluster (slot-routed shard batches keep
  per-node pipelines deep; scales past the standalone numbers in the same
  environment, with a wide run-order-dependent spread). Skips when no
  cluster answers on `:16600-16602`.

## What was deliberately removed

Earlier revisions carried "tuning sweep" benchmarks (batch sizes, flush
delays, buffer sizes) whose numbers were dominated by the configured
`MaxFlushDelay` timer at low parallelism — every swept value reported the same
timer readout, which could only mislead someone tuning from them. They were
removed rather than fixed: `BenchmarkDispatchPath` and the throughput drivers
cover the engine's real knobs. Tune with your own workload shape; the engine's
defaults need no tuning to hit the numbers above.
