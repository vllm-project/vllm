package redis

import (
	"context"
	"errors"
	"fmt"
	"io"
	"runtime"
	"runtime/debug"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"golang.org/x/sys/cpu"

	"github.com/redis/go-redis/v9/internal"
)

// AutoPipelineOptions configures the autopipelining behavior.
//
// EXPERIMENTAL: this API is subject to change, use with caution.
type AutoPipelineOptions struct {
	// MaxBatchSize is the target batch size: the accumulator stops waiting for
	// more commands once the shard queue reaches it, so a batch flushes promptly
	// instead of lingering. It is a soft threshold, not a hard cap — under heavy
	// concurrent enqueue (or while a flush waits on the concurrency semaphore) the
	// queue can grow past it and execute as a single larger pipeline, which is
	// safe and simply yields a deeper pipeline.
	// Default: 200 (the blocking face's no-options preset,
	// DefaultBlockingAutoPipelineOptions, uses 300).
	MaxBatchSize int

	// MaxBatchBytes, when > 0, caps a batch by APPROXIMATE payload volume: the
	// accumulator stops waiting once the queued commands' argument bytes reach
	// it, so many large values flush as several bounded writes instead of one
	// huge burst (300 x 64KiB is ~19MB written down one connection before any
	// reply is read — enough to stall a constrained link past its write
	// deadline). Like MaxBatchSize it is a soft threshold, not a hard cap.
	// The estimate counts string/[]byte argument lengths plus a small
	// per-argument overhead. Default: 0 (no byte cap).
	MaxBatchBytes int

	// MaxConcurrentBatches is the maximum number of pipeline batches that may
	// execute concurrently.
	//
	// Default: 1, which gives a single ordered command stream — batches execute
	// serially in submit order, so even a windowed caller (submit many, read
	// later) sees strict ordering, while still reaching high throughput via deep
	// pipelines (~3M ops/sec locally).
	//
	// Setting this above 1 runs batches in parallel for maximum throughput, but
	// commands then have NO guaranteed execution order. Because that trades away
	// ordering, it is only allowed together with Unordered: true — otherwise the
	// configuration is rejected (see Validate). This makes the trade-off
	// explicit: you cannot accidentally lose ordering by raising concurrency.
	MaxConcurrentBatches int

	// Unordered must be set to true to allow MaxConcurrentBatches > 1. It is the
	// caller's explicit acknowledgement that parallel batch execution gives up
	// command ordering in exchange for throughput. With the default (false),
	// MaxConcurrentBatches is forced to 1 (an ordered stream) and any value > 1
	// is a configuration error.
	Unordered bool

	// contentSharded is set internally by cluster wiring when commands are
	// routed to shards by content (slot), so same-key commands always share a
	// shard and per-key order holds even with several shards. It exempts that
	// wiring from the NumShards ordering check in newAutoPipeliner. Never set
	// by users (unexported).
	contentSharded bool

	// NumShards is the number of independent queue+flusher shards the
	// autopipeliner runs. 0 (the default) means auto: a single shard, which
	// funnels every caller into one queue so batches stay deep — measured
	// throughput and latency are best with one shard even under heavy
	// goroutine concurrency. Cluster clients default to several slot-routed
	// shards instead, so commands for different nodes queue independently
	// (per-key order still holds: a key's slot always maps to the same
	// shard). Raising NumShards splits the queue: it reduces enqueue-mutex
	// contention but fragments batches, which usually costs far more than the
	// contention saves. Every shard always has at least one concurrency
	// permit, so the effective global batch concurrency is
	// max(NumShards, MaxConcurrentBatches) — and because shards flush
	// concurrently, NumShards > 1 on the deferred (async) face requires
	// Unordered: true (construction fails otherwise).
	NumShards int

	// MaxFlushDelay is the maximum delay after flushing before checking for more commands.
	// A small delay (e.g., 100μs) can significantly reduce CPU usage by allowing
	// more commands to batch together, at the cost of slightly higher latency.
	//
	// Trade-off:
	// - 0 (default): Lowest latency, higher CPU usage
	// - 100μs: Balanced (recommended for most workloads)
	// - 500μs: Lower CPU usage, higher latency
	//
	// Based on benchmarks, 100μs can reduce CPU usage by 50%
	// while adding only ~100μs average latency per command.
	// Default: 0, meaning the flusher applies no coalescing wait — it flushes
	// each batch as soon as the queue is ready and lets in-flight backpressure
	// coalesce concurrent callers (see accumulateBatch). Set a value here to add
	// an explicit accumulation window, trading latency for larger batches / less
	// CPU as described above.
	MaxFlushDelay time.Duration

	// AdaptiveDelay enables smart delay calculation based on queue fill level.
	// When enabled, the delay is automatically adjusted:
	// - Queue ≥75% full: No delay (flush immediately to prevent overflow)
	// - Queue ≥50% full: 25% of MaxFlushDelay (queue filling up)
	// - Queue ≥25% full: 50% of MaxFlushDelay (moderate load)
	// - Queue <25% full: 100% of MaxFlushDelay (low load, maximize batching)
	//
	// This provides automatic adaptation to varying load patterns without
	// manual tuning. Uses integer-only arithmetic for optimal performance.
	// Default: false (use fixed MaxFlushDelay)
	AdaptiveDelay bool
}

// autoPipelinePermitBackstop bounds how long a flush waits for a concurrency
// permit when all are busy. It is only a safety net against a wedged semaphore:
// every permit holder releases it (via defer) and each batch Exec is itself
// bounded by the connection's read/write timeout, so in normal operation a
// permit frees long before this. It is set well above the default ReadTimeout
// and a maintnotifications relaxed window so a legitimately slow in-flight batch
// never makes waiters fail spuriously. The wait deliberately does NOT end on
// Close: commands taken from the queue were already accepted, and Close's
// contract is to flush them (it waits via wg/batchWg), so permit waits run on
// a background context bounded only by this backstop.
const autoPipelinePermitBackstop = 30 * time.Second

// autoPipelineCloseBackstop bounds Close's wait for in-flight dispatches. It
// deliberately carries the same value as the permit backstop but its OWN name:
// the two answer different questions, and this one may want tuning on its own.
//
// Why it is generous rather than snappy: the bound is only ever REACHED when a
// dispatch cannot end by itself — a blocking command with no timeout, or a
// stalled read with ReadTimeout disabled. In every other configuration the
// read timeout ends the dispatch and Close returns the moment it does, well
// under this value. A tighter bound would not speed up healthy shutdowns; it
// would instead make Close report failure while legitimate work is still
// finishing (a large final batch, or a maintnotifications relaxed window
// during a failover), turning a correct slow drain into a spurious error.
const autoPipelineCloseBackstop = 30 * time.Second

// numAutoPipelineShards is the shard-count default used by CLUSTER wiring,
// where commands are routed to shards by slot so different nodes' batches
// queue independently (every shard keeps at least one concurrency permit, so
// several shards can flush to their nodes in parallel regardless of
// MaxConcurrentBatches). It is NOT used for standalone clients: those default
// to one shard (see newAutoPipeliner), because a single deep queue pipelines
// far better than a fragmented one. Deliberately NOT derived from
// MaxConcurrentBatches — coupling shard count to the permit budget silently
// collapsed cluster slot routing to a single shard at the default budget.
func numAutoPipelineShards() int {
	n := runtime.GOMAXPROCS(0)
	if n < 1 {
		n = 1
	}
	const maxShards = 16
	if n > maxShards {
		n = maxShards
	}
	return n
}

// DefaultAutoPipelineOptions returns the default autopipelining configuration.
//
// The default is ordered: MaxConcurrentBatches is 1, so batches execute
// serially in submit order (a single ordered command stream) while still
// reaching high throughput via deep pipelines when callers submit in windows.
// To trade ordering for parallel-batch throughput, set MaxConcurrentBatches > 1
// together with Unordered: true.
//
// EXPERIMENTAL: this API is subject to change, use with caution.
func DefaultAutoPipelineOptions() *AutoPipelineOptions {
	return &AutoPipelineOptions{
		MaxBatchSize:         200,
		MaxConcurrentBatches: 1, // ordered by default
		MaxFlushDelay:        0, // lowest latency; no coalescing wait (batch via in-flight backpressure)
	}
}

// DefaultBlockingAutoPipelineOptions returns the default config for the
// blocking face (Client.AutoPipeline). It uses a single ordered batch stream
// (MaxConcurrentBatches: 1). Counterintuitively this maximizes throughput AND
// minimizes latency for the blocking face: with one batch in flight, callers whose
// commands return while it executes re-enqueue and flush together as the next
// batch, so batches stay deep (a near-continuous, double-buffered pipeline),
// while a lone caller flushes promptly in a single round-trip (no coalescing
// wait — see accumulateBatch). More parallel permits (MaxConcurrentBatches>1) do the
// opposite: each command finds a free permit and flushes on its own before
// others accumulate, collapsing batch size — and throughput — toward one command
// per round-trip while latency rises. For maximum throughput use the async face
// (AsyncAutoPipeline) with a window of in-flight commands (inflight>1); it keeps
// MaxConcurrentBatches: 1 as well.
//
// EXPERIMENTAL: this API is subject to change, use with caution.
func DefaultBlockingAutoPipelineOptions() *AutoPipelineOptions {
	return &AutoPipelineOptions{
		MaxBatchSize:         300,
		MaxConcurrentBatches: 1,
	}
}

// Validate reports whether the configuration is self-consistent. It returns an
// error if MaxConcurrentBatches > 1 without Unordered: true — raising
// concurrency gives up command ordering, so the caller must opt in explicitly.
//
// Validate()==nil does not guarantee construction succeeds: rules that need
// the face (e.g. NumShards>1 requires Unordered on the deferred face) are
// enforced by the AutoPipeline/AsyncAutoPipeline getters. Note also that
// Options.AutoPipelineOptions is validated lazily — on the first getter
// call, not in NewClient.
func (cfg *AutoPipelineOptions) Validate() error {
	if cfg.MaxConcurrentBatches > 1 && !cfg.Unordered {
		return fmt.Errorf("redis: AutoPipelineOptions.MaxConcurrentBatches=%d requires Unordered:true "+
			"(parallel batches do not preserve command ordering); set Unordered:true to allow it, "+
			"or keep MaxConcurrentBatches=1 for an ordered stream", cfg.MaxConcurrentBatches)
	}
	// Reject obviously-wrong negatives so a typo surfaces at construction rather
	// than being silently coerced to a default. Zero is allowed and means "use
	// the default" (MaxBatchSize) or "no delay" (MaxFlushDelay).
	if cfg.MaxBatchSize < 0 {
		return fmt.Errorf("redis: AutoPipelineOptions.MaxBatchSize=%d must be >= 0", cfg.MaxBatchSize)
	}
	if cfg.MaxBatchBytes < 0 {
		return fmt.Errorf("redis: AutoPipelineOptions.MaxBatchBytes=%d must be >= 0", cfg.MaxBatchBytes)
	}
	if cfg.MaxConcurrentBatches < 0 {
		return fmt.Errorf("redis: AutoPipelineOptions.MaxConcurrentBatches=%d must be >= 0", cfg.MaxConcurrentBatches)
	}
	if cfg.MaxFlushDelay < 0 {
		return fmt.Errorf("redis: AutoPipelineOptions.MaxFlushDelay=%s must be >= 0", cfg.MaxFlushDelay)
	}
	if cfg.NumShards < 0 {
		return fmt.Errorf("redis: AutoPipelineOptions.NumShards=%d must be >= 0", cfg.NumShards)
	}
	if cfg.AdaptiveDelay && cfg.MaxFlushDelay <= 0 {
		return fmt.Errorf("redis: AutoPipelineOptions.AdaptiveDelay requires MaxFlushDelay > 0 " +
			"(adaptive delay scales MaxFlushDelay by queue fill; with no MaxFlushDelay it would " +
			"silently disable batch accumulation entirely)")
	}
	return nil
}

// cmdableClient is an interface for clients that support pipelining.
// Both Client and ClusterClient implement this interface. It embeds
// UniversalClient (Cmdable + Process + Do + AddHook + Watch + Subscribe... +
// Close + PoolStats) so the AutoPipeliner can delegate the non-batched surface
// back to the underlying client and itself satisfy UniversalClient.
type cmdableClient interface {
	UniversalClient
	// processPipelineHook is the hook-wrapped []Cmder pipeline entry — the same
	// method Pipeline.Exec is wired to (see Client.Pipeline). The flusher
	// dispatches drained batches through it directly, skipping the per-batch
	// Pipeline construction; hooks/OTel see the identical call.
	processPipelineHook(ctx context.Context, cmds []Cmder) error
	// The async faces additionally dispatch through withProcessPipelineHook /
	// withProcessHook with the base processors as the innermost, so the batch
	// can be completed UNDER the user hooks (results ready the moment exec
	// returns, before hooks unwind). Both *Client and *ClusterClient satisfy
	// these via hooksMixin and their base processors.
	withProcessPipelineHook(ctx context.Context, cmds []Cmder, hook ProcessPipelineHook) error
	hookCount() int
	withProcessHook(ctx context.Context, cmd Cmder, hook ProcessHook) error
	processPipeline(ctx context.Context, cmds []Cmder) error
	process(ctx context.Context, cmd Cmder) error
}

// apBatch is the completion signal shared by every command flushed together.
// Its done channel is closed exactly once, when the batch's pipeline has
// executed. Closing one channel wakes all waiters in a single operation,
// instead of doing one buffered-channel send per command — under high
// concurrency the per-command sends dominated CPU (channel-lock contention and
// one goroutine wake-up apiece).
type apBatch struct {
	done chan struct{}
	// closed makes close() idempotent: on the async faces the dispatch closes
	// the batch at the innermost exec seam (under the user hooks, so a hook
	// reading a result after next() does not block on a channel its own
	// goroutine closes — the #3867 deadlock), while the flusher keeps its
	// deferred close as a panic backstop. Whichever runs first wins.
	closed atomic.Bool
	// dispGid is the goroutine id of the dispatcher while the batch is inside
	// the hook chain (0 otherwise). await() consults it before blocking so a
	// hook on the dispatch goroutine reading a result BEFORE next() gets the
	// not-yet-executed view — what a plain pipeline hook sees — instead of a
	// self-deadlock.
	dispGid atomic.Int64
	// nodeGids registers cluster per-node executor goroutines: the cluster
	// pipeline fans a batch out to one goroutine per node, and each runs the
	// NODE client's own hook chain (OnNewNode hooks — redisotel's tracing
	// lives there), which the single dispGid slot cannot vouch for. A node
	// hook reading a result there would block on a batch that completes only
	// after its own return — reproduced as a permanent wedge with a
	// rediscmd-shaped Err() peek. Guarded by nodeMu; entered/left once per
	// node call, consulted only on the guards' slow path (done still open).
	nodeMu   sync.Mutex
	nodeGids []int64
	// nodeCount mirrors len(nodeGids) so isExecutorGoroutine's fast path can
	// skip the goroutine-id parse and the mutex entirely when nobody is
	// registered — which is every standalone batch, always, and a cluster
	// batch outside its node fan-out window.
	nodeCount atomic.Int32
}

// enterNodeDispatch registers the calling goroutine as an executor of this
// batch for the duration of a cluster node call; the returned func
// unregisters it. Registered goroutines get the same treatment as the
// dispatcher in the accessor guards: result reads return the current view
// instead of self-deadlocking on the batch's own completion signal.
func (b *apBatch) enterNodeDispatch() func() {
	gid := curGoroutineID()
	b.nodeMu.Lock()
	b.nodeGids = append(b.nodeGids, gid)
	b.nodeCount.Store(int32(len(b.nodeGids)))
	b.nodeMu.Unlock()
	return func() {
		b.nodeMu.Lock()
		for i, g := range b.nodeGids {
			if g == gid {
				b.nodeGids[i] = b.nodeGids[len(b.nodeGids)-1]
				b.nodeGids = b.nodeGids[:len(b.nodeGids)-1]
				break
			}
		}
		b.nodeCount.Store(int32(len(b.nodeGids)))
		b.nodeMu.Unlock()
	}
}

// isExecutorGoroutine reports whether the CALLING goroutine is currently
// executing this batch: the flusher/dispatch goroutine or a registered
// cluster node executor. The no-executor fast path (dispGid unset and no
// node executors) is two atomic loads — no goroutine-id parse, no lock. That
// laziness is load-bearing: every blocking-face command and every pre-done
// future passes here once per wait, and an earlier revision that parsed the
// goroutine id and took the mutex unconditionally cost the blocking face 6x
// of its throughput (measured 830k -> 138k ops/sec on a loopback bench).
func (b *apBatch) isExecutorGoroutine() bool {
	disp := b.dispGid.Load()
	if disp == 0 && b.nodeCount.Load() == 0 {
		return false
	}
	gid := curGoroutineID()
	if disp != 0 && disp == gid {
		return true
	}
	if b.nodeCount.Load() == 0 {
		return false
	}
	b.nodeMu.Lock()
	defer b.nodeMu.Unlock()
	for _, g := range b.nodeGids {
		if g == gid {
			return true
		}
	}
	return false
}

// noopUnregister is registerBatchExecutors' zero-batch result, shared so the
// plain-pipeline path stays allocation-free.
var noopUnregister = func() {}

// registerBatchExecutors marks the calling goroutine as an executor of every
// deferred-face batch among cmds (plain pipeline commands carry none) and
// returns the combined unregister. The cluster pipeline calls it around each
// node's hook chain.
func registerBatchExecutors(cmds []Cmder) func() {
	var undo []func()
	var seenFirst *apBatch
	var seenMore map[*apBatch]struct{}
	for _, cmd := range cmds {
		bc, ok := cmd.(interface{ readyBatch() *apBatch })
		if !ok {
			continue
		}
		b := bc.readyBatch()
		if b == nil || b == seenFirst {
			continue
		}
		if seenFirst == nil {
			seenFirst = b
		} else {
			if seenMore == nil {
				seenMore = make(map[*apBatch]struct{}, 2)
			}
			if _, dup := seenMore[b]; dup {
				continue
			}
			seenMore[b] = struct{}{}
		}
		undo = append(undo, b.enterNodeDispatch())
	}
	if len(undo) == 0 {
		return noopUnregister
	}
	return func() {
		for _, u := range undo {
			u()
		}
	}
}

func newAPBatch() *apBatch { return &apBatch{done: make(chan struct{})} }

// close completes the batch exactly once, waking every waiter.
func (b *apBatch) close() {
	if b.closed.CompareAndSwap(false, true) {
		close(b.done)
	}
}

// curGoroutineID parses the goroutine id from runtime.Stack's header
// ("goroutine 123 ["). Called only on paths already paying a dispatch or an
// about-to-block round-trip wait — never on await()'s fast path — so the
// microsecond-scale stack read is noise against the batch RTT.
// armSelfDeadlockGuard reports whether async dispatch should stamp the
// dispatcher's goroutine id on the batches (see apBatch.dispGid) — the
// mechanism that lets a hook on the dispatch goroutine read a command
// without deadlocking on a batch only that goroutine completes: before
// next() it sees the not-yet-executed view, after next() the populated
// results (batches complete only when the whole chain has returned). Armed
// when user hooks exist — without hooks nothing can read a command inside
// the chain — and always on cluster clients, whose node clients may carry
// their own hooks (OnNewNode + AddHook, the redisotel pattern) that
// hookCount() cannot see. NOTE: node-level hooks run on node-worker
// goroutines the gid guard cannot identify, so they must not read command
// results on the async face; the same applies to a goroutine a hook spawns
// and joins before returning. A hook added concurrently with an in-flight
// dispatch misses the guard for that one batch. The guard covers result
// READS only: a hook that ISSUES a command on the same AutoPipeliner and
// synchronously waits for it cannot be saved — the nested command needs the
// dispatch slot the hook chain is holding, and the engine recovers only by
// failing the flush after the permit backstops (see
// autoPipelinePermitBackstop) expire.
func (ap *AutoPipeliner) armSelfDeadlockGuard() bool {
	return ap.pipeliner.hookCount() > 0 || ap.config.contentSharded
}

func curGoroutineID() int64 {
	var buf [64]byte
	n := runtime.Stack(buf[:], false)
	const skip = len("goroutine ")
	var id int64
	for _, c := range buf[skip:n] {
		if c < '0' || c > '9' {
			break
		}
		id = id*10 + int64(c-'0')
	}
	return id
}

// The shard queue stores bare Cmders. The batch a command waits on is the
// shard's curBatch at enqueue time — read once to wire the command's ready
// channel and never needed per-command afterward (the flusher closes the one
// shared batch). Storing []Cmder removes a per-command wrapper allocation.

var queueSlicePool = sync.Pool{
	New: func() interface{} { s := make([]Cmder, 0, 100); return &s },
}

func getQueueSlice(capacity int) []Cmder {
	slice := (*queueSlicePool.Get().(*[]Cmder))[:0]
	if cap(slice) < capacity {
		queueSlicePool.Put(&slice)
		return make([]Cmder, 0, capacity)
	}
	return slice
}

func putQueueSlice(slice []Cmder) {
	if cap(slice) <= 1000 {
		// Zero only the used prefix: elements beyond len are already nil —
		// slices enter the pool fully zeroed (here) and are only appended to
		// afterwards, so the tail invariant holds. Zeroing the whole capacity
		// memclr'd up to 8 KB per flush for small batches on large recycled
		// arrays.
		for i := range slice {
			slice[i] = nil
		}
		queueSlicePool.Put(&slice)
	}
}

// AutoPipeliner automatically batches commands and executes them in pipelines.
// It's safe for concurrent use by multiple goroutines.
//
// AutoPipeliner works by collecting commands from multiple goroutines into a
// shared queue and flushing them as one Redis pipeline when the batch reaches
// MaxBatchSize or a configured coalescing window (MaxFlushDelay) elapses. By
// default there is no window: each batch flushes as soon as the queue is ready
// and concurrent callers coalesce via in-flight backpressure, so a lone command
// flushes in a single round-trip while batches stay deep under load.
//
// This provides significant performance improvements for workloads with many
// concurrent small operations, as it reduces the number of network round-trips.
//
// AutoPipeliner implements the Cmdable interface, so you can use it like a
// regular client. Prefer the typed methods (Set, Get, ...); Do runs OUTSIDE
// the pipeline on a normal connection (see Do).
// AutoPipeline / AsyncAutoPipeline return an error for an invalid config, so check it once:
//
//	ap, err := client.AutoPipeline()
//	if err != nil {
//		return err
//	}
//	ap.Set(ctx, "key", "value", 0)
//	ap.Get(ctx, "key")
//	ap.Close()
//
// Per-command contexts: a command is batched and executed on the AutoPipeliner's
// own long-lived context, NOT the context passed to the command. A per-command
// deadline or cancellation is therefore not honored once the command is queued
// (this is deliberate — a per-batch timer per command would cost a goroutine
// each). Use a plain client for commands that need their own deadline.
// The one exception is a blocking command (readTimeout() != nil, e.g. BLPOP):
// it is never batched and runs directly on the caller's context, which is
// honored as usual.
//
// Retries: like any pipeline, a batch that fails on a network error is retried
// as a whole (up to Options.MaxRetries). If the connection drops after the
// server executed part of the batch, non-idempotent commands (INCR, LPUSH, ...)
// may execute twice. Run commands that must not be retransmitted on a plain
// client, or set MaxRetries: -1.
//
// Lifetime: AutoPipeline() returns a single, client-owned instance shared by all
// callers. Close()ing it stops the shared pipeliner for everyone; a later
// AutoPipeline() call on the client builds a fresh one. Closing the CLIENT also
// stops it, but permanently: the getters then return ErrClosed.
//
// Formatting: String()/%v on a command issued by the deferred face WAITS for
// execution, exactly like Err()/Val()/Result() — formatting reads the result
// fields, and reading them unsynchronized would race the dispatcher populating
// them. The one exception is a hook formatting a command from the batch's own
// dispatch goroutine: that returns the not-yet-executed view instead of
// self-deadlocking. Use Name()/Args() if you need to log a submission without
// waiting for it.
//
// EXPERIMENTAL: this API is subject to change, use with caution.
type AutoPipeliner struct {
	cmdable // Embed cmdable to get all Redis command methods

	pipeliner cmdableClient
	config    *AutoPipelineOptions
	// blocking selects how the typed command surface (Set, Get, ...) behaves:
	// when true the command call itself blocks until the command has executed
	// (drop-in, synchronous shape); when false the call returns immediately and
	// the result accessors (Val/Result/Err) block. See AutoPipeline (blocking)
	// vs AsyncAutoPipeline (deferred).
	blocking bool

	// Sharded command queues. Each shard has its own queue, mutex and flusher
	// goroutine, so enqueues from many goroutines spread across shards instead
	// of all contending on a single mutex and being drained by a single
	// flusher. Commands are assigned to shards round-robin; per-goroutine
	// ordering is still guaranteed because Do blocks for each command's result
	// before issuing the next one.
	shards []*apShard
	next   atomic.Uint32 // round-robin shard selector
	// shardFn, when set, picks a command's shard from its content (cluster mode
	// sets it to route by slot so all commands for one node land in the same
	// shard's batch — keeping per-node pipelines deep instead of splitting every
	// batch across nodes). When nil, commands are assigned round-robin.
	shardFn func(Cmder) int

	// preflight, when set, can reject a command at submit time, before it is
	// enqueued or dispatched (cluster mode refuses fan-out-policy commands
	// that cannot ride a pipeline, so one caller's command cannot poison a
	// merged batch). The returned error is set on the command.
	preflight func(ctx context.Context, cmd Cmder) error

	// mustDivert, when set, forces a command off the batching path even though
	// it is otherwise batchable — cluster mode uses it for commands whose
	// routing is NOT slot-derived (ReqSpecial, e.g. FT.CURSOR READ, which is
	// sticky to the node that owns the cursor). Batched, mapCmdsByNode would
	// route them by slot and reach the wrong shard; diverted, they go through
	// Client/ClusterClient.Process and keep their special routing.
	mustDivert func(ctx context.Context, cmd Cmder) bool

	// sharedClosed, when non-nil, is the owning client's pool-set closed flag
	// (shared across WithTimeout clones). The getters refuse to build a fresh
	// pipeliner once it is set; this reference makes an ALREADY-built
	// pipeliner refuse new work too — without it, a clone's Close would leave
	// a cached pipeliner accepting enqueues against closed pools, failing
	// them one dispatch at a time instead of with ErrClosed at submit.
	sharedClosed *atomic.Bool

	// expectedArrivals counts how many commands the engine expects to arrive
	// at any moment: a completed batch of N≥2 commands wakes its N waiters
	// together, and in a closed loop each immediately submits its next command
	// — so completion announces N expected arrivals, and every enqueue accounts
	// for one. The default coalescing wait (awaitExpectedArrivals) holds the
	// flusher while arrivals are still expected, so the whole wakeup wave
	// flushes as one deep pipeline — an exact count, not a smoothed estimate,
	// which cannot ratchet into fragmentation. Single-command batches announce
	// nothing, so a lone caller and open-loop traffic never wait. May
	// transiently go negative (arrivals nobody announced); readers clamp to
	// zero. Pipeliner-global, not per-shard: cluster routing may land a
	// follow-up on a different shard than the batch that woke its caller.
	expectedArrivals atomic.Int64

	// execEWMA is an exponentially-weighted moving average (alpha 1/8) of
	// batch execution time in nanoseconds — the engine's own view of the
	// server round-trip. It scales awaitExpectedArrivals's silence fallback so a
	// wave staggered by scheduling on a slow link is not split mid-landing. Updates
	// are racy read-modify-writes by design: losing an occasional sample is
	// harmless for a smoothing heuristic. 0 means "no sample yet".
	execEWMA atomic.Int64

	// Lifecycle
	ctx     context.Context
	cancel  context.CancelFunc
	wg      sync.WaitGroup // Tracks flusher goroutines
	batchWg sync.WaitGroup // Tracks batch execution goroutines
	// divertWg tracks the goroutines that execute DIVERTED commands (blocking
	// and connection-hostile ones, which never enter a batch). Close waits on
	// it exactly like batchWg so a diverted command's pooled connection is not
	// left in flight after Close returns — bounded, see Close.
	//
	// divertMu serializes "observe not-closed, then register" against Close's
	// "mark closed, then wait": without it a diverted command could pass the
	// closed check, Close could see a zero counter and return, and only then
	// would the goroutine register — leaving an accepted command holding a
	// pooled connection past Close (and racing WaitGroup Add against Wait).
	divertMu sync.Mutex
	divertWg sync.WaitGroup
	closed   atomic.Bool
}

// apShard is one queue + flusher. Its fields are touched only by enqueuing
// goroutines (under mu) and by its own single flusher goroutine.
// apEnqueueStripes is how many enqueue stripes a shard runs when striping is
// safe (unordered configs, and every blocking-face shard — a blocking caller
// waits for each command, so stripes cannot reorder its stream). The
// enqueue mutex is the hottest lock in the engine (128 concurrent callers on
// one shard spend ~half their CPU in lock slow paths); striping the queue
// spreads that contention while the flusher still drains every stripe into ONE
// merged pipeline, so batches stay deep. Ordered shards always use a single
// stripe: with several stripes a caller's consecutive commands can land in
// stripes on opposite sides of an in-progress drain and execute out of order.
const apEnqueueStripes = 8

// apStripe is one striped slice of a shard's enqueue queue. Each stripe has
// its own batch-completion signal so a drain can take stripes one lock at a
// time; every batch taken in one drain completes together after the merged
// pipeline executes. Padded so neighbouring stripes' mutexes do not share a
// cache line.
type apStripe struct {
	mu       sync.Mutex
	queue    []Cmder
	queueLen atomic.Int32
	// queueBytes approximates the queued commands' payload volume; maintained
	// only when MaxBatchBytes is configured (see cmdApproxBytes).
	queueBytes atomic.Int64
	curBatch   *apBatch // completion signal for currently-queued cmds
	// Pad each stripe onto its own cache line(s). Without it, one stripe's hot
	// fields (queueLen/curBatch) share a cache line with the NEXT stripe's
	// contended mutex, so a lock-free counter bump on stripe i invalidates the
	// line a different core is trying to lock stripe i+1 on — false sharing
	// that measured ~16x on a contended microbenchmark. cpu.CacheLinePad is
	// sized per GOARCH (64 B on x86-64/arm64, 128 B on ppc64, 256 B on s390x),
	// so this is correct on every target rather than a hand-tuned constant.
	_ cpu.CacheLinePad
}

type apShard struct {
	ap *AutoPipeliner

	next    atomic.Uint32           // round-robin stripe pick (unordered mode)
	stripes []apStripe              // 1 stripe when ordered, apEnqueueStripes when Unordered
	notify  chan struct{}           // buffered (cap 1) enqueue wake-up
	sem     *internal.FIFOSemaphore // per-shard concurrent-batch budget

	// inFlight counts this shard's dispatched-but-unfinished batches. When it
	// is zero and no arrivals are expected, the shard is idle and a
	// new command flushes immediately; when batches are in flight, arrivals
	// are mid-stream and the flusher holds them briefly to coalesce (see
	// awaitExpectedArrivals).
	inFlight atomic.Int32
}

// stripe picks the enqueue stripe for the next command: the single stripe in
// ordered mode (preserving strict FIFO), round-robin in unordered mode.
func (s *apShard) stripe() *apStripe {
	if len(s.stripes) == 1 {
		return &s.stripes[0]
	}
	return &s.stripes[s.next.Add(1)%uint32(len(s.stripes))]
}

// getOrCreateAutoPipeliner is the shared caching protocol behind the four
// AutoPipeline/AsyncAutoPipeline getters (Client and ClusterClient, each
// face): return the cached live instance, refuse on a closed client, or build
// and cache a new one. The caller supplies its cached-slot pointer, its
// closed flag (both guarded by the mutex), the explicit-config override, the
// fallback config, and a build closure (the cluster one wraps
// clusterAutoPipelineOptions and installs slot sharding).
func getOrCreateAutoPipeliner(
	mu *sync.Mutex,
	slot **AutoPipeliner,
	closed *bool,
	sharedClosed *atomic.Bool,
	override *AutoPipelineOptions,
	fallback func() *AutoPipelineOptions,
	build func(*AutoPipelineOptions) (*AutoPipeliner, error),
) (*AutoPipeliner, error) {
	mu.Lock()
	defer mu.Unlock()
	// closed covers THIS wrapper's Close; sharedClosed covers the shared
	// pools closing through ANY sharer (e.g. a WithTimeout clone falling
	// through to baseClient.Close) — a fresh pipeliner against closed pools
	// would leak flushers that error forever.
	if *closed || (sharedClosed != nil && sharedClosed.Load()) {
		return nil, ErrClosed
	}
	if *slot != nil && !(*slot).closed.Load() {
		return *slot, nil
	}
	cfg := override
	if cfg == nil {
		cfg = fallback()
	}
	ap, err := build(cfg)
	if err != nil {
		return nil, err
	}
	// Thread the shared pool-set closed flag into the pipeliner so an
	// ALREADY-cached instance also refuses enqueues once any sharer closes
	// the pools (the check above only protects fresh builds).
	ap.sharedClosed = sharedClosed
	*slot = ap
	return ap, nil
}

// newAutoPipeliner builds an autopipeliner in either blocking or deferred mode.
// It is unexported on purpose: the public entry points are
// Client/ClusterClient.AutoPipeline and AsyncAutoPipeline, which also install
// cluster slot-sharding. Constructing one directly would skip that wiring and
// give a *ClusterClient degraded (cross-node) batching.
func newAutoPipeliner(pipeliner cmdableClient, config *AutoPipelineOptions, blocking bool) (*AutoPipeliner, error) {
	if config == nil {
		config = DefaultAutoPipelineOptions()
	} else {
		// Copy so default-filling below doesn't mutate the caller's struct — the
		// same *AutoPipelineOptions may be shared across clients (e.g. a reused
		// Options.AutoPipelineOptions), and callers may inspect it afterward.
		cfgCopy := *config
		config = &cfgCopy
	}

	// Validate BEFORE default-filling: Validate treats zero as "use the
	// default" but rejects negatives, and coercing first would silently
	// swallow a negative typo the documented contract promises to error on.
	if err := config.Validate(); err != nil {
		return nil, err
	}

	// Apply defaults for zero values
	if config.MaxBatchSize <= 0 {
		config.MaxBatchSize = 200
	}

	if config.MaxConcurrentBatches <= 0 {
		// Default to an ordered single stream. Callers raise this (with
		// Unordered:true) to opt into parallel-batch throughput.
		config.MaxConcurrentBatches = 1
	}

	// NumShards > 1 on the deferred (async) face distributes commands
	// round-robin across shards that flush concurrently, so submit order is
	// not preserved — require the explicit Unordered opt-in, exactly like
	// MaxConcurrentBatches > 1. The blocking face is exempt (each caller waits
	// per command, and Submit is rejected there), as is cluster slot sharding
	// (contentSharded: same-key commands always land in the same shard, so
	// per-key order holds).
	if config.NumShards > 1 && !config.Unordered && !blocking && !config.contentSharded {
		return nil, fmt.Errorf(
			"redis: AutoPipelineOptions.NumShards=%d requires Unordered:true on the deferred (async) face "+
				"(commands are distributed round-robin across shards, which flush concurrently and do not preserve submit order)",
			config.NumShards)
	}

	ctx, cancel := context.WithCancel(context.Background())

	ap := &AutoPipeliner{
		pipeliner: pipeliner,
		config:    config,
		blocking:  blocking,
		ctx:       ctx,
		cancel:    cancel,
	}

	// Route the typed command surface. Blocking: the command call blocks until
	// executed (synchronous drop-in shape). Deferred: the call returns at once
	// and the result accessors block until the batch executes.
	if blocking {
		ap.cmdable = ap.processBlocking
	} else {
		ap.cmdable = ap.processAsync
	}

	// Pick the shard count. NumShards=0 (auto) means ONE shard: a single deep
	// queue outperforms a sharded one because batches stay large — sharding by
	// core count coupled batch fragmentation to MaxConcurrentBatches and
	// collapsed pipelining (measured: 16 shards cut async throughput ~4x and
	// tripled latency versus one shard at the same permit count). Cluster
	// wiring passes an explicit NumShards so slot-routed shards keep each
	// batch on one node.
	nShards := config.NumShards
	if nShards <= 0 {
		nShards = 1
	}
	// Split the concurrent-batch budget across shards so each shard has its own
	// semaphore. A single shared semaphore became a contention point once the
	// per-shard queue mutexes were no longer the bottleneck. Integer division
	// drops a remainder, so hand the leftover permits to the first shards: the
	// per-shard permits then sum to exactly MaxConcurrentBatches.
	perShard := config.MaxConcurrentBatches / nShards
	remainder := config.MaxConcurrentBatches % nShards
	if perShard < 1 {
		// Budget smaller than the shard count: give every shard one permit so
		// each flusher can still make progress. The sum then exceeds the
		// configured budget, which is unavoidable with per-shard semaphores.
		perShard = 1
		remainder = 0
	}
	ap.shards = make([]*apShard, nShards)
	for i := range ap.shards {
		permits := perShard
		if i < remainder {
			permits++
		}
		// Stripe when reordering is impossible or waived: a BLOCKING caller
		// waits for each command before issuing its next, so its per-goroutine
		// order holds no matter which stripe each command lands in; the async
		// face may only stripe when the user set Unordered. The remaining case
		// (async, ordered) keeps one stripe to preserve strict submit order.
		nStripes := 1
		if config.Unordered || blocking {
			nStripes = apEnqueueStripes
		}
		s := &apShard{
			ap:      ap,
			notify:  make(chan struct{}, 1),
			stripes: make([]apStripe, nStripes),
			sem:     internal.NewFIFOSemaphore(int32(permits)),
		}
		for j := range s.stripes {
			s.stripes[j].queue = getQueueSlice(config.MaxBatchSize)
			s.stripes[j].curBatch = newAPBatch()
		}
		ap.shards[i] = s
		ap.wg.Add(1)
		go s.flusher()
	}

	return ap, nil
}

// Do executes a raw command on a NORMAL connection, outside the pipeline.
// Arbitrary command names can carry connection state (SELECT, MULTI, SUBSCRIBE,
// CLIENT ...) or block the connection (BLPOP ...); batching those onto a shared
// pipeline connection would silently poison it for every later batch, or stall
// unrelated commands. (Submit enforces the same rule for raw Cmders: names in
// the connection-hostile set are diverted off the pipeline automatically.)
// The typed surface (ap.Set, ap.Get, ...) is safe by
// construction and IS batched — prefer it. Do carries the same caveats as
// Client.Do: a stateful command still affects the (normal, non-pipeline)
// pooled connection it runs on. Do keeps each face's call shape: on
// a blocking autopipeliner the call blocks until the command has executed; on a
// deferred (async) one it returns immediately and the command's result
// accessors (Err/Val/Result) block until it completes.
func (ap *AutoPipeliner) Do(ctx context.Context, args ...interface{}) *Cmd {
	cmd := NewCmd(ctx, args...)
	if len(args) == 0 {
		cmd.SetErr(errDoNoArgs)
		return cmd
	}
	if ap.isClosed() {
		cmd.SetErr(ErrClosed)
		return cmd
	}

	// Both faces go through runOutsidePipeline: it applies the divert
	// registration gate, so Close cannot conclude "nothing in flight" while an
	// accepted raw command — a blocking one on the blocking face runs inline on
	// the caller's goroutine — is still holding a pooled connection.
	_ = ap.runOutsidePipeline(ctx, cmd)
	return cmd
}

// runOutsidePipeline executes an escape-hatch command (Do, DoRaw,
// DoRawWriteTo) on a normal pooled connection, outside the batching engine,
// following the face's call shape. Blocking face: synchronous Process.
// Deferred face: returns-immediately — the command runs on a background
// goroutine and a ready batch makes its result accessors block until it
// completes. The batch completes at the innermost seam (under the user
// hooks) so a ProcessHook reading the result cannot self-deadlock; the
// deferred close is the panic backstop. Tracked by divertWg under divertMu,
// so Close waits for accepted diverted work (bounded — see Close) instead of
// returning while it still holds a pooled connection.
func (ap *AutoPipeliner) runOutsidePipeline(ctx context.Context, cmd Cmder) *apBatch {
	if ap.blocking {
		// The blocking face runs it inline, so the caller's own goroutine holds
		// the connection; still take the gate so Close cannot decide "nothing
		// in flight" while this command is executing.
		ap.divertMu.Lock()
		if ap.isClosed() {
			ap.divertMu.Unlock()
			cmd.SetErr(ErrClosed)
			return completedBatch
		}
		ap.divertWg.Add(1)
		ap.divertMu.Unlock()
		defer ap.divertWg.Done()
		_ = ap.pipeliner.Process(ctx, cmd)
		return completedBatch
	}
	// Register under divertMu with a closed re-check, so registration and the
	// close transition cannot interleave (see the divertMu comment). A command
	// that loses the race is rejected here rather than running after Close.
	// The gate comes BEFORE setReady: publishing the fresh batch first and then
	// rejecting would leave the command gated on a batch nobody ever closes,
	// hanging every accessor.
	ap.divertMu.Lock()
	if ap.isClosed() {
		ap.divertMu.Unlock()
		cmd.SetErr(ErrClosed)
		cmd.setReady(completedBatch)
		return completedBatch
	}
	b := newAPBatch()
	cmd.setReady(b)
	ap.divertWg.Add(1)
	ap.divertMu.Unlock()
	go func() {
		defer ap.divertWg.Done()
		defer b.close()
		defer recoverDispatchPanic([]Cmder{cmd})
		if ap.armSelfDeadlockGuard() {
			b.dispGid.Store(curGoroutineID())
		}
		// A hook that returns nil WITHOUT calling next has short-circuited
		// SUCCESSFULLY (it served the command itself); plain Client hooks may do
		// that, so nothing here synthesizes an error for it — see dispatchCmds.
		err := ap.pipeliner.withProcessHook(ctx, cmd, func(ctx context.Context, cmd Cmder) error {
			return ap.pipeliner.process(ctx, cmd)
		})
		// The chain's final verdict, exactly like Client.Process — recorded
		// before the deferred close wakes the reader, so short-circuits,
		// post-next rewrites and suppressions are all honored.
		cmd.SetErr(err)
	}()
	return b
}

// DoRaw mirrors Do for raw RESP access: AutoPipeliner embeds cmdable, so
// without this override DoRaw would ride the batching engine — but raw
// commands carry Do's caveats and DoRawWriteTo-style streaming must not run
// inside a shared batch's reply loop. Runs outside the pipeline, following
// the face's call shape (see Do).
func (ap *AutoPipeliner) DoRaw(ctx context.Context, args ...interface{}) *RawCmd {
	cmd := NewRawCmd(ctx, args...)
	if len(args) == 0 {
		cmd.SetErr(errDoNoArgs)
		return cmd
	}
	if ap.isClosed() {
		cmd.SetErr(ErrClosed)
		return cmd
	}
	_ = ap.runOutsidePipeline(ctx, cmd)
	return cmd
}

// DoRawWriteTo mirrors Do for streamed raw RESP access (see DoRaw). On the
// deferred face the write to w happens when the command executes; use the
// result accessors (Err/Written) to wait before reading w.
func (ap *AutoPipeliner) DoRawWriteTo(ctx context.Context, w io.Writer, args ...interface{}) *RawWriteToCmd {
	cmd := NewRawWriteToCmd(ctx, w, args...)
	if len(args) == 0 {
		cmd.SetErr(errDoNoArgs)
		return cmd
	}
	if ap.isClosed() {
		cmd.SetErr(ErrClosed)
		return cmd
	}
	_ = ap.runOutsidePipeline(ctx, cmd)
	return cmd
}

// Process queues a command for autopipelined execution, following the
// autopipeliner's mode like the typed methods and Do: on a blocking
// autopipeliner the call blocks until the command has executed; on a deferred
// (async) one it returns immediately and reading the command's result
// (Val/Result/Err) blocks until its batch is flushed.
func (ap *AutoPipeliner) Process(ctx context.Context, cmd Cmder) error {
	return ap.cmdable(ctx, cmd)
}

// The methods below complete the UniversalClient surface by delegating to the
// underlying client. They are NOT autopipelined — pub/sub, transactions (Watch),
// hooks, Do and pool stats cannot be batched — so an AutoPipeliner used as a
// UniversalClient batches only the typed data commands; everything here runs on
// the underlying client exactly as it would there.
//
// Note on lifecycle: Close() (defined elsewhere) closes the AUTOPIPELINER —
// drains in-flight batches and stops flushers — but does NOT close the
// underlying client, whose lifecycle is owned by whoever created it.

// AddHook adds a hook to the underlying client. Autopipelined batches are hooked
// too, since dispatch goes through the hook-wrapped pipeline entry.
func (ap *AutoPipeliner) AddHook(hook Hook) { ap.pipeliner.AddHook(hook) }

// The four commands below have CLUSTER-WIDE overrides on ClusterClient
// (DBSize sums every master, the Script commands fan out to every shard).
// The embedded generic cmdable would route them as ordinary keyless commands
// to one picked shard — partial results, scripts missing on other shards —
// so they delegate to the underlying client instead of batching. On a
// standalone client the delegation is semantically identical to the generic
// path; these are rare admin/script-management commands, not data-path.

// DBSize delegates to the underlying client (cluster-wide sum on ClusterClient).
func (ap *AutoPipeliner) DBSize(ctx context.Context) *IntCmd {
	return ap.pipeliner.DBSize(ctx)
}

// ScriptLoad delegates to the underlying client (loads every shard on ClusterClient).
func (ap *AutoPipeliner) ScriptLoad(ctx context.Context, script string) *StringCmd {
	return ap.pipeliner.ScriptLoad(ctx, script)
}

// ScriptFlush delegates to the underlying client (flushes every shard on ClusterClient).
func (ap *AutoPipeliner) ScriptFlush(ctx context.Context) *StatusCmd {
	return ap.pipeliner.ScriptFlush(ctx)
}

// ScriptExists delegates to the underlying client (ANDs results across shards
// on ClusterClient).
func (ap *AutoPipeliner) ScriptExists(ctx context.Context, hashes ...string) *BoolSliceCmd {
	return ap.pipeliner.ScriptExists(ctx, hashes...)
}

// HImportPrepare, HImportDiscard and HImportDiscardAll are the remaining
// cluster-wide overrides (see the delegation note above): ClusterClient fans
// them out to every master and updates the shared fieldset registry, so
// running them on a single routed node would let a later HImportSet for a key
// on another master fail with "no such fieldset". TestAPDelegatesClusterWideOverrides
// fails if a future ClusterClient override is added without a delegate here.
func (ap *AutoPipeliner) HImportPrepare(ctx context.Context, fieldsetName string, fields ...string) *StatusCmd {
	return ap.pipeliner.HImportPrepare(ctx, fieldsetName, fields...)
}

func (ap *AutoPipeliner) HImportDiscard(ctx context.Context, fieldsetName string) *IntCmd {
	return ap.pipeliner.HImportDiscard(ctx, fieldsetName)
}

func (ap *AutoPipeliner) HImportDiscardAll(ctx context.Context) *IntCmd {
	return ap.pipeliner.HImportDiscardAll(ctx)
}

// Watch runs a transactional function on the underlying client (not batched).
func (ap *AutoPipeliner) Watch(ctx context.Context, fn func(*Tx) error, keys ...string) error {
	return ap.pipeliner.Watch(ctx, fn, keys...)
}

// Subscribe opens a pub/sub on the underlying client (not batched — pub/sub
// needs a dedicated connection).
func (ap *AutoPipeliner) Subscribe(ctx context.Context, channels ...string) *PubSub {
	return ap.pipeliner.Subscribe(ctx, channels...)
}

// PSubscribe opens a pattern pub/sub on the underlying client (not batched).
func (ap *AutoPipeliner) PSubscribe(ctx context.Context, channels ...string) *PubSub {
	return ap.pipeliner.PSubscribe(ctx, channels...)
}

// SSubscribe opens a sharded pub/sub on the underlying client (not batched).
func (ap *AutoPipeliner) SSubscribe(ctx context.Context, channels ...string) *PubSub {
	return ap.pipeliner.SSubscribe(ctx, channels...)
}

// PoolStats returns the underlying client's connection pool statistics.
func (ap *AutoPipeliner) PoolStats() *PoolStats { return ap.pipeliner.PoolStats() }

// AutoPipeline delegates to the underlying client, which returns its cached
// autopipeliner (typically this same instance). Present to satisfy the
// UniversalClient surface.
func (ap *AutoPipeliner) AutoPipeline() (*AutoPipeliner, error) {
	return ap.pipeliner.AutoPipeline()
}

// AutoPipelineWithOptions delegates to the underlying client.
func (ap *AutoPipeliner) AutoPipelineWithOptions(config *AutoPipelineOptions) (*AutoPipeliner, error) {
	return ap.pipeliner.AutoPipelineWithOptions(config)
}

// AsyncAutoPipeline delegates to the underlying client. Present to satisfy the
// UniversalClient surface.
func (ap *AutoPipeliner) AsyncAutoPipeline() (*AutoPipeliner, error) {
	return ap.pipeliner.AsyncAutoPipeline()
}

// AsyncAutoPipelineWithOptions delegates to the underlying client.
func (ap *AutoPipeliner) AsyncAutoPipelineWithOptions(config *AutoPipelineOptions) (*AutoPipeliner, error) {
	return ap.pipeliner.AsyncAutoPipelineWithOptions(config)
}

// AutoFuture is the handle returned by Submit. Call Wait (or Result on the
// command after Wait) once the result is needed; it blocks only until the
// command's batch has executed.
type AutoFuture struct {
	cmd   Cmder
	batch *apBatch
}

// Wait blocks until the submitted command has executed, then returns its error.
// The zero AutoFuture (no submitted command) returns an error rather than
// panicking.
func (f AutoFuture) Wait() error {
	if f.batch == nil {
		if f.cmd != nil {
			return f.cmd.Err()
		}
		return errZeroAutoFuture
	}
	select {
	case <-f.batch.done:
	default:
		// Same self-deadlock guard as baseCmd.await(): a pipeline hook on
		// the batch's own dispatch goroutine waiting a future pre-next()
		// would block a channel only its goroutine can close. Give it the
		// not-yet-executed view instead.
		if f.batch.isExecutorGoroutine() {
			return f.cmd.rawErr()
		}
		<-f.batch.done
	}
	return f.cmd.Err()
}

// WaitContext is like Wait but stops waiting when ctx is done. The command
// still executes and its result remains readable once its batch completes —
// ctx abandons only this wait, it does not cancel the command (per-command
// contexts are not honored after enqueue; see the AutoPipeliner doc).
//
// After a ctx error the result may simply not be there YET: the batch is
// still in flight and may populate the command at any moment, so do not read
// Cmd()'s value or error directly — that races the executing batch. Call Wait
// (or WaitContext with a fresh context) again; once it returns a non-context
// error, the command's result is complete and safe to read.
func (f AutoFuture) WaitContext(ctx context.Context) error {
	if f.batch == nil {
		if f.cmd != nil {
			return f.cmd.Err()
		}
		return errZeroAutoFuture
	}
	select {
	case <-f.batch.done:
		return f.cmd.Err()
	default:
		if f.batch.isExecutorGoroutine() {
			return f.cmd.rawErr() // see Wait: dispatch-goroutine self-deadlock guard
		}
	}
	select {
	case <-f.batch.done:
		return f.cmd.Err()
	case <-ctx.Done():
		return ctx.Err()
	}
}

// Cmd returns the underlying command (call Wait first before reading results).
func (f AutoFuture) Cmd() Cmder { return f.cmd }

// outsidePipelineCommands lists commands that must never ride a SHARED
// pipeline connection. SHUTDOWN terminates the server before replying (its
// batchmates would all fail with EOF and the batch would retry against a
// dead server); MONITOR rebinds the connection into a monitor stream,
// desyncing every reply behind it; the rest change per-connection state
// (database, auth, protocol, transaction, subscription mode) that would
// leak to every unrelated caller sharing the pipeline conn afterwards. The
// typed surface cannot produce most of the stateful ones (they live on
// statefulCmdable) — but ReadOnly/ReadWrite ARE on cmdable, and raw
// Submit/Do accept any Cmder. Diverted commands execute directly on their
// own pooled connection — the same semantics (including the same footguns)
// as plain Client.Do.
var outsidePipelineCommands = map[string]struct{}{
	"shutdown": {}, "monitor": {},
	"select": {}, "auth": {}, "hello": {}, "reset": {}, "quit": {},
	"multi": {}, "exec": {}, "discard": {}, "watch": {}, "unwatch": {},
	"subscribe": {}, "unsubscribe": {}, "psubscribe": {}, "punsubscribe": {},
	"ssubscribe": {}, "sunsubscribe": {},
	"client": {},
	// Connection-scoped cluster state: queued onto a shared pipeline conn
	// they would leak replica-reads (or a pending redirect) to every later
	// batch on that conn.
	"readonly": {}, "readwrite": {}, "asking": {},
}

func runsOutsidePipeline(name string) bool {
	_, ok := outsidePipelineCommands[name]
	return ok
}

// blockingCommands are commands that park on the server until data arrives or
// their own timeout expires. The TYPED helpers set a per-command read timeout
// (see cmdable.BLPop), which submit already diverts on; a RAW Cmder built by
// hand — NewCmd(ctx, "blpop", key, 0) via Submit/Process/Do — carries no such
// marker, so without this set it would be queued onto a shared pipeline
// connection and hold the whole batch for the block duration.
// Derived from the typed helpers rather than guessed: every cmdable method that
// calls cmd.setReadTimeout parks the connection, so
//
//	grep -rn 'setReadTimeout' --include='*.go' . | grep -v _test
//
// enumerates exactly the wire names that belong here (the arg-driven ones are
// handled in isBlockingCmd instead). Re-run that grep when adding a blocking
// command.
var blockingCommands = map[string]struct{}{
	"blpop": {}, "brpop": {}, "brpoplpush": {},
	"blmove": {}, "blmovem": {}, "blmpop": {},
	"bzpopmin": {}, "bzpopmax": {}, "bzmpop": {},
	"wait": {}, "waitaof": {},
	// MIGRATE blocks the source instance for up to its timeout.
	"migrate": {},
}

// isBlockingCmd reports whether cmd parks the connection. XREAD/XREADGROUP are
// decided by ARGUMENTS, not by name: only the BLOCK form blocks, and
// blanket-diverting the (far more common) non-blocking form would drop it out
// of batching for nothing.
func isBlockingCmd(cmd Cmder) bool {
	name := cmd.Name()
	if _, ok := blockingCommands[name]; ok {
		return true
	}
	// Arg-driven: these block only in their BLOCK form, and blanket-diverting
	// the far more common non-blocking form would drop it out of batching for
	// nothing. TS.READ takes BLOCK the same way (see TSReadWithArgs).
	if name != "xread" && name != "xreadgroup" && name != "ts.read" {
		return false
	}
	// Match the token the way the encoder does: a raw Cmder may carry RESP
	// tokens as []byte or *string (see baseCmd.stringArg), and a type switch on
	// string alone would let NewCmd(ctx, "xread", []byte("BLOCK"), 0, ...) be
	// batched onto a shared connection.
	for _, arg := range cmd.Args() {
		if internal.ToLower(blockingArgString(arg)) == "block" {
			return true
		}
	}
	return false
}

// blockingArgString renders a command argument as the string the encoder will
// write for the token comparisons above. Only the forms that can carry a RESP
// keyword are handled; anything else cannot be the BLOCK token.
func blockingArgString(arg interface{}) string {
	switch v := arg.(type) {
	case string:
		return v
	case []byte:
		return string(v)
	case *string:
		if v == nil {
			return ""
		}
		return *v
	default:
		return ""
	}
}

// submit queues a command without blocking and returns its completion future.
func (ap *AutoPipeliner) submit(ctx context.Context, cmd Cmder) AutoFuture {
	// finish marks the command ready on the deferred face so its result
	// accessors (Val/Result/Err) self-gate through await() — whether the
	// caller goes through the typed surface or raw Submit. Reading a
	// Submit()-ed command before Wait() was previously a silent data race
	// with the dispatch goroutine. The blocking face deliberately never
	// carries a batch: its callers only regain control after execution, and
	// the dispatcher-gid deadlock guard relies on that.
	finish := func(f AutoFuture) AutoFuture {
		if !ap.blocking {
			cmd.setReady(f.batch)
		}
		return f
	}
	// Decide DIVERSION first. The cluster preflight rejects commands whose
	// request policy cannot ride a pipeline (ReqAllNodes/ReqAllShards), but a
	// diverted command never rides one: it goes through the underlying
	// Client/ClusterClient.Process, which performs the normal cluster-wide
	// fan-out and aggregation. Running the preflight first therefore rejected
	// commands that would have worked — typed WAIT/WAITAOF on a cluster with
	// command policies enabled (review finding by codex on #3942).
	diverted := cmd.readTimeout() != nil || runsOutsidePipeline(cmd.Name()) || isBlockingCmd(cmd) ||
		(ap.mustDivert != nil && ap.mustDivert(ctx, cmd))
	if !diverted && ap.preflight != nil {
		if err := ap.preflight(ctx, cmd); err != nil {
			cmd.SetErr(err)
			return finish(AutoFuture{cmd: cmd, batch: completedBatch})
		}
	}
	if diverted {
		// Blocking commands (and the conn-hostile ones above) are executed
		// directly, outside the pipeline — via runOutsidePipeline, which
		// keeps each face's call shape: the blocking face runs the command
		// synchronously, the deferred face runs it on its own goroutine so
		// this call returns immediately and the result accessors block (a
		// BLPOP submitted on the async face must not stall the submitter,
		// exactly like Do). They still must respect a closed AutoPipeliner:
		// enqueue() rejects on the batched path, so mirror that here instead
		// of running after Close().
		if ap.isClosed() {
			cmd.SetErr(ErrClosed)
			return finish(AutoFuture{cmd: cmd, batch: completedBatch})
		}
		// runOutsidePipeline sets the command ready itself on the deferred
		// face; the returned batch completes when the command has executed.
		return AutoFuture{cmd: cmd, batch: ap.runOutsidePipeline(ctx, cmd)}
	}
	// No finish here: enqueue stamps ready under the stripe lock, before the
	// command is visible to any drain (the error paths above still go through
	// finish for uniform accessor behavior).
	return AutoFuture{cmd: cmd, batch: ap.enqueue(cmd)}
}

// ErrSubmitBlockingFace rejects Submit on the blocking face: Submit does not
// wait, so a windowed caller could have several commands in flight at once —
// but the blocking face stripes its enqueue queue on the strength of every
// caller waiting per command, and a non-waiting window there can be reordered.
// The deferred face (AsyncAutoPipeline) is built for exactly that usage.
//
// EXPERIMENTAL: this API is subject to change, use with caution.
var ErrSubmitBlockingFace = errors.New(
	"redis: Submit requires the deferred autopipeliner (AsyncAutoPipeline); on the blocking face use the typed methods or Do")

// errZeroAutoFuture is returned by Wait/WaitContext on a zero AutoFuture.
var errZeroAutoFuture = errors.New("redis: Wait on a zero AutoFuture")

// errDoNoArgs is returned by Do when called without a command.
var errDoNoArgs = errors.New("redis: AutoPipeliner.Do requires at least one argument")

// ErrAutoPipelineTimeout is set on drained commands when a flush could not
// obtain a batch permit within the engine's internal backstop — the engine is
// overloaded or an in-flight batch is wedged (e.g. read timeouts disabled on
// a dead peer). It is deliberately NOT context.DeadlineExceeded: the caller's
// own context did not expire, and errors.Is(err, context.DeadlineExceeded)
// must not fire for an internal engine timeout.
//
// EXPERIMENTAL: this API is subject to change, use with caution.
var ErrAutoPipelineTimeout = errors.New(
	"redis: autopipeline: no batch permit within the internal backstop (engine overloaded or a batch is wedged)")

// Submit queues a command without blocking and returns an AutoFuture; Wait on
// it when the result is needed. This is the explicit form for working with raw
// Cmders on the deferred (async) face, where the typed methods (Set, Get, ...)
// provide the same deferred behaviour returning the usual *XxxCmd. The
// command's own result accessors (Err/Val/Result) are safe to use instead of
// Wait — they block until the command has executed. Connection-hostile
// command names (SHUTDOWN, MONITOR, SELECT, AUTH, MULTI, SUBSCRIBE, CLIENT,
// ...) never ride a shared pipeline connection: they are diverted to a
// normal pooled connection with plain Client.Do semantics. On a BLOCKING
// autopipeliner Submit is rejected (the future's Wait returns an error): the
// blocking face's ordering relies on every caller waiting for each command
// before issuing the next, which Submit by design does not do.
func (ap *AutoPipeliner) Submit(ctx context.Context, cmd Cmder) AutoFuture {
	if ap.blocking {
		cmd.SetErr(ErrSubmitBlockingFace)
		return AutoFuture{cmd: cmd, batch: completedBatch}
	}
	return ap.submit(ctx, cmd)
}

// processAsync is the cmdable backing the typed command surface: it queues a
// command without blocking the caller and marks it ready so the command's
// result accessors (Val/Result/Err) block until the batch executes. This gives
// the autopipeliner the full typed surface (ap.Set, ap.Get, ...) with the exact
// same call shape as a normal client — only the wait is deferred to the point a
// result is read.
func (ap *AutoPipeliner) processAsync(ctx context.Context, cmd Cmder) error {
	// submit marks the command ready (see the finish closure there): a hook
	// that reads the command before that store lands sees a nil ready — the
	// non-blocking not-yet-executed view — while the caller always sees its
	// own store before any await.
	f := ap.submit(ctx, cmd)
	// Report SUBMIT-time rejections (a closed pipeliner, a cluster preflight
	// refusal): those paths set the error on the command and hand back the
	// shared completed batch without queueing anything, so returning nil made
	// Process claim success for a command that will never run — and callers
	// reaching the engine through UniversalClient.Process see only this return
	// value (review finding by codex on #3942). Execution errors are NOT
	// reported here: the deferred face's contract is that this call does not
	// wait, so those stay on the command for its accessors. rawErr keeps the
	// check non-blocking.
	if f.batch == completedBatch {
		return cmd.rawErr()
	}
	return nil
}

// processBlocking is the cmdable backing the blocking face: it queues the
// command and blocks until its batch has executed, so the command call has the
// same synchronous shape as a normal client (the returned *XxxCmd already holds
// its result). The flusher still batches this command with other concurrent
// callers' commands into a pipeline, so throughput is far above a plain client
// even though each caller waits. Per-goroutine ordering holds regardless of
// MaxConcurrentBatches: a caller cannot issue its next command until this one
// returns, so its commands execute in submit order.
func (ap *AutoPipeliner) processBlocking(ctx context.Context, cmd Cmder) error {
	return ap.submit(ctx, cmd).Wait()
}

// completedBatch is a reusable already-completed batch: returned both for
// commands that already executed directly (blocking commands, Submit-time
// rejections) and for error cases like enqueue-after-Close, so Wait returns
// immediately and the command's own error tells the story.
var completedBatch = func() *apBatch {
	b := newAPBatch()
	b.close()
	return b
}()

// enqueue queues a command and returns the batch whose done channel completes
// when it has executed. On a closed autopipeliner it errors the command and
// returns the already-closed batch.
// isClosed reports whether this pipeliner (or the shared pool set it rides
// on) has been closed. Two atomic loads; no locks.
//
// EVERY closed check that gates accepting new work must go through this, not
// ap.closed directly: a WithTimeout clone's Close sets only the shared flag,
// so a guard reading ap.closed alone would accept commands against pools that
// are already gone and surface pool-closed errors instead of ErrClosed.
// (Close's own CompareAndSwap on ap.closed is the one deliberate direct use:
// it claims the shutdown for this instance.)
func (ap *AutoPipeliner) isClosed() bool {
	return ap.closed.Load() || (ap.sharedClosed != nil && ap.sharedClosed.Load())
}

func (ap *AutoPipeliner) enqueue(cmd Cmder) *apBatch {
	if ap.isClosed() {
		cmd.SetErr(ErrClosed)
		return completedBatch
	}

	// Pick a shard. With shardFn (cluster mode) route by command content so all
	// commands for one node collect in the same shard's batch; otherwise spread
	// round-robin to keep each shard's mutex lightly contended.
	var s *apShard
	if ap.shardFn != nil {
		// uint conversion instead of negation: -math.MinInt overflows back to
		// itself and a negative modulo would panic the index. The unsigned
		// modulo is deterministic for every int, including MinInt.
		idx := ap.shardFn(cmd)
		s = ap.shards[uint(idx)%uint(len(ap.shards))]
	} else if len(ap.shards) == 1 {
		// Single shard (the standalone default): skip the round-robin counter —
		// it is a shared cache line bumped by every enqueue for a pick that is
		// constant. Same guard the stripe pick already has.
		s = ap.shards[0]
	} else {
		// Unsigned modulo: converting to int first goes negative after the
		// uint32 counter passes 2^31 on 32-bit platforms and panics.
		s = ap.shards[int((ap.next.Add(1)-1)%uint32(len(ap.shards)))]
	}

	st := s.stripe()
	st.mu.Lock()
	// Re-check closed under the stripe lock (see Close): either we win the lock
	// first and the shutdown drain flushes us, or the drain ran first and we
	// reject here — so a late enqueue never hangs on an unclosed done.
	if ap.isClosed() {
		st.mu.Unlock()
		cmd.SetErr(ErrClosed)
		return completedBatch
	}
	batch := st.curBatch
	if !ap.blocking {
		// Publish the gating batch BEFORE the command becomes visible to a
		// drain (the drain takes this same stripe lock): a flush racing the
		// submitter's return path must observe ready already set, or the
		// cluster node-executor registration would skip this command's batch
		// and a node hook reading the command mid-dispatch could block on a
		// batch its own call chain completes. The blocking face deliberately
		// never carries a batch (see submit).
		cmd.setReady(batch)
	}
	st.queue = append(st.queue, cmd)
	st.queueLen.Store(int32(len(st.queue)))
	if ap.config.MaxBatchBytes > 0 {
		st.queueBytes.Add(cmdApproxBytes(cmd))
	}
	st.mu.Unlock()

	// One expected arrival has landed (see expectedArrivals).
	ap.expectedArrivals.Add(-1)

	s.wake()
	return batch
}

// wake signals the shard's flusher that work is available without blocking.
func (s *apShard) wake() {
	select {
	case s.notify <- struct{}{}:
	default:
	}
}

// IsBlocking reports which face this autopipeliner is: true for the blocking
// face (Client.AutoPipeline — calls wait for execution), false for the
// deferred face (AsyncAutoPipeline — calls return immediately and result
// accessors block). The two faces reject different usage (Submit is
// blocking-face-rejected), so code handed an *AutoPipeliner can branch on
// this instead of probing with errors.
func (ap *AutoPipeliner) IsBlocking() bool { return ap.blocking }

// Config returns a copy of the effective configuration (defaults filled in).
func (ap *AutoPipeliner) Config() AutoPipelineOptions {
	cfg := *ap.config
	// Strip internal-only fields. contentSharded is set by cluster wiring and
	// tells Validate that shards are slot-routed, so same-key commands cannot
	// be reordered — which exempts the config from the NumShards>1 ordering
	// requirement. Handing that bit back to a caller who copies this config
	// into a STANDALONE async autopipeliner would silence that check for
	// round-robin shards, which really do flush concurrently and really do
	// break submit order (review finding by codex on #3942).
	cfg.contentSharded = false
	return cfg
}

// IsClosed reports whether the AutoPipeliner has been closed, either by an
// explicit Close or by closing the owning client. A closed AutoPipeliner
// rejects new commands with ErrClosed.
func (ap *AutoPipeliner) IsClosed() bool {
	return ap.isClosed()
}

// numShards reports how many shards this autopipeliner runs.
func (ap *AutoPipeliner) numShards() int { return len(ap.shards) }

// setShardFn installs a content-based shard selector. In cluster mode it maps
// a command's SLOT to a shard, which is a batch-depth heuristic, not an
// invariant: slot ranges are assigned to shards proportionally, so when a
// node's slots are non-contiguous one shard's batch can still span nodes and
// mapCmdsByNode splits it (correctness is unaffected — that router resolves
// every command's own slot — but those per-node pipelines are shallower).
// What the mapping DOES guarantee is that a given key always lands on the same
// shard, so a caller's relative order for that key is preserved regardless of
// how the shard's batch is split. Must be called before the autopipeliner is
// used. Not safe to change concurrently with enqueues.
func (ap *AutoPipeliner) setShardFn(fn func(Cmder) int) { ap.shardFn = fn }

// setPreflight installs a submit-time command filter (cluster wiring rejects
// commands whose request policy cannot ride a pipeline). Called once during
// construction, before the AutoPipeliner is published.
func (ap *AutoPipeliner) setPreflight(fn func(ctx context.Context, cmd Cmder) error) {
	ap.preflight = fn
}

// setMustDivert installs a predicate that forces a command off the batching
// path (see the mustDivert field). Called once during construction, before the
// AutoPipeliner is published.
func (ap *AutoPipeliner) setMustDivert(fn func(ctx context.Context, cmd Cmder) bool) {
	ap.mustDivert = fn
}

// Close stops the autopipeliner and flushes any pending commands. Worst
// case it blocks up to the internal permit backstop (~30s) PER SHARD if
// in-flight batches are wedged (e.g. read timeouts disabled against a dead
// peer) — healthy shutdowns take one round trip per shard with commands
// queued, near-zero otherwise.
func (ap *AutoPipeliner) Close() error {
	if !ap.closed.CompareAndSwap(false, true) {
		return nil // Already closed
	}

	// Cancel context to stop flushers
	ap.cancel()

	// Wake every shard's flusher so each observes the cancelled context promptly.
	for _, s := range ap.shards {
		s.wake()
	}

	// Pass through the divert gate once: after the CompareAndSwap above, any
	// registration either completed before this (so the counter already sees
	// it) or will observe closed==true and reject. Without this handshake the
	// wait below could read a zero counter while a diverted command was
	// between its closed check and its Add.
	ap.divertMu.Lock()
	ap.divertMu.Unlock() //nolint:staticcheck // handshake, not a critical section

	// Drain everything that remains, BOUNDED AS ONE UNIT: the flusher exit, the
	// final shard sweep, and the batch/diverted dispatch waits.
	//
	// None of it can be cancelled: commands taken from a queue (or accepted for
	// diverted execution) were already ACCEPTED, and Close's contract is to
	// flush them, so ap.cancel() deliberately does not reach an in-flight
	// dispatch. With ReadTimeout disabled — a supported configuration — a
	// stalled read against a dead peer, or a diverted BLPOP with a zero
	// timeout, has nothing to end it. Bounding only the LAST wait would not
	// help: the wedged dispatch can just as easily sit in a flusher that
	// ap.wg.Wait() is waiting for, or in the shutdown sweep's own dispatch, so
	// Close would hang before ever reaching the bound it documents (review
	// finding by codex on #3942). On expiry, report what is still outstanding
	// instead of blocking the caller: the engine is already closed to new work,
	// and the leaked goroutines end when the server or the OS breaks the
	// connection. See autoPipelineCloseBackstop for why the bound is generous.
	return ap.drainAll(autoPipelineCloseBackstop)
}

// drainAll runs Close's whole drain tail under a single bound and returns an
// error naming every stage that was still outstanding when it expired. Split
// out of Close so the bound is testable without a real stalled connection.
//
// The stages are ordered as Close needs them — the shard sweep must not start
// before the flushers are provably gone — but they are waited on
// CONCURRENTLY with the timer, which is the whole point: any stage can be the
// one that never finishes.
func (ap *AutoPipeliner) drainAll(timeout time.Duration) error {
	flushers := make(chan struct{})
	go func() { defer close(flushers); ap.wg.Wait() }()

	// swept: after the flushers are gone, drain each shard once more under its
	// lock. A command can pass enqueue's under-lock closed-recheck just before
	// Close's CompareAndSwap and append to a shard AFTER that shard's flusher
	// has already drained and exited — leaving its batch.done unclosed and the
	// caller's accessor blocked forever. s.mu serializes the two, so either the
	// late enqueue appends first and this sweep flushes it, or the sweep runs
	// first and the enqueue then observes closed==true and rejects.
	swept := make(chan struct{})
	go func() {
		defer close(swept)
		<-flushers
		for _, s := range ap.shards {
			s.flushBatchSliceShutdown()
		}
	}()

	batches := make(chan struct{})
	go func() {
		defer close(batches)
		<-swept
		ap.batchWg.Wait()
	}()

	diverted := make(chan struct{})
	go func() { defer close(diverted); ap.divertWg.Wait() }()

	timer := time.NewTimer(timeout)
	defer timer.Stop()
	batchesDone, divertedDone := false, false
	for !batchesDone || !divertedDone {
		select {
		case <-batches:
			batchesDone = true
			batches = nil // a closed channel is always ready; stop selecting it
		case <-diverted:
			divertedDone = true
			diverted = nil
		case <-timer.C:
			var outstanding []string
			if !batchesDone {
				// Name the precise stage: a wedged flusher and a wedged batch
				// dispatch need different operator responses.
				select {
				case <-flushers:
					select {
					case <-swept:
						outstanding = append(outstanding, "batch dispatches")
					default:
						outstanding = append(outstanding, "the shutdown flush")
					}
				default:
					outstanding = append(outstanding, "the flusher drain")
				}
			}
			if !divertedDone {
				outstanding = append(outstanding, "diverted (blocking) commands")
			}
			return fmt.Errorf(
				"redis: autopipeline: Close timed out after %s with %s still in flight; "+
					"they hold pooled connections until the server or the OS ends them "+
					"(most often a blocking command with no timeout, or ReadTimeout disabled)",
				timeout, strings.Join(outstanding, " and "))
		}
	}
	return nil
}

// flusher is the per-shard background goroutine that flushes batches.
func (s *apShard) flusher() {
	defer s.ap.wg.Done()
	ap := s.ap

	for {
		// Wait for a command to arrive (or shutdown). The notify channel is a
		// cheap buffered wake-up; no lock is taken on the hot enqueue path.
		if s.Len() == 0 {
			select {
			case <-s.notify:
			case <-ap.ctx.Done():
			}
		}

		// Check if context is cancelled
		if ap.ctx.Err() != nil {
			// Final flush before shutdown - use background context to avoid immediate cancellation
			s.flushBatchSliceShutdown()
			return
		}

		// Apply the coalescing window if one is configured (MaxFlushDelay /
		// AdaptiveDelay). With the default config this returns at once: batching
		// under concurrent load comes from in-flight backpressure, not a wait —
		// see accumulateBatch.
		s.accumulateBatch()

		// Flush all pending commands
		for s.Len() > 0 {
			select {
			case <-ap.ctx.Done():
				// Final flush before shutdown
				s.flushBatchSliceShutdown()
				return
			default:
			}

			s.flushBatchSlice()

			// Between batches, apply the configured window again so the next
			// pipeline is also full. A no-op with the default config (see
			// accumulateBatch); the next drain picks up whatever has queued.
			if s.Len() > 0 && s.Len() < ap.config.MaxBatchSize && !s.bytesFull() {
				s.accumulateBatch()
			}
		}
	}
}

// accumulateBatch lets commands pile up before the flusher drains the queue,
// so pipelines carry many commands instead of one. It returns as soon as any
// of these holds:
//
//   - the queue reaches MaxBatchSize (batch is full);
//   - a configured MaxFlushDelay / AdaptiveDelay window elapses; or
//   - with no configured window (the default), the expected resubmission
//     wave of arrivals has landed — see awaitExpectedArrivals.
//
// A configured MaxFlushDelay / AdaptiveDelay is an intentional accumulation
// window and is waited in full (AdaptiveDelay scales it down as the queue fills
// and returns 0 — flush now — once the queue is ≥75% full).
func (s *apShard) accumulateBatch() {
	ap := s.ap
	batchSize := ap.config.MaxBatchSize
	if batchSize <= 0 {
		batchSize = 1
	}
	if s.Len() >= batchSize || s.bytesFull() {
		return
	}

	// Pick the accumulation window. calculateDelay returns 0 both when no
	// MaxFlushDelay is configured (the default) and when AdaptiveDelay resolves
	// the current fill level to "flush immediately". The fill level is this
	// shard's own length — each shard flushes independently, so a global count
	// would mis-tune a quiet shard while another is busy.
	window := ap.calculateDelay(s.Len())
	if window <= 0 {
		if ap.config.MaxFlushDelay == 0 && !ap.config.AdaptiveDelay {
			// Default: coalesce by expected-arrival count, not by wall-clock.
			s.awaitExpectedArrivals(batchSize)
		}
		return
	}

	// Explicit window: wait the whole delay (or until the batch fills). Each
	// enqueue sends on notify, so we re-check the queue length on every wake-up
	// and return once the batch is full.
	deadline := time.NewTimer(window)
	defer deadline.Stop()
	for {
		select {
		case <-ap.ctx.Done():
			return
		case <-deadline.C:
			return
		case <-s.notify:
			if s.Len() >= batchSize || s.bytesFull() {
				return
			}
		}
	}
}

// silenceGapFloor / silenceGapCeil bound awaitExpectedArrivals's silence fallback.
// The floor covers fast links; the RTT-scaled value (execEWMA/8) takes over on
// slow ones, where a wakeup wave staggered by goroutine scheduling can pause
// longer than the floor mid-landing and a premature flush is expensive (each
// batch fragment occupies a pipeline connection for a full round trip). The
// ceiling bounds how long a stale expectation (callers that left) can delay a
// flush.
const (
	silenceGapFloor = 200 * time.Microsecond
	silenceGapCeil  = 2 * time.Millisecond
)

// coalesceMinFlush is the smallest pipeline worth dispatching while other
// batches are still executing. Below it, a gap-fire holds the queued
// stragglers for the next wave instead of burning a connection on a
// near-empty flush; once nothing is in flight, any size flushes immediately.
const coalesceMinFlush = 8

// observeBatchExec folds one batch execution duration into execEWMA.
func (ap *AutoPipeliner) observeBatchExec(d time.Duration) {
	sample := int64(d)
	if sample <= 0 {
		return
	}
	old := ap.execEWMA.Load()
	if old == 0 {
		ap.execEWMA.Store(sample)
		return
	}
	ap.execEWMA.Store(old + (sample-old)/8)
}

// silenceGap returns the silence fallback for awaitExpectedArrivals, scaled to the
// observed batch round-trip: clamp(execEWMA/8, floor, ceil).
func (ap *AutoPipeliner) silenceGap() time.Duration {
	g := time.Duration(ap.execEWMA.Load() / 8)
	if g < silenceGapFloor {
		return silenceGapFloor
	}
	if g > silenceGapCeil {
		return silenceGapCeil
	}
	return g
}

// awaitExpectedArrivals holds the flusher while related work is in motion, so
// commands flush as deep pipelines instead of fragmenting into small batches
// (each fragment costs a pipeline connection for a full round trip). Two
// signals — both facts the engine already has, not wall-clock guesses — decide
// whether anything is imminent:
//
//   - expectedArrivals: a completed batch of N commands wakes its N waiters
//     together, and in a closed loop each immediately submits its next
//     command. Completion announces the exact count; every enqueue accounts
//     for one; the wait ends the moment the count drains — the wave of
//     arrivals has fully landed. An exact per-wave count has no failure mode
//     where an averaged estimate undershoots the true wave and locks the
//     engine into fragmented flushes.
//   - inFlight: batches still executing mean their waiters will wake shortly
//     and stragglers are mid-stream — worth holding a moment to coalesce with,
//     bounded by the silence gap. This also recovers a fragmented state (many
//     singles in flight, which announce nothing): their staggered returns land
//     within one gap, merge into a real batch, and arrival tracking resumes.
//
// When neither holds, the shard is idle and the flush happens immediately: a
// lone caller pays a single round trip with no timer armed. That is the point
// of the design — the previous fixed ~20µs debounce timer armed on every flush
// fires ~1ms late on an idle or low-core host (wakeup latency dominates the
// requested delay), taxing every low-concurrency command ~5x its round trip.
// Here the gap timer never fires in steady state, closed loop or open; it only
// ends waits for callers that left.
func (s *apShard) awaitExpectedArrivals(batchSize int) {
	ap := s.ap
	expected := ap.expectedArrivals.Load()
	if expected < 0 {
		// Arrivals outran what was announced (open-loop traffic); re-zero so
		// the deficit does not mask the next wave. CAS: only clear the value
		// we saw, never a concurrent announcement.
		ap.expectedArrivals.CompareAndSwap(expected, 0)
		expected = 0
	}
	expectingWave := expected > 0
	if !expectingWave && s.inFlight.Load() == 0 {
		// Idle shard: nothing imminent, flush in one round trip.
		return
	}

	gap := ap.silenceGap()
	// Reset is drain-safe on Go 1.23+ (see go.mod: go 1.24).
	fallback := time.NewTimer(gap)
	defer fallback.Stop()
	lastSeenExpected := expected // count as of the most recent timer (re)arm
	var holdStart time.Time      // set on the first straggler-hold gap fire
	for {
		select {
		case <-ap.ctx.Done():
			return
		case <-fallback.C:
			if !expectingWave && s.Len() < coalesceMinFlush && s.inFlight.Load() > 0 {
				// Only stragglers queued while batches are still executing:
				// flushing a near-empty pipeline burns a connection for a full
				// round trip (measured at high WAN concurrency: straggler
				// flushes of 1-3 commands starved the connection pool and
				// doubled p50). Hold them — the next completed batch's wave
				// sweeps them along, and the wave path below flushes promptly.
				// The hold is bounded like the permit wait: with read timeouts
				// disabled a wedged batch could pin inFlight forever, and the
				// held stragglers must not hang with it.
				if holdStart.IsZero() {
					holdStart = time.Now()
				}
				if time.Since(holdStart) < autoPipelinePermitBackstop {
					lastSeenExpected = ap.expectedArrivals.Load()
					fallback.Reset(gap)
					continue
				}
			}
			if expectingWave {
				// A whole gap passed with no arrivals on this shard: the
				// expected callers left (workload shrank), so clear the stale
				// expectation or future flushes will wait for ghosts. But only
				// if it did not GROW during the silent gap — growth means a
				// batch elsewhere (another shard, or racing this fire)
				// announced a fresh wave, and erasing that would fragment a
				// wave that is really coming. CAS, never a blind store, so an
				// announcement racing the reset itself also survives.
				if d := ap.expectedArrivals.Load(); d > 0 && d <= lastSeenExpected {
					ap.expectedArrivals.CompareAndSwap(d, 0)
				}
			}
			return
		case <-s.notify:
			if s.Len() >= batchSize || s.bytesFull() {
				return
			}
			if d := ap.expectedArrivals.Load(); d > 0 {
				// An in-flight batch completed mid-wait: its wave is now the
				// thing to wait out, with the exact-count exit below.
				expectingWave = true
				lastSeenExpected = d
			} else if expectingWave {
				// The wave has fully landed; flush it as one batch.
				return
			} else if s.inFlight.Load() == 0 {
				// Nothing executing, no wave expected: no completion will
				// wake more callers, so flush what we have now.
				return
			}
			fallback.Reset(gap)
		}
	}
}

// dispatchCmds executes the drained stripe queues as one pipeline without
// constructing a Pipeline object: the queue slices go straight to the client's
// hook-wrapped pipeline processor (the exact entry Pipeline.Exec is wired to),
// so hooks and OTel behave identically while the per-batch Pipeline allocation,
// its append-growth reallocations and the per-command Process calls disappear.
// A single-stripe drain (every ordered shard, and any drain that found one
// non-empty stripe) passes its queue zero-copy; multi-stripe drains merge into
// one pooled slice.
// The batches stay OPEN throughout: completion happens at the caller's
// deferred closes, after the whole hook chain has returned. Hooks on the
// dispatch goroutine can still read results without deadlocking via the
// dispGid guard in await() (pre-next: the not-yet-executed view; post-next:
// the populated results), and — exactly like a plain pipeline — they may
// even adjust results before any waiter wakes.
//
// The innermost records whether execution actually happened. Two hook
// behaviours the chain's return value can carry are surfaced, both while the
// batches are still open (the callers' deferred closes run after this
// returns, so no waiter is reading yet):
//   - short-circuit (hook returned without calling next): nothing set the
//     commands' results — the chain's error, if any, is set
//     on every command;
//   - post-next verdict (exec ran, a hook still returned an error): applied
//     to the commands ONLY when every one of them is error-free — the case
//     where the hook's verdict would otherwise vanish entirely. A plain
//     pipeline hands that verdict to the Exec caller without rewriting
//     per-command results; with no Exec caller here, per-command errors
//     recorded by the exec always win and are never overwritten.
func (ap *AutoPipeliner) dispatchCmds(ctx context.Context, queues [][]Cmder, total int) {
	cmds := queues[0]
	if len(queues) > 1 {
		cmds = getQueueSlice(total)
		for i := range queues {
			cmds = append(cmds, queues[i]...)
		}
	}
	// A command that forbids retries (today: the zero-copy reads, whose reply
	// decodes into a caller buffer that a retry could not un-write) disables
	// retries for the WHOLE slice it is dispatched in — see cmdsContainNoRetry.
	// In a shared batch that would silently strip retries from unrelated
	// callers' ordinary commands, so a mixed batch is dispatched as several
	// pipelines instead of one.
	//
	// Split into CONTIGUOUS RUNS, in order, never into two policy groups:
	// grouping would reorder the stream — a zero-copy read submitted before a
	// SET to the same key would execute after it, so the read observes the new
	// value on a face that promises submit order. Runs preserve every relative
	// position while still keeping each dispatched slice policy-uniform (both
	// findings by codex on #3942; the grouping bug was introduced by the first
	// fix for the retry leak).
	if runs := splitRetryRuns(cmds); runs != nil {
		ap.dispatchSequential(ctx, runs)
		if len(queues) > 1 {
			putQueueSlice(cmds)
		}
		return
	}
	executed := false
	chainErr := ap.pipeliner.withProcessPipelineHook(ctx, cmds, func(ctx context.Context, cmds []Cmder) error {
		executed = true
		return ap.pipeliner.processPipeline(ctx, cmds)
	})
	// NOTE: a hook that returns nil WITHOUT calling next has short-circuited
	// SUCCESSFULLY — it served the batch itself (a cache, a mock) and set the
	// command values. Plain Pipeline/Client hooks are allowed to do exactly
	// that, so no error is synthesized for it: doing so made a hook that works
	// on a pipeline fail on an autopipelined batch (review finding by codex on
	// #3942). Only the hook's own error propagates, below.
	if chainErr != nil {
		if !executed {
			setCmdsErr(cmds, chainErr)
		} else if cmdsFirstErr(cmds) == nil {
			// Post-next error on an all-clean batch: the exec fully succeeded,
			// so the error can only be the hook's own verdict — apply it.
			// On a mixed batch it is applied to nothing: hooks conventionally
			// return next's error (`err := next(...); return err`), so after a
			// partial failure the chain error is presumed to be that echo, and
			// stamping it on the commands that DID succeed would overwrite
			// valid replies with their batchmates' failure. Exec-recorded
			// per-command outcomes always win over a post-next rewrap.
			setCmdsErr(cmds, chainErr)
		}
	}
	if len(queues) > 1 {
		putQueueSlice(cmds)
	}
}

// dispatchCmdsMaybeChunked dispatches a drained batch, splitting it into
// byte-bounded chunks when MaxBatchBytes is configured: each chunk is its own
// pipeline write+read cycle, so a batch of many large values becomes several
// bounded bursts instead of one huge write that can stall a constrained link
// past its deadline. The commands' batches still complete only after ALL
// chunks executed (the caller's deferred closes), exactly like an unchunked
// dispatch — chunking bounds the wire bursts, it does not change completion
// semantics. Each chunk runs the full hook chain, like consecutive pipelines.
func (ap *AutoPipeliner) dispatchCmdsMaybeChunked(ctx context.Context, queues [][]Cmder, total int) {
	limit := int64(ap.config.MaxBatchBytes)
	if limit <= 0 {
		ap.dispatchCmds(ctx, queues, total)
		return
	}

	// Merge (borrowed from dispatchCmds's multi-queue path) so chunk
	// boundaries can cross stripe queues.
	cmds := queues[0]
	merged := false
	if len(queues) > 1 {
		cmds = getQueueSlice(total)
		for i := range queues {
			cmds = append(cmds, queues[i]...)
		}
		merged = true
	}

	// Cut the byte-bounded chunks, then hand the ordered sequence to the shared
	// dispatcher — which stops after a chunk dies on a transport-class failure,
	// so later commands cannot overtake a failed prefix (see
	// dispatchSequential; the retry-policy runs go through the same helper).
	chunks := make([][]Cmder, 0, 4)
	start := 0
	var chunkBytes int64
	for i, cmd := range cmds {
		chunkBytes += cmdApproxBytes(cmd)
		if chunkBytes >= limit && i+1 > start {
			chunks = append(chunks, cmds[start:i+1])
			start = i + 1
			chunkBytes = 0
		}
	}
	if start < len(cmds) {
		chunks = append(chunks, cmds[start:])
	}
	ap.dispatchSequential(ctx, chunks)
	if merged {
		putQueueSlice(cmds)
	}
}

// dispatchSequential dispatches an ORDERED sequence of sub-batches, stopping
// once one of them dies on a transport-class failure and failing the rest with
// that error.
//
// The stop is the same contract the unchunked path has: it fails or retries the
// batch as a UNIT, so in an ordered stream later commands must never overtake a
// prefix that died (retries exhausted, hook abort). Per-command redis errors
// (WRONGTYPE, nil) are normal outcomes and do not stop the sequence.
//
// Both callers that break a batch into ordered pieces — the MaxBatchBytes
// chunker and the retry-policy runs — go through here, because the first
// version of each got this wrong independently (review findings by codex on
// #3942).
func (ap *AutoPipeliner) dispatchSequential(ctx context.Context, groups [][]Cmder) {
	var abortErr error
	for _, group := range groups {
		if len(group) == 0 {
			continue
		}
		if abortErr != nil {
			setCmdsErr(group, abortErr)
			continue
		}
		ap.dispatchCmds(ctx, [][]Cmder{group}, len(group))
		for _, cmd := range group {
			if err := cmd.rawErr(); err != nil && !isRedisError(err) {
				abortErr = err
				break
			}
		}
	}
}

// splitRetryRuns slices cmds into maximal CONTIGUOUS runs of one retry policy,
// preserving order: run i's commands all precede run i+1's, exactly as
// submitted. It returns nil when the whole batch is already policy-uniform —
// the overwhelmingly common case — so uniform batches allocate nothing and are
// dispatched as one pipeline.
//
// Runs are sub-slices of cmds, not copies, so they must be dispatched before
// cmds is recycled and must not be returned to the slice pool individually.
func splitRetryRuns(cmds []Cmder) [][]Cmder {
	if len(cmds) < 2 {
		return nil
	}
	first := cmds[0].NoRetry()
	boundary := -1
	for i := 1; i < len(cmds); i++ {
		if cmds[i].NoRetry() != first {
			boundary = i
			break
		}
	}
	if boundary < 0 {
		return nil // uniform: one dispatch, no split
	}
	runs := make([][]Cmder, 0, 4)
	start := 0
	policy := first
	for i := 1; i < len(cmds); i++ {
		if p := cmds[i].NoRetry(); p != policy {
			runs = append(runs, cmds[start:i])
			start = i
			policy = p
		}
	}
	return append(runs, cmds[start:])
}

// recoverDispatchPanic converts a panic on a dispatch goroutine (a hook or
// command-encoder panic inside Process/Exec) into per-command errors instead
// of crashing the process. On a plain client the same panic unwinds into the
// CALLER, who can recover; the engine's dispatch goroutines have no caller,
// so an unrecovered panic here would kill the whole program on behalf of one
// bad command. Registered LAST at each dispatch site so it runs FIRST on
// unwind (LIFO) — the errors are stamped before the deferred batch closes
// wake the waiters. setCmdsErr fills only commands without an error, so
// exec-recorded outcomes for commands that finished are preserved.
func recoverDispatchPanic(cmds ...[]Cmder) {
	r := recover()
	if r == nil {
		return
	}
	err := fmt.Errorf("redis: autopipeline: panic during dispatch: %v", r)
	for _, batch := range cmds {
		setCmdsErr(batch, err)
	}
	internal.Logger.Printf(context.Background(), "autopipeline: recovered dispatch panic: %v\n%s", r, debug.Stack())
}

// flushBatchSlice takes the shard's currently-queued commands as one batch,
// swaps in a fresh batch for subsequent enqueues, and dispatches the taken
// batch. Completion is signalled by closing the batch's done channel once
// (waking every waiter in a single operation) rather than one channel send
// per command.
func (s *apShard) flushBatchSlice() {
	ap := s.ap

	// Drain every stripe into one combined batch and roll fresh queues for the
	// commands enqueued after this point. Striped enqueue spreads the hot
	// mutex; one merged flush keeps the pipeline deep. accumulateBatch already
	// bounds the total to roughly MaxBatchSize before we get here.
	queues := make([][]Cmder, 0, len(s.stripes))
	batches := make([]*apBatch, 0, len(s.stripes))
	total := 0
	for i := range s.stripes {
		st := &s.stripes[i]
		// Skip provably-empty stripes without taking their mutex. Safe in
		// THIS path only: an enqueue publishes queueLen under the stripe lock
		// and wakes the flusher after unlocking, so a command that appears
		// concurrently with this unlocked read is re-observed by the
		// flusher's Len() loop or the buffered notify — the same protocol the
		// flusher already relies on. The shutdown drain must keep locking
		// unconditionally (see flushBatchSliceShutdown).
		if st.queueLen.Load() == 0 {
			continue
		}
		st.mu.Lock()
		if len(st.queue) > 0 {
			queues = append(queues, st.queue)
			batches = append(batches, st.curBatch)
			total += len(st.queue)
			st.queue = getQueueSlice(ap.config.MaxBatchSize)
			st.curBatch = newAPBatch()
			st.queueLen.Store(0)
			st.queueBytes.Store(0)
		}
		st.mu.Unlock()
	}
	if total == 0 {
		return
	}

	// Acquire a concurrency permit. The wait runs on a background context with
	// a generous backstop deadline against a wedged semaphore: commands taken
	// from the queue were already ACCEPTED, so a concurrent Close must not
	// cancel them mid-acquire — Close's contract is to flush pending commands
	// (it waits for this dispatch via wg/batchWg before tearing anything
	// down). The backstop is deliberately well above both the default
	// ReadTimeout and a maintnotifications relaxed window, so a legitimately
	// slow batch (e.g. during a failover) holding a permit does not cause
	// waiters to spuriously fail.
	if !s.sem.TryAcquire() {
		err := s.sem.Acquire(context.Background(), autoPipelinePermitBackstop, ErrAutoPipelineTimeout)
		if err != nil {
			// A permit not freeing within the backstop means the in-flight
			// batch is wedged well past any configured timeout — leave an
			// operator breadcrumb before failing the drained commands.
			internal.Logger.Printf(context.Background(),
				"redis: autopipeline: no batch permit after %s; failing %d queued commands",
				autoPipelinePermitBackstop, total)
			batchErr := err
			for i := range queues {
				for _, qc := range queues[i] {
					qc.SetErr(batchErr)
				}
				batches[i].close()
				putQueueSlice(queues[i])
			}
			return
		}

		// Wave merge. We took the queue and then waited a full batch round
		// trip for the permit; callers whose replies landed just after our
		// take re-submitted into the FRESH queue during that wait. Executing
		// without them splits the group into two alternating waves — each
		// observing two round trips, at half throughput — a state that is
		// stable once entered (measured: p50 pinned at 2xRTT for entire runs
		// at mid worker counts on a 52ms link). On the default window, let the
		// wave of follow-ups land and fold it into this batch before
		// executing, which merges the waves back into one batch per round
		// trip. Explicit-delay configs keep their own timing.
		if ap.config.MaxFlushDelay == 0 && !ap.config.AdaptiveDelay {
			s.awaitExpectedArrivals(ap.config.MaxBatchSize)
			for i := range s.stripes {
				st := &s.stripes[i]
				if st.queueLen.Load() == 0 {
					continue
				}
				st.mu.Lock()
				if len(st.queue) > 0 {
					queues = append(queues, st.queue)
					batches = append(batches, st.curBatch)
					total += len(st.queue)
					st.queue = getQueueSlice(ap.config.MaxBatchSize)
					st.curBatch = newAPBatch()
					st.queueLen.Store(0)
					st.queueBytes.Store(0)
				}
				st.mu.Unlock()
			}
		}
	}

	// Fast path for single command: skip the pipeline and Process directly, in
	// its own goroutine. The dispatch MUST NOT run inline in the flusher: a
	// synchronous Process blocks the flusher for a full round trip, and on a
	// slow link a solo straggler then holds up an entire landed wave for one
	// RTT — whose flush then delays the straggler's next command in turn, a
	// stable phase-lock where everyone pays 2x RTT (measured: ~25% of runs on
	// a 57ms link locked at exactly 2x RTT until perturbed).
	// No expectedArrivals announcement: a single waiter waking is the
	// lone-caller case, which must keep flushing immediately.
	if total == 1 {
		ap.batchWg.Add(1)
		s.inFlight.Add(1)
		go func() {
			// Defer order matters: the batch close is registered BEFORE the
			// permit release and inFlight decrement so it runs AFTER them
			// (LIFO) — a woken lone caller's next command then observes an
			// idle shard and takes the immediate-flush path instead of
			// arming the silence-gap wait.
			defer ap.batchWg.Done()
			defer batches[0].close()
			defer s.inFlight.Add(-1)
			defer s.sem.Release()
			defer putQueueSlice(queues[0])
			defer recoverDispatchPanic(queues[0])
			// Background for the same reason as the batch goroutine below:
			// accepted commands execute even under a concurrent Close.
			execStart := time.Now()
			b := batches[0]
			if !ap.blocking && ap.armSelfDeadlockGuard() {
				b.dispGid.Store(curGoroutineID())
			}
			solo := queues[0][0]
			// Both faces run the user-hook chain via withProcessHook. The
			// command records the CHAIN's final verdict — exactly what
			// Client.Process does — before the deferred close wakes the
			// waiter, so a hook that short-circuits, rewrites, or suppresses
			// the error is honored. Hooks on this goroutine read the command
			// deadlock-free via the dispGid guard stamped above.
			// A successful short-circuit stays successful (see dispatchCmds).
			err := ap.pipeliner.withProcessHook(context.Background(), solo, func(ctx context.Context, cmd Cmder) error {
				return ap.pipeliner.process(ctx, cmd)
			})
			solo.SetErr(err)
			ap.observeBatchExec(time.Since(execStart))
		}()
		return
	}

	// Track this goroutine in the batchWg so Close() waits for it.
	// IMPORTANT: Add to WaitGroup AFTER semaphore is acquired to avoid deadlock.
	ap.batchWg.Add(1)
	s.inFlight.Add(1)
	go func() {
		defer ap.batchWg.Done()
		defer s.inFlight.Add(-1)
		defer s.sem.Release()
		// Signal completion with one close per taken stripe. Deferred so a
		// panic in Process/Exec (e.g. a malformed command or encoder panic)
		// still wakes every waiter in await() instead of hanging them forever;
		// the closes run after Exec on the happy path, so results are
		// populated first.
		defer func() {
			for i := range queues {
				batches[i].close()
				putQueueSlice(queues[i])
			}
		}()
		defer recoverDispatchPanic(queues...)

		// Execute on a background context: these commands were accepted before
		// any concurrent Close, and Close waits for this goroutine (batchWg)
		// before the client tears down its pools — cancelling here would
		// error already-accepted commands while the shutdown sweep flushes
		// later ones, an inverted outcome. The wire timeouts (Read/Write
		// Timeout, or maintnotifications relaxed windows) still bound the
		// execution; no per-batch timer is allocated.
		ctx := context.Background()

		// The batches complete at the deferred closes, AFTER the whole hook
		// chain has returned — so a hook's post-next verdict is honored and,
		// like a plain pipeline, a hook may adjust results before any waiter
		// wakes. Hooks on this goroutine read results deadlock-free via the
		// dispGid guard in await() (armed below when hooks can exist).
		if !ap.blocking && ap.armSelfDeadlockGuard() {
			gid := curGoroutineID()
			for i := range batches {
				batches[i].dispGid.Store(gid)
			}
		}

		execStart := time.Now()
		ap.dispatchCmdsMaybeChunked(ctx, queues, total)
		ap.observeBatchExec(time.Since(execStart))

		// Announce the expected arrivals BEFORE the deferred closes wake this
		// batch's waiters, so the flusher knows the wave size the moment its
		// first command lands (see expectedArrivals).
		ap.expectedArrivals.Add(int64(total))
	}()
}

// flushBatchSliceShutdown flushes commands during shutdown.
// Unlike flushBatchSlice, this doesn't use ap.ctx for semaphore acquisition
// because ap.ctx is already cancelled during shutdown.
// Executes synchronously to preserve command order.
func (s *apShard) flushBatchSliceShutdown() {
	ap := s.ap
	// Flush all remaining commands synchronously to preserve order.
	//
	// The loop condition is checked UNDER each stripe's lock (not via the
	// unlocked s.Len()): a late enqueue appends to a stripe's queue and updates
	// its queueLen under that stripe's mutex, so reading queueLen without the
	// lock could miss a command that was just appended (seeing 0 and exiting
	// while a command sits in the queue). Locking first makes "is the stripe
	// empty?" and "take the stripe" atomic against that enqueue — this is what
	// closes the lost-command race on Close.
	for {
		// Take every stripe's queue as one merged batch and roll fresh queues.
		queues := make([][]Cmder, 0, len(s.stripes))
		batches := make([]*apBatch, 0, len(s.stripes))
		total := 0
		for i := range s.stripes {
			st := &s.stripes[i]
			st.mu.Lock()
			if len(st.queue) > 0 {
				queues = append(queues, st.queue)
				batches = append(batches, st.curBatch)
				total += len(st.queue)
				st.queue = getQueueSlice(ap.config.MaxBatchSize)
				st.curBatch = newAPBatch()
				st.queueLen.Store(0)
				st.queueBytes.Store(0)
			}
			st.mu.Unlock()
		}
		if total == 0 {
			return
		}

		// Serialize with any still-running in-flight batch: the shutdown drain
		// used to bypass the per-shard permit, so under MaxConcurrentBatches:1
		// a drained command could execute CONCURRENTLY with the in-flight
		// batch during Close and be observed out of order. Acquire the permit
		// (bounded by the backstop, on a background context — ap.ctx is
		// already cancelled here); if the backstop expires the permit holder
		// is wedged and we proceed anyway rather than strand the commands.
		acquired := s.sem.TryAcquire()
		if !acquired {
			acquired = s.sem.Acquire(context.Background(), autoPipelinePermitBackstop, ErrAutoPipelineTimeout) == nil
			if !acquired {
				internal.Logger.Printf(context.Background(),
					"redis: autopipeline: no batch permit after %s during shutdown; flushing unserialized",
					autoPipelinePermitBackstop)
			}
		}

		// Execute each batch in a func so close(batch.done) is deferred: a panic
		// in Process/Exec still signals completion (waking await()) before it
		// propagates, instead of leaving shutdown waiters hung.
		func() {
			if acquired {
				defer s.sem.Release()
			}
			defer func() {
				for i := range queues {
					batches[i].close()
					putQueueSlice(queues[i])
				}
			}()
			defer recoverDispatchPanic(queues...)

			// ap.ctx is already cancelled here (Close cancels it before draining),
			// so use a fresh background context with no artificial deadline. The
			// wire timeout is then governed by the connection's ReadTimeout /
			// WriteTimeout — exactly like the normal flush path and a plain client
			// Exec. Crucially this lets a relaxed timeout (set by maintnotifications
			// during a failover/migration) take effect; a hardcoded short deadline
			// here would cap that relaxed window and time out in-flight commands the
			// relaxation was meant to protect. (A user who wants shutdown bounded
			// sets ReadTimeout/WriteTimeout on the client, as for any command.)
			if !ap.blocking && ap.armSelfDeadlockGuard() {
				gid := curGoroutineID()
				for i := range batches {
					batches[i].dispGid.Store(gid)
				}
			}
			ap.dispatchCmdsMaybeChunked(context.Background(), queues, total)
		}()
	}
}

// Len returns the number of queued commands in this shard.
func (s *apShard) Len() int {
	n := 0
	for i := range s.stripes {
		n += int(s.stripes[i].queueLen.Load())
	}
	return n
}

// bytesFull reports whether the shard's queued payload volume has reached the
// configured MaxBatchBytes (false when the cap is disabled). Like the
// MaxBatchSize trigger it is soft: enqueues racing the check can overshoot.
func (s *apShard) bytesFull() bool {
	limit := int64(s.ap.config.MaxBatchBytes)
	if limit <= 0 {
		return false
	}
	var n int64
	for i := range s.stripes {
		n += s.stripes[i].queueBytes.Load()
		if n >= limit {
			return true
		}
	}
	return false
}

// cmdApproxBytes estimates a command's wire payload for MaxBatchBytes
// accounting: string/[]byte argument lengths plus a small fixed overhead per
// argument (type marker, length line, CRLFs). Exactness doesn't matter — the
// cap bounds burst size, it is not a protocol calculation.
func cmdApproxBytes(cmd Cmder) int64 {
	const perArgOverhead = 16
	n := int64(0)
	for _, a := range cmd.Args() {
		switch v := a.(type) {
		case string:
			n += int64(len(v))
		case []byte:
			n += int64(len(v))
		default:
			n += 8
		}
		n += perArgOverhead
	}
	return n
}

// Len returns the current number of queued commands across all shards.
func (ap *AutoPipeliner) Len() int {
	total := 0
	for _, s := range ap.shards {
		total += s.Len()
	}
	return total
}

// calculateDelay calculates the delay based on the given queue length (the
// caller's own shard, not the global total, so each shard tunes independently).
// Uses integer-only arithmetic for optimal performance (no float operations).
// Returns 0 if MaxFlushDelay is 0.
func (ap *AutoPipeliner) calculateDelay(queueLen int) time.Duration {
	maxDelay := ap.config.MaxFlushDelay
	if maxDelay == 0 {
		return 0
	}

	// If adaptive delay is disabled, return fixed delay
	if !ap.config.AdaptiveDelay {
		return maxDelay
	}

	if queueLen == 0 {
		return 0
	}

	maxBatch := ap.config.MaxBatchSize

	// Use integer arithmetic to avoid float operations
	// Calculate thresholds: 75%, 50%, 25% of maxBatch
	// Multiply by 4 to avoid division: queueLen * 4 vs maxBatch * 3 (75%)
	//
	// Adaptive delay strategy:
	// - ≥75% full: No delay (flush immediately to prevent overflow)
	// - ≥50% full: 25% of max delay (queue filling up)
	// - ≥25% full: 50% of max delay (moderate load)
	// - <25% full: 100% of max delay (low load, maximize batching)
	switch {
	case queueLen*4 >= maxBatch*3: // queueLen >= 75% of maxBatch
		return 0 // Flush immediately
	case queueLen*2 >= maxBatch: // queueLen >= 50% of maxBatch
		return maxDelay >> 2 // Divide by 4 using bit shift (faster)
	case queueLen*4 >= maxBatch: // queueLen >= 25% of maxBatch
		return maxDelay >> 1 // Divide by 2 using bit shift (faster)
	default:
		return maxDelay
	}
}

// Pipeline returns a new pipeline that uses the underlying pipeliner.
// This allows you to create a traditional pipeline from an autopipeliner.
func (ap *AutoPipeliner) Pipeline() Pipeliner {
	return ap.pipeliner.Pipeline()
}

// Pipelined executes a function in a pipeline context.
// This is a convenience method that creates a pipeline, executes the function,
// and returns the results.
func (ap *AutoPipeliner) Pipelined(ctx context.Context, fn func(Pipeliner) error) ([]Cmder, error) {
	return ap.pipeliner.Pipeline().Pipelined(ctx, fn)
}

// TxPipelined executes a function in a transaction pipeline context.
// This is a convenience method that creates a transaction pipeline, executes the function,
// and returns the results. It delegates to the underlying client's TxPipeline.
func (ap *AutoPipeliner) TxPipelined(ctx context.Context, fn func(Pipeliner) error) ([]Cmder, error) {
	return ap.pipeliner.TxPipeline().Pipelined(ctx, fn)
}

// TxPipeline returns a new transaction pipeline that uses the underlying pipeliner.
// This allows you to create a traditional transaction pipeline from an autopipeliner.
// It delegates to the underlying client's TxPipeline.
func (ap *AutoPipeliner) TxPipeline() Pipeliner {
	return ap.pipeliner.TxPipeline()
}

// validate AutoPipeliner implements Cmdable
var _ Cmdable = (*AutoPipeliner)(nil)
