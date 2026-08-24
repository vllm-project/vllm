// Package pool implements the pool management
package pool

import (
	"bufio"
	"context"
	"errors"
	"fmt"
	"net"
	"sync"
	"sync/atomic"
	"time"

	"github.com/redis/go-redis/v9/internal"
	"github.com/redis/go-redis/v9/internal/maintnotifications/logs"
	"github.com/redis/go-redis/v9/internal/proto"
	uberatomic "go.uber.org/atomic"
)

var noDeadline = time.Time{}

// Preallocated errors for hot paths to avoid allocations
var (
	errAlreadyMarkedForHandoff  = errors.New("connection is already marked for handoff")
	errNotMarkedForHandoff      = errors.New("connection was not marked for handoff")
	errHandoffStateChanged      = errors.New("handoff state changed during marking")
	errConnectionNotAvailable   = errors.New("redis: connection not available")
	errConnNotAvailableForWrite = errors.New("redis: connection not available for write operation")
)

// getCachedTimeNs returns the current time in nanoseconds.
// This function previously used a global cache updated by a background goroutine,
// but that caused unnecessary CPU usage when the client was idle (ticker waking up
// the scheduler every 50ms). We now use time.Now() directly, which is fast enough
// on modern systems (vDSO on Linux) and only adds ~1-2% overhead in extreme
// high-concurrency benchmarks while eliminating idle CPU usage.
func getCachedTimeNs() int64 {
	return time.Now().UnixNano()
}

// GetCachedTimeNs returns the current time in nanoseconds.
// Exported for use by other packages that need fast time access.
func GetCachedTimeNs() int64 {
	return getCachedTimeNs()
}

// Global atomic counter for connection IDs
var connIDCounter atomic.Uint64

// HandoffState represents the atomic state for connection handoffs
// This struct is stored atomically to prevent race conditions between
// checking handoff status and reading handoff parameters
type HandoffState struct {
	ShouldHandoff bool   // Whether connection should be handed off
	Endpoint      string // New endpoint for handoff
	SeqID         int64  // Sequence ID from MOVING notification
}

// atomicNetConn is a wrapper to ensure consistent typing in atomic.Value
type atomicNetConn struct {
	conn net.Conn
}

// generateConnID generates a fast unique identifier for a connection with zero allocations
func generateConnID() uint64 {
	return connIDCounter.Add(1)
}

type Conn struct {
	// Connection identifier for unique tracking
	id uint64

	usedAt      atomic.Int64
	lastPutAt   atomic.Int64
	dialStartNs atomic.Int64 // Time when dial started (for connection create time metric)

	// Lock-free netConn access using atomic.Value
	// Contains *atomicNetConn wrapper, accessed atomically for better performance
	netConnAtomic atomic.Value // stores *atomicNetConn

	rd *proto.Reader
	bw *bufio.Writer
	wr *proto.Writer

	// Lightweight mutex to protect reader operations during handoff and health checks
	// Used during:
	// - SetNetConn (write lock for resetting reader state)
	// - HasBufferedData/PeekReplyTypeSafe (read lock for safe concurrent peek operations)
	readerMu sync.RWMutex

	// State machine for connection state management
	// Replaces: usable, Inited, used
	// Provides thread-safe state transitions with FIFO waiting queue
	// States: CREATED → INITIALIZING → IDLE ⇄ IN_USE
	//                                    ↓
	//                                UNUSABLE (handoff/reauth)
	//                                    ↓
	//                                IDLE/CLOSED
	stateMachine *ConnStateMachine

	// Handoff metadata - managed separately from state machine
	// These are atomic for lock-free access during handoff operations
	handoffStateAtomic   atomic.Value  // stores *HandoffState
	handoffRetriesAtomic atomic.Uint32 // retry counter

	pooled    bool
	pubsub    bool
	createdAt time.Time
	expiresAt time.Time
	poolName  string // Name of the pool this connection belongs to (for metrics)

	// preparedFieldsets tracks HIMPORT fieldsets prepared on this
	// connection's current server session: fieldset name -> client-side
	// registry version. The server drops fieldsets when the session ends,
	// so the map is cleared whenever the underlying network connection is
	// replaced. preparedFieldsetsEpoch records the registry's discard-all
	// epoch the session was prepared under; a session behind the current
	// epoch replays HIMPORT DISCARDALL before its next HIMPORT command.
	// Guarded by preparedFieldsetsMu; the map is nil until first use.
	preparedFieldsetsMu    sync.Mutex
	preparedFieldsets      map[string]uint64
	preparedFieldsetsEpoch uint64

	// When a goroutine closes a connection, it usually knows the reason, so closeReason is not needed.
	// closeReason is only used when an in-use connection is closed by another goroutine,
	// to inform the goroutine using the connection why the connection was closed.
	closeReason uberatomic.String

	// closeOnPutReason marks an in-use connection for removal when it is returned
	// to the pool. The socket is left open for the in-flight command and closed
	// by ConnPool.Put.
	closeOnPutReason uberatomic.String

	// maintenanceNotifications upgrade support: relaxed timeouts during migrations/failovers

	// Using atomic operations for lock-free access to avoid mutex contention
	relaxedReadTimeoutNs  atomic.Int64 // time.Duration as nanoseconds
	relaxedWriteTimeoutNs atomic.Int64 // time.Duration as nanoseconds
	relaxedDeadlineNs     atomic.Int64 // time.Time as nanoseconds since epoch

	// Counter to track multiple relaxed timeout setters if we have nested calls
	// will be decremented when ClearRelaxedTimeout is called or deadline is reached
	// if counter reaches 0, we clear the relaxed timeouts
	relaxedCounter atomic.Int32

	// Connection initialization function for reconnections
	initConnFunc func(context.Context, *Conn) error

	onClose func() error

	// onCscClose is the client-side-caching close hook, kept separate from
	// onClose (streaming-credentials cleanup) so neither clobbers the other.
	// Both keep overwrite semantics, so re-running initConn can't accumulate them.
	onCscClose func() error

	// onCscReinit runs after the connection is claimed for reinitialization but
	// before its socket is replaced. CSC uses it to invalidate entries whose
	// server-side tracking coverage belongs to the old socket.
	onCscReinit func()

	// cscReadPending requests one conservative drain after a command read through
	// a transport whose buffered state cannot be fully observed by MaybeHasData.
	cscReadPending atomic.Bool

	// lastCscPeriodicProbeNs throttles bounded fallback reads on platforms and
	// opaque transports without a non-consuming readiness mechanism.
	lastCscPeriodicProbeNs atomic.Int64
}

func NewConn(netConn net.Conn) *Conn {
	return NewConnWithBufferSize(netConn, proto.DefaultBufferSize, proto.DefaultBufferSize)
}

func NewConnWithBufferSize(netConn net.Conn, readBufSize, writeBufSize int) *Conn {
	now := time.Now()
	cn := &Conn{
		createdAt:    now,
		id:           generateConnID(), // Generate unique ID for this connection
		stateMachine: NewConnStateMachine(),
	}

	// Use specified buffer sizes, or fall back to 32KiB defaults if 0
	if readBufSize > 0 {
		cn.rd = proto.NewReaderSize(netConn, readBufSize)
	} else {
		cn.rd = proto.NewReader(netConn) // Uses 32KiB default
	}

	if writeBufSize > 0 {
		cn.bw = bufio.NewWriterSize(netConn, writeBufSize)
	} else {
		cn.bw = bufio.NewWriterSize(netConn, proto.DefaultBufferSize)
	}

	// Store netConn atomically for lock-free access using wrapper
	cn.netConnAtomic.Store(&atomicNetConn{conn: netConn})

	cn.wr = proto.NewWriter(cn.bw)
	cn.SetUsedAt(now)
	// Initialize handoff state atomically
	initialHandoffState := &HandoffState{
		ShouldHandoff: false,
		Endpoint:      "",
		SeqID:         0,
	}
	cn.handoffStateAtomic.Store(initialHandoffState)
	return cn
}

func (cn *Conn) UsedAt() time.Time {
	return time.Unix(0, cn.usedAt.Load())
}
func (cn *Conn) SetUsedAt(tm time.Time) {
	cn.usedAt.Store(tm.UnixNano())
}

func (cn *Conn) UsedAtNs() int64 {
	return cn.usedAt.Load()
}
func (cn *Conn) SetUsedAtNs(ns int64) {
	cn.usedAt.Store(ns)
}

func (cn *Conn) LastPutAtNs() int64 {
	return cn.lastPutAt.Load()
}
func (cn *Conn) SetLastPutAtNs(ns int64) {
	cn.lastPutAt.Store(ns)
}

// GetDialStartNs returns the time when the dial started (in nanoseconds since epoch).
// This is used to calculate the full connection creation time (TCP + handshake).
func (cn *Conn) GetDialStartNs() int64 {
	return cn.dialStartNs.Load()
}

// PoolName returns the name of the pool this connection belongs to.
// This is used for metrics to identify which pool a connection is from.
func (cn *Conn) PoolName() string {
	return cn.poolName
}

// SetPoolName sets the name of the pool this connection belongs to.
// This should be called when the connection is added to a pool.
func (cn *Conn) SetPoolName(name string) {
	cn.poolName = name
}

// Backward-compatible wrapper methods for state machine
// These maintain the existing API while using the new state machine internally

// CompareAndSwapUsable atomically compares and swaps the usable flag (lock-free).
//
// This is used by background operations (handoff, re-auth) to acquire exclusive
// access to a connection. The operation sets usable to false, preventing the pool
// from returning the connection to clients.
//
// Returns true if the swap was successful (old value matched), false otherwise.
//
// Implementation note: This is a compatibility wrapper around the state machine.
// It checks if the current state is "usable" (IDLE or IN_USE) and transitions accordingly.
// Deprecated: Use GetStateMachine().TryTransition() directly for better state management.
func (cn *Conn) CompareAndSwapUsable(old, new bool) bool {
	currentState := cn.stateMachine.GetState()

	// Check if current state matches the "old" usable value
	currentUsable := (currentState == StateIdle || currentState == StateInUse)
	if currentUsable != old {
		return false
	}

	// If we're trying to set to the same value, succeed immediately
	if old == new {
		return true
	}

	// Transition based on new value
	if new {
		// Trying to make usable - transition from UNUSABLE to IDLE
		// This should only work from UNUSABLE or INITIALIZING states
		// Use predefined slice to avoid allocation
		_, err := cn.stateMachine.TryTransition(
			validFromInitializingOrUnusable,
			StateIdle,
		)
		return err == nil
	}
	// Trying to make unusable - transition from IDLE to UNUSABLE
	// This is typically for acquiring the connection for background operations
	// Use predefined slice to avoid allocation
	_, err := cn.stateMachine.TryTransition(
		validFromIdle,
		StateUnusable,
	)
	return err == nil
}

// IsUsable returns true if the connection is safe to use for new commands (lock-free).
//
// A connection is "usable" when it's in a stable state and can be returned to clients.
// It becomes unusable during:
//   - Handoff operations (network connection replacement)
//   - Re-authentication (credential updates)
//   - Other background operations that need exclusive access
//
// Note: CREATED state is considered usable because new connections need to pass OnGet() hook
// before initialization. The initialization happens after OnGet() in the client code.
func (cn *Conn) IsUsable() bool {
	state := cn.stateMachine.GetState()
	// CREATED, IDLE, and IN_USE states are considered usable
	// CREATED: new connection, not yet initialized (will be initialized by client)
	// IDLE: initialized and ready to be acquired
	// IN_USE: usable but currently acquired by someone
	return state == StateCreated || state == StateIdle || state == StateInUse
}

// SetUsable sets the usable flag for the connection (lock-free).
//
// Deprecated: Use GetStateMachine().Transition() directly for better state management.
// This method is kept for backwards compatibility.
//
// This should be called to mark a connection as usable after initialization or
// to release it after a background operation completes.
//
// Prefer CompareAndSwapUsable() when acquiring exclusive access to avoid race conditions.
// Deprecated: Use GetStateMachine().Transition() directly for better state management.
func (cn *Conn) SetUsable(usable bool) {
	if usable {
		// Transition to IDLE state (ready to be acquired)
		cn.stateMachine.Transition(StateIdle)
	} else {
		// Transition to UNUSABLE state (for background operations)
		cn.stateMachine.Transition(StateUnusable)
	}
}

// IsInited returns true if the connection has been initialized.
// This is a backward-compatible wrapper around the state machine.
func (cn *Conn) IsInited() bool {
	state := cn.stateMachine.GetState()
	// Connection is initialized if it's in IDLE or any post-initialization state
	return state != StateCreated && state != StateInitializing && state != StateClosed
}

// Used - State machine based implementation

// CompareAndSwapUsed atomically compares and swaps the used flag (lock-free).
// This method is kept for backwards compatibility.
//
// This is the preferred method for acquiring a connection from the pool, as it
// ensures that only one goroutine marks the connection as used.
//
// Implementation: Uses state machine transitions IDLE ⇄ IN_USE
//
// Returns true if the swap was successful (old value matched), false otherwise.
// Deprecated: Use GetStateMachine().TryTransition() directly for better state management.
func (cn *Conn) CompareAndSwapUsed(old, new bool) bool {
	if old == new {
		// No change needed
		currentState := cn.stateMachine.GetState()
		currentUsed := (currentState == StateInUse)
		return currentUsed == old
	}

	if !old && new {
		// Acquiring: IDLE → IN_USE
		// Use predefined slice to avoid allocation
		_, err := cn.stateMachine.TryTransition(validFromCreatedOrIdle, StateInUse)
		return err == nil
	} else {
		// Releasing: IN_USE → IDLE
		// Use predefined slice to avoid allocation
		_, err := cn.stateMachine.TryTransition(validFromInUse, StateIdle)
		return err == nil
	}
}

// IsUsed returns true if the connection is currently in use (lock-free).
//
// Deprecated: Use GetStateMachine().GetState() == StateInUse directly for better clarity.
// This method is kept for backwards compatibility.
//
// A connection is "used" when it has been retrieved from the pool and is
// actively processing a command. Background operations (like re-auth) should
// wait until the connection is not used before executing commands.
func (cn *Conn) IsUsed() bool {
	return cn.stateMachine.GetState() == StateInUse
}

// SetUsed sets the used flag for the connection (lock-free).
//
// This should be called when returning a connection to the pool (set to false)
// or when a single-connection pool retrieves its connection (set to true).
//
// Prefer CompareAndSwapUsed() when acquiring from a multi-connection pool to
// avoid race conditions.
// Deprecated: Use GetStateMachine().Transition() directly for better state management.
func (cn *Conn) SetUsed(val bool) {
	if val {
		cn.stateMachine.Transition(StateInUse)
	} else {
		cn.stateMachine.Transition(StateIdle)
	}
}

// getNetConn returns the current network connection using atomic load (lock-free).
// This is the fast path for accessing netConn without mutex overhead.
func (cn *Conn) getNetConn() net.Conn {
	if v := cn.netConnAtomic.Load(); v != nil {
		if wrapper, ok := v.(*atomicNetConn); ok {
			return wrapper.conn
		}
	}
	return nil
}

// setNetConn stores the network connection atomically (lock-free).
// This is used for the fast path of connection replacement.
func (cn *Conn) setNetConn(netConn net.Conn) {
	cn.netConnAtomic.Store(&atomicNetConn{conn: netConn})
}

// Handoff state management - atomic access to handoff metadata

// ShouldHandoff returns true if connection needs handoff (lock-free).
func (cn *Conn) ShouldHandoff() bool {
	if v := cn.handoffStateAtomic.Load(); v != nil {
		return v.(*HandoffState).ShouldHandoff
	}
	return false
}

// GetHandoffEndpoint returns the new endpoint for handoff (lock-free).
func (cn *Conn) GetHandoffEndpoint() string {
	if v := cn.handoffStateAtomic.Load(); v != nil {
		return v.(*HandoffState).Endpoint
	}
	return ""
}

// GetMovingSeqID returns the sequence ID from the MOVING notification (lock-free).
func (cn *Conn) GetMovingSeqID() int64 {
	if v := cn.handoffStateAtomic.Load(); v != nil {
		return v.(*HandoffState).SeqID
	}
	return 0
}

// GetHandoffInfo returns all handoff information atomically (lock-free).
// This method prevents race conditions by returning all handoff state in a single atomic operation.
// Returns (shouldHandoff, endpoint, seqID).
func (cn *Conn) GetHandoffInfo() (bool, string, int64) {
	if v := cn.handoffStateAtomic.Load(); v != nil {
		state := v.(*HandoffState)
		return state.ShouldHandoff, state.Endpoint, state.SeqID
	}
	return false, "", 0
}

// HandoffRetries returns the current handoff retry count (lock-free).
func (cn *Conn) HandoffRetries() int {
	return int(cn.handoffRetriesAtomic.Load())
}

// IncrementAndGetHandoffRetries atomically increments and returns handoff retries (lock-free).
func (cn *Conn) IncrementAndGetHandoffRetries(n int) int {
	return int(cn.handoffRetriesAtomic.Add(uint32(n)))
}

// IsPooled returns true if the connection is managed by a pool and will be pooled on Put.
func (cn *Conn) IsPooled() bool {
	return cn.pooled
}

// MarkCloseOnPut marks the connection for removal when it is returned to the pool.
func (cn *Conn) MarkCloseOnPut(reason string) {
	cn.closeOnPutReason.Store(reason)
}

// CloseOnPutReason returns a non-empty reason when the connection should be
// removed instead of pooled on Put.
func (cn *Conn) CloseOnPutReason() string {
	return cn.closeOnPutReason.Load()
}

// IsPubSub returns true if the connection is used for PubSub.
func (cn *Conn) IsPubSub() bool {
	return cn.pubsub
}

// SetRelaxedTimeout sets relaxed timeouts for this connection during maintenanceNotifications upgrades.
// These timeouts will be used for all subsequent commands until the deadline expires.
// Uses atomic operations for lock-free access.
// Note: Metrics should be recorded by the caller (notification handler) which has context about
// the notification type and pool name.
func (cn *Conn) SetRelaxedTimeout(readTimeout, writeTimeout time.Duration) {
	cn.relaxedCounter.Add(1)
	cn.relaxedReadTimeoutNs.Store(int64(readTimeout))
	cn.relaxedWriteTimeoutNs.Store(int64(writeTimeout))
}

// SetRelaxedTimeoutWithDeadline sets relaxed timeouts with an expiration deadline.
// After the deadline, timeouts automatically revert to normal values.
// Uses atomic operations for lock-free access.
func (cn *Conn) SetRelaxedTimeoutWithDeadline(readTimeout, writeTimeout time.Duration, deadline time.Time) {
	cn.SetRelaxedTimeout(readTimeout, writeTimeout)
	cn.relaxedDeadlineNs.Store(deadline.UnixNano())
}

// ClearRelaxedTimeout removes relaxed timeouts, returning to normal timeout behavior.
// Uses atomic operations for lock-free access.
func (cn *Conn) ClearRelaxedTimeout() {
	// Atomically decrement counter and check if we should clear
	newCount := cn.relaxedCounter.Add(-1)
	deadlineNs := cn.relaxedDeadlineNs.Load()
	if newCount <= 0 && (deadlineNs == 0 || time.Now().UnixNano() >= deadlineNs) {
		// Use atomic load to get current value for CAS to avoid stale value race
		current := cn.relaxedCounter.Load()
		if current <= 0 && cn.relaxedCounter.CompareAndSwap(current, 0) {
			cn.clearRelaxedTimeout()
		}
	}
}

func (cn *Conn) clearRelaxedTimeout() {
	cn.relaxedReadTimeoutNs.Store(0)
	cn.relaxedWriteTimeoutNs.Store(0)
	cn.relaxedDeadlineNs.Store(0)
	cn.relaxedCounter.Store(0)

	// Note: Metrics for timeout unrelaxing are not recorded here because we don't have
	// context about which notification type or pool triggered the relaxation.
	// In practice, relaxed timeouts expire automatically via deadline, so explicit
	// unrelaxing metrics are less critical than the initial relaxation metrics.
}

// HasRelaxedTimeout returns true if relaxed timeouts are currently active on this connection.
// This checks both the timeout values and the deadline (if set).
// Uses atomic operations for lock-free access.
func (cn *Conn) HasRelaxedTimeout() bool {
	// Fast path: no relaxed timeouts are set
	if cn.relaxedCounter.Load() <= 0 {
		return false
	}

	readTimeoutNs := cn.relaxedReadTimeoutNs.Load()
	writeTimeoutNs := cn.relaxedWriteTimeoutNs.Load()

	// If no relaxed timeouts are set, return false
	if readTimeoutNs <= 0 && writeTimeoutNs <= 0 {
		return false
	}

	deadlineNs := cn.relaxedDeadlineNs.Load()
	// If no deadline is set, relaxed timeouts are active
	if deadlineNs == 0 {
		return true
	}

	// If deadline is set, check if it's still in the future
	return time.Now().UnixNano() < deadlineNs
}

// getEffectiveReadTimeout returns the timeout to use for read operations.
// If relaxed timeout is set and not expired, it takes precedence over the provided timeout.
// This method automatically clears expired relaxed timeouts using atomic operations.
func (cn *Conn) getEffectiveReadTimeout(normalTimeout time.Duration) time.Duration {
	readTimeoutNs := cn.relaxedReadTimeoutNs.Load()

	// Fast path: no relaxed timeout set
	if readTimeoutNs <= 0 {
		return normalTimeout
	}

	deadlineNs := cn.relaxedDeadlineNs.Load()
	// If no deadline is set, use relaxed timeout
	if deadlineNs == 0 {
		return time.Duration(readTimeoutNs)
	}

	// Use cached time to avoid expensive syscall (max 50ms staleness is acceptable for timeout checks)
	nowNs := getCachedTimeNs()
	// Check if deadline has passed
	if nowNs < deadlineNs {
		// Deadline is in the future, use relaxed timeout
		return time.Duration(readTimeoutNs)
	} else {
		// Deadline has passed, clear relaxed timeouts atomically and use normal timeout
		newCount := cn.relaxedCounter.Add(-1)
		if newCount <= 0 {
			internal.Logger.Printf(context.Background(), logs.UnrelaxedTimeoutAfterDeadline(cn.GetID()))
			cn.clearRelaxedTimeout()
		}
		return normalTimeout
	}
}

// getEffectiveWriteTimeout returns the timeout to use for write operations.
// If relaxed timeout is set and not expired, it takes precedence over the provided timeout.
// This method automatically clears expired relaxed timeouts using atomic operations.
func (cn *Conn) getEffectiveWriteTimeout(normalTimeout time.Duration) time.Duration {
	writeTimeoutNs := cn.relaxedWriteTimeoutNs.Load()

	// Fast path: no relaxed timeout set
	if writeTimeoutNs <= 0 {
		return normalTimeout
	}

	deadlineNs := cn.relaxedDeadlineNs.Load()
	// If no deadline is set, use relaxed timeout
	if deadlineNs == 0 {
		return time.Duration(writeTimeoutNs)
	}

	// Use cached time to avoid expensive syscall (max 50ms staleness is acceptable for timeout checks)
	nowNs := getCachedTimeNs()
	// Check if deadline has passed
	if nowNs < deadlineNs {
		// Deadline is in the future, use relaxed timeout
		return time.Duration(writeTimeoutNs)
	} else {
		// Deadline has passed, clear relaxed timeouts atomically and use normal timeout
		newCount := cn.relaxedCounter.Add(-1)
		if newCount <= 0 {
			internal.Logger.Printf(context.Background(), logs.UnrelaxedTimeoutAfterDeadline(cn.GetID()))
			cn.clearRelaxedTimeout()
		}
		return normalTimeout
	}
}

// SetOnClose installs fn as the callback invoked exactly once when this
// connection is closed (via Conn.Close).
//
// IMPORTANT: SetOnClose OVERWRITES any previously installed callback — it
// does not compose, chain, or deduplicate. A Conn has room for a single
// onClose hook by design, because its lifecycle is bounded (a Conn is
// created, optionally re-initialized on its own net.Conn, and then closed
// once) and the pool's OnRemove hooks handle any registry-level cleanup
// that must survive the net.Conn being swapped.
//
// This has a subtle implication for per-connection subscriptions such as
// the unsubscribe function returned by StreamingCredentialsProvider
// (e.g. EntraID token rotation): if SetOnClose is called twice on the
// same Conn with DIFFERENT unsubscribe closures — for example because
// initConn ran a second time and obtained a fresh Subscribe() —
// the previous unsubscribe is dropped and will NEVER run, leaking a
// subscription on the provider. Callers must therefore ensure either:
//
//   - the provider's Subscribe is idempotent for the same listener (the
//     streaming credentials Manager deduplicates listeners by connection
//     id, so re-Subscribe returns an equivalent unsubscribe), OR
//   - the previous callback has already been invoked before SetOnClose is
//     called again.
//
// Design note: unlike the client-level onCloseHooks registry (see
// redis.baseClient), there is intentionally NO named-hook dedup or
// multi-callback support on Conn. This is a deliberate trade-off to keep
// the Conn object slim — a pool can hold thousands of Conn values and
// each one is a hot allocation, so paying for a sync.Mutex plus a
// map[string]func() error per connection to support a feature that would
// only be used by at most one subsystem today (streaming credentials) is
// not worth the per-connection memory and allocation cost. For a single
// Conn there is at most one meaningful close callback at any point in
// time, and a richer registry here would not even solve the "stale
// closure" hazard described above.
func (cn *Conn) SetOnClose(fn func() error) {
	cn.onClose = fn
}

// SetOnCscClose sets the client-side-caching close hook, overwriting any
// previous one. It runs on Close in addition to the SetOnClose callback.
func (cn *Conn) SetOnCscClose(fn func() error) {
	cn.onCscClose = fn
}

// SetOnCscReinit sets the client-side-caching pre-reinitialization hook,
// overwriting any previous one.
func (cn *Conn) SetOnCscReinit(fn func()) {
	cn.onCscReinit = fn
}

// SetInitConnFunc sets the connection initialization function to be called on reconnections.
func (cn *Conn) SetInitConnFunc(fn func(context.Context, *Conn) error) {
	cn.initConnFunc = fn
}

// ExecuteInitConn runs the stored connection initialization function if available.
func (cn *Conn) ExecuteInitConn(ctx context.Context) error {
	if cn.initConnFunc != nil {
		return cn.initConnFunc(ctx, cn)
	}
	return fmt.Errorf("redis: no initConnFunc set for conn[%d]", cn.GetID())
}

func (cn *Conn) SetNetConn(netConn net.Conn) {
	// Store the new connection atomically first (lock-free)
	cn.setNetConn(netConn)
	// Protect reader reset operations to avoid data races
	// Use write lock since we're modifying the reader state
	cn.readerMu.Lock()
	cn.rd.Reset(netConn)
	cn.readerMu.Unlock()

	cn.bw.Reset(netConn)

	// A new socket is a new server session with no HIMPORT fieldsets and
	// nothing left to discard.
	cn.ClearPreparedFieldsets(0)
}

// FieldsetPreparedVersion returns the client-side registry version at which
// the named HIMPORT fieldset was prepared on this connection's current server
// session, or 0 if it was not prepared on it (registry versions start at 1).
func (cn *Conn) FieldsetPreparedVersion(name string) uint64 {
	cn.preparedFieldsetsMu.Lock()
	version := cn.preparedFieldsets[name]
	cn.preparedFieldsetsMu.Unlock()
	return version
}

// MarkFieldsetPrepared records that the named HIMPORT fieldset was prepared
// on this connection's current server session at the given registry version.
// A session acquiring its first fieldset adopts the given discard-all epoch
// (fieldsets prepared after an HIMPORT DISCARDALL are not subject to it);
// the epoch never moves backwards, so a mark carrying an older snapshot
// cannot regress a session already wiped at a newer epoch.
func (cn *Conn) MarkFieldsetPrepared(name string, version, epoch uint64) {
	cn.preparedFieldsetsMu.Lock()
	if len(cn.preparedFieldsets) == 0 {
		cn.preparedFieldsets = make(map[string]uint64)
		if epoch > cn.preparedFieldsetsEpoch {
			cn.preparedFieldsetsEpoch = epoch
		}
	}
	cn.preparedFieldsets[name] = version
	cn.preparedFieldsetsMu.Unlock()
}

// UnmarkFieldsetPrepared forgets that the named HIMPORT fieldset was prepared
// on this connection, forcing a replay before the next HIMPORT SET using it.
func (cn *Conn) UnmarkFieldsetPrepared(name string) {
	cn.preparedFieldsetsMu.Lock()
	delete(cn.preparedFieldsets, name)
	cn.preparedFieldsetsMu.Unlock()
}

// HasPreparedFieldsets reports whether any HIMPORT fieldset is prepared on
// this connection's current server session.
func (cn *Conn) HasPreparedFieldsets() bool {
	cn.preparedFieldsetsMu.Lock()
	n := len(cn.preparedFieldsets)
	cn.preparedFieldsetsMu.Unlock()
	return n > 0
}

// PreparedFieldsetNames returns the names of the HIMPORT fieldsets prepared
// on this connection's current server session.
func (cn *Conn) PreparedFieldsetNames() []string {
	cn.preparedFieldsetsMu.Lock()
	names := make([]string, 0, len(cn.preparedFieldsets))
	for name := range cn.preparedFieldsets {
		names = append(names, name)
	}
	cn.preparedFieldsetsMu.Unlock()
	return names
}

// FieldsetEpoch returns the discard-all epoch this connection's prepared
// fieldsets belong to (0 when none were ever prepared on the session).
func (cn *Conn) FieldsetEpoch() uint64 {
	cn.preparedFieldsetsMu.Lock()
	epoch := cn.preparedFieldsetsEpoch
	cn.preparedFieldsetsMu.Unlock()
	return epoch
}

// ClearPreparedFieldsets forgets all HIMPORT fieldsets prepared on this
// connection and records the discard-all epoch the wipe corresponds to.
func (cn *Conn) ClearPreparedFieldsets(epoch uint64) {
	cn.preparedFieldsetsMu.Lock()
	cn.preparedFieldsets = nil
	cn.preparedFieldsetsEpoch = epoch
	cn.preparedFieldsetsMu.Unlock()
}

// GetNetConn safely returns the current network connection using atomic load (lock-free).
// This method is used by the pool for health checks and provides better performance.
func (cn *Conn) GetNetConn() net.Conn {
	return cn.getNetConn()
}

// SetNetConnAndInitConn replaces the underlying connection and executes the initialization.
// This method ensures only one initialization can happen at a time by using atomic state transitions.
// If another goroutine is currently initializing, this will wait for it to complete.
func (cn *Conn) SetNetConnAndInitConn(ctx context.Context, netConn net.Conn) error {
	// Wait for and transition to INITIALIZING state - this prevents concurrent initializations
	// Valid from states: CREATED (first init), IDLE (reconnect), UNUSABLE (handoff/reauth)
	// If another goroutine is initializing, we'll wait for it to finish
	// if the context has a deadline, use that, otherwise use the connection read (relaxed) timeout
	// which should be set during handoff. If it is not set, use a 5 second default
	deadline, ok := ctx.Deadline()
	if !ok {
		deadline = time.Now().Add(cn.getEffectiveReadTimeout(5 * time.Second))
	}
	waitCtx, cancel := context.WithDeadline(ctx, deadline)
	defer cancel()
	// Use predefined slice to avoid allocation
	finalState, err := cn.stateMachine.AwaitAndTransition(
		waitCtx,
		validFromCreatedIdleOrUnusable,
		StateInitializing,
	)
	if err != nil {
		return fmt.Errorf("cannot initialize connection from state %s: %w", finalState, err)
	}

	if cn.onCscReinit != nil {
		cn.onCscReinit()
	}

	// Replace the underlying connection
	cn.SetNetConn(netConn)

	// Execute initialization
	// NOTE: ExecuteInitConn (via baseClient.initConn) will transition to IDLE on success
	// or CLOSED on failure. We don't need to do it here.
	// NOTE: Initconn returns conn in IDLE state
	initErr := cn.ExecuteInitConn(ctx)
	if initErr != nil {
		// ExecuteInitConn already transitioned to CLOSED, just return the error
		return initErr
	}

	// ExecuteInitConn already transitioned to IDLE
	return nil
}

// MarkForHandoff marks the connection for handoff due to MOVING notification.
// Returns an error if the connection is already marked for handoff.
// Note: This only sets metadata - the connection state is not changed until OnPut.
// This allows the current user to finish using the connection before handoff.
func (cn *Conn) MarkForHandoff(newEndpoint string, seqID int64) error {
	// Check if already marked for handoff
	if cn.ShouldHandoff() {
		return errAlreadyMarkedForHandoff
	}

	// Set handoff metadata atomically
	cn.handoffStateAtomic.Store(&HandoffState{
		ShouldHandoff: true,
		Endpoint:      newEndpoint,
		SeqID:         seqID,
	})
	return nil
}

// MarkQueuedForHandoff marks the connection as queued for handoff processing.
// This makes the connection unusable until handoff completes.
// This is called from OnPut hook, where the connection is typically in IN_USE state.
// The pool will preserve the UNUSABLE state and not overwrite it with IDLE.
func (cn *Conn) MarkQueuedForHandoff() error {
	// Get current handoff state
	currentState := cn.handoffStateAtomic.Load()
	if currentState == nil {
		return errNotMarkedForHandoff
	}

	state := currentState.(*HandoffState)
	if !state.ShouldHandoff {
		return errNotMarkedForHandoff
	}

	// Create new state with ShouldHandoff=false but preserve endpoint and seqID
	// This prevents the connection from being queued multiple times while still
	// allowing the worker to access the handoff metadata
	newState := &HandoffState{
		ShouldHandoff: false,
		Endpoint:      state.Endpoint, // Preserve endpoint for handoff processing
		SeqID:         state.SeqID,    // Preserve seqID for handoff processing
	}

	// Atomic compare-and-swap to update state
	if !cn.handoffStateAtomic.CompareAndSwap(currentState, newState) {
		// State changed between load and CAS - retry or return error
		return errHandoffStateChanged
	}

	// Transition to UNUSABLE from IN_USE (normal flow), IDLE (edge cases), or CREATED (tests/uninitialized)
	// The connection is typically in IN_USE state when OnPut is called (normal Put flow)
	// But in some edge cases or tests, it might be in IDLE or CREATED state
	// The pool will detect this state change and preserve it (not overwrite with IDLE)
	// Use predefined slice to avoid allocation
	finalState, err := cn.stateMachine.TryTransition(validFromCreatedInUseOrIdle, StateUnusable)
	if err != nil {
		// Check if already in UNUSABLE state (race condition or retry)
		// ShouldHandoff should be false now, but check just in case
		if finalState == StateUnusable && !cn.ShouldHandoff() {
			// Already unusable - this is fine, keep the new handoff state
			return nil
		}
		// Restore the original handoff state only if nothing else changed it
		// since our CAS above. A concurrent handoff worker may have completed
		// the handoff and run ClearHandoffState in this window; a plain Store
		// would clobber that, resurrecting ShouldHandoff=true and wedging the
		// connection so it can never be acquired again. The CAS leaves the
		// worker's state intact when it has taken over.
		cn.handoffStateAtomic.CompareAndSwap(newState, currentState)
		return fmt.Errorf("failed to mark connection as unusable: %w", err)
	}
	return nil
}

// GetID returns the unique identifier for this connection.
func (cn *Conn) GetID() uint64 {
	return cn.id
}

// GetStateMachine returns the connection's state machine for advanced state management.
// This is primarily used by internal packages like maintnotifications for handoff processing.
func (cn *Conn) GetStateMachine() *ConnStateMachine {
	return cn.stateMachine
}

// TryAcquire attempts to acquire the connection for use.
// This is an optimized inline method for the hot path (Get operation).
//
// It tries to transition from IDLE -> IN_USE or CREATED -> CREATED.
// Returns true if the connection was successfully acquired, false otherwise.
// The CREATED->CREATED is done so we can keep the state correct for later
// initialization of the connection in initConn.
//
// Performance: This is faster than calling GetStateMachine() + TryTransitionFast()
//
// NOTE: We directly access cn.stateMachine.state here instead of using the state machine's
// methods. This breaks encapsulation but is necessary for performance.
// The IDLE->IN_USE and CREATED->CREATED transitions don't need
// waiter notification, and benchmarks show 1-3% improvement. If the state machine ever
// needs to notify waiters on these transitions, update this to use TryTransitionFast().
func (cn *Conn) TryAcquire() bool {
	// The || operator short-circuits, so only 1 CAS in the common case
	return cn.stateMachine.state.CompareAndSwap(uint32(StateIdle), uint32(StateInUse)) ||
		cn.stateMachine.state.CompareAndSwap(uint32(StateCreated), uint32(StateCreated))
}

// Release releases the connection back to the pool.
// This is an optimized inline method for the hot path (Put operation).
//
// It tries to transition from IN_USE -> IDLE.
// Returns true if the connection was successfully released, false otherwise.
//
// Performance: This is faster than calling GetStateMachine() + TryTransitionFast().
//
// NOTE: We directly access cn.stateMachine.state here instead of using the state machine's
// methods. This breaks encapsulation but is necessary for performance.
// If the state machine ever needs to notify waiters
// on this transition, update this to use TryTransitionFast().
func (cn *Conn) Release() bool {
	// Inline the hot path - single CAS operation
	return cn.stateMachine.state.CompareAndSwap(uint32(StateInUse), uint32(StateIdle))
}

// ClearHandoffState clears the handoff state after successful handoff.
// Makes the connection usable again.
func (cn *Conn) ClearHandoffState() {
	// Clear handoff metadata
	cn.handoffStateAtomic.Store(&HandoffState{
		ShouldHandoff: false,
		Endpoint:      "",
		SeqID:         0,
	})

	// Reset retry counter
	cn.handoffRetriesAtomic.Store(0)

	// Mark connection as usable again
	// Use state machine directly instead of deprecated SetUsable
	// probably done by initConn
	cn.stateMachine.Transition(StateIdle)
}

// HasBufferedData safely checks if the connection has buffered data.
// This method is used to avoid data races when checking for push notifications.
func (cn *Conn) HasBufferedData() bool {
	// Use read lock for concurrent access to reader state
	cn.readerMu.RLock()
	defer cn.readerMu.RUnlock()
	return cn.rd.Buffered() > 0
}

// PeekReplyTypeSafe safely peeks at the reply type.
// This method is used to avoid data races when checking for push notifications.
func (cn *Conn) PeekReplyTypeSafe() (byte, error) {
	// Use read lock for concurrent access to reader state
	cn.readerMu.RLock()
	defer cn.readerMu.RUnlock()

	if cn.rd.Buffered() <= 0 {
		return 0, fmt.Errorf("redis: can't peek reply type, no data available")
	}
	return cn.rd.PeekReplyType()
}

// PeekReplyTypeForCheck peeks at the reply type while holding readerMu, so it is
// safe against a concurrent SetNetConn resetting the reader during handoff.
// Unlike PeekReplyTypeSafe it does not require the data to already be buffered:
// the pool health check calls it after connCheck reports unexpected socket data,
// and connCheck only MSG_PEEKs, so the byte still has to be pulled from the
// socket into the reader here.
func (cn *Conn) PeekReplyTypeForCheck() (byte, error) {
	cn.readerMu.RLock()
	defer cn.readerMu.RUnlock()
	return cn.rd.PeekReplyType()
}

func (cn *Conn) Write(b []byte) (int, error) {
	// Lock-free netConn access for better performance
	if netConn := cn.getNetConn(); netConn != nil {
		return netConn.Write(b)
	}
	return 0, net.ErrClosed
}

func (cn *Conn) RemoteAddr() net.Addr {
	// Lock-free netConn access for better performance
	if netConn := cn.getNetConn(); netConn != nil {
		return netConn.RemoteAddr()
	}
	return nil
}

func (cn *Conn) WithReader(
	ctx context.Context, timeout time.Duration, fn func(rd *proto.Reader) error,
) error {
	if timeout >= 0 {
		// Use relaxed timeout if set, otherwise use provided timeout
		effectiveTimeout := cn.getEffectiveReadTimeout(timeout)

		// Get the connection directly from atomic storage
		netConn := cn.getNetConn()
		if netConn == nil {
			return errConnectionNotAvailable
		}

		if err := netConn.SetReadDeadline(cn.deadline(ctx, effectiveTimeout)); err != nil {
			return err
		}
	}
	return fn(cn.rd)
}

// WithReaderHardDeadline runs fn under a HARD read deadline of now+timeout,
// bypassing getEffectiveReadTimeout so a relaxed maintenance timeout can't extend
// it (used by the CSC drainer). Takes no context: an expired cycle ctx must not
// become the socket deadline, or the read surfaces context.DeadlineExceeded, which
// isBadConn treats as fatal.
func (cn *Conn) WithReaderHardDeadline(
	timeout time.Duration, fn func(rd *proto.Reader) error,
) (err error) {
	netConn := cn.getNetConn()
	if netConn == nil {
		return errConnectionNotAvailable
	}
	if err := netConn.SetReadDeadline(time.Now().Add(timeout)); err != nil {
		return err
	}
	defer func() {
		if clearErr := netConn.SetReadDeadline(time.Time{}); clearErr != nil {
			err = clearErr
		}
	}()
	return fn(cn.rd)
}

func (cn *Conn) WithWriter(
	ctx context.Context, timeout time.Duration, fn func(wr *proto.Writer) error,
) error {
	if timeout >= 0 {
		// Use relaxed timeout if set, otherwise use provided timeout
		effectiveTimeout := cn.getEffectiveWriteTimeout(timeout)

		// Set write deadline on the connection
		if netConn := cn.getNetConn(); netConn != nil {
			if err := netConn.SetWriteDeadline(cn.deadline(ctx, effectiveTimeout)); err != nil {
				return err
			}
		} else {
			// Connection is not available - return preallocated error
			return errConnNotAvailableForWrite
		}
	}

	// Reset the buffered writer if needed, should not happen
	if cn.bw.Buffered() > 0 {
		if netConn := cn.getNetConn(); netConn != nil {
			cn.bw.Reset(netConn)
		}
	}

	if err := fn(cn.wr); err != nil {
		return err
	}

	return cn.bw.Flush()
}

func (cn *Conn) IsClosed() bool {
	return cn.stateMachine.GetState() == StateClosed
}

func (cn *Conn) Close() error {
	for {
		state := cn.stateMachine.GetState()
		if state == StateClosed {
			return nil
		}
		if cn.stateMachine.TryTransitionFast(state, StateClosed) {
			// TryTransitionFast deliberately skips waiter notification; Close
			// still needs to wake any goroutine waiting on initialization.
			cn.stateMachine.notifyWaiters()
			break
		}
	}

	if cn.onClose != nil {
		// ignore error
		_ = cn.onClose()
		cn.onClose = nil
	}
	if cn.onCscClose != nil {
		// ignore error
		_ = cn.onCscClose()
		cn.onCscClose = nil
	}

	// Lock-free netConn access for better performance
	if netConn := cn.getNetConn(); netConn != nil {
		return netConn.Close()
	}
	return nil
}

// MaybeHasData tries to peek at the next byte in the socket without consuming it
// This is used to check if there are push notifications available
// Important: This will work on Linux, but not on Windows
func (cn *Conn) MaybeHasData() bool {
	// Lock-free netConn access for better performance
	if netConn := cn.getNetConn(); netConn != nil {
		return maybeHasData(netConn)
	}
	return false
}

// CheckForData reports whether the socket has data ready and surfaces a
// detected closed or failed socket.
func (cn *Conn) CheckForData() (bool, error) {
	if netConn := cn.getNetConn(); netConn != nil {
		return checkForData(netConn)
	}
	return false, nil
}

// MarkCscReadPending requests one conservative CSC drain after a command read
// when the transport may retain data that MaybeHasData cannot observe.
func (cn *Conn) MarkCscReadPending() {
	netConn := cn.getNetConn()
	if netConn == nil {
		return
	}
	if needsCscReadProbe(netConn) {
		cn.cscReadPending.Store(true)
	}
}

// TakeCscReadPending consumes the post-command conservative-drain request.
func (cn *Conn) TakeCscReadPending() bool {
	return cn.cscReadPending.Swap(false)
}

// TakeCscPeriodicReadPending schedules a throttled conservative read for
// transports with no readiness mechanism. It returns true at most once per
// interval, including when several drainer passes race.
func (cn *Conn) TakeCscPeriodicReadPending(interval time.Duration) bool {
	netConn := cn.getNetConn()
	if netConn == nil || interval <= 0 || !needsCscPeriodicProbe(netConn) {
		return false
	}

	now := time.Since(cn.createdAt).Nanoseconds()
	if now <= 0 {
		now = 1
	}
	for {
		last := cn.lastCscPeriodicProbeNs.Load()
		if last != 0 && now >= last && now-last < int64(interval) {
			return false
		}
		if cn.lastCscPeriodicProbeNs.CompareAndSwap(last, now) {
			return true
		}
	}
}

// deadline computes the effective deadline time based on context and timeout.
// It updates the usedAt timestamp to now.
// Uses cached time to avoid expensive syscall (max 50ms staleness is acceptable for deadline calculation).
func (cn *Conn) deadline(ctx context.Context, timeout time.Duration) time.Time {
	// Use cached time for deadline calculation (called 2x per command: read + write)
	nowNs := getCachedTimeNs()
	cn.SetUsedAtNs(nowNs)
	tm := time.Unix(0, nowNs)

	if timeout > 0 {
		tm = tm.Add(timeout)
	}

	if ctx != nil {
		deadline, ok := ctx.Deadline()
		if ok {
			if timeout == 0 {
				return deadline
			}
			if deadline.Before(tm) {
				return deadline
			}
			return tm
		}
	}

	if timeout > 0 {
		return tm
	}

	return noDeadline
}
