package pool

import (
	"context"
	"errors"
	"math/rand"
	"net"
	"sync"
	"sync/atomic"
	"time"

	"github.com/redis/go-redis/v9/internal"
	"github.com/redis/go-redis/v9/internal/proto"
)

// Connection close reason constants for metrics.
// These are used as the "reason" parameter in CloseConn() calls.
const (
	// CloseReasonStale indicates the connection was closed because it exceeded
	// the idle timeout or max lifetime.
	CloseReasonStale = "stale"

	// CloseReasonHookError indicates the connection was closed due to an error
	// in a pool hook (OnGet or OnPut).
	CloseReasonHookError = "hook_error"

	// CloseReasonAuthError indicates the connection was closed due to an
	// authentication error during re-authentication.
	CloseReasonAuthError = "auth_error"

	// CloseReasonTest is used in tests when closing connections.
	CloseReasonTest = "test"

	// CloseReasonFailover indicates the connection was closed due to a failover event.
	CloseReasonFailover = "failover"

	// CloseReasonMaintNotificationsDisabled indicates the connection enabled
	// maintenance notifications, but the client later downgraded and disabled
	// maintenance notification handling for the pool.
	CloseReasonMaintNotificationsDisabled = "maintnotifications_disabled"
)

// Metric state constants for connection state tracking.
// These represent the logical state of a connection from a metrics perspective,
// not the internal state machine state (ConnState).
const (
	// MetricStateIdle indicates the connection is idle in the pool,
	// ready to be acquired.
	MetricStateIdle = "idle"

	// MetricStateUsed indicates the connection is currently being used
	// by a client operation.
	MetricStateUsed = "used"
)

var (
	// ErrClosed performs any operation on the closed client will return this error.
	ErrClosed = errors.New("redis: client is closed")

	// ErrPoolExhausted is returned from a pool connection method
	// when the maximum number of database connections in the pool has been reached.
	ErrPoolExhausted = errors.New("redis: connection pool exhausted")

	// ErrPoolTimeout timed out waiting to get a connection from the connection pool.
	ErrPoolTimeout = errors.New("redis: connection pool timeout")

	// ErrConnUnusableTimeout is returned when a connection is not usable and we timed out trying to mark it as unusable.
	ErrConnUnusableTimeout = errors.New("redis: timed out trying to mark connection as unusable")

	// errHookRequestedRemoval is returned when a hook requests connection removal.
	errHookRequestedRemoval = errors.New("hook requested removal")

	// errConnNotPooled is returned when trying to return a non-pooled connection to the pool.
	errConnNotPooled = errors.New("connection not pooled")

	// errConnEvictedIdle is passed to OnRemove hooks when a pooled connection is evicted on
	// Put because the idle pool is already at MaxIdleConns.
	errConnEvictedIdle = errors.New("connection evicted: idle pool at capacity")

	// metricCallbackMu protects all global metric callback functions for thread-safe access.
	metricCallbackMu sync.RWMutex

	// Global metric callbacks for connection state changes
	metricConnectionStateChangeCallback func(ctx context.Context, cn *Conn, fromState, toState string)

	// Global metric callback for connection creation time
	metricConnectionCreateTimeCallback func(ctx context.Context, duration time.Duration, cn *Conn)

	// Global metric callback for connection relaxed timeout changes
	// Parameters: ctx, delta (+1/-1), cn, poolName, notificationType
	metricConnectionRelaxedTimeoutCallback func(ctx context.Context, delta int, cn *Conn, poolName, notificationType string)

	// Global metric callback for connection handoff
	// Parameters: ctx, cn, poolName
	metricConnectionHandoffCallback func(ctx context.Context, cn *Conn, poolName string)

	// Global metric callback for error tracking
	// Parameters: ctx, errorType, cn, statusCode, isInternal, retryAttempts
	metricErrorCallback func(ctx context.Context, errorType string, cn *Conn, statusCode string, isInternal bool, retryAttempts int)

	// Global metric callback for maintenance notifications
	// Parameters: ctx, cn, notificationType
	metricMaintenanceNotificationCallback func(ctx context.Context, cn *Conn, notificationType string)

	// Global metric callback for connection wait time
	// Parameters: ctx, duration, cn
	metricConnectionWaitTimeCallback func(ctx context.Context, duration time.Duration, cn *Conn)

	// Global metric callback for connection timeouts
	// Parameters: ctx, cn, timeoutType
	metricConnectionTimeoutCallback func(ctx context.Context, cn *Conn, timeoutType string)

	// Global metric callback for connection closed
	// Parameters: ctx, cn, reason, err
	metricConnectionClosedCallback func(ctx context.Context, cn *Conn, reason string, err error)

	// Global metric callback for connection count changes (UpDownCounter)
	// Parameters: ctx, delta (+1/-1), cn, state, isPubSub
	metricConnectionCountCallback func(ctx context.Context, delta int, cn *Conn, state string, isPubSub bool)

	// Global metric callback for pending requests changes (UpDownCounter)
	// Parameters: ctx, delta (+1/-1), cn, poolName
	// poolName is passed explicitly because we may not have a connection yet when request starts
	metricPendingRequestsCallback func(ctx context.Context, delta int, cn *Conn, poolName string)

	// errPanicInDial is returned when a panic occurs in the dial function.
	errPanicInQueuedNewConn = errors.New("panic in queuedNewConn")

	// popAttempts is the maximum number of attempts to find a usable connection
	// when popping from the idle connection pool. This handles cases where connections
	// are temporarily marked as unusable (e.g., during maintenanceNotifications upgrades or network issues).
	// Value of 50 provides sufficient resilience without excessive overhead.
	// This is capped by the idle connection count, so we won't loop excessively.
	popAttempts = 50

	// getAttempts is the maximum number of attempts to get a connection that passes
	// hook validation (e.g., maintenanceNotifications upgrade hooks). This protects against race conditions
	// where hooks might temporarily reject connections during cluster transitions.
	// Value of 3 balances resilience with performance - most hook rejections resolve quickly.
	getAttempts = 3

	minTime      = time.Unix(-2208988800, 0) // Jan 1, 1900
	maxTime      = minTime.Add(1<<63 - 1)
	noExpiration = maxTime
)

// MetricCallbacks holds all metric callback functions.
// Use SetAllMetricCallbacks to register all callbacks atomically.
type MetricCallbacks struct {
	// ConnectionCreateTime is called when a new connection is created
	ConnectionCreateTime func(ctx context.Context, duration time.Duration, cn *Conn)

	// ConnectionRelaxedTimeout is called when connection timeout is relaxed/unrelaxed
	// delta: +1 for relaxed, -1 for unrelaxed
	ConnectionRelaxedTimeout func(ctx context.Context, delta int, cn *Conn, poolName, notificationType string)

	// ConnectionHandoff is called when a connection is handed off to another node
	ConnectionHandoff func(ctx context.Context, cn *Conn, poolName string)

	// Error is called when an error occurs
	Error func(ctx context.Context, errorType string, cn *Conn, statusCode string, isInternal bool, retryAttempts int)

	// MaintenanceNotification is called when a maintenance notification is received
	MaintenanceNotification func(ctx context.Context, cn *Conn, notificationType string)

	// ConnectionWaitTime is called to record time spent waiting for a connection
	ConnectionWaitTime func(ctx context.Context, duration time.Duration, cn *Conn)

	// ConnectionClosed is called when a connection is closed
	ConnectionClosed func(ctx context.Context, cn *Conn, reason string, err error)

	// ConnectionCount is called when connection count changes (UpDownCounter)
	// delta: +1 when connection added, -1 when connection removed
	// state: connection state (e.g., "idle", "used")
	// isPubSub: true if this is a PubSub connection
	ConnectionCount func(ctx context.Context, delta int, cn *Conn, state string, isPubSub bool)

	// PendingRequests is called when pending requests count changes (UpDownCounter)
	// delta: +1 when request starts waiting, -1 when request stops waiting
	// poolName is passed explicitly because we may not have a connection yet when request starts
	PendingRequests func(ctx context.Context, delta int, cn *Conn, poolName string)
}

// SetAllMetricCallbacks sets all metric callbacks atomically.
// Pass nil to clear all callbacks (disable metrics).
// This ensures all callbacks are set together under a single lock,
// preventing inconsistent state during registration.
//
// Note on thread safety: After returning, there is a small window where
// concurrent getMetric* calls may return the old callback value. This is
// acceptable for metrics - at most one event may go to the old recorder
// or be missed during the transition. The callbacks themselves are immutable
// function pointers, so calling an "old" callback is safe.
func SetAllMetricCallbacks(callbacks *MetricCallbacks) {
	metricCallbackMu.Lock()
	defer metricCallbackMu.Unlock()

	if callbacks == nil {
		metricConnectionCreateTimeCallback = nil
		metricConnectionRelaxedTimeoutCallback = nil
		metricConnectionHandoffCallback = nil
		metricErrorCallback = nil
		metricMaintenanceNotificationCallback = nil
		metricConnectionWaitTimeCallback = nil
		metricConnectionClosedCallback = nil
		metricConnectionCountCallback = nil
		metricPendingRequestsCallback = nil
		return
	}

	metricConnectionCreateTimeCallback = callbacks.ConnectionCreateTime
	metricConnectionRelaxedTimeoutCallback = callbacks.ConnectionRelaxedTimeout
	metricConnectionHandoffCallback = callbacks.ConnectionHandoff
	metricErrorCallback = callbacks.Error
	metricMaintenanceNotificationCallback = callbacks.MaintenanceNotification
	metricConnectionWaitTimeCallback = callbacks.ConnectionWaitTime
	metricConnectionClosedCallback = callbacks.ConnectionClosed
	metricConnectionCountCallback = callbacks.ConnectionCount
	metricPendingRequestsCallback = callbacks.PendingRequests
}

// getMetricConnectionStateChangeCallback returns the metric callback for connection state changes.
func getMetricConnectionStateChangeCallback() func(ctx context.Context, cn *Conn, fromState, toState string) {
	metricCallbackMu.RLock()
	cb := metricConnectionStateChangeCallback
	metricCallbackMu.RUnlock()
	return cb
}

// GetMetricConnectionCreateTimeCallback returns the metric callback for connection creation time.
func GetMetricConnectionCreateTimeCallback() func(ctx context.Context, duration time.Duration, cn *Conn) {
	metricCallbackMu.RLock()
	cb := metricConnectionCreateTimeCallback
	metricCallbackMu.RUnlock()
	return cb
}

// GetMetricConnectionRelaxedTimeoutCallback returns the metric callback for connection relaxed timeout changes.
// This is used by maintnotifications to record relaxed timeout metrics.
func GetMetricConnectionRelaxedTimeoutCallback() func(ctx context.Context, delta int, cn *Conn, poolName, notificationType string) {
	metricCallbackMu.RLock()
	cb := metricConnectionRelaxedTimeoutCallback
	metricCallbackMu.RUnlock()
	return cb
}

// GetMetricConnectionHandoffCallback returns the metric callback for connection handoffs.
// This is used by maintnotifications to record handoff metrics.
func GetMetricConnectionHandoffCallback() func(ctx context.Context, cn *Conn, poolName string) {
	metricCallbackMu.RLock()
	cb := metricConnectionHandoffCallback
	metricCallbackMu.RUnlock()
	return cb
}

// GetMetricErrorCallback returns the metric callback for error tracking.
// This is used by cluster and client code to record error metrics.
func GetMetricErrorCallback() func(ctx context.Context, errorType string, cn *Conn, statusCode string, isInternal bool, retryAttempts int) {
	metricCallbackMu.RLock()
	cb := metricErrorCallback
	metricCallbackMu.RUnlock()
	return cb
}

// GetMetricMaintenanceNotificationCallback returns the metric callback for maintenance notifications.
// This is used by maintnotifications to record notification metrics.
func GetMetricMaintenanceNotificationCallback() func(ctx context.Context, cn *Conn, notificationType string) {
	metricCallbackMu.RLock()
	cb := metricMaintenanceNotificationCallback
	metricCallbackMu.RUnlock()
	return cb
}

func getMetricConnectionWaitTimeCallback() func(ctx context.Context, duration time.Duration, cn *Conn) {
	metricCallbackMu.RLock()
	cb := metricConnectionWaitTimeCallback
	metricCallbackMu.RUnlock()
	return cb
}

func getMetricConnectionTimeoutCallback() func(ctx context.Context, cn *Conn, timeoutType string) {
	metricCallbackMu.RLock()
	cb := metricConnectionTimeoutCallback
	metricCallbackMu.RUnlock()
	return cb
}

func getMetricConnectionClosedCallback() func(ctx context.Context, cn *Conn, reason string, err error) {
	metricCallbackMu.RLock()
	cb := metricConnectionClosedCallback
	metricCallbackMu.RUnlock()
	return cb
}

// getMetricConnectionCountCallback returns the metric callback for connection count changes (UpDownCounter).
func getMetricConnectionCountCallback() func(ctx context.Context, delta int, cn *Conn, state string, isPubSub bool) {
	metricCallbackMu.RLock()
	cb := metricConnectionCountCallback
	metricCallbackMu.RUnlock()
	return cb
}

// getMetricPendingRequestsCallback returns the metric callback for pending requests changes (UpDownCounter).
func getMetricPendingRequestsCallback() func(ctx context.Context, delta int, cn *Conn, poolName string) {
	metricCallbackMu.RLock()
	cb := metricPendingRequestsCallback
	metricCallbackMu.RUnlock()
	return cb
}

// Stats contains pool state information and accumulated stats.
//
// TODO(cxl): the uint32/int64 fields below will be changed to atomic value
// types (atomic.Uint32/atomic.Int64) in v10, which is a breaking API change.
type Stats struct {
	Hits           uint32 // number of times free connection was found in the pool
	Misses         uint32 // number of times free connection was NOT found in the pool
	Timeouts       uint32 // number of times a wait timeout occurred
	WaitCount      uint32 // number of times a connection was waited
	Unusable       uint32 // number of times a connection was found to be unusable
	WaitDurationNs int64  // total time spent for waiting a connection in nanoseconds

	TotalConns      uint32 // number of total connections in the pool
	IdleConns       uint32 // number of idle connections in the pool
	StaleConns      uint32 // number of stale connections removed from the pool
	PendingRequests uint32 // number of pending requests waiting for a connection

	PubSubStats PubSubStats

	// PipelineStats holds the stats of the separate pipeline connection pool
	// when one is configured (PipelineReadBufferSize/PipelineWriteBufferSize).
	// nil when pipelines share the main pool.
	PipelineStats *Stats
}

type ConnRetirer interface {
	RetireConns(ctx context.Context, conns []*Conn, reason string)
}

type Pooler interface {
	NewConn(context.Context) (*Conn, error)
	CloseConn(ctx context.Context, cn *Conn, reason string, fromState string) error

	Get(context.Context) (*Conn, error)
	Put(context.Context, *Conn)
	Remove(context.Context, *Conn, error)

	Len() int
	IdleLen() int
	Stats() *Stats

	// Size returns the maximum pool size (capacity).
	// This is used by the streaming credentials manager to size the re-auth worker pool.
	Size() int

	AddPoolHook(hook PoolHook)
	RemovePoolHook(hook PoolHook)

	// RemoveWithoutTurn removes a connection from the pool without freeing a turn.
	// This should be used when removing a connection from a context that didn't acquire
	// a turn via Get() (e.g., background workers, cleanup tasks).
	// For normal removal after Get(), use Remove() instead.
	RemoveWithoutTurn(context.Context, *Conn, error)

	Close() error
}

type Options struct {
	Dialer          func(context.Context) (net.Conn, error)
	ReadBufferSize  int
	WriteBufferSize int

	PoolFIFO                 bool
	PoolSize                 int32
	MaxConcurrentDials       int
	DialTimeout              time.Duration
	PoolTimeout              time.Duration
	MinIdleConns             int32
	MaxIdleConns             int32
	MaxActiveConns           int32
	ConnMaxIdleTime          time.Duration
	ConnMaxLifetime          time.Duration
	ConnMaxLifetimeJitter    time.Duration
	PushNotificationsEnabled bool

	// DialerRetries is the maximum number of retry attempts when dialing fails.
	// Default: 5
	DialerRetries int

	// DialerRetryTimeout is the backoff duration between retry attempts.
	// Default: 100ms
	DialerRetryTimeout time.Duration

	// DialerRetryBackoff controls the delay between dial retry attempts.
	// If nil, dial retry backoff is constant and equals DialerRetryTimeout (default: 100ms).
	DialerRetryBackoff func(attempt int) time.Duration

	// Name is a unique identifier for this pool, used in metrics.
	// Format: addr_uniqueID (e.g., "localhost:6379_a1b2c3d4")
	Name string
}

type lastDialErrorWrap struct {
	err error
}

type ConnPool struct {
	cfg *Options

	dialErrorsNum atomic.Uint32
	lastDialError atomic.Value

	dialsInProgress chan struct{}
	dialsQueue      *wantConnQueue
	// Fast semaphore for connection limiting with eventual fairness
	// Uses fast path optimization to avoid timer allocation when tokens are available
	semaphore *internal.FastSemaphore

	connsMu   sync.Mutex
	conns     map[uint64]*Conn
	idleConns []*Conn

	poolSize            atomic.Int32
	idleConnsLen        atomic.Int32
	idleCheckInProgress atomic.Bool
	idleCheckNeeded     atomic.Bool

	stats          Stats
	waitDurationNs atomic.Int64

	_closed atomic.Uint32

	// Pool hooks manager. atomic.Pointer keeps hot-path reads (Get/Put)
	// lock-free; hookMu serializes Add/RemovePoolHook's read-clone-store so
	// concurrent mutators (e.g. maintnotifications and CSC) can't lose an update.
	hookManager atomic.Pointer[PoolHookManager]
	hookMu      sync.Mutex

	// drainMu/drainDone coordinate the CSC drainer's temporary idle-connection
	// claim with Get. The normal semaphore retains its PoolSize capacity (and
	// therefore the established MaxActiveConns/ErrPoolExhausted behavior); a Get
	// that finds the idle list empty only because the drainer borrowed a conn
	// waits for that short claim to finish instead of opening an overflow conn.
	drainMu         sync.Mutex
	drainDone       chan struct{}
	drainBorrowed   int
	drainGeneration atomic.Uint64
}

var _ Pooler = (*ConnPool)(nil)

func NewConnPool(opt *Options) *ConnPool {
	p := &ConnPool{
		cfg:             opt,
		semaphore:       internal.NewFastSemaphore(opt.PoolSize),
		conns:           make(map[uint64]*Conn),
		dialsInProgress: make(chan struct{}, opt.MaxConcurrentDials),
		dialsQueue:      newWantConnQueue(),
		idleConns:       make([]*Conn, 0, opt.PoolSize),
	}

	// Only create MinIdleConns if explicitly requested (> 0)
	// This avoids creating connections during pool initialization for tests
	if opt.MinIdleConns > 0 {
		p.connsMu.Lock()
		p.checkMinIdleConns()
		p.connsMu.Unlock()
	}

	return p
}

// initializeHooks sets up the pool hooks system.
func (p *ConnPool) initializeHooks() {
	manager := NewPoolHookManager()
	p.hookManager.Store(manager)
}

// AddPoolHook adds a pool hook to the pool.
func (p *ConnPool) AddPoolHook(hook PoolHook) {
	// Serialize so a concurrent Add/Remove can't clobber this change.
	p.hookMu.Lock()
	defer p.hookMu.Unlock()

	manager := p.hookManager.Load()
	if manager == nil {
		p.initializeHooks()
		manager = p.hookManager.Load()
	}

	// Create new manager with added hook
	newManager := manager.Clone()
	newManager.AddHook(hook)

	// Atomically swap to new manager (hot-path readers load lock-free)
	p.hookManager.Store(newManager)
}

// SupportsPoolHooks reports that AddPoolHook and RemovePoolHook are functional.
// Pooler adapters with no-op hook methods intentionally do not expose this
// optional capability.
func (p *ConnPool) SupportsPoolHooks() bool {
	return true
}

// RemovePoolHook removes a pool hook from the pool.
func (p *ConnPool) RemovePoolHook(hook PoolHook) {
	p.hookMu.Lock()
	defer p.hookMu.Unlock()

	manager := p.hookManager.Load()
	if manager != nil {
		// Create new manager with removed hook
		newManager := manager.Clone()
		newManager.RemoveHook(hook)

		// Atomically swap to new manager
		p.hookManager.Store(newManager)
	}
}

func (p *ConnPool) checkMinIdleConns() {
	// If a check is already in progress, mark that we need another check and return
	if !p.idleCheckInProgress.CompareAndSwap(false, true) {
		p.idleCheckNeeded.Store(true)
		return
	}

	if p.cfg.MinIdleConns == 0 {
		p.idleCheckInProgress.Store(false)
		return
	}

	// Keep checking until no more checks are needed
	// This handles the case where multiple Remove() calls happen concurrently
	for {
		// Clear the "check needed" flag before we start
		p.idleCheckNeeded.Store(false)

		// Only create idle connections if we haven't reached the total pool size limit
		// MinIdleConns should be a subset of PoolSize, not additional connections
		for p.poolSize.Load() < p.cfg.PoolSize && p.idleConnsLen.Load() < p.cfg.MinIdleConns {
			// Try to acquire a semaphore token
			if !p.semaphore.TryAcquire() {
				// Semaphore is full, can't create more connections right now
				// Break out of inner loop to check if we need to retry
				break
			}

			p.poolSize.Add(1)
			p.idleConnsLen.Add(1)
			go func() {
				defer func() {
					if err := recover(); err != nil {
						p.poolSize.Add(-1)
						p.idleConnsLen.Add(-1)

						p.freeTurn()
						internal.Logger.Printf(context.Background(), "addIdleConn panic: %+v", err)
					}
				}()

				err := p.addIdleConn()
				if err != nil && err != ErrClosed {
					p.poolSize.Add(-1)
					p.idleConnsLen.Add(-1)
				}
				p.freeTurn()
			}()
		}

		// If no one requested another check while we were working, we're done
		if !p.idleCheckNeeded.Load() {
			p.idleCheckInProgress.Store(false)
			return
		}

		// Otherwise, loop again to handle the new requests
	}
}

func (p *ConnPool) addIdleConn() error {
	// Do not apply DialTimeout via context here; dialConn applies DialTimeout per attempt.
	cn, err := p.dialConn(context.Background(), true)
	if err != nil {
		return err
	}

	// NOTE: Connection is in CREATED state and will be initialized by redis.go:initConn()
	// when first acquired from the pool. Do NOT transition to IDLE here - that happens
	// after initialization completes.

	p.connsMu.Lock()
	defer p.connsMu.Unlock()

	// It is not allowed to add new connections to the closed connection pool.
	if p.closed() {
		_ = cn.Close()
		return ErrClosed
	}

	p.conns[cn.GetID()] = cn
	p.idleConns = append(p.idleConns, cn)

	// Record connection count increment (new idle connection from min-idle prewarm)
	if cb := getMetricConnectionCountCallback(); cb != nil {
		cb(context.Background(), 1, cn, "idle", false)
	}

	return nil
}

// NewConn creates a new connection and returns it to the user.
// This will still obey MaxActiveConns but will not include it in the pool and won't increase the pool size.
//
// NOTE: If you directly get a connection from the pool, it won't be pooled and won't support maintnotifications upgrades.
func (p *ConnPool) NewConn(ctx context.Context) (*Conn, error) {
	return p.newConn(ctx, false)
}

func (p *ConnPool) newConn(ctx context.Context, pooled bool) (*Conn, error) {
	if p.closed() {
		return nil, ErrClosed
	}

	if p.cfg.MaxActiveConns > 0 && p.poolSize.Load() >= p.cfg.MaxActiveConns {
		return nil, ErrPoolExhausted
	}

	// Protect against nil context due to race condition in queuedNewConn
	// where the context can be set to nil after timeout/cancellation
	if ctx == nil {
		ctx = context.Background()
	}

	// Do not apply DialTimeout via context here; dialConn applies DialTimeout per attempt.
	// We still propagate ctx so callers can cancel explicitly.
	cn, err := p.dialConn(ctx, pooled)
	if err != nil {
		return nil, err
	}

	// NOTE: Connection is in CREATED state and will be initialized by redis.go:initConn()
	// when first used. Do NOT transition to IDLE here - that happens after initialization completes.
	// The state machine flow is: CREATED → INITIALIZING (in initConn) → IDLE (after init success)

	if p.cfg.MaxActiveConns > 0 && p.poolSize.Load() > p.cfg.MaxActiveConns {
		_ = cn.Close()
		return nil, ErrPoolExhausted
	}

	p.connsMu.Lock()
	defer p.connsMu.Unlock()
	if p.closed() {
		_ = cn.Close()
		return nil, ErrClosed
	}
	// Check if pool was closed while we were waiting for the lock
	if p.conns == nil {
		p.conns = make(map[uint64]*Conn)
	}
	p.conns[cn.GetID()] = cn

	if pooled {
		// If pool is full remove the cn on next Put.
		currentPoolSize := p.poolSize.Load()
		if currentPoolSize >= p.cfg.PoolSize {
			cn.pooled = false
		} else {
			p.poolSize.Add(1)
		}
	}

	// All new connections start as "used" metrically. For the miss path in getConn,
	// this is the final state. For putIdleConn (undelivered conn), a used→idle
	// transition is emitted when it's added to idleConns.
	if cb := getMetricConnectionStateChangeCallback(); cb != nil {
		cb(ctx, cn, "", MetricStateUsed)
	}
	if cb := getMetricConnectionCountCallback(); cb != nil {
		cb(ctx, 1, cn, "used", false)
	}

	return cn, nil
}

func (p *ConnPool) dialConn(ctx context.Context, pooled bool) (*Conn, error) {
	if p.closed() {
		return nil, ErrClosed
	}

	if p.dialErrorsNum.Load() >= uint32(p.cfg.PoolSize) {
		return nil, p.getLastDialError()
	}

	// Record dial start time for connection creation metric
	// This will be used after handshake completes in redis.go _getConn()
	// Only call time.Now() if callback is registered to avoid overhead
	var dialStartNs int64
	if GetMetricConnectionCreateTimeCallback() != nil {
		dialStartNs = time.Now().UnixNano()
	}

	// Retry dialing with backoff
	// Dial timeout is applied per attempt (so retries/backoff don't eat into the next
	// attempt's dial budget), while still honoring caller cancellation via ctx.
	maxRetries := p.cfg.DialerRetries
	if maxRetries <= 0 {
		maxRetries = 5 // Default value
	}

	var lastErr error
	shouldLoop := true
	// when the timeout is reached, we should stop retrying
	// but keep the lastErr to return to the caller
	// instead of a generic context deadline exceeded error
	attempt := 0
	for attempt = 0; (attempt < maxRetries) && shouldLoop; attempt++ {
		attemptCtx := ctx
		var cancel context.CancelFunc
		if p.cfg.DialTimeout > 0 {
			// Apply DialTimeout per attempt, but never extend an existing earlier deadline.
			if deadline, ok := ctx.Deadline(); !ok || time.Until(deadline) > p.cfg.DialTimeout {
				attemptCtx, cancel = context.WithTimeout(ctx, p.cfg.DialTimeout)
			}
		}

		netConn, err := p.cfg.Dialer(attemptCtx)
		if cancel != nil {
			cancel()
		}
		if err != nil {
			lastErr = err
			// Add backoff delay for retry attempts
			// (not for the first attempt, do at least one)
			// Do not sleep after the last attempt.
			if attempt+1 < maxRetries {
				backoffDuration := p.dialRetryBackoff(attempt)
				select {
				case <-ctx.Done():
					shouldLoop = false
				case <-time.After(backoffDuration):
					// Continue with retry
				}
			}
			continue
		}

		cn := NewConnWithBufferSize(netConn, p.cfg.ReadBufferSize, p.cfg.WriteBufferSize)
		cn.pooled = pooled
		// Store dial start time only if we recorded it
		if dialStartNs > 0 {
			cn.dialStartNs.Store(dialStartNs)
		}
		cn.expiresAt = p.calcConnExpiresAt()
		// Set pool name for metrics
		cn.SetPoolName(p.cfg.Name)

		return cn, nil
	}

	internal.Logger.Printf(ctx, "redis: connection pool: failed to dial after %d attempts: %v", attempt, lastErr)
	// All retries failed - handle error tracking
	p.setLastDialError(lastErr)
	if p.dialErrorsNum.Add(1) == uint32(p.cfg.PoolSize) {
		go p.tryDial()
	}
	return nil, lastErr
}

func (p *ConnPool) dialRetryBackoff(attempt int) time.Duration {
	if p.cfg.DialerRetryBackoff != nil {
		d := p.cfg.DialerRetryBackoff(attempt)
		if d < 0 {
			return 0
		}
		return d
	}

	base := p.cfg.DialerRetryTimeout
	if base <= 0 {
		base = 100 * time.Millisecond
	}
	return base
}

// calcConnExpiresAt calculates the expiration time for a connection.
// It applies random jitter to prevent all connections from expiring simultaneously,
// avoiding the "thundering herd" problem where all connections expire at once.
// Returns noExpiration if ConnMaxLifetime is not set.
func (p *ConnPool) calcConnExpiresAt() time.Time {
	if p.cfg.ConnMaxLifetime <= 0 {
		return noExpiration
	}

	if p.cfg.ConnMaxLifetimeJitter <= 0 {
		return time.Now().Add(p.cfg.ConnMaxLifetime)
	}

	jitter := p.cfg.ConnMaxLifetimeJitter
	jitterRange := jitter.Nanoseconds() * 2
	jitterNs := rand.Int63n(jitterRange) - jitter.Nanoseconds()
	return time.Now().Add(p.cfg.ConnMaxLifetime + time.Duration(jitterNs))
}

func (p *ConnPool) tryDial() {
	for {
		if p.closed() {
			return
		}

		// Probe dialing even when dialErrorsNum is saturated. Apply DialTimeout per probe
		// attempt so custom dialers can't hang indefinitely.
		ctx := context.Background()
		var cancel context.CancelFunc
		if p.cfg.DialTimeout > 0 {
			ctx, cancel = context.WithTimeout(ctx, p.cfg.DialTimeout)
		}

		conn, err := p.cfg.Dialer(ctx)
		if cancel != nil {
			cancel()
		}
		if err != nil {
			p.setLastDialError(err)
			time.Sleep(time.Second)
			continue
		}

		p.dialErrorsNum.Store(0)
		_ = conn.Close()
		return
	}
}

func (p *ConnPool) setLastDialError(err error) {
	p.lastDialError.Store(&lastDialErrorWrap{err: err})
}

func (p *ConnPool) getLastDialError() error {
	err, _ := p.lastDialError.Load().(*lastDialErrorWrap)
	if err != nil {
		return err.err
	}
	return nil
}

// Get returns existed connection from the pool or creates a new one.
func (p *ConnPool) Get(ctx context.Context) (*Conn, error) {
	return p.getConn(ctx)
}

// getConn returns a connection from the pool.
func (p *ConnPool) getConn(ctx context.Context) (cn *Conn, err error) {
	if p.closed() {
		return nil, ErrClosed
	}

	// Track pending requests in pool stats
	atomic.AddUint32(&p.stats.PendingRequests, 1)
	// Record pending request increment (UpDownCounter)
	// Pass pool name explicitly since we don't have a connection yet
	poolName := p.cfg.Name
	if cb := getMetricPendingRequestsCallback(); cb != nil {
		cb(ctx, 1, nil, poolName)
	}
	defer func() {
		if err != nil {
			// Failed to get connection, decrement pending requests
			atomic.AddUint32(&p.stats.PendingRequests, ^uint32(0)) // -1
			// Record pending request decrement on failure
			if cb := getMetricPendingRequestsCallback(); cb != nil {
				cb(ctx, -1, nil, poolName)
			}
		}
		if err == ErrPoolTimeout {
			atomic.AddUint32(&p.stats.Timeouts, 1)
			if cb := getMetricConnectionTimeoutCallback(); cb != nil {
				cb(ctx, nil, "pool")
			}
			if cb := GetMetricErrorCallback(); cb != nil {
				cb(ctx, "POOL_TIMEOUT", nil, "POOL_TIMEOUT", true, 0)
			}
		}
	}()

	// PoolTimeout is one budget for both the pool turn and a drainer handoff.
	poolDeadline := time.Now().Add(p.cfg.PoolTimeout)

	// Connection wait time measures only semaphore acquisition.
	var waitStart time.Time
	var waitDuration time.Duration
	waitTimeCallback := getMetricConnectionWaitTimeCallback()
	if waitTimeCallback != nil {
		waitStart = time.Now()
	}
	if err = p.waitTurn(ctx); err != nil {
		return nil, err
	}
	if waitTimeCallback != nil {
		waitDuration = time.Since(waitStart)
	}

	// Use cached time for health checks (max 50ms staleness is acceptable)
	nowNs := getCachedTimeNs()

	// Lock-free atomic read - no mutex overhead!
	hookManager := p.hookManager.Load()

retryIdle:
	drainGeneration := p.drainGeneration.Load()
	for attempts := 0; attempts < getAttempts; attempts++ {

		p.connsMu.Lock()
		cn, err = p.popIdle()
		if cn != nil {
			// Emit idle→used transition inside the lock so Close() sees
			// consistent state (conn removed from idleConns = "used").
			if cb := getMetricConnectionStateChangeCallback(); cb != nil {
				cb(ctx, cn, MetricStateIdle, MetricStateUsed)
			}
			if cb := getMetricConnectionCountCallback(); cb != nil {
				cb(ctx, -1, cn, "idle", false)
				cb(ctx, 1, cn, "used", false)
			}
		}
		p.connsMu.Unlock()

		if err != nil {
			p.freeTurn()
			return nil, err
		}

		if cn == nil {
			break
		}

		if !p.isHealthyConn(cn, nowNs) {
			// Connection was already transitioned to MetricStateUsed under the lock above.
			_ = p.CloseConn(ctx, cn, CloseReasonStale, MetricStateUsed)
			continue
		}

		// Process connection using the hooks system
		// Combine error and rejection checks to reduce branches
		if hookManager != nil {
			acceptConn, hookErr := hookManager.ProcessOnGet(ctx, cn, false)
			if hookErr != nil || !acceptConn {
				if hookErr != nil {
					internal.Logger.Printf(ctx, "redis: connection pool: failed to process idle connection by hook: %v", hookErr)
					// Connection was already transitioned to MetricStateUsed under the lock above.
					_ = p.CloseConn(ctx, cn, CloseReasonHookError, MetricStateUsed)
				} else {
					internal.Logger.Printf(ctx, "redis: connection pool: conn[%d] rejected by hook, returning to pool", cn.GetID())
					// Connection is already in MetricStateUsed (transitioned under the lock above).
					// Return connection to pool without freeing the turn that this Get() call holds.
					// putConnWithoutTurn will emit used→idle transition.
					p.putConnWithoutTurn(ctx, cn)
					cn = nil
				}
				continue
			}
		}

		atomic.AddUint32(&p.stats.Hits, 1)

		// Record wait time (use cached callback from above)
		if waitTimeCallback != nil {
			waitTimeCallback(ctx, waitDuration, cn)
		}

		// Decrement pending requests (connection acquired successfully)
		atomic.AddUint32(&p.stats.PendingRequests, ^uint32(0)) // -1
		// Record pending request decrement (UpDownCounter)
		if cb := getMetricPendingRequestsCallback(); cb != nil {
			cb(ctx, -1, cn, poolName)
		}

		return cn, nil
	}

	// If the CSC drainer removed the only idle connection during this scan,
	// wait for that bounded maintenance claim and retry. The generation closes
	// the race where the drainer returns the connection between popIdle and this
	// check. Normal MaxActiveConns exhaustion still proceeds to newConn and
	// returns ErrPoolExhausted immediately, preserving the existing contract.
	if done, retry := p.drainerWaitState(drainGeneration); done != nil {
		if err = p.waitForDrainer(ctx, done, poolDeadline); err != nil {
			p.freeTurn()
			return nil, err
		}
		goto retryIdle
	} else if retry {
		goto retryIdle
	}

	atomic.AddUint32(&p.stats.Misses, 1)

	var newcn *Conn
	newcn, err = p.queuedNewConn(ctx)
	if err != nil {
		return nil, err
	}

	// Process connection using the hooks system
	// This includes the handshake (HELLO/AUTH) via initConn hook
	if hookManager != nil {
		var acceptConn bool
		acceptConn, err = hookManager.ProcessOnGet(ctx, newcn, true)
		// both errors and accept=false mean a hook rejected the connection
		// this should not happen with a new connection, but we handle it gracefully
		if err != nil || !acceptConn {
			internal.Logger.Printf(ctx, "redis: connection pool: failed to process new connection conn[%d] by hook: accept=%v, err=%v", newcn.GetID(), acceptConn, err)
			// newConn emitted +1 used; CloseConn will emit -1 used if we own the removal.
			_ = p.CloseConn(ctx, newcn, CloseReasonHookError, MetricStateUsed)
			return nil, err
		}

		// Record connection creation time metric when hooks are used.
		// When hookManager is set, ProcessOnGet initializes the connection (AUTH/HELLO),
		// causing IsInited()=true. This means _getConn() in redis.go will take the
		// early return path and never reach its create time recording.
		// When hookManager is nil, _getConn() handles both initialization and create time recording.
		if dialStartNs := newcn.GetDialStartNs(); newcn.IsInited() && dialStartNs > 0 {
			if cb := GetMetricConnectionCreateTimeCallback(); cb != nil {
				duration := time.Duration(time.Now().UnixNano() - dialStartNs)
				cb(ctx, duration, newcn)
			}
		}
	}

	// newConn already emitted +1 used, so no transition needed here.

	// Record wait time (use cached callback from above)
	if waitTimeCallback != nil {
		waitTimeCallback(ctx, waitDuration, newcn)
	}

	// Decrement pending requests (connection acquired successfully)
	atomic.AddUint32(&p.stats.PendingRequests, ^uint32(0)) // -1
	// Record pending request decrement (UpDownCounter)
	if cb := getMetricPendingRequestsCallback(); cb != nil {
		cb(ctx, -1, newcn, poolName)
	}

	return newcn, nil
}

func (p *ConnPool) queuedNewConn(ctx context.Context) (*Conn, error) {
	select {
	case p.dialsInProgress <- struct{}{}:
		// Got permission, proceed to create connection
	case <-ctx.Done():
		p.freeTurn()
		return nil, ctx.Err()
	}

	// Don't apply DialTimeout via context here; dialConn applies DialTimeout per attempt.
	dialCtx, cancel := context.WithCancel(context.Background())

	w := &wantConn{
		ctx:       dialCtx,
		cancelCtx: cancel,
		result:    make(chan wantConnResult, 1),
	}
	var err error
	defer func() {
		if err != nil {
			if cn := w.cancel(); cn != nil && p.putIdleConn(ctx, cn) {
				p.freeTurn()
			}
		}
	}()

	p.dialsQueue.discardDoneAtFront()
	p.dialsQueue.enqueue(w)

	go func(w *wantConn) {
		var freeTurnCalled bool
		defer func() {
			if err := recover(); err != nil {
				w.tryDeliver(nil, errPanicInQueuedNewConn)
				p.dialsQueue.discardDoneAtFront()
				if !freeTurnCalled {
					p.freeTurn()
				}
				internal.Logger.Printf(context.Background(), "queuedNewConn panic: %+v", err)
			}
		}()

		defer w.cancelCtx()
		defer func() { <-p.dialsInProgress }() // Release connection creation permission

		dialCtx := w.getCtxForDial()
		cn, cnErr := p.newConn(dialCtx, true)
		if cnErr != nil {
			w.tryDeliver(nil, cnErr) // deliver error to caller, notify connection creation failed
			p.dialsQueue.discardDoneAtFront()
			p.freeTurn()
			freeTurnCalled = true
			return
		}

		delivered := w.tryDeliver(cn, cnErr)
		p.dialsQueue.discardDoneAtFront()
		if !delivered && p.putIdleConn(dialCtx, cn) {
			p.freeTurn()
			freeTurnCalled = true
		}
	}(w)

	select {
	case <-ctx.Done():
		err = ctx.Err()
		return nil, err
	case result := <-w.result:
		err = result.err
		return result.cn, err
	}
}

// putIdleConn puts a connection back to the pool or passes it to the next waiting request.
//
// It returns true if the connection was put back to the pool,
// which means the turn needs to be freed directly by the caller,
// or false if the connection was passed to the next waiting request,
// which means the turn will be freed by the waiting goroutine after it returns.
func (p *ConnPool) putIdleConn(ctx context.Context, cn *Conn) bool {
	for {
		w, ok := p.dialsQueue.dequeue()
		if !ok {
			break
		}
		if w.tryDeliver(cn, nil) {
			return false
		}
	}

	p.connsMu.Lock()
	defer p.connsMu.Unlock()

	if p.closed() {
		// Don't close here — this connection is still in p.conns and Close()
		// will handle closing it and emitting the correct metric decrements.
		// We just skip adding it to idleConns.
		return true
	}

	p.idleConns = append(p.idleConns, cn)
	p.idleConnsLen.Add(1)

	// Connection was created as "used" in newConn; transition to idle.
	if cb := getMetricConnectionStateChangeCallback(); cb != nil {
		cb(ctx, cn, MetricStateUsed, MetricStateIdle)
	}
	if cb := getMetricConnectionCountCallback(); cb != nil {
		cb(ctx, -1, cn, "used", false)
		cb(ctx, 1, cn, "idle", false)
	}

	return true
}

func (p *ConnPool) waitTurn(ctx context.Context) error {
	// Fast path: check context first
	select {
	case <-ctx.Done():
		return ctx.Err()
	default:
	}

	// Fast path: try to acquire without blocking
	if p.semaphore.TryAcquire() {
		return nil
	}

	// Slow path: need to wait
	start := time.Now()
	err := p.semaphore.Acquire(ctx, p.cfg.PoolTimeout, ErrPoolTimeout)
	if err != nil {
		return err
	}

	p.waitDurationNs.Add(time.Now().UnixNano() - start.UnixNano())
	atomic.AddUint32(&p.stats.WaitCount, 1)
	return nil
}

func (p *ConnPool) freeTurn() {
	p.semaphore.Release()
}

func (p *ConnPool) beginDrainerBorrow() {
	p.drainMu.Lock()
	if p.drainBorrowed == 0 {
		p.drainDone = make(chan struct{})
	}
	p.drainBorrowed++
	p.drainMu.Unlock()
}

func (p *ConnPool) endDrainerBorrow() {
	p.drainMu.Lock()
	p.drainBorrowed--
	if p.drainBorrowed == 0 {
		close(p.drainDone)
		p.drainDone = nil
		p.drainGeneration.Add(1)
	}
	p.drainMu.Unlock()
}

// drainerWaitState returns the current drain epoch's completion channel. If no
// drain is active, retry reports whether an epoch completed during the caller's
// idle scan and the idle list therefore needs to be checked again.
func (p *ConnPool) drainerWaitState(generation uint64) (done <-chan struct{}, retry bool) {
	p.drainMu.Lock()
	defer p.drainMu.Unlock()
	if p.drainBorrowed > 0 {
		return p.drainDone, false
	}
	return nil, p.drainGeneration.Load() != generation
}

func (p *ConnPool) waitForDrainer(
	ctx context.Context, done <-chan struct{}, poolDeadline time.Time,
) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	select {
	case <-done:
		return nil
	default:
	}
	remaining := time.Until(poolDeadline)
	if remaining <= 0 {
		return ErrPoolTimeout
	}
	timer := time.NewTimer(remaining)
	defer timer.Stop()

	select {
	case <-done:
		return nil
	case <-ctx.Done():
		return ctx.Err()
	case <-timer.C:
		// Prefer a caller cancellation that raced with the pool timeout.
		if err := ctx.Err(); err != nil {
			return err
		}
		return ErrPoolTimeout
	}
}

func (p *ConnPool) popIdle() (*Conn, error) {
	if p.closed() {
		return nil, ErrClosed
	}
	defer p.checkMinIdleConns()

	n := len(p.idleConns)
	if n == 0 {
		return nil, nil
	}

	var cn *Conn
	attempts := 0

	maxAttempts := min(popAttempts, n)
	for attempts < maxAttempts {
		if len(p.idleConns) == 0 {
			return nil, nil
		}

		if p.cfg.PoolFIFO {
			cn = p.idleConns[0]
			copy(p.idleConns, p.idleConns[1:])
			p.idleConns = p.idleConns[:len(p.idleConns)-1]
		} else {
			idx := len(p.idleConns) - 1
			cn = p.idleConns[idx]
			p.idleConns = p.idleConns[:idx]
		}
		attempts++

		// Hot path optimization: try IDLE → IN_USE or CREATED → IN_USE transition
		// Using inline TryAcquire() method for better performance (avoids pointer dereference)
		if cn.TryAcquire() {
			// Successfully acquired the connection
			p.idleConnsLen.Add(-1)
			break
		}

		// Connection is in UNUSABLE, INITIALIZING, or other state - skip it

		// Connection is not in a valid state (might be UNUSABLE for handoff/re-auth, INITIALIZING, etc.)
		// Put it back in the pool and try the next one
		if p.cfg.PoolFIFO {
			// FIFO: put at end (will be picked up last since we pop from front)
			p.idleConns = append(p.idleConns, cn)
		} else {
			// LIFO: put at beginning (will be picked up last since we pop from end)
			p.idleConns = append([]*Conn{cn}, p.idleConns...)
		}
		cn = nil
	}

	// If we exhausted all attempts without finding a usable connection, return nil
	if attempts > 1 && attempts >= maxAttempts && int32(attempts) >= p.poolSize.Load() {
		internal.Logger.Printf(context.Background(), "redis: connection pool: failed to get a usable connection after %d attempts", attempts)
		return nil, nil
	}

	return cn, nil
}

func (p *ConnPool) Put(ctx context.Context, cn *Conn) {
	p.putConn(ctx, cn, true)
}

// putConnWithoutTurn is an internal method that puts a connection back to the pool
// without freeing a turn. This is used when returning a rejected connection from
// within Get(), where the turn is still held by the Get() call.
func (p *ConnPool) putConnWithoutTurn(ctx context.Context, cn *Conn) {
	p.putConn(ctx, cn, false)
}

// putConn is the internal implementation of Put that optionally frees a turn.
func (p *ConnPool) putConn(ctx context.Context, cn *Conn, freeTurn bool) {
	// Guard against nil connection
	if cn == nil {
		internal.Logger.Printf(ctx, "putConn called with nil connection")
		if freeTurn {
			p.freeTurn()
		}
		return
	}

	// Process connection using the hooks system
	shouldPool := true
	shouldRemove := false
	var err error

	if reason := cn.CloseOnPutReason(); reason != "" {
		p.removeConnInternal(ctx, cn, errors.New(reason), freeTurn)
		return
	}

	if cn.HasBufferedData() {
		// Peek at the reply type to check if it's a push notification
		if replyType, err := cn.PeekReplyTypeSafe(); err != nil || replyType != proto.RespPush {
			// Not a push notification or error peeking, remove connection
			internal.Logger.Printf(ctx, "Conn has unread data (not push notification), removing it")
			p.removeConnInternal(ctx, cn, err, freeTurn)
			return
		}
		// It's a push notification, allow pooling (client will handle it)
	}

	// Lock-free atomic read - no mutex overhead!
	hookManager := p.hookManager.Load()

	if hookManager != nil {
		shouldPool, shouldRemove, err = hookManager.ProcessOnPut(ctx, cn)
		if err != nil {
			internal.Logger.Printf(ctx, "Connection hook error: %v", err)
			p.removeConnInternal(ctx, cn, err, freeTurn)
			return
		}
	}

	// Combine all removal checks into one - reduces branches
	if shouldRemove || !shouldPool {
		p.removeConnInternal(ctx, cn, errHookRequestedRemoval, freeTurn)
		return
	}

	if !cn.pooled {
		p.removeConnInternal(ctx, cn, errConnNotPooled, freeTurn)
		return
	}

	var shouldCloseConn bool
	var removedFromPool bool

	if p.cfg.MaxIdleConns == 0 || p.idleConnsLen.Load() < p.cfg.MaxIdleConns {
		// Hot path optimization: try fast IN_USE → IDLE transition
		// Using inline Release() method for better performance (avoids pointer dereference)
		transitionedToIdle := cn.Release()

		// Handle unexpected state changes
		if !transitionedToIdle {
			// Fast path failed - hook might have changed state (e.g., to UNUSABLE for handoff)
			// Keep the state set by the hook and pool the connection anyway
			sm := cn.GetStateMachine()
			if sm == nil {
				// State machine is nil - connection is in an invalid state, remove it
				internal.Logger.Printf(ctx, "conn[%d] has nil state machine, removing it", cn.GetID())
				p.removeConnInternal(ctx, cn, errConnNotPooled, freeTurn)
				return
			}
			currentState := sm.GetState()
			switch currentState {
			case StateUnusable:
				// expected state, don't log it
			case StateClosed:
				internal.Logger.Printf(ctx, "Unexpected conn[%d] state changed by hook to %v, closing it", cn.GetID(), currentState)
				if hookManager != nil {
					hookManager.ProcessOnRemove(ctx, cn, errHookRequestedRemoval)
				}
				shouldCloseConn = true
				removedFromPool = p.removeConnWithLock(cn)
			default:
				// Pool as-is
				internal.Logger.Printf(ctx, "Unexpected conn[%d] state changed by hook to %v, pooling as-is", cn.GetID(), currentState)
			}
		}

		// unusable conns are expected to become usable at some point (background process is reconnecting them)
		// put them at the opposite end of the queue
		// Optimization: if we just transitioned to IDLE, we know it's usable - skip the check
		if !transitionedToIdle && !cn.IsUsable() {
			p.connsMu.Lock()
			// Check if Close() already removed this connection from p.conns.
			// If so, skip the append and metrics — Close() already accounted for it.
			if _, inPool := p.conns[cn.GetID()]; inPool {
				if p.cfg.PoolFIFO {
					p.idleConns = append(p.idleConns, cn)
				} else {
					p.idleConns = append([]*Conn{cn}, p.idleConns...)
				}
				if cb := getMetricConnectionStateChangeCallback(); cb != nil {
					cb(ctx, cn, MetricStateUsed, MetricStateIdle)
				}
				if cb := getMetricConnectionCountCallback(); cb != nil {
					cb(ctx, -1, cn, "used", false)
					cb(ctx, 1, cn, "idle", false)
				}
				p.connsMu.Unlock()
				p.idleConnsLen.Add(1)
			} else {
				shouldCloseConn = true
				p.connsMu.Unlock()
			}
		} else if !shouldCloseConn {
			p.connsMu.Lock()
			if _, inPool := p.conns[cn.GetID()]; inPool {
				p.idleConns = append(p.idleConns, cn)
				if cb := getMetricConnectionStateChangeCallback(); cb != nil {
					cb(ctx, cn, MetricStateUsed, MetricStateIdle)
				}
				if cb := getMetricConnectionCountCallback(); cb != nil {
					cb(ctx, -1, cn, "used", false)
					cb(ctx, 1, cn, "idle", false)
				}
				p.connsMu.Unlock()
				p.idleConnsLen.Add(1)
			} else {
				shouldCloseConn = true
				p.connsMu.Unlock()
			}
		}

		if shouldCloseConn {
			// Connection was removed (e.g., hook set state to StateClosed).
			// Only emit if we actually removed it from the map (not already taken by Close()).
			if removedFromPool {
				if cb := getMetricConnectionStateChangeCallback(); cb != nil {
					cb(ctx, cn, MetricStateUsed, "")
				}
				if cb := getMetricConnectionCountCallback(); cb != nil {
					cb(ctx, -1, cn, "used", false)
				}
			}
		}
	} else {
		shouldCloseConn = true
		if hookManager != nil {
			hookManager.ProcessOnRemove(ctx, cn, errConnEvictedIdle)
		}
		removedFromPool = p.removeConnWithLock(cn)

		// Only emit if we actually removed it from the map (not already taken by Close()).
		if removedFromPool {
			// Notify metrics: connection removed (used -> nothing)
			if cb := getMetricConnectionStateChangeCallback(); cb != nil {
				cb(ctx, cn, MetricStateUsed, "")
			}
			// Record connection count decrement (connection removed while in used state)
			if cb := getMetricConnectionCountCallback(); cb != nil {
				cb(ctx, -1, cn, "used", false)
			}
		}
	}

	if freeTurn {
		p.freeTurn()
	}

	if shouldCloseConn {
		// Only emit connection closed if we actually owned the removal.
		// If removedFromPool is false, Close() already emitted connectionClosed for this conn.
		if removedFromPool {
			if cb := getMetricConnectionClosedCallback(); cb != nil {
				reason := "conn_pool_close"
				if r := cn.closeReason.Load(); r != "" {
					reason = r
				}
				cb(ctx, cn, reason, nil)
			}
		}
		_ = p.closeConn(cn)
	}

	cn.SetLastPutAtNs(getCachedTimeNs())
}

func (p *ConnPool) Remove(ctx context.Context, cn *Conn, reason error) {
	p.removeConnInternal(ctx, cn, reason, true)
}

// RemoveWithoutTurn removes a connection from the pool without freeing a turn.
// This should be used when removing a connection from a context that didn't acquire
// a turn via Get() (e.g., background workers, cleanup tasks).
// For normal removal after Get(), use Remove() instead.
func (p *ConnPool) RemoveWithoutTurn(ctx context.Context, cn *Conn, reason error) {
	p.removeConnInternal(ctx, cn, reason, false)
}

// removeConnInternal is the internal implementation of Remove that optionally frees a turn.
func (p *ConnPool) removeConnInternal(ctx context.Context, cn *Conn, reason error, freeTurn bool) {
	// Lock-free atomic read - no mutex overhead!
	hookManager := p.hookManager.Load()

	if hookManager != nil {
		hookManager.ProcessOnRemove(ctx, cn, reason)
	}

	removed := p.removeConnWithLock(cn)

	if freeTurn {
		p.freeTurn()
	}

	// Only emit metric decrements if we actually removed the connection from the map.
	// If removed is false, Close() already removed it and emitted the -1 delta.
	if removed {
		// Notify metrics: connection removed (assume from used state)
		if cb := getMetricConnectionStateChangeCallback(); cb != nil {
			cb(ctx, cn, MetricStateUsed, "")
		}
		// Record connection count decrement (connection removed, assume from used state)
		if cb := getMetricConnectionCountCallback(); cb != nil {
			cb(ctx, -1, cn, "used", false)
		}
	}

	// Only emit connection closed if we actually owned the removal.
	// If removed is false, Close() already emitted connectionClosed for this conn.
	if removed {
		if cb := getMetricConnectionClosedCallback(); cb != nil {
			reasonStr := "unknown"
			if reason != nil {
				reasonStr = reason.Error()
			}
			cb(ctx, cn, reasonStr, reason)
		}
	}

	_ = p.closeConn(cn)

	// Check if we need to create new idle connections to maintain MinIdleConns
	p.checkMinIdleConns()
}

// CloseConn closes a connection and records metrics.
// Parameters:
//   - ctx: context for metric callbacks (enables trace-to-metric correlation)
//   - cn: the connection to close
//   - reason: why the connection is being closed (use CloseReason* constants)
//   - fromState: the metric state the connection was in (use MetricState* constants)
func (p *ConnPool) CloseConn(ctx context.Context, cn *Conn, reason string, fromState string) error {
	if hookManager := p.hookManager.Load(); hookManager != nil {
		hookManager.ProcessOnRemove(ctx, cn, errors.New(reason))
	}

	removed := p.removeConnWithLock(cn)

	// Only emit UpDownCounter decrements if we actually removed the connection.
	// If removed is false, Close() already removed it and emitted the -1 delta.
	// Only emit connection closed if we actually owned the removal.
	// If removed is false, Close() already emitted connectionClosed for this conn.
	if removed {
		p.recordConnectionMetrics(ctx, cn, reason, fromState)
	}

	return p.closeConn(cn)
}

func (p *ConnPool) recordConnectionMetrics(ctx context.Context, cn *Conn, reason string, fromState string) {
	// Record connection state change: connection is being removed from the specified state
	if cb := getMetricConnectionStateChangeCallback(); cb != nil && fromState != "" {
		cb(ctx, cn, fromState, "")
	}

	// Record connection count decrement (UpDownCounter) for the state the connection was in
	if cb := getMetricConnectionCountCallback(); cb != nil && fromState != "" {
		cb(ctx, -1, cn, fromState, false)
	}

	if cb := getMetricConnectionClosedCallback(); cb != nil {
		cb(ctx, cn, reason, nil)
	}
}

// removeConnWithLock removes a connection from the pool under the connsMu lock.
// Returns true if the connection was actually present in p.conns and was removed,
// false if it was already gone (e.g., removed by Close()). Callers must use the
// return value to decide whether to emit metric decrements — this eliminates the
// shutdown race between Close() and concurrent removal paths.
func (p *ConnPool) removeConnWithLock(cn *Conn) bool {
	p.connsMu.Lock()
	defer p.connsMu.Unlock()
	return p.removeConn(cn)
}

// removeConn removes a connection from the pool's internal data structures.
// Returns true if the connection was present and removed, false otherwise.
func (p *ConnPool) removeConn(cn *Conn) bool {
	cid := cn.GetID()
	if _, exists := p.conns[cid]; !exists {
		return false
	}
	delete(p.conns, cid)
	atomic.AddUint32(&p.stats.StaleConns, 1)

	// Decrement pool size counter when removing a connection
	if cn.pooled {
		p.poolSize.Add(-1)
		// this can be idle conn
		for idx, ic := range p.idleConns {
			if ic == cn {
				p.idleConns = append(p.idleConns[:idx], p.idleConns[idx+1:]...)
				p.idleConnsLen.Add(-1)
				break
			}
		}
	}
	return true
}

func (p *ConnPool) closeConn(cn *Conn) error {
	return cn.Close()
}

// Len returns total number of connections.
func (p *ConnPool) Len() int {
	p.connsMu.Lock()
	n := len(p.conns)
	p.connsMu.Unlock()
	return n
}

// IdleLen returns number of idle connections.
func (p *ConnPool) IdleLen() int {
	p.connsMu.Lock()
	n := p.idleConnsLen.Load()
	p.connsMu.Unlock()
	return int(n)
}

// Name returns the pool's configured name, which is stamped on every
// connection it creates (Conn.PoolName). Callers holding a Pooler can type
// assert to interface{ Name() string } to find which pool owns a connection —
// used by maintnotifications to route a handoff to the hook that owns the
// conn's pool rather than always the primary one.
func (p *ConnPool) Name() string { return p.cfg.Name }

// Size returns the maximum pool size (capacity).
//
// This is used by the streaming credentials manager to size the re-auth worker pool,
// ensuring that re-auth operations don't exhaust the connection pool.
func (p *ConnPool) Size() int {
	return int(p.cfg.PoolSize)
}

func (p *ConnPool) Stats() *Stats {
	return &Stats{
		Hits:            atomic.LoadUint32(&p.stats.Hits),
		Misses:          atomic.LoadUint32(&p.stats.Misses),
		Timeouts:        atomic.LoadUint32(&p.stats.Timeouts),
		WaitCount:       atomic.LoadUint32(&p.stats.WaitCount),
		Unusable:        atomic.LoadUint32(&p.stats.Unusable),
		WaitDurationNs:  p.waitDurationNs.Load(),
		PendingRequests: atomic.LoadUint32(&p.stats.PendingRequests),

		TotalConns: uint32(p.Len()),
		IdleConns:  uint32(p.IdleLen()),
		StaleConns: atomic.LoadUint32(&p.stats.StaleConns),
	}
}

func (p *ConnPool) closed() bool {
	return p._closed.Load() == 1
}

func (p *ConnPool) RetireConns(ctx context.Context, conns []*Conn, reason string) {
	if len(conns) == 0 {
		return
	}

	idleConnSet := make(map[*Conn]struct{})
	toClose := make([]*Conn, 0, len(conns))

	p.connsMu.Lock()
	for _, ic := range p.idleConns {
		idleConnSet[ic] = struct{}{}
	}
	for _, cn := range conns {
		if cn == nil {
			continue
		}
		if _, ok := p.conns[cn.GetID()]; !ok {
			continue
		}
		if _, isIdle := idleConnSet[cn]; isIdle {
			if p.removeConn(cn) {
				toClose = append(toClose, cn)
			}
			continue
		}
		cn.MarkCloseOnPut(reason)
	}
	p.connsMu.Unlock()

	if hookManager := p.hookManager.Load(); hookManager != nil {
		for _, cn := range toClose {
			hookManager.ProcessOnRemove(ctx, cn, errors.New(reason))
		}
	}
	for _, cn := range toClose {
		p.recordConnectionMetrics(ctx, cn, reason, MetricStateIdle)
		_ = p.closeConn(cn)
	}
	p.checkMinIdleConns()
}

func (p *ConnPool) Filter(fn func(*Conn) bool) error {
	ctx := context.Background()

	p.connsMu.Lock()
	defer p.connsMu.Unlock()

	idleConnSet := make(map[*Conn]struct{}, len(p.idleConns))
	for _, ic := range p.idleConns {
		idleConnSet[ic] = struct{}{}
	}

	var firstErr error
	for _, cn := range p.conns {
		if fn(cn) {
			var err error
			if _, isIdle := idleConnSet[cn]; isIdle {
				// Idle connection - remove from pool and close.
				p.removeConn(cn)
				p.recordConnectionMetrics(ctx, cn, CloseReasonFailover, MetricStateIdle)
				err = p.closeConn(cn)
			} else {
				// Used connection - set closeReason and close the connection.
				// The connection remains in p.conns. When putConn() is called later,
				// it will close the connection instead of pooling it.
				cn.closeReason.Store(CloseReasonFailover)
				err = cn.Close()
			}
			if err != nil && firstErr == nil {
				firstErr = err
			}
		}
	}
	return firstErr
}

type drainConn struct {
	conn      *Conn
	idleIndex int
}

// DrainState carries the cross-pass round bookkeeping for the CSC drainer.
// It is owned by one drainer goroutine, so no synchronization is needed.
type DrainState struct {
	// round contains only initialized idle connections. Entries are processed
	// from the end so their snapshot indexes remain stable as connections are
	// removed and returned. Connections that go idle mid-round are deferred.
	round []drainConn
	next  int
}

// DrainIdleConns runs one pass of the CSC invalidation drainer over the current
// round, holding AT MOST ONE connection and its pool turn at a time (ctx is the
// per-cycle deadline). A round = idle conn ids snapshotted at start; mid-round
// arrivals are deferred. Each member is drainerPop'd and drained by fn, or — if no
// longer a claimable idle conn — reconciled (marked visited) so it can't hang the
// round. The drainer yields when no turn is immediately available, giving command
// traffic priority. Handles at least one member before honoring ctx (so a tiny
// DrainInterval can't stall it). No-ops if the pool is closed.
func (p *ConnPool) DrainIdleConns(ctx context.Context, st *DrainState, fn func(cn *Conn) error) {
	if st == nil || fn == nil || p.closed() {
		return
	}

	if st.round == nil {
		st.round = p.idleConnsSnapshot()
		st.next = len(st.round)
		if len(st.round) == 0 {
			st.round = nil
			return
		}
	}

	handled := 0
	for st.next > 0 {
		// Min-progress: handle at least one member — drained OR reconciled — before
		// honoring the per-cycle deadline, so a pass does a bounded amount of work
		// while ignoring an expired ctx. A deadline-truncated round resumes on the
		// next pass.
		if handled > 0 && ctx.Err() != nil {
			return
		}

		// Account for the borrowed connection exactly like Get. Without a turn,
		// a concurrent Get can observe the temporarily-empty idle pool and either
		// exceed PoolSize or fail at MaxActiveConns. Maintenance never waits for a
		// turn, so command traffic wins under contention.
		if !p.semaphore.TryAcquire() {
			return
		}
		st.next--
		cn := p.drainerPop(ctx, st.round[st.next])
		if cn == nil {
			p.freeTurn()
			// Reconcile: not a claimable idle member right now (closed, in use,
			// unusable, or moved in idleConns by concurrent traffic). Covered by
			// the command-path drain and/or the next round.
			handled++
			continue
		}

		func() {
			defer p.endDrainerBorrow()
			if err := fn(cn); err != nil {
				// Fatal drain error (read/protocol/connection).
				p.removeConnInternal(ctx, cn, err, true)
			} else {
				// Normal return: runs OnPut (queues any maintenance handoff).
				p.putConn(ctx, cn, true)
			}
		}()
		handled++
	}

	// Every member handled — round complete; snapshot a fresh round next pass.
	st.round = nil
	st.next = 0
}

// idleConnsSnapshot returns initialized idle connections and their current
// indexes. StateCreated MinIdleConns are intentionally excluded.
func (p *ConnPool) idleConnsSnapshot() []drainConn {
	p.connsMu.Lock()
	defer p.connsMu.Unlock()
	if len(p.idleConns) == 0 {
		return nil
	}
	round := make([]drainConn, 0, len(p.idleConns))
	for idx, cn := range p.idleConns {
		if cn.stateMachine.GetState() == StateIdle {
			round = append(round, drainConn{conn: cn, idleIndex: idx})
		}
	}
	return round
}

// drainerPop claims a snapshotted connection (strict IDLE->IN_USE) and removes
// it from idleConns in O(1). Entries are processed in reverse index order, so
// swap removal cannot move an unprocessed round member. If concurrent pool
// traffic changed the slot, the member is deferred to the next round.
func (p *ConnPool) drainerPop(ctx context.Context, member drainConn) *Conn {
	p.connsMu.Lock()
	defer p.connsMu.Unlock()
	if p.closed() {
		return nil
	}
	idx := member.idleIndex
	if idx < 0 || idx >= len(p.idleConns) || p.idleConns[idx] != member.conn {
		return nil
	}
	cn := member.conn
	if !cn.stateMachine.TryTransitionFast(StateIdle, StateInUse) {
		return nil
	}
	p.beginDrainerBorrow()
	last := len(p.idleConns) - 1
	p.idleConns[idx] = p.idleConns[last]
	p.idleConns[last] = nil
	p.idleConns = p.idleConns[:last]
	p.idleConnsLen.Add(-1)
	if cb := getMetricConnectionStateChangeCallback(); cb != nil {
		cb(ctx, cn, MetricStateIdle, MetricStateUsed)
	}
	if cb := getMetricConnectionCountCallback(); cb != nil {
		cb(ctx, -1, cn, "idle", false)
		cb(ctx, 1, cn, "used", false)
	}
	return cn
}

func (p *ConnPool) Close() error {
	if !p._closed.CompareAndSwap(0, 1) {
		return ErrClosed
	}

	var firstErr error
	nowNs := time.Now().UnixNano()
	p.connsMu.Lock()

	// Emit -1 for each connection. Since all idle↔used transitions happen
	// under connsMu, the idleConns slice is the source of truth for state.
	cb := getMetricConnectionCountCallback()
	idleSet := make(map[uint64]struct{}, len(p.idleConns))
	for _, cn := range p.idleConns {
		idleSet[cn.GetID()] = struct{}{}
	}
	ctx := context.Background()
	for _, cn := range p.conns {
		// Check health before closing, since closeConn invalidates the
		// underlying fd and would make connCheck (inside isHealthyConn)
		// always fail with EBADF.
		// Only check health for idle connections to avoid data races when
		// peeking at the socket/reader while another goroutine is reading from it.
		// Non-idle connections are either in use or in transitional states and
		// shouldn't be health-checked during shutdown.
		_, isIdle := idleSet[cn.GetID()]
		var healthy bool
		if isIdle {
			healthy = p.isHealthyConn(cn, nowNs)
		} else {
			healthy = true
		}
		if cb != nil {
			if isIdle {
				cb(ctx, -1, cn, "idle", false)
			} else {
				cb(ctx, -1, cn, "used", false)
			}
		}
		if closedCb := getMetricConnectionClosedCallback(); closedCb != nil {
			closedCb(ctx, cn, "pool_shutdown", nil)
		}
		if err := p.closeConn(cn); err != nil && firstErr == nil {
			// Suppress close errors for stale connections, consistent
			// with how Get() handles them (see CloseReasonStale path).
			if healthy {
				firstErr = err
			}
		}
	}
	p.conns = nil
	p.poolSize.Store(0)
	p.idleConns = nil
	p.idleConnsLen.Store(0)
	p.connsMu.Unlock()

	return firstErr
}

func (p *ConnPool) isHealthyConn(cn *Conn, nowNs int64) bool {
	// Performance optimization: check conditions from cheapest to most expensive,
	// and from most likely to fail to least likely to fail.

	// Only fails if ConnMaxLifetime is set AND connection is old.
	// Most pools don't set ConnMaxLifetime, so this rarely fails.
	if p.cfg.ConnMaxLifetime > 0 {
		if cn.expiresAt.UnixNano() < nowNs {
			return false // Connection has exceeded max lifetime
		}
	}

	// Most pools set ConnMaxIdleTime, and idle connections are common.
	// Checking this first allows us to fail fast without expensive syscalls.
	if p.cfg.ConnMaxIdleTime > 0 {
		if nowNs-cn.UsedAtNs() >= int64(p.cfg.ConnMaxIdleTime) {
			return false // Connection has been idle too long
		}
	}

	// Only run this if the cheap checks passed.
	if err := connCheck(cn.getNetConn()); err != nil {
		// If there's unexpected data, it might be push notifications (RESP3)
		if p.cfg.PushNotificationsEnabled && err == errUnexpectedRead {
			// Peek at the reply type to check if it's a push notification.
			// Use the readerMu-guarded peek: a concurrent handoff may be
			// resetting cn.rd via SetNetConn on a connection popped by Get
			// before the OnGet state check rejects it.
			if replyType, err := cn.PeekReplyTypeForCheck(); err == nil && replyType == proto.RespPush {
				// For RESP3 connections with push notifications, we allow some buffered data
				// The client will process these notifications before using the connection
				internal.Logger.Printf(
					context.Background(),
					"push: conn[%d] has buffered data, likely push notifications - will be processed by client",
					cn.GetID(),
				)

				// Update timestamp for healthy connection
				cn.SetUsedAtNs(nowNs)

				// Connection is healthy, client will handle notifications
				return true
			}
			// Not a push notification - treat as unhealthy
			return false
		}
		// Connection failed health check
		return false
	}

	// Only update UsedAt if connection is healthy (avoids unnecessary atomic store)
	cn.SetUsedAtNs(nowNs)
	return true
}
