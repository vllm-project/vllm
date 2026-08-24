package redis

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"net"
	"sync"
	"sync/atomic"
	"time"

	"github.com/redis/go-redis/v9/auth"
	"github.com/redis/go-redis/v9/internal"
	"github.com/redis/go-redis/v9/internal/auth/streaming"
	"github.com/redis/go-redis/v9/internal/hscan"
	"github.com/redis/go-redis/v9/internal/otel"
	"github.com/redis/go-redis/v9/internal/pool"
	"github.com/redis/go-redis/v9/internal/proto"
	"github.com/redis/go-redis/v9/maintnotifications"
	"github.com/redis/go-redis/v9/push"
)

// Scanner internal/hscan.Scanner exposed interface.
type Scanner = hscan.Scanner

// Nil reply returned by Redis when key does not exist.
const Nil = proto.Nil

// String representations of special float values.
// Values are lowercase for consistency with Redis RESP2 protocol responses.
const (
	NaN  = internal.NaN  // Not a Number
	Inf  = internal.Inf  // Positive infinity
	NInf = internal.NInf // Negative infinity
)

// SetLogger set custom log
// Use with VoidLogger to disable logging.
// If logger is nil, the call is ignored and the existing logger is kept.
func SetLogger(logger internal.Logging) {
	if logger == nil {
		return
	}
	internal.Logger = logger
}

// SetLogLevel sets the log level for the library.
func SetLogLevel(logLevel internal.LogLevelT) {
	internal.LogLevel = logLevel
}

//------------------------------------------------------------------------------

type Hook interface {
	DialHook(next DialHook) DialHook
	ProcessHook(next ProcessHook) ProcessHook
	ProcessPipelineHook(next ProcessPipelineHook) ProcessPipelineHook
}

type (
	DialHook            func(ctx context.Context, network, addr string) (net.Conn, error)
	ProcessHook         func(ctx context.Context, cmd Cmder) error
	ProcessPipelineHook func(ctx context.Context, cmds []Cmder) error
)

type hooksMixin struct {
	// hooksMu serializes writers (AddHook); readers never take it.
	hooksMu *sync.Mutex
	// state holds the immutable hook snapshot. Readers Load it lock-free;
	// writers publish a replacement copy-on-write under hooksMu.
	state *atomic.Pointer[hooksState]
}

// hooksState is an immutable snapshot of the hook configuration. Once stored
// in hooksMixin.state it is never mutated; AddHook builds a fresh copy.
type hooksState struct {
	slice   []Hook
	initial hooks
	current hooks
}

// rebuild recomputes current from initial + slice. It mutates the receiver, so
// it must only run on a state that has not yet been published.
func (s *hooksState) rebuild() {
	s.initial.setDefaults()

	s.current.dial = s.initial.dial
	s.current.process = s.initial.process
	s.current.pipeline = s.initial.pipeline
	s.current.txPipeline = s.initial.txPipeline

	for i := len(s.slice) - 1; i >= 0; i-- {
		if wrapped := s.slice[i].DialHook(s.current.dial); wrapped != nil {
			s.current.dial = wrapped
		}
		if wrapped := s.slice[i].ProcessHook(s.current.process); wrapped != nil {
			s.current.process = wrapped
		}
		if wrapped := s.slice[i].ProcessPipelineHook(s.current.pipeline); wrapped != nil {
			s.current.pipeline = wrapped
		}
		if wrapped := s.slice[i].ProcessPipelineHook(s.current.txPipeline); wrapped != nil {
			s.current.txPipeline = wrapped
		}
	}
}

func (hs *hooksMixin) initHooks(hooks hooks) {
	var slice []Hook
	if hs.state != nil {
		if old := hs.state.Load(); old != nil {
			slice = old.slice
		}
	}

	hs.hooksMu = new(sync.Mutex)
	hs.state = new(atomic.Pointer[hooksState])

	state := &hooksState{slice: slice, initial: hooks}
	state.rebuild()
	hs.state.Store(state)
}

type hooks struct {
	dial       DialHook
	process    ProcessHook
	pipeline   ProcessPipelineHook
	txPipeline ProcessPipelineHook
}

func (h *hooks) setDefaults() {
	if h.dial == nil {
		h.dial = func(ctx context.Context, network, addr string) (net.Conn, error) { return nil, nil }
	}
	if h.process == nil {
		h.process = func(ctx context.Context, cmd Cmder) error { return nil }
	}
	if h.pipeline == nil {
		h.pipeline = func(ctx context.Context, cmds []Cmder) error { return nil }
	}
	if h.txPipeline == nil {
		h.txPipeline = func(ctx context.Context, cmds []Cmder) error { return nil }
	}
}

// AddHook is to add a hook to the queue.
// Hook is a function executed during network connection, command execution, and pipeline,
// it is a first-in-first-out stack queue (FIFO).
// You need to execute the next hook in each hook, unless you want to terminate the execution of the command.
// For example, you added hook-1, hook-2:
//
//	client.AddHook(hook-1, hook-2)
//
// hook-1:
//
//	func (Hook1) ProcessHook(next redis.ProcessHook) redis.ProcessHook {
//	 	return func(ctx context.Context, cmd Cmder) error {
//		 	print("hook-1 start")
//		 	next(ctx, cmd)
//		 	print("hook-1 end")
//		 	return nil
//	 	}
//	}
//
// hook-2:
//
//	func (Hook2) ProcessHook(next redis.ProcessHook) redis.ProcessHook {
//		return func(ctx context.Context, cmd redis.Cmder) error {
//			print("hook-2 start")
//			next(ctx, cmd)
//			print("hook-2 end")
//			return nil
//		}
//	}
//
// The execution sequence is:
//
//	hook-1 start -> hook-2 start -> exec redis cmd -> hook-2 end -> hook-1 end
//
// Please note: "next(ctx, cmd)" is very important, it will call the next hook,
// if "next(ctx, cmd)" is not executed, the redis command will not be executed.
func (hs *hooksMixin) AddHook(hook Hook) {
	hs.hooksMu.Lock()
	defer hs.hooksMu.Unlock()

	old := hs.state.Load()
	state := &hooksState{
		slice:   make([]Hook, len(old.slice)+1),
		initial: old.initial,
	}
	copy(state.slice, old.slice)
	state.slice[len(old.slice)] = hook
	state.rebuild()

	hs.state.Store(state)
}

func (hs *hooksMixin) clone() hooksMixin {
	old := hs.state.Load()
	l := len(old.slice)
	state := &hooksState{
		slice:   old.slice[:l:l],
		initial: old.initial,
		current: old.current,
	}

	clone := hooksMixin{
		hooksMu: new(sync.Mutex),
		state:   new(atomic.Pointer[hooksState]),
	}
	clone.state.Store(state)
	return clone
}

func (hs *hooksMixin) withProcessHook(ctx context.Context, cmd Cmder, hook ProcessHook) error {
	slice := hs.state.Load().slice
	for i := len(slice) - 1; i >= 0; i-- {
		if wrapped := slice[i].ProcessHook(hook); wrapped != nil {
			hook = wrapped
		}
	}
	return hook(ctx, cmd)
}

func (hs *hooksMixin) withProcessPipelineHook(
	ctx context.Context, cmds []Cmder, hook ProcessPipelineHook,
) error {
	slice := hs.state.Load().slice
	for i := len(slice) - 1; i >= 0; i-- {
		if wrapped := slice[i].ProcessPipelineHook(hook); wrapped != nil {
			hook = wrapped
		}
	}
	return hook(ctx, cmds)
}

func (hs *hooksMixin) dialHook(ctx context.Context, network, addr string) (net.Conn, error) {
	return hs.state.Load().current.dial(ctx, network, addr)
}

// hookCount reports how many user hooks are installed. The autopipeliner
// arms its await() self-deadlock guard only when hooks exist, keeping the
// guard a single atomic load on hook-free clients.
func (hs *hooksMixin) hookCount() int {
	return len(hs.state.Load().slice)
}

func (hs *hooksMixin) processHook(ctx context.Context, cmd Cmder) error {
	return hs.state.Load().current.process(ctx, cmd)
}

func (hs *hooksMixin) processPipelineHook(ctx context.Context, cmds []Cmder) error {
	return hs.state.Load().current.pipeline(ctx, cmds)
}

func (hs *hooksMixin) processTxPipelineHook(ctx context.Context, cmds []Cmder) error {
	return hs.state.Load().current.txPipeline(ctx, cmds)
}

//------------------------------------------------------------------------------

// Stable identifiers for baseClient.onClose hooks. Each component that
// registers a close callback owns a dedicated id here so the set of known
// hooks is discoverable in one place and id collisions are caught at
// compile time. New ids should be added as additional constants.
const (
	// onCloseHookIDSentinelFailover identifies the close callback installed
	// by NewFailoverClient to tear down sentinel failover background work.
	onCloseHookIDSentinelFailover = "sentinel-failover"
)

// onCloseHooks is a small registry of named close callbacks attached to a
// baseClient. Each callback is identified by a stable string id; registering
// the same id twice replaces the previous callback rather than chaining onto
// it. This guarantees the registry stays bounded regardless of how often a
// hook is (re)registered and avoids the unbounded closure chain that
// motivated issue #3772.
//
// Hooks are invoked in registration order. All hooks run regardless of
// individual errors; the first non-nil error is returned.
//
// A zero-value onCloseHooks is ready to use. It is safe for concurrent use.
// Clones of a baseClient share the same *onCloseHooks so registrations and
// close semantics are preserved across WithTimeout / WithContext / etc.
type onCloseHooks struct {
	mu    sync.Mutex
	order []string
	hooks map[string]func() error
}

// register adds or replaces the callback associated with id. Re-registering
// an existing id overwrites the previous callback in place; new ids are
// appended to the invocation order.
func (h *onCloseHooks) register(id string, fn func() error) {
	h.mu.Lock()
	defer h.mu.Unlock()
	if h.hooks == nil {
		h.hooks = make(map[string]func() error)
	}
	if _, exists := h.hooks[id]; !exists {
		h.order = append(h.order, id)
	}
	h.hooks[id] = fn
}

// unregister removes the callback associated with id, if any. It is kept
// for API symmetry with register so future callers (e.g. dynamic hook
// owners that need to detach before client Close) do not have to
// reinvent it.
//
//nolint:unused // kept for API symmetry with register; see comment above.
func (h *onCloseHooks) unregister(id string) {
	h.mu.Lock()
	defer h.mu.Unlock()
	if _, exists := h.hooks[id]; !exists {
		return
	}
	delete(h.hooks, id)
	for i, x := range h.order {
		if x == id {
			h.order = append(h.order[:i], h.order[i+1:]...)
			break
		}
	}
}

// run invokes all registered callbacks in registration order and returns
// the first non-nil error encountered. All callbacks are executed even if
// an earlier one returns an error.
func (h *onCloseHooks) run() error {
	if h == nil {
		return nil
	}
	h.mu.Lock()
	fns := make([]func() error, 0, len(h.order))
	for _, id := range h.order {
		if fn := h.hooks[id]; fn != nil {
			fns = append(fns, fn)
		}
	}
	h.mu.Unlock()

	var firstErr error
	for _, fn := range fns {
		if err := fn(); err != nil && firstErr == nil {
			firstErr = err
		}
	}
	return firstErr
}

type baseClient struct {
	// apClosed flips when the shared pools begin closing; every wrapper and
	// every clone SHARING those pools refuses to build a new autopipeliner
	// from then on. A pointer: withTimeout/clone copy it, so the flag is one
	// per pool-set, not one per wrapper. See baseClient.Close.
	apClosed *atomic.Bool

	opt        *Options
	optLock    sync.RWMutex
	connPool   pool.Pooler
	pubSubPool *pool.PubSubPool
	// pipelinePool is an optional separate connection pool for pipelining
	// operations, used when PipelineReadBufferSize/PipelineWriteBufferSize is
	// set so pipelines can use large buffers without bloating the main pool.
	// nil means pipelines use connPool.
	pipelinePool pool.Pooler
	// pipelinePoolName is the pool name assigned to pipelinePool's connections
	// (pool.Conn.PoolName()). It lets poolForConn route a connection back to the
	// pool that owns it — e.g. so streaming-credentials re-auth closes/accounts a
	// failed pipeline connection against pipelinePool, not connPool.
	pipelinePoolName string
	hooksMixin

	// onClose holds named callbacks invoked when the client is closed.
	// Registering a new callback never removes previously registered ones;
	// only re-registering the same id replaces the existing callback. This
	// lets composing components (e.g. sentinel failover) add close logic
	// safely without fear of overwriting each other and without building
	// unbounded closure chains on repeated registration.
	onClose *onCloseHooks

	// Push notification processing
	pushProcessor push.NotificationProcessor

	// Maintenance notifications manager
	maintNotificationsManager     *maintnotifications.Manager
	maintNotificationsManagerLock sync.RWMutex

	// streamingCredentialsManager is used to manage streaming credentials
	streamingCredentialsManager *streaming.Manager

	// himport is the client-side registry of HIMPORT fieldsets, used to
	// lazily replay HIMPORT PREPARE onto pooled connections (see himport.go).
	// Shared by clones and by Conn instances derived from the same pool.
	himport *himportRegistry

	// csc is the shared client-side cache; nil when CSC is disabled.
	csc Cache

	// cscKeyPrefix namespaces a shared cache by DB and fixed authentication
	// identity. It is computed once during attachment and copied with the cache.
	cscKeyPrefix string

	// allowClientTracking exempts a client from the CLIENT TRACKING guard (see
	// process and generalProcessPipeline). Set only on initConn's internal conn
	// wrapper, whose init pipeline legitimately issues CLIENT TRACKING ON;
	// never set on user-visible clients.
	allowClientTracking bool

	// The following are OWNER-ONLY and NOT copied by clone(): derived clients
	// (Conn/WithTimeout) share the cache but must not stop the owner's
	// goroutines or flush its cache on their own Close.

	// cscOwnsCache is true only when this client constructed its LocalCache (not
	// an injected/shared one); it gates the defensive flush on drainer stop.
	cscOwnsCache bool

	// cscDrainHandle is the background drainer handle (nil when none). Held
	// on the client, not a global registry, so an un-Closed client stays
	// GC-collectible and a runtime.AddCleanup net can stop the goroutine. Its
	// presence also identifies the owner (the only client that runs the drainer
	// and thus the one that deregisters cscPoolHook).
	cscDrainHandle *cscDrainHandle

	// cscPoolHook is the evict-on-remove pool hook (nil when CSC is off). Unlike
	// the owner-only fields above it IS copied by clone(): a clone reads it in
	// processCached to attribute fetches to the shared hook. Only the owner (the
	// one with cscDrainHandle) deregisters it when the drainer exits.
	cscPoolHook pool.PoolHook

	// cscActive is allocated only after CSC attaches successfully and becomes
	// false once the drainer stops (owner Close, GC cleanup, or damping). It is
	// shared with derived clients so they initialize borrowed pool connections
	// with tracking only while the parent's CSC is actually operational.
	cscActive *atomic.Bool
}

func (c *baseClient) clone() *baseClient {
	c.maintNotificationsManagerLock.RLock()
	maintNotificationsManager := c.maintNotificationsManager
	c.maintNotificationsManagerLock.RUnlock()

	clone := &baseClient{
		apClosed:                    c.apClosed,
		opt:                         c.opt,
		connPool:                    c.connPool,
		pipelinePool:                c.pipelinePool,
		pipelinePoolName:            c.pipelinePoolName,
		pubSubPool:                  c.pubSubPool,
		onClose:                     c.onClose,
		pushProcessor:               c.pushProcessor,
		maintNotificationsManager:   maintNotificationsManager,
		streamingCredentialsManager: c.streamingCredentialsManager,
		himport:                     c.himport,
		csc:                         c.csc,
		// cscPoolHook and cscActive travel with the cache (read in processCached);
		// the owner-only fields — cscDrainHandle, cscOwnsCache — do not, so a clone's
		// Close never tears down the owner's resources.
		cscPoolHook:  c.cscPoolHook,
		cscActive:    c.cscActive,
		cscKeyPrefix: c.cscKeyPrefix,
	}
	return clone
}

// cloneOpt clones c.opt while holding optLock to prevent races with initConn
// which writes to MaintNotificationsConfig.Mode under the same lock.
func (c *baseClient) cloneOpt() *Options {
	c.optLock.RLock()
	clone := c.opt.clone()
	c.optLock.RUnlock()
	return clone
}

func (c *baseClient) withTimeout(timeout time.Duration) *baseClient {
	opt := c.cloneOpt()
	opt.ReadTimeout = timeout
	opt.WriteTimeout = timeout

	clone := c.clone()
	clone.opt = opt

	return clone
}

func (c *baseClient) String() string {
	return fmt.Sprintf("Redis<%s db:%d>", c.getAddr(), c.opt.DB)
}

func (c *baseClient) getConn(ctx context.Context) (*pool.Conn, error) {
	if c.opt.Limiter != nil {
		err := c.opt.Limiter.Allow()
		if err != nil {
			return nil, err
		}
	}

	cn, err := c._getConn(ctx)
	if err != nil {
		if c.opt.Limiter != nil {
			c.opt.Limiter.ReportResult(err)
		}
		return nil, err
	}

	return cn, nil
}

func (c *baseClient) _getConn(ctx context.Context) (*pool.Conn, error) {
	cn, err := c.connPool.Get(ctx)
	if err != nil {
		return nil, err
	}

	if err := c.initPooledConn(ctx, c.connPool, cn); err != nil {
		return nil, err
	}

	return cn, nil
}

// initPooledConn brings a conn freshly obtained from p to a usable state: it
// runs the connection handshake if needed, records the connection-create-time
// metric, and re-acquires the conn after initConn parks it IDLE. On failure
// the conn is Removed from p (never leaked) and the error is unwrapped to the
// caller-visible cause. Shared by the main-pool path (_getConn) and the
// dedicated pipeline-pool path (withPipelineConn) so the two cannot drift.
func (c *baseClient) initPooledConn(ctx context.Context, p pool.Pooler, cn *pool.Conn) error {
	if cn.IsInited() {
		return nil
	}

	if err := c.initConn(ctx, cn); err != nil {
		p.Remove(ctx, cn, err)
		if unwrapped := errors.Unwrap(err); unwrapped != nil {
			return unwrapped
		}
		return err
	}

	if dialStartNs := cn.GetDialStartNs(); dialStartNs > 0 {
		if cb := pool.GetMetricConnectionCreateTimeCallback(); cb != nil {
			duration := time.Duration(time.Now().UnixNano() - dialStartNs)
			cb(ctx, duration, cn)
		}
	}

	// initConn will transition to IDLE state, so we need to acquire it
	// before returning it to the user.
	if !cn.TryAcquire() {
		err := fmt.Errorf("redis: connection is not usable")
		// Remove rather than abandon: an unacquirable conn left outside the
		// pool's accounting would leak its slot.
		p.Remove(ctx, cn, err)
		return err
	}

	return nil
}

// poolForConn returns the pool that owns cn — the dedicated pipeline pool when
// cn was dialed there, otherwise the main pool. Re-auth close/accounting must
// target the owning pool so a failed pipeline connection is removed from the
// pipeline pool's books, not the main pool's.
func (c *baseClient) poolForConn(cn *pool.Conn) pool.Pooler {
	if c.pipelinePool != nil && c.pipelinePoolName != "" && cn.PoolName() == c.pipelinePoolName {
		return c.pipelinePool
	}
	return c.connPool
}

func (c *baseClient) reAuthConnection() func(poolCn *pool.Conn, credentials auth.Credentials) error {
	return func(poolCn *pool.Conn, credentials auth.Credentials) error {
		var err error
		username, password := credentials.BasicAuth()

		// Use background context - timeout is handled by ReadTimeout in WithReader/WithWriter
		ctx := context.Background()

		connPool := pool.NewSingleConnPool(c.poolForConn(poolCn), poolCn)

		// Pass hooks so that reauth commands are recorded/traced; share the
		// HIMPORT registry for the same reason as in initConn.
		cn := newConn(c.opt, connPool, &c.hooksMixin, c.himport)

		if username != "" {
			err = cn.AuthACL(ctx, username, password).Err()
		} else {
			err = cn.Auth(ctx, password).Err()
		}

		return err
	}
}

func (c *baseClient) onAuthenticationErr() func(poolCn *pool.Conn, err error) {
	return func(poolCn *pool.Conn, err error) {
		if err != nil {
			if isBadConn(err, false, c.opt.Addr) {
				// Close the connection to force a reconnection.
				// Re-auth happens on connections that were idle in the pool (the pool hook
				// waits for IDLE state before transitioning to UNUSABLE for re-auth).
				// From metrics perspective, the connection was never "used" by a client.
				// Note: Using context.Background() as this callback doesn't have access to caller's context.
				err := c.poolForConn(poolCn).CloseConn(context.Background(), poolCn, pool.CloseReasonAuthError, pool.MetricStateIdle)
				if err != nil {
					internal.Logger.Printf(context.Background(), "redis: failed to close connection: %v", err)
					// try to close the network connection directly
					// so that no resource is leaked
					err := poolCn.Close()
					if err != nil {
						internal.Logger.Printf(context.Background(), "redis: failed to close network connection: %v", err)
					}
				}
			}
			internal.Logger.Printf(context.Background(), "redis: re-authentication failed: %v", err)
		}
	}
}

// resolveCredentials returns the username/password to authenticate with, using
// the non-streaming credential sources in precedence order:
// CredentialsProviderContext, then CredentialsProvider, then the static
// Username/Password fields. The StreamingCredentialsProvider path is handled
// separately by initConn (it requires per-connection listener wiring) and is
// intentionally not covered here. Returns empty strings when no credentials
// are configured.
func (opt *Options) resolveCredentials(ctx context.Context) (username, password string, err error) {
	switch {
	case opt.CredentialsProviderContext != nil:
		return opt.CredentialsProviderContext(ctx)
	case opt.CredentialsProvider != nil:
		username, password = opt.CredentialsProvider()
	case opt.Username != "" || opt.Password != "":
		username, password = opt.Username, opt.Password
	}
	return username, password, nil
}

func (c *baseClient) initConn(ctx context.Context, cn *pool.Conn) error {
	// This function is called in two scenarios:
	// 1. First-time init: Connection is in CREATED state (from pool.Get())
	//    - We need to transition CREATED → INITIALIZING and do the initialization
	//    - If another goroutine is already initializing, we WAIT for it to finish
	// 2. Re-initialization: Connection is in INITIALIZING state (from SetNetConnAndInitConn())
	//    - We're already in INITIALIZING, so just proceed with initialization

	currentState := cn.GetStateMachine().GetState()

	// Fast path: Check if already initialized (IDLE or IN_USE)
	if currentState == pool.StateIdle || currentState == pool.StateInUse {
		return nil
	}

	// If in CREATED state, try to transition to INITIALIZING
	if currentState == pool.StateCreated {
		finalState, err := cn.GetStateMachine().TryTransition([]pool.ConnState{pool.StateCreated}, pool.StateInitializing)
		if err != nil {
			// Another goroutine is initializing or connection is in unexpected state
			// Check what state we're in now
			if finalState == pool.StateIdle || finalState == pool.StateInUse {
				// Already initialized by another goroutine
				return nil
			}

			if finalState == pool.StateInitializing {
				// Another goroutine is initializing - WAIT for it to complete
				// Use a context with timeout = min(remaining command timeout, DialTimeout)
				// This prevents waiting too long while respecting the caller's deadline
				var waitCtx context.Context
				var cancel context.CancelFunc
				dialTimeout := c.opt.DialTimeout

				if cmdDeadline, hasCmdDeadline := ctx.Deadline(); hasCmdDeadline {
					// Calculate remaining time until command deadline
					remainingTime := time.Until(cmdDeadline)
					// Use the minimum of remaining time and DialTimeout
					if remainingTime < dialTimeout {
						// Command deadline is sooner, use it
						waitCtx = ctx
					} else {
						// DialTimeout is shorter, cap the wait at DialTimeout
						waitCtx, cancel = context.WithTimeout(ctx, dialTimeout)
					}
				} else {
					// No command deadline, use DialTimeout to prevent waiting indefinitely
					waitCtx, cancel = context.WithTimeout(ctx, dialTimeout)
				}
				if cancel != nil {
					defer cancel()
				}

				finalState, err := cn.GetStateMachine().AwaitAndTransition(
					waitCtx,
					[]pool.ConnState{pool.StateIdle, pool.StateInUse},
					pool.StateIdle, // Target is IDLE (but we're already there, so this is a no-op)
				)
				if err != nil {
					return err
				}
				// Verify we're now initialized
				if finalState == pool.StateIdle || finalState == pool.StateInUse {
					return nil
				}
				// Unexpected state after waiting
				return fmt.Errorf("connection in unexpected state after initialization: %s", finalState)
			}

			// Unexpected state (CLOSED, UNUSABLE, etc.)
			return err
		}
	}

	// At this point, we're in INITIALIZING state and we own the initialization
	// If we fail, we must transition to CLOSED
	var initErr error
	connPool := pool.NewSingleConnPool(c.connPool, cn)
	// The handshake Conn (handed to OnConnect) must share the client's
	// HIMPORT registry: a private registry restarts versions at 1, so an
	// OnConnect prepare would mark the pooled connection with a version
	// number that collides with the client registry's and silently skips
	// the replay of a different fieldset definition.
	conn := newConn(c.opt, connPool, &c.hooksMixin, c.himport)
	// The internal wrapper does not serve cached reads, but it needs the
	// successful-attachment signal both to issue CLIENT TRACKING during init
	// and to guard the user-visible OnConnect callback below.
	conn.baseClient.cscActive = c.cscActive
	// This internal conn's init pipeline issues CLIENT TRACKING ON itself;
	// exempt it from the guard that blocks user-issued CLIENT TRACKING. Setting
	// the field after newConn is safe: initHooks bound the pipeline hook as a
	// method value on the addressable baseClient, so the guard reads the
	// updated field.
	conn.baseClient.allowClientTracking = true

	username, password := "", ""
	if c.opt.StreamingCredentialsProvider != nil {
		credListener, initErr := c.streamingCredentialsManager.Listener(
			cn,
			c.reAuthConnection(),
			c.onAuthenticationErr(),
		)
		if initErr != nil {
			cn.GetStateMachine().Transition(pool.StateClosed)
			return fmt.Errorf("failed to create credentials listener: %w", initErr)
		}

		credentials, unsubscribeFromCredentialsProvider, initErr := c.opt.StreamingCredentialsProvider.
			Subscribe(credListener)
		if initErr != nil {
			cn.GetStateMachine().Transition(pool.StateClosed)
			return fmt.Errorf("failed to subscribe to streaming credentials: %w", initErr)
		}

		// Per-connection unsubscribe is attached to the connection itself so it
		// runs when this specific connection is closed. Do not register it on
		// c.onClose: initConn runs for every (re)initialized connection, and
		// attaching per-connection state to the shared baseClient registry would
		// either leak entries (one per connection id, never trimmed) or — with
		// the pre-fix wrappedOnClose approach — build an unbounded closure chain
		// retaining every prior connection's unsubscribe (see issue #3772).
		//
		// Note: pool.Conn.SetOnClose OVERWRITES any prior callback (see the
		// doc on that method). That is safe here because the streaming
		// credentials Manager deduplicates listeners by connection id, so a
		// second initConn on the same cn re-Subscribes the SAME listener and
		// the returned unsubscribe is equivalent to the one already installed.
		// Any future code path that could hand out a distinct unsubscribe on
		// re-initialization must first invoke the existing one to avoid
		// orphaning the old subscription on the credentials provider.
		cn.SetOnClose(unsubscribeFromCredentialsProvider)

		username, password = credentials.BasicAuth()
	} else {
		username, password, initErr = c.opt.resolveCredentials(ctx)
		if initErr != nil {
			cn.GetStateMachine().Transition(pool.StateClosed)
			return fmt.Errorf("failed to resolve credentials: %w", initErr)
		}
	}

	// for redis-server versions that do not support the HELLO command,
	// RESP2 will continue to be used.
	// helloOK tracks whether HELLO succeeded. If it did not, the connection
	// falls back to RESP2 regardless of c.opt.Protocol, and features that
	// require RESP3 (e.g. maintenance notifications) must be skipped.
	helloOK := false
	// For redis-server versions that do not support HELLO, RESP2 continues to
	// be used. Remember that negotiated fallback: configured Protocol remains 3,
	// but CSC must not serve without RESP3 invalidations.
	helloFallbackToRESP2 := false
	if initErr = conn.Hello(ctx, c.opt.Protocol, username, password, c.opt.ClientName).Err(); initErr == nil {
		// Authentication successful with HELLO command
		helloOK = true
	} else if !isRedisError(initErr) {
		// When the server responds with the RESP protocol and the result is not a normal
		// execution result of the HELLO command, we consider it to be an indication that
		// the server does not support the HELLO command.
		// The server may be a redis-server that does not support the HELLO command,
		// or it could be DragonflyDB or a third-party redis-proxy. They all respond
		// with different error string results for unsupported commands, making it
		// difficult to rely on error strings to determine all results.
		cn.GetStateMachine().Transition(pool.StateClosed)
		return initErr
	} else {
		helloFallbackToRESP2 = c.opt.Protocol == 3
		if password != "" {
			// Try legacy AUTH command if HELLO failed.
			if username != "" {
				initErr = conn.AuthACL(ctx, username, password).Err()
			} else {
				initErr = conn.Auth(ctx, password).Err()
			}
			if initErr != nil {
				cn.GetStateMachine().Transition(pool.StateClosed)
				return fmt.Errorf("failed to authenticate: %w", initErr)
			}
		}
	}
	if helloFallbackToRESP2 {
		c.disableCSCServing(ctx, "HELLO 3 was rejected and the connection negotiated RESP2")
	}

	// trackingEnabled reports whether THIS pool connection must issue
	// CLIENT TRACKING ON during init. True when CSC (SharedTracking) is enabled:
	// the shared cache is fed by per-connection tracking + the background
	// drainer. Once CSC serving stops (owner Close, GC cleanup, or drainer
	// damping), new and re-inited conns skip tracking — nothing consumes the
	// pushes into the cache anymore.
	trackingEnabled := !helloFallbackToRESP2 && !cn.IsPubSub() && c.cscTrackingRequested()
	if trackingEnabled && c.cscConnInitGen(cn.GetID()) == 0 {
		// First initialization establishes generation 1. Reinitialization
		// already bumped and evicted through onCscReinit before replacing the
		// socket, so it must not bump a second time here.
		c.cscEvictOwnedEntries(cn.GetID())
	}
	var trackingCmd *StatusCmd
	initCmds, initErr := conn.Pipelined(ctx, func(pipe Pipeliner) error {
		if c.opt.DB > 0 {
			pipe.Select(ctx, c.opt.DB)
		}

		if c.opt.readOnly {
			pipe.ReadOnly(ctx)
		}

		if c.opt.ClientName != "" {
			pipe.ClientSetName(ctx, c.opt.ClientName)
		}

		if trackingEnabled {
			// Must run before any cacheable command is issued on this conn.
			trackingCmd = pipe.ClientTrackingOn(ctx, nil)
		}

		return nil
	})
	// The exemption is init-only. OnConnect is user code and must go through
	// the same CSC connection-state guard as every other public command path.
	conn.baseClient.allowClientTracking = false
	trackingRejected := trackingCmd != nil && isRedisError(trackingCmd.Err())
	for _, cmd := range initCmds {
		if cmd != trackingCmd && cmd.Err() != nil {
			trackingRejected = false
			break
		}
	}
	if trackingRejected {
		// A server-side rejection means tracking is unavailable, but the
		// connection and the preceding init commands are still usable. Disable
		// CSC globally and continue without caching. Transport and protocol
		// failures still take the normal connection-failure path below.
		c.disableCSCServing(ctx, fmt.Sprintf("CLIENT TRACKING ON was rejected: %v", trackingCmd.Err()))
		c.cscForgetConn(cn.GetID())
		trackingEnabled = false
		initErr = nil
	}
	if initErr != nil {
		if trackingEnabled {
			// cscEvictOwnedEntries above bumped this conn's init generation; a
			// failed init never serves, and the pubsub path has no OnRemove
			// hook (and the close hook below is not yet installed), so drop
			// the entry here to keep the map bounded to live conns.
			c.cscForgetConn(cn.GetID())
		}
		cn.GetStateMachine().Transition(pool.StateClosed)
		return fmt.Errorf("failed to initialize connection options: %w", initErr)
	}

	if trackingEnabled {
		// Evict this conn's entries on any close (incl. the ConnMaxLifetime/idle
		// path that bypasses the OnRemove hook), since the server drops its
		// tracking table on close.
		c.cscInstallConnCloseHook(cn)
		// A handoff replaces the socket before initConn runs. Bump and evict at
		// the pre-swap boundary so fulfillCached cannot publish an old-socket
		// reply during that gap.
		c.cscInstallConnReinitHook(cn)
	}

	// Enable maintnotifications if maintnotifications are configured
	c.optLock.RLock()
	maintNotifEnabled := c.opt.MaintNotificationsConfig != nil && c.opt.MaintNotificationsConfig.Mode != maintnotifications.ModeDisabled
	protocol := c.opt.Protocol
	var endpointType maintnotifications.EndpointType
	var maintNotifMode maintnotifications.Mode
	if maintNotifEnabled {
		endpointType = c.opt.MaintNotificationsConfig.EndpointType
		maintNotifMode = c.opt.MaintNotificationsConfig.Mode
	}
	c.optLock.RUnlock()

	// Maintenance notifications require RESP3 push frames. If HELLO failed
	// and the connection fell back to RESP2, there is no point in sending
	// CLIENT MAINT_NOTIFICATIONS: the server either rejects it (making the
	// error misleading) or accepts it silently, leaving the client unable
	// to receive any notifications. Decide based on the actual negotiated
	// protocol rather than the requested one.
	if maintNotifEnabled && protocol == 3 && !helloOK {
		if maintNotifMode == maintnotifications.ModeEnabled {
			// Explicitly requested - fail fast with a clear reason.
			cn.GetStateMachine().Transition(pool.StateClosed)
			if errorCallback := pool.GetMetricErrorCallback(); errorCallback != nil {
				errorCallback(ctx, "HANDSHAKE_FAILED", cn, "HANDSHAKE_FAILED", true, 0)
			}
			return fmt.Errorf("failed to enable maintnotifications: server does not support RESP3 (HELLO command failed)")
		}
		// auto/other modes: silently disable maintnotifications for this client.
		c.optLock.Lock()
		c.opt.MaintNotificationsConfig.Mode = maintnotifications.ModeDisabled
		c.optLock.Unlock()
		if err := c.disableMaintNotificationsUpgrades(); err != nil {
			internal.Logger.Printf(ctx, "failed to disable maintnotifications in auto mode: %v", err)
		}
		maintNotifEnabled = false
	}

	var maintNotifHandshakeErr error
	if maintNotifEnabled && protocol == 3 {
		// Hold the manager read lock across the handshake and tracking so a
		// concurrent downgrade cannot remove pool-level listeners before a
		// successfully enabled connection is tracked for retirement.
		c.maintNotificationsManagerLock.RLock()
		manager := c.maintNotificationsManager
		maintNotifHandshakeErr = conn.ClientMaintNotifications(
			ctx,
			true,
			endpointType.String(),
		).Err()
		// A successful handshake enables maintnotifications for this connection,
		// but must not promote ModeAuto to ModeEnabled. ModeEnabled is the
		// explicit fail-closed policy; ModeAuto must remain able to downgrade if a
		// later reconnect/failover reaches an endpoint that rejects the command.
		if maintNotifHandshakeErr == nil && manager != nil {
			manager.TrackMaintNotificationsConn(cn)
		}
		c.maintNotificationsManagerLock.RUnlock()
		if maintNotifHandshakeErr != nil {
			if !isRedisError(maintNotifHandshakeErr) {
				// if not redis error, fail the connection
				cn.GetStateMachine().Transition(pool.StateClosed)
				return maintNotifHandshakeErr
			}
			c.optLock.Lock()
			// handshake failed - check and modify config atomically
			switch c.opt.MaintNotificationsConfig.Mode {
			case maintnotifications.ModeEnabled:
				// enabled mode, fail the connection
				c.optLock.Unlock()
				cn.GetStateMachine().Transition(pool.StateClosed)

				// Record handshake failure metric
				if errorCallback := pool.GetMetricErrorCallback(); errorCallback != nil {
					errorCallback(ctx, "HANDSHAKE_FAILED", cn, "HANDSHAKE_FAILED", true, 0)
				}

				return fmt.Errorf("failed to enable maintnotifications: %w", maintNotifHandshakeErr)
			default: // will handle auto and any other
				// Disabling logging here as it's too noisy.
				// TODO: Enable when we have a better logging solution for log levels
				// internal.Logger.Printf(ctx, "auto mode fallback: maintnotifications disabled due to handshake error: %v", maintNotifHandshakeErr)
				c.opt.MaintNotificationsConfig.Mode = maintnotifications.ModeDisabled
				c.optLock.Unlock()
				// auto mode, disable maintnotifications and continue
				if initErr := c.disableMaintNotificationsUpgrades(); initErr != nil {
					// Log error but continue - auto mode should be resilient
					internal.Logger.Printf(ctx, "failed to disable maintnotifications in auto mode: %v", initErr)
				}
			}
		}
	}

	if !c.opt.DisableIdentity && !c.opt.DisableIndentity {
		libName := ""
		libVer := Version()
		if c.opt.IdentitySuffix != "" {
			libName = c.opt.IdentitySuffix
		}
		p := conn.Pipeline()
		p.ClientSetInfo(ctx, WithLibraryName(libName))
		p.ClientSetInfo(ctx, WithLibraryVersion(libVer))
		// Handle network errors (e.g. timeouts) in CLIENT SETINFO to avoid
		// out of order responses later on.
		if _, initErr = p.Exec(ctx); initErr != nil && !isRedisError(initErr) {
			cn.GetStateMachine().Transition(pool.StateClosed)
			return initErr
		}
	}

	// Set the connection initialization function for potential reconnections
	// This must be set before transitioning to IDLE so that handoff/reauth can use it
	cn.SetInitConnFunc(c.createInitConnFunc())

	// Initialization succeeded - transition to IDLE state
	// This marks the connection as initialized and ready for use
	// NOTE: The connection is still owned by the calling goroutine at this point
	// and won't be available to other goroutines until it's Put() back into the pool
	cn.GetStateMachine().Transition(pool.StateIdle)

	// Call OnConnect hook if configured
	// The connection is in IDLE state but still owned by this goroutine
	// If OnConnect needs to send commands, it can use the connection safely
	if c.opt.OnConnect != nil {
		if initErr = c.opt.OnConnect(ctx, conn); initErr != nil {
			// OnConnect failed - transition to closed
			cn.GetStateMachine().Transition(pool.StateClosed)
			return initErr
		}
	}

	return nil
}

func (c *baseClient) releaseConn(ctx context.Context, cn *pool.Conn, err error) {
	if c.opt.Limiter != nil {
		c.opt.Limiter.ReportResult(err)
	}
	c.releaseConnToPool(ctx, c.connPool, cn, err)
}

// releaseConnToPool returns a conn to p after a command or pipeline ran on
// it: bad conns are Removed, pending push notifications are drained (a
// mid-frame drain failure also Removes — the reply stream may be
// desynchronized), and a client-side-cache post-read probe is requested when
// tracking is on. Limiter accounting stays with the callers, whose shapes
// differ. Shared by releaseConn and withPipelineConn so the two cannot drift.
func (c *baseClient) releaseConnToPool(ctx context.Context, p pool.Pooler, cn *pool.Conn, err error) {
	if isBadConn(err, false, c.opt.Addr) {
		p.Remove(ctx, cn, err)
		return
	}
	// process any pending push notifications before returning the connection to the pool
	if err := c.processPushNotifications(ctx, cn); err != nil {
		internal.Logger.Printf(ctx, "push: error processing pending notifications before releasing connection: %v", err)
		if isBadConn(err, false, c.opt.Addr) {
			// A mid-frame read failure may leave the reply stream
			// desynchronized, so the connection cannot be reused.
			p.Remove(ctx, cn, err)
			return
		}
	}
	if c.cscTrackingRequested() {
		// A TLS-like wrapper can retain decrypted bytes after the command
		// reply even when its raw socket is empty. Ask the background
		// drainer for one bounded post-read probe before relying on raw
		// socket peeks again.
		cn.MarkCscReadPending()
	}
	p.Put(ctx, cn)
}

func (c *baseClient) withConn(
	ctx context.Context, fn func(context.Context, *pool.Conn) error,
) error {
	cn, err := c.getConn(ctx)
	if err != nil {
		return err
	}

	var fnErr error
	defer func() {
		c.releaseConn(ctx, cn, fnErr)
	}()

	fnErr = fn(ctx, cn)

	return fnErr
}

// withPipelineConn executes fn with a connection from the pipeline pool when
// one is configured (PipelineReadBufferSize/PipelineWriteBufferSize set),
// otherwise it falls back to the regular pool via withConn.
// withPipelineConn is withConn/releaseConn for the DEDICATED pipeline pool.
// Conn preparation and release go through the shared pool-parameterized
// helpers (initPooledConn, releaseConnToPool) — the paths used to mirror each
// other by hand and drifted three times (a Limiter-ordering divergence, a
// missed drain-error removal, a missed client-side-cache probe), so only the
// Limiter shape is allowed to live here.
func (c *baseClient) withPipelineConn(
	ctx context.Context, fn func(context.Context, *pool.Conn) error,
) (retErr error) {
	// Use pipeline pool if available, otherwise fall back to regular pool.
	if c.pipelinePool == nil {
		return c.withConn(ctx, fn)
	}

	// Honor the Limiter on the dedicated pipeline-pool path too, mirroring
	// getConn/releaseConn: Allow() before acquiring and ReportResult() on every
	// exit (including the early init/re-acquire failures below). Without this,
	// enabling the pipeline pool would silently bypass throttling and failure
	// reporting for callers that set a Limiter.
	if c.opt.Limiter != nil {
		if err := c.opt.Limiter.Allow(); err != nil {
			return err
		}
	}

	// One deferred exit for both concerns, because their ORDER is part of the
	// contract: releaseConn reports the result BEFORE the connection becomes
	// available again, so a limiter or circuit breaker observes the failure
	// before it can admit the next operation. Two separate defers would run
	// LIFO and release first, letting another pipelined operation through
	// against a breaker that has not seen the failure yet (review finding by
	// codex on #3942). cn is nil on the acquire/init failure paths, which still
	// must report.
	var cn *pool.Conn
	var fnErr error
	defer func() {
		if c.opt.Limiter != nil {
			c.opt.Limiter.ReportResult(retErr)
		}
		if cn != nil {
			c.releaseConnToPool(ctx, c.pipelinePool, cn, fnErr)
		}
	}()

	cn, retErr = c.pipelinePool.Get(ctx)
	if retErr != nil {
		cn = nil // nothing acquired: no release, but still report above
		return retErr
	}

	if err := c.initPooledConn(ctx, c.pipelinePool, cn); err != nil {
		// initPooledConn already removed the conn from the pool on failure.
		cn = nil
		retErr = err
		return retErr
	}

	fnErr = fn(ctx, cn)
	retErr = fnErr
	return retErr
}

func (c *baseClient) dial(ctx context.Context, network, addr string) (net.Conn, error) {
	return c.opt.Dialer(ctx, network, addr)
}

// cscTrackingRequested reports whether initConn must issue CLIENT TRACKING ON.
// cscActive is allocated only after attachment succeeds and is shared with
// derived clients: a conn initialized by Conn/Tx may later return to the
// parent's pool, but a configured cache whose attachment failed must not turn
// tracking on.
func (c *baseClient) cscTrackingRequested() bool {
	if c.opt.Protocol != 3 || c.cscActive == nil || !c.cscActive.Load() {
		return false
	}
	return c.opt.DB == 0
}

func (c *baseClient) process(ctx context.Context, cmd Cmder) error {
	opDurationCallback := otel.GetOperationDurationCallback()
	if opDurationCallback == nil {
		return c.processCommand(ctx, cmd, nil)
	}

	start := time.Now()
	var state processState
	err := c.processCommand(ctx, cmd, &state)
	opDurationCallback(ctx, time.Since(start), cmd, state.attempts, err, state.lastConn, c.opt.DB)
	return err
}

type processState struct {
	attempts int
	lastConn *pool.Conn
}

func (c *baseClient) processCommand(ctx context.Context, cmd Cmder, state *processState) error {
	// Reject commands that would make one pooled connection diverge from CSC's
	// tracking or database assumptions. Pipelines mirror this guard below.
	if err := c.cscCommandError(cmd); err != nil {
		return err
	}
	if c.csc != nil && isCacheable(cmd) {
		return c.processCached(ctx, cmd, state)
	}
	return c.processWithRetry(ctx, cmd, nil, state)
}

// processWithRetry runs cmd through the retry loop. capture (optional) is
// filled by the successful attempt's reply read for the CSC fetch path (see
// cscFetchCapture).
func (c *baseClient) processWithRetry(
	ctx context.Context, cmd Cmder, capture *cscFetchCapture, state *processState,
) error {
	var lastConn *pool.Conn

	var lastErr error
	totalAttempts := 0
	maxRetries := c.opt.MaxRetries
	himportRetried := false
	for attempt := 0; attempt <= maxRetries; attempt++ {
		totalAttempts++
		attempt := attempt

		retry, cn, err := c._process(ctx, cmd, attempt, capture)
		if cn != nil {
			lastConn = cn
		}
		if state != nil {
			state.attempts = totalAttempts
			state.lastConn = lastConn
		}
		// A "no such fieldset" reply for a registered fieldset means the
		// connection lost its server session state (e.g. RESET, concurrent
		// discard). The stale prepared flag was invalidated inside _process
		// while the connection was still held; grant a single extra attempt
		// so the retry re-prepares lazily on whichever connection it lands.
		if err != nil && !retry && !himportRetried && !cmd.NoRetry() &&
			c.himportShouldRetrySet(cmd, err) {
			himportRetried = true
			if attempt == maxRetries {
				maxRetries++
			}
			lastErr = err
			continue
		}
		// Don't retry if command explicitly disables retries (e.g., RawWriteToCmd
		// which writes directly to an io.Writer and cannot undo partial writes)
		if err == nil || !retry || cmd.NoRetry() {
			if err != nil {
				if errorCallback := pool.GetMetricErrorCallback(); errorCallback != nil {
					errorType, statusCode, isInternal := classifyCommandError(err)
					errorCallback(ctx, errorType, lastConn, statusCode, isInternal, totalAttempts-1)
				}
			}
			return err
		}

		lastErr = err
	}

	// Record error metric for exhausted retries
	if errorCallback := pool.GetMetricErrorCallback(); errorCallback != nil {
		errorType, statusCode, isInternal := classifyCommandError(lastErr)
		errorCallback(ctx, errorType, lastConn, statusCode, isInternal, totalAttempts-1)
	}

	return lastErr
}

// classifyCommandError classifies an error for metrics reporting.
// Returns: errorType, statusCode, isInternal
// - errorType: A string describing the error type (e.g., "TIMEOUT", "NETWORK", "ERR")
// - statusCode: The Redis error prefix or error category
// - isInternal: true for network/timeout errors, false for Redis server errors
func classifyCommandError(err error) (errorType, statusCode string, isInternal bool) {
	if err == nil {
		return "", "", false
	}

	errStr := err.Error()

	// Check for timeout errors
	if netErr, ok := err.(net.Error); ok && netErr.Timeout() {
		return "TIMEOUT", "TIMEOUT", true
	}

	// Check for network errors
	if _, ok := err.(net.Error); ok {
		return "NETWORK", "NETWORK", true
	}

	// Check for context errors
	if errors.Is(err, context.Canceled) {
		return "CONTEXT_CANCELED", "CONTEXT_CANCELED", true
	}
	if errors.Is(err, context.DeadlineExceeded) {
		return "CONTEXT_TIMEOUT", "CONTEXT_TIMEOUT", true
	}

	// Check for Redis errors
	// Examples: "ERR ...", "WRONGTYPE ...", "CLUSTERDOWN ..."
	if len(errStr) > 0 {
		// Find the first space to extract the prefix
		spaceIdx := 0
		for i, c := range errStr {
			if c == ' ' {
				spaceIdx = i
				break
			}
		}
		if spaceIdx == 0 {
			spaceIdx = len(errStr)
		}
		prefix := errStr[:spaceIdx]
		isUppercase := true
		for _, c := range prefix {
			if c < 'A' || c > 'Z' {
				isUppercase = false
				break
			}
		}
		if isUppercase && len(prefix) > 0 {
			return prefix, prefix, false
		}
	}

	return "UNKNOWN", "UNKNOWN", true
}

func (c *baseClient) _process(ctx context.Context, cmd Cmder, attempt int, capture *cscFetchCapture) (bool, *pool.Conn, error) {
	if attempt > 0 {
		if err := internal.Sleep(ctx, c.retryBackoff(attempt)); err != nil {
			return false, nil, err
		}
	}

	var usedConn *pool.Conn
	var retryTimeout atomic.Uint32
	if err := c.withConn(ctx, func(ctx context.Context, cn *pool.Conn) error {
		usedConn = cn
		// Process any pending push notifications before executing the command
		if err := c.processPushNotifications(ctx, cn); err != nil {
			internal.Logger.Printf(ctx, "push: error processing pending notifications before command: %v", err)
		}

		// HIMPORT bookkeeping: pending discards for this session and the
		// PREPARE for an HIMPORT SET's registered fieldset are written in
		// the same round trip, right before the command.
		var injected []Cmder
		if _, ok := cmd.(himportCmder); ok {
			injected = c.himportInjectedCmds(ctx, cn, []Cmder{cmd})
		}

		if err := cn.WithWriter(c.context(ctx), c.opt.WriteTimeout, func(wr *proto.Writer) error {
			for _, ic := range injected {
				if err := writeCmd(wr, ic); err != nil {
					return err
				}
			}
			return writeCmd(wr, cmd)
		}); err != nil {
			retryTimeout.Store(1)
			return err
		}
		readReplyFunc := cmd.readReply
		// When the caller requested raw-reply capture (client-side cache),
		// read the reply as raw RESP bytes and re-parse them through the
		// command's normal reply handler. This reuses proto.Reader rather
		// than duplicating parsing logic in a bespoke cache serializer.
		if capture != nil {
			origRead := readReplyFunc
			readReplyFunc = func(rd *proto.Reader) error {
				raw, err := rd.ReadRawReply()
				if err != nil {
					return err
				}
				capture.raw = raw
				return origRead(proto.NewReaderSize(bytes.NewReader(raw), len(raw)+1))
			}
		}
		readErr := cn.WithReader(c.context(ctx), c.cmdTimeout(cmd), func(rd *proto.Reader) error {
			// To be sure there are no buffered push notifications, we process them before reading the reply
			if err := c.processPendingPushNotificationWithReader(ctx, cn, rd); err != nil {
				internal.Logger.Printf(ctx, "push: error processing pending notifications before reading reply: %v", err)
			}
			if len(injected) > 0 {
				if err := c.himportReadInjectedReplies(ctx, cn, rd, injected); err != nil {
					return err
				}
				// A push notification can arrive between the injected
				// replies and the command reply; drain again so the
				// reply read below does not consume it as the command's.
				if err := c.processPendingPushNotificationWithReader(ctx, cn, rd); err != nil {
					internal.Logger.Printf(ctx, "push: error processing pending notifications before reading reply: %v", err)
				}
			}
			err := readReplyFunc(rd)
			// Assert the command type before touching the error: the
			// errors.As chain inside himportNoSuchFieldset allocates, and
			// this is the per-command hot path.
			if set, ok := cmd.(*HImportSetCmd); ok && himportNoSuchFieldset(err) {
				// A failed injected PREPARE is the root cause of the
				// command's "no such fieldset" reply (drained above).
				for _, ic := range injected {
					if prep, ok := ic.(*HImportPrepareCmd); ok &&
						prep.fieldsetName == set.fieldsetName && prep.Err() != nil {
						err = prep.Err()
						break
					}
				}
				// The session lost a registered fieldset the flags claim is
				// prepared — and the same event (failover, cross-region
				// switch, reset storm) may have wiped other sessions whose
				// flags also still look current. Bump the fieldset version
				// so every connection re-prepares before its next use,
				// wherever the retry granted by process() lands.
				if himportNoSuchFieldset(err) {
					if fs, registered := c.himport.lookup(set.fieldsetName); registered {
						c.himport.refreshVersion(set.fieldsetName, fs.version)
					}
				}
			}
			return err
		})
		// redis.Nil is a complete, valid negative reply. For a CSC fetch, retain
		// its connection attribution before returning Nil to the caller so the
		// raw reply can be cached and invalidated like any other read result.
		if readErr != nil && (capture == nil || readErr != Nil) {
			if cmd.readTimeout() == nil {
				retryTimeout.Store(1)
			} else {
				retryTimeout.Store(0)
			}
			return readErr
		}
		if capture != nil {
			// Attribute while the conn is still held: once it is released, a
			// queued handoff may swap the socket and bump the generation, and
			// this capture is what fulfillCached compares against.
			capture.connID = cn.GetID()
			capture.initGen = c.cscConnInitGen(capture.connID)
		}

		if hc, ok := cmd.(himportCmder); ok {
			c.himportAfterCmd(cn, hc)
		}
		return readErr
	}); err != nil {
		retry := shouldRetry(err, retryTimeout.Load() == 1)
		return retry, usedConn, err
	}

	return false, usedConn, nil
}

func (c *baseClient) retryBackoff(attempt int) time.Duration {
	return internal.RetryBackoff(attempt, c.opt.MinRetryBackoff, c.opt.MaxRetryBackoff)
}

func (c *baseClient) cmdTimeout(cmd Cmder) time.Duration {
	if timeout := cmd.readTimeout(); timeout != nil {
		t := *timeout
		if t == 0 {
			return 0
		}
		return t + 10*time.Second
	}
	return c.opt.ReadTimeout
}

// context returns the context for the current connection.
// If the context timeout is enabled, it returns the original context.
// Otherwise, it returns a new background context.
func (c *baseClient) context(ctx context.Context) context.Context {
	if c.opt.ContextTimeoutEnabled {
		return ctx
	}
	return context.Background()
}

// createInitConnFunc creates a connection initialization function that can be used for reconnections.
func (c *baseClient) createInitConnFunc() func(context.Context, *pool.Conn) error {
	return func(ctx context.Context, cn *pool.Conn) error {
		return c.initConn(ctx, cn)
	}
}

// enableMaintNotificationsUpgrades initializes the maintnotifications upgrade manager and pool hook.
// This function is called during client initialization.
// will register push notification handlers for all maintenance upgrade events.
// will start background workers for handoff processing in the pool hook.
func (c *baseClient) enableMaintNotificationsUpgrades() error {
	// Create client adapter
	clientAdapterInstance := newClientAdapter(c)

	// Create maintnotifications manager directly
	manager, err := maintnotifications.NewManager(clientAdapterInstance, c.connPool, c.opt.MaintNotificationsConfig)
	if err != nil {
		return err
	}
	// Set the manager reference and initialize pool hook
	c.maintNotificationsManagerLock.Lock()
	c.maintNotificationsManager = manager
	c.maintNotificationsManagerLock.Unlock()

	// Initialize pool hook (safe to call without lock since manager is now set)
	manager.InitPoolHook(c.dialHook)
	// If a dedicated pipeline connection pool is in use, attach an independent
	// maintnotifications hook to it as well. Otherwise autopipelined/pipelined
	// commands run on pipeline-pool connections that never receive MOVING/
	// MIGRATING handoff handling.
	if c.pipelinePool != nil {
		manager.InitPoolHookForPool(c.pipelinePool, c.dialHook)
	}
	return nil
}

func (c *baseClient) disableMaintNotificationsUpgrades() error {
	c.maintNotificationsManagerLock.Lock()
	defer c.maintNotificationsManagerLock.Unlock()

	// Close the maintnotifications manager
	if c.maintNotificationsManager != nil {
		// Closing the manager will also shutdown the pool hook
		// and remove it from the pool
		if err := c.maintNotificationsManager.Close(); err != nil {
			return err
		}
		c.maintNotificationsManager = nil
	}
	return nil
}

// Close closes the client, releasing any open resources.
//
// It is rare to Close a Client, as the Client is meant to be
// long-lived and shared between many goroutines.
func (c *baseClient) Close() error {
	// The pools this baseClient owns are shared with every WithTimeout/
	// WithReadTimeout clone. Once ANY sharer closes them, no wrapper may
	// build a fresh autopipeliner against them — its flushers would run
	// against closed pools forever. The atomic is checked by the
	// AutoPipeline getters of every wrapper sharing this base.
	if c.apClosed != nil {
		c.apClosed.Store(true)
	}
	if h := c.cscDrainHandle; h != nil {
		h.closeOnce.Do(func() {
			h.closeErr = c.closeResources()
		})
		return h.closeErr
	}
	return c.closeResources()
}

func (c *baseClient) closeResources() error {
	var firstErr error

	// CSC teardown (no-op when CSC is not active): stop the background
	// invalidation drainer before the pool it walks is torn down.
	c.stopBackgroundDrainer()

	// Close maintnotifications manager first
	if err := c.disableMaintNotificationsUpgrades(); err != nil {
		firstErr = err
	}

	if err := c.onClose.run(); err != nil && firstErr == nil {
		firstErr = err
	}

	// Unregister pools from OTel before closing them
	otel.UnregisterPools(c.connPool, c.pubSubPool, c.pipelinePool)

	if c.connPool != nil {
		if err := c.connPool.Close(); err != nil && firstErr == nil {
			firstErr = err
		}
	}
	if c.pipelinePool != nil {
		if err := c.pipelinePool.Close(); err != nil && firstErr == nil {
			firstErr = err
		}
	}
	if c.pubSubPool != nil {
		if err := c.pubSubPool.Close(); err != nil && firstErr == nil {
			firstErr = err
		}
	}
	return firstErr
}

func (c *baseClient) getAddr() string {
	return c.opt.Addr
}

func (c *baseClient) processPipeline(ctx context.Context, cmds []Cmder) error {
	if err := c.generalProcessPipeline(ctx, cmds, c.pipelineProcessCmds, "PIPELINE"); err != nil {
		return err
	}
	return cmdsFirstErr(cmds)
}

func (c *baseClient) processTxPipeline(ctx context.Context, cmds []Cmder) error {
	if err := c.generalProcessPipeline(ctx, cmds, c.txPipelineProcessCmds, "MULTI"); err != nil {
		return err
	}
	return cmdsFirstErr(cmds)
}

type pipelineProcessor func(context.Context, *pool.Conn, []Cmder) (bool, error)

func (c *baseClient) generalProcessPipeline(
	ctx context.Context, cmds []Cmder, p pipelineProcessor, operationName string,
) error {
	// Pipeline commands never pass through process, so apply the same CSC state
	// guard here. initConn's internal client is exempt.
	for _, cmd := range cmds {
		if err := c.cscCommandError(cmd); err != nil {
			setCmdsErr(cmds, err)
			return err
		}
	}
	// Only call time.Now() if pipeline operation duration callback is set to avoid overhead
	var operationStart time.Time
	pipelineOpDurationCallback := otel.GetPipelineOperationDurationCallback()
	if pipelineOpDurationCallback != nil {
		operationStart = time.Now()
	}
	var lastConn *pool.Conn
	totalAttempts := 0

	var lastErr error
	for attempt := 0; attempt <= c.opt.MaxRetries; attempt++ {
		totalAttempts++
		if attempt > 0 {
			if err := internal.Sleep(ctx, c.retryBackoff(attempt)); err != nil {
				setCmdsErr(cmds, err)
				if pipelineOpDurationCallback != nil {
					operationDuration := time.Since(operationStart)
					pipelineOpDurationCallback(ctx, operationDuration, operationName, len(cmds), totalAttempts, err, lastConn, c.opt.DB)
				}
				return err
			}
		}

		// Enable retries by default to retry dial errors returned by withConn.
		canRetry := true
		// Route pipelines through the dedicated pipeline pool when configured;
		// withPipelineConn falls back to the regular pool when it is not.
		lastErr = c.withPipelineConn(ctx, func(ctx context.Context, cn *pool.Conn) error {
			lastConn = cn
			// Process any pending push notifications before executing the pipeline
			if err := c.processPushNotifications(ctx, cn); err != nil {
				internal.Logger.Printf(ctx, "push: error processing pending notifications before processing pipeline: %v", err)
			}
			var err error
			canRetry, err = p(ctx, cn, cmds)
			return err
		})
		// Don't retry if any command in the pipeline explicitly disables retries
		// (e.g., RawWriteToCmd which writes directly to an io.Writer and cannot
		// undo partial writes on retry)
		if lastErr == nil || !canRetry || !shouldRetry(lastErr, true) || cmdsContainNoRetry(cmds) {
			// The error should be set here only when failing to obtain the conn.
			if !isRedisError(lastErr) {
				setCmdsErr(cmds, lastErr)
			}
			if pipelineOpDurationCallback != nil {
				operationDuration := time.Since(operationStart)
				pipelineOpDurationCallback(ctx, operationDuration, operationName, len(cmds), totalAttempts, lastErr, lastConn, c.opt.DB)
			}

			if lastErr != nil {
				if errorCallback := pool.GetMetricErrorCallback(); errorCallback != nil {
					errorType, statusCode, isInternal := classifyCommandError(lastErr)
					errorCallback(ctx, errorType, lastConn, statusCode, isInternal, totalAttempts-1)
				}
			}
			return lastErr
		}
	}

	// Retries exhausted on a retryable error: the loop fell through without the
	// early-exit branch running, so the commands were never populated with the
	// failure. Mirror that branch here so callers that observe results only
	// per-command — notably AutoPipeline, which discards this function's returned
	// error — see the error instead of a nil error and a zero value. Guard on
	// !isRedisError so a per-command redis error (e.g. LOADING) keeps its own
	// reply rather than being overwritten.
	if !isRedisError(lastErr) {
		setCmdsErr(cmds, lastErr)
	}

	if pipelineOpDurationCallback != nil {
		operationDuration := time.Since(operationStart)
		pipelineOpDurationCallback(ctx, operationDuration, operationName, len(cmds), totalAttempts, lastErr, lastConn, c.opt.DB)
	}

	if errorCallback := pool.GetMetricErrorCallback(); errorCallback != nil {
		errorType, statusCode, isInternal := classifyCommandError(lastErr)
		errorCallback(ctx, errorType, lastConn, statusCode, isInternal, totalAttempts-1)
	}

	return lastErr
}

func (c *baseClient) pipelineProcessCmds(
	ctx context.Context, cn *pool.Conn, cmds []Cmder,
) (bool, error) {
	// Process any pending push notifications before executing the pipeline
	if err := c.processPushNotifications(ctx, cn); err != nil {
		internal.Logger.Printf(ctx, "push: error processing pending notifications before writing pipeline: %v", err)
	}

	// HIMPORT bookkeeping: pending discards for this session and PREPAREs
	// for registered fieldsets the batch references get written ahead of
	// the batch.
	injected := c.himportInjectedCmds(ctx, cn, cmds)

	if err := cn.WithWriter(c.context(ctx), c.opt.WriteTimeout, func(wr *proto.Writer) error {
		for _, ic := range injected {
			if err := writeCmd(wr, ic); err != nil {
				return err
			}
		}
		return writeCmds(wr, cmds)
	}); err != nil {
		setCmdsErr(cmds, err)
		return true, err
	}

	var readErr error
	if err := cn.WithReader(c.context(ctx), c.opt.ReadTimeout, func(rd *proto.Reader) error {
		if err := c.himportReadInjectedReplies(ctx, cn, rd, injected); err != nil {
			// Transport error with every batch reply unreadXX: stamp the
			// batch like a write failure. The outer retry loop stamps only
			// on its exit branch, not when attempts run out, so without
			// this a batch that keeps dying here would surface an Exec
			// error while every command still reports Err() == nil.
			setCmdsErr(cmds, err)
			return err
		}
		// read all replies
		readErr = c.pipelineReadCmds(ctx, cn, rd, cmds)
		if readErr != nil && !isRedisError(readErr) {
			return readErr
		}
		c.himportAfterBatch(cn, injected, cmds)
		return nil
	}); err != nil {
		return true, err
	}

	// Registered fieldsets whose SETs came back "no such fieldset" (the
	// session was lost between prepare and use) are re-prepared and those
	// SETs re-issued once on the same connection; the error must not
	// surface for managed fieldsets.
	//
	// A transport failure here must neither retry nor fail the batch: the
	// first round trip was fully consumed and its results delivered, so
	// re-executing would double-apply non-idempotent commands and failing
	// would stamp a spurious error onto commands that succeeded. The
	// re-issue errors stay on the retried SETs; the connection, which may
	// hold unread replies, is marked for removal when released.
	if err := c.himportRetryFailedSets(ctx, cn, cmds); err != nil {
		internal.Logger.Printf(ctx, "himport: pipeline set re-issue failed: %v", err)
		cn.MarkCloseOnPut("himport: transport error during set re-issue")
	}

	// Preserve retryable first-command errors (e.g. LOADING) for the outer
	// loop; the re-issue above may have cleared it. rawErr: this runs on the
	// execution path; never await here (an async autopipeline command's ready
	// channel is closed by this very batch — Err() would self-deadlock).
	if readErr != nil {
		readErr = cmds[0].rawErr()
	}
	return readErr != nil, readErr
}

func (c *baseClient) pipelineReadCmds(ctx context.Context, cn *pool.Conn, rd *proto.Reader, cmds []Cmder) error {
	for i, cmd := range cmds {
		// To be sure there are no buffered push notifications, we process them before reading the reply
		if err := c.processPendingPushNotificationWithReader(ctx, cn, rd); err != nil {
			internal.Logger.Printf(ctx, "push: error processing pending notifications before reading reply: %v", err)
		}
		err := cmd.readReply(rd)
		cmd.SetErr(err)
		if err != nil && !isRedisError(err) {
			setCmdsErr(cmds[i+1:], err)
			return err
		}
	}
	// Retry errors like "LOADING redis is loading the dataset in memory".
	// rawErr: this runs on the execution path; never await here (an async
	// autopipeline command's ready channel is closed by this very batch).
	return cmds[0].rawErr()
}

func (c *baseClient) txPipelineProcessCmds(
	ctx context.Context, cn *pool.Conn, cmds []Cmder,
) (bool, error) {
	// Process any pending push notifications before executing the transaction pipeline
	if err := c.processPushNotifications(ctx, cn); err != nil {
		internal.Logger.Printf(ctx, "push: error processing pending notifications before transaction: %v", err)
	}

	// HIMPORT bookkeeping: pending discards for this session and PREPAREs
	// for registered fieldsets the transaction references get written ahead
	// of MULTI; the session state is visible inside the transaction.
	injected := c.himportInjectedCmds(ctx, cn, cmds)

	if err := cn.WithWriter(c.context(ctx), c.opt.WriteTimeout, func(wr *proto.Writer) error {
		for _, ic := range injected {
			if err := writeCmd(wr, ic); err != nil {
				return err
			}
		}
		return writeCmds(wr, cmds)
	}); err != nil {
		setCmdsErr(cmds, err)
		return true, err
	}

	if err := cn.WithReader(c.context(ctx), c.opt.ReadTimeout, func(rd *proto.Reader) error {
		if err := c.himportReadInjectedReplies(ctx, cn, rd, injected); err != nil {
			// Transport error with every transaction reply unread: stamp
			// the batch like a write failure (see pipelineProcessCmds).
			setCmdsErr(cmds, err)
			return err
		}

		statusCmd := cmds[0].(*StatusCmd)
		// Trim multi and exec.
		trimmedCmds := cmds[1 : len(cmds)-1]

		if err := c.txPipelineReadQueued(ctx, cn, rd, statusCmd, trimmedCmds); err != nil {
			setCmdsErr(cmds, err)
			return err
		}

		// Read replies.
		err := c.pipelineReadCmds(ctx, cn, rd, trimmedCmds)
		if err == nil || isRedisError(err) {
			c.himportAfterBatch(cn, injected, trimmedCmds)
		}
		return err
	}); err != nil {
		return false, err
	}

	return false, nil
}

// txPipelineReadQueued reads queued replies from the Redis server.
// It returns an error if the server returns an error or if the number of replies does not match the number of commands.
func (c *baseClient) txPipelineReadQueued(ctx context.Context, cn *pool.Conn, rd *proto.Reader, statusCmd *StatusCmd, cmds []Cmder) error {
	// To be sure there are no buffered push notifications, we process them before reading the reply
	if err := c.processPendingPushNotificationWithReader(ctx, cn, rd); err != nil {
		internal.Logger.Printf(ctx, "push: error processing pending notifications before reading reply: %v", err)
	}
	// Parse +OK.
	if err := statusCmd.readReply(rd); err != nil {
		return err
	}

	// Parse +QUEUED.
	for _, cmd := range cmds {
		// To be sure there are no buffered push notifications, we process them before reading the reply
		if err := c.processPendingPushNotificationWithReader(ctx, cn, rd); err != nil {
			internal.Logger.Printf(ctx, "push: error processing pending notifications before reading reply: %v", err)
		}
		if err := statusCmd.readReply(rd); err != nil {
			cmd.SetErr(err)
			if !isRedisError(err) {
				return err
			}
		}
	}

	// To be sure there are no buffered push notifications, we process them before reading the reply
	if err := c.processPendingPushNotificationWithReader(ctx, cn, rd); err != nil {
		internal.Logger.Printf(ctx, "push: error processing pending notifications before reading reply: %v", err)
	}
	// Parse number of replies.
	line, err := rd.ReadLine()
	if err != nil {
		if err == Nil {
			err = TxFailedErr
		}
		return err
	}

	if line[0] != proto.RespArray {
		return fmt.Errorf("redis: expected '*', but got line %q", line)
	}

	return nil
}

//------------------------------------------------------------------------------

// Client is a Redis client representing a pool of zero or more underlying connections.
// It's safe for concurrent use by multiple goroutines.
//
// Client creates and frees connections automatically; it also maintains a free pool
// of idle connections. You can control the pool size with Config.PoolSize option.
type Client struct {
	*baseClient
	cmdable

	// cscLifecycleOwner keeps the canonical Client wrapper (the one whose GC
	// cleanup owns the drainer) reachable while a WithTimeout clone can still
	// serve from its cache. Nil on the canonical wrapper and on non-CSC clones.
	cscLifecycleOwner *Client

	autopipelinerMu     *sync.Mutex    // guards the autopipeliner fields against concurrent first-call creation
	autopipeliner       *AutoPipeliner // blocking face (Client.AutoPipeline)
	asyncAutopipeliner  *AutoPipeliner // deferred face (Client.AsyncAutoPipeline)
	autopipelinerClosed bool           // set by Close: refuse to resurrect a pipeliner on a closed client
}

// NewClient returns a client to the Redis Server specified by Options.
// Passing nil Options will cause a panic.
func NewClient(opt *Options) *Client {
	if opt == nil {
		panic("redis: NewClient nil options")
	}
	// clone to not share options with the caller
	opt = opt.clone()
	opt.init()

	// Push notifications are always enabled for RESP3 (cannot be disabled)

	c := Client{
		baseClient: &baseClient{
			apClosed: &atomic.Bool{},
			opt:      opt,
			onClose:  &onCloseHooks{},
			himport:  newHImportRegistry(),
		},
	}
	c.init()

	// Initialize push notification processor using shared helper
	// Use void processor for RESP2 connections (push notifications not available)
	c.pushProcessor = initializePushProcessor(opt)
	// set opt push processor for child clients
	c.opt.PushNotificationProcessor = c.pushProcessor

	// Generate unique pool names for metrics
	uniqueID := generateUniqueID()
	mainPoolName := opt.Addr + "_" + uniqueID
	pubsubPoolName := opt.Addr + "_" + uniqueID + "_pubsub"

	// Create connection pools
	var err error
	c.connPool, err = newConnPool(opt, c.dialHook, mainPoolName)
	if err != nil {
		panic(fmt.Errorf("redis: failed to create connection pool: %w", err))
	}
	c.pubSubPool, err = newPubSubPool(opt, c.dialHook, pubsubPoolName)
	if err != nil {
		panic(fmt.Errorf("redis: failed to create pubsub pool: %w", err))
	}

	// Optionally create a separate connection pool for pipelining, with its own
	// (typically larger) buffers, so pipelines can use big buffers without
	// bloating the main pool. Enabled when either pipeline buffer size is set.
	if opt.PipelineReadBufferSize > 0 || opt.PipelineWriteBufferSize > 0 {
		pipelineOpt := opt.clone()
		if opt.PipelineReadBufferSize > 0 {
			pipelineOpt.ReadBufferSize = opt.PipelineReadBufferSize
			// Same clamp Options.init applies to the main pool: RESP3 push
			// parsing needs a minimum read buffer, and a tiny pipeline reader
			// would break push-notification handling on pipeline conns.
			if pipelineOpt.Protocol == 3 && pipelineOpt.ReadBufferSize < proto.MinRESP3ReadBufferSize {
				pipelineOpt.ReadBufferSize = proto.MinRESP3ReadBufferSize
			}
		}
		if opt.PipelineWriteBufferSize > 0 {
			pipelineOpt.WriteBufferSize = opt.PipelineWriteBufferSize
		}
		if opt.PipelinePoolSize > 0 {
			pipelineOpt.PoolSize = opt.PipelinePoolSize
		} else {
			pipelineOpt.PoolSize = 10 // default smaller pool for pipelining
		}
		pipelinePoolName := opt.Addr + "_" + uniqueID + "_pipeline"
		c.pipelinePoolName = pipelinePoolName
		c.pipelinePool, err = newConnPool(pipelineOpt, c.dialHook, pipelinePoolName)
		if err != nil {
			panic(fmt.Errorf("redis: failed to create pipeline connection pool: %w", err))
		}
	}

	if opt.StreamingCredentialsProvider != nil {
		c.streamingCredentialsManager = streaming.NewManager(c.connPool, c.opt.PoolTimeout)
		c.connPool.AddPoolHook(c.streamingCredentialsManager.PoolHook())
		if c.pipelinePool != nil {
			c.pipelinePool.AddPoolHook(c.streamingCredentialsManager.PoolHook())
		}
	}

	// CSC wiring (SharedTracking): shared cache + per-connection CLIENT TRACKING +
	// background drainer. attachCSC is the strategy dispatch entry.
	if opt.Protocol == 3 {
		var cache Cache
		if explicit := opt.ClientSideCache; explicit != nil {
			cache = explicit
		} else if cfg := opt.ClientSideCacheConfig; cfg != nil {
			cache = NewLocalCache(*cfg)
			// We constructed it, so we own it (may flush on drainer stop).
			c.baseClient.cscOwnsCache = true
		}
		c.baseClient.attachCSC(context.Background(), cache)

		// Safety net for a client dropped without Close: the goroutines hold
		// *baseClient (never *Client), so dropping *Client (returned as &c)
		// triggers these cleanups, which stop them. See cscRegisterCleanups.
		cscRegisterCleanups(&c)
	}

	// Initialize maintnotifications first if enabled and protocol is RESP3
	if opt.MaintNotificationsConfig != nil && opt.MaintNotificationsConfig.Mode != maintnotifications.ModeDisabled && opt.Protocol == 3 {
		err := c.enableMaintNotificationsUpgrades()
		if err != nil {
			internal.Logger.Printf(context.Background(), "failed to initialize maintnotifications: %v", err)
			if opt.MaintNotificationsConfig.Mode == maintnotifications.ModeEnabled {
				/*
					Design decision: panic here to fail fast if maintnotifications cannot be enabled when explicitly requested.
					We choose to panic instead of returning an error to avoid breaking the existing client API, which does not expect
					an error from NewClient. This ensures that misconfiguration or critical initialization failures are surfaced
					immediately, rather than allowing the client to continue in a partially initialized or inconsistent state.
					Clients relying on maintnotifications should be aware that initialization errors will cause a panic, and should
					handle this accordingly (e.g., via recover or by validating configuration before calling NewClient).
					This approach is only used when MaintNotificationsConfig.Mode is MaintNotificationsEnabled, indicating that maintnotifications
					upgrades are required for correct operation. In other modes, initialization failures are logged but do not panic.
				*/
				panic(fmt.Errorf("failed to enable maintnotifications: %w", err))
			}
		}
	}

	// Register pools with OTel recorder if it supports pool registration
	// This allows async gauge metrics to pull stats from pools periodically
	otel.RegisterPools(c.connPool, c.pubSubPool, c.pipelinePool, opt.Addr)

	return &c
}

func (c *Client) init() {
	// Fresh per-Client guard and no inherited autopipeliner: a WithTimeout clone
	// (clone := *c) must not share the parent's mutex or AutoPipeliner instance.
	c.autopipelinerMu = &sync.Mutex{}
	c.autopipeliner = nil
	c.asyncAutopipeliner = nil
	c.cmdable = c.Process
	c.initHooks(hooks{
		dial:       c.baseClient.dial,
		process:    c.baseClient.process,
		pipeline:   c.baseClient.processPipeline,
		txPipeline: c.baseClient.processTxPipeline,
	})
}

// WithTimeout returns a clone sharing the parent's connection pools with the
// given read/write timeout. The clone caches its own autopipeliners: an
// AutoPipeline()/AsyncAutoPipeline() created on the clone is NOT stopped by
// the parent's Close — call Close on the clone's autopipeliner explicitly.
func (c *Client) WithTimeout(timeout time.Duration) *Client {
	// Snapshot under the guard: AutoPipeline()/Close() mutate the
	// autopipeliner fields concurrently, so a bare struct copy of them is a
	// data race (init below discards the copied values either way).
	c.autopipelinerMu.Lock()
	clone := *c
	c.autopipelinerMu.Unlock()
	if c.cscLifecycleOwner != nil {
		clone.cscLifecycleOwner = c.cscLifecycleOwner
	} else if c.baseClient.cscDrainHandle != nil {
		clone.cscLifecycleOwner = c
	}
	clone.baseClient = c.baseClient.withTimeout(timeout)
	clone.init()
	return &clone
}

// Close closes the client, stopping both cached autopipeliners (the blocking
// AutoPipeline instance and the async AsyncAutoPipeline instance, if created)
// before releasing the underlying resources, so their background flusher
// goroutines don't outlive the client. AutoPipeliner.Close is idempotent and
// safe to call here even if autopipelining was never used.
// A WithTimeout clone delegates CSC teardown to the canonical wrapper that
// owns the background drainer.
func (c *Client) Close() error {
	c.autopipelinerMu.Lock()
	ap, async := c.autopipeliner, c.asyncAutopipeliner
	c.autopipeliner, c.asyncAutopipeliner = nil, nil
	// A later AutoPipeline()/AsyncAutoPipeline() call must not build a fresh
	// pipeliner against the closed pools: nothing would ever close it and its
	// flusher goroutines would leak. The getters check this flag.
	c.autopipelinerClosed = true
	c.autopipelinerMu.Unlock()
	var firstErr error
	for _, p := range []*AutoPipeliner{ap, async} {
		if p != nil {
			if err := p.Close(); err != nil && firstErr == nil {
				firstErr = err
			}
		}
	}
	if c.cscLifecycleOwner != nil {
		// Delegate through the OWNER's *Client.Close, not its baseClient:
		// the owner may hold cached autopipeliners of its own whose flusher
		// goroutines must stop with the shared pools, and its
		// autopipelinerClosed flag must flip so later owner getters cannot
		// resurrect a pipeliner against closed pools. Client.Close is
		// idempotent through baseClient.Close, so an owner also closed
		// directly is fine.
		if err := c.cscLifecycleOwner.Close(); err != nil && firstErr == nil {
			firstErr = err
		}
		return firstErr
	}
	if err := c.baseClient.Close(); err != nil && firstErr == nil {
		firstErr = err
	}
	return firstErr
}

func (c *Client) Conn() *Conn {
	// Share the HIMPORT fieldset registry: the sticky pool borrows
	// connections from this client's pool, so fieldsets prepared on them
	// stay valid after the connections are returned.
	conn := newConn(c.opt, c.baseClient.newStickyConnPool(), &c.hooksMixin, c.himport)
	// A sticky client does not serve cache hits, but a new pool connection first
	// initialized through it may later be reused by the parent. Share the
	// successful-attachment signal so that connection is tracked exactly when
	// the parent's CSC is active.
	conn.baseClient.cscActive = c.baseClient.cscActive
	// No-op today: the strategy needs an idle-conn drainer and a StickyConnPool
	// has none, so CSC isn't active on a Conn() (its reads hit the server). Kept
	// so a future sticky-pool-capable strategy attaches here.
	conn.baseClient.attachCSC(context.Background(), c.csc)
	// Carry the parent's shared eviction hook so that if this derived client
	// initializes a pool conn, the close hook it installs still evicts from the
	// parent cache (its own csc is nil).
	conn.baseClient.cscPoolHook = c.baseClient.cscPoolHook
	return conn
}

func (c *Client) Process(ctx context.Context, cmd Cmder) error {
	err := c.processHook(ctx, cmd)
	cmd.SetErr(err)
	return err
}

// Options returns read-only *Options that were used to create the client.
// Any alteration of the returned *Options may result in undefined behaviour.
func (c *Client) Options() *Options {
	return c.opt
}

// NodeAddress returns the address of the Redis node as reported by the server.
// For cluster clients, this is the endpoint from CLUSTER SLOTS before any transformation
// (e.g., loopback replacement). For standalone clients, this defaults to Addr.
//
// This is useful for matching the source field in maintenance notifications
// (e.g. SMIGRATED).
func (c *Client) NodeAddress() string {
	return c.opt.NodeAddress
}

// GetMaintNotificationsManager returns the maintnotifications manager instance for monitoring and control.
// Returns nil if maintnotifications are not enabled.
func (c *Client) GetMaintNotificationsManager() *maintnotifications.Manager {
	c.maintNotificationsManagerLock.RLock()
	defer c.maintNotificationsManagerLock.RUnlock()
	return c.maintNotificationsManager
}

// initializePushProcessor initializes the push notification processor for any client type.
// This is a shared helper to avoid duplication across NewClient, NewFailoverClient, and NewSentinelClient.
func initializePushProcessor(opt *Options) push.NotificationProcessor {
	// Always use custom processor if provided
	if opt.PushNotificationProcessor != nil {
		return opt.PushNotificationProcessor
	}

	// Push notifications are always enabled for RESP3, disabled for RESP2
	if opt.Protocol == 3 {
		// Create default processor for RESP3 connections
		return NewPushNotificationProcessor()
	}

	// Create void processor for RESP2 connections (push notifications not available)
	return NewVoidPushNotificationProcessor()
}

// RegisterPushNotificationHandler registers a handler for a specific push notification name.
// Returns an error if a handler is already registered for this push notification name.
// If protected is true, the handler cannot be unregistered.
func (c *Client) RegisterPushNotificationHandler(pushNotificationName string, handler push.NotificationHandler, protected bool) error {
	return c.pushProcessor.RegisterHandler(pushNotificationName, handler, protected)
}

// GetPushNotificationHandler returns the handler for a specific push notification name.
// Returns nil if no handler is registered for the given name.
func (c *Client) GetPushNotificationHandler(pushNotificationName string) push.NotificationHandler {
	return c.pushProcessor.GetHandler(pushNotificationName)
}

type PoolStats pool.Stats

// PoolStats returns connection pool stats.
func (c *Client) PoolStats() *PoolStats {
	stats := c.connPool.Stats()
	stats.PubSubStats = *c.pubSubPool.Stats()
	if c.pipelinePool != nil {
		stats.PipelineStats = c.pipelinePool.Stats()
	}
	return (*PoolStats)(stats)
}

func (c *Client) Pipelined(ctx context.Context, fn func(Pipeliner) error) ([]Cmder, error) {
	return c.Pipeline().Pipelined(ctx, fn)
}

func (c *Client) Pipeline() Pipeliner {
	pipe := Pipeline{
		exec: pipelineExecer(c.processPipelineHook),
	}
	pipe.init()
	return &pipe
}

// AutoPipeline returns the blocking autopipeliner for this client: a drop-in
// replacement for the normal command surface where each command call (ap.Set,
// ap.Get, ...) blocks until executed, exactly like a plain client — but the
// engine batches concurrent callers' commands into pipelines, so throughput is
// far higher (measured locally over loopback: ~1M+ SET/sec vs ~100k; indicative, not a guarantee). Commands keep per-goroutine order.
//
// By default, Options.AutoPipelineOptions is used if set,
// otherwise DefaultBlockingAutoPipelineOptions (a single ordered batch stream,
// which maximizes throughput and minimizes latency for the blocking face — see
// its doc). The instance is cached and shared; the first
// call's config wins and later calls return the same instance until it is closed.
// It must be closed (or close the client) to release its goroutines.
//
// It returns an error if the supplied config is invalid (e.g. MaxConcurrentBatches>1
// without Unordered, or a negative size); on error no instance is cached.
//
// EXPERIMENTAL: this API is subject to change, use with caution.
func (c *Client) AutoPipeline() (*AutoPipeliner, error) {
	return c.AutoPipelineWithOptions(nil)
}

// AutoPipelineWithOptions is AutoPipeline with explicit options instead of
// Options.AutoPipelineOptions / the default. The instance is cached and shared;
// the first call's config wins.
//
// EXPERIMENTAL: this API is subject to change, use with caution.
func (c *Client) AutoPipelineWithOptions(config *AutoPipelineOptions) (*AutoPipeliner, error) {
	return getOrCreateAutoPipeliner(c.autopipelinerMu, &c.autopipeliner, &c.autopipelinerClosed, c.baseClient.apClosed, config,
		func() *AutoPipelineOptions {
			if c.opt.AutoPipelineOptions != nil {
				return c.opt.AutoPipelineOptions
			}
			return DefaultBlockingAutoPipelineOptions()
		},
		func(cfg *AutoPipelineOptions) (*AutoPipeliner, error) { return newAutoPipeliner(c, cfg, true) })
}

// AsyncAutoPipeline returns the deferred (async) autopipeliner: command calls
// return immediately and the result accessors (Val/Result/Err) block until the
// command has executed. Submit a window of commands, then read their results, to
// keep each pipeline deep and reach the highest throughput (measured locally over loopback: ~2-3M SET/sec; indicative).
//
// By default, Options.AutoPipelineOptions is used if set,
// otherwise DefaultAutoPipelineOptions (ordered, MaxConcurrentBatches: 1) — a
// single goroutine's deferred commands execute in submit order. Use AsyncAutoPipelineWithOptions
// to override (and, for parallel batches, set Unordered). The instance is
// cached and shared; the first call's config wins. Close it (or the client) to
// release its goroutines.
//
// It returns an error if the supplied config is invalid (e.g. MaxConcurrentBatches>1
// without Unordered, or a negative size); on error no instance is cached.
//
// EXPERIMENTAL: this API is subject to change, use with caution.
func (c *Client) AsyncAutoPipeline() (*AutoPipeliner, error) {
	return c.AsyncAutoPipelineWithOptions(nil)
}

// AsyncAutoPipelineWithOptions is AsyncAutoPipeline with an explicit config
// instead of Options.AutoPipelineOptions / the default. The instance is cached
// and shared; the first call's config wins.
//
// EXPERIMENTAL: this API is subject to change, use with caution.
func (c *Client) AsyncAutoPipelineWithOptions(config *AutoPipelineOptions) (*AutoPipeliner, error) {
	return getOrCreateAutoPipeliner(c.autopipelinerMu, &c.asyncAutopipeliner, &c.autopipelinerClosed, c.baseClient.apClosed, config,
		func() *AutoPipelineOptions {
			if c.opt.AutoPipelineOptions != nil {
				return c.opt.AutoPipelineOptions
			}
			return DefaultAutoPipelineOptions()
		},
		func(cfg *AutoPipelineOptions) (*AutoPipeliner, error) { return newAutoPipeliner(c, cfg, false) })
}

func (c *Client) TxPipelined(ctx context.Context, fn func(Pipeliner) error) ([]Cmder, error) {
	return c.TxPipeline().Pipelined(ctx, fn)
}

// TxPipeline acts like Pipeline, but wraps queued commands with MULTI/EXEC.
func (c *Client) TxPipeline() Pipeliner {
	pipe := Pipeline{
		exec: func(ctx context.Context, cmds []Cmder) error {
			cmds = wrapMultiExec(ctx, cmds)
			return c.processTxPipelineHook(ctx, cmds)
		},
	}
	pipe.init()
	return &pipe
}

func (c *Client) pubSub() *PubSub {
	pubsub := &PubSub{
		opt: c.cloneOpt(),
		newConn: func(ctx context.Context, addr string, channels []string) (*pool.Conn, error) {
			cn, err := c.pubSubPool.NewConn(ctx, c.opt.Network, addr, channels)
			if err != nil {
				return nil, err
			}
			// will return nil if already initialized
			err = c.initConn(ctx, cn)
			if err != nil {
				_ = cn.Close()
				return nil, err
			}
			// Track connection in PubSubPool
			c.pubSubPool.TrackConn(cn)
			return cn, nil
		},
		closeConn: func(cn *pool.Conn) error {
			// Untrack connection from PubSubPool
			c.pubSubPool.UntrackConn(cn)
			_ = cn.Close()
			return nil
		},
		pushProcessor: c.pushProcessor,
	}
	pubsub.init()

	return pubsub
}

// Subscribe subscribes the client to the specified channels.
// Channels can be omitted to create empty subscription.
// Note that this method does not wait on a response from Redis, so the
// subscription may not be active immediately. To force the connection to wait,
// you may call the Receive() method on the returned *PubSub like so:
//
//	sub := client.Subscribe(queryResp)
//	iface, err := sub.Receive()
//	if err != nil {
//	    // handle error
//	}
//
//	// Should be *Subscription, but others are possible if other actions have been
//	// taken on sub since it was created.
//	switch iface.(type) {
//	case *Subscription:
//	    // subscribe succeeded
//	case *Message:
//	    // received first message
//	case *Pong:
//	    // pong received
//	default:
//	    // handle error
//	}
//
//	ch := sub.Channel()
func (c *Client) Subscribe(ctx context.Context, channels ...string) *PubSub {
	pubsub := c.pubSub()
	if len(channels) > 0 {
		_ = pubsub.Subscribe(ctx, channels...)
	}
	return pubsub
}

// PSubscribe subscribes the client to the given patterns.
// Patterns can be omitted to create empty subscription.
func (c *Client) PSubscribe(ctx context.Context, channels ...string) *PubSub {
	pubsub := c.pubSub()
	if len(channels) > 0 {
		_ = pubsub.PSubscribe(ctx, channels...)
	}
	return pubsub
}

// SSubscribe Subscribes the client to the specified shard channels.
// Channels can be omitted to create empty subscription.
func (c *Client) SSubscribe(ctx context.Context, channels ...string) *PubSub {
	pubsub := c.pubSub()
	if len(channels) > 0 {
		_ = pubsub.SSubscribe(ctx, channels...)
	}
	return pubsub
}

//------------------------------------------------------------------------------

// Conn represents a single Redis connection rather than a pool of connections.
// Prefer running commands from Client unless there is a specific need
// for a continuous single Redis connection.
type Conn struct {
	baseClient
	cmdable
	statefulCmdable
}

// newConn is a helper func to create a new Conn instance.
// The Conn instance is not thread-safe and should not be shared between goroutines.
// The parentHooks will be cloned, no need to clone before passing it.
// himport is the HIMPORT fieldset registry the Conn participates in — pass
// the owning client's registry (a private one would restart versions at 1
// and collide with the client's version space on the shared pooled
// connections); nil disables HIMPORT tracking.
func newConn(opt *Options, connPool pool.Pooler, parentHooks *hooksMixin, himport *himportRegistry) *Conn {
	c := Conn{
		baseClient: baseClient{
			apClosed: &atomic.Bool{},
			opt:      opt,
			connPool: connPool,
			onClose:  &onCloseHooks{},
			himport:  himport,
		},
	}

	if parentHooks != nil {
		c.hooksMixin = parentHooks.clone()
	}

	// Initialize push notification processor using shared helper
	// Use void processor for RESP2 connections (push notifications not available)
	c.pushProcessor = initializePushProcessor(opt)

	c.cmdable = c.Process
	c.statefulCmdable = c.Process
	c.initHooks(hooks{
		dial:       c.baseClient.dial,
		process:    c.baseClient.process,
		pipeline:   c.baseClient.processPipeline,
		txPipeline: c.baseClient.processTxPipeline,
	})

	return &c
}

func (c *Conn) Process(ctx context.Context, cmd Cmder) error {
	err := c.processHook(ctx, cmd)
	cmd.SetErr(err)
	return err
}

// RegisterPushNotificationHandler registers a handler for a specific push notification name.
// Returns an error if a handler is already registered for this push notification name.
// If protected is true, the handler cannot be unregistered.
func (c *Conn) RegisterPushNotificationHandler(pushNotificationName string, handler push.NotificationHandler, protected bool) error {
	return c.pushProcessor.RegisterHandler(pushNotificationName, handler, protected)
}

func (c *Conn) Pipelined(ctx context.Context, fn func(Pipeliner) error) ([]Cmder, error) {
	return c.Pipeline().Pipelined(ctx, fn)
}

func (c *Conn) Pipeline() Pipeliner {
	pipe := Pipeline{
		exec: c.processPipelineHook,
	}
	pipe.init()
	return &pipe
}

func (c *Conn) TxPipelined(ctx context.Context, fn func(Pipeliner) error) ([]Cmder, error) {
	return c.TxPipeline().Pipelined(ctx, fn)
}

// TxPipeline acts like Pipeline, but wraps queued commands with MULTI/EXEC.
func (c *Conn) TxPipeline() Pipeliner {
	pipe := Pipeline{
		exec: func(ctx context.Context, cmds []Cmder) error {
			cmds = wrapMultiExec(ctx, cmds)
			return c.processTxPipelineHook(ctx, cmds)
		},
	}
	pipe.init()
	return &pipe
}

// processPushNotifications processes all pending push notifications on a connection
// This ensures that cluster topology changes are handled immediately before the connection is used
// This method should be called by the client before using WithReader for command execution
//
// Performance optimization: Skip the expensive MaybeHasData() syscall if a health check
// was performed recently (within 5 seconds). The health check already verified the connection
// is healthy and checked for unexpected data (push notifications).
func (c *baseClient) processPushNotifications(ctx context.Context, cn *pool.Conn) error {
	// Only process push notifications for RESP3 connections with a processor
	if c.opt.Protocol != 3 || c.pushProcessor == nil {
		return nil
	}

	// Performance optimization: Skip MaybeHasData() syscall if health check was recent
	// If the connection was health-checked within the last 5 seconds, we can skip the
	// expensive syscall since the health check already verified no unexpected data.
	// This is safe because:
	// 0. lastHealthCheckNs is set in pool/conn.go:putConn() after a successful health check
	// 1. Health check (connCheck) uses the same syscall (Recvfrom with MSG_PEEK)
	// 2. If push notifications arrived, they would have been detected by health check
	// 3. 5 seconds is short enough that connection state is still fresh
	// 4. Push notifications will be processed by the next WithReader call
	// used it is set on getConn, so we should use another timer (lastPutAt?)
	lastHealthCheckNs := cn.LastPutAtNs()
	if lastHealthCheckNs > 0 {
		// Use pool's cached time to avoid expensive time.Now() syscall
		nowNs := pool.GetCachedTimeNs()
		if nowNs-lastHealthCheckNs < int64(5*time.Second) {
			// Recent health check confirmed no unexpected data, skip the syscall
			return nil
		}
	}

	return c.peekAndProcessPushNotifications(ctx, cn)
}

// peekAndProcessPushNotifications peeks the socket and processes any pending
// push notifications on cn unconditionally, bypassing the recent-health-check
// shortcut in processPushNotifications. Required on paths that do not follow
// up with a reply read on the same connection (e.g. the CSC cache-hit drain),
// where the shortcut would otherwise suppress invalidations buffered since the
// last health check.
func (c *baseClient) peekAndProcessPushNotifications(ctx context.Context, cn *pool.Conn) error {
	if c.opt.Protocol != 3 || c.pushProcessor == nil {
		return nil
	}

	if !cn.MaybeHasData() {
		return nil
	}

	// Short read timeout: MaybeHasData confirmed kernel-buffered bytes, so
	// the first read returns immediately — the deadline only needs to cover
	// scheduler pauses, not network waits. 10us was routinely lost to
	// scheduling on loaded machines: the peek then timed out with nothing
	// consumed, the processor treated that as "no pending data", and a
	// connection with buffered push bytes was returned to the pool instead
	// of being drained (or removed, when the frame turns out partial).
	return cn.WithReader(ctx, time.Millisecond, func(rd *proto.Reader) error {
		handlerCtx := c.pushNotificationHandlerContext(cn)
		return c.pushProcessor.ProcessPendingNotifications(ctx, handlerCtx, rd)
	})
}

// cscFallbackProbeInterval bounds how often an idle connection without a
// portable readiness mechanism is subjected to a timed read. Post-command
// probes remain immediate; this is only the eventual invalidation fallback.
const cscFallbackProbeInterval = 100 * time.Millisecond

// drainPushNotifications drains push frames buffered on a connection the CSC
// drainer has claimed, under a HARD read deadline. processorSucceeded reports a
// successful processor invocation; it resets custom-processor damping even when
// the frame was hidden inside a transport wrapper. A non-nil error is
// connection-fatal (the drainer removes the conn), including a read timeout
// after reply consumption starts: the reader may be desynchronized. A custom
// processor's error is also fatal because its contract cannot prove no bytes
// were consumed.
func (c *baseClient) drainPushNotifications(cn *pool.Conn) (processorSucceeded bool, err error) {
	if c.opt.Protocol != 3 || c.pushProcessor == nil {
		return false, nil
	}
	// Skip only when nothing is buffered (reader) AND nothing on the socket:
	// MaybeHasData peeks only the socket, but an invalidate can sit in cn.rd.
	readPending := cn.TakeCscReadPending()
	periodicReadPending := cn.TakeCscPeriodicReadPending(cscFallbackProbeInterval)
	socketData, socketErr := cn.CheckForData()
	if socketErr != nil {
		return false, socketErr
	}
	hasData := cn.HasBufferedData() || socketData
	if !readPending && !periodicReadPending && !hasData {
		return false, nil
	}
	if !hasData {
		// TLS and opaque wrappers can hide bytes from the socket readiness
		// check. Probe one byte without consuming it under a tiny deadline;
		// only a confirmed byte gets the longer fragmented-frame budget below.
		err := cn.WithReaderHardDeadline(cscDrainProbeReadCap, func(rd *proto.Reader) error {
			_, err := rd.Peek(1)
			return err
		})
		if err != nil {
			if isTimeout, hasTimeoutFlag := isTimeoutError(err); isTimeout && hasTimeoutFlag {
				return false, nil
			}
			return false, err
		}
	}

	handlerCtx := c.pushNotificationHandlerContext(cn)
	handlerCtx.Client = cscHandlerClient{baseClient: c}
	err = cn.WithReaderHardDeadline(cscDrainHardReadCap, func(rd *proto.Reader) error {
		if processor, ok := c.pushProcessor.(*push.Processor); ok {
			return processor.ProcessPendingNotificationsBuffered(
				context.Background(), handlerCtx, rd)
		}
		return c.pushProcessor.ProcessPendingNotifications(context.Background(), handlerCtx, rd)
	})
	if err != nil {
		// The built-in processor surfaces mid-frame ReadReply errors (a benign
		// boundary peek timeout returns nil). allowTimeout=false: such an error
		// means bytes were consumed mid-frame, leaving the conn desynced —
		// re-pooling would corrupt the next command's reply, so remove it.
		if _, builtin := c.pushProcessor.(*push.Processor); builtin {
			if isBadConn(err, false, c.opt.Addr) {
				return true, err // fatal read/protocol/connection error — remove the conn
			}
			return true, nil
		}
		// A CUSTOM processor's error contract is unknown: it may have consumed
		// part of a frame before failing, and a mid-frame reader silently
		// corrupts the next command's reply. The conn is idle and held solely
		// by the drainer, so the safe default — removal — costs one reconnect;
		// persistent failures are damped by the drainer (cscDrainCustomErrCap).
		internal.Logger.Printf(context.Background(), "csc: drain: custom push processor error (removing conn): %v", err)
		return true, err
	}
	// The processor ran successfully. This is stronger evidence than a clean
	// connection on which it was never invoked, and prevents successful TLS-
	// buffered drains from being counted as if failures were consecutive.
	return true, nil
}

// processPendingPushNotificationWithReader processes all pending push notifications on a connection
// This method should be called by the client in WithReader before reading the reply
func (c *baseClient) processPendingPushNotificationWithReader(ctx context.Context, cn *pool.Conn, rd *proto.Reader) error {
	// if we have the reader, we don't need to check for data on the socket, we are waiting
	// for either a reply or a push notification, so we can block until we get a reply or reach the timeout
	if c.opt.Protocol != 3 || c.pushProcessor == nil {
		return nil
	}

	// Create handler context with client, connection pool, and connection information
	handlerCtx := c.pushNotificationHandlerContext(cn)
	return c.pushProcessor.ProcessPendingNotifications(ctx, handlerCtx, rd)
}

// pushNotificationHandlerContext creates a handler context for push notification processing
func (c *baseClient) pushNotificationHandlerContext(cn *pool.Conn) push.NotificationHandlerContext {
	return push.NotificationHandlerContext{
		Client:   c,
		ConnPool: c.connPool,
		Conn:     cn, // Wrap in adapter for easier interface access
	}
}
