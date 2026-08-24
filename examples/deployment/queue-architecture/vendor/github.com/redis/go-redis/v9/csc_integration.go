package redis

import (
	"bytes"
	"context"
	"errors"
	"reflect"
	"runtime"
	"strconv"
	"sync"
	"sync/atomic"
	"time"

	"github.com/redis/go-redis/v9/internal"
	"github.com/redis/go-redis/v9/internal/pool"
	"github.com/redis/go-redis/v9/internal/proto"
	"github.com/redis/go-redis/v9/push"
)

// cscRegisterCleanups arranges for a client dropped without Close to stop its
// background CSC drainer. The drainer's exit path revokes its pool's cache
// coverage; the runtime cleanup itself stays non-blocking and never captures
// *Client, so the wrapper remains collectible.
func cscRegisterCleanups(c *Client) {
	h := c.baseClient.cscDrainHandle
	if h == nil {
		return
	}
	// Capture cscActive (a standalone *atomic.Bool, not *Client) so the cleanup
	// also stops clones from serving once the drainer is gone.
	active := c.baseClient.cscActive
	runtime.AddCleanup(c, func(h *cscDrainHandle) {
		if active != nil {
			active.Store(false)
		}
		h.signalStop()
	}, h)
}

// ClientSideCacheConfig configures the built-in client-side cache. Pass a
// non-nil value to Options.ClientSideCacheConfig to enable caching on a RESP3
// client.
//
// Experimental: this API may change in a minor release.
type ClientSideCacheConfig = CacheConfig

const (
	invalidatePushName = "invalidate"
	// cscNamespaceSep separates fixed-width/logically-delimited namespace parts
	// from the command or Redis key.
	cscNamespaceSep = "\x00"
)

// cscNamespacePrefix scopes a shared cache by database and fixed ACL identity.
// Password rotation does not change identity; provider-backed identities are
// rejected before attachment.
func cscNamespacePrefix(db int, username string) string {
	return strconv.Itoa(db) + cscNamespaceSep +
		strconv.Itoa(len(username)) + ":" + username + cscNamespaceSep
}

func cscNamespacedKey(prefix, key string) string {
	return prefix + key
}

// invalidateHandler propagates RESP3 "invalidate" push notifications into the
// shared client-side cache. keyPrefix scopes incoming key names so a shared
// cache cannot collide across databases or fixed ACL identities.
//
// The binding (cache, keyPrefix) is mutable under mu: the owning client's teardown
// RELEASES it (cache=nil) instead of unregistering the handler, so the handler
// can stay registered protected — application code holding the processor
// cannot silently unregister invalidation out from under a live client — while
// a successor client on the same processor can still rebind it (see
// registerInvalidateHandler).
type invalidateHandler struct {
	mu        sync.RWMutex
	cache     Cache
	keyPrefix string
	users     int
}

// HandlePushNotification decodes ["invalidate", <keys>] notifications. A nil
// <keys> payload is emitted on FLUSHDB/FLUSHALL and triggers a full cache flush.
func (h *invalidateHandler) HandlePushNotification(
	_ context.Context, _ push.NotificationHandlerContext, notification []interface{},
) error {
	h.mu.RLock()
	cache, keyPrefix := h.cache, h.keyPrefix
	h.mu.RUnlock()
	if cache == nil || len(notification) < 2 {
		return nil
	}

	switch payload := notification[1].(type) {
	case nil:
		cache.Flush()
	case []interface{}:
		for _, k := range payload {
			var name string
			switch v := k.(type) {
			case string:
				name = v
			case []byte:
				name = string(v)
			default:
				continue
			}
			cache.DeleteByRedisKey(cscNamespacedKey(keyPrefix, name))
		}
	}
	return nil
}

func (h *invalidateHandler) release() {
	h.mu.Lock()
	if h.users > 0 {
		h.releaseLocked()
	}
	h.mu.Unlock()
}

func (h *invalidateHandler) releaseLocked() {
	h.users--
	if h.users == 0 {
		h.cache = nil
		h.keyPrefix = ""
	}
}

// sameCache compares Cache interface values without panicking when an
// implementation uses a non-comparable value type.
func sameCache(a, b Cache) bool {
	if a == nil || b == nil {
		return a == nil && b == nil
	}
	typ := reflect.TypeOf(a)
	return typ == reflect.TypeOf(b) && typ.Comparable() && a == b
}

func isNilCache(cache Cache) bool {
	if cache == nil {
		return true
	}
	v := reflect.ValueOf(cache)
	switch v.Kind() {
	case reflect.Chan, reflect.Func, reflect.Interface, reflect.Map, reflect.Ptr, reflect.Slice:
		return v.IsNil()
	default:
		return false
	}
}

// errInvalidateHandlerBound: piggybacking on a handler bound to a live
// different cache would leave the new cache uninvalidated.
var errInvalidateHandlerBound = errors.New(`csc: a different "invalidate" push handler is already registered`)

// bindTo binds the handler to (cache, keyPrefix). Success when that is already the
// binding (a derived Client.Conn sharing the parent's processor and cache) or
// when the handler was released by a previous owner's teardown (rebind);
// errInvalidateHandlerBound otherwise.
func (h *invalidateHandler) bindTo(cache Cache, keyPrefix string) error {
	h.mu.Lock()
	defer h.mu.Unlock()
	switch {
	case sameCache(h.cache, cache) && h.keyPrefix == keyPrefix:
		h.users++
		return nil
	case h.cache == nil:
		h.cache, h.keyPrefix = cache, keyPrefix
		h.users = 1
		return nil
	default:
		return errInvalidateHandlerBound
	}
}

// lookupInvalidateHandler returns the processor's CSC invalidate handler, nil
// when absent or foreign.
func lookupInvalidateHandler(p push.NotificationProcessor) *invalidateHandler {
	if p == nil {
		return nil
	}
	h, _ := p.GetHandler(invalidatePushName).(*invalidateHandler)
	return h
}

func registerInvalidateHandler(p push.NotificationProcessor, cache Cache, keyPrefix string) error {
	if p == nil || cache == nil {
		return nil
	}
	if existing := p.GetHandler(invalidatePushName); existing != nil {
		h, ok := existing.(*invalidateHandler)
		if !ok {
			return errInvalidateHandlerBound
		}
		return h.bindTo(cache, keyPrefix)
	}
	// VoidProcessor (RESP2) returns an error here; the caller treats it as
	// "CSC not available" rather than fatal. Registered PROTECTED: application
	// code holding the processor must not be able to unregister invalidation
	// under a live client (that would serve unbounded-stale hits with no
	// signal); owner teardown releases the BINDING instead of the handler.
	err := p.RegisterHandler(invalidatePushName, &invalidateHandler{
		cache:     cache,
		keyPrefix: keyPrefix,
		users:     1,
	}, true)
	if err == nil {
		return nil
	}
	// Another client can register the same protected handler between GetHandler
	// and RegisterHandler. Re-read it and accept the compatible binding.
	if existing := p.GetHandler(invalidatePushName); existing != nil {
		h, ok := existing.(*invalidateHandler)
		if !ok {
			return errInvalidateHandlerBound
		}
		return h.bindTo(cache, keyPrefix)
	}
	return err
}

// attachCSC dispatches to the invalidation strategy in
// Options.ClientSideCacheStrategy. Safe with a nil cache; on failure c.csc stays
// nil and commands fall back to normal round-trips. Adding a strategy: a new
// CSCStrategy constant plus cases in Options.init and here.
func (c *baseClient) attachCSC(ctx context.Context, cache Cache) {
	if isNilCache(cache) || c.opt.Protocol != 3 {
		return
	}
	// Credential providers may return a different ACL identity over the
	// client's lifetime (or per context/connection), while the cache namespace
	// is fixed when the client is created. Fixed credentials remain safe because
	// the ACL username is included in the length-delimited namespace below.
	if c.opt.StreamingCredentialsProvider != nil ||
		c.opt.CredentialsProviderContext != nil ||
		c.opt.CredentialsProvider != nil {
		internal.Logger.Printf(ctx,
			"redis: client-side caching is disabled with credential providers")
		return
	}
	c.cscKeyPrefix = cscNamespacePrefix(c.opt.DB, c.opt.Username)
	switch c.opt.ClientSideCacheStrategy {
	case CSCStrategySharedTracking:
		c.attachSharedTrackingCSC(ctx, cache)
	default:
		// Options.init clamps unknown strategies to SharedTracking; delegate anyway.
		c.attachSharedTrackingCSC(ctx, cache)
	}
}

// attachSharedTrackingCSC wires SharedTracking: one shared cache, per-conn CLIENT
// TRACKING, a background drainer, and the owning-conn eviction hook. DB-0 only:
// tracking is bound to the conn's DB and a runtime SELECT does not re-key it.
func (c *baseClient) attachSharedTrackingCSC(ctx context.Context, cache Cache) {
	if c.opt.DB != 0 {
		internal.Logger.Printf(ctx,
			"csc: client-side caching is restricted to DB 0; disabling CSC for client configured with DB=%d. "+
				"Use one client per DB if you need caching against non-zero databases.", c.opt.DB)
		return
	}
	// A pooler without idle-conn draining (e.g. Client.Conn's StickyConnPool)
	// can't apply buffered invalidations, so stay uncached.
	if _, ok := c.connPool.(idleConnDrainer); !ok {
		return
	}
	// The lifecycle hook serializes cache publication with connection removal
	// and socket replacement. Without it, a reply can become visible after its
	// tracking coverage is gone.
	reg, ok := c.connPool.(poolHookSupport)
	if !ok || !reg.SupportsPoolHooks() {
		return
	}
	if err := registerInvalidateHandler(c.pushProcessor, cache, c.cscKeyPrefix); err != nil {
		internal.Logger.Printf(ctx, "csc: failed to register invalidate handler: %v", err)
		return
	}
	c.csc = cache
	c.registerConnEvictHook(cache, reg)
	c.startBackgroundDrainer()
}

// cscHook returns the shared evict-on-remove hook, nil when CSC is off.
func (c *baseClient) cscHook() *cscEvictOnRemoveHook {
	h, _ := c.cscPoolHook.(*cscEvictOnRemoveHook)
	return h
}

// cscInstallConnCloseHook evicts cn's owned entries on any close — including the
// ConnMaxLifetime/idle retirement path (CloseConn) that bypasses the OnRemove
// hook — so entries don't outlive the server tracking dropped at close. Uses the
// onCscClose slot so it doesn't clobber streaming-credentials cleanup.
func (c *baseClient) cscInstallConnCloseHook(cn *pool.Conn) {
	cn.SetOnCscClose(func() error {
		c.cscOnConnClose(cn.GetID())
		return nil
	})
}

// cscInstallConnReinitHook invalidates the old socket's cache coverage before
// SetNetConnAndInitConn replaces it. The later init can then safely enable
// tracking for the new socket without a post-swap publication window.
func (c *baseClient) cscInstallConnReinitHook(cn *pool.Conn) {
	cn.SetOnCscReinit(func() {
		c.cscEvictOwnedEntries(cn.GetID())
	})
}

// cscOnConnClose evicts a closing conn's entries: via the shared hook (which
// records the removed-ring, closing the close-before-fulfill race), else scoped
// EvictByConn on the owning cache.
func (c *baseClient) cscOnConnClose(connID uint64) {
	if h := c.cscHook(); h != nil {
		h.markRemoved(connID)
		return
	}
	if c.csc != nil {
		c.csc.EvictByConn(connID)
	}
}

// poolHookSupport is the pool capability SharedTracking needs to serialize
// cache publication with connection removal and reinitialization.
type poolHookSupport interface {
	AddPoolHook(hook pool.PoolHook)
	RemovePoolHook(hook pool.PoolHook)
	SupportsPoolHooks() bool
}

// cscEvictOnRemoveHook evicts a connection's owned entries when the pool removes
// it (the server stops delivering their invalidations — Window 2), and tracks
// per-conn init generations so fulfillCached can catch a value whose owning
// conn was removed or re-initialized mid-fetch.
type cscEvictOnRemoveHook struct {
	evictor Cache

	mu sync.Mutex
	// initGen counts a live conn's socket (re)initializations: bumped by
	// cscEvictOwnedEntries before its eviction (first init included, so every
	// serving conn has gen >= 1), deleted on removal/close. fulfillCached
	// compares it with the generation captured at reply time.
	initGen map[uint64]uint64
}

func (h *cscEvictOnRemoveHook) OnGet(_ context.Context, _ *pool.Conn, _ bool) (bool, error) {
	return true, nil
}

func (h *cscEvictOnRemoveHook) OnPut(_ context.Context, _ *pool.Conn) (shouldPool, shouldRemove bool, err error) {
	return true, false, nil
}

func (h *cscEvictOnRemoveHook) OnRemove(_ context.Context, cn *pool.Conn, _ error) {
	if cn == nil {
		return
	}
	h.markRemoved(cn.GetID())
}

// markRemoved forgets connID's generation, then evicts. Forgetting before
// evicting lets a racing fulfillCached see the change (a served conn's captured
// generation is >= 1, an absent entry reads 0) and drop an entry created after
// the eviction — closing the close-before-fulfill race.
func (h *cscEvictOnRemoveHook) markRemoved(connID uint64) {
	h.forgetConn(connID)
	h.evictor.EvictByConn(connID)
}

// bumpInitGen advances connID's coverage generation. On reinit it is called by
// the pre-swap hook, before the old socket and its server-side tracking table
// are replaced.
func (h *cscEvictOnRemoveHook) bumpInitGen(connID uint64) {
	h.mu.Lock()
	if h.initGen == nil {
		h.initGen = make(map[uint64]uint64)
	}
	h.initGen[connID]++
	h.mu.Unlock()
}

// invalidateConnCoverage revokes all cache coverage associated with connID.
// Bumping before eviction also rejects an in-flight fetch that completed on the
// connection just before it left the parent's invalidation drainer.
func (h *cscEvictOnRemoveHook) invalidateConnCoverage(connID uint64) {
	h.bumpInitGen(connID)
	h.evictor.EvictByConn(connID)
}

// initGenOf returns connID's current init generation (0 if never bumped).
func (h *cscEvictOnRemoveHook) initGenOf(connID uint64) uint64 {
	h.mu.Lock()
	defer h.mu.Unlock()
	return h.initGen[connID]
}

// forgetConn drops connID's init-generation entry: the conn was removed/closed,
// or its init failed before ever serving (the pubsub path would otherwise leak
// the entry — no OnRemove hook, close hook not yet installed).
func (h *cscEvictOnRemoveHook) forgetConn(connID uint64) {
	h.mu.Lock()
	delete(h.initGen, connID)
	h.mu.Unlock()
}

// fulfillOwnedIfCovered linearizes the final coverage check with connection
// removal/re-init generation changes. Holding h.mu through FulfillOwned means
// either the old generation is rejected before the placeholder becomes valid,
// or publication wins first and the subsequent lifecycle path evicts it before
// closing/replacing the tracked socket.
func (h *cscEvictOnRemoveHook) fulfillOwnedIfCovered(
	cacheKey string,
	token, ownerConnID, capturedGen uint64,
	value []byte,
) bool {
	h.mu.Lock()
	defer h.mu.Unlock()
	if h.initGen[ownerConnID] != capturedGen {
		return false
	}
	return h.evictor.FulfillOwned(cacheKey, token, ownerConnID, value)
}

// invalidateAllCoverage revokes every connection generation known to this
// client's pool and evicts the entries those connections own. Incrementing
// instead of deleting keeps in-flight fetches that captured an old generation
// from publishing after a drainer stops.
func (h *cscEvictOnRemoveHook) invalidateAllCoverage() {
	h.mu.Lock()
	connIDs := make([]uint64, 0, len(h.initGen))
	for connID := range h.initGen {
		h.initGen[connID]++
		connIDs = append(connIDs, connID)
	}
	h.mu.Unlock()

	for _, connID := range connIDs {
		h.evictor.EvictByConn(connID)
	}
}

// registerConnEvictHook wires the required OnRemove eviction hook.
func (c *baseClient) registerConnEvictHook(cache Cache, reg poolHookSupport) {
	h := &cscEvictOnRemoveHook{evictor: cache, initGen: make(map[uint64]uint64)}
	reg.AddPoolHook(h)
	c.cscPoolHook = h
}

// cscEvictOwnedEntries evicts connID's entries on first init or immediately
// before a reinit/handoff replaces the socket and its tracking table. It
// prefers the shared hook (so Conn/Tx, which carry it but have a nil csc, still
// evict from the parent cache). Scoped only — no removed-ring (the conn keeps
// serving, and the ring never ages out); the fulfill-vs-re-init race is closed
// by the init-generation bump instead. No custom-cache flush (this also runs on
// first init).
func (c *baseClient) cscEvictOwnedEntries(connID uint64) {
	if h := c.cscHook(); h != nil {
		h.invalidateConnCoverage(connID)
		return
	}
	if c.csc == nil {
		return
	}
	c.csc.EvictByConn(connID)
}

// newStickyConnPool creates a derived sticky pool and revokes the claimed
// connection's parent-cache ownership before it becomes unreachable to the
// parent's idle-connection drainer.
func (c *baseClient) newStickyConnPool() *pool.StickyConnPool {
	sticky := pool.NewStickyConnPool(c.connPool)
	if h := c.cscHook(); h != nil {
		sticky.SetOnFirstConn(func(cn *pool.Conn) {
			if cn != nil {
				h.invalidateConnCoverage(cn.GetID())
			}
		})
	}
	return sticky
}

// cscFetchCapture receives, from the successful attempt's reply read — while
// the serving connection is still held — everything the CSC fetch path needs to
// attribute the cached entry: the raw RESP reply, the conn id, and the conn's
// CSC init generation. The generation must be captured before the conn is
// released: a handoff queued at Put can re-init the socket (bumping the
// generation) before fulfillCached runs.
type cscFetchCapture struct {
	raw     []byte
	connID  uint64
	initGen uint64
}

// cscConnInitGen returns connID's CSC init generation, captured by _process at
// reply time (while the conn is still held) and compared by fulfillCached via
// fulfillOwnedIfCovered. Zero without an active evict-on-remove hook.
func (c *baseClient) cscConnInitGen(connID uint64) uint64 {
	if h := c.cscHook(); h != nil {
		return h.initGenOf(connID)
	}
	return 0
}

// cscForgetConn drops connID's init-generation entry when initialization does
// not establish tracked coverage, either because init failed or tracking was
// rejected and CSC was disabled.
func (c *baseClient) cscForgetConn(connID uint64) {
	if h := c.cscHook(); h != nil {
		h.forgetConn(connID)
	}
}

// errClientTrackingWithCSC rejects CLIENT TRACKING on clients with built-in CSC
// (see the guards in baseClient.process and generalProcessPipeline). The raw
// escape hatches — Do(ctx, "client", "tracking", ...) with string or []byte
// args, and pipelines — are also caught: the guard matches on the command's
// leading args, not the typed method.
var errClientTrackingWithCSC = errors.New(
	"redis: CLIENT TRACKING is not allowed when client-side caching is enabled")

// errSelectWithCSC rejects runtime SELECT on clients with built-in CSC. Cache
// keys use Options.DB, while SELECT mutates only the chosen pool connection.
var errSelectWithCSC = errors.New(
	"redis: SELECT is not allowed when client-side caching is enabled")

// errAuthWithCSC rejects runtime authentication because it can change one
// connection's ACL identity without changing the client's fixed cache namespace.
var errAuthWithCSC = errors.New(
	"redis: AUTH is not allowed when client-side caching is enabled")

// errHelloWithCSC rejects HELLO with arguments because it can switch a tracked
// connection out of RESP3 (and can also change authentication).
var errHelloWithCSC = errors.New(
	"redis: HELLO with arguments is not allowed when client-side caching is enabled")

// errResetWithCSC rejects RESET because it disables tracking and switches the
// connection to RESP2.
var errResetWithCSC = errors.New(
	"redis: RESET is not allowed when client-side caching is enabled")

// errSubscribeWithCSC rejects raw subscriptions on the ordinary pool. The
// typed Subscribe methods use dedicated PubSub connections and remain allowed.
var errSubscribeWithCSC = errors.New(
	"redis: SUBSCRIBE is not allowed on pooled connections when client-side caching is enabled")

// cscCommandError rejects commands that can make a pooled connection's state
// diverge from the assumptions used by CSC.
func (c *baseClient) cscCommandError(cmd Cmder) error {
	// The successful attachment signal is shared with derived clients.
	// initConn's internal command wrapper is exempt during library setup.
	if !c.cscTrackingRequested() || c.allowClientTracking {
		return nil
	}
	switch {
	case isClientTrackingCmd(cmd):
		return errClientTrackingWithCSC
	case isSelectCmd(cmd):
		return errSelectWithCSC
	case isAuthCmd(cmd):
		return errAuthWithCSC
	case isProtocolChangingHelloCmd(cmd):
		return errHelloWithCSC
	case isResetCmd(cmd):
		return errResetWithCSC
	case isSubscribeCmd(cmd):
		return errSubscribeWithCSC
	default:
		return nil
	}
}

// cscDrainHandle owns the drainer lifecycle and serializes client teardown.
// stop signals shutdown; done is closed on exit so Close can join.
type cscDrainHandle struct {
	stop              chan struct{}
	done              chan struct{}
	stopOnce          sync.Once
	teardownOnce      sync.Once
	handlerCloseOnce  sync.Once
	closeOnce         sync.Once
	closeErr          error
	invalidateHandler *invalidateHandler
}

// signalStop closes stop at most once (so Close and the AddCleanup safety net
// can't double-close) and does not join — a GC cleanup must not block.
func (h *cscDrainHandle) signalStop() {
	h.stopOnce.Do(func() { close(h.stop) })
}

// cscHandlerClient is exposed only through the background drainer's handler
// context. Close must return before the handler does, otherwise it would wait
// for the drainer goroutine that is currently invoking the handler.
type cscHandlerClient struct {
	*baseClient
}

func (c cscHandlerClient) Close() error {
	h := c.cscDrainHandle
	if h == nil {
		return c.baseClient.Close()
	}
	h.handlerCloseOnce.Do(func() {
		// Close has logically started: stop cache hits immediately and let the
		// drainer exit as soon as this handler returns.
		if c.cscActive != nil {
			c.cscActive.Store(false)
		}
		h.signalStop()
		go func() {
			if err := c.baseClient.Close(); err != nil {
				internal.Logger.Printf(context.Background(), "csc: deferred client close failed: %v", err)
			}
		}()
	})
	return nil
}

// cscMinDrainInterval floors a user-supplied DrainInterval: sub-millisecond
// timers are unreliable (https://github.com/golang/go/issues/53824).
const cscMinDrainInterval = time.Millisecond

// cscDrainInterval returns DrainInterval clamped to cscMinDrainInterval, or the
// default (cscDrainSkipWindow) when unset.
func (c *baseClient) cscDrainInterval() time.Duration {
	if cfg := c.opt.ClientSideCacheConfig; cfg != nil && cfg.DrainInterval > 0 {
		if cfg.DrainInterval < cscMinDrainInterval {
			return cscMinDrainInterval
		}
		return cfg.DrainInterval
	}
	return cscDrainSkipWindow
}

// idleConnDrainer is the pooler capability the drainer needs (*pool.ConnPool has
// it). attachSharedTrackingCSC leaves a pooler without it uncached, rather than
// serve entries nothing would invalidate.
type idleConnDrainer interface {
	DrainIdleConns(ctx context.Context, st *pool.DrainState, fn func(cn *pool.Conn) error)
}

// startBackgroundDrainer launches the per-client invalidation drainer: each tick
// runs one pool.DrainIdleConns pass, draining idle conns' buffered push frames.
// No-op for poolers that don't implement idleConnDrainer.
func (c *baseClient) startBackgroundDrainer() {
	cp, ok := c.connPool.(idleConnDrainer)
	if !ok {
		return
	}
	if c.cscDrainHandle != nil {
		return // already running (startBackgroundDrainer runs once, in NewClient)
	}
	h := &cscDrainHandle{
		stop:              make(chan struct{}),
		done:              make(chan struct{}),
		invalidateHandler: lookupInvalidateHandler(c.pushProcessor),
	}
	c.cscDrainHandle = h
	active := &atomic.Bool{}
	active.Store(true)
	c.cscActive = active
	interval := c.cscDrainInterval()
	// Custom-processor drain errors are connection-fatal (drainPushNotifications),
	// so a PERSISTENTLY failing custom processor would turn every tick into a
	// conn removal + redial — a sustained dial storm. Damping: after
	// cscDrainCustomErrCap consecutive fatal custom-processor drains, disable
	// CSC serving and stop the drainer (with one log line) instead of churning.
	// Built-in processor errors are real conn desyncs and are never damped.
	_, builtinProc := c.pushProcessor.(*push.Processor)
	go func() {
		defer func() {
			active.Store(false)
			if c.cscPoolHook != nil {
				if reg, ok := c.connPool.(poolHookSupport); ok {
					reg.RemovePoolHook(c.cscPoolHook)
				}
			}
			if hook := c.cscHook(); hook != nil {
				hook.invalidateAllCoverage()
			}
			if h.invalidateHandler != nil {
				h.invalidateHandler.release()
			}
			close(h.done)
		}()
		ticker := time.NewTicker(interval)
		defer ticker.Stop()
		// st persists round/visited across ticks; single-goroutine, no lock.
		var st pool.DrainState
		consecFatal := 0
		drain := func(cn *pool.Conn) error {
			processorSucceeded, err := c.drainPushNotifications(cn)
			switch {
			case err != nil:
				consecFatal++
			case processorSucceeded:
				// A successful processor invocation resets consecutive
				// failures. A conn skipped without invoking the processor —
				// including a clean replacement after a fatal drain — does
				// not reset the counter.
				consecFatal = 0
			}
			return err
		}
		for {
			select {
			case <-h.stop:
				return
			case <-ticker.C:
				if !active.Load() {
					return
				}
				// ctx bounds the whole pass; the drain read has its own hard deadline.
				cycleCtx, cancel := context.WithTimeout(context.Background(), interval/2)
				cp.DrainIdleConns(cycleCtx, &st, drain)
				cancel()
				if !builtinProc && consecFatal >= cscDrainCustomErrCap {
					internal.Logger.Printf(context.Background(),
						"csc: disabling client-side caching: the custom push notification processor failed %d consecutive drains "+
							"(each failure removes a connection because the reader may be mid-frame); "+
							"caching cannot be kept fresh safely with this processor", consecFatal)
					return
				}
			}
		}
	}()
}

// disableCSCServing atomically stops cache hits and revokes all tracked
// connection coverage. The owner drainer observes the shared active flag on its
// next tick, including when a derived Conn or Tx discovered the incompatibility.
func (c *baseClient) disableCSCServing(ctx context.Context, reason string) {
	active := c.cscActive
	if active == nil || !active.CompareAndSwap(true, false) {
		return
	}
	if hook := c.cscHook(); hook != nil {
		hook.invalidateAllCoverage()
	}
	internal.Logger.Printf(ctx, "csc: disabling client-side caching: %s", reason)
}

// stopBackgroundDrainer joins the drainer goroutine and flushes an owned cache.
// The drainer's exit path releases its handler binding and pool hook, including
// when it stops itself. Owner-only: clones have no handle and return early.
// The fields are never cleared here — fulfillCached reads cscPoolHook on the hot
// path, so niling under a concurrent Close would race; teardownOnce makes repeat
// Close idempotent instead.
func (c *baseClient) stopBackgroundDrainer() {
	h := c.cscDrainHandle
	if h == nil {
		return
	}
	h.teardownOnce.Do(func() {
		// Stop serving cache hits on any clone before the drainer is gone.
		if c.cscActive != nil {
			c.cscActive.Store(false)
		}
		h.signalStop()
		<-h.done
		// The drainer's exit defer revoked and evicted this pool's coverage
		// before closing done, including for injected caches shared elsewhere.
		if c.cscOwnsCache && c.csc != nil {
			c.csc.Flush()
		}
	})
}

// applyCachedReply populates cmd from a previously captured raw RESP reply by
// replaying it through the command's own readReply.
func applyCachedReply(cmd Cmder, raw []byte) error {
	return cmd.readReply(proto.NewReaderSize(bytes.NewReader(raw), len(raw)+1))
}

// isCacheableReplyResult reports whether a fully read Redis reply can be
// cached. redis.Nil is a normal negative lookup, not a transport/protocol
// failure; tracking will invalidate it if the key is later created.
func isCacheableReplyResult(err error) bool {
	return err == nil || err == Nil
}

// cscDrainSkipWindow is the default SharedTracking drain period (overridable via
// ClientSideCacheConfig.DrainInterval). A buffered invalidation is picked up within
// roughly one round; MaxStaleness, when configured, is the hard time-based backstop.
const cscDrainSkipWindow = 5 * time.Millisecond

// cscDrainHardReadCap is the hard socket read deadline the drainer applies via
// Conn.WithReaderHardDeadline. It bounds only a rare partial-frame mid-read. A
// var (not const) so the tuning harness can sweep it.
var cscDrainHardReadCap = 50 * time.Millisecond

// cscDrainProbeReadCap bounds the non-consuming one-byte probe used only when
// an opaque transport may hold data that the socket readiness check cannot see.
const cscDrainProbeReadCap = 50 * time.Microsecond

// cscDrainCustomErrCap is the number of CONSECUTIVE fatal custom-processor
// drain errors after which the drainer disables CSC instead of removing (and
// redialing) a connection per tick indefinitely.
const cscDrainCustomErrCap = 8

// processCached runs the Get-Reserve-Fulfill lifecycle for a cacheable command.
// Only invoked after process has verified that CSC is active and cmd is
// eligible.
func (c *baseClient) processCached(ctx context.Context, cmd Cmder, state *processState) error {
	if err := ctx.Err(); err != nil {
		return err
	}

	// Once the drainer has stopped (owner Close, or the owner dropped without
	// Close), no invalidations flow — a surviving clone must not serve stale hits.
	if a := c.cscActive; a != nil && !a.Load() {
		return c.processWithRetry(ctx, cmd, nil, state)
	}

	rawKey, ok := buildCacheKey(cmd)
	if !ok {
		return c.processWithRetry(ctx, cmd, nil, state)
	}

	redisKeys := extractRedisKeys(cmd)
	if len(redisKeys) == 0 {
		// Without a key list we cannot react to invalidations for this command.
		return c.processWithRetry(ctx, cmd, nil, state)
	}

	keyPrefix := c.cscKeyPrefix
	if keyPrefix == "" {
		// A successfully attached client always has a namespace. Fail closed if
		// an incomplete custom baseClient reaches this path.
		return c.processWithRetry(ctx, cmd, nil, state)
	}
	key := cscNamespacedKey(keyPrefix, rawKey)
	nsRedisKeys := make([]string, len(redisKeys))
	for i, k := range redisKeys {
		nsRedisKeys[i] = cscNamespacedKey(keyPrefix, k)
	}

	// Serve hits straight from the cache.
	if data, ok := c.csc.Get(ctx, key); ok {
		if err := ctx.Err(); err != nil {
			return err
		}
		if err := applyCachedReply(cmd, data); isCacheableReplyResult(err) {
			return err
		}
		c.csc.DeleteByCacheKey(key)
	}

	token, shouldFetch := c.csc.Reserve(key, nsRedisKeys)
	if !shouldFetch {
		// Another goroutine is fetching; Get below waits until it completes.
		if data, ok := c.csc.Get(ctx, key); ok {
			if err := ctx.Err(); err != nil {
				return err
			}
			if err := applyCachedReply(cmd, data); isCacheableReplyResult(err) {
				return err
			}
			c.csc.DeleteByCacheKey(key)
		}
		// Original fetcher cancelled or its value was invalidated; try to take
		// over so later waiters still benefit from the cache.
		token, shouldFetch = c.csc.Reserve(key, nsRedisKeys)
	}

	var fc cscFetchCapture
	var capture *cscFetchCapture
	if shouldFetch {
		capture = &fc
		// Release the placeholder if processWithRetry panics; Cancel on a
		// stale token is a no-op.
		defer func() {
			if capture != nil {
				c.csc.Cancel(key, token)
			}
		}()
	}

	err := c.processWithRetry(ctx, cmd, capture, state)

	if shouldFetch {
		capture = nil // disarm the deferred Cancel
		if isCacheableReplyResult(err) {
			c.fulfillCached(key, token, &fc)
		} else {
			c.csc.Cancel(key, token)
		}
	}
	return err
}

// fulfillCached stores a fetched value, attributing it to its serving conn when
// an evict-on-remove hook is active so EvictByConn can drop it if that conn is
// removed. It also closes the attribute-vs-coverage races: the conn is released
// before this runs, so its OnRemove eviction — or a handoff re-init's scoped
// eviction — may fire before the entry exists. Publication is serialized with
// the hook's init-generation changes, so a reply whose invalidation coverage
// was already lost never becomes visible and never wakes waiters with stale
// data.
func (c *baseClient) fulfillCached(key string, token uint64, fc *cscFetchCapture) bool {
	if active := c.cscActive; active != nil && !active.Load() {
		c.csc.Cancel(key, token)
		return false
	}
	if hook := c.cscHook(); hook != nil {
		if fc.connID == 0 {
			// Invariant: an active hook always gets a real conn id (>=1). A zero id
			// would leave the entry unattributed and un-evictable, so fail closed.
			c.csc.Cancel(key, token)
			return false
		}
		if !hook.fulfillOwnedIfCovered(key, token, fc.connID, fc.initGen, fc.raw) {
			// A coverage mismatch leaves the reservation IN_PROGRESS because
			// FulfillOwned was deliberately skipped. Cancel wakes its waiters
			// as misses so one can safely refetch on a covered connection.
			c.csc.Cancel(key, token)
			return false
		}
		return true
	}
	return c.csc.FulfillOwned(key, token, 0, fc.raw)
}
