package maintnotifications

import (
	"context"
	"errors"
	"fmt"
	"net"
	"sync"
	"sync/atomic"
	"time"

	"github.com/redis/go-redis/v9/internal"
	"github.com/redis/go-redis/v9/internal/interfaces"
	"github.com/redis/go-redis/v9/internal/maintnotifications/logs"
	"github.com/redis/go-redis/v9/internal/pool"
	"github.com/redis/go-redis/v9/push"
)

// Push notification type constants for maintenance
const (
	NotificationMoving      = "MOVING"       // Per-connection handoff notification
	NotificationMigrating   = "MIGRATING"    // Per-connection migration start notification - relaxes timeouts
	NotificationMigrated    = "MIGRATED"     // Per-connection migration complete notification - clears relaxed timeouts
	NotificationFailingOver = "FAILING_OVER" // Per-connection failover start notification - relaxes timeouts
	NotificationFailedOver  = "FAILED_OVER"  // Per-connection failover complete notification - clears relaxed timeouts
	NotificationSMigrating  = "SMIGRATING"   // Cluster slot migrating notification - relaxes timeouts
	NotificationSMigrated   = "SMIGRATED"    // Cluster slot migrated notification -  unrelaxes timeouts and triggers cluster state reload
)

// maintenanceNotificationTypes contains all notification types that maintenance handles
var maintenanceNotificationTypes = []string{
	NotificationMoving,
	NotificationMigrating,
	NotificationMigrated,
	NotificationFailingOver,
	NotificationFailedOver,
	NotificationSMigrating,
	NotificationSMigrated,
}

// NotificationHook is called before and after notification processing
// PreHook can modify the notification and return false to skip processing
// PostHook is called after successful processing
type NotificationHook interface {
	PreHook(ctx context.Context, notificationCtx push.NotificationHandlerContext, notificationType string, notification []interface{}) ([]interface{}, bool)
	PostHook(ctx context.Context, notificationCtx push.NotificationHandlerContext, notificationType string, notification []interface{}, result error)
}

// MovingOperationKey provides a unique key for tracking MOVING operations
// that combines sequence ID with connection identifier to handle duplicate
// sequence IDs across multiple connections to the same node.
type MovingOperationKey struct {
	SeqID  int64  // Sequence ID from MOVING notification
	ConnID uint64 // Unique connection identifier
}

// String returns a string representation of the key for debugging
func (k MovingOperationKey) String() string {
	return fmt.Sprintf("seq:%d-conn:%d", k.SeqID, k.ConnID)
}

// Manager provides a simplified upgrade functionality with hooks and atomic state.
type Manager struct {
	client  interfaces.ClientInterface
	config  *Config
	options interfaces.OptionsInterface
	pool    pool.Pooler

	// MOVING operation tracking - using sync.Map for better concurrent performance
	activeMovingOps sync.Map // map[MovingOperationKey]*MovingOperation

	// SMIGRATED notification deduplication - tracks processed SeqIDs
	// Multiple connections may receive the same SMIGRATED notification
	processedSMigratedSeqIDs sync.Map // map[int64]bool

	// Atomic state tracking - no locks needed for state queries
	activeOperationCount atomic.Int64 // Number of active operations
	closed               atomic.Bool  // Manager closed state

	// shutdownTimeout bounds each pool hook's Shutdown during Close. A field
	// (not a constant) so tests can exercise the failed-Close-then-retry
	// path without waiting out the real budget.
	shutdownTimeout time.Duration

	// Notification hooks for extensibility
	hooks        []NotificationHook
	hooksMu      sync.RWMutex // Protects hooks slice
	poolHooksRef *PoolHook

	// additionalPoolHooks are pool hooks bound to pools other than the primary
	// one (e.g. a dedicated pipeline connection pool). Each is an independent
	// *PoolHook bound to its own pool because the hook's failed-handoff removal
	// target (HandoffRequest.Pool) is taken from the hook's single pool field,
	// so one hook cannot safely serve two pools. They share this Manager as
	// their operations manager, keeping MOVING/MIGRATING tracking centralized.
	additionalPoolHooks []additionalPoolHook

	// Connections that successfully enabled maintnotifications. These need to be
	// retired before the pool-level listeners are removed.
	maintNotificationsConns sync.Map // connID -> *pool.Conn

	// Cluster state reload callback for SMIGRATED notifications.
	// Stored atomically because it is set from the OnNewNode hook while a node
	// client is being created and read from the SMIGRATED push handler on that
	// node's connections, which can overlap during connection init.
	clusterStateReloadCallback atomic.Pointer[ClusterStateReloadCallback]
}

// MovingOperation tracks an active MOVING operation.
type MovingOperation struct {
	SeqID       int64
	NewEndpoint string
	StartTime   time.Time
	Deadline    time.Time
}

// ClusterStateReloadCallback is a callback function that triggers cluster state reload.
// This is used by node clients to notify their parent ClusterClient about SMIGRATED notifications.
// The hostPort parameter indicates the destination node (e.g., "127.0.0.1:6379").
// The slotRanges parameter contains the migrated slots (e.g., ["1234", "5000-6000"]).
// Currently, implementations typically reload the entire cluster state, but in the future
// this could be optimized to reload only the specific slots.
type ClusterStateReloadCallback func(ctx context.Context, hostPort string, slotRanges []string)

// NewManager creates a new simplified manager.
func NewManager(client interfaces.ClientInterface, pool pool.Pooler, config *Config) (*Manager, error) {
	if client == nil {
		return nil, ErrInvalidClient
	}

	hm := &Manager{
		client:          client,
		pool:            pool,
		options:         client.GetOptions(),
		config:          config.Clone(),
		hooks:           make([]NotificationHook, 0),
		shutdownTimeout: 10 * time.Second,
	}

	// Set up push notification handling
	if err := hm.setupPushNotifications(); err != nil {
		return nil, err
	}

	return hm, nil
}

// GetPoolHook creates a pool hook with a custom dialer.
func (hm *Manager) InitPoolHook(baseDialer func(context.Context, string, string) (net.Conn, error)) {
	poolHook := hm.createPoolHook(baseDialer)
	hm.pool.AddPoolHook(poolHook)
}

// additionalPoolHook pairs a pool hook with the pool it was attached to so the
// manager can shut it down and detach it on Close.
type additionalPoolHook struct {
	pool pool.Pooler
	hook *PoolHook
}

// InitPoolHookForPool attaches a maintnotifications pool hook to an additional
// pool (e.g. a client's dedicated pipeline connection pool). A fresh, independent
// *PoolHook is created and bound to the given pool so that connections which fail
// handoff are removed from the correct pool — the hook's removal target is its own
// single pool field, so the primary hook cannot be reused for a second pool. The
// new hook shares this Manager as its operations manager, so MOVING/MIGRATING
// tracking and notification handling stay centralized across both pools.
func (hm *Manager) InitPoolHookForPool(p pool.Pooler, baseDialer func(context.Context, string, string) (net.Conn, error)) {
	if p == nil {
		return
	}
	poolSize := 0
	network := ""
	if hm.options != nil {
		poolSize = hm.options.GetPoolSize()
		network = hm.options.GetNetwork()
	}
	hook := NewPoolHookWithPoolSize(baseDialer, network, hm.config, hm, poolSize)
	hook.SetPool(p)
	// The closed check, the append, AND the AddPoolHook must all be atomic with
	// respect to Close's snapshot: hold the lock across all three so either Close
	// runs first (closed==true here, so we neither register nor attach) or we
	// register+attach first (Close's snapshot then includes this hook and tears
	// it down). Attaching outside the lock left a window where Close could
	// snapshot/tear-down between the append and the attach, then AddPoolHook
	// would re-attach to a closed manager's pool — leaking an active hook.
	// p.AddPoolHook is lock-free (atomic swap on the pool's own hook manager) and
	// never calls back into this manager, so holding hooksMu across it is safe.
	hm.hooksMu.Lock()
	defer hm.hooksMu.Unlock()
	if hm.closed.Load() {
		return
	}
	hm.additionalPoolHooks = append(hm.additionalPoolHooks, additionalPoolHook{pool: p, hook: hook})
	p.AddPoolHook(hook)
}

// hookForConn returns the pool hook that owns cn's pool: an additional hook
// when the connection came from a secondary pool (e.g. a client's dedicated
// pipeline pool), the primary hook otherwise. Handoffs must be queued through
// the owning hook — the HandoffRequest carries that hook's pool, and a failed
// handoff removes the connection from it, so queuing a pipeline-pool
// connection on the primary hook would close the connection without freeing
// its slot in the pipeline pool's bookkeeping.
func (hm *Manager) hookForConn(cn *pool.Conn) *PoolHook {
	if cn == nil {
		return hm.poolHooksRef
	}
	name := cn.PoolName()
	if name == "" {
		return hm.poolHooksRef
	}
	hm.hooksMu.RLock()
	defer hm.hooksMu.RUnlock()
	for _, ah := range hm.additionalPoolHooks {
		// Pooler does not expose the name; the concrete pool does. A Pooler
		// implementation without it simply never matches and falls through to
		// the primary hook — the pre-existing behavior.
		if np, ok := ah.pool.(interface{ Name() string }); ok && np.Name() == name {
			return ah.hook
		}
	}
	return hm.poolHooksRef
}

// setupPushNotifications sets up push notification handling by registering with the client's processor.
func (hm *Manager) setupPushNotifications() error {
	processor := hm.client.GetPushProcessor()
	if processor == nil {
		return ErrInvalidClient // Client doesn't support push notifications
	}

	// Create our notification handler
	handler := &NotificationHandler{manager: hm, operationsManager: hm}

	// Register handlers for all upgrade notifications with the client's processor
	for _, notificationType := range maintenanceNotificationTypes {
		if err := processor.RegisterHandler(notificationType, handler, true); err != nil {
			return errors.New(logs.FailedToRegisterHandler(notificationType, err))
		}
	}

	return nil
}

// TrackMovingOperationWithConnID starts a new MOVING operation with a specific connection ID.
func (hm *Manager) TrackMovingOperationWithConnID(ctx context.Context, newEndpoint string, deadline time.Time, seqID int64, connID uint64) error {
	// Create composite key
	key := MovingOperationKey{
		SeqID:  seqID,
		ConnID: connID,
	}

	// Create MOVING operation record
	movingOp := &MovingOperation{
		SeqID:       seqID,
		NewEndpoint: newEndpoint,
		StartTime:   time.Now(),
		Deadline:    deadline,
	}

	// Use LoadOrStore for atomic check-and-set operation
	if _, loaded := hm.activeMovingOps.LoadOrStore(key, movingOp); loaded {
		// Duplicate MOVING notification, ignore
		if internal.LogLevel.DebugOrAbove() { // Debug level
			internal.Logger.Printf(context.Background(), logs.DuplicateMovingOperation(connID, newEndpoint, seqID))
		}
		return nil
	}
	if internal.LogLevel.DebugOrAbove() { // Debug level
		internal.Logger.Printf(context.Background(), logs.TrackingMovingOperation(connID, newEndpoint, seqID))
	}

	// Increment active operation count atomically
	hm.activeOperationCount.Add(1)

	return nil
}

// UntrackOperationWithConnID completes a MOVING operation with a specific connection ID.
func (hm *Manager) UntrackOperationWithConnID(seqID int64, connID uint64) {
	// Create composite key
	key := MovingOperationKey{
		SeqID:  seqID,
		ConnID: connID,
	}

	// Remove from active operations atomically
	if _, loaded := hm.activeMovingOps.LoadAndDelete(key); loaded {
		if internal.LogLevel.DebugOrAbove() { // Debug level
			internal.Logger.Printf(context.Background(), logs.UntrackingMovingOperation(connID, seqID))
		}
		// Decrement active operation count only if operation existed
		hm.activeOperationCount.Add(-1)
	} else {
		if internal.LogLevel.DebugOrAbove() { // Debug level
			internal.Logger.Printf(context.Background(), logs.OperationNotTracked(connID, seqID))
		}
	}
}

// GetActiveMovingOperations returns active operations with composite keys.
// WARNING: This method creates a new map and copies all operations on every call.
// Use sparingly, especially in hot paths or high-frequency logging.
func (hm *Manager) GetActiveMovingOperations() map[MovingOperationKey]*MovingOperation {
	result := make(map[MovingOperationKey]*MovingOperation)

	// Iterate over sync.Map to build result
	hm.activeMovingOps.Range(func(key, value interface{}) bool {
		k := key.(MovingOperationKey)
		op := value.(*MovingOperation)

		// Create a copy to avoid sharing references
		result[k] = &MovingOperation{
			SeqID:       op.SeqID,
			NewEndpoint: op.NewEndpoint,
			StartTime:   op.StartTime,
			Deadline:    op.Deadline,
		}
		return true // Continue iteration
	})

	return result
}

// IsHandoffInProgress returns true if any handoff is in progress.
// Uses atomic counter for lock-free operation.
func (hm *Manager) IsHandoffInProgress() bool {
	return hm.activeOperationCount.Load() > 0
}

// GetActiveOperationCount returns the number of active operations.
// Uses atomic counter for lock-free operation.
func (hm *Manager) GetActiveOperationCount() int64 {
	return hm.activeOperationCount.Load()
}

// MarkSMigratedSeqIDProcessed attempts to mark a SMIGRATED SeqID as processed.
// Returns true if this is the first time processing this SeqID (should process),
// false if it was already processed (should skip).
// This prevents duplicate processing when multiple connections receive the same notification.
func (hm *Manager) MarkSMigratedSeqIDProcessed(seqID int64) bool {
	_, alreadyProcessed := hm.processedSMigratedSeqIDs.LoadOrStore(seqID, true)
	return !alreadyProcessed // Return true if NOT already processed
}

// TrackMaintNotificationsConn records a connection that successfully enabled
// maintnotifications so it can be retired if the feature is later disabled for
// the pool.
func (hm *Manager) TrackMaintNotificationsConn(cn *pool.Conn) {
	if cn == nil {
		return
	}
	hm.maintNotificationsConns.Store(cn.GetID(), cn)
}

// UntrackMaintNotificationsConn removes a connection from the enabled
// maintnotifications set.
func (hm *Manager) UntrackMaintNotificationsConn(connID uint64) {
	hm.maintNotificationsConns.Delete(connID)
}

func (hm *Manager) maintNotificationsConnSnapshot() []*pool.Conn {
	var conns []*pool.Conn
	hm.maintNotificationsConns.Range(func(_, value interface{}) bool {
		if cn, ok := value.(*pool.Conn); ok {
			conns = append(conns, cn)
		}
		return true
	})
	return conns
}

func (hm *Manager) retireMaintNotificationsConns(ctx context.Context) {
	conns := hm.maintNotificationsConnSnapshot()
	if len(conns) == 0 {
		return
	}

	// Tracked connections can live in the primary pool OR in any additional
	// pool this manager attached a hook to (e.g. a client's dedicated pipeline
	// connection pool — its conns run initConn and are tracked exactly like
	// primary ones). Retire through every pool: RetireConns skips connections
	// a pool does not own, so offering the full snapshot to each pool is safe.
	// Missing the additional pools left pipeline connections in service with
	// maintnotifications enabled but no hook attached after a runtime
	// downgrade — pushes on them were silently dropped.
	pools := make([]pool.Pooler, 0, 1+len(hm.additionalPoolHooks))
	if hm.pool != nil {
		pools = append(pools, hm.pool)
	}
	hm.hooksMu.RLock()
	for _, ah := range hm.additionalPoolHooks {
		if ah.pool != nil {
			pools = append(pools, ah.pool)
		}
	}
	hm.hooksMu.RUnlock()

	for _, pl := range pools {
		if retirer, ok := pl.(pool.ConnRetirer); ok {
			retirer.RetireConns(ctx, conns, pool.CloseReasonMaintNotificationsDisabled)
			continue
		}
		for _, cn := range conns {
			_ = pl.CloseConn(ctx, cn, pool.CloseReasonMaintNotificationsDisabled, pool.MetricStateIdle)
		}
	}
}

// Close closes the manager.
func (hm *Manager) Close() error {
	// Use atomic operation for thread-safe close check
	if !hm.closed.CompareAndSwap(false, true) {
		return nil // Already closed
	}

	// Retire connections that enabled maintnotifications before removing the
	// pool-level listeners that process those push notifications.
	hm.retireMaintNotificationsConns(context.Background())

	// Shutdown the pool hook if it exists
	if hm.poolHooksRef != nil {
		// Use a timeout to prevent hanging indefinitely
		shutdownCtx, cancel := context.WithTimeout(context.Background(), hm.shutdownTimeout)
		defer cancel()

		err := hm.poolHooksRef.Shutdown(shutdownCtx)
		if err != nil {
			// was not able to close pool hook, keep closed state false
			hm.closed.Store(false)
			return err
		}
		// Remove the pool hook from the pool
		if hm.pool != nil {
			hm.pool.RemovePoolHook(hm.poolHooksRef)
		}
	}

	// Shutdown and detach any hooks bound to additional pools (e.g. a dedicated
	// pipeline pool). Snapshot under the lock so we don't iterate concurrently
	// with a registering InitPoolHookForPool; Shutdown itself runs unlocked.
	hm.hooksMu.Lock()
	additional := hm.additionalPoolHooks
	hm.additionalPoolHooks = nil
	hm.hooksMu.Unlock()
	for i, ah := range additional {
		shutdownCtx, cancel := context.WithTimeout(context.Background(), hm.shutdownTimeout)
		err := ah.hook.Shutdown(shutdownCtx)
		cancel()
		if err != nil {
			// Could not cleanly shut down this hook. Put it and the ones not
			// yet processed back so a retried Close still sees them, then stay
			// open so the caller can retry, matching the primary-hook behavior
			// above. Hooks before i already shut down and detached.
			hm.hooksMu.Lock()
			remaining := make([]additionalPoolHook, 0, len(additional)-i+len(hm.additionalPoolHooks))
			remaining = append(remaining, additional[i:]...)
			remaining = append(remaining, hm.additionalPoolHooks...)
			hm.additionalPoolHooks = remaining
			hm.hooksMu.Unlock()
			hm.closed.Store(false)
			return err
		}
		if ah.pool != nil {
			ah.pool.RemovePoolHook(ah.hook)
		}
	}

	// Clear all active operations
	hm.activeMovingOps.Range(func(key, value interface{}) bool {
		hm.activeMovingOps.Delete(key)
		return true
	})

	// Reset counter
	hm.activeOperationCount.Store(0)

	return nil
}

// GetState returns current state using atomic counter for lock-free operation.
func (hm *Manager) GetState() State {
	if hm.activeOperationCount.Load() > 0 {
		return StateMoving
	}
	return StateIdle
}

// processPreHooks calls all pre-hooks and returns the modified notification and whether to continue processing.
func (hm *Manager) processPreHooks(ctx context.Context, notificationCtx push.NotificationHandlerContext, notificationType string, notification []interface{}) ([]interface{}, bool) {
	hm.hooksMu.RLock()
	defer hm.hooksMu.RUnlock()

	currentNotification := notification

	for _, hook := range hm.hooks {
		modifiedNotification, shouldContinue := hook.PreHook(ctx, notificationCtx, notificationType, currentNotification)
		if !shouldContinue {
			return modifiedNotification, false
		}
		currentNotification = modifiedNotification
	}

	return currentNotification, true
}

// processPostHooks calls all post-hooks with the processing result.
func (hm *Manager) processPostHooks(ctx context.Context, notificationCtx push.NotificationHandlerContext, notificationType string, notification []interface{}, result error) {
	hm.hooksMu.RLock()
	defer hm.hooksMu.RUnlock()

	for _, hook := range hm.hooks {
		hook.PostHook(ctx, notificationCtx, notificationType, notification, result)
	}
}

// createPoolHook creates a pool hook with this manager already set.
func (hm *Manager) createPoolHook(baseDialer func(context.Context, string, string) (net.Conn, error)) *PoolHook {
	if hm.poolHooksRef != nil {
		return hm.poolHooksRef
	}
	// Get pool size from client options for better worker defaults
	poolSize := 0
	if hm.options != nil {
		poolSize = hm.options.GetPoolSize()
	}

	hm.poolHooksRef = NewPoolHookWithPoolSize(baseDialer, hm.options.GetNetwork(), hm.config, hm, poolSize)
	hm.poolHooksRef.SetPool(hm.pool)

	return hm.poolHooksRef
}

func (hm *Manager) AddNotificationHook(notificationHook NotificationHook) {
	hm.hooksMu.Lock()
	defer hm.hooksMu.Unlock()
	hm.hooks = append(hm.hooks, notificationHook)
}

// SetClusterStateReloadCallback sets the callback function that will be called when a SMIGRATED notification is received.
// This allows node clients to notify their parent ClusterClient to reload cluster state.
func (hm *Manager) SetClusterStateReloadCallback(callback ClusterStateReloadCallback) {
	hm.clusterStateReloadCallback.Store(&callback)
}

// TriggerClusterStateReload calls the cluster state reload callback if it's set.
// This is called when a SMIGRATED notification is received.
func (hm *Manager) TriggerClusterStateReload(ctx context.Context, hostPort string, slotRanges []string) {
	if cb := hm.clusterStateReloadCallback.Load(); cb != nil {
		(*cb)(ctx, hostPort, slotRanges)
	}
}
