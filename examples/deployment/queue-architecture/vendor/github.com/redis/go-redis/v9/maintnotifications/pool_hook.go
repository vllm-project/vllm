package maintnotifications

import (
	"context"
	"net"
	"sync"
	"time"

	"github.com/redis/go-redis/v9/internal"
	"github.com/redis/go-redis/v9/internal/maintnotifications/logs"
	"github.com/redis/go-redis/v9/internal/pool"
)

// OperationsManagerInterface defines the interface for completing handoff operations
type OperationsManagerInterface interface {
	TrackMovingOperationWithConnID(ctx context.Context, newEndpoint string, deadline time.Time, seqID int64, connID uint64) error
	UntrackOperationWithConnID(seqID int64, connID uint64)
}

type maintNotificationsConnTracker interface {
	UntrackMaintNotificationsConn(connID uint64)
}

// HandoffRequest represents a request to handoff a connection to a new endpoint
type HandoffRequest struct {
	Conn     *pool.Conn
	ConnID   uint64 // Unique connection identifier
	Endpoint string
	SeqID    int64
	Pool     pool.Pooler // Pool to remove connection from on failure
}

// PoolHook implements pool.PoolHook for Redis-specific connection handling
// with maintenance notifications support.
type PoolHook struct {
	// Base dialer for creating connections to new endpoints during handoffs
	// args are network and address
	baseDialer func(context.Context, string, string) (net.Conn, error)

	// Network type (e.g., "tcp", "unix")
	network string

	// Worker manager for background handoff processing
	workerManager *handoffWorkerManager

	// Configuration for the maintenance notifications
	config *Config

	// Operations manager interface for operation completion tracking
	operationsManager OperationsManagerInterface

	// Pool interface for removing connections on handoff failure
	pool pool.Pooler
}

// NewPoolHook creates a new pool hook
func NewPoolHook(baseDialer func(context.Context, string, string) (net.Conn, error), network string, config *Config, operationsManager OperationsManagerInterface) *PoolHook {
	return NewPoolHookWithPoolSize(baseDialer, network, config, operationsManager, 0)
}

// NewPoolHookWithPoolSize creates a new pool hook with pool size for better worker defaults
func NewPoolHookWithPoolSize(baseDialer func(context.Context, string, string) (net.Conn, error), network string, config *Config, operationsManager OperationsManagerInterface, poolSize int) *PoolHook {
	// Apply defaults if config is nil or has zero values
	if config == nil {
		config = config.ApplyDefaultsWithPoolSize(poolSize)
	}

	ph := &PoolHook{
		// baseDialer is used to create connections to new endpoints during handoffs
		baseDialer:        baseDialer,
		network:           network,
		config:            config,
		operationsManager: operationsManager,
	}

	// Create worker manager
	ph.workerManager = newHandoffWorkerManager(config, ph)

	return ph
}

// SetPool sets the pool interface for removing connections on handoff failure
func (ph *PoolHook) SetPool(pooler pool.Pooler) {
	ph.pool = pooler
}

// GetCurrentWorkers returns the current number of active workers (for testing)
func (ph *PoolHook) GetCurrentWorkers() int {
	return ph.workerManager.getCurrentWorkers()
}

// IsHandoffPending returns true if the given connection has a pending handoff
func (ph *PoolHook) IsHandoffPending(conn *pool.Conn) bool {
	return ph.workerManager.isHandoffPending(conn)
}

// GetPendingMap returns the pending map for testing purposes
func (ph *PoolHook) GetPendingMap() *sync.Map {
	return ph.workerManager.getPendingMap()
}

// GetMaxWorkers returns the max workers for testing purposes
func (ph *PoolHook) GetMaxWorkers() int {
	return ph.workerManager.getMaxWorkers()
}

// GetHandoffQueue returns the handoff queue for testing purposes
func (ph *PoolHook) GetHandoffQueue() chan HandoffRequest {
	return ph.workerManager.getHandoffQueue()
}

// GetCircuitBreakerStats returns circuit breaker statistics for monitoring
func (ph *PoolHook) GetCircuitBreakerStats() []CircuitBreakerStats {
	return ph.workerManager.getCircuitBreakerStats()
}

// ResetCircuitBreakers resets all circuit breakers (useful for testing)
func (ph *PoolHook) ResetCircuitBreakers() {
	ph.workerManager.resetCircuitBreakers()
}

// OnGet is called when a connection is retrieved from the pool
func (ph *PoolHook) OnGet(_ context.Context, conn *pool.Conn, _ bool) (accept bool, err error) {
	// Check if connection is marked for handoff
	// This prevents using connections that have received MOVING notifications
	if conn.ShouldHandoff() {
		return false, ErrConnectionMarkedForHandoffWithState
	}

	// Check if connection is usable (not in UNUSABLE or CLOSED state)
	// This ensures we don't return connections that are currently being handed off or re-authenticated.
	if !conn.IsUsable() {
		return false, ErrConnectionMarkedForHandoff
	}

	return true, nil
}

// OnPut is called when a connection is returned to the pool
func (ph *PoolHook) OnPut(ctx context.Context, conn *pool.Conn) (shouldPool bool, shouldRemove bool, err error) {
	// first check if we should handoff for faster rejection
	if !conn.ShouldHandoff() {
		// Default behavior (no handoff): pool the connection
		return true, false, nil
	}

	// check pending handoff to not queue the same connection twice
	if ph.workerManager.isHandoffPending(conn) {
		// Default behavior (pending handoff): pool the connection
		return true, false, nil
	}

	if err := ph.workerManager.queueHandoff(conn); err != nil {
		// Failed to queue handoff, remove the connection
		internal.Logger.Printf(ctx, logs.FailedToQueueHandoff(conn.GetID(), err))
		// Don't pool, remove connection, no error to caller
		return false, true, nil
	}

	// Check if handoff was already processed by a worker before we can mark it as queued
	if !conn.ShouldHandoff() {
		// Handoff was already processed - this is normal and the connection should be pooled
		return true, false, nil
	}

	if err := conn.MarkQueuedForHandoff(); err != nil {
		// Marking can fail if a worker advanced the connection's state between
		// our queueHandoff above and here. Re-check ShouldHandoff: with the CAS
		// rollback in Conn.MarkQueuedForHandoff, a worker that already cleared
		// the handoff state is no longer misreported as ShouldHandoff=true, so a
		// cleared connection is reliably detected and pooled here.
		if !conn.ShouldHandoff() {
			// Handoff was processed - this is normal, pool the connection.
			return true, false, nil
		}
		// Still marked for handoff in an ambiguous state — remove it rather than
		// returning a connection a queued worker may still close or replace.
		return false, true, nil
	}
	internal.Logger.Printf(ctx, logs.MarkedForHandoff(conn.GetID()))
	return true, false, nil
}

func (ph *PoolHook) OnRemove(_ context.Context, conn *pool.Conn, _ error) {
	if tracker, ok := ph.operationsManager.(maintNotificationsConnTracker); ok && conn != nil {
		tracker.UntrackMaintNotificationsConn(conn.GetID())
	}
}

// Shutdown gracefully shuts down the processor, waiting for workers to complete
func (ph *PoolHook) Shutdown(ctx context.Context) error {
	return ph.workerManager.shutdownWorkers(ctx)
}
