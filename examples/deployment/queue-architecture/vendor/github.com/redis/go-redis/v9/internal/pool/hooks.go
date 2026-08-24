package pool

import (
	"context"
	"sync"
)

// PoolHook defines the interface for connection lifecycle hooks.
type PoolHook interface {
	// OnGet is called when a connection is retrieved from the pool.
	// It can modify the connection or return an error to prevent its use.
	// The accept flag can be used to prevent the connection from being used.
	// On Accept = false the connection is rejected and returned to the pool.
	// The error can be used to prevent the connection from being used and returned to the pool.
	// On Errors, the connection is removed from the pool.
	// It has isNewConn flag to indicate if this is a new connection (rather than idle from the pool)
	// The flag can be used for gathering metrics on pool hit/miss ratio.
	OnGet(ctx context.Context, conn *Conn, isNewConn bool) (accept bool, err error)

	// OnPut is called when a connection is returned to the pool.
	// It returns whether the connection should be pooled and whether it should be removed.
	OnPut(ctx context.Context, conn *Conn) (shouldPool bool, shouldRemove bool, err error)

	// OnRemove is called when a connection is removed from the pool.
	// This happens when:
	// - Connection fails health check
	// - Connection exceeds max lifetime
	// - Pool is being closed
	// - Connection encounters an error
	// Implementations should clean up any per-connection state.
	// The reason parameter indicates why the connection was removed.
	OnRemove(ctx context.Context, conn *Conn, reason error)
}

// PoolHookManager manages multiple pool hooks.
type PoolHookManager struct {
	hooks   []PoolHook
	hooksMu sync.RWMutex
}

// NewPoolHookManager creates a new pool hook manager.
func NewPoolHookManager() *PoolHookManager {
	return &PoolHookManager{
		hooks: make([]PoolHook, 0),
	}
}

// AddHook adds a pool hook to the manager.
// Hooks are called in the order they were added.
func (phm *PoolHookManager) AddHook(hook PoolHook) {
	phm.hooksMu.Lock()
	defer phm.hooksMu.Unlock()
	phm.hooks = append(phm.hooks, hook)
}

// RemoveHook removes a pool hook from the manager.
func (phm *PoolHookManager) RemoveHook(hook PoolHook) {
	phm.hooksMu.Lock()
	defer phm.hooksMu.Unlock()

	for i, h := range phm.hooks {
		if h == hook {
			// Remove hook by swapping with last element and truncating
			phm.hooks[i] = phm.hooks[len(phm.hooks)-1]
			phm.hooks = phm.hooks[:len(phm.hooks)-1]
			break
		}
	}
}

// ProcessOnGet calls all OnGet hooks in order.
// If any hook returns an error, processing stops and the error is returned.
func (phm *PoolHookManager) ProcessOnGet(ctx context.Context, conn *Conn, isNewConn bool) (acceptConn bool, err error) {
	// Copy slice reference while holding lock (fast)
	phm.hooksMu.RLock()
	hooks := phm.hooks
	phm.hooksMu.RUnlock()

	// Call hooks without holding lock (slow operations)
	for _, hook := range hooks {
		acceptConn, err := hook.OnGet(ctx, conn, isNewConn)
		if err != nil {
			return false, err
		}

		if !acceptConn {
			return false, nil
		}
	}
	return true, nil
}

// ProcessOnPut calls all OnPut hooks in order.
// The first hook that returns shouldRemove=true or shouldPool=false will stop processing.
func (phm *PoolHookManager) ProcessOnPut(ctx context.Context, conn *Conn) (shouldPool bool, shouldRemove bool, err error) {
	// Copy slice reference while holding lock (fast)
	phm.hooksMu.RLock()
	hooks := phm.hooks
	phm.hooksMu.RUnlock()

	shouldPool = true // Default to pooling the connection

	// Call hooks without holding lock (slow operations)
	for _, hook := range hooks {
		hookShouldPool, hookShouldRemove, hookErr := hook.OnPut(ctx, conn)

		if hookErr != nil {
			return false, true, hookErr
		}

		// If any hook says to remove or not pool, respect that decision
		if hookShouldRemove {
			return false, true, nil
		}

		if !hookShouldPool {
			shouldPool = false
		}
	}

	return shouldPool, false, nil
}

// ProcessOnRemove calls all OnRemove hooks in order.
func (phm *PoolHookManager) ProcessOnRemove(ctx context.Context, conn *Conn, reason error) {
	// Copy slice reference while holding lock (fast)
	phm.hooksMu.RLock()
	hooks := phm.hooks
	phm.hooksMu.RUnlock()

	// Call hooks without holding lock (slow operations)
	for _, hook := range hooks {
		hook.OnRemove(ctx, conn, reason)
	}
}

// GetHookCount returns the number of registered hooks (for testing).
func (phm *PoolHookManager) GetHookCount() int {
	phm.hooksMu.RLock()
	defer phm.hooksMu.RUnlock()
	return len(phm.hooks)
}

// GetHooks returns a copy of all registered hooks.
func (phm *PoolHookManager) GetHooks() []PoolHook {
	phm.hooksMu.RLock()
	defer phm.hooksMu.RUnlock()

	hooks := make([]PoolHook, len(phm.hooks))
	copy(hooks, phm.hooks)
	return hooks
}

// Clone creates a copy of the hook manager with the same hooks.
// This is used for lock-free atomic updates of the hook manager.
func (phm *PoolHookManager) Clone() *PoolHookManager {
	phm.hooksMu.RLock()
	defer phm.hooksMu.RUnlock()

	newManager := &PoolHookManager{
		hooks: make([]PoolHook, len(phm.hooks)),
	}
	copy(newManager.hooks, phm.hooks)
	return newManager
}
