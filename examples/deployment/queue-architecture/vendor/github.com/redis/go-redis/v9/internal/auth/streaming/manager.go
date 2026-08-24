package streaming

import (
	"errors"
	"time"

	"github.com/redis/go-redis/v9/auth"
	"github.com/redis/go-redis/v9/internal/pool"
)

// Manager coordinates streaming credentials and re-authentication for a connection pool.
//
// The manager is responsible for:
//   - Creating and managing per-connection credentials listeners
//   - Providing the pool hook for re-authentication
//   - Coordinating between credentials updates and pool operations
//
// When credentials change via a StreamingCredentialsProvider:
//  1. The credentials listener (ConnReAuthCredentialsListener) receives the update
//  2. It calls MarkForReAuth on the manager
//  3. The manager delegates to the pool hook
//  4. The pool hook schedules background re-authentication
//
// The manager maintains a registry of credentials listeners indexed by connection ID,
// allowing listener reuse when connections are reinitialized (e.g., after handoff).
type Manager struct {
	// credentialsListeners maps connection ID to credentials listener
	credentialsListeners *CredentialsListeners

	// pool is the connection pool being managed
	pool pool.Pooler

	// poolHookRef is the re-authentication pool hook
	poolHookRef *ReAuthPoolHook
}

// NewManager creates a new streaming credentials manager.
//
// Parameters:
//   - pl: The connection pool to manage
//   - reAuthTimeout: Maximum time to wait for acquiring a connection for re-authentication
//
// The manager creates a ReAuthPoolHook sized to match the pool size, ensuring that
// re-auth operations don't exhaust the connection pool.
func NewManager(pl pool.Pooler, reAuthTimeout time.Duration) *Manager {
	m := &Manager{
		pool:                 pl,
		poolHookRef:          NewReAuthPoolHook(pl.Size(), reAuthTimeout),
		credentialsListeners: NewCredentialsListeners(),
	}
	m.poolHookRef.manager = m
	return m
}

// PoolHook returns the pool hook for re-authentication.
//
// This hook should be registered with the connection pool to enable
// automatic re-authentication when credentials change.
func (m *Manager) PoolHook() pool.PoolHook {
	return m.poolHookRef
}

// Listener returns or creates a credentials listener for a connection.
//
// This method is called during connection initialization to set up the
// credentials listener. If a listener already exists for the connection ID
// (e.g., after a handoff), it is reused.
//
// Parameters:
//   - poolCn: The connection to create/get a listener for
//   - reAuth: Function to re-authenticate the connection with new credentials
//   - onErr: Function to call when re-authentication fails
//
// Returns:
//   - auth.CredentialsListener: The listener to subscribe to the credentials provider
//   - error: Non-nil if poolCn is nil
//
// Note: The reAuth and onErr callbacks are captured once when the listener is
// created and reused for the connection's lifetime. They should not change.
//
// Thread-safe: Can be called concurrently during connection initialization.
func (m *Manager) Listener(
	poolCn *pool.Conn,
	reAuth func(*pool.Conn, auth.Credentials) error,
	onErr func(*pool.Conn, error),
) (auth.CredentialsListener, error) {
	if poolCn == nil {
		return nil, errors.New("poolCn cannot be nil")
	}
	connID := poolCn.GetID()
	// if we reconnect the underlying network connection, the streaming credentials listener will continue to work
	// so we can get the old listener from the cache and use it.
	// subscribing the same (an already subscribed) listener for a StreamingCredentialsProvider SHOULD be a no-op
	listener, ok := m.credentialsListeners.Get(connID)
	if !ok || listener == nil {
		// Create new listener for this connection
		// Note: Callbacks (reAuth, onErr) are captured once and reused for the connection's lifetime
		newCredListener := &ConnReAuthCredentialsListener{
			conn:    poolCn,
			reAuth:  reAuth,
			onErr:   onErr,
			manager: m,
		}

		m.credentialsListeners.Add(connID, newCredListener)
		listener = newCredListener
	}
	return listener, nil
}

// MarkForReAuth marks a connection for re-authentication.
//
// This method is called by the credentials listener when new credentials are
// received. It delegates to the pool hook to schedule background re-authentication.
//
// Parameters:
//   - poolCn: The connection to re-authenticate
//   - reAuthFn: Function to call for re-authentication, receives error if acquisition fails
//
// Thread-safe: Called by credentials listeners when credentials change.
func (m *Manager) MarkForReAuth(poolCn *pool.Conn, reAuthFn func(error)) {
	connID := poolCn.GetID()
	m.poolHookRef.MarkForReAuth(connID, reAuthFn)
}

// RemoveListener removes the credentials listener for a connection.
//
// This method is called by the pool hook's OnRemove to clean up listeners
// when connections are removed from the pool.
//
// Parameters:
//   - connID: The connection ID whose listener should be removed
//
// Thread-safe: Called during connection removal.
func (m *Manager) RemoveListener(connID uint64) {
	m.credentialsListeners.Remove(connID)
}
