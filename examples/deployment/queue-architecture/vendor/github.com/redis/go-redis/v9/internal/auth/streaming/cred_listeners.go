package streaming

import (
	"sync"

	"github.com/redis/go-redis/v9/auth"
)

// CredentialsListeners is a thread-safe collection of credentials listeners
// indexed by connection ID.
//
// This collection is used by the Manager to maintain a registry of listeners
// for each connection in the pool. Listeners are reused when connections are
// reinitialized (e.g., after a handoff) to avoid creating duplicate subscriptions
// to the StreamingCredentialsProvider.
//
// The collection supports concurrent access from multiple goroutines during
// connection initialization, credential updates, and connection removal.
type CredentialsListeners struct {
	// listeners maps connection ID to credentials listener
	listeners map[uint64]auth.CredentialsListener

	// lock protects concurrent access to the listeners map
	lock sync.RWMutex
}

// NewCredentialsListeners creates a new thread-safe credentials listeners collection.
func NewCredentialsListeners() *CredentialsListeners {
	return &CredentialsListeners{
		listeners: make(map[uint64]auth.CredentialsListener),
	}
}

// Add adds or updates a credentials listener for a connection.
//
// If a listener already exists for the connection ID, it is replaced.
// This is safe because the old listener should have been unsubscribed
// before the connection was reinitialized.
//
// Thread-safe: Can be called concurrently from multiple goroutines.
func (c *CredentialsListeners) Add(connID uint64, listener auth.CredentialsListener) {
	c.lock.Lock()
	defer c.lock.Unlock()
	if c.listeners == nil {
		c.listeners = make(map[uint64]auth.CredentialsListener)
	}
	c.listeners[connID] = listener
}

// Get retrieves the credentials listener for a connection.
//
// Returns:
//   - listener: The credentials listener for the connection, or nil if not found
//   - ok: true if a listener exists for the connection ID, false otherwise
//
// Thread-safe: Can be called concurrently from multiple goroutines.
func (c *CredentialsListeners) Get(connID uint64) (auth.CredentialsListener, bool) {
	c.lock.RLock()
	defer c.lock.RUnlock()
	if len(c.listeners) == 0 {
		return nil, false
	}
	listener, ok := c.listeners[connID]
	return listener, ok
}

// Remove removes the credentials listener for a connection.
//
// This is called when a connection is removed from the pool to prevent
// memory leaks. If no listener exists for the connection ID, this is a no-op.
//
// Thread-safe: Can be called concurrently from multiple goroutines.
func (c *CredentialsListeners) Remove(connID uint64) {
	c.lock.Lock()
	defer c.lock.Unlock()
	delete(c.listeners, connID)
}
