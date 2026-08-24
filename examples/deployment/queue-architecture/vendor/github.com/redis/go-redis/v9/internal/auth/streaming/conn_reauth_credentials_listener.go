package streaming

import (
	"github.com/redis/go-redis/v9/auth"
	"github.com/redis/go-redis/v9/internal/pool"
)

// ConnReAuthCredentialsListener is a credentials listener for a specific connection
// that triggers re-authentication when credentials change.
//
// This listener implements the auth.CredentialsListener interface and is subscribed
// to a StreamingCredentialsProvider. When new credentials are received via OnNext,
// it marks the connection for re-authentication through the manager.
//
// The re-authentication is always performed asynchronously to avoid blocking the
// credentials provider and to prevent potential deadlocks with the pool semaphore.
// The actual re-auth happens when the connection is returned to the pool in an idle state.
//
// Lifecycle:
//   - Created during connection initialization via Manager.Listener()
//   - Subscribed to the StreamingCredentialsProvider
//   - Receives credential updates via OnNext()
//   - Cleaned up when connection is removed from pool via Manager.RemoveListener()
type ConnReAuthCredentialsListener struct {
	// reAuth is the function to re-authenticate the connection with new credentials
	reAuth func(conn *pool.Conn, credentials auth.Credentials) error

	// onErr is the function to call when re-authentication or acquisition fails
	onErr func(conn *pool.Conn, err error)

	// conn is the connection this listener is associated with
	conn *pool.Conn

	// manager is the streaming credentials manager for coordinating re-auth
	manager *Manager
}

// OnNext is called when new credentials are received from the StreamingCredentialsProvider.
//
// This method marks the connection for asynchronous re-authentication. The actual
// re-authentication happens in the background when the connection is returned to the
// pool and is in an idle state.
//
// Asynchronous re-auth is used to:
//   - Avoid blocking the credentials provider's notification goroutine
//   - Prevent deadlocks with the pool's semaphore (especially with small pool sizes)
//   - Ensure re-auth happens when the connection is safe to use (not processing commands)
//
// The reAuthFn callback receives:
//   - nil if the connection was successfully acquired for re-auth
//   - error if acquisition timed out or failed
//
// Thread-safe: Called by the credentials provider's notification goroutine.
func (c *ConnReAuthCredentialsListener) OnNext(credentials auth.Credentials) {
	if c.conn == nil || c.conn.IsClosed() || c.manager == nil || c.reAuth == nil {
		return
	}

	// Always use async reauth to avoid complex pool semaphore issues
	// The synchronous path can cause deadlocks in the pool's semaphore mechanism
	// when called from the Subscribe goroutine, especially with small pool sizes.
	// The connection pool hook will re-authenticate the connection when it is
	// returned to the pool in a clean, idle state.
	c.manager.MarkForReAuth(c.conn, func(err error) {
		// err is from connection acquisition (timeout, etc.)
		if err != nil {
			// Log the error
			c.OnError(err)
			return
		}
		// err is from reauth command execution
		err = c.reAuth(c.conn, credentials)
		if err != nil {
			// Log the error
			c.OnError(err)
			return
		}
	})
}

// OnError is called when an error occurs during credential streaming or re-authentication.
//
// This method can be called from:
//   - The StreamingCredentialsProvider when there's an error in the credentials stream
//   - The re-auth process when connection acquisition times out
//   - The re-auth process when the AUTH command fails
//
// The error is delegated to the onErr callback provided during listener creation.
//
// Thread-safe: Can be called from multiple goroutines (provider, re-auth worker).
func (c *ConnReAuthCredentialsListener) OnError(err error) {
	if c.onErr == nil {
		return
	}

	c.onErr(c.conn, err)
}

// Ensure ConnReAuthCredentialsListener implements the CredentialsListener interface.
var _ auth.CredentialsListener = (*ConnReAuthCredentialsListener)(nil)
