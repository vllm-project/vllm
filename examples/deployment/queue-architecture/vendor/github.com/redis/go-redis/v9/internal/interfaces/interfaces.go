// Package interfaces provides shared interfaces used by both the main redis package
// and the maintnotifications upgrade package to avoid circular dependencies.
package interfaces

import (
	"context"
	"net"
	"time"
)

// NotificationProcessor is (most probably) a push.NotificationProcessor
// forward declaration to avoid circular imports
type NotificationProcessor interface {
	RegisterHandler(pushNotificationName string, handler interface{}, protected bool) error
	UnregisterHandler(pushNotificationName string) error
	GetHandler(pushNotificationName string) interface{}
}

// ClientInterface defines the interface that clients must implement for maintnotifications upgrades.
type ClientInterface interface {
	// GetOptions returns the client options.
	GetOptions() OptionsInterface

	// GetPushProcessor returns the client's push notification processor.
	GetPushProcessor() NotificationProcessor
}

// OptionsInterface defines the interface for client options.
// Uses an adapter pattern to avoid circular dependencies.
type OptionsInterface interface {
	// GetReadTimeout returns the read timeout.
	GetReadTimeout() time.Duration

	// GetWriteTimeout returns the write timeout.
	GetWriteTimeout() time.Duration

	// GetNetwork returns the network type.
	GetNetwork() string

	// GetAddr returns the connection address.
	GetAddr() string

	// GetNodeAddress returns the address of the Redis node as reported by the server.
	// For cluster clients, this is the endpoint from CLUSTER SLOTS before any transformation.
	// For standalone clients, this defaults to Addr.
	GetNodeAddress() string

	// IsTLSEnabled returns true if TLS is enabled.
	IsTLSEnabled() bool

	// GetProtocol returns the protocol version.
	GetProtocol() int

	// GetPoolSize returns the connection pool size.
	GetPoolSize() int

	// NewDialer returns a new dialer function for the connection.
	NewDialer() func(context.Context) (net.Conn, error)
}
