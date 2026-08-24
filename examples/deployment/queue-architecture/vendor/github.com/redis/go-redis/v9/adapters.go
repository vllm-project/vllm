package redis

import (
	"context"
	"errors"
	"net"
	"time"

	"github.com/redis/go-redis/v9/internal/interfaces"
	"github.com/redis/go-redis/v9/push"
)

// ErrInvalidCommand is returned when an invalid command is passed to ExecuteCommand.
var ErrInvalidCommand = errors.New("invalid command type")

// ErrInvalidPool is returned when the pool type is not supported.
var ErrInvalidPool = errors.New("invalid pool type")

// newClientAdapter creates a new client adapter for regular Redis clients.
func newClientAdapter(client *baseClient) interfaces.ClientInterface {
	return &clientAdapter{client: client}
}

// clientAdapter adapts a Redis client to implement interfaces.ClientInterface.
type clientAdapter struct {
	client *baseClient
}

// GetOptions returns the client options.
func (ca *clientAdapter) GetOptions() interfaces.OptionsInterface {
	return &optionsAdapter{options: ca.client.opt}
}

// GetPushProcessor returns the client's push notification processor.
func (ca *clientAdapter) GetPushProcessor() interfaces.NotificationProcessor {
	return &pushProcessorAdapter{processor: ca.client.pushProcessor}
}

// optionsAdapter adapts Redis options to implement interfaces.OptionsInterface.
type optionsAdapter struct {
	options *Options
}

// GetReadTimeout returns the read timeout.
func (oa *optionsAdapter) GetReadTimeout() time.Duration {
	return oa.options.ReadTimeout
}

// GetWriteTimeout returns the write timeout.
func (oa *optionsAdapter) GetWriteTimeout() time.Duration {
	return oa.options.WriteTimeout
}

// GetNetwork returns the network type.
func (oa *optionsAdapter) GetNetwork() string {
	return oa.options.Network
}

// GetAddr returns the connection address.
func (oa *optionsAdapter) GetAddr() string {
	return oa.options.Addr
}

// GetNodeAddress returns the address of the Redis node as reported by the server.
// For cluster clients, this is the endpoint from CLUSTER SLOTS before any transformation.
// For standalone clients, this defaults to Addr.
func (oa *optionsAdapter) GetNodeAddress() string {
	return oa.options.NodeAddress
}

// IsTLSEnabled returns true if TLS is enabled.
func (oa *optionsAdapter) IsTLSEnabled() bool {
	return oa.options.TLSConfig != nil
}

// GetProtocol returns the protocol version.
func (oa *optionsAdapter) GetProtocol() int {
	return oa.options.Protocol
}

// GetPoolSize returns the connection pool size.
func (oa *optionsAdapter) GetPoolSize() int {
	return oa.options.PoolSize
}

// NewDialer returns a new dialer function for the connection.
func (oa *optionsAdapter) NewDialer() func(context.Context) (net.Conn, error) {
	baseDialer := oa.options.NewDialer()
	return func(ctx context.Context) (net.Conn, error) {
		// Extract network and address from the options
		network := oa.options.Network
		addr := oa.options.Addr
		return baseDialer(ctx, network, addr)
	}
}

// pushProcessorAdapter adapts a push.NotificationProcessor to implement interfaces.NotificationProcessor.
type pushProcessorAdapter struct {
	processor push.NotificationProcessor
}

// RegisterHandler registers a handler for a specific push notification name.
func (ppa *pushProcessorAdapter) RegisterHandler(pushNotificationName string, handler interface{}, protected bool) error {
	if pushHandler, ok := handler.(push.NotificationHandler); ok {
		return ppa.processor.RegisterHandler(pushNotificationName, pushHandler, protected)
	}
	return errors.New("handler must implement push.NotificationHandler")
}

// UnregisterHandler removes a handler for a specific push notification name.
func (ppa *pushProcessorAdapter) UnregisterHandler(pushNotificationName string) error {
	return ppa.processor.UnregisterHandler(pushNotificationName)
}

// GetHandler returns the handler for a specific push notification name.
func (ppa *pushProcessorAdapter) GetHandler(pushNotificationName string) interface{} {
	return ppa.processor.GetHandler(pushNotificationName)
}
