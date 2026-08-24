package push

// No imports needed for this file

// NotificationHandlerContext provides context information about where a push notification was received.
// This struct allows handlers to make informed decisions based on the source of the notification
// with strongly typed access to different client types using concrete types.
type NotificationHandlerContext struct {
	// Client is the Redis client instance that received the notification.
	// It is interface to both allow for future expansion and to avoid
	// circular dependencies. The developer is responsible for type assertion.
	// It can be one of the following types:
	// - *redis.baseClient
	// - *redis.Client
	// - *redis.ClusterClient
	// - *redis.Conn
	Client interface{}

	// ConnPool is the connection pool from which the connection was obtained.
	// It is interface to both allow for future expansion and to avoid
	// circular dependencies. The developer is responsible for type assertion.
	// It can be one of the following types:
	// - *pool.ConnPool
	// - *pool.SingleConnPool
	// - *pool.StickyConnPool
	ConnPool interface{}

	// PubSub is the PubSub instance that received the notification.
	// It is interface to both allow for future expansion and to avoid
	// circular dependencies. The developer is responsible for type assertion.
	// It can be one of the following types:
	// - *redis.PubSub
	PubSub interface{}

	// Conn is the specific connection on which the notification was received.
	// It is interface to both allow for future expansion and to avoid
	// circular dependencies. The developer is responsible for type assertion.
	// It can be one of the following types:
	// - *pool.Conn
	Conn interface{}

	// IsBlocking indicates if the notification was received on a blocking connection.
	IsBlocking bool
}
