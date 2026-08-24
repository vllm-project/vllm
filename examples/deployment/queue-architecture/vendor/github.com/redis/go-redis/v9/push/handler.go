package push

import (
	"context"
)

// NotificationHandler defines the interface for push notification handlers.
type NotificationHandler interface {
	// HandlePushNotification processes a push notification with context information.
	// The handlerCtx provides information about the client, connection pool, and connection
	// on which the notification was received, allowing handlers to make informed decisions.
	// Returns an error if the notification could not be handled.
	HandlePushNotification(ctx context.Context, handlerCtx NotificationHandlerContext, notification []interface{}) error
}
