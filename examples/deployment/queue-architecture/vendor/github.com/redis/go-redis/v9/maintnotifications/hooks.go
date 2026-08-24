package maintnotifications

import (
	"context"
	"slices"

	"github.com/redis/go-redis/v9/internal"
	"github.com/redis/go-redis/v9/internal/maintnotifications/logs"
	"github.com/redis/go-redis/v9/internal/pool"
	"github.com/redis/go-redis/v9/push"
)

// LoggingHook is an example hook implementation that logs all notifications.
type LoggingHook struct {
	LogLevel int // 0=Error, 1=Warn, 2=Info, 3=Debug
}

// PreHook logs the notification before processing and allows modification.
func (lh *LoggingHook) PreHook(ctx context.Context, notificationCtx push.NotificationHandlerContext, notificationType string, notification []interface{}) ([]interface{}, bool) {
	if lh.LogLevel >= 2 { // Info level
		// Log the notification type and content
		connID := uint64(0)
		if conn, ok := notificationCtx.Conn.(*pool.Conn); ok {
			connID = conn.GetID()
		}
		seqID := int64(0)
		if slices.Contains(maintenanceNotificationTypes, notificationType) {
			// seqID is the second element in the notification array
			if len(notification) > 1 {
				if parsedSeqID, ok := notification[1].(int64); !ok {
					seqID = 0
				} else {
					seqID = parsedSeqID
				}
			}

		}
		internal.Logger.Printf(ctx, logs.ProcessingNotification(connID, seqID, notificationType, notification))
	}
	return notification, true // Continue processing with unmodified notification
}

// PostHook logs the result after processing.
func (lh *LoggingHook) PostHook(ctx context.Context, notificationCtx push.NotificationHandlerContext, notificationType string, notification []interface{}, result error) {
	connID := uint64(0)
	if conn, ok := notificationCtx.Conn.(*pool.Conn); ok {
		connID = conn.GetID()
	}
	if result != nil && lh.LogLevel >= 1 { // Warning level
		internal.Logger.Printf(ctx, logs.ProcessingNotificationFailed(connID, notificationType, result, notification))
	} else if lh.LogLevel >= 3 { // Debug level
		internal.Logger.Printf(ctx, logs.ProcessingNotificationSucceeded(connID, notificationType))
	}
}

// NewLoggingHook creates a new logging hook with the specified log level.
// Log levels: 0=Error, 1=Warn, 2=Info, 3=Debug
func NewLoggingHook(logLevel int) *LoggingHook {
	return &LoggingHook{LogLevel: logLevel}
}
