package maintnotifications

import (
	"context"
	"fmt"
	"time"

	"github.com/redis/go-redis/v9/internal"
	"github.com/redis/go-redis/v9/internal/maintnotifications/logs"
	"github.com/redis/go-redis/v9/internal/pool"
	"github.com/redis/go-redis/v9/push"
)

// contextKey is a custom type for context keys to avoid collisions
type contextKey string

const (
	startTimeKey contextKey = "maint_notif_start_time"
)

// MetricsHook collects metrics about notification processing.
type MetricsHook struct {
	NotificationCounts map[string]int64
	ProcessingTimes    map[string]time.Duration
	ErrorCounts        map[string]int64
	HandoffCounts      int64 // Total handoffs initiated
	HandoffSuccesses   int64 // Successful handoffs
	HandoffFailures    int64 // Failed handoffs
}

// NewMetricsHook creates a new metrics collection hook.
func NewMetricsHook() *MetricsHook {
	return &MetricsHook{
		NotificationCounts: make(map[string]int64),
		ProcessingTimes:    make(map[string]time.Duration),
		ErrorCounts:        make(map[string]int64),
	}
}

// PreHook records the start time for processing metrics.
func (mh *MetricsHook) PreHook(ctx context.Context, notificationCtx push.NotificationHandlerContext, notificationType string, notification []interface{}) ([]interface{}, bool) {
	mh.NotificationCounts[notificationType]++

	// Log connection information if available
	if conn, ok := notificationCtx.Conn.(*pool.Conn); ok {
		internal.Logger.Printf(ctx, logs.MetricsHookProcessingNotification(notificationType, conn.GetID()))
	}

	// Store start time in context for duration calculation
	startTime := time.Now()
	_ = context.WithValue(ctx, startTimeKey, startTime) // Context not used further

	return notification, true
}

// PostHook records processing completion and any errors.
func (mh *MetricsHook) PostHook(ctx context.Context, notificationCtx push.NotificationHandlerContext, notificationType string, notification []interface{}, result error) {
	// Calculate processing duration
	if startTime, ok := ctx.Value(startTimeKey).(time.Time); ok {
		duration := time.Since(startTime)
		mh.ProcessingTimes[notificationType] = duration
	}

	// Record errors
	if result != nil {
		mh.ErrorCounts[notificationType]++

		// Log error details with connection information
		if conn, ok := notificationCtx.Conn.(*pool.Conn); ok {
			internal.Logger.Printf(ctx, logs.MetricsHookRecordedError(notificationType, conn.GetID(), result))
		}
	}
}

// GetMetrics returns a summary of collected metrics.
func (mh *MetricsHook) GetMetrics() map[string]interface{} {
	return map[string]interface{}{
		"notification_counts": mh.NotificationCounts,
		"processing_times":    mh.ProcessingTimes,
		"error_counts":        mh.ErrorCounts,
	}
}

// ExampleCircuitBreakerMonitor demonstrates how to monitor circuit breaker status
func ExampleCircuitBreakerMonitor(poolHook *PoolHook) {
	// Get circuit breaker statistics
	stats := poolHook.GetCircuitBreakerStats()

	for _, stat := range stats {
		fmt.Printf("Circuit Breaker for %s:\n", stat.Endpoint)
		fmt.Printf("  State: %s\n", stat.State)
		fmt.Printf("  Failures: %d\n", stat.Failures)
		fmt.Printf("  Last Failure: %v\n", stat.LastFailureTime)
		fmt.Printf("  Last Success: %v\n", stat.LastSuccessTime)

		// Alert if circuit breaker is open
		if stat.State.String() == "open" {
			fmt.Printf("  ⚠️  ALERT: Circuit breaker is OPEN for %s\n", stat.Endpoint)
		}
	}
}
