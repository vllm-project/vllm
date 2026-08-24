package redis

import (
	"context"
	"net"
	"time"

	"github.com/redis/go-redis/v9/internal/otel"
	"github.com/redis/go-redis/v9/internal/pool"
)

// ConnInfo provides information about a Redis connection for metrics.
type ConnInfo interface {
	RemoteAddr() net.Addr
	PoolName() string
}

type Pooler interface {
	PoolStats() *pool.Stats
}

type PubSubPooler interface {
	Stats() *pool.PubSubStats
}

// OTelRecorder is the interface for recording OpenTelemetry metrics.

type OTelRecorder interface {
	// RecordOperationDuration records the total operation duration (including all retries)
	RecordOperationDuration(ctx context.Context, duration time.Duration, cmd Cmder, attempts int, err error, cn ConnInfo, dbIndex int)

	// RecordPipelineOperationDuration records the total pipeline/transaction duration.
	// operationName should be "PIPELINE" for regular pipelines or "MULTI" for transactions.
	RecordPipelineOperationDuration(ctx context.Context, duration time.Duration, operationName string, cmdCount int, attempts int, err error, cn ConnInfo, dbIndex int)

	// RecordConnectionCreateTime records the time it took to create a new connection
	RecordConnectionCreateTime(ctx context.Context, duration time.Duration, cn ConnInfo)

	// RecordConnectionRelaxedTimeout records when connection timeout is relaxed/unrelaxed
	// delta: +1 for relaxed, -1 for unrelaxed
	// poolName: name of the connection pool (e.g., "main", "pubsub")
	// notificationType: the notification type that triggered the timeout relaxation (e.g., "MOVING", "HANDOFF")
	RecordConnectionRelaxedTimeout(ctx context.Context, delta int, cn ConnInfo, poolName, notificationType string)

	// RecordConnectionHandoff records when a connection is handed off to another node
	// poolName: name of the connection pool (e.g., "main", "pubsub")
	RecordConnectionHandoff(ctx context.Context, cn ConnInfo, poolName string)

	// RecordError records client errors (ASK, MOVED, handshake failures, etc.)
	// errorType: type of error (e.g., "ASK", "MOVED", "HANDSHAKE_FAILED")
	// statusCode: Redis response status code if available (e.g., "MOVED", "ASK")
	// isInternal: whether this is an internal error
	// retryAttempts: number of retry attempts made
	RecordError(ctx context.Context, errorType string, cn ConnInfo, statusCode string, isInternal bool, retryAttempts int)

	// RecordMaintenanceNotification records when a maintenance notification is received
	// notificationType: the type of notification (e.g., "MOVING", "MIGRATING", etc.)
	RecordMaintenanceNotification(ctx context.Context, cn ConnInfo, notificationType string)

	// RecordConnectionWaitTime records the time spent waiting for a connection from the pool
	RecordConnectionWaitTime(ctx context.Context, duration time.Duration, cn ConnInfo)

	// RecordConnectionClosed records when a connection is closed
	// reason: reason for closing (e.g., "idle", "max_lifetime", "error", "pool_closed")
	// err: the error that caused the close (nil for non-error closures)
	RecordConnectionClosed(ctx context.Context, cn ConnInfo, reason string, err error)

	// RecordPubSubMessage records a Pub/Sub message
	// direction: "sent" or "received"
	// channel: channel name (may be hidden for cardinality reduction)
	// sharded: true for sharded pub/sub (SPUBLISH/SSUBSCRIBE)
	RecordPubSubMessage(ctx context.Context, cn ConnInfo, direction, channel string, sharded bool)

	// RecordStreamLag records the lag for stream consumer group processing
	// lag: time difference between message creation and consumption
	// streamName: name of the stream (may be hidden for cardinality reduction)
	// consumerGroup: name of the consumer group
	// consumerName: name of the consumer
	RecordStreamLag(ctx context.Context, lag time.Duration, cn ConnInfo, streamName, consumerGroup, consumerName string)
}

// OTelConnectionCounter is an optional capability interface for recording
// connection count and pending request changes via UpDownCounters.
// Implementations of OTelRecorder can optionally implement this interface
// to receive connection count and pending request delta notifications.
// This is kept separate from OTelRecorder to avoid breaking existing
// third-party implementations when new methods are added.
type OTelConnectionCounter interface {
	// RecordConnectionCount records a change in connection count (UpDownCounter)
	// delta: +1 when connection added, -1 when connection removed
	// state: connection state (e.g., "idle", "used")
	// isPubSub: true if this is a PubSub connection
	RecordConnectionCount(ctx context.Context, delta int, cn ConnInfo, state string, isPubSub bool)

	// RecordPendingRequests records a change in pending requests (UpDownCounter)
	// delta: +1 when request starts waiting, -1 when request stops waiting
	// poolName is passed explicitly because we may not have a connection yet when request starts
	RecordPendingRequests(ctx context.Context, delta int, cn ConnInfo, poolName string)
}

// This is used for async gauge metrics that need to pull stats from pools periodically.
type OTelPoolRegistrar interface {
	// RegisterPool is called when a new client is created with its main connection pool.
	// poolName: unique identifier for the pool (e.g., "main_abc123")
	RegisterPool(poolName string, pool Pooler)
	// UnregisterPool is called when a client is closed to remove its pool from the registry.
	UnregisterPool(pool Pooler)
	// RegisterPubSubPool is called when a new client is created with a PubSub pool.
	// poolName: unique identifier for the pool (e.g., "main_abc123_pubsub")
	RegisterPubSubPool(poolName string, pool PubSubPooler)
	// UnregisterPubSubPool is called when a PubSub client is closed to remove its pool.
	UnregisterPubSubPool(pool PubSubPooler)
}

// SetOTelRecorder sets the global OpenTelemetry recorder.
func SetOTelRecorder(r OTelRecorder) {
	if r == nil {
		otel.SetGlobalRecorder(nil)
		return
	}
	otel.SetGlobalRecorder(&otelRecorderAdapter{r})
}

type otelRecorderAdapter struct {
	recorder OTelRecorder
}

// toConnInfo converts *pool.Conn to ConnInfo interface properly.
// This ensures that a nil *pool.Conn becomes a true nil interface,
// not a non-nil interface containing a nil pointer.
func toConnInfo(cn *pool.Conn) ConnInfo {
	if cn == nil {
		return nil
	}
	return cn
}

func (a *otelRecorderAdapter) RecordOperationDuration(ctx context.Context, duration time.Duration, cmd otel.Cmder, attempts int, err error, cn *pool.Conn, dbIndex int) {
	// Convert internal Cmder to public Cmder
	if publicCmd, ok := cmd.(Cmder); ok {
		a.recorder.RecordOperationDuration(ctx, duration, publicCmd, attempts, err, toConnInfo(cn), dbIndex)
	}
}

func (a *otelRecorderAdapter) RecordPipelineOperationDuration(ctx context.Context, duration time.Duration, operationName string, cmdCount int, attempts int, err error, cn *pool.Conn, dbIndex int) {
	a.recorder.RecordPipelineOperationDuration(ctx, duration, operationName, cmdCount, attempts, err, toConnInfo(cn), dbIndex)
}

func (a *otelRecorderAdapter) RecordConnectionCreateTime(ctx context.Context, duration time.Duration, cn *pool.Conn) {
	a.recorder.RecordConnectionCreateTime(ctx, duration, toConnInfo(cn))
}

func (a *otelRecorderAdapter) RecordConnectionRelaxedTimeout(ctx context.Context, delta int, cn *pool.Conn, poolName, notificationType string) {
	a.recorder.RecordConnectionRelaxedTimeout(ctx, delta, toConnInfo(cn), poolName, notificationType)
}

func (a *otelRecorderAdapter) RecordConnectionHandoff(ctx context.Context, cn *pool.Conn, poolName string) {
	a.recorder.RecordConnectionHandoff(ctx, toConnInfo(cn), poolName)
}

func (a *otelRecorderAdapter) RecordError(ctx context.Context, errorType string, cn *pool.Conn, statusCode string, isInternal bool, retryAttempts int) {
	a.recorder.RecordError(ctx, errorType, toConnInfo(cn), statusCode, isInternal, retryAttempts)
}

func (a *otelRecorderAdapter) RecordMaintenanceNotification(ctx context.Context, cn *pool.Conn, notificationType string) {
	a.recorder.RecordMaintenanceNotification(ctx, toConnInfo(cn), notificationType)
}

func (a *otelRecorderAdapter) RecordConnectionWaitTime(ctx context.Context, duration time.Duration, cn *pool.Conn) {
	a.recorder.RecordConnectionWaitTime(ctx, duration, toConnInfo(cn))
}

func (a *otelRecorderAdapter) RecordConnectionClosed(ctx context.Context, cn *pool.Conn, reason string, err error) {
	a.recorder.RecordConnectionClosed(ctx, toConnInfo(cn), reason, err)
}

func (a *otelRecorderAdapter) RecordPubSubMessage(ctx context.Context, cn *pool.Conn, direction, channel string, sharded bool) {
	a.recorder.RecordPubSubMessage(ctx, toConnInfo(cn), direction, channel, sharded)
}

func (a *otelRecorderAdapter) RecordStreamLag(ctx context.Context, lag time.Duration, cn *pool.Conn, streamName, consumerGroup, consumerName string) {
	a.recorder.RecordStreamLag(ctx, lag, toConnInfo(cn), streamName, consumerGroup, consumerName)
}

func (a *otelRecorderAdapter) RecordConnectionCount(ctx context.Context, delta int, cn *pool.Conn, state string, isPubSub bool) {
	if counter, ok := a.recorder.(OTelConnectionCounter); ok {
		counter.RecordConnectionCount(ctx, delta, toConnInfo(cn), state, isPubSub)
	}
}

func (a *otelRecorderAdapter) RecordPendingRequests(ctx context.Context, delta int, cn *pool.Conn, poolName string) {
	if counter, ok := a.recorder.(OTelConnectionCounter); ok {
		counter.RecordPendingRequests(ctx, delta, toConnInfo(cn), poolName)
	}
}

func (a *otelRecorderAdapter) RegisterPool(poolName string, p pool.Pooler) {
	if registrar, ok := a.recorder.(OTelPoolRegistrar); ok {
		registrar.RegisterPool(poolName, &poolerAdapter{p})
	}
}

func (a *otelRecorderAdapter) UnregisterPool(p pool.Pooler) {
	if registrar, ok := a.recorder.(OTelPoolRegistrar); ok {
		registrar.UnregisterPool(&poolerAdapter{p})
	}
}

func (a *otelRecorderAdapter) RegisterPubSubPool(poolName string, p otel.PubSubPooler) {
	if registrar, ok := a.recorder.(OTelPoolRegistrar); ok {
		registrar.RegisterPubSubPool(poolName, &pubSubPoolerAdapter{p})
	}
}

func (a *otelRecorderAdapter) UnregisterPubSubPool(p otel.PubSubPooler) {
	if registrar, ok := a.recorder.(OTelPoolRegistrar); ok {
		registrar.UnregisterPubSubPool(&pubSubPoolerAdapter{p})
	}
}

type poolerAdapter struct {
	p pool.Pooler
}

func (a *poolerAdapter) PoolStats() *pool.Stats {
	return a.p.Stats()
}

type pubSubPoolerAdapter struct {
	p otel.PubSubPooler
}

func (a *pubSubPoolerAdapter) Stats() *pool.PubSubStats {
	return a.p.Stats()
}
