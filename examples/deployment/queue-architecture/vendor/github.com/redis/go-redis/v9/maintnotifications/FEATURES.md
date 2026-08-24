# Maintenance Notifications - FEATURES

## Overview

The Maintenance Notifications feature enables seamless Redis connection handoffs during cluster maintenance operations without dropping active connections. This feature leverages Redis RESP3 push notifications to provide zero-downtime maintenance for Redis Enterprise and compatible Redis deployments.

## Important

Using Maintenance Notifications may affect the read and write timeouts by relaxing them during maintenance operations.
This is necessary to prevent false failures due to increased latency during handoffs. The relaxed timeouts are automatically applied and removed as needed.

## Key Features

### Seamless Connection Handoffs
- **Zero-Downtime Maintenance**: Automatically handles connection transitions during cluster operations
- **Active Operation Preservation**: Transfers in-flight operations to new connections without interruption
- **Graceful Degradation**: Falls back to standard reconnection if handoff fails

### Push Notification Support
Supports all Redis Enterprise maintenance notification types:
- **MOVING** - Slot moving to a new node
- **MIGRATING** - Slot in migration state
- **MIGRATED** - Migration completed
- **FAILING_OVER** - Node failing over
- **FAILED_OVER** - Failover completed

### Circuit Breaker Pattern
- **Endpoint-Specific Failure Tracking**: Prevents repeated connection attempts to failing endpoints
- **Automatic Recovery Testing**: Half-open state allows gradual recovery validation
- **Configurable Thresholds**: Customize failure thresholds and reset timeouts

### Flexible Configuration
- **Auto-Detection Mode**: Automatically detects server support for maintenance notifications
- **Multiple Endpoint Types**: Support for internal/external IP/FQDN endpoint resolution
- **Auto-Scaling Workers**: Automatically sizes worker pool based on connection pool size
- **Timeout Management**: Separate timeouts for relaxed (during maintenance) and normal operations

### Extensible Hook System
- **Pre/Post Processing Hooks**: Monitor and customize notification handling
- **Built-in Hooks**: Logging and metrics collection hooks included
- **Custom Hook Support**: Implement custom business logic around maintenance events

### Comprehensive Monitoring
- **Metrics Collection**: Track notification counts, processing times, and error rates
- **Circuit Breaker Stats**: Monitor endpoint health and circuit breaker states
- **Operation Tracking**: Track active handoff operations and their lifecycle

## Architecture Highlights

### Event-Driven Handoff System
- **Asynchronous Processing**: Non-blocking handoff operations using worker pool pattern
- **Queue-Based Architecture**: Configurable queue size with auto-scaling support
- **Retry Mechanism**: Configurable retry attempts with exponential backoff

### Connection Pool Integration
- **Pool Hook Interface**: Seamless integration with go-redis connection pool
- **Connection State Management**: Atomic flags for connection usability tracking
- **Graceful Shutdown**: Ensures all in-flight handoffs complete before shutdown

### Thread-Safe Design
- **Lock-Free Operations**: Atomic operations for high-performance state tracking
- **Concurrent-Safe Maps**: sync.Map for tracking active operations
- **Minimal Lock Contention**: Read-write locks only where necessary

## Configuration Options

### Operation Modes
- **`ModeDisabled`**: Maintenance notifications completely disabled
- **`ModeEnabled`**: Forcefully enabled (fails if server doesn't support)
- **`ModeAuto`**: Auto-detect server support (recommended default)

### Endpoint Types
- **`EndpointTypeAuto`**: Auto-detect based on current connection
- **`EndpointTypeInternalIP`**: Use internal IP addresses
- **`EndpointTypeInternalFQDN`**: Use internal fully qualified domain names
- **`EndpointTypeExternalIP`**: Use external IP addresses
- **`EndpointTypeExternalFQDN`**: Use external fully qualified domain names
- **`EndpointTypeNone`**: No endpoint (reconnect with current configuration)

### Timeout Configuration
- **`RelaxedTimeout`**: Extended timeout during maintenance operations (default: 10s)
- **`HandoffTimeout`**: Maximum time for handoff completion (default: 15s)
- **`PostHandoffRelaxedDuration`**: Relaxed period after handoff (default: 2×RelaxedTimeout)

### Worker Pool Configuration
- **`MaxWorkers`**: Maximum concurrent handoff workers (auto-calculated if 0)
- **`HandoffQueueSize`**: Handoff queue capacity (auto-calculated if 0)
- **`MaxHandoffRetries`**: Maximum retry attempts for failed handoffs (default: 3)

### Circuit Breaker Configuration
- **`CircuitBreakerFailureThreshold`**: Failures before opening circuit (default: 5)
- **`CircuitBreakerResetTimeout`**: Time before testing recovery (default: 60s)
- **`CircuitBreakerMaxRequests`**: Max requests in half-open state (default: 3)

## Auto-Scaling Formulas

### Worker Pool Sizing
When `MaxWorkers = 0` (auto-calculate):
```
MaxWorkers = min(PoolSize/2, max(10, PoolSize/3))
```

### Queue Sizing
When `HandoffQueueSize = 0` (auto-calculate):
```
QueueSize = max(20 × MaxWorkers, PoolSize)
Capped by: min(MaxActiveConns + 1, 5 × PoolSize)
```

### Examples
- **Pool Size 100**: 33 workers, 660 queue (capped at 500)
- **Pool Size 100 + MaxActiveConns 150**: 33 workers, 151 queue
- **Pool Size 50**: 16 workers, 320 queue (capped at 250)

## Performance Characteristics

### Throughput
- **Non-Blocking Handoffs**: Client operations continue during handoffs
- **Concurrent Processing**: Multiple handoffs processed in parallel
- **Minimal Overhead**: Lock-free atomic operations for state tracking

### Latency
- **Relaxed Timeouts**: Extended timeouts during maintenance prevent false failures
- **Fast Path**: Connections not undergoing handoff have zero overhead
- **Graceful Degradation**: Failed handoffs fall back to standard reconnection

### Resource Usage
- **Memory Efficient**: Bounded queue sizes prevent memory exhaustion
- **Worker Pool**: Fixed worker count prevents goroutine explosion
- **Connection Reuse**: Handoff reuses existing connection objects

## Testing

### Unit Tests
- Comprehensive unit test coverage for all components
- Mock-based testing for isolation
- Concurrent operation testing

### Integration Tests
- Pool integration tests with real connection handoffs
- Circuit breaker behavior validation
- Hook system integration testing

### E2E Tests
- Real Redis Enterprise cluster testing
- Multiple scenario coverage (timeouts, endpoint types, stress tests)
- Fault injection testing
- TLS configuration testing

## Compatibility

### Requirements
- **Redis Protocol**: RESP3 required for push notifications
- **Redis Version**: Redis Enterprise or compatible Redis with maintenance notifications
- **Go Version**: Go 1.18+ (uses generics and atomic types)

### Client Support
#### Currently Supported
- **Standalone Client** (`redis.NewClient`) - Full support for MOVING, MIGRATING, MIGRATED, FAILING_OVER, FAILED_OVER notifications
- **Cluster Client** (`redis.NewClusterClient`) - Support for SMIGRATING and SMIGRATED notifications for hitless slot migrations

#### Will Not Support
- **Failover Client** (no planned support)
- **Ring Client** (no planned support)

## Migration Guide

### Enabling Maintenance Notifications (Standalone Client)

**Before:**
```go
client := redis.NewClient(&redis.Options{
    Addr:     "localhost:6379",
    Protocol: 2, // RESP2
})
```

**After:**
```go
client := redis.NewClient(&redis.Options{
    Addr:     "localhost:6379",
    Protocol: 3, // RESP3 required
    MaintNotificationsConfig: &maintnotifications.Config{
        Mode: maintnotifications.ModeAuto,
    },
})
```

### Enabling Hitless Upgrades (Cluster Client)

For Redis Cluster with hitless slot migration support:

```go
client := redis.NewClusterClient(&redis.ClusterOptions{
    Addrs:    []string{"localhost:7000", "localhost:7001", "localhost:7002"},
    Protocol: 3, // RESP3 required for push notifications
    MaintNotificationsConfig: &maintnotifications.Config{
        Mode:           maintnotifications.ModeAuto,
        RelaxedTimeout: 10 * time.Second, // Extended timeout during slot migrations
    },
})
```

The cluster client automatically handles:
- **SMIGRATING**: Relaxes timeouts when slots are being migrated
- **SMIGRATED**: Triggers lazy cluster state reload when migration completes
- **SeqID Deduplication**: Same notification from multiple nodes triggers only one reload

### Adding Monitoring

```go
// Get the manager from the client
manager := client.GetMaintNotificationsManager()
if manager != nil {
    // Add logging hook
    loggingHook := maintnotifications.NewLoggingHook(2) // Info level
    manager.AddNotificationHook(loggingHook)
    
    // Add metrics hook
    metricsHook := maintnotifications.NewMetricsHook()
    manager.AddNotificationHook(metricsHook)
}
```

## Known Limitations

1. **RESP3 Required**: Push notifications require RESP3 protocol
2. **Server Support**: Requires Redis Enterprise or compatible Redis with maintenance notifications
3. **Single Connection Commands**: Some commands (MULTI/EXEC, WATCH) may need special handling
4. **No Failover/Ring Client Support**: Failover and Ring clients are not supported and there are no plans to add support

## Future Enhancements

- Enhanced metrics and observability
- TTL-based cleanup for SeqID deduplication map