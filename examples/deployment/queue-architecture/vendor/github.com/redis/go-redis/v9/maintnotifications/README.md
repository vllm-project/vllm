# Maintenance Notifications

Seamless Redis connection handoffs during cluster maintenance operations without dropping connections.

## Cluster Support

**Cluster notifications are now supported for ClusterClient!**

- **SMIGRATING**: `["SMIGRATING", SeqID, slot/range, ...]` - Relaxes timeouts when slots are being migrated
- **SMIGRATED**: `["SMIGRATED", SeqID, src host:port, dst host:port, slot/range, ...]` - Reloads cluster state when slot migration completes

**Note:** Other maintenance notifications (MOVING, MIGRATING, MIGRATED, FAILING_OVER, FAILED_OVER) are supported only in standalone Redis clients. Cluster clients support SMIGRATING and SMIGRATED for cluster-specific slot migration handling.

## Quick Start

```go
client := redis.NewClient(&redis.Options{
    Addr:     "localhost:6379",
    Protocol: 3, // RESP3 required
	MaintNotificationsConfig: &maintnotifications.Config{
        Mode: maintnotifications.ModeEnabled,
    },
})
```

## Modes

- **`ModeDisabled`** - Maintenance notifications disabled
- **`ModeEnabled`** - Forcefully enabled (fails if server doesn't support)
- **`ModeAuto`** - Auto-detect server support (default)

## Configuration

```go
&maintnotifications.Config{
    Mode:                       maintnotifications.ModeAuto,
    EndpointType:               maintnotifications.EndpointTypeAuto,
    RelaxedTimeout:             10 * time.Second,
    HandoffTimeout:             15 * time.Second,
    MaxHandoffRetries:          3,
    MaxWorkers:                 0,    // Auto-calculated
    HandoffQueueSize:           0,    // Auto-calculated
    PostHandoffRelaxedDuration: 0,    // 2 * RelaxedTimeout
}
```

### Endpoint Types

- **`EndpointTypeAuto`** - Auto-detect based on connection (default)
- **`EndpointTypeInternalIP`** - Internal IP address
- **`EndpointTypeInternalFQDN`** - Internal FQDN
- **`EndpointTypeExternalIP`** - External IP address
- **`EndpointTypeExternalFQDN`** - External FQDN
- **`EndpointTypeNone`** - No endpoint (reconnect with current config)

### Auto-Scaling

**Workers**: `min(PoolSize/2, max(10, PoolSize/3))` when auto-calculated
**Queue**: `max(20×Workers, PoolSize)` capped by `MaxActiveConns+1` or `5×PoolSize`

**Examples:**
- Pool 100: 33 workers, 660 queue (capped at 500)
- Pool 100 + MaxActiveConns 150: 33 workers, 151 queue

## How It Works

1. Redis sends push notifications about cluster maintenance operations
2. Client creates new connections to updated endpoints
3. Active operations transfer to new connections
4. Old connections close gracefully


## For more information, see [FEATURES](FEATURES.md)
