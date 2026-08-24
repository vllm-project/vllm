package redis

import "context"

// Cluster and ring support for the HIMPORT command family.
//
// Correctness comes from the shared registry: every node/shard client holds
// the same himportRegistry (wired at client construction), so any connection
// executing an HIMPORT SET lazily replays the PREPARE, MOVED/ASK redirects
// re-prepare on the target node, and discards propagate through tombstones
// and the discard-all epoch. Replicas share the registry too — roles change
// with the topology, and a promoted replica's connections simply carry no
// prepared flags, so their first SET self-prepares.
//
// On top of that, user-issued PREPARE/DISCARD/DISCARDALL fan out eagerly to
// all masters (R.4): one connection per master is prepared/cleaned up front,
// server-side validation surfaces immediately, and leftover session state is
// bounded. The fan-out is best-effort — any connection it does not reach is
// covered by the lazy replay.

// The fan-out helpers execute the per-node copies through each node client's
// Process, so node-level hooks observe them; the cluster/ring-level
// ProcessHook chain sees only the user's command object, not the fan-out.
//
// Known limitation: an HImportPrepare pipelined together with HImportSets of
// the same new fieldset in one ClusterClient Exec is not ordered across
// nodes — per-node sub-batches run concurrently, and the registration
// happens when the PREPARE's node completes, so SETs routed to other nodes
// can race it and fail with "no such fieldset". Register the fieldset with
// the client-level HImportPrepare before pipelining (the HLD's back-to-back
// PREPARE+SET pattern is a single-connection guarantee).

// himportForEach runs fn on a set of clients (all cluster masters, or all
// ring shards).
type himportForEach func(ctx context.Context, fn func(ctx context.Context, client *Client) error) error

// himportRequeueFailedSets re-queues HIMPORT SETs of registered fieldsets
// that failed with "no such fieldset" — their stale prepared flags were just
// invalidated by himportAfterBatch, so the next pipeline attempt re-prepares
// lazily and re-executes only those SETs (a full replace, so idempotent).
// Bounded by the cluster pipeline's attempt budget.
func (c *ClusterClient) himportRequeueFailedSets(ctx context.Context, cmds []Cmder, failedCmds *cmdsMap) {
	for _, cmd := range cmds {
		// rawErr: runs on the per-node execution goroutine, same
		// self-deadlock rule as himportAfterBatch.
		if set, ok := cmd.(*HImportSetCmd); ok && himportNoSuchFieldset(set.rawErr()) {
			if _, registered := c.himport.lookup(set.fieldsetName); registered {
				_ = c.mapCmdsByNode(ctx, failedCmds, []Cmder{set})
			}
		}
	}
}

// himportFanOutPrepare registers the fieldset once in the shared registry
// and executes a pre-versioned PREPARE copy on every client; each copy marks
// its executing connection without registering again. A deterministic server
// rejection (e.g. duplicate field name) withdraws the registration; a
// transport failure keeps it, and lazy replay covers the connections the
// fan-out missed (all-succeeded semantics: the first error is reported).
func himportFanOutPrepare(ctx context.Context, registry *himportRegistry, forEach himportForEach, cmd *HImportPrepareCmd) {
	version, epoch := registry.register(cmd.fieldsetName, cmd.fields)
	err := forEach(ctx, func(ctx context.Context, client *Client) error {
		fanCmd := NewHImportPrepareCmd(ctx, cmd.fieldsetName, cmd.fields...)
		fanCmd.registryVersion = version
		fanCmd.registryEpoch = epoch
		return client.Process(ctx, fanCmd)
	})
	if err != nil {
		if isRedisError(err) {
			// Withdraw the registration; the tombstone cleans the sessions
			// on which the fan-out succeeded before the rejection.
			registry.discardVersion(cmd.fieldsetName, version)
		}
		cmd.SetErr(err)
		return
	}
	cmd.SetVal("OK")
}

// himportFanOutDiscard removes the fieldset from the shared registry
// (leaving the tombstone that lazily cleans the connections the fan-out does
// not reach) and discards it on one connection of every client.
func himportFanOutDiscard(ctx context.Context, registry *himportRegistry, forEach himportForEach, cmd *HImportDiscardCmd) {
	registered := registry.discard(cmd.fieldsetName)
	err := forEach(ctx, func(ctx context.Context, client *Client) error {
		return client.Process(ctx, NewHImportDiscardCmd(ctx, cmd.fieldsetName))
	})
	if err != nil {
		cmd.SetErr(err)
		return
	}
	if registered {
		cmd.SetVal(1)
	} else {
		cmd.SetVal(0)
	}
}

// himportFanOutDiscardAll wipes the shared registry once and executes a
// pre-epoch DISCARDALL copy on every client; each copy moves its executing
// connection to the new epoch without bumping the registry again.
func himportFanOutDiscardAll(ctx context.Context, registry *himportRegistry, forEach himportForEach, cmd *HImportDiscardAllCmd) {
	epoch, removed := registry.discardAll()
	err := forEach(ctx, func(ctx context.Context, client *Client) error {
		fanCmd := NewHImportDiscardAllCmd(ctx)
		fanCmd.registryEpoch = epoch
		return client.Process(ctx, fanCmd)
	})
	if err != nil {
		cmd.SetErr(err)
		return
	}
	cmd.SetVal(int64(removed))
}

// HImportPrepare registers the fieldset in the cluster-wide registry and
// eagerly prepares one connection on every master; all other connections —
// including those of replicas promoted later and masters added by
// resharding — are prepared lazily before their first HImportSet. See
// HashCmdable.HImportPrepare (cmdable) for the fieldset semantics.
//
// The fan-out is best-effort and reports the first error: on a server
// rejection (e.g. duplicate field name) the registration is withdrawn and
// any sessions the fan-out already prepared are cleaned lazily; on a
// transport failure the registration is kept and lazy replay covers the
// connections the fan-out missed.
//
// Requires Redis 8.10 or newer.
//
// note: the API is experimental and may be subject to change.
func (c *ClusterClient) HImportPrepare(ctx context.Context, fieldsetName string, fields ...string) *StatusCmd {
	cmd := NewHImportPrepareCmd(ctx, fieldsetName, fields...)
	himportFanOutPrepare(ctx, c.himport, c.ForEachMaster, cmd)
	return &cmd.StatusCmd
}

// HImportDiscard removes the fieldset from the cluster-wide registry and
// discards it on every master; connections the fan-out does not reach
// replay the discard before their next HIMPORT command. It returns 1 if the
// fieldset was registered on this client and is now removed.
//
// Requires Redis 8.10 or newer.
//
// note: the API is experimental and may be subject to change.
func (c *ClusterClient) HImportDiscard(ctx context.Context, fieldsetName string) *IntCmd {
	cmd := NewHImportDiscardCmd(ctx, fieldsetName)
	himportFanOutDiscard(ctx, c.himport, c.ForEachMaster, cmd)
	return &cmd.IntCmd
}

// HImportDiscardAll removes all fieldsets from the cluster-wide registry and
// wipes them on every master; connections the fan-out does not reach replay
// the wipe before their next HIMPORT command. It returns the number of
// fieldsets removed from the registry.
//
// Requires Redis 8.10 or newer.
//
// note: the API is experimental and may be subject to change.
func (c *ClusterClient) HImportDiscardAll(ctx context.Context) *IntCmd {
	cmd := NewHImportDiscardAllCmd(ctx)
	himportFanOutDiscardAll(ctx, c.himport, c.ForEachMaster, cmd)
	return &cmd.IntCmd
}

// HImportPrepare registers the fieldset in the ring-wide registry and
// eagerly prepares one connection on every shard; all other connections are
// prepared lazily before their first HImportSet. The fan-out is best-effort
// with the same failure semantics as ClusterClient.HImportPrepare. See
// HashCmdable.HImportPrepare (cmdable) for the fieldset semantics.
//
// Requires Redis 8.10 or newer.
//
// note: the API is experimental and may be subject to change.
func (c *Ring) HImportPrepare(ctx context.Context, fieldsetName string, fields ...string) *StatusCmd {
	cmd := NewHImportPrepareCmd(ctx, fieldsetName, fields...)
	himportFanOutPrepare(ctx, c.opt.himport, c.ForEachShard, cmd)
	return &cmd.StatusCmd
}

// HImportDiscard removes the fieldset from the ring-wide registry and
// discards it on every shard; connections the fan-out does not reach replay
// the discard before their next HIMPORT command. It returns 1 if the
// fieldset was registered on this client and is now removed.
//
// Requires Redis 8.10 or newer.
//
// note: the API is experimental and may be subject to change.
func (c *Ring) HImportDiscard(ctx context.Context, fieldsetName string) *IntCmd {
	cmd := NewHImportDiscardCmd(ctx, fieldsetName)
	himportFanOutDiscard(ctx, c.opt.himport, c.ForEachShard, cmd)
	return &cmd.IntCmd
}

// HImportDiscardAll removes all fieldsets from the ring-wide registry and
// wipes them on every shard; connections the fan-out does not reach replay
// the wipe before their next HIMPORT command. It returns the number of
// fieldsets removed from the registry.
//
// Requires Redis 8.10 or newer.
//
// note: the API is experimental and may be subject to change.
func (c *Ring) HImportDiscardAll(ctx context.Context) *IntCmd {
	cmd := NewHImportDiscardAllCmd(ctx)
	himportFanOutDiscardAll(ctx, c.opt.himport, c.ForEachShard, cmd)
	return &cmd.IntCmd
}
