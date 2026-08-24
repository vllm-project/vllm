package redis

import "context"

type ClusterCmdable interface {
	ClusterMyShardID(ctx context.Context) *StringCmd
	ClusterMyID(ctx context.Context) *StringCmd
	ClusterSlots(ctx context.Context) *ClusterSlotsCmd
	ClusterShards(ctx context.Context) *ClusterShardsCmd
	ClusterLinks(ctx context.Context) *ClusterLinksCmd
	ClusterNodes(ctx context.Context) *StringCmd
	ClusterMeet(ctx context.Context, host, port string) *StatusCmd
	ClusterForget(ctx context.Context, nodeID string) *StatusCmd
	ClusterReplicate(ctx context.Context, nodeID string) *StatusCmd
	ClusterResetSoft(ctx context.Context) *StatusCmd
	ClusterResetHard(ctx context.Context) *StatusCmd
	ClusterInfo(ctx context.Context) *StringCmd
	ClusterKeySlot(ctx context.Context, key string) *IntCmd
	ClusterGetKeysInSlot(ctx context.Context, slot int, count int) *StringSliceCmd
	ClusterCountFailureReports(ctx context.Context, nodeID string) *IntCmd
	ClusterCountKeysInSlot(ctx context.Context, slot int) *IntCmd
	ClusterDelSlots(ctx context.Context, slots ...int) *StatusCmd
	ClusterDelSlotsRange(ctx context.Context, min, max int) *StatusCmd
	ClusterSaveConfig(ctx context.Context) *StatusCmd
	ClusterSlaves(ctx context.Context, nodeID string) *StringSliceCmd
	ClusterFailover(ctx context.Context) *StatusCmd
	ClusterAddSlots(ctx context.Context, slots ...int) *StatusCmd
	ClusterAddSlotsRange(ctx context.Context, min, max int) *StatusCmd
	ReadOnly(ctx context.Context) *StatusCmd
	ReadWrite(ctx context.Context) *StatusCmd
}

func (c cmdable) ClusterMyShardID(ctx context.Context) *StringCmd {
	cmd := NewStringCmd(ctx, "cluster", "myshardid")
	_ = c(ctx, cmd)
	return cmd
}

func (c cmdable) ClusterMyID(ctx context.Context) *StringCmd {
	cmd := NewStringCmd(ctx, "cluster", "myid")
	_ = c(ctx, cmd)
	return cmd
}

// ClusterSlots returns the mapping of cluster slots to nodes.
//
// Deprecated: Use ClusterShards instead as of Redis 7.0.0.
func (c cmdable) ClusterSlots(ctx context.Context) *ClusterSlotsCmd {
	cmd := NewClusterSlotsCmd(ctx, "cluster", "slots")
	_ = c(ctx, cmd)
	return cmd
}

func (c cmdable) ClusterShards(ctx context.Context) *ClusterShardsCmd {
	cmd := NewClusterShardsCmd(ctx, "cluster", "shards")
	_ = c(ctx, cmd)
	return cmd
}

func (c cmdable) ClusterLinks(ctx context.Context) *ClusterLinksCmd {
	cmd := NewClusterLinksCmd(ctx, "cluster", "links")
	_ = c(ctx, cmd)
	return cmd
}

func (c cmdable) ClusterNodes(ctx context.Context) *StringCmd {
	cmd := NewStringCmd(ctx, "cluster", "nodes")
	_ = c(ctx, cmd)
	return cmd
}

func (c cmdable) ClusterMeet(ctx context.Context, host, port string) *StatusCmd {
	cmd := NewStatusCmd(ctx, "cluster", "meet", host, port)
	_ = c(ctx, cmd)
	return cmd
}

func (c cmdable) ClusterForget(ctx context.Context, nodeID string) *StatusCmd {
	cmd := NewStatusCmd(ctx, "cluster", "forget", nodeID)
	_ = c(ctx, cmd)
	return cmd
}

func (c cmdable) ClusterReplicate(ctx context.Context, nodeID string) *StatusCmd {
	cmd := NewStatusCmd(ctx, "cluster", "replicate", nodeID)
	_ = c(ctx, cmd)
	return cmd
}

func (c cmdable) ClusterResetSoft(ctx context.Context) *StatusCmd {
	cmd := NewStatusCmd(ctx, "cluster", "reset", "soft")
	_ = c(ctx, cmd)
	return cmd
}

func (c cmdable) ClusterResetHard(ctx context.Context) *StatusCmd {
	cmd := NewStatusCmd(ctx, "cluster", "reset", "hard")
	_ = c(ctx, cmd)
	return cmd
}

func (c cmdable) ClusterInfo(ctx context.Context) *StringCmd {
	cmd := NewStringCmd(ctx, "cluster", "info")
	_ = c(ctx, cmd)
	return cmd
}

func (c cmdable) ClusterKeySlot(ctx context.Context, key string) *IntCmd {
	cmd := NewIntCmd(ctx, "cluster", "keyslot", key)
	_ = c(ctx, cmd)
	return cmd
}

func (c cmdable) ClusterGetKeysInSlot(ctx context.Context, slot int, count int) *StringSliceCmd {
	cmd := NewStringSliceCmd(ctx, "cluster", "getkeysinslot", slot, count)
	_ = c(ctx, cmd)
	return cmd
}

func (c cmdable) ClusterCountFailureReports(ctx context.Context, nodeID string) *IntCmd {
	cmd := NewIntCmd(ctx, "cluster", "count-failure-reports", nodeID)
	_ = c(ctx, cmd)
	return cmd
}

func (c cmdable) ClusterCountKeysInSlot(ctx context.Context, slot int) *IntCmd {
	cmd := NewIntCmd(ctx, "cluster", "countkeysinslot", slot)
	_ = c(ctx, cmd)
	return cmd
}

func (c cmdable) ClusterDelSlots(ctx context.Context, slots ...int) *StatusCmd {
	args := make([]interface{}, 2+len(slots))
	args[0] = "cluster"
	args[1] = "delslots"
	for i, slot := range slots {
		args[2+i] = slot
	}
	cmd := NewStatusCmd(ctx, args...)
	_ = c(ctx, cmd)
	return cmd
}

func (c cmdable) ClusterDelSlotsRange(ctx context.Context, min, max int) *StatusCmd {
	size := max - min + 1
	slots := make([]int, size)
	for i := 0; i < size; i++ {
		slots[i] = min + i
	}
	return c.ClusterDelSlots(ctx, slots...)
}

func (c cmdable) ClusterSaveConfig(ctx context.Context) *StatusCmd {
	cmd := NewStatusCmd(ctx, "cluster", "saveconfig")
	_ = c(ctx, cmd)
	return cmd
}

// ClusterSlaves lists the replica nodes of a master node.
//
// Deprecated: Use ClusterReplicas instead as of Redis 5.0.0.
func (c cmdable) ClusterSlaves(ctx context.Context, nodeID string) *StringSliceCmd {
	cmd := NewStringSliceCmd(ctx, "cluster", "slaves", nodeID)
	_ = c(ctx, cmd)
	return cmd
}

func (c cmdable) ClusterFailover(ctx context.Context) *StatusCmd {
	cmd := NewStatusCmd(ctx, "cluster", "failover")
	_ = c(ctx, cmd)
	return cmd
}

func (c cmdable) ClusterAddSlots(ctx context.Context, slots ...int) *StatusCmd {
	args := make([]interface{}, 2+len(slots))
	args[0] = "cluster"
	args[1] = "addslots"
	for i, num := range slots {
		args[2+i] = num
	}
	cmd := NewStatusCmd(ctx, args...)
	_ = c(ctx, cmd)
	return cmd
}

func (c cmdable) ClusterAddSlotsRange(ctx context.Context, min, max int) *StatusCmd {
	size := max - min + 1
	slots := make([]int, size)
	for i := 0; i < size; i++ {
		slots[i] = min + i
	}
	return c.ClusterAddSlots(ctx, slots...)
}

func (c cmdable) ReadOnly(ctx context.Context) *StatusCmd {
	cmd := NewStatusCmd(ctx, "readonly")
	_ = c(ctx, cmd)
	return cmd
}

func (c cmdable) ReadWrite(ctx context.Context) *StatusCmd {
	cmd := NewStatusCmd(ctx, "readwrite")
	_ = c(ctx, cmd)
	return cmd
}
