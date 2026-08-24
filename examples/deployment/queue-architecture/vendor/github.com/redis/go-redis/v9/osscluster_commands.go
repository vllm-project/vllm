package redis

import (
	"context"
	"sync"
	"sync/atomic"
)

func (c *ClusterClient) DBSize(ctx context.Context) *IntCmd {
	cmd := NewIntCmd(ctx, "dbsize")
	_ = c.withProcessHook(ctx, cmd, func(ctx context.Context, _ Cmder) error {
		var size atomic.Int64
		err := c.ForEachMaster(ctx, func(ctx context.Context, master *Client) error {
			n, err := master.DBSize(ctx).Result()
			if err != nil {
				return err
			}
			size.Add(n)
			return nil
		})
		if err != nil {
			cmd.SetErr(err)
		} else {
			cmd.val = size.Load()
		}
		return nil
	})
	return cmd
}

func (c *ClusterClient) ScriptLoad(ctx context.Context, script string) *StringCmd {
	cmd := NewStringCmd(ctx, "script", "load", script)
	_ = c.withProcessHook(ctx, cmd, func(ctx context.Context, _ Cmder) error {
		var mu sync.Mutex
		err := c.ForEachShard(ctx, func(ctx context.Context, shard *Client) error {
			val, err := shard.ScriptLoad(ctx, script).Result()
			if err != nil {
				return err
			}

			mu.Lock()
			if cmd.Val() == "" {
				cmd.val = val
			}
			mu.Unlock()

			return nil
		})
		if err != nil {
			cmd.SetErr(err)
		}
		return nil
	})
	return cmd
}

func (c *ClusterClient) ScriptFlush(ctx context.Context) *StatusCmd {
	cmd := NewStatusCmd(ctx, "script", "flush")
	_ = c.withProcessHook(ctx, cmd, func(ctx context.Context, _ Cmder) error {
		err := c.ForEachShard(ctx, func(ctx context.Context, shard *Client) error {
			return shard.ScriptFlush(ctx).Err()
		})
		if err != nil {
			cmd.SetErr(err)
		}
		return nil
	})
	return cmd
}

func (c *ClusterClient) ScriptExists(ctx context.Context, hashes ...string) *BoolSliceCmd {
	args := make([]interface{}, 2+len(hashes))
	args[0] = "script"
	args[1] = "exists"
	for i, hash := range hashes {
		args[2+i] = hash
	}
	cmd := NewBoolSliceCmd(ctx, args...)

	result := make([]bool, len(hashes))
	for i := range result {
		result[i] = true
	}

	_ = c.withProcessHook(ctx, cmd, func(ctx context.Context, _ Cmder) error {
		var mu sync.Mutex
		err := c.ForEachShard(ctx, func(ctx context.Context, shard *Client) error {
			val, err := shard.ScriptExists(ctx, hashes...).Result()
			if err != nil {
				return err
			}

			mu.Lock()
			for i, v := range val {
				result[i] = result[i] && v
			}
			mu.Unlock()

			return nil
		})
		if err != nil {
			cmd.SetErr(err)
		} else {
			cmd.val = result
		}
		return nil
	})
	return cmd
}
