package redis

import "context"

type HyperLogLogCmdable interface {
	PFAdd(ctx context.Context, key string, els ...interface{}) *IntCmd
	PFCount(ctx context.Context, keys ...string) *IntCmd
	PFMerge(ctx context.Context, dest string, keys ...string) *StatusCmd
}

func (c cmdable) PFAdd(ctx context.Context, key string, els ...interface{}) *IntCmd {
	args := make([]interface{}, 2, 2+len(els))
	args[0] = "pfadd"
	args[1] = key
	args = appendArgs(args, els)
	cmd := NewIntCmd(ctx, args...)
	_ = c(ctx, cmd)
	return cmd
}

func (c cmdable) PFCount(ctx context.Context, keys ...string) *IntCmd {
	args := make([]interface{}, 1+len(keys))
	args[0] = "pfcount"
	for i, key := range keys {
		args[1+i] = key
	}
	cmd := NewIntCmd(ctx, args...)
	_ = c(ctx, cmd)
	return cmd
}

func (c cmdable) PFMerge(ctx context.Context, dest string, keys ...string) *StatusCmd {
	args := make([]interface{}, 2+len(keys))
	args[0] = "pfmerge"
	args[1] = dest
	for i, key := range keys {
		args[2+i] = key
	}
	cmd := NewStatusCmd(ctx, args...)
	_ = c(ctx, cmd)
	return cmd
}
