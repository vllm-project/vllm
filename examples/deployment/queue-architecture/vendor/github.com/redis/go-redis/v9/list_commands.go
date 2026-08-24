package redis

import (
	"context"
	"strings"
	"time"
)

type ListCmdable interface {
	BLPop(ctx context.Context, timeout time.Duration, keys ...string) *StringSliceCmd
	BLMPop(ctx context.Context, timeout time.Duration, direction string, count int64, keys ...string) *KeyValuesCmd
	BRPop(ctx context.Context, timeout time.Duration, keys ...string) *StringSliceCmd
	BRPopLPush(ctx context.Context, source, destination string, timeout time.Duration) *StringCmd
	LIndex(ctx context.Context, key string, index int64) *StringCmd
	LInsert(ctx context.Context, key, op string, pivot, value interface{}) *IntCmd
	LInsertBefore(ctx context.Context, key string, pivot, value interface{}) *IntCmd
	LInsertAfter(ctx context.Context, key string, pivot, value interface{}) *IntCmd
	LLen(ctx context.Context, key string) *IntCmd
	LMPop(ctx context.Context, direction string, count int64, keys ...string) *KeyValuesCmd
	LPop(ctx context.Context, key string) *StringCmd
	LPopCount(ctx context.Context, key string, count int) *StringSliceCmd
	LPos(ctx context.Context, key string, value string, args LPosArgs) *IntCmd
	LPosCount(ctx context.Context, key string, value string, count int64, args LPosArgs) *IntSliceCmd
	LPush(ctx context.Context, key string, values ...interface{}) *IntCmd
	LPushX(ctx context.Context, key string, values ...interface{}) *IntCmd
	LRange(ctx context.Context, key string, start, stop int64) *StringSliceCmd
	LRem(ctx context.Context, key string, count int64, value interface{}) *IntCmd
	LSet(ctx context.Context, key string, index int64, value interface{}) *StatusCmd
	LTrim(ctx context.Context, key string, start, stop int64) *StatusCmd
	RPop(ctx context.Context, key string) *StringCmd
	RPopCount(ctx context.Context, key string, count int) *StringSliceCmd
	RPopLPush(ctx context.Context, source, destination string) *StringCmd
	RPush(ctx context.Context, key string, values ...interface{}) *IntCmd
	RPushX(ctx context.Context, key string, values ...interface{}) *IntCmd
	LMove(ctx context.Context, source, destination, srcpos, destpos string) *StringCmd
	BLMove(ctx context.Context, source, destination, srcpos, destpos string, timeout time.Duration) *StringCmd
	LMoveM(ctx context.Context, source, destination, srcpos, destpos string, args LMoveMArgs) *StringSliceCmd
	BLMoveM(ctx context.Context, source, destination, srcpos, destpos string, timeout time.Duration, args LMoveMArgs) *StringSliceCmd
}

func (c cmdable) BLPop(ctx context.Context, timeout time.Duration, keys ...string) *StringSliceCmd {
	args := make([]interface{}, 1+len(keys)+1)
	args[0] = "blpop"
	for i, key := range keys {
		args[1+i] = key
	}
	args[len(args)-1] = formatSec(ctx, timeout)
	cmd := NewStringSliceCmd(ctx, args...)
	cmd.setReadTimeout(timeout)
	_ = c(ctx, cmd)
	return cmd
}

func (c cmdable) BLMPop(ctx context.Context, timeout time.Duration, direction string, count int64, keys ...string) *KeyValuesCmd {
	args := make([]interface{}, 3+len(keys), 6+len(keys))
	args[0] = "blmpop"
	args[1] = formatSec(ctx, timeout)
	args[2] = len(keys)
	for i, key := range keys {
		args[3+i] = key
	}
	args = append(args, strings.ToLower(direction), "count", count)
	cmd := NewKeyValuesCmd(ctx, args...)
	cmd.setReadTimeout(timeout)
	_ = c(ctx, cmd)
	return cmd
}

func (c cmdable) BRPop(ctx context.Context, timeout time.Duration, keys ...string) *StringSliceCmd {
	args := make([]interface{}, 1+len(keys)+1)
	args[0] = "brpop"
	for i, key := range keys {
		args[1+i] = key
	}
	args[len(keys)+1] = formatSec(ctx, timeout)
	cmd := NewStringSliceCmd(ctx, args...)
	cmd.setReadTimeout(timeout)
	_ = c(ctx, cmd)
	return cmd
}

// BRPopLPush pops an element from a list, pushes it to another list and returns it.
// Blocks until an element is available or timeout is reached.
//
// Deprecated: Use BLMove with RIGHT and LEFT arguments instead as of Redis 6.2.0.
func (c cmdable) BRPopLPush(ctx context.Context, source, destination string, timeout time.Duration) *StringCmd {
	cmd := NewStringCmd(
		ctx,
		"brpoplpush",
		source,
		destination,
		formatSec(ctx, timeout),
	)
	cmd.setReadTimeout(timeout)
	_ = c(ctx, cmd)
	return cmd
}

func (c cmdable) LIndex(ctx context.Context, key string, index int64) *StringCmd {
	cmd := NewStringCmd(ctx, "lindex", key, index)
	_ = c(ctx, cmd)
	return cmd
}

// LMPop Pops one or more elements from the first non-empty list key from the list of provided key names.
// direction: left or right, count: > 0
// example: client.LMPop(ctx, "left", 3, "key1", "key2")
func (c cmdable) LMPop(ctx context.Context, direction string, count int64, keys ...string) *KeyValuesCmd {
	args := make([]interface{}, 2+len(keys), 5+len(keys))
	args[0] = "lmpop"
	args[1] = len(keys)
	for i, key := range keys {
		args[2+i] = key
	}
	args = append(args, strings.ToLower(direction), "count", count)
	cmd := NewKeyValuesCmd(ctx, args...)
	_ = c(ctx, cmd)
	return cmd
}

func (c cmdable) LInsert(ctx context.Context, key, op string, pivot, value interface{}) *IntCmd {
	cmd := NewIntCmd(ctx, "linsert", key, op, pivot, value)
	_ = c(ctx, cmd)
	return cmd
}

func (c cmdable) LInsertBefore(ctx context.Context, key string, pivot, value interface{}) *IntCmd {
	cmd := NewIntCmd(ctx, "linsert", key, "before", pivot, value)
	_ = c(ctx, cmd)
	return cmd
}

func (c cmdable) LInsertAfter(ctx context.Context, key string, pivot, value interface{}) *IntCmd {
	cmd := NewIntCmd(ctx, "linsert", key, "after", pivot, value)
	_ = c(ctx, cmd)
	return cmd
}

func (c cmdable) LLen(ctx context.Context, key string) *IntCmd {
	cmd := NewIntCmd(ctx, "llen", key)
	_ = c(ctx, cmd)
	return cmd
}

func (c cmdable) LPop(ctx context.Context, key string) *StringCmd {
	cmd := NewStringCmd(ctx, "lpop", key)
	_ = c(ctx, cmd)
	return cmd
}

func (c cmdable) LPopCount(ctx context.Context, key string, count int) *StringSliceCmd {
	cmd := NewStringSliceCmd(ctx, "lpop", key, count)
	_ = c(ctx, cmd)
	return cmd
}

type LPosArgs struct {
	Rank, MaxLen int64
}

// LMoveMMode is the count semantics for LMOVEM/BLMOVEM.
type LMoveMMode string

const (
	LMoveMCount   LMoveMMode = "COUNT"   // up to Count
	LMoveMExactly LMoveMMode = "EXACTLY" // exactly Count, or nothing
)

// LMoveMOrder is the destination ordering for LMOVEM/BLMOVEM.
type LMoveMOrder string

const (
	LMoveMOBO  LMoveMOrder = "OBO"  // one-by-one, order reversed
	LMoveMBulk LMoveMOrder = "BULK" // preserve order
)

// LMoveMArgs configures the optional count group of LMOVEM/BLMOVEM.
// Count <= 0 moves a single element. Mode defaults to COUNT, Order to BULK.
type LMoveMArgs struct {
	Mode  LMoveMMode
	Count int64
	Order LMoveMOrder
}

func (a LMoveMArgs) appendArgs(args []interface{}) []interface{} {
	if a.Count <= 0 {
		return args
	}
	mode := a.Mode
	if mode == "" {
		mode = LMoveMCount
	}
	order := a.Order
	if order == "" {
		order = LMoveMBulk
	}
	return append(args, string(mode), a.Count, string(order))
}

func (c cmdable) LPos(ctx context.Context, key string, value string, a LPosArgs) *IntCmd {
	args := []interface{}{"lpos", key, value}
	if a.Rank != 0 {
		args = append(args, "rank", a.Rank)
	}
	if a.MaxLen != 0 {
		args = append(args, "maxlen", a.MaxLen)
	}

	cmd := NewIntCmd(ctx, args...)
	_ = c(ctx, cmd)
	return cmd
}

func (c cmdable) LPosCount(ctx context.Context, key string, value string, count int64, a LPosArgs) *IntSliceCmd {
	args := []interface{}{"lpos", key, value, "count", count}
	if a.Rank != 0 {
		args = append(args, "rank", a.Rank)
	}
	if a.MaxLen != 0 {
		args = append(args, "maxlen", a.MaxLen)
	}
	cmd := NewIntSliceCmd(ctx, args...)
	_ = c(ctx, cmd)
	return cmd
}

func (c cmdable) LPush(ctx context.Context, key string, values ...interface{}) *IntCmd {
	args := make([]interface{}, 2, 2+len(values))
	args[0] = "lpush"
	args[1] = key
	args = appendArgs(args, values)
	cmd := NewIntCmd(ctx, args...)
	_ = c(ctx, cmd)
	return cmd
}

func (c cmdable) LPushX(ctx context.Context, key string, values ...interface{}) *IntCmd {
	args := make([]interface{}, 2, 2+len(values))
	args[0] = "lpushx"
	args[1] = key
	args = appendArgs(args, values)
	cmd := NewIntCmd(ctx, args...)
	_ = c(ctx, cmd)
	return cmd
}

func (c cmdable) LRange(ctx context.Context, key string, start, stop int64) *StringSliceCmd {
	cmd := NewStringSliceCmd(
		ctx,
		"lrange",
		key,
		start,
		stop,
	)
	_ = c(ctx, cmd)
	return cmd
}

func (c cmdable) LRem(ctx context.Context, key string, count int64, value interface{}) *IntCmd {
	cmd := NewIntCmd(ctx, "lrem", key, count, value)
	_ = c(ctx, cmd)
	return cmd
}

func (c cmdable) LSet(ctx context.Context, key string, index int64, value interface{}) *StatusCmd {
	cmd := NewStatusCmd(ctx, "lset", key, index, value)
	_ = c(ctx, cmd)
	return cmd
}

func (c cmdable) LTrim(ctx context.Context, key string, start, stop int64) *StatusCmd {
	cmd := NewStatusCmd(
		ctx,
		"ltrim",
		key,
		start,
		stop,
	)
	_ = c(ctx, cmd)
	return cmd
}

func (c cmdable) RPop(ctx context.Context, key string) *StringCmd {
	cmd := NewStringCmd(ctx, "rpop", key)
	_ = c(ctx, cmd)
	return cmd
}

func (c cmdable) RPopCount(ctx context.Context, key string, count int) *StringSliceCmd {
	cmd := NewStringSliceCmd(ctx, "rpop", key, count)
	_ = c(ctx, cmd)
	return cmd
}

// RPopLPush atomically returns and removes the last element of the source list,
// and pushes the element as the first element of the destination list.
//
// Deprecated: Use LMove with RIGHT and LEFT arguments instead as of Redis 6.2.0.
func (c cmdable) RPopLPush(ctx context.Context, source, destination string) *StringCmd {
	cmd := NewStringCmd(ctx, "rpoplpush", source, destination)
	_ = c(ctx, cmd)
	return cmd
}

func (c cmdable) RPush(ctx context.Context, key string, values ...interface{}) *IntCmd {
	args := make([]interface{}, 2, 2+len(values))
	args[0] = "rpush"
	args[1] = key
	args = appendArgs(args, values)
	cmd := NewIntCmd(ctx, args...)
	_ = c(ctx, cmd)
	return cmd
}

func (c cmdable) RPushX(ctx context.Context, key string, values ...interface{}) *IntCmd {
	args := make([]interface{}, 2, 2+len(values))
	args[0] = "rpushx"
	args[1] = key
	args = appendArgs(args, values)
	cmd := NewIntCmd(ctx, args...)
	_ = c(ctx, cmd)
	return cmd
}

func (c cmdable) LMove(ctx context.Context, source, destination, srcpos, destpos string) *StringCmd {
	cmd := NewStringCmd(ctx, "lmove", source, destination, srcpos, destpos)
	_ = c(ctx, cmd)
	return cmd
}

func (c cmdable) BLMove(
	ctx context.Context, source, destination, srcpos, destpos string, timeout time.Duration,
) *StringCmd {
	cmd := NewStringCmd(ctx, "blmove", source, destination, srcpos, destpos, formatSec(ctx, timeout))
	cmd.setReadTimeout(timeout)
	_ = c(ctx, cmd)
	return cmd
}

// LMoveM atomically moves multiple elements between lists (Redis 8.10+).
// srcpos/destpos are "LEFT" or "RIGHT". Returns moved elements, or redis.Nil if none.
func (c cmdable) LMoveM(ctx context.Context, source, destination, srcpos, destpos string, a LMoveMArgs) *StringSliceCmd {
	args := make([]interface{}, 5, 8)
	args[0], args[1], args[2], args[3], args[4] = "lmovem", source, destination, srcpos, destpos
	args = a.appendArgs(args)
	cmd := NewStringSliceCmd(ctx, args...)
	_ = c(ctx, cmd)
	return cmd
}

// BLMoveM is the blocking variant of LMoveM (Redis 8.10+); timeout 0 blocks forever.
// Returns moved elements, or redis.Nil on timeout.
func (c cmdable) BLMoveM(ctx context.Context, source, destination, srcpos, destpos string, timeout time.Duration, a LMoveMArgs) *StringSliceCmd {
	args := make([]interface{}, 6, 9)
	args[0], args[1], args[2], args[3], args[4] = "blmovem", source, destination, srcpos, destpos
	args[5] = formatSec(ctx, timeout)
	args = a.appendArgs(args)
	cmd := NewStringSliceCmd(ctx, args...)
	cmd.setReadTimeout(timeout)
	_ = c(ctx, cmd)
	return cmd
}
