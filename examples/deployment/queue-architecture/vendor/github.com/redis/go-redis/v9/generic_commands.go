package redis

import (
	"context"
	"time"

	"github.com/redis/go-redis/v9/internal/hashtag"
)

type GenericCmdable interface {
	Del(ctx context.Context, keys ...string) *IntCmd
	Dump(ctx context.Context, key string) *StringCmd
	Exists(ctx context.Context, keys ...string) *IntCmd
	Expire(ctx context.Context, key string, expiration time.Duration) *BoolCmd
	ExpireAt(ctx context.Context, key string, tm time.Time) *BoolCmd
	ExpireTime(ctx context.Context, key string) *DurationCmd
	ExpireNX(ctx context.Context, key string, expiration time.Duration) *BoolCmd
	ExpireXX(ctx context.Context, key string, expiration time.Duration) *BoolCmd
	ExpireGT(ctx context.Context, key string, expiration time.Duration) *BoolCmd
	ExpireLT(ctx context.Context, key string, expiration time.Duration) *BoolCmd
	Keys(ctx context.Context, pattern string) *StringSliceCmd
	Migrate(ctx context.Context, host, port, key string, db int, timeout time.Duration) *StatusCmd
	Move(ctx context.Context, key string, db int) *BoolCmd
	ObjectFreq(ctx context.Context, key string) *IntCmd
	ObjectRefCount(ctx context.Context, key string) *IntCmd
	ObjectEncoding(ctx context.Context, key string) *StringCmd
	ObjectIdleTime(ctx context.Context, key string) *DurationCmd
	Persist(ctx context.Context, key string) *BoolCmd
	PExpire(ctx context.Context, key string, expiration time.Duration) *BoolCmd
	PExpireAt(ctx context.Context, key string, tm time.Time) *BoolCmd
	PExpireTime(ctx context.Context, key string) *DurationCmd
	PTTL(ctx context.Context, key string) *DurationCmd
	RandomKey(ctx context.Context) *StringCmd
	Rename(ctx context.Context, key, newkey string) *StatusCmd
	RenameNX(ctx context.Context, key, newkey string) *BoolCmd
	Restore(ctx context.Context, key string, ttl time.Duration, value string) *StatusCmd
	RestoreReplace(ctx context.Context, key string, ttl time.Duration, value string) *StatusCmd
	Sort(ctx context.Context, key string, sort *Sort) *StringSliceCmd
	SortRO(ctx context.Context, key string, sort *Sort) *StringSliceCmd
	SortStore(ctx context.Context, key, store string, sort *Sort) *IntCmd
	SortInterfaces(ctx context.Context, key string, sort *Sort) *SliceCmd
	Touch(ctx context.Context, keys ...string) *IntCmd
	TTL(ctx context.Context, key string) *DurationCmd
	Type(ctx context.Context, key string) *StatusCmd
	Copy(ctx context.Context, sourceKey string, destKey string, db int, replace bool) *IntCmd

	Scan(ctx context.Context, cursor uint64, match string, count int64) *ScanCmd
	ScanType(ctx context.Context, cursor uint64, match string, count int64, keyType string) *ScanCmd
}

func (c cmdable) Del(ctx context.Context, keys ...string) *IntCmd {
	args := make([]interface{}, 1+len(keys))
	args[0] = "del"
	for i, key := range keys {
		args[1+i] = key
	}
	cmd := NewIntCmd(ctx, args...)
	_ = c(ctx, cmd)
	return cmd
}

func (c cmdable) Unlink(ctx context.Context, keys ...string) *IntCmd {
	args := make([]interface{}, 1+len(keys))
	args[0] = "unlink"
	for i, key := range keys {
		args[1+i] = key
	}
	cmd := NewIntCmd(ctx, args...)
	_ = c(ctx, cmd)
	return cmd
}

func (c cmdable) Dump(ctx context.Context, key string) *StringCmd {
	cmd := NewStringCmd(ctx, "dump", key)
	_ = c(ctx, cmd)
	return cmd
}

func (c cmdable) Exists(ctx context.Context, keys ...string) *IntCmd {
	args := make([]interface{}, 1+len(keys))
	args[0] = "exists"
	for i, key := range keys {
		args[1+i] = key
	}
	cmd := NewIntCmd(ctx, args...)
	_ = c(ctx, cmd)
	return cmd
}

func (c cmdable) Expire(ctx context.Context, key string, expiration time.Duration) *BoolCmd {
	return c.expire(ctx, key, expiration, "")
}

func (c cmdable) ExpireNX(ctx context.Context, key string, expiration time.Duration) *BoolCmd {
	return c.expire(ctx, key, expiration, "NX")
}

func (c cmdable) ExpireXX(ctx context.Context, key string, expiration time.Duration) *BoolCmd {
	return c.expire(ctx, key, expiration, "XX")
}

func (c cmdable) ExpireGT(ctx context.Context, key string, expiration time.Duration) *BoolCmd {
	return c.expire(ctx, key, expiration, "GT")
}

func (c cmdable) ExpireLT(ctx context.Context, key string, expiration time.Duration) *BoolCmd {
	return c.expire(ctx, key, expiration, "LT")
}

func (c cmdable) expire(
	ctx context.Context, key string, expiration time.Duration, mode string,
) *BoolCmd {
	args := make([]interface{}, 3, 4)
	args[0] = "expire"
	args[1] = key
	args[2] = formatSec(ctx, expiration)
	if mode != "" {
		args = append(args, mode)
	}

	cmd := NewBoolCmd(ctx, args...)
	_ = c(ctx, cmd)
	return cmd
}

func (c cmdable) ExpireAt(ctx context.Context, key string, tm time.Time) *BoolCmd {
	cmd := NewBoolCmd(ctx, "expireat", key, tm.Unix())
	_ = c(ctx, cmd)
	return cmd
}

// ExpireTime returns the absolute expiration time of key as a Unix timestamp
// encoded in *DurationCmd (seconds since the epoch), not a remaining TTL.
// Convert with: time.Unix(int64(d/time.Second), 0). Use TTL/PTTL for remaining TTL.
func (c cmdable) ExpireTime(ctx context.Context, key string) *DurationCmd {
	cmd := NewDurationCmd(ctx, time.Second, "expiretime", key)
	_ = c(ctx, cmd)
	return cmd
}

func (c cmdable) Keys(ctx context.Context, pattern string) *StringSliceCmd {
	cmd := NewStringSliceCmd(ctx, "keys", pattern)
	_ = c(ctx, cmd)
	return cmd
}

func (c cmdable) Migrate(ctx context.Context, host, port, key string, db int, timeout time.Duration) *StatusCmd {
	cmd := NewStatusCmd(
		ctx,
		"migrate",
		host,
		port,
		key,
		db,
		formatMs(ctx, timeout),
	)
	cmd.setReadTimeout(timeout)
	_ = c(ctx, cmd)
	return cmd
}

func (c cmdable) Move(ctx context.Context, key string, db int) *BoolCmd {
	cmd := NewBoolCmd(ctx, "move", key, db)
	_ = c(ctx, cmd)
	return cmd
}

func (c cmdable) ObjectFreq(ctx context.Context, key string) *IntCmd {
	cmd := NewIntCmd(ctx, "object", "freq", key)
	_ = c(ctx, cmd)
	return cmd
}

func (c cmdable) ObjectRefCount(ctx context.Context, key string) *IntCmd {
	cmd := NewIntCmd(ctx, "object", "refcount", key)
	_ = c(ctx, cmd)
	return cmd
}

func (c cmdable) ObjectEncoding(ctx context.Context, key string) *StringCmd {
	cmd := NewStringCmd(ctx, "object", "encoding", key)
	_ = c(ctx, cmd)
	return cmd
}

func (c cmdable) ObjectIdleTime(ctx context.Context, key string) *DurationCmd {
	cmd := NewDurationCmd(ctx, time.Second, "object", "idletime", key)
	_ = c(ctx, cmd)
	return cmd
}

func (c cmdable) Persist(ctx context.Context, key string) *BoolCmd {
	cmd := NewBoolCmd(ctx, "persist", key)
	_ = c(ctx, cmd)
	return cmd
}

func (c cmdable) PExpire(ctx context.Context, key string, expiration time.Duration) *BoolCmd {
	cmd := NewBoolCmd(ctx, "pexpire", key, formatMs(ctx, expiration))
	_ = c(ctx, cmd)
	return cmd
}

func (c cmdable) PExpireAt(ctx context.Context, key string, tm time.Time) *BoolCmd {
	cmd := NewBoolCmd(
		ctx,
		"pexpireat",
		key,
		tm.UnixNano()/int64(time.Millisecond),
	)
	_ = c(ctx, cmd)
	return cmd
}

// PExpireTime returns the absolute expiration time of key as a Unix timestamp
// encoded in *DurationCmd (milliseconds since the epoch), not a remaining TTL.
// Convert with: time.UnixMilli(int64(d/time.Millisecond)). Use TTL/PTTL for remaining TTL.
func (c cmdable) PExpireTime(ctx context.Context, key string) *DurationCmd {
	cmd := NewDurationCmd(ctx, time.Millisecond, "pexpiretime", key)
	_ = c(ctx, cmd)
	return cmd
}

func (c cmdable) PTTL(ctx context.Context, key string) *DurationCmd {
	cmd := NewDurationCmd(ctx, time.Millisecond, "pttl", key)
	_ = c(ctx, cmd)
	return cmd
}

func (c cmdable) RandomKey(ctx context.Context) *StringCmd {
	cmd := NewStringCmd(ctx, "randomkey")
	_ = c(ctx, cmd)
	return cmd
}

func (c cmdable) Rename(ctx context.Context, key, newkey string) *StatusCmd {
	cmd := NewStatusCmd(ctx, "rename", key, newkey)
	_ = c(ctx, cmd)
	return cmd
}

func (c cmdable) RenameNX(ctx context.Context, key, newkey string) *BoolCmd {
	cmd := NewBoolCmd(ctx, "renamenx", key, newkey)
	_ = c(ctx, cmd)
	return cmd
}

func (c cmdable) Restore(ctx context.Context, key string, ttl time.Duration, value string) *StatusCmd {
	cmd := NewStatusCmd(
		ctx,
		"restore",
		key,
		formatMs(ctx, ttl),
		value,
	)
	_ = c(ctx, cmd)
	return cmd
}

func (c cmdable) RestoreReplace(ctx context.Context, key string, ttl time.Duration, value string) *StatusCmd {
	cmd := NewStatusCmd(
		ctx,
		"restore",
		key,
		formatMs(ctx, ttl),
		value,
		"replace",
	)
	_ = c(ctx, cmd)
	return cmd
}

type Sort struct {
	By            string
	Offset, Count int64
	Get           []string
	Order         string
	Alpha         bool
}

func (sort *Sort) args(command, key string) []interface{} {
	args := []interface{}{command, key}

	if sort.By != "" {
		args = append(args, "by", sort.By)
	}
	if sort.Offset != 0 || sort.Count != 0 {
		args = append(args, "limit", sort.Offset, sort.Count)
	}
	for _, get := range sort.Get {
		args = append(args, "get", get)
	}
	if sort.Order != "" {
		args = append(args, sort.Order)
	}
	if sort.Alpha {
		args = append(args, "alpha")
	}
	return args
}

func (c cmdable) SortRO(ctx context.Context, key string, sort *Sort) *StringSliceCmd {
	cmd := NewStringSliceCmd(ctx, sort.args("sort_ro", key)...)
	_ = c(ctx, cmd)
	return cmd
}

func (c cmdable) Sort(ctx context.Context, key string, sort *Sort) *StringSliceCmd {
	cmd := NewStringSliceCmd(ctx, sort.args("sort", key)...)
	_ = c(ctx, cmd)
	return cmd
}

func (c cmdable) SortStore(ctx context.Context, key, store string, sort *Sort) *IntCmd {
	args := sort.args("sort", key)
	if store != "" {
		args = append(args, "store", store)
	}
	cmd := NewIntCmd(ctx, args...)
	_ = c(ctx, cmd)
	return cmd
}

func (c cmdable) SortInterfaces(ctx context.Context, key string, sort *Sort) *SliceCmd {
	cmd := NewSliceCmd(ctx, sort.args("sort", key)...)
	_ = c(ctx, cmd)
	return cmd
}

func (c cmdable) Touch(ctx context.Context, keys ...string) *IntCmd {
	args := make([]interface{}, len(keys)+1)
	args[0] = "touch"
	for i, key := range keys {
		args[i+1] = key
	}
	cmd := NewIntCmd(ctx, args...)
	_ = c(ctx, cmd)
	return cmd
}

func (c cmdable) TTL(ctx context.Context, key string) *DurationCmd {
	cmd := NewDurationCmd(ctx, time.Second, "ttl", key)
	_ = c(ctx, cmd)
	return cmd
}

func (c cmdable) Type(ctx context.Context, key string) *StatusCmd {
	cmd := NewStatusCmd(ctx, "type", key)
	_ = c(ctx, cmd)
	return cmd
}

func (c cmdable) Copy(ctx context.Context, sourceKey, destKey string, db int, replace bool) *IntCmd {
	args := []interface{}{"copy", sourceKey, destKey, "DB", db}
	if replace {
		args = append(args, "REPLACE")
	}
	cmd := NewIntCmd(ctx, args...)
	_ = c(ctx, cmd)
	return cmd
}

//------------------------------------------------------------------------------

func (c cmdable) Scan(ctx context.Context, cursor uint64, match string, count int64) *ScanCmd {
	args := []interface{}{"scan", cursor}
	if match != "" {
		args = append(args, "match", match)
	}
	if count > 0 {
		args = append(args, "count", count)
	}
	cmd := NewScanCmd(ctx, c, args...)
	if hashtag.Present(match) {
		cmd.SetFirstKeyPos(3)
	}
	_ = c(ctx, cmd)
	return cmd
}

func (c cmdable) ScanType(ctx context.Context, cursor uint64, match string, count int64, keyType string) *ScanCmd {
	args := []interface{}{"scan", cursor}
	if match != "" {
		args = append(args, "match", match)
	}
	if count > 0 {
		args = append(args, "count", count)
	}
	if keyType != "" {
		args = append(args, "type", keyType)
	}
	cmd := NewScanCmd(ctx, c, args...)
	if hashtag.Present(match) {
		cmd.SetFirstKeyPos(3)
	}
	_ = c(ctx, cmd)
	return cmd
}
