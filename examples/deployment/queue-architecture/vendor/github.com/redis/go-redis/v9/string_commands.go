package redis

import (
	"context"
	"fmt"
	"strings"
	"time"
)

type StringCmdable interface {
	Append(ctx context.Context, key, value string) *IntCmd
	Decr(ctx context.Context, key string) *IntCmd
	DecrBy(ctx context.Context, key string, decrement int64) *IntCmd
	DelExArgs(ctx context.Context, key string, a DelExArgs) *IntCmd
	Digest(ctx context.Context, key string) *DigestCmd
	Get(ctx context.Context, key string) *StringCmd
	GetRange(ctx context.Context, key string, start, end int64) *StringCmd
	GetSet(ctx context.Context, key string, value interface{}) *StringCmd
	GetEx(ctx context.Context, key string, expiration time.Duration) *StringCmd
	GetDel(ctx context.Context, key string) *StringCmd
	GetToBuffer(ctx context.Context, key string, buf []byte) *ZeroCopyStringCmd
	Incr(ctx context.Context, key string) *IntCmd
	IncrBy(ctx context.Context, key string, value int64) *IntCmd
	IncrByFloat(ctx context.Context, key string, value float64) *FloatCmd
	IncrEXInt(ctx context.Context, key string, args IncrEXIntArgs) *IncrEXIntCmd
	IncrEXFloat(ctx context.Context, key string, args IncrEXFloatArgs) *IncrEXFloatCmd
	LCS(ctx context.Context, q *LCSQuery) *LCSCmd
	MGet(ctx context.Context, keys ...string) *SliceCmd
	MSet(ctx context.Context, values ...interface{}) *StatusCmd
	MSetNX(ctx context.Context, values ...interface{}) *BoolCmd
	MSetEX(ctx context.Context, args MSetEXArgs, values ...interface{}) *IntCmd
	Set(ctx context.Context, key string, value interface{}, expiration time.Duration) *StatusCmd
	SetArgs(ctx context.Context, key string, value interface{}, a SetArgs) *StatusCmd
	SetEx(ctx context.Context, key string, value interface{}, expiration time.Duration) *StatusCmd
	SetFromBuffer(ctx context.Context, key string, buf []byte) *StatusCmd
	SetIFEQ(ctx context.Context, key string, value interface{}, matchValue interface{}, expiration time.Duration) *StatusCmd
	SetIFEQGet(ctx context.Context, key string, value interface{}, matchValue interface{}, expiration time.Duration) *StringCmd
	SetIFNE(ctx context.Context, key string, value interface{}, matchValue interface{}, expiration time.Duration) *StatusCmd
	SetIFNEGet(ctx context.Context, key string, value interface{}, matchValue interface{}, expiration time.Duration) *StringCmd
	SetIFDEQ(ctx context.Context, key string, value interface{}, matchDigest uint64, expiration time.Duration) *StatusCmd
	SetIFDEQGet(ctx context.Context, key string, value interface{}, matchDigest uint64, expiration time.Duration) *StringCmd
	SetIFDNE(ctx context.Context, key string, value interface{}, matchDigest uint64, expiration time.Duration) *StatusCmd
	SetIFDNEGet(ctx context.Context, key string, value interface{}, matchDigest uint64, expiration time.Duration) *StringCmd
	SetNX(ctx context.Context, key string, value interface{}, expiration time.Duration) *BoolCmd
	SetXX(ctx context.Context, key string, value interface{}, expiration time.Duration) *BoolCmd
	SetRange(ctx context.Context, key string, offset int64, value string) *IntCmd
	StrLen(ctx context.Context, key string) *IntCmd
}

func (c cmdable) Append(ctx context.Context, key, value string) *IntCmd {
	cmd := NewIntCmd(ctx, "append", key, value)
	_ = c(ctx, cmd)
	return cmd
}

func (c cmdable) Decr(ctx context.Context, key string) *IntCmd {
	cmd := NewIntCmd(ctx, "decr", key)
	_ = c(ctx, cmd)
	return cmd
}

func (c cmdable) DecrBy(ctx context.Context, key string, decrement int64) *IntCmd {
	cmd := NewIntCmd(ctx, "decrby", key, decrement)
	_ = c(ctx, cmd)
	return cmd
}

// DelExArgs provides arguments for the DelExArgs function.
type DelExArgs struct {
	// Mode can be `IFEQ`, `IFNE`, `IFDEQ`, or `IFDNE`.
	Mode string

	// MatchValue is used with IFEQ/IFNE modes for compare-and-delete operations.
	// - IFEQ: only delete if current value equals MatchValue
	// - IFNE: only delete if current value does not equal MatchValue
	MatchValue interface{}

	// MatchDigest is used with IFDEQ/IFDNE modes for digest-based compare-and-delete.
	// - IFDEQ: only delete if current value's digest equals MatchDigest
	// - IFDNE: only delete if current value's digest does not equal MatchDigest
	//
	// The digest is a uint64 xxh3 hash value.
	//
	// For examples of client-side digest generation, see:
	// example/digest-optimistic-locking/
	MatchDigest uint64
}

// DelExArgs Redis `DELEX key [IFEQ|IFNE|IFDEQ|IFDNE] match-value` command.
// Compare-and-delete with flexible conditions.
//
// Returns the number of keys that were removed (0 or 1).
//
// NOTE DelExArgs is still experimental
// it's signature and behaviour may change
func (c cmdable) DelExArgs(ctx context.Context, key string, a DelExArgs) *IntCmd {
	args := []interface{}{"delex", key}

	if a.Mode != "" {
		args = append(args, a.Mode)

		// Add match value/digest based on mode
		switch a.Mode {
		case "ifeq", "IFEQ", "ifne", "IFNE":
			if a.MatchValue != nil {
				args = append(args, a.MatchValue)
			}
		case "ifdeq", "IFDEQ", "ifdne", "IFDNE":
			if a.MatchDigest != 0 {
				args = append(args, fmt.Sprintf("%016x", a.MatchDigest))
			}
		}
	}

	cmd := NewIntCmd(ctx, args...)
	_ = c(ctx, cmd)
	return cmd
}

// Digest returns the xxh3 hash (uint64) of the specified key's value.
//
// The digest is a 64-bit xxh3 hash that can be used for optimistic locking
// with SetIFDEQ, SetIFDNE, and DelExArgs commands.
//
// For examples of client-side digest generation and usage patterns, see:
// example/digest-optimistic-locking/
//
// Redis 8.4+. See https://redis.io/commands/digest/
//
// NOTE Digest is still experimental
// it's signature and behaviour may change
func (c cmdable) Digest(ctx context.Context, key string) *DigestCmd {
	cmd := NewDigestCmd(ctx, "digest", key)
	_ = c(ctx, cmd)
	return cmd
}

// Get Redis `GET key` command. It returns redis.Nil error when key does not exist.
func (c cmdable) Get(ctx context.Context, key string) *StringCmd {
	cmd := NewStringCmd(ctx, "get", key)
	_ = c(ctx, cmd)
	return cmd
}

func (c cmdable) GetRange(ctx context.Context, key string, start, end int64) *StringCmd {
	cmd := NewStringCmd(ctx, "getrange", key, start, end)
	_ = c(ctx, cmd)
	return cmd
}

// GetSet returns the old value stored at key and sets it to the new value.
//
// Deprecated: Use SetArgs with Get option instead as of Redis 6.2.0.
func (c cmdable) GetSet(ctx context.Context, key string, value interface{}) *StringCmd {
	cmd := NewStringCmd(ctx, "getset", key, value)
	_ = c(ctx, cmd)
	return cmd
}

// GetEx An expiration of zero removes the TTL associated with the key (i.e. GETEX key persist).
// Requires Redis >= 6.2.0.
func (c cmdable) GetEx(ctx context.Context, key string, expiration time.Duration) *StringCmd {
	args := make([]interface{}, 0, 4)
	args = append(args, "getex", key)
	if expiration > 0 {
		if usePrecise(expiration) {
			args = append(args, "px", formatMs(ctx, expiration))
		} else {
			args = append(args, "ex", formatSec(ctx, expiration))
		}
	} else if expiration == 0 {
		args = append(args, "persist")
	}

	cmd := NewStringCmd(ctx, args...)
	_ = c(ctx, cmd)
	return cmd
}

// GetDel redis-server version >= 6.2.0.
func (c cmdable) GetDel(ctx context.Context, key string) *StringCmd {
	cmd := NewStringCmd(ctx, "getdel", key)
	_ = c(ctx, cmd)
	return cmd
}

// GetToBuffer executes GET and reads the reply directly into buf, avoiding
// the intermediate string allocation that *StringCmd would produce. For values
// larger than the connection's read buffer, the payload is read straight from
// the socket into buf — effectively zero-copy on the receive path.
//
// The returned *ZeroCopyStringCmd reports the number of bytes read via Val(),
// the populated slice via Bytes() (which is buf[:Val()]), and any error
// (including redis.Nil when the key does not exist) via Err(). If buf is too
// small to hold the value, Err() returns a "buffer too small" error.
//
// Nothing is ever written past len(buf). When len(buf) >= value length + 2,
// the read takes a fast path that pulls the payload and the protocol's
// trailing CRLF in a single socket read, using the two bytes after the
// payload as scratch — size buffers with 2 spare bytes to opt in (see
// example/zerocopy-buffer).
//
// This command opts out of automatic retries because partial data from a
// failed attempt would already be sitting in the caller's buffer.
func (c cmdable) GetToBuffer(ctx context.Context, key string, buf []byte) *ZeroCopyStringCmd {
	cmd := NewZeroCopyStringCmd(ctx, buf, "get", key)
	_ = c(ctx, cmd)
	return cmd
}

func (c cmdable) Incr(ctx context.Context, key string) *IntCmd {
	cmd := NewIntCmd(ctx, "incr", key)
	_ = c(ctx, cmd)
	return cmd
}

func (c cmdable) IncrBy(ctx context.Context, key string, value int64) *IntCmd {
	cmd := NewIntCmd(ctx, "incrby", key, value)
	_ = c(ctx, cmd)
	return cmd
}

func (c cmdable) IncrByFloat(ctx context.Context, key string, value float64) *FloatCmd {
	cmd := NewFloatCmd(ctx, "incrbyfloat", key, value)
	_ = c(ctx, cmd)
	return cmd
}

// IncrEXIntArgs are the arguments to IncrEXInt (the BYINT variant of INCREX).
//
// If By is zero and HasBy is false, the server increments by 1.
// HasLBound/HasUBound gate the optional LBOUND/UBOUND clauses so that 0 is a
// valid bound. Expiration is shared with the SET command via ExpirationOption.
type IncrEXIntArgs struct {
	By    int64
	HasBy bool

	LBound, UBound       int64
	HasLBound, HasUBound bool

	// Saturate clamps the result to LBOUND/UBOUND (or LLONG_MAX/MIN when no
	// explicit bound is given) when the increment would exceed it. Without
	// this flag, out-of-bounds operations are rejected: the key and TTL are
	// left unchanged and the reply is [current_value, 0].
	Saturate bool

	// Expiration sets the TTL semantics: EX, PX, EXAT, PXAT, or PERSIST.
	Expiration *ExpirationOption

	// ENX applies the expiration only when the key does not already have an
	// expiration. Requires Expiration to set one of EX/PX/EXAT/PXAT.
	ENX bool
}

// IncrEXFloatArgs are the arguments to IncrEXFloat (the BYFLOAT variant of
// INCREX). BYFLOAT is always sent — even when By is zero — to keep the
// operation in float mode on the server side; omitting BYFLOAT would cause
// the server to treat the call as an integer increment by 1.
// HasLBound/HasUBound gate the optional LBOUND/UBOUND clauses so that 0 is
// a valid bound.
type IncrEXFloatArgs struct {
	By float64

	LBound, UBound       float64
	HasLBound, HasUBound bool

	// Saturate clamps the result to LBOUND/UBOUND (or ±LDBL_MAX when no
	// explicit bound is given) when the increment would exceed it. Without
	// this flag, out-of-bounds operations are rejected: the key and TTL are
	// left unchanged and the reply is [current_value, 0].
	Saturate bool

	Expiration *ExpirationOption

	ENX bool
}

// IncrEXInt Redis `INCREX key [BYINT amount] [LBOUND value] [UBOUND value]
// [SATURATE] [EX seconds | PX ms | EXAT ts | PXAT ts | PERSIST] [ENX]`
// command.
//
// Atomically increments the integer value stored at key, optionally
// constraining the result to a range and applying expiration semantics.
// Returns the new value and the increment that was actually applied. When
// the increment would exceed LBOUND/UBOUND and SATURATE is not set, the key
// and TTL are left unchanged and the reply is [current_value, 0].
//
// Available since Redis 8.8.
// For more information, see https://redis.io/commands/increx
func (c cmdable) IncrEXInt(ctx context.Context, key string, a IncrEXIntArgs) *IncrEXIntCmd {
	args := make([]interface{}, 0, 14)
	args = append(args, "increx", key)
	if a.HasBy {
		args = append(args, "byint", a.By)
	}
	if a.HasLBound {
		args = append(args, "lbound", a.LBound)
	}
	if a.HasUBound {
		args = append(args, "ubound", a.UBound)
	}
	if a.Saturate {
		args = append(args, "saturate")
	}
	args = appendIncrEXTail(args, a.Expiration, a.ENX)

	cmd := NewIncrEXIntCmd(ctx, args...)
	_ = c(ctx, cmd)
	return cmd
}

// IncrEXFloat Redis `INCREX key [BYFLOAT amount] [LBOUND value] [UBOUND value]
// [SATURATE] [EX seconds | PX ms | EXAT ts | PXAT ts | PERSIST] [ENX]`
// command.
//
// Available since Redis 8.8.
// For more information, see https://redis.io/commands/increx
func (c cmdable) IncrEXFloat(ctx context.Context, key string, a IncrEXFloatArgs) *IncrEXFloatCmd {
	args := make([]interface{}, 0, 14)
	args = append(args, "increx", key, "byfloat", a.By)
	if a.HasLBound {
		args = append(args, "lbound", a.LBound)
	}
	if a.HasUBound {
		args = append(args, "ubound", a.UBound)
	}
	if a.Saturate {
		args = append(args, "saturate")
	}
	args = appendIncrEXTail(args, a.Expiration, a.ENX)

	cmd := NewIncrEXFloatCmd(ctx, args...)
	_ = c(ctx, cmd)
	return cmd
}

func appendIncrEXTail(args []interface{}, exp *ExpirationOption, enx bool) []interface{} {
	if exp != nil {
		switch exp.Mode {
		case EX, PX, EXAT, PXAT:
			args = append(args, strings.ToLower(string(exp.Mode)), exp.Value)
		case PERSIST:
			args = append(args, "persist")
		}
	}
	if enx {
		args = append(args, "enx")
	}
	return args
}

type SetCondition string

const (
	// NX only set the keys and their expiration if none exist
	NX SetCondition = "NX"
	// XX only set the keys and their expiration if all already exist
	XX SetCondition = "XX"
)

type ExpirationMode string

const (
	// EX sets expiration in seconds
	EX ExpirationMode = "EX"
	// PX sets expiration in milliseconds
	PX ExpirationMode = "PX"
	// EXAT sets expiration as Unix timestamp in seconds
	EXAT ExpirationMode = "EXAT"
	// PXAT sets expiration as Unix timestamp in milliseconds
	PXAT ExpirationMode = "PXAT"
	// KEEPTTL keeps the existing TTL
	KEEPTTL ExpirationMode = "KEEPTTL"
	// PERSIST removes the existing TTL. Used by INCREX.
	PERSIST ExpirationMode = "PERSIST"
)

type ExpirationOption struct {
	Mode  ExpirationMode
	Value int64
}

func (c cmdable) LCS(ctx context.Context, q *LCSQuery) *LCSCmd {
	cmd := NewLCSCmd(ctx, q)
	_ = c(ctx, cmd)
	return cmd
}

func (c cmdable) MGet(ctx context.Context, keys ...string) *SliceCmd {
	args := make([]interface{}, 1+len(keys))
	args[0] = "mget"
	for i, key := range keys {
		args[1+i] = key
	}
	cmd := NewSliceCmd(ctx, args...)
	_ = c(ctx, cmd)
	return cmd
}

// MSet is like Set but accepts multiple values:
//   - MSet("key1", "value1", "key2", "value2")
//   - MSet([]string{"key1", "value1", "key2", "value2"})
//   - MSet(map[string]interface{}{"key1": "value1", "key2": "value2"})
//   - MSet(struct), For struct types, see HSet description.
func (c cmdable) MSet(ctx context.Context, values ...interface{}) *StatusCmd {
	args := make([]interface{}, 1, 1+len(values))
	args[0] = "mset"
	args = appendArgs(args, values)
	cmd := NewStatusCmd(ctx, args...)
	_ = c(ctx, cmd)
	return cmd
}

// MSetNX is like SetNX but accepts multiple values:
//   - MSetNX("key1", "value1", "key2", "value2")
//   - MSetNX([]string{"key1", "value1", "key2", "value2"})
//   - MSetNX(map[string]interface{}{"key1": "value1", "key2": "value2"})
//   - MSetNX(struct), For struct types, see HSet description.
func (c cmdable) MSetNX(ctx context.Context, values ...interface{}) *BoolCmd {
	args := make([]interface{}, 1, 1+len(values))
	args[0] = "msetnx"
	args = appendArgs(args, values)
	cmd := NewBoolCmd(ctx, args...)
	_ = c(ctx, cmd)
	return cmd
}

type MSetEXArgs struct {
	Condition  SetCondition
	Expiration *ExpirationOption
}

// MSetEX sets the given keys to their respective values.
// This command is an extension of the MSETNX that adds expiration and XX options.
// Available since Redis 8.4
// Important: When this method is used with Cluster clients, all keys
// must be in the same hash slot, otherwise CROSSSLOT error will be returned.
// For more information, see https://redis.io/commands/msetex
func (c cmdable) MSetEX(ctx context.Context, args MSetEXArgs, values ...interface{}) *IntCmd {
	expandedArgs := appendArgs([]interface{}{}, values)
	numkeys := len(expandedArgs) / 2

	cmdArgs := make([]interface{}, 0, 2+len(expandedArgs)+3)
	cmdArgs = append(cmdArgs, "msetex", numkeys)
	cmdArgs = append(cmdArgs, expandedArgs...)

	if args.Condition != "" {
		cmdArgs = append(cmdArgs, string(args.Condition))
	}

	if args.Expiration != nil {
		switch args.Expiration.Mode {
		case EX:
			cmdArgs = append(cmdArgs, "ex", args.Expiration.Value)
		case PX:
			cmdArgs = append(cmdArgs, "px", args.Expiration.Value)
		case EXAT:
			cmdArgs = append(cmdArgs, "exat", args.Expiration.Value)
		case PXAT:
			cmdArgs = append(cmdArgs, "pxat", args.Expiration.Value)
		case KEEPTTL:
			cmdArgs = append(cmdArgs, "keepttl")
		}
	}

	cmd := NewIntCmd(ctx, cmdArgs...)
	_ = c(ctx, cmd)
	return cmd
}

// Set Redis `SET key value [expiration]` command.
// Use expiration for `SETEx`-like behavior.
//
// Zero expiration means the key has no expiration time.
// KeepTTL is a Redis KEEPTTL option to keep existing TTL, it requires your redis-server version >= 6.0,
// otherwise you will receive an error: (error) ERR syntax error.
func (c cmdable) Set(ctx context.Context, key string, value interface{}, expiration time.Duration) *StatusCmd {
	args := make([]interface{}, 3, 5)
	args[0] = "set"
	args[1] = key
	args[2] = value
	if expiration > 0 {
		if usePrecise(expiration) {
			args = append(args, "px", formatMs(ctx, expiration))
		} else {
			args = append(args, "ex", formatSec(ctx, expiration))
		}
	} else if expiration == KeepTTL {
		args = append(args, "keepttl")
	}

	cmd := NewStatusCmd(ctx, args...)
	_ = c(ctx, cmd)
	return cmd
}

// SetArgs provides arguments for the SetArgs function.
type SetArgs struct {
	// Mode can be `NX`, `XX`, `IFEQ`, `IFNE`, `IFDEQ`, `IFDNE` or empty.
	Mode string

	// MatchValue is used with IFEQ/IFNE modes for compare-and-set operations.
	// - IFEQ: only set if current value equals MatchValue
	// - IFNE: only set if current value does not equal MatchValue
	MatchValue interface{}

	// MatchDigest is used with IFDEQ/IFDNE modes for digest-based compare-and-set.
	// - IFDEQ: only set if current value's digest equals MatchDigest
	// - IFDNE: only set if current value's digest does not equal MatchDigest
	//
	// The digest is a uint64 xxh3 hash value.
	//
	// For examples of client-side digest generation, see:
	// example/digest-optimistic-locking/
	MatchDigest uint64

	// Zero `TTL` or `Expiration` means that the key has no expiration time.
	TTL      time.Duration
	ExpireAt time.Time

	// When Get is true, the command returns the old value stored at key, or nil when key did not exist.
	Get bool

	// KeepTTL is a Redis KEEPTTL option to keep existing TTL, it requires your redis-server version >= 6.0,
	// otherwise you will receive an error: (error) ERR syntax error.
	KeepTTL bool
}

// SetArgs supports all the options that the SET command supports.
// It is the alternative to the Set function when you want
// to have more control over the options.
func (c cmdable) SetArgs(ctx context.Context, key string, value interface{}, a SetArgs) *StatusCmd {
	args := []interface{}{"set", key, value}

	if a.KeepTTL {
		args = append(args, "keepttl")
	}

	if !a.ExpireAt.IsZero() {
		args = append(args, "exat", a.ExpireAt.Unix())
	}
	if a.TTL > 0 {
		if usePrecise(a.TTL) {
			args = append(args, "px", formatMs(ctx, a.TTL))
		} else {
			args = append(args, "ex", formatSec(ctx, a.TTL))
		}
	}

	if a.Mode != "" {
		args = append(args, a.Mode)

		// Add match value/digest for CAS modes
		switch a.Mode {
		case "ifeq", "IFEQ", "ifne", "IFNE":
			if a.MatchValue != nil {
				args = append(args, a.MatchValue)
			}
		case "ifdeq", "IFDEQ", "ifdne", "IFDNE":
			if a.MatchDigest != 0 {
				args = append(args, fmt.Sprintf("%016x", a.MatchDigest))
			}
		}
	}

	if a.Get {
		args = append(args, "get")
	}

	cmd := NewStatusCmd(ctx, args...)
	_ = c(ctx, cmd)
	return cmd
}

// SetEx sets the value and expiration of a key.
//
// Deprecated: Use Set with expiration instead as of Redis 2.6.12.
func (c cmdable) SetEx(ctx context.Context, key string, value interface{}, expiration time.Duration) *StatusCmd {
	cmd := NewStatusCmd(ctx, "setex", key, formatSec(ctx, expiration), value)
	_ = c(ctx, cmd)
	return cmd
}

// SetFromBuffer executes SET writing the value directly from buf. For values
// larger than the connection's write buffer, bufio.Writer.Write flushes its
// internal buffer (containing the RESP header) and then writes buf straight
// to the socket — effectively zero-copy on the send path.
//
// Expiration is not supported; use Expire separately if a TTL is required.
//
// Note: SetFromBuffer is exposed for API symmetry with GetToBuffer and is
// functionally equivalent to Set(ctx, key, buf, 0) — both dispatch to the
// same []byte case in the RESP writer and produce identical bytes on the
// wire. The zero-copy property on the send path comes from
// bufio.Writer.Write bypassing its internal buffer for large payloads,
// which Set([]byte) gets automatically. Prefer SetFromBuffer in code that
// also uses GetToBuffer so the buffer-based pattern reads coherently;
// otherwise Set(ctx, key, buf, 0) is equally efficient.
func (c cmdable) SetFromBuffer(ctx context.Context, key string, buf []byte) *StatusCmd {
	cmd := NewStatusCmd(ctx, "set", key, buf)
	_ = c(ctx, cmd)
	return cmd
}

// SetNX sets the value of a key only if the key does not exist.
//
// Zero expiration means the key has no expiration time.
// KeepTTL is a Redis KEEPTTL option to keep existing TTL, it requires your redis-server version >= 6.0,
// otherwise you will receive an error: (error) ERR syntax error.
func (c cmdable) SetNX(ctx context.Context, key string, value interface{}, expiration time.Duration) *BoolCmd {
	var cmd *BoolCmd
	switch expiration {
	case 0:
		cmd = NewBoolCmd(ctx, "set", key, value, "nx")
	case KeepTTL:
		cmd = NewBoolCmd(ctx, "set", key, value, "keepttl", "nx")
	default:
		if usePrecise(expiration) {
			cmd = NewBoolCmd(ctx, "set", key, value, "px", formatMs(ctx, expiration), "nx")
		} else {
			cmd = NewBoolCmd(ctx, "set", key, value, "ex", formatSec(ctx, expiration), "nx")
		}
	}

	_ = c(ctx, cmd)
	return cmd
}

// SetXX Redis `SET key value [expiration] XX` command.
//
// Zero expiration means the key has no expiration time.
// KeepTTL is a Redis KEEPTTL option to keep existing TTL, it requires your redis-server version >= 6.0,
// otherwise you will receive an error: (error) ERR syntax error.
func (c cmdable) SetXX(ctx context.Context, key string, value interface{}, expiration time.Duration) *BoolCmd {
	var cmd *BoolCmd
	switch expiration {
	case 0:
		cmd = NewBoolCmd(ctx, "set", key, value, "xx")
	case KeepTTL:
		cmd = NewBoolCmd(ctx, "set", key, value, "keepttl", "xx")
	default:
		if usePrecise(expiration) {
			cmd = NewBoolCmd(ctx, "set", key, value, "px", formatMs(ctx, expiration), "xx")
		} else {
			cmd = NewBoolCmd(ctx, "set", key, value, "ex", formatSec(ctx, expiration), "xx")
		}
	}

	_ = c(ctx, cmd)
	return cmd
}

// SetIFEQ Redis `SET key value [expiration] IFEQ match-value` command.
// Compare-and-set: only sets the value if the current value equals matchValue.
//
// Returns "OK" on success.
// Returns nil if the operation was aborted due to condition not matching.
// Zero expiration means the key has no expiration time.
//
// NOTE SetIFEQ is still experimental
// it's signature and behaviour may change
func (c cmdable) SetIFEQ(ctx context.Context, key string, value interface{}, matchValue interface{}, expiration time.Duration) *StatusCmd {
	args := []interface{}{"set", key, value}

	if expiration > 0 {
		if usePrecise(expiration) {
			args = append(args, "px", formatMs(ctx, expiration))
		} else {
			args = append(args, "ex", formatSec(ctx, expiration))
		}
	} else if expiration == KeepTTL {
		args = append(args, "keepttl")
	}

	args = append(args, "ifeq", matchValue)

	cmd := NewStatusCmd(ctx, args...)
	_ = c(ctx, cmd)
	return cmd
}

// SetIFEQGet Redis `SET key value [expiration] IFEQ match-value GET` command.
// Compare-and-set with GET: only sets the value if the current value equals matchValue,
// and returns the previous value.
//
// Returns the previous value on success.
// Returns nil if the operation was aborted due to condition not matching.
// Zero expiration means the key has no expiration time.
//
// NOTE SetIFEQGet is still experimental
// it's signature and behaviour may change
func (c cmdable) SetIFEQGet(ctx context.Context, key string, value interface{}, matchValue interface{}, expiration time.Duration) *StringCmd {
	args := []interface{}{"set", key, value}

	if expiration > 0 {
		if usePrecise(expiration) {
			args = append(args, "px", formatMs(ctx, expiration))
		} else {
			args = append(args, "ex", formatSec(ctx, expiration))
		}
	} else if expiration == KeepTTL {
		args = append(args, "keepttl")
	}

	args = append(args, "ifeq", matchValue, "get")

	cmd := NewStringCmd(ctx, args...)
	_ = c(ctx, cmd)
	return cmd
}

// SetIFNE Redis `SET key value [expiration] IFNE match-value` command.
// Compare-and-set: only sets the value if the current value does not equal matchValue.
//
// Returns "OK" on success.
// Returns nil if the operation was aborted due to condition not matching.
// Zero expiration means the key has no expiration time.
//
// NOTE SetIFNE is still experimental
// it's signature and behaviour may change
func (c cmdable) SetIFNE(ctx context.Context, key string, value interface{}, matchValue interface{}, expiration time.Duration) *StatusCmd {
	args := []interface{}{"set", key, value}

	if expiration > 0 {
		if usePrecise(expiration) {
			args = append(args, "px", formatMs(ctx, expiration))
		} else {
			args = append(args, "ex", formatSec(ctx, expiration))
		}
	} else if expiration == KeepTTL {
		args = append(args, "keepttl")
	}

	args = append(args, "ifne", matchValue)

	cmd := NewStatusCmd(ctx, args...)
	_ = c(ctx, cmd)
	return cmd
}

// SetIFNEGet Redis `SET key value [expiration] IFNE match-value GET` command.
// Compare-and-set with GET: only sets the value if the current value does not equal matchValue,
// and returns the previous value.
//
// Returns the previous value on success.
// Returns nil if the operation was aborted due to condition not matching.
// Zero expiration means the key has no expiration time.
//
// NOTE SetIFNEGet is still experimental
// it's signature and behaviour may change
func (c cmdable) SetIFNEGet(ctx context.Context, key string, value interface{}, matchValue interface{}, expiration time.Duration) *StringCmd {
	args := []interface{}{"set", key, value}

	if expiration > 0 {
		if usePrecise(expiration) {
			args = append(args, "px", formatMs(ctx, expiration))
		} else {
			args = append(args, "ex", formatSec(ctx, expiration))
		}
	} else if expiration == KeepTTL {
		args = append(args, "keepttl")
	}

	args = append(args, "ifne", matchValue, "get")

	cmd := NewStringCmd(ctx, args...)
	_ = c(ctx, cmd)
	return cmd
}

// SetIFDEQ sets the value only if the current value's digest equals matchDigest.
//
// This is a compare-and-set operation using xxh3 digest for optimistic locking.
// The matchDigest parameter is a uint64 xxh3 hash value.
//
// Returns "OK" on success.
// Returns redis.Nil if the digest doesn't match (value was modified).
// Zero expiration means the key has no expiration time.
//
// For examples of client-side digest generation and usage patterns, see:
// example/digest-optimistic-locking/
//
// Redis 8.4+. See https://redis.io/commands/set/
//
// NOTE SetIFNEQ is still experimental
// it's signature and behaviour may change
func (c cmdable) SetIFDEQ(ctx context.Context, key string, value interface{}, matchDigest uint64, expiration time.Duration) *StatusCmd {
	args := []interface{}{"set", key, value}

	if expiration > 0 {
		if usePrecise(expiration) {
			args = append(args, "px", formatMs(ctx, expiration))
		} else {
			args = append(args, "ex", formatSec(ctx, expiration))
		}
	} else if expiration == KeepTTL {
		args = append(args, "keepttl")
	}

	args = append(args, "ifdeq", fmt.Sprintf("%016x", matchDigest))

	cmd := NewStatusCmd(ctx, args...)
	_ = c(ctx, cmd)
	return cmd
}

// SetIFDEQGet sets the value only if the current value's digest equals matchDigest,
// and returns the previous value.
//
// This is a compare-and-set operation using xxh3 digest for optimistic locking.
// The matchDigest parameter is a uint64 xxh3 hash value.
//
// Returns the previous value on success.
// Returns redis.Nil if the digest doesn't match (value was modified).
// Zero expiration means the key has no expiration time.
//
// For examples of client-side digest generation and usage patterns, see:
// example/digest-optimistic-locking/
//
// Redis 8.4+. See https://redis.io/commands/set/
//
// NOTE SetIFNEQGet is still experimental
// it's signature and behaviour may change
func (c cmdable) SetIFDEQGet(ctx context.Context, key string, value interface{}, matchDigest uint64, expiration time.Duration) *StringCmd {
	args := []interface{}{"set", key, value}

	if expiration > 0 {
		if usePrecise(expiration) {
			args = append(args, "px", formatMs(ctx, expiration))
		} else {
			args = append(args, "ex", formatSec(ctx, expiration))
		}
	} else if expiration == KeepTTL {
		args = append(args, "keepttl")
	}

	args = append(args, "ifdeq", fmt.Sprintf("%016x", matchDigest), "get")

	cmd := NewStringCmd(ctx, args...)
	_ = c(ctx, cmd)
	return cmd
}

// SetIFDNE sets the value only if the current value's digest does NOT equal matchDigest.
//
// This is a compare-and-set operation using xxh3 digest for optimistic locking.
// The matchDigest parameter is a uint64 xxh3 hash value.
//
// Returns "OK" on success (digest didn't match, value was set).
// Returns redis.Nil if the digest matches (value was not modified).
// Zero expiration means the key has no expiration time.
//
// For examples of client-side digest generation and usage patterns, see:
// example/digest-optimistic-locking/
//
// Redis 8.4+. See https://redis.io/commands/set/
//
// NOTE SetIFDNE is still experimental
// it's signature and behaviour may change
func (c cmdable) SetIFDNE(ctx context.Context, key string, value interface{}, matchDigest uint64, expiration time.Duration) *StatusCmd {
	args := []interface{}{"set", key, value}

	if expiration > 0 {
		if usePrecise(expiration) {
			args = append(args, "px", formatMs(ctx, expiration))
		} else {
			args = append(args, "ex", formatSec(ctx, expiration))
		}
	} else if expiration == KeepTTL {
		args = append(args, "keepttl")
	}

	args = append(args, "ifdne", fmt.Sprintf("%016x", matchDigest))

	cmd := NewStatusCmd(ctx, args...)
	_ = c(ctx, cmd)
	return cmd
}

// SetIFDNEGet sets the value only if the current value's digest does NOT equal matchDigest,
// and returns the previous value.
//
// This is a compare-and-set operation using xxh3 digest for optimistic locking.
// The matchDigest parameter is a uint64 xxh3 hash value.
//
// Returns the previous value on success (digest didn't match, value was set).
// Returns redis.Nil if the digest matches (value was not modified).
// Zero expiration means the key has no expiration time.
//
// For examples of client-side digest generation and usage patterns, see:
// example/digest-optimistic-locking/
//
// Redis 8.4+. See https://redis.io/commands/set/
//
// NOTE SetIFDNEGet is still experimental
// it's signature and behaviour may change
func (c cmdable) SetIFDNEGet(ctx context.Context, key string, value interface{}, matchDigest uint64, expiration time.Duration) *StringCmd {
	args := []interface{}{"set", key, value}

	if expiration > 0 {
		if usePrecise(expiration) {
			args = append(args, "px", formatMs(ctx, expiration))
		} else {
			args = append(args, "ex", formatSec(ctx, expiration))
		}
	} else if expiration == KeepTTL {
		args = append(args, "keepttl")
	}

	args = append(args, "ifdne", fmt.Sprintf("%016x", matchDigest), "get")

	cmd := NewStringCmd(ctx, args...)
	_ = c(ctx, cmd)
	return cmd
}

func (c cmdable) SetRange(ctx context.Context, key string, offset int64, value string) *IntCmd {
	cmd := NewIntCmd(ctx, "setrange", key, offset, value)
	_ = c(ctx, cmd)
	return cmd
}

func (c cmdable) StrLen(ctx context.Context, key string) *IntCmd {
	cmd := NewIntCmd(ctx, "strlen", key)
	_ = c(ctx, cmd)
	return cmd
}
