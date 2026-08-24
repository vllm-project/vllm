package redis

import (
	"context"
	"errors"
)

type BitMapCmdable interface {
	GetBit(ctx context.Context, key string, offset int64) *IntCmd
	SetBit(ctx context.Context, key string, offset int64, value int) *IntCmd
	BitCount(ctx context.Context, key string, bitCount *BitCount) *IntCmd
	BitOpAnd(ctx context.Context, destKey string, keys ...string) *IntCmd
	BitOpOr(ctx context.Context, destKey string, keys ...string) *IntCmd
	BitOpXor(ctx context.Context, destKey string, keys ...string) *IntCmd
	BitOpDiff(ctx context.Context, destKey string, keys ...string) *IntCmd
	BitOpDiff1(ctx context.Context, destKey string, keys ...string) *IntCmd
	BitOpAndOr(ctx context.Context, destKey string, keys ...string) *IntCmd
	BitOpOne(ctx context.Context, destKey string, keys ...string) *IntCmd
	BitOpNot(ctx context.Context, destKey string, key string) *IntCmd
	BitPos(ctx context.Context, key string, bit int64, pos ...int64) *IntCmd
	BitPosSpan(ctx context.Context, key string, bit int8, start, end int64, span string) *IntCmd
	BitField(ctx context.Context, key string, values ...interface{}) *IntSliceCmd
	BitFieldRO(ctx context.Context, key string, values ...interface{}) *IntSliceCmd
}

func (c cmdable) GetBit(ctx context.Context, key string, offset int64) *IntCmd {
	cmd := NewIntCmd(ctx, "getbit", key, offset)
	_ = c(ctx, cmd)
	return cmd
}

func (c cmdable) SetBit(ctx context.Context, key string, offset int64, value int) *IntCmd {
	cmd := NewIntCmd(
		ctx,
		"setbit",
		key,
		offset,
		value,
	)
	_ = c(ctx, cmd)
	return cmd
}

type BitCount struct {
	Start, End int64
	Unit       string // BYTE(default) | BIT
}

const BitCountIndexByte string = "BYTE"
const BitCountIndexBit string = "BIT"

func (c cmdable) BitCount(ctx context.Context, key string, bitCount *BitCount) *IntCmd {
	args := make([]any, 2, 5)
	args[0] = "bitcount"
	args[1] = key
	if bitCount != nil {
		args = append(args, bitCount.Start, bitCount.End)
		if bitCount.Unit != "" {
			if bitCount.Unit != BitCountIndexByte && bitCount.Unit != BitCountIndexBit {
				cmd := NewIntCmd(ctx)
				cmd.SetErr(errors.New("redis: invalid bitcount index"))
				return cmd
			}
			args = append(args, bitCount.Unit)
		}
	}
	cmd := NewIntCmd(ctx, args...)
	_ = c(ctx, cmd)
	return cmd
}

func (c cmdable) bitOp(ctx context.Context, op, destKey string, keys ...string) *IntCmd {
	args := make([]interface{}, 3+len(keys))
	args[0] = "bitop"
	args[1] = op
	args[2] = destKey
	for i, key := range keys {
		args[3+i] = key
	}
	cmd := NewIntCmd(ctx, args...)
	_ = c(ctx, cmd)
	return cmd
}

// BitOpAnd creates a new bitmap in which users are members of all given bitmaps
func (c cmdable) BitOpAnd(ctx context.Context, destKey string, keys ...string) *IntCmd {
	return c.bitOp(ctx, "and", destKey, keys...)
}

// BitOpOr creates a new bitmap in which users are member of at least one given bitmap
func (c cmdable) BitOpOr(ctx context.Context, destKey string, keys ...string) *IntCmd {
	return c.bitOp(ctx, "or", destKey, keys...)
}

// BitOpXor creates a new bitmap in which users are the result of XORing all given bitmaps
func (c cmdable) BitOpXor(ctx context.Context, destKey string, keys ...string) *IntCmd {
	return c.bitOp(ctx, "xor", destKey, keys...)
}

// BitOpNot creates a new bitmap in which users are not members of a given bitmap
func (c cmdable) BitOpNot(ctx context.Context, destKey string, key string) *IntCmd {
	return c.bitOp(ctx, "not", destKey, key)
}

// BitOpDiff creates a new bitmap in which users are members of bitmap X but not of any of bitmaps Y1, Y2, …
// Introduced with Redis 8.2
func (c cmdable) BitOpDiff(ctx context.Context, destKey string, keys ...string) *IntCmd {
	return c.bitOp(ctx, "diff", destKey, keys...)
}

// BitOpDiff1 creates a new bitmap in which users are members of one or more of bitmaps Y1, Y2, … but not members of bitmap X
// Introduced with Redis 8.2
func (c cmdable) BitOpDiff1(ctx context.Context, destKey string, keys ...string) *IntCmd {
	return c.bitOp(ctx, "diff1", destKey, keys...)
}

// BitOpAndOr creates a new bitmap in which users are members of bitmap X and also members of one or more of bitmaps Y1, Y2, …
// Introduced with Redis 8.2
func (c cmdable) BitOpAndOr(ctx context.Context, destKey string, keys ...string) *IntCmd {
	return c.bitOp(ctx, "andor", destKey, keys...)
}

// BitOpOne creates a new bitmap in which users are members of exactly one of the given bitmaps
// Introduced with Redis 8.2
func (c cmdable) BitOpOne(ctx context.Context, destKey string, keys ...string) *IntCmd {
	return c.bitOp(ctx, "one", destKey, keys...)
}

// BitPos is an API before Redis version 7.0, cmd: bitpos key bit start end
// if you need the `byte | bit` parameter, please use `BitPosSpan`.
func (c cmdable) BitPos(ctx context.Context, key string, bit int64, pos ...int64) *IntCmd {
	args := make([]interface{}, 3+len(pos))
	args[0] = "bitpos"
	args[1] = key
	args[2] = bit
	switch len(pos) {
	case 0:
	case 1:
		args[3] = pos[0]
	case 2:
		args[3] = pos[0]
		args[4] = pos[1]
	default:
		cmd := NewIntCmd(ctx)
		cmd.SetErr(errors.New("too many arguments"))
		return cmd
	}
	cmd := NewIntCmd(ctx, args...)
	_ = c(ctx, cmd)
	return cmd
}

// BitPosSpan supports the `byte | bit` parameters in redis version 7.0,
// the bitpos command defaults to using byte type for the `start-end` range,
// which means it counts in bytes from start to end. you can set the value
// of "span" to determine the type of `start-end`.
// span = "bit", cmd: bitpos key bit start end bit
// span = "byte", cmd: bitpos key bit start end byte
func (c cmdable) BitPosSpan(ctx context.Context, key string, bit int8, start, end int64, span string) *IntCmd {
	cmd := NewIntCmd(ctx, "bitpos", key, bit, start, end, span)
	_ = c(ctx, cmd)
	return cmd
}

// BitField accepts multiple values:
//   - BitField("set", "i1", "offset1", "value1","cmd2", "type2", "offset2", "value2")
//   - BitField([]string{"cmd1", "type1", "offset1", "value1","cmd2", "type2", "offset2", "value2"})
//   - BitField([]interface{}{"cmd1", "type1", "offset1", "value1","cmd2", "type2", "offset2", "value2"})
func (c cmdable) BitField(ctx context.Context, key string, values ...interface{}) *IntSliceCmd {
	args := make([]interface{}, 2, 2+len(values))
	args[0] = "bitfield"
	args[1] = key
	args = appendArgs(args, values)
	cmd := NewIntSliceCmd(ctx, args...)
	_ = c(ctx, cmd)
	return cmd
}

// BitFieldRO - Read-only variant of the BITFIELD command.
// It is like the original BITFIELD but only accepts GET subcommand and can safely be used in read-only replicas.
// - BitFieldRO(ctx, key, "<Encoding0>", "<Offset0>", "<Encoding1>","<Offset1>")
func (c cmdable) BitFieldRO(ctx context.Context, key string, values ...interface{}) *IntSliceCmd {
	args := make([]interface{}, 2, 2+len(values))
	args[0] = "BITFIELD_RO"
	args[1] = key
	if len(values)%2 != 0 {
		c := NewIntSliceCmd(ctx)
		c.SetErr(errors.New("BitFieldRO: invalid number of arguments, must be even"))
		return c
	}
	for i := 0; i < len(values); i += 2 {
		args = append(args, "GET", values[i], values[i+1])
	}
	cmd := NewIntSliceCmd(ctx, args...)
	_ = c(ctx, cmd)
	return cmd
}
