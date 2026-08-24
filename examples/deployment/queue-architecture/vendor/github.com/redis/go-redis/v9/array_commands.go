package redis

import (
	"context"
)

// note: the APIs is experimental and may be subject to change.
//
// ArrayCmdable defines the interface for Redis Array data structure commands
// available in Redis 8.8.0+.
//
// Redis array supports index range [0, math.MaxUint64-1), so index parameters use uint64.
type ArrayCmdable interface {
	ARSet(ctx context.Context, key string, index uint64, values ...string) *IntCmd
	ARGet(ctx context.Context, key string, index uint64) *StringCmd
	ARGetRange(ctx context.Context, key string, start, end uint64) *SliceCmd
	ARMGet(ctx context.Context, key string, indexes ...uint64) *SliceCmd
	ARMSet(ctx context.Context, key string, members ...AREntry) *IntCmd
	ARInsert(ctx context.Context, key string, values ...string) *UintCmd
	ARDel(ctx context.Context, key string, indexes ...uint64) *IntCmd
	ARDelRange(ctx context.Context, key string, ranges ...ARRange) *UintCmd
	ARLen(ctx context.Context, key string) *UintCmd
	ARCount(ctx context.Context, key string) *UintCmd
	ARNext(ctx context.Context, key string) *UintCmd
	ARSeek(ctx context.Context, key string, index uint64) *IntCmd
	ARInfo(ctx context.Context, key string) *MapStringInterfaceCmd
	ARInfoFull(ctx context.Context, key string) *MapStringInterfaceCmd
	ARScan(ctx context.Context, key string, start, end uint64, args *ARScanArgs) *AREntrySliceCmd
	AROpSum(ctx context.Context, key string, start, end uint64) *StringCmd
	AROpMin(ctx context.Context, key string, start, end uint64) *StringCmd
	AROpMax(ctx context.Context, key string, start, end uint64) *StringCmd
	AROpAnd(ctx context.Context, key string, start, end uint64) *IntCmd
	AROpOr(ctx context.Context, key string, start, end uint64) *IntCmd
	AROpXor(ctx context.Context, key string, start, end uint64) *IntCmd
	AROpMatch(ctx context.Context, key string, start, end uint64, value string) *IntCmd
	AROpUsed(ctx context.Context, key string, start, end uint64) *IntCmd
	ARGrep(ctx context.Context, key string, start, end string, args *ARGrepArgs) *UintSliceCmd
	ARGrepWithValues(ctx context.Context, key string, start, end string, args *ARGrepArgs) *AREntrySliceCmd
	ARRing(ctx context.Context, key string, size uint64, values ...string) *UintCmd
	ARLastItems(ctx context.Context, key string, count uint64, rev bool) *SliceCmd
}

// AREntry represents an index-value pair for ARMSET.
type AREntry struct {
	Index uint64
	Value string
}

// ARRange represents a start-end range for ARDELRANGE.
type ARRange struct {
	Start uint64
	End   uint64
}

// ARScanArgs contains optional arguments for ARSCAN.
type ARScanArgs struct {
	Limit uint64
}

// ARGrepPredicateType defines the type of predicate for ARGREP.
type ARGrepPredicateType string

const (
	ARGrepExact ARGrepPredicateType = "EXACT"
	ARGrepMatch ARGrepPredicateType = "MATCH"
	ARGrepGlob  ARGrepPredicateType = "GLOB"
	ARGrepRegex ARGrepPredicateType = "RE"
)

// ARGrepPredicate represents a search predicate for ARGREP.
type ARGrepPredicate struct {
	Type  ARGrepPredicateType
	Value string
}

// ARGrepArgs contains optional arguments for ARGREP.
// Redis ARGREP defaults to OR when multiple predicates are given.
// Set CombineAnd to true to combine predicates with AND instead.
type ARGrepArgs struct {
	Predicates []ARGrepPredicate
	CombineAnd bool
	Limit      uint64
	NoCase     bool
}

// ARSet sets one or more contiguous values starting at an index in an array.
// Returns the number of new slots that were set (previously empty).
func (c cmdable) ARSet(ctx context.Context, key string, index uint64, values ...string) *IntCmd {
	args := make([]any, 3, 3+len(values))
	args[0] = "arset"
	args[1] = key
	args[2] = index
	for _, v := range values {
		args = append(args, v)
	}
	cmd := NewIntCmd(ctx, args...)
	_ = c(ctx, cmd)
	return cmd
}

// ARGet gets the value at an index in an array.
// Returns redis.Nil if the key or index does not exist.
func (c cmdable) ARGet(ctx context.Context, key string, index uint64) *StringCmd {
	cmd := NewStringCmd(ctx, "arget", key, index)
	_ = c(ctx, cmd)
	return cmd
}

// ARGetRange gets values in a range of indexes.
// Returns values in the range, with nil for unset indexes.
func (c cmdable) ARGetRange(ctx context.Context, key string, start, end uint64) *SliceCmd {
	cmd := NewSliceCmd(ctx, "argetrange", key, start, end)
	_ = c(ctx, cmd)
	return cmd
}

// ARMGet gets values at multiple indexes in an array.
// Returns values at the specified indexes, with nil for unset indexes.
func (c cmdable) ARMGet(ctx context.Context, key string, indexes ...uint64) *SliceCmd {
	args := make([]any, 2+len(indexes))
	args[0] = "armget"
	args[1] = key
	for i, idx := range indexes {
		args[2+i] = idx
	}
	cmd := NewSliceCmd(ctx, args...)
	_ = c(ctx, cmd)
	return cmd
}

// ARMSet sets multiple index-value pairs in an array.
// Returns the number of new slots that were set (previously empty).
func (c cmdable) ARMSet(ctx context.Context, key string, members ...AREntry) *IntCmd {
	args := make([]any, 2, 2+2*len(members))
	args[0] = "armset"
	args[1] = key
	for _, m := range members {
		args = append(args, m.Index, m.Value)
	}
	cmd := NewIntCmd(ctx, args...)
	_ = c(ctx, cmd)
	return cmd
}

// ARInsert inserts one or more values at consecutive indexes.
// Returns the last index where a value was inserted.
func (c cmdable) ARInsert(ctx context.Context, key string, values ...string) *UintCmd {
	args := make([]any, 2, 2+len(values))
	args[0] = "arinsert"
	args[1] = key
	for _, v := range values {
		args = append(args, v)
	}
	cmd := NewUintCmd(ctx, args...)
	_ = c(ctx, cmd)
	return cmd
}

// ARDel deletes elements at the specified indexes in an array.
// Returns the number of elements deleted.
func (c cmdable) ARDel(ctx context.Context, key string, indexes ...uint64) *IntCmd {
	args := make([]any, 2+len(indexes))
	args[0] = "ardel"
	args[1] = key
	for i, idx := range indexes {
		args[2+i] = idx
	}
	cmd := NewIntCmd(ctx, args...)
	_ = c(ctx, cmd)
	return cmd
}

// ARDelRange deletes elements in one or more ranges.
// Returns the number of elements deleted.
func (c cmdable) ARDelRange(ctx context.Context, key string, ranges ...ARRange) *UintCmd {
	args := make([]any, 2, 2+2*len(ranges))
	args[0] = "ardelrange"
	args[1] = key
	for _, r := range ranges {
		args = append(args, r.Start, r.End)
	}
	cmd := NewUintCmd(ctx, args...)
	_ = c(ctx, cmd)
	return cmd
}

// ARLen returns the length of an array (max index + 1).
// Returns 0 if the key does not exist.
func (c cmdable) ARLen(ctx context.Context, key string) *UintCmd {
	cmd := NewUintCmd(ctx, "arlen", key)
	_ = c(ctx, cmd)
	return cmd
}

// ARCount returns the number of non-empty elements in an array.
// Returns 0 if the key does not exist.
func (c cmdable) ARCount(ctx context.Context, key string) *UintCmd {
	cmd := NewUintCmd(ctx, "arcount", key)
	_ = c(ctx, cmd)
	return cmd
}

// ARNext returns the next index ARINSERT would use.
// Returns 0 for missing keys or when no insert happened yet.
// Returns nil when the insertion cursor is exhausted / would overflow.
func (c cmdable) ARNext(ctx context.Context, key string) *UintCmd {
	cmd := NewUintCmd(ctx, "arnext", key)
	_ = c(ctx, cmd)
	return cmd
}

// ARSeek sets the ARINSERT / ARRING cursor to a specific index.
// Returns 1 if the cursor was set, 0 if the key does not exist.
func (c cmdable) ARSeek(ctx context.Context, key string, index uint64) *IntCmd {
	cmd := NewIntCmd(ctx, "arseek", key, index)
	_ = c(ctx, cmd)
	return cmd
}

// ARInfo returns metadata about an array.
func (c cmdable) ARInfo(ctx context.Context, key string) *MapStringInterfaceCmd {
	cmd := NewMapStringInterfaceCmd(ctx, "arinfo", key)
	_ = c(ctx, cmd)
	return cmd
}

// ARInfoFull returns detailed metadata about an array including slice statistics.
func (c cmdable) ARInfoFull(ctx context.Context, key string) *MapStringInterfaceCmd {
	cmd := NewMapStringInterfaceCmd(ctx, "arinfo", key, "full")
	_ = c(ctx, cmd)
	return cmd
}

// ARScan iterates existing elements in a range, returning index-value pairs.
func (c cmdable) ARScan(ctx context.Context, key string, start, end uint64, scanArgs *ARScanArgs) *AREntrySliceCmd {
	args := make([]any, 4, 6)
	args[0], args[1], args[2], args[3] = "arscan", key, start, end
	if scanArgs != nil && scanArgs.Limit > 0 {
		args = append(args, "limit", scanArgs.Limit)
	}
	cmd := NewAREntrySliceCmd(ctx, args...)
	_ = c(ctx, cmd)
	return cmd
}

// AROpSum returns the sum of numeric elements in a range.
func (c cmdable) AROpSum(ctx context.Context, key string, start, end uint64) *StringCmd {
	cmd := NewStringCmd(ctx, "arop", key, start, end, "SUM")
	_ = c(ctx, cmd)
	return cmd
}

// AROpMin returns the minimum numeric element in a range.
func (c cmdable) AROpMin(ctx context.Context, key string, start, end uint64) *StringCmd {
	cmd := NewStringCmd(ctx, "arop", key, start, end, "MIN")
	_ = c(ctx, cmd)
	return cmd
}

// AROpMax returns the maximum numeric element in a range.
func (c cmdable) AROpMax(ctx context.Context, key string, start, end uint64) *StringCmd {
	cmd := NewStringCmd(ctx, "arop", key, start, end, "MAX")
	_ = c(ctx, cmd)
	return cmd
}

// AROpAnd returns the bitwise AND of integer elements in a range.
func (c cmdable) AROpAnd(ctx context.Context, key string, start, end uint64) *IntCmd {
	cmd := NewIntCmd(ctx, "arop", key, start, end, "AND")
	_ = c(ctx, cmd)
	return cmd
}

// AROpOr returns the bitwise OR of integer elements in a range.
func (c cmdable) AROpOr(ctx context.Context, key string, start, end uint64) *IntCmd {
	cmd := NewIntCmd(ctx, "arop", key, start, end, "OR")
	_ = c(ctx, cmd)
	return cmd
}

// AROpXor returns the bitwise XOR of integer elements in a range.
func (c cmdable) AROpXor(ctx context.Context, key string, start, end uint64) *IntCmd {
	cmd := NewIntCmd(ctx, "arop", key, start, end, "XOR")
	_ = c(ctx, cmd)
	return cmd
}

// AROpMatch returns the count of elements matching a target string in a range.
func (c cmdable) AROpMatch(ctx context.Context, key string, start, end uint64, value string) *IntCmd {
	cmd := NewIntCmd(ctx, "arop", key, start, end, "MATCH", value)
	_ = c(ctx, cmd)
	return cmd
}

// AROpUsed returns the count of non-empty slots in a range.
func (c cmdable) AROpUsed(ctx context.Context, key string, start, end uint64) *IntCmd {
	cmd := NewIntCmd(ctx, "arop", key, start, end, "USED")
	_ = c(ctx, cmd)
	return cmd
}

// ARGrep searches array elements in a range using textual predicates.
// Returns matching indexes only. Use ARGrepWithValues to also get the values.
func (c cmdable) ARGrep(ctx context.Context, key string, start, end string, grepArgs *ARGrepArgs) *UintSliceCmd {
	args := make([]any, 4, 4+grepArgs.Len())
	args[0], args[1], args[2], args[3] = "argrep", key, start, end
	args = grepArgs.Append(args)
	cmd := NewUintSliceCmd(ctx, args...)
	_ = c(ctx, cmd)
	return cmd
}

// ARGrepWithValues searches array elements in a range using textual predicates.
// Returns matching indexes and their values as index-value pairs.
func (c cmdable) ARGrepWithValues(ctx context.Context, key string, start, end string, grepArgs *ARGrepArgs) *AREntrySliceCmd {
	args := make([]any, 4, 5+grepArgs.Len())
	args[0], args[1], args[2], args[3] = "argrep", key, start, end
	args = grepArgs.Append(args)
	args = append(args, "withvalues")
	cmd := NewAREntrySliceCmd(ctx, args...)
	_ = c(ctx, cmd)
	return cmd
}

func (args *ARGrepArgs) Len() int {
	if args == nil {
		return 0
	}
	n := 2 * len(args.Predicates)
	if args.CombineAnd {
		n++
	}
	if args.Limit > 0 {
		n += 2
	}
	if args.NoCase {
		n++
	}
	return n
}

func (args *ARGrepArgs) Append(a []any) []any {
	if args == nil {
		return a
	}
	for _, p := range args.Predicates {
		a = append(a, string(p.Type), p.Value)
	}
	if args.CombineAnd {
		a = append(a, "and")
	}
	if args.Limit > 0 {
		a = append(a, "limit", args.Limit)
	}
	if args.NoCase {
		a = append(a, "nocase")
	}
	return a
}

// ARRing inserts values into a ring buffer of specified size, wrapping and truncating as needed.
// Returns the last index where a value was inserted.
func (c cmdable) ARRing(ctx context.Context, key string, size uint64, values ...string) *UintCmd {
	args := make([]any, 3, 3+len(values))
	args[0] = "arring"
	args[1] = key
	args[2] = size
	for _, v := range values {
		args = append(args, v)
	}
	cmd := NewUintCmd(ctx, args...)
	_ = c(ctx, cmd)
	return cmd
}

// ARLastItems returns the most recently inserted elements.
// When rev is true, returns items in reverse order.
func (c cmdable) ARLastItems(ctx context.Context, key string, count uint64, rev bool) *SliceCmd {
	args := make([]any, 3, 4)
	args[0], args[1], args[2] = "arlastitems", key, count
	if rev {
		args = append(args, "rev")
	}
	cmd := NewSliceCmd(ctx, args...)
	_ = c(ctx, cmd)
	return cmd
}
