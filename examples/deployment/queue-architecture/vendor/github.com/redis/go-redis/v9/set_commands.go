package redis

import (
	"context"

	"github.com/redis/go-redis/v9/internal/hashtag"
)

// SetCmdable is an interface for Redis set commands.
// Sets are unordered collections of unique strings.
type SetCmdable interface {
	SAdd(ctx context.Context, key string, members ...interface{}) *IntCmd
	SCard(ctx context.Context, key string) *IntCmd
	SDiff(ctx context.Context, keys ...string) *StringSliceCmd
	SDiffCard(ctx context.Context, opts *SDiffCardOptions, keys ...string) *IntCmd
	SDiffStore(ctx context.Context, destination string, keys ...string) *IntCmd
	SInter(ctx context.Context, keys ...string) *StringSliceCmd
	SInterCard(ctx context.Context, limit int64, keys ...string) *IntCmd
	SInterStore(ctx context.Context, destination string, keys ...string) *IntCmd
	SIsMember(ctx context.Context, key string, member interface{}) *BoolCmd
	SMIsMember(ctx context.Context, key string, members ...interface{}) *BoolSliceCmd
	SMembers(ctx context.Context, key string) *StringSliceCmd
	SMembersMap(ctx context.Context, key string) *StringStructMapCmd
	SMove(ctx context.Context, source, destination string, member interface{}) *BoolCmd
	SPop(ctx context.Context, key string) *StringCmd
	SPopN(ctx context.Context, key string, count int64) *StringSliceCmd
	SRandMember(ctx context.Context, key string) *StringCmd
	SRandMemberN(ctx context.Context, key string, count int64) *StringSliceCmd
	SRem(ctx context.Context, key string, members ...interface{}) *IntCmd
	SScan(ctx context.Context, key string, cursor uint64, match string, count int64) *ScanCmd
	SUnion(ctx context.Context, keys ...string) *StringSliceCmd
	SUnionCard(ctx context.Context, opts *SUnionCardOptions, keys ...string) *IntCmd
	SUnionStore(ctx context.Context, destination string, keys ...string) *IntCmd
}

// SUnionCardOptions are the options for SUnionCard.
type SUnionCardOptions struct {
	Approx bool  // use an approximate (HyperLogLog) count.
	Limit  int64 // cap the result; 0 means no limit.
}

// SDiffCardOptions are the options for SDiffCard.
type SDiffCardOptions struct {
	Limit int64 // cap the result; 0 means no limit.
}

// Returns the number of elements that were added to the set, not including all
// the elements already present in the set.
//
// For more information about the command please refer to [SADD].
//
// [SADD]: (https://redis.io/docs/latest/commands/sadd/)
func (c cmdable) SAdd(ctx context.Context, key string, members ...interface{}) *IntCmd {
	args := make([]interface{}, 2, 2+len(members))
	args[0] = "sadd"
	args[1] = key
	args = appendArgs(args, members)
	cmd := NewIntCmd(ctx, args...)
	_ = c(ctx, cmd)
	return cmd
}

// Returns the set cardinality (number of elements) of the set stored at key.
// Returns 0 if key does not exist.
//
// For more information about the command please refer to [SCARD].
//
// [SCARD]: (https://redis.io/docs/latest/commands/scard/)
func (c cmdable) SCard(ctx context.Context, key string) *IntCmd {
	cmd := NewIntCmd(ctx, "scard", key)
	_ = c(ctx, cmd)
	return cmd
}

// Returns the members of the set resulting from the difference between the first set
// and all the successive sets.
// Keys that do not exist are considered to be empty sets.
//
// For more information about the command please refer to [SDIFF].
//
// [SDIFF]: (https://redis.io/docs/latest/commands/sdiff/)
func (c cmdable) SDiff(ctx context.Context, keys ...string) *StringSliceCmd {
	args := make([]interface{}, 1+len(keys))
	args[0] = "sdiff"
	for i, key := range keys {
		args[1+i] = key
	}
	cmd := NewStringSliceCmd(ctx, args...)
	_ = c(ctx, cmd)
	return cmd
}

// Stores the members of the set resulting from the difference between the first set
// and all the successive sets into destination.
// If destination already exists, it is overwritten.
//
// For more information about the command please refer to [SDIFFSTORE].
//
// [SDIFFSTORE]: (https://redis.io/docs/latest/commands/sdiffstore/)
func (c cmdable) SDiffStore(ctx context.Context, destination string, keys ...string) *IntCmd {
	args := make([]interface{}, 2+len(keys))
	args[0] = "sdiffstore"
	args[1] = destination
	for i, key := range keys {
		args[2+i] = key
	}
	cmd := NewIntCmd(ctx, args...)
	_ = c(ctx, cmd)
	return cmd
}

// Returns the cardinality of the difference of the first set and the rest.
// Missing keys are treated as empty sets.
//
// For more information about the command please refer to [SDIFFCARD].
//
// [SDIFFCARD]: (https://redis.io/docs/latest/commands/sdiffcard/)
func (c cmdable) SDiffCard(ctx context.Context, opts *SDiffCardOptions, keys ...string) *IntCmd {
	if opts == nil {
		opts = &SDiffCardOptions{}
	}
	numKeys := len(keys)
	args := make([]interface{}, 0, 4+numKeys)
	args = append(args, "sdiffcard", numKeys)
	for _, key := range keys {
		args = append(args, key)
	}
	args = append(args, "limit", opts.Limit)
	cmd := NewIntCmd(ctx, args...)
	// Keys start after the numkeys arg: ["sdiffcard", numKeys, key1, ...].
	cmd.SetFirstKeyPos(2)
	_ = c(ctx, cmd)
	return cmd
}

// Returns the members of the set resulting from the intersection of all the given sets.
// Keys that do not exist are considered to be empty sets.
// With one of the keys being an empty set, the resulting set is also empty.
//
// For more information about the command please refer to [SINTER].
//
// [SINTER]: (https://redis.io/docs/latest/commands/sinter/)
func (c cmdable) SInter(ctx context.Context, keys ...string) *StringSliceCmd {
	args := make([]interface{}, 1+len(keys))
	args[0] = "sinter"
	for i, key := range keys {
		args[1+i] = key
	}
	cmd := NewStringSliceCmd(ctx, args...)
	_ = c(ctx, cmd)
	return cmd
}

// Returns the cardinality of the set resulting from the intersection of all the given sets.
// Keys that do not exist are considered to be empty sets.
// With one of the keys being an empty set, the resulting set is also empty.
//
// The limit parameter sets an upper bound on the number of results returned.
// If limit is 0, no limit is applied.
//
// For more information about the command please refer to [SINTERCARD].
//
// [SINTERCARD]: (https://redis.io/docs/latest/commands/sintercard/)
func (c cmdable) SInterCard(ctx context.Context, limit int64, keys ...string) *IntCmd {
	numKeys := len(keys)
	args := make([]interface{}, 4+numKeys)
	args[0] = "sintercard"
	args[1] = numKeys
	for i, key := range keys {
		args[2+i] = key
	}
	args[2+numKeys] = "limit"
	args[3+numKeys] = limit
	cmd := NewIntCmd(ctx, args...)
	_ = c(ctx, cmd)
	return cmd
}

// Stores the members of the set resulting from the intersection of all the given sets
// into destination.
// If destination already exists, it is overwritten.
//
// For more information about the command please refer to [SINTERSTORE].
//
// [SINTERSTORE]: (https://redis.io/docs/latest/commands/sinterstore/)
func (c cmdable) SInterStore(ctx context.Context, destination string, keys ...string) *IntCmd {
	args := make([]interface{}, 2+len(keys))
	args[0] = "sinterstore"
	args[1] = destination
	for i, key := range keys {
		args[2+i] = key
	}
	cmd := NewIntCmd(ctx, args...)
	_ = c(ctx, cmd)
	return cmd
}

// Returns if member is a member of the set stored at key.
// Returns true if the element is a member of the set, false if it is not a member
// or if key does not exist.
//
// For more information about the command please refer to [SISMEMBER].
//
// [SISMEMBER]: (https://redis.io/docs/latest/commands/sismember/)
func (c cmdable) SIsMember(ctx context.Context, key string, member interface{}) *BoolCmd {
	cmd := NewBoolCmd(ctx, "sismember", key, member)
	_ = c(ctx, cmd)
	return cmd
}

// Returns whether each member is a member of the set stored at key.
// For each member, returns true if the element is a member of the set, false if it is not
// a member or if key does not exist.
//
// For more information about the command please refer to [SMISMEMBER].
//
// [SMISMEMBER]: (https://redis.io/docs/latest/commands/smismember/)
func (c cmdable) SMIsMember(ctx context.Context, key string, members ...interface{}) *BoolSliceCmd {
	args := make([]interface{}, 2, 2+len(members))
	args[0] = "smismember"
	args[1] = key
	args = appendArgs(args, members)
	cmd := NewBoolSliceCmd(ctx, args...)
	_ = c(ctx, cmd)
	return cmd
}

// Returns all the members of the set value stored at key.
// Returns an empty slice if key does not exist.
//
// For more information about the command please refer to [SMEMBERS].
//
// [SMEMBERS]: (https://redis.io/docs/latest/commands/smembers/)
func (c cmdable) SMembers(ctx context.Context, key string) *StringSliceCmd {
	cmd := NewStringSliceCmd(ctx, "smembers", key)
	_ = c(ctx, cmd)
	return cmd
}

// Returns all the members of the set value stored at key as a map.
// Returns an empty map if key does not exist.
//
// For more information about the command please refer to [SMEMBERS].
//
// [SMEMBERS]: (https://redis.io/docs/latest/commands/smembers/)
func (c cmdable) SMembersMap(ctx context.Context, key string) *StringStructMapCmd {
	cmd := NewStringStructMapCmd(ctx, "smembers", key)
	_ = c(ctx, cmd)
	return cmd
}

// Moves member from the set at source to the set at destination.
// This operation is atomic. In every given moment the element will appear to be a member
// of source or destination for other clients.
//
// For more information about the command please refer to [SMOVE].
//
// [SMOVE]: (https://redis.io/docs/latest/commands/smove/)
func (c cmdable) SMove(ctx context.Context, source, destination string, member interface{}) *BoolCmd {
	cmd := NewBoolCmd(ctx, "smove", source, destination, member)
	_ = c(ctx, cmd)
	return cmd
}

// Removes and returns one or more random members from the set value stored at key.
// This version returns a single random member.
//
// For more information about the command please refer to [SPOP].
//
// [SPOP]: (https://redis.io/docs/latest/commands/spop/)
func (c cmdable) SPop(ctx context.Context, key string) *StringCmd {
	cmd := NewStringCmd(ctx, "spop", key)
	_ = c(ctx, cmd)
	return cmd
}

// Removes and returns one or more random members from the set value stored at key.
// This version returns up to count random members.
//
// For more information about the command please refer to [SPOP].
//
// [SPOP]: (https://redis.io/docs/latest/commands/spop/)
func (c cmdable) SPopN(ctx context.Context, key string, count int64) *StringSliceCmd {
	cmd := NewStringSliceCmd(ctx, "spop", key, count)
	_ = c(ctx, cmd)
	return cmd
}

// Returns a random member from the set value stored at key.
// This version returns a single random member without removing it.
//
// For more information about the command please refer to [SRANDMEMBER].
//
// [SRANDMEMBER]: (https://redis.io/docs/latest/commands/srandmember/)
func (c cmdable) SRandMember(ctx context.Context, key string) *StringCmd {
	cmd := NewStringCmd(ctx, "srandmember", key)
	_ = c(ctx, cmd)
	return cmd
}

// Returns an array of random members from the set value stored at key.
// This version returns up to count random members without removing them.
// When called with a positive count, returns distinct elements.
// When called with a negative count, allows for repeated elements.
//
// For more information about the command please refer to [SRANDMEMBER].
//
// [SRANDMEMBER]: (https://redis.io/docs/latest/commands/srandmember/)
func (c cmdable) SRandMemberN(ctx context.Context, key string, count int64) *StringSliceCmd {
	cmd := NewStringSliceCmd(ctx, "srandmember", key, count)
	_ = c(ctx, cmd)
	return cmd
}

// Removes the specified members from the set stored at key.
// Specified members that are not a member of this set are ignored.
// If key does not exist, it is treated as an empty set and this command returns 0.
//
// For more information about the command please refer to [SREM].
//
// [SREM]: (https://redis.io/docs/latest/commands/srem/)
func (c cmdable) SRem(ctx context.Context, key string, members ...interface{}) *IntCmd {
	args := make([]interface{}, 2, 2+len(members))
	args[0] = "srem"
	args[1] = key
	args = appendArgs(args, members)
	cmd := NewIntCmd(ctx, args...)
	_ = c(ctx, cmd)
	return cmd
}

// Returns the members of the set resulting from the union of all the given sets.
// Keys that do not exist are considered to be empty sets.
//
// For more information about the command please refer to [SUNION].
//
// [SUNION]: (https://redis.io/docs/latest/commands/sunion/)
func (c cmdable) SUnion(ctx context.Context, keys ...string) *StringSliceCmd {
	args := make([]interface{}, 1+len(keys))
	args[0] = "sunion"
	for i, key := range keys {
		args[1+i] = key
	}
	cmd := NewStringSliceCmd(ctx, args...)
	_ = c(ctx, cmd)
	return cmd
}

// Stores the members of the set resulting from the union of all the given sets
// into destination.
// If destination already exists, it is overwritten.
//
// For more information about the command please refer to [SUNIONSTORE].
//
// [SUNIONSTORE]: (https://redis.io/docs/latest/commands/sunionstore/)
func (c cmdable) SUnionStore(ctx context.Context, destination string, keys ...string) *IntCmd {
	args := make([]interface{}, 2+len(keys))
	args[0] = "sunionstore"
	args[1] = destination
	for i, key := range keys {
		args[2+i] = key
	}
	cmd := NewIntCmd(ctx, args...)
	_ = c(ctx, cmd)
	return cmd
}

// Returns the cardinality of the union of all the given sets.
// Missing keys are treated as empty sets.
//
// For more information about the command please refer to [SUNIONCARD].
//
// [SUNIONCARD]: (https://redis.io/docs/latest/commands/sunioncard/)
func (c cmdable) SUnionCard(ctx context.Context, opts *SUnionCardOptions, keys ...string) *IntCmd {
	if opts == nil {
		opts = &SUnionCardOptions{}
	}
	numKeys := len(keys)
	args := make([]interface{}, 0, 4+numKeys+1)
	args = append(args, "sunioncard", numKeys)
	for _, key := range keys {
		args = append(args, key)
	}
	if opts.Approx {
		args = append(args, "approx")
	}
	args = append(args, "limit", opts.Limit)
	cmd := NewIntCmd(ctx, args...)
	// Keys start after the numkeys arg: ["sunioncard", numKeys, key1, ...].
	cmd.SetFirstKeyPos(2)
	_ = c(ctx, cmd)
	return cmd
}

// Incrementally iterates the set elements stored at key.
// This is a cursor-based iterator that allows scanning large sets efficiently.
//
// Parameters:
//   - cursor: The cursor value for the iteration (use 0 to start a new scan)
//   - match: Optional pattern to match elements (empty string means no pattern)
//   - count: Optional hint about how many elements to return per iteration
//
// For more information about the command please refer to [SSCAN].
//
// [SSCAN]: (https://redis.io/docs/latest/commands/sscan/)
func (c cmdable) SScan(ctx context.Context, key string, cursor uint64, match string, count int64) *ScanCmd {
	args := []interface{}{"sscan", key, cursor}
	if match != "" {
		args = append(args, "match", match)
	}
	if count > 0 {
		args = append(args, "count", count)
	}
	cmd := NewScanCmd(ctx, c, args...)
	if hashtag.Present(match) {
		cmd.SetFirstKeyPos(4)
	}
	_ = c(ctx, cmd)
	return cmd
}
