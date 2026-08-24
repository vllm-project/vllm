package redis

import (
	"bytes"
	"strconv"
	"strings"

	"github.com/redis/go-redis/v9/internal/proto"
)

// defaultCacheableCommands is the allow-list of read-only, deterministic
// commands whose responses may be stored in the client-side cache. Keys are
// lowercase to match baseCmd.Name() on the hot path.
var defaultCacheableCommands = map[string]struct{}{
	// String commands
	"get": {}, "mget": {}, "getbit": {}, "getrange": {},
	"strlen": {}, "substr": {},
	// Hash commands
	"hget": {}, "hgetall": {}, "hmget": {},
	"hkeys": {}, "hvals": {}, "hlen": {},
	"hexists": {}, "hstrlen": {},
	// List commands
	"lindex": {}, "llen": {}, "lpos": {}, "lrange": {},
	// Set commands
	"scard": {}, "sismember": {}, "smembers": {}, "smismember": {},
	"sdiff": {}, "sinter": {}, "sintercard": {}, "sunion": {},
	// Sorted-set commands
	"zcard": {}, "zcount": {}, "zlexcount": {}, "zmscore": {},
	"zrange": {}, "zrangebylex": {}, "zrangebyscore": {},
	"zrank": {}, "zrevrange": {}, "zrevrangebylex": {},
	"zrevrangebyscore": {}, "zrevrank": {}, "zscore": {},
	"zdiff": {}, "zinter": {}, "zunion": {},
	// Bit commands
	"bitcount": {}, "bitfield_ro": {}, "bitpos": {},
	// Key/generic commands
	"exists": {}, "type": {}, "sort_ro": {}, "lcs": {},
	// Geo commands
	"geodist": {}, "geohash": {}, "geopos": {}, "geosearch": {},
	"georadiusbymember_ro": {}, "georadius_ro": {},
	// Stream commands. XREAD is deliberately excluded: it supports BLOCK, and
	// its $/+ IDs are state-relative, so identical args are not deterministic.
	// XPENDING is excluded for the same class of reason: its extended form
	// returns wall-clock-relative idle times and its IDLE filter is
	// time-dependent, so identical args yield different correct results with
	// no key modification (and therefore no invalidation).
	"xlen": {}, "xrange": {}, "xrevrange": {},
	// JSON (RedisJSON) commands
	"json.get": {}, "json.mget": {}, "json.arrindex": {}, "json.arrlen": {},
	"json.objkeys": {}, "json.objlen": {}, "json.resp": {},
	"json.strlen": {}, "json.type": {},
	// TimeSeries commands
	"ts.get": {}, "ts.info": {}, "ts.range": {}, "ts.revrange": {},
}

// isCacheable reports whether cmd is eligible for client-side caching: its
// name is on the allow-list and it operates on at least one key.
func isCacheable(cmd Cmder) bool {
	// Commands such as RawWriteToCmd stream replies directly to an io.Writer.
	// Capturing their replies for CSC would buffer the entire response first,
	// defeating their streaming and allocation guarantees.
	if cmd.NoRetry() {
		return false
	}
	if _, ok := defaultCacheableCommands[cmd.Name()]; !ok {
		return false
	}
	// SORT_RO ... BY/GET reads pattern keys that extractRedisKeys can't
	// enumerate, so its invalidations would be dropped and the result go stale.
	// Plain SORT_RO is fine.
	if cmd.Name() == "sort_ro" && sortROHasByGet(cmd) {
		return false
	}
	return cmdFirstKeyPosWithInfo(cmd, nil) != 0
}

// sortROHasByGet reports whether a SORT_RO invocation uses BY or GET
// (case-insensitive), scanning past the command name and key. stringArg
// normalizes string, *string, and []byte tokens.
func sortROHasByGet(cmd Cmder) bool {
	for i := 2; i < len(cmd.Args()); i++ {
		if s := cmd.stringArg(i); strings.EqualFold(s, "by") || strings.EqualFold(s, "get") {
			return true
		}
	}
	return false
}

// isClientTrackingCmd reports whether cmd is a CLIENT TRACKING subcommand (any
// mode: ON, OFF, or with options). Name and stringArg normalize string,
// *string, and []byte arguments.
func isClientTrackingCmd(cmd Cmder) bool {
	return cmd.Name() == "client" && strings.EqualFold(cmd.stringArg(1), "tracking")
}

// isSelectCmd reports whether cmd changes the selected database on its
// connection. CSC keys are namespaced with Options.DB, so a runtime SELECT
// would make the connection's actual database diverge from the cache namespace.
func isSelectCmd(cmd Cmder) bool {
	return cmd.Name() == "select"
}

// isAuthCmd reports whether cmd changes the authenticated user on its
// connection. The cache namespace is fixed from Options.Username, so runtime
// authentication would make the connection identity diverge from it.
func isAuthCmd(cmd Cmder) bool {
	return cmd.Name() == "auth"
}

// isProtocolChangingHelloCmd reports whether HELLO includes a protocol version
// (and can therefore switch a tracked RESP3 connection to RESP2). A bare HELLO
// only reports connection properties and is safe.
func isProtocolChangingHelloCmd(cmd Cmder) bool {
	return cmd.Name() == "hello" && len(cmd.Args()) > 1
}

// isResetCmd reports whether cmd resets all server-side connection state.
// RESET disables tracking, switches to RESP2, deauthenticates, and changes
// other state that a pooled CSC connection relies on.
func isResetCmd(cmd Cmder) bool {
	return cmd.Name() == "reset"
}

// isSubscribeCmd reports whether a raw command would turn an ordinary pooled
// connection into a Pub/Sub connection. Pub/Sub pushes are deliberately left
// for the dedicated PubSub reader, so the CSC drainer cannot safely own such a
// connection.
func isSubscribeCmd(cmd Cmder) bool {
	switch cmd.Name() {
	case "subscribe", "psubscribe", "ssubscribe":
		return true
	default:
		return false
	}
}

// buildCacheKey returns the RESP-encoded form of the command's argument list,
// used as a collision-free canonical cache key. ok is false when the writer
// cannot marshal the arguments, in which case the caller must skip caching
// rather than bucket the command under an empty key.
func buildCacheKey(cmd Cmder) (string, bool) {
	args := cmd.Args()
	if len(args) == 0 {
		return "", false
	}
	var buf bytes.Buffer
	if err := proto.NewWriter(&buf).WriteArgs(args); err != nil {
		return "", false
	}
	return buf.String(), true
}

// keyArg renders the key argument at pos exactly as proto.Writer sends it to
// the server, so invalidation lookups match the key names in the server's
// "invalidate" pushes. Only types whose stringArg rendering is byte-identical
// to the wire encoding are accepted (fmt.Sprint of any integer matches the
// writer's base-10 strconv output); for anything else — pointers, bools,
// times, durations, floats, BinaryMarshaler values — the rendering can
// diverge, the invalidation would never match, and the entry would be served
// stale forever, so ok=false and the caller skips caching (see processCached).
func keyArg(cmd Cmder, pos int) (string, bool) {
	args := cmd.Args()
	if pos < 0 || pos >= len(args) {
		return "", false
	}
	switch args[pos].(type) {
	case string, []byte,
		int, int8, int16, int32, int64,
		uint, uint8, uint16, uint32, uint64:
		return cmd.stringArg(pos), true
	}
	return "", false
}

// extractRedisKeys returns the Redis key arguments from cmd. The result lets
// the cache map incoming invalidations back to affected entries. Returns nil
// (caller skips caching) when any key
// argument cannot be rendered in its wire form (see keyArg).
func extractRedisKeys(cmd Cmder) []string {
	firstKey := cmdFirstKeyPosWithInfo(cmd, nil)
	if firstKey == 0 {
		return nil
	}

	argsLen := len(cmd.Args())
	if firstKey >= argsLen {
		return nil
	}

	switch cmd.Name() {
	// All remaining args from firstKeyPos are keys.
	case "mget", "exists", "sdiff", "sinter", "sunion":
		keys := make([]string, 0, argsLen-firstKey)
		for i := firstKey; i < argsLen; i++ {
			k, ok := keyArg(cmd, i)
			if !ok {
				return nil
			}
			keys = append(keys, k)
		}
		return keys

	// Numkeys pattern: numkeys at args[1], keys from args[2].
	case "sintercard", "zdiff", "zinter", "zunion":
		if argsLen < 3 {
			return nil
		}
		numKeys, err := strconv.Atoi(cmd.stringArg(1))
		if err != nil || numKeys <= 0 {
			return nil
		}
		keys := make([]string, 0, numKeys)
		for i := 2; i < 2+numKeys && i < argsLen; i++ {
			k, ok := keyArg(cmd, i)
			if !ok {
				return nil
			}
			keys = append(keys, k)
		}
		return keys

	// LCS: exactly two consecutive keys starting at firstKeyPos.
	case "lcs":
		if firstKey+1 >= argsLen {
			return nil
		}
		k1, ok1 := keyArg(cmd, firstKey)
		k2, ok2 := keyArg(cmd, firstKey+1)
		if !ok1 || !ok2 {
			return nil
		}
		return []string{k1, k2}

	// JSON.MGET: keys from firstKeyPos to second-to-last (last arg is the
	// JSON path, not a key).
	case "json.mget":
		lastKey := argsLen - 2
		if lastKey < firstKey {
			return nil
		}
		keys := make([]string, 0, lastKey-firstKey+1)
		for i := firstKey; i <= lastKey; i++ {
			k, ok := keyArg(cmd, i)
			if !ok {
				return nil
			}
			keys = append(keys, k)
		}
		return keys
	}

	// Single key at firstKeyPos (GET, HGET, LRANGE, ...).
	k, ok := keyArg(cmd, firstKey)
	if !ok {
		return nil
	}
	return []string{k}
}
