package redis

import (
	"context"
	"strconv"
	"strings"
	"time"

	"github.com/redis/go-redis/v9/internal/otel"
)

// XTrimLimitDisabled is a sentinel value for the LIMIT argument of stream
// trimming (XAddArgs.Limit and the XTrim*Approx* commands). Passing it emits
// an explicit "LIMIT 0", which tells Redis to disable the trimming effort cap
// entirely. This differs from passing 0, which keeps the historical behavior:
// no LIMIT clause is sent and Redis applies its implicit default
// (100 * stream-node-max-entries examined entries).
//
// LIMIT is only valid together with the "~" (approximate) trimming flag;
// Redis rejects LIMIT used with exact ("=") trimming.
const XTrimLimitDisabled = -1

// appendXTrimLimit appends the LIMIT clause used by XADD and XTRIM trimming:
//   - limit > 0 emits "LIMIT <limit>";
//   - limit < 0 (see XTrimLimitDisabled) emits "LIMIT 0", disabling the
//     trimming effort cap;
//   - limit == 0 omits the clause, so Redis applies its implicit default.
func appendXTrimLimit(args []interface{}, limit int64) []interface{} {
	switch {
	case limit > 0:
		return append(args, "limit", limit)
	case limit < 0:
		return append(args, "limit", int64(0))
	default:
		return args
	}
}

type StreamCmdable interface {
	XAdd(ctx context.Context, a *XAddArgs) *StringCmd
	XAckDel(ctx context.Context, stream string, group string, mode string, ids ...string) *SliceCmd
	XDel(ctx context.Context, stream string, ids ...string) *IntCmd
	XDelEx(ctx context.Context, stream string, mode string, ids ...string) *SliceCmd
	XLen(ctx context.Context, stream string) *IntCmd
	XRange(ctx context.Context, stream, start, stop string) *XMessageSliceCmd
	XRangeN(ctx context.Context, stream, start, stop string, count int64) *XMessageSliceCmd
	XRevRange(ctx context.Context, stream string, start, stop string) *XMessageSliceCmd
	XRevRangeN(ctx context.Context, stream string, start, stop string, count int64) *XMessageSliceCmd
	XRead(ctx context.Context, a *XReadArgs) *XStreamSliceCmd
	XReadStreams(ctx context.Context, streams ...string) *XStreamSliceCmd
	XGroupCreate(ctx context.Context, stream, group, start string) *StatusCmd
	XGroupCreateMkStream(ctx context.Context, stream, group, start string) *StatusCmd
	XGroupSetID(ctx context.Context, stream, group, start string) *StatusCmd
	XGroupDestroy(ctx context.Context, stream, group string) *IntCmd
	XGroupCreateConsumer(ctx context.Context, stream, group, consumer string) *IntCmd
	XGroupDelConsumer(ctx context.Context, stream, group, consumer string) *IntCmd
	XReadGroup(ctx context.Context, a *XReadGroupArgs) *XStreamSliceCmd
	XAck(ctx context.Context, stream, group string, ids ...string) *IntCmd
	XNack(ctx context.Context, a *XNackArgs) *IntCmd
	XPending(ctx context.Context, stream, group string) *XPendingCmd
	XPendingExt(ctx context.Context, a *XPendingExtArgs) *XPendingExtCmd
	XClaim(ctx context.Context, a *XClaimArgs) *XMessageSliceCmd
	XClaimJustID(ctx context.Context, a *XClaimArgs) *StringSliceCmd
	XAutoClaim(ctx context.Context, a *XAutoClaimArgs) *XAutoClaimCmd
	XAutoClaimWithDeleted(ctx context.Context, a *XAutoClaimArgs) *XAutoClaimWithDeletedCmd
	XAutoClaimJustID(ctx context.Context, a *XAutoClaimArgs) *XAutoClaimJustIDCmd
	XTrimMaxLen(ctx context.Context, key string, maxLen int64) *IntCmd
	XTrimMaxLenApprox(ctx context.Context, key string, maxLen, limit int64) *IntCmd
	XTrimMaxLenMode(ctx context.Context, key string, maxLen int64, mode string) *IntCmd
	XTrimMaxLenApproxMode(ctx context.Context, key string, maxLen, limit int64, mode string) *IntCmd
	XTrimMinID(ctx context.Context, key string, minID string) *IntCmd
	XTrimMinIDApprox(ctx context.Context, key string, minID string, limit int64) *IntCmd
	XTrimMinIDMode(ctx context.Context, key string, minID string, mode string) *IntCmd
	XTrimMinIDApproxMode(ctx context.Context, key string, minID string, limit int64, mode string) *IntCmd
	XInfoGroups(ctx context.Context, key string) *XInfoGroupsCmd
	XInfoStream(ctx context.Context, key string) *XInfoStreamCmd
	XInfoStreamFull(ctx context.Context, key string, count int) *XInfoStreamFullCmd
	XInfoConsumers(ctx context.Context, key string, group string) *XInfoConsumersCmd
	XCfgSet(ctx context.Context, a *XCfgSetArgs) *StatusCmd
}

// XAddArgs accepts values in the following formats:
//   - XAddArgs.Values = []interface{}{"key1", "value1", "key2", "value2"}
//   - XAddArgs.Values = []string("key1", "value1", "key2", "value2")
//   - XAddArgs.Values = map[string]interface{}{"key1": "value1", "key2": "value2"}
//
// Note that map will not preserve the order of key-value pairs.
// MaxLen/MaxLenApprox and MinID are in conflict, only one of them can be used.
//
// For idempotent production (at-most-once production):
//   - ProducerID: A unique identifier for the producer (required for both IDMP and IDMPAUTO)
//   - IdempotentID: A unique identifier for the message (used with IDMP)
//   - IdempotentAuto: If true, Redis will auto-generate an idempotent ID based on message content (IDMPAUTO)
//
// ProducerID and IdempotentID are mutually exclusive with IdempotentAuto.
// When using idempotent production, ID must be "*" or empty.
type XAddArgs struct {
	Stream     string
	NoMkStream bool
	MaxLen     int64 // MAXLEN N
	MinID      string
	// Approx causes MaxLen and MinID to use "~" matcher (instead of "=").
	Approx bool
	// Limit caps the trimming effort:
	//   - 0 omits the LIMIT clause (Redis applies its implicit default);
	//   - a positive value emits "LIMIT <n>";
	//   - a negative value (see XTrimLimitDisabled) emits "LIMIT 0",
	//     disabling the effort cap entirely.
	// LIMIT requires Approx to be true; Redis rejects LIMIT together with
	// exact ("=") trimming.
	Limit          int64
	Mode           string
	ID             string
	Values         interface{}
	ProducerID     string // Producer ID for idempotent production (IDMP or IDMPAUTO)
	IdempotentID   string // Idempotent ID for IDMP
	IdempotentAuto bool   // Use IDMPAUTO to auto-generate idempotent ID based on content
}

func (c cmdable) XAdd(ctx context.Context, a *XAddArgs) *StringCmd {
	args := make([]interface{}, 0, 15)
	args = append(args, "xadd", a.Stream)
	if a.NoMkStream {
		args = append(args, "nomkstream")
	}

	if a.Mode != "" {
		args = append(args, a.Mode)
	}

	if a.ProducerID != "" {
		if a.IdempotentAuto {
			// IDMPAUTO pid
			args = append(args, "idmpauto", a.ProducerID)
		} else if a.IdempotentID != "" {
			// IDMP pid iid
			args = append(args, "idmp", a.ProducerID, a.IdempotentID)
		}
	}

	switch {
	case a.MaxLen > 0:
		if a.Approx {
			args = append(args, "maxlen", "~", a.MaxLen)
		} else {
			args = append(args, "maxlen", "=", a.MaxLen)
		}
	case a.MinID != "":
		if a.Approx {
			args = append(args, "minid", "~", a.MinID)
		} else {
			args = append(args, "minid", "=", a.MinID)
		}
	}
	args = appendXTrimLimit(args, a.Limit)

	if a.ID != "" {
		args = append(args, a.ID)
	} else {
		args = append(args, "*")
	}
	args = appendArg(args, a.Values)

	cmd := NewStringCmd(ctx, args...)
	_ = c(ctx, cmd)
	return cmd
}

func (c cmdable) XAckDel(ctx context.Context, stream string, group string, mode string, ids ...string) *SliceCmd {
	args := []interface{}{"xackdel", stream, group, mode, "ids", len(ids)}
	for _, id := range ids {
		args = append(args, id)
	}
	cmd := NewSliceCmd(ctx, args...)
	_ = c(ctx, cmd)
	return cmd
}

func (c cmdable) XDel(ctx context.Context, stream string, ids ...string) *IntCmd {
	args := []interface{}{"xdel", stream}
	for _, id := range ids {
		args = append(args, id)
	}
	cmd := NewIntCmd(ctx, args...)
	_ = c(ctx, cmd)
	return cmd
}

func (c cmdable) XDelEx(ctx context.Context, stream string, mode string, ids ...string) *SliceCmd {
	args := []interface{}{"xdelex", stream, mode, "ids", len(ids)}
	for _, id := range ids {
		args = append(args, id)
	}
	cmd := NewSliceCmd(ctx, args...)
	_ = c(ctx, cmd)
	return cmd
}

func (c cmdable) XLen(ctx context.Context, stream string) *IntCmd {
	cmd := NewIntCmd(ctx, "xlen", stream)
	_ = c(ctx, cmd)
	return cmd
}

func (c cmdable) XRange(ctx context.Context, stream, start, stop string) *XMessageSliceCmd {
	cmd := NewXMessageSliceCmd(ctx, "xrange", stream, start, stop)
	_ = c(ctx, cmd)
	return cmd
}

func (c cmdable) XRangeN(ctx context.Context, stream, start, stop string, count int64) *XMessageSliceCmd {
	cmd := NewXMessageSliceCmd(ctx, "xrange", stream, start, stop, "count", count)
	_ = c(ctx, cmd)
	return cmd
}

func (c cmdable) XRevRange(ctx context.Context, stream, start, stop string) *XMessageSliceCmd {
	cmd := NewXMessageSliceCmd(ctx, "xrevrange", stream, start, stop)
	_ = c(ctx, cmd)
	return cmd
}

func (c cmdable) XRevRangeN(ctx context.Context, stream, start, stop string, count int64) *XMessageSliceCmd {
	cmd := NewXMessageSliceCmd(ctx, "xrevrange", stream, start, stop, "count", count)
	_ = c(ctx, cmd)
	return cmd
}

type XReadArgs struct {
	Streams  []string // list of streams and ids, e.g. stream1 stream2 id1 id2
	Count    int64
	MaxCount int64 // cumulative cap on total entries across all streams (Redis >= 8.10)
	MaxSize  int64 // soft cumulative cap on total reply size in bytes across all streams (Redis >= 8.10)
	Block    time.Duration
	ID       string
}

func (c cmdable) XRead(ctx context.Context, a *XReadArgs) *XStreamSliceCmd {
	args := make([]interface{}, 0, 2*len(a.Streams)+10)
	args = append(args, "xread")

	keyPos := int8(1)
	if a.Count > 0 {
		args = append(args, "count")
		args = append(args, a.Count)
		keyPos += 2
	}
	if a.MaxCount > 0 {
		args = append(args, "maxcount", a.MaxCount)
		keyPos += 2
	}
	if a.MaxSize > 0 {
		args = append(args, "maxsize", a.MaxSize)
		keyPos += 2
	}
	if a.Block >= 0 {
		args = append(args, "block")
		args = append(args, int64(a.Block/time.Millisecond))
		keyPos += 2
	}
	args = append(args, "streams")
	keyPos++
	for _, s := range a.Streams {
		args = append(args, s)
	}
	if a.ID != "" {
		for range a.Streams {
			args = append(args, a.ID)
		}
	}

	cmd := NewXStreamSliceCmd(ctx, args...)
	if a.Block >= 0 {
		cmd.setReadTimeout(a.Block)
	}
	cmd.SetFirstKeyPos(keyPos)
	_ = c(ctx, cmd)
	return cmd
}

func (c cmdable) XReadStreams(ctx context.Context, streams ...string) *XStreamSliceCmd {
	return c.XRead(ctx, &XReadArgs{
		Streams: streams,
		Block:   -1,
	})
}

func (c cmdable) XGroupCreate(ctx context.Context, stream, group, start string) *StatusCmd {
	cmd := NewStatusCmd(ctx, "xgroup", "create", stream, group, start)
	cmd.SetFirstKeyPos(2)
	_ = c(ctx, cmd)
	return cmd
}

func (c cmdable) XGroupCreateMkStream(ctx context.Context, stream, group, start string) *StatusCmd {
	cmd := NewStatusCmd(ctx, "xgroup", "create", stream, group, start, "mkstream")
	cmd.SetFirstKeyPos(2)
	_ = c(ctx, cmd)
	return cmd
}

func (c cmdable) XGroupSetID(ctx context.Context, stream, group, start string) *StatusCmd {
	cmd := NewStatusCmd(ctx, "xgroup", "setid", stream, group, start)
	cmd.SetFirstKeyPos(2)
	_ = c(ctx, cmd)
	return cmd
}

func (c cmdable) XGroupDestroy(ctx context.Context, stream, group string) *IntCmd {
	cmd := NewIntCmd(ctx, "xgroup", "destroy", stream, group)
	cmd.SetFirstKeyPos(2)
	_ = c(ctx, cmd)
	return cmd
}

func (c cmdable) XGroupCreateConsumer(ctx context.Context, stream, group, consumer string) *IntCmd {
	cmd := NewIntCmd(ctx, "xgroup", "createconsumer", stream, group, consumer)
	cmd.SetFirstKeyPos(2)
	_ = c(ctx, cmd)
	return cmd
}

func (c cmdable) XGroupDelConsumer(ctx context.Context, stream, group, consumer string) *IntCmd {
	cmd := NewIntCmd(ctx, "xgroup", "delconsumer", stream, group, consumer)
	cmd.SetFirstKeyPos(2)
	_ = c(ctx, cmd)
	return cmd
}

type XReadGroupArgs struct {
	Group    string
	Consumer string
	Streams  []string // list of streams and ids, e.g. stream1 stream2 id1 id2
	Count    int64
	MaxCount int64 // cumulative cap on total entries across all streams (Redis >= 8.10)
	MaxSize  int64 // soft cumulative cap on total reply size in bytes across all streams (Redis >= 8.10)
	Block    time.Duration
	NoAck    bool
	Claim    time.Duration // Claim idle pending entries older than this duration
}

func (c cmdable) XReadGroup(ctx context.Context, a *XReadGroupArgs) *XStreamSliceCmd {
	args := make([]interface{}, 0, 14+len(a.Streams))
	args = append(args, "xreadgroup", "group", a.Group, a.Consumer)

	keyPos := int8(4)
	if a.Count > 0 {
		args = append(args, "count", a.Count)
		keyPos += 2
	}
	if a.MaxCount > 0 {
		args = append(args, "maxcount", a.MaxCount)
		keyPos += 2
	}
	if a.MaxSize > 0 {
		args = append(args, "maxsize", a.MaxSize)
		keyPos += 2
	}
	if a.Block >= 0 {
		args = append(args, "block", int64(a.Block/time.Millisecond))
		keyPos += 2
	}
	if a.NoAck {
		args = append(args, "noack")
		keyPos++
	}
	if a.Claim > 0 {
		args = append(args, "claim", int64(a.Claim/time.Millisecond))
		keyPos += 2
	}
	args = append(args, "streams")
	keyPos++
	for _, s := range a.Streams {
		args = append(args, s)
	}

	cmd := NewXStreamSliceCmd(ctx, args...)
	if a.Block >= 0 {
		cmd.setReadTimeout(a.Block)
	}
	cmd.SetFirstKeyPos(keyPos)
	_ = c(ctx, cmd)

	// Record stream lag for each message (if command succeeded). Gated on the
	// result being readable WITHOUT blocking: this command carries a
	// read-timeout marker, so on the deferred autopipeline face it is diverted
	// and still running when we get here — and the default Block: 0 form can
	// wait indefinitely for messages, so reading the outcome would block the
	// submit call instead of returning a future (review finding by codex on
	// #3942). Skipped for a submission that has not executed yet; emitting
	// this from the execution path is a follow-up in the OTel wiring.
	if otel.Enabled() && cmd.resultReady() && cmd.rawErr() == nil {
		streams := cmd.Val()
		for _, stream := range streams {
			for _, msg := range stream.Messages {
				// Parse message ID to extract timestamp (format: "millisecondsTime-sequenceNumber")
				if parts := strings.SplitN(msg.ID, "-", 2); len(parts) == 2 {
					if timestampMs, err := strconv.ParseInt(parts[0], 10, 64); err == nil {
						// Calculate lag (time since message was created)
						messageTime := time.Unix(0, timestampMs*int64(time.Millisecond))
						lag := time.Since(messageTime)
						// Record lag metric
						otel.RecordStreamLag(ctx, lag, nil, stream.Stream, a.Group, a.Consumer)
					}
				}
			}
		}
	}

	return cmd
}

func (c cmdable) XAck(ctx context.Context, stream, group string, ids ...string) *IntCmd {
	args := []interface{}{"xack", stream, group}
	for _, id := range ids {
		args = append(args, id)
	}
	cmd := NewIntCmd(ctx, args...)
	_ = c(ctx, cmd)
	return cmd
}

// XNACK modes. See [XNackArgs.Mode].
const (
	XNackModeSilent = "SILENT"
	XNackModeFail   = "FAIL"
	XNackModeFatal  = "FATAL"
)

// XNackArgs represents the arguments for the XNACK command (Redis >= 8.8).
//
// XNACK negatively acknowledges one or more messages in a consumer group's
// Pending Entries List (PEL), releasing them back to the group so they can be
// redelivered to another consumer via XREADGROUP.
type XNackArgs struct {
	Stream string
	Group  string

	// Mode controls how the delivery counter is adjusted for each NACKed entry.
	// Must be one of [XNackModeSilent], [XNackModeFail], or [XNackModeFatal]:
	//   - SILENT: the consumer is shutting down or experiencing internal errors
	//     unrelated to the message. The delivery counter is decremented by 1,
	//     undoing the increment that happened when the message was delivered.
	//   - FAIL: the consumer could not process the message (e.g. insufficient
	//     memory), but another consumer might succeed. The delivery counter is
	//     left unchanged.
	//   - FATAL: the message is invalid or suspected malicious. The delivery
	//     counter is set to MAXINT, which will immediately move the message to
	//     the Dead Letter Queue (DLQ) if one is configured for the group.
	Mode string

	// IDs is the list of message IDs to NACK. All IDs must already be in the
	// group's PEL (i.e. previously delivered via XREADGROUP), unless Force is set.
	IDs []string

	// RetryCount sets the delivery counter to an explicit value, overriding the
	// counter adjustment that would otherwise be applied by Mode.
	// Leave nil to let Mode control the counter (the common case).
	RetryCount *uint64

	// Force allows NACKing message IDs that are not yet in the group's PEL,
	// creating new unowned NACKed PEL entries for them directly.
	// This is analogous to the FORCE flag in XCLAIM.
	// Primarily used internally by Redis during AOF rewrite to reconstruct
	// NACKed entries, but can also be used to manually inject entries.
	Force bool
}

// XNack executes the XNACK command. See [XNackArgs] for the full argument documentation.
// Requires Redis >= 8.8.
func (c cmdable) XNack(ctx context.Context, a *XNackArgs) *IntCmd {
	args := make([]interface{}, 0, 9+len(a.IDs))
	args = append(args, "xnack", a.Stream, a.Group, a.Mode, "ids", len(a.IDs))
	for _, id := range a.IDs {
		args = append(args, id)
	}
	if a.RetryCount != nil {
		args = append(args, "retrycount", *a.RetryCount)
	}
	if a.Force {
		args = append(args, "force")
	}
	cmd := NewIntCmd(ctx, args...)
	_ = c(ctx, cmd)
	return cmd
}

func (c cmdable) XPending(ctx context.Context, stream, group string) *XPendingCmd {
	cmd := NewXPendingCmd(ctx, "xpending", stream, group)
	_ = c(ctx, cmd)
	return cmd
}

type XPendingExtArgs struct {
	Stream   string
	Group    string
	Idle     time.Duration
	Start    string
	End      string
	Count    int64
	Consumer string
}

func (c cmdable) XPendingExt(ctx context.Context, a *XPendingExtArgs) *XPendingExtCmd {
	args := make([]interface{}, 0, 9)
	args = append(args, "xpending", a.Stream, a.Group)
	if a.Idle != 0 {
		args = append(args, "idle", formatMs(ctx, a.Idle))
	}
	args = append(args, a.Start, a.End, a.Count)
	if a.Consumer != "" {
		args = append(args, a.Consumer)
	}
	cmd := NewXPendingExtCmd(ctx, args...)
	_ = c(ctx, cmd)
	return cmd
}

type XAutoClaimArgs struct {
	Stream   string
	Group    string
	MinIdle  time.Duration
	Start    string
	Count    int64
	Consumer string
}

func (c cmdable) XAutoClaim(ctx context.Context, a *XAutoClaimArgs) *XAutoClaimCmd {
	args := xAutoClaimArgs(ctx, a)
	cmd := NewXAutoClaimCmd(ctx, args...)
	_ = c(ctx, cmd)
	return cmd
}

func (c cmdable) XAutoClaimWithDeleted(ctx context.Context, a *XAutoClaimArgs) *XAutoClaimWithDeletedCmd {
	args := xAutoClaimArgs(ctx, a)
	cmd := NewXAutoClaimWithDeletedCmd(ctx, args...)
	_ = c(ctx, cmd)
	return cmd
}

func (c cmdable) XAutoClaimJustID(ctx context.Context, a *XAutoClaimArgs) *XAutoClaimJustIDCmd {
	args := xAutoClaimArgs(ctx, a)
	args = append(args, "justid")
	cmd := NewXAutoClaimJustIDCmd(ctx, args...)
	_ = c(ctx, cmd)
	return cmd
}

func xAutoClaimArgs(ctx context.Context, a *XAutoClaimArgs) []interface{} {
	args := make([]interface{}, 0, 8)
	args = append(args, "xautoclaim", a.Stream, a.Group, a.Consumer, formatMs(ctx, a.MinIdle), a.Start)
	if a.Count > 0 {
		args = append(args, "count", a.Count)
	}
	return args
}

type XClaimArgs struct {
	Stream   string
	Group    string
	Consumer string
	MinIdle  time.Duration
	Messages []string
}

func (c cmdable) XClaim(ctx context.Context, a *XClaimArgs) *XMessageSliceCmd {
	args := xClaimArgs(a)
	cmd := NewXMessageSliceCmd(ctx, args...)
	_ = c(ctx, cmd)
	return cmd
}

func (c cmdable) XClaimJustID(ctx context.Context, a *XClaimArgs) *StringSliceCmd {
	args := xClaimArgs(a)
	args = append(args, "justid")
	cmd := NewStringSliceCmd(ctx, args...)
	_ = c(ctx, cmd)
	return cmd
}

func xClaimArgs(a *XClaimArgs) []interface{} {
	args := make([]interface{}, 0, 5+len(a.Messages))
	args = append(args,
		"xclaim",
		a.Stream,
		a.Group, a.Consumer,
		int64(a.MinIdle/time.Millisecond))
	for _, id := range a.Messages {
		args = append(args, id)
	}
	return args
}

// TODO: refactor xTrim, xTrimMode and the wrappers over the functions

// xTrim If approx is true, add the "~" parameter, otherwise it is the default "=" (redis default).
// example:
//
//	XTRIM key MAXLEN/MINID threshold LIMIT limit.
//	XTRIM key MAXLEN/MINID ~ threshold LIMIT limit.
//
// The redis-server version is lower than 6.2, please set limit to 0.
//
// limit == 0 omits the LIMIT clause, a positive limit emits "LIMIT <limit>",
// and a negative limit (see XTrimLimitDisabled) emits "LIMIT 0" to disable
// the trimming effort cap. LIMIT requires approx; Redis rejects it otherwise.
func (c cmdable) xTrim(
	ctx context.Context, key, strategy string,
	approx bool, threshold interface{}, limit int64,
) *IntCmd {
	args := make([]interface{}, 0, 7)
	args = append(args, "xtrim", key, strategy)
	if approx {
		args = append(args, "~")
	} else {
		args = append(args, "=")
	}
	args = append(args, threshold)
	args = appendXTrimLimit(args, limit)
	cmd := NewIntCmd(ctx, args...)
	_ = c(ctx, cmd)
	return cmd
}

// XTrimMaxLen No `~` rules are used, `limit` cannot be used.
// cmd: XTRIM key MAXLEN maxLen
func (c cmdable) XTrimMaxLen(ctx context.Context, key string, maxLen int64) *IntCmd {
	return c.xTrim(ctx, key, "maxlen", false, maxLen, 0)
}

// XTrimMaxLenApprox trims the stream using the `~` rule.
// cmd: XTRIM key MAXLEN ~ maxLen [LIMIT limit]
//
// limit == 0 omits the LIMIT clause, limit > 0 emits "LIMIT <limit>", and a
// negative limit (see XTrimLimitDisabled) emits "LIMIT 0" to disable the
// trimming effort cap.
func (c cmdable) XTrimMaxLenApprox(ctx context.Context, key string, maxLen, limit int64) *IntCmd {
	return c.xTrim(ctx, key, "maxlen", true, maxLen, limit)
}

func (c cmdable) XTrimMinID(ctx context.Context, key string, minID string) *IntCmd {
	return c.xTrim(ctx, key, "minid", false, minID, 0)
}

// XTrimMinIDApprox trims the stream using the `~` rule.
// cmd: XTRIM key MINID ~ minID [LIMIT limit]
//
// limit == 0 omits the LIMIT clause, limit > 0 emits "LIMIT <limit>", and a
// negative limit (see XTrimLimitDisabled) emits "LIMIT 0" to disable the
// trimming effort cap.
func (c cmdable) XTrimMinIDApprox(ctx context.Context, key string, minID string, limit int64) *IntCmd {
	return c.xTrim(ctx, key, "minid", true, minID, limit)
}

// xTrimMode is xTrim with a trailing trimming mode argument (e.g. KEEPREF).
// The limit semantics are the same as xTrim's.
func (c cmdable) xTrimMode(
	ctx context.Context, key, strategy string,
	approx bool, threshold interface{}, limit int64,
	mode string,
) *IntCmd {
	args := make([]interface{}, 0, 7)
	args = append(args, "xtrim", key, strategy)
	if approx {
		args = append(args, "~")
	} else {
		args = append(args, "=")
	}
	args = append(args, threshold)
	args = appendXTrimLimit(args, limit)
	args = append(args, mode)
	cmd := NewIntCmd(ctx, args...)
	_ = c(ctx, cmd)
	return cmd
}

func (c cmdable) XTrimMaxLenMode(ctx context.Context, key string, maxLen int64, mode string) *IntCmd {
	return c.xTrimMode(ctx, key, "maxlen", false, maxLen, 0, mode)
}

func (c cmdable) XTrimMaxLenApproxMode(ctx context.Context, key string, maxLen, limit int64, mode string) *IntCmd {
	return c.xTrimMode(ctx, key, "maxlen", true, maxLen, limit, mode)
}

func (c cmdable) XTrimMinIDMode(ctx context.Context, key string, minID string, mode string) *IntCmd {
	return c.xTrimMode(ctx, key, "minid", false, minID, 0, mode)
}

func (c cmdable) XTrimMinIDApproxMode(ctx context.Context, key string, minID string, limit int64, mode string) *IntCmd {
	return c.xTrimMode(ctx, key, "minid", true, minID, limit, mode)
}

func (c cmdable) XInfoConsumers(ctx context.Context, key string, group string) *XInfoConsumersCmd {
	cmd := NewXInfoConsumersCmd(ctx, key, group)
	_ = c(ctx, cmd)
	return cmd
}

func (c cmdable) XInfoGroups(ctx context.Context, key string) *XInfoGroupsCmd {
	cmd := NewXInfoGroupsCmd(ctx, key)
	_ = c(ctx, cmd)
	return cmd
}

func (c cmdable) XInfoStream(ctx context.Context, key string) *XInfoStreamCmd {
	cmd := NewXInfoStreamCmd(ctx, key)
	_ = c(ctx, cmd)
	return cmd
}

// XInfoStreamFull XINFO STREAM FULL [COUNT count]
// redis-server >= 6.0.
func (c cmdable) XInfoStreamFull(ctx context.Context, key string, count int) *XInfoStreamFullCmd {
	args := make([]interface{}, 0, 6)
	args = append(args, "xinfo", "stream", key, "full")
	if count > 0 {
		args = append(args, "count", count)
	}
	cmd := NewXInfoStreamFullCmd(ctx, args...)
	_ = c(ctx, cmd)
	return cmd
}

// XCfgSetArgs represents the arguments for the XCFGSET command.
// Duration is the duration, in seconds, that Redis keeps each idempotent ID.
// MaxSize is the maximum number of most recent idempotent IDs that Redis keeps for each producer ID.
type XCfgSetArgs struct {
	Stream   string
	Duration int64
	MaxSize  int64
}

// XCfgSet sets the idempotent production configuration for a stream.
// XCFGSET key [IDMP-DURATION duration] [IDMP-MAXSIZE maxsize]
func (c cmdable) XCfgSet(ctx context.Context, a *XCfgSetArgs) *StatusCmd {
	args := make([]interface{}, 0, 6)
	args = append(args, "xcfgset", a.Stream)
	if a.Duration > 0 {
		args = append(args, "idmp-duration", a.Duration)
	}
	if a.MaxSize > 0 {
		args = append(args, "idmp-maxsize", a.MaxSize)
	}
	cmd := NewStatusCmd(ctx, args...)
	_ = c(ctx, cmd)
	return cmd
}
