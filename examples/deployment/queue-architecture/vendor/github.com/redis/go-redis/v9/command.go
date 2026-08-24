package redis

import (
	"bufio"
	"context"
	"fmt"
	"io"
	"maps"
	"net"
	"regexp"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/redis/go-redis/v9/internal"
	"github.com/redis/go-redis/v9/internal/hscan"
	"github.com/redis/go-redis/v9/internal/proto"
	"github.com/redis/go-redis/v9/internal/routing"
	"github.com/redis/go-redis/v9/internal/util"
)

// keylessCommands contains Redis commands that have empty key specifications (9th slot empty)
// Only includes core Redis commands, excludes FT.*, ts.*, timeseries.*, search.* and subcommands
var keylessCommands = map[string]struct{}{
	"acl":          {},
	"asking":       {},
	"auth":         {},
	"bgrewriteaof": {},
	"bgsave":       {},
	"client":       {},
	"cluster":      {},
	"config":       {},
	"debug":        {},
	"discard":      {},
	"echo":         {},
	"exec":         {},
	"failover":     {},
	"function":     {},
	"hello":        {},
	"hotkeys":      {},
	"latency":      {},
	"lolwut":       {},
	"module":       {},
	"monitor":      {},
	"multi":        {},
	"pfselftest":   {},
	"ping":         {},
	"psubscribe":   {},
	"psync":        {},
	"publish":      {},
	"pubsub":       {},
	"punsubscribe": {},
	"quit":         {},
	"readonly":     {},
	"readwrite":    {},
	"replconf":     {},
	"replicaof":    {},
	"role":         {},
	"save":         {},
	"script":       {},
	"select":       {},
	"shutdown":     {},
	"slaveof":      {},
	"slowlog":      {},
	"subscribe":    {},
	"swapdb":       {},
	"sync":         {},
	"time":         {},
	"unsubscribe":  {},
	"unwatch":      {},
	"wait":         {},
}

// CmdTyper interface for getting command type
type CmdTyper interface {
	GetCmdType() CmdType
}

// CmdTypeGetter interface for getting command type without circular imports
type CmdTypeGetter interface {
	GetCmdType() CmdType
}

type CmdType uint8

const (
	CmdTypeGeneric CmdType = iota
	CmdTypeString
	CmdTypeInt
	CmdTypeBool
	CmdTypeFloat
	CmdTypeStringSlice
	CmdTypeIntSlice
	CmdTypeFloatSlice
	CmdTypeBoolSlice
	CmdTypeMapStringString
	CmdTypeMapStringInt
	CmdTypeMapStringInterface
	CmdTypeMapStringInterfaceSlice
	CmdTypeSlice
	CmdTypeStatus
	CmdTypeDuration
	CmdTypeTime
	CmdTypeKeyValueSlice
	CmdTypeStringStructMap
	CmdTypeXMessageSlice
	CmdTypeXStreamSlice
	CmdTypeXPending
	CmdTypeXPendingExt
	CmdTypeXAutoClaim
	CmdTypeXAutoClaimWithDeleted
	CmdTypeXAutoClaimJustID
	CmdTypeXInfoConsumers
	CmdTypeXInfoGroups
	CmdTypeXInfoStream
	CmdTypeXInfoStreamFull
	CmdTypeZSlice
	CmdTypeZWithKey
	CmdTypeScan
	CmdTypeClusterSlots
	CmdTypeGeoLocation
	CmdTypeGeoSearchLocation
	CmdTypeGeoPos
	CmdTypeCommandsInfo
	CmdTypeSlowLog
	CmdTypeMapStringStringSlice
	CmdTypeMapMapStringInterface
	CmdTypeKeyValues
	CmdTypeZSliceWithKey
	CmdTypeFunctionList
	CmdTypeFunctionStats
	CmdTypeLCS
	CmdTypeKeyFlags
	CmdTypeClusterLinks
	CmdTypeClusterShards
	CmdTypeRankWithScore
	CmdTypeClientInfo
	CmdTypeACLLog
	CmdTypeInfo
	CmdTypeMonitor
	CmdTypeJSON
	CmdTypeJSONSlice
	CmdTypeIntPointerSlice
	CmdTypeScanDump
	CmdTypeBFInfo
	CmdTypeCFInfo
	CmdTypeCMSInfo
	CmdTypeTopKInfo
	CmdTypeTDigestInfo
	CmdTypeFTSynDump
	CmdTypeAggregate
	CmdTypeFTInfo
	CmdTypeFTSpellCheck
	CmdTypeFTSearch
	CmdTypeTSTimestampValue
	CmdTypeTSTimestampValueSlice
	CmdTypeTSNRangePivotRowSlice
	CmdTypeHotKeys
	CmdTypeIncrEXInt
	CmdTypeIncrEXFloat
	CmdTypeUint
	CmdTypeUintSlice
	CmdTypeAREntrySlice
)

type (
	CmdTypeXAutoClaimValue struct {
		messages []XMessage
		start    string
	}

	CmdTypeXAutoClaimWithDeletedValue struct {
		messages   []XMessage
		start      string
		deletedIDs []string
	}

	CmdTypeXAutoClaimJustIDValue struct {
		ids   []string
		start string
	}

	CmdTypeScanValue struct {
		keys   []string
		cursor uint64
	}

	CmdTypeKeyValuesValue struct {
		key    string
		values []string
	}

	CmdTypeZSliceWithKeyValue struct {
		key    string
		zSlice []Z
	}
)

type Cmder interface {
	// command name.
	// e.g. "set k v ex 10" -> "set", "cluster info" -> "cluster".
	Name() string

	// full command name.
	// e.g. "set k v ex 10" -> "set", "cluster info" -> "cluster info".
	FullName() string

	// all args of the command.
	// e.g. "set k v ex 10" -> "[set k v ex 10]".
	Args() []interface{}

	// format request and response string.
	// e.g. "set k v ex 10" -> "set k v ex 10: OK", "get k" -> "get k: v".
	String() string

	// Clone creates a copy of the command.
	Clone() Cmder

	stringArg(int) string
	firstKeyPos() int8
	SetFirstKeyPos(int8)
	stepCount() int8
	SetStepCount(int8)

	// cachedSlot/setCachedSlot memoize the cluster slot so it is computed once
	// (in the autopipeline shard router) and reused at pipeline-flush routing.
	cachedSlot() (int, bool)
	setCachedSlot(int)

	readTimeout() *time.Duration
	readReply(rd *proto.Reader) error
	readRawReply(rd *proto.Reader) error
	SetErr(error)
	Err() error

	// setReady marks a command as asynchronously pending (autopipeline async
	// faces); await blocks the public accessors until it has executed; rawErr
	// reads the error without awaiting (internal execution path).
	setReady(*apBatch)
	await()
	rawErr() error

	// NoRetry returns true if the command should not be retried on failure.
	// Commands that write directly to an io.Writer should return true since
	// partial writes cannot be undone on retry.
	NoRetry() bool

	// GetCmdType returns the command type for fast value extraction
	GetCmdType() CmdType
}

func setCmdsErr(cmds []Cmder, e error) {
	for _, cmd := range cmds {
		// rawErr: this runs on the execution path; never await here.
		if cmd.rawErr() == nil {
			cmd.SetErr(e)
		}
	}
}

func cmdsFirstErr(cmds []Cmder) error {
	for _, cmd := range cmds {
		// rawErr: this runs on the execution path; never await here.
		if err := cmd.rawErr(); err != nil {
			return err
		}
	}
	return nil
}

// cmdsContainNoRetry returns true if any command in the slice has NoRetry() == true.
// If a pipeline contains a non-retryable command (e.g., RawWriteToCmd), the entire
// pipeline must not be retried to prevent data corruption from partial writes.
func cmdsContainNoRetry(cmds []Cmder) bool {
	for _, cmd := range cmds {
		if cmd.NoRetry() {
			return true
		}
	}
	return false
}

func writeCmds(wr *proto.Writer, cmds []Cmder) error {
	for _, cmd := range cmds {
		if err := writeCmd(wr, cmd); err != nil {
			return err
		}
	}
	return nil
}

func writeCmd(wr *proto.Writer, cmd Cmder) error {
	return wr.WriteArgs(cmd.Args())
}

// cmdFirstKeyPosWithInfo returns the first key position in a command's args (0 if none).
// Uses CommandInfo.FirstKeyPos when available (via cache peek, no network call), falling
// back to a hardcoded table. eval/evalsha variants are resolved from the runtime numkeys arg.
func cmdFirstKeyPosWithInfo(cmd Cmder, info *CommandInfo) int {
	if pos := cmd.firstKeyPos(); pos != 0 {
		return int(pos)
	}

	name := cmd.Name()

	// first check if the command is keyless
	if _, ok := keylessCommands[name]; ok {
		return 0
	}

	// Module commands registered keyless in the static policy table (e.g.
	// ft.aliaslist) route as keyless even while the command-info cache is
	// cold, so the first calls of a process don't hash a non-key argument
	// (such as an index name) into a slot.
	if defaultPolicyKeyless(name) {
		return 0
	}

	switch name {
	case "eval", "evalsha", "eval_ro", "evalsha_ro":
		if cmd.stringArg(2) != "0" {
			return 3
		}

		return 0
	case "memory":
		// https://github.com/redis/redis/issues/7493
		if cmd.stringArg(1) == "usage" {
			return 2
		}
		// CommandInfo (if available) gives the correct answer
		// otherwise the hardcoded fallback applies.
	}

	// Use CommandInfo cache when warm (in-memory only, no extra round-trips).
	if info != nil {
		return int(info.FirstKeyPos)
	}

	return 1
}

func cmdString(cmd Cmder, val interface{}) string {
	b := make([]byte, 0, 64)

	for i, arg := range cmd.Args() {
		if i > 0 {
			b = append(b, ' ')
		}
		b = internal.AppendArg(b, arg)
	}

	if err := cmd.rawErr(); err != nil {
		b = append(b, ": "...)
		b = append(b, err.Error()...)
	} else if val != nil {
		b = append(b, ": "...)
		b = internal.AppendArg(b, val)
	}

	return util.BytesToString(b)
}

//------------------------------------------------------------------------------

type baseCmd struct {
	ctx          context.Context
	args         []interface{}
	err          error
	keyPos       int8
	_stepCount   int8
	rawVal       interface{}
	_readTimeout *time.Duration
	cmdType      CmdType
	// slotCache memoizes the cluster slot once computed, so the cluster
	// autopipeline shard router and the pipeline flush router don't each
	// recompute it. 0 = not computed; it stores slot+1 so a real slot of 0 is
	// distinguishable from unset. A plain field is safe by construction: it
	// is written at most once, on the submitting goroutine BEFORE the command
	// is published to a stripe queue (the stripe mutex is the happens-before
	// edge to the flusher that later reads it). Do not write it from any
	// other point in the command's life.
	slotCache uint16

	// ready, when non-nil, is the batch whose done channel closes once the
	// command has executed. It is set only by the deferred (async)
	// autopipeliner, which hands the command back to the caller before it
	// runs. The public result accessors (Err/Val/Result/String) call await
	// so they transparently block until execution; internal execution-path
	// reads use rawErr to avoid awaiting the very batch they are producing
	// (formatting included: cmdString reads rawErr and receives the value
	// snapshot from its caller, so String methods await BEFORE reading their
	// val field — otherwise formatting an in-flight async command would race
	// with reply processing). ready stays
	// nil for ordinary synchronous commands, whose accessors never block.
	ready atomic.Pointer[apBatch]
}

// setReady publishes the batch gating this command's result accessors. The
// field is atomic, NOT lock-ordered with the enqueue: a dispatch-side hook
// racing this store simply reads nil and takes the non-blocking path — the
// correct "not executed yet" view — while the setting goroutine always sees
// its own store before it awaits.
func (cmd *baseCmd) setReady(b *apBatch) { cmd.ready.Store(b) }

// await blocks until an asynchronously-submitted command has executed. It is a
// single nil-pointer load for synchronous commands, so the common path stays
// allocation- and contention-free.
func (cmd *baseCmd) await() {
	b := cmd.ready.Load()
	if b == nil {
		return
	}
	select {
	case <-b.done:
		return
	default:
	}
	if b.isExecutorGoroutine() {
		// A hook on one of the batch's own executor goroutines (the
		// dispatcher, or a cluster per-node executor) is reading this
		// command's result BEFORE next() has executed it. Blocking would
		// self-deadlock (the batch completes only after that goroutine
		// returns); return the not-yet-executed state instead — the same
		// view a plain pipeline hook has before next().
		return
	}
	<-b.done
}

// rawErr returns the command error WITHOUT awaiting. The internal
// execution/serialization path (setCmdsErr, cmdsFirstErr, and the cmdString
// formatter — public String methods await before calling it) uses it so that
// reading errors while a batch is being executed does not deadlock on the
// batch's own completion signal.
func (cmd *baseCmd) rawErr() error { return cmd.err }

// readyBatch exposes the deferred-face batch gating this command (nil for
// synchronous commands) to the cluster fan-out, which registers its per-node
// goroutines as executors of every batch they carry.
func (cmd *baseCmd) readyBatch() *apBatch { return cmd.ready.Load() }

// resultReady reports whether the command's result can be read WITHOUT
// blocking: either it never rode the deferred autopipeline face (no gating
// batch) or that batch has already completed. Post-execution bookkeeping in
// the command wrappers — the OTel metric emissions — consults it so that
// enabling telemetry cannot turn a deferred submission into a blocking call.
func (cmd *baseCmd) resultReady() bool {
	b := cmd.ready.Load()
	if b == nil {
		return true
	}
	select {
	case <-b.done:
		return true
	default:
		return false
	}
}

var _ Cmder = (*Cmd)(nil)

func (cmd *baseCmd) Name() string {
	if len(cmd.args) == 0 {
		return ""
	}
	// Cmd name must be lower cased.
	return internal.ToLower(cmd.stringArg(0))
}

func (cmd *baseCmd) FullName() string {
	switch name := cmd.Name(); name {
	case "cluster", "command":
		if len(cmd.args) == 1 {
			return name
		}
		if s2, ok := cmd.args[1].(string); ok {
			return name + " " + s2
		}
		return name
	default:
		return name
	}
}

func (cmd *baseCmd) Args() []interface{} {
	return cmd.args
}

func (cmd *baseCmd) stringArg(pos int) string {
	if pos < 0 || pos >= len(cmd.args) {
		return ""
	}
	arg := cmd.args[pos]
	switch v := arg.(type) {
	case string:
		return v
	case *string:
		if v == nil {
			return ""
		}
		return *v
	case []byte:
		return string(v)
	default:
		// TODO: consider using appendArg
		return fmt.Sprint(v)
	}
}

func (cmd *baseCmd) firstKeyPos() int8 {
	return cmd.keyPos
}

func (cmd *baseCmd) SetFirstKeyPos(keyPos int8) {
	cmd.keyPos = keyPos
}

// cachedSlot returns the cached cluster slot and whether one was set.
func (cmd *baseCmd) cachedSlot() (int, bool) {
	if cmd.slotCache == 0 {
		return 0, false
	}
	return int(cmd.slotCache - 1), true
}

// setCachedSlot stores the computed cluster slot (0..16383) for reuse.
func (cmd *baseCmd) setCachedSlot(slot int) {
	if slot >= 0 && slot < 16384 {
		cmd.slotCache = uint16(slot + 1)
	}
}

func (cmd *baseCmd) stepCount() int8 {
	return cmd._stepCount
}

func (cmd *baseCmd) SetStepCount(stepCount int8) {
	cmd._stepCount = stepCount
}

func (cmd *baseCmd) SetErr(e error) {
	cmd.err = e
}

func (cmd *baseCmd) Err() error {
	cmd.await()
	return cmd.err
}

func (cmd *baseCmd) readTimeout() *time.Duration {
	return cmd._readTimeout
}

func (cmd *baseCmd) setReadTimeout(d time.Duration) {
	cmd._readTimeout = &d
}

func (cmd *baseCmd) readRawReply(rd *proto.Reader) (err error) {
	cmd.rawVal, err = rd.ReadReply()
	return err
}

// NoRetry returns true if the command should not be retried on failure.
// By default, commands can be retried. Commands that write directly to an
// io.Writer (like RawWriteToCmd) should override this to return true since
// partial writes cannot be undone on retry.
func (cmd *baseCmd) NoRetry() bool {
	return false
}

func (cmd *baseCmd) GetCmdType() CmdType {
	return cmd.cmdType
}

func (cmd *baseCmd) cloneBaseCmd() baseCmd {
	var readTimeout *time.Duration
	if cmd._readTimeout != nil {
		timeout := *cmd._readTimeout
		readTimeout = &timeout
	}

	// Create a copy of args slice
	args := make([]interface{}, len(cmd.args))
	copy(args, cmd.args)

	return baseCmd{
		ctx:          cmd.ctx,
		args:         args,
		err:          cmd.err,
		keyPos:       cmd.keyPos,
		_stepCount:   cmd._stepCount,
		rawVal:       cmd.rawVal,
		_readTimeout: readTimeout,
		cmdType:      cmd.cmdType,
	}
}

//------------------------------------------------------------------------------

type Cmd struct {
	baseCmd

	val interface{}
}

func NewCmd(ctx context.Context, args ...interface{}) *Cmd {
	return &Cmd{
		baseCmd: baseCmd{
			ctx:     ctx,
			args:    args,
			cmdType: CmdTypeGeneric,
		},
	}
}

func (cmd *Cmd) String() string {
	cmd.await()
	return cmdString(cmd, cmd.val)
}

func (cmd *Cmd) SetVal(val interface{}) {
	cmd.val = val
}

func (cmd *Cmd) Val() interface{} {
	cmd.await()
	return cmd.val
}

func (cmd *Cmd) Result() (interface{}, error) {
	cmd.await()
	return cmd.val, cmd.err
}

func (cmd *Cmd) Text() (string, error) {
	cmd.await()
	if cmd.err != nil {
		return "", cmd.err
	}
	return toString(cmd.val)
}

func toString(val interface{}) (string, error) {
	switch val := val.(type) {
	case string:
		return val, nil
	default:
		err := fmt.Errorf("redis: unexpected type=%T for String", val)
		return "", err
	}
}

func (cmd *Cmd) Int() (int, error) {
	cmd.await()
	if cmd.err != nil {
		return 0, cmd.err
	}
	switch val := cmd.val.(type) {
	case int64:
		return int(val), nil
	case string:
		return strconv.Atoi(val)
	default:
		err := fmt.Errorf("redis: unexpected type=%T for Int", val)
		return 0, err
	}
}

func (cmd *Cmd) Int64() (int64, error) {
	cmd.await()
	if cmd.err != nil {
		return 0, cmd.err
	}
	return toInt64(cmd.val)
}

func toInt64(val interface{}) (int64, error) {
	switch val := val.(type) {
	case int64:
		return val, nil
	case string:
		return strconv.ParseInt(val, 10, 64)
	default:
		err := fmt.Errorf("redis: unexpected type=%T for Int64", val)
		return 0, err
	}
}

func (cmd *Cmd) Uint64() (uint64, error) {
	cmd.await()
	if cmd.err != nil {
		return 0, cmd.err
	}
	return toUint64(cmd.val)
}

func toUint64(val interface{}) (uint64, error) {
	switch val := val.(type) {
	case int64:
		return uint64(val), nil
	case string:
		return strconv.ParseUint(val, 10, 64)
	default:
		err := fmt.Errorf("redis: unexpected type=%T for Uint64", val)
		return 0, err
	}
}

func (cmd *Cmd) Float32() (float32, error) {
	cmd.await()
	if cmd.err != nil {
		return 0, cmd.err
	}
	return toFloat32(cmd.val)
}

func toFloat32(val interface{}) (float32, error) {
	switch val := val.(type) {
	case int64:
		return float32(val), nil
	case string:
		f, err := strconv.ParseFloat(val, 32)
		if err != nil {
			return 0, err
		}
		return float32(f), nil
	default:
		err := fmt.Errorf("redis: unexpected type=%T for Float32", val)
		return 0, err
	}
}

func (cmd *Cmd) Float64() (float64, error) {
	cmd.await()
	if cmd.err != nil {
		return 0, cmd.err
	}
	return toFloat64(cmd.val)
}

func toFloat64(val interface{}) (float64, error) {
	switch val := val.(type) {
	case int64:
		return float64(val), nil
	case string:
		return strconv.ParseFloat(val, 64)
	default:
		err := fmt.Errorf("redis: unexpected type=%T for Float64", val)
		return 0, err
	}
}

func (cmd *Cmd) Bool() (bool, error) {
	cmd.await()
	if cmd.err != nil {
		return false, cmd.err
	}
	return toBool(cmd.val)
}

func toBool(val interface{}) (bool, error) {
	switch val := val.(type) {
	case bool:
		return val, nil
	case int64:
		return val != 0, nil
	case string:
		return strconv.ParseBool(val)
	default:
		err := fmt.Errorf("redis: unexpected type=%T for Bool", val)
		return false, err
	}
}

func (cmd *Cmd) Slice() ([]interface{}, error) {
	cmd.await()
	if cmd.err != nil {
		return nil, cmd.err
	}
	switch val := cmd.val.(type) {
	case []interface{}:
		return val, nil
	default:
		return nil, fmt.Errorf("redis: unexpected type=%T for Slice", val)
	}
}

func (cmd *Cmd) StringSlice() ([]string, error) {
	slice, err := cmd.Slice()
	if err != nil {
		return nil, err
	}

	ss := make([]string, len(slice))
	for i, iface := range slice {
		val, err := toString(iface)
		if err != nil {
			return nil, err
		}
		ss[i] = val
	}
	return ss, nil
}

func (cmd *Cmd) Int64Slice() ([]int64, error) {
	slice, err := cmd.Slice()
	if err != nil {
		return nil, err
	}

	nums := make([]int64, len(slice))
	for i, iface := range slice {
		val, err := toInt64(iface)
		if err != nil {
			return nil, err
		}
		nums[i] = val
	}
	return nums, nil
}

func (cmd *Cmd) Uint64Slice() ([]uint64, error) {
	slice, err := cmd.Slice()
	if err != nil {
		return nil, err
	}

	nums := make([]uint64, len(slice))
	for i, iface := range slice {
		val, err := toUint64(iface)
		if err != nil {
			return nil, err
		}
		nums[i] = val
	}
	return nums, nil
}

func (cmd *Cmd) Float32Slice() ([]float32, error) {
	slice, err := cmd.Slice()
	if err != nil {
		return nil, err
	}

	floats := make([]float32, len(slice))
	for i, iface := range slice {
		val, err := toFloat32(iface)
		if err != nil {
			return nil, err
		}
		floats[i] = val
	}
	return floats, nil
}

func (cmd *Cmd) Float64Slice() ([]float64, error) {
	slice, err := cmd.Slice()
	if err != nil {
		return nil, err
	}

	floats := make([]float64, len(slice))
	for i, iface := range slice {
		val, err := toFloat64(iface)
		if err != nil {
			return nil, err
		}
		floats[i] = val
	}
	return floats, nil
}

func (cmd *Cmd) BoolSlice() ([]bool, error) {
	slice, err := cmd.Slice()
	if err != nil {
		return nil, err
	}

	bools := make([]bool, len(slice))
	for i, iface := range slice {
		val, err := toBool(iface)
		if err != nil {
			return nil, err
		}
		bools[i] = val
	}
	return bools, nil
}

func (cmd *Cmd) readReply(rd *proto.Reader) (err error) {
	cmd.val, err = rd.ReadReply()
	return err
}

func (cmd *Cmd) Clone() Cmder {
	return &Cmd{
		baseCmd: cmd.cloneBaseCmd(),
		val:     cmd.val,
	}
}

//------------------------------------------------------------------------------

// RawCmd returns raw RESP protocol bytes without parsing.
type RawCmd struct {
	baseCmd
	val []byte
}

var _ Cmder = (*RawCmd)(nil)

func NewRawCmd(ctx context.Context, args ...interface{}) *RawCmd {
	return &RawCmd{
		baseCmd: baseCmd{
			ctx:     ctx,
			args:    args,
			cmdType: CmdTypeGeneric,
		},
	}
}

func (cmd *RawCmd) SetVal(val []byte) {
	cmd.val = val
}

func (cmd *RawCmd) Val() []byte {
	cmd.await()
	return cmd.val
}

func (cmd *RawCmd) Result() ([]byte, error) {
	cmd.await()
	return cmd.val, cmd.err
}

func (cmd *RawCmd) Bytes() ([]byte, error) {
	cmd.await()
	return cmd.val, cmd.err
}

func (cmd *RawCmd) String() string {
	cmd.await()
	return cmdString(cmd, cmd.val)
}

func (cmd *RawCmd) readReply(rd *proto.Reader) (err error) {
	cmd.val, err = rd.ReadRawReply()
	return err
}

func (cmd *RawCmd) Clone() Cmder {
	var val []byte
	if cmd.val != nil {
		val = make([]byte, len(cmd.val))
		copy(val, cmd.val)
	}
	return &RawCmd{
		baseCmd: cmd.cloneBaseCmd(),
		val:     val,
	}
}

//------------------------------------------------------------------------------

// RawWriteToCmd streams raw RESP protocol bytes directly to an io.Writer without intermediate allocations.
type RawWriteToCmd struct {
	baseCmd
	w       io.Writer
	written int64
}

var _ Cmder = (*RawWriteToCmd)(nil)

func NewRawWriteToCmd(ctx context.Context, w io.Writer, args ...interface{}) *RawWriteToCmd {
	return &RawWriteToCmd{
		baseCmd: baseCmd{
			ctx:     ctx,
			args:    args,
			cmdType: CmdTypeGeneric,
		},
		w: w,
	}
}

func (cmd *RawWriteToCmd) SetVal(written int64) {
	cmd.written = written
}

func (cmd *RawWriteToCmd) Val() int64 {
	cmd.await()
	return cmd.written
}

func (cmd *RawWriteToCmd) Result() (int64, error) {
	cmd.await()
	return cmd.written, cmd.err
}

func (cmd *RawWriteToCmd) String() string {
	cmd.await()
	return cmdString(cmd, cmd.written)
}

func (cmd *RawWriteToCmd) readReply(rd *proto.Reader) (err error) {
	cmd.written, err = rd.ReadRawReplyWriteTo(cmd.w)
	return err
}

// NoRetry returns true because RawWriteToCmd writes directly to an io.Writer.
// If a retry occurs, partial data from failed attempts would be appended to
// the writer, causing data corruption. The caller must handle retries manually
// if needed, using a fresh writer for each attempt.
func (cmd *RawWriteToCmd) NoRetry() bool {
	return true
}

func (cmd *RawWriteToCmd) Clone() Cmder {
	return &RawWriteToCmd{
		baseCmd: cmd.cloneBaseCmd(),
		w:       cmd.w,
		written: cmd.written,
	}
}

//------------------------------------------------------------------------------

// ZeroCopyStringCmd reads a bulk string response directly into a user-provided
// buffer, avoiding intermediate allocations. The RESP header is parsed through
// the buffered reader, then bulk data is read straight into the caller's buffer
// via proto.Reader.ReadStringInto — for values larger than the bufio buffer,
// this is effectively zero-copy from the socket to the user buffer.
//
// The buffer must be sized to fit the value; if it is too small an error is
// returned and the payload plus trailing CRLF are drained from the reader so
// the connection stays aligned for subsequent commands.
type ZeroCopyStringCmd struct {
	baseCmd
	buf    []byte // user-provided buffer to read into
	n      int    // number of bytes read into buf
	cloned bool   // set by Clone(); causes readReply to drain + error
}

var _ Cmder = (*ZeroCopyStringCmd)(nil)

func NewZeroCopyStringCmd(ctx context.Context, buf []byte, args ...interface{}) *ZeroCopyStringCmd {
	return &ZeroCopyStringCmd{
		baseCmd: baseCmd{
			ctx:     ctx,
			args:    args,
			cmdType: CmdTypeString,
		},
		buf: buf,
	}
}

func (cmd *ZeroCopyStringCmd) SetVal(n int) {
	cmd.n = n
}

func (cmd *ZeroCopyStringCmd) Val() int {
	cmd.await()
	return cmd.n
}

// Result returns the number of bytes read and any error.
func (cmd *ZeroCopyStringCmd) Result() (int, error) {
	cmd.await()
	return cmd.n, cmd.err
}

// Bytes returns the slice of the user-provided buffer containing the read data.
func (cmd *ZeroCopyStringCmd) Bytes() []byte {
	cmd.await()
	return cmd.buf[:cmd.n]
}

func (cmd *ZeroCopyStringCmd) String() string {
	cmd.await()
	return cmdString(cmd, cmd.n)
}

func (cmd *ZeroCopyStringCmd) readReply(rd *proto.Reader) error {
	// Reset the byte count before reading so a previous successful run
	// can't leak its data through Bytes() if this call errors out before
	// updating cmd.n.
	cmd.n = 0
	if cmd.cloned {
		// A cloned ZeroCopyStringCmd has no usable destination buffer
		// (see Clone for the rationale). Drain the network reply so the
		// connection stays aligned for the next command, then surface a
		// clear error rather than silently producing a wrong result.
		if err := rd.DiscardNext(); err != nil {
			return err
		}
		return fmt.Errorf("redis: ZeroCopyStringCmd cannot be cloned (cmd writes into caller-owned memory)")
	}
	n, err := rd.ReadStringInto(cmd.buf)
	if err != nil {
		return err
	}
	cmd.n = n
	return nil
}

// NoRetry returns true because the response is written directly into the
// caller's buffer. A retry could leave partial data from a failed attempt in
// the buffer, so the caller must handle retries explicitly if needed.
func (cmd *ZeroCopyStringCmd) NoRetry() bool {
	return true
}

// Clone returns a clone that is intentionally non-functional. Cloning a
// ZeroCopyStringCmd has no well-defined semantics: the cmd writes into
// caller-owned memory (the buf passed to GetToBuffer), and a clone can
// neither safely share that buf (concurrent writes from sibling clones
// would race, last-writer wins) nor allocate its own buf (the result
// would be invisible to whoever asked for the original cmd's reply).
//
// The Cmder interface requires Clone, so we return a clone marked so
// that its readReply drains the network reply (keeping the connection
// aligned) and fails the cmd with a clear error. Whoever processes the
// clone gets an explicit error instead of silently-wrong bytes.
//
// In practice this path is unreachable through normal client flows:
// Clone is only called from cluster fan-out routing
// (osscluster_router.go) for multi-shard commands like DBSIZE / KEYS /
// FLUSHDB, and ZeroCopyStringCmd is only produced by GetToBuffer which
// issues GET — a single-key command routed to one shard, never fanned
// out. Combined with NoRetry() returning true, the retry path also will
// not clone this cmd.
func (cmd *ZeroCopyStringCmd) Clone() Cmder {
	return &ZeroCopyStringCmd{
		baseCmd: cmd.cloneBaseCmd(),
		cloned:  true,
	}
}

//------------------------------------------------------------------------------

type SliceCmd struct {
	baseCmd

	val []interface{}
}

var _ Cmder = (*SliceCmd)(nil)

func NewSliceCmd(ctx context.Context, args ...interface{}) *SliceCmd {
	return &SliceCmd{
		baseCmd: baseCmd{
			ctx:     ctx,
			args:    args,
			cmdType: CmdTypeSlice,
		},
	}
}

func (cmd *SliceCmd) SetVal(val []interface{}) {
	cmd.val = val
}

func (cmd *SliceCmd) Val() []interface{} {
	cmd.await()
	return cmd.val
}

func (cmd *SliceCmd) Result() ([]interface{}, error) {
	cmd.await()
	return cmd.val, cmd.err
}

func (cmd *SliceCmd) String() string {
	cmd.await()
	return cmdString(cmd, cmd.val)
}

// Scan scans the results from the map into a destination struct. The map keys
// are matched in the Redis struct fields by the `redis:"field"` tag.
func (cmd *SliceCmd) Scan(dst interface{}) error {
	cmd.await()
	if cmd.err != nil {
		return cmd.err
	}

	// Pass the list of keys and values.
	// Skip the first two args for: HMGET key
	var args []interface{}
	if cmd.args[0] == "hmget" {
		args = cmd.args[2:]
	} else {
		// Otherwise, it's: MGET field field ...
		args = cmd.args[1:]
	}

	return hscan.Scan(dst, args, cmd.val)
}

func (cmd *SliceCmd) readReply(rd *proto.Reader) (err error) {
	cmd.val, err = rd.ReadSlice()
	return err
}

func (cmd *SliceCmd) Clone() Cmder {
	var val []interface{}
	if cmd.val != nil {
		val = make([]interface{}, len(cmd.val))
		copy(val, cmd.val)
	}
	return &SliceCmd{
		baseCmd: cmd.cloneBaseCmd(),
		val:     val,
	}
}

//------------------------------------------------------------------------------

type StatusCmd struct {
	baseCmd

	val string
}

var _ Cmder = (*StatusCmd)(nil)

func NewStatusCmd(ctx context.Context, args ...interface{}) *StatusCmd {
	return &StatusCmd{
		baseCmd: baseCmd{
			ctx:     ctx,
			args:    args,
			cmdType: CmdTypeStatus,
		},
	}
}

func (cmd *StatusCmd) SetVal(val string) {
	cmd.val = val
}

func (cmd *StatusCmd) Val() string {
	cmd.await()
	return cmd.val
}

func (cmd *StatusCmd) Result() (string, error) {
	cmd.await()
	return cmd.val, cmd.err
}

func (cmd *StatusCmd) Bytes() ([]byte, error) {
	cmd.await()
	return util.StringToBytes(cmd.val), cmd.err
}

func (cmd *StatusCmd) String() string {
	cmd.await()
	return cmdString(cmd, cmd.val)
}

func (cmd *StatusCmd) readReply(rd *proto.Reader) (err error) {
	cmd.val, err = rd.ReadString()
	return err
}

func (cmd *StatusCmd) Clone() Cmder {
	return &StatusCmd{
		baseCmd: cmd.cloneBaseCmd(),
		val:     cmd.val,
	}
}

//------------------------------------------------------------------------------

type IntCmd struct {
	baseCmd

	val int64
}

var _ Cmder = (*IntCmd)(nil)

func NewIntCmd(ctx context.Context, args ...interface{}) *IntCmd {
	return &IntCmd{
		baseCmd: baseCmd{
			ctx:     ctx,
			args:    args,
			cmdType: CmdTypeInt,
		},
	}
}

func (cmd *IntCmd) SetVal(val int64) {
	cmd.val = val
}

func (cmd *IntCmd) Val() int64 {
	cmd.await()
	return cmd.val
}

func (cmd *IntCmd) Result() (int64, error) {
	cmd.await()
	return cmd.val, cmd.err
}

func (cmd *IntCmd) Uint64() (uint64, error) {
	cmd.await()
	return uint64(cmd.val), cmd.err
}

func (cmd *IntCmd) String() string {
	cmd.await()
	return cmdString(cmd, cmd.val)
}

func (cmd *IntCmd) readReply(rd *proto.Reader) (err error) {
	cmd.val, err = rd.ReadInt()
	return err
}

func (cmd *IntCmd) Clone() Cmder {
	return &IntCmd{
		baseCmd: cmd.cloneBaseCmd(),
		val:     cmd.val,
	}
}

type UintCmd struct {
	baseCmd

	val uint64
}

var _ Cmder = (*UintCmd)(nil)

func NewUintCmd(ctx context.Context, args ...any) *UintCmd {
	return &UintCmd{
		baseCmd: baseCmd{
			ctx:     ctx,
			args:    args,
			cmdType: CmdTypeUint,
		},
	}
}

func (cmd *UintCmd) SetVal(val uint64) {
	cmd.val = val
}

func (cmd *UintCmd) Val() uint64 {
	cmd.await()
	return cmd.val
}

func (cmd *UintCmd) Result() (uint64, error) {
	cmd.await()
	return cmd.val, cmd.err
}

func (cmd *UintCmd) String() string {
	cmd.await()
	return cmdString(cmd, cmd.val)
}

func (cmd *UintCmd) readReply(rd *proto.Reader) (err error) {
	cmd.val, err = rd.ReadUint()
	return err
}

func (cmd *UintCmd) Clone() Cmder {
	return &UintCmd{
		baseCmd: cmd.cloneBaseCmd(),
		val:     cmd.val,
	}
}

//------------------------------------------------------------------------------

// DigestCmd is a command that returns a uint64 xxh3 hash digest.
//
// This command is specifically designed for the Redis DIGEST command,
// which returns the xxh3 hash of a key's value as a hex string.
// The hex string is automatically parsed to a uint64 value.
//
// The digest can be used for optimistic locking with SetIFDEQ, SetIFDNE,
// and DelExArgs commands.
//
// For examples of client-side digest generation and usage patterns, see:
// example/digest-optimistic-locking/
//
// Redis 8.4+. See https://redis.io/commands/digest/
type DigestCmd struct {
	baseCmd

	val uint64
}

var _ Cmder = (*DigestCmd)(nil)

func NewDigestCmd(ctx context.Context, args ...interface{}) *DigestCmd {
	return &DigestCmd{
		baseCmd: baseCmd{
			ctx:  ctx,
			args: args,
		},
	}
}

func (cmd *DigestCmd) SetVal(val uint64) {
	cmd.val = val
}

func (cmd *DigestCmd) Val() uint64 {
	cmd.await()
	return cmd.val
}

func (cmd *DigestCmd) Result() (uint64, error) {
	cmd.await()
	return cmd.val, cmd.err
}

func (cmd *DigestCmd) String() string {
	cmd.await()
	return cmdString(cmd, cmd.val)
}

func (cmd *DigestCmd) Clone() Cmder {
	return &DigestCmd{
		baseCmd: cmd.cloneBaseCmd(),
		val:     cmd.val,
	}
}

func (cmd *DigestCmd) readReply(rd *proto.Reader) (err error) {
	// Redis DIGEST command returns a hex string (e.g., "a1b2c3d4e5f67890")
	// We parse it as a uint64 xxh3 hash value
	var hexStr string
	hexStr, err = rd.ReadString()
	if err != nil {
		return err
	}

	// Parse hex string to uint64
	cmd.val, err = strconv.ParseUint(hexStr, 16, 64)
	return err
}

//------------------------------------------------------------------------------

type IntSliceCmd struct {
	baseCmd

	val []int64
}

var _ Cmder = (*IntSliceCmd)(nil)

func NewIntSliceCmd(ctx context.Context, args ...interface{}) *IntSliceCmd {
	return &IntSliceCmd{
		baseCmd: baseCmd{
			ctx:     ctx,
			args:    args,
			cmdType: CmdTypeIntSlice,
		},
	}
}

func (cmd *IntSliceCmd) SetVal(val []int64) {
	cmd.val = val
}

func (cmd *IntSliceCmd) Val() []int64 {
	cmd.await()
	return cmd.val
}

func (cmd *IntSliceCmd) Result() ([]int64, error) {
	cmd.await()
	return cmd.val, cmd.err
}

func (cmd *IntSliceCmd) String() string {
	cmd.await()
	return cmdString(cmd, cmd.val)
}

func (cmd *IntSliceCmd) readReply(rd *proto.Reader) error {
	n, err := rd.ReadArrayLen()
	if err != nil {
		return err
	}
	cmd.val = make([]int64, n)
	for i := 0; i < len(cmd.val); i++ {
		switch num, err := rd.ReadInt(); {
		case err == Nil:
			cmd.val[i] = 0
		case err != nil:
			return err
		default:
			cmd.val[i] = num
		}
	}
	return nil
}

func (cmd *IntSliceCmd) Clone() Cmder {
	var val []int64
	if cmd.val != nil {
		val = make([]int64, len(cmd.val))
		copy(val, cmd.val)
	}
	return &IntSliceCmd{
		baseCmd: cmd.cloneBaseCmd(),
		val:     val,
	}
}

type UintSliceCmd struct {
	baseCmd

	val []uint64
}

var _ Cmder = (*UintSliceCmd)(nil)

func NewUintSliceCmd(ctx context.Context, args ...any) *UintSliceCmd {
	return &UintSliceCmd{
		baseCmd: baseCmd{
			ctx:     ctx,
			args:    args,
			cmdType: CmdTypeUintSlice,
		},
	}
}

func (cmd *UintSliceCmd) SetVal(val []uint64) {
	cmd.val = val
}

func (cmd *UintSliceCmd) Val() []uint64 {
	cmd.await()
	return cmd.val
}

func (cmd *UintSliceCmd) Result() ([]uint64, error) {
	cmd.await()
	return cmd.val, cmd.err
}

func (cmd *UintSliceCmd) String() string {
	cmd.await()
	return cmdString(cmd, cmd.val)
}

func (cmd *UintSliceCmd) readReply(rd *proto.Reader) error {
	n, err := rd.ReadArrayLen()
	if err != nil {
		return err
	}
	cmd.val = make([]uint64, n)
	for i := range cmd.val {
		switch num, err := rd.ReadUint(); {
		case err == Nil:
			cmd.val[i] = 0
		case err != nil:
			return err
		default:
			cmd.val[i] = num
		}
	}
	return nil
}

func (cmd *UintSliceCmd) Clone() Cmder {
	var val []uint64
	if cmd.val != nil {
		val = make([]uint64, len(cmd.val))
		copy(val, cmd.val)
	}
	return &UintSliceCmd{
		baseCmd: cmd.cloneBaseCmd(),
		val:     val,
	}
}

//------------------------------------------------------------------------------

type DurationCmd struct {
	baseCmd

	val       time.Duration
	precision time.Duration
}

var _ Cmder = (*DurationCmd)(nil)

func NewDurationCmd(ctx context.Context, precision time.Duration, args ...interface{}) *DurationCmd {
	return &DurationCmd{
		baseCmd: baseCmd{
			ctx:     ctx,
			args:    args,
			cmdType: CmdTypeDuration,
		},
		precision: precision,
	}
}

func (cmd *DurationCmd) SetVal(val time.Duration) {
	cmd.val = val
}

func (cmd *DurationCmd) Val() time.Duration {
	cmd.await()
	return cmd.val
}

func (cmd *DurationCmd) Result() (time.Duration, error) {
	cmd.await()
	return cmd.val, cmd.err
}

func (cmd *DurationCmd) String() string {
	cmd.await()
	return cmdString(cmd, cmd.val)
}

func (cmd *DurationCmd) readReply(rd *proto.Reader) error {
	n, err := rd.ReadInt()
	if err != nil {
		return err
	}
	switch n {
	// -2 if the key does not exist
	// -1 if the key exists but has no associated expire
	case -2, -1:
		cmd.val = time.Duration(n)
	default:
		cmd.val = time.Duration(n) * cmd.precision
	}
	return nil
}

func (cmd *DurationCmd) Clone() Cmder {
	return &DurationCmd{
		baseCmd:   cmd.cloneBaseCmd(),
		val:       cmd.val,
		precision: cmd.precision,
	}
}

//------------------------------------------------------------------------------

type TimeCmd struct {
	baseCmd

	val time.Time
}

var _ Cmder = (*TimeCmd)(nil)

func NewTimeCmd(ctx context.Context, args ...interface{}) *TimeCmd {
	return &TimeCmd{
		baseCmd: baseCmd{
			ctx:     ctx,
			args:    args,
			cmdType: CmdTypeTime,
		},
	}
}

func (cmd *TimeCmd) SetVal(val time.Time) {
	cmd.val = val
}

func (cmd *TimeCmd) Val() time.Time {
	cmd.await()
	return cmd.val
}

func (cmd *TimeCmd) Result() (time.Time, error) {
	cmd.await()
	return cmd.val, cmd.err
}

func (cmd *TimeCmd) String() string {
	cmd.await()
	return cmdString(cmd, cmd.val)
}

func (cmd *TimeCmd) readReply(rd *proto.Reader) error {
	if err := rd.ReadFixedArrayLen(2); err != nil {
		return err
	}
	second, err := rd.ReadInt()
	if err != nil {
		return err
	}
	microsecond, err := rd.ReadInt()
	if err != nil {
		return err
	}
	cmd.val = time.Unix(second, microsecond*1000)
	return nil
}

func (cmd *TimeCmd) Clone() Cmder {
	return &TimeCmd{
		baseCmd: cmd.cloneBaseCmd(),
		val:     cmd.val,
	}
}

//------------------------------------------------------------------------------

type BoolCmd struct {
	baseCmd

	val bool
}

var _ Cmder = (*BoolCmd)(nil)

func NewBoolCmd(ctx context.Context, args ...interface{}) *BoolCmd {
	return &BoolCmd{
		baseCmd: baseCmd{
			ctx:     ctx,
			args:    args,
			cmdType: CmdTypeBool,
		},
	}
}

func (cmd *BoolCmd) SetVal(val bool) {
	cmd.val = val
}

func (cmd *BoolCmd) Val() bool {
	cmd.await()
	return cmd.val
}

func (cmd *BoolCmd) Result() (bool, error) {
	cmd.await()
	return cmd.val, cmd.err
}

func (cmd *BoolCmd) String() string {
	cmd.await()
	return cmdString(cmd, cmd.val)
}

func (cmd *BoolCmd) readReply(rd *proto.Reader) (err error) {
	cmd.val, err = rd.ReadBool()

	// `SET key value NX` returns nil when key already exists. But
	// `SETNX key value` returns bool (0/1). So convert nil to bool.
	if err == Nil {
		cmd.val = false
		err = nil
	}
	return err
}

func (cmd *BoolCmd) Clone() Cmder {
	return &BoolCmd{
		baseCmd: cmd.cloneBaseCmd(),
		val:     cmd.val,
	}
}

//------------------------------------------------------------------------------

type StringCmd struct {
	baseCmd

	val string
}

var _ Cmder = (*StringCmd)(nil)

func NewStringCmd(ctx context.Context, args ...interface{}) *StringCmd {
	return &StringCmd{
		baseCmd: baseCmd{
			ctx:     ctx,
			args:    args,
			cmdType: CmdTypeString,
		},
	}
}

func (cmd *StringCmd) SetVal(val string) {
	cmd.val = val
}

func (cmd *StringCmd) Val() string {
	cmd.await()
	return cmd.val
}

func (cmd *StringCmd) Result() (string, error) {
	cmd.await()
	return cmd.val, cmd.err
}

func (cmd *StringCmd) Bytes() ([]byte, error) {
	cmd.await()
	return util.StringToBytes(cmd.val), cmd.err
}

func (cmd *StringCmd) Bool() (bool, error) {
	cmd.await()
	if cmd.err != nil {
		return false, cmd.err
	}
	return strconv.ParseBool(cmd.val)
}

func (cmd *StringCmd) Int() (int, error) {
	cmd.await()
	if cmd.err != nil {
		return 0, cmd.err
	}
	return strconv.Atoi(cmd.val)
}

func (cmd *StringCmd) Int64() (int64, error) {
	cmd.await()
	if cmd.err != nil {
		return 0, cmd.err
	}
	return strconv.ParseInt(cmd.val, 10, 64)
}

func (cmd *StringCmd) Uint64() (uint64, error) {
	cmd.await()
	if cmd.err != nil {
		return 0, cmd.err
	}
	return strconv.ParseUint(cmd.val, 10, 64)
}

func (cmd *StringCmd) Float32() (float32, error) {
	cmd.await()
	if cmd.err != nil {
		return 0, cmd.err
	}
	f, err := strconv.ParseFloat(cmd.val, 32)
	if err != nil {
		return 0, err
	}
	return float32(f), nil
}

func (cmd *StringCmd) Float64() (float64, error) {
	cmd.await()
	if cmd.err != nil {
		return 0, cmd.err
	}
	return strconv.ParseFloat(cmd.val, 64)
}

func (cmd *StringCmd) Time() (time.Time, error) {
	cmd.await()
	if cmd.err != nil {
		return time.Time{}, cmd.err
	}
	return time.Parse(time.RFC3339Nano, cmd.val)
}

func (cmd *StringCmd) Scan(val interface{}) error {
	cmd.await()
	if cmd.err != nil {
		return cmd.err
	}
	return proto.Scan([]byte(cmd.val), val)
}

func (cmd *StringCmd) String() string {
	cmd.await()
	return cmdString(cmd, cmd.val)
}

func (cmd *StringCmd) readReply(rd *proto.Reader) (err error) {
	cmd.val, err = rd.ReadString()
	return err
}

func (cmd *StringCmd) Clone() Cmder {
	return &StringCmd{
		baseCmd: cmd.cloneBaseCmd(),
		val:     cmd.val,
	}
}

//------------------------------------------------------------------------------

type FloatCmd struct {
	baseCmd

	val float64
}

var _ Cmder = (*FloatCmd)(nil)

func NewFloatCmd(ctx context.Context, args ...interface{}) *FloatCmd {
	return &FloatCmd{
		baseCmd: baseCmd{
			ctx:     ctx,
			args:    args,
			cmdType: CmdTypeFloat,
		},
	}
}

func (cmd *FloatCmd) SetVal(val float64) {
	cmd.val = val
}

func (cmd *FloatCmd) Val() float64 {
	cmd.await()
	return cmd.val
}

func (cmd *FloatCmd) Result() (float64, error) {
	cmd.await()
	return cmd.val, cmd.err
}

func (cmd *FloatCmd) String() string {
	cmd.await()
	return cmdString(cmd, cmd.val)
}

func (cmd *FloatCmd) readReply(rd *proto.Reader) (err error) {
	cmd.val, err = rd.ReadFloat()
	return err
}

func (cmd *FloatCmd) Clone() Cmder {
	return &FloatCmd{
		baseCmd: cmd.cloneBaseCmd(),
		val:     cmd.val,
	}
}

//------------------------------------------------------------------------------

type FloatSliceCmd struct {
	baseCmd

	val []float64
}

var _ Cmder = (*FloatSliceCmd)(nil)

func NewFloatSliceCmd(ctx context.Context, args ...interface{}) *FloatSliceCmd {
	return &FloatSliceCmd{
		baseCmd: baseCmd{
			ctx:     ctx,
			args:    args,
			cmdType: CmdTypeFloatSlice,
		},
	}
}

func (cmd *FloatSliceCmd) SetVal(val []float64) {
	cmd.val = val
}

func (cmd *FloatSliceCmd) Val() []float64 {
	cmd.await()
	return cmd.val
}

func (cmd *FloatSliceCmd) Result() ([]float64, error) {
	cmd.await()
	return cmd.val, cmd.err
}

func (cmd *FloatSliceCmd) String() string {
	cmd.await()
	return cmdString(cmd, cmd.val)
}

func (cmd *FloatSliceCmd) readReply(rd *proto.Reader) error {
	n, err := rd.ReadArrayLen()
	if err != nil {
		return err
	}

	cmd.val = make([]float64, n)
	for i := 0; i < len(cmd.val); i++ {
		switch num, err := rd.ReadFloat(); {
		case err == Nil:
			cmd.val[i] = 0
		case err != nil:
			return err
		default:
			cmd.val[i] = num
		}
	}
	return nil
}

func (cmd *FloatSliceCmd) Clone() Cmder {
	var val []float64
	if cmd.val != nil {
		val = make([]float64, len(cmd.val))
		copy(val, cmd.val)
	}
	return &FloatSliceCmd{
		baseCmd: cmd.cloneBaseCmd(),
		val:     val,
	}
}

//------------------------------------------------------------------------------

type StringSliceCmd struct {
	baseCmd

	val []string
}

var _ Cmder = (*StringSliceCmd)(nil)

func NewStringSliceCmd(ctx context.Context, args ...interface{}) *StringSliceCmd {
	return &StringSliceCmd{
		baseCmd: baseCmd{
			ctx:     ctx,
			args:    args,
			cmdType: CmdTypeStringSlice,
		},
	}
}

func (cmd *StringSliceCmd) SetVal(val []string) {
	cmd.val = val
}

func (cmd *StringSliceCmd) Val() []string {
	cmd.await()
	return cmd.val
}

func (cmd *StringSliceCmd) Result() ([]string, error) {
	cmd.await()
	return cmd.val, cmd.err
}

func (cmd *StringSliceCmd) String() string {
	cmd.await()
	return cmdString(cmd, cmd.val)
}

func (cmd *StringSliceCmd) ScanSlice(container interface{}) error {
	cmd.await()
	return proto.ScanSlice(cmd.val, container)
}

func (cmd *StringSliceCmd) readReply(rd *proto.Reader) error {
	n, err := rd.ReadArrayLen()
	if err != nil {
		return err
	}
	cmd.val = make([]string, n)
	for i := 0; i < len(cmd.val); i++ {
		switch s, err := rd.ReadString(); {
		case err == Nil:
			cmd.val[i] = ""
		case err != nil:
			return err
		default:
			cmd.val[i] = s
		}
	}
	return nil
}

func (cmd *StringSliceCmd) Clone() Cmder {
	var val []string
	if cmd.val != nil {
		val = make([]string, len(cmd.val))
		copy(val, cmd.val)
	}
	return &StringSliceCmd{
		baseCmd: cmd.cloneBaseCmd(),
		val:     val,
	}
}

//------------------------------------------------------------------------------

// StringSliceSliceCmd returns a slice of string slices ([][]string).
// This is used for commands like VLINKS that return an array of arrays.
type StringSliceSliceCmd struct {
	baseCmd

	val [][]string
}

var _ Cmder = (*StringSliceSliceCmd)(nil)

func NewStringSliceSliceCmd(ctx context.Context, args ...any) *StringSliceSliceCmd {
	return &StringSliceSliceCmd{
		baseCmd: baseCmd{
			ctx:  ctx,
			args: args,
		},
	}
}

func (cmd *StringSliceSliceCmd) SetVal(val [][]string) {
	cmd.val = val
}

func (cmd *StringSliceSliceCmd) Val() [][]string {
	cmd.await()
	return cmd.val
}

func (cmd *StringSliceSliceCmd) Result() ([][]string, error) {
	cmd.await()
	return cmd.val, cmd.err
}

func (cmd *StringSliceSliceCmd) String() string {
	cmd.await()
	return cmdString(cmd, cmd.val)
}

func (cmd *StringSliceSliceCmd) readReply(rd *proto.Reader) error {
	n, err := rd.ReadArrayLen()
	if err != nil {
		return err
	}
	cmd.val = make([][]string, n)
	for i := range n {
		// Read inner array
		innerN, err := rd.ReadArrayLen()
		if err != nil {
			return err
		}
		cmd.val[i] = make([]string, innerN)
		for j := range innerN {
			switch s, err := rd.ReadString(); {
			case err == Nil:
				cmd.val[i][j] = ""
			case err != nil:
				return err
			default:
				cmd.val[i][j] = s
			}
		}
	}
	return nil
}

func (cmd *StringSliceSliceCmd) Clone() Cmder {
	var val [][]string
	if cmd.val != nil {
		val = make([][]string, len(cmd.val))
		for i, slice := range cmd.val {
			if slice != nil {
				val[i] = make([]string, len(slice))
				copy(val[i], slice)
			}
		}
	}
	return &StringSliceSliceCmd{
		baseCmd: cmd.cloneBaseCmd(),
		val:     val,
	}
}

//------------------------------------------------------------------------------

type KeyValue struct {
	Key   string
	Value string
}

type KeyValueSliceCmd struct {
	baseCmd

	val []KeyValue
}

var _ Cmder = (*KeyValueSliceCmd)(nil)

func NewKeyValueSliceCmd(ctx context.Context, args ...interface{}) *KeyValueSliceCmd {
	return &KeyValueSliceCmd{
		baseCmd: baseCmd{
			ctx:     ctx,
			args:    args,
			cmdType: CmdTypeKeyValueSlice,
		},
	}
}

func (cmd *KeyValueSliceCmd) SetVal(val []KeyValue) {
	cmd.val = val
}

func (cmd *KeyValueSliceCmd) Val() []KeyValue {
	cmd.await()
	return cmd.val
}

func (cmd *KeyValueSliceCmd) Result() ([]KeyValue, error) {
	cmd.await()
	return cmd.val, cmd.err
}

func (cmd *KeyValueSliceCmd) String() string {
	cmd.await()
	return cmdString(cmd, cmd.val)
}

// Many commands will respond to two formats:
//  1. 1) "one"
//  2. (double) 1
//  2. 1) "two"
//  2. (double) 2
//
// OR:
//  1. "two"
//  2. (double) 2
//  3. "one"
//  4. (double) 1
func (cmd *KeyValueSliceCmd) readReply(rd *proto.Reader) error { // nolint:dupl
	n, err := rd.ReadArrayLen()
	if err != nil {
		return err
	}

	// If the n is 0, can't continue reading.
	if n == 0 {
		cmd.val = make([]KeyValue, 0)
		return nil
	}

	typ, err := rd.PeekReplyType()
	if err != nil {
		return err
	}
	array := typ == proto.RespArray

	if array {
		cmd.val = make([]KeyValue, n)
	} else {
		if n%2 != 0 {
			return fmt.Errorf("redis: got %d elements in the key-value array, wanted a multiple of 2", n)
		}
		cmd.val = make([]KeyValue, n/2)
	}

	for i := 0; i < len(cmd.val); i++ {
		if array {
			if err = rd.ReadFixedArrayLen(2); err != nil {
				return err
			}
		}

		if cmd.val[i].Key, err = rd.ReadString(); err != nil {
			return err
		}

		if cmd.val[i].Value, err = rd.ReadString(); err != nil {
			return err
		}
	}

	return nil
}

func (cmd *KeyValueSliceCmd) Clone() Cmder {
	var val []KeyValue
	if cmd.val != nil {
		val = make([]KeyValue, len(cmd.val))
		copy(val, cmd.val)
	}
	return &KeyValueSliceCmd{
		baseCmd: cmd.cloneBaseCmd(),
		val:     val,
	}
}

//------------------------------------------------------------------------------

type BoolSliceCmd struct {
	baseCmd

	val []bool
}

var _ Cmder = (*BoolSliceCmd)(nil)

func NewBoolSliceCmd(ctx context.Context, args ...interface{}) *BoolSliceCmd {
	return &BoolSliceCmd{
		baseCmd: baseCmd{
			ctx:     ctx,
			args:    args,
			cmdType: CmdTypeBoolSlice,
		},
	}
}

func (cmd *BoolSliceCmd) SetVal(val []bool) {
	cmd.val = val
}

func (cmd *BoolSliceCmd) Val() []bool {
	cmd.await()
	return cmd.val
}

func (cmd *BoolSliceCmd) Result() ([]bool, error) {
	cmd.await()
	return cmd.val, cmd.err
}

func (cmd *BoolSliceCmd) String() string {
	cmd.await()
	return cmdString(cmd, cmd.val)
}

func (cmd *BoolSliceCmd) readReply(rd *proto.Reader) error {
	n, err := rd.ReadArrayLen()
	if err != nil {
		return err
	}
	cmd.val = make([]bool, n)
	for i := 0; i < len(cmd.val); i++ {
		switch b, err := rd.ReadBool(); {
		case err == Nil:
			cmd.val[i] = false
		case err != nil:
			return err
		default:
			cmd.val[i] = b
		}
	}
	return nil
}

func (cmd *BoolSliceCmd) Clone() Cmder {
	var val []bool
	if cmd.val != nil {
		val = make([]bool, len(cmd.val))
		copy(val, cmd.val)
	}
	return &BoolSliceCmd{
		baseCmd: cmd.cloneBaseCmd(),
		val:     val,
	}
}

//------------------------------------------------------------------------------

type MapStringStringCmd struct {
	baseCmd

	val map[string]string
}

var _ Cmder = (*MapStringStringCmd)(nil)

func NewMapStringStringCmd(ctx context.Context, args ...interface{}) *MapStringStringCmd {
	return &MapStringStringCmd{
		baseCmd: baseCmd{
			ctx:     ctx,
			args:    args,
			cmdType: CmdTypeMapStringString,
		},
	}
}

func (cmd *MapStringStringCmd) Val() map[string]string {
	cmd.await()
	return cmd.val
}

func (cmd *MapStringStringCmd) SetVal(val map[string]string) {
	cmd.val = val
}

func (cmd *MapStringStringCmd) Result() (map[string]string, error) {
	cmd.await()
	return cmd.val, cmd.err
}

func (cmd *MapStringStringCmd) String() string {
	cmd.await()
	return cmdString(cmd, cmd.val)
}

// Scan scans the results from the map into a destination struct. The map keys
// are matched in the Redis struct fields by the `redis:"field"` tag.
func (cmd *MapStringStringCmd) Scan(dest interface{}) error {
	cmd.await()
	if cmd.err != nil {
		return cmd.err
	}

	strct, err := hscan.Struct(dest)
	if err != nil {
		return err
	}

	for k, v := range cmd.val {
		if err := strct.Scan(k, v); err != nil {
			return err
		}
	}

	return nil
}

func (cmd *MapStringStringCmd) readReply(rd *proto.Reader) error {
	n, err := rd.ReadMapLen()
	if err != nil {
		return err
	}

	cmd.val = make(map[string]string, n)
	for i := 0; i < n; i++ {
		key, err := rd.ReadString()
		if err != nil {
			return err
		}

		value, err := rd.ReadString()
		if err != nil {
			return err
		}

		cmd.val[key] = value
	}
	return nil
}

func (cmd *MapStringStringCmd) Clone() Cmder {
	var val map[string]string
	if cmd.val != nil {
		val = make(map[string]string, len(cmd.val))
		for k, v := range cmd.val {
			val[k] = v
		}
	}
	return &MapStringStringCmd{
		baseCmd: cmd.cloneBaseCmd(),
		val:     val,
	}
}

//------------------------------------------------------------------------------

type MapStringIntCmd struct {
	baseCmd

	val map[string]int64
}

var _ Cmder = (*MapStringIntCmd)(nil)

func NewMapStringIntCmd(ctx context.Context, args ...interface{}) *MapStringIntCmd {
	return &MapStringIntCmd{
		baseCmd: baseCmd{
			ctx:     ctx,
			args:    args,
			cmdType: CmdTypeMapStringInt,
		},
	}
}

func (cmd *MapStringIntCmd) SetVal(val map[string]int64) {
	cmd.val = val
}

func (cmd *MapStringIntCmd) Val() map[string]int64 {
	cmd.await()
	return cmd.val
}

func (cmd *MapStringIntCmd) Result() (map[string]int64, error) {
	cmd.await()
	return cmd.val, cmd.err
}

func (cmd *MapStringIntCmd) String() string {
	cmd.await()
	return cmdString(cmd, cmd.val)
}

func (cmd *MapStringIntCmd) readReply(rd *proto.Reader) error {
	n, err := rd.ReadMapLen()
	if err != nil {
		return err
	}

	cmd.val = make(map[string]int64, n)
	for i := 0; i < n; i++ {
		key, err := rd.ReadString()
		if err != nil {
			return err
		}

		nn, err := rd.ReadInt()
		if err != nil {
			return err
		}
		cmd.val[key] = nn
	}
	return nil
}

func (cmd *MapStringIntCmd) Clone() Cmder {
	var val map[string]int64
	if cmd.val != nil {
		val = make(map[string]int64, len(cmd.val))
		for k, v := range cmd.val {
			val[k] = v
		}
	}
	return &MapStringIntCmd{
		baseCmd: cmd.cloneBaseCmd(),
		val:     val,
	}
}

// ------------------------------------------------------------------------------
type MapStringSliceInterfaceCmd struct {
	baseCmd
	val map[string][]interface{}
}

func NewMapStringSliceInterfaceCmd(ctx context.Context, args ...interface{}) *MapStringSliceInterfaceCmd {
	return &MapStringSliceInterfaceCmd{
		baseCmd: baseCmd{
			ctx:     ctx,
			args:    args,
			cmdType: CmdTypeMapStringInterfaceSlice,
		},
	}
}

func (cmd *MapStringSliceInterfaceCmd) String() string {
	cmd.await()
	return cmdString(cmd, cmd.val)
}

func (cmd *MapStringSliceInterfaceCmd) SetVal(val map[string][]interface{}) {
	cmd.val = val
}

func (cmd *MapStringSliceInterfaceCmd) Result() (map[string][]interface{}, error) {
	cmd.await()
	return cmd.val, cmd.err
}

func (cmd *MapStringSliceInterfaceCmd) Val() map[string][]interface{} {
	cmd.await()
	return cmd.val
}

func (cmd *MapStringSliceInterfaceCmd) readReply(rd *proto.Reader) (err error) {
	readType, err := rd.PeekReplyType()
	if err != nil {
		return err
	}

	cmd.val = make(map[string][]interface{})

	switch readType {
	case proto.RespMap:
		n, err := rd.ReadMapLen()
		if err != nil {
			return err
		}
		for i := 0; i < n; i++ {
			k, err := rd.ReadString()
			if err != nil {
				return err
			}
			nn, err := rd.ReadArrayLen()
			if err != nil {
				return err
			}
			cmd.val[k] = make([]interface{}, nn)
			for j := 0; j < nn; j++ {
				value, err := rd.ReadReply()
				if err != nil {
					return err
				}
				cmd.val[k][j] = value
			}
		}
	case proto.RespArray:
		// RESP2 response
		n, err := rd.ReadArrayLen()
		if err != nil {
			return err
		}

		for i := 0; i < n; i++ {
			// Each entry in this array is itself an array with key details
			itemLen, err := rd.ReadArrayLen()
			if err != nil {
				return err
			}

			key, err := rd.ReadString()
			if err != nil {
				return err
			}
			cmd.val[key] = make([]interface{}, 0, itemLen-1)
			for j := 1; j < itemLen; j++ {
				// Read the inner array for timestamp-value pairs
				data, err := rd.ReadReply()
				if err != nil {
					return err
				}
				cmd.val[key] = append(cmd.val[key], data)
			}
		}
	default:
		// Any other reply type leaves the peeked frame unread. Returning nil
		// here would put the connection back in the pool with those bytes
		// buffered, so the next command reads them as its own reply.
		return fmt.Errorf("redis: can't parse map-string-slice-interface reply: unexpected type %c", readType)
	}

	return nil
}

func (cmd *MapStringSliceInterfaceCmd) Clone() Cmder {
	var val map[string][]interface{}
	if cmd.val != nil {
		val = make(map[string][]interface{}, len(cmd.val))
		for k, v := range cmd.val {
			if v != nil {
				newSlice := make([]interface{}, len(v))
				copy(newSlice, v)
				val[k] = newSlice
			}
		}
	}
	return &MapStringSliceInterfaceCmd{
		baseCmd: cmd.cloneBaseCmd(),
		val:     val,
	}
}

//------------------------------------------------------------------------------

type StringStructMapCmd struct {
	baseCmd

	val map[string]struct{}
}

var _ Cmder = (*StringStructMapCmd)(nil)

func NewStringStructMapCmd(ctx context.Context, args ...interface{}) *StringStructMapCmd {
	return &StringStructMapCmd{
		baseCmd: baseCmd{
			ctx:     ctx,
			args:    args,
			cmdType: CmdTypeStringStructMap,
		},
	}
}

func (cmd *StringStructMapCmd) SetVal(val map[string]struct{}) {
	cmd.val = val
}

func (cmd *StringStructMapCmd) Val() map[string]struct{} {
	cmd.await()
	return cmd.val
}

func (cmd *StringStructMapCmd) Result() (map[string]struct{}, error) {
	cmd.await()
	return cmd.val, cmd.err
}

func (cmd *StringStructMapCmd) String() string {
	cmd.await()
	return cmdString(cmd, cmd.val)
}

func (cmd *StringStructMapCmd) readReply(rd *proto.Reader) error {
	n, err := rd.ReadArrayLen()
	if err != nil {
		return err
	}

	cmd.val = make(map[string]struct{}, n)
	for i := 0; i < n; i++ {
		key, err := rd.ReadString()
		if err != nil {
			return err
		}
		cmd.val[key] = struct{}{}
	}
	return nil
}

func (cmd *StringStructMapCmd) Clone() Cmder {
	var val map[string]struct{}
	if cmd.val != nil {
		val = maps.Clone(cmd.val)
	}
	return &StringStructMapCmd{
		baseCmd: cmd.cloneBaseCmd(),
		val:     val,
	}
}

//------------------------------------------------------------------------------

type XMessage struct {
	ID     string
	Values map[string]interface{}
	// MillisElapsedFromDelivery is the number of milliseconds since the entry was last delivered.
	// Only populated when using XREADGROUP with CLAIM argument for claimed entries.
	MillisElapsedFromDelivery int64
	// DeliveredCount is the number of times the entry was delivered.
	// Only populated when using XREADGROUP with CLAIM argument for claimed entries.
	DeliveredCount int64
}

type XMessageSliceCmd struct {
	baseCmd

	val []XMessage
}

var _ Cmder = (*XMessageSliceCmd)(nil)

func NewXMessageSliceCmd(ctx context.Context, args ...interface{}) *XMessageSliceCmd {
	return &XMessageSliceCmd{
		baseCmd: baseCmd{
			ctx:     ctx,
			args:    args,
			cmdType: CmdTypeXMessageSlice,
		},
	}
}

func (cmd *XMessageSliceCmd) SetVal(val []XMessage) {
	cmd.val = val
}

func (cmd *XMessageSliceCmd) Val() []XMessage {
	cmd.await()
	return cmd.val
}

func (cmd *XMessageSliceCmd) Result() ([]XMessage, error) {
	cmd.await()
	return cmd.val, cmd.err
}

func (cmd *XMessageSliceCmd) String() string {
	cmd.await()
	return cmdString(cmd, cmd.val)
}

func (cmd *XMessageSliceCmd) readReply(rd *proto.Reader) (err error) {
	cmd.val, err = readXMessageSlice(rd)
	return err
}

func (cmd *XMessageSliceCmd) Clone() Cmder {
	var val []XMessage
	if cmd.val != nil {
		val = make([]XMessage, len(cmd.val))
		for i, msg := range cmd.val {
			val[i] = XMessage{
				ID: msg.ID,
			}
			if msg.Values != nil {
				val[i].Values = maps.Clone(msg.Values)
			}
		}
	}
	return &XMessageSliceCmd{
		baseCmd: cmd.cloneBaseCmd(),
		val:     val,
	}
}

func readXMessageSlice(rd *proto.Reader) ([]XMessage, error) {
	n, err := rd.ReadArrayLen()
	if err != nil {
		return nil, err
	}

	msgs := make([]XMessage, n)
	for i := 0; i < len(msgs); i++ {
		if msgs[i], err = readXMessage(rd); err != nil {
			return nil, err
		}
	}
	return msgs, nil
}

func readXMessage(rd *proto.Reader) (XMessage, error) {
	// Read array length can be 2 or 4 (with CLAIM metadata)
	n, err := rd.ReadArrayLen()
	if err != nil {
		return XMessage{}, err
	}

	if n != 2 && n != 4 {
		return XMessage{}, fmt.Errorf("redis: got %d elements in the XMessage array, expected 2 or 4", n)
	}

	id, err := rd.ReadString()
	if err != nil {
		return XMessage{}, err
	}

	v, err := stringInterfaceMapParser(rd)
	if err != nil {
		if err != proto.Nil {
			return XMessage{}, err
		}
	}

	msg := XMessage{
		ID:     id,
		Values: v,
	}

	if n == 4 {
		msg.MillisElapsedFromDelivery, err = rd.ReadInt()
		if err != nil {
			return XMessage{}, err
		}

		msg.DeliveredCount, err = rd.ReadInt()
		if err != nil {
			return XMessage{}, err
		}
	}

	return msg, nil
}

func stringInterfaceMapParser(rd *proto.Reader) (map[string]interface{}, error) {
	n, err := rd.ReadMapLen()
	if err != nil {
		return nil, err
	}

	m := make(map[string]interface{}, n)
	for i := 0; i < n; i++ {
		key, err := rd.ReadString()
		if err != nil {
			return nil, err
		}

		value, err := rd.ReadString()
		if err != nil {
			return nil, err
		}

		m[key] = value
	}
	return m, nil
}

//------------------------------------------------------------------------------

type XStream struct {
	Stream   string
	Messages []XMessage
}

type XStreamSliceCmd struct {
	baseCmd

	val []XStream
}

var _ Cmder = (*XStreamSliceCmd)(nil)

func NewXStreamSliceCmd(ctx context.Context, args ...interface{}) *XStreamSliceCmd {
	return &XStreamSliceCmd{
		baseCmd: baseCmd{
			ctx:     ctx,
			args:    args,
			cmdType: CmdTypeXStreamSlice,
		},
	}
}

func (cmd *XStreamSliceCmd) SetVal(val []XStream) {
	cmd.val = val
}

func (cmd *XStreamSliceCmd) Val() []XStream {
	cmd.await()
	return cmd.val
}

func (cmd *XStreamSliceCmd) Result() ([]XStream, error) {
	cmd.await()
	return cmd.val, cmd.err
}

func (cmd *XStreamSliceCmd) String() string {
	cmd.await()
	return cmdString(cmd, cmd.val)
}

func (cmd *XStreamSliceCmd) readReply(rd *proto.Reader) error {
	typ, err := rd.PeekReplyType()
	if err != nil {
		return err
	}

	var n int
	if typ == proto.RespMap {
		n, err = rd.ReadMapLen()
	} else {
		n, err = rd.ReadArrayLen()
	}
	if err != nil {
		return err
	}
	cmd.val = make([]XStream, n)
	for i := 0; i < len(cmd.val); i++ {
		if typ != proto.RespMap {
			if err = rd.ReadFixedArrayLen(2); err != nil {
				return err
			}
		}
		if cmd.val[i].Stream, err = rd.ReadString(); err != nil {
			return err
		}
		if cmd.val[i].Messages, err = readXMessageSlice(rd); err != nil {
			return err
		}
	}
	return nil
}

func (cmd *XStreamSliceCmd) Clone() Cmder {
	var val []XStream
	if cmd.val != nil {
		val = make([]XStream, len(cmd.val))
		for i, stream := range cmd.val {
			val[i] = XStream{
				Stream: stream.Stream,
			}
			if stream.Messages != nil {
				val[i].Messages = make([]XMessage, len(stream.Messages))
				for j, msg := range stream.Messages {
					val[i].Messages[j] = XMessage{
						ID: msg.ID,
					}
					if msg.Values != nil {
						val[i].Messages[j].Values = make(map[string]interface{}, len(msg.Values))
						for k, v := range msg.Values {
							val[i].Messages[j].Values[k] = v
						}
					}
				}
			}
		}
	}
	return &XStreamSliceCmd{
		baseCmd: cmd.cloneBaseCmd(),
		val:     val,
	}
}

//------------------------------------------------------------------------------

type XPending struct {
	Count     int64
	Lower     string
	Higher    string
	Consumers map[string]int64
}

type XPendingCmd struct {
	baseCmd
	val *XPending
}

var _ Cmder = (*XPendingCmd)(nil)

func NewXPendingCmd(ctx context.Context, args ...interface{}) *XPendingCmd {
	return &XPendingCmd{
		baseCmd: baseCmd{
			ctx:     ctx,
			args:    args,
			cmdType: CmdTypeXPending,
		},
	}
}

func (cmd *XPendingCmd) SetVal(val *XPending) {
	cmd.val = val
}

func (cmd *XPendingCmd) Val() *XPending {
	cmd.await()
	return cmd.val
}

func (cmd *XPendingCmd) Result() (*XPending, error) {
	cmd.await()
	return cmd.val, cmd.err
}

func (cmd *XPendingCmd) String() string {
	cmd.await()
	return cmdString(cmd, cmd.val)
}

func (cmd *XPendingCmd) readReply(rd *proto.Reader) error {
	var err error
	if err = rd.ReadFixedArrayLen(4); err != nil {
		return err
	}
	cmd.val = &XPending{}

	if cmd.val.Count, err = rd.ReadInt(); err != nil {
		return err
	}

	if cmd.val.Lower, err = rd.ReadString(); err != nil && err != Nil {
		return err
	}

	if cmd.val.Higher, err = rd.ReadString(); err != nil && err != Nil {
		return err
	}

	n, err := rd.ReadArrayLen()
	if err != nil && err != Nil {
		return err
	}
	cmd.val.Consumers = make(map[string]int64, n)
	for i := 0; i < n; i++ {
		if err = rd.ReadFixedArrayLen(2); err != nil {
			return err
		}

		consumerName, err := rd.ReadString()
		if err != nil {
			return err
		}
		consumerPending, err := rd.ReadInt()
		if err != nil {
			return err
		}
		cmd.val.Consumers[consumerName] = consumerPending
	}
	return nil
}

func (cmd *XPendingCmd) Clone() Cmder {
	var val *XPending
	if cmd.val != nil {
		val = &XPending{
			Count:  cmd.val.Count,
			Lower:  cmd.val.Lower,
			Higher: cmd.val.Higher,
		}
		if cmd.val.Consumers != nil {
			val.Consumers = make(map[string]int64, len(cmd.val.Consumers))
			for k, v := range cmd.val.Consumers {
				val.Consumers[k] = v
			}
		}
	}
	return &XPendingCmd{
		baseCmd: cmd.cloneBaseCmd(),
		val:     val,
	}
}

//------------------------------------------------------------------------------

type XPendingExt struct {
	ID         string
	Consumer   string
	Idle       time.Duration
	RetryCount int64
}

type XPendingExtCmd struct {
	baseCmd
	val []XPendingExt
}

var _ Cmder = (*XPendingExtCmd)(nil)

func NewXPendingExtCmd(ctx context.Context, args ...interface{}) *XPendingExtCmd {
	return &XPendingExtCmd{
		baseCmd: baseCmd{
			ctx:     ctx,
			args:    args,
			cmdType: CmdTypeXPendingExt,
		},
	}
}

func (cmd *XPendingExtCmd) SetVal(val []XPendingExt) {
	cmd.val = val
}

func (cmd *XPendingExtCmd) Val() []XPendingExt {
	cmd.await()
	return cmd.val
}

func (cmd *XPendingExtCmd) Result() ([]XPendingExt, error) {
	cmd.await()
	return cmd.val, cmd.err
}

func (cmd *XPendingExtCmd) String() string {
	cmd.await()
	return cmdString(cmd, cmd.val)
}

func (cmd *XPendingExtCmd) readReply(rd *proto.Reader) error {
	n, err := rd.ReadArrayLen()
	if err != nil {
		return err
	}
	cmd.val = make([]XPendingExt, n)

	for i := 0; i < len(cmd.val); i++ {
		if err = rd.ReadFixedArrayLen(4); err != nil {
			return err
		}

		if cmd.val[i].ID, err = rd.ReadString(); err != nil {
			return err
		}

		if cmd.val[i].Consumer, err = rd.ReadString(); err != nil && err != Nil {
			return err
		}

		idle, err := rd.ReadInt()
		if err != nil && err != Nil {
			return err
		}
		cmd.val[i].Idle = time.Duration(idle) * time.Millisecond

		if cmd.val[i].RetryCount, err = rd.ReadInt(); err != nil && err != Nil {
			return err
		}
	}

	return nil
}

func (cmd *XPendingExtCmd) Clone() Cmder {
	var val []XPendingExt
	if cmd.val != nil {
		val = make([]XPendingExt, len(cmd.val))
		copy(val, cmd.val)
	}
	return &XPendingExtCmd{
		baseCmd: cmd.cloneBaseCmd(),
		val:     val,
	}
}

//------------------------------------------------------------------------------

type XAutoClaimCmd struct {
	baseCmd

	start string
	val   []XMessage
}

var _ Cmder = (*XAutoClaimCmd)(nil)

func NewXAutoClaimCmd(ctx context.Context, args ...interface{}) *XAutoClaimCmd {
	return &XAutoClaimCmd{
		baseCmd: baseCmd{
			ctx:     ctx,
			args:    args,
			cmdType: CmdTypeXAutoClaim,
		},
	}
}

func (cmd *XAutoClaimCmd) SetVal(val []XMessage, start string) {
	cmd.val = val
	cmd.start = start
}

func (cmd *XAutoClaimCmd) Val() (messages []XMessage, start string) {
	cmd.await()
	return cmd.val, cmd.start
}

func (cmd *XAutoClaimCmd) Result() (messages []XMessage, start string, err error) {
	cmd.await()
	return cmd.val, cmd.start, cmd.err
}

func (cmd *XAutoClaimCmd) String() string {
	cmd.await()
	return cmdString(cmd, cmd.val)
}

func (cmd *XAutoClaimCmd) readReply(rd *proto.Reader) error {
	n, err := rd.ReadArrayLen()
	if err != nil {
		return err
	}

	switch n {
	case 2, // Redis 6
		3: // Redis 7:
		// ok
	default:
		return fmt.Errorf("redis: got %d elements in XAutoClaim reply, wanted 2/3", n)
	}

	cmd.start, err = rd.ReadString()
	if err != nil {
		return err
	}

	cmd.val, err = readXMessageSlice(rd)
	if err != nil {
		return err
	}

	if n >= 3 {
		return rd.DiscardNext()
	}

	return nil
}

func (cmd *XAutoClaimCmd) Clone() Cmder {
	var val []XMessage
	if cmd.val != nil {
		val = make([]XMessage, len(cmd.val))
		for i, msg := range cmd.val {
			val[i] = XMessage{
				ID: msg.ID,
			}
			if msg.Values != nil {
				val[i].Values = make(map[string]interface{}, len(msg.Values))
				for k, v := range msg.Values {
					val[i].Values[k] = v
				}
			}
		}
	}
	return &XAutoClaimCmd{
		baseCmd: cmd.cloneBaseCmd(),
		start:   cmd.start,
		val:     val,
	}
}

//------------------------------------------------------------------------------

type XAutoClaimWithDeletedCmd struct {
	baseCmd

	start      string
	val        []XMessage
	deletedIDs []string
}

var _ Cmder = (*XAutoClaimWithDeletedCmd)(nil)

func NewXAutoClaimWithDeletedCmd(ctx context.Context, args ...interface{}) *XAutoClaimWithDeletedCmd {
	return &XAutoClaimWithDeletedCmd{
		baseCmd: baseCmd{
			ctx:     ctx,
			args:    args,
			cmdType: CmdTypeXAutoClaimWithDeleted,
		},
	}
}

func (cmd *XAutoClaimWithDeletedCmd) SetVal(val []XMessage, start string, deletedIDs []string) {
	cmd.val = val
	cmd.start = start
	cmd.deletedIDs = deletedIDs
}

func (cmd *XAutoClaimWithDeletedCmd) Val() (messages []XMessage, start string, deletedIDs []string) {
	cmd.await()
	return cmd.val, cmd.start, cmd.deletedIDs
}

func (cmd *XAutoClaimWithDeletedCmd) Result() (messages []XMessage, start string, deletedIDs []string, err error) {
	cmd.await()
	return cmd.val, cmd.start, cmd.deletedIDs, cmd.err
}

func (cmd *XAutoClaimWithDeletedCmd) String() string {
	cmd.await()
	return cmdString(cmd, cmd.val)
}

func (cmd *XAutoClaimWithDeletedCmd) readReply(rd *proto.Reader) error {
	n, err := rd.ReadArrayLen()
	if err != nil {
		return err
	}

	switch n {
	case 2, // Redis 6
		3: // Redis 7:
		// ok
	default:
		return fmt.Errorf("redis: got %d elements in XAutoClaim reply, wanted 2/3", n)
	}

	cmd.start, err = rd.ReadString()
	if err != nil {
		return err
	}

	cmd.val, err = readXMessageSlice(rd)
	if err != nil {
		return err
	}

	if n < 3 {
		return nil
	}

	nn, err := rd.ReadArrayLen()
	if err != nil {
		return err
	}

	cmd.deletedIDs = make([]string, nn)
	for i := 0; i < nn; i++ {
		cmd.deletedIDs[i], err = rd.ReadString()
		if err != nil {
			return err
		}
	}

	return nil
}

func (cmd *XAutoClaimWithDeletedCmd) Clone() Cmder {
	var val []XMessage
	if cmd.val != nil {
		val = make([]XMessage, len(cmd.val))
		for i, msg := range cmd.val {
			val[i] = XMessage{
				ID: msg.ID,
			}
			if msg.Values != nil {
				val[i].Values = make(map[string]interface{}, len(msg.Values))
				for k, v := range msg.Values {
					val[i].Values[k] = v
				}
			}
		}
	}
	var deletedIDs []string
	if cmd.deletedIDs != nil {
		deletedIDs = make([]string, len(cmd.deletedIDs))
		copy(deletedIDs, cmd.deletedIDs)
	}
	return &XAutoClaimWithDeletedCmd{
		baseCmd:    cmd.cloneBaseCmd(),
		start:      cmd.start,
		val:        val,
		deletedIDs: deletedIDs,
	}
}

//------------------------------------------------------------------------------

type XAutoClaimJustIDCmd struct {
	baseCmd

	start string
	val   []string
}

var _ Cmder = (*XAutoClaimJustIDCmd)(nil)

func NewXAutoClaimJustIDCmd(ctx context.Context, args ...interface{}) *XAutoClaimJustIDCmd {
	return &XAutoClaimJustIDCmd{
		baseCmd: baseCmd{
			ctx:     ctx,
			args:    args,
			cmdType: CmdTypeXAutoClaimJustID,
		},
	}
}

func (cmd *XAutoClaimJustIDCmd) SetVal(val []string, start string) {
	cmd.val = val
	cmd.start = start
}

func (cmd *XAutoClaimJustIDCmd) Val() (ids []string, start string) {
	cmd.await()
	return cmd.val, cmd.start
}

func (cmd *XAutoClaimJustIDCmd) Result() (ids []string, start string, err error) {
	cmd.await()
	return cmd.val, cmd.start, cmd.err
}

func (cmd *XAutoClaimJustIDCmd) String() string {
	cmd.await()
	return cmdString(cmd, cmd.val)
}

func (cmd *XAutoClaimJustIDCmd) readReply(rd *proto.Reader) error {
	n, err := rd.ReadArrayLen()
	if err != nil {
		return err
	}

	switch n {
	case 2, // Redis 6
		3: // Redis 7:
		// ok
	default:
		return fmt.Errorf("redis: got %d elements in XAutoClaimJustID reply, wanted 2/3", n)
	}

	cmd.start, err = rd.ReadString()
	if err != nil {
		return err
	}

	nn, err := rd.ReadArrayLen()
	if err != nil {
		return err
	}

	cmd.val = make([]string, nn)
	for i := 0; i < nn; i++ {
		cmd.val[i], err = rd.ReadString()
		if err != nil {
			return err
		}
	}

	if n >= 3 {
		if err := rd.DiscardNext(); err != nil {
			return err
		}
	}

	return nil
}

func (cmd *XAutoClaimJustIDCmd) Clone() Cmder {
	var val []string
	if cmd.val != nil {
		val = make([]string, len(cmd.val))
		copy(val, cmd.val)
	}
	return &XAutoClaimJustIDCmd{
		baseCmd: cmd.cloneBaseCmd(),
		start:   cmd.start,
		val:     val,
	}
}

//------------------------------------------------------------------------------

type XInfoConsumersCmd struct {
	baseCmd
	val []XInfoConsumer
}

type XInfoConsumer struct {
	Name     string
	Pending  int64
	Idle     time.Duration
	Inactive time.Duration
}

var _ Cmder = (*XInfoConsumersCmd)(nil)

func NewXInfoConsumersCmd(ctx context.Context, stream string, group string) *XInfoConsumersCmd {
	return &XInfoConsumersCmd{
		baseCmd: baseCmd{
			ctx:     ctx,
			args:    []interface{}{"xinfo", "consumers", stream, group},
			cmdType: CmdTypeXInfoConsumers,
		},
	}
}

func (cmd *XInfoConsumersCmd) SetVal(val []XInfoConsumer) {
	cmd.val = val
}

func (cmd *XInfoConsumersCmd) Val() []XInfoConsumer {
	cmd.await()
	return cmd.val
}

func (cmd *XInfoConsumersCmd) Result() ([]XInfoConsumer, error) {
	cmd.await()
	return cmd.val, cmd.err
}

func (cmd *XInfoConsumersCmd) String() string {
	cmd.await()
	return cmdString(cmd, cmd.val)
}

func (cmd *XInfoConsumersCmd) readReply(rd *proto.Reader) error {
	n, err := rd.ReadArrayLen()
	if err != nil {
		return err
	}
	cmd.val = make([]XInfoConsumer, n)

	for i := 0; i < len(cmd.val); i++ {
		nn, err := rd.ReadMapLen()
		if err != nil {
			return err
		}

		var key string
		for f := 0; f < nn; f++ {
			key, err = rd.ReadString()
			if err != nil {
				return err
			}

			switch key {
			case "name":
				cmd.val[i].Name, err = rd.ReadString()
			case "pending":
				cmd.val[i].Pending, err = rd.ReadInt()
			case "idle":
				var idle int64
				idle, err = rd.ReadInt()
				cmd.val[i].Idle = time.Duration(idle) * time.Millisecond
			case "inactive":
				var inactive int64
				inactive, err = rd.ReadInt()
				cmd.val[i].Inactive = time.Duration(inactive) * time.Millisecond
			default:
				// skip unknown fields
				if err = rd.DiscardNext(); err != nil {
					return err
				}
			}
			if err != nil {
				return err
			}
		}
	}

	return nil
}

func (cmd *XInfoConsumersCmd) Clone() Cmder {
	var val []XInfoConsumer
	if cmd.val != nil {
		val = make([]XInfoConsumer, len(cmd.val))
		copy(val, cmd.val)
	}
	return &XInfoConsumersCmd{
		baseCmd: cmd.cloneBaseCmd(),
		val:     val,
	}
}

//------------------------------------------------------------------------------

type XInfoGroupsCmd struct {
	baseCmd
	val []XInfoGroup
}

type XInfoGroup struct {
	Name            string
	Consumers       int64
	Pending         int64
	LastDeliveredID string
	EntriesRead     int64
	// Lag represents the number of pending messages in the stream not yet
	// delivered to this consumer group. Returns -1 when the lag cannot be determined.
	Lag int64
}

var _ Cmder = (*XInfoGroupsCmd)(nil)

func NewXInfoGroupsCmd(ctx context.Context, stream string) *XInfoGroupsCmd {
	return &XInfoGroupsCmd{
		baseCmd: baseCmd{
			ctx:     ctx,
			args:    []interface{}{"xinfo", "groups", stream},
			cmdType: CmdTypeXInfoGroups,
		},
	}
}

func (cmd *XInfoGroupsCmd) SetVal(val []XInfoGroup) {
	cmd.val = val
}

func (cmd *XInfoGroupsCmd) Val() []XInfoGroup {
	cmd.await()
	return cmd.val
}

func (cmd *XInfoGroupsCmd) Result() ([]XInfoGroup, error) {
	cmd.await()
	return cmd.val, cmd.err
}

func (cmd *XInfoGroupsCmd) String() string {
	cmd.await()
	return cmdString(cmd, cmd.val)
}

func (cmd *XInfoGroupsCmd) readReply(rd *proto.Reader) error {
	n, err := rd.ReadArrayLen()
	if err != nil {
		return err
	}
	cmd.val = make([]XInfoGroup, n)

	for i := 0; i < len(cmd.val); i++ {
		group := &cmd.val[i]

		nn, err := rd.ReadMapLen()
		if err != nil {
			return err
		}

		var key string
		for j := 0; j < nn; j++ {
			key, err = rd.ReadString()
			if err != nil {
				return err
			}

			switch key {
			case "name":
				group.Name, err = rd.ReadString()
				if err != nil {
					return err
				}
			case "consumers":
				group.Consumers, err = rd.ReadInt()
				if err != nil {
					return err
				}
			case "pending":
				group.Pending, err = rd.ReadInt()
				if err != nil {
					return err
				}
			case "last-delivered-id":
				group.LastDeliveredID, err = rd.ReadString()
				if err != nil {
					return err
				}
			case "entries-read":
				group.EntriesRead, err = rd.ReadInt()
				if err != nil && err != Nil {
					return err
				}
			case "lag":
				group.Lag, err = rd.ReadInt()

				// lag: the number of entries in the stream that are still waiting to be delivered
				// to the group's consumers, or a NULL(Nil) when that number can't be determined.
				// In that case, we return -1.
				if err != nil && err != Nil {
					return err
				} else if err == Nil {
					group.Lag = -1
				}
			default:
				// skip unknown fields
				if err = rd.DiscardNext(); err != nil {
					return err
				}
			}
		}
	}

	return nil
}

func (cmd *XInfoGroupsCmd) Clone() Cmder {
	var val []XInfoGroup
	if cmd.val != nil {
		val = make([]XInfoGroup, len(cmd.val))
		copy(val, cmd.val)
	}
	return &XInfoGroupsCmd{
		baseCmd: cmd.cloneBaseCmd(),
		val:     val,
	}
}

//------------------------------------------------------------------------------

type XInfoStreamCmd struct {
	baseCmd
	val *XInfoStream
}

type XInfoStream struct {
	Length               int64
	RadixTreeKeys        int64
	RadixTreeNodes       int64
	Groups               int64
	LastGeneratedID      string
	MaxDeletedEntryID    string
	EntriesAdded         int64
	FirstEntry           XMessage
	LastEntry            XMessage
	RecordedFirstEntryID string

	IDMPDuration   int64
	IDMPMaxSize    int64
	PIDsTracked    int64
	IIDsTracked    int64
	IIDsAdded      int64
	IIDsDuplicates int64
}

var _ Cmder = (*XInfoStreamCmd)(nil)

func NewXInfoStreamCmd(ctx context.Context, stream string) *XInfoStreamCmd {
	return &XInfoStreamCmd{
		baseCmd: baseCmd{
			ctx:     ctx,
			args:    []interface{}{"xinfo", "stream", stream},
			cmdType: CmdTypeXInfoStream,
		},
	}
}

func (cmd *XInfoStreamCmd) SetVal(val *XInfoStream) {
	cmd.val = val
}

func (cmd *XInfoStreamCmd) Val() *XInfoStream {
	cmd.await()
	return cmd.val
}

func (cmd *XInfoStreamCmd) Result() (*XInfoStream, error) {
	cmd.await()
	return cmd.val, cmd.err
}

func (cmd *XInfoStreamCmd) String() string {
	cmd.await()
	return cmdString(cmd, cmd.val)
}

func (cmd *XInfoStreamCmd) readReply(rd *proto.Reader) error {
	n, err := rd.ReadMapLen()
	if err != nil {
		return err
	}
	cmd.val = &XInfoStream{}

	for i := 0; i < n; i++ {
		key, err := rd.ReadString()
		if err != nil {
			return err
		}
		switch key {
		case "length":
			cmd.val.Length, err = rd.ReadInt()
			if err != nil {
				return err
			}
		case "radix-tree-keys":
			cmd.val.RadixTreeKeys, err = rd.ReadInt()
			if err != nil {
				return err
			}
		case "radix-tree-nodes":
			cmd.val.RadixTreeNodes, err = rd.ReadInt()
			if err != nil {
				return err
			}
		case "groups":
			cmd.val.Groups, err = rd.ReadInt()
			if err != nil {
				return err
			}
		case "last-generated-id":
			cmd.val.LastGeneratedID, err = rd.ReadString()
			if err != nil {
				return err
			}
		case "max-deleted-entry-id":
			cmd.val.MaxDeletedEntryID, err = rd.ReadString()
			if err != nil {
				return err
			}
		case "entries-added":
			cmd.val.EntriesAdded, err = rd.ReadInt()
			if err != nil {
				return err
			}
		case "first-entry":
			cmd.val.FirstEntry, err = readXMessage(rd)
			if err != nil && err != Nil {
				return err
			}
		case "last-entry":
			cmd.val.LastEntry, err = readXMessage(rd)
			if err != nil && err != Nil {
				return err
			}
		case "recorded-first-entry-id":
			cmd.val.RecordedFirstEntryID, err = rd.ReadString()
			if err != nil {
				return err
			}
		case "idmp-duration":
			cmd.val.IDMPDuration, err = rd.ReadInt()
			if err != nil {
				return err
			}
		case "idmp-maxsize":
			cmd.val.IDMPMaxSize, err = rd.ReadInt()
			if err != nil {
				return err
			}
		case "pids-tracked":
			cmd.val.PIDsTracked, err = rd.ReadInt()
			if err != nil {
				return err
			}
		case "iids-tracked":
			cmd.val.IIDsTracked, err = rd.ReadInt()
			if err != nil {
				return err
			}
		case "iids-added":
			cmd.val.IIDsAdded, err = rd.ReadInt()
			if err != nil {
				return err
			}
		case "iids-duplicates":
			cmd.val.IIDsDuplicates, err = rd.ReadInt()
			if err != nil {
				return err
			}
		default:
			// skip unknown fields
			if err = rd.DiscardNext(); err != nil {
				return err
			}
		}
	}
	return nil
}

func (cmd *XInfoStreamCmd) Clone() Cmder {
	var val *XInfoStream
	if cmd.val != nil {
		val = &XInfoStream{
			Length:               cmd.val.Length,
			RadixTreeKeys:        cmd.val.RadixTreeKeys,
			RadixTreeNodes:       cmd.val.RadixTreeNodes,
			Groups:               cmd.val.Groups,
			LastGeneratedID:      cmd.val.LastGeneratedID,
			MaxDeletedEntryID:    cmd.val.MaxDeletedEntryID,
			EntriesAdded:         cmd.val.EntriesAdded,
			RecordedFirstEntryID: cmd.val.RecordedFirstEntryID,
		}
		// Clone XMessage fields
		val.FirstEntry = XMessage{
			ID: cmd.val.FirstEntry.ID,
		}
		if cmd.val.FirstEntry.Values != nil {
			val.FirstEntry.Values = make(map[string]interface{}, len(cmd.val.FirstEntry.Values))
			for k, v := range cmd.val.FirstEntry.Values {
				val.FirstEntry.Values[k] = v
			}
		}
		val.LastEntry = XMessage{
			ID: cmd.val.LastEntry.ID,
		}
		if cmd.val.LastEntry.Values != nil {
			val.LastEntry.Values = make(map[string]interface{}, len(cmd.val.LastEntry.Values))
			for k, v := range cmd.val.LastEntry.Values {
				val.LastEntry.Values[k] = v
			}
		}
	}
	return &XInfoStreamCmd{
		baseCmd: cmd.cloneBaseCmd(),
		val:     val,
	}
}

//------------------------------------------------------------------------------

type XInfoStreamFullCmd struct {
	baseCmd
	val *XInfoStreamFull
}

type XInfoStreamFull struct {
	Length               int64
	RadixTreeKeys        int64
	RadixTreeNodes       int64
	LastGeneratedID      string
	MaxDeletedEntryID    string
	EntriesAdded         int64
	Entries              []XMessage
	Groups               []XInfoStreamGroup
	RecordedFirstEntryID string
	IDMPDuration         int64
	IDMPMaxSize          int64
	PIDsTracked          int64
	IIDsTracked          int64
	IIDsAdded            int64
	IIDsDuplicates       int64
}

type XInfoStreamGroup struct {
	Name            string
	LastDeliveredID string
	EntriesRead     int64
	Lag             int64
	PelCount        int64
	NackedCount     uint64 // redis version 8.8, number of NACK'd messages in the group
	Pending         []XInfoStreamGroupPending
	Consumers       []XInfoStreamConsumer
}

type XInfoStreamGroupPending struct {
	ID            string
	Consumer      string
	DeliveryTime  time.Time
	DeliveryCount int64
}

type XInfoStreamConsumer struct {
	Name       string
	SeenTime   time.Time
	ActiveTime time.Time
	PelCount   int64
	Pending    []XInfoStreamConsumerPending
}

type XInfoStreamConsumerPending struct {
	ID            string
	DeliveryTime  time.Time
	DeliveryCount int64
}

var _ Cmder = (*XInfoStreamFullCmd)(nil)

func NewXInfoStreamFullCmd(ctx context.Context, args ...interface{}) *XInfoStreamFullCmd {
	return &XInfoStreamFullCmd{
		baseCmd: baseCmd{
			ctx:     ctx,
			args:    args,
			cmdType: CmdTypeXInfoStreamFull,
		},
	}
}

func (cmd *XInfoStreamFullCmd) SetVal(val *XInfoStreamFull) {
	cmd.val = val
}

func (cmd *XInfoStreamFullCmd) Val() *XInfoStreamFull {
	cmd.await()
	return cmd.val
}

func (cmd *XInfoStreamFullCmd) Result() (*XInfoStreamFull, error) {
	cmd.await()
	return cmd.val, cmd.err
}

func (cmd *XInfoStreamFullCmd) String() string {
	cmd.await()
	return cmdString(cmd, cmd.val)
}

func (cmd *XInfoStreamFullCmd) readReply(rd *proto.Reader) error {
	n, err := rd.ReadMapLen()
	if err != nil {
		return err
	}

	cmd.val = &XInfoStreamFull{}

	for i := 0; i < n; i++ {
		key, err := rd.ReadString()
		if err != nil {
			return err
		}

		switch key {
		case "length":
			cmd.val.Length, err = rd.ReadInt()
			if err != nil {
				return err
			}
		case "radix-tree-keys":
			cmd.val.RadixTreeKeys, err = rd.ReadInt()
			if err != nil {
				return err
			}
		case "radix-tree-nodes":
			cmd.val.RadixTreeNodes, err = rd.ReadInt()
			if err != nil {
				return err
			}
		case "last-generated-id":
			cmd.val.LastGeneratedID, err = rd.ReadString()
			if err != nil {
				return err
			}
		case "entries-added":
			cmd.val.EntriesAdded, err = rd.ReadInt()
			if err != nil {
				return err
			}
		case "entries":
			cmd.val.Entries, err = readXMessageSlice(rd)
			if err != nil {
				return err
			}
		case "groups":
			cmd.val.Groups, err = readStreamGroups(rd)
			if err != nil {
				return err
			}
		case "max-deleted-entry-id":
			cmd.val.MaxDeletedEntryID, err = rd.ReadString()
			if err != nil {
				return err
			}
		case "recorded-first-entry-id":
			cmd.val.RecordedFirstEntryID, err = rd.ReadString()
			if err != nil {
				return err
			}
		case "idmp-duration":
			cmd.val.IDMPDuration, err = rd.ReadInt()
			if err != nil {
				return err
			}
		case "idmp-maxsize":
			cmd.val.IDMPMaxSize, err = rd.ReadInt()
			if err != nil {
				return err
			}
		case "pids-tracked":
			cmd.val.PIDsTracked, err = rd.ReadInt()
			if err != nil {
				return err
			}
		case "iids-tracked":
			cmd.val.IIDsTracked, err = rd.ReadInt()
			if err != nil {
				return err
			}
		case "iids-added":
			cmd.val.IIDsAdded, err = rd.ReadInt()
			if err != nil {
				return err
			}
		case "iids-duplicates":
			cmd.val.IIDsDuplicates, err = rd.ReadInt()
			if err != nil {
				return err
			}
		default:
			// skip unknown fields
			if err = rd.DiscardNext(); err != nil {
				return err
			}
		}
	}
	return nil
}

func readStreamGroups(rd *proto.Reader) ([]XInfoStreamGroup, error) {
	n, err := rd.ReadArrayLen()
	if err != nil {
		return nil, err
	}
	groups := make([]XInfoStreamGroup, 0, n)
	for i := 0; i < n; i++ {
		nn, err := rd.ReadMapLen()
		if err != nil {
			return nil, err
		}

		group := XInfoStreamGroup{}

		for j := 0; j < nn; j++ {
			key, err := rd.ReadString()
			if err != nil {
				return nil, err
			}

			switch key {
			case "name":
				group.Name, err = rd.ReadString()
				if err != nil {
					return nil, err
				}
			case "last-delivered-id":
				group.LastDeliveredID, err = rd.ReadString()
				if err != nil {
					return nil, err
				}
			case "entries-read":
				group.EntriesRead, err = rd.ReadInt()
				if err != nil && err != Nil {
					return nil, err
				}
			case "lag":
				// lag: the number of entries in the stream that are still waiting to be delivered
				// to the group's consumers, or a NULL(Nil) when that number can't be determined.
				group.Lag, err = rd.ReadInt()
				if err != nil && err != Nil {
					return nil, err
				}
			case "pel-count":
				group.PelCount, err = rd.ReadInt()
				if err != nil {
					return nil, err
				}
			case "nacked-count":
				group.NackedCount, err = rd.ReadUint()
				if err != nil {
					return nil, err
				}
			case "pending":
				group.Pending, err = readXInfoStreamGroupPending(rd)
				if err != nil {
					return nil, err
				}
			case "consumers":
				group.Consumers, err = readXInfoStreamConsumers(rd)
				if err != nil {
					return nil, err
				}
			default:
				// skip unknown fields
				if err = rd.DiscardNext(); err != nil {
					return nil, err
				}
			}
		}

		groups = append(groups, group)
	}

	return groups, nil
}

func readXInfoStreamGroupPending(rd *proto.Reader) ([]XInfoStreamGroupPending, error) {
	n, err := rd.ReadArrayLen()
	if err != nil {
		return nil, err
	}

	pending := make([]XInfoStreamGroupPending, 0, n)

	for i := 0; i < n; i++ {
		if err = rd.ReadFixedArrayLen(4); err != nil {
			return nil, err
		}

		p := XInfoStreamGroupPending{}

		p.ID, err = rd.ReadString()
		if err != nil {
			return nil, err
		}

		p.Consumer, err = rd.ReadString()
		if err != nil {
			return nil, err
		}

		delivery, err := rd.ReadInt()
		if err != nil {
			return nil, err
		}
		p.DeliveryTime = time.Unix(delivery/1000, delivery%1000*int64(time.Millisecond))

		p.DeliveryCount, err = rd.ReadInt()
		if err != nil {
			return nil, err
		}

		pending = append(pending, p)
	}

	return pending, nil
}

func readXInfoStreamConsumers(rd *proto.Reader) ([]XInfoStreamConsumer, error) {
	n, err := rd.ReadArrayLen()
	if err != nil {
		return nil, err
	}

	consumers := make([]XInfoStreamConsumer, 0, n)

	for i := 0; i < n; i++ {
		nn, err := rd.ReadMapLen()
		if err != nil {
			return nil, err
		}

		c := XInfoStreamConsumer{}

		for f := 0; f < nn; f++ {
			cKey, err := rd.ReadString()
			if err != nil {
				return nil, err
			}

			switch cKey {
			case "name":
				c.Name, err = rd.ReadString()
			case "seen-time":
				seen, err := rd.ReadInt()
				if err != nil {
					return nil, err
				}
				c.SeenTime = time.UnixMilli(seen)
			case "active-time":
				active, err := rd.ReadInt()
				if err != nil {
					return nil, err
				}
				c.ActiveTime = time.UnixMilli(active)
			case "pel-count":
				c.PelCount, err = rd.ReadInt()
			case "pending":
				pendingNumber, err := rd.ReadArrayLen()
				if err != nil {
					return nil, err
				}

				c.Pending = make([]XInfoStreamConsumerPending, 0, pendingNumber)

				for pn := 0; pn < pendingNumber; pn++ {
					if err = rd.ReadFixedArrayLen(3); err != nil {
						return nil, err
					}

					p := XInfoStreamConsumerPending{}

					p.ID, err = rd.ReadString()
					if err != nil {
						return nil, err
					}

					delivery, err := rd.ReadInt()
					if err != nil {
						return nil, err
					}
					p.DeliveryTime = time.Unix(delivery/1000, delivery%1000*int64(time.Millisecond))

					p.DeliveryCount, err = rd.ReadInt()
					if err != nil {
						return nil, err
					}

					c.Pending = append(c.Pending, p)
				}
			default:
				// skip unknown fields
				if err = rd.DiscardNext(); err != nil {
					return nil, err
				}
			}
			if err != nil {
				return nil, err
			}
		}
		consumers = append(consumers, c)
	}

	return consumers, nil
}

func (cmd *XInfoStreamFullCmd) Clone() Cmder {
	var val *XInfoStreamFull
	if cmd.val != nil {
		val = &XInfoStreamFull{
			Length:               cmd.val.Length,
			RadixTreeKeys:        cmd.val.RadixTreeKeys,
			RadixTreeNodes:       cmd.val.RadixTreeNodes,
			LastGeneratedID:      cmd.val.LastGeneratedID,
			MaxDeletedEntryID:    cmd.val.MaxDeletedEntryID,
			EntriesAdded:         cmd.val.EntriesAdded,
			RecordedFirstEntryID: cmd.val.RecordedFirstEntryID,
		}
		// Clone Entries
		if cmd.val.Entries != nil {
			val.Entries = make([]XMessage, len(cmd.val.Entries))
			for i, msg := range cmd.val.Entries {
				val.Entries[i] = XMessage{
					ID: msg.ID,
				}
				if msg.Values != nil {
					val.Entries[i].Values = make(map[string]interface{}, len(msg.Values))
					for k, v := range msg.Values {
						val.Entries[i].Values[k] = v
					}
				}
			}
		}
		// Clone Groups - simplified copy for now due to complexity
		if cmd.val.Groups != nil {
			val.Groups = make([]XInfoStreamGroup, len(cmd.val.Groups))
			copy(val.Groups, cmd.val.Groups)
		}
	}
	return &XInfoStreamFullCmd{
		baseCmd: cmd.cloneBaseCmd(),
		val:     val,
	}
}

//------------------------------------------------------------------------------

type ZSliceCmd struct {
	baseCmd

	val []Z
}

var _ Cmder = (*ZSliceCmd)(nil)

func NewZSliceCmd(ctx context.Context, args ...interface{}) *ZSliceCmd {
	return &ZSliceCmd{
		baseCmd: baseCmd{
			ctx:     ctx,
			args:    args,
			cmdType: CmdTypeZSlice,
		},
	}
}

func (cmd *ZSliceCmd) SetVal(val []Z) {
	cmd.val = val
}

func (cmd *ZSliceCmd) Val() []Z {
	cmd.await()
	return cmd.val
}

func (cmd *ZSliceCmd) Result() ([]Z, error) {
	cmd.await()
	return cmd.val, cmd.err
}

func (cmd *ZSliceCmd) String() string {
	cmd.await()
	return cmdString(cmd, cmd.val)
}

func (cmd *ZSliceCmd) readReply(rd *proto.Reader) error { // nolint:dupl
	n, err := rd.ReadArrayLen()
	if err != nil {
		return err
	}

	// If the n is 0, can't continue reading.
	if n == 0 {
		cmd.val = make([]Z, 0)
		return nil
	}

	typ, err := rd.PeekReplyType()
	if err != nil {
		return err
	}
	array := typ == proto.RespArray

	if array {
		cmd.val = make([]Z, n)
	} else {
		if n%2 != 0 {
			return fmt.Errorf("redis: got %d elements in the sorted set array, wanted a multiple of 2", n)
		}
		cmd.val = make([]Z, n/2)
	}

	for i := 0; i < len(cmd.val); i++ {
		if array {
			if err = rd.ReadFixedArrayLen(2); err != nil {
				return err
			}
		}

		if cmd.val[i].Member, err = rd.ReadString(); err != nil {
			return err
		}

		if cmd.val[i].Score, err = rd.ReadFloat(); err != nil {
			return err
		}
	}

	return nil
}

func (cmd *ZSliceCmd) Clone() Cmder {
	var val []Z
	if cmd.val != nil {
		val = make([]Z, len(cmd.val))
		copy(val, cmd.val)
	}
	return &ZSliceCmd{
		baseCmd: cmd.cloneBaseCmd(),
		val:     val,
	}
}

//------------------------------------------------------------------------------

type ZWithKeyCmd struct {
	baseCmd

	val *ZWithKey
}

var _ Cmder = (*ZWithKeyCmd)(nil)

func NewZWithKeyCmd(ctx context.Context, args ...interface{}) *ZWithKeyCmd {
	return &ZWithKeyCmd{
		baseCmd: baseCmd{
			ctx:     ctx,
			args:    args,
			cmdType: CmdTypeZWithKey,
		},
	}
}

func (cmd *ZWithKeyCmd) SetVal(val *ZWithKey) {
	cmd.val = val
}

func (cmd *ZWithKeyCmd) Val() *ZWithKey {
	cmd.await()
	return cmd.val
}

func (cmd *ZWithKeyCmd) Result() (*ZWithKey, error) {
	cmd.await()
	return cmd.val, cmd.err
}

func (cmd *ZWithKeyCmd) String() string {
	cmd.await()
	return cmdString(cmd, cmd.val)
}

func (cmd *ZWithKeyCmd) readReply(rd *proto.Reader) (err error) {
	if err = rd.ReadFixedArrayLen(3); err != nil {
		return err
	}
	cmd.val = &ZWithKey{}

	if cmd.val.Key, err = rd.ReadString(); err != nil {
		return err
	}
	if cmd.val.Member, err = rd.ReadString(); err != nil {
		return err
	}
	if cmd.val.Score, err = rd.ReadFloat(); err != nil {
		return err
	}

	return nil
}

func (cmd *ZWithKeyCmd) Clone() Cmder {
	var val *ZWithKey
	if cmd.val != nil {
		val = &ZWithKey{
			Z: Z{
				Score:  cmd.val.Score,
				Member: cmd.val.Member,
			},
			Key: cmd.val.Key,
		}
	}
	return &ZWithKeyCmd{
		baseCmd: cmd.cloneBaseCmd(),
		val:     val,
	}
}

//------------------------------------------------------------------------------

type ScanCmd struct {
	baseCmd

	page   []string
	cursor uint64

	process cmdable
}

var _ Cmder = (*ScanCmd)(nil)

func NewScanCmd(ctx context.Context, process cmdable, args ...interface{}) *ScanCmd {
	return &ScanCmd{
		baseCmd: baseCmd{
			ctx:     ctx,
			args:    args,
			cmdType: CmdTypeScan,
		},
		process: process,
	}
}

func (cmd *ScanCmd) SetVal(page []string, cursor uint64) {
	cmd.page = page
	cmd.cursor = cursor
}

func (cmd *ScanCmd) Val() (keys []string, cursor uint64) {
	cmd.await()
	return cmd.page, cmd.cursor
}

func (cmd *ScanCmd) Result() (keys []string, cursor uint64, err error) {
	cmd.await()
	return cmd.page, cmd.cursor, cmd.err
}

func (cmd *ScanCmd) String() string {
	cmd.await()
	return cmdString(cmd, cmd.page)
}

func (cmd *ScanCmd) readReply(rd *proto.Reader) error {
	if err := rd.ReadFixedArrayLen(2); err != nil {
		return err
	}

	cursor, err := rd.ReadUint()
	if err != nil {
		return err
	}
	cmd.cursor = cursor

	n, err := rd.ReadArrayLen()
	if err != nil {
		return err
	}
	cmd.page = make([]string, n)

	for i := 0; i < len(cmd.page); i++ {
		if cmd.page[i], err = rd.ReadString(); err != nil {
			return err
		}
	}
	return nil
}

func (cmd *ScanCmd) Clone() Cmder {
	var page []string
	if cmd.page != nil {
		page = make([]string, len(cmd.page))
		copy(page, cmd.page)
	}
	return &ScanCmd{
		baseCmd: cmd.cloneBaseCmd(),
		page:    page,
		cursor:  cmd.cursor,
		process: cmd.process,
	}
}

// Iterator creates a new ScanIterator.
func (cmd *ScanCmd) Iterator() *ScanIterator {
	return &ScanIterator{
		cmd: cmd,
	}
}

//------------------------------------------------------------------------------

type ClusterNode struct {
	ID                 string
	Addr               string
	NetworkingMetadata map[string]string
}

type ClusterSlot struct {
	Start int
	End   int
	Nodes []ClusterNode
}

type ClusterSlotsCmd struct {
	baseCmd

	val []ClusterSlot
}

var _ Cmder = (*ClusterSlotsCmd)(nil)

func NewClusterSlotsCmd(ctx context.Context, args ...interface{}) *ClusterSlotsCmd {
	return &ClusterSlotsCmd{
		baseCmd: baseCmd{
			ctx:     ctx,
			args:    args,
			cmdType: CmdTypeClusterSlots,
		},
	}
}

func (cmd *ClusterSlotsCmd) SetVal(val []ClusterSlot) {
	cmd.val = val
}

func (cmd *ClusterSlotsCmd) Val() []ClusterSlot {
	cmd.await()
	return cmd.val
}

func (cmd *ClusterSlotsCmd) Result() ([]ClusterSlot, error) {
	cmd.await()
	return cmd.val, cmd.err
}

func (cmd *ClusterSlotsCmd) String() string {
	cmd.await()
	return cmdString(cmd, cmd.val)
}

func (cmd *ClusterSlotsCmd) readReply(rd *proto.Reader) error {
	n, err := rd.ReadArrayLen()
	if err != nil {
		return err
	}
	cmd.val = make([]ClusterSlot, n)

	for i := 0; i < len(cmd.val); i++ {
		n, err = rd.ReadArrayLen()
		if err != nil {
			return err
		}
		if n < 2 {
			return fmt.Errorf("redis: got %d elements in cluster info, expected at least 2", n)
		}

		start, err := rd.ReadInt()
		if err != nil {
			return err
		}

		end, err := rd.ReadInt()
		if err != nil {
			return err
		}

		// subtract start and end.
		nodes := make([]ClusterNode, n-2)

		for j := 0; j < len(nodes); j++ {
			nn, err := rd.ReadArrayLen()
			if err != nil {
				return err
			}
			if nn < 2 || nn > 4 {
				return fmt.Errorf("got %d elements in cluster info address, expected 2, 3, or 4", n)
			}

			ip, err := rd.ReadString()
			if err != nil {
				return err
			}

			port, err := rd.ReadString()
			if err != nil {
				return err
			}

			nodes[j].Addr = net.JoinHostPort(ip, port)

			if nn >= 3 {
				id, err := rd.ReadString()
				if err != nil {
					return err
				}
				nodes[j].ID = id
			}

			if nn >= 4 {
				metadataLength, err := rd.ReadMapLen()
				if err != nil {
					return err
				}

				networkingMetadata := make(map[string]string, metadataLength)

				for i := 0; i < metadataLength; i++ {
					key, err := rd.ReadString()
					if err != nil {
						return err
					}
					value, err := rd.ReadString()
					if err != nil {
						return err
					}
					networkingMetadata[key] = value
				}

				nodes[j].NetworkingMetadata = networkingMetadata
			}
		}

		cmd.val[i] = ClusterSlot{
			Start: int(start),
			End:   int(end),
			Nodes: nodes,
		}
	}

	return nil
}

func (cmd *ClusterSlotsCmd) Clone() Cmder {
	var val []ClusterSlot
	if cmd.val != nil {
		val = make([]ClusterSlot, len(cmd.val))
		for i, slot := range cmd.val {
			val[i] = ClusterSlot{
				Start: slot.Start,
				End:   slot.End,
			}
			if slot.Nodes != nil {
				val[i].Nodes = make([]ClusterNode, len(slot.Nodes))
				for j, node := range slot.Nodes {
					val[i].Nodes[j] = ClusterNode{
						ID:   node.ID,
						Addr: node.Addr,
					}
					if node.NetworkingMetadata != nil {
						val[i].Nodes[j].NetworkingMetadata = make(map[string]string, len(node.NetworkingMetadata))
						for k, v := range node.NetworkingMetadata {
							val[i].Nodes[j].NetworkingMetadata[k] = v
						}
					}
				}
			}
		}
	}
	return &ClusterSlotsCmd{
		baseCmd: cmd.cloneBaseCmd(),
		val:     val,
	}
}

//------------------------------------------------------------------------------

// GeoLocation is used with GeoAdd to add geospatial location.
type GeoLocation struct {
	Name                      string
	Longitude, Latitude, Dist float64
	GeoHash                   int64
}

// GeoRadiusQuery is used with GeoRadius to query geospatial index.
type GeoRadiusQuery struct {
	Radius float64
	// Can be m, km, ft, or mi. Default is km.
	Unit        string
	WithCoord   bool
	WithDist    bool
	WithGeoHash bool
	Count       int
	// Can be ASC or DESC. Default is no sort order.
	Sort      string
	Store     string
	StoreDist string

	// WithCoord+WithDist+WithGeoHash
	withLen int
}

type GeoLocationCmd struct {
	baseCmd

	q         *GeoRadiusQuery
	locations []GeoLocation
}

var _ Cmder = (*GeoLocationCmd)(nil)

func NewGeoLocationCmd(ctx context.Context, q *GeoRadiusQuery, args ...interface{}) *GeoLocationCmd {
	return &GeoLocationCmd{
		baseCmd: baseCmd{
			ctx:     ctx,
			args:    geoLocationArgs(q, args...),
			cmdType: CmdTypeGeoLocation,
		},
		q: q,
	}
}

func geoLocationArgs(q *GeoRadiusQuery, args ...interface{}) []interface{} {
	args = append(args, q.Radius)
	if q.Unit != "" {
		args = append(args, q.Unit)
	} else {
		args = append(args, "km")
	}
	if q.WithCoord {
		args = append(args, "withcoord")
		q.withLen++
	}
	if q.WithDist {
		args = append(args, "withdist")
		q.withLen++
	}
	if q.WithGeoHash {
		args = append(args, "withhash")
		q.withLen++
	}
	if q.Count > 0 {
		args = append(args, "count", q.Count)
	}
	if q.Sort != "" {
		args = append(args, q.Sort)
	}
	if q.Store != "" {
		args = append(args, "store")
		args = append(args, q.Store)
	}
	if q.StoreDist != "" {
		args = append(args, "storedist")
		args = append(args, q.StoreDist)
	}
	return args
}

func (cmd *GeoLocationCmd) SetVal(locations []GeoLocation) {
	cmd.locations = locations
}

func (cmd *GeoLocationCmd) Val() []GeoLocation {
	cmd.await()
	return cmd.locations
}

func (cmd *GeoLocationCmd) Result() ([]GeoLocation, error) {
	cmd.await()
	return cmd.locations, cmd.err
}

func (cmd *GeoLocationCmd) String() string {
	cmd.await()
	return cmdString(cmd, cmd.locations)
}

func (cmd *GeoLocationCmd) readReply(rd *proto.Reader) error {
	n, err := rd.ReadArrayLen()
	if err != nil {
		return err
	}
	cmd.locations = make([]GeoLocation, n)

	for i := 0; i < len(cmd.locations); i++ {
		// only name
		if cmd.q.withLen == 0 {
			if cmd.locations[i].Name, err = rd.ReadString(); err != nil {
				return err
			}
			continue
		}

		// +name
		if err = rd.ReadFixedArrayLen(cmd.q.withLen + 1); err != nil {
			return err
		}

		if cmd.locations[i].Name, err = rd.ReadString(); err != nil {
			return err
		}
		if cmd.q.WithDist {
			if cmd.locations[i].Dist, err = rd.ReadFloat(); err != nil {
				return err
			}
		}
		if cmd.q.WithGeoHash {
			if cmd.locations[i].GeoHash, err = rd.ReadInt(); err != nil {
				return err
			}
		}
		if cmd.q.WithCoord {
			if err = rd.ReadFixedArrayLen(2); err != nil {
				return err
			}
			if cmd.locations[i].Longitude, err = rd.ReadFloat(); err != nil {
				return err
			}
			if cmd.locations[i].Latitude, err = rd.ReadFloat(); err != nil {
				return err
			}
		}
	}

	return nil
}

func (cmd *GeoLocationCmd) Clone() Cmder {
	var q *GeoRadiusQuery
	if cmd.q != nil {
		q = &GeoRadiusQuery{
			Radius:      cmd.q.Radius,
			Unit:        cmd.q.Unit,
			WithCoord:   cmd.q.WithCoord,
			WithDist:    cmd.q.WithDist,
			WithGeoHash: cmd.q.WithGeoHash,
			Count:       cmd.q.Count,
			Sort:        cmd.q.Sort,
			Store:       cmd.q.Store,
			StoreDist:   cmd.q.StoreDist,
			withLen:     cmd.q.withLen,
		}
	}
	var locations []GeoLocation
	if cmd.locations != nil {
		locations = make([]GeoLocation, len(cmd.locations))
		copy(locations, cmd.locations)
	}
	return &GeoLocationCmd{
		baseCmd:   cmd.cloneBaseCmd(),
		q:         q,
		locations: locations,
	}
}

//------------------------------------------------------------------------------

// GeoSearchQuery is used for GEOSearch/GEOSearchStore command query.
type GeoSearchQuery struct {
	Member string

	// Latitude and Longitude when using FromLonLat option.
	Longitude float64
	Latitude  float64

	// Distance and unit when using ByRadius option.
	// Can use m, km, ft, or mi. Default is km.
	Radius     float64
	RadiusUnit string

	// Height, width and unit when using ByBox option.
	// Can be m, km, ft, or mi. Default is km.
	BoxWidth  float64
	BoxHeight float64
	BoxUnit   string

	// Can be ASC or DESC. Default is no sort order.
	Sort     string
	Count    int
	CountAny bool
}

type GeoSearchLocationQuery struct {
	GeoSearchQuery

	WithCoord bool
	WithDist  bool
	WithHash  bool
}

type GeoSearchStoreQuery struct {
	GeoSearchQuery

	// When using the StoreDist option, the command stores the items in a
	// sorted set populated with their distance from the center of the circle or box,
	// as a floating-point number, in the same unit specified for that shape.
	StoreDist bool
}

func geoSearchLocationArgs(q *GeoSearchLocationQuery, args []interface{}) []interface{} {
	args = geoSearchArgs(&q.GeoSearchQuery, args)

	if q.WithCoord {
		args = append(args, "withcoord")
	}
	if q.WithDist {
		args = append(args, "withdist")
	}
	if q.WithHash {
		args = append(args, "withhash")
	}

	return args
}

func geoSearchArgs(q *GeoSearchQuery, args []interface{}) []interface{} {
	if q.Member != "" {
		args = append(args, "frommember", q.Member)
	} else {
		args = append(args, "fromlonlat", q.Longitude, q.Latitude)
	}

	if q.Radius > 0 {
		if q.RadiusUnit == "" {
			q.RadiusUnit = "km"
		}
		args = append(args, "byradius", q.Radius, q.RadiusUnit)
	} else {
		if q.BoxUnit == "" {
			q.BoxUnit = "km"
		}
		args = append(args, "bybox", q.BoxWidth, q.BoxHeight, q.BoxUnit)
	}

	if q.Sort != "" {
		args = append(args, q.Sort)
	}

	if q.Count > 0 {
		args = append(args, "count", q.Count)
		if q.CountAny {
			args = append(args, "any")
		}
	}

	return args
}

type GeoSearchLocationCmd struct {
	baseCmd

	opt *GeoSearchLocationQuery
	val []GeoLocation
}

var _ Cmder = (*GeoSearchLocationCmd)(nil)

func NewGeoSearchLocationCmd(
	ctx context.Context, opt *GeoSearchLocationQuery, args ...interface{},
) *GeoSearchLocationCmd {
	return &GeoSearchLocationCmd{
		baseCmd: baseCmd{
			ctx:     ctx,
			args:    geoSearchLocationArgs(opt, args),
			cmdType: CmdTypeGeoSearchLocation,
		},
		opt: opt,
	}
}

func (cmd *GeoSearchLocationCmd) SetVal(val []GeoLocation) {
	cmd.val = val
}

func (cmd *GeoSearchLocationCmd) Val() []GeoLocation {
	cmd.await()
	return cmd.val
}

func (cmd *GeoSearchLocationCmd) Result() ([]GeoLocation, error) {
	cmd.await()
	return cmd.val, cmd.err
}

func (cmd *GeoSearchLocationCmd) String() string {
	cmd.await()
	return cmdString(cmd, cmd.val)
}

func (cmd *GeoSearchLocationCmd) readReply(rd *proto.Reader) error {
	n, err := rd.ReadArrayLen()
	if err != nil {
		return err
	}

	cmd.val = make([]GeoLocation, n)
	// Each element is an array of [name, ...] whose minimum length is set by
	// the requested WITH flags. Entries shorter than that would make the
	// parser read into the next reply; extra elements are drained below so a
	// longer entry (e.g. from a newer server) can't leave frames on the wire.
	withLen := 1
	if cmd.opt.WithDist {
		withLen++
	}
	if cmd.opt.WithHash {
		withLen++
	}
	if cmd.opt.WithCoord {
		withLen++
	}
	for i := 0; i < n; i++ {
		nn, err := rd.ReadArrayLen()
		if err != nil {
			return err
		}
		if nn < withLen {
			return fmt.Errorf("redis: got %d elements in GEOSEARCH reply, expected at least %d", nn, withLen)
		}

		var loc GeoLocation

		loc.Name, err = rd.ReadString()
		if err != nil {
			return err
		}
		if cmd.opt.WithDist {
			loc.Dist, err = rd.ReadFloat()
			if err != nil {
				return err
			}
		}
		if cmd.opt.WithHash {
			loc.GeoHash, err = rd.ReadInt()
			if err != nil {
				return err
			}
		}
		if cmd.opt.WithCoord {
			if err = rd.ReadFixedArrayLen(2); err != nil {
				return err
			}
			loc.Longitude, err = rd.ReadFloat()
			if err != nil {
				return err
			}
			loc.Latitude, err = rd.ReadFloat()
			if err != nil {
				return err
			}
		}
		for j := withLen; j < nn; j++ {
			if err := rd.DiscardNext(); err != nil {
				return err
			}
		}

		cmd.val[i] = loc
	}

	return nil
}

func (cmd *GeoSearchLocationCmd) Clone() Cmder {
	var opt *GeoSearchLocationQuery
	if cmd.opt != nil {
		opt = &GeoSearchLocationQuery{
			GeoSearchQuery: GeoSearchQuery{
				Member:     cmd.opt.Member,
				Longitude:  cmd.opt.Longitude,
				Latitude:   cmd.opt.Latitude,
				Radius:     cmd.opt.Radius,
				RadiusUnit: cmd.opt.RadiusUnit,
				BoxWidth:   cmd.opt.BoxWidth,
				BoxHeight:  cmd.opt.BoxHeight,
				BoxUnit:    cmd.opt.BoxUnit,
				Sort:       cmd.opt.Sort,
				Count:      cmd.opt.Count,
				CountAny:   cmd.opt.CountAny,
			},
			WithCoord: cmd.opt.WithCoord,
			WithDist:  cmd.opt.WithDist,
			WithHash:  cmd.opt.WithHash,
		}
	}
	var val []GeoLocation
	if cmd.val != nil {
		val = make([]GeoLocation, len(cmd.val))
		copy(val, cmd.val)
	}
	return &GeoSearchLocationCmd{
		baseCmd: cmd.cloneBaseCmd(),
		opt:     opt,
		val:     val,
	}
}

//------------------------------------------------------------------------------

type GeoPos struct {
	Longitude, Latitude float64
}

type GeoPosCmd struct {
	baseCmd

	val []*GeoPos
}

var _ Cmder = (*GeoPosCmd)(nil)

func NewGeoPosCmd(ctx context.Context, args ...interface{}) *GeoPosCmd {
	return &GeoPosCmd{
		baseCmd: baseCmd{
			ctx:     ctx,
			args:    args,
			cmdType: CmdTypeGeoPos,
		},
	}
}

func (cmd *GeoPosCmd) SetVal(val []*GeoPos) {
	cmd.val = val
}

func (cmd *GeoPosCmd) Val() []*GeoPos {
	cmd.await()
	return cmd.val
}

func (cmd *GeoPosCmd) Result() ([]*GeoPos, error) {
	cmd.await()
	return cmd.val, cmd.err
}

func (cmd *GeoPosCmd) String() string {
	cmd.await()
	return cmdString(cmd, cmd.val)
}

func (cmd *GeoPosCmd) readReply(rd *proto.Reader) error {
	n, err := rd.ReadArrayLen()
	if err != nil {
		return err
	}
	cmd.val = make([]*GeoPos, n)

	for i := 0; i < len(cmd.val); i++ {
		err = rd.ReadFixedArrayLen(2)
		if err != nil {
			if err == Nil {
				cmd.val[i] = nil
				continue
			}
			return err
		}

		longitude, err := rd.ReadFloat()
		if err != nil {
			return err
		}
		latitude, err := rd.ReadFloat()
		if err != nil {
			return err
		}

		cmd.val[i] = &GeoPos{
			Longitude: longitude,
			Latitude:  latitude,
		}
	}

	return nil
}

func (cmd *GeoPosCmd) Clone() Cmder {
	var val []*GeoPos
	if cmd.val != nil {
		val = make([]*GeoPos, len(cmd.val))
		for i, pos := range cmd.val {
			if pos != nil {
				val[i] = &GeoPos{
					Longitude: pos.Longitude,
					Latitude:  pos.Latitude,
				}
			}
		}
	}
	return &GeoPosCmd{
		baseCmd: cmd.cloneBaseCmd(),
		val:     val,
	}
}

//------------------------------------------------------------------------------

type CommandInfo struct {
	Name          string
	Arity         int8
	Flags         []string
	ACLFlags      []string
	FirstKeyPos   int8
	LastKeyPos    int8
	StepCount     int8
	ReadOnly      bool
	CommandPolicy *routing.CommandPolicy
}

type CommandsInfoCmd struct {
	baseCmd

	val map[string]*CommandInfo
}

var _ Cmder = (*CommandsInfoCmd)(nil)

func NewCommandsInfoCmd(ctx context.Context, args ...interface{}) *CommandsInfoCmd {
	return &CommandsInfoCmd{
		baseCmd: baseCmd{
			ctx:     ctx,
			args:    args,
			cmdType: CmdTypeCommandsInfo,
		},
	}
}

func (cmd *CommandsInfoCmd) SetVal(val map[string]*CommandInfo) {
	cmd.val = val
}

func (cmd *CommandsInfoCmd) Val() map[string]*CommandInfo {
	cmd.await()
	return cmd.val
}

func (cmd *CommandsInfoCmd) Result() (map[string]*CommandInfo, error) {
	cmd.await()
	return cmd.val, cmd.err
}

func (cmd *CommandsInfoCmd) String() string {
	cmd.await()
	return cmdString(cmd, cmd.val)
}

func (cmd *CommandsInfoCmd) readReply(rd *proto.Reader) error {
	const numArgRedis5 = 6
	const numArgRedis6 = 7
	const numArgRedis7 = 10 // Also matches redis 8

	n, err := rd.ReadArrayLen()
	if err != nil {
		return err
	}
	cmd.val = make(map[string]*CommandInfo, n)

	for i := 0; i < n; i++ {
		nn, err := rd.ReadArrayLen()
		if err != nil {
			return err
		}

		switch nn {
		case numArgRedis5, numArgRedis6, numArgRedis7:
			// ok
		default:
			return fmt.Errorf("redis: got %d elements in COMMAND reply, wanted 6/7/10", nn)
		}

		cmdInfo := &CommandInfo{}
		if cmdInfo.Name, err = rd.ReadString(); err != nil {
			return err
		}

		arity, err := rd.ReadInt()
		if err != nil {
			return err
		}
		cmdInfo.Arity = int8(arity)

		flagLen, err := rd.ReadArrayLen()
		if err != nil {
			return err
		}
		cmdInfo.Flags = make([]string, flagLen)
		for f := 0; f < len(cmdInfo.Flags); f++ {
			switch s, err := rd.ReadString(); {
			case err == Nil:
				cmdInfo.Flags[f] = ""
			case err != nil:
				return err
			default:
				if !cmdInfo.ReadOnly && s == "readonly" {
					cmdInfo.ReadOnly = true
				}
				cmdInfo.Flags[f] = s
			}
		}

		firstKeyPos, err := rd.ReadInt()
		if err != nil {
			return err
		}
		cmdInfo.FirstKeyPos = int8(firstKeyPos)

		lastKeyPos, err := rd.ReadInt()
		if err != nil {
			return err
		}
		cmdInfo.LastKeyPos = int8(lastKeyPos)

		stepCount, err := rd.ReadInt()
		if err != nil {
			return err
		}
		cmdInfo.StepCount = int8(stepCount)

		if nn >= numArgRedis6 {
			aclFlagLen, err := rd.ReadArrayLen()
			if err != nil {
				return err
			}
			cmdInfo.ACLFlags = make([]string, aclFlagLen)
			for f := 0; f < len(cmdInfo.ACLFlags); f++ {
				switch s, err := rd.ReadString(); {
				case err == Nil:
					cmdInfo.ACLFlags[f] = ""
				case err != nil:
					return err
				default:
					cmdInfo.ACLFlags[f] = s
				}
			}
		}

		if nn >= numArgRedis7 {
			// The 8th argument is an array of tips.
			tipsLen, err := rd.ReadArrayLen()
			if err != nil {
				return err
			}

			rawTips := make(map[string]string, tipsLen)
			if cmdInfo.ReadOnly {
				rawTips[routing.ReadOnlyCMD] = ""
			}
			for f := 0; f < tipsLen; f++ {
				tip, err := rd.ReadString()
				if err != nil {
					return err
				}

				k, v, ok := strings.Cut(tip, ":")
				if !ok {
					// Handle tips that don't have a colon (like "nondeterministic_output")
					rawTips[tip] = ""
				} else {
					// Handle normal key:value tips
					rawTips[k] = v
				}
			}
			cmdInfo.CommandPolicy = parseCommandPolicies(rawTips, cmdInfo.FirstKeyPos)

			if err := rd.DiscardNext(); err != nil {
				return err
			}
			if err := rd.DiscardNext(); err != nil {
				return err
			}
		}

		cmd.val[cmdInfo.Name] = cmdInfo
	}

	return nil
}

func (cmd *CommandsInfoCmd) Clone() Cmder {
	var val map[string]*CommandInfo
	if cmd.val != nil {
		val = make(map[string]*CommandInfo, len(cmd.val))
		for k, v := range cmd.val {
			if v != nil {
				newInfo := &CommandInfo{
					Name:          v.Name,
					Arity:         v.Arity,
					FirstKeyPos:   v.FirstKeyPos,
					LastKeyPos:    v.LastKeyPos,
					StepCount:     v.StepCount,
					ReadOnly:      v.ReadOnly,
					CommandPolicy: v.CommandPolicy, // CommandPolicy can be shared as it's immutable
				}
				if v.Flags != nil {
					newInfo.Flags = make([]string, len(v.Flags))
					copy(newInfo.Flags, v.Flags)
				}
				if v.ACLFlags != nil {
					newInfo.ACLFlags = make([]string, len(v.ACLFlags))
					copy(newInfo.ACLFlags, v.ACLFlags)
				}
				val[k] = newInfo
			}
		}
	}
	return &CommandsInfoCmd{
		baseCmd: cmd.cloneBaseCmd(),
		val:     val,
	}
}

//------------------------------------------------------------------------------

type cmdsInfoCache struct {
	fn func(ctx context.Context) (map[string]*CommandInfo, error)

	once        internal.Once
	refreshLock sync.RWMutex
	cmds        map[string]*CommandInfo
	// cmdsAtomic mirrors cmds for lock-free reads via Peek. cmds is only ever
	// replaced wholesale (never mutated in place), so an atomic pointer load is a
	// safe, contention-free read — Peek is on the hot per-command cluster routing
	// path where the RWMutex.RLock showed up as a bottleneck under heavy load.
	cmdsAtomic atomic.Pointer[map[string]*CommandInfo]
}

func newCmdsInfoCache(fn func(ctx context.Context) (map[string]*CommandInfo, error)) *cmdsInfoCache {
	return &cmdsInfoCache{
		fn: fn,
	}
}

func (c *cmdsInfoCache) Get(ctx context.Context) (map[string]*CommandInfo, error) {
	c.refreshLock.Lock()
	defer c.refreshLock.Unlock()

	err := c.once.Do(func() error {
		cmds, err := c.fn(ctx)
		if err != nil {
			return err
		}

		lowerCmds := make(map[string]*CommandInfo, len(cmds))

		// Extensions have cmd names in upper case. Convert them to lower case.
		for k, v := range cmds {
			lowerCmds[internal.ToLower(k)] = v
		}

		c.cmds = lowerCmds
		c.cmdsAtomic.Store(&lowerCmds)
		return nil
	})
	return c.cmds, err
}

func (c *cmdsInfoCache) Refresh() {
	c.refreshLock.Lock()
	defer c.refreshLock.Unlock()

	c.once = internal.Once{}
}

// Peek returns the cached CommandInfo map without triggering a Redis round-trip.
// Returns nil when the cache is cold; callers should fall back to other heuristics.
// The read is lock-free (a single atomic load) and never blocks, even while a
// concurrent Get() is populating the cache — it simply returns nil until the
// first population publishes the map.
// The returned map and its entries MUST NOT be mutated by the caller.
func (c *cmdsInfoCache) Peek() map[string]*CommandInfo {
	if c == nil {
		return nil
	}
	// Lock-free read: cmds is replaced wholesale, never mutated in place.
	if p := c.cmdsAtomic.Load(); p != nil {
		return *p
	}
	return nil
}

// ------------------------------------------------------------------------------
const (
	requestPolicy  = "request_policy"
	responsePolicy = "response_policy"
)

func parseCommandPolicies(commandInfoTips map[string]string, firstKeyPos int8) *routing.CommandPolicy {
	req := routing.ReqDefault
	resp := routing.RespDefaultKeyless
	if firstKeyPos > 0 {
		resp = routing.RespDefaultHashSlot
	}

	tips := make(map[string]string, len(commandInfoTips))
	for k, v := range commandInfoTips {
		if k == requestPolicy {
			if p, err := routing.ParseRequestPolicy(v); err == nil {
				req = p
			}
			continue
		}
		if k == responsePolicy {
			if p, err := routing.ParseResponsePolicy(v); err == nil {
				resp = p
			}
			continue
		}
		tips[k] = v
	}

	return &routing.CommandPolicy{Request: req, Response: resp, Tips: tips}
}

//------------------------------------------------------------------------------

type SlowLog struct {
	ID       int64
	Time     time.Time
	Duration time.Duration
	Args     []string
	// These are also optional fields emitted only by Redis 4.0 or greater:
	// https://redis.io/commands/slowlog#output-format
	ClientAddr string
	ClientName string
	// CommandArgc is the command's total argument count (including the command
	// name), emitted only by Redis 8.10 or greater. It may exceed len(Args) when
	// the slow log truncates the stored arguments (slowlog-max-argc, default 32).
	CommandArgc int64
}

type SlowLogCmd struct {
	baseCmd

	val []SlowLog
}

var _ Cmder = (*SlowLogCmd)(nil)

func NewSlowLogCmd(ctx context.Context, args ...interface{}) *SlowLogCmd {
	return &SlowLogCmd{
		baseCmd: baseCmd{
			ctx:     ctx,
			args:    args,
			cmdType: CmdTypeSlowLog,
		},
	}
}

func (cmd *SlowLogCmd) SetVal(val []SlowLog) {
	cmd.val = val
}

func (cmd *SlowLogCmd) Val() []SlowLog {
	cmd.await()
	return cmd.val
}

func (cmd *SlowLogCmd) Result() ([]SlowLog, error) {
	cmd.await()
	return cmd.val, cmd.err
}

func (cmd *SlowLogCmd) String() string {
	cmd.await()
	return cmdString(cmd, cmd.val)
}

func (cmd *SlowLogCmd) readReply(rd *proto.Reader) error {
	n, err := rd.ReadArrayLen()
	if err != nil {
		return err
	}
	cmd.val = make([]SlowLog, n)

	for i := 0; i < len(cmd.val); i++ {
		nn, err := rd.ReadArrayLen()
		if err != nil {
			return err
		}
		if nn < 4 {
			return fmt.Errorf("redis: got %d elements in slowlog get, expected at least 4", nn)
		}

		if cmd.val[i].ID, err = rd.ReadInt(); err != nil {
			return err
		}

		createdAt, err := rd.ReadInt()
		if err != nil {
			return err
		}
		cmd.val[i].Time = time.Unix(createdAt, 0)

		costs, err := rd.ReadInt()
		if err != nil {
			return err
		}
		cmd.val[i].Duration = time.Duration(costs) * time.Microsecond

		cmdLen, err := rd.ReadArrayLen()
		if err != nil {
			return err
		}
		if cmdLen < 1 {
			return fmt.Errorf("redis: got %d elements commands reply in slowlog get, expected at least 1", cmdLen)
		}

		cmd.val[i].Args = make([]string, cmdLen)
		for f := 0; f < len(cmd.val[i].Args); f++ {
			cmd.val[i].Args[f], err = rd.ReadString()
			if err != nil {
				return err
			}
		}

		if nn >= 5 {
			if cmd.val[i].ClientAddr, err = rd.ReadString(); err != nil {
				return err
			}
		}

		if nn >= 6 {
			if cmd.val[i].ClientName, err = rd.ReadString(); err != nil {
				return err
			}
		}

		// Redis 8.10+ appends a 7th field: the command's total argument count.
		if nn >= 7 {
			if cmd.val[i].CommandArgc, err = rd.ReadInt(); err != nil {
				return err
			}
		}

		// Drain any elements past the 7 this parser knows about so a server
		// that declares a longer entry array doesn't leave frames on the wire.
		for j := 7; j < nn; j++ {
			if err = rd.DiscardNext(); err != nil {
				return err
			}
		}
	}

	return nil
}

func (cmd *SlowLogCmd) Clone() Cmder {
	var val []SlowLog
	if cmd.val != nil {
		val = make([]SlowLog, len(cmd.val))
		for i, log := range cmd.val {
			val[i] = SlowLog{
				ID:          log.ID,
				Time:        log.Time,
				Duration:    log.Duration,
				ClientAddr:  log.ClientAddr,
				ClientName:  log.ClientName,
				CommandArgc: log.CommandArgc,
			}
			if log.Args != nil {
				val[i].Args = make([]string, len(log.Args))
				copy(val[i].Args, log.Args)
			}
		}
	}
	return &SlowLogCmd{
		baseCmd: cmd.cloneBaseCmd(),
		val:     val,
	}
}

//-----------------------------------------------------------------------

type Latency struct {
	Name   string
	Time   time.Time
	Latest time.Duration
	Max    time.Duration
}

type LatencyCmd struct {
	baseCmd
	val []Latency
}

var _ Cmder = (*LatencyCmd)(nil)

func NewLatencyCmd(ctx context.Context, args ...interface{}) *LatencyCmd {
	return &LatencyCmd{
		baseCmd: baseCmd{
			ctx:  ctx,
			args: args,
		},
	}
}

func (cmd *LatencyCmd) SetVal(val []Latency) {
	cmd.val = val
}

func (cmd *LatencyCmd) Val() []Latency {
	cmd.await()
	return cmd.val
}

func (cmd *LatencyCmd) Result() ([]Latency, error) {
	cmd.await()
	return cmd.val, cmd.err
}

func (cmd *LatencyCmd) String() string {
	cmd.await()
	return cmdString(cmd, cmd.val)
}

func (cmd *LatencyCmd) readReply(rd *proto.Reader) error {
	n, err := rd.ReadArrayLen()
	if err != nil {
		return err
	}
	cmd.val = make([]Latency, n)
	for i := 0; i < len(cmd.val); i++ {
		nn, err := rd.ReadArrayLen()
		if err != nil {
			return err
		}
		if nn < 4 {
			return fmt.Errorf("redis: got %d elements in latency get, expected at least 4", nn)
		}
		if cmd.val[i].Name, err = rd.ReadString(); err != nil {
			return err
		}
		createdAt, err := rd.ReadInt()
		if err != nil {
			return err
		}
		cmd.val[i].Time = time.Unix(createdAt, 0)
		latest, err := rd.ReadInt()
		if err != nil {
			return err
		}
		cmd.val[i].Latest = time.Duration(latest) * time.Millisecond
		maximum, err := rd.ReadInt()
		if err != nil {
			return err
		}
		cmd.val[i].Max = time.Duration(maximum) * time.Millisecond
		// Drain any elements beyond the 4 this parser reads so a server that
		// declares a longer entry array can't leave frames on the wire.
		for j := 4; j < nn; j++ {
			if err = rd.DiscardNext(); err != nil {
				return err
			}
		}
	}
	return nil
}

func (cmd *LatencyCmd) Clone() Cmder {
	var val []Latency
	if cmd.val != nil {
		val = make([]Latency, len(cmd.val))
		copy(val, cmd.val)
	}
	return &LatencyCmd{
		baseCmd: cmd.cloneBaseCmd(),
		val:     val,
	}
}

//-----------------------------------------------------------------------

// HotKeysSlotRange represents a slot or slot range in the response.
// Single element slice = individual slot, two element slice = slot range [start, end].
type HotKeysSlotRange []int64

// HotKeysKeyEntry represents a hot key entry with its metric value.
type HotKeysKeyEntry struct {
	Key   string
	Value interface{} // Can be int64 or string
}

// HotKeysResult represents the response data from HOTKEYS GET command.
// Field names match the Redis response format.
type HotKeysResult struct {
	TrackingActive                       bool
	SampleRatio                          uint8
	SelectedSlots                        []HotKeysSlotRange
	SampledCommandsSelectedSlots         time.Duration // Present when sample-ratio > 1 and selected-slots is not empty
	AllCommandsSelectedSlots             time.Duration // Present when selected-slots is not empty
	AllCommandsAllSlots                  time.Duration
	NetBytesSampledCommandsSelectedSlots int64 // Present when sample-ratio > 1 and selected-slots is not empty
	NetBytesAllCommandsSelectedSlots     int64 // Present when selected-slots is not empty
	NetBytesAllCommandsAllSlots          int64
	CollectionStartTime                  time.Time
	CollectionDuration                   time.Duration
	UsedCPUSys                           time.Duration
	UsedCPUUser                          time.Duration
	TotalNetBytes                        int64
	ByCPUTime                            []HotKeysKeyEntry
	ByNetBytes                           []HotKeysKeyEntry
}

type HotKeysCmd struct {
	baseCmd

	val *HotKeysResult
}

var _ Cmder = (*HotKeysCmd)(nil)

func NewHotKeysCmd(ctx context.Context, args ...interface{}) *HotKeysCmd {
	return &HotKeysCmd{
		baseCmd: baseCmd{
			ctx:     ctx,
			args:    args,
			cmdType: CmdTypeHotKeys,
		},
	}
}

func (cmd *HotKeysCmd) SetVal(val *HotKeysResult) {
	cmd.val = val
}

func (cmd *HotKeysCmd) Val() *HotKeysResult {
	cmd.await()
	return cmd.val
}

func (cmd *HotKeysCmd) Result() (*HotKeysResult, error) {
	cmd.await()
	return cmd.val, cmd.err
}

func (cmd *HotKeysCmd) String() string {
	cmd.await()
	return cmdString(cmd, cmd.val)
}

func (cmd *HotKeysCmd) readReply(rd *proto.Reader) error {
	// HOTKEYS GET response is wrapped in an array for aggregation support
	arrayLen, err := rd.ReadArrayLen()
	if err != nil {
		return err
	}

	if arrayLen == 0 {
		// Empty array means no tracking was started or after reset
		cmd.val = nil
		return nil
	}

	// Read the first (and typically only) element which is a map
	n, err := rd.ReadMapLen()
	if err != nil {
		return err
	}

	result := &HotKeysResult{}
	data := make(map[string]interface{}, n)

	for i := 0; i < n; i++ {
		k, err := rd.ReadString()
		if err != nil {
			return err
		}
		v, err := rd.ReadReply()
		if err != nil {
			if err == Nil {
				data[k] = Nil
				continue
			}
			if err, ok := err.(proto.RedisError); ok {
				data[k] = err
				continue
			}
			return err
		}
		data[k] = v
	}

	if v, ok := data["tracking-active"].(int64); ok {
		result.TrackingActive = v == 1
	}
	if v, ok := data["sample-ratio"].(int64); ok {
		result.SampleRatio = uint8(v)
	}
	if v, ok := data["selected-slots"].([]interface{}); ok {
		result.SelectedSlots = make([]HotKeysSlotRange, 0, len(v))
		for _, slot := range v {
			switch s := slot.(type) {
			case int64:
				// Single slot
				result.SelectedSlots = append(result.SelectedSlots, HotKeysSlotRange{s})
			case []interface{}:
				// Slot range
				slotRange := make(HotKeysSlotRange, 0, len(s))
				for _, sr := range s {
					if val, ok := sr.(int64); ok {
						slotRange = append(slotRange, val)
					}
				}
				result.SelectedSlots = append(result.SelectedSlots, slotRange)
			}
		}
	}
	if v, ok := data["sampled-commands-selected-slots-us"].(int64); ok {
		result.SampledCommandsSelectedSlots = time.Duration(v) * time.Microsecond
	}
	if v, ok := data["all-commands-selected-slots-us"].(int64); ok {
		result.AllCommandsSelectedSlots = time.Duration(v) * time.Microsecond
	}
	if v, ok := data["all-commands-all-slots-us"].(int64); ok {
		result.AllCommandsAllSlots = time.Duration(v) * time.Microsecond
	}
	if v, ok := data["net-bytes-sampled-commands-selected-slots"].(int64); ok {
		result.NetBytesSampledCommandsSelectedSlots = v
	}
	if v, ok := data["net-bytes-all-commands-selected-slots"].(int64); ok {
		result.NetBytesAllCommandsSelectedSlots = v
	}
	if v, ok := data["net-bytes-all-commands-all-slots"].(int64); ok {
		result.NetBytesAllCommandsAllSlots = v
	}
	if v, ok := data["collection-start-time-unix-ms"].(int64); ok {
		result.CollectionStartTime = time.UnixMilli(v)
	}
	if v, ok := data["collection-duration-ms"].(int64); ok {
		result.CollectionDuration = time.Duration(v) * time.Millisecond
	}
	if v, ok := data["used-cpu-sys-ms"].(int64); ok {
		result.UsedCPUSys = time.Duration(v) * time.Millisecond
	}
	if v, ok := data["used-cpu-user-ms"].(int64); ok {
		result.UsedCPUUser = time.Duration(v) * time.Millisecond
	}
	if v, ok := data["total-net-bytes"].(int64); ok {
		result.TotalNetBytes = v
	}

	if v, ok := data["by-cpu-time-us"].([]interface{}); ok {
		result.ByCPUTime = parseHotKeysKeyEntries(v)
	}

	if v, ok := data["by-net-bytes"].([]interface{}); ok {
		result.ByNetBytes = parseHotKeysKeyEntries(v)
	}

	// Only the first element of the outer array is parsed; drain the rest so a
	// server that wraps more than one element doesn't leave frames on the wire.
	for i := 1; i < arrayLen; i++ {
		if err := rd.DiscardNext(); err != nil {
			return err
		}
	}

	cmd.val = result
	return nil
}

// parseHotKeysKeyEntries parses the key-value pairs from HOTKEYS GET response.
func parseHotKeysKeyEntries(v []interface{}) []HotKeysKeyEntry {
	entries := make([]HotKeysKeyEntry, 0, len(v)/2)
	for i := 0; i < len(v); i += 2 {
		if i+1 < len(v) {
			key, keyOk := v[i].(string)
			if keyOk {
				entries = append(entries, HotKeysKeyEntry{
					Key:   key,
					Value: v[i+1], // Can be int64 or string
				})
			}
		}
	}
	return entries
}

func (cmd *HotKeysCmd) Clone() Cmder {
	var val *HotKeysResult
	if cmd.val != nil {
		val = &HotKeysResult{
			TrackingActive:                       cmd.val.TrackingActive,
			SampleRatio:                          cmd.val.SampleRatio,
			SampledCommandsSelectedSlots:         cmd.val.SampledCommandsSelectedSlots,
			AllCommandsSelectedSlots:             cmd.val.AllCommandsSelectedSlots,
			AllCommandsAllSlots:                  cmd.val.AllCommandsAllSlots,
			NetBytesSampledCommandsSelectedSlots: cmd.val.NetBytesSampledCommandsSelectedSlots,
			NetBytesAllCommandsSelectedSlots:     cmd.val.NetBytesAllCommandsSelectedSlots,
			NetBytesAllCommandsAllSlots:          cmd.val.NetBytesAllCommandsAllSlots,
			CollectionStartTime:                  cmd.val.CollectionStartTime,
			CollectionDuration:                   cmd.val.CollectionDuration,
			UsedCPUSys:                           cmd.val.UsedCPUSys,
			UsedCPUUser:                          cmd.val.UsedCPUUser,
			TotalNetBytes:                        cmd.val.TotalNetBytes,
		}
		if cmd.val.SelectedSlots != nil {
			val.SelectedSlots = make([]HotKeysSlotRange, len(cmd.val.SelectedSlots))
			for i, sr := range cmd.val.SelectedSlots {
				val.SelectedSlots[i] = make(HotKeysSlotRange, len(sr))
				copy(val.SelectedSlots[i], sr)
			}
		}
		if cmd.val.ByCPUTime != nil {
			val.ByCPUTime = make([]HotKeysKeyEntry, len(cmd.val.ByCPUTime))
			copy(val.ByCPUTime, cmd.val.ByCPUTime)
		}
		if cmd.val.ByNetBytes != nil {
			val.ByNetBytes = make([]HotKeysKeyEntry, len(cmd.val.ByNetBytes))
			copy(val.ByNetBytes, cmd.val.ByNetBytes)
		}
	}
	return &HotKeysCmd{
		baseCmd: cmd.cloneBaseCmd(),
		val:     val,
	}
}

//-----------------------------------------------------------------------

type MapStringInterfaceCmd struct {
	baseCmd

	val map[string]interface{}
}

var _ Cmder = (*MapStringInterfaceCmd)(nil)

func NewMapStringInterfaceCmd(ctx context.Context, args ...interface{}) *MapStringInterfaceCmd {
	return &MapStringInterfaceCmd{
		baseCmd: baseCmd{
			ctx:     ctx,
			args:    args,
			cmdType: CmdTypeMapStringInterface,
		},
	}
}

func (cmd *MapStringInterfaceCmd) SetVal(val map[string]interface{}) {
	cmd.val = val
}

func (cmd *MapStringInterfaceCmd) Val() map[string]interface{} {
	cmd.await()
	return cmd.val
}

func (cmd *MapStringInterfaceCmd) Result() (map[string]interface{}, error) {
	cmd.await()
	return cmd.val, cmd.err
}

func (cmd *MapStringInterfaceCmd) String() string {
	cmd.await()
	return cmdString(cmd, cmd.val)
}

func (cmd *MapStringInterfaceCmd) readReply(rd *proto.Reader) error {
	n, err := rd.ReadMapLen()
	if err != nil {
		return err
	}

	cmd.val = make(map[string]interface{}, n)
	for i := 0; i < n; i++ {
		k, err := rd.ReadString()
		if err != nil {
			return err
		}
		v, err := rd.ReadReply()
		if err != nil {
			if err == Nil {
				cmd.val[k] = Nil
				continue
			}
			if err, ok := err.(proto.RedisError); ok {
				cmd.val[k] = err
				continue
			}
			return err
		}
		cmd.val[k] = v
	}
	return nil
}

func (cmd *MapStringInterfaceCmd) Clone() Cmder {
	var val map[string]interface{}
	if cmd.val != nil {
		val = make(map[string]interface{}, len(cmd.val))
		for k, v := range cmd.val {
			val[k] = v
		}
	}
	return &MapStringInterfaceCmd{
		baseCmd: cmd.cloneBaseCmd(),
		val:     val,
	}
}

//-----------------------------------------------------------------------

type MapStringStringSliceCmd struct {
	baseCmd

	val []map[string]string
}

var _ Cmder = (*MapStringStringSliceCmd)(nil)

func NewMapStringStringSliceCmd(ctx context.Context, args ...interface{}) *MapStringStringSliceCmd {
	return &MapStringStringSliceCmd{
		baseCmd: baseCmd{
			ctx:     ctx,
			args:    args,
			cmdType: CmdTypeMapStringStringSlice,
		},
	}
}

func (cmd *MapStringStringSliceCmd) SetVal(val []map[string]string) {
	cmd.val = val
}

func (cmd *MapStringStringSliceCmd) Val() []map[string]string {
	cmd.await()
	return cmd.val
}

func (cmd *MapStringStringSliceCmd) Result() ([]map[string]string, error) {
	cmd.await()
	return cmd.val, cmd.err
}

func (cmd *MapStringStringSliceCmd) String() string {
	cmd.await()
	return cmdString(cmd, cmd.val)
}

func (cmd *MapStringStringSliceCmd) readReply(rd *proto.Reader) error {
	n, err := rd.ReadArrayLen()
	if err != nil {
		return err
	}

	cmd.val = make([]map[string]string, n)
	for i := 0; i < n; i++ {
		nn, err := rd.ReadMapLen()
		if err != nil {
			return err
		}
		cmd.val[i] = make(map[string]string, nn)
		for f := 0; f < nn; f++ {
			k, err := rd.ReadString()
			if err != nil {
				return err
			}

			v, err := rd.ReadString()
			if err != nil {
				return err
			}
			cmd.val[i][k] = v
		}
	}
	return nil
}

func (cmd *MapStringStringSliceCmd) Clone() Cmder {
	var val []map[string]string
	if cmd.val != nil {
		val = make([]map[string]string, len(cmd.val))
		for i, m := range cmd.val {
			if m != nil {
				val[i] = make(map[string]string, len(m))
				for k, v := range m {
					val[i][k] = v
				}
			}
		}
	}
	return &MapStringStringSliceCmd{
		baseCmd: cmd.cloneBaseCmd(),
		val:     val,
	}
}

// -----------------------------------------------------------------------

// MapMapStringInterfaceCmd represents a command that returns a map of strings to interface{}.
type MapMapStringInterfaceCmd struct {
	baseCmd
	val map[string]interface{}
}

func NewMapMapStringInterfaceCmd(ctx context.Context, args ...interface{}) *MapMapStringInterfaceCmd {
	return &MapMapStringInterfaceCmd{
		baseCmd: baseCmd{
			ctx:     ctx,
			args:    args,
			cmdType: CmdTypeMapMapStringInterface,
		},
	}
}

func (cmd *MapMapStringInterfaceCmd) String() string {
	cmd.await()
	return cmdString(cmd, cmd.val)
}

func (cmd *MapMapStringInterfaceCmd) SetVal(val map[string]interface{}) {
	cmd.val = val
}

func (cmd *MapMapStringInterfaceCmd) Result() (map[string]interface{}, error) {
	cmd.await()
	return cmd.val, cmd.err
}

func (cmd *MapMapStringInterfaceCmd) Val() map[string]interface{} {
	cmd.await()
	return cmd.val
}

// readReply will try to parse the reply from the proto.Reader for both resp2 and resp3
func (cmd *MapMapStringInterfaceCmd) readReply(rd *proto.Reader) (err error) {
	data, err := rd.ReadReply()
	if err != nil {
		return err
	}
	resultMap := map[string]interface{}{}

	switch midResponse := data.(type) {
	case map[interface{}]interface{}: // resp3 will return map
		for k, v := range midResponse {
			stringKey, ok := k.(string)
			if !ok {
				return fmt.Errorf("redis: invalid map key %#v", k)
			}
			resultMap[stringKey] = v
		}
	case []interface{}: // resp2 will return array of arrays
		n := len(midResponse)
		for i := 0; i < n; i++ {
			finalArr, ok := midResponse[i].([]interface{}) // final array that we need to transform to map
			if !ok {
				return fmt.Errorf("redis: unexpected response %#v", data)
			}
			m := len(finalArr)
			if m%2 != 0 { // since this should be map, keys should be even number
				return fmt.Errorf("redis: unexpected response %#v", data)
			}

			for j := 0; j < m; j += 2 {
				stringKey, ok := finalArr[j].(string) // the first one
				if !ok {
					return fmt.Errorf("redis: invalid map key %#v", finalArr[i])
				}
				resultMap[stringKey] = finalArr[j+1] // second one is value
			}
		}
	default:
		return fmt.Errorf("redis: unexpected response %#v", data)
	}

	cmd.val = resultMap
	return nil
}

func (cmd *MapMapStringInterfaceCmd) Clone() Cmder {
	var val map[string]interface{}
	if cmd.val != nil {
		val = make(map[string]interface{}, len(cmd.val))
		for k, v := range cmd.val {
			val[k] = v
		}
	}
	return &MapMapStringInterfaceCmd{
		baseCmd: cmd.cloneBaseCmd(),
		val:     val,
	}
}

//-----------------------------------------------------------------------

type MapStringInterfaceSliceCmd struct {
	baseCmd

	val []map[string]interface{}
}

var _ Cmder = (*MapStringInterfaceSliceCmd)(nil)

func NewMapStringInterfaceSliceCmd(ctx context.Context, args ...interface{}) *MapStringInterfaceSliceCmd {
	return &MapStringInterfaceSliceCmd{
		baseCmd: baseCmd{
			ctx:     ctx,
			args:    args,
			cmdType: CmdTypeMapStringInterfaceSlice,
		},
	}
}

func (cmd *MapStringInterfaceSliceCmd) SetVal(val []map[string]interface{}) {
	cmd.val = val
}

func (cmd *MapStringInterfaceSliceCmd) Val() []map[string]interface{} {
	cmd.await()
	return cmd.val
}

func (cmd *MapStringInterfaceSliceCmd) Result() ([]map[string]interface{}, error) {
	cmd.await()
	return cmd.val, cmd.err
}

func (cmd *MapStringInterfaceSliceCmd) String() string {
	cmd.await()
	return cmdString(cmd, cmd.val)
}

func (cmd *MapStringInterfaceSliceCmd) readReply(rd *proto.Reader) error {
	n, err := rd.ReadArrayLen()
	if err != nil {
		return err
	}

	cmd.val = make([]map[string]interface{}, n)
	for i := 0; i < n; i++ {
		nn, err := rd.ReadMapLen()
		if err != nil {
			return err
		}
		cmd.val[i] = make(map[string]interface{}, nn)
		for f := 0; f < nn; f++ {
			k, err := rd.ReadString()
			if err != nil {
				return err
			}
			v, err := rd.ReadReply()
			if err != nil {
				if err != Nil {
					return err
				}
			}
			cmd.val[i][k] = v
		}
	}
	return nil
}

func (cmd *MapStringInterfaceSliceCmd) Clone() Cmder {
	var val []map[string]interface{}
	if cmd.val != nil {
		val = make([]map[string]interface{}, len(cmd.val))
		for i, m := range cmd.val {
			if m != nil {
				val[i] = make(map[string]interface{}, len(m))
				for k, v := range m {
					val[i][k] = v
				}
			}
		}
	}
	return &MapStringInterfaceSliceCmd{
		baseCmd: cmd.cloneBaseCmd(),
		val:     val,
	}
}

//------------------------------------------------------------------------------

type KeyValuesCmd struct {
	baseCmd

	key string
	val []string
}

var _ Cmder = (*KeyValuesCmd)(nil)

func NewKeyValuesCmd(ctx context.Context, args ...interface{}) *KeyValuesCmd {
	return &KeyValuesCmd{
		baseCmd: baseCmd{
			ctx:     ctx,
			args:    args,
			cmdType: CmdTypeKeyValues,
		},
	}
}

func (cmd *KeyValuesCmd) SetVal(key string, val []string) {
	cmd.key = key
	cmd.val = val
}

func (cmd *KeyValuesCmd) Val() (string, []string) {
	cmd.await()
	return cmd.key, cmd.val
}

func (cmd *KeyValuesCmd) Result() (string, []string, error) {
	cmd.await()
	return cmd.key, cmd.val, cmd.err
}

func (cmd *KeyValuesCmd) String() string {
	cmd.await()
	return cmdString(cmd, cmd.val)
}

func (cmd *KeyValuesCmd) readReply(rd *proto.Reader) (err error) {
	if err = rd.ReadFixedArrayLen(2); err != nil {
		return err
	}

	cmd.key, err = rd.ReadString()
	if err != nil {
		return err
	}

	n, err := rd.ReadArrayLen()
	if err != nil {
		return err
	}
	cmd.val = make([]string, n)
	for i := 0; i < n; i++ {
		cmd.val[i], err = rd.ReadString()
		if err != nil {
			return err
		}
	}

	return nil
}

func (cmd *KeyValuesCmd) Clone() Cmder {
	var val []string
	if cmd.val != nil {
		val = make([]string, len(cmd.val))
		copy(val, cmd.val)
	}
	return &KeyValuesCmd{
		baseCmd: cmd.cloneBaseCmd(),
		key:     cmd.key,
		val:     val,
	}
}

//------------------------------------------------------------------------------

type ZSliceWithKeyCmd struct {
	baseCmd

	key string
	val []Z
}

var _ Cmder = (*ZSliceWithKeyCmd)(nil)

func NewZSliceWithKeyCmd(ctx context.Context, args ...interface{}) *ZSliceWithKeyCmd {
	return &ZSliceWithKeyCmd{
		baseCmd: baseCmd{
			ctx:     ctx,
			args:    args,
			cmdType: CmdTypeZSliceWithKey,
		},
	}
}

func (cmd *ZSliceWithKeyCmd) SetVal(key string, val []Z) {
	cmd.key = key
	cmd.val = val
}

func (cmd *ZSliceWithKeyCmd) Val() (string, []Z) {
	cmd.await()
	return cmd.key, cmd.val
}

func (cmd *ZSliceWithKeyCmd) Result() (string, []Z, error) {
	cmd.await()
	return cmd.key, cmd.val, cmd.err
}

func (cmd *ZSliceWithKeyCmd) String() string {
	cmd.await()
	return cmdString(cmd, cmd.val)
}

func (cmd *ZSliceWithKeyCmd) readReply(rd *proto.Reader) (err error) {
	if err = rd.ReadFixedArrayLen(2); err != nil {
		return err
	}

	cmd.key, err = rd.ReadString()
	if err != nil {
		return err
	}

	n, err := rd.ReadArrayLen()
	if err != nil {
		return err
	}

	typ, err := rd.PeekReplyType()
	if err != nil {
		return err
	}
	array := typ == proto.RespArray

	if array {
		cmd.val = make([]Z, n)
	} else {
		if n%2 != 0 {
			return fmt.Errorf("redis: got %d elements in the sorted set array, wanted a multiple of 2", n)
		}
		cmd.val = make([]Z, n/2)
	}

	for i := 0; i < len(cmd.val); i++ {
		if array {
			if err = rd.ReadFixedArrayLen(2); err != nil {
				return err
			}
		}

		if cmd.val[i].Member, err = rd.ReadString(); err != nil {
			return err
		}

		if cmd.val[i].Score, err = rd.ReadFloat(); err != nil {
			return err
		}
	}

	return nil
}

func (cmd *ZSliceWithKeyCmd) Clone() Cmder {
	var val []Z
	if cmd.val != nil {
		val = make([]Z, len(cmd.val))
		copy(val, cmd.val)
	}
	return &ZSliceWithKeyCmd{
		baseCmd: cmd.cloneBaseCmd(),
		key:     cmd.key,
		val:     val,
	}
}

type Function struct {
	Name        string
	Description string
	Flags       []string
}

type Library struct {
	Name      string
	Engine    string
	Functions []Function
	Code      string
}

type FunctionListCmd struct {
	baseCmd

	val []Library
}

var _ Cmder = (*FunctionListCmd)(nil)

func NewFunctionListCmd(ctx context.Context, args ...interface{}) *FunctionListCmd {
	return &FunctionListCmd{
		baseCmd: baseCmd{
			ctx:     ctx,
			args:    args,
			cmdType: CmdTypeFunctionList,
		},
	}
}

func (cmd *FunctionListCmd) SetVal(val []Library) {
	cmd.val = val
}

func (cmd *FunctionListCmd) String() string {
	cmd.await()
	return cmdString(cmd, cmd.val)
}

func (cmd *FunctionListCmd) Val() []Library {
	cmd.await()
	return cmd.val
}

func (cmd *FunctionListCmd) Result() ([]Library, error) {
	cmd.await()
	return cmd.val, cmd.err
}

func (cmd *FunctionListCmd) First() (*Library, error) {
	cmd.await()
	if cmd.err != nil {
		return nil, cmd.err
	}
	if len(cmd.val) > 0 {
		return &cmd.val[0], nil
	}
	return nil, Nil
}

func (cmd *FunctionListCmd) readReply(rd *proto.Reader) (err error) {
	n, err := rd.ReadArrayLen()
	if err != nil {
		return err
	}

	libraries := make([]Library, n)
	for i := 0; i < n; i++ {
		nn, err := rd.ReadMapLen()
		if err != nil {
			return err
		}

		library := Library{}
		for f := 0; f < nn; f++ {
			key, err := rd.ReadString()
			if err != nil {
				return err
			}

			switch key {
			case "library_name":
				library.Name, err = rd.ReadString()
			case "engine":
				library.Engine, err = rd.ReadString()
			case "functions":
				library.Functions, err = cmd.readFunctions(rd)
			case "library_code":
				library.Code, err = rd.ReadString()
			default:
				return fmt.Errorf("redis: function list unexpected key %s", key)
			}

			if err != nil {
				return err
			}
		}

		libraries[i] = library
	}
	cmd.val = libraries
	return nil
}

func (cmd *FunctionListCmd) readFunctions(rd *proto.Reader) ([]Function, error) {
	n, err := rd.ReadArrayLen()
	if err != nil {
		return nil, err
	}

	functions := make([]Function, n)
	for i := 0; i < n; i++ {
		nn, err := rd.ReadMapLen()
		if err != nil {
			return nil, err
		}

		function := Function{}
		for f := 0; f < nn; f++ {
			key, err := rd.ReadString()
			if err != nil {
				return nil, err
			}

			switch key {
			case "name":
				if function.Name, err = rd.ReadString(); err != nil {
					return nil, err
				}
			case "description":
				if function.Description, err = rd.ReadString(); err != nil && err != Nil {
					return nil, err
				}
			case "flags":
				// resp set
				nx, err := rd.ReadArrayLen()
				if err != nil {
					return nil, err
				}

				function.Flags = make([]string, nx)
				for j := 0; j < nx; j++ {
					if function.Flags[j], err = rd.ReadString(); err != nil {
						return nil, err
					}
				}
			default:
				return nil, fmt.Errorf("redis: function list unexpected key %s", key)
			}
		}

		functions[i] = function
	}
	return functions, nil
}

func (cmd *FunctionListCmd) Clone() Cmder {
	var val []Library
	if cmd.val != nil {
		val = make([]Library, len(cmd.val))
		for i, lib := range cmd.val {
			val[i] = Library{
				Name:   lib.Name,
				Engine: lib.Engine,
				Code:   lib.Code,
			}
			if lib.Functions != nil {
				val[i].Functions = make([]Function, len(lib.Functions))
				for j, fn := range lib.Functions {
					val[i].Functions[j] = Function{
						Name:        fn.Name,
						Description: fn.Description,
					}
					if fn.Flags != nil {
						val[i].Functions[j].Flags = make([]string, len(fn.Flags))
						copy(val[i].Functions[j].Flags, fn.Flags)
					}
				}
			}
		}
	}
	return &FunctionListCmd{
		baseCmd: cmd.cloneBaseCmd(),
		val:     val,
	}
}

// FunctionStats contains information about the scripts currently executing on the server, and the available engines
//   - Engines:
//     Statistics about the engine like number of functions and number of libraries
//   - RunningScript:
//     The script currently running on the shard we're connecting to.
//     For Redis Enterprise and Redis Cloud, this represents the
//     function with the longest running time, across all the running functions, on all shards
//   - RunningScripts
//     All scripts currently running in a Redis Enterprise clustered database.
//     Only available on Redis Enterprise
type FunctionStats struct {
	Engines   []Engine
	isRunning bool
	rs        RunningScript
	allrs     []RunningScript
}

func (fs *FunctionStats) Running() bool {
	return fs.isRunning
}

func (fs *FunctionStats) RunningScript() (RunningScript, bool) {
	return fs.rs, fs.isRunning
}

// AllRunningScripts returns all scripts currently running in a Redis Enterprise clustered database.
// Only available on Redis Enterprise
func (fs *FunctionStats) AllRunningScripts() []RunningScript {
	return fs.allrs
}

type RunningScript struct {
	Name     string
	Command  []string
	Duration time.Duration
}

type Engine struct {
	Language       string
	LibrariesCount int64
	FunctionsCount int64
}

type FunctionStatsCmd struct {
	baseCmd
	val FunctionStats
}

var _ Cmder = (*FunctionStatsCmd)(nil)

func NewFunctionStatsCmd(ctx context.Context, args ...interface{}) *FunctionStatsCmd {
	return &FunctionStatsCmd{
		baseCmd: baseCmd{
			ctx:     ctx,
			args:    args,
			cmdType: CmdTypeFunctionStats,
		},
	}
}

func (cmd *FunctionStatsCmd) SetVal(val FunctionStats) {
	cmd.val = val
}

func (cmd *FunctionStatsCmd) String() string {
	cmd.await()
	return cmdString(cmd, cmd.val)
}

func (cmd *FunctionStatsCmd) Val() FunctionStats {
	cmd.await()
	return cmd.val
}

func (cmd *FunctionStatsCmd) Result() (FunctionStats, error) {
	cmd.await()
	return cmd.val, cmd.err
}

func (cmd *FunctionStatsCmd) readReply(rd *proto.Reader) (err error) {
	n, err := rd.ReadMapLen()
	if err != nil {
		return err
	}

	var key string
	var result FunctionStats
	for f := 0; f < n; f++ {
		key, err = rd.ReadString()
		if err != nil {
			return err
		}

		switch key {
		case "running_script":
			result.rs, result.isRunning, err = cmd.readRunningScript(rd)
		case "engines":
			result.Engines, err = cmd.readEngines(rd)
		case "all_running_scripts": // Redis Enterprise only
			result.allrs, result.isRunning, err = cmd.readRunningScripts(rd)
		default:
			return fmt.Errorf("redis: function stats unexpected key %s", key)
		}

		if err != nil {
			return err
		}
	}

	cmd.val = result
	return nil
}

func (cmd *FunctionStatsCmd) readRunningScript(rd *proto.Reader) (RunningScript, bool, error) {
	err := rd.ReadFixedMapLen(3)
	if err != nil {
		if err == Nil {
			return RunningScript{}, false, nil
		}
		return RunningScript{}, false, err
	}

	var runningScript RunningScript
	for i := 0; i < 3; i++ {
		key, err := rd.ReadString()
		if err != nil {
			return RunningScript{}, false, err
		}

		switch key {
		case "name":
			runningScript.Name, err = rd.ReadString()
		case "duration_ms":
			runningScript.Duration, err = cmd.readDuration(rd)
		case "command":
			runningScript.Command, err = cmd.readCommand(rd)
		default:
			return RunningScript{}, false, fmt.Errorf("redis: function stats unexpected running_script key %s", key)
		}

		if err != nil {
			return RunningScript{}, false, err
		}
	}

	return runningScript, true, nil
}

func (cmd *FunctionStatsCmd) readEngines(rd *proto.Reader) ([]Engine, error) {
	n, err := rd.ReadMapLen()
	if err != nil {
		return nil, err
	}

	engines := make([]Engine, 0, n)
	for i := 0; i < n; i++ {
		engine := Engine{}
		engine.Language, err = rd.ReadString()
		if err != nil {
			return nil, err
		}

		err = rd.ReadFixedMapLen(2)
		if err != nil {
			return nil, fmt.Errorf("redis: function stats unexpected %s engine map length", engine.Language)
		}

		for i := 0; i < 2; i++ {
			key, err := rd.ReadString()
			if err != nil {
				return nil, err
			}
			switch key {
			case "libraries_count":
				engine.LibrariesCount, err = rd.ReadInt()
			case "functions_count":
				engine.FunctionsCount, err = rd.ReadInt()
			default:
				// Unknown field: drain its value so the reader stays aligned
				// with the rest of the reply.
				err = rd.DiscardNext()
			}
			if err != nil {
				return nil, err
			}
		}

		engines = append(engines, engine)
	}
	return engines, nil
}

func (cmd *FunctionStatsCmd) readDuration(rd *proto.Reader) (time.Duration, error) {
	t, err := rd.ReadInt()
	if err != nil {
		return time.Duration(0), err
	}
	return time.Duration(t) * time.Millisecond, nil
}

func (cmd *FunctionStatsCmd) readCommand(rd *proto.Reader) ([]string, error) {
	n, err := rd.ReadArrayLen()
	if err != nil {
		return nil, err
	}

	command := make([]string, 0, n)
	for i := 0; i < n; i++ {
		x, err := rd.ReadString()
		if err != nil {
			return nil, err
		}
		command = append(command, x)
	}

	return command, nil
}

func (cmd *FunctionStatsCmd) readRunningScripts(rd *proto.Reader) ([]RunningScript, bool, error) {
	n, err := rd.ReadArrayLen()
	if err != nil {
		return nil, false, err
	}

	runningScripts := make([]RunningScript, 0, n)
	for i := 0; i < n; i++ {
		rs, _, err := cmd.readRunningScript(rd)
		if err != nil {
			return nil, false, err
		}
		runningScripts = append(runningScripts, rs)
	}

	return runningScripts, len(runningScripts) > 0, nil
}

func (cmd *FunctionStatsCmd) Clone() Cmder {
	val := FunctionStats{
		isRunning: cmd.val.isRunning,
		rs:        cmd.val.rs, // RunningScript is a simple struct, can be copied directly
	}
	if cmd.val.Engines != nil {
		val.Engines = make([]Engine, len(cmd.val.Engines))
		copy(val.Engines, cmd.val.Engines)
	}
	if cmd.val.allrs != nil {
		val.allrs = make([]RunningScript, len(cmd.val.allrs))
		for i, rs := range cmd.val.allrs {
			val.allrs[i] = RunningScript{
				Name:     rs.Name,
				Duration: rs.Duration,
			}
			if rs.Command != nil {
				val.allrs[i].Command = make([]string, len(rs.Command))
				copy(val.allrs[i].Command, rs.Command)
			}
		}
	}
	return &FunctionStatsCmd{
		baseCmd: cmd.cloneBaseCmd(),
		val:     val,
	}
}

//------------------------------------------------------------------------------

// LCSQuery is a parameter used for the LCS command
type LCSQuery struct {
	Key1         string
	Key2         string
	Len          bool
	Idx          bool
	MinMatchLen  int
	WithMatchLen bool
}

// LCSMatch is the result set of the LCS command.
type LCSMatch struct {
	MatchString string
	Matches     []LCSMatchedPosition
	Len         int64
}

type LCSMatchedPosition struct {
	Key1 LCSPosition
	Key2 LCSPosition

	// only for withMatchLen is true
	MatchLen int64
}

type LCSPosition struct {
	Start int64
	End   int64
}

type LCSCmd struct {
	baseCmd

	// 1: match string
	// 2: match len
	// 3: match idx LCSMatch
	readType uint8
	val      *LCSMatch
}

func NewLCSCmd(ctx context.Context, q *LCSQuery) *LCSCmd {
	args := make([]interface{}, 3, 7)
	args[0] = "lcs"
	args[1] = q.Key1
	args[2] = q.Key2

	cmd := &LCSCmd{readType: 1}
	if q.Len {
		cmd.readType = 2
		args = append(args, "len")
	} else if q.Idx {
		cmd.readType = 3
		args = append(args, "idx")
		if q.MinMatchLen != 0 {
			args = append(args, "minmatchlen", q.MinMatchLen)
		}
		if q.WithMatchLen {
			args = append(args, "withmatchlen")
		}
	}
	cmd.baseCmd = baseCmd{
		ctx:     ctx,
		args:    args,
		cmdType: CmdTypeLCS,
	}

	return cmd
}

func (cmd *LCSCmd) SetVal(val *LCSMatch) {
	cmd.val = val
}

func (cmd *LCSCmd) String() string {
	cmd.await()
	return cmdString(cmd, cmd.val)
}

func (cmd *LCSCmd) Val() *LCSMatch {
	cmd.await()
	return cmd.val
}

func (cmd *LCSCmd) Result() (*LCSMatch, error) {
	cmd.await()
	return cmd.val, cmd.err
}

func (cmd *LCSCmd) readReply(rd *proto.Reader) (err error) {
	lcs := &LCSMatch{}
	switch cmd.readType {
	case 1:
		// match string
		if lcs.MatchString, err = rd.ReadString(); err != nil {
			return err
		}
	case 2:
		// match len
		if lcs.Len, err = rd.ReadInt(); err != nil {
			return err
		}
	case 3:
		// read LCSMatch
		if err = rd.ReadFixedMapLen(2); err != nil {
			return err
		}

		// read matches or len field
		for i := 0; i < 2; i++ {
			key, err := rd.ReadString()
			if err != nil {
				return err
			}

			switch key {
			case "matches":
				// read array of matched positions
				if lcs.Matches, err = cmd.readMatchedPositions(rd); err != nil {
					return err
				}
			case "len":
				// read match length
				if lcs.Len, err = rd.ReadInt(); err != nil {
					return err
				}
			default:
				// Unknown field: drain its value so the reader stays aligned
				// with the rest of the reply.
				if err = rd.DiscardNext(); err != nil {
					return err
				}
			}
		}
	}

	cmd.val = lcs
	return nil
}

func (cmd *LCSCmd) readMatchedPositions(rd *proto.Reader) ([]LCSMatchedPosition, error) {
	n, err := rd.ReadArrayLen()
	if err != nil {
		return nil, err
	}

	positions := make([]LCSMatchedPosition, n)
	for i := 0; i < n; i++ {
		pn, err := rd.ReadArrayLen()
		if err != nil {
			return nil, err
		}

		if positions[i].Key1, err = cmd.readPosition(rd); err != nil {
			return nil, err
		}
		if positions[i].Key2, err = cmd.readPosition(rd); err != nil {
			return nil, err
		}

		// read match length if WithMatchLen is true
		if pn > 2 {
			if positions[i].MatchLen, err = rd.ReadInt(); err != nil {
				return nil, err
			}
		}
	}

	return positions, nil
}

func (cmd *LCSCmd) readPosition(rd *proto.Reader) (pos LCSPosition, err error) {
	if err = rd.ReadFixedArrayLen(2); err != nil {
		return pos, err
	}
	if pos.Start, err = rd.ReadInt(); err != nil {
		return pos, err
	}
	if pos.End, err = rd.ReadInt(); err != nil {
		return pos, err
	}

	return pos, nil
}

func (cmd *LCSCmd) Clone() Cmder {
	var val *LCSMatch
	if cmd.val != nil {
		val = &LCSMatch{
			MatchString: cmd.val.MatchString,
			Len:         cmd.val.Len,
		}
		if cmd.val.Matches != nil {
			val.Matches = make([]LCSMatchedPosition, len(cmd.val.Matches))
			copy(val.Matches, cmd.val.Matches)
		}
	}
	return &LCSCmd{
		baseCmd:  cmd.cloneBaseCmd(),
		readType: cmd.readType,
		val:      val,
	}
}

// ------------------------------------------------------------------------

type KeyFlags struct {
	Key   string
	Flags []string
}

type KeyFlagsCmd struct {
	baseCmd

	val []KeyFlags
}

var _ Cmder = (*KeyFlagsCmd)(nil)

func NewKeyFlagsCmd(ctx context.Context, args ...interface{}) *KeyFlagsCmd {
	return &KeyFlagsCmd{
		baseCmd: baseCmd{
			ctx:     ctx,
			args:    args,
			cmdType: CmdTypeKeyFlags,
		},
	}
}

func (cmd *KeyFlagsCmd) SetVal(val []KeyFlags) {
	cmd.val = val
}

func (cmd *KeyFlagsCmd) Val() []KeyFlags {
	cmd.await()
	return cmd.val
}

func (cmd *KeyFlagsCmd) Result() ([]KeyFlags, error) {
	cmd.await()
	return cmd.val, cmd.err
}

func (cmd *KeyFlagsCmd) String() string {
	cmd.await()
	return cmdString(cmd, cmd.val)
}

func (cmd *KeyFlagsCmd) readReply(rd *proto.Reader) error {
	n, err := rd.ReadArrayLen()
	if err != nil {
		return err
	}

	if n == 0 {
		cmd.val = make([]KeyFlags, 0)
		return nil
	}

	cmd.val = make([]KeyFlags, n)

	for i := 0; i < len(cmd.val); i++ {

		if err = rd.ReadFixedArrayLen(2); err != nil {
			return err
		}

		if cmd.val[i].Key, err = rd.ReadString(); err != nil {
			return err
		}
		flagsLen, err := rd.ReadArrayLen()
		if err != nil {
			return err
		}
		cmd.val[i].Flags = make([]string, flagsLen)

		for j := 0; j < flagsLen; j++ {
			if cmd.val[i].Flags[j], err = rd.ReadString(); err != nil {
				return err
			}
		}
	}

	return nil
}

func (cmd *KeyFlagsCmd) Clone() Cmder {
	var val []KeyFlags
	if cmd.val != nil {
		val = make([]KeyFlags, len(cmd.val))
		for i, kf := range cmd.val {
			val[i] = KeyFlags{
				Key: kf.Key,
			}
			if kf.Flags != nil {
				val[i].Flags = make([]string, len(kf.Flags))
				copy(val[i].Flags, kf.Flags)
			}
		}
	}
	return &KeyFlagsCmd{
		baseCmd: cmd.cloneBaseCmd(),
		val:     val,
	}
}

// ---------------------------------------------------------------------------------------------------

type ClusterLink struct {
	Direction           string
	Node                string
	CreateTime          int64
	Events              string
	SendBufferAllocated int64
	SendBufferUsed      int64
}

type ClusterLinksCmd struct {
	baseCmd

	val []ClusterLink
}

var _ Cmder = (*ClusterLinksCmd)(nil)

func NewClusterLinksCmd(ctx context.Context, args ...interface{}) *ClusterLinksCmd {
	return &ClusterLinksCmd{
		baseCmd: baseCmd{
			ctx:     ctx,
			args:    args,
			cmdType: CmdTypeClusterLinks,
		},
	}
}

func (cmd *ClusterLinksCmd) SetVal(val []ClusterLink) {
	cmd.val = val
}

func (cmd *ClusterLinksCmd) Val() []ClusterLink {
	cmd.await()
	return cmd.val
}

func (cmd *ClusterLinksCmd) Result() ([]ClusterLink, error) {
	cmd.await()
	return cmd.val, cmd.err
}

func (cmd *ClusterLinksCmd) String() string {
	cmd.await()
	return cmdString(cmd, cmd.val)
}

func (cmd *ClusterLinksCmd) readReply(rd *proto.Reader) error {
	n, err := rd.ReadArrayLen()
	if err != nil {
		return err
	}
	cmd.val = make([]ClusterLink, n)

	for i := 0; i < len(cmd.val); i++ {
		m, err := rd.ReadMapLen()
		if err != nil {
			return err
		}

		for j := 0; j < m; j++ {
			key, err := rd.ReadString()
			if err != nil {
				return err
			}

			switch key {
			case "direction":
				cmd.val[i].Direction, err = rd.ReadString()
			case "node":
				cmd.val[i].Node, err = rd.ReadString()
			case "create-time":
				cmd.val[i].CreateTime, err = rd.ReadInt()
			case "events":
				cmd.val[i].Events, err = rd.ReadString()
			case "send-buffer-allocated":
				cmd.val[i].SendBufferAllocated, err = rd.ReadInt()
			case "send-buffer-used":
				cmd.val[i].SendBufferUsed, err = rd.ReadInt()
			default:
				return fmt.Errorf("redis: unexpected key %q in CLUSTER LINKS reply", key)
			}

			if err != nil {
				return err
			}
		}
	}

	return nil
}

func (cmd *ClusterLinksCmd) Clone() Cmder {
	var val []ClusterLink
	if cmd.val != nil {
		val = make([]ClusterLink, len(cmd.val))
		copy(val, cmd.val)
	}
	return &ClusterLinksCmd{
		baseCmd: cmd.cloneBaseCmd(),
		val:     val,
	}
}

// ------------------------------------------------------------------------------------------------------------------

type SlotRange struct {
	Start int64
	End   int64
}

type Node struct {
	ID                string
	Endpoint          string
	IP                string
	Hostname          string
	Port              int64
	TLSPort           int64
	Role              string
	ReplicationOffset int64
	Health            string
}

type ClusterShard struct {
	Slots []SlotRange
	Nodes []Node
}

type ClusterShardsCmd struct {
	baseCmd

	val []ClusterShard
}

var _ Cmder = (*ClusterShardsCmd)(nil)

func NewClusterShardsCmd(ctx context.Context, args ...interface{}) *ClusterShardsCmd {
	return &ClusterShardsCmd{
		baseCmd: baseCmd{
			ctx:     ctx,
			args:    args,
			cmdType: CmdTypeClusterShards,
		},
	}
}

func (cmd *ClusterShardsCmd) SetVal(val []ClusterShard) {
	cmd.val = val
}

func (cmd *ClusterShardsCmd) Val() []ClusterShard {
	cmd.await()
	return cmd.val
}

func (cmd *ClusterShardsCmd) Result() ([]ClusterShard, error) {
	cmd.await()
	return cmd.val, cmd.err
}

func (cmd *ClusterShardsCmd) String() string {
	cmd.await()
	return cmdString(cmd, cmd.val)
}

func (cmd *ClusterShardsCmd) readReply(rd *proto.Reader) error {
	n, err := rd.ReadArrayLen()
	if err != nil {
		return err
	}
	cmd.val = make([]ClusterShard, n)

	for i := 0; i < n; i++ {
		m, err := rd.ReadMapLen()
		if err != nil {
			return err
		}

		for j := 0; j < m; j++ {
			key, err := rd.ReadString()
			if err != nil {
				return err
			}

			switch key {
			case "slots":
				l, err := rd.ReadArrayLen()
				if err != nil {
					return err
				}
				for k := 0; k < l; k += 2 {
					start, err := rd.ReadInt()
					if err != nil {
						return err
					}

					end, err := rd.ReadInt()
					if err != nil {
						return err
					}

					cmd.val[i].Slots = append(cmd.val[i].Slots, SlotRange{Start: start, End: end})
				}
			case "nodes":
				nodesLen, err := rd.ReadArrayLen()
				if err != nil {
					return err
				}
				cmd.val[i].Nodes = make([]Node, nodesLen)
				for k := 0; k < nodesLen; k++ {
					nodeMapLen, err := rd.ReadMapLen()
					if err != nil {
						return err
					}

					for l := 0; l < nodeMapLen; l++ {
						nodeKey, err := rd.ReadString()
						if err != nil {
							return err
						}

						switch nodeKey {
						case "id":
							cmd.val[i].Nodes[k].ID, err = rd.ReadString()
						case "endpoint":
							cmd.val[i].Nodes[k].Endpoint, err = rd.ReadString()
						case "ip":
							cmd.val[i].Nodes[k].IP, err = rd.ReadString()
						case "hostname":
							cmd.val[i].Nodes[k].Hostname, err = rd.ReadString()
						case "port":
							cmd.val[i].Nodes[k].Port, err = rd.ReadInt()
						case "tls-port":
							cmd.val[i].Nodes[k].TLSPort, err = rd.ReadInt()
						case "role":
							cmd.val[i].Nodes[k].Role, err = rd.ReadString()
						case "replication-offset":
							cmd.val[i].Nodes[k].ReplicationOffset, err = rd.ReadInt()
						case "health":
							cmd.val[i].Nodes[k].Health, err = rd.ReadString()
						default:
							if err = rd.DiscardNext(); err != nil {
								return err
							}
						}

						if err != nil {
							return err
						}
					}
				}
			default:
				if err = rd.DiscardNext(); err != nil {
					return err
				}
			}
		}
	}

	return nil
}

func (cmd *ClusterShardsCmd) Clone() Cmder {
	var val []ClusterShard
	if cmd.val != nil {
		val = make([]ClusterShard, len(cmd.val))
		for i, shard := range cmd.val {
			val[i] = ClusterShard{}
			if shard.Slots != nil {
				val[i].Slots = make([]SlotRange, len(shard.Slots))
				copy(val[i].Slots, shard.Slots)
			}
			if shard.Nodes != nil {
				val[i].Nodes = make([]Node, len(shard.Nodes))
				copy(val[i].Nodes, shard.Nodes)
			}
		}
	}
	return &ClusterShardsCmd{
		baseCmd: cmd.cloneBaseCmd(),
		val:     val,
	}
}

// -----------------------------------------

type RankScore struct {
	Rank  int64
	Score float64
}

type RankWithScoreCmd struct {
	baseCmd

	val RankScore
}

var _ Cmder = (*RankWithScoreCmd)(nil)

func NewRankWithScoreCmd(ctx context.Context, args ...interface{}) *RankWithScoreCmd {
	return &RankWithScoreCmd{
		baseCmd: baseCmd{
			ctx:     ctx,
			args:    args,
			cmdType: CmdTypeRankWithScore,
		},
	}
}

func (cmd *RankWithScoreCmd) SetVal(val RankScore) {
	cmd.val = val
}

func (cmd *RankWithScoreCmd) Val() RankScore {
	cmd.await()
	return cmd.val
}

func (cmd *RankWithScoreCmd) Result() (RankScore, error) {
	cmd.await()
	return cmd.val, cmd.err
}

func (cmd *RankWithScoreCmd) String() string {
	cmd.await()
	return cmdString(cmd, cmd.val)
}

func (cmd *RankWithScoreCmd) readReply(rd *proto.Reader) error {
	if err := rd.ReadFixedArrayLen(2); err != nil {
		return err
	}

	rank, err := rd.ReadInt()
	if err != nil {
		return err
	}

	score, err := rd.ReadFloat()
	if err != nil {
		return err
	}

	cmd.val = RankScore{Rank: rank, Score: score}

	return nil
}

func (cmd *RankWithScoreCmd) Clone() Cmder {
	return &RankWithScoreCmd{
		baseCmd: cmd.cloneBaseCmd(),
		val:     cmd.val, // RankScore is a simple struct, can be copied directly
	}
}

// --------------------------------------------------------------------------------------------------

// ClientFlags is redis-server client flags, copy from redis/src/server.h (redis 7.0)
type ClientFlags uint64

const (
	ClientSlave            ClientFlags = 1 << 0  /* This client is a replica */
	ClientMaster           ClientFlags = 1 << 1  /* This client is a master */
	ClientMonitor          ClientFlags = 1 << 2  /* This client is a slave monitor, see MONITOR */
	ClientMulti            ClientFlags = 1 << 3  /* This client is in a MULTI context */
	ClientBlocked          ClientFlags = 1 << 4  /* The client is waiting in a blocking operation */
	ClientDirtyCAS         ClientFlags = 1 << 5  /* Watched keys modified. EXEC will fail. */
	ClientCloseAfterReply  ClientFlags = 1 << 6  /* Close after writing entire reply. */
	ClientUnBlocked        ClientFlags = 1 << 7  /* This client was unblocked and is stored in server.unblocked_clients */
	ClientScript           ClientFlags = 1 << 8  /* This is a non-connected client used by Lua */
	ClientAsking           ClientFlags = 1 << 9  /* Client issued the ASKING command */
	ClientCloseASAP        ClientFlags = 1 << 10 /* Close this client ASAP */
	ClientUnixSocket       ClientFlags = 1 << 11 /* Client connected via Unix domain socket */
	ClientDirtyExec        ClientFlags = 1 << 12 /* EXEC will fail for errors while queueing */
	ClientMasterForceReply ClientFlags = 1 << 13 /* Queue replies even if is master */
	ClientForceAOF         ClientFlags = 1 << 14 /* Force AOF propagation of current cmd. */
	ClientForceRepl        ClientFlags = 1 << 15 /* Force replication of current cmd. */
	ClientPrePSync         ClientFlags = 1 << 16 /* Instance don't understand PSYNC. */
	ClientReadOnly         ClientFlags = 1 << 17 /* Cluster client is in read-only state. */
	ClientPubSub           ClientFlags = 1 << 18 /* Client is in Pub/Sub mode. */
	ClientPreventAOFProp   ClientFlags = 1 << 19 /* Don't propagate to AOF. */
	ClientPreventReplProp  ClientFlags = 1 << 20 /* Don't propagate to slaves. */
	ClientPreventProp      ClientFlags = ClientPreventAOFProp | ClientPreventReplProp
	ClientPendingWrite     ClientFlags = 1 << 21 /* Client has output to send but a-write handler is yet not installed. */
	ClientReplyOff         ClientFlags = 1 << 22 /* Don't send replies to client. */
	ClientReplySkipNext    ClientFlags = 1 << 23 /* Set ClientREPLY_SKIP for next cmd */
	ClientReplySkip        ClientFlags = 1 << 24 /* Don't send just this reply. */
	ClientLuaDebug         ClientFlags = 1 << 25 /* Run EVAL in debug mode. */
	ClientLuaDebugSync     ClientFlags = 1 << 26 /* EVAL debugging without fork() */
	ClientModule           ClientFlags = 1 << 27 /* Non connected client used by some module. */
	ClientProtected        ClientFlags = 1 << 28 /* Client should not be freed for now. */
	ClientExecutingCommand ClientFlags = 1 << 29 /* Indicates that the client is currently in the process of handling
	   a command. usually this will be marked only during call()
	   however, blocked clients might have this flag kept until they
	   will try to reprocess the command. */
	ClientPendingCommand      ClientFlags = 1 << 30 /* Indicates the client has a fully * parsed command ready for execution. */
	ClientTracking            ClientFlags = 1 << 31 /* Client enabled keys tracking in order to perform client side caching. */
	ClientTrackingBrokenRedir ClientFlags = 1 << 32 /* Target client is invalid. */
	ClientTrackingBCAST       ClientFlags = 1 << 33 /* Tracking in BCAST mode. */
	ClientTrackingOptIn       ClientFlags = 1 << 34 /* Tracking in opt-in mode. */
	ClientTrackingOptOut      ClientFlags = 1 << 35 /* Tracking in opt-out mode. */
	ClientTrackingCaching     ClientFlags = 1 << 36 /* CACHING yes/no was given, depending on optin/optout mode. */
	ClientTrackingNoLoop      ClientFlags = 1 << 37 /* Don't send invalidation messages about writes performed by myself.*/
	ClientInTimeoutTable      ClientFlags = 1 << 38 /* This client is in the timeout table. */
	ClientProtocolError       ClientFlags = 1 << 39 /* Protocol error chatting with it. */
	ClientCloseAfterCommand   ClientFlags = 1 << 40 /* Close after executing commands * and writing entire reply. */
	ClientDenyBlocking        ClientFlags = 1 << 41 /* Indicate that the client should not be blocked. currently, turned on inside MULTI, Lua, RM_Call, and AOF client */
	ClientReplRDBOnly         ClientFlags = 1 << 42 /* This client is a replica that only wants RDB without replication buffer. */
	ClientNoEvict             ClientFlags = 1 << 43 /* This client is protected against client memory eviction. */
	ClientAllowOOM            ClientFlags = 1 << 44 /* Client used by RM_Call is allowed to fully execute scripts even when in OOM */
	ClientNoTouch             ClientFlags = 1 << 45 /* This client will not touch LFU/LRU stats. */
	ClientPushing             ClientFlags = 1 << 46 /* This client is pushing notifications. */
)

// ClientInfo is redis-server ClientInfo, not go-redis *Client
type ClientInfo struct {
	ID                 int64         // redis version 2.8.12, a unique 64-bit client ID
	Addr               string        // address/port of the client
	LAddr              string        // address/port of local address client connected to (bind address)
	FD                 int64         // file descriptor corresponding to the socket
	Name               string        // the name set by the client with CLIENT SETNAME
	Age                time.Duration // total duration of the connection in seconds
	Idle               time.Duration // idle time of the connection in seconds
	Flags              ClientFlags   // client flags (see below)
	DB                 int           // current database ID
	Sub                int           // number of channel subscriptions
	PSub               int           // number of pattern matching subscriptions
	SSub               int           // redis version 7.0.3, number of shard channel subscriptions
	Multi              int           // number of commands in a MULTI/EXEC context
	Watch              int           // redis version 7.4 RC1, number of keys this client is currently watching.
	QueryBuf           int           // qbuf, query buffer length (0 means no query pending)
	QueryBufFree       int           // qbuf-free, free space of the query buffer (0 means the buffer is full)
	ArgvMem            int           // incomplete arguments for the next command (already extracted from query buffer)
	MultiMem           int           // redis version 7.0, memory is used up by buffered multi commands
	BufferSize         int           // rbs, usable size of buffer
	BufferPeak         int           // rbp, peak used size of buffer in last 5 sec interval
	OutputBufferLength int           // obl, output buffer length
	OutputListLength   int           // oll, output list length (replies are queued in this list when the buffer is full)
	OutputMemory       int           // omem, output buffer memory usage
	TotalMemory        int           // tot-mem, total memory consumed by this client in its various buffers
	TotalNetIn         int           // tot-net-in, total network input
	TotalNetOut        int           // tot-net-out, total network output
	TotalCmds          int           // tot-cmds, total number of commands processed
	IoThread           int           // io-thread id
	Events             string        // file descriptor events (see below)
	LastCmd            string        // cmd, last command played
	User               string        // the authenticated username of the client
	Redir              int64         // client id of current client tracking redirection
	Resp               int           // redis version 7.0, client RESP protocol version
	LibName            string        // redis version 7.2, client library name
	LibVer             string        // redis version 7.2, client library version
	ReadEvents         uint64        // redis version 8.8, number of read events processed
	AvgPipelineLenSum  uint64        // redis version 8.8, sum of pipeline lengths
	AvgPipelineLenCnt  uint64        // redis version 8.8, count of pipeline operations
}

type ClientInfoCmd struct {
	baseCmd

	val *ClientInfo
}

var _ Cmder = (*ClientInfoCmd)(nil)

func NewClientInfoCmd(ctx context.Context, args ...interface{}) *ClientInfoCmd {
	return &ClientInfoCmd{
		baseCmd: baseCmd{
			ctx:     ctx,
			args:    args,
			cmdType: CmdTypeClientInfo,
		},
	}
}

func (cmd *ClientInfoCmd) SetVal(val *ClientInfo) {
	cmd.val = val
}

func (cmd *ClientInfoCmd) String() string {
	cmd.await()
	return cmdString(cmd, cmd.val)
}

func (cmd *ClientInfoCmd) Val() *ClientInfo {
	cmd.await()
	return cmd.val
}

func (cmd *ClientInfoCmd) Result() (*ClientInfo, error) {
	cmd.await()
	return cmd.val, cmd.err
}

func (cmd *ClientInfoCmd) readReply(rd *proto.Reader) (err error) {
	txt, err := rd.ReadString()
	if err != nil {
		return err
	}

	// sds o = catClientInfoString(sdsempty(), c);
	// o = sdscatlen(o,"\n",1);
	// addReplyVerbatim(c,o,sdslen(o),"txt");
	// sdsfree(o);
	cmd.val, err = parseClientInfo(strings.TrimSpace(txt))
	return err
}

// fmt.Sscanf() cannot handle null values
func parseClientInfo(txt string) (info *ClientInfo, err error) {
	info = &ClientInfo{}
	for _, s := range strings.Split(txt, " ") {
		kv := strings.Split(s, "=")
		if len(kv) != 2 {
			return nil, fmt.Errorf("redis: unexpected client info data (%s)", s)
		}
		key, val := kv[0], kv[1]

		switch key {
		case "id":
			info.ID, err = strconv.ParseInt(val, 10, 64)
		case "addr":
			info.Addr = val
		case "laddr":
			info.LAddr = val
		case "fd":
			info.FD, err = strconv.ParseInt(val, 10, 64)
		case "name":
			info.Name = val
		case "age":
			var age int
			if age, err = strconv.Atoi(val); err == nil {
				info.Age = time.Duration(age) * time.Second
			}
		case "idle":
			var idle int
			if idle, err = strconv.Atoi(val); err == nil {
				info.Idle = time.Duration(idle) * time.Second
			}
		case "flags":
			if val == "N" {
				break
			}

			for i := 0; i < len(val); i++ {
				switch val[i] {
				case 'S':
					info.Flags |= ClientSlave
				case 'O':
					info.Flags |= ClientSlave | ClientMonitor
				case 'M':
					info.Flags |= ClientMaster
				case 'P':
					info.Flags |= ClientPubSub
				case 'x':
					info.Flags |= ClientMulti
				case 'b':
					info.Flags |= ClientBlocked
				case 't':
					info.Flags |= ClientTracking
				case 'R':
					info.Flags |= ClientTrackingBrokenRedir
				case 'B':
					info.Flags |= ClientTrackingBCAST
				case 'd':
					info.Flags |= ClientDirtyCAS
				case 'c':
					info.Flags |= ClientCloseAfterCommand
				case 'u':
					info.Flags |= ClientUnBlocked
				case 'A':
					info.Flags |= ClientCloseASAP
				case 'U':
					info.Flags |= ClientUnixSocket
				case 'r':
					info.Flags |= ClientReadOnly
				case 'e':
					info.Flags |= ClientNoEvict
				case 'T':
					info.Flags |= ClientNoTouch
				default:
					return nil, fmt.Errorf("redis: unexpected client info flags(%s)", string(val[i]))
				}
			}
		case "db":
			info.DB, err = strconv.Atoi(val)
		case "sub":
			info.Sub, err = strconv.Atoi(val)
		case "psub":
			info.PSub, err = strconv.Atoi(val)
		case "ssub":
			info.SSub, err = strconv.Atoi(val)
		case "multi":
			info.Multi, err = strconv.Atoi(val)
		case "watch":
			info.Watch, err = strconv.Atoi(val)
		case "qbuf":
			info.QueryBuf, err = strconv.Atoi(val)
		case "qbuf-free":
			info.QueryBufFree, err = strconv.Atoi(val)
		case "argv-mem":
			info.ArgvMem, err = strconv.Atoi(val)
		case "multi-mem":
			info.MultiMem, err = strconv.Atoi(val)
		case "rbs":
			info.BufferSize, err = strconv.Atoi(val)
		case "rbp":
			info.BufferPeak, err = strconv.Atoi(val)
		case "obl":
			info.OutputBufferLength, err = strconv.Atoi(val)
		case "oll":
			info.OutputListLength, err = strconv.Atoi(val)
		case "omem":
			info.OutputMemory, err = strconv.Atoi(val)
		case "tot-mem":
			info.TotalMemory, err = strconv.Atoi(val)
		case "tot-net-in":
			info.TotalNetIn, err = strconv.Atoi(val)
		case "tot-net-out":
			info.TotalNetOut, err = strconv.Atoi(val)
		case "tot-cmds":
			info.TotalCmds, err = strconv.Atoi(val)
		case "events":
			info.Events = val
		case "cmd":
			info.LastCmd = val
		case "user":
			info.User = val
		case "redir":
			info.Redir, err = strconv.ParseInt(val, 10, 64)
		case "resp":
			info.Resp, err = strconv.Atoi(val)
		case "lib-name":
			info.LibName = val
		case "lib-ver":
			info.LibVer = val
		case "io-thread":
			info.IoThread, err = strconv.Atoi(val)
		case "read-events":
			info.ReadEvents, err = strconv.ParseUint(val, 10, 64)
		case "avg-pipeline-len-sum":
			info.AvgPipelineLenSum, err = strconv.ParseUint(val, 10, 64)
		case "avg-pipeline-len-cnt":
			info.AvgPipelineLenCnt, err = strconv.ParseUint(val, 10, 64)
		default:
			// skip unknown fields
		}

		if err != nil {
			return nil, err
		}
	}

	return info, nil
}

func (cmd *ClientInfoCmd) Clone() Cmder {
	var val *ClientInfo
	if cmd.val != nil {
		val = &ClientInfo{
			ID:                 cmd.val.ID,
			Addr:               cmd.val.Addr,
			LAddr:              cmd.val.LAddr,
			FD:                 cmd.val.FD,
			Name:               cmd.val.Name,
			Age:                cmd.val.Age,
			Idle:               cmd.val.Idle,
			Flags:              cmd.val.Flags,
			DB:                 cmd.val.DB,
			Sub:                cmd.val.Sub,
			PSub:               cmd.val.PSub,
			SSub:               cmd.val.SSub,
			Multi:              cmd.val.Multi,
			Watch:              cmd.val.Watch,
			QueryBuf:           cmd.val.QueryBuf,
			QueryBufFree:       cmd.val.QueryBufFree,
			ArgvMem:            cmd.val.ArgvMem,
			MultiMem:           cmd.val.MultiMem,
			BufferSize:         cmd.val.BufferSize,
			BufferPeak:         cmd.val.BufferPeak,
			OutputBufferLength: cmd.val.OutputBufferLength,
			OutputListLength:   cmd.val.OutputListLength,
			OutputMemory:       cmd.val.OutputMemory,
			TotalMemory:        cmd.val.TotalMemory,
			IoThread:           cmd.val.IoThread,
			Events:             cmd.val.Events,
			LastCmd:            cmd.val.LastCmd,
			User:               cmd.val.User,
			Redir:              cmd.val.Redir,
			Resp:               cmd.val.Resp,
			LibName:            cmd.val.LibName,
			LibVer:             cmd.val.LibVer,
			ReadEvents:         cmd.val.ReadEvents,
			AvgPipelineLenSum:  cmd.val.AvgPipelineLenSum,
			AvgPipelineLenCnt:  cmd.val.AvgPipelineLenCnt,
		}
	}
	return &ClientInfoCmd{
		baseCmd: cmd.cloneBaseCmd(),
		val:     val,
	}
}

// -------------------------------------------

type ACLLogEntry struct {
	Count                int64
	Reason               string
	Context              string
	Object               string
	Username             string
	AgeSeconds           float64
	ClientInfo           *ClientInfo
	EntryID              int64
	TimestampCreated     int64
	TimestampLastUpdated int64
}

type ACLLogCmd struct {
	baseCmd

	val []*ACLLogEntry
}

var _ Cmder = (*ACLLogCmd)(nil)

func NewACLLogCmd(ctx context.Context, args ...interface{}) *ACLLogCmd {
	return &ACLLogCmd{
		baseCmd: baseCmd{
			ctx:     ctx,
			args:    args,
			cmdType: CmdTypeACLLog,
		},
	}
}

func (cmd *ACLLogCmd) SetVal(val []*ACLLogEntry) {
	cmd.val = val
}

func (cmd *ACLLogCmd) Val() []*ACLLogEntry {
	cmd.await()
	return cmd.val
}

func (cmd *ACLLogCmd) Result() ([]*ACLLogEntry, error) {
	cmd.await()
	return cmd.val, cmd.err
}

func (cmd *ACLLogCmd) String() string {
	cmd.await()
	return cmdString(cmd, cmd.val)
}

func (cmd *ACLLogCmd) readReply(rd *proto.Reader) error {
	n, err := rd.ReadArrayLen()
	if err != nil {
		return err
	}

	cmd.val = make([]*ACLLogEntry, n)
	for i := 0; i < n; i++ {
		cmd.val[i] = &ACLLogEntry{}
		entry := cmd.val[i]
		respLen, err := rd.ReadMapLen()
		if err != nil {
			return err
		}
		for j := 0; j < respLen; j++ {
			key, err := rd.ReadString()
			if err != nil {
				return err
			}

			switch key {
			case "count":
				entry.Count, err = rd.ReadInt()
			case "reason":
				entry.Reason, err = rd.ReadString()
			case "context":
				entry.Context, err = rd.ReadString()
			case "object":
				entry.Object, err = rd.ReadString()
			case "username":
				entry.Username, err = rd.ReadString()
			case "age-seconds":
				entry.AgeSeconds, err = rd.ReadFloat()
			case "client-info":
				txt, err := rd.ReadString()
				if err != nil {
					return err
				}
				entry.ClientInfo, err = parseClientInfo(strings.TrimSpace(txt))
				if err != nil {
					return err
				}
			case "entry-id":
				entry.EntryID, err = rd.ReadInt()
			case "timestamp-created":
				entry.TimestampCreated, err = rd.ReadInt()
			case "timestamp-last-updated":
				entry.TimestampLastUpdated, err = rd.ReadInt()
			default:
				// skip unknown fields
				if err := rd.DiscardNext(); err != nil {
					return err
				}
			}

			if err != nil {
				return err
			}
		}
	}

	return nil
}

func (cmd *ACLLogCmd) Clone() Cmder {
	var val []*ACLLogEntry
	if cmd.val != nil {
		val = make([]*ACLLogEntry, len(cmd.val))
		for i, entry := range cmd.val {
			if entry != nil {
				val[i] = &ACLLogEntry{
					Count:                entry.Count,
					Reason:               entry.Reason,
					Context:              entry.Context,
					Object:               entry.Object,
					Username:             entry.Username,
					AgeSeconds:           entry.AgeSeconds,
					EntryID:              entry.EntryID,
					TimestampCreated:     entry.TimestampCreated,
					TimestampLastUpdated: entry.TimestampLastUpdated,
				}
				// Clone ClientInfo if present
				if entry.ClientInfo != nil {
					val[i].ClientInfo = &ClientInfo{
						ID:                 entry.ClientInfo.ID,
						Addr:               entry.ClientInfo.Addr,
						LAddr:              entry.ClientInfo.LAddr,
						FD:                 entry.ClientInfo.FD,
						Name:               entry.ClientInfo.Name,
						Age:                entry.ClientInfo.Age,
						Idle:               entry.ClientInfo.Idle,
						Flags:              entry.ClientInfo.Flags,
						DB:                 entry.ClientInfo.DB,
						Sub:                entry.ClientInfo.Sub,
						PSub:               entry.ClientInfo.PSub,
						SSub:               entry.ClientInfo.SSub,
						Multi:              entry.ClientInfo.Multi,
						Watch:              entry.ClientInfo.Watch,
						QueryBuf:           entry.ClientInfo.QueryBuf,
						QueryBufFree:       entry.ClientInfo.QueryBufFree,
						ArgvMem:            entry.ClientInfo.ArgvMem,
						MultiMem:           entry.ClientInfo.MultiMem,
						BufferSize:         entry.ClientInfo.BufferSize,
						BufferPeak:         entry.ClientInfo.BufferPeak,
						OutputBufferLength: entry.ClientInfo.OutputBufferLength,
						OutputListLength:   entry.ClientInfo.OutputListLength,
						OutputMemory:       entry.ClientInfo.OutputMemory,
						TotalMemory:        entry.ClientInfo.TotalMemory,
						IoThread:           entry.ClientInfo.IoThread,
						Events:             entry.ClientInfo.Events,
						LastCmd:            entry.ClientInfo.LastCmd,
						User:               entry.ClientInfo.User,
						Redir:              entry.ClientInfo.Redir,
						Resp:               entry.ClientInfo.Resp,
						LibName:            entry.ClientInfo.LibName,
						LibVer:             entry.ClientInfo.LibVer,
						ReadEvents:         entry.ClientInfo.ReadEvents,
						AvgPipelineLenSum:  entry.ClientInfo.AvgPipelineLenSum,
						AvgPipelineLenCnt:  entry.ClientInfo.AvgPipelineLenCnt,
					}
				}
			}
		}
	}
	return &ACLLogCmd{
		baseCmd: cmd.cloneBaseCmd(),
		val:     val,
	}
}

// LibraryInfo holds the library info.
type LibraryInfo struct {
	LibName *string
	LibVer  *string
}

// WithLibraryName returns a valid LibraryInfo with library name only.
func WithLibraryName(libName string) LibraryInfo {
	return LibraryInfo{LibName: &libName}
}

// WithLibraryVersion returns a valid LibraryInfo with library version only.
func WithLibraryVersion(libVer string) LibraryInfo {
	return LibraryInfo{LibVer: &libVer}
}

// -------------------------------------------

type InfoCmd struct {
	baseCmd
	val map[string]map[string]string
}

var _ Cmder = (*InfoCmd)(nil)

func NewInfoCmd(ctx context.Context, args ...interface{}) *InfoCmd {
	return &InfoCmd{
		baseCmd: baseCmd{
			ctx:     ctx,
			args:    args,
			cmdType: CmdTypeInfo,
		},
	}
}

func (cmd *InfoCmd) SetVal(val map[string]map[string]string) {
	cmd.val = val
}

func (cmd *InfoCmd) Val() map[string]map[string]string {
	cmd.await()
	return cmd.val
}

func (cmd *InfoCmd) Result() (map[string]map[string]string, error) {
	cmd.await()
	return cmd.val, cmd.err
}

func (cmd *InfoCmd) String() string {
	cmd.await()
	return cmdString(cmd, cmd.val)
}

func (cmd *InfoCmd) readReply(rd *proto.Reader) error {
	val, err := rd.ReadString()
	if err != nil {
		return err
	}

	section := ""
	scanner := bufio.NewScanner(strings.NewReader(val))
	for scanner.Scan() {
		line := scanner.Text()
		if strings.HasPrefix(line, "#") {
			if cmd.val == nil {
				cmd.val = make(map[string]map[string]string)
			}
			section = strings.TrimPrefix(line, "# ")
			cmd.val[section] = make(map[string]string)
		} else if line != "" {
			if section == "Modules" {
				moduleRe := regexp.MustCompile(`module:name=(.+?),(.+)$`)
				kv := moduleRe.FindStringSubmatch(line)
				if len(kv) == 3 {
					cmd.val[section][kv[1]] = kv[2]
				}
			} else {
				kv := strings.SplitN(line, ":", 2)
				if len(kv) == 2 {
					cmd.val[section][kv[0]] = kv[1]
				}
			}
		}
	}

	return nil
}

func (cmd *InfoCmd) Item(section, key string) string {
	cmd.await()
	if cmd.val == nil {
		return ""
	} else if cmd.val[section] == nil {
		return ""
	} else {
		return cmd.val[section][key]
	}
}

func (cmd *InfoCmd) Clone() Cmder {
	var val map[string]map[string]string
	if cmd.val != nil {
		val = make(map[string]map[string]string, len(cmd.val))
		for section, sectionMap := range cmd.val {
			if sectionMap != nil {
				val[section] = make(map[string]string, len(sectionMap))
				for k, v := range sectionMap {
					val[section][k] = v
				}
			}
		}
	}
	return &InfoCmd{
		baseCmd: cmd.cloneBaseCmd(),
		val:     val,
	}
}

type MonitorStatus int

const (
	monitorStatusIdle MonitorStatus = iota
	monitorStatusStart
	monitorStatusStop
)

type MonitorCmd struct {
	baseCmd
	ch     chan string
	status MonitorStatus
	mu     sync.Mutex
}

func newMonitorCmd(ctx context.Context, ch chan string) *MonitorCmd {
	return &MonitorCmd{
		baseCmd: baseCmd{
			ctx:     ctx,
			args:    []interface{}{"monitor"},
			cmdType: CmdTypeMonitor,
		},
		ch:     ch,
		status: monitorStatusIdle,
		mu:     sync.Mutex{},
	}
}

func (cmd *MonitorCmd) String() string {
	cmd.await()
	return cmdString(cmd, nil)
}

func (cmd *MonitorCmd) readReply(rd *proto.Reader) error {
	ctx, cancel := context.WithCancel(cmd.ctx)
	go func(ctx context.Context) {
		for {
			select {
			case <-ctx.Done():
				return
			default:
				err := cmd.readMonitor(rd, cancel)
				if err != nil {
					cmd.err = err
					return
				}
			}
		}
	}(ctx)
	return nil
}

func (cmd *MonitorCmd) readMonitor(rd *proto.Reader, cancel context.CancelFunc) error {
	for {
		cmd.mu.Lock()
		st := cmd.status
		pk, _ := rd.Peek(1)
		cmd.mu.Unlock()
		if len(pk) != 0 && st == monitorStatusStart {
			cmd.mu.Lock()
			line, err := rd.ReadString()
			cmd.mu.Unlock()
			if err != nil {
				return err
			}
			cmd.ch <- line
		}
		if st == monitorStatusStop {
			cancel()
			break
		}
	}
	return nil
}

func (cmd *MonitorCmd) Start() {
	cmd.mu.Lock()
	defer cmd.mu.Unlock()
	cmd.status = monitorStatusStart
}

func (cmd *MonitorCmd) Stop() {
	cmd.mu.Lock()
	defer cmd.mu.Unlock()
	cmd.status = monitorStatusStop
}

type VectorScoreSliceCmd struct {
	baseCmd

	val []VectorScore
}

var _ Cmder = (*VectorScoreSliceCmd)(nil)

func NewVectorScoreSliceCmd(ctx context.Context, args ...any) *VectorScoreSliceCmd {
	return &VectorScoreSliceCmd{
		baseCmd: baseCmd{
			ctx:  ctx,
			args: args,
		},
	}
}

// NewVectorInfoSliceCmd is an alias for NewVectorScoreSliceCmd kept for backwards compatibility.
func NewVectorInfoSliceCmd(ctx context.Context, args ...any) *VectorScoreSliceCmd {
	return NewVectorScoreSliceCmd(ctx, args...)
}

func (cmd *VectorScoreSliceCmd) SetVal(val []VectorScore) {
	cmd.val = val
}

func (cmd *VectorScoreSliceCmd) Val() []VectorScore {
	cmd.await()
	return cmd.val
}

func (cmd *VectorScoreSliceCmd) Result() ([]VectorScore, error) {
	cmd.await()
	return cmd.val, cmd.err
}

func (cmd *VectorScoreSliceCmd) String() string {
	cmd.await()
	return cmdString(cmd, cmd.val)
}

func (cmd *VectorScoreSliceCmd) readReply(rd *proto.Reader) error {
	typ, err := rd.PeekReplyType()
	if err != nil {
		return err
	}

	var n int
	if typ == proto.RespMap {
		n, err = rd.ReadMapLen()
		if err != nil {
			return err
		}
	} else {
		// RESP2 returns a flat array [name, score, name, score, ...]
		n, err = rd.ReadArrayLen()
		if err != nil {
			return err
		}
		if n%2 != 0 {
			return fmt.Errorf("redis: VectorScoreSliceCmd expects even number of elements, got %d", n)
		}
		n /= 2
	}

	cmd.val = make([]VectorScore, n)
	for i := 0; i < n; i++ {
		name, err := rd.ReadString()
		if err != nil {
			return err
		}
		cmd.val[i].Name = name

		score, err := rd.ReadFloat()
		if err != nil {
			return err
		}
		cmd.val[i].Score = score
	}

	return nil
}

func (cmd *VectorScoreSliceCmd) Clone() Cmder {
	return &VectorScoreSliceCmd{
		baseCmd: cmd.cloneBaseCmd(),
		val:     cmd.val,
	}
}

// VectorScoreSliceSliceCmd is used for VLINKS WITHSCORES which returns an array of arrays.
// In RESP3, each inner array contains maps of element -> score.
type VectorScoreSliceSliceCmd struct {
	baseCmd

	val [][]VectorScore
}

var _ Cmder = (*VectorScoreSliceSliceCmd)(nil)

func NewVectorScoreSliceSliceCmd(ctx context.Context, args ...any) *VectorScoreSliceSliceCmd {
	return &VectorScoreSliceSliceCmd{
		baseCmd: baseCmd{
			ctx:  ctx,
			args: args,
		},
	}
}

func (cmd *VectorScoreSliceSliceCmd) SetVal(val [][]VectorScore) {
	cmd.val = val
}

func (cmd *VectorScoreSliceSliceCmd) Val() [][]VectorScore {
	cmd.await()
	return cmd.val
}

func (cmd *VectorScoreSliceSliceCmd) Result() ([][]VectorScore, error) {
	cmd.await()
	return cmd.val, cmd.err
}

func (cmd *VectorScoreSliceSliceCmd) String() string {
	cmd.await()
	return cmdString(cmd, cmd.val)
}

func (cmd *VectorScoreSliceSliceCmd) readReply(rd *proto.Reader) error {
	n, err := rd.ReadArrayLen()
	if err != nil {
		return err
	}

	cmd.val = make([][]VectorScore, n)
	for i := range n {
		// Each level can be either a map (RESP3) or an array (RESP2)
		levelTyp, err := rd.PeekReplyType()
		if err != nil {
			return err
		}

		if levelTyp == proto.RespMap {
			// RESP3 format: each level is a map {element: score, element: score, ...}
			mapLen, err := rd.ReadMapLen()
			if err != nil {
				return err
			}

			cmd.val[i] = make([]VectorScore, mapLen)
			for j := range mapLen {
				name, err := rd.ReadString()
				if err != nil {
					return err
				}
				score, err := rd.ReadFloat()
				if err != nil {
					return err
				}
				cmd.val[i][j] = VectorScore{Name: name, Score: score}
			}
		} else {
			// RESP2 format: each level is an array of [element, score, element, score, ...] pairs
			innerLen, err := rd.ReadArrayLen()
			if err != nil {
				return err
			}

			if innerLen%2 != 0 {
				return fmt.Errorf("redis: got %d elements in the VLINKS array, wanted a multiple of 2", innerLen)
			}

			cmd.val[i] = make([]VectorScore, innerLen/2)
			for j := 0; j < innerLen; j += 2 {
				name, err := rd.ReadString()
				if err != nil {
					return err
				}
				score, err := rd.ReadFloat()
				if err != nil {
					return err
				}
				cmd.val[i][j/2] = VectorScore{Name: name, Score: score}
			}
		}
	}

	return nil
}

func (cmd *VectorScoreSliceSliceCmd) Clone() Cmder {
	var val [][]VectorScore
	if cmd.val != nil {
		val = make([][]VectorScore, len(cmd.val))
		for i, slice := range cmd.val {
			if slice != nil {
				val[i] = make([]VectorScore, len(slice))
				copy(val[i], slice)
			}
		}
	}
	return &VectorScoreSliceSliceCmd{
		baseCmd: cmd.cloneBaseCmd(),
		val:     val,
	}
}

func readVectorAttribStringOrNil(rd *proto.Reader) (*string, error) {
	v, err := rd.ReadReply()
	if err != nil {
		if err == proto.Nil {
			return nil, nil
		}
		return nil, err
	}
	s, ok := v.(string)
	if !ok {
		return nil, fmt.Errorf("redis: can't parse reply=%T reading string", v)
	}
	return &s, nil
}

type VectorAttribSliceCmd struct {
	baseCmd

	val []VectorAttrib
}

var _ Cmder = (*VectorAttribSliceCmd)(nil)

func NewVectorAttribSliceCmd(ctx context.Context, args ...any) *VectorAttribSliceCmd {
	return &VectorAttribSliceCmd{
		baseCmd: baseCmd{
			ctx:  ctx,
			args: args,
		},
	}
}

func (cmd *VectorAttribSliceCmd) SetVal(val []VectorAttrib) {
	cmd.val = val
}

func (cmd *VectorAttribSliceCmd) Val() []VectorAttrib {
	cmd.await()
	return cmd.val
}

func (cmd *VectorAttribSliceCmd) Result() ([]VectorAttrib, error) {
	cmd.await()
	return cmd.val, cmd.err
}

func (cmd *VectorAttribSliceCmd) String() string {
	cmd.await()
	return cmdString(cmd, cmd.val)
}

func (cmd *VectorAttribSliceCmd) readReply(rd *proto.Reader) error {
	replyType, err := rd.PeekReplyType()
	if err != nil {
		return err
	}

	if replyType == proto.RespMap {
		n, err := rd.ReadMapLen()
		if err != nil {
			return err
		}
		cmd.val = make([]VectorAttrib, n)
		for i := 0; i < n; i++ {
			name, err := rd.ReadString()
			if err != nil {
				return err
			}
			attrib, err := readVectorAttribStringOrNil(rd)
			if err != nil {
				return err
			}
			cmd.val[i] = VectorAttrib{Name: name, Attribs: attrib}
		}
		return nil
	}

	n, err := rd.ReadArrayLen()
	if err != nil {
		return err
	}
	if n%2 != 0 {
		return fmt.Errorf("redis: got %d elements in the VSIM array, wanted a multiple of 2", n)
	}
	cmd.val = make([]VectorAttrib, n/2)
	for i := range cmd.val {
		name, err := rd.ReadString()
		if err != nil {
			return err
		}
		attrib, err := readVectorAttribStringOrNil(rd)
		if err != nil {
			return err
		}
		cmd.val[i] = VectorAttrib{Name: name, Attribs: attrib}
	}
	return nil
}

func (cmd *VectorAttribSliceCmd) Clone() Cmder {
	return &VectorAttribSliceCmd{
		baseCmd: cmd.cloneBaseCmd(),
		val:     cmd.val,
	}
}

type VectorScoreAttribSliceCmd struct {
	baseCmd

	val []VectorScoreAttrib
}

var _ Cmder = (*VectorScoreAttribSliceCmd)(nil)

func NewVectorScoreAttribSliceCmd(ctx context.Context, args ...any) *VectorScoreAttribSliceCmd {
	return &VectorScoreAttribSliceCmd{
		baseCmd: baseCmd{
			ctx:  ctx,
			args: args,
		},
	}
}

func (cmd *VectorScoreAttribSliceCmd) SetVal(val []VectorScoreAttrib) {
	cmd.val = val
}

func (cmd *VectorScoreAttribSliceCmd) Val() []VectorScoreAttrib {
	cmd.await()
	return cmd.val
}

func (cmd *VectorScoreAttribSliceCmd) Result() ([]VectorScoreAttrib, error) {
	cmd.await()
	return cmd.val, cmd.err
}

func (cmd *VectorScoreAttribSliceCmd) String() string {
	cmd.await()
	return cmdString(cmd, cmd.val)
}

func (cmd *VectorScoreAttribSliceCmd) readReply(rd *proto.Reader) error {
	replyType, err := rd.PeekReplyType()
	if err != nil {
		return err
	}

	if replyType == proto.RespMap {
		n, err := rd.ReadMapLen()
		if err != nil {
			return err
		}
		cmd.val = make([]VectorScoreAttrib, n)
		for i := 0; i < n; i++ {
			name, err := rd.ReadString()
			if err != nil {
				return err
			}
			if err := rd.ReadFixedArrayLen(2); err != nil {
				return err
			}
			score, err := rd.ReadFloat()
			if err != nil {
				return err
			}
			attrib, err := readVectorAttribStringOrNil(rd)
			if err != nil {
				return err
			}
			cmd.val[i] = VectorScoreAttrib{Name: name, Score: score, Attribs: attrib}
		}
		return nil
	}

	n, err := rd.ReadArrayLen()
	if err != nil {
		return err
	}
	if n%3 != 0 {
		return fmt.Errorf("redis: got %d elements in the VSIM array, wanted a multiple of 3", n)
	}
	cmd.val = make([]VectorScoreAttrib, n/3)
	for i := range cmd.val {
		name, err := rd.ReadString()
		if err != nil {
			return err
		}
		score, err := rd.ReadFloat()
		if err != nil {
			return err
		}
		attrib, err := readVectorAttribStringOrNil(rd)
		if err != nil {
			return err
		}
		cmd.val[i] = VectorScoreAttrib{Name: name, Score: score, Attribs: attrib}
	}
	return nil
}

func (cmd *VectorScoreAttribSliceCmd) Clone() Cmder {
	return &VectorScoreAttribSliceCmd{
		baseCmd: cmd.cloneBaseCmd(),
		val:     cmd.val,
	}
}

func (cmd *MonitorCmd) Clone() Cmder {
	// MonitorCmd cannot be safely cloned due to channels and goroutines
	// Return a new MonitorCmd with the same channel
	return newMonitorCmd(cmd.ctx, cmd.ch)
}

// ExtractCommandValue extracts the value from a command result using the fast enum-based approach
func ExtractCommandValue(cmd interface{}) (interface{}, error) {
	// First try to get the command type using the interface
	if cmdTypeGetter, ok := cmd.(CmdTypeGetter); ok {
		cmdType := cmdTypeGetter.GetCmdType()

		// Use fast type-based extraction
		switch cmdType {
		case CmdTypeGeneric:
			if genericCmd, ok := cmd.(interface {
				Val() interface{}
				Err() error
			}); ok {
				return genericCmd.Val(), genericCmd.Err()
			}
		case CmdTypeString:
			if stringCmd, ok := cmd.(interface {
				Val() string
				Err() error
			}); ok {
				return stringCmd.Val(), stringCmd.Err()
			}
		case CmdTypeInt:
			if intCmd, ok := cmd.(interface {
				Val() int64
				Err() error
			}); ok {
				return intCmd.Val(), intCmd.Err()
			}
		case CmdTypeUint:
			if uintCmd, ok := cmd.(interface {
				Val() uint64
				Err() error
			}); ok {
				return uintCmd.Val(), uintCmd.Err()
			}
		case CmdTypeBool:
			if boolCmd, ok := cmd.(interface {
				Val() bool
				Err() error
			}); ok {
				return boolCmd.Val(), boolCmd.Err()
			}
		case CmdTypeFloat:
			if floatCmd, ok := cmd.(interface {
				Val() float64
				Err() error
			}); ok {
				return floatCmd.Val(), floatCmd.Err()
			}
		case CmdTypeStatus:
			if statusCmd, ok := cmd.(interface {
				Val() string
				Err() error
			}); ok {
				return statusCmd.Val(), statusCmd.Err()
			}
		case CmdTypeDuration:
			if durationCmd, ok := cmd.(interface {
				Val() time.Duration
				Err() error
			}); ok {
				return durationCmd.Val(), durationCmd.Err()
			}
		case CmdTypeTime:
			if timeCmd, ok := cmd.(interface {
				Val() time.Time
				Err() error
			}); ok {
				return timeCmd.Val(), timeCmd.Err()
			}
		case CmdTypeStringStructMap:
			if structMapCmd, ok := cmd.(interface {
				Val() map[string]struct{}
				Err() error
			}); ok {
				return structMapCmd.Val(), structMapCmd.Err()
			}
		case CmdTypeXMessageSlice:
			if xMessageSliceCmd, ok := cmd.(interface {
				Val() []XMessage
				Err() error
			}); ok {
				return xMessageSliceCmd.Val(), xMessageSliceCmd.Err()
			}
		case CmdTypeXStreamSlice:
			if xStreamSliceCmd, ok := cmd.(interface {
				Val() []XStream
				Err() error
			}); ok {
				return xStreamSliceCmd.Val(), xStreamSliceCmd.Err()
			}
		case CmdTypeXPending:
			if xPendingCmd, ok := cmd.(interface {
				Val() *XPending
				Err() error
			}); ok {
				return xPendingCmd.Val(), xPendingCmd.Err()
			}
		case CmdTypeXPendingExt:
			if xPendingExtCmd, ok := cmd.(interface {
				Val() []XPendingExt
				Err() error
			}); ok {
				return xPendingExtCmd.Val(), xPendingExtCmd.Err()
			}
		case CmdTypeXAutoClaim:
			if xAutoClaimCmd, ok := cmd.(interface {
				Val() ([]XMessage, string)
				Err() error
			}); ok {
				messages, start := xAutoClaimCmd.Val()
				return CmdTypeXAutoClaimValue{messages: messages, start: start}, xAutoClaimCmd.Err()
			}
		case CmdTypeXAutoClaimWithDeleted:
			if xAutoClaimWithDeletedCmd, ok := cmd.(interface {
				Val() ([]XMessage, string, []string)
				Err() error
			}); ok {
				messages, start, deletedIDs := xAutoClaimWithDeletedCmd.Val()
				return CmdTypeXAutoClaimWithDeletedValue{messages: messages, start: start, deletedIDs: deletedIDs}, xAutoClaimWithDeletedCmd.Err()
			}
		case CmdTypeXAutoClaimJustID:
			if xAutoClaimJustIDCmd, ok := cmd.(interface {
				Val() ([]string, string)
				Err() error
			}); ok {
				ids, start := xAutoClaimJustIDCmd.Val()
				return CmdTypeXAutoClaimJustIDValue{ids: ids, start: start}, xAutoClaimJustIDCmd.Err()
			}
		case CmdTypeXInfoConsumers:
			if xInfoConsumersCmd, ok := cmd.(interface {
				Val() []XInfoConsumer
				Err() error
			}); ok {
				return xInfoConsumersCmd.Val(), xInfoConsumersCmd.Err()
			}
		case CmdTypeXInfoGroups:
			if xInfoGroupsCmd, ok := cmd.(interface {
				Val() []XInfoGroup
				Err() error
			}); ok {
				return xInfoGroupsCmd.Val(), xInfoGroupsCmd.Err()
			}
		case CmdTypeXInfoStream:
			if xInfoStreamCmd, ok := cmd.(interface {
				Val() *XInfoStream
				Err() error
			}); ok {
				return xInfoStreamCmd.Val(), xInfoStreamCmd.Err()
			}
		case CmdTypeXInfoStreamFull:
			if xInfoStreamFullCmd, ok := cmd.(interface {
				Val() *XInfoStreamFull
				Err() error
			}); ok {
				return xInfoStreamFullCmd.Val(), xInfoStreamFullCmd.Err()
			}
		case CmdTypeZSlice:
			if zSliceCmd, ok := cmd.(interface {
				Val() []Z
				Err() error
			}); ok {
				return zSliceCmd.Val(), zSliceCmd.Err()
			}
		case CmdTypeZWithKey:
			if zWithKeyCmd, ok := cmd.(interface {
				Val() *ZWithKey
				Err() error
			}); ok {
				return zWithKeyCmd.Val(), zWithKeyCmd.Err()
			}
		case CmdTypeScan:
			if scanCmd, ok := cmd.(interface {
				Val() ([]string, uint64)
				Err() error
			}); ok {
				keys, cursor := scanCmd.Val()
				return CmdTypeScanValue{keys: keys, cursor: cursor}, scanCmd.Err()
			}
		case CmdTypeClusterSlots:
			if clusterSlotsCmd, ok := cmd.(interface {
				Val() []ClusterSlot
				Err() error
			}); ok {
				return clusterSlotsCmd.Val(), clusterSlotsCmd.Err()
			}
		case CmdTypeGeoLocation:
			if geoLocationCmd, ok := cmd.(interface {
				Val() []GeoLocation
				Err() error
			}); ok {
				return geoLocationCmd.Val(), geoLocationCmd.Err()
			}
		case CmdTypeGeoSearchLocation:
			if geoSearchLocationCmd, ok := cmd.(interface {
				Val() []GeoLocation
				Err() error
			}); ok {
				return geoSearchLocationCmd.Val(), geoSearchLocationCmd.Err()
			}
		case CmdTypeGeoPos:
			if geoPosCmd, ok := cmd.(interface {
				Val() []*GeoPos
				Err() error
			}); ok {
				return geoPosCmd.Val(), geoPosCmd.Err()
			}
		case CmdTypeCommandsInfo:
			if commandsInfoCmd, ok := cmd.(interface {
				Val() map[string]*CommandInfo
				Err() error
			}); ok {
				return commandsInfoCmd.Val(), commandsInfoCmd.Err()
			}
		case CmdTypeSlowLog:
			if slowLogCmd, ok := cmd.(interface {
				Val() []SlowLog
				Err() error
			}); ok {
				return slowLogCmd.Val(), slowLogCmd.Err()
			}
		case CmdTypeHotKeys:
			if hotKeysCmd, ok := cmd.(interface {
				Val() *HotKeysResult
				Err() error
			}); ok {
				return hotKeysCmd.Val(), hotKeysCmd.Err()
			}
		case CmdTypeIncrEXInt:
			if incrEXCmd, ok := cmd.(interface {
				Val() IncrEXIntResult
				Err() error
			}); ok {
				return incrEXCmd.Val(), incrEXCmd.Err()
			}
		case CmdTypeIncrEXFloat:
			if incrEXCmd, ok := cmd.(interface {
				Val() IncrEXFloatResult
				Err() error
			}); ok {
				return incrEXCmd.Val(), incrEXCmd.Err()
			}
		case CmdTypeKeyValues:
			if keyValuesCmd, ok := cmd.(interface {
				Val() (string, []string)
				Err() error
			}); ok {
				key, values := keyValuesCmd.Val()
				return CmdTypeKeyValuesValue{key: key, values: values}, keyValuesCmd.Err()
			}
		case CmdTypeZSliceWithKey:
			if zSliceWithKeyCmd, ok := cmd.(interface {
				Val() (string, []Z)
				Err() error
			}); ok {
				key, zSlice := zSliceWithKeyCmd.Val()
				return CmdTypeZSliceWithKeyValue{key: key, zSlice: zSlice}, zSliceWithKeyCmd.Err()
			}
		case CmdTypeFunctionList:
			if functionListCmd, ok := cmd.(interface {
				Val() []Library
				Err() error
			}); ok {
				return functionListCmd.Val(), functionListCmd.Err()
			}
		case CmdTypeFunctionStats:
			if functionStatsCmd, ok := cmd.(interface {
				Val() FunctionStats
				Err() error
			}); ok {
				return functionStatsCmd.Val(), functionStatsCmd.Err()
			}
		case CmdTypeLCS:
			if lcsCmd, ok := cmd.(interface {
				Val() *LCSMatch
				Err() error
			}); ok {
				return lcsCmd.Val(), lcsCmd.Err()
			}
		case CmdTypeKeyFlags:
			if keyFlagsCmd, ok := cmd.(interface {
				Val() []KeyFlags
				Err() error
			}); ok {
				return keyFlagsCmd.Val(), keyFlagsCmd.Err()
			}
		case CmdTypeClusterLinks:
			if clusterLinksCmd, ok := cmd.(interface {
				Val() []ClusterLink
				Err() error
			}); ok {
				return clusterLinksCmd.Val(), clusterLinksCmd.Err()
			}
		case CmdTypeClusterShards:
			if clusterShardsCmd, ok := cmd.(interface {
				Val() []ClusterShard
				Err() error
			}); ok {
				return clusterShardsCmd.Val(), clusterShardsCmd.Err()
			}
		case CmdTypeRankWithScore:
			if rankWithScoreCmd, ok := cmd.(interface {
				Val() RankScore
				Err() error
			}); ok {
				return rankWithScoreCmd.Val(), rankWithScoreCmd.Err()
			}
		case CmdTypeClientInfo:
			if clientInfoCmd, ok := cmd.(interface {
				Val() *ClientInfo
				Err() error
			}); ok {
				return clientInfoCmd.Val(), clientInfoCmd.Err()
			}
		case CmdTypeACLLog:
			if aclLogCmd, ok := cmd.(interface {
				Val() []*ACLLogEntry
				Err() error
			}); ok {
				return aclLogCmd.Val(), aclLogCmd.Err()
			}
		case CmdTypeInfo:
			if infoCmd, ok := cmd.(interface {
				Val() string
				Err() error
			}); ok {
				return infoCmd.Val(), infoCmd.Err()
			}
		case CmdTypeMonitor:
			if monitorCmd, ok := cmd.(interface {
				Val() string
				Err() error
			}); ok {
				return monitorCmd.Val(), monitorCmd.Err()
			}
		case CmdTypeJSON:
			if jsonCmd, ok := cmd.(interface {
				Val() string
				Err() error
			}); ok {
				return jsonCmd.Val(), jsonCmd.Err()
			}
		case CmdTypeJSONSlice:
			if jsonSliceCmd, ok := cmd.(interface {
				Val() []interface{}
				Err() error
			}); ok {
				return jsonSliceCmd.Val(), jsonSliceCmd.Err()
			}
		case CmdTypeIntPointerSlice:
			if intPointerSliceCmd, ok := cmd.(interface {
				Val() []*int64
				Err() error
			}); ok {
				return intPointerSliceCmd.Val(), intPointerSliceCmd.Err()
			}
		case CmdTypeScanDump:
			if scanDumpCmd, ok := cmd.(interface {
				Val() ScanDump
				Err() error
			}); ok {
				return scanDumpCmd.Val(), scanDumpCmd.Err()
			}
		case CmdTypeBFInfo:
			if bfInfoCmd, ok := cmd.(interface {
				Val() BFInfo
				Err() error
			}); ok {
				return bfInfoCmd.Val(), bfInfoCmd.Err()
			}
		case CmdTypeCFInfo:
			if cfInfoCmd, ok := cmd.(interface {
				Val() CFInfo
				Err() error
			}); ok {
				return cfInfoCmd.Val(), cfInfoCmd.Err()
			}
		case CmdTypeCMSInfo:
			if cmsInfoCmd, ok := cmd.(interface {
				Val() CMSInfo
				Err() error
			}); ok {
				return cmsInfoCmd.Val(), cmsInfoCmd.Err()
			}
		case CmdTypeTopKInfo:
			if topKInfoCmd, ok := cmd.(interface {
				Val() TopKInfo
				Err() error
			}); ok {
				return topKInfoCmd.Val(), topKInfoCmd.Err()
			}
		case CmdTypeTDigestInfo:
			if tDigestInfoCmd, ok := cmd.(interface {
				Val() TDigestInfo
				Err() error
			}); ok {
				return tDigestInfoCmd.Val(), tDigestInfoCmd.Err()
			}
		case CmdTypeFTSearch:
			if ftSearchCmd, ok := cmd.(interface {
				Val() FTSearchResult
				Err() error
			}); ok {
				return ftSearchCmd.Val(), ftSearchCmd.Err()
			}
		case CmdTypeFTInfo:
			if ftInfoCmd, ok := cmd.(interface {
				Val() FTInfoResult
				Err() error
			}); ok {
				return ftInfoCmd.Val(), ftInfoCmd.Err()
			}
		case CmdTypeFTSpellCheck:
			if ftSpellCheckCmd, ok := cmd.(interface {
				Val() []SpellCheckResult
				Err() error
			}); ok {
				return ftSpellCheckCmd.Val(), ftSpellCheckCmd.Err()
			}
		case CmdTypeFTSynDump:
			if ftSynDumpCmd, ok := cmd.(interface {
				Val() []FTSynDumpResult
				Err() error
			}); ok {
				return ftSynDumpCmd.Val(), ftSynDumpCmd.Err()
			}
		case CmdTypeAggregate:
			if aggregateCmd, ok := cmd.(interface {
				Val() *FTAggregateResult
				Err() error
			}); ok {
				return aggregateCmd.Val(), aggregateCmd.Err()
			}
		case CmdTypeTSTimestampValue:
			if tsTimestampValueCmd, ok := cmd.(interface {
				Val() TSTimestampValue
				Err() error
			}); ok {
				return tsTimestampValueCmd.Val(), tsTimestampValueCmd.Err()
			}
		case CmdTypeTSTimestampValueSlice:
			if tsTimestampValueSliceCmd, ok := cmd.(interface {
				Val() []TSTimestampValue
				Err() error
			}); ok {
				return tsTimestampValueSliceCmd.Val(), tsTimestampValueSliceCmd.Err()
			}
		case CmdTypeTSNRangePivotRowSlice:
			if tsNRangePivotRowSliceCmd, ok := cmd.(interface {
				Val() []TSNRangePivotRow
				Err() error
			}); ok {
				return tsNRangePivotRowSliceCmd.Val(), tsNRangePivotRowSliceCmd.Err()
			}
		case CmdTypeStringSlice:
			if stringSliceCmd, ok := cmd.(interface {
				Val() []string
				Err() error
			}); ok {
				return stringSliceCmd.Val(), stringSliceCmd.Err()
			}
		case CmdTypeIntSlice:
			if intSliceCmd, ok := cmd.(interface {
				Val() []int64
				Err() error
			}); ok {
				return intSliceCmd.Val(), intSliceCmd.Err()
			}
		case CmdTypeUintSlice:
			if uintSliceCmd, ok := cmd.(interface {
				Val() []uint64
				Err() error
			}); ok {
				return uintSliceCmd.Val(), uintSliceCmd.Err()
			}
		case CmdTypeBoolSlice:
			if boolSliceCmd, ok := cmd.(interface {
				Val() []bool
				Err() error
			}); ok {
				return boolSliceCmd.Val(), boolSliceCmd.Err()
			}
		case CmdTypeFloatSlice:
			if floatSliceCmd, ok := cmd.(interface {
				Val() []float64
				Err() error
			}); ok {
				return floatSliceCmd.Val(), floatSliceCmd.Err()
			}
		case CmdTypeSlice:
			if sliceCmd, ok := cmd.(interface {
				Val() []interface{}
				Err() error
			}); ok {
				return sliceCmd.Val(), sliceCmd.Err()
			}
		case CmdTypeKeyValueSlice:
			if keyValueSliceCmd, ok := cmd.(interface {
				Val() []KeyValue
				Err() error
			}); ok {
				return keyValueSliceCmd.Val(), keyValueSliceCmd.Err()
			}
		case CmdTypeAREntrySlice:
			if arEntrySliceCmd, ok := cmd.(interface {
				Val() []AREntry
				Err() error
			}); ok {
				return arEntrySliceCmd.Val(), arEntrySliceCmd.Err()
			}
		case CmdTypeMapStringString:
			if mapCmd, ok := cmd.(interface {
				Val() map[string]string
				Err() error
			}); ok {
				return mapCmd.Val(), mapCmd.Err()
			}
		case CmdTypeMapStringInt:
			if mapCmd, ok := cmd.(interface {
				Val() map[string]int64
				Err() error
			}); ok {
				return mapCmd.Val(), mapCmd.Err()
			}
		case CmdTypeMapStringInterfaceSlice:
			if mapCmd, ok := cmd.(interface {
				Val() []map[string]interface{}
				Err() error
			}); ok {
				return mapCmd.Val(), mapCmd.Err()
			}
		case CmdTypeMapStringInterface:
			if mapCmd, ok := cmd.(interface {
				Val() map[string]interface{}
				Err() error
			}); ok {
				return mapCmd.Val(), mapCmd.Err()
			}
		case CmdTypeMapStringStringSlice:
			if mapCmd, ok := cmd.(interface {
				Val() []map[string]string
				Err() error
			}); ok {
				return mapCmd.Val(), mapCmd.Err()
			}
		case CmdTypeMapMapStringInterface:
			if mapCmd, ok := cmd.(interface {
				Val() map[string]interface{}
				Err() error
			}); ok {
				return mapCmd.Val(), mapCmd.Err()
			}
		default:
			// For unknown command types, return nil
			return nil, nil
		}
	}

	// If we can't get the command type, return nil
	return nil, nil
}

//------------------------------------------------------------------------------

// IncrEXIntResult is the reply of an INCREX command issued via IncrEXInt.
// Value is the new value of the key; AppliedIncrement is the increment that
// the server actually applied (0 when an out-of-bounds operation was
// rejected, clamped when SATURATE was set).
type IncrEXIntResult struct {
	Value            int64
	AppliedIncrement int64
}

type IncrEXIntCmd struct {
	baseCmd

	val IncrEXIntResult
}

var _ Cmder = (*IncrEXIntCmd)(nil)

func NewIncrEXIntCmd(ctx context.Context, args ...interface{}) *IncrEXIntCmd {
	return &IncrEXIntCmd{
		baseCmd: baseCmd{
			ctx:     ctx,
			args:    args,
			cmdType: CmdTypeIncrEXInt,
		},
	}
}

func (cmd *IncrEXIntCmd) SetVal(val IncrEXIntResult) { cmd.val = val }
func (cmd *IncrEXIntCmd) Val() IncrEXIntResult {
	cmd.await()
	return cmd.val
}
func (cmd *IncrEXIntCmd) Result() (IncrEXIntResult, error) {
	cmd.await()
	return cmd.val, cmd.err
}
func (cmd *IncrEXIntCmd) String() string {
	cmd.await()
	return cmdString(cmd, cmd.val)
}

func (cmd *IncrEXIntCmd) readReply(rd *proto.Reader) error {
	if err := rd.ReadFixedArrayLen(2); err != nil {
		return err
	}
	value, err := rd.ReadInt()
	if err != nil {
		return err
	}
	applied, err := rd.ReadInt()
	if err != nil {
		return err
	}
	cmd.val = IncrEXIntResult{Value: value, AppliedIncrement: applied}
	return nil
}

func (cmd *IncrEXIntCmd) Clone() Cmder {
	return &IncrEXIntCmd{
		baseCmd: cmd.cloneBaseCmd(),
		val:     cmd.val,
	}
}

// IncrEXFloatResult is the reply of an INCREX command issued via IncrEXFloat.
type IncrEXFloatResult struct {
	Value            float64
	AppliedIncrement float64
}

type IncrEXFloatCmd struct {
	baseCmd

	val IncrEXFloatResult
}

var _ Cmder = (*IncrEXFloatCmd)(nil)

func NewIncrEXFloatCmd(ctx context.Context, args ...interface{}) *IncrEXFloatCmd {
	return &IncrEXFloatCmd{
		baseCmd: baseCmd{
			ctx:     ctx,
			args:    args,
			cmdType: CmdTypeIncrEXFloat,
		},
	}
}

func (cmd *IncrEXFloatCmd) SetVal(val IncrEXFloatResult) { cmd.val = val }
func (cmd *IncrEXFloatCmd) Val() IncrEXFloatResult {
	cmd.await()
	return cmd.val
}
func (cmd *IncrEXFloatCmd) Result() (IncrEXFloatResult, error) {
	cmd.await()
	return cmd.val, cmd.err
}
func (cmd *IncrEXFloatCmd) String() string {
	cmd.await()
	return cmdString(cmd, cmd.val)
}

func (cmd *IncrEXFloatCmd) readReply(rd *proto.Reader) error {
	if err := rd.ReadFixedArrayLen(2); err != nil {
		return err
	}
	value, err := rd.ReadFloat()
	if err != nil {
		return err
	}
	applied, err := rd.ReadFloat()
	if err != nil {
		return err
	}
	cmd.val = IncrEXFloatResult{Value: value, AppliedIncrement: applied}
	return nil
}

func (cmd *IncrEXFloatCmd) Clone() Cmder {
	return &IncrEXFloatCmd{
		baseCmd: cmd.cloneBaseCmd(),
		val:     cmd.val,
	}
}

//------------------------------------------------------------------------------

// AREntrySliceCmd is a command that returns index-value pairs from ARSCAN or ARGREP.
type AREntrySliceCmd struct {
	baseCmd
	val []AREntry
}

var _ Cmder = (*AREntrySliceCmd)(nil)

func NewAREntrySliceCmd(ctx context.Context, args ...any) *AREntrySliceCmd {
	return &AREntrySliceCmd{
		baseCmd: baseCmd{
			ctx:     ctx,
			args:    args,
			cmdType: CmdTypeAREntrySlice,
		},
	}
}

func (cmd *AREntrySliceCmd) SetVal(val []AREntry) {
	cmd.val = val
}

func (cmd *AREntrySliceCmd) Val() []AREntry {
	cmd.await()
	return cmd.val
}

func (cmd *AREntrySliceCmd) Result() ([]AREntry, error) {
	cmd.await()
	return cmd.val, cmd.err
}

func (cmd *AREntrySliceCmd) String() string {
	cmd.await()
	return cmdString(cmd, cmd.val)
}

func (cmd *AREntrySliceCmd) readReply(rd *proto.Reader) error {
	n, err := rd.ReadArrayLen()
	if err != nil {
		return err
	}
	if n == 0 {
		cmd.val = make([]AREntry, 0)
		return nil
	}

	cmd.val = make([]AREntry, n)
	for i := range n {
		if err = rd.ReadFixedArrayLen(2); err != nil {
			return err
		}

		cmd.val[i].Index, err = rd.ReadUint()
		if err != nil {
			return err
		}

		cmd.val[i].Value, err = rd.ReadString()
		if err != nil {
			return err
		}
	}
	return nil
}

func (cmd *AREntrySliceCmd) Clone() Cmder {
	var val []AREntry
	if cmd.val != nil {
		val = make([]AREntry, len(cmd.val))
		copy(val, cmd.val)
	}
	return &AREntrySliceCmd{
		baseCmd: cmd.cloneBaseCmd(),
		val:     val,
	}
}
