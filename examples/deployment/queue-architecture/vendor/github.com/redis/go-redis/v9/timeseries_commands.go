package redis

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"time"

	"github.com/redis/go-redis/v9/internal/proto"
	"github.com/redis/go-redis/v9/internal/util"
)

type TimeseriesCmdable interface {
	TSAdd(ctx context.Context, key string, timestamp interface{}, value float64) *IntCmd
	TSAddWithArgs(ctx context.Context, key string, timestamp interface{}, value float64, options *TSOptions) *IntCmd
	TSCreate(ctx context.Context, key string) *StatusCmd
	TSCreateWithArgs(ctx context.Context, key string, options *TSOptions) *StatusCmd
	TSAlter(ctx context.Context, key string, options *TSAlterOptions) *StatusCmd
	TSCreateRule(ctx context.Context, sourceKey string, destKey string, aggregator Aggregator, bucketDuration int) *StatusCmd
	TSCreateRuleWithArgs(ctx context.Context, sourceKey string, destKey string, aggregator Aggregator, bucketDuration int, options *TSCreateRuleOptions) *StatusCmd
	TSIncrBy(ctx context.Context, Key string, timestamp float64) *IntCmd
	TSIncrByWithArgs(ctx context.Context, key string, timestamp float64, options *TSIncrDecrOptions) *IntCmd
	TSDecrBy(ctx context.Context, Key string, timestamp float64) *IntCmd
	TSDecrByWithArgs(ctx context.Context, key string, timestamp float64, options *TSIncrDecrOptions) *IntCmd
	TSDel(ctx context.Context, Key string, fromTimestamp int, toTimestamp int) *IntCmd
	TSDeleteRule(ctx context.Context, sourceKey string, destKey string) *StatusCmd
	TSGet(ctx context.Context, key string) *TSTimestampValueCmd
	TSGetWithArgs(ctx context.Context, key string, options *TSGetOptions) *TSTimestampValueCmd
	TSInfo(ctx context.Context, key string) *MapStringInterfaceCmd
	TSInfoWithArgs(ctx context.Context, key string, options *TSInfoOptions) *MapStringInterfaceCmd
	TSMAdd(ctx context.Context, ktvSlices [][]interface{}) *IntSliceCmd
	TSQueryIndex(ctx context.Context, filterExpr []string) *StringSliceCmd
	TSQueryLabels(ctx context.Context, filterExpr []string) *StringSliceCmd
	TSQueryLabelValues(ctx context.Context, label string, filterExpr []string) *StringSliceCmd
	TSRevRange(ctx context.Context, key string, fromTimestamp int, toTimestamp int) *TSTimestampValueSliceCmd
	TSRevRangeWithArgs(ctx context.Context, key string, fromTimestamp int, toTimestamp int, options *TSRevRangeOptions) *TSTimestampValueSliceCmd
	TSRange(ctx context.Context, key string, fromTimestamp int, toTimestamp int) *TSTimestampValueSliceCmd
	TSRangeWithArgs(ctx context.Context, key string, fromTimestamp int, toTimestamp int, options *TSRangeOptions) *TSTimestampValueSliceCmd
	TSMRange(ctx context.Context, fromTimestamp int, toTimestamp int, filterExpr []string) *MapStringSliceInterfaceCmd
	TSMRangeWithArgs(ctx context.Context, fromTimestamp int, toTimestamp int, filterExpr []string, options *TSMRangeOptions) *MapStringSliceInterfaceCmd
	TSMRevRange(ctx context.Context, fromTimestamp int, toTimestamp int, filterExpr []string) *MapStringSliceInterfaceCmd
	TSMRevRangeWithArgs(ctx context.Context, fromTimestamp int, toTimestamp int, filterExpr []string, options *TSMRevRangeOptions) *MapStringSliceInterfaceCmd
	TSMGet(ctx context.Context, filters []string) *MapStringSliceInterfaceCmd
	TSMGetWithArgs(ctx context.Context, filters []string, options *TSMGetOptions) *MapStringSliceInterfaceCmd
	TSNRange(ctx context.Context, keys []string, fromTimestamp interface{}, toTimestamp interface{}) *TSNRangePivotRowSliceCmd
	TSNRangeWithArgs(ctx context.Context, keys []string, fromTimestamp interface{}, toTimestamp interface{}, options *TSNRangeOptions) *TSNRangePivotRowSliceCmd
	TSNRevRange(ctx context.Context, keys []string, fromTimestamp interface{}, toTimestamp interface{}) *TSNRangePivotRowSliceCmd
	TSNRevRangeWithArgs(ctx context.Context, keys []string, fromTimestamp interface{}, toTimestamp interface{}, options *TSNRevRangeOptions) *TSNRangePivotRowSliceCmd
	TSRead(ctx context.Context, key string, timestamp interface{}) *TSTimestampValueSliceCmd
	TSReadWithArgs(ctx context.Context, key string, timestamp interface{}, options *TSReadOptions) *TSTimestampValueSliceCmd
}

// TS.READ timestamp cursor sentinels.
const (
	TSReadEarliest = "-" // read from the earliest sample
	TSReadLatest   = "+" // latest sample, inclusive
	TSReadNew      = "$" // only samples added after the call
)

// TSReadOptions holds the optional TS.READ arguments.
type TSReadOptions struct {
	Block    bool          // wait for samples (emits the BLOCK group)
	Timeout  time.Duration // max wait; 0 blocks indefinitely
	MinCount int           // unblock threshold; defaults to 1
	MaxCount int           // reply cap; 0 is unlimited
}

type TSOptions struct {
	Retention         int
	ChunkSize         int
	Encoding          string
	DuplicatePolicy   string
	Labels            map[string]string
	IgnoreMaxTimeDiff int64
	IgnoreMaxValDiff  float64
}
type TSIncrDecrOptions struct {
	Timestamp         int64
	Retention         int
	ChunkSize         int
	Uncompressed      bool
	DuplicatePolicy   string
	Labels            map[string]string
	IgnoreMaxTimeDiff int64
	IgnoreMaxValDiff  float64
}

type TSAlterOptions struct {
	Retention         int
	ChunkSize         int
	DuplicatePolicy   string
	Labels            map[string]string
	IgnoreMaxTimeDiff int64
	IgnoreMaxValDiff  float64
}

type TSCreateRuleOptions struct {
	alignTimestamp int64
}

type TSGetOptions struct {
	Latest bool
}

type TSInfoOptions struct {
	Debug bool
}
type Aggregator int

const (
	Invalid = Aggregator(iota)
	Avg
	Sum
	Min
	Max
	Range
	Count
	First
	Last
	StdP
	StdS
	VarP
	VarS
	Twa
	CountNaN
	CountAll
)

func (a Aggregator) String() string {
	switch a {
	case Invalid:
		return ""
	case Avg:
		return "AVG"
	case Sum:
		return "SUM"
	case Min:
		return "MIN"
	case Max:
		return "MAX"
	case Range:
		return "RANGE"
	case Count:
		return "COUNT"
	case First:
		return "FIRST"
	case Last:
		return "LAST"
	case StdP:
		return "STD.P"
	case StdS:
		return "STD.S"
	case VarP:
		return "VAR.P"
	case VarS:
		return "VAR.S"
	case Twa:
		return "TWA"
	case CountNaN:
		return "COUNTNAN"
	case CountAll:
		return "COUNTALL"
	default:
		return ""
	}
}

var (
	errTSMultiAggregationGroupBy = errors.New("redis: GROUPBY is not allowed when multiple aggregators are specified")
	errTSAggregationConflict     = errors.New("redis: setting both Aggregator and Aggregators is not allowed; use Aggregators instead because Aggregator is deprecated")
	errTSExcludeEmptyGroupBy     = errors.New("redis: EXCLUDEEMPTY is not allowed with GROUPBY")
)

func formatAggregationArgs(aggregator Aggregator, aggregators []Aggregator) (string, int, error) {
	if aggregator != Invalid && len(aggregators) > 0 {
		return "", 0, errTSAggregationConflict
	}
	if len(aggregators) == 0 {
		if aggregator == Invalid {
			return "", 0, nil
		}
		aggregationArg, err := formatAggregatorArg(aggregator)
		if err != nil {
			return "", 0, err
		}
		return aggregationArg, 1, nil
	}

	parts := make([]string, len(aggregators))
	for i, agg := range aggregators {
		if agg == Invalid {
			return "", 0, fmt.Errorf("redis: invalid timeseries aggregator at index %d: Invalid (%d)", i, agg)
		}
		aggregationArg, err := formatAggregatorArg(agg)
		if err != nil {
			return "", 0, fmt.Errorf("redis: invalid timeseries aggregator at index %d: %d", i, agg)
		}
		parts[i] = aggregationArg
	}

	return strings.Join(parts, ","), len(parts), nil
}

func formatAggregatorArg(aggregator Aggregator) (string, error) {
	aggregationArg := aggregator.String()
	if aggregationArg == "" {
		return "", fmt.Errorf("redis: invalid timeseries aggregator: %d", aggregator)
	}
	return aggregationArg, nil
}

type TSRangeOptions struct {
	Latest        bool
	FilterByTS    []int
	FilterByValue []int
	Count         int
	Align         interface{}
	// Deprecated: use Aggregators instead.
	Aggregator      Aggregator
	Aggregators     []Aggregator
	BucketDuration  int
	BucketTimestamp interface{}
	Empty           bool
}

type TSRevRangeOptions struct {
	Latest        bool
	FilterByTS    []int
	FilterByValue []int
	Count         int
	Align         interface{}
	// Deprecated: use Aggregators instead.
	Aggregator      Aggregator
	Aggregators     []Aggregator
	BucketDuration  int
	BucketTimestamp interface{}
	Empty           bool
}

type TSMRangeOptions struct {
	Latest         bool
	FilterByTS     []int
	FilterByValue  []int
	WithLabels     bool
	SelectedLabels []interface{}
	Count          int
	Align          interface{}
	// Deprecated: use Aggregators instead.
	Aggregator      Aggregator
	Aggregators     []Aggregator
	BucketDuration  int
	BucketTimestamp interface{}
	Empty           bool
	// ExcludeEmpty omits matching series that have no samples. Not allowed with GroupByLabel/Reducer. Redis 8.10+.
	ExcludeEmpty bool
	GroupByLabel interface{}
	Reducer      interface{}
}

type TSMRevRangeOptions struct {
	Latest         bool
	FilterByTS     []int
	FilterByValue  []int
	WithLabels     bool
	SelectedLabels []interface{}
	Count          int
	Align          interface{}
	// Deprecated: use Aggregators instead.
	Aggregator      Aggregator
	Aggregators     []Aggregator
	BucketDuration  int
	BucketTimestamp interface{}
	Empty           bool
	// ExcludeEmpty omits matching series that have no samples. Not allowed with GroupByLabel/Reducer. Redis 8.10+.
	ExcludeEmpty bool
	GroupByLabel interface{}
	Reducer      interface{}
}

type TSMGetOptions struct {
	Latest         bool
	WithLabels     bool
	SelectedLabels []interface{}
}

type TSNRangeOptions struct {
	Latest        bool
	FilterByTS    []int
	FilterByValue []float64 // exactly two elements: [min, max]
	Count         int
	Align         interface{}
	// Aggregators holds exactly one aggregator spec per key. Each spec lists one or
	// more aggregators applied to that key and is sent as a single comma-joined token
	// (e.g. {{Min, Max}, {Sum}} -> AGGREGATION MIN,MAX SUM <bucketDuration>).
	Aggregators     [][]Aggregator
	BucketDuration  int
	BucketTimestamp interface{}
	Empty           bool
}

type TSNRevRangeOptions struct {
	Latest        bool
	FilterByTS    []int
	FilterByValue []float64 // exactly two elements: [min, max]
	Count         int
	Align         interface{}
	// Aggregators holds exactly one aggregator spec per key. Each spec lists one or
	// more aggregators applied to that key and is sent as a single comma-joined token
	// (e.g. {{Min, Max}, {Sum}} -> AGGREGATION MIN,MAX SUM <bucketDuration>).
	Aggregators     [][]Aggregator
	BucketDuration  int
	BucketTimestamp interface{}
	Empty           bool
}

// TSAdd - Adds one or more observations to a t-digest sketch.
// For more information - https://redis.io/commands/ts.add/
func (c cmdable) TSAdd(ctx context.Context, key string, timestamp interface{}, value float64) *IntCmd {
	args := []interface{}{"TS.ADD", key, timestamp, value}
	cmd := NewIntCmd(ctx, args...)
	_ = c(ctx, cmd)
	return cmd
}

// TSAddWithArgs - Adds one or more observations to a t-digest sketch.
// This function also allows for specifying additional options such as:
// Retention, ChunkSize, Encoding, DuplicatePolicy and Labels.
// For more information - https://redis.io/commands/ts.add/
func (c cmdable) TSAddWithArgs(ctx context.Context, key string, timestamp interface{}, value float64, options *TSOptions) *IntCmd {
	args := []interface{}{"TS.ADD", key, timestamp, value}
	if options != nil {
		if options.Retention != 0 {
			args = append(args, "RETENTION", options.Retention)
		}
		if options.ChunkSize != 0 {
			args = append(args, "CHUNK_SIZE", options.ChunkSize)
		}
		if options.Encoding != "" {
			args = append(args, "ENCODING", options.Encoding)
		}

		if options.DuplicatePolicy != "" {
			args = append(args, "DUPLICATE_POLICY", options.DuplicatePolicy)
		}
		if options.Labels != nil {
			args = append(args, "LABELS")
			for label, value := range options.Labels {
				args = append(args, label, value)
			}
		}
		if options.IgnoreMaxTimeDiff != 0 || options.IgnoreMaxValDiff != 0 {
			args = append(args, "IGNORE", options.IgnoreMaxTimeDiff, options.IgnoreMaxValDiff)
		}
	}
	cmd := NewIntCmd(ctx, args...)
	_ = c(ctx, cmd)
	return cmd
}

// TSCreate - Creates a new time-series key.
// For more information - https://redis.io/commands/ts.create/
func (c cmdable) TSCreate(ctx context.Context, key string) *StatusCmd {
	args := []interface{}{"TS.CREATE", key}
	cmd := NewStatusCmd(ctx, args...)
	_ = c(ctx, cmd)
	return cmd
}

// TSCreateWithArgs - Creates a new time-series key with additional options.
// This function allows for specifying additional options such as:
// Retention, ChunkSize, Encoding, DuplicatePolicy and Labels.
// For more information - https://redis.io/commands/ts.create/
func (c cmdable) TSCreateWithArgs(ctx context.Context, key string, options *TSOptions) *StatusCmd {
	args := []interface{}{"TS.CREATE", key}
	if options != nil {
		if options.Retention != 0 {
			args = append(args, "RETENTION", options.Retention)
		}
		if options.ChunkSize != 0 {
			args = append(args, "CHUNK_SIZE", options.ChunkSize)
		}
		if options.Encoding != "" {
			args = append(args, "ENCODING", options.Encoding)
		}

		if options.DuplicatePolicy != "" {
			args = append(args, "DUPLICATE_POLICY", options.DuplicatePolicy)
		}
		if options.Labels != nil {
			args = append(args, "LABELS")
			for label, value := range options.Labels {
				args = append(args, label, value)
			}
		}
		if options.IgnoreMaxTimeDiff != 0 || options.IgnoreMaxValDiff != 0 {
			args = append(args, "IGNORE", options.IgnoreMaxTimeDiff, options.IgnoreMaxValDiff)
		}
	}
	cmd := NewStatusCmd(ctx, args...)
	_ = c(ctx, cmd)
	return cmd
}

// TSAlter - Alters an existing time-series key with additional options.
// This function allows for specifying additional options such as:
// Retention, ChunkSize and DuplicatePolicy.
// For more information - https://redis.io/commands/ts.alter/
func (c cmdable) TSAlter(ctx context.Context, key string, options *TSAlterOptions) *StatusCmd {
	args := []interface{}{"TS.ALTER", key}
	if options != nil {
		if options.Retention != 0 {
			args = append(args, "RETENTION", options.Retention)
		}
		if options.ChunkSize != 0 {
			args = append(args, "CHUNK_SIZE", options.ChunkSize)
		}
		if options.DuplicatePolicy != "" {
			args = append(args, "DUPLICATE_POLICY", options.DuplicatePolicy)
		}
		if options.Labels != nil {
			args = append(args, "LABELS")
			for label, value := range options.Labels {
				args = append(args, label, value)
			}
		}
		if options.IgnoreMaxTimeDiff != 0 || options.IgnoreMaxValDiff != 0 {
			args = append(args, "IGNORE", options.IgnoreMaxTimeDiff, options.IgnoreMaxValDiff)
		}
	}
	cmd := NewStatusCmd(ctx, args...)
	_ = c(ctx, cmd)
	return cmd
}

// TSCreateRule - Creates a compaction rule from sourceKey to destKey.
// For more information - https://redis.io/commands/ts.createrule/
func (c cmdable) TSCreateRule(ctx context.Context, sourceKey string, destKey string, aggregator Aggregator, bucketDuration int) *StatusCmd {
	args := []interface{}{"TS.CREATERULE", sourceKey, destKey, "AGGREGATION", aggregator.String(), bucketDuration}
	cmd := NewStatusCmd(ctx, args...)
	_ = c(ctx, cmd)
	return cmd
}

// TSCreateRuleWithArgs - Creates a compaction rule from sourceKey to destKey with additional option.
// This function allows for specifying additional option such as:
// alignTimestamp.
// For more information - https://redis.io/commands/ts.createrule/
func (c cmdable) TSCreateRuleWithArgs(ctx context.Context, sourceKey string, destKey string, aggregator Aggregator, bucketDuration int, options *TSCreateRuleOptions) *StatusCmd {
	args := []interface{}{"TS.CREATERULE", sourceKey, destKey, "AGGREGATION", aggregator.String(), bucketDuration}
	if options != nil {
		if options.alignTimestamp != 0 {
			args = append(args, options.alignTimestamp)
		}
	}
	cmd := NewStatusCmd(ctx, args...)
	_ = c(ctx, cmd)
	return cmd
}

// TSIncrBy - Increments the value of a time-series key by the specified timestamp.
// For more information - https://redis.io/commands/ts.incrby/
func (c cmdable) TSIncrBy(ctx context.Context, Key string, timestamp float64) *IntCmd {
	args := []interface{}{"TS.INCRBY", Key, timestamp}
	cmd := NewIntCmd(ctx, args...)
	_ = c(ctx, cmd)
	return cmd
}

// TSIncrByWithArgs - Increments the value of a time-series key by the specified timestamp with additional options.
// This function allows for specifying additional options such as:
// Timestamp, Retention, ChunkSize, Uncompressed and Labels.
// For more information - https://redis.io/commands/ts.incrby/
func (c cmdable) TSIncrByWithArgs(ctx context.Context, key string, timestamp float64, options *TSIncrDecrOptions) *IntCmd {
	args := []interface{}{"TS.INCRBY", key, timestamp}
	if options != nil {
		if options.Timestamp != 0 {
			args = append(args, "TIMESTAMP", options.Timestamp)
		}
		if options.Retention != 0 {
			args = append(args, "RETENTION", options.Retention)
		}
		if options.ChunkSize != 0 {
			args = append(args, "CHUNK_SIZE", options.ChunkSize)
		}
		if options.Uncompressed {
			args = append(args, "UNCOMPRESSED")
		}
		if options.DuplicatePolicy != "" {
			args = append(args, "DUPLICATE_POLICY", options.DuplicatePolicy)
		}
		if options.Labels != nil {
			args = append(args, "LABELS")
			for label, value := range options.Labels {
				args = append(args, label, value)
			}
		}
		if options.IgnoreMaxTimeDiff != 0 || options.IgnoreMaxValDiff != 0 {
			args = append(args, "IGNORE", options.IgnoreMaxTimeDiff, options.IgnoreMaxValDiff)
		}
	}
	cmd := NewIntCmd(ctx, args...)
	_ = c(ctx, cmd)
	return cmd
}

// TSDecrBy - Decrements the value of a time-series key by the specified timestamp.
// For more information - https://redis.io/commands/ts.decrby/
func (c cmdable) TSDecrBy(ctx context.Context, Key string, timestamp float64) *IntCmd {
	args := []interface{}{"TS.DECRBY", Key, timestamp}
	cmd := NewIntCmd(ctx, args...)
	_ = c(ctx, cmd)
	return cmd
}

// TSDecrByWithArgs - Decrements the value of a time-series key by the specified timestamp with additional options.
// This function allows for specifying additional options such as:
// Timestamp, Retention, ChunkSize, Uncompressed and Labels.
// For more information - https://redis.io/commands/ts.decrby/
func (c cmdable) TSDecrByWithArgs(ctx context.Context, key string, timestamp float64, options *TSIncrDecrOptions) *IntCmd {
	args := []interface{}{"TS.DECRBY", key, timestamp}
	if options != nil {
		if options.Timestamp != 0 {
			args = append(args, "TIMESTAMP", options.Timestamp)
		}
		if options.Retention != 0 {
			args = append(args, "RETENTION", options.Retention)
		}
		if options.ChunkSize != 0 {
			args = append(args, "CHUNK_SIZE", options.ChunkSize)
		}
		if options.Uncompressed {
			args = append(args, "UNCOMPRESSED")
		}
		if options.DuplicatePolicy != "" {
			args = append(args, "DUPLICATE_POLICY", options.DuplicatePolicy)
		}
		if options.Labels != nil {
			args = append(args, "LABELS")
			for label, value := range options.Labels {
				args = append(args, label, value)
			}
		}
		if options.IgnoreMaxTimeDiff != 0 || options.IgnoreMaxValDiff != 0 {
			args = append(args, "IGNORE", options.IgnoreMaxTimeDiff, options.IgnoreMaxValDiff)
		}
	}
	cmd := NewIntCmd(ctx, args...)
	_ = c(ctx, cmd)
	return cmd
}

// TSDel - Deletes a range of samples from a time-series key.
// For more information - https://redis.io/commands/ts.del/
func (c cmdable) TSDel(ctx context.Context, Key string, fromTimestamp int, toTimestamp int) *IntCmd {
	args := []interface{}{"TS.DEL", Key, fromTimestamp, toTimestamp}
	cmd := NewIntCmd(ctx, args...)
	_ = c(ctx, cmd)
	return cmd
}

// TSDeleteRule - Deletes a compaction rule from sourceKey to destKey.
// For more information - https://redis.io/commands/ts.deleterule/
func (c cmdable) TSDeleteRule(ctx context.Context, sourceKey string, destKey string) *StatusCmd {
	args := []interface{}{"TS.DELETERULE", sourceKey, destKey}
	cmd := NewStatusCmd(ctx, args...)
	_ = c(ctx, cmd)
	return cmd
}

// TSGetWithArgs - Gets the last sample of a time-series key with additional option.
// This function allows for specifying additional option such as:
// Latest.
// For more information - https://redis.io/commands/ts.get/
func (c cmdable) TSGetWithArgs(ctx context.Context, key string, options *TSGetOptions) *TSTimestampValueCmd {
	args := []interface{}{"TS.GET", key}
	if options != nil {
		if options.Latest {
			args = append(args, "LATEST")
		}
	}
	cmd := newTSTimestampValueCmd(ctx, args...)
	_ = c(ctx, cmd)
	return cmd
}

// TSGet - Gets the last sample of a time-series key.
// For more information - https://redis.io/commands/ts.get/
func (c cmdable) TSGet(ctx context.Context, key string) *TSTimestampValueCmd {
	args := []interface{}{"TS.GET", key}
	cmd := newTSTimestampValueCmd(ctx, args...)
	_ = c(ctx, cmd)
	return cmd
}

type TSTimestampValue struct {
	Timestamp int64
	Value     float64
	Values    []float64
}

func (tv TSTimestampValue) String() string {
	if len(tv.Values) > 0 {
		return fmt.Sprintf("{%d %v}", tv.Timestamp, tv.Values)
	}
	return fmt.Sprintf("{%d %v}", tv.Timestamp, tv.Value)
}

type TSTimestampValueCmd struct {
	baseCmd
	val TSTimestampValue
}

func newTSTimestampValueCmd(ctx context.Context, args ...interface{}) *TSTimestampValueCmd {
	return &TSTimestampValueCmd{
		baseCmd: baseCmd{
			ctx:     ctx,
			args:    args,
			cmdType: CmdTypeTSTimestampValue,
		},
	}
}

func (cmd *TSTimestampValueCmd) String() string {
	cmd.await()
	return cmdString(cmd, cmd.val)
}

func (cmd *TSTimestampValueCmd) SetVal(val TSTimestampValue) {
	cmd.val = val
}

func (cmd *TSTimestampValueCmd) Result() (TSTimestampValue, error) {
	cmd.await()
	return cmd.val, cmd.err
}

func (cmd *TSTimestampValueCmd) Val() TSTimestampValue {
	cmd.await()
	return cmd.val
}

func (cmd *TSTimestampValueCmd) readReply(rd *proto.Reader) (err error) {
	n, err := rd.ReadMapLen()
	if err != nil {
		return err
	}
	cmd.val = TSTimestampValue{}
	for i := 0; i < n; i++ {
		timestamp, err := rd.ReadInt()
		if err != nil {
			return err
		}
		value, err := rd.ReadString()
		if err != nil {
			return err
		}
		cmd.val.Timestamp = timestamp
		cmd.val.Value, err = util.ParseStringToFloat(value)
		if err != nil {
			return err
		}
	}

	return nil
}

func (cmd *TSTimestampValueCmd) Clone() Cmder {
	val := cmd.val
	if cmd.val.Values != nil {
		val.Values = make([]float64, len(cmd.val.Values))
		copy(val.Values, cmd.val.Values)
	}
	return &TSTimestampValueCmd{
		baseCmd: cmd.cloneBaseCmd(),
		val:     val,
	}
}

// TSInfo - Returns information about a time-series key.
// For more information - https://redis.io/commands/ts.info/
func (c cmdable) TSInfo(ctx context.Context, key string) *MapStringInterfaceCmd {
	args := []interface{}{"TS.INFO", key}
	cmd := NewMapStringInterfaceCmd(ctx, args...)
	_ = c(ctx, cmd)
	return cmd
}

// TSInfoWithArgs - Returns information about a time-series key with additional option.
// This function allows for specifying additional option such as:
// Debug.
// For more information - https://redis.io/commands/ts.info/
func (c cmdable) TSInfoWithArgs(ctx context.Context, key string, options *TSInfoOptions) *MapStringInterfaceCmd {
	args := []interface{}{"TS.INFO", key}
	if options != nil {
		if options.Debug {
			args = append(args, "DEBUG")
		}
	}
	cmd := NewMapStringInterfaceCmd(ctx, args...)
	_ = c(ctx, cmd)
	return cmd
}

// TSMAdd - Adds multiple samples to multiple time-series keys.
// It accepts a slice of 'ktv' slices, each containing exactly three elements: key, timestamp, and value.
// This struct must be provided for this command to work.
// For more information - https://redis.io/commands/ts.madd/
func (c cmdable) TSMAdd(ctx context.Context, ktvSlices [][]interface{}) *IntSliceCmd {
	args := []interface{}{"TS.MADD"}
	for _, ktv := range ktvSlices {
		args = append(args, ktv...)
	}
	cmd := NewIntSliceCmd(ctx, args...)
	_ = c(ctx, cmd)
	return cmd
}

// TSQueryIndex - Returns all the keys matching the filter expression.
// For more information - https://redis.io/commands/ts.queryindex/
func (c cmdable) TSQueryIndex(ctx context.Context, filterExpr []string) *StringSliceCmd {
	args := []interface{}{"TS.QUERYINDEX"}
	for _, f := range filterExpr {
		args = append(args, f)
	}
	cmd := NewStringSliceCmd(ctx, args...)
	_ = c(ctx, cmd)
	return cmd
}

// TSQueryLabels - Returns the set of label names present on the time series
// matching the filter expressions. Passing no filter expressions queries all
// indexed series. The reply is unordered and already deduplicated by the
// server; it includes the label names used in the filter itself, and an
// empty reply is a valid result, not an error.
// filterExpr uses the same filter language as TSQueryIndex and is passed to
// the server verbatim. Available since Redis 8.10.
// For more information - https://redis.io/commands/ts.querylabels/
func (c cmdable) TSQueryLabels(ctx context.Context, filterExpr []string) *StringSliceCmd {
	args := []interface{}{"TS.QUERYLABELS", "LABELS"}
	args = appendTSFilter(args, filterExpr)
	cmd := NewStringSliceCmd(ctx, args...)
	_ = c(ctx, cmd)
	return cmd
}

// TSQueryLabelValues - Returns the set of values assigned to the given label
// name across the time series matching the filter expressions. Passing no
// filter expressions queries all indexed series. The label name is matched
// byte-exactly; a label present on no matching series yields an empty reply,
// not an error. The reply is unordered and already deduplicated by the
// server.
// filterExpr uses the same filter language as TSQueryIndex and is passed to
// the server verbatim. Available since Redis 8.10.
// For more information - https://redis.io/commands/ts.querylabels/
func (c cmdable) TSQueryLabelValues(ctx context.Context, label string, filterExpr []string) *StringSliceCmd {
	args := []interface{}{"TS.QUERYLABELS", "VALUES", label}
	args = appendTSFilter(args, filterExpr)
	cmd := NewStringSliceCmd(ctx, args...)
	_ = c(ctx, cmd)
	return cmd
}

// appendTSFilter appends the FILTER token followed by the filter expressions,
// or nothing when no expressions are given: the server rejects a bare FILTER
// token, and omitting it is the documented way to query all indexed series.
func appendTSFilter(args []interface{}, filterExpr []string) []interface{} {
	if len(filterExpr) == 0 {
		return args
	}
	args = append(args, "FILTER")
	for _, f := range filterExpr {
		args = append(args, f)
	}
	return args
}

// TSRevRange - Returns a range of samples from a time-series key in reverse order.
// For more information - https://redis.io/commands/ts.revrange/
func (c cmdable) TSRevRange(ctx context.Context, key string, fromTimestamp int, toTimestamp int) *TSTimestampValueSliceCmd {
	args := []interface{}{"TS.REVRANGE", key, fromTimestamp, toTimestamp}
	cmd := newTSTimestampValueSliceCmd(ctx, args...)
	_ = c(ctx, cmd)
	return cmd
}

// TSRevRangeWithArgs - Returns a range of samples from a time-series key in reverse order with additional options.
// This function allows for specifying additional options such as:
// Latest, FilterByTS, FilterByValue, Count, Align, Aggregator,
// BucketDuration, BucketTimestamp and Empty.
// For more information - https://redis.io/commands/ts.revrange/
func (c cmdable) TSRevRangeWithArgs(ctx context.Context, key string, fromTimestamp int, toTimestamp int, options *TSRevRangeOptions) *TSTimestampValueSliceCmd {
	args := []interface{}{"TS.REVRANGE", key, fromTimestamp, toTimestamp}
	if options != nil {
		if options.Latest {
			args = append(args, "LATEST")
		}
		if options.FilterByTS != nil {
			args = append(args, "FILTER_BY_TS")
			for _, f := range options.FilterByTS {
				args = append(args, f)
			}
		}
		if options.FilterByValue != nil {
			args = append(args, "FILTER_BY_VALUE")
			for _, f := range options.FilterByValue {
				args = append(args, f)
			}
		}
		if options.Count != 0 {
			args = append(args, "COUNT", options.Count)
		}
		if options.Align != nil {
			args = append(args, "ALIGN", options.Align)
		}
		aggregationArg, _, err := formatAggregationArgs(options.Aggregator, options.Aggregators)
		if err != nil {
			cmd := newTSTimestampValueSliceCmd(ctx, args...)
			cmd.SetErr(err)
			return cmd
		}
		if aggregationArg != "" {
			args = append(args, "AGGREGATION", aggregationArg)
		}
		if options.BucketDuration != 0 {
			args = append(args, options.BucketDuration)
		}
		if options.BucketTimestamp != nil {
			args = append(args, "BUCKETTIMESTAMP", options.BucketTimestamp)
		}
		if options.Empty {
			args = append(args, "EMPTY")
		}
	}
	cmd := newTSTimestampValueSliceCmd(ctx, args...)
	_ = c(ctx, cmd)
	return cmd
}

// TSRange - Returns a range of samples from a time-series key.
// For more information - https://redis.io/commands/ts.range/
func (c cmdable) TSRange(ctx context.Context, key string, fromTimestamp int, toTimestamp int) *TSTimestampValueSliceCmd {
	args := []interface{}{"TS.RANGE", key, fromTimestamp, toTimestamp}
	cmd := newTSTimestampValueSliceCmd(ctx, args...)
	_ = c(ctx, cmd)
	return cmd
}

// TSRangeWithArgs - Returns a range of samples from a time-series key with additional options.
// This function allows for specifying additional options such as:
// Latest, FilterByTS, FilterByValue, Count, Align, Aggregator,
// BucketDuration, BucketTimestamp and Empty.
// For more information - https://redis.io/commands/ts.range/
func (c cmdable) TSRangeWithArgs(ctx context.Context, key string, fromTimestamp int, toTimestamp int, options *TSRangeOptions) *TSTimestampValueSliceCmd {
	args := []interface{}{"TS.RANGE", key, fromTimestamp, toTimestamp}
	if options != nil {
		if options.Latest {
			args = append(args, "LATEST")
		}
		if options.FilterByTS != nil {
			args = append(args, "FILTER_BY_TS")
			for _, f := range options.FilterByTS {
				args = append(args, f)
			}
		}
		if options.FilterByValue != nil {
			args = append(args, "FILTER_BY_VALUE")
			for _, f := range options.FilterByValue {
				args = append(args, f)
			}
		}
		if options.Count != 0 {
			args = append(args, "COUNT", options.Count)
		}
		if options.Align != nil {
			args = append(args, "ALIGN", options.Align)
		}
		aggregationArg, _, err := formatAggregationArgs(options.Aggregator, options.Aggregators)
		if err != nil {
			cmd := newTSTimestampValueSliceCmd(ctx, args...)
			cmd.SetErr(err)
			return cmd
		}
		if aggregationArg != "" {
			args = append(args, "AGGREGATION", aggregationArg)
		}
		if options.BucketDuration != 0 {
			args = append(args, options.BucketDuration)
		}
		if options.BucketTimestamp != nil {
			args = append(args, "BUCKETTIMESTAMP", options.BucketTimestamp)
		}
		if options.Empty {
			args = append(args, "EMPTY")
		}
	}
	cmd := newTSTimestampValueSliceCmd(ctx, args...)
	_ = c(ctx, cmd)
	return cmd
}

// TSRead - Returns samples at or after timestamp, in ascending order.
// timestamp is a non-negative Unix-ms integer or a sentinel (TSReadEarliest,
// TSReadLatest, TSReadNew).
// For more information - https://redis.io/commands/ts.read/
func (c cmdable) TSRead(ctx context.Context, key string, timestamp interface{}) *TSTimestampValueSliceCmd {
	args := []interface{}{"TS.READ", key, timestamp}
	cmd := newTSTimestampValueSliceCmd(ctx, args...)
	_ = c(ctx, cmd)
	return cmd
}

// TSReadWithArgs - TS.READ with the optional BLOCK and MAX_COUNT groups.
// When options.Block is set it waits for options.MinCount samples or until
// options.Timeout elapses. Blocking calls must not be used in a pipeline or MULTI.
// For more information - https://redis.io/commands/ts.read/
func (c cmdable) TSReadWithArgs(ctx context.Context, key string, timestamp interface{}, options *TSReadOptions) *TSTimestampValueSliceCmd {
	args := []interface{}{"TS.READ", key, timestamp}
	blocking := false
	var blockTimeout time.Duration
	if options != nil {
		if options.Block {
			blocking = true
			blockTimeout = options.Timeout
			minCount := options.MinCount
			if minCount <= 0 {
				minCount = 1
			}
			args = append(args, "BLOCK", formatMs(ctx, options.Timeout), minCount)
		}
		if options.MaxCount != 0 {
			args = append(args, "MAX_COUNT", options.MaxCount)
		}
	}
	cmd := newTSTimestampValueSliceCmd(ctx, args...)
	if blocking {
		cmd.setReadTimeout(blockTimeout)
	}
	_ = c(ctx, cmd)
	return cmd
}

type TSTimestampValueSliceCmd struct {
	baseCmd
	val []TSTimestampValue
}

func newTSTimestampValueSliceCmd(ctx context.Context, args ...interface{}) *TSTimestampValueSliceCmd {
	return &TSTimestampValueSliceCmd{
		baseCmd: baseCmd{
			ctx:     ctx,
			args:    args,
			cmdType: CmdTypeTSTimestampValueSlice,
		},
	}
}

func (cmd *TSTimestampValueSliceCmd) String() string {
	cmd.await()
	return cmdString(cmd, cmd.val)
}

func (cmd *TSTimestampValueSliceCmd) SetVal(val []TSTimestampValue) {
	cmd.val = val
}

func (cmd *TSTimestampValueSliceCmd) Result() ([]TSTimestampValue, error) {
	cmd.await()
	return cmd.val, cmd.err
}

func (cmd *TSTimestampValueSliceCmd) Val() []TSTimestampValue {
	cmd.await()
	return cmd.val
}

func (cmd *TSTimestampValueSliceCmd) readReply(rd *proto.Reader) (err error) {
	n, err := rd.ReadArrayLen()
	if err != nil {
		return err
	}
	cmd.val = make([]TSTimestampValue, n)
	for i := 0; i < n; i++ {
		itemLen, err := rd.ReadArrayLen()
		if err != nil {
			return err
		}

		timestamp, err := rd.ReadInt()
		if err != nil {
			return err
		}
		cmd.val[i].Timestamp = timestamp
		if itemLen == 2 {
			value, err := rd.ReadString()
			if err != nil {
				return err
			}
			cmd.val[i].Value, err = util.ParseStringToFloat(value)
			if err != nil {
				return err
			}
			continue
		}

		cmd.val[i].Values = make([]float64, itemLen-1)
		for j := 0; j < itemLen-1; j++ {
			value, err := rd.ReadString()
			if err != nil {
				return err
			}
			cmd.val[i].Values[j], err = util.ParseStringToFloat(value)
			if err != nil {
				return err
			}
		}
	}

	return nil
}

func (cmd *TSTimestampValueSliceCmd) Clone() Cmder {
	var val []TSTimestampValue
	if cmd.val != nil {
		val = make([]TSTimestampValue, len(cmd.val))
		copy(val, cmd.val)
		for i := range cmd.val {
			if cmd.val[i].Values != nil {
				val[i].Values = make([]float64, len(cmd.val[i].Values))
				copy(val[i].Values, cmd.val[i].Values)
			}
		}
	}
	return &TSTimestampValueSliceCmd{
		baseCmd: cmd.cloneBaseCmd(),
		val:     val,
	}
}

// TSMRange - Returns a range of samples from multiple time-series keys.
// For more information - https://redis.io/commands/ts.mrange/
func (c cmdable) TSMRange(ctx context.Context, fromTimestamp int, toTimestamp int, filterExpr []string) *MapStringSliceInterfaceCmd {
	args := []interface{}{"TS.MRANGE", fromTimestamp, toTimestamp, "FILTER"}
	for _, f := range filterExpr {
		args = append(args, f)
	}
	cmd := NewMapStringSliceInterfaceCmd(ctx, args...)
	_ = c(ctx, cmd)
	return cmd
}

// TSMRangeWithArgs - Returns a range of samples from multiple time-series keys.
// Options are set via TSMRangeOptions.
// For more information - https://redis.io/commands/ts.mrange/
func (c cmdable) TSMRangeWithArgs(ctx context.Context, fromTimestamp int, toTimestamp int, filterExpr []string, options *TSMRangeOptions) *MapStringSliceInterfaceCmd {
	args := []interface{}{"TS.MRANGE", fromTimestamp, toTimestamp}
	multiAggregationCount := 0
	if options != nil {
		if options.Latest {
			args = append(args, "LATEST")
		}
		if options.FilterByTS != nil {
			args = append(args, "FILTER_BY_TS")
			for _, f := range options.FilterByTS {
				args = append(args, f)
			}
		}
		if options.FilterByValue != nil {
			args = append(args, "FILTER_BY_VALUE")
			for _, f := range options.FilterByValue {
				args = append(args, f)
			}
		}
		if options.WithLabels {
			args = append(args, "WITHLABELS")
		}
		if options.SelectedLabels != nil {
			args = append(args, "SELECTED_LABELS")
			args = append(args, options.SelectedLabels...)
		}
		if options.Count != 0 {
			args = append(args, "COUNT", options.Count)
		}
		if options.Align != nil {
			args = append(args, "ALIGN", options.Align)
		}
		aggregationArg, count, err := formatAggregationArgs(options.Aggregator, options.Aggregators)
		if err != nil {
			cmd := NewMapStringSliceInterfaceCmd(ctx, args...)
			cmd.SetErr(err)
			return cmd
		}
		multiAggregationCount = count
		if aggregationArg != "" {
			args = append(args, "AGGREGATION", aggregationArg)
		}
		if options.BucketDuration != 0 {
			args = append(args, options.BucketDuration)
		}
		if options.BucketTimestamp != nil {
			args = append(args, "BUCKETTIMESTAMP", options.BucketTimestamp)
		}
		if options.Empty {
			args = append(args, "EMPTY")
		}
		if options.ExcludeEmpty {
			args = append(args, "EXCLUDEEMPTY")
		}
	}
	args = append(args, "FILTER")
	for _, f := range filterExpr {
		args = append(args, f)
	}
	if options != nil {
		if options.ExcludeEmpty && (options.GroupByLabel != nil || options.Reducer != nil) {
			cmd := NewMapStringSliceInterfaceCmd(ctx, args...)
			cmd.SetErr(errTSExcludeEmptyGroupBy)
			return cmd
		}
		if multiAggregationCount > 1 && (options.GroupByLabel != nil || options.Reducer != nil) {
			cmd := NewMapStringSliceInterfaceCmd(ctx, args...)
			cmd.SetErr(errTSMultiAggregationGroupBy)
			return cmd
		}
		if options.GroupByLabel != nil {
			args = append(args, "GROUPBY", options.GroupByLabel)
		}
		if options.Reducer != nil {
			args = append(args, "REDUCE", options.Reducer)
		}
	}
	cmd := NewMapStringSliceInterfaceCmd(ctx, args...)
	_ = c(ctx, cmd)
	return cmd
}

// TSMRevRange - Returns a range of samples from multiple time-series keys in reverse order.
// For more information - https://redis.io/commands/ts.mrevrange/
func (c cmdable) TSMRevRange(ctx context.Context, fromTimestamp int, toTimestamp int, filterExpr []string) *MapStringSliceInterfaceCmd {
	args := []interface{}{"TS.MREVRANGE", fromTimestamp, toTimestamp, "FILTER"}
	for _, f := range filterExpr {
		args = append(args, f)
	}
	cmd := NewMapStringSliceInterfaceCmd(ctx, args...)
	_ = c(ctx, cmd)
	return cmd
}

// TSMRevRangeWithArgs - Returns a range of samples from multiple time-series keys in reverse order.
// Options are set via TSMRevRangeOptions.
// For more information - https://redis.io/commands/ts.mrevrange/
func (c cmdable) TSMRevRangeWithArgs(ctx context.Context, fromTimestamp int, toTimestamp int, filterExpr []string, options *TSMRevRangeOptions) *MapStringSliceInterfaceCmd {
	args := []interface{}{"TS.MREVRANGE", fromTimestamp, toTimestamp}
	multiAggregationCount := 0
	if options != nil {
		if options.Latest {
			args = append(args, "LATEST")
		}
		if options.FilterByTS != nil {
			args = append(args, "FILTER_BY_TS")
			for _, f := range options.FilterByTS {
				args = append(args, f)
			}
		}
		if options.FilterByValue != nil {
			args = append(args, "FILTER_BY_VALUE")
			for _, f := range options.FilterByValue {
				args = append(args, f)
			}
		}
		if options.WithLabels {
			args = append(args, "WITHLABELS")
		}
		if options.SelectedLabels != nil {
			args = append(args, "SELECTED_LABELS")
			args = append(args, options.SelectedLabels...)
		}
		if options.Count != 0 {
			args = append(args, "COUNT", options.Count)
		}
		if options.Align != nil {
			args = append(args, "ALIGN", options.Align)
		}
		aggregationArg, count, err := formatAggregationArgs(options.Aggregator, options.Aggregators)
		if err != nil {
			cmd := NewMapStringSliceInterfaceCmd(ctx, args...)
			cmd.SetErr(err)
			return cmd
		}
		multiAggregationCount = count
		if aggregationArg != "" {
			args = append(args, "AGGREGATION", aggregationArg)
		}
		if options.BucketDuration != 0 {
			args = append(args, options.BucketDuration)
		}
		if options.BucketTimestamp != nil {
			args = append(args, "BUCKETTIMESTAMP", options.BucketTimestamp)
		}
		if options.Empty {
			args = append(args, "EMPTY")
		}
		if options.ExcludeEmpty {
			args = append(args, "EXCLUDEEMPTY")
		}
	}
	args = append(args, "FILTER")
	for _, f := range filterExpr {
		args = append(args, f)
	}
	if options != nil {
		if options.ExcludeEmpty && (options.GroupByLabel != nil || options.Reducer != nil) {
			cmd := NewMapStringSliceInterfaceCmd(ctx, args...)
			cmd.SetErr(errTSExcludeEmptyGroupBy)
			return cmd
		}
		if multiAggregationCount > 1 && (options.GroupByLabel != nil || options.Reducer != nil) {
			cmd := NewMapStringSliceInterfaceCmd(ctx, args...)
			cmd.SetErr(errTSMultiAggregationGroupBy)
			return cmd
		}
		if options.GroupByLabel != nil {
			args = append(args, "GROUPBY", options.GroupByLabel)
		}
		if options.Reducer != nil {
			args = append(args, "REDUCE", options.Reducer)
		}
	}
	cmd := NewMapStringSliceInterfaceCmd(ctx, args...)
	_ = c(ctx, cmd)
	return cmd
}

// TSMGet - Returns the last sample of multiple time-series keys.
// For more information - https://redis.io/commands/ts.mget/
func (c cmdable) TSMGet(ctx context.Context, filters []string) *MapStringSliceInterfaceCmd {
	args := []interface{}{"TS.MGET", "FILTER"}
	for _, f := range filters {
		args = append(args, f)
	}
	cmd := NewMapStringSliceInterfaceCmd(ctx, args...)
	_ = c(ctx, cmd)
	return cmd
}

// TSMGetWithArgs - Returns the last sample of multiple time-series keys with additional options.
// This function allows for specifying additional options such as:
// Latest, WithLabels and SelectedLabels.
// For more information - https://redis.io/commands/ts.mget/
func (c cmdable) TSMGetWithArgs(ctx context.Context, filters []string, options *TSMGetOptions) *MapStringSliceInterfaceCmd {
	args := []interface{}{"TS.MGET"}
	if options != nil {
		if options.Latest {
			args = append(args, "LATEST")
		}
		if options.WithLabels {
			args = append(args, "WITHLABELS")
		}
		if options.SelectedLabels != nil {
			args = append(args, "SELECTED_LABELS")
			args = append(args, options.SelectedLabels...)
		}
	}
	args = append(args, "FILTER")
	for _, f := range filters {
		args = append(args, f)
	}
	cmd := NewMapStringSliceInterfaceCmd(ctx, args...)
	_ = c(ctx, cmd)
	return cmd
}

// TSNRangePivotRow represents a single row in the pivot response from TS.NRANGE / TS.NREVRANGE.
// Timestamp is the row's timestamp. Without aggregation, Values holds one float64 per input key
// in input-key order. With aggregation, Values holds one float64 per requested (key, aggregator)
// pair, flattened in input-key order with each key's aggregators in spec order.
// Missing samples and missing aggregation buckets are represented as NaN.
type TSNRangePivotRow struct {
	Timestamp int64
	Values    []float64
}

type TSNRangePivotRowSliceCmd struct {
	baseCmd
	val []TSNRangePivotRow
}

func newTSNRangePivotRowSliceCmd(ctx context.Context, args ...interface{}) *TSNRangePivotRowSliceCmd {
	return &TSNRangePivotRowSliceCmd{
		baseCmd: baseCmd{
			ctx:     ctx,
			args:    args,
			cmdType: CmdTypeTSNRangePivotRowSlice,
		},
	}
}

func (cmd *TSNRangePivotRowSliceCmd) String() string {
	cmd.await()
	return cmdString(cmd, cmd.val)
}

func (cmd *TSNRangePivotRowSliceCmd) SetVal(val []TSNRangePivotRow) {
	cmd.val = val
}

func (cmd *TSNRangePivotRowSliceCmd) Result() ([]TSNRangePivotRow, error) {
	cmd.await()
	return cmd.val, cmd.err
}

func (cmd *TSNRangePivotRowSliceCmd) Val() []TSNRangePivotRow {
	cmd.await()
	return cmd.val
}

func (cmd *TSNRangePivotRowSliceCmd) readReply(rd *proto.Reader) error {
	n, err := rd.ReadArrayLen()
	if err != nil {
		return err
	}
	cmd.val = make([]TSNRangePivotRow, n)
	for i := 0; i < n; i++ {
		// Each row is a 2-element array: [timestamp, [value_0, value_1, ...]]
		if _, err = rd.ReadArrayLen(); err != nil {
			return err
		}
		timestamp, err := rd.ReadInt()
		if err != nil {
			return err
		}
		cmd.val[i].Timestamp = timestamp

		valCount, err := rd.ReadArrayLen()
		if err != nil {
			return err
		}
		cmd.val[i].Values = make([]float64, valCount)
		for j := 0; j < valCount; j++ {
			s, err := rd.ReadString()
			if err != nil {
				return err
			}
			cmd.val[i].Values[j], err = util.ParseStringToFloat(s)
			if err != nil {
				return err
			}
		}
	}
	return nil
}

func (cmd *TSNRangePivotRowSliceCmd) Clone() Cmder {
	var val []TSNRangePivotRow
	if cmd.val != nil {
		val = make([]TSNRangePivotRow, len(cmd.val))
		copy(val, cmd.val)
		for i := range cmd.val {
			if cmd.val[i].Values != nil {
				val[i].Values = make([]float64, len(cmd.val[i].Values))
				copy(val[i].Values, cmd.val[i].Values)
			}
		}
	}
	return &TSNRangePivotRowSliceCmd{
		baseCmd: cmd.cloneBaseCmd(),
		val:     val,
	}
}

// buildNRangeAggregationArgs validates and returns one aggregator spec string per key for
// TS.NRANGE / TS.NREVRANGE. The number of specs must equal the number of keys. Each spec
// lists one or more aggregators for its key and is emitted as a single comma-joined wire
// token; specs for different keys are separate wire tokens.
func buildNRangeAggregationArgs(keys []string, aggregators [][]Aggregator) ([]string, error) {
	if len(aggregators) != len(keys) {
		return nil, fmt.Errorf("redis: TS.NRANGE/TS.NREVRANGE requires exactly %d aggregator spec(s), got %d", len(keys), len(aggregators))
	}
	parts := make([]string, len(aggregators))
	for i, spec := range aggregators {
		if len(spec) == 0 {
			return nil, fmt.Errorf("redis: empty timeseries aggregator spec at index %d", i)
		}
		names := make([]string, len(spec))
		for j, agg := range spec {
			if agg == Invalid {
				return nil, fmt.Errorf("redis: invalid timeseries aggregator at index %d[%d]: Invalid (%d)", i, j, agg)
			}
			s := agg.String()
			if s == "" {
				return nil, fmt.Errorf("redis: invalid timeseries aggregator at index %d[%d]: %d", i, j, agg)
			}
			names[j] = s
		}
		parts[i] = strings.Join(names, ",")
	}
	return parts, nil
}

// appendNRangeOptions appends optional TS.NRANGE / TS.NREVRANGE arguments to args.
func appendNRangeOptions(
	args []interface{},
	keys []string,
	latest bool,
	filterByTS []int,
	filterByValue []float64,
	count int,
	align interface{},
	aggregators [][]Aggregator,
	bucketDuration int,
	bucketTimestamp interface{},
	empty bool,
) ([]interface{}, error) {
	if latest {
		args = append(args, "LATEST")
	}
	if len(filterByTS) > 0 {
		args = append(args, "FILTER_BY_TS")
		for _, ts := range filterByTS {
			args = append(args, ts)
		}
	}
	if len(filterByValue) > 0 {
		if len(filterByValue) != 2 {
			return args, fmt.Errorf("redis: FILTER_BY_VALUE requires exactly 2 elements [min, max], got %d", len(filterByValue))
		}
		args = append(args, "FILTER_BY_VALUE", filterByValue[0], filterByValue[1])
	}
	if count != 0 {
		args = append(args, "COUNT", count)
	}
	if align != nil {
		args = append(args, "ALIGN", align)
	}
	if len(aggregators) > 0 {
		aggParts, err := buildNRangeAggregationArgs(keys, aggregators)
		if err != nil {
			return args, err
		}
		args = append(args, "AGGREGATION")
		for _, a := range aggParts {
			args = append(args, a)
		}
		if bucketDuration != 0 {
			args = append(args, bucketDuration)
		}
		if bucketTimestamp != nil {
			args = append(args, "BUCKETTIMESTAMP", bucketTimestamp)
		}
		if empty {
			args = append(args, "EMPTY")
		}
	}
	return args, nil
}

// TSNRange - Queries multiple time-series keys and returns a pivot response in forward (ascending) order.
// For more information - https://redis.io/commands/ts.nrange/
func (c cmdable) TSNRange(ctx context.Context, keys []string, fromTimestamp interface{}, toTimestamp interface{}) *TSNRangePivotRowSliceCmd {
	args := make([]interface{}, 0, 3+len(keys))
	args = append(args, "TS.NRANGE", len(keys))
	for _, k := range keys {
		args = append(args, k)
	}
	args = append(args, fromTimestamp, toTimestamp)
	cmd := newTSNRangePivotRowSliceCmd(ctx, args...)
	cmd.SetFirstKeyPos(2)
	_ = c(ctx, cmd)
	return cmd
}

// TSNRangeWithArgs - Queries multiple time-series keys and returns a pivot response in forward (ascending) order with additional options.
// This function allows for specifying additional options such as:
// Latest, FilterByTS, FilterByValue, Count, Align, Aggregators, BucketDuration, BucketTimestamp and Empty.
// Aggregators must contain exactly one spec per key; each spec lists one or more aggregators
// for its key and is emitted as a single comma-joined wire token.
// For more information - https://redis.io/commands/ts.nrange/
func (c cmdable) TSNRangeWithArgs(ctx context.Context, keys []string, fromTimestamp interface{}, toTimestamp interface{}, options *TSNRangeOptions) *TSNRangePivotRowSliceCmd {
	args := make([]interface{}, 0, 3+len(keys))
	args = append(args, "TS.NRANGE", len(keys))
	for _, k := range keys {
		args = append(args, k)
	}
	args = append(args, fromTimestamp, toTimestamp)
	if options != nil {
		var err error
		args, err = appendNRangeOptions(args, keys,
			options.Latest, options.FilterByTS, options.FilterByValue,
			options.Count, options.Align, options.Aggregators,
			options.BucketDuration, options.BucketTimestamp, options.Empty)
		if err != nil {
			cmd := newTSNRangePivotRowSliceCmd(ctx, args...)
			cmd.SetErr(err)
			return cmd
		}
	}
	cmd := newTSNRangePivotRowSliceCmd(ctx, args...)
	cmd.SetFirstKeyPos(2)
	_ = c(ctx, cmd)
	return cmd
}

// TSNRevRange - Queries multiple time-series keys and returns a pivot response in reverse (descending) order.
// For more information - https://redis.io/commands/ts.nrevrange/
func (c cmdable) TSNRevRange(ctx context.Context, keys []string, fromTimestamp interface{}, toTimestamp interface{}) *TSNRangePivotRowSliceCmd {
	args := make([]interface{}, 0, 3+len(keys))
	args = append(args, "TS.NREVRANGE", len(keys))
	for _, k := range keys {
		args = append(args, k)
	}
	args = append(args, fromTimestamp, toTimestamp)
	cmd := newTSNRangePivotRowSliceCmd(ctx, args...)
	cmd.SetFirstKeyPos(2)
	_ = c(ctx, cmd)
	return cmd
}

// TSNRevRangeWithArgs - Queries multiple time-series keys and returns a pivot response in reverse (descending) order with additional options.
// This function allows for specifying additional options such as:
// Latest, FilterByTS, FilterByValue, Count, Align, Aggregators, BucketDuration, BucketTimestamp and Empty.
// Aggregators must contain exactly one spec per key; each spec lists one or more aggregators
// for its key and is emitted as a single comma-joined wire token.
// For more information - https://redis.io/commands/ts.nrevrange/
func (c cmdable) TSNRevRangeWithArgs(ctx context.Context, keys []string, fromTimestamp interface{}, toTimestamp interface{}, options *TSNRevRangeOptions) *TSNRangePivotRowSliceCmd {
	args := make([]interface{}, 0, 3+len(keys))
	args = append(args, "TS.NREVRANGE", len(keys))
	for _, k := range keys {
		args = append(args, k)
	}
	args = append(args, fromTimestamp, toTimestamp)
	if options != nil {
		var err error
		args, err = appendNRangeOptions(args, keys,
			options.Latest, options.FilterByTS, options.FilterByValue,
			options.Count, options.Align, options.Aggregators,
			options.BucketDuration, options.BucketTimestamp, options.Empty)
		if err != nil {
			cmd := newTSNRangePivotRowSliceCmd(ctx, args...)
			cmd.SetErr(err)
			return cmd
		}
	}
	cmd := newTSNRangePivotRowSliceCmd(ctx, args...)
	cmd.SetFirstKeyPos(2)
	_ = c(ctx, cmd)
	return cmd
}
