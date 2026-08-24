package redis

import (
	"context"
	"fmt"

	"github.com/redis/go-redis/v9/internal/proto"
)

type ProbabilisticCmdable interface {
	BFAdd(ctx context.Context, key string, element interface{}) *BoolCmd
	BFCard(ctx context.Context, key string) *IntCmd
	BFExists(ctx context.Context, key string, element interface{}) *BoolCmd
	BFInfo(ctx context.Context, key string) *BFInfoCmd
	BFInfoArg(ctx context.Context, key, option string) *BFInfoCmd
	BFInfoCapacity(ctx context.Context, key string) *BFInfoCmd
	BFInfoSize(ctx context.Context, key string) *BFInfoCmd
	BFInfoFilters(ctx context.Context, key string) *BFInfoCmd
	BFInfoItems(ctx context.Context, key string) *BFInfoCmd
	BFInfoExpansion(ctx context.Context, key string) *BFInfoCmd
	BFInsert(ctx context.Context, key string, options *BFInsertOptions, elements ...interface{}) *BoolSliceCmd
	BFMAdd(ctx context.Context, key string, elements ...interface{}) *BoolSliceCmd
	BFMExists(ctx context.Context, key string, elements ...interface{}) *BoolSliceCmd
	BFReserve(ctx context.Context, key string, errorRate float64, capacity int64) *StatusCmd
	BFReserveExpansion(ctx context.Context, key string, errorRate float64, capacity, expansion int64) *StatusCmd
	BFReserveNonScaling(ctx context.Context, key string, errorRate float64, capacity int64) *StatusCmd
	BFReserveWithArgs(ctx context.Context, key string, options *BFReserveOptions) *StatusCmd
	BFScanDump(ctx context.Context, key string, iterator int64) *ScanDumpCmd
	BFLoadChunk(ctx context.Context, key string, iterator int64, data interface{}) *StatusCmd

	CFAdd(ctx context.Context, key string, element interface{}) *BoolCmd
	CFAddNX(ctx context.Context, key string, element interface{}) *BoolCmd
	CFCount(ctx context.Context, key string, element interface{}) *IntCmd
	CFDel(ctx context.Context, key string, element interface{}) *BoolCmd
	CFExists(ctx context.Context, key string, element interface{}) *BoolCmd
	CFInfo(ctx context.Context, key string) *CFInfoCmd
	CFInsert(ctx context.Context, key string, options *CFInsertOptions, elements ...interface{}) *BoolSliceCmd
	CFInsertNX(ctx context.Context, key string, options *CFInsertOptions, elements ...interface{}) *IntSliceCmd
	CFMExists(ctx context.Context, key string, elements ...interface{}) *BoolSliceCmd
	CFReserve(ctx context.Context, key string, capacity int64) *StatusCmd
	CFReserveWithArgs(ctx context.Context, key string, options *CFReserveOptions) *StatusCmd
	CFReserveExpansion(ctx context.Context, key string, capacity int64, expansion int64) *StatusCmd
	CFReserveBucketSize(ctx context.Context, key string, capacity int64, bucketsize int64) *StatusCmd
	CFReserveMaxIterations(ctx context.Context, key string, capacity int64, maxiterations int64) *StatusCmd
	CFScanDump(ctx context.Context, key string, iterator int64) *ScanDumpCmd
	CFLoadChunk(ctx context.Context, key string, iterator int64, data interface{}) *StatusCmd

	CMSIncrBy(ctx context.Context, key string, elements ...interface{}) *IntSliceCmd
	CMSInfo(ctx context.Context, key string) *CMSInfoCmd
	CMSInitByDim(ctx context.Context, key string, width, height int64) *StatusCmd
	CMSInitByProb(ctx context.Context, key string, errorRate, probability float64) *StatusCmd
	CMSMerge(ctx context.Context, destKey string, sourceKeys ...string) *StatusCmd
	CMSMergeWithWeight(ctx context.Context, destKey string, sourceKeys map[string]int64) *StatusCmd
	CMSQuery(ctx context.Context, key string, elements ...interface{}) *IntSliceCmd

	TopKAdd(ctx context.Context, key string, elements ...interface{}) *StringSliceCmd
	TopKCount(ctx context.Context, key string, elements ...interface{}) *IntSliceCmd
	TopKIncrBy(ctx context.Context, key string, elements ...interface{}) *StringSliceCmd
	TopKInfo(ctx context.Context, key string) *TopKInfoCmd
	TopKList(ctx context.Context, key string) *StringSliceCmd
	TopKListWithCount(ctx context.Context, key string) *MapStringIntCmd
	TopKQuery(ctx context.Context, key string, elements ...interface{}) *BoolSliceCmd
	TopKReserve(ctx context.Context, key string, k int64) *StatusCmd
	TopKReserveWithOptions(ctx context.Context, key string, k int64, width, depth int64, decay float64) *StatusCmd

	TDigestAdd(ctx context.Context, key string, elements ...float64) *StatusCmd
	TDigestByRank(ctx context.Context, key string, rank ...uint64) *FloatSliceCmd
	TDigestByRevRank(ctx context.Context, key string, rank ...uint64) *FloatSliceCmd
	TDigestCDF(ctx context.Context, key string, elements ...float64) *FloatSliceCmd
	TDigestCreate(ctx context.Context, key string) *StatusCmd
	TDigestCreateWithCompression(ctx context.Context, key string, compression int64) *StatusCmd
	TDigestInfo(ctx context.Context, key string) *TDigestInfoCmd
	TDigestMax(ctx context.Context, key string) *FloatCmd
	TDigestMin(ctx context.Context, key string) *FloatCmd
	TDigestMerge(ctx context.Context, destKey string, options *TDigestMergeOptions, sourceKeys ...string) *StatusCmd
	TDigestQuantile(ctx context.Context, key string, elements ...float64) *FloatSliceCmd
	TDigestRank(ctx context.Context, key string, values ...float64) *IntSliceCmd
	TDigestReset(ctx context.Context, key string) *StatusCmd
	TDigestRevRank(ctx context.Context, key string, values ...float64) *IntSliceCmd
	TDigestTrimmedMean(ctx context.Context, key string, lowCutQuantile, highCutQuantile float64) *FloatCmd
}

type BFInsertOptions struct {
	Capacity   int64
	Error      float64
	Expansion  int64
	NonScaling bool
	NoCreate   bool
}

type BFReserveOptions struct {
	Capacity   int64
	Error      float64
	Expansion  int64
	NonScaling bool
}

type CFReserveOptions struct {
	Capacity      int64
	BucketSize    int64
	MaxIterations int64
	Expansion     int64
}

type CFInsertOptions struct {
	Capacity int64
	NoCreate bool
}

// -------------------------------------------
// Bloom filter commands
//-------------------------------------------

// BFReserve creates an empty Bloom filter with a single sub-filter
// for the initial specified capacity and with an upper bound error_rate.
// For more information - https://redis.io/commands/bf.reserve/
func (c cmdable) BFReserve(ctx context.Context, key string, errorRate float64, capacity int64) *StatusCmd {
	args := []interface{}{"BF.RESERVE", key, errorRate, capacity}
	cmd := NewStatusCmd(ctx, args...)
	_ = c(ctx, cmd)
	return cmd
}

// BFReserveExpansion creates an empty Bloom filter with a single sub-filter
// for the initial specified capacity and with an upper bound error_rate.
// This function also allows for specifying an expansion rate for the filter.
// For more information - https://redis.io/commands/bf.reserve/
func (c cmdable) BFReserveExpansion(ctx context.Context, key string, errorRate float64, capacity, expansion int64) *StatusCmd {
	args := []interface{}{"BF.RESERVE", key, errorRate, capacity, "EXPANSION", expansion}
	cmd := NewStatusCmd(ctx, args...)
	_ = c(ctx, cmd)
	return cmd
}

// BFReserveNonScaling creates an empty Bloom filter with a single sub-filter
// for the initial specified capacity and with an upper bound error_rate.
// This function also allows for specifying that the filter should not scale.
// For more information - https://redis.io/commands/bf.reserve/
func (c cmdable) BFReserveNonScaling(ctx context.Context, key string, errorRate float64, capacity int64) *StatusCmd {
	args := []interface{}{"BF.RESERVE", key, errorRate, capacity, "NONSCALING"}
	cmd := NewStatusCmd(ctx, args...)
	_ = c(ctx, cmd)
	return cmd
}

// BFReserveWithArgs creates an empty Bloom filter with a single sub-filter
// for the initial specified capacity and with an upper bound error_rate.
// This function also allows for specifying additional options such as expansion rate and non-scaling behavior.
// For more information - https://redis.io/commands/bf.reserve/
func (c cmdable) BFReserveWithArgs(ctx context.Context, key string, options *BFReserveOptions) *StatusCmd {
	args := []interface{}{"BF.RESERVE", key}
	if options != nil {
		args = append(args, options.Error, options.Capacity)
		if options.Expansion != 0 {
			args = append(args, "EXPANSION", options.Expansion)
		}
		if options.NonScaling {
			args = append(args, "NONSCALING")
		}
	}
	cmd := NewStatusCmd(ctx, args...)
	_ = c(ctx, cmd)
	return cmd
}

// BFAdd adds an item to a Bloom filter.
// For more information - https://redis.io/commands/bf.add/
func (c cmdable) BFAdd(ctx context.Context, key string, element interface{}) *BoolCmd {
	args := []interface{}{"BF.ADD", key, element}
	cmd := NewBoolCmd(ctx, args...)
	_ = c(ctx, cmd)
	return cmd
}

// BFCard returns the cardinality of a Bloom filter -
// number of items that were added to a Bloom filter and detected as unique
// (items that caused at least one bit to be set in at least one sub-filter).
// For more information - https://redis.io/commands/bf.card/
func (c cmdable) BFCard(ctx context.Context, key string) *IntCmd {
	args := []interface{}{"BF.CARD", key}
	cmd := NewIntCmd(ctx, args...)
	_ = c(ctx, cmd)
	return cmd
}

// BFExists determines whether a given item was added to a Bloom filter.
// For more information - https://redis.io/commands/bf.exists/
func (c cmdable) BFExists(ctx context.Context, key string, element interface{}) *BoolCmd {
	args := []interface{}{"BF.EXISTS", key, element}
	cmd := NewBoolCmd(ctx, args...)
	_ = c(ctx, cmd)
	return cmd
}

// BFLoadChunk restores a Bloom filter previously saved using BF.SCANDUMP.
// For more information - https://redis.io/commands/bf.loadchunk/
func (c cmdable) BFLoadChunk(ctx context.Context, key string, iterator int64, data interface{}) *StatusCmd {
	args := []interface{}{"BF.LOADCHUNK", key, iterator, data}
	cmd := NewStatusCmd(ctx, args...)
	_ = c(ctx, cmd)
	return cmd
}

// Begins an incremental save of the Bloom filter.
// This command is useful for large Bloom filters that cannot fit into the DUMP and RESTORE model.
// For more information - https://redis.io/commands/bf.scandump/
func (c cmdable) BFScanDump(ctx context.Context, key string, iterator int64) *ScanDumpCmd {
	args := []interface{}{"BF.SCANDUMP", key, iterator}
	cmd := newScanDumpCmd(ctx, args...)
	_ = c(ctx, cmd)
	return cmd
}

type ScanDump struct {
	Iter int64
	Data string
}

type ScanDumpCmd struct {
	baseCmd

	val ScanDump
}

func newScanDumpCmd(ctx context.Context, args ...interface{}) *ScanDumpCmd {
	return &ScanDumpCmd{
		baseCmd: baseCmd{
			ctx:     ctx,
			args:    args,
			cmdType: CmdTypeScanDump,
		},
	}
}

func (cmd *ScanDumpCmd) String() string {
	cmd.await()
	return cmdString(cmd, cmd.val)
}

func (cmd *ScanDumpCmd) SetVal(val ScanDump) {
	cmd.val = val
}

func (cmd *ScanDumpCmd) Result() (ScanDump, error) {
	cmd.await()
	return cmd.val, cmd.err
}

func (cmd *ScanDumpCmd) Val() ScanDump {
	cmd.await()
	return cmd.val
}

func (cmd *ScanDumpCmd) readReply(rd *proto.Reader) (err error) {
	n, err := rd.ReadMapLen()
	if err != nil {
		return err
	}
	cmd.val = ScanDump{}
	for i := 0; i < n; i++ {
		iter, err := rd.ReadInt()
		if err != nil {
			return err
		}
		data, err := rd.ReadString()
		if err != nil {
			return err
		}
		cmd.val.Data = data
		cmd.val.Iter = iter

	}

	return nil
}

func (cmd *ScanDumpCmd) Clone() Cmder {
	return &ScanDumpCmd{
		baseCmd: cmd.cloneBaseCmd(),
		val:     cmd.val, // ScanDump is a simple struct, can be copied directly
	}
}

// Returns information about a Bloom filter.
// For more information - https://redis.io/commands/bf.info/
func (c cmdable) BFInfo(ctx context.Context, key string) *BFInfoCmd {
	args := []interface{}{"BF.INFO", key}
	cmd := NewBFInfoCmd(ctx, args...)
	_ = c(ctx, cmd)
	return cmd
}

type BFInfo struct {
	Capacity      int64
	Size          int64
	Filters       int64
	ItemsInserted int64
	ExpansionRate int64
}

type BFInfoCmd struct {
	baseCmd

	val BFInfo
}

func NewBFInfoCmd(ctx context.Context, args ...interface{}) *BFInfoCmd {
	return &BFInfoCmd{
		baseCmd: baseCmd{
			ctx:     ctx,
			args:    args,
			cmdType: CmdTypeBFInfo,
		},
	}
}

func (cmd *BFInfoCmd) SetVal(val BFInfo) {
	cmd.val = val
}

func (cmd *BFInfoCmd) String() string {
	cmd.await()
	return cmdString(cmd, cmd.val)
}

func (cmd *BFInfoCmd) Val() BFInfo {
	cmd.await()
	return cmd.val
}

func (cmd *BFInfoCmd) Result() (BFInfo, error) {
	cmd.await()
	return cmd.val, cmd.err
}

func (cmd *BFInfoCmd) readReply(rd *proto.Reader) (err error) {
	result := BFInfo{}

	// Create a mapping from key names to pointers of struct fields
	respMapping := map[string]*int64{
		"Capacity":                 &result.Capacity,
		"CAPACITY":                 &result.Capacity,
		"Size":                     &result.Size,
		"SIZE":                     &result.Size,
		"Number of filters":        &result.Filters,
		"FILTERS":                  &result.Filters,
		"Number of items inserted": &result.ItemsInserted,
		"ITEMS":                    &result.ItemsInserted,
		"Expansion rate":           &result.ExpansionRate,
		"EXPANSION":                &result.ExpansionRate,
	}

	// Helper function to read and assign a value based on the key
	readAndAssignValue := func(key string) error {
		fieldPtr, exists := respMapping[key]
		if !exists {
			return fmt.Errorf("redis: BLOOM.INFO unexpected key %s", key)
		}

		// Read the integer and assign to the field via pointer dereferencing
		val, err := rd.ReadInt()
		if err != nil {
			return err
		}
		*fieldPtr = val
		return nil
	}

	readType, err := rd.PeekReplyType()
	if err != nil {
		return err
	}

	if len(cmd.args) > 2 && readType == proto.RespArray {
		n, err := rd.ReadArrayLen()
		if err != nil {
			return err
		}
		if key, ok := cmd.args[2].(string); ok && n == 1 {
			if err := readAndAssignValue(key); err != nil {
				return err
			}
		} else {
			return fmt.Errorf("redis: BLOOM.INFO invalid argument key type")
		}
	} else {
		n, err := rd.ReadMapLen()
		if err != nil {
			return err
		}
		for i := 0; i < n; i++ {
			key, err := rd.ReadString()
			if err != nil {
				return err
			}
			if err := readAndAssignValue(key); err != nil {
				return err
			}
		}
	}

	cmd.val = result
	return nil
}

func (cmd *BFInfoCmd) Clone() Cmder {
	return &BFInfoCmd{
		baseCmd: cmd.cloneBaseCmd(),
		val:     cmd.val, // BFInfo is a simple struct, can be copied directly
	}
}

// BFInfoCapacity returns information about the capacity of a Bloom filter.
// For more information - https://redis.io/commands/bf.info/
func (c cmdable) BFInfoCapacity(ctx context.Context, key string) *BFInfoCmd {
	return c.BFInfoArg(ctx, key, "CAPACITY")
}

// BFInfoSize returns information about the size of a Bloom filter.
// For more information - https://redis.io/commands/bf.info/
func (c cmdable) BFInfoSize(ctx context.Context, key string) *BFInfoCmd {
	return c.BFInfoArg(ctx, key, "SIZE")
}

// BFInfoFilters returns information about the filters of a Bloom filter.
// For more information - https://redis.io/commands/bf.info/
func (c cmdable) BFInfoFilters(ctx context.Context, key string) *BFInfoCmd {
	return c.BFInfoArg(ctx, key, "FILTERS")
}

// BFInfoItems returns information about the items of a Bloom filter.
// For more information - https://redis.io/commands/bf.info/
func (c cmdable) BFInfoItems(ctx context.Context, key string) *BFInfoCmd {
	return c.BFInfoArg(ctx, key, "ITEMS")
}

// BFInfoExpansion returns information about the expansion rate of a Bloom filter.
// For more information - https://redis.io/commands/bf.info/
func (c cmdable) BFInfoExpansion(ctx context.Context, key string) *BFInfoCmd {
	return c.BFInfoArg(ctx, key, "EXPANSION")
}

// BFInfoArg returns information about a specific option of a Bloom filter.
// For more information - https://redis.io/commands/bf.info/
func (c cmdable) BFInfoArg(ctx context.Context, key, option string) *BFInfoCmd {
	args := []interface{}{"BF.INFO", key, option}
	cmd := NewBFInfoCmd(ctx, args...)
	_ = c(ctx, cmd)
	return cmd
}

// BFInsert inserts elements into a Bloom filter.
// This function also allows for specifying additional options such as:
// capacity, error rate, expansion rate, and non-scaling behavior.
// For more information - https://redis.io/commands/bf.insert/
func (c cmdable) BFInsert(ctx context.Context, key string, options *BFInsertOptions, elements ...interface{}) *BoolSliceCmd {
	args := []interface{}{"BF.INSERT", key}
	if options != nil {
		if options.Capacity != 0 {
			args = append(args, "CAPACITY", options.Capacity)
		}
		if options.Error != 0 {
			args = append(args, "ERROR", options.Error)
		}
		if options.Expansion != 0 {
			args = append(args, "EXPANSION", options.Expansion)
		}
		if options.NoCreate {
			args = append(args, "NOCREATE")
		}
		if options.NonScaling {
			args = append(args, "NONSCALING")
		}
	}
	args = append(args, "ITEMS")
	args = append(args, elements...)

	cmd := NewBoolSliceCmd(ctx, args...)
	_ = c(ctx, cmd)
	return cmd
}

// BFMAdd adds multiple elements to a Bloom filter.
// Returns an array of booleans indicating whether each element was added to the filter or not.
// For more information - https://redis.io/commands/bf.madd/
func (c cmdable) BFMAdd(ctx context.Context, key string, elements ...interface{}) *BoolSliceCmd {
	args := []interface{}{"BF.MADD", key}
	args = append(args, elements...)
	cmd := NewBoolSliceCmd(ctx, args...)
	_ = c(ctx, cmd)
	return cmd
}

// BFMExists check if multiple elements exist in a Bloom filter.
// Returns an array of booleans indicating whether each element exists in the filter or not.
// For more information - https://redis.io/commands/bf.mexists/
func (c cmdable) BFMExists(ctx context.Context, key string, elements ...interface{}) *BoolSliceCmd {
	args := []interface{}{"BF.MEXISTS", key}
	args = append(args, elements...)

	cmd := NewBoolSliceCmd(ctx, args...)
	_ = c(ctx, cmd)
	return cmd
}

// -------------------------------------------
// Cuckoo filter commands
//-------------------------------------------

// CFReserve creates an empty Cuckoo filter with the specified capacity.
// For more information - https://redis.io/commands/cf.reserve/
func (c cmdable) CFReserve(ctx context.Context, key string, capacity int64) *StatusCmd {
	args := []interface{}{"CF.RESERVE", key, capacity}
	cmd := NewStatusCmd(ctx, args...)
	_ = c(ctx, cmd)
	return cmd
}

// CFReserveExpansion creates an empty Cuckoo filter with the specified capacity and expansion rate.
// For more information - https://redis.io/commands/cf.reserve/
func (c cmdable) CFReserveExpansion(ctx context.Context, key string, capacity int64, expansion int64) *StatusCmd {
	args := []interface{}{"CF.RESERVE", key, capacity, "EXPANSION", expansion}
	cmd := NewStatusCmd(ctx, args...)
	_ = c(ctx, cmd)
	return cmd
}

// CFReserveBucketSize creates an empty Cuckoo filter with the specified capacity and bucket size.
// For more information - https://redis.io/commands/cf.reserve/
func (c cmdable) CFReserveBucketSize(ctx context.Context, key string, capacity int64, bucketsize int64) *StatusCmd {
	args := []interface{}{"CF.RESERVE", key, capacity, "BUCKETSIZE", bucketsize}
	cmd := NewStatusCmd(ctx, args...)
	_ = c(ctx, cmd)
	return cmd
}

// CFReserveMaxIterations creates an empty Cuckoo filter with the specified capacity and maximum number of iterations.
// For more information - https://redis.io/commands/cf.reserve/
func (c cmdable) CFReserveMaxIterations(ctx context.Context, key string, capacity int64, maxiterations int64) *StatusCmd {
	args := []interface{}{"CF.RESERVE", key, capacity, "MAXITERATIONS", maxiterations}
	cmd := NewStatusCmd(ctx, args...)
	_ = c(ctx, cmd)
	return cmd
}

// CFReserveWithArgs creates an empty Cuckoo filter with the specified options.
// This function allows for specifying additional options such as bucket size and maximum number of iterations.
// For more information - https://redis.io/commands/cf.reserve/
func (c cmdable) CFReserveWithArgs(ctx context.Context, key string, options *CFReserveOptions) *StatusCmd {
	args := []interface{}{"CF.RESERVE", key, options.Capacity}
	if options.BucketSize != 0 {
		args = append(args, "BUCKETSIZE", options.BucketSize)
	}
	if options.MaxIterations != 0 {
		args = append(args, "MAXITERATIONS", options.MaxIterations)
	}
	if options.Expansion != 0 {
		args = append(args, "EXPANSION", options.Expansion)
	}
	cmd := NewStatusCmd(ctx, args...)
	_ = c(ctx, cmd)
	return cmd
}

// CFAdd adds an element to a Cuckoo filter.
// Returns true if the element was added to the filter or false if it already exists in the filter.
// For more information - https://redis.io/commands/cf.add/
func (c cmdable) CFAdd(ctx context.Context, key string, element interface{}) *BoolCmd {
	args := []interface{}{"CF.ADD", key, element}
	cmd := NewBoolCmd(ctx, args...)
	_ = c(ctx, cmd)
	return cmd
}

// CFAddNX adds an element to a Cuckoo filter only if it does not already exist in the filter.
// Returns true if the element was added to the filter or false if it already exists in the filter.
// For more information - https://redis.io/commands/cf.addnx/
func (c cmdable) CFAddNX(ctx context.Context, key string, element interface{}) *BoolCmd {
	args := []interface{}{"CF.ADDNX", key, element}
	cmd := NewBoolCmd(ctx, args...)
	_ = c(ctx, cmd)
	return cmd
}

// CFCount returns an estimate of the number of times an element may be in a Cuckoo Filter.
// For more information - https://redis.io/commands/cf.count/
func (c cmdable) CFCount(ctx context.Context, key string, element interface{}) *IntCmd {
	args := []interface{}{"CF.COUNT", key, element}
	cmd := NewIntCmd(ctx, args...)
	_ = c(ctx, cmd)
	return cmd
}

// CFDel deletes an item once from the cuckoo filter.
// For more information - https://redis.io/commands/cf.del/
func (c cmdable) CFDel(ctx context.Context, key string, element interface{}) *BoolCmd {
	args := []interface{}{"CF.DEL", key, element}
	cmd := NewBoolCmd(ctx, args...)
	_ = c(ctx, cmd)
	return cmd
}

// CFExists determines whether an item may exist in the Cuckoo Filter or not.
// For more information - https://redis.io/commands/cf.exists/
func (c cmdable) CFExists(ctx context.Context, key string, element interface{}) *BoolCmd {
	args := []interface{}{"CF.EXISTS", key, element}
	cmd := NewBoolCmd(ctx, args...)
	_ = c(ctx, cmd)
	return cmd
}

// CFLoadChunk restores a filter previously saved using SCANDUMP.
// For more information - https://redis.io/commands/cf.loadchunk/
func (c cmdable) CFLoadChunk(ctx context.Context, key string, iterator int64, data interface{}) *StatusCmd {
	args := []interface{}{"CF.LOADCHUNK", key, iterator, data}
	cmd := NewStatusCmd(ctx, args...)
	_ = c(ctx, cmd)
	return cmd
}

// CFScanDump begins an incremental save of the cuckoo filter.
// For more information - https://redis.io/commands/cf.scandump/
func (c cmdable) CFScanDump(ctx context.Context, key string, iterator int64) *ScanDumpCmd {
	args := []interface{}{"CF.SCANDUMP", key, iterator}
	cmd := newScanDumpCmd(ctx, args...)
	_ = c(ctx, cmd)
	return cmd
}

type CFInfo struct {
	Size             int64
	NumBuckets       int64
	NumFilters       int64
	NumItemsInserted int64
	NumItemsDeleted  int64
	BucketSize       int64
	ExpansionRate    int64
	MaxIteration     int64
}

type CFInfoCmd struct {
	baseCmd

	val CFInfo
}

func NewCFInfoCmd(ctx context.Context, args ...interface{}) *CFInfoCmd {
	return &CFInfoCmd{
		baseCmd: baseCmd{
			ctx:     ctx,
			args:    args,
			cmdType: CmdTypeCFInfo,
		},
	}
}

func (cmd *CFInfoCmd) SetVal(val CFInfo) {
	cmd.val = val
}

func (cmd *CFInfoCmd) String() string {
	cmd.await()
	return cmdString(cmd, cmd.val)
}

func (cmd *CFInfoCmd) Val() CFInfo {
	cmd.await()
	return cmd.val
}

func (cmd *CFInfoCmd) Result() (CFInfo, error) {
	cmd.await()
	return cmd.val, cmd.err
}

func (cmd *CFInfoCmd) readReply(rd *proto.Reader) (err error) {
	n, err := rd.ReadMapLen()
	if err != nil {
		return err
	}

	var key string
	var result CFInfo
	for f := 0; f < n; f++ {
		key, err = rd.ReadString()
		if err != nil {
			return err
		}

		switch key {
		case "Size":
			result.Size, err = rd.ReadInt()
		case "Number of buckets":
			result.NumBuckets, err = rd.ReadInt()
		case "Number of filters":
			result.NumFilters, err = rd.ReadInt()
		case "Number of items inserted":
			result.NumItemsInserted, err = rd.ReadInt()
		case "Number of items deleted":
			result.NumItemsDeleted, err = rd.ReadInt()
		case "Bucket size":
			result.BucketSize, err = rd.ReadInt()
		case "Expansion rate":
			result.ExpansionRate, err = rd.ReadInt()
		case "Max iterations":
			result.MaxIteration, err = rd.ReadInt()

		default:
			return fmt.Errorf("redis: CF.INFO unexpected key %s", key)
		}

		if err != nil {
			return err
		}
	}

	cmd.val = result
	return nil
}

func (cmd *CFInfoCmd) Clone() Cmder {
	return &CFInfoCmd{
		baseCmd: cmd.cloneBaseCmd(),
		val:     cmd.val, // CFInfo is a simple struct, can be copied directly
	}
}

// CFInfo returns information about a Cuckoo filter.
// For more information - https://redis.io/commands/cf.info/
func (c cmdable) CFInfo(ctx context.Context, key string) *CFInfoCmd {
	args := []interface{}{"CF.INFO", key}
	cmd := NewCFInfoCmd(ctx, args...)
	_ = c(ctx, cmd)
	return cmd
}

// CFInsert inserts elements into a Cuckoo filter.
// This function also allows for specifying additional options such as capacity, error rate, expansion rate, and non-scaling behavior.
// Returns an array of booleans indicating whether each element was added to the filter or not.
// For more information - https://redis.io/commands/cf.insert/
func (c cmdable) CFInsert(ctx context.Context, key string, options *CFInsertOptions, elements ...interface{}) *BoolSliceCmd {
	args := []interface{}{"CF.INSERT", key}
	args = c.getCfInsertWithArgs(args, options, elements...)

	cmd := NewBoolSliceCmd(ctx, args...)
	_ = c(ctx, cmd)
	return cmd
}

// CFInsertNX inserts elements into a Cuckoo filter only if they do not already exist in the filter.
// This function also allows for specifying additional options such as:
// capacity, error rate, expansion rate, and non-scaling behavior.
// Returns an array of integers indicating whether each element was added to the filter or not.
// For more information - https://redis.io/commands/cf.insertnx/
func (c cmdable) CFInsertNX(ctx context.Context, key string, options *CFInsertOptions, elements ...interface{}) *IntSliceCmd {
	args := []interface{}{"CF.INSERTNX", key}
	args = c.getCfInsertWithArgs(args, options, elements...)

	cmd := NewIntSliceCmd(ctx, args...)
	_ = c(ctx, cmd)
	return cmd
}

func (c cmdable) getCfInsertWithArgs(args []interface{}, options *CFInsertOptions, elements ...interface{}) []interface{} {
	if options != nil {
		if options.Capacity != 0 {
			args = append(args, "CAPACITY", options.Capacity)
		}
		if options.NoCreate {
			args = append(args, "NOCREATE")
		}
	}
	args = append(args, "ITEMS")
	args = append(args, elements...)

	return args
}

// CFMExists check if multiple elements exist in a Cuckoo filter.
// Returns an array of booleans indicating whether each element exists in the filter or not.
// For more information - https://redis.io/commands/cf.mexists/
func (c cmdable) CFMExists(ctx context.Context, key string, elements ...interface{}) *BoolSliceCmd {
	args := []interface{}{"CF.MEXISTS", key}
	args = append(args, elements...)
	cmd := NewBoolSliceCmd(ctx, args...)
	_ = c(ctx, cmd)
	return cmd
}

// -------------------------------------------
// CMS commands
//-------------------------------------------

// CMSIncrBy increments the count of one or more items in a Count-Min Sketch filter.
// Returns an array of integers representing the updated count of each item.
// For more information - https://redis.io/commands/cms.incrby/
func (c cmdable) CMSIncrBy(ctx context.Context, key string, elements ...interface{}) *IntSliceCmd {
	args := make([]interface{}, 2, 2+len(elements))
	args[0] = "CMS.INCRBY"
	args[1] = key
	args = appendArgs(args, elements)

	cmd := NewIntSliceCmd(ctx, args...)
	_ = c(ctx, cmd)
	return cmd
}

type CMSInfo struct {
	Width int64
	Depth int64
	Count int64
}

type CMSInfoCmd struct {
	baseCmd

	val CMSInfo
}

func NewCMSInfoCmd(ctx context.Context, args ...interface{}) *CMSInfoCmd {
	return &CMSInfoCmd{
		baseCmd: baseCmd{
			ctx:     ctx,
			args:    args,
			cmdType: CmdTypeCMSInfo,
		},
	}
}

func (cmd *CMSInfoCmd) SetVal(val CMSInfo) {
	cmd.val = val
}

func (cmd *CMSInfoCmd) String() string {
	cmd.await()
	return cmdString(cmd, cmd.val)
}

func (cmd *CMSInfoCmd) Val() CMSInfo {
	cmd.await()
	return cmd.val
}

func (cmd *CMSInfoCmd) Result() (CMSInfo, error) {
	cmd.await()
	return cmd.val, cmd.err
}

func (cmd *CMSInfoCmd) readReply(rd *proto.Reader) (err error) {
	n, err := rd.ReadMapLen()
	if err != nil {
		return err
	}

	var key string
	var result CMSInfo
	for f := 0; f < n; f++ {
		key, err = rd.ReadString()
		if err != nil {
			return err
		}

		switch key {
		case "width":
			result.Width, err = rd.ReadInt()
		case "depth":
			result.Depth, err = rd.ReadInt()
		case "count":
			result.Count, err = rd.ReadInt()
		default:
			return fmt.Errorf("redis: CMS.INFO unexpected key %s", key)
		}

		if err != nil {
			return err
		}
	}

	cmd.val = result
	return nil
}

func (cmd *CMSInfoCmd) Clone() Cmder {
	return &CMSInfoCmd{
		baseCmd: cmd.cloneBaseCmd(),
		val:     cmd.val, // CMSInfo is a simple struct, can be copied directly
	}
}

// CMSInfo returns information about a Count-Min Sketch filter.
// For more information - https://redis.io/commands/cms.info/
func (c cmdable) CMSInfo(ctx context.Context, key string) *CMSInfoCmd {
	args := []interface{}{"CMS.INFO", key}
	cmd := NewCMSInfoCmd(ctx, args...)
	_ = c(ctx, cmd)
	return cmd
}

// CMSInitByDim creates an empty Count-Min Sketch filter with the specified dimensions.
// For more information - https://redis.io/commands/cms.initbydim/
func (c cmdable) CMSInitByDim(ctx context.Context, key string, width, depth int64) *StatusCmd {
	args := []interface{}{"CMS.INITBYDIM", key, width, depth}
	cmd := NewStatusCmd(ctx, args...)
	_ = c(ctx, cmd)
	return cmd
}

// CMSInitByProb creates an empty Count-Min Sketch filter with the specified error rate and probability.
// For more information - https://redis.io/commands/cms.initbyprob/
func (c cmdable) CMSInitByProb(ctx context.Context, key string, errorRate, probability float64) *StatusCmd {
	args := []interface{}{"CMS.INITBYPROB", key, errorRate, probability}
	cmd := NewStatusCmd(ctx, args...)
	_ = c(ctx, cmd)
	return cmd
}

// CMSMerge merges multiple Count-Min Sketch filters into a single filter.
// The destination filter must not exist and will be created with the dimensions of the first source filter.
// The number of items in each source filter must be equal.
// Returns OK on success or an error if the filters could not be merged.
// For more information - https://redis.io/commands/cms.merge/
func (c cmdable) CMSMerge(ctx context.Context, destKey string, sourceKeys ...string) *StatusCmd {
	args := []interface{}{"CMS.MERGE", destKey, len(sourceKeys)}
	for _, s := range sourceKeys {
		args = append(args, s)
	}
	cmd := NewStatusCmd(ctx, args...)
	_ = c(ctx, cmd)
	return cmd
}

// CMSMergeWithWeight merges multiple Count-Min Sketch filters into a single filter with weights for each source filter.
// The destination filter must not exist and will be created with the dimensions of the first source filter.
// The number of items in each source filter must be equal.
// Returns OK on success or an error if the filters could not be merged.
// For more information - https://redis.io/commands/cms.merge/
func (c cmdable) CMSMergeWithWeight(ctx context.Context, destKey string, sourceKeys map[string]int64) *StatusCmd {
	args := make([]interface{}, 0, 4+(len(sourceKeys)*2+1))
	args = append(args, "CMS.MERGE", destKey, len(sourceKeys))

	if len(sourceKeys) > 0 {
		sk := make([]interface{}, len(sourceKeys))
		sw := make([]interface{}, len(sourceKeys))

		i := 0
		for k, w := range sourceKeys {
			sk[i] = k
			sw[i] = w
			i++
		}

		args = append(args, sk...)
		args = append(args, "WEIGHTS")
		args = append(args, sw...)
	}

	cmd := NewStatusCmd(ctx, args...)
	_ = c(ctx, cmd)
	return cmd
}

// CMSQuery returns count for item(s).
// For more information - https://redis.io/commands/cms.query/
func (c cmdable) CMSQuery(ctx context.Context, key string, elements ...interface{}) *IntSliceCmd {
	args := []interface{}{"CMS.QUERY", key}
	args = append(args, elements...)
	cmd := NewIntSliceCmd(ctx, args...)
	_ = c(ctx, cmd)
	return cmd
}

// -------------------------------------------
// TopK commands
//--------------------------------------------

// TopKAdd adds one or more elements to a Top-K filter.
// Returns an array of strings representing the items that were removed from the filter, if any.
// For more information - https://redis.io/commands/topk.add/
func (c cmdable) TopKAdd(ctx context.Context, key string, elements ...interface{}) *StringSliceCmd {
	args := make([]interface{}, 2, 2+len(elements))
	args[0] = "TOPK.ADD"
	args[1] = key
	args = appendArgs(args, elements)

	cmd := NewStringSliceCmd(ctx, args...)
	_ = c(ctx, cmd)
	return cmd
}

// TopKReserve creates an empty Top-K filter with the specified number of top items to keep.
// For more information - https://redis.io/commands/topk.reserve/
func (c cmdable) TopKReserve(ctx context.Context, key string, k int64) *StatusCmd {
	args := []interface{}{"TOPK.RESERVE", key, k}

	cmd := NewStatusCmd(ctx, args...)
	_ = c(ctx, cmd)
	return cmd
}

// TopKReserveWithOptions creates an empty Top-K filter with the specified number of top items to keep and additional options.
// This function allows for specifying additional options such as width, depth and decay.
// For more information - https://redis.io/commands/topk.reserve/
func (c cmdable) TopKReserveWithOptions(ctx context.Context, key string, k int64, width, depth int64, decay float64) *StatusCmd {
	args := []interface{}{"TOPK.RESERVE", key, k, width, depth, decay}

	cmd := NewStatusCmd(ctx, args...)
	_ = c(ctx, cmd)
	return cmd
}

type TopKInfo struct {
	K     int64
	Width int64
	Depth int64
	Decay float64
}

type TopKInfoCmd struct {
	baseCmd

	val TopKInfo
}

func NewTopKInfoCmd(ctx context.Context, args ...interface{}) *TopKInfoCmd {
	return &TopKInfoCmd{
		baseCmd: baseCmd{
			ctx:     ctx,
			args:    args,
			cmdType: CmdTypeTopKInfo,
		},
	}
}

func (cmd *TopKInfoCmd) SetVal(val TopKInfo) {
	cmd.val = val
}

func (cmd *TopKInfoCmd) String() string {
	cmd.await()
	return cmdString(cmd, cmd.val)
}

func (cmd *TopKInfoCmd) Val() TopKInfo {
	cmd.await()
	return cmd.val
}

func (cmd *TopKInfoCmd) Result() (TopKInfo, error) {
	cmd.await()
	return cmd.val, cmd.err
}

func (cmd *TopKInfoCmd) readReply(rd *proto.Reader) (err error) {
	n, err := rd.ReadMapLen()
	if err != nil {
		return err
	}

	var key string
	var result TopKInfo
	for f := 0; f < n; f++ {
		key, err = rd.ReadString()
		if err != nil {
			return err
		}

		switch key {
		case "k":
			result.K, err = rd.ReadInt()
		case "width":
			result.Width, err = rd.ReadInt()
		case "depth":
			result.Depth, err = rd.ReadInt()
		case "decay":
			result.Decay, err = rd.ReadFloat()
		default:
			return fmt.Errorf("redis: topk.info unexpected key %s", key)
		}

		if err != nil {
			return err
		}
	}

	cmd.val = result
	return nil
}

func (cmd *TopKInfoCmd) Clone() Cmder {
	return &TopKInfoCmd{
		baseCmd: cmd.cloneBaseCmd(),
		val:     cmd.val, // TopKInfo is a simple struct, can be copied directly
	}
}

// TopKInfo returns information about a Top-K filter.
// For more information - https://redis.io/commands/topk.info/
func (c cmdable) TopKInfo(ctx context.Context, key string) *TopKInfoCmd {
	args := []interface{}{"TOPK.INFO", key}

	cmd := NewTopKInfoCmd(ctx, args...)
	_ = c(ctx, cmd)
	return cmd
}

// TopKQuery check if multiple elements exist in a Top-K filter.
// Returns an array of booleans indicating whether each element exists in the filter or not.
// For more information - https://redis.io/commands/topk.query/
func (c cmdable) TopKQuery(ctx context.Context, key string, elements ...interface{}) *BoolSliceCmd {
	args := make([]interface{}, 2, 2+len(elements))
	args[0] = "TOPK.QUERY"
	args[1] = key
	args = appendArgs(args, elements)

	cmd := NewBoolSliceCmd(ctx, args...)
	_ = c(ctx, cmd)
	return cmd
}

// TopKCount returns an estimate of the number of times an item may be in a Top-K filter.
// For more information - https://redis.io/commands/topk.count/
func (c cmdable) TopKCount(ctx context.Context, key string, elements ...interface{}) *IntSliceCmd {
	args := make([]interface{}, 2, 2+len(elements))
	args[0] = "TOPK.COUNT"
	args[1] = key
	args = appendArgs(args, elements)

	cmd := NewIntSliceCmd(ctx, args...)
	_ = c(ctx, cmd)
	return cmd
}

// TopKIncrBy increases the count of one or more items in a Top-K filter.
// For more information - https://redis.io/commands/topk.incrby/
func (c cmdable) TopKIncrBy(ctx context.Context, key string, elements ...interface{}) *StringSliceCmd {
	args := make([]interface{}, 2, 2+len(elements))
	args[0] = "TOPK.INCRBY"
	args[1] = key
	args = appendArgs(args, elements)

	cmd := NewStringSliceCmd(ctx, args...)
	_ = c(ctx, cmd)
	return cmd
}

// TopKList returns all items in Top-K list.
// For more information - https://redis.io/commands/topk.list/
func (c cmdable) TopKList(ctx context.Context, key string) *StringSliceCmd {
	args := []interface{}{"TOPK.LIST", key}

	cmd := NewStringSliceCmd(ctx, args...)
	_ = c(ctx, cmd)
	return cmd
}

// TopKListWithCount returns all items in Top-K list with their respective count.
// For more information - https://redis.io/commands/topk.list/
func (c cmdable) TopKListWithCount(ctx context.Context, key string) *MapStringIntCmd {
	args := []interface{}{"TOPK.LIST", key, "WITHCOUNT"}

	cmd := NewMapStringIntCmd(ctx, args...)
	_ = c(ctx, cmd)
	return cmd
}

// -------------------------------------------
// t-digest commands
// --------------------------------------------

// TDigestAdd adds one or more elements to a t-Digest data structure.
// Returns OK on success or an error if the operation could not be completed.
// For more information - https://redis.io/commands/tdigest.add/
func (c cmdable) TDigestAdd(ctx context.Context, key string, elements ...float64) *StatusCmd {
	args := make([]interface{}, 2+len(elements))
	args[0] = "TDIGEST.ADD"
	args[1] = key

	for i, v := range elements {
		args[2+i] = v
	}

	cmd := NewStatusCmd(ctx, args...)
	_ = c(ctx, cmd)
	return cmd
}

// TDigestByRank returns an array of values from a t-Digest data structure based on their rank.
// The rank of an element is its position in the sorted list of all elements in the t-Digest.
// Returns an array of floats representing the values at the specified ranks or an error if the operation could not be completed.
// For more information - https://redis.io/commands/tdigest.byrank/
func (c cmdable) TDigestByRank(ctx context.Context, key string, rank ...uint64) *FloatSliceCmd {
	args := make([]interface{}, 2+len(rank))
	args[0] = "TDIGEST.BYRANK"
	args[1] = key

	for i, r := range rank {
		args[2+i] = r
	}

	cmd := NewFloatSliceCmd(ctx, args...)
	_ = c(ctx, cmd)
	return cmd
}

// TDigestByRevRank returns an array of values from a t-Digest data structure based on their reverse rank.
// The reverse rank of an element is its position in the sorted list of all elements in the t-Digest when sorted in descending order.
// Returns an array of floats representing the values at the specified ranks or an error if the operation could not be completed.
// For more information - https://redis.io/commands/tdigest.byrevrank/
func (c cmdable) TDigestByRevRank(ctx context.Context, key string, rank ...uint64) *FloatSliceCmd {
	args := make([]interface{}, 2+len(rank))
	args[0] = "TDIGEST.BYREVRANK"
	args[1] = key

	for i, r := range rank {
		args[2+i] = r
	}

	cmd := NewFloatSliceCmd(ctx, args...)
	_ = c(ctx, cmd)
	return cmd
}

// TDigestCDF returns an array of cumulative distribution function (CDF) values for one or more elements in a t-Digest data structure.
// The CDF value for an element is the fraction of all elements in the t-Digest that are less than or equal to it.
// Returns an array of floats representing the CDF values for each element or an error if the operation could not be completed.
// For more information - https://redis.io/commands/tdigest.cdf/
func (c cmdable) TDigestCDF(ctx context.Context, key string, elements ...float64) *FloatSliceCmd {
	args := make([]interface{}, 2+len(elements))
	args[0] = "TDIGEST.CDF"
	args[1] = key

	for i, v := range elements {
		args[2+i] = v
	}

	cmd := NewFloatSliceCmd(ctx, args...)
	_ = c(ctx, cmd)
	return cmd
}

// TDigestCreate creates an empty t-Digest data structure with default parameters.
// Returns OK on success or an error if the operation could not be completed.
// For more information - https://redis.io/commands/tdigest.create/
func (c cmdable) TDigestCreate(ctx context.Context, key string) *StatusCmd {
	args := []interface{}{"TDIGEST.CREATE", key}

	cmd := NewStatusCmd(ctx, args...)
	_ = c(ctx, cmd)
	return cmd
}

// TDigestCreateWithCompression creates an empty t-Digest data structure with a specified compression parameter.
// The compression parameter controls the accuracy and memory usage of the t-Digest.
// Returns OK on success or an error if the operation could not be completed.
// For more information - https://redis.io/commands/tdigest.create/
func (c cmdable) TDigestCreateWithCompression(ctx context.Context, key string, compression int64) *StatusCmd {
	args := []interface{}{"TDIGEST.CREATE", key, "COMPRESSION", compression}

	cmd := NewStatusCmd(ctx, args...)
	_ = c(ctx, cmd)
	return cmd
}

type TDigestInfo struct {
	Compression       int64
	Capacity          int64
	MergedNodes       int64
	UnmergedNodes     int64
	MergedWeight      int64
	UnmergedWeight    int64
	Observations      int64
	TotalCompressions int64
	MemoryUsage       int64
}

type TDigestInfoCmd struct {
	baseCmd

	val TDigestInfo
}

func NewTDigestInfoCmd(ctx context.Context, args ...interface{}) *TDigestInfoCmd {
	return &TDigestInfoCmd{
		baseCmd: baseCmd{
			ctx:     ctx,
			args:    args,
			cmdType: CmdTypeTDigestInfo,
		},
	}
}

func (cmd *TDigestInfoCmd) SetVal(val TDigestInfo) {
	cmd.val = val
}

func (cmd *TDigestInfoCmd) String() string {
	cmd.await()
	return cmdString(cmd, cmd.val)
}

func (cmd *TDigestInfoCmd) Val() TDigestInfo {
	cmd.await()
	return cmd.val
}

func (cmd *TDigestInfoCmd) Result() (TDigestInfo, error) {
	cmd.await()
	return cmd.val, cmd.err
}

func (cmd *TDigestInfoCmd) readReply(rd *proto.Reader) (err error) {
	n, err := rd.ReadMapLen()
	if err != nil {
		return err
	}

	var key string
	var result TDigestInfo
	for f := 0; f < n; f++ {
		key, err = rd.ReadString()
		if err != nil {
			return err
		}

		switch key {
		case "Compression":
			result.Compression, err = rd.ReadInt()
		case "Capacity":
			result.Capacity, err = rd.ReadInt()
		case "Merged nodes":
			result.MergedNodes, err = rd.ReadInt()
		case "Unmerged nodes":
			result.UnmergedNodes, err = rd.ReadInt()
		case "Merged weight":
			result.MergedWeight, err = rd.ReadInt()
		case "Unmerged weight":
			result.UnmergedWeight, err = rd.ReadInt()
		case "Observations":
			result.Observations, err = rd.ReadInt()
		case "Total compressions":
			result.TotalCompressions, err = rd.ReadInt()
		case "Memory usage":
			result.MemoryUsage, err = rd.ReadInt()
		default:
			return fmt.Errorf("redis: tdigest.info unexpected key %s", key)
		}

		if err != nil {
			return err
		}
	}

	cmd.val = result
	return nil
}

func (cmd *TDigestInfoCmd) Clone() Cmder {
	return &TDigestInfoCmd{
		baseCmd: cmd.cloneBaseCmd(),
		val:     cmd.val, // TDigestInfo is a simple struct, can be copied directly
	}
}

// TDigestInfo returns information about a t-Digest data structure.
// For more information - https://redis.io/commands/tdigest.info/
func (c cmdable) TDigestInfo(ctx context.Context, key string) *TDigestInfoCmd {
	args := []interface{}{"TDIGEST.INFO", key}

	cmd := NewTDigestInfoCmd(ctx, args...)
	_ = c(ctx, cmd)
	return cmd
}

// TDigestMax returns the maximum value from a t-Digest data structure.
// For more information - https://redis.io/commands/tdigest.max/
func (c cmdable) TDigestMax(ctx context.Context, key string) *FloatCmd {
	args := []interface{}{"TDIGEST.MAX", key}

	cmd := NewFloatCmd(ctx, args...)
	_ = c(ctx, cmd)
	return cmd
}

type TDigestMergeOptions struct {
	Compression int64
	Override    bool
}

// TDigestMerge merges multiple t-Digest data structures into a single t-Digest.
// This function also allows for specifying additional options such as compression and override behavior.
// Returns OK on success or an error if the operation could not be completed.
// For more information - https://redis.io/commands/tdigest.merge/
func (c cmdable) TDigestMerge(ctx context.Context, destKey string, options *TDigestMergeOptions, sourceKeys ...string) *StatusCmd {
	args := []interface{}{"TDIGEST.MERGE", destKey, len(sourceKeys)}

	for _, sourceKey := range sourceKeys {
		args = append(args, sourceKey)
	}

	if options != nil {
		if options.Compression != 0 {
			args = append(args, "COMPRESSION", options.Compression)
		}
		if options.Override {
			args = append(args, "OVERRIDE")
		}
	}

	cmd := NewStatusCmd(ctx, args...)
	_ = c(ctx, cmd)
	return cmd
}

// TDigestMin returns the minimum value from a t-Digest data structure.
// For more information - https://redis.io/commands/tdigest.min/
func (c cmdable) TDigestMin(ctx context.Context, key string) *FloatCmd {
	args := []interface{}{"TDIGEST.MIN", key}

	cmd := NewFloatCmd(ctx, args...)
	_ = c(ctx, cmd)
	return cmd
}

// TDigestQuantile returns an array of quantile values for one or more elements in a t-Digest data structure.
// The quantile value for an element is the fraction of all elements in the t-Digest that are less than or equal to it.
// Returns an array of floats representing the quantile values for each element or an error if the operation could not be completed.
// For more information - https://redis.io/commands/tdigest.quantile/
func (c cmdable) TDigestQuantile(ctx context.Context, key string, elements ...float64) *FloatSliceCmd {
	args := make([]interface{}, 2+len(elements))
	args[0] = "TDIGEST.QUANTILE"
	args[1] = key

	for i, v := range elements {
		args[2+i] = v
	}

	cmd := NewFloatSliceCmd(ctx, args...)
	_ = c(ctx, cmd)
	return cmd
}

// TDigestRank returns an array of rank values for one or more elements in a t-Digest data structure.
// The rank of an element is its position in the sorted list of all elements in the t-Digest.
// Returns an array of integers representing the rank values for each element or an error if the operation could not be completed.
// For more information - https://redis.io/commands/tdigest.rank/
func (c cmdable) TDigestRank(ctx context.Context, key string, values ...float64) *IntSliceCmd {
	args := make([]interface{}, 2+len(values))
	args[0] = "TDIGEST.RANK"
	args[1] = key

	for i, v := range values {
		args[i+2] = v
	}

	cmd := NewIntSliceCmd(ctx, args...)
	_ = c(ctx, cmd)
	return cmd
}

// TDigestReset resets a t-Digest data structure to its initial state.
// Returns OK on success or an error if the operation could not be completed.
// For more information - https://redis.io/commands/tdigest.reset/
func (c cmdable) TDigestReset(ctx context.Context, key string) *StatusCmd {
	args := []interface{}{"TDIGEST.RESET", key}

	cmd := NewStatusCmd(ctx, args...)
	_ = c(ctx, cmd)
	return cmd
}

// TDigestRevRank returns an array of reverse rank values for one or more elements in a t-Digest data structure.
// The reverse rank of an element is its position in the sorted list of all elements in the t-Digest when sorted in descending order.
// Returns an array of integers representing the reverse rank values for each element or an error if the operation could not be completed.
// For more information - https://redis.io/commands/tdigest.revrank/
func (c cmdable) TDigestRevRank(ctx context.Context, key string, values ...float64) *IntSliceCmd {
	args := make([]interface{}, 2+len(values))
	args[0] = "TDIGEST.REVRANK"
	args[1] = key

	for i, v := range values {
		args[2+i] = v
	}

	cmd := NewIntSliceCmd(ctx, args...)
	_ = c(ctx, cmd)
	return cmd
}

// TDigestTrimmedMean returns the trimmed mean value from a t-Digest data structure.
// The trimmed mean is calculated by removing a specified fraction of the highest and lowest values from the t-Digest and then calculating the mean of the remaining values.
// Returns a float representing the trimmed mean value or an error if the operation could not be completed.
// For more information - https://redis.io/commands/tdigest.trimmed_mean/
func (c cmdable) TDigestTrimmedMean(ctx context.Context, key string, lowCutQuantile, highCutQuantile float64) *FloatCmd {
	args := []interface{}{"TDIGEST.TRIMMED_MEAN", key, lowCutQuantile, highCutQuantile}

	cmd := NewFloatCmd(ctx, args...)
	_ = c(ctx, cmd)
	return cmd
}
