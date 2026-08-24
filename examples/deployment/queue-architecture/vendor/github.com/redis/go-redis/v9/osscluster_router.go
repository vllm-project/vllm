package redis

import (
	"context"
	"errors"
	"fmt"
	"reflect"
	"sync"
	"time"

	"github.com/redis/go-redis/v9/internal/hashtag"
	"github.com/redis/go-redis/v9/internal/routing"
)

var (
	errInvalidCmdPointer         = errors.New("redis: invalid command pointer")
	errNoCmdsToAggregate         = errors.New("redis: no commands to aggregate")
	errNoResToAggregate          = errors.New("redis: no results to aggregate")
	errInvalidCursorCmdArgsCount = errors.New("redis: FT.CURSOR command requires at least 3 arguments")
	errInvalidCursorIdType       = errors.New("redis: invalid cursor ID type")
)

// slotResult represents the result of executing a command on a specific slot
type slotResult struct {
	cmd  Cmder
	keys []string
	err  error
}

// routeAndRun routes a command to the appropriate cluster nodes and executes it
func (c *ClusterClient) routeAndRun(ctx context.Context, cmd Cmder, node *clusterNode) error {
	var policy *routing.CommandPolicy
	if c.cmdInfoResolver != nil {
		policy = c.cmdInfoResolver.GetCommandPolicy(ctx, cmd)
	}

	// Set stepCount from cmdInfo if not already set
	if cmd.stepCount() == 0 {
		if cmdInfo := c.cmdInfo(ctx, cmd.Name()); cmdInfo != nil && cmdInfo.StepCount > 0 {
			cmd.SetStepCount(cmdInfo.StepCount)
		}
	}

	if policy == nil {
		return c.executeDefault(ctx, cmd, policy, node)
	}
	switch policy.Request {
	case routing.ReqAllNodes:
		return c.executeOnAllNodes(ctx, cmd, policy)
	case routing.ReqAllShards:
		return c.executeOnAllShards(ctx, cmd, policy)
	case routing.ReqMultiShard:
		return c.executeMultiShard(ctx, cmd, policy)
	case routing.ReqSpecial:
		return c.executeSpecialCommand(ctx, cmd, policy, node)
	default:
		return c.executeDefault(ctx, cmd, policy, node)
	}
}

// executeDefault handles standard command routing based on keys
func (c *ClusterClient) executeDefault(ctx context.Context, cmd Cmder, policy *routing.CommandPolicy, node *clusterNode) error {
	if policy != nil && !c.hasKeys(cmd) {
		if c.readOnlyEnabled() && policy.IsReadOnly() {
			return c.executeOnArbitraryNode(ctx, cmd)
		}
	}

	return node.Client.Process(ctx, cmd)
}

// executeOnArbitraryNode routes command to an arbitrary node
func (c *ClusterClient) executeOnArbitraryNode(ctx context.Context, cmd Cmder) error {
	node := c.pickArbitraryNode(ctx)
	if node == nil {
		return errClusterNoNodes
	}
	return node.Client.Process(ctx, cmd)
}

// executeOnAllNodes executes command on all nodes (masters and replicas)
func (c *ClusterClient) executeOnAllNodes(ctx context.Context, cmd Cmder, policy *routing.CommandPolicy) error {
	state, err := c.state.Get(ctx)
	if err != nil {
		return err
	}

	nodes := make([]*clusterNode, 0, len(state.Masters)+len(state.Slaves))
	nodes = append(nodes, state.Masters...)
	nodes = append(nodes, state.Slaves...)
	if len(nodes) == 0 {
		return errClusterNoNodes
	}

	return c.executeParallel(ctx, cmd, nodes, policy)
}

// executeOnAllShards executes command on all master shards
func (c *ClusterClient) executeOnAllShards(ctx context.Context, cmd Cmder, policy *routing.CommandPolicy) error {
	state, err := c.state.Get(ctx)
	if err != nil {
		return err
	}

	if len(state.Masters) == 0 {
		return errClusterNoNodes
	}

	return c.executeParallel(ctx, cmd, state.Masters, policy)
}

// executeMultiShard handles commands that operate on multiple keys across shards
func (c *ClusterClient) executeMultiShard(ctx context.Context, cmd Cmder, policy *routing.CommandPolicy) error {
	args := cmd.Args()
	firstKeyPos := cmdFirstKeyPosWithInfo(cmd, c.cmdInfoPeek(cmd.Name()))
	stepCount := int(cmd.stepCount())
	if stepCount == 0 {
		stepCount = 1 // Default to 1 if not set
	}

	if firstKeyPos == 0 || firstKeyPos >= len(args) {
		return fmt.Errorf("redis: multi-shard command %s has no key arguments", cmd.Name())
	}

	// Group keys by slot
	slotMap := make(map[int][]string)
	keyOrder := make([]string, 0)

	for i := firstKeyPos; i < len(args); i += stepCount {
		key, ok := args[i].(string)
		if !ok {
			return fmt.Errorf("redis: non-string key at position %d: %v", i, args[i])
		}

		slot := hashtag.Slot(key)
		slotMap[slot] = append(slotMap[slot], key)
		for j := 1; j < stepCount; j++ {
			if i+j >= len(args) {
				break
			}
			slotMap[slot] = append(slotMap[slot], args[i+j].(string))
		}
		keyOrder = append(keyOrder, key)
	}

	return c.executeMultiSlot(ctx, cmd, slotMap, keyOrder, policy, firstKeyPos)
}

// executeMultiSlot executes commands across multiple slots concurrently
func (c *ClusterClient) executeMultiSlot(ctx context.Context, cmd Cmder, slotMap map[int][]string, keyOrder []string, policy *routing.CommandPolicy, firstKeyPos int) error {
	results := make(chan slotResult, len(slotMap))
	var wg sync.WaitGroup

	// Execute on each slot concurrently
	for slot, keys := range slotMap {
		wg.Add(1)
		go func(slot int, keys []string) {
			defer wg.Done()

			node, err := c.cmdNodeWithShardPicker(ctx, cmd.Name(), slot, c.opt.ShardPicker)
			if err != nil {
				results <- slotResult{nil, keys, err}
				return
			}

			// Create a command for this specific slot's keys
			subCmd := c.createSlotSpecificCommand(ctx, cmd, keys, firstKeyPos)
			err = node.Client.Process(ctx, subCmd)
			results <- slotResult{subCmd, keys, err}
		}(slot, keys)
	}

	go func() {
		wg.Wait()
		close(results)
	}()

	return c.aggregateMultiSlotResults(ctx, cmd, results, keyOrder, policy)
}

// createSlotSpecificCommand creates a new command for a specific slot's keys.
// firstKeyPos is passed in from the caller (computed once in executeMultiShard)
// so this function never independently re-peeks the cache — avoids the
// cold --> warm inconsistency the reviewer flagged.
func (c *ClusterClient) createSlotSpecificCommand(ctx context.Context, originalCmd Cmder, keys []string, firstKeyPos int) Cmder {
	originalArgs := originalCmd.Args()

	// Build new args with only the specified keys
	newArgs := make([]interface{}, 0, firstKeyPos+len(keys))

	// Copy command name and arguments before the keys
	newArgs = append(newArgs, originalArgs[:firstKeyPos]...)

	// Add the slot-specific keys
	for _, key := range keys {
		newArgs = append(newArgs, key)
	}

	// Create a new command of the same type using the helper function
	return createCommandByType(ctx, originalCmd.GetCmdType(), newArgs...)
}

// createCommandByType creates a new command of the specified type with the given arguments
func createCommandByType(ctx context.Context, cmdType CmdType, args ...interface{}) Cmder {
	switch cmdType {
	case CmdTypeString:
		return NewStringCmd(ctx, args...)
	case CmdTypeInt:
		return NewIntCmd(ctx, args...)
	case CmdTypeBool:
		return NewBoolCmd(ctx, args...)
	case CmdTypeFloat:
		return NewFloatCmd(ctx, args...)
	case CmdTypeStringSlice:
		return NewStringSliceCmd(ctx, args...)
	case CmdTypeIntSlice:
		return NewIntSliceCmd(ctx, args...)
	case CmdTypeFloatSlice:
		return NewFloatSliceCmd(ctx, args...)
	case CmdTypeBoolSlice:
		return NewBoolSliceCmd(ctx, args...)
	case CmdTypeStatus:
		return NewStatusCmd(ctx, args...)
	case CmdTypeTime:
		return NewTimeCmd(ctx, args...)
	case CmdTypeMapStringString:
		return NewMapStringStringCmd(ctx, args...)
	case CmdTypeMapStringInt:
		return NewMapStringIntCmd(ctx, args...)
	case CmdTypeMapStringInterface:
		return NewMapStringInterfaceCmd(ctx, args...)
	case CmdTypeMapStringInterfaceSlice:
		return NewMapStringInterfaceSliceCmd(ctx, args...)
	case CmdTypeSlice:
		return NewSliceCmd(ctx, args...)
	case CmdTypeStringStructMap:
		return NewStringStructMapCmd(ctx, args...)
	case CmdTypeXMessageSlice:
		return NewXMessageSliceCmd(ctx, args...)
	case CmdTypeXStreamSlice:
		return NewXStreamSliceCmd(ctx, args...)
	case CmdTypeXPending:
		return NewXPendingCmd(ctx, args...)
	case CmdTypeXPendingExt:
		return NewXPendingExtCmd(ctx, args...)
	case CmdTypeXAutoClaim:
		return NewXAutoClaimCmd(ctx, args...)
	case CmdTypeXAutoClaimWithDeleted:
		return NewXAutoClaimWithDeletedCmd(ctx, args...)
	case CmdTypeXAutoClaimJustID:
		return NewXAutoClaimJustIDCmd(ctx, args...)
	case CmdTypeXInfoStreamFull:
		return NewXInfoStreamFullCmd(ctx, args...)
	case CmdTypeZSlice:
		return NewZSliceCmd(ctx, args...)
	case CmdTypeZWithKey:
		return NewZWithKeyCmd(ctx, args...)
	case CmdTypeClusterSlots:
		return NewClusterSlotsCmd(ctx, args...)
	case CmdTypeGeoPos:
		return NewGeoPosCmd(ctx, args...)
	case CmdTypeCommandsInfo:
		return NewCommandsInfoCmd(ctx, args...)
	case CmdTypeSlowLog:
		return NewSlowLogCmd(ctx, args...)
	case CmdTypeKeyValues:
		return NewKeyValuesCmd(ctx, args...)
	case CmdTypeZSliceWithKey:
		return NewZSliceWithKeyCmd(ctx, args...)
	case CmdTypeFunctionList:
		return NewFunctionListCmd(ctx, args...)
	case CmdTypeFunctionStats:
		return NewFunctionStatsCmd(ctx, args...)
	case CmdTypeKeyFlags:
		return NewKeyFlagsCmd(ctx, args...)
	case CmdTypeDuration:
		return NewDurationCmd(ctx, time.Millisecond, args...)
	}
	return NewCmd(ctx, args...)
}

// executeSpecialCommand handles commands with special routing requirements
func (c *ClusterClient) executeSpecialCommand(ctx context.Context, cmd Cmder, policy *routing.CommandPolicy, node *clusterNode) error {
	switch cmd.Name() {
	case "ft.cursor":
		return c.executeCursorCommand(ctx, cmd)
	default:
		return c.executeDefault(ctx, cmd, policy, node)
	}
}

// executeCursorCommand handles FT.CURSOR commands with sticky routing
func (c *ClusterClient) executeCursorCommand(ctx context.Context, cmd Cmder) error {
	args := cmd.Args()
	if len(args) < 4 {
		return errInvalidCursorCmdArgsCount
	}

	cursorID, ok := args[3].(string)
	if !ok {
		return errInvalidCursorIdType
	}

	// Route based on cursor ID to maintain stickiness
	slot := hashtag.Slot(cursorID)
	node, err := c.cmdNodeWithShardPicker(ctx, cmd.Name(), slot, c.opt.ShardPicker)
	if err != nil {
		return err
	}

	return node.Client.Process(ctx, cmd)
}

// executeParallel executes a command on multiple nodes concurrently
func (c *ClusterClient) executeParallel(ctx context.Context, cmd Cmder, nodes []*clusterNode, policy *routing.CommandPolicy) error {
	if len(nodes) == 0 {
		return errClusterNoNodes
	}

	if len(nodes) == 1 {
		return nodes[0].Client.Process(ctx, cmd)
	}

	type nodeResult struct {
		cmd Cmder
		err error
	}

	results := make(chan nodeResult, len(nodes))
	var wg sync.WaitGroup

	for _, node := range nodes {
		wg.Add(1)
		go func(n *clusterNode) {
			defer wg.Done()
			cmdCopy := cmd.Clone()
			err := n.Client.Process(ctx, cmdCopy)
			results <- nodeResult{cmdCopy, err}
		}(node)
	}

	go func() {
		wg.Wait()
		close(results)
	}()

	// Collect results and check for errors
	cmds := make([]Cmder, 0, len(nodes))
	var firstErr error

	for result := range results {
		if result.err != nil && firstErr == nil {
			firstErr = result.err
		}
		cmds = append(cmds, result.cmd)
	}

	// If there was an error and no policy specified, fail fast
	if firstErr != nil && (policy == nil || policy.Response == routing.RespDefaultKeyless) {
		cmd.SetErr(firstErr)
		return firstErr
	}

	return c.aggregateResponses(cmd, cmds, policy)
}

// aggregateMultiSlotResults aggregates results from multi-slot execution
func (c *ClusterClient) aggregateMultiSlotResults(ctx context.Context, cmd Cmder, results <-chan slotResult, keyOrder []string, policy *routing.CommandPolicy) error {
	keyedResults := make(map[string]routing.AggregatorResErr)
	var firstErr error

	for result := range results {
		if result.err != nil && firstErr == nil {
			firstErr = result.err
		}
		if result.cmd != nil && result.err == nil {
			value, err := ExtractCommandValue(result.cmd)

			// Check if the result is a slice (e.g., from MGET)
			if sliceValue, ok := value.([]interface{}); ok {
				// Map each element to its corresponding key
				for i, key := range result.keys {
					if i < len(sliceValue) {
						keyedResults[key] = routing.AggregatorResErr{Result: sliceValue[i], Err: err}
					} else {
						keyedResults[key] = routing.AggregatorResErr{Result: nil, Err: err}
					}
				}
			} else {
				// For non-slice results, map the entire result to each key
				for _, key := range result.keys {
					keyedResults[key] = routing.AggregatorResErr{Result: value, Err: err}
				}
			}
		}

		// TODO: return multiple errors by order when we will implement multiple errors returning
		if result.err != nil {
			firstErr = result.err
		}
	}

	return c.aggregateKeyedValues(cmd, keyedResults, keyOrder, policy)
}

// aggregateKeyedValues aggregates individual key-value pairs while preserving key order
func (c *ClusterClient) aggregateKeyedValues(cmd Cmder, keyedResults map[string]routing.AggregatorResErr, keyOrder []string, policy *routing.CommandPolicy) error {
	if len(keyedResults) == 0 {
		return errNoResToAggregate
	}

	aggregator := c.createAggregator(policy, cmd, true)

	// Set key order for keyed aggregators
	var keyedAgg *routing.DefaultKeyedAggregator
	var isKeyedAgg bool
	var err error
	if keyedAgg, isKeyedAgg = aggregator.(*routing.DefaultKeyedAggregator); isKeyedAgg {
		err = keyedAgg.BatchAddWithKeyOrder(keyedResults, keyOrder)
	} else {
		err = aggregator.BatchAdd(keyedResults)
	}

	if err != nil {
		return err
	}

	return c.finishAggregation(cmd, aggregator)
}

// aggregateResponses aggregates multiple shard responses
func (c *ClusterClient) aggregateResponses(cmd Cmder, cmds []Cmder, policy *routing.CommandPolicy) error {
	if len(cmds) == 0 {
		return errNoCmdsToAggregate
	}

	if len(cmds) == 1 {
		shardCmd := cmds[0]
		if err := shardCmd.Err(); err != nil {
			cmd.SetErr(err)
			return err
		}
		value, _ := ExtractCommandValue(shardCmd)
		return c.setCommandValue(cmd, value)
	}

	aggregator := c.createAggregator(policy, cmd, false)

	batchWithErrs := []routing.AggregatorResErr{}
	// Add all results to aggregator
	for _, shardCmd := range cmds {
		value, err := ExtractCommandValue(shardCmd)
		batchWithErrs = append(batchWithErrs, routing.AggregatorResErr{
			Result: value,
			Err:    err,
		})
	}

	err := aggregator.BatchSlice(batchWithErrs)
	if err != nil {
		return err
	}

	return c.finishAggregation(cmd, aggregator)
}

// createAggregator creates the appropriate response aggregator
func (c *ClusterClient) createAggregator(policy *routing.CommandPolicy, cmd Cmder, isKeyed bool) routing.ResponseAggregator {
	if policy != nil {
		return routing.NewResponseAggregator(policy.Response, cmd.Name())
	}

	if !isKeyed {
		firstKeyPos := cmdFirstKeyPosWithInfo(cmd, c.cmdInfoPeek(cmd.Name()))
		isKeyed = firstKeyPos > 0
	}

	return routing.NewDefaultAggregator(isKeyed)
}

// finishAggregation completes the aggregation process and sets the result
func (c *ClusterClient) finishAggregation(cmd Cmder, aggregator routing.ResponseAggregator) error {
	finalValue, finalErr := aggregator.Result()
	if finalErr != nil {
		cmd.SetErr(finalErr)
		return finalErr
	}

	return c.setCommandValue(cmd, finalValue)
}

// pickArbitraryNode selects a master or slave shard using the configured ShardPicker
func (c *ClusterClient) pickArbitraryNode(ctx context.Context) *clusterNode {
	state, err := c.state.Get(ctx)
	if err != nil || len(state.Masters) == 0 {
		return nil
	}

	// Index into masters+slaves without materializing a combined slice.
	// append(state.Masters, state.Slaves...) writes into the shared snapshot's
	// spare capacity and races other routers, so pick directly.
	idx := c.opt.ShardPicker.Next(len(state.Masters) + len(state.Slaves))
	if idx < len(state.Masters) {
		return state.Masters[idx]
	}
	return state.Slaves[idx-len(state.Masters)]
}

// hasKeys checks if a command operates on keys
func (c *ClusterClient) hasKeys(cmd Cmder) bool {
	firstKeyPos := cmdFirstKeyPosWithInfo(cmd, c.cmdInfoPeek(cmd.Name()))
	return firstKeyPos > 0
}

func (c *ClusterClient) readOnlyEnabled() bool {
	return c.opt.ReadOnly
}

// setCommandValue sets the aggregated value on a command using the enum-based approach
func (c *ClusterClient) setCommandValue(cmd Cmder, value interface{}) error {
	// If value is nil, it might mean ExtractCommandValue couldn't extract the value
	// but the command might have executed successfully. In this case, don't set an error.
	if value == nil {
		// ExtractCommandValue returned nil - this means the command type is not supported
		// in the aggregation flow. This is a programming error, not a runtime error.
		if cmd.Err() != nil {
			// Command already has an error, preserve it
			return cmd.Err()
		}
		// Command executed successfully but we can't extract/set the aggregated value
		// This indicates the command type needs to be added to ExtractCommandValue
		return fmt.Errorf("redis: cannot aggregate command %s: unsupported command type %d",
			cmd.Name(), cmd.GetCmdType())
	}

	switch cmd.GetCmdType() {
	case CmdTypeGeneric:
		if c, ok := cmd.(*Cmd); ok {
			c.SetVal(value)
		}
	case CmdTypeString:
		if c, ok := cmd.(*StringCmd); ok {
			if v, ok := value.(string); ok {
				c.SetVal(v)
			}
		}
	case CmdTypeInt:
		if c, ok := cmd.(*IntCmd); ok {
			if v, ok := value.(int64); ok {
				c.SetVal(v)
			} else if v, ok := value.(float64); ok {
				c.SetVal(int64(v))
			}
		}
	case CmdTypeBool:
		if c, ok := cmd.(*BoolCmd); ok {
			if v, ok := value.(bool); ok {
				c.SetVal(v)
			}
		}
	case CmdTypeFloat:
		if c, ok := cmd.(*FloatCmd); ok {
			if v, ok := value.(float64); ok {
				c.SetVal(v)
			}
		}
	case CmdTypeStringSlice:
		if c, ok := cmd.(*StringSliceCmd); ok {
			if v, ok := value.([]string); ok {
				c.SetVal(v)
			}
		}
	case CmdTypeIntSlice:
		if c, ok := cmd.(*IntSliceCmd); ok {
			if v, ok := value.([]int64); ok {
				c.SetVal(v)
			} else if v, ok := value.([]float64); ok {
				els := len(v)
				intSlc := make([]int, els)
				for i := range v {
					intSlc[i] = int(v[i])
				}
			}
		}
	case CmdTypeFloatSlice:
		if c, ok := cmd.(*FloatSliceCmd); ok {
			if v, ok := value.([]float64); ok {
				c.SetVal(v)
			}
		}
	case CmdTypeBoolSlice:
		if c, ok := cmd.(*BoolSliceCmd); ok {
			if v, ok := value.([]bool); ok {
				c.SetVal(v)
			}
		}
	case CmdTypeMapStringString:
		if c, ok := cmd.(*MapStringStringCmd); ok {
			if v, ok := value.(map[string]string); ok {
				c.SetVal(v)
			}
		}
	case CmdTypeMapStringInt:
		if c, ok := cmd.(*MapStringIntCmd); ok {
			if v, ok := value.(map[string]int64); ok {
				c.SetVal(v)
			}
		}
	case CmdTypeMapStringInterface:
		if c, ok := cmd.(*MapStringInterfaceCmd); ok {
			if v, ok := value.(map[string]interface{}); ok {
				c.SetVal(v)
			}
		}
	case CmdTypeSlice:
		if c, ok := cmd.(*SliceCmd); ok {
			if v, ok := value.([]interface{}); ok {
				c.SetVal(v)
			}
		}
	case CmdTypeStatus:
		if c, ok := cmd.(*StatusCmd); ok {
			if v, ok := value.(string); ok {
				c.SetVal(v)
			}
		}
	case CmdTypeDuration:
		if c, ok := cmd.(*DurationCmd); ok {
			if v, ok := value.(time.Duration); ok {
				c.SetVal(v)
			}
		}
	case CmdTypeTime:
		if c, ok := cmd.(*TimeCmd); ok {
			if v, ok := value.(time.Time); ok {
				c.SetVal(v)
			}
		}
	case CmdTypeKeyValueSlice:
		if c, ok := cmd.(*KeyValueSliceCmd); ok {
			if v, ok := value.([]KeyValue); ok {
				c.SetVal(v)
			}
		}
	case CmdTypeStringStructMap:
		if c, ok := cmd.(*StringStructMapCmd); ok {
			if v, ok := value.(map[string]struct{}); ok {
				c.SetVal(v)
			}
		}
	case CmdTypeXMessageSlice:
		if c, ok := cmd.(*XMessageSliceCmd); ok {
			if v, ok := value.([]XMessage); ok {
				c.SetVal(v)
			}
		}
	case CmdTypeXStreamSlice:
		if c, ok := cmd.(*XStreamSliceCmd); ok {
			if v, ok := value.([]XStream); ok {
				c.SetVal(v)
			}
		}
	case CmdTypeXPending:
		if c, ok := cmd.(*XPendingCmd); ok {
			if v, ok := value.(*XPending); ok {
				c.SetVal(v)
			}
		}
	case CmdTypeXPendingExt:
		if c, ok := cmd.(*XPendingExtCmd); ok {
			if v, ok := value.([]XPendingExt); ok {
				c.SetVal(v)
			}
		}
	case CmdTypeXAutoClaim:
		if c, ok := cmd.(*XAutoClaimCmd); ok {
			if v, ok := value.(CmdTypeXAutoClaimValue); ok {
				c.SetVal(v.messages, v.start)
			}
		}
	case CmdTypeXAutoClaimWithDeleted:
		if c, ok := cmd.(*XAutoClaimWithDeletedCmd); ok {
			if v, ok := value.(CmdTypeXAutoClaimWithDeletedValue); ok {
				c.SetVal(v.messages, v.start, v.deletedIDs)
			}
		}
	case CmdTypeXAutoClaimJustID:
		if c, ok := cmd.(*XAutoClaimJustIDCmd); ok {
			if v, ok := value.(CmdTypeXAutoClaimJustIDValue); ok {
				c.SetVal(v.ids, v.start)
			}
		}
	case CmdTypeXInfoConsumers:
		if c, ok := cmd.(*XInfoConsumersCmd); ok {
			if v, ok := value.([]XInfoConsumer); ok {
				c.SetVal(v)
			}
		}
	case CmdTypeXInfoGroups:
		if c, ok := cmd.(*XInfoGroupsCmd); ok {
			if v, ok := value.([]XInfoGroup); ok {
				c.SetVal(v)
			}
		}
	case CmdTypeXInfoStream:
		if c, ok := cmd.(*XInfoStreamCmd); ok {
			if v, ok := value.(*XInfoStream); ok {
				c.SetVal(v)
			}
		}
	case CmdTypeXInfoStreamFull:
		if c, ok := cmd.(*XInfoStreamFullCmd); ok {
			if v, ok := value.(*XInfoStreamFull); ok {
				c.SetVal(v)
			}
		}
	case CmdTypeZSlice:
		if c, ok := cmd.(*ZSliceCmd); ok {
			if v, ok := value.([]Z); ok {
				c.SetVal(v)
			}
		}
	case CmdTypeZWithKey:
		if c, ok := cmd.(*ZWithKeyCmd); ok {
			if v, ok := value.(*ZWithKey); ok {
				c.SetVal(v)
			}
		}
	case CmdTypeScan:
		if c, ok := cmd.(*ScanCmd); ok {
			if v, ok := value.(CmdTypeScanValue); ok {
				c.SetVal(v.keys, v.cursor)
			}
		}
	case CmdTypeClusterSlots:
		if c, ok := cmd.(*ClusterSlotsCmd); ok {
			if v, ok := value.([]ClusterSlot); ok {
				c.SetVal(v)
			}
		}
	case CmdTypeGeoLocation:
		if c, ok := cmd.(*GeoLocationCmd); ok {
			if v, ok := value.([]GeoLocation); ok {
				c.SetVal(v)
			}
		}
	case CmdTypeGeoSearchLocation:
		if c, ok := cmd.(*GeoSearchLocationCmd); ok {
			if v, ok := value.([]GeoLocation); ok {
				c.SetVal(v)
			}
		}
	case CmdTypeGeoPos:
		if c, ok := cmd.(*GeoPosCmd); ok {
			if v, ok := value.([]*GeoPos); ok {
				c.SetVal(v)
			}
		}
	case CmdTypeCommandsInfo:
		if c, ok := cmd.(*CommandsInfoCmd); ok {
			if v, ok := value.(map[string]*CommandInfo); ok {
				c.SetVal(v)
			}
		}
	case CmdTypeSlowLog:
		if c, ok := cmd.(*SlowLogCmd); ok {
			if v, ok := value.([]SlowLog); ok {
				c.SetVal(v)
			}
		}
	case CmdTypeMapStringStringSlice:
		if c, ok := cmd.(*MapStringStringSliceCmd); ok {
			if v, ok := value.([]map[string]string); ok {
				c.SetVal(v)
			}
		}
	case CmdTypeMapMapStringInterface:
		if c, ok := cmd.(*MapMapStringInterfaceCmd); ok {
			if v, ok := value.(map[string]interface{}); ok {
				c.SetVal(v)
			}
		}
	case CmdTypeMapStringInterfaceSlice:
		if c, ok := cmd.(*MapStringInterfaceSliceCmd); ok {
			if v, ok := value.([]map[string]interface{}); ok {
				c.SetVal(v)
			}
		}
	case CmdTypeKeyValues:
		if c, ok := cmd.(*KeyValuesCmd); ok {
			// KeyValuesCmd needs a key string and values slice
			if v, ok := value.(CmdTypeKeyValuesValue); ok {
				c.SetVal(v.key, v.values)
			}
		}
	case CmdTypeZSliceWithKey:
		if c, ok := cmd.(*ZSliceWithKeyCmd); ok {
			// ZSliceWithKeyCmd needs a key string and Z slice
			if v, ok := value.(CmdTypeZSliceWithKeyValue); ok {
				c.SetVal(v.key, v.zSlice)
			}
		}
	case CmdTypeFunctionList:
		if c, ok := cmd.(*FunctionListCmd); ok {
			if v, ok := value.([]Library); ok {
				c.SetVal(v)
			}
		}
	case CmdTypeFunctionStats:
		if c, ok := cmd.(*FunctionStatsCmd); ok {
			if v, ok := value.(FunctionStats); ok {
				c.SetVal(v)
			}
		}
	case CmdTypeLCS:
		if c, ok := cmd.(*LCSCmd); ok {
			if v, ok := value.(*LCSMatch); ok {
				c.SetVal(v)
			}
		}
	case CmdTypeKeyFlags:
		if c, ok := cmd.(*KeyFlagsCmd); ok {
			if v, ok := value.([]KeyFlags); ok {
				c.SetVal(v)
			}
		}
	case CmdTypeClusterLinks:
		if c, ok := cmd.(*ClusterLinksCmd); ok {
			if v, ok := value.([]ClusterLink); ok {
				c.SetVal(v)
			}
		}
	case CmdTypeClusterShards:
		if c, ok := cmd.(*ClusterShardsCmd); ok {
			if v, ok := value.([]ClusterShard); ok {
				c.SetVal(v)
			}
		}
	case CmdTypeRankWithScore:
		if c, ok := cmd.(*RankWithScoreCmd); ok {
			if v, ok := value.(RankScore); ok {
				c.SetVal(v)
			}
		}
	case CmdTypeClientInfo:
		if c, ok := cmd.(*ClientInfoCmd); ok {
			if v, ok := value.(*ClientInfo); ok {
				c.SetVal(v)
			}
		}
	case CmdTypeACLLog:
		if c, ok := cmd.(*ACLLogCmd); ok {
			if v, ok := value.([]*ACLLogEntry); ok {
				c.SetVal(v)
			}
		}
	case CmdTypeInfo:
		if c, ok := cmd.(*InfoCmd); ok {
			if v, ok := value.(map[string]map[string]string); ok {
				c.SetVal(v)
			}
		}
	case CmdTypeMonitor:
		// MonitorCmd doesn't have SetVal method
		// Skip setting value for MonitorCmd
	case CmdTypeJSON:
		if c, ok := cmd.(*JSONCmd); ok {
			if v, ok := value.(string); ok {
				c.SetVal(v)
			}
		}
	case CmdTypeJSONSlice:
		if c, ok := cmd.(*JSONSliceCmd); ok {
			if v, ok := value.([]interface{}); ok {
				c.SetVal(v)
			}
		}
	case CmdTypeIntPointerSlice:
		if c, ok := cmd.(*IntPointerSliceCmd); ok {
			if v, ok := value.([]*int64); ok {
				c.SetVal(v)
			}
		}
	case CmdTypeScanDump:
		if c, ok := cmd.(*ScanDumpCmd); ok {
			if v, ok := value.(ScanDump); ok {
				c.SetVal(v)
			}
		}
	case CmdTypeBFInfo:
		if c, ok := cmd.(*BFInfoCmd); ok {
			if v, ok := value.(BFInfo); ok {
				c.SetVal(v)
			}
		}
	case CmdTypeCFInfo:
		if c, ok := cmd.(*CFInfoCmd); ok {
			if v, ok := value.(CFInfo); ok {
				c.SetVal(v)
			}
		}
	case CmdTypeCMSInfo:
		if c, ok := cmd.(*CMSInfoCmd); ok {
			if v, ok := value.(CMSInfo); ok {
				c.SetVal(v)
			}
		}
	case CmdTypeTopKInfo:
		if c, ok := cmd.(*TopKInfoCmd); ok {
			if v, ok := value.(TopKInfo); ok {
				c.SetVal(v)
			}
		}
	case CmdTypeTDigestInfo:
		if c, ok := cmd.(*TDigestInfoCmd); ok {
			if v, ok := value.(TDigestInfo); ok {
				c.SetVal(v)
			}
		}
	case CmdTypeFTSynDump:
		if c, ok := cmd.(*FTSynDumpCmd); ok {
			if v, ok := value.([]FTSynDumpResult); ok {
				c.SetVal(v)
			}
		}
	case CmdTypeAggregate:
		if c, ok := cmd.(*AggregateCmd); ok {
			if v, ok := value.(*FTAggregateResult); ok {
				c.SetVal(v)
			}
		}
	case CmdTypeFTInfo:
		if c, ok := cmd.(*FTInfoCmd); ok {
			if v, ok := value.(FTInfoResult); ok {
				c.SetVal(v)
			}
		}
	case CmdTypeFTSpellCheck:
		if c, ok := cmd.(*FTSpellCheckCmd); ok {
			if v, ok := value.([]SpellCheckResult); ok {
				c.SetVal(v)
			}
		}
	case CmdTypeFTSearch:
		if c, ok := cmd.(*FTSearchCmd); ok {
			if v, ok := value.(FTSearchResult); ok {
				c.SetVal(v)
			}
		}
	case CmdTypeTSTimestampValue:
		if c, ok := cmd.(*TSTimestampValueCmd); ok {
			if v, ok := value.(TSTimestampValue); ok {
				c.SetVal(v)
			}
		}
	case CmdTypeTSTimestampValueSlice:
		if c, ok := cmd.(*TSTimestampValueSliceCmd); ok {
			if v, ok := value.([]TSTimestampValue); ok {
				c.SetVal(v)
			}
		}
	default:
		// Fallback to reflection for unknown types
		return c.setCommandValueReflection(cmd, value)
	}

	return nil
}

// setCommandValueReflection is a fallback function that uses reflection
func (c *ClusterClient) setCommandValueReflection(cmd Cmder, value interface{}) error {
	cmdValue := reflect.ValueOf(cmd)
	if cmdValue.Kind() != reflect.Ptr || cmdValue.IsNil() {
		return errInvalidCmdPointer
	}

	setValMethod := cmdValue.MethodByName("SetVal")
	if !setValMethod.IsValid() {
		return fmt.Errorf("redis: command %T does not have SetVal method", cmd)
	}

	args := []reflect.Value{reflect.ValueOf(value)}

	switch cmd.(type) {
	case *XAutoClaimCmd, *XAutoClaimJustIDCmd:
		args = append(args, reflect.ValueOf(""))
	case *ScanCmd:
		args = append(args, reflect.ValueOf(uint64(0)))
	case *KeyValuesCmd, *ZSliceWithKeyCmd:
		if key, ok := value.(string); ok {
			args = []reflect.Value{reflect.ValueOf(key)}
			if _, ok := cmd.(*ZSliceWithKeyCmd); ok {
				args = append(args, reflect.ValueOf([]Z{}))
			} else {
				args = append(args, reflect.ValueOf([]string{}))
			}
		}
	}

	defer func() {
		if r := recover(); r != nil {
			cmd.SetErr(fmt.Errorf("redis: failed to set command value: %v", r))
		}
	}()

	setValMethod.Call(args)
	return nil
}
