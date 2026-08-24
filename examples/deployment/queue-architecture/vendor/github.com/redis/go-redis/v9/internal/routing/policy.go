package routing

import (
	"fmt"
	"strings"
)

type RequestPolicy uint8

const (
	ReqDefault RequestPolicy = iota

	ReqAllNodes

	ReqAllShards

	ReqMultiShard

	ReqSpecial
)

const (
	ReadOnlyCMD string = "readonly"
)

func (p RequestPolicy) String() string {
	switch p {
	case ReqDefault:
		return "default"
	case ReqAllNodes:
		return "all_nodes"
	case ReqAllShards:
		return "all_shards"
	case ReqMultiShard:
		return "multi_shard"
	case ReqSpecial:
		return "special"
	default:
		return fmt.Sprintf("unknown_request_policy(%d)", p)
	}
}

func ParseRequestPolicy(raw string) (RequestPolicy, error) {
	switch strings.ToLower(raw) {
	case "", "default", "none":
		return ReqDefault, nil
	case "all_nodes":
		return ReqAllNodes, nil
	case "all_shards":
		return ReqAllShards, nil
	case "multi_shard":
		return ReqMultiShard, nil
	case "special":
		return ReqSpecial, nil
	default:
		return ReqDefault, fmt.Errorf("routing: unknown request_policy %q", raw)
	}
}

type ResponsePolicy uint8

const (
	RespDefaultKeyless ResponsePolicy = iota
	RespDefaultHashSlot
	RespAllSucceeded
	RespOneSucceeded
	RespAggSum
	RespAggMin
	RespAggMax
	RespAggLogicalAnd
	RespAggLogicalOr
	RespSpecial
)

func (p ResponsePolicy) String() string {
	switch p {
	case RespDefaultKeyless:
		return "default(keyless)"
	case RespDefaultHashSlot:
		return "default(hashslot)"
	case RespAllSucceeded:
		return "all_succeeded"
	case RespOneSucceeded:
		return "one_succeeded"
	case RespAggSum:
		return "agg_sum"
	case RespAggMin:
		return "agg_min"
	case RespAggMax:
		return "agg_max"
	case RespAggLogicalAnd:
		return "agg_logical_and"
	case RespAggLogicalOr:
		return "agg_logical_or"
	case RespSpecial:
		return "special"
	default:
		return "all_succeeded"
	}
}

func ParseResponsePolicy(raw string) (ResponsePolicy, error) {
	switch strings.ToLower(raw) {
	case "default(keyless)":
		return RespDefaultKeyless, nil
	case "default(hashslot)":
		return RespDefaultHashSlot, nil
	case "all_succeeded":
		return RespAllSucceeded, nil
	case "one_succeeded":
		return RespOneSucceeded, nil
	case "agg_sum":
		return RespAggSum, nil
	case "agg_min":
		return RespAggMin, nil
	case "agg_max":
		return RespAggMax, nil
	case "agg_logical_and":
		return RespAggLogicalAnd, nil
	case "agg_logical_or":
		return RespAggLogicalOr, nil
	case "special":
		return RespSpecial, nil
	default:
		return RespDefaultKeyless, fmt.Errorf("routing: unknown response_policy %q", raw)
	}
}

type CommandPolicy struct {
	Request  RequestPolicy
	Response ResponsePolicy
	// Tips that are not request_policy or response_policy
	// e.g nondeterministic_output, nondeterministic_output_order.
	Tips map[string]string
}

func (p *CommandPolicy) CanBeUsedInPipeline() bool {
	return p.Request != ReqAllNodes && p.Request != ReqAllShards && p.Request != ReqMultiShard
}

func (p *CommandPolicy) IsReadOnly() bool {
	_, readOnly := p.Tips[ReadOnlyCMD]
	return readOnly
}
