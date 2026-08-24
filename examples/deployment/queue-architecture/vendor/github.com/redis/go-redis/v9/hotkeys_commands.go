package redis

import (
	"context"
	"errors"
	"strings"
)

// HOTKEYS commands are only available on standalone *Client instances.
// They are NOT available on ClusterClient, Ring, or UniversalClient because
// HOTKEYS is a stateful command requiring session affinity - all operations
// (START, GET, STOP, RESET) must be sent to the same Redis node.
//
// If you are using UniversalClient and need HOTKEYS functionality, you must
// type assert to *Client first:
//
//	if client, ok := universalClient.(*redis.Client); ok {
//	    result, err := client.HotKeysStart(ctx, args)
//	    // ...
//	}

// HotKeysMetric represents the metrics that can be tracked by the HOTKEYS command.
type HotKeysMetric string

const (
	// HotKeysMetricCPU tracks CPU time spent on the key (in microseconds).
	HotKeysMetricCPU HotKeysMetric = "CPU"
	// HotKeysMetricNET tracks network bytes used by the key (ingress + egress + replication).
	HotKeysMetricNET HotKeysMetric = "NET"
)

// HotKeysStartArgs contains the arguments for the HOTKEYS START command.
// This command is only available on standalone clients due to its stateful nature
// requiring session affinity. It must NOT be used on cluster or pooled clients.
type HotKeysStartArgs struct {
	// Metrics to track. At least one must be specified.
	Metrics []HotKeysMetric
	// Count is the number of top keys to report.
	// Default: 10, Min: 10, Max: 64
	Count uint8
	// Duration is the auto-stop tracking after this many seconds.
	// Default: 0 (no auto-stop)
	Duration int64
	// Sample is the sample ratio - track keys with probability 1/sample.
	// Default: 1 (track every key), Min: 1
	Sample int64
	// Slots specifies specific hash slots to track (0-16383).
	// All specified slots must be hosted by the receiving node.
	// If not specified, all slots are tracked.
	Slots []uint16
}

// ErrHotKeysNoMetrics is returned when HotKeysStart is called without any metrics specified.
var ErrHotKeysNoMetrics = errors.New("redis: at least one metric must be specified for HOTKEYS START")

// HotKeysStart starts collecting hotkeys data.
// At least one metric must be specified in args.Metrics.
// This command is only available on standalone clients.
func (c *Client) HotKeysStart(ctx context.Context, args *HotKeysStartArgs) *StatusCmd {
	cmdArgs := make([]interface{}, 0, 16)
	cmdArgs = append(cmdArgs, "hotkeys", "start")

	// Validate that at least one metric is specified
	if len(args.Metrics) == 0 {
		cmd := NewStatusCmd(ctx, cmdArgs...)
		cmd.SetErr(ErrHotKeysNoMetrics)
		return cmd
	}

	cmdArgs = append(cmdArgs, "metrics", len(args.Metrics))
	for _, metric := range args.Metrics {
		cmdArgs = append(cmdArgs, strings.ToLower(string(metric)))
	}

	if args.Count > 0 {
		cmdArgs = append(cmdArgs, "count", args.Count)
	}

	if args.Duration > 0 {
		cmdArgs = append(cmdArgs, "duration", args.Duration)
	}

	if args.Sample > 0 {
		cmdArgs = append(cmdArgs, "sample", args.Sample)
	}

	if len(args.Slots) > 0 {
		cmdArgs = append(cmdArgs, "slots", len(args.Slots))
		for _, slot := range args.Slots {
			cmdArgs = append(cmdArgs, slot)
		}
	}

	cmd := NewStatusCmd(ctx, cmdArgs...)
	_ = c.Process(ctx, cmd)
	return cmd
}

// HotKeysStop stops the ongoing hotkeys collection session.
// This command is only available on standalone clients.
func (c *Client) HotKeysStop(ctx context.Context) *StatusCmd {
	cmd := NewStatusCmd(ctx, "hotkeys", "stop")
	_ = c.Process(ctx, cmd)
	return cmd
}

// HotKeysReset discards the last hotkeys collection session results.
// Returns an error if tracking is currently active.
// This command is only available on standalone clients.
func (c *Client) HotKeysReset(ctx context.Context) *StatusCmd {
	cmd := NewStatusCmd(ctx, "hotkeys", "reset")
	_ = c.Process(ctx, cmd)
	return cmd
}

// HotKeysGet retrieves the results of the ongoing or last hotkeys collection session.
// This command is only available on standalone clients.
func (c *Client) HotKeysGet(ctx context.Context) *HotKeysCmd {
	cmd := NewHotKeysCmd(ctx, "hotkeys", "get")
	_ = c.Process(ctx, cmd)
	return cmd
}
