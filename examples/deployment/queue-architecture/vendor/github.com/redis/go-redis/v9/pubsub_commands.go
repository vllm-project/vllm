package redis

import (
	"context"

	"github.com/redis/go-redis/v9/internal/otel"
)

type PubSubCmdable interface {
	Publish(ctx context.Context, channel string, message interface{}) *IntCmd
	SPublish(ctx context.Context, channel string, message interface{}) *IntCmd
	PubSubChannels(ctx context.Context, pattern string) *StringSliceCmd
	PubSubNumSub(ctx context.Context, channels ...string) *MapStringIntCmd
	PubSubNumPat(ctx context.Context) *IntCmd
	PubSubShardChannels(ctx context.Context, pattern string) *StringSliceCmd
	PubSubShardNumSub(ctx context.Context, channels ...string) *MapStringIntCmd
}

// Publish posts the message to the channel.
func (c cmdable) Publish(ctx context.Context, channel string, message interface{}) *IntCmd {
	cmd := NewIntCmd(ctx, "publish", channel, message)
	_ = c(ctx, cmd)
	// Record PubSub message sent (if command succeeded). Gated on the result
	// being readable WITHOUT blocking: on the deferred autopipeline face the
	// call above only enqueues, so reading the outcome here would await the
	// batch and turn a fire-and-forget publish into a blocking call — i.e.
	// enabling telemetry would change the async call shape (review finding by
	// codex on #3942). The metric is therefore skipped for a submission that
	// has not executed yet; recording it from the execution path instead is a
	// follow-up in the OTel wiring, not something the command wrapper can do.
	if otel.Enabled() && cmd.resultReady() && cmd.rawErr() == nil {
		otel.RecordPubSubMessage(ctx, nil, "sent", channel, false)
	}
	return cmd
}

func (c cmdable) SPublish(ctx context.Context, channel string, message interface{}) *IntCmd {
	cmd := NewIntCmd(ctx, "spublish", channel, message)
	_ = c(ctx, cmd)
	// Record PubSub message sent (if command succeeded). See Publish for why
	// this is gated on the result being readable without blocking.
	if otel.Enabled() && cmd.resultReady() && cmd.rawErr() == nil {
		otel.RecordPubSubMessage(ctx, nil, "sent", channel, true)
	}
	return cmd
}

func (c cmdable) PubSubChannels(ctx context.Context, pattern string) *StringSliceCmd {
	args := []interface{}{"pubsub", "channels"}
	if pattern != "*" {
		args = append(args, pattern)
	}
	cmd := NewStringSliceCmd(ctx, args...)
	_ = c(ctx, cmd)
	return cmd
}

func (c cmdable) PubSubNumSub(ctx context.Context, channels ...string) *MapStringIntCmd {
	args := make([]interface{}, 2+len(channels))
	args[0] = "pubsub"
	args[1] = "numsub"
	for i, channel := range channels {
		args[2+i] = channel
	}
	cmd := NewMapStringIntCmd(ctx, args...)
	_ = c(ctx, cmd)
	return cmd
}

func (c cmdable) PubSubShardChannels(ctx context.Context, pattern string) *StringSliceCmd {
	args := []interface{}{"pubsub", "shardchannels"}
	if pattern != "*" {
		args = append(args, pattern)
	}
	cmd := NewStringSliceCmd(ctx, args...)
	_ = c(ctx, cmd)
	return cmd
}

func (c cmdable) PubSubShardNumSub(ctx context.Context, channels ...string) *MapStringIntCmd {
	args := make([]interface{}, 2+len(channels))
	args[0] = "pubsub"
	args[1] = "shardnumsub"
	for i, channel := range channels {
		args[2+i] = channel
	}
	cmd := NewMapStringIntCmd(ctx, args...)
	_ = c(ctx, cmd)
	return cmd
}

func (c cmdable) PubSubNumPat(ctx context.Context) *IntCmd {
	cmd := NewIntCmd(ctx, "pubsub", "numpat")
	_ = c(ctx, cmd)
	return cmd
}
