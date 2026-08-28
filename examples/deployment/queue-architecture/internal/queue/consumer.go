package queue

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"time"

	"github.com/nats-io/nats.go/jetstream"
)

const (
	defaultConsumerName  = "vllm-sidecars"
	defaultMaxAckPending = 2
	defaultMaxDeliver    = 2
	defaultAckWait       = 30 * time.Second
	defaultAckPolicy     = jetstream.AckExplicitPolicy
	defaultDeliverPolicy = jetstream.DeliverAllPolicy

	defaultFetchPollInterval = 1 * time.Second
	defaultBatchSize         = 1
)

type ConsumerConfig struct {
	ConsumerName      string
	MaxDeliver        int
	AckWait           time.Duration
	MaxAckPending     int
	AckPolicy         jetstream.AckPolicy
	DeliverPolicy     jetstream.DeliverPolicy
	FetchPollInterval time.Duration
	FetchBatchSize    int
}

type ConsumerOpts = func(ConsumerConfig) ConsumerConfig

func WithConsumerName(name string) ConsumerOpts {
	return func(c ConsumerConfig) ConsumerConfig {
		c.ConsumerName = name
		return c
	}
}

func WithMaxDeliver(maxDeliver int) ConsumerOpts {
	return func(c ConsumerConfig) ConsumerConfig {
		c.MaxDeliver = maxDeliver
		return c
	}
}

func WithAckWait(ackWait time.Duration) ConsumerOpts {
	return func(c ConsumerConfig) ConsumerConfig {
		c.AckWait = ackWait
		return c
	}
}

func WithMaxAckPending(maxAckPending int) ConsumerOpts {
	return func(c ConsumerConfig) ConsumerConfig {
		c.MaxAckPending = maxAckPending
		return c
	}
}

func WithAckPolicy(ackPolicy jetstream.AckPolicy) ConsumerOpts {
	return func(c ConsumerConfig) ConsumerConfig {
		c.AckPolicy = ackPolicy
		return c
	}
}

func WithDeliverPolicy(deliverPolicy jetstream.DeliverPolicy) ConsumerOpts {
	return func(c ConsumerConfig) ConsumerConfig {
		c.DeliverPolicy = deliverPolicy
		return c
	}
}

func WithFetchPollInterval(interval time.Duration) ConsumerOpts {
	return func(c ConsumerConfig) ConsumerConfig {
		c.FetchPollInterval = interval
		return c
	}
}

func WithFetchBatchSize(batchSize int) ConsumerOpts {
	return func(c ConsumerConfig) ConsumerConfig {
		c.FetchBatchSize = batchSize
		return c
	}
}

type Consumer struct {
	client   *Client
	cfg      ConsumerConfig
	consumer jetstream.Consumer
}

func NewConsumer(client *Client, opts ...ConsumerOpts) *Consumer {
	cfg := ConsumerConfig{
		ConsumerName:      defaultConsumerName,
		MaxDeliver:        defaultMaxDeliver,
		AckWait:           defaultAckWait,
		MaxAckPending:     defaultMaxAckPending,
		AckPolicy:         defaultAckPolicy,
		DeliverPolicy:     defaultDeliverPolicy,
		FetchPollInterval: defaultFetchPollInterval,
		FetchBatchSize:    defaultBatchSize,
	}

	for _, o := range opts {
		cfg = o(cfg)
	}
	return &Consumer{
		client: client,
		cfg:    cfg,
	}
}

func (c *Consumer) Connect(ctx context.Context) error {
	err := c.client.dial(ctx)
	if err != nil {
		return fmt.Errorf("error dialing: %w", err)
	}

	err = c.client.EnsureStream(ctx)
	if err != nil {
		c.client.Close()
		return fmt.Errorf("error ensuring stream: %w", err)
	}

	consumer, err := c.client.js.CreateOrUpdateConsumer(
		ctx,
		c.client.cfg.StreamName,
		jetstream.ConsumerConfig{
			Durable:       c.cfg.ConsumerName,
			FilterSubject: c.client.cfg.StreamSubject,
			AckPolicy:     c.cfg.AckPolicy,
			DeliverPolicy: c.cfg.DeliverPolicy,
			MaxDeliver:    c.cfg.MaxDeliver,
			AckWait:       c.cfg.AckWait,
			MaxAckPending: c.cfg.MaxAckPending,
		},
	)
	if err != nil {
		c.client.Close()
		return fmt.Errorf("error running ensure consumer %s: %w", c.cfg.ConsumerName, err)
	}

	c.consumer = consumer
	return nil
}

func (c *Consumer) Close() {
	c.client.Close()
}

func (c *Consumer) Config() ConsumerConfig {
	return c.cfg
}

// Fetch pulls up to batchSize messages from the shared durable consumer.
// batchSize <= 0 uses the consumer's configured FetchBatchSize.
// Blocks until at least one valid message is available or ctx is cancelled.
func (c *Consumer) Fetch(ctx context.Context, batchSize int) ([]Message, error) {
	if c.consumer == nil {
		return nil, fmt.Errorf("consumer not initialized; call Connect")
	}
	if batchSize <= 0 {
		batchSize = c.cfg.FetchBatchSize
	}
	if batchSize <= 0 {
		batchSize = 1
	}

	poll := c.cfg.FetchPollInterval
	if poll <= 0 {
		poll = defaultFetchPollInterval
	}

	msgs := make([]Message, 0, batchSize)
	for {
		if err := ctx.Err(); err != nil {
			return nil, err
		}

		batch, err := c.consumer.Fetch(batchSize, jetstream.FetchMaxWait(poll))
		if err != nil {
			return nil, fmt.Errorf("fetch message error: %w", err)
		}

		var termErr error
		for m := range batch.Messages() {
			if termErr != nil {
				continue
			}

			msg := Message{Msg: m}
			if err := json.Unmarshal(m.Data(), &msg.Job); err != nil {
				if err := msg.Term(ctx); err != nil {
					termErr = fmt.Errorf("term error: %w", err)
					continue
				}
				slog.ErrorContext(ctx, "bad message", "err", err)
				continue
			}
			msgs = append(msgs, msg)
		}

		if termErr != nil {
			return nil, termErr
		}
		if err := batch.Error(); err != nil {
			if len(msgs) > 0 {
				return msgs, nil
			}
			if errors.Is(err, context.Canceled) || errors.Is(err, context.DeadlineExceeded) {
				if ctx.Err() != nil {
					return nil, ctx.Err()
				}
				continue
			}
			return nil, fmt.Errorf("batch error: %w", err)
		}

		if len(msgs) > 0 {
			return msgs, nil
		}
	}
}

func (c *Consumer) Mail(recipient string, data []byte) error {
	return c.client.Mail(recipient, data)
}
