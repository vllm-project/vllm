package queue

import (
	"context"
	"encoding/json"
	"fmt"

	"github.com/nats-io/nats.go"
	"github.com/rvo-redplatform/vllm/examples/deployment/queue-architecture/internal/model"
)

const natsReplyToHeader = "Nats-Reply-To"

type Producer struct {
	client *Client
}

func NewProducer(client *Client) *Producer {
	return &Producer{
		client: client,
	}
}

func (p *Producer) Connect(ctx context.Context) error {
	err := p.client.dial(ctx)
	if err != nil {
		return err
	}

	err = p.client.EnsureStream(ctx)
	if err != nil {
		p.client.Close()
		return err
	}

	return nil
}

// Enqueue publishes a job to the JetStream work queue.
func (c *Producer) Enqueue(ctx context.Context, job model.Job) error {
	jobJSON, err := json.Marshal(job)
	if err != nil {
		return fmt.Errorf("marshal job: %w", err)
	}

	msg := &nats.Msg{
		Subject: c.client.cfg.StreamSubject,
		Data:    jobJSON,
	}
	if job.ReplyTo != "" {
		msg.Header = nats.Header{
			natsReplyToHeader: []string{job.ReplyTo},
		}
	}

	if _, err := c.client.js.PublishMsg(ctx, msg); err != nil {
		return fmt.Errorf("publish to %s: %w", c.client.cfg.StreamSubject, err)
	}
	return nil
}

func (c *Producer) Close() {
	c.client.Close()
}

func (c *Producer) SubscribeSync() (string, *nats.Subscription, error) {
	inbox := nats.NewInbox()
	sub, err := c.client.Conn().SubscribeSync(inbox)
	return inbox, sub, err
}
