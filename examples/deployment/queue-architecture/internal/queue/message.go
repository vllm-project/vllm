package queue

import (
	"context"
	"errors"
	"fmt"

	"github.com/nats-io/nats.go/jetstream"
	"github.com/rvo-redplatform/vllm/examples/deployment/queue-architecture/internal/model"
)

// FetchedMessage is a pulled work item and its JetStream message handle.
type Message struct {
	Job model.Job
	Msg jetstream.Msg
}

// ReplyTo returns the reply inbox from the NATS header, falling back to the
// JSON field on the job.
func (m Message) ReplyTo() string {
	if m.Msg != nil {
		if values := m.Msg.Headers().Get(natsReplyToHeader); values != "" {
			return values
		}
	}
	return m.Job.ReplyTo
}

// Ack acknowledges successful processing of a work message.
func (m Message) Ack(ctx context.Context) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	if m.Msg == nil {
		return errors.New("ack error: nil message")
	}
	if err := m.Msg.Ack(); err != nil {
		return fmt.Errorf("ack message error: %w", err)
	}
	return nil
}

// Term terminates a work message so it will not be redelivered.
func (m Message) Term(ctx context.Context) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	if m.Msg == nil {
		return errors.New("term: nil message")
	}
	if err := m.Msg.Term(); err != nil {
		return fmt.Errorf("term message: %w", err)
	}
	return nil
}

// InProgress resets the ack wait timer for an in-flight work message.
func (m Message) InProgress(ctx context.Context) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	if m.Msg == nil {
		return errors.New("in progress: nil message")
	}
	if err := m.Msg.InProgress(); err != nil {
		return fmt.Errorf("in progress message: %w", err)
	}
	return nil
}
