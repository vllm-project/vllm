package queue

import (
	"context"
	"fmt"
	"sync"
	"time"

	"github.com/nats-io/nats.go"
	"github.com/nats-io/nats.go/jetstream"
)

const (
	defaultStreamName    = "vllm_requests"
	defaultStreamSubject = "vllm.requests"

	maxMsgSizeBytes = 10 << 20 // 10 MiB
	maxStreamBytes  = 256 << 20
)

// Config holds NATS JetStream queue settings, typically loaded from env.
type Config struct {
	NATSURL       string
	StreamName    string
	StreamSubject string
	ConsumerName  string
}

// Client is a NATS JetStream-backed work queue.
type Client struct {
	nc  *nats.Conn
	js  jetstream.JetStream
	cfg Config
	o   sync.Once
}

type ClientOpts = func(Config) Config

func WithNatsURL(url string) ClientOpts {
	return func(c Config) Config {
		c.NATSURL = url
		return c
	}
}

func WithStreamName(name string) ClientOpts {
	return func(c Config) Config {
		if name != "" {
			c.StreamName = name
		}
		return c
	}
}

func WithStreamSubject(subj string) ClientOpts {
	return func(c Config) Config {
		if subj != "" {
			c.StreamSubject = subj
		}
		return c
	}
}

func NewClient(ctx context.Context, opts ...ClientOpts) *Client {
	cfg := Config{
		StreamName:    defaultStreamName,
		StreamSubject: defaultStreamSubject,
	}

	for _, o := range opts {
		cfg = o(cfg)
	}

	return &Client{
		cfg: cfg,
	}
}

// EnsureStream idempotently creates or updates the work-queue stream.
func (c *Client) EnsureStream(ctx context.Context) error {
	_, err := c.js.CreateOrUpdateStream(ctx, jetstream.StreamConfig{
		Name:       c.cfg.StreamName,
		Subjects:   []string{c.cfg.StreamSubject},
		Storage:    jetstream.MemoryStorage,
		Retention:  jetstream.WorkQueuePolicy,
		Discard:    jetstream.DiscardNew,
		MaxMsgSize: maxMsgSizeBytes,
		MaxBytes:   maxStreamBytes,
		MaxAge:     time.Hour,
		Replicas:   1,
	})
	if err != nil {
		return fmt.Errorf("ensure stream %s: %w", c.cfg.StreamName, err)
	}
	return nil
}

// Close shuts down the underlying NATS connection.
func (c *Client) Close() {
	if c.nc != nil {
		c.nc.Close()
	}
}

// Config returns the client configuration.
func (c *Client) Config() Config {
	return c.cfg
}

// Conn returns the core NATS connection (e.g. for inbox subscriptions).
func (c *Client) Conn() *nats.Conn {
	return c.nc
}

func (c *Client) Mail(replyTo string, data []byte) error {
	if replyTo == "" {
		return fmt.Errorf("replyTo required")
	}

	if c.nc == nil {
		return fmt.Errorf("no connection found")
	}

	err := c.nc.Publish(replyTo, data)
	if err != nil {
		return fmt.Errorf("error publishing data: %w", err)
	}

	return nil
}

func (c *Client) dial(ctx context.Context) error {
	var err error

	c.o.Do(func() {
		conn, js, dialErr := c._dial()
		if dialErr != nil {
			err = dialErr
			return
		}

		c.nc = conn
		c.js = js
	})

	return err
}

func (c *Client) _dial() (*nats.Conn, jetstream.JetStream, error) {
	if c.cfg.NATSURL == "" {
		return nil, nil, fmt.Errorf("NATS_URL is required")
	}

	nc, err := nats.Connect(c.cfg.NATSURL)
	if err != nil {
		return nil, nil, fmt.Errorf("connect to nats failed: %w", err)
	}

	js, err := jetstream.New(nc)
	if err != nil {
		nc.Close()
		return nil, nil, fmt.Errorf("create jetstream failed: %w", err)
	}

	return nc, js, nil
}
