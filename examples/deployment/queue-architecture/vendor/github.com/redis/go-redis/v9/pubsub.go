package redis

import (
	"context"
	"fmt"
	"maps"
	"slices"
	"strings"
	"sync"
	"time"

	"github.com/redis/go-redis/v9/internal"
	"github.com/redis/go-redis/v9/internal/otel"
	"github.com/redis/go-redis/v9/internal/pool"
	"github.com/redis/go-redis/v9/internal/proto"
	"github.com/redis/go-redis/v9/push"
)

// PubSub implements Pub/Sub commands as described in
// https://redis.io/docs/latest/develop/pubsub. Message receiving is NOT safe
// for concurrent use by multiple goroutines.
//
// PubSub automatically reconnects to Redis Server and resubscribes
// to the channels in case of network errors.
type PubSub struct {
	opt *Options

	newConn   func(ctx context.Context, addr string, channels []string) (*pool.Conn, error)
	closeConn func(*pool.Conn) error

	mu        sync.Mutex
	cn        *pool.Conn
	channels  map[string]struct{}
	patterns  map[string]struct{}
	schannels map[string]struct{}

	closed bool
	exit   chan struct{}

	cmd *Cmd

	chOnce sync.Once
	msgCh  *channel
	allCh  *channel

	// Push notification processor for handling generic push notifications
	pushProcessor push.NotificationProcessor

	// Cleanup callback for maintenanceNotifications upgrade tracking
	onClose func()
}

func (c *PubSub) init() {
	c.exit = make(chan struct{})
}

func (c *PubSub) String() string {
	c.mu.Lock()
	defer c.mu.Unlock()

	channels := slices.Collect(maps.Keys(c.channels))
	channels = append(channels, slices.Collect(maps.Keys(c.patterns))...)
	channels = append(channels, slices.Collect(maps.Keys(c.schannels))...)
	return fmt.Sprintf("PubSub(%s)", strings.Join(channels, ", "))
}

func (c *PubSub) connWithLock(ctx context.Context) (*pool.Conn, error) {
	c.mu.Lock()
	cn, err := c.conn(ctx, nil)
	c.mu.Unlock()
	return cn, err
}

func (c *PubSub) conn(ctx context.Context, newChannels []string) (*pool.Conn, error) {
	if c.closed {
		return nil, pool.ErrClosed
	}
	if c.cn != nil {
		return c.cn, nil
	}

	if c.opt.Addr == "" {
		// TODO(maintenanceNotifications):
		// this is probably cluster client
		// c.newConn will ignore the addr argument
		// will be changed when we have maintenanceNotifications upgrades for cluster clients
		c.opt.Addr = internal.RedisNull
	}

	// Include c.schannels so reconnect-time routing of an SSubscribe-only
	// PubSub picks the slot owner (channels[0] in ClusterClient.pubSub()'s
	// newConn closure) instead of a random node.
	// See https://github.com/redis/go-redis/issues/3806.
	// c.patterns is intentionally NOT included: patterns are not slot-
	// addressable, and adding them would force PSubscribe-only PubSubs to
	// pin to a single node based on pattern-string hash, regressing the
	// existing random-node behaviour.
	channels := slices.Collect(maps.Keys(c.channels))
	channels = append(channels, slices.Collect(maps.Keys(c.schannels))...)
	channels = append(channels, newChannels...)

	cn, err := c.newConn(ctx, c.opt.Addr, channels)
	if err != nil {
		return nil, err
	}

	if err := c.resubscribe(ctx, cn); err != nil {
		_ = c.closeConn(cn)
		return nil, err
	}

	c.cn = cn
	return cn, nil
}

func (c *PubSub) writeCmd(ctx context.Context, cn *pool.Conn, cmd Cmder) error {
	return cn.WithWriter(ctx, c.opt.WriteTimeout, func(wr *proto.Writer) error {
		return writeCmd(wr, cmd)
	})
}

func (c *PubSub) resubscribe(ctx context.Context, cn *pool.Conn) error {
	var firstErr error

	if len(c.channels) > 0 {
		firstErr = c._subscribe(ctx, cn, "subscribe", slices.Collect(maps.Keys(c.channels)))
	}

	if len(c.patterns) > 0 {
		err := c._subscribe(ctx, cn, "psubscribe", slices.Collect(maps.Keys(c.patterns)))
		if err != nil && firstErr == nil {
			firstErr = err
		}
	}

	if len(c.schannels) > 0 {
		err := c._subscribe(ctx, cn, "ssubscribe", slices.Collect(maps.Keys(c.schannels)))
		if err != nil && firstErr == nil {
			firstErr = err
		}
	}

	return firstErr
}

func (c *PubSub) _subscribe(
	ctx context.Context, cn *pool.Conn, redisCmd string, channels []string,
) error {
	args := make([]interface{}, 0, 1+len(channels))
	args = append(args, redisCmd)
	for _, channel := range channels {
		args = append(args, channel)
	}
	cmd := NewSliceCmd(ctx, args...)
	return c.writeCmd(ctx, cn, cmd)
}

func (c *PubSub) releaseConnWithLock(
	ctx context.Context,
	cn *pool.Conn,
	err error,
	allowTimeout bool,
) {
	c.mu.Lock()
	c.releaseConn(ctx, cn, err, allowTimeout)
	c.mu.Unlock()
}

func (c *PubSub) releaseConn(ctx context.Context, cn *pool.Conn, err error, allowTimeout bool) {
	if c.cn != cn {
		return
	}

	if !cn.IsUsable() || cn.ShouldHandoff() {
		c.reconnect(ctx, fmt.Errorf("pubsub: connection is not usable"))
		return
	}

	if isBadConn(err, allowTimeout, c.opt.Addr) {
		c.reconnect(ctx, err)
	}
}

func (c *PubSub) reconnect(ctx context.Context, reason error) {
	if c.cn != nil && c.cn.ShouldHandoff() {
		newEndpoint := c.cn.GetHandoffEndpoint()
		// If new endpoint is NULL, use the original address
		if newEndpoint == internal.RedisNull {
			newEndpoint = c.opt.Addr
		}

		if newEndpoint != "" {
			// Update the address in the options
			oldAddr := c.cn.RemoteAddr().String()
			c.opt.Addr = newEndpoint
			internal.Logger.Printf(ctx, "pubsub: reconnecting to new endpoint %s (was %s)", newEndpoint, oldAddr)
		}
	}
	_ = c.closeTheCn(reason)
	_, _ = c.conn(ctx, nil)
}

func (c *PubSub) closeTheCn(reason error) error {
	if c.cn == nil {
		return nil
	}
	err := c.closeConn(c.cn)
	c.cn = nil
	return err
}

func (c *PubSub) Close() error {
	c.mu.Lock()
	defer c.mu.Unlock()

	if c.closed {
		return pool.ErrClosed
	}
	c.closed = true
	close(c.exit)

	// Call cleanup callback if set
	if c.onClose != nil {
		c.onClose()
	}

	return c.closeTheCn(pool.ErrClosed)
}

// Subscribe the client to the specified channels. It returns
// empty subscription if there are no channels.
func (c *PubSub) Subscribe(ctx context.Context, channels ...string) error {
	c.mu.Lock()
	defer c.mu.Unlock()

	err := c.subscribe(ctx, "subscribe", channels...)
	if c.channels == nil {
		c.channels = make(map[string]struct{})
	}
	for _, s := range channels {
		c.channels[s] = struct{}{}
	}
	return err
}

// PSubscribe the client to the given patterns. It returns
// empty subscription if there are no patterns.
func (c *PubSub) PSubscribe(ctx context.Context, patterns ...string) error {
	c.mu.Lock()
	defer c.mu.Unlock()

	err := c.subscribe(ctx, "psubscribe", patterns...)
	if c.patterns == nil {
		c.patterns = make(map[string]struct{})
	}
	for _, s := range patterns {
		c.patterns[s] = struct{}{}
	}
	return err
}

// SSubscribe Subscribes the client to the specified shard channels.
func (c *PubSub) SSubscribe(ctx context.Context, channels ...string) error {
	c.mu.Lock()
	defer c.mu.Unlock()

	err := c.subscribe(ctx, "ssubscribe", channels...)
	if c.schannels == nil {
		c.schannels = make(map[string]struct{})
	}
	for _, s := range channels {
		c.schannels[s] = struct{}{}
	}
	return err
}

// Unsubscribe the client from the given channels, or from all of
// them if none is given.
func (c *PubSub) Unsubscribe(ctx context.Context, channels ...string) error {
	c.mu.Lock()
	defer c.mu.Unlock()

	if len(channels) > 0 {
		for _, channel := range channels {
			delete(c.channels, channel)
		}
	} else {
		// Unsubscribe from all channels.
		clear(c.channels)
	}

	err := c.subscribe(ctx, "unsubscribe", channels...)
	return err
}

// PUnsubscribe the client from the given patterns, or from all of
// them if none is given.
func (c *PubSub) PUnsubscribe(ctx context.Context, patterns ...string) error {
	c.mu.Lock()
	defer c.mu.Unlock()

	if len(patterns) > 0 {
		for _, pattern := range patterns {
			delete(c.patterns, pattern)
		}
	} else {
		// Unsubscribe from all patterns.
		clear(c.patterns)
	}

	err := c.subscribe(ctx, "punsubscribe", patterns...)
	return err
}

// SUnsubscribe unsubscribes the client from the given shard channels,
// or from all of them if none is given.
func (c *PubSub) SUnsubscribe(ctx context.Context, channels ...string) error {
	c.mu.Lock()
	defer c.mu.Unlock()

	if len(channels) > 0 {
		for _, channel := range channels {
			delete(c.schannels, channel)
		}
	} else {
		// Unsubscribe from all channels.
		clear(c.schannels)
	}

	err := c.subscribe(ctx, "sunsubscribe", channels...)
	return err
}

func (c *PubSub) subscribe(ctx context.Context, redisCmd string, channels ...string) error {
	cn, err := c.conn(ctx, channels)
	if err != nil {
		return err
	}

	err = c._subscribe(ctx, cn, redisCmd, channels)
	c.releaseConn(ctx, cn, err, false)
	return err
}

func (c *PubSub) Ping(ctx context.Context, payload ...string) error {
	args := []interface{}{"ping"}
	if len(payload) == 1 {
		args = append(args, payload[0])
	}
	cmd := NewCmd(ctx, args...)

	c.mu.Lock()
	defer c.mu.Unlock()

	cn, err := c.conn(ctx, nil)
	if err != nil {
		return err
	}

	err = c.writeCmd(ctx, cn, cmd)
	c.releaseConn(ctx, cn, err, false)
	return err
}

// ClientSetName assigns  a namee to the PubSub connection using  CLIENT SETNAME,
// The name is visible in CLIENT LIST output and is useful for debugging
// and identifying connections in a redis instance.
func (c *PubSub) ClientSetName(ctx context.Context, name string) error {
	cmd := NewStatusCmd(ctx, "client", "setname", name)

	c.mu.Lock()
	defer c.mu.Unlock()

	cn, err := c.conn(ctx, nil)
	if err != nil {
		return err
	}

	err = c.writeCmd(ctx, cn, cmd)
	c.releaseConn(ctx, cn, err, false)
	return err
}

// Subscription received after a successful subscription to channel.
type Subscription struct {
	// Can be "subscribe", "unsubscribe", "psubscribe" or "punsubscribe".
	Kind string
	// Channel name we have subscribed to.
	Channel string
	// Number of channels we are currently subscribed to.
	Count int
}

func (m *Subscription) String() string {
	return fmt.Sprintf("%s: %s", m.Kind, m.Channel)
}

// Message received as result of a PUBLISH command issued by another client.
type Message struct {
	Channel      string
	Pattern      string
	Payload      string
	PayloadSlice []string
}

func (m *Message) String() string {
	return fmt.Sprintf("Message<%s: %s>", m.Channel, m.Payload)
}

// Pong received as result of a PING command issued by another client.
type Pong struct {
	Payload string
}

func (p *Pong) String() string {
	if p.Payload != "" {
		return fmt.Sprintf("Pong<%s>", p.Payload)
	}
	return "Pong"
}

func (c *PubSub) newMessage(ctx context.Context, cn *pool.Conn, reply interface{}) (interface{}, error) {
	switch reply := reply.(type) {
	case string:
		return &Pong{
			Payload: reply,
		}, nil
	case []interface{}:
		switch kind := reply[0].(string); kind {
		case "subscribe", "unsubscribe", "psubscribe", "punsubscribe", "ssubscribe", "sunsubscribe":
			// Can be nil in case of "unsubscribe".
			channel, _ := reply[1].(string)
			return &Subscription{
				Kind:    kind,
				Channel: channel,
				Count:   int(reply[2].(int64)),
			}, nil
		case "message", "smessage":
			channel := reply[1].(string)
			sharded := kind == "smessage"
			switch payload := reply[2].(type) {
			case string:
				msg := &Message{
					Channel: channel,
					Payload: payload,
				}
				// Record PubSub message received
				otel.RecordPubSubMessage(ctx, cn, "received", channel, sharded)
				return msg, nil
			case []interface{}:
				ss := make([]string, len(payload))
				for i, s := range payload {
					ss[i] = s.(string)
				}
				msg := &Message{
					Channel:      channel,
					PayloadSlice: ss,
				}
				// Record PubSub message received
				otel.RecordPubSubMessage(ctx, cn, "received", channel, sharded)
				return msg, nil
			default:
				return nil, fmt.Errorf("redis: unsupported pubsub message payload: %T", payload)
			}
		case "pmessage":
			channel := reply[2].(string)
			msg := &Message{
				Pattern: reply[1].(string),
				Channel: channel,
				Payload: reply[3].(string),
			}
			// Record PubSub message received (pattern message, not sharded)
			otel.RecordPubSubMessage(ctx, cn, "received", channel, false)
			return msg, nil
		case "pong":
			return &Pong{
				Payload: reply[1].(string),
			}, nil
		default:
			return nil, fmt.Errorf("redis: unsupported pubsub message: %q", kind)
		}
	default:
		return nil, fmt.Errorf("redis: unsupported pubsub message: %#v", reply)
	}
}

// ReceiveTimeout acts like Receive but returns an error if message
// is not received in time. This is low-level API and in most cases
// Channel should be used instead.
func (c *PubSub) ReceiveTimeout(ctx context.Context, timeout time.Duration) (interface{}, error) {
	if c.cmd == nil {
		c.cmd = NewCmd(ctx)
	}

	// Don't hold the lock to allow subscriptions and pings.
	cn, err := c.connWithLock(ctx)
	if err != nil {
		return nil, err
	}

	err = cn.WithReader(ctx, timeout, func(rd *proto.Reader) error {
		// To be sure there are no buffered push notifications, we process them before reading the reply
		if err := c.processPendingPushNotificationWithReader(ctx, cn, rd); err != nil {
			// Log the error but don't fail the command execution
			// Push notification processing errors shouldn't break normal Redis operations
			internal.Logger.Printf(ctx, "push: conn[%d] error processing pending notifications before reading reply: %v", cn.GetID(), err)
		}
		return c.cmd.readReply(rd)
	})
	c.releaseConnWithLock(ctx, cn, err, timeout > 0)

	if err != nil {
		return nil, err
	}

	return c.newMessage(ctx, cn, c.cmd.Val())
}

// Receive returns a message as a Subscription, Message, Pong or error.
// See PubSub example for details. This is low-level API and in most cases
// Channel should be used instead.
// Receive returns a message as a Subscription, Message, Pong, or an error.
// See PubSub example for details. This is a low-level API and in most cases
// Channel should be used instead.
// This method blocks until a message is received or an error occurs.
// It may return early with an error if the context is canceled, the connection fails,
// or other internal errors occur.
func (c *PubSub) Receive(ctx context.Context) (interface{}, error) {
	return c.ReceiveTimeout(ctx, 0)
}

// ReceiveMessage returns a Message or error ignoring Subscription and Pong
// messages. This is low-level API and in most cases Channel should be used
// instead.
func (c *PubSub) ReceiveMessage(ctx context.Context) (*Message, error) {
	for {
		msg, err := c.Receive(ctx)
		if err != nil {
			return nil, err
		}

		switch msg := msg.(type) {
		case *Subscription:
			// Ignore.
		case *Pong:
			// Ignore.
		case *Message:
			return msg, nil
		default:
			err := fmt.Errorf("redis: unknown message: %T", msg)
			return nil, err
		}
	}
}

func (c *PubSub) getContext() context.Context {
	if c.cmd != nil {
		return c.cmd.ctx
	}
	return context.Background()
}

//------------------------------------------------------------------------------

// Channel returns a Go channel for concurrently receiving messages.
// The channel is closed together with the PubSub. If the Go channel
// is blocked full for 1 minute the message is dropped.
// Receive* APIs can not be used after channel is created.
//
// go-redis periodically sends ping messages to test connection health
// and re-subscribes if ping can not received for 1 minute.
func (c *PubSub) Channel(opts ...ChannelOption) <-chan *Message {
	c.chOnce.Do(func() {
		c.msgCh = newChannel(c, opts...)
		c.msgCh.initMsgChan()
	})
	if c.msgCh == nil {
		err := fmt.Errorf("redis: Channel can't be called after ChannelWithSubscriptions")
		panic(err)
	}
	return c.msgCh.msgCh
}

// ChannelSize is like Channel, but creates a Go channel
// with specified buffer size.
//
// Deprecated: use Channel(WithChannelSize(size)), remove in v9.
func (c *PubSub) ChannelSize(size int) <-chan *Message {
	return c.Channel(WithChannelSize(size))
}

// ChannelWithSubscriptions is like Channel, but message type can be either
// *Subscription or *Message. Subscription messages can be used to detect
// reconnections.
//
// ChannelWithSubscriptions can not be used together with Channel or ChannelSize.
func (c *PubSub) ChannelWithSubscriptions(opts ...ChannelOption) <-chan interface{} {
	c.chOnce.Do(func() {
		c.allCh = newChannel(c, opts...)
		c.allCh.initAllChan()
	})
	if c.allCh == nil {
		err := fmt.Errorf("redis: ChannelWithSubscriptions can't be called after Channel")
		panic(err)
	}
	return c.allCh.allCh
}

func (c *PubSub) processPendingPushNotificationWithReader(ctx context.Context, cn *pool.Conn, rd *proto.Reader) error {
	// Only process push notifications for RESP3 connections with a processor
	if c.opt.Protocol != 3 || c.pushProcessor == nil {
		return nil
	}

	// Create handler context with client, connection pool, and connection information
	handlerCtx := c.pushNotificationHandlerContext(cn)
	return c.pushProcessor.ProcessPendingNotifications(ctx, handlerCtx, rd)
}

func (c *PubSub) pushNotificationHandlerContext(cn *pool.Conn) push.NotificationHandlerContext {
	// PubSub doesn't have a client or connection pool, so we pass nil for those
	// PubSub connections are blocking
	return push.NotificationHandlerContext{
		PubSub:     c,
		Conn:       cn,
		IsBlocking: true,
	}
}

type ChannelOption func(c *channel)

// WithChannelSize specifies the Go chan size that is used to buffer incoming messages.
//
// The default is 100 messages.
func WithChannelSize(size int) ChannelOption {
	return func(c *channel) {
		c.chanSize = size
	}
}

// WithChannelHealthCheckInterval specifies the health check interval.
// PubSub will ping Redis Server if it does not receive any messages within the interval.
// To disable health check, use zero interval.
//
// The default is 3 seconds.
func WithChannelHealthCheckInterval(d time.Duration) ChannelOption {
	return func(c *channel) {
		c.checkInterval = d
	}
}

// WithChannelSendTimeout specifies the channel send timeout after which
// the message is dropped.
//
// The default is 60 seconds.
func WithChannelSendTimeout(d time.Duration) ChannelOption {
	return func(c *channel) {
		c.chanSendTimeout = d
	}
}

// WithChannelPingTimeout specifies the timeout for the health-check ping.
//
// The default is 5 seconds.
func WithChannelPingTimeout(d time.Duration) ChannelOption {
	return func(c *channel) {
		c.pingTimeout = d
	}
}

// WithChannelReconnectTimeout specifies the timeout for reconnecting after
// a failed health-check ping.
//
// The default is 10 seconds.
func WithChannelReconnectTimeout(d time.Duration) ChannelOption {
	return func(c *channel) {
		c.reconnectTimeout = d
	}
}

type channel struct {
	pubSub *PubSub

	msgCh chan *Message
	allCh chan interface{}
	ping  chan struct{}

	chanSize         int
	chanSendTimeout  time.Duration
	checkInterval    time.Duration
	pingTimeout      time.Duration
	reconnectTimeout time.Duration
}

func newChannel(pubSub *PubSub, opts ...ChannelOption) *channel {
	c := &channel{
		pubSub: pubSub,

		chanSize:         100,
		chanSendTimeout:  time.Minute,
		checkInterval:    3 * time.Second,
		pingTimeout:      5 * time.Second,
		reconnectTimeout: 10 * time.Second,
	}
	for _, opt := range opts {
		opt(c)
	}
	if c.checkInterval > 0 {
		c.initHealthCheck()
	}
	return c
}

func (c *channel) initHealthCheck() {
	c.ping = make(chan struct{}, 1)

	go func() {
		timer := time.NewTimer(time.Minute)
		timer.Stop()

		for {
			timer.Reset(c.checkInterval)
			select {
			case <-c.ping:
				select {
				case <-timer.C:
				default:
				}
			case <-timer.C:
				ctx, cancel := context.WithTimeout(context.Background(), c.pingTimeout)
				pingErr := c.pubSub.Ping(ctx)
				cancel()

				if pingErr != nil {
					c.pubSub.mu.Lock()
					reconnectCtx, reconnectCancel := context.WithTimeout(context.Background(), c.reconnectTimeout)
					c.pubSub.reconnect(reconnectCtx, pingErr)
					reconnectCancel()
					c.pubSub.mu.Unlock()
				}
			case <-c.pubSub.exit:
				return
			}
		}
	}()
}

// initMsgChan must be in sync with initAllChan.
func (c *channel) initMsgChan() {
	ctx := context.TODO()
	c.msgCh = make(chan *Message, c.chanSize)

	go func() {
		timer := time.NewTimer(time.Minute)
		timer.Stop()

		var errCount int
		for {
			msg, err := c.pubSub.Receive(ctx)
			if err != nil {
				if err == pool.ErrClosed {
					close(c.msgCh)
					return
				}
				if errCount > 0 {
					time.Sleep(100 * time.Millisecond)
				}
				errCount++
				continue
			}

			errCount = 0

			// Any message is as good as a ping.
			select {
			case c.ping <- struct{}{}:
			default:
			}

			switch msg := msg.(type) {
			case *Subscription:
				// Ignore.
			case *Pong:
				// Ignore.
			case *Message:
				timer.Reset(c.chanSendTimeout)
				select {
				case c.msgCh <- msg:
					if !timer.Stop() {
						<-timer.C
					}
				case <-timer.C:
					internal.Logger.Printf(
						ctx, "redis: %v channel is full for %s (message is dropped)",
						c, c.chanSendTimeout)
				}
			default:
				internal.Logger.Printf(ctx, "redis: unknown message type: %T", msg)
			}
		}
	}()
}

// initAllChan must be in sync with initMsgChan.
func (c *channel) initAllChan() {
	ctx := context.TODO()
	c.allCh = make(chan interface{}, c.chanSize)

	go func() {
		timer := time.NewTimer(time.Minute)
		timer.Stop()

		var errCount int
		for {
			msg, err := c.pubSub.Receive(ctx)
			if err != nil {
				if err == pool.ErrClosed {
					close(c.allCh)
					return
				}
				if errCount > 0 {
					time.Sleep(100 * time.Millisecond)
				}
				errCount++
				continue
			}

			errCount = 0

			// Any message is as good as a ping.
			select {
			case c.ping <- struct{}{}:
			default:
			}

			switch msg := msg.(type) {
			case *Pong:
				// Ignore.
			case *Subscription, *Message:
				timer.Reset(c.chanSendTimeout)
				select {
				case c.allCh <- msg:
					if !timer.Stop() {
						<-timer.C
					}
				case <-timer.C:
					internal.Logger.Printf(
						ctx, "redis: %v channel is full for %s (message is dropped)",
						c, c.chanSendTimeout)
				}
			default:
				internal.Logger.Printf(ctx, "redis: unknown message type: %T", msg)
			}
		}
	}()
}
