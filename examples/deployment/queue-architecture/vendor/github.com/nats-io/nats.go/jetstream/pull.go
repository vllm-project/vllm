// Copyright 2022-2025 The NATS Authors
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package jetstream

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"slices"
	"sync"
	"sync/atomic"
	"time"

	"github.com/nats-io/nats.go"
	"github.com/nats-io/nats.go/internal/syncx"
	"github.com/nats-io/nuid"
)

type (
	// MessagesContext supports iterating over a messages on a stream.
	// It is returned by [Consumer.Messages] method.
	MessagesContext interface {
		// Next retrieves next message on a stream. If MessagesContext is closed
		// (either stopped or drained), Next will return ErrMsgIteratorClosed
		// error. An optional timeout or context can be provided using NextOpt
		// options. If none are provided, Next will block indefinitely until a
		// message is available, iterator is closed or a heartbeat error occurs.
		Next(opts ...NextOpt) (Msg, error)

		// Stop unsubscribes from the stream and cancels subscription. Calling
		// Next after calling Stop will return ErrMsgIteratorClosed error.
		// All messages that are already in the buffer are discarded.
		Stop()

		// Drain unsubscribes from the stream and cancels subscription. All
		// messages that are already in the buffer will be available on
		// subsequent calls to Next. After the buffer is drained, Next will
		// return ErrMsgIteratorClosed error.
		Drain()
	}

	// ConsumeContext supports processing incoming messages from a stream.
	// It is returned by [Consumer.Consume] method.
	ConsumeContext interface {
		// Stop unsubscribes from the stream and cancels subscription.
		// No more messages will be received after calling this method.
		// All messages that are already in the buffer are discarded.
		Stop()

		// Drain unsubscribes from the stream and cancels subscription.
		// All messages that are already in the buffer will be processed in callback function.
		Drain()

		// Closed returns a channel that is closed when the consuming is
		// fully stopped/drained. When the channel is closed, no more messages
		// will be received and processing is complete.
		Closed() <-chan struct{}
	}

	// MessageHandler is a handler function used as callback in [Consume].
	MessageHandler func(msg Msg)

	// PullConsumeOpt represent additional options used in [Consume] for pull consumers.
	PullConsumeOpt interface {
		configureConsume(*consumeOpts) error
	}

	// PullMessagesOpt represent additional options used in [Messages] for pull consumers.
	PullMessagesOpt interface {
		configureMessages(*consumeOpts) error
	}

	pullConsumer struct {
		sync.Mutex
		js      *jetStream
		stream  string
		durable bool
		name    string
		info    *ConsumerInfo
		subs    syncx.Map[string, *pullSubscription]
		pinID   string
	}

	pullRequest struct {
		Expires       time.Duration   `json:"expires,omitempty"`
		Batch         int             `json:"batch,omitempty"`
		MaxBytes      int             `json:"max_bytes,omitempty"`
		NoWait        bool            `json:"no_wait,omitempty"`
		Heartbeat     time.Duration   `json:"idle_heartbeat,omitempty"`
		MinPending    int64           `json:"min_pending,omitempty"`
		MinAckPending int64           `json:"min_ack_pending,omitempty"`
		PinID         string          `json:"id,omitempty"`
		Group         string          `json:"group,omitempty"`
		Priority      uint8           `json:"priority,omitempty"`
		ctx           context.Context `json:"-"`
		maxWaitSet    bool            `json:"-"`
	}

	consumeOpts struct {
		Expires                 time.Duration
		MaxMessages             int
		MaxBytes                int
		LimitSize               bool
		MinPending              int64
		MinAckPending           int64
		Priority                uint8
		Group                   string
		Heartbeat               time.Duration
		ErrHandler              ConsumeErrHandler
		ReportMissingHeartbeats bool
		ThresholdMessages       int
		ThresholdBytes          int
		StopAfter               int
		stopAfterMsgsLeft       chan int
		notifyOnReconnect       bool
	}

	ConsumeErrHandlerFunc func(consumeCtx ConsumeContext, err error)

	pullSubscription struct {
		sync.Mutex
		id                string
		consumer          *pullConsumer
		subscription      *nats.Subscription
		msgs              chan *nats.Msg
		msgsClosed        atomic.Uint32
		errs              chan error
		pending           pendingMsgs
		hbMonitor         *hbMonitor
		fetchInProgress   atomic.Uint32
		closed            atomic.Uint32
		draining          atomic.Uint32
		done              chan struct{}
		connStatusChanged chan nats.Status
		fetchNext         chan *pullRequest
		consumeOpts       *consumeOpts
		delivered         int
		closedCh          chan struct{}
	}

	pendingMsgs struct {
		msgCount  int
		byteCount int
	}

	MessageBatch interface {
		Messages() <-chan Msg
		Error() error
	}

	fetchResult struct {
		sync.Mutex
		msgs chan Msg
		err  error
		done bool
		sseq uint64
	}

	FetchOpt func(*pullRequest) error

	hbMonitor struct {
		timer *time.Timer
		sync.Mutex
	}

	// NextOpt is an option for configuring the behavior of MessagesContext.Next.
	NextOpt interface {
		configureNext(*nextOpts)
	}

	nextOpts struct {
		timeout time.Duration
		ctx     context.Context
	}
)

const (
	DefaultMaxMessages       = 500
	DefaultExpires           = 30 * time.Second
	defaultBatchMaxBytesOnly = 1_000_000
	unset                    = -1
)

// Consume can be used to continuously receive messages and handle them
// with the provided callback function. Consume cannot be used concurrently
// when using ordered consumer.
//
// See [Consumer.Consume] for more details.
func (p *pullConsumer) Consume(handler MessageHandler, opts ...PullConsumeOpt) (ConsumeContext, error) {
	if handler == nil {
		return nil, ErrHandlerRequired
	}
	consumeOpts, err := parseConsumeOpts(false, opts...)
	if err != nil {
		return nil, fmt.Errorf("%w: %s", ErrInvalidOption, err)
	}

	if len(p.info.Config.PriorityGroups) != 0 {
		if consumeOpts.Group == "" {
			return nil, fmt.Errorf("%w: %s", ErrInvalidOption, "priority group is required for priority consumer")
		}

		if !slices.Contains(p.info.Config.PriorityGroups, consumeOpts.Group) {
			return nil, fmt.Errorf("%w: %s", ErrInvalidOption, "invalid priority group")
		}
	} else if consumeOpts.Group != "" {
		return nil, fmt.Errorf("%w: %s", ErrInvalidOption, "priority group is not supported for this consumer")
	}

	p.Lock()

	subject := p.js.apiSubject(fmt.Sprintf(apiRequestNextT, p.stream, p.name))

	consumeID := nuid.Next()
	sub := &pullSubscription{
		id:          consumeID,
		consumer:    p,
		errs:        make(chan error, 10),
		done:        make(chan struct{}, 1),
		fetchNext:   make(chan *pullRequest, 1),
		consumeOpts: consumeOpts,
	}
	sub.connStatusChanged = p.js.conn.StatusChanged(nats.CONNECTED, nats.RECONNECTING, nats.CLOSED)

	sub.hbMonitor = sub.scheduleHeartbeatCheck(consumeOpts.Heartbeat)

	p.subs.Store(sub.id, sub)
	p.Unlock()

	internalHandler := func(msg *nats.Msg) {
		if sub.hbMonitor != nil {
			sub.hbMonitor.Stop()
		}
		userMsg, msgErr := checkMsg(msg)
		if !userMsg && msgErr == nil {
			if sub.hbMonitor != nil {
				sub.hbMonitor.Reset(2 * consumeOpts.Heartbeat)
			}
			return
		}
		if !userMsg {
			// heartbeat message
			if msgErr == nil {
				return
			}

			sub.Lock()
			termErr, notifyErr := sub.handleStatusMsg(msg, msgErr)
			if termErr == nil {
				sub.checkPending()
				if sub.hbMonitor != nil {
					sub.hbMonitor.Reset(2 * consumeOpts.Heartbeat)
				}
			}
			sub.Unlock()

			if sub.consumeOpts.ErrHandler != nil && notifyErr != nil {
				sub.consumeOpts.ErrHandler(sub, notifyErr)
			}
			if termErr != nil {
				if sub.closed.Load() == 1 {
					return
				}
				if sub.consumeOpts.ErrHandler != nil {
					sub.consumeOpts.ErrHandler(sub, termErr)
				}
				sub.Stop()
			}
			return
		}
		if pinId := msg.Header.Get("Nats-Pin-Id"); pinId != "" {
			p.setPinID(pinId)
		}
		handler(p.js.toJSMsg(msg))
		sub.Lock()
		sub.decrementPendingMsgs(msg)
		sub.incrementDeliveredMsgs()
		sub.checkPending()
		if sub.hbMonitor != nil {
			sub.hbMonitor.Reset(2 * consumeOpts.Heartbeat)
		}
		sub.Unlock()

		if sub.consumeOpts.StopAfter > 0 && sub.consumeOpts.StopAfter == sub.delivered {
			sub.Stop()
		}
	}
	inbox := p.js.conn.NewInbox()
	sub.subscription, err = p.js.conn.Subscribe(inbox, internalHandler)
	if err != nil {
		return nil, err
	}
	sub.subscription.SetClosedHandler(func(sid string) func(string) {
		return func(subject string) {
			p.subs.Delete(sid)
			sub.draining.CompareAndSwap(1, 0)
			sub.Lock()
			if sub.closedCh != nil {
				close(sub.closedCh)
				sub.closedCh = nil
			}
			sub.Unlock()
		}
	}(sub.id))

	sub.Lock()
	// initial pull
	sub.resetPendingMsgs()
	batchSize := sub.consumeOpts.MaxMessages
	if sub.consumeOpts.StopAfter > 0 {
		batchSize = min(batchSize, sub.consumeOpts.StopAfter-sub.delivered)
	}
	if err := sub.pull(&pullRequest{
		Expires:       consumeOpts.Expires,
		Batch:         batchSize,
		MaxBytes:      consumeOpts.MaxBytes,
		Heartbeat:     consumeOpts.Heartbeat,
		MinPending:    consumeOpts.MinPending,
		MinAckPending: consumeOpts.MinAckPending,
		Priority:      consumeOpts.Priority,
		Group:         consumeOpts.Group,
		PinID:         p.getPinID(),
	}, subject); err != nil {
		sub.errs <- err
	}
	sub.Unlock()

	go func() {
		isConnected := true
		for {
			if sub.closed.Load() == 1 {
				return
			}
			select {
			case status, ok := <-sub.connStatusChanged:
				if !ok {
					continue
				}
				switch status {
				case nats.RECONNECTING:
					if sub.hbMonitor != nil {
						sub.hbMonitor.Stop()
					}
					isConnected = false
				case nats.CONNECTED:
					sub.Lock()
					if !isConnected {
						isConnected = true
						if sub.consumeOpts.notifyOnReconnect {
							sub.errs <- errConnected
						}

						sub.fetchNext <- &pullRequest{
							Expires:       sub.consumeOpts.Expires,
							Batch:         sub.consumeOpts.MaxMessages,
							MaxBytes:      sub.consumeOpts.MaxBytes,
							Heartbeat:     sub.consumeOpts.Heartbeat,
							MinPending:    sub.consumeOpts.MinPending,
							MinAckPending: sub.consumeOpts.MinAckPending,
							Priority:      sub.consumeOpts.Priority,
							Group:         sub.consumeOpts.Group,
							PinID:         p.getPinID(),
						}
						if sub.hbMonitor != nil {
							sub.hbMonitor.Reset(2 * sub.consumeOpts.Heartbeat)
						}
						sub.resetPendingMsgs()
					}
					sub.Unlock()

				case nats.CLOSED:
					sub.errs <- ErrConnectionClosed
				}
			case err := <-sub.errs:
				sub.Lock()
				if errors.Is(err, ErrNoHeartbeat) {
					batchSize := sub.consumeOpts.MaxMessages
					if sub.consumeOpts.StopAfter > 0 {
						batchSize = min(batchSize, sub.consumeOpts.StopAfter-sub.delivered)
					}
					sub.fetchNext <- &pullRequest{
						Expires:       sub.consumeOpts.Expires,
						Batch:         batchSize,
						MaxBytes:      sub.consumeOpts.MaxBytes,
						Heartbeat:     sub.consumeOpts.Heartbeat,
						MinPending:    sub.consumeOpts.MinPending,
						MinAckPending: sub.consumeOpts.MinAckPending,
						Priority:      sub.consumeOpts.Priority,
						Group:         sub.consumeOpts.Group,
						PinID:         p.getPinID(),
					}
					if sub.hbMonitor != nil {
						sub.hbMonitor.Reset(2 * sub.consumeOpts.Heartbeat)
					}
					sub.resetPendingMsgs()
				}
				sub.Unlock()
				if sub.consumeOpts.ErrHandler != nil {
					sub.consumeOpts.ErrHandler(sub, err)
				}
				if errors.Is(err, ErrConnectionClosed) {
					sub.Stop()
				}
			case <-sub.done:
				return
			}
		}
	}()

	go sub.pullMessages(subject)

	return sub, nil
}

// resetPendingMsgs resets pending message count and byte count
// to the values set in consumeOpts
// lock should be held before calling this method
func (s *pullSubscription) resetPendingMsgs() {
	s.pending.msgCount = s.consumeOpts.MaxMessages
	s.pending.byteCount = s.consumeOpts.MaxBytes
}

// decrementPendingMsgs decrements pending message count and byte count
// lock should be held before calling this method
func (s *pullSubscription) decrementPendingMsgs(msg *nats.Msg) {
	s.pending.msgCount--
	if s.consumeOpts.MaxBytes != 0 && !s.consumeOpts.LimitSize {
		s.pending.byteCount -= msg.Size()
	}
}

// incrementDeliveredMsgs increments delivered message count
// lock should be held before calling this method
func (s *pullSubscription) incrementDeliveredMsgs() {
	s.delivered++
}

// checkPending verifies whether there are enough messages in
// the buffer to trigger a new pull request.
// lock should be held before calling this method
func (s *pullSubscription) checkPending() {
	// check if we went below any threshold
	// we don't want to track bytes threshold if either it's not set or we used
	// PullMaxMessagesWithBytesLimit
	if (s.pending.msgCount < s.consumeOpts.ThresholdMessages ||
		(s.pending.byteCount < s.consumeOpts.ThresholdBytes && s.consumeOpts.MaxBytes != 0 && !s.consumeOpts.LimitSize)) &&
		s.fetchInProgress.Load() == 0 {

		var batchSize, maxBytes int
		batchSize = s.consumeOpts.MaxMessages - s.pending.msgCount
		if s.consumeOpts.MaxBytes != 0 {
			if s.consumeOpts.LimitSize {
				maxBytes = s.consumeOpts.MaxBytes
			} else {
				maxBytes = s.consumeOpts.MaxBytes - s.pending.byteCount
				// when working with max bytes only, always ask for full batch
				batchSize = s.consumeOpts.MaxMessages
			}
		}
		if s.consumeOpts.StopAfter > 0 {
			batchSize = min(batchSize, s.consumeOpts.StopAfter-s.delivered-s.pending.msgCount)
		}
		if batchSize > 0 {
			pinID := ""
			if s.consumer != nil {
				pinID = s.consumer.getPinID()
			}
			s.fetchNext <- &pullRequest{
				Expires:       s.consumeOpts.Expires,
				Batch:         batchSize,
				MaxBytes:      maxBytes,
				Heartbeat:     s.consumeOpts.Heartbeat,
				PinID:         pinID,
				Group:         s.consumeOpts.Group,
				MinPending:    s.consumeOpts.MinPending,
				MinAckPending: s.consumeOpts.MinAckPending,
				Priority:      s.consumeOpts.Priority,
			}

			s.pending.msgCount = s.consumeOpts.MaxMessages
			s.pending.byteCount = s.consumeOpts.MaxBytes
		}
	}
}

// Messages returns MessagesContext, allowing continuously iterating
// over messages on a stream. Messages cannot be used concurrently
// when using ordered consumer.
//
// See [Consumer.Messages] for more details.
func (p *pullConsumer) Messages(opts ...PullMessagesOpt) (MessagesContext, error) {
	consumeOpts, err := parseMessagesOpts(false, opts...)
	if err != nil {
		return nil, fmt.Errorf("%w: %s", ErrInvalidOption, err)
	}

	if len(p.info.Config.PriorityGroups) != 0 {
		if consumeOpts.Group == "" {
			return nil, fmt.Errorf("%w: %s", ErrInvalidOption, "priority group is required for priority consumer")
		}

		if !slices.Contains(p.info.Config.PriorityGroups, consumeOpts.Group) {
			return nil, fmt.Errorf("%w: %s", ErrInvalidOption, "invalid priority group")
		}
	} else if consumeOpts.Group != "" {
		return nil, fmt.Errorf("%w: %s", ErrInvalidOption, "priority group is not supported for this consumer")
	}

	p.Lock()
	subject := p.js.apiSubject(fmt.Sprintf(apiRequestNextT, p.stream, p.name))

	msgs := make(chan *nats.Msg, consumeOpts.MaxMessages)

	consumeID := nuid.Next()
	sub := &pullSubscription{
		id:          consumeID,
		consumer:    p,
		done:        make(chan struct{}, 1),
		msgs:        msgs,
		errs:        make(chan error, 10),
		fetchNext:   make(chan *pullRequest, 1),
		consumeOpts: consumeOpts,
	}
	sub.connStatusChanged = p.js.conn.StatusChanged(nats.CONNECTED, nats.RECONNECTING)
	inbox := p.js.conn.NewInbox()
	sub.subscription, err = p.js.conn.ChanSubscribe(inbox, sub.msgs)
	if err != nil {
		p.Unlock()
		return nil, err
	}
	sub.subscription.SetClosedHandler(func(sid string) func(string) {
		return func(subject string) {
			if sub.draining.Load() != 1 {
				// if we're not draining, subscription can be closed as soon
				// as closed handler is called
				// otherwise, we need to wait until all messages are drained
				// in Next
				p.subs.Delete(sid)
			}
			sub.closeMsgs()
		}
	}(sub.id))

	p.subs.Store(sub.id, sub)
	p.Unlock()

	go sub.pullMessages(subject)

	go func() {
		for {
			select {
			case status, ok := <-sub.connStatusChanged:
				if !ok {
					return
				}
				if status == nats.CONNECTED {
					sub.errs <- errConnected
				}
				if status == nats.RECONNECTING {
					sub.errs <- errDisconnected
				}
			case <-sub.done:
				return
			}
		}
	}()

	return sub, nil
}

var (
	errConnected    = errors.New("connected")
	errDisconnected = errors.New("disconnected")
)

// Next retrieves next message on a stream. If MessagesContext is closed
// (either stopped or drained), Next will return ErrMsgIteratorClosed
// error. An optional timeout or context can be provided using NextOpt
// options. If none are provided, Next will block indefinitely until a
// message is available, iterator is closed or a heartbeat error occurs.
func (s *pullSubscription) Next(opts ...NextOpt) (Msg, error) {
	var nextOpts nextOpts
	for _, opt := range opts {
		opt.configureNext(&nextOpts)
	}

	if nextOpts.timeout > 0 && nextOpts.ctx != nil {
		return nil, fmt.Errorf("%w: cannot specify both NextMaxWait and NextContext", ErrInvalidOption)
	}

	// Create timeout channel if needed
	var timeoutCh <-chan time.Time
	if nextOpts.timeout > 0 {
		timer := time.NewTimer(nextOpts.timeout)
		defer timer.Stop()
		timeoutCh = timer.C
	}

	// Use context if provided
	var ctxDone <-chan struct{}
	if nextOpts.ctx != nil {
		ctxDone = nextOpts.ctx.Done()
	}

	s.Lock()
	defer s.Unlock()
	drainMode := s.draining.Load() == 1
	closed := s.closed.Load() == 1
	if closed && !drainMode {
		// Check if iterator was closed due to connection closure
		if s.consumer.js.conn.IsClosed() {
			return nil, fmt.Errorf("%w: %w", ErrMsgIteratorClosed, ErrConnectionClosed)
		}
		return nil, ErrMsgIteratorClosed
	}
	hbMonitor := s.scheduleHeartbeatCheck(s.consumeOpts.Heartbeat)
	defer func() {
		if hbMonitor != nil {
			hbMonitor.Stop()
		}
	}()

	isConnected := true
	if s.consumeOpts.StopAfter > 0 && s.delivered >= s.consumeOpts.StopAfter {
		s.Stop()
		return nil, ErrMsgIteratorClosed
	}

	for {
		s.checkPending()
		select {
		case msg, ok := <-s.msgs:
			if !ok {
				// if msgs channel is closed, it means that subscription was either drained or stopped
				s.consumer.subs.Delete(s.id)
				s.draining.CompareAndSwap(1, 0)
				// Check if iterator was closed due to connection closure
				if s.consumer.js.conn.IsClosed() {
					return nil, fmt.Errorf("%w: %w", ErrMsgIteratorClosed, ErrConnectionClosed)
				}
				return nil, ErrMsgIteratorClosed
			}
			if hbMonitor != nil {
				hbMonitor.Reset(2 * s.consumeOpts.Heartbeat)
			}
			userMsg, msgErr := checkMsg(msg)
			if !userMsg {
				// heartbeat message
				if msgErr == nil {
					continue
				}
				if termErr, _ := s.handleStatusMsg(msg, msgErr); termErr != nil {
					s.Stop()
					return nil, termErr
				}
				continue
			}
			if pinId := msg.Header.Get("Nats-Pin-Id"); pinId != "" {
				s.consumer.setPinID(pinId)
			}
			s.decrementPendingMsgs(msg)
			s.incrementDeliveredMsgs()
			return s.consumer.js.toJSMsg(msg), nil
		case err := <-s.errs:
			if errors.Is(err, ErrNoHeartbeat) {
				s.pending.msgCount = 0
				s.pending.byteCount = 0
				if s.consumeOpts.ReportMissingHeartbeats {
					return nil, err
				}
				if hbMonitor != nil {
					hbMonitor.Reset(2 * s.consumeOpts.Heartbeat)
				}
			}
			if errors.Is(err, errConnected) {
				if !isConnected {
					isConnected = true

					if s.consumeOpts.notifyOnReconnect {
						return nil, errConnected
					}
					s.pending.msgCount = 0
					s.pending.byteCount = 0
					if hbMonitor != nil {
						hbMonitor.Reset(2 * s.consumeOpts.Heartbeat)
					}
				}
			}
			if errors.Is(err, errDisconnected) {
				if hbMonitor != nil {
					hbMonitor.Stop()
				}
				isConnected = false
			}
		case <-timeoutCh:
			return nil, nats.ErrTimeout
		case <-ctxDone:
			return nil, nextOpts.ctx.Err()
		}
	}
}

// handleStatusMsg processes a status message from the server.
// It returns a terminal error (caller should stop) and a non-terminal
// error to notify the user about via ErrHandler. The caller should invoke
// ErrHandler outside the lock to avoid deadlocks.
func (s *pullSubscription) handleStatusMsg(msg *nats.Msg, msgErr error) (error, error) {
	if !errors.Is(msgErr, nats.ErrTimeout) && !errors.Is(msgErr, ErrMaxBytesExceeded) && !errors.Is(msgErr, ErrBatchCompleted) {
		if errors.Is(msgErr, ErrConsumerDeleted) || errors.Is(msgErr, ErrBadRequest) {
			return msgErr, nil
		}
		if errors.Is(msgErr, ErrPinIDMismatch) {
			s.consumer.setPinID("")
			s.pending.msgCount = 0
			s.pending.byteCount = 0
		}
		if errors.Is(msgErr, ErrConsumerLeadershipChanged) {
			s.pending.msgCount = 0
			s.pending.byteCount = 0
		}
		return nil, msgErr
	}
	msgsLeft, bytesLeft, err := parsePending(msg)
	if err != nil {
		return err, nil
	}
	s.pending.msgCount -= msgsLeft
	if s.pending.msgCount < 0 {
		s.pending.msgCount = 0
	}
	if s.consumeOpts.MaxBytes > 0 && !s.consumeOpts.LimitSize {
		s.pending.byteCount -= bytesLeft
		if s.pending.byteCount < 0 {
			s.pending.byteCount = 0
		}
	}
	return nil, nil
}

func (hb *hbMonitor) Stop() {
	hb.Lock()
	hb.timer.Stop()
	hb.Unlock()
}

func (hb *hbMonitor) Reset(dur time.Duration) {
	hb.Lock()
	hb.timer.Reset(dur)
	hb.Unlock()
}

// Stop unsubscribes from the stream and cancels subscription. Calling
// Next after calling Stop will return ErrMsgIteratorClosed error.
// All messages that are already in the buffer are discarded.
func (s *pullSubscription) Stop() {
	if !s.closed.CompareAndSwap(0, 1) {
		return
	}
	close(s.done)
	if s.consumeOpts.stopAfterMsgsLeft != nil {
		if s.delivered >= s.consumeOpts.StopAfter {
			close(s.consumeOpts.stopAfterMsgsLeft)
		} else {
			s.consumeOpts.stopAfterMsgsLeft <- s.consumeOpts.StopAfter - s.delivered
		}
	}
}

// Drain unsubscribes from the stream and cancels subscription. All
// messages that are already in the buffer will be available on
// subsequent calls to Next. After the buffer is drained, Next will
// return ErrMsgIteratorClosed error.
func (s *pullSubscription) Drain() {
	if !s.closed.CompareAndSwap(0, 1) {
		return
	}
	s.draining.Store(1)
	close(s.done)
	if s.consumeOpts.stopAfterMsgsLeft != nil {
		if s.delivered >= s.consumeOpts.StopAfter {
			close(s.consumeOpts.stopAfterMsgsLeft)
		} else {
			s.consumeOpts.stopAfterMsgsLeft <- s.consumeOpts.StopAfter - s.delivered
		}
	}
}

// Closed returns a channel that is closed when consuming is
// fully stopped/drained. When the channel is closed, no more messages
// will be received and processing is complete.
func (s *pullSubscription) Closed() <-chan struct{} {
	s.Lock()
	defer s.Unlock()
	closedCh := s.closedCh
	if closedCh == nil {
		closedCh = make(chan struct{})
		s.closedCh = closedCh
	}
	if !s.subscription.IsValid() {
		close(s.closedCh)
		s.closedCh = nil
	}
	return closedCh
}

// Fetch sends a single request to retrieve given number of messages.
// It will wait up to provided expiry time if not all messages are available.
func (p *pullConsumer) Fetch(batch int, opts ...FetchOpt) (MessageBatch, error) {
	req := &pullRequest{
		Batch:     batch,
		Expires:   DefaultExpires,
		Heartbeat: unset,
	}
	for _, opt := range opts {
		if err := opt(req); err != nil {
			return nil, err
		}
	}

	if req.ctx != nil && req.maxWaitSet {
		return nil, fmt.Errorf("%w: cannot specify both FetchContext and FetchMaxWait", ErrInvalidOption)
	}

	// if heartbeat was not explicitly set, set it to 5 seconds for longer pulls
	// and disable it for shorter pulls
	if req.Heartbeat == unset {
		if req.Expires >= 10*time.Second {
			req.Heartbeat = 5 * time.Second
		} else {
			req.Heartbeat = 0
		}
	}
	if req.Expires < 2*req.Heartbeat {
		return nil, fmt.Errorf("%w: expiry time should be at least 2 times the heartbeat", ErrInvalidOption)
	}

	return p.fetch(req)
}

// FetchBytes is used to retrieve up to a provided bytes from the stream.
func (p *pullConsumer) FetchBytes(maxBytes int, opts ...FetchOpt) (MessageBatch, error) {
	req := &pullRequest{
		Batch:     defaultBatchMaxBytesOnly,
		MaxBytes:  maxBytes,
		Expires:   DefaultExpires,
		Heartbeat: unset,
	}
	for _, opt := range opts {
		if err := opt(req); err != nil {
			return nil, err
		}
	}

	if req.ctx != nil && req.maxWaitSet {
		return nil, fmt.Errorf("%w: cannot specify both FetchContext and FetchMaxWait", ErrInvalidOption)
	}

	// if heartbeat was not explicitly set, set it to 5 seconds for longer pulls
	// and disable it for shorter pulls
	if req.Heartbeat == unset {
		if req.Expires >= 10*time.Second {
			req.Heartbeat = 5 * time.Second
		} else {
			req.Heartbeat = 0
		}
	}
	if req.Expires < 2*req.Heartbeat {
		return nil, fmt.Errorf("%w: expiry time should be at least 2 times the heartbeat", ErrInvalidOption)
	}

	return p.fetch(req)
}

// FetchNoWait sends a single request to retrieve given number of messages.
// FetchNoWait will only return messages that are available at the time of the
// request. It will not wait for more messages to arrive.
func (p *pullConsumer) FetchNoWait(batch int) (MessageBatch, error) {
	req := &pullRequest{
		Batch:  batch,
		NoWait: true,
	}

	return p.fetch(req)
}

func (p *pullConsumer) fetch(req *pullRequest) (MessageBatch, error) {
	res := &fetchResult{
		msgs: make(chan Msg, req.Batch),
	}
	msgs := make(chan *nats.Msg, 2*req.Batch)
	subject := p.js.apiSubject(fmt.Sprintf(apiRequestNextT, p.stream, p.name))

	sub := &pullSubscription{
		consumer: p,
		done:     make(chan struct{}, 1),
		msgs:     msgs,
		errs:     make(chan error, 10),
	}
	inbox := p.js.conn.NewInbox()
	var err error
	sub.subscription, err = p.js.conn.ChanSubscribe(inbox, sub.msgs)
	if err != nil {
		return nil, err
	}
	req.PinID = p.getPinID()
	if err := sub.pull(req, subject); err != nil {
		return nil, err
	}

	var receivedMsgs, receivedBytes int
	hbTimer := sub.scheduleHeartbeatCheck(req.Heartbeat)

	// Use context if provided
	var ctxDone <-chan struct{}
	if req.ctx != nil {
		ctxDone = req.ctx.Done()
	}

	go func(res *fetchResult) {
		defer sub.subscription.Unsubscribe()
		defer close(res.msgs)
		for {
			select {
			case msg := <-msgs:
				res.Lock()
				if hbTimer != nil {
					hbTimer.Reset(2 * req.Heartbeat)
				}
				userMsg, err := checkMsg(msg)
				if err != nil {
					errNotTimeoutOrNoMsgs := !errors.Is(err, nats.ErrTimeout) && !errors.Is(err, ErrNoMessages)
					if errNotTimeoutOrNoMsgs && !errors.Is(err, ErrMaxBytesExceeded) {
						res.err = err
					}
					if errors.Is(err, ErrPinIDMismatch) {
						p.setPinID("")
					}
					res.done = true
					res.Unlock()
					return
				}
				if !userMsg {
					res.Unlock()
					continue
				}
				if pinId := msg.Header.Get("Nats-Pin-Id"); pinId != "" {
					p.setPinID(pinId)
				}
				res.msgs <- p.js.toJSMsg(msg)
				meta, err := msg.Metadata()
				if err != nil {
					res.err = fmt.Errorf("parsing message metadata: %s", err)
					res.done = true
					res.Unlock()
					return
				}
				res.sseq = meta.Sequence.Stream
				receivedMsgs++
				if req.MaxBytes != 0 {
					receivedBytes += msg.Size()
				}
				if receivedMsgs == req.Batch || (req.MaxBytes != 0 && receivedBytes >= req.MaxBytes) {
					res.done = true
					res.Unlock()
					return
				}
				res.Unlock()
			case err := <-sub.errs:
				res.Lock()
				res.err = err
				res.done = true
				res.Unlock()
				return
			case <-time.After(req.Expires + 1*time.Second):
				res.Lock()
				res.done = true
				res.Unlock()
				return
			case <-ctxDone:
				res.Lock()
				res.err = req.ctx.Err()
				res.done = true
				res.Unlock()
				return
			}
		}
	}(res)
	return res, nil
}

func (fr *fetchResult) Messages() <-chan Msg {
	fr.Lock()
	defer fr.Unlock()
	return fr.msgs
}

func (fr *fetchResult) Error() error {
	fr.Lock()
	defer fr.Unlock()
	return fr.err
}

func (fr *fetchResult) closed() bool {
	fr.Lock()
	defer fr.Unlock()
	return fr.done
}

// Next is used to retrieve the next message from the stream. This
// method will block until the message is retrieved or timeout is
// reached.
func (p *pullConsumer) Next(opts ...FetchOpt) (Msg, error) {
	res, err := p.Fetch(1, opts...)
	if err != nil {
		return nil, err
	}
	msg := <-res.Messages()
	if msg != nil {
		return msg, nil
	}
	if res.Error() == nil {
		return nil, nats.ErrTimeout
	}
	return nil, res.Error()
}

func (s *pullSubscription) pullMessages(subject string) {
	for {
		select {
		case req := <-s.fetchNext:
			s.fetchInProgress.Store(1)

			if err := s.pull(req, subject); err != nil {
				if errors.Is(err, ErrMsgIteratorClosed) {
					s.cleanup()
					return
				}
				s.errs <- err
			}
			s.fetchInProgress.Store(0)
		case <-s.done:
			s.cleanup()
			return
		}
	}
}

func (s *pullSubscription) closeMsgs() {
	if s.msgsClosed.CompareAndSwap(0, 1) {
		close(s.msgs)
	}
}

func (s *pullSubscription) scheduleHeartbeatCheck(dur time.Duration) *hbMonitor {
	if dur == 0 {
		return nil
	}
	return &hbMonitor{
		timer: time.AfterFunc(2*dur, func() {
			s.errs <- ErrNoHeartbeat
		}),
	}
}

func (s *pullSubscription) cleanup() {
	// For now this function does not need to hold the lock.
	// Holding the lock here might cause a deadlock if Next()
	// is already holding the lock and waiting.
	// The fields that are read (subscription, hbMonitor)
	// are read only (Only written on creation of pullSubscription).
	if s.subscription == nil || !s.subscription.IsValid() {
		return
	}
	if s.consumer != nil {
		nc := s.consumer.js.conn
		nc.RemoveStatusListener(s.connStatusChanged)
	}
	if s.hbMonitor != nil {
		s.hbMonitor.Stop()
	}
	drainMode := s.draining.Load() == 1
	if drainMode {
		s.subscription.Drain()
	} else {
		s.subscription.Unsubscribe()
	}
	s.closed.Store(1)
}

// pull sends a pull request to the server and waits for messages using a subscription from [pullSubscription].
// Messages will be fetched up to given batch_size or until there are no more messages or timeout is returned
func (s *pullSubscription) pull(req *pullRequest, subject string) error {
	s.consumer.Lock()
	defer s.consumer.Unlock()
	if s.closed.Load() == 1 {
		return ErrMsgIteratorClosed
	}
	if req.Batch < 1 {
		return fmt.Errorf("%w: batch size must be at least 1", nats.ErrInvalidArg)
	}
	reqJSON, err := json.Marshal(req)
	if err != nil {
		return err
	}
	reply := s.subscription.Subject

	if err := s.consumer.js.conn.PublishRequest(subject, reply, reqJSON); err != nil {
		return err
	}
	return nil
}

func parseConsumeOpts(ordered bool, opts ...PullConsumeOpt) (*consumeOpts, error) {
	consumeOpts := &consumeOpts{
		MaxMessages:             unset,
		MaxBytes:                unset,
		Expires:                 DefaultExpires,
		Heartbeat:               unset,
		ReportMissingHeartbeats: true,
		StopAfter:               unset,
	}
	for _, opt := range opts {
		if err := opt.configureConsume(consumeOpts); err != nil {
			return nil, err
		}
	}
	if err := consumeOpts.setDefaults(ordered); err != nil {
		return nil, err
	}
	return consumeOpts, nil
}

func parseMessagesOpts(ordered bool, opts ...PullMessagesOpt) (*consumeOpts, error) {
	consumeOpts := &consumeOpts{
		MaxMessages:             unset,
		MaxBytes:                unset,
		Expires:                 DefaultExpires,
		Heartbeat:               unset,
		ReportMissingHeartbeats: true,
		StopAfter:               unset,
	}
	for _, opt := range opts {
		if err := opt.configureMessages(consumeOpts); err != nil {
			return nil, err
		}
	}
	if err := consumeOpts.setDefaults(ordered); err != nil {
		return nil, err
	}
	return consumeOpts, nil
}

func (consumeOpts *consumeOpts) setDefaults(ordered bool) error {
	// we cannot use both max messages and max bytes unless we're using max bytes as fetch size limiter
	if consumeOpts.MaxBytes != unset && consumeOpts.MaxMessages != unset && !consumeOpts.LimitSize {
		return errors.New("only one of MaxMessages and MaxBytes can be specified")
	}
	if consumeOpts.MaxBytes != unset && !consumeOpts.LimitSize {
		// we used PullMaxBytes setting, set MaxMessages to a high value
		consumeOpts.MaxMessages = defaultBatchMaxBytesOnly
	} else if consumeOpts.MaxMessages == unset {
		// otherwise, if max messages is not set, set it to default value
		consumeOpts.MaxMessages = DefaultMaxMessages
	}
	// if user did not set max bytes, set it to 0
	if consumeOpts.MaxBytes == unset {
		consumeOpts.MaxBytes = 0
	}

	if consumeOpts.ThresholdMessages == 0 {
		// half of the max messages, rounded up
		consumeOpts.ThresholdMessages = int(math.Ceil(float64(consumeOpts.MaxMessages) / 2))
	}
	if consumeOpts.ThresholdBytes == 0 {
		// half of the max bytes, rounded up
		consumeOpts.ThresholdBytes = int(math.Ceil(float64(consumeOpts.MaxBytes) / 2))
	}

	// set default heartbeats
	if consumeOpts.Heartbeat == unset {
		// by default, use 50% of expiry time
		consumeOpts.Heartbeat = consumeOpts.Expires / 2
		if ordered {
			// for ordered consumers, the default heartbeat is 5 seconds
			if consumeOpts.Expires < 10*time.Second {
				consumeOpts.Heartbeat = consumeOpts.Expires / 2
			} else {
				consumeOpts.Heartbeat = 5 * time.Second
			}
		} else if consumeOpts.Heartbeat > 30*time.Second {
			// cap the heartbeat to 30 seconds
			consumeOpts.Heartbeat = 30 * time.Second
		}
	}
	if consumeOpts.Heartbeat > consumeOpts.Expires/2 {
		return fmt.Errorf("%w: the value of Heartbeat must be less than 50%% of expiry", ErrInvalidOption)
	}
	return nil
}

func (c *pullConsumer) getPinID() string {
	c.Lock()
	defer c.Unlock()
	return c.pinID
}

func (c *pullConsumer) setPinID(pinID string) {
	c.Lock()
	defer c.Unlock()
	c.pinID = pinID
}
