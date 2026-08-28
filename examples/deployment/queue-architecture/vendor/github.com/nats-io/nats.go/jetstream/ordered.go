// Copyright 2022-2024 The NATS Authors
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
	"errors"
	"fmt"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/nats-io/nats.go"
)

type (
	orderedConsumer struct {
		js                *jetStream
		cfg               *OrderedConsumerConfig
		stream            string
		currentConsumer   *pullConsumer
		currentSub        *pullSubscription
		cursor            cursor
		namePrefix        string
		serial            int
		consumerType      consumerType
		doReset           chan struct{}
		resetInProgress   atomic.Uint32
		userErrHandler    ConsumeErrHandler
		stopAfter         int
		stopAfterMsgsLeft chan int
		withStopAfter     bool
		runningFetch      *fetchResult
		subscription      *orderedSubscription
		sync.Mutex
	}

	orderedSubscription struct {
		consumer *orderedConsumer
		opts     []PullMessagesOpt
		done     chan struct{}
		closed   atomic.Uint32
	}

	cursor struct {
		streamSeq  uint64
		deliverSeq uint64
	}

	consumerType int
)

const (
	consumerTypeNotSet consumerType = iota
	consumerTypeConsume
	consumerTypeFetch
)

var (
	errOrderedSequenceMismatch = errors.New("sequence mismatch")
	errOrderedConsumerClosed   = errors.New("ordered consumer closed")
)

// Consume can be used to continuously receive messages and handle them
// with the provided callback function. Consume cannot be used concurrently
// when using ordered consumer.
//
// See [Consumer.Consume] for more details.
func (c *orderedConsumer) Consume(handler MessageHandler, opts ...PullConsumeOpt) (ConsumeContext, error) {
	c.Lock()
	defer c.Unlock()
	if (c.consumerType == consumerTypeNotSet || c.consumerType == consumerTypeConsume) && c.currentConsumer == nil {
		err := c.reset()
		if err != nil {
			return nil, err
		}
	} else if c.consumerType == consumerTypeConsume && c.currentConsumer != nil {
		return nil, ErrOrderedConsumerConcurrentRequests
	}
	if c.consumerType == consumerTypeFetch {
		return nil, ErrOrderConsumerUsedAsFetch
	}
	c.consumerType = consumerTypeConsume
	consumeOpts, err := parseConsumeOpts(true, opts...)
	if err != nil {
		return nil, fmt.Errorf("%w: %s", ErrInvalidOption, err)
	}
	c.userErrHandler = consumeOpts.ErrHandler
	opts = append(opts, consumeReconnectNotify(),
		ConsumeErrHandler(c.errHandler(c.serial)))
	if consumeOpts.StopAfter > 0 {
		c.withStopAfter = true
		c.stopAfter = consumeOpts.StopAfter
	}
	c.stopAfterMsgsLeft = make(chan int, 1)
	if c.stopAfter > 0 {
		opts = append(opts, consumeStopAfterNotify(c.stopAfter, c.stopAfterMsgsLeft))
	}
	sub := &orderedSubscription{
		consumer: c,
		done:     make(chan struct{}, 1),
	}
	c.subscription = sub
	internalHandler := func(serial int) func(msg Msg) {
		return func(msg Msg) {
			c.Lock()
			// handler is a noop if message was delivered for a consumer with different serial
			if serial != c.serial {
				c.Unlock()
				return
			}
			meta, err := msg.Metadata()
			if err != nil {
				currentSub := c.currentSub
				c.Unlock()
				c.errHandler(serial)(currentSub, err)
				return
			}
			dseq := meta.Sequence.Consumer
			if dseq != c.cursor.deliverSeq+1 {
				c.Unlock()
				c.errHandler(serial)(sub, errOrderedSequenceMismatch)
				return
			}
			c.cursor.deliverSeq = dseq
			c.cursor.streamSeq = meta.Sequence.Stream
			c.Unlock()
			handler(msg)
		}
	}

	cc, err := c.currentConsumer.Consume(internalHandler(c.serial), opts...)
	if err != nil {
		return nil, err
	}
	c.currentSub = cc.(*pullSubscription)

	go func() {
		for {
			select {
			case <-c.doReset:
				if err := c.reset(); err != nil {
					if errors.Is(err, errOrderedConsumerClosed) {
						continue
					}
					c.errHandler(c.serial)(c.currentSub, err)
				}
				if c.withStopAfter {
					select {
					case c.stopAfter = <-c.stopAfterMsgsLeft:
					default:
					}
					if c.stopAfter <= 0 {
						sub.Stop()
						return
					}
				}
				if c.stopAfter > 0 {
					opts = opts[:len(opts)-2]
				} else {
					opts = opts[:len(opts)-1]
				}

				// overwrite the previous err handler to use the new serial
				opts = append(opts, ConsumeErrHandler(c.errHandler(c.serial)))
				if c.withStopAfter {
					opts = append(opts, consumeStopAfterNotify(c.stopAfter, c.stopAfterMsgsLeft))
				}
				if cc, err := c.currentConsumer.Consume(internalHandler(c.serial), opts...); err != nil {
					c.errHandler(c.serial)(cc, err)
				} else {
					c.Lock()
					c.currentSub = cc.(*pullSubscription)
					c.Unlock()
				}
			case <-sub.done:
				s := sub.consumer.currentSub
				if s != nil {
					sub.consumer.Lock()
					s.Stop()
					sub.consumer.Unlock()
				}
				return
			case msgsLeft, ok := <-c.stopAfterMsgsLeft:
				if !ok {
					close(sub.done)
				}
				c.stopAfter = msgsLeft
				return
			}
		}
	}()
	return sub, nil
}

func (c *orderedConsumer) errHandler(serial int) func(cc ConsumeContext, err error) {
	return func(cc ConsumeContext, err error) {
		c.Lock()

		if c.userErrHandler != nil && !errors.Is(err, errOrderedSequenceMismatch) && !errors.Is(err, errConnected) {
			c.userErrHandler(cc, err)
		}
		if errors.Is(err, ErrConnectionClosed) {
			if c.subscription != nil {
				c.Unlock()
				c.subscription.Stop()
				return
			}
			c.Unlock()
			return
		}

		if errors.Is(err, ErrNoHeartbeat) ||
			errors.Is(err, errOrderedSequenceMismatch) ||
			errors.Is(err, ErrConsumerDeleted) ||
			errors.Is(err, errConnected) ||
			errors.Is(err, nats.ErrNoResponders) {
			// only reset if serial matches the current consumer serial and there is no reset in progress
			if serial == c.serial && c.resetInProgress.Load() == 0 {
				c.resetInProgress.Store(1)
				c.doReset <- struct{}{}
			}
		}
		c.Unlock()
	}
}

// Messages returns MessagesContext, allowing continuously iterating
// over messages on a stream. Messages cannot be used concurrently
// when using ordered consumer.
//
// See [Consumer.Messages] for more details.
func (c *orderedConsumer) Messages(opts ...PullMessagesOpt) (MessagesContext, error) {
	if (c.consumerType == consumerTypeNotSet || c.consumerType == consumerTypeConsume) && c.currentConsumer == nil {
		err := c.reset()
		if err != nil {
			return nil, err
		}
	} else if c.consumerType == consumerTypeConsume && c.currentConsumer != nil {
		return nil, ErrOrderedConsumerConcurrentRequests
	}
	if c.consumerType == consumerTypeFetch {
		return nil, ErrOrderConsumerUsedAsFetch
	}
	c.consumerType = consumerTypeConsume
	consumeOpts, err := parseMessagesOpts(true, opts...)
	if err != nil {
		return nil, fmt.Errorf("%w: %s", ErrInvalidOption, err)
	}
	opts = append(opts,
		WithMessagesErrOnMissingHeartbeat(true),
		messagesReconnectNotify())
	c.stopAfterMsgsLeft = make(chan int, 1)
	if consumeOpts.StopAfter > 0 {
		c.withStopAfter = true
		c.stopAfter = consumeOpts.StopAfter
	}
	c.userErrHandler = consumeOpts.ErrHandler
	if c.stopAfter > 0 {
		opts = append(opts, messagesStopAfterNotify(c.stopAfter, c.stopAfterMsgsLeft))
	}
	cc, err := c.currentConsumer.Messages(opts...)
	if err != nil {
		return nil, err
	}
	c.currentSub = cc.(*pullSubscription)

	sub := &orderedSubscription{
		consumer: c,
		opts:     opts,
		done:     make(chan struct{}, 1),
	}
	c.subscription = sub

	return sub, nil
}

func (s *orderedSubscription) Next(opts ...NextOpt) (Msg, error) {
	for {
		msg, err := s.consumer.currentSub.Next(opts...)
		if err != nil {
			// Check for errors which should be returned directly
			// without resetting the consumer
			if errors.Is(err, ErrInvalidOption) {
				return nil, err
			}
			if errors.Is(err, nats.ErrTimeout) {
				return nil, err
			}
			if errors.Is(err, context.Canceled) || errors.Is(err, context.DeadlineExceeded) {
				return nil, err
			}
			if errors.Is(err, ErrMsgIteratorClosed) {
				s.Stop()
				return nil, err
			}
			if s.consumer.withStopAfter {
				select {
				case s.consumer.stopAfter = <-s.consumer.stopAfterMsgsLeft:
				default:
				}
				if s.consumer.stopAfter <= 0 {
					s.Stop()
					return nil, ErrMsgIteratorClosed
				}
				s.opts[len(s.opts)-1] = StopAfter(s.consumer.stopAfter)
			}
			if err := s.consumer.reset(); err != nil {
				if errors.Is(err, errOrderedConsumerClosed) {
					return nil, ErrMsgIteratorClosed
				}
				return nil, err
			}
			cc, err := s.consumer.currentConsumer.Messages(s.opts...)
			if err != nil {
				return nil, err
			}
			s.consumer.currentSub = cc.(*pullSubscription)
			continue
		}

		meta, err := msg.Metadata()
		if err != nil {
			return nil, err
		}
		serial := serialNumberFromConsumer(meta.Consumer)
		if serial != s.consumer.serial {
			continue
		}
		dseq := meta.Sequence.Consumer
		if dseq != s.consumer.cursor.deliverSeq+1 {
			if err := s.consumer.reset(); err != nil {
				if errors.Is(err, errOrderedConsumerClosed) {
					return nil, ErrMsgIteratorClosed
				}
				return nil, err
			}
			cc, err := s.consumer.currentConsumer.Messages(s.opts...)
			if err != nil {
				return nil, err
			}
			s.consumer.currentSub = cc.(*pullSubscription)
			continue
		}
		s.consumer.cursor.deliverSeq = dseq
		s.consumer.cursor.streamSeq = meta.Sequence.Stream
		return msg, nil
	}
}

func (s *orderedSubscription) Stop() {
	if !s.closed.CompareAndSwap(0, 1) {
		return
	}
	s.consumer.Lock()
	defer s.consumer.Unlock()
	if s.consumer.currentSub != nil {
		s.consumer.currentSub.Stop()
	}
	close(s.done)
}

func (s *orderedSubscription) Drain() {
	if !s.closed.CompareAndSwap(0, 1) {
		return
	}
	s.consumer.Lock()
	defer s.consumer.Unlock()
	if s.consumer.currentSub != nil {
		s.consumer.currentSub.Drain()
	}
	close(s.done)
}

// Closed returns a channel that is closed when the consuming is
// fully stopped/drained. When the channel is closed, no more messages
// will be received and processing is complete.
func (s *orderedSubscription) Closed() <-chan struct{} {
	closedCh := make(chan struct{})

	go func() {
		// First wait for s.done to be closed
		<-s.done

		// Then ensure underlying consumer is also closed (it may still be draining)
		s.consumer.Lock()
		if s.consumer.currentSub != nil {
			closed := s.consumer.currentSub.Closed()
			s.consumer.Unlock()
			<-closed
		} else {
			s.consumer.Unlock()
		}

		close(closedCh)
	}()
	return closedCh
}

// Fetch is used to retrieve up to a provided number of messages from a
// stream. This method will always send a single request and wait until
// either all messages are retrieved or request times out.
//
// It is not efficient to use Fetch with on an ordered consumer, as it will
// reset the consumer for each subsequent Fetch call.
// Consider using [Consumer.Consume] or [Consumer.Messages] instead.
func (c *orderedConsumer) Fetch(batch int, opts ...FetchOpt) (MessageBatch, error) {
	c.Lock()
	if c.consumerType == consumerTypeConsume {
		c.Unlock()
		return nil, ErrOrderConsumerUsedAsConsume
	}
	if c.runningFetch != nil {
		if !c.runningFetch.closed() {
			return nil, ErrOrderedConsumerConcurrentRequests
		}
		if c.runningFetch.sseq != 0 {
			c.cursor.streamSeq = c.runningFetch.sseq
		}
	}
	c.consumerType = consumerTypeFetch
	sub := orderedSubscription{
		consumer: c,
		done:     make(chan struct{}),
	}
	c.subscription = &sub
	c.Unlock()
	err := c.reset()
	if err != nil {
		return nil, err
	}
	msgs, err := c.currentConsumer.Fetch(batch, opts...)
	if err != nil {
		return nil, err
	}
	c.runningFetch = msgs.(*fetchResult)
	return msgs, nil
}

// FetchBytes is used to retrieve up to a provided bytes from the
// stream. This method will always send a single request and wait until
// provided number of bytes is exceeded or request times out.
//
// It is not efficient to use FetchBytes with on an ordered consumer, as it will
// reset the consumer for each subsequent Fetch call.
// Consider using [Consumer.Consume] or [Consumer.Messages] instead.
func (c *orderedConsumer) FetchBytes(maxBytes int, opts ...FetchOpt) (MessageBatch, error) {
	c.Lock()
	if c.consumerType == consumerTypeConsume {
		c.Unlock()
		return nil, ErrOrderConsumerUsedAsConsume
	}
	if c.runningFetch != nil {
		if !c.runningFetch.closed() {
			return nil, ErrOrderedConsumerConcurrentRequests
		}
		if c.runningFetch.sseq != 0 {
			c.cursor.streamSeq = c.runningFetch.sseq
		}
	}
	c.consumerType = consumerTypeFetch
	sub := orderedSubscription{
		consumer: c,
		done:     make(chan struct{}),
	}
	c.subscription = &sub
	c.Unlock()
	err := c.reset()
	if err != nil {
		return nil, err
	}
	msgs, err := c.currentConsumer.FetchBytes(maxBytes, opts...)
	if err != nil {
		return nil, err
	}
	c.runningFetch = msgs.(*fetchResult)
	return msgs, nil
}

// FetchNoWait is used to retrieve up to a provided number of messages
// from a stream. This method will always send a single request and
// immediately return up to a provided number of messages or wait until
// at least one message is available or request times out.
//
// It is not efficient to use FetchNoWait with on an ordered consumer, as it will
// reset the consumer for each subsequent Fetch call.
// Consider using [Consumer.Consume] or [Consumer.Messages] instead.
func (c *orderedConsumer) FetchNoWait(batch int) (MessageBatch, error) {
	if c.consumerType == consumerTypeConsume {
		return nil, ErrOrderConsumerUsedAsConsume
	}
	if c.runningFetch != nil && !c.runningFetch.done {
		return nil, ErrOrderedConsumerConcurrentRequests
	}
	c.consumerType = consumerTypeFetch
	sub := orderedSubscription{
		consumer: c,
		done:     make(chan struct{}),
	}
	c.subscription = &sub
	err := c.reset()
	if err != nil {
		return nil, err
	}
	return c.currentConsumer.FetchNoWait(batch)
}

// Next is used to retrieve the next message from the stream. This
// method will block until the message is retrieved or timeout is
// reached.
//
// It is not efficient to use Next with on an ordered consumer, as it will
// reset the consumer for each subsequent Fetch call.
// Consider using [Consumer.Consume] or [Consumer.Messages] instead.
func (c *orderedConsumer) Next(opts ...FetchOpt) (Msg, error) {
	res, err := c.Fetch(1, opts...)
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

func serialNumberFromConsumer(name string) int {
	if len(name) == 0 {
		return 0
	}
	parts := strings.Split(name, "_")
	if len(parts) < 2 {
		return 0
	}
	serial, err := strconv.Atoi(parts[len(parts)-1])
	if err != nil {
		return 0
	}
	return serial
}

func (c *orderedConsumer) reset() error {
	c.Lock()
	defer c.Unlock()
	defer c.resetInProgress.Store(0)
	if c.currentConsumer != nil {
		c.currentConsumer.Lock()
		if c.currentSub != nil {
			c.currentSub.Stop()
		}
		consName := c.currentConsumer.CachedInfo().Name
		c.currentConsumer.Unlock()
		go func() {
			ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
			_ = c.js.DeleteConsumer(ctx, c.stream, consName)
			cancel()
		}()
	}

	c.cursor.deliverSeq = 0
	consumerConfig := c.getConsumerConfig()

	var err error
	var cons Consumer

	backoffOpts := backoffOpts{
		attempts:        c.cfg.MaxResetAttempts,
		initialInterval: time.Second,
		factor:          2,
		maxInterval:     10 * time.Second,
		cancel:          c.subscription.done,
	}
	err = retryWithBackoff(func(attempt int) (bool, error) {
		isClosed := c.subscription.closed.Load() == 1
		if isClosed {
			return false, errOrderedConsumerClosed
		}
		ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
		defer cancel()
		cons, err = c.js.CreateOrUpdateConsumer(ctx, c.stream, *consumerConfig)
		if err != nil {
			return true, err
		}
		return false, nil
	}, backoffOpts)
	if err != nil {
		return err
	}
	c.currentConsumer = cons.(*pullConsumer)
	return nil
}

func (c *orderedConsumer) getConsumerConfig() *ConsumerConfig {
	c.serial++
	var nextSeq uint64

	// if stream sequence is not initialized, no message was consumed yet
	// therefore, start from the beginning (either from 1 or from the provided sequence)
	if c.cursor.streamSeq == 0 {
		if c.cfg.OptStartSeq != 0 {
			nextSeq = c.cfg.OptStartSeq
		} else {
			nextSeq = 1
		}
	} else {
		// otherwise, start from the next sequence
		nextSeq = c.cursor.streamSeq + 1
	}

	if c.cfg.MaxResetAttempts == 0 {
		c.cfg.MaxResetAttempts = -1
	}
	name := fmt.Sprintf("%s_%d", c.namePrefix, c.serial)
	cfg := &ConsumerConfig{
		Name:              name,
		DeliverPolicy:     DeliverByStartSequencePolicy,
		OptStartSeq:       nextSeq,
		AckPolicy:         AckNonePolicy,
		InactiveThreshold: 5 * time.Minute,
		Replicas:          1,
		HeadersOnly:       c.cfg.HeadersOnly,
		MemoryStorage:     true,
		Metadata:          c.cfg.Metadata,
	}
	if len(c.cfg.FilterSubjects) == 1 {
		cfg.FilterSubject = c.cfg.FilterSubjects[0]
	} else {
		cfg.FilterSubjects = c.cfg.FilterSubjects
	}
	if c.cfg.InactiveThreshold != 0 {
		cfg.InactiveThreshold = c.cfg.InactiveThreshold
	}

	// if the cursor is not yet set, use the provided deliver policy
	if c.cursor.streamSeq != 0 {
		return cfg
	}

	// initial request, some options may be modified at that point
	cfg.DeliverPolicy = c.cfg.DeliverPolicy
	if c.cfg.DeliverPolicy == DeliverLastPerSubjectPolicy ||
		c.cfg.DeliverPolicy == DeliverLastPolicy ||
		c.cfg.DeliverPolicy == DeliverNewPolicy ||
		c.cfg.DeliverPolicy == DeliverAllPolicy {

		cfg.OptStartSeq = 0
	} else if c.cfg.DeliverPolicy == DeliverByStartTimePolicy {
		cfg.OptStartSeq = 0
		cfg.OptStartTime = c.cfg.OptStartTime
	} else {
		cfg.OptStartSeq = c.cfg.OptStartSeq
	}

	if cfg.DeliverPolicy == DeliverLastPerSubjectPolicy && len(c.cfg.FilterSubjects) == 0 {
		cfg.FilterSubjects = []string{">"}
	}

	return cfg
}

func consumeStopAfterNotify(numMsgs int, msgsLeftAfterStop chan int) PullConsumeOpt {
	return pullOptFunc(func(opts *consumeOpts) error {
		opts.StopAfter = numMsgs
		opts.stopAfterMsgsLeft = msgsLeftAfterStop
		return nil
	})
}

func messagesStopAfterNotify(numMsgs int, msgsLeftAfterStop chan int) PullMessagesOpt {
	return pullOptFunc(func(opts *consumeOpts) error {
		opts.StopAfter = numMsgs
		opts.stopAfterMsgsLeft = msgsLeftAfterStop
		return nil
	})
}

func consumeReconnectNotify() PullConsumeOpt {
	return pullOptFunc(func(opts *consumeOpts) error {
		opts.notifyOnReconnect = true
		return nil
	})
}

func messagesReconnectNotify() PullMessagesOpt {
	return pullOptFunc(func(opts *consumeOpts) error {
		opts.notifyOnReconnect = true
		return nil
	})
}

// Info returns information about the ordered consumer.
// Note that this method will fetch the latest instance of the
// consumer from the server, which can be deleted by the library at any time.
func (c *orderedConsumer) Info(ctx context.Context) (*ConsumerInfo, error) {
	c.Lock()
	defer c.Unlock()
	if c.currentConsumer == nil {
		return nil, ErrOrderedConsumerNotCreated
	}
	infoSubject := fmt.Sprintf(apiConsumerInfoT, c.stream, c.currentConsumer.name)
	var resp consumerInfoResponse

	if _, err := c.js.apiRequestJSON(ctx, infoSubject, &resp); err != nil {
		return nil, err
	}
	if resp.Error != nil {
		if resp.Error.ErrorCode == JSErrCodeConsumerNotFound {
			return nil, ErrConsumerNotFound
		}
		return nil, resp.Error
	}
	if resp.Error == nil && resp.ConsumerInfo == nil {
		return nil, ErrConsumerNotFound
	}

	c.currentConsumer.info = resp.ConsumerInfo
	return resp.ConsumerInfo, nil
}

// CachedInfo returns cached information about the consumer currently
// used by the ordered consumer. Cached info will be updated on every call
// to [Consumer.Info] or on consumer reset.
func (c *orderedConsumer) CachedInfo() *ConsumerInfo {
	c.Lock()
	defer c.Unlock()
	if c.currentConsumer == nil {
		return nil
	}
	return c.currentConsumer.info
}

type backoffOpts struct {
	// total retry attempts
	// -1 for unlimited
	attempts int
	// initial interval after which first retry will be performed
	// defaults to 1s
	initialInterval time.Duration
	// determines whether first function execution should be performed immediately
	disableInitialExecution bool
	// multiplier on each attempt
	// defaults to 2
	factor float64
	// max interval between retries
	// after reaching this value, all subsequent
	// retries will be performed with this interval
	// defaults to 1 minute
	maxInterval time.Duration
	// custom backoff intervals
	// if set, overrides all other options except attempts
	// if attempts are set, then the last interval will be used
	// for all subsequent retries after reaching the limit
	customBackoff []time.Duration
	// cancel channel
	// if set, retry will be canceled when this channel is closed
	cancel <-chan struct{}
}

func retryWithBackoff(f func(int) (bool, error), opts backoffOpts) error {
	var err error
	var shouldContinue bool
	// if custom backoff is set, use it instead of other options
	if len(opts.customBackoff) > 0 {
		if opts.attempts != 0 {
			return errors.New("cannot use custom backoff intervals when attempts are set")
		}
		for i, interval := range opts.customBackoff {
			select {
			case <-opts.cancel:
				return nil
			case <-time.After(interval):
			}
			shouldContinue, err = f(i)
			if !shouldContinue {
				return err
			}
		}
		return err
	}

	// set default options
	if opts.initialInterval == 0 {
		opts.initialInterval = 1 * time.Second
	}
	if opts.factor == 0 {
		opts.factor = 2
	}
	if opts.maxInterval == 0 {
		opts.maxInterval = 1 * time.Minute
	}
	if opts.attempts == 0 {
		return errors.New("retry attempts have to be set when not using custom backoff intervals")
	}
	interval := opts.initialInterval
	for i := 0; ; i++ {
		if i == 0 && opts.disableInitialExecution {
			time.Sleep(interval)
			continue
		}
		shouldContinue, err = f(i)
		if !shouldContinue {
			return err
		}
		if opts.attempts > 0 && i >= opts.attempts-1 {
			break
		}
		select {
		case <-opts.cancel:
			return nil
		case <-time.After(interval):
		}
		interval = time.Duration(float64(interval) * opts.factor)
		if interval >= opts.maxInterval {
			interval = opts.maxInterval
		}
	}
	return err
}
