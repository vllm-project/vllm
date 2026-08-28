package jetstream

import (
	"errors"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/nats-io/nats.go"
	"github.com/nats-io/nuid"
)

type (
	pushConsumer struct {
		sync.Mutex
		js      *jetStream
		stream  string
		name    string
		info    *ConsumerInfo
		started atomic.Bool
	}

	pushSubscription struct {
		sync.Mutex
		id                string
		errs              chan error
		subscription      *nats.Subscription
		connStatusChanged chan nats.Status
		closedCh          chan struct{}
		done              chan struct{}
		closed            atomic.Bool
		consumeOpts       *pushConsumeOpts
		hbMonitor         *hbMonitor
		idleHeartbeat     time.Duration
	}

	pushConsumeOpts struct {
		ErrHandler ConsumeErrHandler
	}

	PushConsumeOpt interface {
		configurePushConsume(*pushConsumeOpts) error
	}
)

func (p *pushConsumer) Consume(handler MessageHandler, opts ...PushConsumeOpt) (ConsumeContext, error) {
	if handler == nil {
		return nil, ErrHandlerRequired
	}
	consumeOpts := &pushConsumeOpts{}
	for _, opt := range opts {
		if err := opt.configurePushConsume(consumeOpts); err != nil {
			return nil, err
		}
	}

	p.Lock()
	defer p.Unlock()

	if p.info == nil {
		return nil, ErrConsumerNotFound
	}

	if p.started.Load() {
		return nil, ErrConsumerAlreadyConsuming
	}

	consumeID := nuid.Next()
	sub := &pushSubscription{
		id:                consumeID,
		errs:              make(chan error, 1),
		done:              make(chan struct{}, 1),
		consumeOpts:       consumeOpts,
		connStatusChanged: p.js.conn.StatusChanged(nats.CONNECTED, nats.RECONNECTING),
		idleHeartbeat:     p.info.Config.IdleHeartbeat,
	}

	sub.hbMonitor = sub.scheduleHeartbeatCheck(sub.idleHeartbeat)
	internalHandler := func(msg *nats.Msg) {
		if sub.hbMonitor != nil {
			sub.hbMonitor.Stop()
		}
		defer func() {
			if sub.hbMonitor != nil {
				sub.hbMonitor.Reset(2 * sub.idleHeartbeat)
			}
		}()
		status, descr := msg.Header.Get("Status"), msg.Header.Get("Description")
		if status == "" {
			jsMsg := p.js.toJSMsg(msg)
			handler(jsMsg)
			return
		}
		sub.Lock()
		if err, terminate := sub.handleStatusMsg(msg, status, descr); err != nil {
			if sub.consumeOpts.ErrHandler != nil {
				sub.consumeOpts.ErrHandler(sub, err)
			}
			if terminate {
				sub.Stop()
			}
		}
		sub.Unlock()
	}

	var err error
	if p.info.Config.DeliverGroup != "" {
		sub.subscription, err = p.js.conn.QueueSubscribe(p.info.Config.DeliverSubject, p.info.Config.DeliverGroup, internalHandler)
	} else {
		sub.subscription, err = p.js.conn.Subscribe(p.info.Config.DeliverSubject, internalHandler)
	}
	if err != nil {
		return nil, err
	}

	sub.subscription.SetClosedHandler(func(sid string) func(string) {
		return func(subject string) {
			p.started.Store(false)
			sub.Lock()
			defer sub.Unlock()
			if sub.closedCh != nil {
				close(sub.closedCh)
				sub.closedCh = nil
			}
		}
	}(sub.id))

	go func() {
		isConnected := true
		for {
			if sub.closed.Load() {
				return
			}
			select {
			case status, ok := <-sub.connStatusChanged:
				if !ok {
					continue
				}
				if status == nats.RECONNECTING {
					if sub.hbMonitor != nil {
						sub.hbMonitor.Stop()
					}
					isConnected = false
				}
				if status == nats.CONNECTED {
					sub.Lock()
					if !isConnected {
						isConnected = true

						if sub.hbMonitor != nil {
							sub.hbMonitor.Reset(2 * sub.idleHeartbeat)
						}
					}
					sub.Unlock()
				}
			case err := <-sub.errs:
				sub.Lock()
				if sub.consumeOpts.ErrHandler != nil {
					sub.consumeOpts.ErrHandler(sub, err)
				}
				if errors.Is(err, ErrNoHeartbeat) {
					if sub.hbMonitor != nil {
						sub.hbMonitor.Reset(2 * sub.idleHeartbeat)
					}
				}
				sub.Unlock()
			case <-sub.done:
				return
			}
		}
	}()

	p.started.Store(true)

	return sub, nil

}

func (s *pushSubscription) handleStatusMsg(msg *nats.Msg, status, description string) (error, bool) {
	switch status {
	case statusControlMsg:
		switch strings.ToLower(description) {
		case idleHeartbeatDescr:
			return nil, false
		case fcRequestDescr:
			if err := msg.Respond(nil); err != nil {
				if s.consumeOpts.ErrHandler != nil {
					s.consumeOpts.ErrHandler(s, err)
				}
			}
			return nil, false
		}
	case statusConflict:
		if description == consumerDeleted {
			return ErrConsumerDeleted, true
		}
		if description == leadershipChange {
			if s.consumeOpts.ErrHandler != nil {
				s.consumeOpts.ErrHandler(s, ErrConsumerLeadershipChanged)
				return ErrConsumerLeadershipChanged, false
			}
		}
	}
	return nil, false
}

// Stop unsubscribes from the stream and cancels subscription.
// No more messages will be received after calling this method.
// All messages that are already in the buffer are discarded.
func (s *pushSubscription) Stop() {
	if !s.closed.CompareAndSwap(false, true) {
		return
	}
	s.Lock()
	defer s.Unlock()
	close(s.done)
	s.subscription.Unsubscribe()
	if s.hbMonitor != nil {
		s.hbMonitor.Stop()
	}
}

// Drain unsubscribes from the stream and cancels subscription.
// All messages that are already in the buffer will be processed in callback function.
func (s *pushSubscription) Drain() {
	if !s.closed.CompareAndSwap(false, true) {
		return
	}
	s.Lock()
	defer s.Unlock()
	close(s.done)
	s.subscription.Drain()
	if s.hbMonitor != nil {
		s.hbMonitor.Stop()
	}
}

// Closed returns a channel that is closed when consuming is
// fully stopped/drained. When the channel is closed, no more messages
// will be received and processing is complete.
func (s *pushSubscription) Closed() <-chan struct{} {
	s.Lock()
	defer s.Unlock()
	ch := s.closedCh
	if ch == nil {
		ch = make(chan struct{})
		s.closedCh = ch
	}
	if !s.subscription.IsValid() {
		close(s.closedCh)
		s.closedCh = nil
	}
	return ch
}

func (s *pushSubscription) scheduleHeartbeatCheck(dur time.Duration) *hbMonitor {
	if dur == 0 {
		return nil
	}
	return &hbMonitor{
		timer: time.AfterFunc(2*dur, func() {
			s.Lock()
			defer s.Unlock()
			s.errs <- ErrNoHeartbeat
		}),
	}
}
