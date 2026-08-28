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
	"crypto/sha256"
	"encoding/json"
	"errors"
	"fmt"
	"math/rand"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/nats-io/nats.go"
	"github.com/nats-io/nuid"
)

type (
	asyncPublisherOpts struct {
		// For async publish error handling.
		aecb MsgErrHandler
		// For async publish ack handling.
		ackcb MsgAckHandler
		// Max async pub ack in flight
		maxpa int
		// ackTimeout is the max time to wait for an ack.
		ackTimeout time.Duration
	}

	// PublishOpt are the options that can be passed to Publish methods.
	PublishOpt func(*pubOpts) error

	pubOpts struct {
		id             string
		lastMsgID      string        // Expected last msgId
		stream         string        // Expected stream name
		lastSeq        *uint64       // Expected last sequence
		lastSubjectSeq *uint64       // Expected last sequence for subject
		lastSubject    string        // Expected subject for last sequence
		ttl            time.Duration // Message TTL
		schedule       string        // Schedule expression
		scheduleTarget string        // Target subject for scheduled messages
		scheduleSource string        // Source subject for sampling
		scheduleTTL    string        // TTL for generated messages
		scheduleTZ     string        // Time zone for cron schedules

		// Publish retries for NoResponders err.
		retryWait     time.Duration // Retry wait between attempts
		retryAttempts int           // Retry attempts

		// stallWait is the max wait of a async pub ack.
		stallWait time.Duration

		// internal option to re-use existing paf in case of retry.
		pafRetry *pubAckFuture
	}

	// PubAckFuture is a future for a PubAck.
	// It can be used to wait for a PubAck or an error after an async publish.
	PubAckFuture interface {
		// Ok returns a receive only channel that can be used to get a PubAck.
		Ok() <-chan *PubAck

		// Err returns a receive only channel that can be used to get the error from an async publish.
		Err() <-chan error

		// Msg returns the message that was sent to the server.
		Msg() *nats.Msg
	}

	pubAckFuture struct {
		jsClient   *jetStreamClient
		msg        *nats.Msg
		retries    int
		maxRetries int
		retryWait  time.Duration
		ack        *PubAck
		err        error
		errCh      chan error
		doneCh     chan *PubAck
		reply      string
		timeout    *time.Timer
	}

	jetStreamClient struct {
		asyncPublishContext
		asyncPublisherOpts
	}

	// MsgErrHandler is used to process asynchronous errors from JetStream
	// PublishAsync. It will return the original message sent to the server for
	// possible retransmitting and the error encountered.
	MsgErrHandler func(JetStream, *nats.Msg, error)

	// MsgAckHandler is used to process asynchronous acks from JetStream
	// PublishAsync. It will return the original message sent to the server
	// and the resulting PubAck.
	//
	// The handler is invoked synchronously on the ack-processing goroutine,
	// so it must not block or call methods on the PubAckFuture returned by
	// the corresponding PublishAsync/PublishMsgAsync call, as doing so may
	// deadlock.
	MsgAckHandler func(JetStream, *nats.Msg, *PubAck)

	asyncPublishContext struct {
		sync.RWMutex
		replyPrefix string
		replySub    *nats.Subscription
		acks        map[string]*pubAckFuture
		stallCh     chan struct{}
		doneCh      chan struct{}
		rr          *rand.Rand
		// channel to signal when server is disconnected or conn is closed
		connStatusCh chan (nats.Status)
	}

	pubAckResponse struct {
		apiResponse
		*PubAck
	}

	// PubAck is an ack received after successfully publishing a message.
	PubAck struct {
		// Stream is the stream name the message was published to.
		Stream string `json:"stream"`

		// Sequence is the stream sequence number of the message.
		Sequence uint64 `json:"seq"`

		// Duplicate indicates whether the message was a duplicate.
		// Duplicate can be detected using the [MsgIDHeader] and [StreamConfig.Duplicates].
		Duplicate bool `json:"duplicate,omitempty"`

		// Domain is the domain the message was published to.
		Domain string `json:"domain,omitempty"`

		Value string `json:"val,omitempty"`
	}
)

const (
	// Default time wait between retries on Publish if err is ErrNoResponders.
	DefaultPubRetryWait = 250 * time.Millisecond

	// Default number of retries
	DefaultPubRetryAttempts = 2
)

const (
	statusHdr = "Status"

	rdigits = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
	base    = 62
)

// Publish performs a synchronous publish to a stream and waits for ack
// from server. It accepts subject name (which must be bound to a stream)
// and message payload.
func (js *jetStream) Publish(ctx context.Context, subj string, data []byte, opts ...PublishOpt) (*PubAck, error) {
	return js.PublishMsg(ctx, &nats.Msg{Subject: subj, Data: data}, opts...)
}

// PublishMsg performs a synchronous publish to a stream and waits for
// ack from server. It accepts subject name (which must be bound to a
// stream) and nats.Message.
func (js *jetStream) PublishMsg(ctx context.Context, m *nats.Msg, opts ...PublishOpt) (*PubAck, error) {
	ctx, cancel := js.wrapContextWithoutDeadline(ctx)
	if cancel != nil {
		defer cancel()
	}
	o := pubOpts{
		retryWait:     DefaultPubRetryWait,
		retryAttempts: DefaultPubRetryAttempts,
	}
	if len(opts) > 0 {
		if m.Header == nil {
			m.Header = nats.Header{}
		}
		for _, opt := range opts {
			if err := opt(&o); err != nil {
				return nil, err
			}
		}
	}
	if o.stallWait > 0 {
		return nil, fmt.Errorf("%w: stall wait cannot be set to sync publish", ErrInvalidOption)
	}

	if o.id != "" {
		m.Header.Set(MsgIDHeader, o.id)
	}
	if o.lastMsgID != "" {
		m.Header.Set(ExpectedLastMsgIDHeader, o.lastMsgID)
	}
	if o.stream != "" {
		m.Header.Set(ExpectedStreamHeader, o.stream)
	}
	if o.lastSeq != nil {
		m.Header.Set(ExpectedLastSeqHeader, strconv.FormatUint(*o.lastSeq, 10))
	}
	if o.lastSubjectSeq != nil {
		m.Header.Set(ExpectedLastSubjSeqHeader, strconv.FormatUint(*o.lastSubjectSeq, 10))
	}
	if o.lastSubject != "" {
		m.Header.Set(ExpectedLastSubjSeqSubjHeader, o.lastSubject)
		m.Header.Set(ExpectedLastSubjSeqHeader, strconv.FormatUint(*o.lastSubjectSeq, 10))
	}
	if o.ttl > 0 {
		m.Header.Set(MsgTTLHeader, o.ttl.String())
	}
	if o.schedule != "" {
		m.Header.Set(ScheduleHeader, o.schedule)
	}
	if o.scheduleTarget != "" {
		m.Header.Set(ScheduleTargetHeader, o.scheduleTarget)
	}
	if o.scheduleSource != "" {
		m.Header.Set(ScheduleSourceHeader, o.scheduleSource)
	}
	if o.scheduleTTL != "" {
		m.Header.Set(ScheduleTTLHeader, o.scheduleTTL)
	}
	if o.scheduleTZ != "" {
		m.Header.Set(ScheduleTimeZoneHeader, o.scheduleTZ)
	}

	var resp *nats.Msg
	var err error

	resp, err = js.conn.RequestMsgWithContext(ctx, m)

	if err != nil {
		for r := 0; errors.Is(err, nats.ErrNoResponders) && (r < o.retryAttempts || o.retryAttempts < 0); r++ {
			// To protect against small blips in leadership changes etc, if we get a no responders here retry.
			select {
			case <-ctx.Done():
			case <-time.After(o.retryWait):
			}
			resp, err = js.conn.RequestMsgWithContext(ctx, m)
		}
		if err != nil {
			if errors.Is(err, nats.ErrNoResponders) {
				return nil, ErrNoStreamResponse
			}
			return nil, err
		}
	}

	var ackResp pubAckResponse
	if err := json.Unmarshal(resp.Data, &ackResp); err != nil {
		return nil, ErrInvalidJSAck
	}
	if ackResp.Error != nil {
		return nil, fmt.Errorf("nats: %w", ackResp.Error)
	}
	if ackResp.PubAck == nil || ackResp.PubAck.Stream == "" {
		return nil, ErrInvalidJSAck
	}
	return ackResp.PubAck, nil
}

// PublishAsync performs an asynchronous publish to a stream and returns
// [PubAckFuture] interface. It accepts subject name (which must be bound
// to a stream) and message payload.
func (js *jetStream) PublishAsync(subj string, data []byte, opts ...PublishOpt) (PubAckFuture, error) {
	return js.PublishMsgAsync(&nats.Msg{Subject: subj, Data: data}, opts...)
}

// PublishMsgAsync performs an asynchronous publish to a stream and
// returns [PubAckFuture] interface. It accepts subject name (which must
// be bound to a stream) and nats.Message.
func (js *jetStream) PublishMsgAsync(m *nats.Msg, opts ...PublishOpt) (PubAckFuture, error) {
	o := pubOpts{
		retryWait:     DefaultPubRetryWait,
		retryAttempts: DefaultPubRetryAttempts,
	}
	if len(opts) > 0 {
		if m.Header == nil {
			m.Header = nats.Header{}
		}
		for _, opt := range opts {
			if err := opt(&o); err != nil {
				return nil, err
			}
		}
	}
	defaultStallWait := 200 * time.Millisecond

	stallWait := defaultStallWait
	if o.stallWait > 0 {
		stallWait = o.stallWait
	}

	if o.id != "" {
		m.Header.Set(MsgIDHeader, o.id)
	}
	if o.lastMsgID != "" {
		m.Header.Set(ExpectedLastMsgIDHeader, o.lastMsgID)
	}
	if o.stream != "" {
		m.Header.Set(ExpectedStreamHeader, o.stream)
	}
	if o.lastSeq != nil {
		m.Header.Set(ExpectedLastSeqHeader, strconv.FormatUint(*o.lastSeq, 10))
	}
	if o.lastSubjectSeq != nil {
		m.Header.Set(ExpectedLastSubjSeqHeader, strconv.FormatUint(*o.lastSubjectSeq, 10))
	}
	if o.lastSubject != "" {
		m.Header.Set(ExpectedLastSubjSeqSubjHeader, o.lastSubject)
		m.Header.Set(ExpectedLastSubjSeqHeader, strconv.FormatUint(*o.lastSubjectSeq, 10))
	}
	if o.ttl > 0 {
		m.Header.Set(MsgTTLHeader, o.ttl.String())
	}
	if o.schedule != "" {
		m.Header.Set(ScheduleHeader, o.schedule)
	}
	if o.scheduleTarget != "" {
		m.Header.Set(ScheduleTargetHeader, o.scheduleTarget)
	}
	if o.scheduleSource != "" {
		m.Header.Set(ScheduleSourceHeader, o.scheduleSource)
	}
	if o.scheduleTTL != "" {
		m.Header.Set(ScheduleTTLHeader, o.scheduleTTL)
	}
	if o.scheduleTZ != "" {
		m.Header.Set(ScheduleTimeZoneHeader, o.scheduleTZ)
	}

	paf := o.pafRetry
	if paf == nil && m.Reply != "" {
		return nil, ErrAsyncPublishReplySubjectSet
	}

	var id string
	var reply string

	// register new paf if not retrying
	if paf == nil {
		var err error
		reply, err = js.newAsyncReply()
		if err != nil {
			return nil, fmt.Errorf("nats: error creating async reply handler: %s", err)
		}
		id = reply[js.opts.replyPrefixLen:]
		paf = &pubAckFuture{msg: m, jsClient: js.publisher, maxRetries: o.retryAttempts, retryWait: o.retryWait, reply: reply}
		numPending, maxPending := js.registerPAF(id, paf)

		if maxPending > 0 && numPending > maxPending {
			select {
			case <-js.asyncStall():
			case <-time.After(stallWait):
				js.clearPAF(id)
				return nil, ErrTooManyStalledMsgs
			}
		}
		if js.publisher.ackTimeout > 0 {
			paf.timeout = time.AfterFunc(js.publisher.ackTimeout, func() {
				js.publisher.Lock()
				defer js.publisher.Unlock()

				if _, ok := js.publisher.acks[id]; !ok {
					// paf has already been resolved
					// while waiting for the lock
					return
				}

				// ack timed out, remove from pending acks
				delete(js.publisher.acks, id)

				// check on anyone stalled and waiting.
				if js.publisher.stallCh != nil && len(js.publisher.acks) < js.publisher.maxpa {
					close(js.publisher.stallCh)
					js.publisher.stallCh = nil
				}

				// send error to user
				paf.err = ErrAsyncPublishTimeout
				if paf.errCh != nil {
					paf.errCh <- paf.err
				}

				// call error callback if set
				if js.publisher.asyncPublisherOpts.aecb != nil {
					js.publisher.asyncPublisherOpts.aecb(js, paf.msg, ErrAsyncPublishTimeout)
				}

				// check on anyone one waiting on done status.
				if js.publisher.doneCh != nil && len(js.publisher.acks) == 0 {
					close(js.publisher.doneCh)
					js.publisher.doneCh = nil
				}
			})
		}
	} else {
		// when retrying, get the ID from existing reply subject
		reply = paf.reply
		if paf.timeout != nil {
			paf.timeout.Reset(js.publisher.ackTimeout)
		}
		id = reply[js.opts.replyPrefixLen:]
	}

	pubMsg := &nats.Msg{
		Subject: m.Subject,
		Reply:   reply,
		Data:    m.Data,
		Header:  m.Header,
	}
	if err := js.conn.PublishMsg(pubMsg); err != nil {
		js.clearPAF(id)
		return nil, err
	}

	return paf, nil
}

// For quick token lookup etc.
const (
	aReplyTokensize = 6
)

func (js *jetStream) newAsyncReply() (string, error) {
	js.publisher.Lock()
	if js.publisher.replySub == nil {
		// Create our wildcard reply subject.
		sha := sha256.New()
		sha.Write([]byte(nuid.Next()))
		b := sha.Sum(nil)
		for i := 0; i < aReplyTokensize; i++ {
			b[i] = rdigits[int(b[i]%base)]
		}
		js.publisher.replyPrefix = fmt.Sprintf("%s%s.", js.opts.replyPrefix, b[:aReplyTokensize])
		sub, err := js.conn.Subscribe(fmt.Sprintf("%s*", js.publisher.replyPrefix), js.handleAsyncReply)
		if err != nil {
			js.publisher.Unlock()
			return "", err
		}
		js.publisher.replySub = sub
		js.publisher.rr = rand.New(rand.NewSource(time.Now().UnixNano()))
	}
	if js.publisher.connStatusCh == nil {
		js.publisher.connStatusCh = js.conn.StatusChanged(nats.RECONNECTING, nats.CLOSED)
		go js.resetPendingAcksOnReconnect()
	}
	var sb strings.Builder
	sb.WriteString(js.publisher.replyPrefix)
	for {
		rn := js.publisher.rr.Int63()
		var b [aReplyTokensize]byte
		for i, l := 0, rn; i < len(b); i++ {
			b[i] = rdigits[l%base]
			l /= base
		}
		if _, ok := js.publisher.acks[string(b[:])]; ok {
			continue
		}
		sb.Write(b[:])
		break
	}

	js.publisher.Unlock()
	return sb.String(), nil
}

// Handle an async reply from PublishAsync.
func (js *jetStream) handleAsyncReply(m *nats.Msg) {
	if len(m.Subject) <= js.opts.replyPrefixLen {
		return
	}
	id := m.Subject[js.opts.replyPrefixLen:]

	js.publisher.Lock()

	paf := js.getPAF(id)
	if paf == nil {
		js.publisher.Unlock()
		return
	}

	closeStc := func() {
		// Check on anyone stalled and waiting.
		if js.publisher.stallCh != nil && len(js.publisher.acks) < js.publisher.maxpa {
			close(js.publisher.stallCh)
			js.publisher.stallCh = nil
		}
	}

	closeDchFn := func() func() {
		var dch chan struct{}
		// Check on anyone one waiting on done status.
		if js.publisher.doneCh != nil && len(js.publisher.acks) == 0 {
			dch = js.publisher.doneCh
			js.publisher.doneCh = nil
		}
		// Return function to close done channel which
		// should be deferred so that error is processed and
		// can be checked.
		return func() {
			if dch != nil {
				close(dch)
			}
		}
	}

	doErr := func(err error) {
		paf.err = err
		if paf.errCh != nil {
			paf.errCh <- paf.err
		}
		cb := js.publisher.asyncPublisherOpts.aecb
		js.publisher.Unlock()
		if cb != nil {
			cb(js, paf.msg, err)
		}
	}

	if paf.timeout != nil {
		paf.timeout.Stop()
	}

	// Process no responders etc.
	if len(m.Data) == 0 && m.Header.Get(statusHdr) == statusNoResponders {
		if paf.retries < paf.maxRetries {
			paf.retries++
			time.AfterFunc(paf.retryWait, func() {
				js.publisher.Lock()
				paf := js.getPAF(id)
				js.publisher.Unlock()
				if paf == nil {
					return
				}
				_, err := js.PublishMsgAsync(paf.msg, func(po *pubOpts) error {
					po.pafRetry = paf
					return nil
				})
				if err != nil {
					js.publisher.Lock()
					doErr(err)
				}
			})
			js.publisher.Unlock()
			return
		}
		delete(js.publisher.acks, id)
		closeStc()
		defer closeDchFn()()
		doErr(ErrNoStreamResponse)
		return
	}

	// Remove
	delete(js.publisher.acks, id)
	closeStc()
	defer closeDchFn()()

	var pa pubAckResponse
	if err := json.Unmarshal(m.Data, &pa); err != nil {
		doErr(ErrInvalidJSAck)
		return
	}
	if pa.Error != nil {
		doErr(pa.Error)
		return
	}
	if pa.PubAck == nil || pa.PubAck.Stream == "" {
		doErr(ErrInvalidJSAck)
		return
	}

	// So here we have received a proper puback.
	paf.ack = pa.PubAck
	if paf.doneCh != nil {
		paf.doneCh <- paf.ack
	}
	cb := js.publisher.asyncPublisherOpts.ackcb
	js.publisher.Unlock()
	if cb != nil {
		cb(js, paf.msg, paf.ack)
	}
}

func (js *jetStream) resetPendingAcksOnReconnect() {
	js.publisher.Lock()
	connStatusCh := js.publisher.connStatusCh
	js.publisher.Unlock()
	for {
		newStatus, ok := <-connStatusCh
		if !ok || newStatus == nats.CLOSED {
			return
		}
		js.publisher.Lock()
		errCb := js.publisher.asyncPublisherOpts.aecb
		for id, paf := range js.publisher.acks {
			paf.err = nats.ErrDisconnected
			if paf.errCh != nil {
				paf.errCh <- paf.err
			}
			if errCb != nil {
				defer errCb(js, paf.msg, nats.ErrDisconnected)
			}
			delete(js.publisher.acks, id)
		}
		if js.publisher.doneCh != nil {
			close(js.publisher.doneCh)
			js.publisher.doneCh = nil
		}
		js.publisher.Unlock()
	}
}

// registerPAF will register for a PubAckFuture.
func (js *jetStream) registerPAF(id string, paf *pubAckFuture) (int, int) {
	js.publisher.Lock()
	if js.publisher.acks == nil {
		js.publisher.acks = make(map[string]*pubAckFuture)
	}
	js.publisher.acks[id] = paf
	np := len(js.publisher.acks)
	maxpa := js.publisher.asyncPublisherOpts.maxpa
	js.publisher.Unlock()
	return np, maxpa
}

// Lock should be held.
func (js *jetStream) getPAF(id string) *pubAckFuture {
	if js.publisher.acks == nil {
		return nil
	}
	return js.publisher.acks[id]
}

// clearPAF will remove a PubAckFuture that was registered.
func (js *jetStream) clearPAF(id string) {
	js.publisher.Lock()
	delete(js.publisher.acks, id)
	js.publisher.Unlock()
}

func (js *jetStream) asyncStall() <-chan struct{} {
	js.publisher.Lock()
	if js.publisher.stallCh == nil {
		js.publisher.stallCh = make(chan struct{})
	}
	stc := js.publisher.stallCh
	js.publisher.Unlock()
	return stc
}

func (paf *pubAckFuture) Ok() <-chan *PubAck {
	paf.jsClient.Lock()
	defer paf.jsClient.Unlock()

	if paf.doneCh == nil {
		paf.doneCh = make(chan *PubAck, 1)
		if paf.ack != nil {
			paf.doneCh <- paf.ack
		}
	}

	return paf.doneCh
}

func (paf *pubAckFuture) Err() <-chan error {
	paf.jsClient.Lock()
	defer paf.jsClient.Unlock()

	if paf.errCh == nil {
		paf.errCh = make(chan error, 1)
		if paf.err != nil {
			paf.errCh <- paf.err
		}
	}

	return paf.errCh
}

func (paf *pubAckFuture) Msg() *nats.Msg {
	paf.jsClient.RLock()
	defer paf.jsClient.RUnlock()
	return paf.msg
}

// PublishAsyncPending returns the number of async publishes outstanding
// for this context.
func (js *jetStream) PublishAsyncPending() int {
	js.publisher.RLock()
	defer js.publisher.RUnlock()
	return len(js.publisher.acks)
}

// PublishAsyncComplete returns a channel that will be closed when all
// outstanding asynchronously published messages are acknowledged by the
// server.
func (js *jetStream) PublishAsyncComplete() <-chan struct{} {
	js.publisher.Lock()
	defer js.publisher.Unlock()
	if js.publisher.doneCh == nil {
		js.publisher.doneCh = make(chan struct{})
	}
	dch := js.publisher.doneCh
	if len(js.publisher.acks) == 0 {
		close(js.publisher.doneCh)
		js.publisher.doneCh = nil
	}
	return dch
}
