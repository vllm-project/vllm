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
	"fmt"
	"time"
)

type pullOptFunc func(*consumeOpts) error

func (fn pullOptFunc) configureConsume(opts *consumeOpts) error {
	return fn(opts)
}

func (fn pullOptFunc) configureMessages(opts *consumeOpts) error {
	return fn(opts)
}

// WithClientTrace enables request/response API calls tracing.
func WithClientTrace(ct *ClientTrace) JetStreamOpt {
	return func(opts *JetStreamOptions) error {
		opts.ClientTrace = ct
		return nil
	}
}

// WithPublishAsyncErrHandler sets error handler for async message publish.
func WithPublishAsyncErrHandler(cb MsgErrHandler) JetStreamOpt {
	return func(opts *JetStreamOptions) error {
		opts.publisherOpts.aecb = cb
		return nil
	}
}

// WithPublishAsyncAckHandler sets an ack handler for async message publish,
// invoked when a publish is successfully acked by the server. See
// [MsgAckHandler] for constraints on the handler.
func WithPublishAsyncAckHandler(cb MsgAckHandler) JetStreamOpt {
	return func(opts *JetStreamOptions) error {
		opts.publisherOpts.ackcb = cb
		return nil
	}
}

// WithPublishAsyncMaxPending sets the maximum outstanding async publishes that
// can be inflight at one time.
func WithPublishAsyncMaxPending(max int) JetStreamOpt {
	return func(opts *JetStreamOptions) error {
		if max < 1 {
			return fmt.Errorf("%w: max ack pending should be >= 1", ErrInvalidOption)
		}
		opts.publisherOpts.maxpa = max
		return nil
	}
}

// WithPublishAsyncTimeout sets the timeout for async message publish.
// If not provided, timeout is disabled.
func WithPublishAsyncTimeout(dur time.Duration) JetStreamOpt {
	return func(opts *JetStreamOptions) error {
		opts.publisherOpts.ackTimeout = dur
		return nil
	}
}

// WithDefaultTimeout sets the default timeout for JetStream API requests.
// It is used when context used for the request does not have a deadline set.
// If not provided, a default of 5 seconds will be used.
func WithDefaultTimeout(timeout time.Duration) JetStreamOpt {
	return func(opts *JetStreamOptions) error {
		if timeout <= 0 {
			return fmt.Errorf("%w: timeout value must be greater than 0", ErrInvalidOption)
		}
		opts.DefaultTimeout = timeout
		return nil
	}
}

// WithPurgeSubject sets a specific subject for which messages on a stream will
// be purged
func WithPurgeSubject(subject string) StreamPurgeOpt {
	return func(req *StreamPurgeRequest) error {
		req.Subject = subject
		return nil
	}
}

// WithPurgeSequence is used to set a specific sequence number up to which (but
// not including) messages will be purged from a stream Can be combined with
// [WithPurgeSubject] option, but not with [WithPurgeKeep]
func WithPurgeSequence(sequence uint64) StreamPurgeOpt {
	return func(req *StreamPurgeRequest) error {
		if req.Keep != 0 {
			return fmt.Errorf("%w: both 'keep' and 'sequence' cannot be provided in purge request", ErrInvalidOption)
		}
		req.Sequence = sequence
		return nil
	}
}

// WithPurgeKeep sets the number of messages to be kept in the stream after
// purge. Can be combined with [WithPurgeSubject] option, but not with
// [WithPurgeSequence]
func WithPurgeKeep(keep uint64) StreamPurgeOpt {
	return func(req *StreamPurgeRequest) error {
		if req.Sequence != 0 {
			return fmt.Errorf("%w: both 'keep' and 'sequence' cannot be provided in purge request", ErrInvalidOption)
		}
		req.Keep = keep
		return nil
	}
}

// WithGetMsgSubject sets the stream subject from which the message should be
// retrieved. Server will return a first message with a seq >= to the input seq
// that has the specified subject.
func WithGetMsgSubject(subject string) GetMsgOpt {
	return func(req *apiMsgGetRequest) error {
		req.NextFor = subject
		return nil
	}
}

// PullMaxMessages limits the number of messages to be buffered in the client.
// If not provided, a default of 500 messages will be used.
// This option is exclusive with PullMaxBytes.
//
// PullMaxMessages implements both PullConsumeOpt and PullMessagesOpt, allowing
// it to configure Consumer.Consume and Consumer.Messages.
type PullMaxMessages int

func (max PullMaxMessages) configureConsume(opts *consumeOpts) error {
	if max <= 0 {
		return fmt.Errorf("%w: maxMessages size must be at least 1", ErrInvalidOption)
	}
	opts.MaxMessages = int(max)
	return nil
}

func (max PullMaxMessages) configureMessages(opts *consumeOpts) error {
	if max <= 0 {
		return fmt.Errorf("%w: maxMessages size must be at least 1", ErrInvalidOption)
	}
	opts.MaxMessages = int(max)
	return nil
}

type pullMaxMessagesWithBytesLimit struct {
	maxMessages int
	maxBytes    int
}

// PullMaxMessagesWithBytesLimit limits the number of messages to be buffered
// in the client. Additionally, it sets the maximum size a single fetch request
// can have. Note that this will not limit the total size of messages buffered
// in the client, but rather can serve as a way to limit what nats server will
// have to internally buffer for a single fetch request.
//
// The byte limit should never be set to a value lower than the maximum message
// size that can be expected from the server. If the byte limit is lower than
// the maximum message size, the consumer will stall and not be able to consume
// messages.
//
// This is an advanced option and should be used with caution. Most users should
// use [PullMaxMessages] or [PullMaxBytes] instead.
//
// PullMaxMessagesWithBytesLimit implements both PullConsumeOpt and
// PullMessagesOpt, allowing it to configure Consumer.Consume and Consumer.Messages.
func PullMaxMessagesWithBytesLimit(maxMessages, byteLimit int) pullMaxMessagesWithBytesLimit {
	return pullMaxMessagesWithBytesLimit{maxMessages, byteLimit}
}

func (m pullMaxMessagesWithBytesLimit) configureConsume(opts *consumeOpts) error {
	if m.maxMessages <= 0 {
		return fmt.Errorf("%w: maxMessages size must be at least 1", ErrInvalidOption)
	}
	if m.maxBytes <= 0 {
		return fmt.Errorf("%w: maxBytes size must be at least 1", ErrInvalidOption)
	}
	if opts.MaxMessages > 0 {
		return fmt.Errorf("%w: maxMessages already set", ErrInvalidOption)
	}
	opts.MaxMessages = m.maxMessages
	opts.MaxBytes = m.maxBytes
	opts.LimitSize = true

	return nil
}

func (m pullMaxMessagesWithBytesLimit) configureMessages(opts *consumeOpts) error {
	if m.maxMessages <= 0 {
		return fmt.Errorf("%w: maxMessages size must be at least 1", ErrInvalidOption)
	}
	if m.maxBytes <= 0 {
		return fmt.Errorf("%w: maxBytes size must be at least 1", ErrInvalidOption)
	}
	if opts.MaxMessages > 0 {
		return fmt.Errorf("%w: maxMessages already set", ErrInvalidOption)
	}
	opts.MaxMessages = m.maxMessages
	opts.MaxBytes = m.maxBytes
	opts.LimitSize = true

	return nil
}

// PullExpiry sets timeout on a single pull request, waiting until at least one
// message is available.
// If not provided, a default of 30 seconds will be used.
//
// PullExpiry implements both PullConsumeOpt and PullMessagesOpt, allowing
// it to configure Consumer.Consume and Consumer.Messages.
type PullExpiry time.Duration

func (exp PullExpiry) configureConsume(opts *consumeOpts) error {
	expiry := time.Duration(exp)
	if expiry < time.Second {
		return fmt.Errorf("%w: expires value must be at least 1s", ErrInvalidOption)
	}
	opts.Expires = expiry
	return nil
}

func (exp PullExpiry) configureMessages(opts *consumeOpts) error {
	expiry := time.Duration(exp)
	if expiry < time.Second {
		return fmt.Errorf("%w: expires value must be at least 1s", ErrInvalidOption)
	}
	opts.Expires = expiry
	return nil
}

// PullMaxBytes limits the number of bytes to be buffered in the client.
// If not provided, the limit is not set (max messages will be used instead).
// This option is exclusive with PullMaxMessages.
//
// The value should be set to a high enough value to accommodate the largest
// message expected from the server. Note that it may not be sufficient to set
// this value to the maximum message size, as this setting controls the client
// buffer size, not the max bytes requested from the server within a single pull
// request. If the value is set too low, the consumer will stall and not be able
// to consume messages.
//
// PullMaxBytes implements both PullConsumeOpt and PullMessagesOpt, allowing
// it to configure Consumer.Consume and Consumer.Messages.
type PullMaxBytes int

func (max PullMaxBytes) configureConsume(opts *consumeOpts) error {
	if max <= 0 {
		return fmt.Errorf("%w: max bytes must be greater than 0", ErrInvalidOption)
	}
	opts.MaxBytes = int(max)
	return nil
}

func (max PullMaxBytes) configureMessages(opts *consumeOpts) error {
	if max <= 0 {
		return fmt.Errorf("%w: max bytes must be greater than 0", ErrInvalidOption)
	}
	opts.MaxBytes = int(max)
	return nil
}

// PullThresholdMessages sets the message count on which consuming will trigger
// new pull request to the server. Defaults to 50% of MaxMessages.
//
// PullThresholdMessages implements both PullConsumeOpt and PullMessagesOpt,
// allowing it to configure Consumer.Consume and Consumer.Messages.
type PullThresholdMessages int

func (t PullThresholdMessages) configureConsume(opts *consumeOpts) error {
	opts.ThresholdMessages = int(t)
	return nil
}

func (t PullThresholdMessages) configureMessages(opts *consumeOpts) error {
	opts.ThresholdMessages = int(t)
	return nil
}

// PullThresholdBytes sets the byte count on which consuming will trigger
// new pull request to the server. Defaults to 50% of MaxBytes (if set).
//
// PullThresholdBytes implements both PullConsumeOpt and PullMessagesOpt,
// allowing it to configure Consumer.Consume and Consumer.Messages.
type PullThresholdBytes int

func (t PullThresholdBytes) configureConsume(opts *consumeOpts) error {
	opts.ThresholdBytes = int(t)
	return nil
}

func (t PullThresholdBytes) configureMessages(opts *consumeOpts) error {
	opts.ThresholdBytes = int(t)
	return nil
}

// PullMinPending sets the minimum number of messages that should be pending for
// a consumer with PriorityPolicyOverflow to be considered for delivery.
// If provided, PullPriorityGroup must be set as well and the consumer has to have
// PriorityPolicy set to PriorityPolicyOverflow.
//
// PullMinPending implements both PullConsumeOpt and PullMessagesOpt, allowing
// it to configure Consumer.Consume and Consumer.Messages.
type PullMinPending int

func (min PullMinPending) configureConsume(opts *consumeOpts) error {
	if min < 1 {
		return fmt.Errorf("%w: min pending should be more than 0", ErrInvalidOption)
	}
	opts.MinPending = int64(min)
	return nil
}

func (min PullMinPending) configureMessages(opts *consumeOpts) error {
	if min < 1 {
		return fmt.Errorf("%w: min pending should be more than 0", ErrInvalidOption)
	}
	opts.MinPending = int64(min)
	return nil
}

// PullMinAckPending sets the minimum number of pending acks that should be
// present for a consumer with PriorityPolicyOverflow to be considered for
// delivery. If provided, PullPriorityGroup must be set as well and the consumer
// has to have PriorityPolicy set to PriorityPolicyOverflow.
//
// PullMinAckPending implements both PullConsumeOpt and PullMessagesOpt, allowing
// it to configure Consumer.Consume and Consumer.Messages.
type PullMinAckPending int

func (min PullMinAckPending) configureConsume(opts *consumeOpts) error {
	if min < 1 {
		return fmt.Errorf("%w: min pending should be more than 0", ErrInvalidOption)
	}
	opts.MinAckPending = int64(min)
	return nil
}

func (min PullMinAckPending) configureMessages(opts *consumeOpts) error {
	if min < 1 {
		return fmt.Errorf("%w: min pending should be more than 0", ErrInvalidOption)
	}
	opts.MinAckPending = int64(min)
	return nil
}

// PullPrioritized sets the priority used when sending pull requests for consumer with
// PriorityPolicyPrioritized. Lower values indicate higher priority (0 is the
// highest priority). Maximum priority value is 9.
//
// If provided, PullPriorityGroup must be set as well and the consumer has to
// have PriorityPolicy set to PriorityPolicyPrioritized.
//
// PullPrioritized implements both PullConsumeOpt and PullMessagesOpt, allowing
// it to configure Consumer.Consume and Consumer.Messages.
type PullPrioritized uint8

func (p PullPrioritized) configureConsume(opts *consumeOpts) error {
	opts.Priority = uint8(p)
	return nil
}
func (p PullPrioritized) configureMessages(opts *consumeOpts) error {
	opts.Priority = uint8(p)
	return nil
}

// PullPriorityGroup sets the priority group for a consumer.
// It has to match one of the priority groups set on the consumer.
//
// PullPriorityGroup implements both PullConsumeOpt and PullMessagesOpt, allowing
// it to configure Consumer.Consume and Consumer.Messages.
type PullPriorityGroup string

func (g PullPriorityGroup) configureConsume(opts *consumeOpts) error {
	opts.Group = string(g)
	return nil
}

func (g PullPriorityGroup) configureMessages(opts *consumeOpts) error {
	opts.Group = string(g)
	return nil
}

// PullHeartbeat sets the idle heartbeat duration for a pull subscription
// If a client does not receive a heartbeat message from a stream for more
// than the idle heartbeat setting, the subscription will be removed
// and error will be passed to the message handler.
// If not provided, a default PullExpiry / 2 will be used (capped at 30 seconds)
//
// PullHeartbeat implements both PullConsumeOpt and PullMessagesOpt, allowing
// it to configure Consumer.Consume and Consumer.Messages.
type PullHeartbeat time.Duration

func (hb PullHeartbeat) configureConsume(opts *consumeOpts) error {
	hbTime := time.Duration(hb)
	if hbTime < 500*time.Millisecond || hbTime > 30*time.Second {
		return fmt.Errorf("%w: idle_heartbeat value must be within 500ms-30s range", ErrInvalidOption)
	}
	opts.Heartbeat = hbTime
	return nil
}

func (hb PullHeartbeat) configureMessages(opts *consumeOpts) error {
	hbTime := time.Duration(hb)
	if hbTime < 500*time.Millisecond || hbTime > 30*time.Second {
		return fmt.Errorf("%w: idle_heartbeat value must be within 500ms-30s range", ErrInvalidOption)
	}
	opts.Heartbeat = hbTime
	return nil
}

// StopAfter sets the number of messages after which the consumer is
// automatically stopped and no more messages are pulled from the server.
//
// StopAfter implements both PullConsumeOpt and PullMessagesOpt, allowing
// it to configure Consumer.Consume and Consumer.Messages.
type StopAfter int

func (nMsgs StopAfter) configureConsume(opts *consumeOpts) error {
	if nMsgs <= 0 {
		return fmt.Errorf("%w: auto stop after value cannot be less than 1", ErrInvalidOption)
	}
	opts.StopAfter = int(nMsgs)
	return nil
}

func (nMsgs StopAfter) configureMessages(opts *consumeOpts) error {
	if nMsgs <= 0 {
		return fmt.Errorf("%w: auto stop after value cannot be less than 1", ErrInvalidOption)
	}
	opts.StopAfter = int(nMsgs)
	return nil
}

// ConsumeErrHandler sets custom error handler invoked when an error was
// encountered while consuming messages It will be invoked for both terminal
// (Consumer Deleted, invalid request body) and non-terminal (e.g. missing
// heartbeats) errors.
type ConsumeErrHandler ConsumeErrHandlerFunc

func (c ConsumeErrHandler) configureConsume(opts *consumeOpts) error {
	opts.ErrHandler = c
	return nil
}

func (c ConsumeErrHandler) configurePushConsume(opts *pushConsumeOpts) error {
	opts.ErrHandler = c
	return nil
}

// WithMessagesErrOnMissingHeartbeat sets whether a missing heartbeat error
// should be reported when calling [MessagesContext.Next] (Default: true).
func WithMessagesErrOnMissingHeartbeat(hbErr bool) PullMessagesOpt {
	return pullOptFunc(func(cfg *consumeOpts) error {
		cfg.ReportMissingHeartbeats = hbErr
		return nil
	})
}

// FetchMinPending sets the minimum number of messages that should be pending for
// a consumer with PriorityPolicyOverflow to be considered for delivery.
// If provided, FetchPriorityGroup must be set as well and the consumer has to have
// PriorityPolicy set to PriorityPolicyOverflow.
func FetchMinPending(min int64) FetchOpt {
	return func(req *pullRequest) error {
		if min < 1 {
			return fmt.Errorf("%w: min pending should be more than 0", ErrInvalidOption)
		}
		req.MinPending = min
		return nil
	}
}

// FetchMinAckPending sets the minimum number of pending acks that should be
// present for a consumer with PriorityPolicyOverflow to be considered for
// delivery. If provided, FetchPriorityGroup must be set as well and the consumer
// has to have PriorityPolicy set to PriorityPolicyOverflow.
func FetchMinAckPending(min int64) FetchOpt {
	return func(req *pullRequest) error {
		if min < 1 {
			return fmt.Errorf("%w: min ack pending should be more than 0", ErrInvalidOption)
		}
		req.MinAckPending = min
		return nil
	}
}

// FetchPrioritized sets the priority used when sending fetch requests for consumer with
// PriorityPolicyPrioritized. Lower values indicate higher priority (0 is the
// highest priority). Maximum priority value is 9.
//
// If provided, FetchPriorityGroup must be set as well and the consumer has to
// have PriorityPolicy set to PriorityPolicyPrioritized.
func FetchPrioritized(priority uint8) FetchOpt {
	return func(req *pullRequest) error {
		req.Priority = priority
		return nil
	}
}

// FetchPriorityGroup sets the priority group for a consumer.
// It has to match one of the priority groups set on the consumer.
func FetchPriorityGroup(group string) FetchOpt {
	return func(req *pullRequest) error {
		req.Group = group
		return nil
	}
}

// FetchMaxWait sets custom timeout for fetching predefined batch of messages.
//
// If not provided, a default of 30 seconds will be used.
func FetchMaxWait(timeout time.Duration) FetchOpt {
	return func(req *pullRequest) error {
		if timeout <= 0 {
			return fmt.Errorf("%w: timeout value must be greater than 0", ErrInvalidOption)
		}
		req.Expires = timeout
		req.maxWaitSet = true
		return nil
	}
}

// FetchHeartbeat sets custom heartbeat for individual fetch request. If a
// client does not receive a heartbeat message from a stream for more than 2
// times the idle heartbeat setting, Fetch will return [ErrNoHeartbeat].
//
// Heartbeat value has to be lower than FetchMaxWait / 2.
//
// If not provided, heartbeat will is set to 5s for requests with FetchMaxWait > 10s
// and disabled otherwise.
func FetchHeartbeat(hb time.Duration) FetchOpt {
	return func(req *pullRequest) error {
		if hb <= 0 {
			return fmt.Errorf("%w: timeout value must be greater than 0", ErrInvalidOption)
		}
		req.Heartbeat = hb
		return nil
	}
}

// FetchContext sets a context for the Fetch operation.
// The Fetch operation will be canceled if the context is canceled.
// If the context has a deadline, it will be used to set expiry on pull request.
func FetchContext(ctx context.Context) FetchOpt {
	return func(req *pullRequest) error {
		req.ctx = ctx

		// If context has a deadline, use it to set expiry
		if deadline, ok := ctx.Deadline(); ok {
			remaining := time.Until(deadline)
			if remaining <= 0 {
				return fmt.Errorf("%w: context deadline already exceeded", ErrInvalidOption)
			}
			// Use 90% of remaining time for server (capped at 1s)
			buffer := time.Duration(float64(remaining) * 0.1)
			if buffer > time.Second {
				buffer = time.Second
			}
			req.Expires = remaining - buffer
		}

		return nil
	}
}

// WithDeletedDetails can be used to display the information about messages
// deleted from a stream on a stream info request
func WithDeletedDetails(deletedDetails bool) StreamInfoOpt {
	return func(req *streamInfoRequest) error {
		req.DeletedDetails = deletedDetails
		return nil
	}
}

// WithSubjectFilter can be used to display the information about messages
// stored on given subjects.
// NOTE: if the subject filter matches over 100k
// subjects, this will result in multiple requests to the server to retrieve all
// the information, and all of the returned subjects will be kept in memory.
func WithSubjectFilter(subject string) StreamInfoOpt {
	return func(req *streamInfoRequest) error {
		req.SubjectFilter = subject
		return nil
	}
}

// WithStreamListSubject can be used to filter results of ListStreams and
// StreamNames requests to only streams that have given subject in their
// configuration.
func WithStreamListSubject(subject string) StreamListOpt {
	return func(req *streamsRequest) error {
		req.Subject = subject
		return nil
	}
}

// WithMsgID sets the message ID used for deduplication.
func WithMsgID(id string) PublishOpt {
	return func(opts *pubOpts) error {
		opts.id = id
		return nil
	}
}

// WithMsgTTL sets per msg TTL.
// Requires [StreamConfig.AllowMsgTTL] to be enabled.
func WithMsgTTL(dur time.Duration) PublishOpt {
	return func(opts *pubOpts) error {
		opts.ttl = dur
		return nil
	}
}

// WithExpectStream sets the expected stream the message should be published to.
// If the message is published to a different stream server will reject the
// message and publish will fail.
func WithExpectStream(stream string) PublishOpt {
	return func(opts *pubOpts) error {
		opts.stream = stream
		return nil
	}
}

// WithExpectLastSequence sets the expected sequence number the last message
// on a stream should have. If the last message has a different sequence number
// server will reject the message and publish will fail.
func WithExpectLastSequence(seq uint64) PublishOpt {
	return func(opts *pubOpts) error {
		opts.lastSeq = &seq
		return nil
	}
}

// WithExpectLastSequencePerSubject sets the expected sequence number the last
// message on a subject the message is published to. If the last message on a
// subject has a different sequence number server will reject the message and
// publish will fail.
func WithExpectLastSequencePerSubject(seq uint64) PublishOpt {
	return func(opts *pubOpts) error {
		opts.lastSubjectSeq = &seq
		return nil
	}
}

// WithExpectLastSequenceForSubject sets the sequence and subject for which the
// last sequence number should be checked. If the last message on a subject
// has a different sequence number server will reject the message and publish
// will fail.
func WithExpectLastSequenceForSubject(seq uint64, subject string) PublishOpt {
	return func(opts *pubOpts) error {
		if subject == "" {
			return fmt.Errorf("%w: subject cannot be empty", ErrInvalidOption)
		}
		opts.lastSubjectSeq = &seq
		opts.lastSubject = subject
		return nil
	}
}

// WithExpectLastMsgID sets the expected message ID the last message on a stream
// should have. If the last message has a different message ID server will
// reject the message and publish will fail.
func WithExpectLastMsgID(id string) PublishOpt {
	return func(opts *pubOpts) error {
		opts.lastMsgID = id
		return nil
	}
}

// WithRetryWait sets the retry wait time when ErrNoResponders is encountered.
// Defaults to 250ms.
func WithRetryWait(dur time.Duration) PublishOpt {
	return func(opts *pubOpts) error {
		if dur <= 0 {
			return fmt.Errorf("%w: retry wait should be more than 0", ErrInvalidOption)
		}
		opts.retryWait = dur
		return nil
	}
}

// WithRetryAttempts sets the retry number of attempts when ErrNoResponders is
// encountered. Defaults to 2
func WithRetryAttempts(num int) PublishOpt {
	return func(opts *pubOpts) error {
		if num < 0 {
			return fmt.Errorf("%w: retry attempts cannot be negative", ErrInvalidOption)
		}
		opts.retryAttempts = num
		return nil
	}
}

// WithStallWait sets the max wait when the producer becomes stall producing
// messages. If a publish call is blocked for this long, ErrTooManyStalledMsgs
// is returned.
func WithStallWait(ttl time.Duration) PublishOpt {
	return func(opts *pubOpts) error {
		if ttl <= 0 {
			return fmt.Errorf("%w: stall wait should be more than 0", ErrInvalidOption)
		}
		opts.stallWait = ttl
		return nil
	}
}

// Predefined schedule constants for use with [WithScheduleCron].
const (
	ScheduleYearly  = "@yearly"
	ScheduleMonthly = "@monthly"
	ScheduleWeekly  = "@weekly"
	ScheduleDaily   = "@daily"
	ScheduleHourly  = "@hourly"
)

// WithScheduleAt schedules a message for one-time delivery at a specific time.
// Requires [StreamConfig.AllowMsgSchedules] to be enabled.
func WithScheduleAt(t time.Time) PublishOpt {
	return func(opts *pubOpts) error {
		opts.schedule = "@at " + t.UTC().Format(time.RFC3339)
		return nil
	}
}

// WithScheduleEvery schedules a message for repeated delivery at the given
// interval. The minimum interval is 1 second.
// Requires [StreamConfig.AllowMsgSchedules] to be enabled.
func WithScheduleEvery(d time.Duration) PublishOpt {
	return func(opts *pubOpts) error {
		if d < time.Second {
			return fmt.Errorf("%w: schedule interval must be at least 1s", ErrInvalidOption)
		}
		opts.schedule = "@every " + d.String()
		return nil
	}
}

// WithScheduleCron schedules a message using a cron expression.
// Accepts 6-field cron format (seconds minutes hours day-of-month month
// day-of-week) or predefined schedules (e.g. [ScheduleHourly],
// [ScheduleDaily]).
// Requires [StreamConfig.AllowMsgSchedules] to be enabled.
func WithScheduleCron(expr string) PublishOpt {
	return func(opts *pubOpts) error {
		if expr == "" {
			return fmt.Errorf("%w: cron expression cannot be empty", ErrInvalidOption)
		}
		opts.schedule = expr
		return nil
	}
}

// WithScheduleTarget sets the subject where scheduled messages will be
// delivered.
// This option is required for scheduled messages.
// Requires [StreamConfig.AllowMsgSchedules] to be enabled.
func WithScheduleTarget(subject string) PublishOpt {
	return func(opts *pubOpts) error {
		if subject == "" {
			return fmt.Errorf("%w: schedule target subject cannot be empty", ErrInvalidOption)
		}
		opts.scheduleTarget = subject
		return nil
	}
}

// WithScheduleSource instructs the scheduler to read the latest message
// from the given subject and republish it on schedule. Used for data
// sampling use cases.
// Requires [StreamConfig.AllowMsgSchedules] to be enabled.
func WithScheduleSource(subject string) PublishOpt {
	return func(opts *pubOpts) error {
		if subject == "" {
			return fmt.Errorf("%w: schedule source subject cannot be empty", ErrInvalidOption)
		}
		opts.scheduleSource = subject
		return nil
	}
}

// WithScheduleTTL sets the TTL on messages generated by the scheduler.
// Requires [StreamConfig.AllowMsgTTL] to be enabled on the stream.
func WithScheduleTTL(d time.Duration) PublishOpt {
	return func(opts *pubOpts) error {
		if d <= 0 {
			return fmt.Errorf("%w: schedule TTL must be greater than 0", ErrInvalidOption)
		}
		opts.scheduleTTL = d.String()
		return nil
	}
}

// WithScheduleTTLNever marks messages generated by the scheduler as never
// expiring. This overrides the stream's MaxAge for delivered messages.
// Requires [StreamConfig.AllowMsgTTL] to be enabled on the stream.
func WithScheduleTTLNever() PublishOpt {
	return func(opts *pubOpts) error {
		opts.scheduleTTL = "never"
		return nil
	}
}

// WithScheduleTimeZone sets the time zone for cron schedule expressions.
// The zone must be a valid IANA time zone name (e.g., "America/New_York").
// Only valid with cron schedule expressions; the server will reject this
// header if used with @at or @every schedules.
// Requires [StreamConfig.AllowMsgSchedules] to be enabled.
func WithScheduleTimeZone(zone string) PublishOpt {
	return func(opts *pubOpts) error {
		if zone == "" {
			return fmt.Errorf("%w: schedule time zone cannot be empty", ErrInvalidOption)
		}
		opts.scheduleTZ = zone
		return nil
	}
}

type nextOptFunc func(*nextOpts)

func (fn nextOptFunc) configureNext(opts *nextOpts) {
	fn(opts)
}

// NextMaxWait sets a timeout for the Next operation.
// If the timeout is reached before a message is available, a timeout error is returned.
func NextMaxWait(timeout time.Duration) NextOpt {
	return nextOptFunc(func(opts *nextOpts) {
		opts.timeout = timeout
	})
}

// NextContext sets a context for the Next operation.
// The Next operation will be canceled if the context is canceled.
func NextContext(ctx context.Context) NextOpt {
	return nextOptFunc(func(opts *nextOpts) {
		opts.ctx = ctx
	})
}
