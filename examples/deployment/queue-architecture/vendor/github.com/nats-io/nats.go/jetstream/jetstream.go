// Copyright 2022-2026 The NATS Authors
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
	"regexp"
	"slices"
	"strings"
	"time"

	"github.com/nats-io/nats.go"
	"github.com/nats-io/nuid"
)

type (

	// JetStream is the top-level interface for interacting with JetStream.
	// The capabilities of JetStream include:
	//
	// - Publishing messages to a stream using [Publisher].
	// - Managing streams using [StreamManager].
	// - Managing consumers using [StreamConsumerManager]. These are the same
	//   methods available on [Stream], but exposed here as a shortcut that bypasses
	//   stream lookup.
	// - Managing KeyValue stores using [KeyValueManager].
	// - Managing Object Stores using [ObjectStoreManager].
	//
	// JetStream can be created using [New], [NewWithAPIPrefix] or
	// [NewWithDomain] methods.
	JetStream interface {
		// AccountInfo fetches account information from the server, containing details
		// about the account associated with this JetStream connection. If account is
		// not enabled for JetStream, ErrJetStreamNotEnabledForAccount is returned. If
		// the server does not have JetStream enabled, ErrJetStreamNotEnabled is
		// returned.
		AccountInfo(ctx context.Context) (*AccountInfo, error)

		// Conn returns the underlying NATS connection.
		Conn() *nats.Conn

		// Options returns read-only JetStreamOptions used
		// when making requests to JetStream.
		Options() JetStreamOptions

		StreamConsumerManager
		StreamManager
		Publisher
		KeyValueManager
		ObjectStoreManager
	}

	// Publisher provides methods for publishing messages to a stream.
	// It is available as a part of [JetStream] interface.
	// The behavior of Publisher can be customized using [PublishOpt] options.
	Publisher interface {
		// Publish performs a synchronous publish to a stream and waits for ack
		// from server. It accepts subject name (which must be bound to a stream)
		// and message payload.
		Publish(ctx context.Context, subject string, payload []byte, opts ...PublishOpt) (*PubAck, error)

		// PublishMsg performs a synchronous publish to a stream and waits for
		// ack from server. It accepts subject name (which must be bound to a
		// stream) and nats.Message.
		PublishMsg(ctx context.Context, msg *nats.Msg, opts ...PublishOpt) (*PubAck, error)

		// PublishAsync performs a publish to a stream and returns
		// [PubAckFuture] interface, not blocking while waiting for an
		// acknowledgement. It accepts subject name (which must be bound to a
		// stream) and message payload.
		//
		// PublishAsync does not guarantee that the message has been
		// received by the server. It only guarantees that the message has been
		// sent to the server and thus messages can be stored in the stream
		// out of order in case of retries.
		PublishAsync(subject string, payload []byte, opts ...PublishOpt) (PubAckFuture, error)

		// PublishMsgAsync performs a publish to a stream and returns
		// [PubAckFuture] interface, not blocking while waiting for an
		// acknowledgement. It accepts subject name (which must
		// be bound to a stream) and nats.Message.
		//
		// PublishMsgAsync does not guarantee that the message has been
		// received by the server. It only guarantees that the message has been
		// sent to the server and thus messages can be stored in the stream
		// out of order in case of retries.
		PublishMsgAsync(msg *nats.Msg, opts ...PublishOpt) (PubAckFuture, error)

		// PublishAsyncPending returns the number of async publishes outstanding
		// for this context. An outstanding publish is one that has been
		// sent by the publisher but has not yet received an ack.
		PublishAsyncPending() int

		// PublishAsyncComplete returns a channel that will be closed when all
		// outstanding asynchronously published messages are acknowledged by the
		// server.
		PublishAsyncComplete() <-chan struct{}

		// CleanupPublisher will clean up the publishing side of JetStreamContext.
		//
		// This will unsubscribe from the internal reply subject if needed.
		// All pending async publishes will fail with ErrJetStreamContextClosed.
		//
		// If an error handler was provided, it will be called for each pending async
		// publish and PublishAsyncComplete will be closed.
		//
		// After completing JetStreamContext is still usable - internal subscription
		// will be recreated on next publish, but the acks from previous publishes will
		// be lost.
		CleanupPublisher()
	}

	// StreamManager provides CRUD API for managing streams. It is available as
	// a part of [JetStream] interface. CreateStream, UpdateStream,
	// CreateOrUpdateStream and Stream methods return a [Stream] interface, allowing
	// to operate on a stream.
	StreamManager interface {
		// CreateStream creates a new stream with the given config and returns an
		// interface to operate on it. If stream with given name already exists
		// and its configuration differs from the provided one,
		// ErrStreamNameAlreadyInUse is returned.
		CreateStream(ctx context.Context, cfg StreamConfig) (Stream, error)

		// UpdateStream updates an existing stream. If stream does not exist,
		// ErrStreamNotFound is returned.
		UpdateStream(ctx context.Context, cfg StreamConfig) (Stream, error)

		// CreateOrUpdateStream creates a stream with the given config. If stream
		// already exists, it will be updated (if possible).
		CreateOrUpdateStream(ctx context.Context, cfg StreamConfig) (Stream, error)

		// Stream fetches [StreamInfo] and returns a [Stream] interface for a given stream name.
		// If stream does not exist, ErrStreamNotFound is returned.
		Stream(ctx context.Context, stream string) (Stream, error)

		// StreamNameBySubject returns the name of the stream that listens on the given
		// subject. If no stream is bound to given subject, ErrStreamNotFound
		// is returned.
		StreamNameBySubject(ctx context.Context, subject string) (string, error)

		// DeleteStream removes a stream with given name. If stream does not
		// exist, ErrStreamNotFound is returned.
		DeleteStream(ctx context.Context, stream string) error

		// ListStreams returns StreamInfoLister, enabling iteration over a
		// channel of stream infos.
		ListStreams(context.Context, ...StreamListOpt) StreamInfoLister

		// StreamNames returns a StreamNameLister, enabling iteration over a
		// channel of stream names.
		StreamNames(context.Context, ...StreamListOpt) StreamNameLister
	}

	// StreamConsumerManager provides CRUD API for managing consumers. It is
	// available as a part of [JetStream] interface. This is an alternative to
	// [Stream] interface, allowing to bypass stream lookup. CreateConsumer,
	// UpdateConsumer, CreateOrUpdateConsumer and Consumer methods return a
	// [Consumer] interface, allowing operation on a consumer (e.g. consume
	// messages).
	StreamConsumerManager interface {
		// CreateOrUpdateConsumer creates a consumer on a given stream with
		// the given config. If consumer already exists, it will be updated (if
		// possible). Consumer interface is returned, allowing to operate on a
		// consumer (e.g. fetch messages).
		CreateOrUpdateConsumer(ctx context.Context, stream string, cfg ConsumerConfig) (Consumer, error)

		// CreateConsumer creates a consumer on a given stream with given
		// config. If consumer already exists and the provided configuration
		// differs from its configuration, ErrConsumerExists is returned. If the
		// provided configuration is the same as the existing consumer, the
		// existing consumer is returned. Consumer interface is returned,
		// allowing to operate on a consumer (e.g. fetch messages).
		CreateConsumer(ctx context.Context, stream string, cfg ConsumerConfig) (Consumer, error)

		// UpdateConsumer updates an existing consumer. If consumer does not
		// exist, ErrConsumerDoesNotExist is returned. Consumer interface is
		// returned, allowing to operate on a consumer (e.g. fetch messages).
		UpdateConsumer(ctx context.Context, stream string, cfg ConsumerConfig) (Consumer, error)

		// OrderedConsumer returns a client-managed ordered consumer for the given stream.
		// Ordered consumers use ephemeral pull consumers and automatically reset
		// themselves when ordering is lost. The client tracks state in memory and
		// recreates the underlying consumer as needed, making them resilient to deletes
		// and restarts while ensuring message order is preserved.
		OrderedConsumer(ctx context.Context, stream string, cfg OrderedConsumerConfig) (Consumer, error)

		// Consumer returns an interface to an existing consumer, allowing processing
		// of messages. If consumer does not exist, ErrConsumerNotFound is
		// returned.
		Consumer(ctx context.Context, stream string, consumer string) (Consumer, error)

		// DeleteConsumer removes a consumer with given name from a stream.
		// If consumer does not exist, ErrConsumerNotFound is returned.
		DeleteConsumer(ctx context.Context, stream string, consumer string) error

		// PauseConsumer pauses a consumer until the given time.
		PauseConsumer(ctx context.Context, stream string, consumer string, pauseUntil time.Time) (*ConsumerPauseResponse, error)

		// ResumeConsumer resumes a paused consumer.
		ResumeConsumer(ctx context.Context, stream string, consumer string) (*ConsumerPauseResponse, error)

		// ResetConsumer resets a consumer's delivery state. The consumer is
		// reset to deliver from ack_floor + 1.
		ResetConsumer(ctx context.Context, stream, consumer string) (*ConsumerResetResponse, error)

		// ResetConsumerToSequence resets a consumer's delivery state to the
		// given stream sequence. The seq must be compatible with the
		// consumer's DeliverPolicy. If incompatible, ErrConsumerInvalidReset
		// is returned.
		ResetConsumerToSequence(ctx context.Context, stream, consumer string, seq uint64) (*ConsumerResetResponse, error)

		// CreateOrUpdatePushConsumer creates a push consumer on a given stream with
		// the given config. If consumer already exists, it will be updated (if
		// possible). Consumer interface is returned, allowing to consume messages.
		CreateOrUpdatePushConsumer(ctx context.Context, stream string, cfg ConsumerConfig) (PushConsumer, error)

		// CreatePushConsumer creates a push consumer on a given stream with given
		// config. If consumer already exists and the provided configuration
		// differs from its configuration, ErrConsumerExists is returned. If the
		// provided configuration is the same as the existing consumer, the
		// existing consumer is returned. Consumer interface is returned,
		// allowing to consume messages.
		CreatePushConsumer(ctx context.Context, stream string, cfg ConsumerConfig) (PushConsumer, error)

		// UpdatePushConsumer updates an existing push consumer. If consumer does not
		// exist, ErrConsumerDoesNotExist is returned. Consumer interface is
		// returned, allowing to consume messages.
		UpdatePushConsumer(ctx context.Context, stream string, cfg ConsumerConfig) (PushConsumer, error)

		// PushConsumer returns an interface to an existing push consumer, allowing processing
		// of messages. If consumer does not exist, ErrConsumerNotFound is
		// returned.
		//
		// It returns ErrNotPushConsumer if the consumer is not a push consumer (delivery subject is not set).
		PushConsumer(ctx context.Context, stream string, consumer string) (PushConsumer, error)
	}

	// StreamListOpt is a functional option for [StreamManager.ListStreams] and
	// [StreamManager.StreamNames] methods.
	StreamListOpt func(*streamsRequest) error

	// AccountInfo contains information about the JetStream usage from the
	// current account.
	AccountInfo struct {
		// Tier is the current account usage tier.
		Tier

		// Domain is the domain name associated with this account.
		Domain string `json:"domain"`

		// API is the API usage statistics for this account.
		API APIStats `json:"api"`

		// Tiers is the list of available tiers for this account.
		Tiers map[string]Tier `json:"tiers"`
	}

	// Tier represents a JetStream account usage tier.
	Tier struct {
		// Memory is the memory storage being used for Stream Message storage.
		Memory uint64 `json:"memory"`

		// Store is the disk storage being used for Stream Message storage.
		Store uint64 `json:"storage"`

		// ReservedMemory is the number of bytes reserved for memory usage by
		// this account on the server
		ReservedMemory uint64 `json:"reserved_memory"`

		// ReservedStore is the number of bytes reserved for disk usage by this
		// account on the server
		ReservedStore uint64 `json:"reserved_storage"`

		// Streams is the number of streams currently defined for this account.
		Streams int `json:"streams"`

		// Consumers is the number of consumers currently defined for this
		// account.
		Consumers int `json:"consumers"`

		// Limits are the JetStream limits for this account.
		Limits AccountLimits `json:"limits"`
	}

	// APIStats reports on API calls to JetStream for this account.
	APIStats struct {
		// Level is the API level for this account.
		Level int `json:"level"`

		// Total is the total number of API calls.
		Total uint64 `json:"total"`

		// Errors is the total number of API errors.
		Errors uint64 `json:"errors"`

		// Inflight is the number of API calls currently in flight.
		Inflight uint64 `json:"inflight,omitempty"`
	}

	// AccountLimits includes the JetStream limits of the current account.
	AccountLimits struct {
		// MaxMemory is the maximum amount of memory available for this account.
		MaxMemory int64 `json:"max_memory"`

		// MaxStore is the maximum amount of disk storage available for this
		// account.
		MaxStore int64 `json:"max_storage"`

		// MaxStreams is the maximum number of streams allowed for this account.
		MaxStreams int `json:"max_streams"`

		// MaxConsumers is the maximum number of consumers allowed for this
		// account.
		MaxConsumers int `json:"max_consumers"`

		// MaxAckPending is the maximum number of outstanding ACKs any consumer
		// may configure.
		MaxAckPending int `json:"max_ack_pending"`

		// MemoryMaxStreamBytes is is the maximum size any single memory based
		// stream may be.
		MemoryMaxStreamBytes int64 `json:"memory_max_stream_bytes"`

		// StoreMaxStreamBytes is the maximum size any single storage based
		// stream may be.
		StoreMaxStreamBytes int64 `json:"storage_max_stream_bytes"`

		// MaxBytesRequired indicates if max bytes is required to be set for
		// streams in this account.
		MaxBytesRequired bool `json:"max_bytes_required"`
	}

	jetStream struct {
		conn *nats.Conn
		opts JetStreamOptions

		publisher *jetStreamClient
	}

	// JetStreamOpt is a functional option for [New], [NewWithAPIPrefix] and
	// [NewWithDomain] methods.
	JetStreamOpt func(*JetStreamOptions) error

	// JetStreamOptions are used to configure JetStream.
	JetStreamOptions struct {
		// APIPrefix is the prefix used for JetStream API requests.
		APIPrefix string

		// Domain is the domain name token used when sending JetStream requests.
		Domain string

		// DefaultTimeout is the default timeout used for JetStream API requests.
		// This applies when the context passed to JetStream methods does not have
		// a deadline set.
		DefaultTimeout time.Duration

		// ClientTrace enables request/response API calls tracing.
		ClientTrace *ClientTrace

		publisherOpts asyncPublisherOpts

		// this is the actual prefix used in the API requests
		// it is either APIPrefix or a domain specific prefix
		apiPrefix      string
		replyPrefix    string
		replyPrefixLen int
	}

	// ClientTrace can be used to trace API interactions for [JetStream].
	ClientTrace struct {
		// RequestSent is called when an API request is sent to the server.
		RequestSent func(subj string, payload []byte)

		// ResponseReceived is called when a response is received from the
		// server.
		ResponseReceived func(subj string, payload []byte, hdr nats.Header)
	}
	streamInfoResponse struct {
		apiResponse
		apiPaged
		*StreamInfo
	}

	accountInfoResponse struct {
		apiResponse
		AccountInfo
	}

	streamDeleteResponse struct {
		apiResponse
		Success bool `json:"success,omitempty"`
	}

	// StreamInfoLister is used to iterate over a channel of stream infos.
	// Err method can be used to check for errors encountered during iteration.
	// Info channel is always closed and therefore can be used in a range loop.
	StreamInfoLister interface {
		Info() <-chan *StreamInfo
		Err() error
	}

	// StreamNameLister is used to iterate over a channel of stream names.
	// Err method can be used to check for errors encountered during iteration.
	// Name channel is always closed and therefore can be used in a range loop.
	StreamNameLister interface {
		Name() <-chan string
		Err() error
	}

	apiPagedRequest struct {
		Offset int `json:"offset"`
	}

	streamLister struct {
		js       *jetStream
		offset   int
		pageInfo *apiPaged

		streams chan *StreamInfo
		names   chan string
		err     error
	}

	streamListResponse struct {
		apiResponse
		apiPaged
		Streams []*StreamInfo `json:"streams"`
	}

	streamNamesResponse struct {
		apiResponse
		apiPaged
		Streams []string `json:"streams"`
	}

	streamsRequest struct {
		apiPagedRequest
		Subject string `json:"subject,omitempty"`
	}
)

// defaultAPITimeout is used if context.Background() or context.TODO() is passed to API calls.
const defaultAPITimeout = 5 * time.Second

var subjectRegexp = regexp.MustCompile(`^[^ >]*[>]?$`)

// New returns a new JetStream instance.
// It uses default API prefix ($JS.API) for JetStream API requests.
// If a custom API prefix is required, use [NewWithAPIPrefix] or [NewWithDomain].
//
// Available options:
//   - [WithClientTrace] - enables request/response tracing.
//   - [WithPublishAsyncErrHandler] - sets error handler for async message publish.
//   - [WithPublishAsyncAckHandler] - sets ack handler for async message publish.
//   - [WithPublishAsyncMaxPending] - sets the maximum outstanding async publishes
//     that can be inflight at one time.
func New(nc *nats.Conn, opts ...JetStreamOpt) (JetStream, error) {
	jsOpts := JetStreamOptions{
		apiPrefix: DefaultAPIPrefix,
		publisherOpts: asyncPublisherOpts{
			maxpa: defaultAsyncPubAckInflight,
		},
		DefaultTimeout: defaultAPITimeout,
	}
	setReplyPrefix(nc, &jsOpts)
	for _, opt := range opts {
		if err := opt(&jsOpts); err != nil {
			return nil, err
		}
	}
	js := &jetStream{
		conn:      nc,
		opts:      jsOpts,
		publisher: &jetStreamClient{asyncPublisherOpts: jsOpts.publisherOpts},
	}

	return js, nil
}

const (
	// defaultAsyncPubAckInflight is the number of async pub acks inflight.
	defaultAsyncPubAckInflight = 4000
)

func setReplyPrefix(nc *nats.Conn, jsOpts *JetStreamOptions) {
	jsOpts.replyPrefix = nats.InboxPrefix
	if nc.Opts.InboxPrefix != "" {
		jsOpts.replyPrefix = nc.Opts.InboxPrefix + "."
	}
	// Add 1 for the dot separator.
	jsOpts.replyPrefixLen = len(jsOpts.replyPrefix) + aReplyTokensize + 1

}

// NewWithAPIPrefix returns a new JetStream instance and sets the API prefix to be used in requests to JetStream API.
// The API prefix will be used in API requests to JetStream, e.g. <prefix>.STREAM.INFO.<stream>.
//
// Available options:
//   - [WithClientTrace] - enables request/response tracing.
//   - [WithPublishAsyncErrHandler] - sets error handler for async message publish.
//   - [WithPublishAsyncAckHandler] - sets ack handler for async message publish.
//   - [WithPublishAsyncMaxPending] - sets the maximum outstanding async publishes
//     that can be inflight at one time.
func NewWithAPIPrefix(nc *nats.Conn, apiPrefix string, opts ...JetStreamOpt) (JetStream, error) {
	jsOpts := JetStreamOptions{
		publisherOpts: asyncPublisherOpts{
			maxpa: defaultAsyncPubAckInflight,
		},
		APIPrefix:      apiPrefix,
		DefaultTimeout: defaultAPITimeout,
	}
	setReplyPrefix(nc, &jsOpts)
	for _, opt := range opts {
		if err := opt(&jsOpts); err != nil {
			return nil, err
		}
	}
	if apiPrefix == "" {
		return nil, errors.New("API prefix cannot be empty")
	}
	if !strings.HasSuffix(apiPrefix, ".") {
		jsOpts.apiPrefix = fmt.Sprintf("%s.", apiPrefix)
	} else {
		jsOpts.apiPrefix = apiPrefix
	}
	js := &jetStream{
		conn:      nc,
		opts:      jsOpts,
		publisher: &jetStreamClient{asyncPublisherOpts: jsOpts.publisherOpts},
	}
	return js, nil
}

// NewWithDomain returns a new JetStream instance and sets the domain name token used when sending JetStream requests.
// The domain name token will be used in API requests to JetStream, e.g. $JS.<domain>.API.STREAM.INFO.<stream>.
//
// Available options:
//   - [WithClientTrace] - enables request/response tracing.
//   - [WithPublishAsyncErrHandler] - sets error handler for async message publish.
//   - [WithPublishAsyncAckHandler] - sets ack handler for async message publish.
//   - [WithPublishAsyncMaxPending] - sets the maximum outstanding async publishes
//     that can be inflight at one time.
func NewWithDomain(nc *nats.Conn, domain string, opts ...JetStreamOpt) (JetStream, error) {
	jsOpts := JetStreamOptions{
		publisherOpts: asyncPublisherOpts{
			maxpa: defaultAsyncPubAckInflight,
		},
		Domain:         domain,
		DefaultTimeout: defaultAPITimeout,
	}
	setReplyPrefix(nc, &jsOpts)
	for _, opt := range opts {
		if err := opt(&jsOpts); err != nil {
			return nil, err
		}
	}
	if domain == "" {
		return nil, errors.New("domain cannot be empty")
	}
	jsOpts.apiPrefix = fmt.Sprintf(jsDomainT, domain)
	js := &jetStream{
		conn:      nc,
		opts:      jsOpts,
		publisher: &jetStreamClient{asyncPublisherOpts: jsOpts.publisherOpts},
	}
	return js, nil
}

// Conn returns the underlying NATS connection.
func (js *jetStream) Conn() *nats.Conn {
	return js.conn
}

func (js *jetStream) Options() JetStreamOptions {
	opts := js.opts
	// Return a copy of ClientTrace to prevent modification
	if opts.ClientTrace != nil {
		clientTraceCopy := *opts.ClientTrace
		opts.ClientTrace = &clientTraceCopy
	}
	return opts
}

// CreateStream creates a new stream with the given config and returns an
// interface to operate on it. If stream with given name already exists,
// ErrStreamNameAlreadyInUse is returned.
func (js *jetStream) CreateStream(ctx context.Context, cfg StreamConfig) (Stream, error) {
	if err := validateStreamName(cfg.Name); err != nil {
		return nil, err
	}
	ctx, cancel := js.wrapContextWithoutDeadline(ctx)
	if cancel != nil {
		defer cancel()
	}

	ncfg, err := convertStreamConfigDomains(cfg)
	if err != nil {
		return nil, err
	}

	req, err := json.Marshal(ncfg)
	if err != nil {
		return nil, err
	}

	createSubject := fmt.Sprintf(apiStreamCreateT, cfg.Name)
	var resp streamInfoResponse

	if _, err = js.apiRequestJSON(ctx, createSubject, &resp, req); err != nil {
		return nil, err
	}
	if resp.Error != nil {
		if resp.Error.ErrorCode == JSErrCodeStreamNameInUse {
			return nil, ErrStreamNameAlreadyInUse
		}
		return nil, resp.Error
	}
	if resp.StreamInfo == nil {
		return nil, ErrInvalidJetStreamResponse
	}

	// check that input subject transform (if used) is reflected in the returned StreamInfo
	if cfg.SubjectTransform != nil && resp.StreamInfo.Config.SubjectTransform == nil {
		return nil, ErrStreamSubjectTransformNotSupported
	}

	if len(cfg.Sources) != 0 {
		if len(cfg.Sources) != len(resp.Config.Sources) {
			return nil, ErrStreamSourceNotSupported
		}

		// the sources list in the response is not ordered
		cfgNumTransforms := make([]int, len(cfg.Sources))
		respNumTransforms := make([]int, len(resp.Config.Sources))
		for i, cfgSource := range cfg.Sources {
			cfgNumTransforms[i] = len(cfgSource.SubjectTransforms)
			respNumTransforms[i] = len(resp.Config.Sources[i].SubjectTransforms)
		}
		slices.Sort(cfgNumTransforms)
		slices.Sort(respNumTransforms)
		if !slices.Equal(cfgNumTransforms, respNumTransforms) {
			return nil, ErrStreamSubjectTransformNotSupported
		}
	}

	return &stream{
		js:   js,
		name: cfg.Name,
		info: resp.StreamInfo,
	}, nil
}

// If we have a Domain, convert to the appropriate ext.APIPrefix.
// This will change the stream source, so should be a copy passed in.
func (ss *StreamSource) convertDomain() error {
	if ss.Domain == "" {
		return nil
	}
	if ss.External != nil {
		return errors.New("nats: domain and external are both set")
	}
	ss.External = &ExternalStream{APIPrefix: fmt.Sprintf(jsExtDomainT, ss.Domain)}
	return nil
}

// Helper for copying when we do not want to change user's version.
func (ss *StreamSource) copy() *StreamSource {
	nss := *ss
	// Check pointers
	if ss.OptStartTime != nil {
		t := *ss.OptStartTime
		nss.OptStartTime = &t
	}
	if ss.External != nil {
		ext := *ss.External
		nss.External = &ext
	}
	return &nss
}

// convertStreamConfigDomains converts domain configurations to external configurations
// in both mirror and sources of a StreamConfig. It creates a copy of the config to avoid
// modifying the caller's version.
func convertStreamConfigDomains(cfg StreamConfig) (StreamConfig, error) {
	ncfg := cfg
	// If we have a mirror and an external domain, convert to ext.APIPrefix.
	if ncfg.Mirror != nil && ncfg.Mirror.Domain != "" {
		// Copy so we do not change the caller's version.
		ncfg.Mirror = ncfg.Mirror.copy()
		if err := ncfg.Mirror.convertDomain(); err != nil {
			return StreamConfig{}, err
		}
	}

	// Check sources for the same conversion.
	if len(ncfg.Sources) > 0 {
		ncfg.Sources = append([]*StreamSource(nil), ncfg.Sources...)
		for i, ss := range ncfg.Sources {
			if ss.Domain != "" {
				ncfg.Sources[i] = ss.copy()
				if err := ncfg.Sources[i].convertDomain(); err != nil {
					return StreamConfig{}, err
				}
			}
		}
	}

	return ncfg, nil
}

// UpdateStream updates an existing stream. If stream does not exist,
// ErrStreamNotFound is returned.
func (js *jetStream) UpdateStream(ctx context.Context, cfg StreamConfig) (Stream, error) {
	if err := validateStreamName(cfg.Name); err != nil {
		return nil, err
	}
	ctx, cancel := js.wrapContextWithoutDeadline(ctx)
	if cancel != nil {
		defer cancel()
	}

	ncfg, err := convertStreamConfigDomains(cfg)
	if err != nil {
		return nil, err
	}

	req, err := json.Marshal(ncfg)
	if err != nil {
		return nil, err
	}

	updateSubject := fmt.Sprintf(apiStreamUpdateT, cfg.Name)
	var resp streamInfoResponse

	if _, err = js.apiRequestJSON(ctx, updateSubject, &resp, req); err != nil {
		return nil, err
	}
	if resp.Error != nil {
		if resp.Error.ErrorCode == JSErrCodeStreamNotFound {
			return nil, ErrStreamNotFound
		}
		return nil, resp.Error
	}
	if resp.StreamInfo == nil {
		return nil, ErrInvalidJetStreamResponse
	}

	// check that input subject transform (if used) is reflected in the returned StreamInfo
	if cfg.SubjectTransform != nil && resp.StreamInfo.Config.SubjectTransform == nil {
		return nil, ErrStreamSubjectTransformNotSupported
	}

	if len(cfg.Sources) != 0 {
		if len(cfg.Sources) != len(resp.Config.Sources) {
			return nil, ErrStreamSourceNotSupported
		}

		// the sources list in the response is not ordered
		cfgNumTransforms := make([]int, len(cfg.Sources))
		respNumTransforms := make([]int, len(resp.Config.Sources))
		for i, cfgSource := range cfg.Sources {
			cfgNumTransforms[i] = len(cfgSource.SubjectTransforms)
			respNumTransforms[i] = len(resp.Config.Sources[i].SubjectTransforms)
		}
		slices.Sort(cfgNumTransforms)
		slices.Sort(respNumTransforms)
		if !slices.Equal(cfgNumTransforms, respNumTransforms) {
			return nil, ErrStreamSubjectTransformNotSupported
		}
	}

	return &stream{
		js:   js,
		name: cfg.Name,
		info: resp.StreamInfo,
	}, nil
}

// CreateOrUpdateStream creates a stream with the given config. If stream
// already exists, it will be updated (if possible).
func (js *jetStream) CreateOrUpdateStream(ctx context.Context, cfg StreamConfig) (Stream, error) {
	s, err := js.UpdateStream(ctx, cfg)
	if err != nil {
		if !errors.Is(err, ErrStreamNotFound) {
			return nil, err
		}
		return js.CreateStream(ctx, cfg)
	}

	return s, nil
}

// Stream fetches [StreamInfo] and returns a [Stream] interface for a given stream name.
// If stream does not exist, ErrStreamNotFound is returned.
func (js *jetStream) Stream(ctx context.Context, name string) (Stream, error) {
	if err := validateStreamName(name); err != nil {
		return nil, err
	}
	ctx, cancel := js.wrapContextWithoutDeadline(ctx)
	if cancel != nil {
		defer cancel()
	}
	infoSubject := fmt.Sprintf(apiStreamInfoT, name)

	var resp streamInfoResponse

	if _, err := js.apiRequestJSON(ctx, infoSubject, &resp); err != nil {
		return nil, err
	}
	if resp.Error != nil {
		if resp.Error.ErrorCode == JSErrCodeStreamNotFound {
			return nil, ErrStreamNotFound
		}
		return nil, resp.Error
	}
	if resp.StreamInfo == nil {
		return nil, ErrInvalidJetStreamResponse
	}
	return &stream{
		js:   js,
		name: name,
		info: resp.StreamInfo,
	}, nil
}

// DeleteStream removes a stream with given name
func (js *jetStream) DeleteStream(ctx context.Context, name string) error {
	if err := validateStreamName(name); err != nil {
		return err
	}
	ctx, cancel := js.wrapContextWithoutDeadline(ctx)
	if cancel != nil {
		defer cancel()
	}
	deleteSubject := fmt.Sprintf(apiStreamDeleteT, name)
	var resp streamDeleteResponse

	if _, err := js.apiRequestJSON(ctx, deleteSubject, &resp); err != nil {
		return err
	}
	if resp.Error != nil {
		if resp.Error.ErrorCode == JSErrCodeStreamNotFound {
			return ErrStreamNotFound
		}
		return resp.Error
	}
	return nil
}

// CreateOrUpdateConsumer creates a consumer on a given stream with
// the given config. If consumer already exists, it will be updated (if
// possible). Consumer interface is returned, allowing to operate on a
// consumer (e.g. fetch messages).
func (js *jetStream) CreateOrUpdateConsumer(ctx context.Context, stream string, cfg ConsumerConfig) (Consumer, error) {
	if err := validateStreamName(stream); err != nil {
		return nil, err
	}
	return upsertPullConsumer(ctx, js, stream, cfg, consumerActionCreateOrUpdate)
}

// CreateConsumer creates a consumer on a given stream with given
// config. If consumer already exists and the provided configuration
// differs from its configuration, ErrConsumerExists is returned. If the
// provided configuration is the same as the existing consumer, the
// existing consumer is returned. Consumer interface is returned,
// allowing to operate on a consumer (e.g. fetch messages).
func (js *jetStream) CreateConsumer(ctx context.Context, stream string, cfg ConsumerConfig) (Consumer, error) {
	if err := validateStreamName(stream); err != nil {
		return nil, err
	}
	return upsertPullConsumer(ctx, js, stream, cfg, consumerActionCreate)
}

// UpdateConsumer updates an existing consumer. If consumer does not
// exist, ErrConsumerDoesNotExist is returned. Consumer interface is
// returned, allowing to operate on a consumer (e.g. fetch messages).
func (js *jetStream) UpdateConsumer(ctx context.Context, stream string, cfg ConsumerConfig) (Consumer, error) {
	if err := validateStreamName(stream); err != nil {
		return nil, err
	}
	return upsertPullConsumer(ctx, js, stream, cfg, consumerActionUpdate)
}

// OrderedConsumer returns an OrderedConsumer instance. OrderedConsumer
// are managed by the library and provide a simple way to consume
// messages from a stream. Ordered consumers are ephemeral in-memory
// pull consumers and are resilient to deletes and restarts.
func (js *jetStream) OrderedConsumer(ctx context.Context, stream string, cfg OrderedConsumerConfig) (Consumer, error) {
	if err := validateStreamName(stream); err != nil {
		return nil, err
	}
	namePrefix := cfg.NamePrefix
	if namePrefix == "" {
		namePrefix = nuid.Next()
	}
	oc := &orderedConsumer{
		js:         js,
		cfg:        &cfg,
		stream:     stream,
		namePrefix: namePrefix,
		doReset:    make(chan struct{}, 1),
	}
	consCfg := oc.getConsumerConfig()
	cons, err := js.CreateOrUpdateConsumer(ctx, stream, *consCfg)
	if err != nil {
		return nil, err
	}
	oc.currentConsumer = cons.(*pullConsumer)

	return oc, nil
}

// Consumer returns an interface to an existing consumer, allowing processing
// of messages. If consumer does not exist, ErrConsumerNotFound is
// returned.
func (js *jetStream) Consumer(ctx context.Context, stream string, name string) (Consumer, error) {
	if err := validateStreamName(stream); err != nil {
		return nil, err
	}
	return getConsumer(ctx, js, stream, name)
}

// DeleteConsumer removes a consumer with given name from a stream.
// If consumer does not exist, ErrConsumerNotFound is returned.
func (js *jetStream) DeleteConsumer(ctx context.Context, stream string, name string) error {
	if err := validateStreamName(stream); err != nil {
		return err
	}
	return deleteConsumer(ctx, js, stream, name)
}

// CreateOrUpdatePushConsumer creates a push consumer on a given stream with
// the given config. If consumer already exists, it will be updated (if
// possible). Consumer interface is returned, allowing to consume messages.
func (js *jetStream) CreateOrUpdatePushConsumer(ctx context.Context, stream string, cfg ConsumerConfig) (PushConsumer, error) {
	if err := validateStreamName(stream); err != nil {
		return nil, err
	}
	return upsertPushConsumer(ctx, js, stream, cfg, consumerActionCreateOrUpdate)
}

// CreatePushConsumer creates a push consumer on a given stream with given
// config. If consumer already exists and the provided configuration
// differs from its configuration, ErrConsumerExists is returned. If the
// provided configuration is the same as the existing consumer, the
// existing consumer is returned. Consumer interface is returned,
// allowing to consume messages.
func (js *jetStream) CreatePushConsumer(ctx context.Context, stream string, cfg ConsumerConfig) (PushConsumer, error) {
	if err := validateStreamName(stream); err != nil {
		return nil, err
	}
	return upsertPushConsumer(ctx, js, stream, cfg, consumerActionCreate)
}

// UpdatePushConsumer updates an existing push consumer. If consumer does not
// exist, ErrConsumerDoesNotExist is returned. Consumer interface is
// returned, allowing to consume messages.
func (js *jetStream) UpdatePushConsumer(ctx context.Context, stream string, cfg ConsumerConfig) (PushConsumer, error) {
	if err := validateStreamName(stream); err != nil {
		return nil, err
	}
	return upsertPushConsumer(ctx, js, stream, cfg, consumerActionUpdate)
}

// PushConsumer returns an interface to an existing consumer, allowing processing
// of messages. If consumer does not exist, ErrConsumerNotFound is
// returned.
func (js *jetStream) PushConsumer(ctx context.Context, stream string, name string) (PushConsumer, error) {
	if err := validateStreamName(stream); err != nil {
		return nil, err
	}
	return getPushConsumer(ctx, js, stream, name)
}

func (js *jetStream) PauseConsumer(ctx context.Context, stream string, consumer string, pauseUntil time.Time) (*ConsumerPauseResponse, error) {
	if err := validateStreamName(stream); err != nil {
		return nil, err
	}
	return pauseConsumer(ctx, js, stream, consumer, &pauseUntil)
}

func (js *jetStream) ResumeConsumer(ctx context.Context, stream string, consumer string) (*ConsumerPauseResponse, error) {
	if err := validateStreamName(stream); err != nil {
		return nil, err
	}
	return resumeConsumer(ctx, js, stream, consumer)
}

func (js *jetStream) ResetConsumer(ctx context.Context, stream, consumer string) (*ConsumerResetResponse, error) {
	if err := validateStreamName(stream); err != nil {
		return nil, err
	}
	return resetConsumer(ctx, js, stream, consumer, 0)
}

func (js *jetStream) ResetConsumerToSequence(ctx context.Context, stream, consumer string, seq uint64) (*ConsumerResetResponse, error) {
	if err := validateStreamName(stream); err != nil {
		return nil, err
	}
	return resetConsumer(ctx, js, stream, consumer, seq)
}

func validateStreamName(stream string) error {
	if stream == "" {
		return ErrStreamNameRequired
	}
	if strings.ContainsAny(stream, ">*. /\\\t\r\n") {
		return fmt.Errorf("%w: %q", ErrInvalidStreamName, stream)
	}
	return nil
}

func validateSubject(subject string) error {
	if subject == "" {
		return fmt.Errorf("%w: %s", ErrInvalidSubject, "subject cannot be empty")
	}
	if subject[0] == '.' || subject[len(subject)-1] == '.' || !subjectRegexp.MatchString(subject) {
		return fmt.Errorf("%w: %s", ErrInvalidSubject, subject)
	}
	return nil
}

// AccountInfo fetches account information from the server, containing details
// about the account associated with this JetStream connection. If account is
// not enabled for JetStream, ErrJetStreamNotEnabledForAccount is returned.
//
// If the server does not have JetStream enabled, ErrJetStreamNotEnabled is
// returned (for a single server setup). For clustered topologies, AccountInfo
// will time out.
func (js *jetStream) AccountInfo(ctx context.Context) (*AccountInfo, error) {
	ctx, cancel := js.wrapContextWithoutDeadline(ctx)
	if cancel != nil {
		defer cancel()
	}
	var resp accountInfoResponse

	if _, err := js.apiRequestJSON(ctx, apiAccountInfo, &resp); err != nil {
		if errors.Is(err, nats.ErrNoResponders) {
			return nil, ErrJetStreamNotEnabled
		}
		return nil, err
	}
	if resp.Error != nil {
		if resp.Error.ErrorCode == JSErrCodeJetStreamNotEnabledForAccount {
			return nil, ErrJetStreamNotEnabledForAccount
		}
		if resp.Error.ErrorCode == JSErrCodeJetStreamNotEnabled {
			return nil, ErrJetStreamNotEnabled
		}
		return nil, resp.Error
	}

	return &resp.AccountInfo, nil
}

// ListStreams returns StreamInfoLister, enabling iteration over a
// channel of stream infos.
func (js *jetStream) ListStreams(ctx context.Context, opts ...StreamListOpt) StreamInfoLister {
	l := &streamLister{
		js:      js,
		streams: make(chan *StreamInfo),
	}
	var streamsReq streamsRequest
	for _, opt := range opts {
		if err := opt(&streamsReq); err != nil {
			l.err = err
			close(l.streams)
			return l
		}
	}
	go func() {
		defer close(l.streams)
		ctx, cancel := js.wrapContextWithoutDeadline(ctx)
		if cancel != nil {
			defer cancel()
		}
		for {
			page, err := l.streamInfos(ctx, streamsReq)
			if err != nil && !errors.Is(err, ErrEndOfData) {
				l.err = err
				return
			}
			for _, info := range page {
				select {
				case l.streams <- info:
				case <-ctx.Done():
					l.err = ctx.Err()
					return
				}
			}
			if errors.Is(err, ErrEndOfData) {
				return
			}
		}
	}()

	return l
}

// Info returns a channel allowing retrieval of stream infos returned by [ListStreams]
func (s *streamLister) Info() <-chan *StreamInfo {
	return s.streams
}

// Err returns an error channel which will be populated with error from [ListStreams] or [StreamNames] request
func (s *streamLister) Err() error {
	return s.err
}

// StreamNames returns a StreamNameLister, enabling iteration over a
// channel of stream names.
func (js *jetStream) StreamNames(ctx context.Context, opts ...StreamListOpt) StreamNameLister {
	l := &streamLister{
		js:    js,
		names: make(chan string),
	}
	var streamsReq streamsRequest
	for _, opt := range opts {
		if err := opt(&streamsReq); err != nil {
			l.err = err
			close(l.names)
			return l
		}
	}
	go func() {
		ctx, cancel := js.wrapContextWithoutDeadline(ctx)
		if cancel != nil {
			defer cancel()
		}
		defer close(l.names)
		for {
			page, err := l.streamNames(ctx, streamsReq)
			if err != nil && !errors.Is(err, ErrEndOfData) {
				l.err = err
				return
			}
			for _, info := range page {
				select {
				case l.names <- info:
				case <-ctx.Done():
					l.err = ctx.Err()
					return
				}
			}
			if errors.Is(err, ErrEndOfData) {
				return
			}
		}
	}()

	return l
}

// StreamNameBySubject returns the name of the stream bound to the given
// subject. If no stream is bound to given subject, ErrStreamNotFound
// is returned.
func (js *jetStream) StreamNameBySubject(ctx context.Context, subject string) (string, error) {
	ctx, cancel := js.wrapContextWithoutDeadline(ctx)
	if cancel != nil {
		defer cancel()
	}
	if err := validateSubject(subject); err != nil {
		return "", err
	}

	r := &streamsRequest{Subject: subject}
	req, err := json.Marshal(r)
	if err != nil {
		return "", err
	}
	var resp streamNamesResponse
	_, err = js.apiRequestJSON(ctx, apiStreams, &resp, req)
	if err != nil {
		return "", err
	}
	if resp.Error != nil {
		return "", resp.Error
	}
	if len(resp.Streams) == 0 {
		return "", ErrStreamNotFound
	}

	return resp.Streams[0], nil
}

// Name returns a channel allowing retrieval of stream names returned by [StreamNames]
func (s *streamLister) Name() <-chan string {
	return s.names
}

// infos fetches the next [StreamInfo] page
func (s *streamLister) streamInfos(ctx context.Context, streamsReq streamsRequest) ([]*StreamInfo, error) {
	if s.pageInfo != nil && s.offset >= s.pageInfo.Total {
		return nil, ErrEndOfData
	}

	req := streamsRequest{
		apiPagedRequest: apiPagedRequest{
			Offset: s.offset,
		},
		Subject: streamsReq.Subject,
	}
	reqJSON, err := json.Marshal(req)
	if err != nil {
		return nil, err
	}

	var resp streamListResponse
	_, err = s.js.apiRequestJSON(ctx, apiStreamListT, &resp, reqJSON)
	if err != nil {
		return nil, err
	}
	if resp.Error != nil {
		return nil, resp.Error
	}

	s.pageInfo = &resp.apiPaged
	s.offset += len(resp.Streams)
	return resp.Streams, nil
}

// streamNames fetches the next stream names page
func (s *streamLister) streamNames(ctx context.Context, streamsReq streamsRequest) ([]string, error) {
	if s.pageInfo != nil && s.offset >= s.pageInfo.Total {
		return nil, ErrEndOfData
	}

	req := streamsRequest{
		apiPagedRequest: apiPagedRequest{
			Offset: s.offset,
		},
		Subject: streamsReq.Subject,
	}
	reqJSON, err := json.Marshal(req)
	if err != nil {
		return nil, err
	}

	var resp streamNamesResponse
	_, err = s.js.apiRequestJSON(ctx, apiStreams, &resp, reqJSON)
	if err != nil {
		return nil, err
	}
	if resp.Error != nil {
		return nil, resp.Error
	}

	s.pageInfo = &resp.apiPaged
	s.offset += len(resp.Streams)
	return resp.Streams, nil
}

// wrapContextWithoutDeadline wraps context without deadline with default timeout.
// If deadline is already set, it will be returned as is, and cancel() will be nil.
// Caller should check if cancel() is nil before calling it.
func (js *jetStream) wrapContextWithoutDeadline(ctx context.Context) (context.Context, context.CancelFunc) {
	if _, ok := ctx.Deadline(); ok {
		return ctx, nil
	}
	return context.WithTimeout(ctx, js.opts.DefaultTimeout)
}

// CleanupPublisher will cleanup the publishing side of JetStreamContext.
//
// This will unsubscribe from the internal reply subject if needed.
// All pending async publishes will fail with ErrJetStreamContextClosed.
//
// If an error handler was provided, it will be called for each pending async
// publish and PublishAsyncComplete will be closed.
//
// After completing JetStreamContext is still usable - internal subscription
// will be recreated on next publish, but the acks from previous publishes will
// be lost.
func (js *jetStream) CleanupPublisher() {
	js.cleanupReplySub()
	js.publisher.Lock()
	errCb := js.publisher.aecb
	for id, paf := range js.publisher.acks {
		paf.err = ErrJetStreamPublisherClosed
		if paf.errCh != nil {
			paf.errCh <- paf.err
		}
		if errCb != nil {
			// call error handler after releasing the mutex to avoid contention
			defer errCb(js, paf.msg, ErrJetStreamPublisherClosed)
		}
		delete(js.publisher.acks, id)
	}
	if js.publisher.doneCh != nil {
		close(js.publisher.doneCh)
		js.publisher.doneCh = nil
	}
	js.publisher.Unlock()
}

func (js *jetStream) cleanupReplySub() {
	if js.publisher == nil {
		return
	}
	js.publisher.Lock()
	if js.publisher.replySub != nil {
		js.publisher.replySub.Unsubscribe()
		js.publisher.replySub = nil
	}
	if js.publisher.connStatusCh != nil {
		js.conn.RemoveStatusListener(js.publisher.connStatusCh)
		js.publisher.connStatusCh = nil
	}
	js.publisher.Unlock()
}
