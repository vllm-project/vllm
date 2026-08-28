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
	"strconv"
	"time"

	"github.com/nats-io/nats.go"
	"github.com/nats-io/nuid"
)

type (
	// Stream contains CRUD methods on a consumer via [ConsumerManager], as well
	// as operations on an existing stream. It allows fetching and removing
	// messages from a stream, as well as purging a stream.
	Stream interface {
		ConsumerManager

		// Info returns StreamInfo from the server.
		Info(ctx context.Context, opts ...StreamInfoOpt) (*StreamInfo, error)

		// CachedInfo returns ConsumerInfo currently cached on this stream.
		// This method does not perform any network requests. The cached
		// StreamInfo is updated on every call to Info and Update.
		CachedInfo() *StreamInfo

		// Purge removes messages from a stream. It is a destructive operation.
		// Use with caution. See StreamPurgeOpt for available options.
		Purge(ctx context.Context, opts ...StreamPurgeOpt) error

		// GetMsg retrieves a raw stream message stored in JetStream by sequence number.
		GetMsg(ctx context.Context, seq uint64, opts ...GetMsgOpt) (*RawStreamMsg, error)

		// GetLastMsgForSubject retrieves the last raw stream message stored in
		// JetStream on a given subject.
		GetLastMsgForSubject(ctx context.Context, subject string) (*RawStreamMsg, error)

		// DeleteMsg deletes a message from a stream.
		// On the server, the message is marked as erased, but not overwritten.
		DeleteMsg(ctx context.Context, seq uint64) error

		// SecureDeleteMsg deletes a message from a stream. The deleted message
		// is overwritten with random data. As a result, this operation is slower
		// than DeleteMsg.
		SecureDeleteMsg(ctx context.Context, seq uint64) error
	}

	// ConsumerManager provides CRUD API for managing consumers. It is
	// available as a part of [Stream] interface. CreateConsumer,
	// UpdateConsumer, CreateOrUpdateConsumer and Consumer methods return a
	// [Consumer] interface, allowing to operate on a consumer (e.g. consume
	// messages).
	ConsumerManager interface {
		// CreateOrUpdateConsumer creates a pull consumer on a given stream with
		// given config. If consumer already exists, it will be updated (if
		// possible). Consumer interface is returned, allowing to operate on a
		// consumer (e.g. fetch messages).
		CreateOrUpdateConsumer(ctx context.Context, cfg ConsumerConfig) (Consumer, error)

		// CreateConsumer creates a pull consumer on a given stream with given
		// config. If consumer already exists and the provided configuration
		// differs from its configuration, ErrConsumerExists is returned. If the
		// provided configuration is the same as the existing consumer, the
		// existing consumer is returned. Consumer interface is returned,
		// allowing to operate on a consumer (e.g. fetch messages).
		CreateConsumer(ctx context.Context, cfg ConsumerConfig) (Consumer, error)

		// UpdateConsumer updates an existing pull consumer. If consumer does not
		// exist, ErrConsumerDoesNotExist is returned. Consumer interface is
		// returned, allowing to operate on a consumer (e.g. fetch messages).
		UpdateConsumer(ctx context.Context, cfg ConsumerConfig) (Consumer, error)

		// OrderedConsumer returns an OrderedConsumer instance. OrderedConsumer
		// are managed by the library and provide a simple way to consume
		// messages from a stream. Ordered consumers are ephemeral in-memory
		// pull consumers and are resilient to deletes and restarts.
		OrderedConsumer(ctx context.Context, cfg OrderedConsumerConfig) (Consumer, error)

		// Consumer returns an interface to an existing consumer, allowing processing
		// of messages. If consumer does not exist, ErrConsumerNotFound is
		// returned.
		//
		// It returns ErrNotPullConsumer if the consumer is not a pull consumer (deliver subject is not set).
		Consumer(ctx context.Context, consumer string) (Consumer, error)

		// DeleteConsumer removes a consumer with given name from a stream.
		// If consumer does not exist, ErrConsumerNotFound is returned.
		DeleteConsumer(ctx context.Context, consumer string) error

		// PauseConsumer pauses a consumer.
		PauseConsumer(ctx context.Context, consumer string, pauseUntil time.Time) (*ConsumerPauseResponse, error)

		// ResumeConsumer resumes a consumer.
		ResumeConsumer(ctx context.Context, consumer string) (*ConsumerPauseResponse, error)

		// ListConsumers returns ConsumerInfoLister enabling iterating over a
		// channel of consumer infos.
		ListConsumers(context.Context) ConsumerInfoLister

		// ConsumerNames returns a ConsumerNameLister enabling iterating over a
		// channel of consumer names.
		ConsumerNames(context.Context) ConsumerNameLister

		// UnpinConsumer unpins the currently pinned client for a consumer for the given group name.
		// If consumer does not exist, ErrConsumerNotFound is returned.
		UnpinConsumer(ctx context.Context, consumer string, group string) error

		// ResetConsumer resets a consumer's delivery state without deleting
		// and recreating it. The consumer is reset to deliver from
		// ack_floor + 1.
		ResetConsumer(ctx context.Context, consumer string) (*ConsumerResetResponse, error)

		// ResetConsumerToSequence resets a consumer's delivery state to the
		// given stream sequence. The seq must be compatible with the
		// consumer's DeliverPolicy (e.g. for DeliverByStartSequencePolicy,
		// seq must be >= OptStartSeq). If incompatible,
		// ErrConsumerInvalidReset is returned.
		ResetConsumerToSequence(ctx context.Context, consumer string, seq uint64) (*ConsumerResetResponse, error)

		// CreateOrUpdatePushConsumer creates a push consumer on a given stream with
		// given config. If consumer already exists, it will be updated (if
		// possible). Consumer interface is returned, allowing to consume messages.
		CreateOrUpdatePushConsumer(ctx context.Context, cfg ConsumerConfig) (PushConsumer, error)

		// CreatePushConsumer creates a push consumer on a given stream with given
		// config. If consumer already exists and the provided configuration
		// differs from its configuration, ErrConsumerExists is returned. If the
		// provided configuration is the same as the existing consumer, the
		// existing consumer is returned. Consumer interface is returned,
		// allowing to consume messages.
		CreatePushConsumer(ctx context.Context, cfg ConsumerConfig) (PushConsumer, error)

		// UpdatePushConsumer updates an existing push consumer. If consumer does not
		// exist, ErrConsumerDoesNotExist is returned. Consumer interface is
		// returned, allowing to consume messages.
		UpdatePushConsumer(ctx context.Context, cfg ConsumerConfig) (PushConsumer, error)

		// PushConsumer returns an interface to an existing push consumer, allowing processing
		// of messages. If consumer does not exist, ErrConsumerNotFound is
		// returned.
		//
		// It returns ErrNotPushConsumer if the consumer is not a push consumer (deliver subject is not set).
		PushConsumer(ctx context.Context, consumer string) (PushConsumer, error)
	}

	RawStreamMsg struct {
		Subject  string
		Sequence uint64
		Header   nats.Header
		Data     []byte
		Time     time.Time
	}

	stream struct {
		name string
		info *StreamInfo
		js   *jetStream
	}

	// StreamInfoOpt is a function setting options for [Stream.Info]
	StreamInfoOpt func(*streamInfoRequest) error

	streamInfoRequest struct {
		apiPagedRequest
		DeletedDetails bool   `json:"deleted_details,omitempty"`
		SubjectFilter  string `json:"subjects_filter,omitempty"`
	}

	consumerInfoResponse struct {
		apiResponse
		*ConsumerInfo
	}

	// StreamPurgeOpt is a function setting options for [Stream.Purge]
	StreamPurgeOpt func(*StreamPurgeRequest) error

	// StreamPurgeRequest is an API request body to purge a stream.

	StreamPurgeRequest struct {
		// Purge up to but not including sequence.
		Sequence uint64 `json:"seq,omitempty"`
		// Subject to match against messages for the purge command.
		Subject string `json:"filter,omitempty"`
		// Number of messages to keep.
		Keep uint64 `json:"keep,omitempty"`
	}

	streamPurgeResponse struct {
		apiResponse
		Success bool   `json:"success,omitempty"`
		Purged  uint64 `json:"purged"`
	}

	consumerDeleteResponse struct {
		apiResponse
		Success bool `json:"success,omitempty"`
	}

	consumerPauseRequest struct {
		PauseUntil *time.Time `json:"pause_until,omitempty"`
	}

	ConsumerPauseResponse struct {
		// Paused is true if the consumer is paused.
		Paused bool `json:"paused"`
		// PauseUntil is the time until the consumer is paused.
		PauseUntil time.Time `json:"pause_until"`
		// PauseRemaining is the time remaining until the consumer is paused.
		PauseRemaining time.Duration `json:"pause_remaining,omitempty"`
	}

	consumerPauseApiResponse struct {
		apiResponse
		ConsumerPauseResponse
	}

	consumerResetRequest struct {
		Seq uint64 `json:"seq,omitempty"`
	}

	// ConsumerResetResponse contains the result of a consumer reset operation.
	// It carries the updated ConsumerInfo together with the stream sequence the
	// server actually reset the consumer to.
	//
	// The embedded *ConsumerInfo is guaranteed non-nil on a successful response;
	// the helper returns ErrConsumerResetResponseEmpty if the server omits it.
	ConsumerResetResponse struct {
		*ConsumerInfo
		// ResetSeq is the stream sequence the consumer was reset to. When
		// ResetConsumer is called without an explicit sequence, this is the
		// server-resolved value (typically ack_floor + 1).
		ResetSeq uint64 `json:"reset_seq"`
	}

	consumerResetApiResponse struct {
		apiResponse
		ConsumerResetResponse
	}

	// GetMsgOpt is a function setting options for [Stream.GetMsg]
	GetMsgOpt func(*apiMsgGetRequest) error

	apiMsgGetRequest struct {
		Seq     uint64 `json:"seq,omitempty"`
		LastFor string `json:"last_by_subj,omitempty"`
		NextFor string `json:"next_by_subj,omitempty"`
	}

	// apiMsgGetResponse is the response for a Stream get request.
	apiMsgGetResponse struct {
		apiResponse
		Message *storedMsg `json:"message,omitempty"`
	}

	// storedMsg is a raw message stored in JetStream.
	storedMsg struct {
		Subject  string    `json:"subject"`
		Sequence uint64    `json:"seq"`
		Header   []byte    `json:"hdrs,omitempty"`
		Data     []byte    `json:"data,omitempty"`
		Time     time.Time `json:"time"`
	}

	msgDeleteRequest struct {
		Seq     uint64 `json:"seq"`
		NoErase bool   `json:"no_erase,omitempty"`
	}

	msgDeleteResponse struct {
		apiResponse
		Success bool `json:"success,omitempty"`
	}

	// ConsumerInfoLister is used to iterate over a channel of consumer infos.
	// Err method can be used to check for errors encountered during iteration.
	// Info channel is always closed and therefore can be used in a range loop.
	ConsumerInfoLister interface {
		Info() <-chan *ConsumerInfo
		Err() error
	}

	// ConsumerNameLister is used to iterate over a channel of consumer names.
	// Err method can be used to check for errors encountered during iteration.
	// Name channel is always closed and therefore can be used in a range loop.
	ConsumerNameLister interface {
		Name() <-chan string
		Err() error
	}

	consumerLister struct {
		js       *jetStream
		offset   int
		pageInfo *apiPaged

		consumers chan *ConsumerInfo
		names     chan string
		err       error
	}

	consumerListResponse struct {
		apiResponse
		apiPaged
		Consumers []*ConsumerInfo `json:"consumers"`
	}

	consumerNamesResponse struct {
		apiResponse
		apiPaged
		Consumers []string `json:"consumers"`
	}

	consumerUnpinRequest struct {
		Group string `json:"group"`
	}
)

// CreateOrUpdateConsumer creates a consumer on a given stream with
// given config. If consumer already exists, it will be updated (if
// possible). Consumer interface is returned, allowing to operate on a
// consumer (e.g. fetch messages).
func (s *stream) CreateOrUpdateConsumer(ctx context.Context, cfg ConsumerConfig) (Consumer, error) {
	return upsertPullConsumer(ctx, s.js, s.name, cfg, consumerActionCreateOrUpdate)
}

// CreateConsumer creates a consumer on a given stream with given
// config. If consumer already exists and the provided configuration
// differs from its configuration, ErrConsumerExists is returned. If the
// provided configuration is the same as the existing consumer, the
// existing consumer is returned. Consumer interface is returned,
// allowing to operate on a consumer (e.g. fetch messages).
func (s *stream) CreateConsumer(ctx context.Context, cfg ConsumerConfig) (Consumer, error) {
	return upsertPullConsumer(ctx, s.js, s.name, cfg, consumerActionCreate)
}

// UpdateConsumer updates an existing consumer. If consumer does not
// exist, ErrConsumerDoesNotExist is returned. Consumer interface is
// returned, allowing to operate on a consumer (e.g. fetch messages).
func (s *stream) UpdateConsumer(ctx context.Context, cfg ConsumerConfig) (Consumer, error) {
	return upsertPullConsumer(ctx, s.js, s.name, cfg, consumerActionUpdate)
}

// CreateOrUpdatePushConsumer creates a consumer on a given stream with
// given config. If consumer already exists, it will be updated (if
// possible). Consumer interface is returned, allowing to consume messages.
func (s *stream) CreateOrUpdatePushConsumer(ctx context.Context, cfg ConsumerConfig) (PushConsumer, error) {
	return upsertPushConsumer(ctx, s.js, s.name, cfg, consumerActionCreateOrUpdate)
}

// CreatePushConsumer creates a consumer on a given stream with given
// config. If consumer already exists and the provided configuration
// differs from its configuration, ErrConsumerExists is returned. If the
// provided configuration is the same as the existing consumer, the
// existing consumer is returned. Consumer interface is returned,
// allowing to consume messages.
func (s *stream) CreatePushConsumer(ctx context.Context, cfg ConsumerConfig) (PushConsumer, error) {
	return upsertPushConsumer(ctx, s.js, s.name, cfg, consumerActionCreate)
}

// UpdatePushConsumer updates an existing consumer. If consumer does not
// exist, ErrConsumerDoesNotExist is returned. Consumer interface is
// returned, allowing to consume messages.
func (s *stream) UpdatePushConsumer(ctx context.Context, cfg ConsumerConfig) (PushConsumer, error) {
	return upsertPushConsumer(ctx, s.js, s.name, cfg, consumerActionUpdate)
}

// OrderedConsumer returns an OrderedConsumer instance. OrderedConsumer
// are managed by the library and provide a simple way to consume
// messages from a stream. Ordered consumers are ephemeral in-memory
// pull consumers and are resilient to deletes and restarts.
func (s *stream) OrderedConsumer(ctx context.Context, cfg OrderedConsumerConfig) (Consumer, error) {
	namePrefix := cfg.NamePrefix
	if namePrefix == "" {
		namePrefix = nuid.Next()
	}
	oc := &orderedConsumer{
		js:         s.js,
		cfg:        &cfg,
		stream:     s.name,
		namePrefix: namePrefix,
		doReset:    make(chan struct{}, 1),
	}
	consCfg := oc.getConsumerConfig()
	cons, err := s.CreateOrUpdateConsumer(ctx, *consCfg)
	if err != nil {
		return nil, err
	}
	oc.currentConsumer = cons.(*pullConsumer)

	return oc, nil
}

// Consumer returns an interface to an existing consumer, allowing processing
// of messages. If consumer does not exist, ErrConsumerNotFound is
// returned.
func (s *stream) Consumer(ctx context.Context, name string) (Consumer, error) {
	return getConsumer(ctx, s.js, s.name, name)
}

func (s *stream) PushConsumer(ctx context.Context, name string) (PushConsumer, error) {
	return getPushConsumer(ctx, s.js, s.name, name)
}

// DeleteConsumer removes a consumer with given name from a stream.
// If consumer does not exist, ErrConsumerNotFound is returned.
func (s *stream) DeleteConsumer(ctx context.Context, name string) error {
	return deleteConsumer(ctx, s.js, s.name, name)
}

// PauseConsumer pauses a consumer.
func (s *stream) PauseConsumer(ctx context.Context, name string, pauseUntil time.Time) (*ConsumerPauseResponse, error) {
	return pauseConsumer(ctx, s.js, s.name, name, &pauseUntil)
}

// ResumeConsumer resumes a consumer.
func (s *stream) ResumeConsumer(ctx context.Context, name string) (*ConsumerPauseResponse, error) {
	return resumeConsumer(ctx, s.js, s.name, name)
}

// Info returns StreamInfo from the server.
func (s *stream) Info(ctx context.Context, opts ...StreamInfoOpt) (*StreamInfo, error) {
	ctx, cancel := s.js.wrapContextWithoutDeadline(ctx)
	if cancel != nil {
		defer cancel()
	}
	var infoReq *streamInfoRequest
	for _, opt := range opts {
		if infoReq == nil {
			infoReq = &streamInfoRequest{}
		}
		if err := opt(infoReq); err != nil {
			return nil, err
		}
	}
	var req []byte
	var err error
	var subjectMap map[string]uint64
	var offset int

	infoSubject := fmt.Sprintf(apiStreamInfoT, s.name)
	var info *StreamInfo
	for {
		if infoReq != nil {
			if infoReq.SubjectFilter != "" {
				if subjectMap == nil {
					subjectMap = make(map[string]uint64)
				}
				infoReq.Offset = offset
			}
			req, err = json.Marshal(infoReq)
			if err != nil {
				return nil, err
			}
		}
		var resp streamInfoResponse
		if _, err = s.js.apiRequestJSON(ctx, infoSubject, &resp, req); err != nil {
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
		info = resp.StreamInfo
		var total int
		if resp.Total != 0 {
			total = resp.Total
		}
		if len(resp.StreamInfo.State.Subjects) > 0 {
			for subj, msgs := range resp.StreamInfo.State.Subjects {
				subjectMap[subj] = msgs
			}
			offset = len(subjectMap)
		}
		if total == 0 || total <= offset {
			info.State.Subjects = nil
			// we don't want to store subjects in cache
			cached := *info
			s.info = &cached
			info.State.Subjects = subjectMap
			break
		}
	}

	return info, nil
}

// CachedInfo returns ConsumerInfo currently cached on this stream.
// This method does not perform any network requests. The cached
// StreamInfo is updated on every call to Info and Update.
func (s *stream) CachedInfo() *StreamInfo {
	return s.info
}

// Purge removes messages from a stream. It is a destructive operation.
// Use with caution. See StreamPurgeOpt for available options.
func (s *stream) Purge(ctx context.Context, opts ...StreamPurgeOpt) error {
	ctx, cancel := s.js.wrapContextWithoutDeadline(ctx)
	if cancel != nil {
		defer cancel()
	}
	var purgeReq StreamPurgeRequest
	for _, opt := range opts {
		if err := opt(&purgeReq); err != nil {
			return err
		}
	}
	var req []byte
	var err error
	req, err = json.Marshal(purgeReq)
	if err != nil {
		return err
	}

	purgeSubject := fmt.Sprintf(apiStreamPurgeT, s.name)

	var resp streamPurgeResponse
	if _, err = s.js.apiRequestJSON(ctx, purgeSubject, &resp, req); err != nil {
		return err
	}
	if resp.Error != nil {
		return resp.Error
	}

	return nil
}

// GetMsg retrieves a raw stream message stored in JetStream by sequence number.
func (s *stream) GetMsg(ctx context.Context, seq uint64, opts ...GetMsgOpt) (*RawStreamMsg, error) {
	req := &apiMsgGetRequest{Seq: seq}
	for _, opt := range opts {
		if err := opt(req); err != nil {
			return nil, err
		}
	}
	return s.getMsg(ctx, req)
}

// GetLastMsgForSubject retrieves the last raw stream message stored in
// JetStream on a given subject subject.
func (s *stream) GetLastMsgForSubject(ctx context.Context, subject string) (*RawStreamMsg, error) {
	return s.getMsg(ctx, &apiMsgGetRequest{LastFor: subject})
}

func (s *stream) getMsg(ctx context.Context, mreq *apiMsgGetRequest) (*RawStreamMsg, error) {
	ctx, cancel := s.js.wrapContextWithoutDeadline(ctx)
	if cancel != nil {
		defer cancel()
	}
	req, err := json.Marshal(mreq)
	if err != nil {
		return nil, err
	}

	var gmSubj string

	// handle direct gets
	if s.info.Config.AllowDirect {
		if mreq.LastFor != "" {
			gmSubj = fmt.Sprintf(apiDirectMsgGetLastBySubjectT, s.name, mreq.LastFor)
			r, err := s.js.apiRequest(ctx, gmSubj, nil)
			if err != nil {
				return nil, err
			}
			return convertDirectGetMsgResponseToMsg(r.msg)
		}
		gmSubj = fmt.Sprintf(apiDirectMsgGetT, s.name)
		r, err := s.js.apiRequest(ctx, gmSubj, req)
		if err != nil {
			return nil, err
		}
		return convertDirectGetMsgResponseToMsg(r.msg)
	}

	var resp apiMsgGetResponse
	dsSubj := fmt.Sprintf(apiMsgGetT, s.name)
	_, err = s.js.apiRequestJSON(ctx, dsSubj, &resp, req)
	if err != nil {
		return nil, err
	}

	if resp.Error != nil {
		if resp.Error.ErrorCode == JSErrCodeMessageNotFound {
			return nil, ErrMsgNotFound
		}
		return nil, resp.Error
	}
	if resp.Message == nil {
		return nil, ErrInvalidJetStreamResponse
	}

	msg := resp.Message

	var hdr nats.Header
	if len(msg.Header) > 0 {
		hdr, err = nats.DecodeHeadersMsg(msg.Header)
		if err != nil {
			return nil, err
		}
	}

	return &RawStreamMsg{
		Subject:  msg.Subject,
		Sequence: msg.Sequence,
		Header:   hdr,
		Data:     msg.Data,
		Time:     msg.Time,
	}, nil
}

func convertDirectGetMsgResponseToMsg(r *nats.Msg) (*RawStreamMsg, error) {
	// Check for 404/408. We would get a no-payload message and a "Status" header
	if len(r.Data) == 0 {
		val := r.Header.Get(statusHdr)
		if val != "" {
			switch val {
			case statusNoMsgs:
				return nil, ErrMsgNotFound
			default:
				desc := r.Header.Get("Description")
				if desc == "" {
					desc = "unable to get message"
				}
				return nil, fmt.Errorf("nats: %s", desc)
			}
		}
	}

	// Check for headers that give us the required information to
	// reconstruct the message.
	if len(r.Header) == 0 {
		return nil, errors.New("nats: response should have headers")
	}
	stream := r.Header.Get(StreamHeader)
	if stream == "" {
		return nil, errors.New("nats: missing stream header")
	}

	seqStr := r.Header.Get(SequenceHeader)
	if seqStr == "" {
		return nil, errors.New("nats: missing sequence header")
	}
	seq, err := strconv.ParseUint(seqStr, 10, 64)
	if err != nil {
		return nil, fmt.Errorf("nats: invalid sequence header '%s': %v", seqStr, err)
	}
	timeStr := r.Header.Get(TimeStampHeaer)
	if timeStr == "" {
		return nil, errors.New("nats: missing timestamp header")
	}

	tm, err := time.Parse(time.RFC3339Nano, timeStr)
	if err != nil {
		return nil, fmt.Errorf("nats: invalid timestamp header '%s': %v", timeStr, err)
	}
	subj := r.Header.Get(SubjectHeader)
	if subj == "" {
		return nil, errors.New("nats: missing subject header")
	}
	return &RawStreamMsg{
		Subject:  subj,
		Sequence: seq,
		Header:   r.Header,
		Data:     r.Data,
		Time:     tm,
	}, nil
}

// DeleteMsg deletes a message from a stream.
// On the server, the message is marked as erased, but not overwritten.
func (s *stream) DeleteMsg(ctx context.Context, seq uint64) error {
	return s.deleteMsg(ctx, &msgDeleteRequest{Seq: seq, NoErase: true})
}

// SecureDeleteMsg deletes a message from a stream. The deleted message
// is overwritten with random data. As a result, this operation is slower
// than DeleteMsg.
func (s *stream) SecureDeleteMsg(ctx context.Context, seq uint64) error {
	return s.deleteMsg(ctx, &msgDeleteRequest{Seq: seq})
}

func (s *stream) deleteMsg(ctx context.Context, req *msgDeleteRequest) error {
	ctx, cancel := s.js.wrapContextWithoutDeadline(ctx)
	if cancel != nil {
		defer cancel()
	}
	r, err := json.Marshal(req)
	if err != nil {
		return err
	}
	subj := fmt.Sprintf(apiMsgDeleteT, s.name)
	var resp msgDeleteResponse
	if _, err = s.js.apiRequestJSON(ctx, subj, &resp, r); err != nil {
		return err
	}
	if !resp.Success {
		return fmt.Errorf("%w: %s", ErrMsgDeleteUnsuccessful, resp.Error.Error())
	}
	return nil
}

// ListConsumers returns ConsumerInfoLister enabling iterating over a
// channel of consumer infos.
func (s *stream) ListConsumers(ctx context.Context) ConsumerInfoLister {
	l := &consumerLister{
		js:        s.js,
		consumers: make(chan *ConsumerInfo),
	}
	go func() {
		defer close(l.consumers)
		ctx, cancel := s.js.wrapContextWithoutDeadline(ctx)
		if cancel != nil {
			defer cancel()
		}
		for {
			page, err := l.consumerInfos(ctx, s.name)
			if err != nil && !errors.Is(err, ErrEndOfData) {
				l.err = err
				return
			}
			for _, info := range page {
				select {
				case <-ctx.Done():
					l.err = ctx.Err()
					return
				default:
				}
				if info != nil {
					l.consumers <- info
				}
			}
			if errors.Is(err, ErrEndOfData) {
				return
			}
		}
	}()

	return l
}

func (s *consumerLister) Info() <-chan *ConsumerInfo {
	return s.consumers
}

func (s *consumerLister) Err() error {
	return s.err
}

// ConsumerNames returns a ConsumerNameLister enabling iterating over a
// channel of consumer names.
func (s *stream) ConsumerNames(ctx context.Context) ConsumerNameLister {
	l := &consumerLister{
		js:    s.js,
		names: make(chan string),
	}
	go func() {
		defer close(l.names)
		ctx, cancel := s.js.wrapContextWithoutDeadline(ctx)
		if cancel != nil {
			defer cancel()
		}
		for {
			page, err := l.consumerNames(ctx, s.name)
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

func (s *consumerLister) Name() <-chan string {
	return s.names
}

// consumerInfos fetches the next ConsumerInfo page
func (s *consumerLister) consumerInfos(ctx context.Context, stream string) ([]*ConsumerInfo, error) {
	if s.pageInfo != nil && s.offset >= s.pageInfo.Total {
		return nil, ErrEndOfData
	}

	req, err := json.Marshal(
		apiPagedRequest{Offset: s.offset},
	)
	if err != nil {
		return nil, err
	}

	slSubj := fmt.Sprintf(apiConsumerListT, stream)
	var resp consumerListResponse
	_, err = s.js.apiRequestJSON(ctx, slSubj, &resp, req)
	if err != nil {
		return nil, err
	}
	if resp.Error != nil {
		return nil, resp.Error
	}

	s.pageInfo = &resp.apiPaged
	s.offset += len(resp.Consumers)
	return resp.Consumers, nil
}

// consumerNames fetches the next consumer names page
func (s *consumerLister) consumerNames(ctx context.Context, stream string) ([]string, error) {
	if s.pageInfo != nil && s.offset >= s.pageInfo.Total {
		return nil, ErrEndOfData
	}

	req, err := json.Marshal(
		apiPagedRequest{Offset: s.offset},
	)
	if err != nil {
		return nil, err
	}

	slSubj := fmt.Sprintf(apiConsumerNamesT, stream)
	var resp consumerNamesResponse
	_, err = s.js.apiRequestJSON(ctx, slSubj, &resp, req)
	if err != nil {
		return nil, err
	}
	if resp.Error != nil {
		return nil, resp.Error
	}

	s.pageInfo = &resp.apiPaged
	s.offset += len(resp.Consumers)
	return resp.Consumers, nil
}

// UnpinConsumer unpins the currently pinned client for a consumer for the given group name.
// If consumer does not exist, ErrConsumerNotFound is returned.
func (s *stream) UnpinConsumer(ctx context.Context, consumer string, group string) error {
	return unpinConsumer(ctx, s.js, s.name, consumer, group)
}

// ResetConsumer resets a consumer's delivery state. See [Stream.ResetConsumer].
func (s *stream) ResetConsumer(ctx context.Context, consumer string) (*ConsumerResetResponse, error) {
	return resetConsumer(ctx, s.js, s.name, consumer, 0)
}

// ResetConsumerToSequence resets a consumer's delivery state to the given
// stream sequence. See [Stream.ResetConsumerToSequence].
func (s *stream) ResetConsumerToSequence(ctx context.Context, consumer string, seq uint64) (*ConsumerResetResponse, error) {
	return resetConsumer(ctx, s.js, s.name, consumer, seq)
}
