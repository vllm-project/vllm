// Copyright 2021-2023 The NATS Authors
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

package nats

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"strconv"
	"strings"
	"time"
)

// JetStreamManager manages JetStream Streams and Consumers.
type JetStreamManager interface {
	// AddStream creates a stream.
	AddStream(cfg *StreamConfig, opts ...JSOpt) (*StreamInfo, error)

	// UpdateStream updates a stream.
	UpdateStream(cfg *StreamConfig, opts ...JSOpt) (*StreamInfo, error)

	// DeleteStream deletes a stream.
	DeleteStream(name string, opts ...JSOpt) error

	// StreamInfo retrieves information from a stream.
	StreamInfo(stream string, opts ...JSOpt) (*StreamInfo, error)

	// PurgeStream purges a stream messages.
	PurgeStream(name string, opts ...JSOpt) error

	// StreamsInfo can be used to retrieve a list of StreamInfo objects.
	// Deprecated: Use Streams() instead.
	StreamsInfo(opts ...JSOpt) <-chan *StreamInfo

	// Streams can be used to retrieve a list of StreamInfo objects.
	Streams(opts ...JSOpt) <-chan *StreamInfo

	// StreamNames is used to retrieve a list of Stream names.
	StreamNames(opts ...JSOpt) <-chan string

	// GetMsg retrieves a raw stream message stored in JetStream by sequence number.
	// Use options nats.DirectGet() or nats.DirectGetNext() to trigger retrieval
	// directly from a distributed group of servers (leader and replicas).
	// The stream must have been created/updated with the AllowDirect boolean.
	GetMsg(name string, seq uint64, opts ...JSOpt) (*RawStreamMsg, error)

	// GetLastMsg retrieves the last raw stream message stored in JetStream by subject.
	// Use option nats.DirectGet() to trigger retrieval
	// directly from a distributed group of servers (leader and replicas).
	// The stream must have been created/updated with the AllowDirect boolean.
	GetLastMsg(name, subject string, opts ...JSOpt) (*RawStreamMsg, error)

	// DeleteMsg deletes a message from a stream. The message is marked as erased, but its value is not overwritten.
	DeleteMsg(name string, seq uint64, opts ...JSOpt) error

	// SecureDeleteMsg deletes a message from a stream. The deleted message is overwritten with random data
	// As a result, this operation is slower than DeleteMsg()
	SecureDeleteMsg(name string, seq uint64, opts ...JSOpt) error

	// AddConsumer adds a consumer to a stream.
	// If the consumer already exists, and the configuration is the same, it
	// will return the existing consumer.
	// If the consumer already exists, and the configuration is different, it
	// will return ErrConsumerNameAlreadyInUse.
	AddConsumer(stream string, cfg *ConsumerConfig, opts ...JSOpt) (*ConsumerInfo, error)

	// UpdateConsumer updates an existing consumer.
	UpdateConsumer(stream string, cfg *ConsumerConfig, opts ...JSOpt) (*ConsumerInfo, error)

	// DeleteConsumer deletes a consumer.
	DeleteConsumer(stream, consumer string, opts ...JSOpt) error

	// ConsumerInfo retrieves information of a consumer from a stream.
	ConsumerInfo(stream, name string, opts ...JSOpt) (*ConsumerInfo, error)

	// ConsumersInfo is used to retrieve a list of ConsumerInfo objects.
	// Deprecated: Use Consumers() instead.
	ConsumersInfo(stream string, opts ...JSOpt) <-chan *ConsumerInfo

	// Consumers is used to retrieve a list of ConsumerInfo objects.
	Consumers(stream string, opts ...JSOpt) <-chan *ConsumerInfo

	// ConsumerNames is used to retrieve a list of Consumer names.
	ConsumerNames(stream string, opts ...JSOpt) <-chan string

	// AccountInfo retrieves info about the JetStream usage from an account.
	AccountInfo(opts ...JSOpt) (*AccountInfo, error)

	// StreamNameBySubject returns a stream matching given subject.
	StreamNameBySubject(string, ...JSOpt) (string, error)
}

// StreamConfig will determine the properties for a stream.
// There are sensible defaults for most. If no subjects are
// given the name will be used as the only subject.
type StreamConfig struct {
	// Name is the name of the stream. It is required and must be unique
	// across the JetStream account.
	//
	// Name Names cannot contain whitespace, ., *, >, path separators
	// (forward or backwards slash), and non-printable characters.
	Name string `json:"name"`

	// Description is an optional description of the stream.
	Description string `json:"description,omitempty"`

	// Subjects is a list of subjects that the stream is listening on.
	// Wildcards are supported. Subjects cannot be set if the stream is
	// created as a mirror.
	Subjects []string `json:"subjects,omitempty"`

	// Retention defines the message retention policy for the stream.
	// Defaults to LimitsPolicy.
	Retention RetentionPolicy `json:"retention"`

	// MaxConsumers specifies the maximum number of consumers allowed for
	// the stream.
	MaxConsumers int `json:"max_consumers"`

	// MaxMsgs is the maximum number of messages the stream will store.
	// After reaching the limit, stream adheres to the discard policy.
	// If not set, server default is -1 (unlimited).
	MaxMsgs int64 `json:"max_msgs"`

	// MaxBytes is the maximum total size of messages the stream will store.
	// After reaching the limit, stream adheres to the discard policy.
	// If not set, server default is -1 (unlimited).
	MaxBytes int64 `json:"max_bytes"`

	// Discard defines the policy for handling messages when the stream
	// reaches its limits in terms of number of messages or total bytes.
	Discard DiscardPolicy `json:"discard"`

	// DiscardNewPerSubject is a flag to enable discarding new messages per
	// subject when limits are reached. Requires DiscardPolicy to be
	// DiscardNew and the MaxMsgsPerSubject to be set.
	DiscardNewPerSubject bool `json:"discard_new_per_subject,omitempty"`

	// MaxAge is the maximum age of messages that the stream will retain.
	MaxAge time.Duration `json:"max_age"`

	// MaxMsgsPerSubject is the maximum number of messages per subject that
	// the stream will retain.
	MaxMsgsPerSubject int64 `json:"max_msgs_per_subject"`

	// MaxMsgSize is the maximum size of any single message in the stream.
	MaxMsgSize int32 `json:"max_msg_size,omitempty"`

	// Storage specifies the type of storage backend used for the stream
	// (file or memory).
	Storage StorageType `json:"storage"`

	// Replicas is the number of stream replicas in clustered JetStream.
	// Defaults to 1, maximum is 5.
	Replicas int `json:"num_replicas"`

	// NoAck is a flag to disable acknowledging messages received by this
	// stream.
	//
	// If set to true, publish methods from the JetStream client will not
	// work as expected, since they rely on acknowledgements. Core NATS
	// publish methods should be used instead. Note that this will make
	// message delivery less reliable.
	NoAck bool `json:"no_ack,omitempty"`

	// Duplicates is the window within which to track duplicate messages.
	// If not set, server default is 2 minutes.
	Duplicates time.Duration `json:"duplicate_window,omitempty"`

	// Placement is used to declare where the stream should be placed via
	// tags and/or an explicit cluster name.
	Placement *Placement `json:"placement,omitempty"`

	// Mirror defines the configuration for mirroring another stream.
	Mirror *StreamSource `json:"mirror,omitempty"`

	// Sources is a list of other streams this stream sources messages from.
	Sources []*StreamSource `json:"sources,omitempty"`

	// Sealed streams do not allow messages to be published or deleted via limits or API,
	// sealed streams can not be unsealed via configuration update. Can only
	// be set on already created streams via the Update API.
	Sealed bool `json:"sealed,omitempty"`

	// DenyDelete restricts the ability to delete messages from a stream via
	// the API. Defaults to false.
	DenyDelete bool `json:"deny_delete,omitempty"`

	// DenyPurge restricts the ability to purge messages from a stream via
	// the API. Defaults to false.
	DenyPurge bool `json:"deny_purge,omitempty"`

	// AllowRollup allows the use of the Nats-Rollup header to replace all
	// contents of a stream, or subject in a stream, with a single new
	// message.
	AllowRollup bool `json:"allow_rollup_hdrs,omitempty"`

	// Compression specifies the message storage compression algorithm.
	// Defaults to NoCompression.
	Compression StoreCompression `json:"compression"`

	// FirstSeq is the initial sequence number of the first message in the
	// stream.
	FirstSeq uint64 `json:"first_seq,omitempty"`

	// SubjectTransform allows applying a transformation to matching
	// messages' subjects.
	SubjectTransform *SubjectTransformConfig `json:"subject_transform,omitempty"`

	// RePublish allows immediate republishing a message to the configured
	// subject after it's stored.
	RePublish *RePublish `json:"republish,omitempty"`

	// AllowDirect enables direct access to individual messages using direct
	// get API. Defaults to false.
	AllowDirect bool `json:"allow_direct"`

	// MirrorDirect enables direct access to individual messages from the
	// origin stream using direct get API. Defaults to false.
	MirrorDirect bool `json:"mirror_direct"`

	// ConsumerLimits defines limits of certain values that consumers can
	// set, defaults for those who don't set these settings
	ConsumerLimits StreamConsumerLimits `json:"consumer_limits,omitempty"`

	// Metadata is a set of application-defined key-value pairs for
	// associating metadata on the stream. This feature requires nats-server
	// v2.10.0 or later.
	Metadata map[string]string `json:"metadata,omitempty"`

	// Template identifies the template that manages the Stream. Deprecated:
	// This feature is no longer supported.
	Template string `json:"template_owner,omitempty"`

	// AllowMsgTTL allows header initiated per-message TTLs.
	// This feature requires nats-server v2.11.0 or later.
	AllowMsgTTL bool `json:"allow_msg_ttl"`

	// Enables and sets a duration for adding server markers for delete, purge and max age limits.
	// This feature requires nats-server v2.11.0 or later.
	SubjectDeleteMarkerTTL time.Duration `json:"subject_delete_marker_ttl,omitempty"`
}

// SubjectTransformConfig is for applying a subject transform (to matching messages) before doing anything else when a new message is received.
type SubjectTransformConfig struct {
	Source      string `json:"src,omitempty"`
	Destination string `json:"dest"`
}

// RePublish is for republishing messages once committed to a stream. The original
// subject cis remapped from the subject pattern to the destination pattern.
type RePublish struct {
	Source      string `json:"src,omitempty"`
	Destination string `json:"dest"`
	HeadersOnly bool   `json:"headers_only,omitempty"`
}

// Placement is used to guide placement of streams in clustered JetStream.
type Placement struct {
	Cluster string   `json:"cluster"`
	Tags    []string `json:"tags,omitempty"`
}

// StreamSource dictates how streams can source from other streams.
type StreamSource struct {
	Name              string                   `json:"name"`
	OptStartSeq       uint64                   `json:"opt_start_seq,omitempty"`
	OptStartTime      *time.Time               `json:"opt_start_time,omitempty"`
	FilterSubject     string                   `json:"filter_subject,omitempty"`
	SubjectTransforms []SubjectTransformConfig `json:"subject_transforms,omitempty"`
	External          *ExternalStream          `json:"external,omitempty"`
	Domain            string                   `json:"-"`
}

// ExternalStream allows you to qualify access to a stream source in another
// account.
type ExternalStream struct {
	APIPrefix     string `json:"api"`
	DeliverPrefix string `json:"deliver,omitempty"`
}

// StreamConsumerLimits are the limits for a consumer on a stream.
// These can be overridden on a per consumer basis.
type StreamConsumerLimits struct {
	InactiveThreshold time.Duration `json:"inactive_threshold,omitempty"`
	MaxAckPending     int           `json:"max_ack_pending,omitempty"`
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

// If we have a Domain, convert to the appropriate ext.APIPrefix.
// This will change the stream source, so should be a copy passed in.
func (ss *StreamSource) convertDomain() error {
	if ss.Domain == _EMPTY_ {
		return nil
	}
	if ss.External != nil {
		// These should be mutually exclusive.
		// TODO(dlc) - Make generic?
		return errors.New("nats: domain and external are both set")
	}
	ss.External = &ExternalStream{APIPrefix: fmt.Sprintf(jsExtDomainT, ss.Domain)}
	return nil
}

// apiResponse is a standard response from the JetStream JSON API
type apiResponse struct {
	Type  string    `json:"type"`
	Error *APIError `json:"error,omitempty"`
}

// apiPaged includes variables used to create paged responses from the JSON API
type apiPaged struct {
	Total  int `json:"total"`
	Offset int `json:"offset"`
	Limit  int `json:"limit"`
}

// apiPagedRequest includes parameters allowing specific pages to be requested
// from APIs responding with apiPaged.
type apiPagedRequest struct {
	Offset int `json:"offset,omitempty"`
}

// AccountInfo contains info about the JetStream usage from the current account.
type AccountInfo struct {
	Tier
	Domain string          `json:"domain"`
	API    APIStats        `json:"api"`
	Tiers  map[string]Tier `json:"tiers"`
}

type Tier struct {
	Memory         uint64        `json:"memory"`
	Store          uint64        `json:"storage"`
	ReservedMemory uint64        `json:"reserved_memory"`
	ReservedStore  uint64        `json:"reserved_storage"`
	Streams        int           `json:"streams"`
	Consumers      int           `json:"consumers"`
	Limits         AccountLimits `json:"limits"`
}

// APIStats reports on API calls to JetStream for this account.
type APIStats struct {
	Level    int    `json:"level"`
	Total    uint64 `json:"total"`
	Errors   uint64 `json:"errors"`
	Inflight uint64 `json:"inflight,omitempty"`
}

// AccountLimits includes the JetStream limits of the current account.
type AccountLimits struct {
	MaxMemory            int64 `json:"max_memory"`
	MaxStore             int64 `json:"max_storage"`
	MaxStreams           int   `json:"max_streams"`
	MaxConsumers         int   `json:"max_consumers"`
	MaxAckPending        int   `json:"max_ack_pending"`
	MemoryMaxStreamBytes int64 `json:"memory_max_stream_bytes"`
	StoreMaxStreamBytes  int64 `json:"storage_max_stream_bytes"`
	MaxBytesRequired     bool  `json:"max_bytes_required"`
}

type accountInfoResponse struct {
	apiResponse
	AccountInfo
}

// AccountInfo fetches account information from the server, containing details
// about the account associated with this JetStream connection. If account is
// not enabled for JetStream, ErrJetStreamNotEnabledForAccount is returned.
//
// If the server does not have JetStream enabled, ErrJetStreamNotEnabled is
// returned (for a single server setup). For clustered topologies, AccountInfo
// will time out.
func (js *js) AccountInfo(opts ...JSOpt) (*AccountInfo, error) {
	o, cancel, err := getJSContextOpts(js.opts, opts...)
	if err != nil {
		return nil, err
	}
	if cancel != nil {
		defer cancel()
	}

	resp, err := js.apiRequestWithContext(o.ctx, o.apiSubj(apiAccountInfo), nil)
	if err != nil {
		// todo maybe nats server should never have no responder on this subject and always respond if they know there is no js to be had
		if errors.Is(err, ErrNoResponders) {
			err = ErrJetStreamNotEnabled
		}
		return nil, err
	}
	var info accountInfoResponse
	if err := json.Unmarshal(resp.Data, &info); err != nil {
		return nil, err
	}
	if info.Error != nil {
		// Internally checks based on error code instead of description match.
		if errors.Is(info.Error, ErrJetStreamNotEnabledForAccount) {
			return nil, ErrJetStreamNotEnabledForAccount
		}
		return nil, info.Error
	}

	return &info.AccountInfo, nil
}

type createConsumerRequest struct {
	Stream string          `json:"stream_name"`
	Config *ConsumerConfig `json:"config"`
}

type consumerResponse struct {
	apiResponse
	*ConsumerInfo
}

// AddConsumer adds a consumer to a stream.
// If the consumer already exists, and the configuration is the same, it
// will return the existing consumer.
// If the consumer already exists, and the configuration is different, it
// will return ErrConsumerNameAlreadyInUse.
func (js *js) AddConsumer(stream string, cfg *ConsumerConfig, opts ...JSOpt) (*ConsumerInfo, error) {
	if cfg == nil {
		cfg = &ConsumerConfig{}
	}
	consumerName := cfg.Name
	if consumerName == _EMPTY_ {
		consumerName = cfg.Durable
	}
	if consumerName != _EMPTY_ {
		consInfo, err := js.ConsumerInfo(stream, consumerName, opts...)
		if err != nil && !errors.Is(err, ErrConsumerNotFound) && !errors.Is(err, ErrStreamNotFound) {
			return nil, err
		}

		if consInfo != nil {
			sameConfig := checkConfig(&consInfo.Config, cfg)
			if sameConfig != nil {
				return nil, fmt.Errorf("%w: creating consumer %q on stream %q", ErrConsumerNameAlreadyInUse, consumerName, stream)
			} else {
				return consInfo, nil
			}
		}
	}

	return js.upsertConsumer(stream, consumerName, cfg, opts...)
}

func (js *js) UpdateConsumer(stream string, cfg *ConsumerConfig, opts ...JSOpt) (*ConsumerInfo, error) {
	if cfg == nil {
		return nil, ErrConsumerConfigRequired
	}
	consumerName := cfg.Name
	if consumerName == _EMPTY_ {
		consumerName = cfg.Durable
	}
	if consumerName == _EMPTY_ {
		return nil, ErrConsumerNameRequired
	}
	return js.upsertConsumer(stream, consumerName, cfg, opts...)
}

func (js *js) upsertConsumer(stream, consumerName string, cfg *ConsumerConfig, opts ...JSOpt) (*ConsumerInfo, error) {
	if err := checkStreamName(stream); err != nil {
		return nil, err
	}
	o, cancel, err := getJSContextOpts(js.opts, opts...)
	if err != nil {
		return nil, err
	}
	if cancel != nil {
		defer cancel()
	}

	req, err := json.Marshal(&createConsumerRequest{Stream: stream, Config: cfg})
	if err != nil {
		return nil, err
	}

	var ccSubj string
	if consumerName == _EMPTY_ {
		// if consumer name is empty (neither Durable nor Name is set), use the legacy ephemeral endpoint
		ccSubj = fmt.Sprintf(apiLegacyConsumerCreateT, stream)
	} else if err := checkConsumerName(consumerName); err != nil {
		return nil, err
	} else if js.nc.serverMinVersion(2, 9, 0) {
		if cfg.Durable != "" && js.opts.featureFlags.useDurableConsumerCreate {
			// if user set the useDurableConsumerCreate flag, use the legacy DURABLE.CREATE endpoint
			ccSubj = fmt.Sprintf(apiDurableCreateT, stream, consumerName)
		} else if cfg.FilterSubject == _EMPTY_ || cfg.FilterSubject == ">" {
			// if filter subject is empty or ">", use the endpoint without filter subject
			ccSubj = fmt.Sprintf(apiConsumerCreateT, stream, consumerName)
		} else {
			// safeguard against passing invalid filter subject in request subject
			if cfg.FilterSubject[0] == '.' || cfg.FilterSubject[len(cfg.FilterSubject)-1] == '.' {
				return nil, fmt.Errorf("%w: %q", ErrInvalidFilterSubject, cfg.FilterSubject)
			}
			// if filter subject is not empty, use the endpoint with filter subject
			ccSubj = fmt.Sprintf(apiConsumerCreateWithFilterSubjectT, stream, consumerName, cfg.FilterSubject)
		}
	} else {
		if cfg.Durable != "" {
			// if Durable is set, use the DURABLE.CREATE endpoint
			ccSubj = fmt.Sprintf(apiDurableCreateT, stream, consumerName)
		} else {
			// if Durable is not set, use the legacy ephemeral endpoint
			ccSubj = fmt.Sprintf(apiLegacyConsumerCreateT, stream)
		}
	}

	resp, err := js.apiRequestWithContext(o.ctx, o.apiSubj(ccSubj), req)
	if err != nil {
		if errors.Is(err, ErrNoResponders) {
			err = ErrJetStreamNotEnabled
		}
		return nil, err
	}
	var info consumerResponse
	err = json.Unmarshal(resp.Data, &info)
	if err != nil {
		return nil, err
	}
	if info.Error != nil {
		if errors.Is(info.Error, ErrStreamNotFound) {
			return nil, ErrStreamNotFound
		}
		if errors.Is(info.Error, ErrConsumerNotFound) {
			return nil, ErrConsumerNotFound
		}
		return nil, info.Error
	}

	if info.Error == nil && info.ConsumerInfo == nil {
		return nil, ErrConsumerCreationResponseEmpty
	}

	// check whether multiple filter subjects (if used) are reflected in the returned ConsumerInfo
	if len(cfg.FilterSubjects) != 0 && len(info.Config.FilterSubjects) == 0 {
		return nil, ErrConsumerMultipleFilterSubjectsNotSupported
	}
	return info.ConsumerInfo, nil
}

// consumerDeleteResponse is the response for a Consumer delete request.
type consumerDeleteResponse struct {
	apiResponse
	Success bool `json:"success,omitempty"`
}

func checkStreamName(stream string) error {
	if stream == _EMPTY_ {
		return ErrStreamNameRequired
	}
	if strings.ContainsAny(stream, ". ") {
		return ErrInvalidStreamName
	}
	return nil
}

// Check that the consumer name is not empty and is valid (does not contain "." and " ").
// Additional consumer name validation is done in nats-server.
// Returns ErrConsumerNameRequired if consumer name is empty, ErrInvalidConsumerName is invalid, otherwise nil
func checkConsumerName(consumer string) error {
	if consumer == _EMPTY_ {
		return ErrConsumerNameRequired
	}
	if strings.ContainsAny(consumer, ". ") {
		return ErrInvalidConsumerName
	}
	return nil
}

// DeleteConsumer deletes a Consumer.
func (js *js) DeleteConsumer(stream, consumer string, opts ...JSOpt) error {
	if err := checkStreamName(stream); err != nil {
		return err
	}
	if err := checkConsumerName(consumer); err != nil {
		return err
	}
	o, cancel, err := getJSContextOpts(js.opts, opts...)
	if err != nil {
		return err
	}
	if cancel != nil {
		defer cancel()
	}

	dcSubj := o.apiSubj(fmt.Sprintf(apiConsumerDeleteT, stream, consumer))
	r, err := js.apiRequestWithContext(o.ctx, dcSubj, nil)
	if err != nil {
		return err
	}
	var resp consumerDeleteResponse
	if err := json.Unmarshal(r.Data, &resp); err != nil {
		return err
	}

	if resp.Error != nil {
		if errors.Is(resp.Error, ErrConsumerNotFound) {
			return ErrConsumerNotFound
		}
		return resp.Error
	}
	return nil
}

// ConsumerInfo returns information about a Consumer.
func (js *js) ConsumerInfo(stream, consumer string, opts ...JSOpt) (*ConsumerInfo, error) {
	if err := checkStreamName(stream); err != nil {
		return nil, err
	}
	if err := checkConsumerName(consumer); err != nil {
		return nil, err
	}
	o, cancel, err := getJSContextOpts(js.opts, opts...)
	if err != nil {
		return nil, err
	}
	if cancel != nil {
		defer cancel()
	}
	return js.getConsumerInfoContext(o.ctx, stream, consumer, o)
}

// consumerLister fetches pages of ConsumerInfo objects. This object is not
// safe to use for multiple threads.
type consumerLister struct {
	stream string
	js     *js

	err      error
	offset   int
	page     []*ConsumerInfo
	pageInfo *apiPaged
}

// consumersRequest is the type used for Consumers requests.
type consumersRequest struct {
	apiPagedRequest
}

// consumerListResponse is the response for a Consumers List request.
type consumerListResponse struct {
	apiResponse
	apiPaged
	Consumers []*ConsumerInfo `json:"consumers"`
}

// Next fetches the next ConsumerInfo page.
func (c *consumerLister) Next() bool {
	if c.err != nil {
		return false
	}
	if err := checkStreamName(c.stream); err != nil {
		c.err = err
		return false
	}
	if c.pageInfo != nil && c.offset >= c.pageInfo.Total {
		return false
	}

	req, err := json.Marshal(consumersRequest{
		apiPagedRequest: apiPagedRequest{Offset: c.offset},
	})
	if err != nil {
		c.err = err
		return false
	}

	var cancel context.CancelFunc
	ctx := c.js.opts.ctx
	if ctx == nil {
		ctx, cancel = context.WithTimeout(context.Background(), c.js.opts.wait)
		defer cancel()
	}

	clSubj := c.js.apiSubj(fmt.Sprintf(apiConsumerListT, c.stream))
	r, err := c.js.apiRequestWithContext(ctx, clSubj, req)
	if err != nil {
		c.err = err
		return false
	}
	var resp consumerListResponse
	if err := json.Unmarshal(r.Data, &resp); err != nil {
		c.err = err
		return false
	}
	if resp.Error != nil {
		c.err = resp.Error
		return false
	}

	c.pageInfo = &resp.apiPaged
	c.page = resp.Consumers
	c.offset += len(c.page)
	return true
}

// Page returns the current ConsumerInfo page.
func (c *consumerLister) Page() []*ConsumerInfo {
	return c.page
}

// Err returns any errors found while fetching pages.
func (c *consumerLister) Err() error {
	return c.err
}

// Consumers is used to retrieve a list of ConsumerInfo objects.
func (jsc *js) Consumers(stream string, opts ...JSOpt) <-chan *ConsumerInfo {
	o, cancel, err := getJSContextOpts(jsc.opts, opts...)
	if err != nil {
		return nil
	}

	ch := make(chan *ConsumerInfo)
	l := &consumerLister{js: &js{nc: jsc.nc, opts: o}, stream: stream}
	go func() {
		if cancel != nil {
			defer cancel()
		}
		defer close(ch)
		for l.Next() {
			for _, info := range l.Page() {
				select {
				case ch <- info:
				case <-o.ctx.Done():
					return
				}
			}
		}
	}()

	return ch
}

// ConsumersInfo is used to retrieve a list of ConsumerInfo objects.
// Deprecated: Use Consumers() instead.
func (jsc *js) ConsumersInfo(stream string, opts ...JSOpt) <-chan *ConsumerInfo {
	return jsc.Consumers(stream, opts...)
}

type consumerNamesLister struct {
	stream string
	js     *js

	err      error
	offset   int
	page     []string
	pageInfo *apiPaged
}

// consumerNamesListResponse is the response for a Consumers Names List request.
type consumerNamesListResponse struct {
	apiResponse
	apiPaged
	Consumers []string `json:"consumers"`
}

// Next fetches the next consumer names page.
func (c *consumerNamesLister) Next() bool {
	if c.err != nil {
		return false
	}
	if err := checkStreamName(c.stream); err != nil {
		c.err = err
		return false
	}
	if c.pageInfo != nil && c.offset >= c.pageInfo.Total {
		return false
	}

	var cancel context.CancelFunc
	ctx := c.js.opts.ctx
	if ctx == nil {
		ctx, cancel = context.WithTimeout(context.Background(), c.js.opts.wait)
		defer cancel()
	}

	req, err := json.Marshal(consumersRequest{
		apiPagedRequest: apiPagedRequest{Offset: c.offset},
	})
	if err != nil {
		c.err = err
		return false
	}
	clSubj := c.js.apiSubj(fmt.Sprintf(apiConsumerNamesT, c.stream))
	r, err := c.js.apiRequestWithContext(ctx, clSubj, req)
	if err != nil {
		c.err = err
		return false
	}
	var resp consumerNamesListResponse
	if err := json.Unmarshal(r.Data, &resp); err != nil {
		c.err = err
		return false
	}
	if resp.Error != nil {
		c.err = resp.Error
		return false
	}

	c.pageInfo = &resp.apiPaged
	c.page = resp.Consumers
	c.offset += len(c.page)
	return true
}

// Page returns the current ConsumerInfo page.
func (c *consumerNamesLister) Page() []string {
	return c.page
}

// Err returns any errors found while fetching pages.
func (c *consumerNamesLister) Err() error {
	return c.err
}

// ConsumerNames is used to retrieve a list of Consumer names.
func (jsc *js) ConsumerNames(stream string, opts ...JSOpt) <-chan string {
	o, cancel, err := getJSContextOpts(jsc.opts, opts...)
	if err != nil {
		return nil
	}

	ch := make(chan string)
	l := &consumerNamesLister{stream: stream, js: &js{nc: jsc.nc, opts: o}}
	go func() {
		if cancel != nil {
			defer cancel()
		}
		defer close(ch)
		for l.Next() {
			for _, info := range l.Page() {
				select {
				case ch <- info:
				case <-o.ctx.Done():
					return
				}
			}
		}
	}()

	return ch
}

// streamCreateResponse stream creation.
type streamCreateResponse struct {
	apiResponse
	*StreamInfo
}

func (js *js) AddStream(cfg *StreamConfig, opts ...JSOpt) (*StreamInfo, error) {
	if cfg == nil {
		return nil, ErrStreamConfigRequired
	}
	if err := checkStreamName(cfg.Name); err != nil {
		return nil, err
	}
	o, cancel, err := getJSContextOpts(js.opts, opts...)
	if err != nil {
		return nil, err
	}
	if cancel != nil {
		defer cancel()
	}

	// In case we need to change anything, copy so we do not change the caller's version.
	ncfg := *cfg

	// If we have a mirror and an external domain, convert to ext.APIPrefix.
	if cfg.Mirror != nil && cfg.Mirror.Domain != _EMPTY_ {
		// Copy so we do not change the caller's version.
		ncfg.Mirror = ncfg.Mirror.copy()
		if err := ncfg.Mirror.convertDomain(); err != nil {
			return nil, err
		}
	}
	// Check sources for the same.
	if len(ncfg.Sources) > 0 {
		ncfg.Sources = append([]*StreamSource(nil), ncfg.Sources...)
		for i, ss := range ncfg.Sources {
			if ss.Domain != _EMPTY_ {
				ncfg.Sources[i] = ss.copy()
				if err := ncfg.Sources[i].convertDomain(); err != nil {
					return nil, err
				}
			}
		}
	}

	req, err := json.Marshal(&ncfg)
	if err != nil {
		return nil, err
	}

	csSubj := o.apiSubj(fmt.Sprintf(apiStreamCreateT, cfg.Name))
	r, err := js.apiRequestWithContext(o.ctx, csSubj, req)
	if err != nil {
		return nil, err
	}
	var resp streamCreateResponse
	if err := json.Unmarshal(r.Data, &resp); err != nil {
		return nil, err
	}
	if resp.Error != nil {
		if errors.Is(resp.Error, ErrStreamNameAlreadyInUse) {
			return nil, ErrStreamNameAlreadyInUse
		}
		return nil, resp.Error
	}

	// check that input subject transform (if used) is reflected in the returned ConsumerInfo
	if cfg.SubjectTransform != nil && resp.StreamInfo.Config.SubjectTransform == nil {
		return nil, ErrStreamSubjectTransformNotSupported
	}
	if len(cfg.Sources) != 0 {
		if len(cfg.Sources) != len(resp.Config.Sources) {
			return nil, ErrStreamSourceNotSupported
		}
		for i := range cfg.Sources {
			if len(cfg.Sources[i].SubjectTransforms) != 0 && len(resp.Sources[i].SubjectTransforms) == 0 {
				return nil, ErrStreamSourceMultipleSubjectTransformsNotSupported
			}
		}
	}

	return resp.StreamInfo, nil
}

type (
	// StreamInfoRequest contains additional option to return
	StreamInfoRequest struct {
		apiPagedRequest
		// DeletedDetails when true includes information about deleted messages
		DeletedDetails bool `json:"deleted_details,omitempty"`
		// SubjectsFilter when set, returns information on the matched subjects
		SubjectsFilter string `json:"subjects_filter,omitempty"`
	}
	streamInfoResponse = struct {
		apiResponse
		apiPaged
		*StreamInfo
	}
)

func (js *js) StreamInfo(stream string, opts ...JSOpt) (*StreamInfo, error) {
	if err := checkStreamName(stream); err != nil {
		return nil, err
	}
	o, cancel, err := getJSContextOpts(js.opts, opts...)
	if err != nil {
		return nil, err
	}
	if cancel != nil {
		defer cancel()
	}

	var i int
	var subjectMessagesMap map[string]uint64
	var req []byte
	var requestPayload bool

	var siOpts StreamInfoRequest
	if o.streamInfoOpts != nil {
		requestPayload = true
		siOpts = *o.streamInfoOpts
	}

	for {
		if requestPayload {
			siOpts.Offset = i
			if req, err = json.Marshal(&siOpts); err != nil {
				return nil, err
			}
		}

		siSubj := o.apiSubj(fmt.Sprintf(apiStreamInfoT, stream))

		r, err := js.apiRequestWithContext(o.ctx, siSubj, req)
		if err != nil {
			return nil, err
		}

		var resp streamInfoResponse
		if err := json.Unmarshal(r.Data, &resp); err != nil {
			return nil, err
		}

		if resp.Error != nil {
			if errors.Is(resp.Error, ErrStreamNotFound) {
				return nil, ErrStreamNotFound
			}
			return nil, resp.Error
		}

		var total int
		// for backwards compatibility
		if resp.Total != 0 {
			total = resp.Total
		} else {
			total = len(resp.State.Subjects)
		}

		if requestPayload && len(resp.StreamInfo.State.Subjects) > 0 {
			if subjectMessagesMap == nil {
				subjectMessagesMap = make(map[string]uint64, total)
			}

			for k, j := range resp.State.Subjects {
				subjectMessagesMap[k] = j
				i++
			}
		}

		if i >= total {
			if requestPayload {
				resp.StreamInfo.State.Subjects = subjectMessagesMap
			}
			return resp.StreamInfo, nil
		}
	}
}

// StreamInfo shows config and current state for this stream.
type StreamInfo struct {
	Config     StreamConfig        `json:"config"`
	Created    time.Time           `json:"created"`
	State      StreamState         `json:"state"`
	Cluster    *ClusterInfo        `json:"cluster,omitempty"`
	Mirror     *StreamSourceInfo   `json:"mirror,omitempty"`
	Sources    []*StreamSourceInfo `json:"sources,omitempty"`
	Alternates []*StreamAlternate  `json:"alternates,omitempty"`
}

// StreamAlternate is an alternate stream represented by a mirror.
type StreamAlternate struct {
	Name    string `json:"name"`
	Domain  string `json:"domain,omitempty"`
	Cluster string `json:"cluster"`
}

// StreamSourceInfo shows information about an upstream stream source.
type StreamSourceInfo struct {
	Name              string                   `json:"name"`
	Lag               uint64                   `json:"lag"`
	Active            time.Duration            `json:"active"`
	External          *ExternalStream          `json:"external"`
	Error             *APIError                `json:"error"`
	FilterSubject     string                   `json:"filter_subject,omitempty"`
	SubjectTransforms []SubjectTransformConfig `json:"subject_transforms,omitempty"`
}

// StreamState is information about the given stream.
type StreamState struct {
	Msgs        uint64            `json:"messages"`
	Bytes       uint64            `json:"bytes"`
	FirstSeq    uint64            `json:"first_seq"`
	FirstTime   time.Time         `json:"first_ts"`
	LastSeq     uint64            `json:"last_seq"`
	LastTime    time.Time         `json:"last_ts"`
	Consumers   int               `json:"consumer_count"`
	Deleted     []uint64          `json:"deleted"`
	NumDeleted  int               `json:"num_deleted"`
	NumSubjects uint64            `json:"num_subjects"`
	Subjects    map[string]uint64 `json:"subjects"`
}

// ClusterInfo shows information about the underlying set of servers
// that make up the stream or consumer.
type ClusterInfo struct {
	Name        string      `json:"name,omitempty"`
	RaftGroup   string      `json:"raft_group,omitempty"`
	Leader      string      `json:"leader,omitempty"`
	LeaderSince *time.Time  `json:"leader_since,omitempty"`
	SystemAcc   bool        `json:"system_account,omitempty"`
	TrafficAcc  string      `json:"traffic_account,omitempty"`
	Replicas    []*PeerInfo `json:"replicas,omitempty"`
}

// PeerInfo shows information about all the peers in the cluster that
// are supporting the stream or consumer.
type PeerInfo struct {
	Name    string        `json:"name"`
	Current bool          `json:"current"`
	Offline bool          `json:"offline,omitempty"`
	Active  time.Duration `json:"active"`
	Lag     uint64        `json:"lag,omitempty"`
}

// UpdateStream updates a Stream.
func (js *js) UpdateStream(cfg *StreamConfig, opts ...JSOpt) (*StreamInfo, error) {
	if cfg == nil {
		return nil, ErrStreamConfigRequired
	}
	if err := checkStreamName(cfg.Name); err != nil {
		return nil, err
	}
	o, cancel, err := getJSContextOpts(js.opts, opts...)
	if err != nil {
		return nil, err
	}
	if cancel != nil {
		defer cancel()
	}

	req, err := json.Marshal(cfg)
	if err != nil {
		return nil, err
	}

	usSubj := o.apiSubj(fmt.Sprintf(apiStreamUpdateT, cfg.Name))
	r, err := js.apiRequestWithContext(o.ctx, usSubj, req)
	if err != nil {
		return nil, err
	}
	var resp streamInfoResponse
	if err := json.Unmarshal(r.Data, &resp); err != nil {
		return nil, err
	}
	if resp.Error != nil {
		if errors.Is(resp.Error, ErrStreamNotFound) {
			return nil, ErrStreamNotFound
		}
		return nil, resp.Error
	}

	// check that input subject transform (if used) is reflected in the returned StreamInfo
	if cfg.SubjectTransform != nil && resp.StreamInfo.Config.SubjectTransform == nil {
		return nil, ErrStreamSubjectTransformNotSupported
	}

	if len(cfg.Sources) != 0 {
		if len(cfg.Sources) != len(resp.Config.Sources) {
			return nil, ErrStreamSourceNotSupported
		}
		for i := range cfg.Sources {
			if len(cfg.Sources[i].SubjectTransforms) != 0 && len(resp.Sources[i].SubjectTransforms) == 0 {
				return nil, ErrStreamSourceMultipleSubjectTransformsNotSupported
			}
		}
	}

	return resp.StreamInfo, nil
}

// streamDeleteResponse is the response for a Stream delete request.
type streamDeleteResponse struct {
	apiResponse
	Success bool `json:"success,omitempty"`
}

// DeleteStream deletes a Stream.
func (js *js) DeleteStream(name string, opts ...JSOpt) error {
	if err := checkStreamName(name); err != nil {
		return err
	}
	o, cancel, err := getJSContextOpts(js.opts, opts...)
	if err != nil {
		return err
	}
	if cancel != nil {
		defer cancel()
	}

	dsSubj := o.apiSubj(fmt.Sprintf(apiStreamDeleteT, name))
	r, err := js.apiRequestWithContext(o.ctx, dsSubj, nil)
	if err != nil {
		return err
	}
	var resp streamDeleteResponse
	if err := json.Unmarshal(r.Data, &resp); err != nil {
		return err
	}

	if resp.Error != nil {
		if errors.Is(resp.Error, ErrStreamNotFound) {
			return ErrStreamNotFound
		}
		return resp.Error
	}
	return nil
}

type apiMsgGetRequest struct {
	Seq     uint64 `json:"seq,omitempty"`
	LastFor string `json:"last_by_subj,omitempty"`
	NextFor string `json:"next_by_subj,omitempty"`
}

// RawStreamMsg is a raw message stored in JetStream.
type RawStreamMsg struct {
	Subject  string
	Sequence uint64
	Header   Header
	Data     []byte
	Time     time.Time
}

// storedMsg is a raw message stored in JetStream.
type storedMsg struct {
	Subject  string    `json:"subject"`
	Sequence uint64    `json:"seq"`
	Header   []byte    `json:"hdrs,omitempty"`
	Data     []byte    `json:"data,omitempty"`
	Time     time.Time `json:"time"`
}

// apiMsgGetResponse is the response for a Stream get request.
type apiMsgGetResponse struct {
	apiResponse
	Message *storedMsg `json:"message,omitempty"`
}

// GetLastMsg retrieves the last raw stream message stored in JetStream by subject.
func (js *js) GetLastMsg(name, subject string, opts ...JSOpt) (*RawStreamMsg, error) {
	return js.getMsg(name, &apiMsgGetRequest{LastFor: subject}, opts...)
}

// GetMsg retrieves a raw stream message stored in JetStream by sequence number.
func (js *js) GetMsg(name string, seq uint64, opts ...JSOpt) (*RawStreamMsg, error) {
	return js.getMsg(name, &apiMsgGetRequest{Seq: seq}, opts...)
}

// Low level getMsg
func (js *js) getMsg(name string, mreq *apiMsgGetRequest, opts ...JSOpt) (*RawStreamMsg, error) {
	o, cancel, err := getJSContextOpts(js.opts, opts...)
	if err != nil {
		return nil, err
	}
	if cancel != nil {
		defer cancel()
	}

	if err := checkStreamName(name); err != nil {
		return nil, err
	}

	var apiSubj string
	if o.directGet && mreq.LastFor != _EMPTY_ {
		apiSubj = apiDirectMsgGetLastBySubjectT
		dsSubj := o.apiSubj(fmt.Sprintf(apiSubj, name, mreq.LastFor))
		r, err := js.apiRequestWithContext(o.ctx, dsSubj, nil)
		if err != nil {
			return nil, err
		}
		return convertDirectGetMsgResponseToMsg(name, r)
	}

	if o.directGet {
		apiSubj = apiDirectMsgGetT
		mreq.NextFor = o.directNextFor
	} else {
		apiSubj = apiMsgGetT
	}

	req, err := json.Marshal(mreq)
	if err != nil {
		return nil, err
	}

	dsSubj := o.apiSubj(fmt.Sprintf(apiSubj, name))
	r, err := js.apiRequestWithContext(o.ctx, dsSubj, req)
	if err != nil {
		return nil, err
	}

	if o.directGet {
		return convertDirectGetMsgResponseToMsg(name, r)
	}

	var resp apiMsgGetResponse
	if err := json.Unmarshal(r.Data, &resp); err != nil {
		return nil, err
	}
	if resp.Error != nil {
		if errors.Is(resp.Error, ErrMsgNotFound) {
			return nil, ErrMsgNotFound
		}
		if errors.Is(resp.Error, ErrStreamNotFound) {
			return nil, ErrStreamNotFound
		}
		return nil, resp.Error
	}

	msg := resp.Message

	var hdr Header
	if len(msg.Header) > 0 {
		hdr, err = DecodeHeadersMsg(msg.Header)
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

func convertDirectGetMsgResponseToMsg(name string, r *Msg) (*RawStreamMsg, error) {
	// Check for 404/408. We would get a no-payload message and a "Status" header
	if len(r.Data) == 0 {
		val := r.Header.Get(statusHdr)
		if val != _EMPTY_ {
			switch val {
			case noMessagesSts:
				return nil, ErrMsgNotFound
			default:
				desc := r.Header.Get(descrHdr)
				if desc == _EMPTY_ {
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
	stream := r.Header.Get(JSStream)
	if stream == _EMPTY_ {
		return nil, errors.New("nats: missing stream header")
	}

	// Mirrors can now answer direct gets, so removing check for name equality.
	// TODO(dlc) - We could have server also have a header with origin and check that?

	seqStr := r.Header.Get(JSSequence)
	if seqStr == _EMPTY_ {
		return nil, errors.New("nats: missing sequence header")
	}
	seq, err := strconv.ParseUint(seqStr, 10, 64)
	if err != nil {
		return nil, fmt.Errorf("nats: invalid sequence header '%s': %v", seqStr, err)
	}
	timeStr := r.Header.Get(JSTimeStamp)
	if timeStr == _EMPTY_ {
		return nil, errors.New("nats: missing timestamp header")
	}
	// Temporary code: the server in main branch is sending with format
	// "2006-01-02 15:04:05.999999999 +0000 UTC", but will be changed
	// to use format RFC3339Nano. Because of server test deps/cycle,
	// support both until the server PR lands.
	tm, err := time.Parse(time.RFC3339Nano, timeStr)
	if err != nil {
		tm, err = time.Parse("2006-01-02 15:04:05.999999999 +0000 UTC", timeStr)
		if err != nil {
			return nil, fmt.Errorf("nats: invalid timestamp header '%s': %v", timeStr, err)
		}
	}
	subj := r.Header.Get(JSSubject)
	if subj == _EMPTY_ {
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

type msgDeleteRequest struct {
	Seq     uint64 `json:"seq"`
	NoErase bool   `json:"no_erase,omitempty"`
}

// msgDeleteResponse is the response for a Stream delete request.
type msgDeleteResponse struct {
	apiResponse
	Success bool `json:"success,omitempty"`
}

// DeleteMsg deletes a message from a stream.
// The message is marked as erased, but not overwritten
func (js *js) DeleteMsg(name string, seq uint64, opts ...JSOpt) error {
	o, cancel, err := getJSContextOpts(js.opts, opts...)
	if err != nil {
		return err
	}
	if cancel != nil {
		defer cancel()
	}

	return js.deleteMsg(o, name, &msgDeleteRequest{Seq: seq, NoErase: true})
}

// SecureDeleteMsg deletes a message from a stream. The deleted message is overwritten with random data
// As a result, this operation is slower than DeleteMsg()
func (js *js) SecureDeleteMsg(name string, seq uint64, opts ...JSOpt) error {
	o, cancel, err := getJSContextOpts(js.opts, opts...)
	if err != nil {
		return err
	}
	if cancel != nil {
		defer cancel()
	}

	return js.deleteMsg(o, name, &msgDeleteRequest{Seq: seq})
}

func (js *js) deleteMsg(o *jsOpts, stream string, req *msgDeleteRequest) error {
	if err := checkStreamName(stream); err != nil {
		return err
	}
	reqJSON, err := json.Marshal(req)
	if err != nil {
		return err
	}

	dsSubj := o.apiSubj(fmt.Sprintf(apiMsgDeleteT, stream))
	r, err := js.apiRequestWithContext(o.ctx, dsSubj, reqJSON)
	if err != nil {
		return err
	}
	var resp msgDeleteResponse
	if err := json.Unmarshal(r.Data, &resp); err != nil {
		return err
	}
	if resp.Error != nil {
		return resp.Error
	}
	return nil
}

// StreamPurgeRequest is optional request information to the purge API.
type StreamPurgeRequest struct {
	// Purge up to but not including sequence.
	Sequence uint64 `json:"seq,omitempty"`
	// Subject to match against messages for the purge command.
	Subject string `json:"filter,omitempty"`
	// Number of messages to keep.
	Keep uint64 `json:"keep,omitempty"`
}

type streamPurgeResponse struct {
	apiResponse
	Success bool   `json:"success,omitempty"`
	Purged  uint64 `json:"purged"`
}

// PurgeStream purges messages on a Stream.
func (js *js) PurgeStream(stream string, opts ...JSOpt) error {
	if err := checkStreamName(stream); err != nil {
		return err
	}
	var req *StreamPurgeRequest
	var ok bool
	for _, opt := range opts {
		// For PurgeStream, only request body opt is relevant
		if req, ok = opt.(*StreamPurgeRequest); ok {
			break
		}
	}
	return js.purgeStream(stream, req)
}

func (js *js) purgeStream(stream string, req *StreamPurgeRequest, opts ...JSOpt) error {
	o, cancel, err := getJSContextOpts(js.opts, opts...)
	if err != nil {
		return err
	}
	if cancel != nil {
		defer cancel()
	}

	var b []byte
	if req != nil {
		if b, err = json.Marshal(req); err != nil {
			return err
		}
	}

	psSubj := o.apiSubj(fmt.Sprintf(apiStreamPurgeT, stream))
	r, err := js.apiRequestWithContext(o.ctx, psSubj, b)
	if err != nil {
		return err
	}
	var resp streamPurgeResponse
	if err := json.Unmarshal(r.Data, &resp); err != nil {
		return err
	}
	if resp.Error != nil {
		if errors.Is(resp.Error, ErrBadRequest) {
			return fmt.Errorf("%w: %s", ErrBadRequest, "invalid purge request body")
		}
		return resp.Error
	}
	return nil
}

// streamLister fetches pages of StreamInfo objects. This object is not safe
// to use for multiple threads.
type streamLister struct {
	js   *js
	page []*StreamInfo
	err  error

	offset   int
	pageInfo *apiPaged
}

// streamListResponse list of detailed stream information.
// A nil request is valid and means all streams.
type streamListResponse struct {
	apiResponse
	apiPaged
	Streams []*StreamInfo `json:"streams"`
}

// streamNamesRequest is used for Stream Name requests.
type streamNamesRequest struct {
	apiPagedRequest
	// These are filters that can be applied to the list.
	Subject string `json:"subject,omitempty"`
}

// Next fetches the next StreamInfo page.
func (s *streamLister) Next() bool {
	if s.err != nil {
		return false
	}
	if s.pageInfo != nil && s.offset >= s.pageInfo.Total {
		return false
	}

	req, err := json.Marshal(streamNamesRequest{
		apiPagedRequest: apiPagedRequest{Offset: s.offset},
		Subject:         s.js.opts.streamListSubject,
	})
	if err != nil {
		s.err = err
		return false
	}

	var cancel context.CancelFunc
	ctx := s.js.opts.ctx
	if ctx == nil {
		ctx, cancel = context.WithTimeout(context.Background(), s.js.opts.wait)
		defer cancel()
	}

	slSubj := s.js.apiSubj(apiStreamListT)
	r, err := s.js.apiRequestWithContext(ctx, slSubj, req)
	if err != nil {
		s.err = err
		return false
	}
	var resp streamListResponse
	if err := json.Unmarshal(r.Data, &resp); err != nil {
		s.err = err
		return false
	}
	if resp.Error != nil {
		s.err = resp.Error
		return false
	}

	s.pageInfo = &resp.apiPaged
	s.page = resp.Streams
	s.offset += len(s.page)
	return true
}

// Page returns the current StreamInfo page.
func (s *streamLister) Page() []*StreamInfo {
	return s.page
}

// Err returns any errors found while fetching pages.
func (s *streamLister) Err() error {
	return s.err
}

// Streams can be used to retrieve a list of StreamInfo objects.
func (jsc *js) Streams(opts ...JSOpt) <-chan *StreamInfo {
	o, cancel, err := getJSContextOpts(jsc.opts, opts...)
	if err != nil {
		return nil
	}

	ch := make(chan *StreamInfo)
	l := &streamLister{js: &js{nc: jsc.nc, opts: o}}
	go func() {
		if cancel != nil {
			defer cancel()
		}
		defer close(ch)
		for l.Next() {
			for _, info := range l.Page() {
				select {
				case ch <- info:
				case <-o.ctx.Done():
					return
				}
			}
		}
	}()

	return ch
}

// StreamsInfo can be used to retrieve a list of StreamInfo objects.
// Deprecated: Use Streams() instead.
func (jsc *js) StreamsInfo(opts ...JSOpt) <-chan *StreamInfo {
	return jsc.Streams(opts...)
}

type streamNamesLister struct {
	js *js

	err      error
	offset   int
	page     []string
	pageInfo *apiPaged
}

// Next fetches the next stream names page.
func (l *streamNamesLister) Next() bool {
	if l.err != nil {
		return false
	}
	if l.pageInfo != nil && l.offset >= l.pageInfo.Total {
		return false
	}

	var cancel context.CancelFunc
	ctx := l.js.opts.ctx
	if ctx == nil {
		ctx, cancel = context.WithTimeout(context.Background(), l.js.opts.wait)
		defer cancel()
	}

	req, err := json.Marshal(streamNamesRequest{
		apiPagedRequest: apiPagedRequest{Offset: l.offset},
		Subject:         l.js.opts.streamListSubject,
	})
	if err != nil {
		l.err = err
		return false
	}
	r, err := l.js.apiRequestWithContext(ctx, l.js.apiSubj(apiStreams), req)
	if err != nil {
		l.err = err
		return false
	}
	var resp streamNamesResponse
	if err := json.Unmarshal(r.Data, &resp); err != nil {
		l.err = err
		return false
	}
	if resp.Error != nil {
		l.err = resp.Error
		return false
	}

	l.pageInfo = &resp.apiPaged
	l.page = resp.Streams
	l.offset += len(l.page)
	return true
}

// Page returns the current ConsumerInfo page.
func (l *streamNamesLister) Page() []string {
	return l.page
}

// Err returns any errors found while fetching pages.
func (l *streamNamesLister) Err() error {
	return l.err
}

// StreamNames is used to retrieve a list of Stream names.
func (jsc *js) StreamNames(opts ...JSOpt) <-chan string {
	o, cancel, err := getJSContextOpts(jsc.opts, opts...)
	if err != nil {
		return nil
	}

	ch := make(chan string)
	l := &streamNamesLister{js: &js{nc: jsc.nc, opts: o}}
	go func() {
		if cancel != nil {
			defer cancel()
		}
		defer close(ch)
		for l.Next() {
			for _, info := range l.Page() {
				select {
				case ch <- info:
				case <-o.ctx.Done():
					return
				}
			}
		}
	}()

	return ch
}

// StreamNameBySubject returns a stream name that matches the subject.
func (jsc *js) StreamNameBySubject(subj string, opts ...JSOpt) (string, error) {
	o, cancel, err := getJSContextOpts(jsc.opts, opts...)
	if err != nil {
		return "", err
	}
	if cancel != nil {
		defer cancel()
	}

	var slr streamNamesResponse
	req := &streamRequest{subj}
	j, err := json.Marshal(req)
	if err != nil {
		return _EMPTY_, err
	}

	resp, err := jsc.apiRequestWithContext(o.ctx, o.apiSubj(apiStreams), j)
	if err != nil {
		if errors.Is(err, ErrNoResponders) {
			err = ErrJetStreamNotEnabled
		}
		return _EMPTY_, err
	}
	if err := json.Unmarshal(resp.Data, &slr); err != nil {
		return _EMPTY_, err
	}

	if slr.Error != nil || len(slr.Streams) != 1 {
		return _EMPTY_, ErrNoMatchingStream
	}
	return slr.Streams[0], nil
}

func getJSContextOpts(defs *jsOpts, opts ...JSOpt) (*jsOpts, context.CancelFunc, error) {
	var o jsOpts
	for _, opt := range opts {
		if err := opt.configureJSContext(&o); err != nil {
			return nil, nil, err
		}
	}

	// Check for option collisions. Right now just timeout and context.
	if o.ctx != nil && o.wait != 0 {
		return nil, nil, ErrContextAndTimeout
	}
	if o.wait == 0 && o.ctx == nil {
		o.wait = defs.wait
	}
	var cancel context.CancelFunc
	if o.ctx == nil && o.wait > 0 {
		o.ctx, cancel = context.WithTimeout(context.Background(), o.wait)
	}
	if o.pre == _EMPTY_ {
		o.pre = defs.pre
	}
	if o.ctx != nil {
		// if context does not have a deadline, use timeout from js context
		if _, hasDeadline := o.ctx.Deadline(); !hasDeadline {
			o.ctx, cancel = context.WithTimeout(o.ctx, defs.wait)
		}
	}
	return &o, cancel, nil
}
