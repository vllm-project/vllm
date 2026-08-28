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
	"encoding/json"
	"fmt"
	"time"
)

type (
	// ConsumerInfo is the detailed information about a JetStream consumer.
	ConsumerInfo struct {
		// Stream specifies the name of the stream that the consumer is bound
		// to.
		Stream string `json:"stream_name"`

		// Name represents the unique identifier for the consumer. This can be
		// either set explicitly by the client or generated automatically if not
		// set.
		Name string `json:"name"`

		// Created is the timestamp when the consumer was created.
		Created time.Time `json:"created"`

		// Config contains the configuration settings of the consumer, set when
		// creating or updating the consumer.
		Config ConsumerConfig `json:"config"`

		// Delivered holds information about the most recently delivered
		// message, including its sequence numbers and timestamp.
		Delivered SequenceInfo `json:"delivered"`

		// AckFloor indicates the message before the first unacknowledged
		// message.
		AckFloor SequenceInfo `json:"ack_floor"`

		// NumAckPending is the number of messages that have been delivered but
		// not yet acknowledged.
		NumAckPending int `json:"num_ack_pending"`

		// NumRedelivered counts the number of messages that have been
		// redelivered and not yet acknowledged. Each message is counted only
		// once, even if it has been redelivered multiple times. This count is
		// reset when the message is eventually acknowledged.
		NumRedelivered int `json:"num_redelivered"`

		// NumWaiting is the count of active pull requests. It is only relevant
		// for pull-based consumers.
		NumWaiting int `json:"num_waiting"`

		// NumPending is the number of messages that match the consumer's
		// filter, but have not been delivered yet.
		NumPending uint64 `json:"num_pending"`

		// Cluster contains information about the cluster to which this consumer
		// belongs (if applicable).
		Cluster *ClusterInfo `json:"cluster,omitempty"`

		// PushBound indicates whether at least one subscription exists for the
		// delivery subject of this consumer. This is only applicable to
		// push-based consumers.
		PushBound bool `json:"push_bound,omitempty"`

		// TimeStamp indicates when the info was gathered by the server.
		TimeStamp time.Time `json:"ts"`

		// PriorityGroups contains the information about the currently defined priority groups
		PriorityGroups []PriorityGroupState `json:"priority_groups,omitempty"`

		// Paused indicates whether the consumer is paused.
		Paused bool `json:"paused,omitempty"`

		// PauseRemaining contains the amount of time left until the consumer
		// unpauses. It will only be non-zero if the consumer is currently paused.
		PauseRemaining time.Duration `json:"pause_remaining,omitempty"`
	}

	PriorityGroupState struct {
		// Group this status is for.
		Group string `json:"group"`

		// PinnedClientID is the generated ID of the pinned client.
		PinnedClientID string `json:"pinned_client_id,omitempty"`

		// PinnedTS is the timestamp when the client was pinned.
		PinnedTS time.Time `json:"pinned_ts,omitempty"`
	}

	// ConsumerConfig represents the configuration of a JetStream consumer,
	// encompassing both push and pull consumer settings
	ConsumerConfig struct {
		// Name is an optional name for the consumer. If not set, one is
		// generated automatically.
		//
		// Name cannot contain whitespace, ., *, >, path separators (forward or
		// backwards slash), and non-printable characters.
		Name string `json:"name,omitempty"`

		// Durable is an optional durable name for the consumer. If both Durable
		// and Name are set, they have to be equal. Unless InactiveThreshold is set, a
		// durable consumer will not be cleaned up automatically.
		//
		// Durable cannot contain whitespace, ., *, >, path separators (forward or
		// backwards slash), and non-printable characters.
		Durable string `json:"durable_name,omitempty"`

		// Description provides an optional description of the consumer.
		Description string `json:"description,omitempty"`

		// DeliverPolicy defines from which point to start delivering messages
		// from the stream. Defaults to DeliverAllPolicy.
		DeliverPolicy DeliverPolicy `json:"deliver_policy"`

		// OptStartSeq is an optional sequence number from which to start
		// message delivery. Only applicable when DeliverPolicy is set to
		// DeliverByStartSequencePolicy.
		OptStartSeq uint64 `json:"opt_start_seq,omitempty"`

		// OptStartTime is an optional time from which to start message
		// delivery. Only applicable when DeliverPolicy is set to
		// DeliverByStartTimePolicy.
		OptStartTime *time.Time `json:"opt_start_time,omitempty"`

		// AckPolicy defines the acknowledgement policy for the consumer.
		// Defaults to AckExplicitPolicy.
		AckPolicy AckPolicy `json:"ack_policy"`

		// AckWait defines how long the server will wait for an acknowledgement
		// before resending a message. If not set, server default is 30 seconds.
		AckWait time.Duration `json:"ack_wait,omitempty"`

		// MaxDeliver defines the maximum number of delivery attempts for a
		// message. Applies to any message that is re-sent due to ack policy.
		//  If not set, server default is -1 (unlimited).
		MaxDeliver int `json:"max_deliver,omitempty"`

		// BackOff specifies the optional back-off intervals for retrying
		// message delivery after a failed acknowledgement. It overrides
		// AckWait.
		//
		// BackOff only applies to messages not acknowledged in specified time,
		// not messages that were nack'ed.
		//
		// The number of intervals specified must be lower or equal to
		// MaxDeliver. If the number of intervals is lower, the last interval is
		// used for all remaining attempts.
		BackOff []time.Duration `json:"backoff,omitempty"`

		// FilterSubject can be used to filter messages delivered from the
		// stream. FilterSubject is exclusive with FilterSubjects.
		FilterSubject string `json:"filter_subject,omitempty"`

		// ReplayPolicy defines the rate at which messages are sent to the
		// consumer. If ReplayOriginalPolicy is set, messages are sent in the
		// same intervals in which they were stored on stream. This can be used
		// e.g. to simulate production traffic in development environments. If
		// ReplayInstantPolicy is set, messages are sent as fast as possible.
		// Defaults to ReplayInstantPolicy.
		ReplayPolicy ReplayPolicy `json:"replay_policy"`

		// RateLimit specifies an optional maximum rate of message delivery in
		// bits per second.
		RateLimit uint64 `json:"rate_limit_bps,omitempty"`

		// SampleFrequency is an optional frequency for sampling how often
		// acknowledgements are sampled for observability. See
		// https://docs.nats.io/running-a-nats-service/nats_admin/monitoring/monitoring_jetstream
		SampleFrequency string `json:"sample_freq,omitempty"`

		// MaxWaiting is a maximum number of pull requests waiting to be
		// fulfilled. If not set, this will inherit settings from stream's
		// ConsumerLimits or (if those are not set) from account settings.  If
		// neither are set, server default is 512.
		MaxWaiting int `json:"max_waiting,omitempty"`

		// MaxAckPending is a maximum number of outstanding unacknowledged
		// messages. Once this limit is reached, the server will suspend sending
		// messages to the consumer. If not set, server default is 1000.
		// Set to -1 for unlimited.
		MaxAckPending int `json:"max_ack_pending,omitempty"`

		// HeadersOnly indicates whether only headers of messages should be sent
		// (and no payload). Defaults to false.
		HeadersOnly bool `json:"headers_only,omitempty"`

		// MaxRequestBatch is the optional maximum batch size a single pull
		// request can make. When set with MaxRequestMaxBytes, the batch size
		// will be constrained by whichever limit is hit first.
		MaxRequestBatch int `json:"max_batch,omitempty"`

		// MaxRequestExpires is the maximum duration a single pull request will
		// wait for messages to be available to pull.
		MaxRequestExpires time.Duration `json:"max_expires,omitempty"`

		// MaxRequestMaxBytes is the optional maximum total bytes that can be
		// requested in a given batch. When set with MaxRequestBatch, the batch
		// size will be constrained by whichever limit is hit first.
		MaxRequestMaxBytes int `json:"max_bytes,omitempty"`

		// InactiveThreshold is a duration which instructs the server to clean
		// up the consumer if it has been inactive for the specified duration.
		// Durable consumers will not be cleaned up by default, but if
		// InactiveThreshold is set, they will be. If not set, this will inherit
		// settings from stream's ConsumerLimits. If neither are set, server
		// default is 5 seconds.
		//
		// A consumer is considered inactive if no pull requests are received by
		// the server (for pull consumers), or no interest is detected on the
		// deliver subject (for push consumers), not if there are no
		// messages to be delivered.
		InactiveThreshold time.Duration `json:"inactive_threshold,omitempty"`

		// Replicas is the number of replicas for the consumer's state. By
		// default, consumers inherit the number of replicas from the stream.
		Replicas int `json:"num_replicas"`

		// MemoryStorage is a flag to force the consumer to use memory storage
		// rather than inherit the storage type from the stream.
		MemoryStorage bool `json:"mem_storage,omitempty"`

		// FilterSubjects allows filtering messages from a stream by subject.
		// This field is exclusive with FilterSubject. Requires nats-server
		// v2.10.0 or later.
		FilterSubjects []string `json:"filter_subjects,omitempty"`

		// Metadata is a set of application-defined key-value pairs for
		// associating metadata on the consumer. This feature requires
		// nats-server v2.10.0 or later.
		Metadata map[string]string `json:"metadata,omitempty"`

		// PauseUntil is for suspending the consumer until the deadline.
		PauseUntil *time.Time `json:"pause_until,omitempty"`

		// PriorityPolicy represents the priority policy the consumer is set to.
		// Requires nats-server v2.11.0 or later.
		PriorityPolicy PriorityPolicy `json:"priority_policy,omitempty"`

		// PinnedTTL represents the time after which the client will be unpinned
		// if no new pull requests are sent. Used with PriorityPolicyPinned.
		// Requires nats-server v2.11.0 or later.
		PinnedTTL time.Duration `json:"priority_timeout,omitempty"`

		// PriorityGroups is a list of priority groups this consumer supports.
		PriorityGroups []string `json:"priority_groups,omitempty"`

		// Fields specific to push consumers:

		// DeliverSubject is the subject to deliver messages to for push consumers
		DeliverSubject string `json:"deliver_subject,omitempty"`

		// DeliverGroup is the group name for push consumers
		DeliverGroup string `json:"deliver_group,omitempty"`

		// FlowControl is a flag to enable flow control for the consumer.
		// When set, the server will regularly send an empty message with status
		// header 100 and a reply subject. Consumers must reply to these
		// messages to control the rate of message delivery.
		FlowControl bool `json:"flow_control,omitempty"`

		// IdleHeartbeat enables push consumer idle heartbeat messages.
		// If the Consumer is idle for more than the set value, an empty message
		// with Status header 100 will be sent indicating the consumer is still
		// alive.
		IdleHeartbeat time.Duration `json:"idle_heartbeat,omitempty"`
	}

	// OrderedConsumerConfig is the configuration of an ordered JetStream
	// consumer. For more information, see [Ordered Consumers] in README
	//
	// [Ordered Consumers]: https://github.com/nats-io/nats.go/blob/main/jetstream/README.md#ordered-consumers
	OrderedConsumerConfig struct {
		// FilterSubjects allows filtering messages from a stream by subject.
		// This field is exclusive with FilterSubject. Requires nats-server
		// v2.10.0 or later.
		FilterSubjects []string `json:"filter_subjects,omitempty"`

		// DeliverPolicy defines from which point to start delivering messages
		// from the stream. Defaults to DeliverAllPolicy.
		DeliverPolicy DeliverPolicy `json:"deliver_policy"`

		// OptStartSeq is an optional sequence number from which to start
		// message delivery. Only applicable when DeliverPolicy is set to
		// DeliverByStartSequencePolicy.
		OptStartSeq uint64 `json:"opt_start_seq,omitempty"`

		// OptStartTime is an optional time from which to start message
		// delivery. Only applicable when DeliverPolicy is set to
		// DeliverByStartTimePolicy.
		OptStartTime *time.Time `json:"opt_start_time,omitempty"`

		// ReplayPolicy defines the rate at which messages are sent to the
		// consumer. If ReplayOriginalPolicy is set, messages are sent in the
		// same intervals in which they were stored on the stream. This can be
		// used e.g. to simulate production traffic in development environments.
		// If ReplayInstantPolicy is set, messages are sent as fast as possible.
		// Defaults to ReplayInstantPolicy.
		ReplayPolicy ReplayPolicy `json:"replay_policy"`

		// InactiveThreshold is a duration which instructs the server to clean
		// up the consumer if it has been inactive for the specified duration.
		// Defaults to 5m.
		InactiveThreshold time.Duration `json:"inactive_threshold,omitempty"`

		// HeadersOnly indicates whether only headers of messages should be sent
		// (and no payload). Defaults to false.
		HeadersOnly bool `json:"headers_only,omitempty"`

		// MaxResetAttempts is the maximum number of attempts to recreate the
		// consumer in a single recovery cycle. Defaults to unlimited.
		MaxResetAttempts int

		// Metadata is a set of application-defined key-value pairs for
		// associating metadata with the consumer. This feature requires
		// nats-server v2.10.0 or later.
		Metadata map[string]string `json:"metadata,omitempty"`

		// NamePrefix is an optional custom prefix for the consumer name.
		// If provided, ordered consumer names will be generated as:
		// {NamePrefix}_{sequence_number} (e.g., "custom_1", "custom_2").
		// If not provided, a unique ID (NUID) will be used as the prefix.
		NamePrefix string `json:"-"`
	}

	// DeliverPolicy determines from which point to start delivering messages.
	DeliverPolicy int

	// AckPolicy determines how the consumer should acknowledge delivered
	// messages.
	AckPolicy int

	// ReplayPolicy determines how the consumer should replay messages it
	// already has queued in the stream.
	ReplayPolicy int

	// SequenceInfo has both the consumer and the stream sequence and last
	// activity.
	SequenceInfo struct {
		Consumer uint64     `json:"consumer_seq"`
		Stream   uint64     `json:"stream_seq"`
		Last     *time.Time `json:"last_active,omitempty"`
	}

	// PriorityPolicy determines the priority policy the consumer is set to.
	PriorityPolicy int
)

const (
	// PriorityPolicyNone is the default priority policy.
	PriorityPolicyNone PriorityPolicy = iota

	// PriorityPolicyPinned is the priority policy that pins a consumer to a
	// specific client.
	PriorityPolicyPinned

	// PriorityPolicyOverflow is the priority policy that allows for
	// restricting when a consumer will receive messages based on the number of
	// pending messages or acks.
	PriorityPolicyOverflow

	// PriorityPolicyPrioritized is the priority policy that allows for the
	// server to deliver messages to clients based on their priority (instead
	// of round-robin). Requires nats-server v2.12.0 or later.
	PriorityPolicyPrioritized
)

func (p *PriorityPolicy) UnmarshalJSON(data []byte) error {
	switch string(data) {
	case jsonString(""):
		*p = PriorityPolicyNone
	case jsonString("pinned_client"):
		*p = PriorityPolicyPinned
	case jsonString("overflow"):
		*p = PriorityPolicyOverflow
	case jsonString("prioritized"):
		*p = PriorityPolicyPrioritized
	default:
		return fmt.Errorf("nats: cannot unmarshal %q", data)
	}
	return nil
}

func (p PriorityPolicy) MarshalJSON() ([]byte, error) {
	switch p {
	case PriorityPolicyNone:
		return json.Marshal("")
	case PriorityPolicyPinned:
		return json.Marshal("pinned_client")
	case PriorityPolicyOverflow:
		return json.Marshal("overflow")
	case PriorityPolicyPrioritized:
		return json.Marshal("prioritized")
	}
	return nil, fmt.Errorf("nats: unknown priority policy %v", p)
}

const (
	// DeliverAllPolicy starts delivering messages from the very beginning of a
	// stream. This is the default.
	DeliverAllPolicy DeliverPolicy = iota

	// DeliverLastPolicy will start the consumer with the last sequence
	// received.
	DeliverLastPolicy

	// DeliverNewPolicy will only deliver new messages that are sent after the
	// consumer is created.
	DeliverNewPolicy

	// DeliverByStartSequencePolicy will deliver messages starting from a given
	// sequence configured with OptStartSeq in ConsumerConfig.
	DeliverByStartSequencePolicy

	// DeliverByStartTimePolicy will deliver messages starting from a given time
	// configured with OptStartTime in ConsumerConfig.
	DeliverByStartTimePolicy

	// DeliverLastPerSubjectPolicy will start the consumer with the last message
	// for all subjects received.
	DeliverLastPerSubjectPolicy
)

func (p *DeliverPolicy) UnmarshalJSON(data []byte) error {
	switch string(data) {
	case jsonString("all"), jsonString("undefined"):
		*p = DeliverAllPolicy
	case jsonString("last"):
		*p = DeliverLastPolicy
	case jsonString("new"):
		*p = DeliverNewPolicy
	case jsonString("by_start_sequence"):
		*p = DeliverByStartSequencePolicy
	case jsonString("by_start_time"):
		*p = DeliverByStartTimePolicy
	case jsonString("last_per_subject"):
		*p = DeliverLastPerSubjectPolicy
	default:
		return fmt.Errorf("nats: cannot unmarshal %q", data)
	}

	return nil
}

func (p DeliverPolicy) MarshalJSON() ([]byte, error) {
	switch p {
	case DeliverAllPolicy:
		return json.Marshal("all")
	case DeliverLastPolicy:
		return json.Marshal("last")
	case DeliverNewPolicy:
		return json.Marshal("new")
	case DeliverByStartSequencePolicy:
		return json.Marshal("by_start_sequence")
	case DeliverByStartTimePolicy:
		return json.Marshal("by_start_time")
	case DeliverLastPerSubjectPolicy:
		return json.Marshal("last_per_subject")
	}
	return nil, fmt.Errorf("nats: unknown deliver policy %v", p)
}

func (p DeliverPolicy) String() string {
	switch p {
	case DeliverAllPolicy:
		return "all"
	case DeliverLastPolicy:
		return "last"
	case DeliverNewPolicy:
		return "new"
	case DeliverByStartSequencePolicy:
		return "by_start_sequence"
	case DeliverByStartTimePolicy:
		return "by_start_time"
	case DeliverLastPerSubjectPolicy:
		return "last_per_subject"
	}
	return ""
}

const (
	// AckExplicitPolicy requires ack or nack for all messages.
	AckExplicitPolicy AckPolicy = iota

	// AckAllPolicy when acking a sequence number, this implicitly acks all
	// sequences below this one as well.
	AckAllPolicy

	// AckNonePolicy requires no acks for delivered messages.
	AckNonePolicy

	// AckFlowControlPolicy functions like AckAllPolicy, but acks based on
	// responses to flow control. Used by durable stream sourcing and
	// mirroring against an existing durable push consumer.
	//
	// This feature requires nats-server v2.14.0 or later.
	AckFlowControlPolicy
)

func (p *AckPolicy) UnmarshalJSON(data []byte) error {
	switch string(data) {
	case jsonString("none"):
		*p = AckNonePolicy
	case jsonString("all"):
		*p = AckAllPolicy
	case jsonString("explicit"):
		*p = AckExplicitPolicy
	case jsonString("flow_control"):
		*p = AckFlowControlPolicy
	default:
		return fmt.Errorf("nats: cannot unmarshal %q", data)
	}
	return nil
}

func (p AckPolicy) MarshalJSON() ([]byte, error) {
	switch p {
	case AckNonePolicy:
		return json.Marshal("none")
	case AckAllPolicy:
		return json.Marshal("all")
	case AckExplicitPolicy:
		return json.Marshal("explicit")
	case AckFlowControlPolicy:
		return json.Marshal("flow_control")
	}
	return nil, fmt.Errorf("nats: unknown acknowledgement policy %v", p)
}

func (p AckPolicy) String() string {
	switch p {
	case AckNonePolicy:
		return "AckNone"
	case AckAllPolicy:
		return "AckAll"
	case AckExplicitPolicy:
		return "AckExplicit"
	case AckFlowControlPolicy:
		return "AckFlowControl"
	}
	return "Unknown AckPolicy"
}

const (
	// ReplayInstantPolicy will replay messages as fast as possible.
	ReplayInstantPolicy ReplayPolicy = iota

	// ReplayOriginalPolicy will maintain the same timing as the messages were
	// received.
	ReplayOriginalPolicy
)

func (p *ReplayPolicy) UnmarshalJSON(data []byte) error {
	switch string(data) {
	case jsonString("instant"):
		*p = ReplayInstantPolicy
	case jsonString("original"):
		*p = ReplayOriginalPolicy
	default:
		return fmt.Errorf("nats: cannot unmarshal %q", data)
	}
	return nil
}

func (p ReplayPolicy) MarshalJSON() ([]byte, error) {
	switch p {
	case ReplayOriginalPolicy:
		return json.Marshal("original")
	case ReplayInstantPolicy:
		return json.Marshal("instant")
	}
	return nil, fmt.Errorf("nats: unknown replay policy %v", p)
}

func (p ReplayPolicy) String() string {
	switch p {
	case ReplayOriginalPolicy:
		return "original"
	case ReplayInstantPolicy:
		return "instant"
	}
	return ""
}
