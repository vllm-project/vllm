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
	"encoding/json"
	"errors"
	"fmt"
	"strings"
	"time"
)

type (
	// StreamInfo shows config and current state for this stream.
	StreamInfo struct {
		// Config contains the configuration settings of the stream, set when
		// creating or updating the stream.
		Config StreamConfig `json:"config"`

		// Created is the timestamp when the stream was created.
		Created time.Time `json:"created"`

		// State provides the state of the stream at the time of request,
		// including metrics like the number of messages in the stream, total
		// bytes, etc.
		State StreamState `json:"state"`

		// Cluster contains information about the cluster to which this stream
		// belongs (if applicable).
		Cluster *ClusterInfo `json:"cluster,omitempty"`

		// Mirror contains information about another stream this one is
		// mirroring. Mirroring is used to create replicas of another stream's
		// data. This field is omitted if the stream is not mirroring another
		// stream.
		Mirror *StreamSourceInfo `json:"mirror,omitempty"`

		// Sources is a list of source streams from which this stream collects
		// data.
		Sources []*StreamSourceInfo `json:"sources,omitempty"`

		// TimeStamp indicates when the info was gathered by the server.
		TimeStamp time.Time `json:"ts"`
	}

	// StreamConfig is the configuration of a JetStream stream.
	StreamConfig struct {
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

		// Template identifies the template that manages the Stream.
		// Deprecated: This feature is no longer supported.
		Template string `json:"template_owner,omitempty"`

		// AllowMsgTTL allows header initiated per-message TTLs.
		// This feature requires nats-server v2.11.0 or later.
		AllowMsgTTL bool `json:"allow_msg_ttl,omitempty"`

		// Enables and sets a duration for adding server markers for delete, purge and max age limits.
		// This feature requires nats-server v2.11.0 or later.
		SubjectDeleteMarkerTTL time.Duration `json:"subject_delete_marker_ttl,omitempty"`

		// AllowMsgCounter enables the feature
		AllowMsgCounter bool `json:"allow_msg_counter,omitempty"`

		// AllowAtomicPublish allows atomic batch publishing into the stream.
		AllowAtomicPublish bool `json:"allow_atomic,omitempty"`

		// AllowMsgSchedules enables the scheduling of messages
		AllowMsgSchedules bool `json:"allow_msg_schedules,omitempty"`

		// PersistMode allows to opt-in to different persistence mode settings.
		PersistMode PersistModeType `json:"persist_mode,omitempty"`

		// AllowBatchPublish allows fast batch publishing into the stream.
		// This feature requires nats-server v2.14.0 or later.
		AllowBatchPublish bool `json:"allow_batched,omitempty"`
	}

	// StreamSourceInfo shows information about an upstream stream
	// source/mirror.
	StreamSourceInfo struct {
		// Name is the name of the stream that is being replicated.
		Name string `json:"name"`

		// Lag informs how many messages behind the source/mirror operation is.
		// This will only show correctly if there is active communication
		// with stream/mirror.
		Lag uint64 `json:"lag"`

		// Active informs when last the mirror or sourced stream had activity.
		// Value will be -1 when there has been no activity.
		Active time.Duration `json:"active"`

		// FilterSubject is the subject filter defined for this source/mirror.
		FilterSubject string `json:"filter_subject,omitempty"`

		// SubjectTransforms is a list of subject transforms defined for this
		// source/mirror.
		SubjectTransforms []SubjectTransformConfig `json:"subject_transforms,omitempty"`
	}

	// StreamState is the state of a JetStream stream at the time of request.
	StreamState struct {
		// Msgs is the number of messages stored in the stream.
		Msgs uint64 `json:"messages"`

		// Bytes is the number of bytes stored in the stream.
		Bytes uint64 `json:"bytes"`

		// FirstSeq is the sequence number of the first message in the stream.
		FirstSeq uint64 `json:"first_seq"`

		// FirstTime is the timestamp of the first message in the stream.
		FirstTime time.Time `json:"first_ts"`

		// LastSeq is the sequence number of the last message in the stream.
		LastSeq uint64 `json:"last_seq"`

		// LastTime is the timestamp of the last message in the stream.
		LastTime time.Time `json:"last_ts"`

		// Consumers is the number of consumers on the stream.
		Consumers int `json:"consumer_count"`

		// Deleted is a list of sequence numbers that have been removed from the
		// stream. This field will only be returned if the stream has been
		// fetched with the DeletedDetails option.
		Deleted []uint64 `json:"deleted"`

		// NumDeleted is the number of messages that have been removed from the
		// stream. Only deleted messages causing a gap in stream sequence numbers
		// are counted. Messages deleted at the beginning or end of the stream
		// are not counted.
		NumDeleted int `json:"num_deleted"`

		// NumSubjects is the number of unique subjects the stream has received
		// messages on.
		NumSubjects uint64 `json:"num_subjects"`

		// Subjects is a map of subjects the stream has received messages on
		// with message count per subject. This field will only be returned if
		// the stream has been fetched with the SubjectFilter option.
		Subjects map[string]uint64 `json:"subjects"`
	}

	// ClusterInfo shows information about the underlying set of servers that
	// make up the stream or consumer.
	ClusterInfo struct {
		// Name is the name of the cluster.
		Name string `json:"name,omitempty"`

		// RaftGroup is the name of the Raft group managing the asset (in
		// clustered environments).
		RaftGroup string `json:"raft_group,omitempty"`

		// Leader is the server name of the RAFT leader.
		Leader string `json:"leader,omitempty"`

		// LeaderSince is the time that it was elected as leader in RFC3339
		// format, absent when not the leader.
		LeaderSince *time.Time `json:"leader_since,omitempty"`

		// SystemAcc indicates if the traffic_account is the system account.
		// When true, replication traffic goes over the system account.
		SystemAcc bool `json:"system_account,omitempty"`

		// TrafficAcc is the account where the replication traffic goes over.
		TrafficAcc string `json:"traffic_account,omitempty"`

		// Replicas is the list of members of the RAFT cluster.
		Replicas []*PeerInfo `json:"replicas,omitempty"`
	}

	// PeerInfo shows information about the peers in the cluster that are
	// supporting the stream or consumer.
	PeerInfo struct {
		// Name is the server name of the peer.
		Name string `json:"name"`

		// Current indicates if the peer is up to date and synchronized with the
		// leader.
		Current bool `json:"current"`

		// Offline indicates if the peer is considered offline by the group.
		Offline bool `json:"offline,omitempty"`

		// Active it the duration since this peer was last seen.
		Active time.Duration `json:"active"`

		// Lag is the number of uncommitted operations this peer is behind the
		// leader.
		Lag uint64 `json:"lag,omitempty"`
	}

	// SubjectTransformConfig is for applying a subject transform (to matching
	// messages) before doing anything else when a new message is received.
	SubjectTransformConfig struct {
		// Source is the subject pattern to match incoming messages against.
		Source string `json:"src"`

		// Destination is the subject pattern to remap the subject to.
		Destination string `json:"dest"`
	}

	// RePublish is for republishing messages once committed to a stream. The
	// original subject is remapped from the subject pattern to the destination
	// pattern.
	RePublish struct {
		// Source is the subject pattern to match incoming messages against.
		Source string `json:"src,omitempty"`

		// Destination is the subject pattern to republish the subject to.
		Destination string `json:"dest"`

		// HeadersOnly is a flag to indicate that only the headers should be
		// republished.
		HeadersOnly bool `json:"headers_only,omitempty"`
	}

	// Placement is used to guide placement of streams in clustered JetStream.
	Placement struct {
		// Cluster is the name of the cluster to which the stream should be
		// assigned.
		Cluster string `json:"cluster"`

		// Tags are used to match streams to servers in the cluster. A stream
		// will be assigned to a server with a matching tag.
		Tags []string `json:"tags,omitempty"`
	}

	// StreamSource dictates how streams can source from other streams.
	StreamSource struct {
		// Name is the name of the stream to source from.
		Name string `json:"name"`

		// OptStartSeq is the sequence number to start sourcing from.
		OptStartSeq uint64 `json:"opt_start_seq,omitempty"`

		// OptStartTime is the timestamp of messages to start sourcing from.
		OptStartTime *time.Time `json:"opt_start_time,omitempty"`

		// FilterSubject is the subject filter used to only replicate messages
		// with matching subjects.
		FilterSubject string `json:"filter_subject,omitempty"`

		// SubjectTransforms is a list of subject transforms to apply to
		// matching messages.
		//
		// Subject transforms on sources and mirrors are also used as subject
		// filters with optional transformations.
		SubjectTransforms []SubjectTransformConfig `json:"subject_transforms,omitempty"`

		// External is a configuration referencing a stream source in another
		// account or JetStream domain.
		External *ExternalStream `json:"external,omitempty"`

		// Consumer is consumer information for durable sourcing.
		//
		// This feature requires nats-server v2.14.0 or later.
		Consumer *StreamConsumerSource `json:"consumer,omitempty"`

		// Domain is used to configure a stream source in another JetStream
		// domain. This setting will set the External field with the appropriate
		// APIPrefix.
		Domain string `json:"-"`
	}

	// StreamConsumerSource is consumer information for durable sourcing.
	//
	// This feature requires nats-server v2.14.0 or later.
	StreamConsumerSource struct {
		// Name is the consumer name.
		Name string `json:"name,omitempty"`

		// DeliverSubject is the subject to deliver messages to.
		DeliverSubject string `json:"deliver_subject,omitempty"`
	}

	// ExternalStream allows you to qualify access to a stream source in another
	// account.
	ExternalStream struct {
		// APIPrefix is the subject prefix that imports the other account/domain
		// $JS.API.CONSUMER.> subjects.
		APIPrefix string `json:"api"`

		// DeliverPrefix is the delivery subject to use for the push consumer.
		DeliverPrefix string `json:"deliver"`
	}

	// StreamConsumerLimits are the limits for a consumer on a stream. These can
	// be overridden on a per consumer basis.
	StreamConsumerLimits struct {
		// InactiveThreshold is a duration which instructs the server to clean
		// up the consumer if it has been inactive for the specified duration.
		InactiveThreshold time.Duration `json:"inactive_threshold,omitempty"`

		// MaxAckPending is a maximum number of outstanding unacknowledged
		// messages for a consumer.
		MaxAckPending int `json:"max_ack_pending,omitempty"`
	}

	// DiscardPolicy determines how to proceed when limits of messages or bytes
	// are reached.
	DiscardPolicy int

	// RetentionPolicy determines how messages in a stream are retained.
	RetentionPolicy int

	// StorageType determines how messages are stored for retention.
	StorageType int

	// StoreCompression determines how messages are compressed.
	StoreCompression uint8

	// PersistModeType determines what persistence mode the stream uses.
	PersistModeType int
)

const (
	// LimitsPolicy (default) means that messages are retained until any given
	// limit is reached. This could be one of MaxMsgs, MaxBytes, or MaxAge.
	LimitsPolicy RetentionPolicy = iota

	// InterestPolicy specifies that when all known observables have
	// acknowledged a message it can be removed.
	InterestPolicy

	// WorkQueuePolicy specifies that when the first worker or subscriber
	// acknowledges the message it can be removed.
	WorkQueuePolicy
)

const (
	// DiscardOld will remove older messages to return to the limits. This is
	// the default.
	DiscardOld DiscardPolicy = iota

	// DiscardNew will fail to store new messages once the limits are reached.
	DiscardNew
)

const (
	limitsPolicyString    = "limits"
	interestPolicyString  = "interest"
	workQueuePolicyString = "workqueue"
)

const (
	// DefaultPersistMode specifies the default persist mode. Writes to the stream will immediately be flushed.
	// The publish acknowledgement will be sent after the persisting completes.
	DefaultPersistMode = PersistModeType(iota)
	// AsyncPersistMode specifies writes to the stream will be flushed asynchronously.
	// The publish acknowledgement may be sent before the persisting completes.
	// This means writes could be lost if they weren't flushed prior to a hard kill of the server.
	AsyncPersistMode
)

func (rp RetentionPolicy) String() string {
	switch rp {
	case LimitsPolicy:
		return "Limits"
	case InterestPolicy:
		return "Interest"
	case WorkQueuePolicy:
		return "WorkQueue"
	default:
		return "Unknown Retention Policy"
	}
}

func (rp RetentionPolicy) MarshalJSON() ([]byte, error) {
	switch rp {
	case LimitsPolicy:
		return json.Marshal(limitsPolicyString)
	case InterestPolicy:
		return json.Marshal(interestPolicyString)
	case WorkQueuePolicy:
		return json.Marshal(workQueuePolicyString)
	default:
		return nil, fmt.Errorf("nats: can not marshal %v", rp)
	}
}

func (rp *RetentionPolicy) UnmarshalJSON(data []byte) error {
	switch string(data) {
	case jsonString(limitsPolicyString):
		*rp = LimitsPolicy
	case jsonString(interestPolicyString):
		*rp = InterestPolicy
	case jsonString(workQueuePolicyString):
		*rp = WorkQueuePolicy
	default:
		return fmt.Errorf("nats: can not unmarshal %q", data)
	}
	return nil
}

func (dp DiscardPolicy) String() string {
	switch dp {
	case DiscardOld:
		return "DiscardOld"
	case DiscardNew:
		return "DiscardNew"
	default:
		return "Unknown Discard Policy"
	}
}

func (dp DiscardPolicy) MarshalJSON() ([]byte, error) {
	switch dp {
	case DiscardOld:
		return json.Marshal("old")
	case DiscardNew:
		return json.Marshal("new")
	default:
		return nil, fmt.Errorf("nats: can not marshal %v", dp)
	}
}

func (dp *DiscardPolicy) UnmarshalJSON(data []byte) error {
	switch strings.ToLower(string(data)) {
	case jsonString("old"):
		*dp = DiscardOld
	case jsonString("new"):
		*dp = DiscardNew
	default:
		return fmt.Errorf("nats: can not unmarshal %q", data)
	}
	return nil
}

func (pm PersistModeType) String() string {
	switch pm {
	case DefaultPersistMode:
		return "Default"
	case AsyncPersistMode:
		return "Async"
	default:
		return "Unknown Persist Mode"
	}
}

func (pm PersistModeType) MarshalJSON() ([]byte, error) {
	switch pm {
	case DefaultPersistMode:
		return json.Marshal("default")
	case AsyncPersistMode:
		return json.Marshal("async")
	default:
		return nil, fmt.Errorf("nats: can not marshal %v", pm)
	}
}

func (pm *PersistModeType) UnmarshalJSON(data []byte) error {
	switch strings.ToLower(string(data)) {
	case jsonString("default"):
		*pm = DefaultPersistMode
	case jsonString("async"):
		*pm = AsyncPersistMode
	default:
		return fmt.Errorf("nats: can not unmarshal %q", data)
	}
	return nil
}

const (
	// FileStorage specifies on disk storage. It's the default.
	FileStorage StorageType = iota
	// MemoryStorage specifies in memory only.
	MemoryStorage
)

const (
	memoryStorageString = "memory"
	fileStorageString   = "file"
)

func (st StorageType) String() string {
	switch st {
	case MemoryStorage:
		return "Memory"
	case FileStorage:
		return "File"
	default:
		return "Unknown Storage Type"
	}
}

func (st StorageType) MarshalJSON() ([]byte, error) {
	switch st {
	case MemoryStorage:
		return json.Marshal(memoryStorageString)
	case FileStorage:
		return json.Marshal(fileStorageString)
	default:
		return nil, fmt.Errorf("nats: can not marshal %v", st)
	}
}

func (st *StorageType) UnmarshalJSON(data []byte) error {
	switch string(data) {
	case jsonString(memoryStorageString):
		*st = MemoryStorage
	case jsonString(fileStorageString):
		*st = FileStorage
	default:
		return fmt.Errorf("nats: can not unmarshal %q", data)
	}
	return nil
}

func jsonString(s string) string {
	return "\"" + s + "\""
}

const (
	// NoCompression disables compression on the stream. This is the default.
	NoCompression StoreCompression = iota

	// S2Compression enables S2 compression on the stream.
	S2Compression
)

func (alg StoreCompression) String() string {
	switch alg {
	case NoCompression:
		return "None"
	case S2Compression:
		return "S2"
	default:
		return "Unknown StoreCompression"
	}
}

func (alg StoreCompression) MarshalJSON() ([]byte, error) {
	var str string
	switch alg {
	case S2Compression:
		str = "s2"
	case NoCompression:
		str = "none"
	default:
		return nil, errors.New("unknown compression algorithm")
	}
	return json.Marshal(str)
}

func (alg *StoreCompression) UnmarshalJSON(b []byte) error {
	var str string
	if err := json.Unmarshal(b, &str); err != nil {
		return err
	}
	switch str {
	case "s2":
		*alg = S2Compression
	case "none":
		*alg = NoCompression
	default:
		return errors.New("unknown compression algorithm")
	}
	return nil
}
