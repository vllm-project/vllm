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
	"errors"
	"fmt"
)

type (
	// JetStreamError is an error result that happens when using JetStream.
	// In case of client-side error, [APIError] returns nil.
	JetStreamError interface {
		APIError() *APIError
		error
	}

	jsError struct {
		apiErr  *APIError
		message string
	}

	// APIError is included in all API responses if there was an error.
	APIError struct {
		Code        int       `json:"code"`
		ErrorCode   ErrorCode `json:"err_code"`
		Description string    `json:"description,omitempty"`
	}

	// ErrorCode represents error_code returned in response from JetStream API.
	ErrorCode uint16
)

const (
	JSErrCodeBadRequest            ErrorCode = 10003
	JSErrCodeConsumerCreate        ErrorCode = 10012
	JSErrCodeConsumerNameExists    ErrorCode = 10013
	JSErrCodeConsumerNotFound      ErrorCode = 10014
	JSErrCodeMaximumConsumersLimit ErrorCode = 10026

	JSErrCodeMessageNotFound               ErrorCode = 10037
	JSErrCodeJetStreamNotEnabledForAccount ErrorCode = 10039

	JSErrCodeStreamNameInUse ErrorCode = 10058
	JSErrCodeStreamNotFound  ErrorCode = 10059

	JSErrCodeStreamWrongLastSequence ErrorCode = 10071
	JSErrCodeJetStreamNotEnabled     ErrorCode = 10076

	// JSErrCodeStreamWrongLastSequenceConstant is returned instead of
	// JSErrCodeStreamWrongLastSequence for CAS conflicts on replicated (R>1)
	// streams. The two are equivalent "wrong last sequence" responses.
	JSErrCodeStreamWrongLastSequenceConstant ErrorCode = 10164

	JSErrCodeConsumerAlreadyExists ErrorCode = 10105

	JSErrCodeDuplicateFilterSubjects   ErrorCode = 10136
	JSErrCodeOverlappingFilterSubjects ErrorCode = 10138
	JSErrCodeConsumerEmptyFilter       ErrorCode = 10139
	JSErrCodeConsumerExists            ErrorCode = 10148
	JSErrCodeConsumerDoesNotExist      ErrorCode = 10149

	JSErrCodeMirrorWithMsgSchedules   ErrorCode = 10186
	JSErrCodeSourceWithMsgSchedules   ErrorCode = 10187
	JSErrCodeMessageSchedulesDisabled ErrorCode = 10188
	JSErrCodeSchedulePatternInvalid   ErrorCode = 10189
	JSErrCodeScheduleTargetInvalid    ErrorCode = 10190
	JSErrCodeScheduleTTLInvalid       ErrorCode = 10191
	JSErrCodeScheduleRollupInvalid    ErrorCode = 10192
	JSErrCodeScheduleSourceInvalid    ErrorCode = 10203

	JSErrCodeConsumerInvalidReset ErrorCode = 10204
)

var (
	// JetStream API errors

	// ErrJetStreamNotEnabled is an error returned when JetStream is not
	// enabled.
	//
	// Note: This error will not be returned in clustered mode, even if each
	// server in the cluster does not have JetStream enabled. In clustered mode,
	// requests will time out instead.
	ErrJetStreamNotEnabled JetStreamError = &jsError{apiErr: &APIError{ErrorCode: JSErrCodeJetStreamNotEnabled, Description: "jetstream not enabled", Code: 503}}

	// ErrJetStreamNotEnabledForAccount is an error returned when JetStream is
	// not enabled for an account.
	ErrJetStreamNotEnabledForAccount JetStreamError = &jsError{apiErr: &APIError{ErrorCode: JSErrCodeJetStreamNotEnabledForAccount, Description: "jetstream not enabled for account", Code: 503}}

	// ErrStreamNotFound is an error returned when stream with given name does
	// not exist.
	ErrStreamNotFound JetStreamError = &jsError{apiErr: &APIError{ErrorCode: JSErrCodeStreamNotFound, Description: "stream not found", Code: 404}}

	// ErrStreamNameAlreadyInUse is returned when a stream with given name
	// already exists and has a different configuration.
	ErrStreamNameAlreadyInUse JetStreamError = &jsError{apiErr: &APIError{ErrorCode: JSErrCodeStreamNameInUse, Description: "stream name already in use", Code: 400}}

	// ErrStreamSubjectTransformNotSupported is returned when the connected
	// nats-server version does not support setting the stream subject
	// transform. If this error is returned when executing CreateStream(), the
	// stream with invalid configuration was already created in the server.
	ErrStreamSubjectTransformNotSupported JetStreamError = &jsError{message: "stream subject transformation not supported by nats-server"}

	// ErrStreamSourceSubjectTransformNotSupported is returned when the
	// connected nats-server version does not support setting the stream source
	// subject transform. If this error is returned when executing
	// CreateStream(), the stream with invalid configuration was already created
	// in the server.
	ErrStreamSourceSubjectTransformNotSupported JetStreamError = &jsError{message: "stream subject transformation not supported by nats-server"}

	// ErrStreamSourceNotSupported is returned when the connected nats-server
	// version does not support setting the stream sources. If this error is
	// returned when executing CreateStream(), the stream with invalid
	// configuration was already created in the server.
	ErrStreamSourceNotSupported JetStreamError = &jsError{message: "stream sourcing is not supported by nats-server"}

	// ErrStreamSourceMultipleFilterSubjectsNotSupported is returned when the
	// connected nats-server version does not support setting the stream
	// sources. If this error is returned when executing CreateStream(), the
	// stream with invalid configuration was already created in the server.
	ErrStreamSourceMultipleFilterSubjectsNotSupported JetStreamError = &jsError{message: "stream sourcing with multiple subject filters not supported by nats-server"}

	// ErrConsumerNotFound is an error returned when consumer with given name
	// does not exist.
	ErrConsumerNotFound JetStreamError = &jsError{apiErr: &APIError{ErrorCode: JSErrCodeConsumerNotFound, Description: "consumer not found", Code: 404}}

	// ErrConsumerCreationResponseEmpty is an error returned when the response from the server
	// when creating a consumer is empty. This means that the state of the consumer is unknown and
	// the consumer may not have been created successfully.
	ErrConsumerCreationResponseEmpty JetStreamError = &jsError{message: "consumer creation response is empty"}

	// ErrInvalidJetStreamResponse is returned when the response from the server
	// to a JetStream API call (stream CRUD, message get, etc.) does not contain
	// the expected data payload. The operation may or may not have succeeded on
	// the server side.
	ErrInvalidJetStreamResponse JetStreamError = &jsError{message: "invalid jetstream api response"}

	// ErrConsumerExists is returned when attempting to create a consumer with
	// CreateConsumer but a consumer with given name already exists.
	ErrConsumerExists JetStreamError = &jsError{apiErr: &APIError{ErrorCode: JSErrCodeConsumerExists, Description: "consumer already exists", Code: 400}}

	// ErrConsumerNameExists is returned when attempting to update a consumer
	// with UpdateConsumer but a consumer with given name does not exist.
	ErrConsumerDoesNotExist JetStreamError = &jsError{apiErr: &APIError{ErrorCode: JSErrCodeConsumerDoesNotExist, Description: "consumer does not exist", Code: 400}}

	// ErrConsumerResetResponseEmpty is returned when the response from the
	// server to a consumer reset request is missing the ConsumerInfo
	// payload. The reset may or may not have taken effect.
	ErrConsumerResetResponseEmpty JetStreamError = &jsError{message: "consumer reset response is empty"}

	// ErrConsumerInvalidReset is returned when ResetConsumerToSequence is
	// called with a sequence that violates the consumer's DeliverPolicy
	// constraints (e.g. seq below OptStartSeq, or non-zero seq with a
	// DeliverPolicy other than all/by-start-sequence/by-start-time).
	ErrConsumerInvalidReset JetStreamError = &jsError{apiErr: &APIError{ErrorCode: JSErrCodeConsumerInvalidReset, Description: "invalid reset", Code: 400}}

	// ErrMsgNotFound is returned when message with provided sequence number
	// does not exist.
	ErrMsgNotFound JetStreamError = &jsError{apiErr: &APIError{ErrorCode: JSErrCodeMessageNotFound, Description: "message not found", Code: 404}}

	// ErrBadRequest is returned when invalid request is sent to JetStream API.
	ErrBadRequest JetStreamError = &jsError{apiErr: &APIError{ErrorCode: JSErrCodeBadRequest, Description: "bad request", Code: 400}}

	// ErrConsumerCreate is returned when nats-server reports error when
	// creating consumer (e.g. illegal update).
	ErrConsumerCreate JetStreamError = &jsError{apiErr: &APIError{ErrorCode: JSErrCodeConsumerCreate, Description: "could not create consumer", Code: 500}}

	// ErrMaximumConsumersLimit is returned when user limit of allowed
	// consumers for stream is reached
	ErrMaximumConsumersLimit JetStreamError = &jsError{apiErr: &APIError{ErrorCode: JSErrCodeMaximumConsumersLimit, Description: "maximum consumers limit reached", Code: 400}}

	// ErrDuplicateFilterSubjects is returned when both FilterSubject and
	// FilterSubjects are specified when creating consumer.
	ErrDuplicateFilterSubjects JetStreamError = &jsError{apiErr: &APIError{ErrorCode: JSErrCodeDuplicateFilterSubjects, Description: "consumer cannot have both FilterSubject and FilterSubjects specified", Code: 500}}

	// ErrOverlappingFilterSubjects is returned when filter subjects overlap when
	// creating consumer.
	ErrOverlappingFilterSubjects JetStreamError = &jsError{apiErr: &APIError{ErrorCode: JSErrCodeOverlappingFilterSubjects, Description: "consumer subject filters cannot overlap", Code: 500}}

	// ErrEmptyFilter is returned when a filter in FilterSubjects is empty.
	ErrEmptyFilter JetStreamError = &jsError{apiErr: &APIError{ErrorCode: JSErrCodeConsumerEmptyFilter, Description: "consumer filter in FilterSubjects cannot be empty", Code: 500}}

	// ErrScheduleTargetInvalid is returned when publishing a scheduled
	// message without a valid target subject.
	ErrScheduleTargetInvalid JetStreamError = &jsError{apiErr: &APIError{ErrorCode: JSErrCodeScheduleTargetInvalid, Description: "message schedules target is invalid", Code: 400}}

	// ErrSchedulePatternInvalid is returned when the schedule expression
	// is not a valid @at, @every, or cron pattern.
	ErrSchedulePatternInvalid JetStreamError = &jsError{apiErr: &APIError{ErrorCode: JSErrCodeSchedulePatternInvalid, Description: "message schedules pattern is invalid", Code: 400}}

	// ErrMessageSchedulesDisabled is returned when publishing a scheduled
	// message to a stream that does not have AllowMsgSchedules enabled.
	ErrMessageSchedulesDisabled JetStreamError = &jsError{apiErr: &APIError{ErrorCode: JSErrCodeMessageSchedulesDisabled, Description: "message schedules is disabled", Code: 400}}

	// ErrScheduleSourceInvalid is returned when the schedule source
	// subject is invalid.
	ErrScheduleSourceInvalid JetStreamError = &jsError{apiErr: &APIError{ErrorCode: JSErrCodeScheduleSourceInvalid, Description: "message schedules source is invalid", Code: 400}}

	// ErrScheduleTTLInvalid is returned when the schedule TTL value
	// is not a valid duration or "never".
	ErrScheduleTTLInvalid JetStreamError = &jsError{apiErr: &APIError{ErrorCode: JSErrCodeScheduleTTLInvalid, Description: "message schedules invalid per-message TTL", Code: 400}}

	// ErrScheduleRollupInvalid is returned when a scheduled message
	// has an invalid rollup configuration.
	ErrScheduleRollupInvalid JetStreamError = &jsError{apiErr: &APIError{ErrorCode: JSErrCodeScheduleRollupInvalid, Description: "message schedules invalid rollup", Code: 400}}

	// ErrMirrorWithMsgSchedules is returned when attempting to enable
	// message scheduling on a mirror stream.
	ErrMirrorWithMsgSchedules JetStreamError = &jsError{apiErr: &APIError{ErrorCode: JSErrCodeMirrorWithMsgSchedules, Description: "stream mirrors can not also schedule messages", Code: 400}}

	// ErrSourceWithMsgSchedules is returned when attempting to enable
	// message scheduling on a stream with sources.
	ErrSourceWithMsgSchedules JetStreamError = &jsError{apiErr: &APIError{ErrorCode: JSErrCodeSourceWithMsgSchedules, Description: "stream source can not also schedule messages", Code: 400}}

	// Client errors

	// ErrConsumerMultipleFilterSubjectsNotSupported is returned when the
	// connected nats-server version does not support setting multiple filter
	// subjects with filter_subjects field. If this error is returned when
	// executing AddConsumer(), the consumer with invalid configuration was
	// already created in the server.
	ErrConsumerMultipleFilterSubjectsNotSupported JetStreamError = &jsError{message: "multiple consumer filter subjects not supported by nats-server"}

	// ErrConsumerNameAlreadyInUse is an error returned when attempting to create
	// a consumer with a name that is already in use.
	ErrConsumerNameAlreadyInUse JetStreamError = &jsError{message: "consumer name already in use"}

	// ErrNotPullConsumer is returned when attempting to fetch or create pull
	// consumer and the returned consumer is a push consumer.
	ErrNotPullConsumer JetStreamError = &jsError{message: "consumer is not a pull consumer"}

	// ErrNotPushConsumer is returned when attempting to fetch or create push
	// consumer and the returned consumer is a pull consumer.
	ErrNotPushConsumer JetStreamError = &jsError{message: "consumer is not a push consumer"}

	// ErrConsumerAlreadyConsuming is returned when attempting to consume from
	// the same push consumer more than once.
	ErrConsumerAlreadyConsuming JetStreamError = &jsError{message: "consumer is already consuming"}

	// ErrInvalidJSAck is returned when JetStream ack from message publish is
	// invalid.
	ErrInvalidJSAck JetStreamError = &jsError{message: "invalid jetstream publish response"}

	// ErrStreamNameRequired is returned when the provided stream name is empty.
	ErrStreamNameRequired JetStreamError = &jsError{message: "stream name is required"}

	// ErrMsgAlreadyAckd is returned when attempting to acknowledge message more
	// than once.
	ErrMsgAlreadyAckd JetStreamError = &jsError{message: "message was already acknowledged"}

	// ErrNoStreamResponse is returned when there is no response from stream
	// (e.g. no responders error).
	ErrNoStreamResponse JetStreamError = &jsError{message: "no response from stream"}

	// ErrNotJSMessage is returned when attempting to get metadata from non
	// JetStream message.
	ErrNotJSMessage JetStreamError = &jsError{message: "not a jetstream message"}

	// ErrInvalidStreamName is returned when the provided stream name is invalid
	// (contains '.').
	ErrInvalidStreamName JetStreamError = &jsError{message: "invalid stream name"}

	// ErrInvalidSubject is returned when the provided subject name is invalid.
	ErrInvalidSubject JetStreamError = &jsError{message: "invalid subject name"}

	// ErrInvalidConsumerName is returned when the provided consumer name is
	// invalid (contains '.').
	ErrInvalidConsumerName JetStreamError = &jsError{message: "invalid consumer name"}

	// ErrNoMessages is returned when no messages are currently available for a
	// consumer.
	ErrNoMessages JetStreamError = &jsError{message: "no messages"}

	// ErrPinIDMismatch is returned when Pin ID sent in the request does not match
	// the currently pinned consumer subscriber ID on the server.
	ErrPinIDMismatch JetStreamError = &jsError{message: "pin ID mismatch"}

	// ErrMaxBytesExceeded is returned when a message would exceed MaxBytes set
	// on a pull request.
	ErrMaxBytesExceeded JetStreamError = &jsError{message: "message size exceeds max bytes"}

	// ErrBatchCompleted is returned when a fetch request sent the whole batch,
	// but there are still bytes left. This is applicable only when MaxBytes is
	// set on a pull request.
	ErrBatchCompleted JetStreamError = &jsError{message: "batch completed"}

	// ErrConsumerDeleted is returned when attempting to send pull request to a
	// consumer which does not exist.
	ErrConsumerDeleted JetStreamError = &jsError{message: "consumer deleted"}

	// ErrConsumerLeadershipChanged is returned when pending requests are no
	// longer valid after leadership has changed.
	ErrConsumerLeadershipChanged JetStreamError = &jsError{message: "leadership change"}

	// ErrHandlerRequired is returned when no handler func is provided in
	// Stream().
	ErrHandlerRequired JetStreamError = &jsError{message: "handler cannot be empty"}

	// ErrEndOfData is returned when iterating over paged API from JetStream
	// reaches end of data.
	ErrEndOfData JetStreamError = &jsError{message: "end of data reached"}

	// ErrNoHeartbeat is received when no message is received in IdleHeartbeat
	// time (if set).
	ErrNoHeartbeat JetStreamError = &jsError{message: "no heartbeat received"}

	// ErrConsumerHasActiveSubscription is returned when a consumer is already
	// subscribed to a stream.
	ErrConsumerHasActiveSubscription JetStreamError = &jsError{message: "consumer has active subscription"}

	// ErrMsgNotBound is returned when given message is not bound to any
	// subscription.
	ErrMsgNotBound JetStreamError = &jsError{message: "message is not bound to subscription/connection"}

	// ErrMsgNoReply is returned when attempting to reply to a message without a
	// reply subject.
	ErrMsgNoReply JetStreamError = &jsError{message: "message does not have a reply"}

	// ErrMsgDeleteUnsuccessful is returned when an attempt to delete a message
	// is unsuccessful.
	ErrMsgDeleteUnsuccessful JetStreamError = &jsError{message: "message deletion unsuccessful"}

	// ErrAsyncPublishReplySubjectSet is returned when reply subject is set on
	// async message publish.
	ErrAsyncPublishReplySubjectSet JetStreamError = &jsError{message: "reply subject should be empty"}

	// ErrTooManyStalledMsgs is returned when too many outstanding async
	// messages are waiting for ack.
	ErrTooManyStalledMsgs JetStreamError = &jsError{message: "stalled with too many outstanding async published messages"}

	// ErrInvalidOption is returned when there is a collision between options.
	ErrInvalidOption JetStreamError = &jsError{message: "invalid jetstream option"}

	// ErrMsgIteratorClosed is returned when attempting to get message from a
	// closed iterator.
	ErrMsgIteratorClosed JetStreamError = &jsError{message: "messages iterator closed"}

	// ErrConnectionClosed is returned when JetStream operations fail due to
	// underlying connection being closed.
	ErrConnectionClosed JetStreamError = &jsError{message: "connection closed"}

	// ErrServerShutdown is returned when pull request fails due to server
	// shutdown.
	ErrServerShutdown JetStreamError = &jsError{message: "server shutdown"}

	// ErrOrderedConsumerReset indicates that the ordered consumer was
	// automatically reset and recreated to preserve message ordering.
	ErrOrderedConsumerReset JetStreamError = &jsError{message: "recreating ordered consumer"}

	// ErrOrderConsumerUsedAsFetch is returned when ordered consumer was already
	// used to process messages using Fetch (or FetchBytes).
	ErrOrderConsumerUsedAsFetch JetStreamError = &jsError{message: "ordered consumer initialized as fetch"}

	// ErrOrderConsumerUsedAsConsume is returned when ordered consumer was
	// already used to process messages using Consume or Messages.
	ErrOrderConsumerUsedAsConsume JetStreamError = &jsError{message: "ordered consumer initialized as consume"}

	// ErrOrderedConsumerConcurrentRequests is returned when attempting to run
	// concurrent operations on ordered consumers.
	ErrOrderedConsumerConcurrentRequests JetStreamError = &jsError{message: "cannot run concurrent processing using ordered consumer"}

	// ErrOrderedConsumerNotCreated is returned when trying to get consumer info
	// of an ordered consumer which was not yet created.
	ErrOrderedConsumerNotCreated JetStreamError = &jsError{message: "consumer instance not yet created"}

	// ErrJetStreamPublisherClosed is returned for each unfinished ack future when JetStream.Cleanup is called.
	ErrJetStreamPublisherClosed JetStreamError = &jsError{message: "jetstream context closed"}

	// ErrAsyncPublishTimeout is returned when waiting for ack on async publish
	ErrAsyncPublishTimeout JetStreamError = &jsError{message: "timeout waiting for ack"}

	// KeyValue Errors

	// ErrKeyExists is returned when attempting to create a key that already
	// exists.
	//
	// Note: ErrKeyExists matches errors by code 10071, which CAS conflicts
	// from Update/Delete/Purge also carry on non-replicated streams;
	// replicated (R>1) streams report code 10164 instead and will not match.
	// Do not use ErrKeyExists to detect revision conflicts - use
	// ErrKeyRevisionMismatch.
	ErrKeyExists JetStreamError = &jsError{apiErr: &APIError{ErrorCode: JSErrCodeStreamWrongLastSequence, Code: 400}, message: "key exists"}

	// ErrKeyRevisionMismatch is returned by Update, and by Delete/Purge when
	// the LastRevision option is used, if the provided revision does not
	// match the key's current revision (an optimistic-concurrency conflict).
	// Replicated (R>1) streams report this as error code 10164 instead of
	// 10071; both map to this error.
	ErrKeyRevisionMismatch JetStreamError = &jsError{message: "key revision mismatch"}

	// ErrKeyValueConfigRequired is returned when attempting to create a bucket
	// without a config.
	ErrKeyValueConfigRequired JetStreamError = &jsError{message: "config required"}

	// ErrInvalidBucketName is returned when attempting to create a bucket with
	// an invalid name.
	ErrInvalidBucketName JetStreamError = &jsError{message: "invalid bucket name"}

	// ErrInvalidKey is returned when attempting to create a key with an invalid
	// name.
	ErrInvalidKey JetStreamError = &jsError{message: "invalid key"}

	// ErrBucketExists is returned when attempting to create a bucket that
	// already exists and has a different configuration.
	ErrBucketExists JetStreamError = &jsError{message: "bucket name already in use"}

	// ErrBucketNotFound is returned when attempting to access a bucket that
	// does not exist.
	ErrBucketNotFound JetStreamError = &jsError{message: "bucket not found"}

	// ErrBadBucket is returned when attempting to access a bucket that is not a
	// key-value store.
	ErrBadBucket JetStreamError = &jsError{message: "bucket not valid key-value store"}

	// ErrKeyNotFound is returned when attempting to access a key that does not
	// exist.
	ErrKeyNotFound JetStreamError = &jsError{message: "key not found"}

	// ErrKeyDeleted is returned when attempting to access a key that was
	// deleted.
	ErrKeyDeleted JetStreamError = &jsError{message: "key was deleted"}

	// ErrHistoryTooLarge is returned when provided history limit is larger than
	// 64.
	ErrHistoryTooLarge JetStreamError = &jsError{message: "history limited to a max of 64"}

	// ErrNoKeysFound is returned when no keys are found.
	ErrNoKeysFound JetStreamError = &jsError{message: "no keys found"}

	// ErrTTLOnDeleteNotSupported is returned when attempting to set a TTL
	// on a delete operation.
	ErrTTLOnDeleteNotSupported JetStreamError = &jsError{message: "TTL is not supported on delete"}

	// ErrLimitMarkerTTLNotSupported is returned when the connected jetstream API
	// does not support setting the LimitMarkerTTL.
	ErrLimitMarkerTTLNotSupported JetStreamError = &jsError{message: "limit marker TTLs not supported by server"}

	// ErrObjectConfigRequired is returned when attempting to create an object
	// without a config.
	ErrObjectConfigRequired JetStreamError = &jsError{message: "object-store config required"}

	// ErrBadObjectMeta is returned when the meta information of an object is
	// invalid.
	ErrBadObjectMeta JetStreamError = &jsError{message: "object-store meta information invalid"}

	// ErrObjectNotFound is returned when an object is not found.
	ErrObjectNotFound JetStreamError = &jsError{message: "object not found"}

	// ErrInvalidStoreName is returned when the name of an object-store is
	// invalid.
	ErrInvalidStoreName JetStreamError = &jsError{message: "invalid object-store name"}

	// ErrDigestMismatch is returned when the digests of an object do not match.
	ErrDigestMismatch JetStreamError = &jsError{message: "received a corrupt object, digests do not match"}

	// ErrInvalidDigestFormat is returned when the digest hash of an object has
	// an invalid format.
	ErrInvalidDigestFormat JetStreamError = &jsError{message: "object digest hash has invalid format"}

	// ErrNoObjectsFound is returned when no objects are found.
	ErrNoObjectsFound JetStreamError = &jsError{message: "no objects found"}

	// ErrObjectAlreadyExists is returned when an object with the same name
	// already exists.
	ErrObjectAlreadyExists JetStreamError = &jsError{message: "an object already exists with that name"}

	// ErrNameRequired is returned when a name is required.
	ErrNameRequired JetStreamError = &jsError{message: "name is required"}

	// ErrLinkNotAllowed is returned when a link cannot be set when putting the
	// object in a bucket.
	ErrLinkNotAllowed JetStreamError = &jsError{message: "link cannot be set when putting the object in bucket"}

	// ErrObjectRequired is returned when an object is required.
	ErrObjectRequired = &jsError{message: "object required"}

	// ErrNoLinkToDeleted is returned when it is not allowed to link to a
	// deleted object.
	ErrNoLinkToDeleted JetStreamError = &jsError{message: "not allowed to link to a deleted object"}

	// ErrNoLinkToLink is returned when it is not allowed to link to another
	// link.
	ErrNoLinkToLink JetStreamError = &jsError{message: "not allowed to link to another link"}

	// ErrCantGetBucket is returned when an invalid Get is attempted on an
	// object that is a link to a bucket.
	ErrCantGetBucket JetStreamError = &jsError{message: "invalid Get, object is a link to a bucket"}

	// ErrBucketRequired is returned when a bucket is required.
	ErrBucketRequired JetStreamError = &jsError{message: "bucket required"}

	// ErrBucketMalformed is returned when a bucket is malformed.
	ErrBucketMalformed JetStreamError = &jsError{message: "bucket malformed"}

	// ErrUpdateMetaDeleted is returned when the meta information of a deleted
	// object cannot be updated.
	ErrUpdateMetaDeleted JetStreamError = &jsError{message: "cannot update meta for a deleted object"}
)

// Error prints the JetStream API error code and description.
func (e *APIError) Error() string {
	return fmt.Sprintf("nats: API error: code=%d err_code=%d description=%s", e.Code, e.ErrorCode, e.Description)
}

// APIError implements the JetStreamError interface.
func (e *APIError) APIError() *APIError {
	return e
}

// Is matches against an APIError.
func (e *APIError) Is(err error) bool {
	if e == nil {
		return false
	}
	// Extract internal APIError to match against.
	var aerr *APIError
	ok := errors.As(err, &aerr)
	if !ok {
		return ok
	}
	return e.ErrorCode == aerr.ErrorCode
}

func (err *jsError) APIError() *APIError {
	return err.apiErr
}

func (err *jsError) Error() string {
	if err.apiErr != nil && err.apiErr.Description != "" {
		return err.apiErr.Error()
	}
	return fmt.Sprintf("nats: %s", err.message)
}

func (err *jsError) Unwrap() error {
	// Allow matching to embedded APIError in case there is one.
	if err.apiErr == nil {
		return nil
	}
	return err.apiErr
}
