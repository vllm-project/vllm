// Copyright 2023-2024 The NATS Authors
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
	"reflect"
	"regexp"
	"slices"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/nats-io/nats.go"
	"github.com/nats-io/nats.go/internal/parser"
)

type (
	// KeyValueManager is used to manage KeyValue stores. It provides methods to
	// create, delete, and retrieve KeyValue stores.
	KeyValueManager interface {
		// KeyValue will lookup and bind to an existing KeyValue store.
		//
		// If the KeyValue store with given name does not exist,
		// ErrBucketNotFound will be returned.
		KeyValue(ctx context.Context, bucket string) (KeyValue, error)

		// CreateKeyValue will create a KeyValue store with the given
		// configuration.
		//
		// If a KeyValue store with the same name already exists and the
		// configuration is different, ErrBucketExists will be returned.
		CreateKeyValue(ctx context.Context, cfg KeyValueConfig) (KeyValue, error)

		// UpdateKeyValue will update an existing KeyValue store with the given
		// configuration.
		//
		// If a KeyValue store with the given name does not exist, ErrBucketNotFound
		// will be returned.
		UpdateKeyValue(ctx context.Context, cfg KeyValueConfig) (KeyValue, error)

		// CreateOrUpdateKeyValue will create a KeyValue store if it does not
		// exist or update an existing KeyValue store with the given
		// configuration (if possible).
		CreateOrUpdateKeyValue(ctx context.Context, cfg KeyValueConfig) (KeyValue, error)

		// DeleteKeyValue will delete this KeyValue store.
		//
		// If the KeyValue store with given name does not exist,
		// ErrBucketNotFound will be returned.
		DeleteKeyValue(ctx context.Context, bucket string) error

		// KeyValueStoreNames is used to retrieve a list of key value store
		// names. It returns a KeyValueNamesLister exposing a channel to read
		// the names from. The lister will always close the channel when done
		// (either all names have been read or an error occurred) and therefore
		// can be used in range loops.
		KeyValueStoreNames(ctx context.Context) KeyValueNamesLister

		// KeyValueStores is used to retrieve a list of key value store
		// statuses. It returns a KeyValueLister exposing a channel to read the
		// statuses from. The lister will always close the channel when done
		// (either all statuses have been read or an error occurred) and
		// therefore can be used in range loops.
		KeyValueStores(ctx context.Context) KeyValueLister
	}

	// KeyValue contains methods to operate on a KeyValue store.
	// Using the KeyValue interface, it is possible to:
	//
	// - Get, Put, Create, Update, Delete and Purge a key
	// - Watch for updates to keys
	// - List all keys
	// - Retrieve historical values for a key
	// - Retrieve status and configuration of a key value bucket
	// - Purge all delete markers
	// - Close the KeyValue store
	KeyValue interface {
		// Get returns the latest value for the key. If the key does not exist,
		// ErrKeyNotFound will be returned.
		Get(ctx context.Context, key string) (KeyValueEntry, error)

		// GetRevision returns a specific revision value for the key. If the key
		// does not exist or the provided revision does not exists,
		// ErrKeyNotFound will be returned.
		GetRevision(ctx context.Context, key string, revision uint64) (KeyValueEntry, error)

		// Put will place the new value for the key into the store. If the key
		// does not exist, it will be created. If the key exists, the value will
		// be updated.
		//
		// A key has to consist of alphanumeric characters, dashes, underscores,
		// equal signs, and dots.
		Put(ctx context.Context, key string, value []byte) (uint64, error)

		// PutString will place the string for the key into the store. If the
		// key does not exist, it will be created. If the key exists, the value
		// will be updated.
		//
		// A key has to consist of alphanumeric characters, dashes, underscores,
		// equal signs, and dots.
		PutString(ctx context.Context, key string, value string) (uint64, error)

		// Create will add the key/value pair if it does not exist. If the key
		// already exists, ErrKeyExists will be returned.
		//
		// A key has to consist of alphanumeric characters, dashes, underscores,
		// equal signs, and dots.
		Create(ctx context.Context, key string, value []byte, opts ...KVCreateOpt) (uint64, error)

		// Update will update the value if the latest revision matches.
		// If the provided revision does not match the key's current revision,
		// ErrKeyRevisionMismatch is returned.
		// Update also resets the TTL associated with the key (if any).
		Update(ctx context.Context, key string, value []byte, revision uint64) (uint64, error)

		// Delete will place a delete marker and leave all revisions. A history
		// of a deleted key can still be retrieved by using the History method
		// or a watch on the key. [Delete] is a non-destructive operation and
		// will not remove any previous revisions from the underlying stream.
		//
		// [LastRevision] option can be specified to only perform delete if the
		// latest revision matches the provided one; if it does not,
		// ErrKeyRevisionMismatch is returned.
		Delete(ctx context.Context, key string, opts ...KVDeleteOpt) error

		// Purge will place a delete marker and remove all previous revisions.
		// Only the latest revision will be preserved (with a delete marker).
		// Unlike [Delete], Purge is a destructive operation and will remove all
		// previous revisions from the underlying streams.
		//
		// [LastRevision] option can be specified to only perform purge if the
		// latest revision matches the provided one; if it does not,
		// ErrKeyRevisionMismatch is returned.
		Purge(ctx context.Context, key string, opts ...KVDeleteOpt) error

		// Watch for any updates to keys that match the keys argument which
		// could include wildcards. By default, the watcher will send the latest
		// value for each key and all future updates. Watch will send a nil
		// entry when it has received all initial values. There are a few ways
		// to configure the watcher:
		//
		// - IncludeHistory will have the key watcher send all historical values
		// for each key (up to KeyValueMaxHistory).
		// - IgnoreDeletes will have the key watcher not pass any keys with
		// delete markers.
		// - UpdatesOnly will have the key watcher only pass updates on values
		// (without latest values when started).
		// - MetaOnly will have the key watcher retrieve only the entry meta
		// data, not the entry value.
		// - ResumeFromRevision instructs the key watcher to resume from a
		// specific revision number.
		Watch(ctx context.Context, keys string, opts ...WatchOpt) (KeyWatcher, error)

		// WatchAll will watch for any updates to all keys. It can be configured
		// with the same options as Watch.
		WatchAll(ctx context.Context, opts ...WatchOpt) (KeyWatcher, error)

		// WatchFiltered will watch for any updates to keys that match the keys
		// argument. It can be configured with the same options as Watch.
		WatchFiltered(ctx context.Context, keys []string, opts ...WatchOpt) (KeyWatcher, error)

		// Keys will return all keys, filtering out any duplicates.
		// For large datasets, this can be memory-heavy as all keys are loaded
		// into memory. Use ListKeys for a streaming alternative.
		Keys(ctx context.Context, opts ...WatchOpt) ([]string, error)

		// ListKeys will return KeyLister, allowing to retrieve all keys from
		// the key value store in a streaming fashion (on a channel).
		// Note: On buckets with a large number of keys and frequent writes,
		// duplicate keys may be reported during listing.
		ListKeys(ctx context.Context, opts ...WatchOpt) (KeyLister, error)

		// ListKeysFiltered returns a KeyLister for filtered keys in the bucket.
		// Note: On buckets with a large number of keys and frequent writes,
		// duplicate keys may be reported during listing.
		ListKeysFiltered(ctx context.Context, filters ...string) (KeyLister, error)

		// History will return all historical values for the key (up to
		// KeyValueMaxHistory).
		History(ctx context.Context, key string, opts ...WatchOpt) ([]KeyValueEntry, error)

		// Bucket returns the KV store name.
		Bucket() string

		// PurgeDeletes will remove all current delete markers. It can be
		// configured using DeleteMarkersOlderThan option to only remove delete
		// markers older than a certain duration.
		//
		// [PurgeDeletes] is a destructive operation and will remove all entries
		// with delete markers from the underlying stream.
		PurgeDeletes(ctx context.Context, opts ...KVPurgeOpt) error

		// Status retrieves the status and configuration of a bucket.
		Status(ctx context.Context) (KeyValueStatus, error)
	}

	// KeyValueConfig is the configuration for a KeyValue store.
	KeyValueConfig struct {
		// Bucket is the name of the KeyValue store. Bucket name has to be
		// unique and can only contain alphanumeric characters, dashes, and
		// underscores.
		Bucket string `json:"bucket"`

		// Description is an optional description for the KeyValue store.
		Description string `json:"description,omitempty"`

		// MaxValueSize is the maximum size of a value in bytes. If not
		// specified, the default is -1 (unlimited).
		MaxValueSize int32 `json:"max_value_size,omitempty"`

		// History is the number of historical values to keep per key. If not
		// specified, the default is 1. Max is 64.
		History uint8 `json:"history,omitempty"`

		// TTL is the expiry time for keys. By default, keys do not expire.
		TTL time.Duration `json:"ttl,omitempty"`

		// MaxBytes is the maximum size in bytes of the KeyValue store. If not
		// specified, the default is -1 (unlimited).
		MaxBytes int64 `json:"max_bytes,omitempty"`

		// Storage is the type of storage to use for the KeyValue store. If not
		// specified, the default is FileStorage.
		Storage StorageType `json:"storage,omitempty"`

		// Replicas is the number of replicas to keep for the KeyValue store in
		// clustered jetstream. Defaults to 1, maximum is 5.
		Replicas int `json:"num_replicas,omitempty"`

		// Placement is used to declare where the stream should be placed via
		// tags and/or an explicit cluster name.
		Placement *Placement `json:"placement,omitempty"`

		// RePublish allows immediate republishing a message to the configured
		// subject after it's stored.
		RePublish *RePublish `json:"republish,omitempty"`

		// Mirror defines the configuration for mirroring another KeyValue
		// store.
		Mirror *StreamSource `json:"mirror,omitempty"`

		// Sources defines the configuration for sources of a KeyValue store.
		// If no subject transforms are defined, it is assumed that a source is
		// also a KV store and subject transforms will be set to correctly map
		// keys from the source KV to the current one. If subject transforms are
		// defined, they will be used as is. This allows using non-kv streams as
		// sources.
		Sources []*StreamSource `json:"sources,omitempty"`

		// Compression sets the underlying stream compression.
		// NOTE: Compression is supported for nats-server 2.10.0+
		Compression bool `json:"compression,omitempty"`

		// LimitMarkerTTL is how long the bucket keeps markers when keys are
		// removed by the TTL setting.
		// It is required for per-key TTL to work and for watcher to notify
		// about TTL expirations (both per key and per bucket)
		LimitMarkerTTL time.Duration `json:"limit_marker_ttl,omitempty"`

		// Metadata is a set of application-defined key-value pairs that can be
		// used to store arbitrary metadata about the bucket.
		Metadata map[string]string `json:"metadata,omitempty"`
	}

	// KeyLister is used to retrieve a list of key value store keys. It returns
	// a channel to read the keys from. The lister will always close the channel
	// when done (either all keys have been read or an error occurred) and
	// therefore can be used in range loops. Stop can be used to stop the lister
	// when not all keys have been read.
	KeyLister interface {
		Keys() <-chan string
		Stop() error
	}

	// KeyValueLister is used to retrieve a list of key value stores. It returns
	// a channel to read the KV store statuses from. The lister will always
	// close the channel when done (either all stores have been retrieved or an
	// error occurred) and therefore can be used in range loops. Stop can be
	// used to stop the lister when not all KeyValue stores have been read.
	KeyValueLister interface {
		Status() <-chan KeyValueStatus
		Error() error
	}

	// KeyValueNamesLister is used to retrieve a list of key value store names.
	// It returns a channel to read the KV bucket names from. The lister will
	// always close the channel when done (either all stores have been retrieved
	// or an error occurred) and therefore can be used in range loops. Stop can
	// be used to stop the lister when not all bucket names have been read.
	KeyValueNamesLister interface {
		Name() <-chan string
		Error() error
	}

	// KeyValueStatus is run-time status about a Key-Value bucket.
	KeyValueStatus interface {
		// Bucket returns the name of the KeyValue store.
		Bucket() string

		// Values is how many messages are in the bucket, including historical values.
		Values() uint64

		// History returns the configured history kept per key.
		History() int64

		// TTL returns the duration for which keys are kept in the bucket.
		TTL() time.Duration

		// BackingStore indicates what technology is used for storage of the bucket.
		// Currently only JetStream is supported.
		BackingStore() string

		// Bytes returns the size of the bucket in bytes.
		Bytes() uint64

		// IsCompressed indicates if the data is compressed on disk.
		IsCompressed() bool

		// LimitMarkerTTL is how long the bucket keeps markers when keys are
		// removed by the TTL setting, 0 meaning markers are not supported.
		LimitMarkerTTL() time.Duration

		// Metadata returns the metadata associated with the bucket.
		Metadata() map[string]string

		// Config returns the configuration for the bucket.
		Config() KeyValueConfig
	}

	// KeyWatcher is what is returned when doing a watch. It can be used to
	// retrieve updates to keys. If not using UpdatesOnly option, it will also
	// send the latest value for each key. After all initial values have been
	// sent, a nil entry will be sent. Stop can be used to stop the watcher and
	// close the underlying channel. Watcher will not close the channel until
	// Stop is called or connection is closed.
	KeyWatcher interface {
		Updates() <-chan KeyValueEntry
		Stop() error
	}

	// KeyValueEntry is a retrieved entry for Get, List or Watch.
	KeyValueEntry interface {
		// Bucket is the bucket the data was loaded from.
		Bucket() string

		// Key is the name of the key that was retrieved.
		Key() string

		// Value is the retrieved value.
		Value() []byte

		// Revision is a unique sequence for this value.
		Revision() uint64

		// Created is the time the data was put in the bucket.
		Created() time.Time

		// Delta is distance from the latest value (how far the current sequence
		// is from the latest).
		Delta() uint64

		// Operation returns Put or Delete or Purge, depending on the manner in
		// which the current revision was created.
		Operation() KeyValueOp
	}
)

type (
	WatchOpt interface {
		configureWatcher(opts *watchOpts) error
	}

	watchOpts struct {
		// Do not send delete markers to the update channel.
		ignoreDeletes bool
		// Include all history per subject, not just last one.
		includeHistory bool
		// Include only updates for keys.
		updatesOnly bool
		// retrieve only the meta data of the entry
		metaOnly bool
		// resumeFromRevision is the revision to resume from.
		resumeFromRevision uint64
	}

	// KVDeleteOpt is used to configure delete and purge operations.
	KVDeleteOpt interface {
		configureDelete(opts *deleteOpts) error
	}

	deleteOpts struct {
		// Remove all previous revisions.
		purge bool

		// Delete only if the latest revision matches.
		revision uint64

		// purge ttl
		ttl time.Duration
	}

	// KVCreateOpt is used to configure Create.
	KVCreateOpt interface {
		configureCreate(opts *createOpts) error
	}

	createOpts struct {
		ttl time.Duration // TTL for the key
	}

	// KVPurgeOpt is used to configure PurgeDeletes.
	KVPurgeOpt interface {
		configurePurge(opts *purgeOpts) error
	}

	purgeOpts struct {
		dmthr time.Duration // Delete markers threshold
	}
)

// kvs is the implementation of KeyValue
type kvs struct {
	name       string
	streamName string
	pre        string
	putPre     string
	pushJS     nats.JetStreamContext
	js         *jetStream
	stream     Stream
	// If true, it means that APIPrefix/Domain was set in the context
	// and we need to add something to some of our high level protocols
	// (such as Put, etc..)
	useJSPfx bool
	// To know if we can use the stream direct get API
	useDirect bool
}

// KeyValueOp represents the type of KV operation (Put, Delete, Purge). It is a
// part of KeyValueEntry.
type KeyValueOp uint8

// Available KeyValueOp values.
const (
	// KeyValuePut is a set on a revision which creates or updates a value for a
	// key.
	KeyValuePut KeyValueOp = iota

	// KeyValueDelete is a set on a revision which adds a delete marker for a
	// key.
	KeyValueDelete

	// KeyValuePurge is a set on a revision which removes all previous revisions
	// for a key.
	KeyValuePurge
)

func (op KeyValueOp) String() string {
	switch op {
	case KeyValuePut:
		return "KeyValuePutOp"
	case KeyValueDelete:
		return "KeyValueDeleteOp"
	case KeyValuePurge:
		return "KeyValuePurgeOp"
	default:
		return "Unknown Operation"
	}
}

const (
	kvBucketNamePre         = "KV_"
	kvBucketNameTmpl        = "KV_%s"
	kvSubjectsTmpl          = "$KV.%s.>"
	kvSubjectsPreTmpl       = "$KV.%s."
	kvSubjectsPreDomainTmpl = "%s.$KV.%s."
)

const (
	KeyValueMaxHistory = 64
	AllKeys            = ">"
	kvLatestRevision   = 0
	kvop               = "KV-Operation"
	kvdel              = "DEL"
	kvpurge            = "PURGE"
)

// Regex for valid keys and buckets.
var (
	validBucketRe    = regexp.MustCompile(`^[a-zA-Z0-9_-]+$`)
	validKeyRe       = regexp.MustCompile(`^[-/_=\.a-zA-Z0-9]+$`)
	validSearchKeyRe = regexp.MustCompile(`^[-/_=\.a-zA-Z0-9*]*[>]?$`)
)

func (js *jetStream) KeyValue(ctx context.Context, bucket string) (KeyValue, error) {
	if !bucketValid(bucket) {
		return nil, ErrInvalidBucketName
	}
	streamName := fmt.Sprintf(kvBucketNameTmpl, bucket)
	stream, err := js.Stream(ctx, streamName)
	if err != nil {
		if errors.Is(err, ErrStreamNotFound) {
			err = ErrBucketNotFound
		}
		return nil, err
	}
	// Do some quick sanity checks that this is a correctly formed stream for KV.
	// Max msgs per subject should be > 0.
	if stream.CachedInfo().Config.MaxMsgsPerSubject < 1 {
		return nil, ErrBadBucket
	}
	pushJS, err := js.legacyJetStream()
	if err != nil {
		return nil, err
	}

	return mapStreamToKVS(js, pushJS, stream), nil
}

func (js *jetStream) CreateKeyValue(ctx context.Context, cfg KeyValueConfig) (KeyValue, error) {
	scfg, err := js.prepareKeyValueConfig(ctx, cfg)
	if err != nil {
		return nil, err
	}

	stream, err := js.CreateStream(ctx, scfg)
	if err != nil {
		if errors.Is(err, ErrStreamNameAlreadyInUse) {
			// errors are joined so that backwards compatibility is retained
			// and previous checks for ErrStreamNameAlreadyInUse will still work.
			err = errors.Join(fmt.Errorf("%w: %s", ErrBucketExists, cfg.Bucket), err)

			// If we have a failure to add, it could be because we have
			// a config change if the KV was created against before a bug fix
			// that changed the value of discard policy.
			// We will check if the stream exists and if the only difference
			// is the discard policy, we will update the stream.
			// The same logic applies for KVs created pre 2.9.x and
			// the AllowDirect setting.
			if stream, _ = js.Stream(ctx, scfg.Name); stream != nil {
				cfg := stream.CachedInfo().Config
				cfg.Discard = scfg.Discard
				cfg.AllowDirect = scfg.AllowDirect
				if reflect.DeepEqual(cfg, scfg) {
					stream, err = js.UpdateStream(ctx, scfg)
				}
			}
		}
		if err != nil {
			return nil, err
		}
	}
	pushJS, err := js.legacyJetStream()
	if err != nil {
		return nil, err
	}

	return mapStreamToKVS(js, pushJS, stream), nil
}

func (js *jetStream) UpdateKeyValue(ctx context.Context, cfg KeyValueConfig) (KeyValue, error) {
	scfg, err := js.prepareKeyValueConfig(ctx, cfg)
	if err != nil {
		return nil, err
	}

	stream, err := js.UpdateStream(ctx, scfg)
	if err != nil {
		if errors.Is(err, ErrStreamNotFound) {
			err = fmt.Errorf("%w: %s", ErrBucketNotFound, cfg.Bucket)
		}
		return nil, err
	}
	pushJS, err := js.legacyJetStream()
	if err != nil {
		return nil, err
	}

	return mapStreamToKVS(js, pushJS, stream), nil
}

func (js *jetStream) CreateOrUpdateKeyValue(ctx context.Context, cfg KeyValueConfig) (KeyValue, error) {
	scfg, err := js.prepareKeyValueConfig(ctx, cfg)
	if err != nil {
		return nil, err
	}

	stream, err := js.CreateOrUpdateStream(ctx, scfg)
	if err != nil {
		return nil, err
	}
	pushJS, err := js.legacyJetStream()
	if err != nil {
		return nil, err
	}

	return mapStreamToKVS(js, pushJS, stream), nil
}

func (js *jetStream) prepareKeyValueConfig(ctx context.Context, cfg KeyValueConfig) (StreamConfig, error) {
	if !bucketValid(cfg.Bucket) {
		return StreamConfig{}, ErrInvalidBucketName
	}
	if _, err := js.AccountInfo(ctx); err != nil {
		return StreamConfig{}, err
	}

	// Default to 1 for history. Max is 64 for now.
	history := int64(1)
	if cfg.History > 0 {
		if cfg.History > KeyValueMaxHistory {
			return StreamConfig{}, ErrHistoryTooLarge
		}
		history = int64(cfg.History)
	}

	replicas := cfg.Replicas
	if replicas == 0 {
		replicas = 1
	}

	// We will set explicitly some values so that we can do comparison
	// if we get an "already in use" error and need to check if it is same.
	maxBytes := cfg.MaxBytes
	if maxBytes == 0 {
		maxBytes = -1
	}
	maxMsgSize := cfg.MaxValueSize
	if maxMsgSize == 0 {
		maxMsgSize = -1
	}
	// When stream's MaxAge is not set, server uses 2 minutes as the default
	// for the duplicate window. If MaxAge is set, and lower than 2 minutes,
	// then the duplicate window will be set to that. If MaxAge is greater,
	// we will cap the duplicate window to 2 minutes (to be consistent with
	// previous behavior).
	duplicateWindow := 2 * time.Minute
	if cfg.TTL > 0 && cfg.TTL < duplicateWindow {
		duplicateWindow = cfg.TTL
	}
	var compression StoreCompression
	if cfg.Compression {
		compression = S2Compression
	}
	var allowMsgTTL bool
	var subjectDeleteMarkerTTL time.Duration
	if cfg.LimitMarkerTTL != 0 {
		info, err := js.AccountInfo(ctx)
		if err != nil {
			return StreamConfig{}, err
		}
		if info.API.Level < 1 {
			return StreamConfig{}, ErrLimitMarkerTTLNotSupported
		}
		allowMsgTTL = true
		subjectDeleteMarkerTTL = cfg.LimitMarkerTTL
	}
	scfg := StreamConfig{
		Name:                   fmt.Sprintf(kvBucketNameTmpl, cfg.Bucket),
		Description:            cfg.Description,
		MaxMsgsPerSubject:      history,
		MaxBytes:               maxBytes,
		MaxAge:                 cfg.TTL,
		MaxMsgSize:             maxMsgSize,
		Storage:                cfg.Storage,
		Replicas:               replicas,
		Placement:              cfg.Placement,
		AllowRollup:            true,
		DenyDelete:             true,
		Duplicates:             duplicateWindow,
		MaxMsgs:                -1,
		MaxConsumers:           -1,
		AllowDirect:            true,
		RePublish:              cfg.RePublish,
		Compression:            compression,
		Discard:                DiscardNew,
		AllowMsgTTL:            allowMsgTTL,
		SubjectDeleteMarkerTTL: subjectDeleteMarkerTTL,
		Metadata:               cfg.Metadata,
	}
	if cfg.Mirror != nil {
		// Copy in case we need to make changes so we do not change caller's version.
		m := cfg.Mirror.copy()
		if !strings.HasPrefix(m.Name, kvBucketNamePre) {
			m.Name = fmt.Sprintf(kvBucketNameTmpl, m.Name)
		}
		scfg.Mirror = m
		scfg.MirrorDirect = true
	} else if len(cfg.Sources) > 0 {
		for _, ss := range cfg.Sources {
			// if subject transforms are already set, then use as is.
			// this allows for full control of the source, e.g. using non-KV streams.
			// Note that in this case, the Name is not modified and full stream name must be provided.
			if len(ss.SubjectTransforms) > 0 {
				scfg.Sources = append(scfg.Sources, ss)
				continue
			}
			var sourceBucketName string
			if strings.HasPrefix(ss.Name, kvBucketNamePre) {
				sourceBucketName = ss.Name[len(kvBucketNamePre):]
			} else {
				sourceBucketName = ss.Name
				ss.Name = fmt.Sprintf(kvBucketNameTmpl, ss.Name)
			}

			if ss.External == nil || sourceBucketName != cfg.Bucket {
				ss.SubjectTransforms = []SubjectTransformConfig{{Source: fmt.Sprintf(kvSubjectsTmpl, sourceBucketName), Destination: fmt.Sprintf(kvSubjectsTmpl, cfg.Bucket)}}
			}
			scfg.Sources = append(scfg.Sources, ss)
		}
		scfg.Subjects = []string{fmt.Sprintf(kvSubjectsTmpl, cfg.Bucket)}
	} else {
		scfg.Subjects = []string{fmt.Sprintf(kvSubjectsTmpl, cfg.Bucket)}
	}

	return scfg, nil
}

// DeleteKeyValue will delete this KeyValue store (JetStream stream).
func (js *jetStream) DeleteKeyValue(ctx context.Context, bucket string) error {
	if !bucketValid(bucket) {
		return ErrInvalidBucketName
	}
	stream := fmt.Sprintf(kvBucketNameTmpl, bucket)
	if err := js.DeleteStream(ctx, stream); err != nil {
		if errors.Is(err, ErrStreamNotFound) {
			err = errors.Join(fmt.Errorf("%w: %s", ErrBucketNotFound, bucket), err)
		}
		return err
	}
	return nil
}

// KeyValueStoreNames is used to retrieve a list of key value store names
func (js *jetStream) KeyValueStoreNames(ctx context.Context) KeyValueNamesLister {
	res := &kvLister{
		kvNames: make(chan string),
	}
	l := &streamLister{js: js}
	streamsReq := streamsRequest{
		Subject: fmt.Sprintf(kvSubjectsTmpl, "*"),
	}
	go func() {
		defer close(res.kvNames)
		for {
			page, err := l.streamNames(ctx, streamsReq)
			if err != nil && !errors.Is(err, ErrEndOfData) {
				res.err = err
				return
			}
			for _, name := range page {
				if !strings.HasPrefix(name, kvBucketNamePre) {
					continue
				}
				res.kvNames <- strings.TrimPrefix(name, kvBucketNamePre)
			}
			if errors.Is(err, ErrEndOfData) {
				return
			}
		}
	}()
	return res
}

// KeyValueStores is used to retrieve a list of key value store statuses
func (js *jetStream) KeyValueStores(ctx context.Context) KeyValueLister {
	res := &kvLister{
		kvs: make(chan KeyValueStatus),
	}
	l := &streamLister{js: js}
	streamsReq := streamsRequest{
		Subject: fmt.Sprintf(kvSubjectsTmpl, "*"),
	}
	go func() {
		defer close(res.kvs)
		for {
			page, err := l.streamInfos(ctx, streamsReq)
			if err != nil && !errors.Is(err, ErrEndOfData) {
				res.err = err
				return
			}
			for _, info := range page {
				if !strings.HasPrefix(info.Config.Name, kvBucketNamePre) {
					continue
				}
				res.kvs <- &KeyValueBucketStatus{info: info, bucket: strings.TrimPrefix(info.Config.Name, kvBucketNamePre)}
			}
			if errors.Is(err, ErrEndOfData) {
				return
			}
		}
	}()
	return res
}

// KeyValueBucketStatus represents status of a Bucket, implements KeyValueStatus
type KeyValueBucketStatus struct {
	info   *StreamInfo
	bucket string
}

// Bucket the name of the bucket
func (s *KeyValueBucketStatus) Bucket() string { return s.bucket }

// Values is how many messages are in the bucket, including historical values
func (s *KeyValueBucketStatus) Values() uint64 { return s.info.State.Msgs }

// History returns the configured history kept per key
func (s *KeyValueBucketStatus) History() int64 { return s.info.Config.MaxMsgsPerSubject }

// TTL is how long the bucket keeps values for
func (s *KeyValueBucketStatus) TTL() time.Duration { return s.info.Config.MaxAge }

// BackingStore indicates what technology is used for storage of the bucket
func (s *KeyValueBucketStatus) BackingStore() string { return "JetStream" }

// StreamInfo is the stream info retrieved to create the status
func (s *KeyValueBucketStatus) StreamInfo() *StreamInfo { return s.info }

// Bytes is the size of the stream
func (s *KeyValueBucketStatus) Bytes() uint64 { return s.info.State.Bytes }

// IsCompressed indicates if the data is compressed on disk
func (s *KeyValueBucketStatus) IsCompressed() bool { return s.info.Config.Compression != NoCompression }

// LimitMarkerTTL is how long the bucket keeps markers when keys are
// removed by the TTL setting, 0 meaning markers are not supported.
func (s *KeyValueBucketStatus) LimitMarkerTTL() time.Duration {
	return s.info.Config.SubjectDeleteMarkerTTL
}

// Config returns the configuration for the bucket.
func (s *KeyValueBucketStatus) Config() KeyValueConfig {
	return KeyValueConfig{
		Bucket:         s.bucket,
		Description:    s.info.Config.Description,
		MaxValueSize:   s.info.Config.MaxMsgSize,
		History:        uint8(s.info.Config.MaxMsgsPerSubject),
		TTL:            s.info.Config.MaxAge,
		MaxBytes:       s.info.Config.MaxBytes,
		Storage:        s.info.Config.Storage,
		Replicas:       s.info.Config.Replicas,
		Placement:      s.info.Config.Placement,
		RePublish:      s.info.Config.RePublish,
		Mirror:         s.info.Config.Mirror,
		Sources:        s.info.Config.Sources,
		Compression:    s.info.Config.Compression != NoCompression,
		Metadata:       s.info.Config.Metadata,
		LimitMarkerTTL: s.info.Config.SubjectDeleteMarkerTTL,
	}
}

// Metadata returns the metadata associated with the bucket.
func (s *KeyValueBucketStatus) Metadata() map[string]string {
	return s.info.Config.Metadata
}

type kvLister struct {
	kvs     chan KeyValueStatus
	kvNames chan string
	err     error
}

func (kl *kvLister) Status() <-chan KeyValueStatus {
	return kl.kvs
}

func (kl *kvLister) Name() <-chan string {
	return kl.kvNames
}

func (kl *kvLister) Error() error {
	return kl.err
}

func (js *jetStream) legacyJetStream() (nats.JetStreamContext, error) {
	opts := make([]nats.JSOpt, 0)
	if js.opts.apiPrefix != "" {
		opts = append(opts, nats.APIPrefix(js.opts.apiPrefix))
	}
	if js.opts.ClientTrace != nil {
		opts = append(opts, nats.ClientTrace{
			RequestSent:      js.opts.ClientTrace.RequestSent,
			ResponseReceived: js.opts.ClientTrace.ResponseReceived,
		})
	}
	return js.conn.JetStream(opts...)
}

func bucketValid(bucket string) bool {
	if len(bucket) == 0 {
		return false
	}
	return validBucketRe.MatchString(bucket)
}

func keyValid(key string) bool {
	if len(key) == 0 || key[0] == '.' || key[len(key)-1] == '.' || strings.Contains(key, "..") {
		return false
	}
	return validKeyRe.MatchString(key)
}

func searchKeyValid(key string) bool {
	if len(key) == 0 || key[0] == '.' || key[len(key)-1] == '.' || strings.Contains(key, "..") {
		return false
	}
	return validSearchKeyRe.MatchString(key)
}

func (kv *kvs) get(ctx context.Context, key string, revision uint64) (KeyValueEntry, error) {
	if !keyValid(key) {
		return nil, ErrInvalidKey
	}

	var b strings.Builder
	b.WriteString(kv.pre)
	b.WriteString(key)

	var m *RawStreamMsg
	var err error

	if revision == kvLatestRevision {
		m, err = kv.stream.GetLastMsgForSubject(ctx, b.String())
	} else {
		m, err = kv.stream.GetMsg(ctx, revision)
		// If a sequence was provided, just make sure that the retrieved
		// message subject matches the request.
		if err == nil && m.Subject != b.String() {
			return nil, ErrKeyNotFound
		}
	}
	if err != nil {
		if errors.Is(err, ErrMsgNotFound) {
			err = ErrKeyNotFound
		}
		return nil, err
	}

	entry := &kve{
		bucket:   kv.name,
		key:      key,
		value:    m.Data,
		revision: m.Sequence,
		created:  m.Time,
	}

	// Double check here that this is not a DEL Operation marker.
	if len(m.Header) > 0 {
		if m.Header.Get(kvop) != "" {
			switch m.Header.Get(kvop) {
			case kvdel:
				entry.op = KeyValueDelete
			case kvpurge:
				entry.op = KeyValuePurge
			}
		} else if m.Header.Get(MarkerReasonHeader) != "" {
			switch m.Header.Get(MarkerReasonHeader) {
			case "MaxAge", "Purge":
				entry.op = KeyValuePurge
			case "Remove":
				entry.op = KeyValueDelete
			}
		}
		if entry.op != KeyValuePut {
			return entry, ErrKeyDeleted
		}
	}

	return entry, nil
}

// kve is the implementation of KeyValueEntry
type kve struct {
	bucket   string
	key      string
	value    []byte
	revision uint64
	delta    uint64
	created  time.Time
	op       KeyValueOp
}

func (e *kve) Bucket() string        { return e.bucket }
func (e *kve) Key() string           { return e.key }
func (e *kve) Value() []byte         { return e.value }
func (e *kve) Revision() uint64      { return e.revision }
func (e *kve) Created() time.Time    { return e.created }
func (e *kve) Delta() uint64         { return e.delta }
func (e *kve) Operation() KeyValueOp { return e.op }

// Get returns the latest value for the key.
func (kv *kvs) Get(ctx context.Context, key string) (KeyValueEntry, error) {
	e, err := kv.get(ctx, key, kvLatestRevision)
	if err != nil {
		if errors.Is(err, ErrKeyDeleted) {
			return nil, ErrKeyNotFound
		}
		return nil, err
	}

	return e, nil
}

// GetRevision returns a specific revision value for the key.
func (kv *kvs) GetRevision(ctx context.Context, key string, revision uint64) (KeyValueEntry, error) {
	e, err := kv.get(ctx, key, revision)
	if err != nil {
		if errors.Is(err, ErrKeyDeleted) {
			return nil, ErrKeyNotFound
		}
		return nil, err
	}

	return e, nil
}

// Put will place the new value for the key into the store.
func (kv *kvs) Put(ctx context.Context, key string, value []byte) (uint64, error) {
	if !keyValid(key) {
		return 0, ErrInvalidKey
	}

	var b strings.Builder
	if kv.useJSPfx {
		b.WriteString(kv.js.opts.apiPrefix)
	}
	if kv.putPre != "" {
		b.WriteString(kv.putPre)
	} else {
		b.WriteString(kv.pre)
	}
	b.WriteString(key)

	pa, err := kv.js.Publish(ctx, b.String(), value)
	if err != nil {
		return 0, err
	}
	return pa.Sequence, err
}

// PutString will place the string for the key into the store.
func (kv *kvs) PutString(ctx context.Context, key string, value string) (uint64, error) {
	return kv.Put(ctx, key, []byte(value))
}

// Create will add the key/value pair if it does not exist.
func (kv *kvs) Create(ctx context.Context, key string, value []byte, opts ...KVCreateOpt) (revision uint64, err error) {
	var o createOpts
	for _, opt := range opts {
		if opt != nil {
			if err := opt.configureCreate(&o); err != nil {
				return 0, err
			}
		}
	}

	v, err := kv.updateRevision(ctx, key, value, 0, o.ttl)
	if err == nil {
		return v, nil
	}

	if e, err := kv.get(ctx, key, kvLatestRevision); errors.Is(err, ErrKeyDeleted) {
		return kv.updateRevision(ctx, key, value, e.Revision(), o.ttl)
	}

	// A wrong-last-sequence response means the key already exists.
	if isWrongLastSeqErr(err) {
		// 10071 matches ErrKeyExists via its code and keeps its original
		// message; 10164 (replicated streams) must wrap ErrKeyExists explicitly.
		if errors.Is(err, ErrKeyExists) {
			jserr := ErrKeyExists.(*jsError)
			return 0, fmt.Errorf("%w: %s", err, jserr.message)
		}
		return 0, fmt.Errorf("%w: %w", err, ErrKeyExists)
	}

	return 0, err
}

// isWrongLastSeqErr reports whether err is a "wrong last sequence" API error.
// Replicated (R>1) streams report CAS conflicts as 10164 instead of 10071.
func isWrongLastSeqErr(err error) bool {
	var apiErr *APIError
	if !errors.As(err, &apiErr) {
		return false
	}
	return apiErr.ErrorCode == JSErrCodeStreamWrongLastSequence ||
		apiErr.ErrorCode == JSErrCodeStreamWrongLastSequenceConstant
}

// mapRevisionMismatch wraps a wrong-last-sequence error with
// ErrKeyRevisionMismatch so callers can detect CAS conflicts regardless of
// whether the stream reported code 10071 or 10164.
func mapRevisionMismatch(err error) error {
	if err != nil && isWrongLastSeqErr(err) {
		return fmt.Errorf("%w: %w", err, ErrKeyRevisionMismatch)
	}
	return err
}

// Update will update the value if the latest revision matches.
func (kv *kvs) Update(ctx context.Context, key string, value []byte, revision uint64) (uint64, error) {
	rev, err := kv.updateRevision(ctx, key, value, revision, 0)
	return rev, mapRevisionMismatch(err)
}

func (kv *kvs) updateRevision(ctx context.Context, key string, value []byte, revision uint64, ttl time.Duration) (uint64, error) {
	if !keyValid(key) {
		return 0, ErrInvalidKey
	}

	var b strings.Builder
	if kv.useJSPfx {
		b.WriteString(kv.js.opts.apiPrefix)
	}
	if kv.putPre != "" {
		b.WriteString(kv.putPre)
	} else {
		b.WriteString(kv.pre)
	}
	b.WriteString(key)

	m := nats.Msg{Subject: b.String(), Header: nats.Header{}, Data: value}
	opts := []PublishOpt{
		WithExpectLastSequencePerSubject(revision),
	}
	if ttl > 0 {
		opts = append(opts, WithMsgTTL(ttl))
	}

	pa, err := kv.js.PublishMsg(ctx, &m, opts...)
	if err != nil {
		return 0, err
	}
	return pa.Sequence, err
}

// Delete will place a delete marker and leave all revisions.
func (kv *kvs) Delete(ctx context.Context, key string, opts ...KVDeleteOpt) error {
	if !keyValid(key) {
		return ErrInvalidKey
	}

	var b strings.Builder
	if kv.useJSPfx {
		b.WriteString(kv.js.opts.apiPrefix)
	}
	if kv.putPre != "" {
		b.WriteString(kv.putPre)
	} else {
		b.WriteString(kv.pre)
	}
	b.WriteString(key)

	// DEL op marker. For watch functionality.
	m := nats.NewMsg(b.String())

	var o deleteOpts
	for _, opt := range opts {
		if opt != nil {
			if err := opt.configureDelete(&o); err != nil {
				return err
			}
		}
	}

	if o.purge {
		m.Header.Set(kvop, kvpurge)
		m.Header.Set(MsgRollup, MsgRollupSubject)
	} else {
		m.Header.Set(kvop, kvdel)
	}
	pubOpts := make([]PublishOpt, 0)
	if o.ttl > 0 && o.purge {
		pubOpts = append(pubOpts, WithMsgTTL(o.ttl))
	} else if o.ttl > 0 {
		return ErrTTLOnDeleteNotSupported
	}

	if o.revision != 0 {
		m.Header.Set(ExpectedLastSubjSeqHeader, strconv.FormatUint(o.revision, 10))
	}

	_, err := kv.js.PublishMsg(ctx, m, pubOpts...)
	return mapRevisionMismatch(err)
}

// Purge will place a delete marker and remove all previous revisions.
func (kv *kvs) Purge(ctx context.Context, key string, opts ...KVDeleteOpt) error {
	return kv.Delete(ctx, key, append(opts, purge())...)
}

// purge removes all previous revisions.
func purge() KVDeleteOpt {
	return deleteOptFn(func(opts *deleteOpts) error {
		opts.purge = true
		return nil
	})
}

// Implementation for Watch
type watcher struct {
	mu          sync.Mutex
	updates     chan KeyValueEntry
	sub         *nats.Subscription
	initDone    bool
	initPending uint64
	received    uint64
}

// Updates returns the interior channel.
func (w *watcher) Updates() <-chan KeyValueEntry {
	if w == nil {
		return nil
	}
	return w.updates
}

// Stop will unsubscribe from the watcher.
func (w *watcher) Stop() error {
	if w == nil {
		return nil
	}
	return w.sub.Unsubscribe()
}

// WatchFiltered will watch for any updates to keys that match the provided
// key filters. It can be configured with the same options as Watch.
func (kv *kvs) WatchFiltered(ctx context.Context, keys []string, opts ...WatchOpt) (KeyWatcher, error) {
	for _, key := range keys {
		if !searchKeyValid(key) {
			return nil, fmt.Errorf("%w: %s", ErrInvalidKey, "key cannot be empty and must be a valid NATS subject")
		}
	}
	var o watchOpts
	for _, opt := range opts {
		if opt != nil {
			if err := opt.configureWatcher(&o); err != nil {
				return nil, err
			}
		}
	}

	// Could be a pattern so don't check for validity as we normally do.
	for i, key := range keys {
		var b strings.Builder
		b.WriteString(kv.pre)
		b.WriteString(key)
		keys[i] = b.String()
	}

	// if no keys are provided, watch all keys
	if len(keys) == 0 {
		var b strings.Builder
		b.WriteString(kv.pre)
		b.WriteString(AllKeys)
		keys = []string{b.String()}
	}

	// We will block below on placing items on the chan. That is by design.
	w := &watcher{updates: make(chan KeyValueEntry, 256)}

	update := func(m *nats.Msg) {
		tokens, err := parser.GetMetadataFields(m.Reply)
		if err != nil {
			return
		}
		if len(m.Subject) <= len(kv.pre) {
			return
		}
		subj := m.Subject[len(kv.pre):]

		var op KeyValueOp
		if len(m.Header) > 0 {
			if m.Header.Get(kvop) != "" {
				switch m.Header.Get(kvop) {
				case kvdel:
					op = KeyValueDelete
				case kvpurge:
					op = KeyValuePurge
				}
			} else if m.Header.Get(MarkerReasonHeader) != "" {
				switch m.Header.Get(MarkerReasonHeader) {
				case "MaxAge", "Purge":
					op = KeyValuePurge
				case "Remove":
					op = KeyValueDelete
				}
			}
		}
		delta := parser.ParseNum(tokens[parser.AckNumPendingTokenPos])
		w.mu.Lock()
		defer w.mu.Unlock()
		if !o.ignoreDeletes || (op != KeyValueDelete && op != KeyValuePurge) {
			entry := &kve{
				bucket:   kv.name,
				key:      subj,
				value:    m.Data,
				revision: parser.ParseNum(tokens[parser.AckStreamSeqTokenPos]),
				created:  time.Unix(0, int64(parser.ParseNum(tokens[parser.AckTimestampSeqTokenPos]))),
				delta:    delta,
				op:       op,
			}
			w.updates <- entry
		}
		// Check if done and initial values.
		if !w.initDone {
			w.received++
			// Use the stable initPending value set at consumer creation.
			// We're done if we've received all expected messages OR there are no more pending.
			if w.received >= w.initPending || delta == 0 {
				w.initDone = true
				w.updates <- nil
			}
		}
	}

	// Used ordered consumer to deliver results.
	subOpts := []nats.SubOpt{nats.BindStream(kv.streamName), nats.OrderedConsumer()}
	if !o.includeHistory {
		subOpts = append(subOpts, nats.DeliverLastPerSubject())
	}
	if o.updatesOnly {
		subOpts = append(subOpts, nats.DeliverNew())
	}
	if o.metaOnly {
		subOpts = append(subOpts, nats.HeadersOnly())
	}
	if o.resumeFromRevision > 0 {
		subOpts = append(subOpts, nats.StartSequence(o.resumeFromRevision))
	}
	subOpts = append(subOpts, nats.Context(ctx))
	// Create the sub and rest of initialization under the lock.
	// We want to prevent the race between this code and the
	// update() callback.
	w.mu.Lock()
	defer w.mu.Unlock()
	var sub *nats.Subscription
	var err error
	if len(keys) == 1 {
		sub, err = kv.pushJS.Subscribe(keys[0], update, subOpts...)
	} else {
		subOpts = append(subOpts, nats.ConsumerFilterSubjects(keys...))
		sub, err = kv.pushJS.Subscribe("", update, subOpts...)
	}
	if err != nil {
		return nil, err
	}
	sub.SetClosedHandler(func(_ string) {
		w.mu.Lock()
		defer w.mu.Unlock()
		close(w.updates)
	})
	// If there were no pending messages at the time of the creation
	// of the consumer, send the marker.
	// Skip if UpdatesOnly() is set, since there will never be updates initially.
	if !o.updatesOnly {
		initialPending, err := sub.InitialConsumerPending()
		if err == nil {
			if initialPending == 0 {
				w.initDone = true
				w.updates <- nil
			} else {
				w.initPending = initialPending
			}
		}
	} else {
		// if UpdatesOnly was used, mark initialization as complete
		w.initDone = true
	}
	w.sub = sub
	return w, nil
}

// Watch for any updates to keys that match the keys argument which could include wildcards.
// Watch will send a nil entry when it has received all initial values.
func (kv *kvs) Watch(ctx context.Context, keys string, opts ...WatchOpt) (KeyWatcher, error) {
	return kv.WatchFiltered(ctx, []string{keys}, opts...)
}

// WatchAll will invoke the callback for all updates.
func (kv *kvs) WatchAll(ctx context.Context, opts ...WatchOpt) (KeyWatcher, error) {
	return kv.Watch(ctx, AllKeys, opts...)
}

// Keys will return all keys, filtering out any duplicates.
func (kv *kvs) Keys(ctx context.Context, opts ...WatchOpt) ([]string, error) {
	opts = append(opts, IgnoreDeletes(), MetaOnly())
	watcher, err := kv.WatchAll(ctx, opts...)
	if err != nil {
		return nil, err
	}
	defer watcher.Stop()

	var keys []string
	for entry := range watcher.Updates() {
		if entry == nil {
			break
		}
		keys = append(keys, entry.Key())
	}
	if len(keys) == 0 {
		return nil, ErrNoKeysFound
	}
	slices.Sort(keys)
	return slices.Compact(keys), nil
}

type keyLister struct {
	watcher KeyWatcher
	keys    chan string
}

// ListKeys will return all keys.
// Note: On buckets with a large number of keys and frequent writes,
// duplicate keys may be reported during listing.
func (kv *kvs) ListKeys(ctx context.Context, opts ...WatchOpt) (KeyLister, error) {
	opts = append(opts, IgnoreDeletes(), MetaOnly())
	watcher, err := kv.WatchAll(ctx, opts...)
	if err != nil {
		return nil, err
	}
	kl := &keyLister{watcher: watcher, keys: make(chan string, 256)}

	go func() {
		defer close(kl.keys)
		defer watcher.Stop()
		for {
			select {
			case entry := <-watcher.Updates():
				if entry == nil {
					return
				}
				kl.keys <- entry.Key()
			case <-ctx.Done():
				return
			}
		}
	}()
	return kl, nil
}

// ListKeysFiltered returns a KeyLister for filtered keys in the bucket.
// Note: On buckets with a large number of keys and frequent writes,
// duplicate keys may be reported during listing.
func (kv *kvs) ListKeysFiltered(ctx context.Context, filters ...string) (KeyLister, error) {
	watcher, err := kv.WatchFiltered(ctx, filters, IgnoreDeletes(), MetaOnly())
	if err != nil {
		return nil, err
	}

	// Reuse the existing keyLister implementation
	kl := &keyLister{watcher: watcher, keys: make(chan string, 256)}

	go func() {
		defer close(kl.keys)
		defer watcher.Stop()

		for {
			select {
			case entry := <-watcher.Updates():
				if entry == nil { // Indicates all initial values are received
					return
				}
				kl.keys <- entry.Key()
			case <-ctx.Done():
				return
			}
		}
	}()

	return kl, nil
}

func (kl *keyLister) Keys() <-chan string {
	return kl.keys
}

func (kl *keyLister) Stop() error {
	return kl.watcher.Stop()
}

// History will return all historical values for the key.
func (kv *kvs) History(ctx context.Context, key string, opts ...WatchOpt) ([]KeyValueEntry, error) {
	opts = append(opts, IncludeHistory())
	watcher, err := kv.Watch(ctx, key, opts...)
	if err != nil {
		return nil, err
	}
	defer watcher.Stop()

	var entries []KeyValueEntry
	for entry := range watcher.Updates() {
		if entry == nil {
			break
		}
		entries = append(entries, entry)
	}
	if len(entries) == 0 {
		return nil, ErrKeyNotFound
	}
	return entries, nil
}

// Bucket returns the current bucket name.
func (kv *kvs) Bucket() string {
	return kv.name
}

const kvDefaultPurgeDeletesMarkerThreshold = 30 * time.Minute

// PurgeDeletes will remove all current delete markers.
func (kv *kvs) PurgeDeletes(ctx context.Context, opts ...KVPurgeOpt) error {
	var o purgeOpts
	for _, opt := range opts {
		if opt != nil {
			if err := opt.configurePurge(&o); err != nil {
				return err
			}
		}
	}
	watcher, err := kv.WatchAll(ctx)
	if err != nil {
		return err
	}
	defer watcher.Stop()

	var limit time.Time
	olderThan := o.dmthr
	// Negative value is used to instruct to always remove markers, regardless
	// of age. If set to 0 (or not set), use our default value.
	if olderThan == 0 {
		olderThan = kvDefaultPurgeDeletesMarkerThreshold
	}
	if olderThan > 0 {
		limit = time.Now().Add(-olderThan)
	}

	var deleteMarkers []KeyValueEntry
	for entry := range watcher.Updates() {
		if entry == nil {
			break
		}
		if op := entry.Operation(); op == KeyValueDelete || op == KeyValuePurge {
			deleteMarkers = append(deleteMarkers, entry)
		}
	}
	// Stop watcher here so as we purge we do not have the system continually updating numPending.
	watcher.Stop()

	var b strings.Builder
	// Do actual purges here.
	for _, entry := range deleteMarkers {
		b.WriteString(kv.pre)
		b.WriteString(entry.Key())
		purgeOpts := []StreamPurgeOpt{WithPurgeSubject(b.String())}
		if olderThan > 0 && entry.Created().After(limit) {
			purgeOpts = append(purgeOpts, WithPurgeKeep(1))
		}
		if err := kv.stream.Purge(ctx, purgeOpts...); err != nil {
			return err
		}
		b.Reset()
	}
	return nil
}

// Status retrieves the status and configuration of a bucket
func (kv *kvs) Status(ctx context.Context) (KeyValueStatus, error) {
	nfo, err := kv.stream.Info(ctx)
	if err != nil {
		return nil, err
	}

	return &KeyValueBucketStatus{info: nfo, bucket: kv.name}, nil
}

func mapStreamToKVS(js *jetStream, pushJS nats.JetStreamContext, stream Stream) *kvs {
	info := stream.CachedInfo()
	bucket := strings.TrimPrefix(info.Config.Name, kvBucketNamePre)
	kv := &kvs{
		name:       bucket,
		streamName: info.Config.Name,
		pre:        fmt.Sprintf(kvSubjectsPreTmpl, bucket),
		js:         js,
		pushJS:     pushJS,
		stream:     stream,
		// Determine if we need to use the JS prefix in front of Put and Delete operations
		useJSPfx:  js.opts.apiPrefix != DefaultAPIPrefix,
		useDirect: info.Config.AllowDirect,
	}

	// If we are mirroring, we will have mirror direct on, so just use the mirror name
	// and override use
	if m := info.Config.Mirror; m != nil {
		bucket := strings.TrimPrefix(m.Name, kvBucketNamePre)
		if m.External != nil && m.External.APIPrefix != "" {
			kv.useJSPfx = false
			kv.pre = fmt.Sprintf(kvSubjectsPreTmpl, bucket)
			kv.putPre = fmt.Sprintf(kvSubjectsPreDomainTmpl, m.External.APIPrefix, bucket)
		} else {
			kv.putPre = fmt.Sprintf(kvSubjectsPreTmpl, bucket)
		}
	}

	return kv
}
