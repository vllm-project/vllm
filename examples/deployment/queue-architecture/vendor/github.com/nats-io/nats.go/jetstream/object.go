// Copyright 2023-2025 The NATS Authors
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
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"hash"
	"io"
	"net"
	"os"
	"strings"
	"sync"
	"time"

	"github.com/nats-io/nats.go"
	"github.com/nats-io/nats.go/internal/parser"
	"github.com/nats-io/nuid"
)

type (
	// ObjectStoreManager is used to manage object stores. It provides methods
	// CRUD operations on object stores.
	ObjectStoreManager interface {
		// ObjectStore will look up and bind to an existing object store
		// instance.
		//
		// If the object store with given name does not exist, ErrBucketNotFound
		// will be returned.
		ObjectStore(ctx context.Context, bucket string) (ObjectStore, error)

		// CreateObjectStore will create a new object store with the given
		// configuration.
		//
		// If the object store with given name already exists, ErrBucketExists
		// will be returned.
		CreateObjectStore(ctx context.Context, cfg ObjectStoreConfig) (ObjectStore, error)

		// UpdateObjectStore will update an existing object store with the given
		// configuration.
		//
		// If the object store with given name does not exist, ErrBucketNotFound
		// will be returned.
		UpdateObjectStore(ctx context.Context, cfg ObjectStoreConfig) (ObjectStore, error)

		// CreateOrUpdateObjectStore will create a new object store with the given
		// configuration if it does not exist, or update an existing object store
		// with the given configuration.
		CreateOrUpdateObjectStore(ctx context.Context, cfg ObjectStoreConfig) (ObjectStore, error)

		// DeleteObjectStore will delete the provided object store.
		//
		// If the object store with given name does not exist, ErrBucketNotFound
		// will be returned.
		DeleteObjectStore(ctx context.Context, bucket string) error

		// ObjectStoreNames is used to retrieve a list of bucket names.
		// It returns an ObjectStoreNamesLister exposing a channel to receive
		// the names of the object stores.
		//
		// The lister will always close the channel when done (either all names
		// have been read or an error occurred) and therefore can be used in a
		// for-range loop.
		ObjectStoreNames(ctx context.Context) ObjectStoreNamesLister

		// ObjectStores is used to retrieve a list of bucket statuses.
		// It returns an ObjectStoresLister exposing a channel to receive
		// the statuses of the object stores.
		//
		// The lister will always close the channel when done (either all statuses
		// have been read or an error occurred) and therefore can be used in a
		// for-range loop.
		ObjectStores(ctx context.Context) ObjectStoresLister
	}

	// ObjectStore contains methods to operate on an object store.
	// Using the ObjectStore interface, it is possible to:
	//
	// - Perform CRUD operations on objects (Get, Put, Delete).
	//   Get and put expose convenience methods to work with
	//   byte slices, strings and files, in addition to streaming [io.Reader]
	// - Get information about an object without retrieving it.
	// - Update the metadata of an object.
	// - Add links to other objects or object stores.
	// - Watch for updates to a store
	// - List information about objects in a store
	// - Retrieve status and configuration of an object store.
	ObjectStore interface {
		// Put will place the contents from the reader into a new object. If the
		// object already exists, it will be overwritten. The object name is
		// required and is taken from the ObjectMeta.Name field.
		//
		// The reader will be read until EOF. ObjectInfo will be returned, containing
		// the object's metadata, digest and instance information.
		Put(ctx context.Context, obj ObjectMeta, reader io.Reader) (*ObjectInfo, error)

		// PutBytes is convenience function to put a byte slice into this object
		// store under the given name.
		//
		// ObjectInfo will be returned, containing the object's metadata, digest
		// and instance information.
		PutBytes(ctx context.Context, name string, data []byte) (*ObjectInfo, error)

		// PutString is convenience function to put a string into this object
		// store under the given name.
		//
		// ObjectInfo will be returned, containing the object's metadata, digest
		// and instance information.
		PutString(ctx context.Context, name string, data string) (*ObjectInfo, error)

		// PutFile is convenience function to put a file contents into this
		// object store. The name of the object will be the path of the file.
		//
		// ObjectInfo will be returned, containing the object's metadata, digest
		// and instance information.
		PutFile(ctx context.Context, file string) (*ObjectInfo, error)

		// Get will pull the named object from the object store. If the object
		// does not exist, ErrObjectNotFound will be returned.
		//
		// The returned ObjectResult will contain the object's metadata and a
		// reader to read the object's contents. The reader will be closed when
		// all data has been read or an error occurs.
		//
		// A GetObjectShowDeleted option can be supplied to return an object
		// even if it was marked as deleted.
		Get(ctx context.Context, name string, opts ...GetObjectOpt) (ObjectResult, error)

		// GetBytes is a convenience function to pull an object from this object
		// store and return it as a byte slice.
		//
		// If the object does not exist, ErrObjectNotFound will be returned.
		//
		// A GetObjectShowDeleted option can be supplied to return an object
		// even if it was marked as deleted.
		GetBytes(ctx context.Context, name string, opts ...GetObjectOpt) ([]byte, error)

		// GetString is a convenience function to pull an object from this
		// object store and return it as a string.
		//
		// If the object does not exist, ErrObjectNotFound will be returned.
		//
		// A GetObjectShowDeleted option can be supplied to return an object
		// even if it was marked as deleted.
		GetString(ctx context.Context, name string, opts ...GetObjectOpt) (string, error)

		// GetFile is a convenience function to pull an object from this object
		// store and place it in a file. If the file already exists, it will be
		// overwritten, otherwise it will be created.
		//
		// If the object does not exist, ErrObjectNotFound will be returned.
		// A GetObjectShowDeleted option can be supplied to return an object
		// even if it was marked as deleted.
		GetFile(ctx context.Context, name, file string, opts ...GetObjectOpt) error

		// GetInfo will retrieve the current information for the object, containing
		// the object's metadata and instance information.
		//
		// If the object does not exist, ErrObjectNotFound will be returned.
		//
		// A GetObjectInfoShowDeleted option can be supplied to return an object
		// even if it was marked as deleted.
		GetInfo(ctx context.Context, name string, opts ...GetObjectInfoOpt) (*ObjectInfo, error)

		// UpdateMeta will update the metadata for the object.
		//
		// If the object does not exist, ErrUpdateMetaDeleted will be returned.
		// If the new name is different from the old name, and an object with the
		// new name already exists, ErrObjectAlreadyExists will be returned.
		UpdateMeta(ctx context.Context, name string, meta ObjectMeta) error

		// Delete will delete the named object from the object store. If the object
		// does not exist, ErrObjectNotFound will be returned. If the object is
		// already deleted, no error will be returned.
		//
		// All chunks for the object will be purged, and the object will be marked
		// as deleted.
		Delete(ctx context.Context, name string) error

		// AddLink will add a link to another object. A link is a reference to
		// another object. The provided name is the name of the link object.
		// The provided ObjectInfo is the info of the object being linked to.
		//
		// If an object with given name already exists, ErrObjectAlreadyExists
		// will be returned.
		// If object being linked to is deleted, ErrNoLinkToDeleted will be
		// returned.
		// If the provided object is a link, ErrNoLinkToLink will be returned.
		// If the provided object is nil or the name is empty, ErrObjectRequired
		// will be returned.
		AddLink(ctx context.Context, name string, obj *ObjectInfo) (*ObjectInfo, error)

		// AddBucketLink will add a link to another object store. A link is a
		// reference to another object store. The provided name is the name of
		// the link object.
		// The provided ObjectStore is the object store being linked to.
		//
		// If an object with given name already exists, ErrObjectAlreadyExists
		// will be returned.
		// If the provided object store is nil ErrBucketRequired will be returned.
		AddBucketLink(ctx context.Context, name string, bucket ObjectStore) (*ObjectInfo, error)

		// Seal will seal the object store, no further modifications will be allowed.
		Seal(ctx context.Context) error

		// Watch for any updates to objects in the store. By default, the watcher will send the latest
		// info for each object and all future updates. Watch will send a nil
		// entry when it has received all initial values. There are a few ways
		// to configure the watcher:
		//
		// - IncludeHistory will have the watcher send all historical information
		// for each object.
		// - IgnoreDeletes will have the watcher not pass any objects with
		// delete markers.
		// - UpdatesOnly will have the watcher only pass updates on objects
		// (without latest info when started).
		Watch(ctx context.Context, opts ...WatchOpt) (ObjectWatcher, error)

		// List will list information about objects in the store.
		//
		// If the object store is empty, ErrNoObjectsFound will be returned.
		List(ctx context.Context, opts ...ListObjectsOpt) ([]*ObjectInfo, error)

		// Status retrieves the status and configuration of the bucket.
		Status(ctx context.Context) (ObjectStoreStatus, error)
	}

	// ObjectWatcher is what is returned when doing a watch. It can be used to
	// retrieve updates to objects in a bucket. If not using UpdatesOnly option,
	// it will also send the latest value for each key. After all initial values
	// have been sent, a nil entry will be sent. Stop can be used to stop the
	// watcher and close the underlying channel. Watcher will not close the
	// channel until Stop is called or connection is closed.
	ObjectWatcher interface {
		Updates() <-chan *ObjectInfo
		Stop() error
	}

	// ObjectStoreConfig is the configuration for the object store.
	ObjectStoreConfig struct {
		// Bucket is the name of the object store. Bucket name has to be
		// unique and can only contain alphanumeric characters, dashes, and
		// underscores.
		Bucket string `json:"bucket"`

		// Description is an optional description for the object store.
		Description string `json:"description,omitempty"`

		// TTL is the maximum age of objects in the store. If an object is not
		// updated within this time, it will be removed from the store.
		// By default, objects do not expire.
		TTL time.Duration `json:"max_age,omitempty"`

		// MaxBytes is the maximum size of the object store. If not specified,
		// the default is -1 (unlimited).
		MaxBytes int64 `json:"max_bytes,omitempty"`

		// Storage is the type of storage to use for the object store. If not
		// specified, the default is FileStorage.
		Storage StorageType `json:"storage,omitempty"`

		// Replicas is the number of replicas to keep for the object store in
		// clustered jetstream. Defaults to 1, maximum is 5.
		Replicas int `json:"num_replicas,omitempty"`

		// Placement is used to declare where the object store should be placed via
		// tags and/or an explicit cluster name.
		Placement *Placement `json:"placement,omitempty"`

		// Compression enables the underlying stream compression.
		// NOTE: Compression is supported for nats-server 2.10.0+
		Compression bool `json:"compression,omitempty"`

		// Bucket-specific metadata
		// NOTE: Metadata requires nats-server v2.10.0+
		Metadata map[string]string `json:"metadata,omitempty"`
	}

	// ObjectStoresLister is used to retrieve a list of object stores. It returns
	// a channel to read the bucket store statuses from. The lister will always
	// close the channel when done (either all stores have been retrieved or an
	// error occurred) and therefore can be used in range loops. Stop can be
	// used to stop the lister when not all object stores have been read.
	ObjectStoresLister interface {
		Status() <-chan ObjectStoreStatus
		Error() error
	}

	// ObjectStoreNamesLister is used to retrieve a list of object store names.
	// It returns a channel to read the bucket names from. The lister will
	// always close the channel when done (either all stores have been retrieved
	// or an error occurred) and therefore can be used in range loops. Stop can
	// be used to stop the lister when not all bucket names have been read.
	ObjectStoreNamesLister interface {
		Name() <-chan string
		Error() error
	}

	// ObjectStoreStatus is run-time status about a bucket.
	ObjectStoreStatus interface {
		// Bucket returns the name of the object store.
		Bucket() string

		// Description is the description supplied when creating the bucket.
		Description() string

		// TTL indicates how long objects are kept in the bucket.
		TTL() time.Duration

		// Storage indicates the underlying JetStream storage technology used to
		// store data.
		Storage() StorageType

		// Replicas indicates how many storage replicas are kept for the data in
		// the bucket.
		Replicas() int

		// Sealed indicates the stream is sealed and cannot be modified in any
		// way.
		Sealed() bool

		// Size is the combined size of all data in the bucket including
		// metadata, in bytes.
		Size() uint64

		// BackingStore indicates what technology is used for storage of the
		// bucket. Currently only JetStream is supported.
		BackingStore() string

		// Metadata is the user supplied metadata for the bucket.
		Metadata() map[string]string

		// IsCompressed indicates if the data is compressed on disk.
		IsCompressed() bool
	}

	// ObjectMetaOptions is used to set additional options when creating an object.
	ObjectMetaOptions struct {
		// Link contains information about a link to another object or object store.
		// It should not be set manually, but rather by using the AddLink or
		// AddBucketLink methods.
		Link *ObjectLink `json:"link,omitempty"`

		// ChunkSize is the maximum size of each chunk in bytes. If not specified,
		// the default is 128k.
		ChunkSize uint32 `json:"max_chunk_size,omitempty"`
	}

	// ObjectMeta is high level information about an object.
	ObjectMeta struct {
		// Name is the name of the object. The name is required when adding an
		// object and has to be unique within the object store.
		Name string `json:"name"`

		// Description is an optional description for the object.
		Description string `json:"description,omitempty"`

		// Headers is an optional set of user-defined headers for the object.
		Headers nats.Header `json:"headers,omitempty"`

		// Metadata is the user supplied metadata for the object.
		Metadata map[string]string `json:"metadata,omitempty"`

		// Additional options for the object.
		Opts *ObjectMetaOptions `json:"options,omitempty"`
	}

	// ObjectInfo contains ObjectMeta and additional information about an
	// object.
	ObjectInfo struct {
		// ObjectMeta contains high level information about the object.
		ObjectMeta

		// Bucket is the name of the object store.
		Bucket string `json:"bucket"`

		// NUID is the unique identifier for the object set when putting the
		// object into the store.
		NUID string `json:"nuid"`

		// Size is the size of the object in bytes. It only includes the size of
		// the object itself, not the metadata.
		Size uint64 `json:"size"`

		// ModTime is the last modification time of the object.
		ModTime time.Time `json:"mtime"`

		// Chunks is the number of chunks the object is split into. Maximum size
		// of each chunk can be specified in ObjectMetaOptions.
		Chunks uint32 `json:"chunks"`

		// Digest is the SHA-256 digest of the object. It is used to verify the
		// integrity of the object.
		Digest string `json:"digest,omitempty"`

		// Deleted indicates if the object is marked as deleted.
		Deleted bool `json:"deleted,omitempty"`
	}

	// ObjectLink is used to embed links to other buckets and objects.
	ObjectLink struct {
		// Bucket is the name of the object store the link is pointing to.
		Bucket string `json:"bucket"`

		// Name can be used to link to a single object.
		// If empty means this is a link to the whole store, like a directory.
		Name string `json:"name,omitempty"`
	}

	// ObjectResult will return the object info and a reader to read the object's
	// contents. The reader will be closed when all data has been read or an
	// error occurs.
	ObjectResult interface {
		io.ReadCloser
		Info() (*ObjectInfo, error)
		Error() error
	}

	// GetObjectOpt is used to set additional options when getting an object.
	GetObjectOpt func(opts *getObjectOpts) error

	// GetObjectInfoOpt is used to set additional options when getting object info.
	GetObjectInfoOpt func(opts *getObjectInfoOpts) error

	// ListObjectsOpt is used to set additional options when listing objects.
	ListObjectsOpt func(opts *listObjectOpts) error

	getObjectOpts struct {
		// Include deleted object in the result.
		showDeleted bool
	}

	getObjectInfoOpts struct {
		// Include deleted object in the result.
		showDeleted bool
	}

	listObjectOpts struct {
		// Include deleted objects in the result channel.
		showDeleted bool
	}

	obs struct {
		name       string
		streamName string
		stream     Stream
		pushJS     nats.JetStreamContext
		js         *jetStream
	}

	// ObjectResult impl.
	objResult struct {
		sync.Mutex
		info   *ObjectInfo
		r      io.ReadCloser
		err    error
		ctx    context.Context
		cancel context.CancelFunc
		digest hash.Hash
	}
)

const (
	objNameTmpl         = "OBJ_%s"     // OBJ_<bucket> // stream name
	objAllChunksPreTmpl = "$O.%s.C.>"  // $O.<bucket>.C.> // chunk stream subject
	objAllMetaPreTmpl   = "$O.%s.M.>"  // $O.<bucket>.M.> // meta stream subject
	objChunksPreTmpl    = "$O.%s.C.%s" // $O.<bucket>.C.<object-nuid> // chunk message subject
	objMetaPreTmpl      = "$O.%s.M.%s" // $O.<bucket>.M.<name-encoded> // meta message subject
	objNoPending        = "0"
	objDefaultChunkSize = uint32(128 * 1024) // 128k
	objDigestType       = "SHA-256="
	objDigestTmpl       = objDigestType + "%s"
)

func (js *jetStream) CreateObjectStore(ctx context.Context, cfg ObjectStoreConfig) (ObjectStore, error) {
	scfg, err := js.prepareObjectStoreConfig(cfg)
	if err != nil {
		return nil, err
	}

	stream, err := js.CreateStream(ctx, scfg)
	if err != nil {
		if errors.Is(err, ErrStreamNameAlreadyInUse) {
			// errors are joined so that backwards compatibility is retained
			// and previous checks for ErrStreamNameAlreadyInUse will still work.
			err = errors.Join(fmt.Errorf("%w: %s", ErrBucketExists, cfg.Bucket), err)
		}
		return nil, err
	}
	pushJS, err := js.legacyJetStream()
	if err != nil {
		return nil, err
	}

	return mapStreamToObjectStore(js, pushJS, cfg.Bucket, stream), nil
}

func (js *jetStream) UpdateObjectStore(ctx context.Context, cfg ObjectStoreConfig) (ObjectStore, error) {
	scfg, err := js.prepareObjectStoreConfig(cfg)
	if err != nil {
		return nil, err
	}

	// Attempt to update the stream.
	stream, err := js.UpdateStream(ctx, scfg)
	if err != nil {
		if errors.Is(err, ErrStreamNotFound) {
			return nil, fmt.Errorf("%w: %s", ErrBucketNotFound, cfg.Bucket)
		}
		return nil, err
	}
	pushJS, err := js.legacyJetStream()
	if err != nil {
		return nil, err
	}

	return mapStreamToObjectStore(js, pushJS, cfg.Bucket, stream), nil
}

func (js *jetStream) CreateOrUpdateObjectStore(ctx context.Context, cfg ObjectStoreConfig) (ObjectStore, error) {
	scfg, err := js.prepareObjectStoreConfig(cfg)
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

	return mapStreamToObjectStore(js, pushJS, cfg.Bucket, stream), nil
}

func (js *jetStream) prepareObjectStoreConfig(cfg ObjectStoreConfig) (StreamConfig, error) {
	if !validBucketRe.MatchString(cfg.Bucket) {
		return StreamConfig{}, ErrInvalidStoreName
	}

	name := cfg.Bucket
	chunks := fmt.Sprintf(objAllChunksPreTmpl, name)
	meta := fmt.Sprintf(objAllMetaPreTmpl, name)

	// We will set explicitly some values so that we can do comparison
	// if we get an "already in use" error and need to check if it is same.
	// See kv
	replicas := cfg.Replicas
	if replicas == 0 {
		replicas = 1
	}
	maxBytes := cfg.MaxBytes
	if maxBytes == 0 {
		maxBytes = -1
	}
	var compression StoreCompression
	if cfg.Compression {
		compression = S2Compression
	}
	scfg := StreamConfig{
		Name:        fmt.Sprintf(objNameTmpl, name),
		Description: cfg.Description,
		Subjects:    []string{chunks, meta},
		MaxAge:      cfg.TTL,
		MaxBytes:    maxBytes,
		Storage:     cfg.Storage,
		Replicas:    replicas,
		Placement:   cfg.Placement,
		Discard:     DiscardNew,
		AllowRollup: true,
		AllowDirect: true,
		Metadata:    cfg.Metadata,
		Compression: compression,
	}

	return scfg, nil
}

// ObjectStore will look up and bind to an existing object store instance.
func (js *jetStream) ObjectStore(ctx context.Context, bucket string) (ObjectStore, error) {
	if !validBucketRe.MatchString(bucket) {
		return nil, ErrInvalidStoreName
	}

	streamName := fmt.Sprintf(objNameTmpl, bucket)
	stream, err := js.Stream(ctx, streamName)
	if err != nil {
		if errors.Is(err, ErrStreamNotFound) {
			err = ErrBucketNotFound
		}
		return nil, err
	}
	pushJS, err := js.legacyJetStream()
	if err != nil {
		return nil, err
	}
	return mapStreamToObjectStore(js, pushJS, bucket, stream), nil
}

// DeleteObjectStore will delete the underlying stream for the named object.
func (js *jetStream) DeleteObjectStore(ctx context.Context, bucket string) error {
	if !validBucketRe.MatchString(bucket) {
		return ErrInvalidStoreName
	}
	stream := fmt.Sprintf(objNameTmpl, bucket)
	if err := js.DeleteStream(ctx, stream); err != nil {
		if errors.Is(err, ErrStreamNotFound) {
			err = errors.Join(fmt.Errorf("%w: %s", ErrBucketNotFound, bucket), err)
		}
		return err
	}
	return nil
}

func encodeName(name string) string {
	return base64.URLEncoding.EncodeToString([]byte(name))
}

// Put will place the contents from the reader into this object-store.
func (obs *obs) Put(ctx context.Context, meta ObjectMeta, r io.Reader) (*ObjectInfo, error) {
	if meta.Name == "" {
		return nil, ErrBadObjectMeta
	}

	if meta.Opts == nil {
		meta.Opts = &ObjectMetaOptions{ChunkSize: objDefaultChunkSize}
	} else if meta.Opts.Link != nil {
		return nil, ErrLinkNotAllowed
	} else if meta.Opts.ChunkSize == 0 {
		meta.Opts.ChunkSize = objDefaultChunkSize
	}

	// Create the new nuid so chunks go on a new subject if the name is re-used
	newnuid := nuid.Next()

	// These will be used in more than one place
	chunkSubj := fmt.Sprintf(objChunksPreTmpl, obs.name, newnuid)

	// Grab existing meta info (einfo). Ok to be found or not found, any other error is a problem
	// Chunks on the old nuid can be cleaned up at the end
	einfo, err := obs.GetInfo(ctx, meta.Name, GetObjectInfoShowDeleted()) // GetInfo will encode the name
	if err != nil && err != ErrObjectNotFound {
		return nil, err
	}

	// For async error handling
	var perr error
	var mu sync.Mutex
	setErr := func(err error) {
		mu.Lock()
		defer mu.Unlock()
		perr = err
	}
	getErr := func() error {
		mu.Lock()
		defer mu.Unlock()
		return perr
	}

	opts := []JetStreamOpt{
		WithPublishAsyncErrHandler(func(js JetStream, _ *nats.Msg, err error) { setErr(err) }),
	}

	// if context deadline is not set, use default JetStream timeout (per publish)
	if _, ok := ctx.Deadline(); !ok {
		opts = append(opts, WithPublishAsyncTimeout(obs.js.opts.DefaultTimeout))
	}

	// Create our own JS context to handle errors etc.
	pubJS, err := New(obs.js.conn, opts...)
	if err != nil {
		return nil, err
	}

	defer pubJS.(*jetStream).cleanupReplySub()

	purgePartial := func() {
		// wait until all pubs are complete or up to default timeout before attempting purge
		select {
		case <-pubJS.PublishAsyncComplete():
		case <-ctx.Done():
		}
		_ = obs.stream.Purge(ctx, WithPurgeSubject(chunkSubj))
	}

	m, h := nats.NewMsg(chunkSubj), sha256.New()
	chunk, sent, total := make([]byte, meta.Opts.ChunkSize), 0, uint64(0)

	// set up the info object. The chunk upload sets the size and digest
	info := &ObjectInfo{Bucket: obs.name, NUID: newnuid, ObjectMeta: meta}

	for r != nil {
		if ctx != nil {
			select {
			case <-ctx.Done():
				if ctx.Err() == context.Canceled {
					err = ctx.Err()
				} else {
					err = nats.ErrTimeout
				}
			default:
			}
			if err != nil {
				purgePartial()
				return nil, err
			}
		}

		// Actual read.
		// TODO(dlc) - Deadline?
		n, readErr := r.Read(chunk)

		// Handle all non EOF errors
		if readErr != nil && readErr != io.EOF {
			purgePartial()
			return nil, readErr
		}

		// Add chunk only if we received data
		if n > 0 {
			// Chunk processing.
			m.Data = chunk[:n]
			h.Write(m.Data)

			// Send msg itself.
			if _, err := pubJS.PublishMsgAsync(m); err != nil {
				purgePartial()
				return nil, err
			}
			if err := getErr(); err != nil {
				purgePartial()
				return nil, err
			}
			// Update totals.
			sent++
			total += uint64(n)
		}

		// EOF Processing.
		if readErr == io.EOF {
			// Place meta info.
			info.Size, info.Chunks = uint64(total), uint32(sent)
			info.Digest = GetObjectDigestValue(h)
			break
		}
	}

	// Prepare the meta message
	metaSubj := fmt.Sprintf(objMetaPreTmpl, obs.name, encodeName(meta.Name))
	mm := nats.NewMsg(metaSubj)
	mm.Header.Set(MsgRollup, MsgRollupSubject)
	mm.Data, err = json.Marshal(info)
	if err != nil {
		if r != nil {
			purgePartial()
		}
		return nil, err
	}

	// Publish the meta message.
	_, err = pubJS.PublishMsgAsync(mm)
	if err != nil {
		if r != nil {
			purgePartial()
		}
		return nil, err
	}

	// Wait for all to be processed.
	select {
	case <-pubJS.PublishAsyncComplete():
		if err := getErr(); err != nil {
			if r != nil {
				purgePartial()
			}
			return nil, err
		}
	case <-ctx.Done():
		return nil, nats.ErrTimeout
	}

	info.ModTime = time.Now().UTC() // This time is not actually the correct time

	// Delete any original chunks.
	if einfo != nil && !einfo.Deleted {
		echunkSubj := fmt.Sprintf(objChunksPreTmpl, obs.name, einfo.NUID)
		_ = obs.stream.Purge(ctx, WithPurgeSubject(echunkSubj))
	}

	// TODO would it be okay to do this to return the info with the correct time?
	// With the understanding that it is an extra call to the server.
	// Otherwise the time the user gets back is the client time, not the server time.
	// return obs.GetInfo(info.Name)

	return info, nil
}

// GetObjectDigestValue calculates the base64 value of hashed data
func GetObjectDigestValue(data hash.Hash) string {
	sha := data.Sum(nil)
	return fmt.Sprintf(objDigestTmpl, base64.URLEncoding.EncodeToString(sha[:]))
}

// DecodeObjectDigest decodes base64 hash
func DecodeObjectDigest(data string) ([]byte, error) {
	digest := strings.SplitN(data, "=", 2)
	if len(digest) != 2 {
		return nil, ErrInvalidDigestFormat
	}
	return base64.URLEncoding.DecodeString(digest[1])
}

func (info *ObjectInfo) isLink() bool {
	return info.ObjectMeta.Opts != nil && info.ObjectMeta.Opts.Link != nil
}

// Get will pull the object from the underlying stream.
func (obs *obs) Get(ctx context.Context, name string, opts ...GetObjectOpt) (ObjectResult, error) {
	ctx, cancel := obs.js.wrapContextWithoutDeadline(ctx)
	var o getObjectOpts
	for _, opt := range opts {
		if opt != nil {
			if err := opt(&o); err != nil {
				return nil, err
			}
		}
	}
	infoOpts := make([]GetObjectInfoOpt, 0)
	if o.showDeleted {
		infoOpts = append(infoOpts, GetObjectInfoShowDeleted())
	}

	// Grab meta info.
	info, err := obs.GetInfo(ctx, name, infoOpts...)
	if err != nil {
		return nil, err
	}
	if info.NUID == "" {
		return nil, ErrBadObjectMeta
	}

	// Check for object links. If single objects we do a pass through.
	if info.isLink() {
		if info.ObjectMeta.Opts.Link.Name == "" {
			return nil, ErrCantGetBucket
		}

		// is the link in the same bucket?
		lbuck := info.ObjectMeta.Opts.Link.Bucket
		if lbuck == obs.name {
			return obs.Get(ctx, info.ObjectMeta.Opts.Link.Name)
		}

		// different bucket
		lobs, err := obs.js.ObjectStore(ctx, lbuck)
		if err != nil {
			return nil, err
		}
		return lobs.Get(ctx, info.ObjectMeta.Opts.Link.Name)
	}

	result := &objResult{info: info, ctx: ctx, cancel: cancel}
	if info.Size == 0 {
		return result, nil
	}

	pr, pw := net.Pipe()
	result.r = pr

	gotErr := func(m *nats.Msg, err error) {
		pw.Close()
		m.Sub.Unsubscribe()
		result.setErr(err)
	}

	// For calculating sum256
	result.digest = sha256.New()

	processChunk := func(m *nats.Msg) {
		var err error
		if ctx != nil {
			select {
			case <-ctx.Done():
				if ctx.Err() == context.Canceled {
					err = ctx.Err()
				} else {
					err = nats.ErrTimeout
				}
			default:
			}
			if err != nil {
				gotErr(m, err)
				return
			}
		}

		tokens, err := parser.GetMetadataFields(m.Reply)
		if err != nil {
			gotErr(m, err)
			return
		}

		// Write to our pipe.
		for b := m.Data; len(b) > 0; {
			n, err := pw.Write(b)
			if err != nil {
				gotErr(m, err)
				return
			}
			b = b[n:]
		}
		// Update sha256
		result.digest.Write(m.Data)

		// Check if we are done.
		if tokens[parser.AckNumPendingTokenPos] == objNoPending {
			pw.Close()
			m.Sub.Unsubscribe()
		}
	}

	chunkSubj := fmt.Sprintf(objChunksPreTmpl, obs.name, info.NUID)
	streamName := fmt.Sprintf(objNameTmpl, obs.name)
	subscribeOpts := []nats.SubOpt{
		nats.OrderedConsumer(),
		nats.Context(ctx),
		nats.BindStream(streamName),
	}
	_, err = obs.pushJS.Subscribe(chunkSubj, processChunk, subscribeOpts...)
	if err != nil {
		return nil, err
	}

	return result, nil
}

// Delete will delete the object.
func (obs *obs) Delete(ctx context.Context, name string) error {
	// Grab meta info.
	info, err := obs.GetInfo(ctx, name, GetObjectInfoShowDeleted())
	if err != nil {
		return err
	}
	if info.NUID == "" {
		return ErrBadObjectMeta
	}

	// Place a rollup delete marker and publish the info
	info.Deleted = true
	info.Size, info.Chunks, info.Digest = 0, 0, ""

	if err = publishMeta(ctx, info, obs.js); err != nil {
		return err
	}

	// Purge chunks for the object.
	chunkSubj := fmt.Sprintf(objChunksPreTmpl, obs.name, info.NUID)
	return obs.stream.Purge(ctx, WithPurgeSubject(chunkSubj))
}

func publishMeta(ctx context.Context, info *ObjectInfo, js *jetStream) error {
	// marshal the object into json, don't store an actual time
	info.ModTime = time.Time{}
	data, err := json.Marshal(info)
	if err != nil {
		return err
	}

	// Prepare and publish the message.
	mm := nats.NewMsg(fmt.Sprintf(objMetaPreTmpl, info.Bucket, encodeName(info.ObjectMeta.Name)))
	mm.Header.Set(MsgRollup, MsgRollupSubject)
	mm.Data = data
	if _, err := js.PublishMsg(ctx, mm); err != nil {
		return err
	}

	// set the ModTime in case it's returned to the user, even though it's not the correct time.
	info.ModTime = time.Now().UTC()
	return nil
}

// AddLink will add a link to another object if it's not deleted and not another link
// name is the name of this link object
// obj is what is being linked too
func (obs *obs) AddLink(ctx context.Context, name string, obj *ObjectInfo) (*ObjectInfo, error) {
	if name == "" {
		return nil, ErrNameRequired
	}

	// TODO Handle stale info

	if obj == nil || obj.Name == "" {
		return nil, ErrObjectRequired
	}
	if obj.Deleted {
		return nil, ErrNoLinkToDeleted
	}
	if obj.isLink() {
		return nil, ErrNoLinkToLink
	}

	// If object with link's name is found, error.
	// If link with link's name is found, that's okay to overwrite.
	// If there was an error that was not ErrObjectNotFound, error.
	einfo, err := obs.GetInfo(ctx, name, GetObjectInfoShowDeleted())
	if einfo != nil {
		if !einfo.isLink() {
			return nil, ErrObjectAlreadyExists
		}
	} else if err != ErrObjectNotFound {
		return nil, err
	}

	// create the meta for the link
	meta := &ObjectMeta{
		Name: name,
		Opts: &ObjectMetaOptions{Link: &ObjectLink{Bucket: obj.Bucket, Name: obj.Name}},
	}
	info := &ObjectInfo{Bucket: obs.name, NUID: nuid.Next(), ModTime: time.Now().UTC(), ObjectMeta: *meta}

	// put the link object
	if err = publishMeta(ctx, info, obs.js); err != nil {
		return nil, err
	}

	return info, nil
}

// AddBucketLink will add a link to another object store.
func (ob *obs) AddBucketLink(ctx context.Context, name string, bucket ObjectStore) (*ObjectInfo, error) {
	if name == "" {
		return nil, ErrNameRequired
	}
	if bucket == nil {
		return nil, ErrBucketRequired
	}
	bos, ok := bucket.(*obs)
	if !ok {
		return nil, ErrBucketMalformed
	}

	// If object with link's name is found, error.
	// If link with link's name is found, that's okay to overwrite.
	// If there was an error that was not ErrObjectNotFound, error.
	einfo, err := ob.GetInfo(ctx, name, GetObjectInfoShowDeleted())
	if einfo != nil {
		if !einfo.isLink() {
			return nil, ErrObjectAlreadyExists
		}
	} else if err != ErrObjectNotFound {
		return nil, err
	}

	// create the meta for the link
	meta := &ObjectMeta{
		Name: name,
		Opts: &ObjectMetaOptions{Link: &ObjectLink{Bucket: bos.name}},
	}
	info := &ObjectInfo{Bucket: ob.name, NUID: nuid.Next(), ObjectMeta: *meta}

	// put the link object
	err = publishMeta(ctx, info, ob.js)
	if err != nil {
		return nil, err
	}

	return info, nil
}

// PutBytes is convenience function to put a byte slice into this object store.
func (obs *obs) PutBytes(ctx context.Context, name string, data []byte) (*ObjectInfo, error) {
	return obs.Put(ctx, ObjectMeta{Name: name}, bytes.NewReader(data))
}

// GetBytes is a convenience function to pull an object from this object store and return it as a byte slice.
func (obs *obs) GetBytes(ctx context.Context, name string, opts ...GetObjectOpt) ([]byte, error) {
	result, err := obs.Get(ctx, name, opts...)
	if err != nil {
		return nil, err
	}
	defer result.Close()

	var b bytes.Buffer
	if _, err := b.ReadFrom(result); err != nil {
		return nil, err
	}
	return b.Bytes(), nil
}

// PutString is convenience function to put a string into this object store.
func (obs *obs) PutString(ctx context.Context, name string, data string) (*ObjectInfo, error) {
	return obs.Put(ctx, ObjectMeta{Name: name}, strings.NewReader(data))
}

// GetString is a convenience function to pull an object from this object store and return it as a string.
func (obs *obs) GetString(ctx context.Context, name string, opts ...GetObjectOpt) (string, error) {
	result, err := obs.Get(ctx, name, opts...)
	if err != nil {
		return "", err
	}
	defer result.Close()

	var b bytes.Buffer
	if _, err := b.ReadFrom(result); err != nil {
		return "", err
	}
	return b.String(), nil
}

// PutFile is convenience function to put a file into an object store.
func (obs *obs) PutFile(ctx context.Context, file string) (*ObjectInfo, error) {
	f, err := os.Open(file)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	return obs.Put(ctx, ObjectMeta{Name: file}, f)
}

// GetFile is a convenience function to pull and object and place in a file.
func (obs *obs) GetFile(ctx context.Context, name, file string, opts ...GetObjectOpt) error {
	// Expect file to be new.
	f, err := os.OpenFile(file, os.O_WRONLY|os.O_CREATE, 0600)
	if err != nil {
		return err
	}
	defer f.Close()

	result, err := obs.Get(ctx, name, opts...)
	if err != nil {
		os.Remove(f.Name())
		return err
	}
	defer result.Close()

	// Stream copy to the file.
	_, err = io.Copy(f, result)
	return err
}

// GetInfo will retrieve the current information for the object.
func (obs *obs) GetInfo(ctx context.Context, name string, opts ...GetObjectInfoOpt) (*ObjectInfo, error) {
	// Grab last meta value we have.
	if name == "" {
		return nil, ErrNameRequired
	}
	var o getObjectInfoOpts
	for _, opt := range opts {
		if opt != nil {
			if err := opt(&o); err != nil {
				return nil, err
			}
		}
	}

	metaSubj := fmt.Sprintf(objMetaPreTmpl, obs.name, encodeName(name)) // used as data in a JS API call

	m, err := obs.stream.GetLastMsgForSubject(ctx, metaSubj)
	if err != nil {
		if errors.Is(err, ErrMsgNotFound) {
			err = ErrObjectNotFound
		}
		if errors.Is(err, ErrStreamNotFound) {
			err = ErrBucketNotFound
		}
		return nil, err
	}
	var info ObjectInfo
	if err := json.Unmarshal(m.Data, &info); err != nil {
		return nil, ErrBadObjectMeta
	}
	if !o.showDeleted && info.Deleted {
		return nil, ErrObjectNotFound
	}
	info.ModTime = m.Time
	return &info, nil
}

// UpdateMeta will update the meta for the object.
func (obs *obs) UpdateMeta(ctx context.Context, name string, meta ObjectMeta) error {
	// Grab the current meta.
	info, err := obs.GetInfo(ctx, name)
	if err != nil {
		if errors.Is(err, ErrObjectNotFound) {
			return ErrUpdateMetaDeleted
		}
		return err
	}

	// If the new name is different from the old, and it exists, error
	// If there was an error that was not ErrObjectNotFound, error.
	if name != meta.Name {
		existingInfo, err := obs.GetInfo(ctx, meta.Name, GetObjectInfoShowDeleted())
		if err != nil && !errors.Is(err, ErrObjectNotFound) {
			return err
		}
		if err == nil && !existingInfo.Deleted {
			return ErrObjectAlreadyExists
		}
	}

	// Update Meta prevents update of ObjectMetaOptions (Link, ChunkSize)
	// These should only be updated internally when appropriate.
	info.Name = meta.Name
	info.Description = meta.Description
	info.Headers = meta.Headers
	info.Metadata = meta.Metadata

	// Prepare the meta message
	if err = publishMeta(ctx, info, obs.js); err != nil {
		return err
	}

	// did the name of this object change? We just stored the meta under the new name
	// so delete the meta from the old name via purge stream for subject
	if name != meta.Name {
		metaSubj := fmt.Sprintf(objMetaPreTmpl, obs.name, encodeName(name))
		return obs.stream.Purge(ctx, WithPurgeSubject(metaSubj))
	}

	return nil
}

// Seal will seal the object store, no further modifications will be allowed.
func (obs *obs) Seal(ctx context.Context) error {
	si, err := obs.stream.Info(ctx)
	if err != nil {
		return err
	}
	// Seal the stream from being able to take on more messages.
	cfg := si.Config
	cfg.Sealed = true
	_, err = obs.js.UpdateStream(ctx, cfg)
	return err
}

// Implementation for Watch
type objWatcher struct {
	updates chan *ObjectInfo
	sub     *nats.Subscription
}

// Updates returns the interior channel.
func (w *objWatcher) Updates() <-chan *ObjectInfo {
	if w == nil {
		return nil
	}
	return w.updates
}

// Stop will unsubscribe from the watcher.
func (w *objWatcher) Stop() error {
	if w == nil {
		return nil
	}
	return w.sub.Unsubscribe()
}

// Watch for changes in the underlying store and receive meta information updates.
func (obs *obs) Watch(ctx context.Context, opts ...WatchOpt) (ObjectWatcher, error) {
	var o watchOpts
	for _, opt := range opts {
		if opt != nil {
			if err := opt.configureWatcher(&o); err != nil {
				return nil, err
			}
		}
	}

	var initDoneMarker bool

	w := &objWatcher{updates: make(chan *ObjectInfo, 32)}

	update := func(m *nats.Msg) {
		var info ObjectInfo
		if err := json.Unmarshal(m.Data, &info); err != nil {
			return // TODO(dlc) - Communicate this upwards?
		}
		meta, err := m.Metadata()
		if err != nil {
			return
		}

		if !o.ignoreDeletes || !info.Deleted {
			info.ModTime = meta.Timestamp
			w.updates <- &info
		}

		// if UpdatesOnly is set, no not send nil to the channel
		// as it would always be triggered after initializing the watcher
		if !initDoneMarker && meta.NumPending == 0 {
			initDoneMarker = true
			w.updates <- nil
		}
	}

	allMeta := fmt.Sprintf(objAllMetaPreTmpl, obs.name)
	_, err := obs.stream.GetLastMsgForSubject(ctx, allMeta)
	// if there are no messages on the stream and we are not watching
	// updates only, send nil to the channel to indicate that the initial
	// watch is done
	if !o.updatesOnly {
		if errors.Is(err, ErrMsgNotFound) {
			initDoneMarker = true
			w.updates <- nil
		}
	} else {
		// if UpdatesOnly was used, mark initialization as complete
		initDoneMarker = true
	}

	// Used ordered consumer to deliver results.
	streamName := fmt.Sprintf(objNameTmpl, obs.name)
	subOpts := []nats.SubOpt{nats.OrderedConsumer(), nats.BindStream(streamName)}
	if !o.includeHistory {
		subOpts = append(subOpts, nats.DeliverLastPerSubject())
	}
	if o.updatesOnly {
		subOpts = append(subOpts, nats.DeliverNew())
	}
	subOpts = append(subOpts, nats.Context(ctx))
	sub, err := obs.pushJS.Subscribe(allMeta, update, subOpts...)
	if err != nil {
		return nil, err
	}
	sub.SetClosedHandler(func(_ string) {
		close(w.updates)
	})
	w.sub = sub
	return w, nil
}

// List will list all the objects in this store.
func (obs *obs) List(ctx context.Context, opts ...ListObjectsOpt) ([]*ObjectInfo, error) {
	var o listObjectOpts
	for _, opt := range opts {
		if opt != nil {
			if err := opt(&o); err != nil {
				return nil, err
			}
		}
	}
	watchOpts := make([]WatchOpt, 0)
	if !o.showDeleted {
		watchOpts = append(watchOpts, IgnoreDeletes())
	}
	watcher, err := obs.Watch(ctx, watchOpts...)
	if err != nil {
		return nil, err
	}
	defer watcher.Stop()

	var objs []*ObjectInfo
	updates := watcher.Updates()
Updates:
	for {
		select {
		case entry := <-updates:
			if entry == nil {
				break Updates
			}
			objs = append(objs, entry)
		case <-ctx.Done():
			return nil, ctx.Err()
		}
	}
	if len(objs) == 0 {
		return nil, ErrNoObjectsFound
	}
	return objs, nil
}

// ObjectBucketStatus  represents status of a Bucket, implements ObjectStoreStatus
type ObjectBucketStatus struct {
	nfo    *StreamInfo
	bucket string
}

// Bucket is the name of the bucket
func (s *ObjectBucketStatus) Bucket() string { return s.bucket }

// Description is the description supplied when creating the bucket
func (s *ObjectBucketStatus) Description() string { return s.nfo.Config.Description }

// TTL indicates how long objects are kept in the bucket
func (s *ObjectBucketStatus) TTL() time.Duration { return s.nfo.Config.MaxAge }

// Storage indicates the underlying JetStream storage technology used to store data
func (s *ObjectBucketStatus) Storage() StorageType { return s.nfo.Config.Storage }

// Replicas indicates how many storage replicas are kept for the data in the bucket
func (s *ObjectBucketStatus) Replicas() int { return s.nfo.Config.Replicas }

// Sealed indicates the stream is sealed and cannot be modified in any way
func (s *ObjectBucketStatus) Sealed() bool { return s.nfo.Config.Sealed }

// Size is the combined size of all data in the bucket including metadata, in bytes
func (s *ObjectBucketStatus) Size() uint64 { return s.nfo.State.Bytes }

// BackingStore indicates what technology is used for storage of the bucket
func (s *ObjectBucketStatus) BackingStore() string { return "JetStream" }

// Metadata is the metadata supplied when creating the bucket
func (s *ObjectBucketStatus) Metadata() map[string]string { return s.nfo.Config.Metadata }

// StreamInfo is the stream info retrieved to create the status
func (s *ObjectBucketStatus) StreamInfo() *StreamInfo { return s.nfo }

// IsCompressed indicates if the data is compressed on disk
func (s *ObjectBucketStatus) IsCompressed() bool { return s.nfo.Config.Compression != NoCompression }

// Status retrieves run-time status about a bucket
func (obs *obs) Status(ctx context.Context) (ObjectStoreStatus, error) {
	nfo, err := obs.stream.Info(ctx)
	if err != nil {
		return nil, err
	}

	status := &ObjectBucketStatus{
		nfo:    nfo,
		bucket: obs.name,
	}

	return status, nil
}

// Read impl.
func (o *objResult) Read(p []byte) (n int, err error) {
	o.Lock()
	defer o.Unlock()
	readDeadline := time.Now().Add(defaultAPITimeout)
	if ctx := o.ctx; ctx != nil {
		if deadline, ok := ctx.Deadline(); ok {
			readDeadline = deadline
		}
		select {
		case <-ctx.Done():
			if ctx.Err() == context.Canceled {
				o.err = ctx.Err()
			} else {
				o.err = nats.ErrTimeout
			}
		default:
		}
	}
	if o.err != nil {
		return 0, o.err
	}
	if o.r == nil {
		return 0, io.EOF
	}

	r := o.r.(net.Conn)
	_ = r.SetReadDeadline(readDeadline)
	n, err = r.Read(p)
	if err, ok := err.(net.Error); ok && err.Timeout() {
		if ctx := o.ctx; ctx != nil {
			select {
			case <-ctx.Done():
				if ctx.Err() == context.Canceled {
					return 0, ctx.Err()
				} else {
					return 0, nats.ErrTimeout
				}
			default:
				err = nil
			}
		}
	}
	if err == io.EOF {
		// Make sure the digest matches.
		sha := o.digest.Sum(nil)
		rsha, decodeErr := DecodeObjectDigest(o.info.Digest)
		if decodeErr != nil {
			o.err = decodeErr
			return 0, o.err
		}
		if !bytes.Equal(sha[:], rsha) {
			o.err = ErrDigestMismatch
			return 0, o.err
		}
	}
	return n, err
}

// Close impl.
func (o *objResult) Close() error {
	o.Lock()
	defer o.Unlock()
	if o.cancel != nil {
		o.cancel()
	}
	if o.r == nil {
		return nil
	}
	return o.r.Close()
}

func (o *objResult) setErr(err error) {
	o.Lock()
	defer o.Unlock()
	o.err = err
}

func (o *objResult) Info() (*ObjectInfo, error) {
	o.Lock()
	defer o.Unlock()
	return o.info, o.err
}

func (o *objResult) Error() error {
	o.Lock()
	defer o.Unlock()
	return o.err
}

// ObjectStoreNames is used to retrieve a list of bucket names
func (js *jetStream) ObjectStoreNames(ctx context.Context) ObjectStoreNamesLister {
	res := &obsLister{
		obsNames: make(chan string),
	}
	l := &streamLister{js: js}
	streamsReq := streamsRequest{
		Subject: fmt.Sprintf(objAllChunksPreTmpl, "*"),
	}

	go func() {
		defer close(res.obsNames)
		for {
			page, err := l.streamNames(ctx, streamsReq)
			if err != nil && !errors.Is(err, ErrEndOfData) {
				res.err = err
				return
			}
			for _, name := range page {
				if !strings.HasPrefix(name, "OBJ_") {
					continue
				}
				res.obsNames <- strings.TrimPrefix(name, "OBJ_")
			}
			if errors.Is(err, ErrEndOfData) {
				return
			}
		}
	}()

	return res
}

// ObjectStores is used to retrieve a list of bucket statuses
func (js *jetStream) ObjectStores(ctx context.Context) ObjectStoresLister {
	res := &obsLister{
		obs: make(chan ObjectStoreStatus),
	}
	l := &streamLister{js: js}
	streamsReq := streamsRequest{
		Subject: fmt.Sprintf(objAllChunksPreTmpl, "*"),
	}
	go func() {
		defer close(res.obs)
		for {
			page, err := l.streamInfos(ctx, streamsReq)
			if err != nil && !errors.Is(err, ErrEndOfData) {
				res.err = err
				return
			}
			for _, info := range page {
				if !strings.HasPrefix(info.Config.Name, "OBJ_") {
					continue
				}
				res.obs <- &ObjectBucketStatus{
					nfo:    info,
					bucket: strings.TrimPrefix(info.Config.Name, "OBJ_"),
				}
			}
			if errors.Is(err, ErrEndOfData) {
				return
			}
		}
	}()

	return res
}

type obsLister struct {
	obs      chan ObjectStoreStatus
	obsNames chan string
	err      error
}

func (ol *obsLister) Status() <-chan ObjectStoreStatus {
	return ol.obs
}

func (ol *obsLister) Name() <-chan string {
	return ol.obsNames
}

func (ol *obsLister) Error() error {
	return ol.err
}

func mapStreamToObjectStore(js *jetStream, pushJS nats.JetStreamContext, bucket string, stream Stream) *obs {
	info := stream.CachedInfo()

	obs := &obs{
		name:       bucket,
		js:         js,
		pushJS:     pushJS,
		streamName: info.Config.Name,
		stream:     stream,
	}

	return obs
}
