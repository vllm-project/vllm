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

	"github.com/nats-io/nats.go/internal/parser"
	"github.com/nats-io/nuid"
)

// ObjectStoreManager creates, loads and deletes Object Stores
type ObjectStoreManager interface {
	// ObjectStore will look up and bind to an existing object store instance.
	ObjectStore(bucket string) (ObjectStore, error)
	// CreateObjectStore will create an object store.
	CreateObjectStore(cfg *ObjectStoreConfig) (ObjectStore, error)
	// DeleteObjectStore will delete the underlying stream for the named object.
	DeleteObjectStore(bucket string) error
	// ObjectStoreNames is used to retrieve a list of bucket names
	ObjectStoreNames(opts ...ObjectOpt) <-chan string
	// ObjectStores is used to retrieve a list of bucket statuses
	ObjectStores(opts ...ObjectOpt) <-chan ObjectStoreStatus
}

// ObjectStore is a blob store capable of storing large objects efficiently in
// JetStream streams
type ObjectStore interface {
	// Put will place the contents from the reader into a new object.
	Put(obj *ObjectMeta, reader io.Reader, opts ...ObjectOpt) (*ObjectInfo, error)
	// Get will pull the named object from the object store.
	Get(name string, opts ...GetObjectOpt) (ObjectResult, error)

	// PutBytes is convenience function to put a byte slice into this object store.
	PutBytes(name string, data []byte, opts ...ObjectOpt) (*ObjectInfo, error)
	// GetBytes is a convenience function to pull an object from this object store and return it as a byte slice.
	GetBytes(name string, opts ...GetObjectOpt) ([]byte, error)

	// PutString is convenience function to put a string into this object store.
	PutString(name string, data string, opts ...ObjectOpt) (*ObjectInfo, error)
	// GetString is a convenience function to pull an object from this object store and return it as a string.
	GetString(name string, opts ...GetObjectOpt) (string, error)

	// PutFile is convenience function to put a file into this object store.
	PutFile(file string, opts ...ObjectOpt) (*ObjectInfo, error)
	// GetFile is a convenience function to pull an object from this object store and place it in a file.
	GetFile(name, file string, opts ...GetObjectOpt) error

	// GetInfo will retrieve the current information for the object.
	GetInfo(name string, opts ...GetObjectInfoOpt) (*ObjectInfo, error)
	// UpdateMeta will update the metadata for the object.
	UpdateMeta(name string, meta *ObjectMeta) error

	// Delete will delete the named object.
	Delete(name string) error

	// AddLink will add a link to another object.
	AddLink(name string, obj *ObjectInfo) (*ObjectInfo, error)

	// AddBucketLink will add a link to another object store.
	AddBucketLink(name string, bucket ObjectStore) (*ObjectInfo, error)

	// Seal will seal the object store, no further modifications will be allowed.
	Seal() error

	// Watch for changes in the underlying store and receive meta information updates.
	Watch(opts ...WatchOpt) (ObjectWatcher, error)

	// List will list all the objects in this store.
	List(opts ...ListObjectsOpt) ([]*ObjectInfo, error)

	// Status retrieves run-time status about the backing store of the bucket.
	Status() (ObjectStoreStatus, error)
}

type ObjectOpt interface {
	configureObject(opts *objOpts) error
}

type objOpts struct {
	ctx context.Context
}

// For nats.Context() support.
func (ctx ContextOpt) configureObject(opts *objOpts) error {
	opts.ctx = ctx
	return nil
}

// ObjectWatcher is what is returned when doing a watch.
type ObjectWatcher interface {
	// Updates returns a channel to read any updates to entries.
	Updates() <-chan *ObjectInfo
	// Stop will stop this watcher.
	Stop() error
}

var (
	ErrObjectConfigRequired = errors.New("nats: object-store config required")
	ErrBadObjectMeta        = errors.New("nats: object-store meta information invalid")
	ErrObjectNotFound       = errors.New("nats: object not found")
	ErrInvalidStoreName     = errors.New("nats: invalid object-store name")
	ErrDigestMismatch       = errors.New("nats: received a corrupt object, digests do not match")
	ErrInvalidDigestFormat  = errors.New("nats: object digest hash has invalid format")
	ErrNoObjectsFound       = errors.New("nats: no objects found")
	ErrObjectAlreadyExists  = errors.New("nats: an object already exists with that name")
	ErrNameRequired         = errors.New("nats: name is required")
	ErrNeeds262             = errors.New("nats: object-store requires at least server version 2.6.2")
	ErrLinkNotAllowed       = errors.New("nats: link cannot be set when putting the object in bucket")
	ErrObjectRequired       = errors.New("nats: object required")
	ErrNoLinkToDeleted      = errors.New("nats: not allowed to link to a deleted object")
	ErrNoLinkToLink         = errors.New("nats: not allowed to link to another link")
	ErrCantGetBucket        = errors.New("nats: invalid Get, object is a link to a bucket")
	ErrBucketRequired       = errors.New("nats: bucket required")
	ErrBucketMalformed      = errors.New("nats: bucket malformed")
	ErrUpdateMetaDeleted    = errors.New("nats: cannot update meta for a deleted object")
)

// ObjectStoreConfig is the config for the object store.
type ObjectStoreConfig struct {
	Bucket      string        `json:"bucket"`
	Description string        `json:"description,omitempty"`
	TTL         time.Duration `json:"max_age,omitempty"`
	MaxBytes    int64         `json:"max_bytes,omitempty"`
	Storage     StorageType   `json:"storage,omitempty"`
	Replicas    int           `json:"num_replicas,omitempty"`
	Placement   *Placement    `json:"placement,omitempty"`

	// Bucket-specific metadata
	// NOTE: Metadata requires nats-server v2.10.0+
	Metadata map[string]string `json:"metadata,omitempty"`
	// Enable underlying stream compression.
	// NOTE: Compression is supported for nats-server 2.10.0+
	Compression bool `json:"compression,omitempty"`
}

type ObjectStoreStatus interface {
	// Bucket is the name of the bucket
	Bucket() string
	// Description is the description supplied when creating the bucket
	Description() string
	// TTL indicates how long objects are kept in the bucket
	TTL() time.Duration
	// Storage indicates the underlying JetStream storage technology used to store data
	Storage() StorageType
	// Replicas indicates how many storage replicas are kept for the data in the bucket
	Replicas() int
	// Sealed indicates the stream is sealed and cannot be modified in any way
	Sealed() bool
	// Size is the combined size of all data in the bucket including metadata, in bytes
	Size() uint64
	// BackingStore provides details about the underlying storage
	BackingStore() string
	// Metadata is the user supplied metadata for the bucket
	Metadata() map[string]string
	// IsCompressed indicates if the data is compressed on disk
	IsCompressed() bool
}

// ObjectMetaOptions
type ObjectMetaOptions struct {
	Link      *ObjectLink `json:"link,omitempty"`
	ChunkSize uint32      `json:"max_chunk_size,omitempty"`
}

// ObjectMeta is high level information about an object.
type ObjectMeta struct {
	Name        string            `json:"name"`
	Description string            `json:"description,omitempty"`
	Headers     Header            `json:"headers,omitempty"`
	Metadata    map[string]string `json:"metadata,omitempty"`

	// Optional options.
	Opts *ObjectMetaOptions `json:"options,omitempty"`
}

// ObjectInfo is meta plus instance information.
type ObjectInfo struct {
	ObjectMeta
	Bucket  string    `json:"bucket"`
	NUID    string    `json:"nuid"`
	Size    uint64    `json:"size"`
	ModTime time.Time `json:"mtime"`
	Chunks  uint32    `json:"chunks"`
	Digest  string    `json:"digest,omitempty"`
	Deleted bool      `json:"deleted,omitempty"`
}

// ObjectLink is used to embed links to other buckets and objects.
type ObjectLink struct {
	// Bucket is the name of the other object store.
	Bucket string `json:"bucket"`
	// Name can be used to link to a single object.
	// If empty means this is a link to the whole store, like a directory.
	Name string `json:"name,omitempty"`
}

// ObjectResult will return the underlying stream info and also be an io.ReadCloser.
type ObjectResult interface {
	io.ReadCloser
	Info() (*ObjectInfo, error)
	Error() error
}

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

type obs struct {
	name   string
	stream string
	js     *js
}

// CreateObjectStore will create an object store.
func (js *js) CreateObjectStore(cfg *ObjectStoreConfig) (ObjectStore, error) {
	if !js.nc.serverMinVersion(2, 6, 2) {
		return nil, ErrNeeds262
	}
	if cfg == nil {
		return nil, ErrObjectConfigRequired
	}
	if !validBucketRe.MatchString(cfg.Bucket) {
		return nil, ErrInvalidStoreName
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
	scfg := &StreamConfig{
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

	// Create our stream.
	_, err := js.AddStream(scfg)
	if err != nil {
		return nil, err
	}

	return &obs{name: name, stream: scfg.Name, js: js}, nil
}

// ObjectStore will look up and bind to an existing object store instance.
func (js *js) ObjectStore(bucket string) (ObjectStore, error) {
	if !validBucketRe.MatchString(bucket) {
		return nil, ErrInvalidStoreName
	}
	if !js.nc.serverMinVersion(2, 6, 2) {
		return nil, ErrNeeds262
	}

	stream := fmt.Sprintf(objNameTmpl, bucket)
	si, err := js.StreamInfo(stream)
	if err != nil {
		return nil, err
	}
	return &obs{name: bucket, stream: si.Config.Name, js: js}, nil
}

// DeleteObjectStore will delete the underlying stream for the named object.
func (js *js) DeleteObjectStore(bucket string) error {
	stream := fmt.Sprintf(objNameTmpl, bucket)
	return js.DeleteStream(stream)
}

func encodeName(name string) string {
	return base64.URLEncoding.EncodeToString([]byte(name))
}

// Put will place the contents from the reader into this object-store.
func (obs *obs) Put(meta *ObjectMeta, r io.Reader, opts ...ObjectOpt) (*ObjectInfo, error) {
	if meta == nil || meta.Name == "" {
		return nil, ErrBadObjectMeta
	}

	if meta.Opts == nil {
		meta.Opts = &ObjectMetaOptions{ChunkSize: objDefaultChunkSize}
	} else if meta.Opts.Link != nil {
		return nil, ErrLinkNotAllowed
	} else if meta.Opts.ChunkSize == 0 {
		meta.Opts.ChunkSize = objDefaultChunkSize
	}

	var o objOpts
	for _, opt := range opts {
		if opt != nil {
			if err := opt.configureObject(&o); err != nil {
				return nil, err
			}
		}
	}
	ctx := o.ctx

	// Create the new nuid so chunks go on a new subject if the name is re-used
	newnuid := nuid.Next()

	// These will be used in more than one place
	chunkSubj := fmt.Sprintf(objChunksPreTmpl, obs.name, newnuid)

	// Grab existing meta info (einfo). Ok to be found or not found, any other error is a problem
	// Chunks on the old nuid can be cleaned up at the end
	einfo, err := obs.GetInfo(meta.Name, GetObjectInfoShowDeleted()) // GetInfo will encode the name
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

	// Create our own JS context to handle errors etc.
	jetStream, err := obs.js.nc.JetStream(PublishAsyncErrHandler(func(js JetStream, _ *Msg, err error) { setErr(err) }))
	if err != nil {
		return nil, err
	}

	defer jetStream.(*js).cleanupReplySub()

	purgePartial := func() error {
		// wait until all pubs are complete or up to default timeout before attempting purge
		select {
		case <-jetStream.PublishAsyncComplete():
		case <-time.After(obs.js.opts.wait):
		}
		if err := obs.js.purgeStream(obs.stream, &StreamPurgeRequest{Subject: chunkSubj}); err != nil {
			return fmt.Errorf("could not cleanup bucket after erroneous put operation: %w", err)
		}
		return nil
	}

	m, h := NewMsg(chunkSubj), sha256.New()
	chunk, sent, total := make([]byte, meta.Opts.ChunkSize), 0, uint64(0)

	// set up the info object. The chunk upload sets the size and digest
	info := &ObjectInfo{Bucket: obs.name, NUID: newnuid, ObjectMeta: *meta}

	for r != nil {
		if ctx != nil {
			select {
			case <-ctx.Done():
				if ctx.Err() == context.Canceled {
					err = ctx.Err()
				} else {
					err = ErrTimeout
				}
			default:
			}
			if err != nil {
				if purgeErr := purgePartial(); purgeErr != nil {
					return nil, errors.Join(err, purgeErr)
				}
				return nil, err
			}
		}

		// Actual read.
		// TODO(dlc) - Deadline?
		n, readErr := r.Read(chunk)

		// Handle all non EOF errors
		if readErr != nil && readErr != io.EOF {
			if purgeErr := purgePartial(); purgeErr != nil {
				return nil, errors.Join(readErr, purgeErr)
			}
			return nil, readErr
		}

		// Add chunk only if we received data
		if n > 0 {
			// Chunk processing.
			m.Data = chunk[:n]
			h.Write(m.Data)

			// Send msg itself.
			if _, err := jetStream.PublishMsgAsync(m); err != nil {
				if purgeErr := purgePartial(); purgeErr != nil {
					return nil, errors.Join(err, purgeErr)
				}
				return nil, err
			}
			if err := getErr(); err != nil {
				if purgeErr := purgePartial(); purgeErr != nil {
					return nil, errors.Join(err, purgeErr)
				}
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
	mm := NewMsg(metaSubj)
	mm.Header.Set(MsgRollup, MsgRollupSubject)
	mm.Data, err = json.Marshal(info)
	if err != nil {
		if r != nil {
			if purgeErr := purgePartial(); purgeErr != nil {
				return nil, errors.Join(err, purgeErr)
			}
		}
		return nil, err
	}

	// Publish the meta message.
	_, err = jetStream.PublishMsgAsync(mm)
	if err != nil {
		if r != nil {
			if purgeErr := purgePartial(); purgeErr != nil {
				return nil, errors.Join(err, purgeErr)
			}
		}
		return nil, err
	}

	// Wait for all to be processed.
	select {
	case <-jetStream.PublishAsyncComplete():
		if err := getErr(); err != nil {
			if r != nil {
				if purgeErr := purgePartial(); purgeErr != nil {
					return nil, errors.Join(err, purgeErr)
				}
			}
			return nil, err
		}
	case <-time.After(obs.js.opts.wait):
		return nil, ErrTimeout
	}

	info.ModTime = time.Now().UTC() // This time is not actually the correct time

	// Delete any original chunks.
	if einfo != nil && !einfo.Deleted {
		echunkSubj := fmt.Sprintf(objChunksPreTmpl, obs.name, einfo.NUID)
		if err := obs.js.purgeStream(obs.stream, &StreamPurgeRequest{Subject: echunkSubj}); err != nil {
			return info, err
		}
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

// ObjectResult impl.
type objResult struct {
	sync.Mutex
	info        *ObjectInfo
	r           io.ReadCloser
	err         error
	ctx         context.Context
	digest      hash.Hash
	readTimeout time.Duration
}

func (info *ObjectInfo) isLink() bool {
	return info.ObjectMeta.Opts != nil && info.ObjectMeta.Opts.Link != nil
}

type GetObjectOpt interface {
	configureGetObject(opts *getObjectOpts) error
}
type getObjectOpts struct {
	ctx context.Context
	// Include deleted object in the result.
	showDeleted bool
}

type getObjectFn func(opts *getObjectOpts) error

func (opt getObjectFn) configureGetObject(opts *getObjectOpts) error {
	return opt(opts)
}

// GetObjectShowDeleted makes Get() return object if it was marked as deleted.
func GetObjectShowDeleted() GetObjectOpt {
	return getObjectFn(func(opts *getObjectOpts) error {
		opts.showDeleted = true
		return nil
	})
}

// For nats.Context() support.
func (ctx ContextOpt) configureGetObject(opts *getObjectOpts) error {
	opts.ctx = ctx
	return nil
}

// Get will pull the object from the underlying stream.
func (obs *obs) Get(name string, opts ...GetObjectOpt) (ObjectResult, error) {
	var o getObjectOpts
	for _, opt := range opts {
		if opt != nil {
			if err := opt.configureGetObject(&o); err != nil {
				return nil, err
			}
		}
	}
	ctx := o.ctx
	infoOpts := make([]GetObjectInfoOpt, 0)
	if ctx != nil {
		infoOpts = append(infoOpts, Context(ctx))
	}
	if o.showDeleted {
		infoOpts = append(infoOpts, GetObjectInfoShowDeleted())
	}

	// Grab meta info.
	info, err := obs.GetInfo(name, infoOpts...)
	if err != nil {
		return nil, err
	}
	if info.NUID == _EMPTY_ {
		return nil, ErrBadObjectMeta
	}

	// Check for object links. If single objects we do a pass through.
	if info.isLink() {
		if info.ObjectMeta.Opts.Link.Name == _EMPTY_ {
			return nil, ErrCantGetBucket
		}

		// is the link in the same bucket?
		lbuck := info.ObjectMeta.Opts.Link.Bucket
		if lbuck == obs.name {
			return obs.Get(info.ObjectMeta.Opts.Link.Name)
		}

		// different bucket
		lobs, err := obs.js.ObjectStore(lbuck)
		if err != nil {
			return nil, err
		}
		return lobs.Get(info.ObjectMeta.Opts.Link.Name)
	}

	result := &objResult{info: info, ctx: ctx, readTimeout: obs.js.opts.wait}
	if info.Size == 0 {
		return result, nil
	}

	pr, pw := net.Pipe()
	result.r = pr

	gotErr := func(m *Msg, err error) {
		pw.Close()
		m.Sub.Unsubscribe()
		result.setErr(err)
	}

	// For calculating sum256
	result.digest = sha256.New()

	processChunk := func(m *Msg) {
		var err error
		if ctx != nil {
			select {
			case <-ctx.Done():
				if errors.Is(ctx.Err(), context.Canceled) {
					err = ctx.Err()
				} else {
					err = ErrTimeout
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
	subscribeOpts := []SubOpt{
		OrderedConsumer(),
		BindStream(streamName),
	}
	_, err = obs.js.Subscribe(chunkSubj, processChunk, subscribeOpts...)
	if err != nil {
		return nil, err
	}

	return result, nil
}

// Delete will delete the object.
func (obs *obs) Delete(name string) error {
	// Grab meta info.
	info, err := obs.GetInfo(name, GetObjectInfoShowDeleted())
	if err != nil {
		return err
	}
	if info.NUID == _EMPTY_ {
		return ErrBadObjectMeta
	}

	// Place a rollup delete marker and publish the info
	info.Deleted = true
	info.Size, info.Chunks, info.Digest = 0, 0, _EMPTY_

	if err = publishMeta(info, obs.js); err != nil {
		return err
	}

	// Purge chunks for the object.
	chunkSubj := fmt.Sprintf(objChunksPreTmpl, obs.name, info.NUID)
	return obs.js.purgeStream(obs.stream, &StreamPurgeRequest{Subject: chunkSubj})
}

func publishMeta(info *ObjectInfo, js JetStreamContext) error {
	// marshal the object into json, don't store an actual time
	info.ModTime = time.Time{}
	data, err := json.Marshal(info)
	if err != nil {
		return err
	}

	// Prepare and publish the message.
	mm := NewMsg(fmt.Sprintf(objMetaPreTmpl, info.Bucket, encodeName(info.ObjectMeta.Name)))
	mm.Header.Set(MsgRollup, MsgRollupSubject)
	mm.Data = data
	if _, err := js.PublishMsg(mm); err != nil {
		return err
	}

	// set the ModTime in case it's returned to the user, even though it's not the correct time.
	info.ModTime = time.Now().UTC()
	return nil
}

// AddLink will add a link to another object if it's not deleted and not another link
// name is the name of this link object
// obj is what is being linked too
func (obs *obs) AddLink(name string, obj *ObjectInfo) (*ObjectInfo, error) {
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
	einfo, err := obs.GetInfo(name, GetObjectInfoShowDeleted())
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
	if err = publishMeta(info, obs.js); err != nil {
		return nil, err
	}

	return info, nil
}

// AddBucketLink will add a link to another object store.
func (ob *obs) AddBucketLink(name string, bucket ObjectStore) (*ObjectInfo, error) {
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
	einfo, err := ob.GetInfo(name, GetObjectInfoShowDeleted())
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
	err = publishMeta(info, ob.js)
	if err != nil {
		return nil, err
	}

	return info, nil
}

// PutBytes is convenience function to put a byte slice into this object store.
func (obs *obs) PutBytes(name string, data []byte, opts ...ObjectOpt) (*ObjectInfo, error) {
	return obs.Put(&ObjectMeta{Name: name}, bytes.NewReader(data), opts...)
}

// GetBytes is a convenience function to pull an object from this object store and return it as a byte slice.
func (obs *obs) GetBytes(name string, opts ...GetObjectOpt) ([]byte, error) {
	result, err := obs.Get(name, opts...)
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
func (obs *obs) PutString(name string, data string, opts ...ObjectOpt) (*ObjectInfo, error) {
	return obs.Put(&ObjectMeta{Name: name}, strings.NewReader(data), opts...)
}

// GetString is a convenience function to pull an object from this object store and return it as a string.
func (obs *obs) GetString(name string, opts ...GetObjectOpt) (string, error) {
	result, err := obs.Get(name, opts...)
	if err != nil {
		return _EMPTY_, err
	}
	defer result.Close()

	var b bytes.Buffer
	if _, err := b.ReadFrom(result); err != nil {
		return _EMPTY_, err
	}
	return b.String(), nil
}

// PutFile is convenience function to put a file into an object store.
func (obs *obs) PutFile(file string, opts ...ObjectOpt) (*ObjectInfo, error) {
	f, err := os.Open(file)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	return obs.Put(&ObjectMeta{Name: file}, f, opts...)
}

// GetFile is a convenience function to pull and object and place in a file.
func (obs *obs) GetFile(name, file string, opts ...GetObjectOpt) error {
	// Expect file to be new.
	f, err := os.OpenFile(file, os.O_WRONLY|os.O_CREATE, 0600)
	if err != nil {
		return err
	}
	defer f.Close()

	result, err := obs.Get(name, opts...)
	if err != nil {
		os.Remove(f.Name())
		return err
	}
	defer result.Close()

	// Stream copy to the file.
	_, err = io.Copy(f, result)
	return err
}

type GetObjectInfoOpt interface {
	configureGetInfo(opts *getObjectInfoOpts) error
}
type getObjectInfoOpts struct {
	ctx context.Context
	// Include deleted object in the result.
	showDeleted bool
}

type getObjectInfoFn func(opts *getObjectInfoOpts) error

func (opt getObjectInfoFn) configureGetInfo(opts *getObjectInfoOpts) error {
	return opt(opts)
}

// GetObjectInfoShowDeleted makes GetInfo() return object if it was marked as deleted.
func GetObjectInfoShowDeleted() GetObjectInfoOpt {
	return getObjectInfoFn(func(opts *getObjectInfoOpts) error {
		opts.showDeleted = true
		return nil
	})
}

// For nats.Context() support.
func (ctx ContextOpt) configureGetInfo(opts *getObjectInfoOpts) error {
	opts.ctx = ctx
	return nil
}

// GetInfo will retrieve the current information for the object.
func (obs *obs) GetInfo(name string, opts ...GetObjectInfoOpt) (*ObjectInfo, error) {
	// Grab last meta value we have.
	if name == "" {
		return nil, ErrNameRequired
	}
	var o getObjectInfoOpts
	for _, opt := range opts {
		if opt != nil {
			if err := opt.configureGetInfo(&o); err != nil {
				return nil, err
			}
		}
	}

	metaSubj := fmt.Sprintf(objMetaPreTmpl, obs.name, encodeName(name)) // used as data in a JS API call
	stream := fmt.Sprintf(objNameTmpl, obs.name)

	m, err := obs.js.GetLastMsg(stream, metaSubj)
	if err != nil {
		if errors.Is(err, ErrMsgNotFound) {
			err = ErrObjectNotFound
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
func (obs *obs) UpdateMeta(name string, meta *ObjectMeta) error {
	if meta == nil {
		return ErrBadObjectMeta
	}

	// Grab the current meta.
	info, err := obs.GetInfo(name)
	if err != nil {
		if errors.Is(err, ErrObjectNotFound) {
			return ErrUpdateMetaDeleted
		}
		return err
	}

	// If the new name is different from the old, and it exists, error
	// If there was an error that was not ErrObjectNotFound, error.
	if name != meta.Name {
		existingInfo, err := obs.GetInfo(meta.Name, GetObjectInfoShowDeleted())
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
	if err = publishMeta(info, obs.js); err != nil {
		return err
	}

	// did the name of this object change? We just stored the meta under the new name
	// so delete the meta from the old name via purge stream for subject
	if name != meta.Name {
		metaSubj := fmt.Sprintf(objMetaPreTmpl, obs.name, encodeName(name))
		return obs.js.purgeStream(obs.stream, &StreamPurgeRequest{Subject: metaSubj})
	}

	return nil
}

// Seal will seal the object store, no further modifications will be allowed.
func (obs *obs) Seal() error {
	stream := fmt.Sprintf(objNameTmpl, obs.name)
	si, err := obs.js.StreamInfo(stream)
	if err != nil {
		return err
	}
	// Seal the stream from being able to take on more messages.
	cfg := si.Config
	cfg.Sealed = true
	_, err = obs.js.UpdateStream(&cfg)
	return err
}

// Implementation for Watch
type objWatcher struct {
	updates chan *ObjectInfo
	sub     *Subscription
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
func (obs *obs) Watch(opts ...WatchOpt) (ObjectWatcher, error) {
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

	update := func(m *Msg) {
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
	_, err := obs.js.GetLastMsg(obs.stream, allMeta)
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
	subOpts := []SubOpt{OrderedConsumer(), BindStream(streamName)}
	if !o.includeHistory {
		subOpts = append(subOpts, DeliverLastPerSubject())
	}
	if o.updatesOnly {
		subOpts = append(subOpts, DeliverNew())
	}
	sub, err := obs.js.Subscribe(allMeta, update, subOpts...)
	if err != nil {
		return nil, err
	}
	// Set us up to close when the waitForMessages func returns.
	sub.pDone = func(_ string) {
		close(w.updates)
	}
	w.sub = sub
	return w, nil
}

type ListObjectsOpt interface {
	configureListObjects(opts *listObjectOpts) error
}
type listObjectOpts struct {
	ctx context.Context
	// Include deleted objects in the result channel.
	showDeleted bool
}

type listObjectsFn func(opts *listObjectOpts) error

func (opt listObjectsFn) configureListObjects(opts *listObjectOpts) error {
	return opt(opts)
}

// ListObjectsShowDeleted makes ListObjects() return deleted objects.
func ListObjectsShowDeleted() ListObjectsOpt {
	return listObjectsFn(func(opts *listObjectOpts) error {
		opts.showDeleted = true
		return nil
	})
}

// For nats.Context() support.
func (ctx ContextOpt) configureListObjects(opts *listObjectOpts) error {
	opts.ctx = ctx
	return nil
}

// List will list all the objects in this store.
func (obs *obs) List(opts ...ListObjectsOpt) ([]*ObjectInfo, error) {
	var o listObjectOpts
	for _, opt := range opts {
		if opt != nil {
			if err := opt.configureListObjects(&o); err != nil {
				return nil, err
			}
		}
	}
	watchOpts := make([]WatchOpt, 0)
	if !o.showDeleted {
		watchOpts = append(watchOpts, IgnoreDeletes())
	}
	watcher, err := obs.Watch(watchOpts...)
	if err != nil {
		return nil, err
	}
	defer watcher.Stop()
	if o.ctx == nil {
		o.ctx = context.Background()
	}

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
		case <-o.ctx.Done():
			return nil, o.ctx.Err()
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
func (obs *obs) Status() (ObjectStoreStatus, error) {
	nfo, err := obs.js.StreamInfo(obs.stream)
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
	readDeadline := time.Now().Add(o.readTimeout)
	if ctx := o.ctx; ctx != nil {
		if deadline, ok := ctx.Deadline(); ok {
			readDeadline = deadline
		}
		select {
		case <-ctx.Done():
			if ctx.Err() == context.Canceled {
				o.err = ctx.Err()
			} else {
				o.err = ErrTimeout
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
	r.SetReadDeadline(readDeadline)
	n, err = r.Read(p)
	if err, ok := err.(net.Error); ok && err.Timeout() {
		if ctx := o.ctx; ctx != nil {
			select {
			case <-ctx.Done():
				if ctx.Err() == context.Canceled {
					return 0, ctx.Err()
				} else {
					return 0, ErrTimeout
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
func (js *js) ObjectStoreNames(opts ...ObjectOpt) <-chan string {
	var o objOpts
	for _, opt := range opts {
		if opt != nil {
			if err := opt.configureObject(&o); err != nil {
				return nil
			}
		}
	}
	ch := make(chan string)
	var cancel context.CancelFunc
	if o.ctx == nil {
		o.ctx, cancel = context.WithTimeout(context.Background(), defaultRequestWait)
	}
	l := &streamLister{js: js}
	l.js.opts.streamListSubject = fmt.Sprintf(objAllChunksPreTmpl, "*")
	l.js.opts.ctx = o.ctx
	go func() {
		if cancel != nil {
			defer cancel()
		}
		defer close(ch)
		for l.Next() {
			for _, info := range l.Page() {
				if !strings.HasPrefix(info.Config.Name, "OBJ_") {
					continue
				}
				select {
				case ch <- info.Config.Name:
				case <-o.ctx.Done():
					return
				}
			}
		}
	}()

	return ch
}

// ObjectStores is used to retrieve a list of bucket statuses
func (js *js) ObjectStores(opts ...ObjectOpt) <-chan ObjectStoreStatus {
	var o objOpts
	for _, opt := range opts {
		if opt != nil {
			if err := opt.configureObject(&o); err != nil {
				return nil
			}
		}
	}
	ch := make(chan ObjectStoreStatus)
	var cancel context.CancelFunc
	if o.ctx == nil {
		o.ctx, cancel = context.WithTimeout(context.Background(), defaultRequestWait)
	}
	l := &streamLister{js: js}
	l.js.opts.streamListSubject = fmt.Sprintf(objAllChunksPreTmpl, "*")
	l.js.opts.ctx = o.ctx
	go func() {
		if cancel != nil {
			defer cancel()
		}
		defer close(ch)
		for l.Next() {
			for _, info := range l.Page() {
				if !strings.HasPrefix(info.Config.Name, "OBJ_") {
					continue
				}
				select {
				case ch <- &ObjectBucketStatus{
					nfo:    info,
					bucket: strings.TrimPrefix(info.Config.Name, "OBJ_"),
				}:
				case <-o.ctx.Done():
					return
				}
			}
		}
	}()

	return ch
}
