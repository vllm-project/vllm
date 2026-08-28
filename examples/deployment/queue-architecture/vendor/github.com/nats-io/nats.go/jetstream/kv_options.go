// Copyright 2024 The NATS Authors
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
	"fmt"
	"time"
)

type watchOptFn func(opts *watchOpts) error

func (opt watchOptFn) configureWatcher(opts *watchOpts) error {
	return opt(opts)
}

// IncludeHistory instructs the key watcher to include historical values as
// well (up to KeyValueMaxHistory).
func IncludeHistory() WatchOpt {
	return watchOptFn(func(opts *watchOpts) error {
		if opts.updatesOnly {
			return fmt.Errorf("%w: include history cannot be used with updates only", ErrInvalidOption)
		}
		opts.includeHistory = true
		return nil
	})
}

// UpdatesOnly instructs the key watcher to only include updates on values
// (without latest values when started).
func UpdatesOnly() WatchOpt {
	return watchOptFn(func(opts *watchOpts) error {
		if opts.includeHistory {
			return fmt.Errorf("%w: updates only cannot be used with include history", ErrInvalidOption)
		}
		opts.updatesOnly = true
		return nil
	})
}

// IgnoreDeletes will prevent the key watcher from passing any deleted keys.
func IgnoreDeletes() WatchOpt {
	return watchOptFn(func(opts *watchOpts) error {
		opts.ignoreDeletes = true
		return nil
	})
}

// MetaOnly instructs the key watcher to retrieve only the entry metadata, not
// the entry value.
func MetaOnly() WatchOpt {
	return watchOptFn(func(opts *watchOpts) error {
		opts.metaOnly = true
		return nil
	})
}

// ResumeFromRevision instructs the key watcher to resume from a specific
// revision number.
func ResumeFromRevision(revision uint64) WatchOpt {
	return watchOptFn(func(opts *watchOpts) error {
		opts.resumeFromRevision = revision
		return nil
	})
}

// DeleteMarkersOlderThan indicates that delete or purge markers older than that
// will be deleted as part of [KeyValue.PurgeDeletes] operation, otherwise, only the data
// will be removed but markers that are recent will be kept.
// Note that if no option is specified, the default is 30 minutes. You can set
// this option to a negative value to instruct to always remove the markers,
// regardless of their age.
type DeleteMarkersOlderThan time.Duration

func (ttl DeleteMarkersOlderThan) configurePurge(opts *purgeOpts) error {
	opts.dmthr = time.Duration(ttl)
	return nil
}

type deleteOptFn func(opts *deleteOpts) error

func (opt deleteOptFn) configureDelete(opts *deleteOpts) error {
	return opt(opts)
}

// LastRevision deletes if the latest revision matches the provided one. If the
// provided revision is not the latest, the delete will return an error.
func LastRevision(revision uint64) KVDeleteOpt {
	return deleteOptFn(func(opts *deleteOpts) error {
		opts.revision = revision
		return nil
	})
}

// PurgeTTL sets the TTL for the purge operation.
// After the TTL expires, the delete markers will be removed.
// This requires LimitMarkerTTL to be enabled on the bucket.
// Note that this is not the same as the TTL for the key itself, which is set
// using the KeyTTL option when creating the key.
func PurgeTTL(ttl time.Duration) KVDeleteOpt {
	return deleteOptFn(func(opts *deleteOpts) error {
		opts.ttl = ttl
		return nil
	})
}

type createOptFn func(opts *createOpts) error

func (opt createOptFn) configureCreate(opts *createOpts) error {
	return opt(opts)
}

// KeyTTL sets the TTL for the key. This is the time after which the key will be
// automatically deleted. The TTL is set when the key is created and cannot be
// changed later. This requires LimitMarkerTTL to be enabled on the bucket.
func KeyTTL(ttl time.Duration) KVCreateOpt {
	return createOptFn(func(opts *createOpts) error {
		opts.ttl = ttl
		return nil
	})
}
