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

// GetObjectShowDeleted makes [ObjectStore.Get] return object even if it was
// marked as deleted.
func GetObjectShowDeleted() GetObjectOpt {
	return func(opts *getObjectOpts) error {
		opts.showDeleted = true
		return nil
	}
}

// GetObjectInfoShowDeleted makes [ObjectStore.GetInfo] return object info event
// if it was marked as deleted.
func GetObjectInfoShowDeleted() GetObjectInfoOpt {
	return func(opts *getObjectInfoOpts) error {
		opts.showDeleted = true
		return nil
	}
}

// ListObjectsShowDeleted makes [ObjectStore.ListObjects] also return deleted
// objects.
func ListObjectsShowDeleted() ListObjectsOpt {
	return func(opts *listObjectOpts) error {
		opts.showDeleted = true
		return nil
	}
}
