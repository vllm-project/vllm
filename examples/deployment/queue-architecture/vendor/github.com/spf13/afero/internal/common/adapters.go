// Copyright © 2022 Steve Francia <spf@spf13.com>.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package common

import "io/fs"

// FileInfoDirEntry provides an adapter from os.FileInfo to fs.DirEntry
type FileInfoDirEntry struct {
	fs.FileInfo
}

var _ fs.DirEntry = FileInfoDirEntry{}

func (d FileInfoDirEntry) Type() fs.FileMode { return d.FileInfo.Mode().Type() }

func (d FileInfoDirEntry) Info() (fs.FileInfo, error) { return d.FileInfo, nil }
