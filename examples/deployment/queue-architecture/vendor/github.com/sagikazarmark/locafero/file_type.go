package locafero

import "io/fs"

// FileType represents the kind of entries [Finder] can return.
type FileType int

// FileType represents the kind of entries [Finder] can return.
const (
	FileTypeAny FileType = iota
	FileTypeFile
	FileTypeDir

	// Deprecated: Use [FileTypeAny] instead.
	FileTypeAll = FileTypeAny
)

func (ft FileType) match(info fs.FileInfo) bool {
	switch ft {
	case FileTypeAny:
		return true

	case FileTypeFile:
		return info.Mode().IsRegular()

	case FileTypeDir:
		return info.IsDir()

	default:
		return false
	}
}
