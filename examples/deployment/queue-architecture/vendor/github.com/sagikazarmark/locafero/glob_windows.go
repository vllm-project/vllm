//go:build windows

package locafero

// See [filepath.Match]:
//
//	On Windows, escaping is disabled. Instead, '\\' is treated as path separator.
const globMatch = "*?[]^"
