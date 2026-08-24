//go:build !appengine

package util

import (
	"unsafe"
)

// BytesToString converts byte slice to string.
func BytesToString(b []byte) string {
	return unsafe.String(unsafe.SliceData(b), len(b))
}

// StringToBytes converts string to byte slice.
func StringToBytes(s string) []byte {
	return unsafe.Slice(unsafe.StringData(s), len(s))
}
