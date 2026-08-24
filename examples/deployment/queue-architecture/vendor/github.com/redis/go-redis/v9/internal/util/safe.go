//go:build appengine

package util

func BytesToString(b []byte) string {
	return string(b)
}

func StringToBytes(s string) []byte {
	return []byte(s)
}
