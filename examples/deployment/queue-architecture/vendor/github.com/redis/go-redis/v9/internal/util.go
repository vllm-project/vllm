package internal

import (
	"context"
	"math"
	"net"
	"strconv"
	"strings"
	"time"

	"github.com/redis/go-redis/v9/internal/util"
)

// String representations of special float values.
// Values are lowercase for consistency with Redis RESP2 protocol responses.
const (
	NaN  = "nan"  // Not a Number
	Inf  = "inf"  // Positive infinity
	NInf = "-inf" // Negative infinity
)

// FormatFloat formats a float64 to string, normalizing special values
// (NaN, Inf) to lowercase for consistency with Redis RESP2 protocol.
func FormatFloat(f float64) string {
	switch {
	case math.IsNaN(f):
		return NaN
	case math.IsInf(f, 1):
		return Inf
	case math.IsInf(f, -1):
		return NInf
	default:
		return strconv.FormatFloat(f, 'f', -1, 64)
	}
}

func Sleep(ctx context.Context, dur time.Duration) error {
	t := time.NewTimer(dur)
	defer t.Stop()

	select {
	case <-t.C:
		return nil
	case <-ctx.Done():
		return ctx.Err()
	}
}

func ToLower(s string) string {
	if isLower(s) {
		return s
	}

	b := make([]byte, len(s))
	for i := range b {
		c := s[i]
		if c >= 'A' && c <= 'Z' {
			c += 'a' - 'A'
		}
		b[i] = c
	}
	return util.BytesToString(b)
}

func isLower(s string) bool {
	for i := 0; i < len(s); i++ {
		c := s[i]
		if c >= 'A' && c <= 'Z' {
			return false
		}
	}
	return true
}

func ReplaceSpaces(s string) string {
	return strings.ReplaceAll(s, " ", "-")
}

func GetAddr(addr string) string {
	ind := strings.LastIndexByte(addr, ':')
	if ind == -1 {
		return ""
	}

	if strings.IndexByte(addr, '.') != -1 {
		return addr
	}

	if addr[0] == '[' {
		return addr
	}
	return net.JoinHostPort(addr[:ind], addr[ind+1:])
}

func ToInteger(val interface{}) int {
	switch v := val.(type) {
	case int:
		return v
	case int64:
		return int(v)
	case string:
		i, _ := strconv.Atoi(v)
		return i
	default:
		return 0
	}
}

func ToFloat(val interface{}) float64 {
	switch v := val.(type) {
	case float64:
		return v
	case string:
		f, _ := strconv.ParseFloat(v, 64)
		return f
	default:
		return 0.0
	}
}

func ToString(val interface{}) string {
	if str, ok := val.(string); ok {
		return str
	}
	return ""
}

func ToStringSlice(val interface{}) []string {
	if arr, ok := val.([]interface{}); ok {
		result := make([]string, len(arr))
		for i, v := range arr {
			result[i] = ToString(v)
		}
		return result
	}
	return nil
}
