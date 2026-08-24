package util

import (
	"fmt"
	"math"
	"strconv"
)

// ParseFloat parses a Redis RESP3 float reply into a Go float64,
// handling "inf", "-inf", "nan" per Redis conventions.
func ParseStringToFloat(s string) (float64, error) {
	switch s {
	case "inf":
		return math.Inf(1), nil
	case "-inf":
		return math.Inf(-1), nil
	case "nan", "-nan":
		return math.NaN(), nil
	}
	return strconv.ParseFloat(s, 64)
}

// MustParseFloat is like ParseFloat but panics on parse errors.
func MustParseFloat(s string) float64 {
	f, err := ParseStringToFloat(s)
	if err != nil {
		panic(fmt.Sprintf("redis: failed to parse float %q: %v", s, err))
	}
	return f
}

// SafeIntToInt32 safely converts an int to int32, returning an error if overflow would occur.
func SafeIntToInt32(value int, fieldName string) (int32, error) {
	if value > math.MaxInt32 {
		return 0, fmt.Errorf("redis: %s value %d exceeds maximum allowed value %d", fieldName, value, math.MaxInt32)
	}
	if value < math.MinInt32 {
		return 0, fmt.Errorf("redis: %s value %d is below minimum allowed value %d", fieldName, value, math.MinInt32)
	}
	return int32(value), nil
}
