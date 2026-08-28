// Copyright © 2014 Steve Francia <spf@spf13.com>.
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file.

package cast

import (
	"encoding/json"
	"errors"
	"fmt"
	"strings"
	"time"

	"github.com/spf13/cast/internal"
)

// ToTimeE any value to a [time.Time] type.
func ToTimeE(i any) (time.Time, error) {
	return ToTimeInDefaultLocationE(i, time.UTC)
}

// ToTimeInDefaultLocationE casts an empty interface to [time.Time],
// interpreting inputs without a timezone to be in the given location,
// or the local timezone if nil.
func ToTimeInDefaultLocationE(i any, location *time.Location) (tim time.Time, err error) {
	i, _ = indirect(i)

	switch v := i.(type) {
	case time.Time:
		return v, nil
	case string:
		return StringToDateInDefaultLocation(v, location)
	case json.Number:
		// Originally this used ToInt64E, but adding string float conversion broke ToTime.
		// the behavior of ToTime would have changed if we continued using it.
		// For now, using json.Number's own Int64 method should be good enough to preserve backwards compatibility.
		v = json.Number(trimZeroDecimal(string(v)))
		s, err1 := v.Int64()
		if err1 != nil {
			return time.Time{}, fmt.Errorf(errorMsg, i, i, time.Time{})
		}
		return time.Unix(s, 0), nil
	case int:
		return time.Unix(int64(v), 0), nil
	case int32:
		return time.Unix(int64(v), 0), nil
	case int64:
		return time.Unix(v, 0), nil
	case uint:
		return time.Unix(int64(v), 0), nil
	case uint32:
		return time.Unix(int64(v), 0), nil
	case uint64:
		return time.Unix(int64(v), 0), nil
	case nil:
		return time.Time{}, nil
	default:
		return time.Time{}, fmt.Errorf(errorMsg, i, i, time.Time{})
	}
}

// ToDurationE casts any value to a [time.Duration] type.
func ToDurationE(i any) (time.Duration, error) {
	i, _ = indirect(i)

	switch s := i.(type) {
	case time.Duration:
		return s, nil
	case int, int8, int16, int32, int64, uint, uint8, uint16, uint32, uint64:
		v, err := ToInt64E(s)
		if err != nil {
			// TODO: once there is better error handling, this should be easier
			return 0, errors.New(strings.ReplaceAll(err.Error(), " int64", "time.Duration"))
		}

		return time.Duration(v), nil
	case float32, float64, float64EProvider, float64Provider:
		v, err := ToFloat64E(s)
		if err != nil {
			// TODO: once there is better error handling, this should be easier
			return 0, errors.New(strings.ReplaceAll(err.Error(), " float64", "time.Duration"))
		}

		return time.Duration(v), nil
	case string:
		if !strings.ContainsAny(s, "nsuµmh") {
			return time.ParseDuration(s + "ns")
		}

		return time.ParseDuration(s)
	case nil:
		return time.Duration(0), nil
	default:
		if i, ok := resolveAlias(i); ok {
			return ToDurationE(i)
		}

		return 0, fmt.Errorf(errorMsg, i, i, time.Duration(0))
	}
}

// StringToDate attempts to parse a string into a [time.Time] type using a
// predefined list of formats.
//
// If no suitable format is found, an error is returned.
func StringToDate(s string) (time.Time, error) {
	return internal.ParseDateWith(s, time.UTC, internal.TimeFormats)
}

// StringToDateInDefaultLocation casts an empty interface to a [time.Time],
// interpreting inputs without a timezone to be in the given location,
// or the local timezone if nil.
func StringToDateInDefaultLocation(s string, location *time.Location) (time.Time, error) {
	return internal.ParseDateWith(s, location, internal.TimeFormats)
}
