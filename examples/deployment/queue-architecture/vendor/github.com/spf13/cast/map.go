// Copyright © 2014 Steve Francia <spf@spf13.com>.
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file.

package cast

import (
	"encoding/json"
	"fmt"
	"reflect"
)

func toMapE[K comparable, V any](i any, keyFn func(any) K, valFn func(any) V) (map[K]V, error) {
	m := map[K]V{}

	if i == nil {
		return m, fmt.Errorf(errorMsg, i, i, m)
	}

	switch v := i.(type) {
	case map[K]V:
		return v, nil

	case map[K]any:
		for k, val := range v {
			m[k] = valFn(val)
		}

		return m, nil

	case map[any]V:
		for k, val := range v {
			m[keyFn(k)] = val
		}

		return m, nil

	case map[any]any:
		for k, val := range v {
			m[keyFn(k)] = valFn(val)
		}

		return m, nil

	case string:
		err := jsonStringToObject(v, &m)
		return m, err

	default:
		return m, fmt.Errorf(errorMsg, i, i, m)
	}
}

func toStringMapE[T any](i any, fn func(any) T) (map[string]T, error) {
	return toMapE(i, ToString, fn)
}

// ToStringMapStringE casts any value to a map[string]string type.
func ToStringMapStringE(i any) (map[string]string, error) {
	return toStringMapE(i, ToString)
}

// ToStringMapStringSliceE casts any value to a map[string][]string type.
func ToStringMapStringSliceE(i any) (map[string][]string, error) {
	m := map[string][]string{}

	switch v := i.(type) {
	case map[string][]string:
		return v, nil
	case map[string][]any:
		for k, val := range v {
			m[ToString(k)] = ToStringSlice(val)
		}
		return m, nil
	case map[string]string:
		for k, val := range v {
			m[ToString(k)] = []string{val}
		}
	case map[string]any:
		for k, val := range v {
			switch vt := val.(type) {
			case []any:
				m[ToString(k)] = ToStringSlice(vt)
			case []string:
				m[ToString(k)] = vt
			default:
				m[ToString(k)] = []string{ToString(val)}
			}
		}
		return m, nil
	case map[any][]string:
		for k, val := range v {
			m[ToString(k)] = ToStringSlice(val)
		}
		return m, nil
	case map[any]string:
		for k, val := range v {
			m[ToString(k)] = ToStringSlice(val)
		}
		return m, nil
	case map[any][]any:
		for k, val := range v {
			m[ToString(k)] = ToStringSlice(val)
		}
		return m, nil
	case map[any]any:
		for k, val := range v {
			key, err := ToStringE(k)
			if err != nil {
				return m, fmt.Errorf(errorMsg, i, i, m)
			}
			value, err := ToStringSliceE(val)
			if err != nil {
				return m, fmt.Errorf(errorMsg, i, i, m)
			}
			m[key] = value
		}
	case string:
		err := jsonStringToObject(v, &m)
		return m, err
	default:
		return m, fmt.Errorf(errorMsg, i, i, m)
	}

	return m, nil
}

// ToStringMapBoolE casts any value to a map[string]bool type.
func ToStringMapBoolE(i any) (map[string]bool, error) {
	return toStringMapE(i, ToBool)
}

// ToStringMapE casts any value to a map[string]any type.
func ToStringMapE(i any) (map[string]any, error) {
	fn := func(i any) any { return i }

	return toStringMapE(i, fn)
}

func toStringMapIntE[T int | int64](i any, fn func(any) T, fnE func(any) (T, error)) (map[string]T, error) {
	m := map[string]T{}

	if i == nil {
		return nil, fmt.Errorf(errorMsg, i, i, m)
	}

	switch v := i.(type) {
	case map[string]T:
		return v, nil

	case map[string]any:
		for k, val := range v {
			m[k] = fn(val)
		}

		return m, nil

	case map[any]T:
		for k, val := range v {
			m[ToString(k)] = val
		}

		return m, nil

	case map[any]any:
		for k, val := range v {
			m[ToString(k)] = fn(val)
		}

		return m, nil

	case string:
		err := jsonStringToObject(v, &m)
		return m, err
	}

	if reflect.TypeOf(i).Kind() != reflect.Map {
		return m, fmt.Errorf(errorMsg, i, i, m)
	}

	mVal := reflect.ValueOf(m)
	v := reflect.ValueOf(i)

	for _, keyVal := range v.MapKeys() {
		val, err := fnE(v.MapIndex(keyVal).Interface())
		if err != nil {
			return m, fmt.Errorf(errorMsg, i, i, m)
		}

		mVal.SetMapIndex(keyVal, reflect.ValueOf(val))
	}

	return m, nil
}

// ToStringMapIntE casts any value to a map[string]int type.
func ToStringMapIntE(i any) (map[string]int, error) {
	return toStringMapIntE(i, ToInt, ToIntE)
}

// ToStringMapInt64E casts any value to a map[string]int64 type.
func ToStringMapInt64E(i any) (map[string]int64, error) {
	return toStringMapIntE(i, ToInt64, ToInt64E)
}

// jsonStringToObject attempts to unmarshall a string as JSON into
// the object passed as pointer.
func jsonStringToObject(s string, v any) error {
	data := []byte(s)
	return json.Unmarshal(data, v)
}
