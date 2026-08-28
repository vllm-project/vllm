//go:build go1.20

package mapstructure

import "reflect"

// TODO: remove once we drop support for Go <1.20
func isComparable(v reflect.Value) bool {
	return v.Comparable()
}
