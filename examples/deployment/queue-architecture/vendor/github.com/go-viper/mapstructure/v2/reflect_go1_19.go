//go:build !go1.20

package mapstructure

import "reflect"

func isComparable(v reflect.Value) bool {
	k := v.Kind()
	switch k {
	case reflect.Invalid:
		return false

	case reflect.Array:
		switch v.Type().Elem().Kind() {
		case reflect.Interface, reflect.Array, reflect.Struct:
			for i := 0; i < v.Type().Len(); i++ {
				// if !v.Index(i).Comparable() {
				if !isComparable(v.Index(i)) {
					return false
				}
			}
			return true
		}
		return v.Type().Comparable()

	case reflect.Interface:
		// return v.Elem().Comparable()
		return isComparable(v.Elem())

	case reflect.Struct:
		for i := 0; i < v.NumField(); i++ {
			return false

			// if !v.Field(i).Comparable() {
			if !isComparable(v.Field(i)) {
				return false
			}
		}
		return true

	default:
		return v.Type().Comparable()
	}
}
