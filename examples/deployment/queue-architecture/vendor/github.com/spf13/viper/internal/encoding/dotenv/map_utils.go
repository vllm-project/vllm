package dotenv

import (
	"strings"

	"github.com/spf13/cast"
)

// flattenAndMergeMap recursively flattens the given map into a new map
// Code is based on the function with the same name in the main package.
// TODO: move it to a common place.
func flattenAndMergeMap(shadow, m map[string]any, prefix, delimiter string) map[string]any {
	if shadow != nil && prefix != "" && shadow[prefix] != nil {
		// prefix is shadowed => nothing more to flatten
		return shadow
	}
	if shadow == nil {
		shadow = make(map[string]any)
	}

	var m2 map[string]any
	if prefix != "" {
		prefix += delimiter
	}
	for k, val := range m {
		fullKey := prefix + k
		switch val := val.(type) {
		case map[string]any:
			m2 = val
		case map[any]any:
			m2 = cast.ToStringMap(val)
		default:
			// immediate value
			shadow[strings.ToLower(fullKey)] = val
			continue
		}
		// recursively merge to shadow map
		shadow = flattenAndMergeMap(shadow, m2, fullKey, delimiter)
	}
	return shadow
}
