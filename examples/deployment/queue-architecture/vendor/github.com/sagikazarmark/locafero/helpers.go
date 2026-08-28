package locafero

import "fmt"

// NameWithExtensions creates a list of names from a base name and a list of extensions.
//
// TODO: find a better name for this function.
func NameWithExtensions(baseName string, extensions ...string) []string {
	var names []string

	if baseName == "" {
		return names
	}

	for _, ext := range extensions {
		if ext == "" {
			continue
		}

		names = append(names, fmt.Sprintf("%s.%s", baseName, ext))
	}

	return names
}

// NameWithOptionalExtensions creates a list of names from a base name and a list of extensions,
// plus it adds the base name (without any extensions) to the end of the list.
//
// TODO: find a better name for this function.
func NameWithOptionalExtensions(baseName string, extensions ...string) []string {
	var names []string

	if baseName == "" {
		return names
	}

	names = NameWithExtensions(baseName, extensions...)
	names = append(names, baseName)

	return names
}
