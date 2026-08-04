/*
Copyright 2026.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package resources

import "maps"

// SelectorLabels returns the immutable label subset used for pod selectors.
func SelectorLabels(name string) map[string]string {
	return map[string]string{
		"app.kubernetes.io/name":       "lmcache",
		"app.kubernetes.io/instance":   name,
		"app.kubernetes.io/managed-by": "lmcache-operator",
	}
}

// StandardLabels returns the full label set for all owned resources.
func StandardLabels(name string) map[string]string {
	labels := SelectorLabels(name)
	labels["app.kubernetes.io/component"] = "cache-engine"
	return labels
}

// MergeLabels merges multiple label maps. Later maps take precedence.
func MergeLabels(labelMaps ...map[string]string) map[string]string {
	result := make(map[string]string)
	for _, m := range labelMaps {
		maps.Copy(result, m)
	}
	return result
}
