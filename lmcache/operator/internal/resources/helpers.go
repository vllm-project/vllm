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

// derefInt32 returns the value pointed to by p, or def if p is nil.
func derefInt32(p *int32, def int32) int32 {
	if p != nil {
		return *p
	}
	return def
}

// derefString returns the value pointed to by p, or def if p is nil.
func derefString(p *string, def string) string {
	if p != nil {
		return *p
	}
	return def
}

// derefBool returns the value pointed to by p, or def if p is nil.
func derefBool(p *bool, def bool) bool {
	if p != nil {
		return *p
	}
	return def
}

// derefFloat64 returns the value pointed to by p, or def if p is nil.
func derefFloat64(p *float64, def float64) float64 {
	if p != nil {
		return *p
	}
	return def
}

// RESPAuthSecretName returns the name of the managed local copy of the
// RESP auth secret for a given LMCacheEngine.
func RESPAuthSecretName(engineName string) string {
	return engineName + "-resp-auth"
}
