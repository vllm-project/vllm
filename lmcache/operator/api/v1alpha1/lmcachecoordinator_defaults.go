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

package v1alpha1

// defaultCoordinatorReplicas is the replica count applied when spec.replicas is
// unset (belt-and-suspenders with the kubebuilder default).
const defaultCoordinatorReplicas int32 = 1

// SetDefaults applies defaults that cannot be expressed purely via kubebuilder
// markers. The numeric coordinator knobs (port, timeouts, ratios) are defaulted
// by their kubebuilder markers, matching MPCoordinatorConfig.
func (c *LMCacheCoordinator) SetDefaults() {
	spec := &c.Spec

	if spec.LogLevel == nil {
		info := defaultLogLevel
		spec.LogLevel = &info
	}

	if spec.Replicas == nil {
		r := defaultCoordinatorReplicas
		spec.Replicas = &r
	}
}
