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

// defaultLogLevel is the log level applied when spec.logLevel is unset (it
// mirrors the kubebuilder default on both engine kinds).
const defaultLogLevel = "INFO"

// labelValueTrue is the string value of boolean-style node-selector labels
// (e.g. nvidia.com/gpu.present: "true").
const labelValueTrue = "true"

// SetDefaults applies defaults that cannot be expressed purely via kubebuilder markers.
func (e *LMCacheEngine) SetDefaults() {
	spec := &e.Spec

	// Default logLevel to INFO if unset (belt-and-suspenders with kubebuilder default).
	if spec.LogLevel == nil {
		info := defaultLogLevel
		spec.LogLevel = &info
	}

	if spec.GPUVendor == nil {
		v := GPUVendorNvidia
		spec.GPUVendor = &v
	}

	if spec.NodeSelector == nil && *spec.GPUVendor == GPUVendorNvidia {
		spec.NodeSelector = map[string]string{
			"nvidia.com/gpu.present": labelValueTrue,
		}
	}
}
