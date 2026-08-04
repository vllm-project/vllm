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

// SetDefaults applies defaults that cannot be expressed purely via kubebuilder
// markers. It mirrors LMCacheEngine.SetDefaults and additionally pins the blend
// tunables, injection defaults, and the CacheBlend-required chunk size of 256.
func (e *CacheBlendEngine) SetDefaults() {
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

	// CacheBlend requires chunk_size == 256 (block_size 64 * 4). Default the
	// server block and pin chunkSize to 256 if unset.
	if spec.Server == nil {
		spec.Server = &ServerSpec{}
	}
	if spec.Server.ChunkSize == nil {
		cs := CacheBlendChunkSize
		spec.Server.ChunkSize = &cs
	}

	// Blend tunables.
	if spec.Blend == nil {
		spec.Blend = &BlendSpec{}
	}
	if spec.Blend.CheckLayer == nil {
		cl := int32(1)
		spec.Blend.CheckLayer = &cl
	}
	if spec.Blend.RecompRatio == nil {
		rr := 0.15
		spec.Blend.RecompRatio = &rr
	}

	// Injection defaults.
	if spec.Injection == nil {
		spec.Injection = &InjectionSpec{}
	}
	// payloadImage tag/pullPolicy default when the image is set; repository is
	// required (no cluster-wide default for the private payload image).
	if spec.Injection.PayloadImage != nil {
		if spec.Injection.PayloadImage.Tag == nil {
			t := "latest"
			spec.Injection.PayloadImage.Tag = &t
		}
		if spec.Injection.PayloadImage.PullPolicy == nil {
			pp := "IfNotPresent"
			spec.Injection.PayloadImage.PullPolicy = &pp
		}
	}
	if spec.Injection.Cudagraph == nil {
		cg := CudagraphEager
		spec.Injection.Cudagraph = &cg
	}
}
