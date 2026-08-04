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

import (
	"k8s.io/apimachinery/pkg/util/validation/field"
)

// ValidateSpec validates the CacheBlendEngineSpec and returns any validation
// errors. It mirrors LMCacheEngine.ValidateSpec and additionally enforces the
// CacheBlend invariants: chunkSize == 256, recompRatio in (0, 1], and
// checkLayer >= 0.
func (e *CacheBlendEngine) ValidateSpec() field.ErrorList {
	var errs field.ErrorList
	spec := &e.Spec

	// l1.sizeGB must be > 0
	if spec.L1.SizeGB <= 0 {
		errs = append(errs, field.Invalid(field.NewPath("spec", "l1", "sizeGB"), spec.L1.SizeGB, "must be greater than 0"))
	}

	errs = append(errs, validateEvictionSpec(spec.Eviction)...)

	// Server validation
	if spec.Server != nil {
		serverPath := field.NewPath("spec", "server")

		if spec.Server.Port != nil {
			port := *spec.Server.Port
			if port < 1024 || port > 65535 {
				errs = append(errs, field.Invalid(serverPath.Child("port"), port, "must be in [1024, 65535]"))
			}
		}

		// CacheBlend requires chunk_size == 256 (block_size 64 * 4).
		if spec.Server.ChunkSize != nil && *spec.Server.ChunkSize != CacheBlendChunkSize {
			errs = append(errs, field.Invalid(serverPath.Child("chunkSize"), *spec.Server.ChunkSize,
				"must be 256 for CacheBlend (chunk_size == block_size 64 * 4)"))
		}
	}

	// Blend validation
	if spec.Blend != nil {
		blendPath := field.NewPath("spec", "blend")

		if spec.Blend.CheckLayer != nil && *spec.Blend.CheckLayer < 0 {
			errs = append(errs, field.Invalid(blendPath.Child("checkLayer"), *spec.Blend.CheckLayer, "must be >= 0"))
		}

		if spec.Blend.RecompRatio != nil {
			rr := *spec.Blend.RecompRatio
			if rr <= 0.0 || rr > 1.0 {
				errs = append(errs, field.Invalid(blendPath.Child("recompRatio"), rr, "must be in (0.0, 1.0]"))
			}
		}
	}

	// Injection validation. injection.payloadImage.repository is functionally
	// required: the mutating webhook needs it to inject a valid init container.
	// Without it the webhook would produce a Pod with an empty init-container
	// image, which the API server rejects at Pod creation. Enforce it here so
	// the misconfiguration is caught at `kubectl apply` time instead.
	injPath := field.NewPath("spec", "injection")
	if spec.Injection == nil {
		errs = append(errs, field.Required(injPath, "must be specified for CacheBlend injection"))
	} else if spec.Injection.PayloadImage == nil {
		errs = append(errs, field.Required(injPath.Child("payloadImage"),
			"must be specified for CacheBlend injection"))
	} else if spec.Injection.PayloadImage.Repository == nil || *spec.Injection.PayloadImage.Repository == "" {
		errs = append(errs, field.Required(injPath.Child("payloadImage", "repository"),
			"must be a non-empty string"))
	}

	errs = append(errs, validateL2BackendSpec(spec.L2Backend)...)
	errs = append(errs, validateCoordinatorConnectionSpec(spec.Coordinator)...)

	return errs
}
