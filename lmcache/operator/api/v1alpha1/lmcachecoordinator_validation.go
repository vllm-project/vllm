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

// ValidateSpec validates the LMCacheCoordinatorSpec and returns any validation
// errors. The checks mirror MPCoordinatorConfig.__post_init__
// (lmcache/v1/mp_coordinator/config.py) so an invalid spec is rejected before
// the coordinator pod would fail to start.
func (c *LMCacheCoordinator) ValidateSpec() field.ErrorList {
	var errs field.ErrorList
	spec := &c.Spec
	specPath := field.NewPath("spec")

	if spec.Port != nil {
		port := *spec.Port
		if port < 1 || port > 65535 {
			errs = append(errs, field.Invalid(specPath.Child("port"), port, "must be in [1, 65535]"))
		}
	}

	if spec.Replicas != nil && *spec.Replicas < 0 {
		errs = append(errs, field.Invalid(specPath.Child("replicas"), *spec.Replicas, "must be >= 0"))
	}

	if spec.InstanceTimeout != nil && *spec.InstanceTimeout <= 0 {
		errs = append(errs, field.Invalid(specPath.Child("instanceTimeout"), *spec.InstanceTimeout, "must be greater than 0"))
	}

	if spec.HealthCheckInterval != nil && *spec.HealthCheckInterval < 0 {
		errs = append(errs, field.Invalid(specPath.Child("healthCheckInterval"), *spec.HealthCheckInterval, "must be >= 0"))
	}

	if spec.EvictionCheckInterval != nil && *spec.EvictionCheckInterval < 0 {
		errs = append(errs, field.Invalid(specPath.Child("evictionCheckInterval"), *spec.EvictionCheckInterval, "must be >= 0"))
	}

	if spec.EvictionRatio != nil {
		er := *spec.EvictionRatio
		if er < 0.0 || er > 1.0 {
			errs = append(errs, field.Invalid(specPath.Child("evictionRatio"), er, "must be in [0.0, 1.0]"))
		}
	}

	if spec.TriggerWatermark != nil {
		tw := *spec.TriggerWatermark
		if tw <= 0.0 || tw > 1.0 {
			errs = append(errs, field.Invalid(specPath.Child("triggerWatermark"), tw, "must be in (0.0, 1.0]"))
		}
	}

	if spec.BlendChunkSize != nil && *spec.BlendChunkSize < 1 {
		errs = append(errs, field.Invalid(specPath.Child("blendChunkSize"), *spec.BlendChunkSize, "must be > 0"))
	}

	if spec.BlendProbeStride != nil && *spec.BlendProbeStride < 1 {
		errs = append(errs, field.Invalid(specPath.Child("blendProbeStride"), *spec.BlendProbeStride, "must be > 0"))
	}

	return errs
}
