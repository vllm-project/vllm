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

// ValidateSpec validates the LMCacheEngineSpec and returns any validation errors.
func (e *LMCacheEngine) ValidateSpec() field.ErrorList {
	var errs field.ErrorList
	spec := &e.Spec

	// l1.sizeGB must be > 0
	if spec.L1.SizeGB <= 0 {
		errs = append(errs, field.Invalid(field.NewPath("spec", "l1", "sizeGB"), spec.L1.SizeGB, "must be greater than 0"))
	}

	errs = append(errs, validateEvictionSpec(spec.Eviction)...)

	// Server port validation
	if spec.Server != nil && spec.Server.Port != nil {
		port := *spec.Server.Port
		if port < 1024 || port > 65535 {
			errs = append(errs, field.Invalid(field.NewPath("spec", "server", "port"), port, "must be in [1024, 65535]"))
		}
	}

	errs = append(errs, validateL2BackendSpec(spec.L2Backend)...)
	errs = append(errs, validateCoordinatorConnectionSpec(spec.Coordinator)...)

	return errs
}

// validateCoordinatorConnectionSpec validates the shared coordinator connection
// configuration used by both LMCacheEngine and CacheBlendEngine. A nil spec is
// valid (registration is disabled). Exactly one of ref or url must be set, and
// the intervals must be positive. Errors are rooted at spec.coordinator.
func validateCoordinatorConnectionSpec(conn *CoordinatorConnectionSpec) field.ErrorList {
	var errs field.ErrorList
	if conn == nil {
		return errs
	}
	connPath := field.NewPath("spec", "coordinator")

	refSet := conn.Ref != nil && conn.Ref.Name != ""
	urlSet := conn.URL != nil && *conn.URL != ""
	switch {
	case refSet && urlSet:
		errs = append(errs, field.Invalid(connPath, "", "exactly one of ref or url must be set, got both"))
	case !refSet && !urlSet:
		errs = append(errs, field.Required(connPath, "exactly one of ref or url must be set"))
	}

	if conn.HeartbeatInterval != nil && *conn.HeartbeatInterval <= 0 {
		errs = append(errs, field.Invalid(connPath.Child("heartbeatInterval"), *conn.HeartbeatInterval, "must be greater than 0"))
	}
	if conn.L2EventFlushInterval != nil && *conn.L2EventFlushInterval <= 0 {
		errs = append(errs, field.Invalid(connPath.Child("l2EventFlushInterval"), *conn.L2EventFlushInterval, "must be greater than 0"))
	}

	return errs
}

// validateEvictionSpec validates the shared eviction configuration used by both
// LMCacheEngine and CacheBlendEngine. A nil eviction is valid (defaults apply).
// Returned errors are rooted at spec.eviction.
func validateEvictionSpec(eviction *EvictionSpec) field.ErrorList {
	var errs field.ErrorList
	if eviction == nil {
		return errs
	}
	evPath := field.NewPath("spec", "eviction")

	if eviction.Policy != nil && *eviction.Policy != "LRU" {
		errs = append(errs, field.NotSupported(evPath.Child("policy"), *eviction.Policy, []string{"LRU"}))
	}

	if eviction.TriggerWatermark != nil {
		tw := *eviction.TriggerWatermark
		if tw <= 0.0 || tw > 1.0 {
			errs = append(errs, field.Invalid(evPath.Child("triggerWatermark"), tw, "must be in (0.0, 1.0]"))
		}
	}

	if eviction.EvictionRatio != nil {
		er := *eviction.EvictionRatio
		if er <= 0.0 || er > 1.0 {
			errs = append(errs, field.Invalid(evPath.Child("evictionRatio"), er, "must be in (0.0, 1.0]"))
		}
	}

	return errs
}

// validateL2BackendSpec validates the shared L2 backend configuration used by
// both LMCacheEngine and CacheBlendEngine. A nil backend is valid (L2 is
// optional). Returned errors are rooted at spec.l2Backend.
func validateL2BackendSpec(b *L2BackendSpec) field.ErrorList {
	var errs field.ErrorList
	if b == nil {
		return errs
	}
	l2Path := field.NewPath("spec", "l2Backend")

	setCount := 0
	if b.RESP != nil {
		setCount++
	}
	if b.Raw != nil {
		setCount++
	}
	// For now we only support one kind at each LMCache server. LMCache
	// MP mode is designed to support multiple ones at the same time
	// but tests and performance validation is needed before we ship
	// it into operator.
	if setCount == 0 {
		errs = append(errs, field.Required(l2Path, "exactly one of resp or raw must be set"))
	} else if setCount > 1 {
		errs = append(errs, field.Invalid(l2Path, "", "exactly one of resp or raw must be set, got multiple"))
	}

	if b.RESP != nil {
		respPath := l2Path.Child("resp")
		if b.RESP.Host == "" {
			errs = append(errs, field.Required(respPath.Child("host"), "must be a non-empty string"))
		}
		if b.RESP.Port < 1 || b.RESP.Port > 65535 {
			errs = append(errs, field.Invalid(respPath.Child("port"), b.RESP.Port, "must be in [1, 65535]"))
		}
		if b.RESP.AuthSecretRef != nil && b.RESP.AuthSecretRef.Name == "" {
			errs = append(errs, field.Required(respPath.Child("authSecretRef", "name"), "must be non-empty"))
		}
	}

	if b.Raw != nil {
		rawPath := l2Path.Child("raw")
		if b.Raw.Type == "" {
			errs = append(errs, field.Required(rawPath.Child("type"), "must be a non-empty string"))
		}
	}

	if b.Serde != nil {
		serdePath := l2Path.Child("serde")
		if b.Serde.AESGCM == nil {
			errs = append(errs, field.Required(serdePath, "exactly one serde type must be set"))
		} else if b.Serde.AESGCM.MasterKeySecretRef.Name == "" {
			errs = append(errs, field.Required(serdePath.Child("aesgcm", "masterKeySecretRef", "name"), "must be non-empty"))
		}
		// A raw adapter spec may already carry a hand-written "serde"
		// config; combining it with the serde field would silently
		// overwrite one of the two.
		if b.Raw != nil {
			if _, ok := b.Raw.Config["serde"]; ok {
				errs = append(errs, field.Invalid(serdePath, "",
					"cannot be combined with a raw adapter that sets a 'serde' config key"))
			}
		}
	}

	return errs
}
